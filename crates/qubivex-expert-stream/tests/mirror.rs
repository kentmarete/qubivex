//! Failover between copies of a model's experts (ADR-006 amendment 4).

#![cfg(not(loom))]
#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use qubivex_core::{ExpertIdx, LayerIdx};
use qubivex_expert_stream::fault::{Fault, FaultySource, Rule, Target, VerifiedSource, checksum};
use qubivex_expert_stream::mirror::{MirrorPolicy, MirroredSource};
use qubivex_expert_stream::{ExpertKey, ExpertSource, ExpertStore, Policy, PoolSpec, StoreError};

const SIZE: u64 = 1024;

fn key(e: u32) -> ExpertKey {
    (LayerIdx(0), ExpertIdx(e))
}

fn byte(k: ExpertKey, i: usize) -> u8 {
    (k.1.0 as usize * 7 + i * 13) as u8
}

struct Pattern;

impl ExpertSource for Pattern {
    fn expert_size(&self, _: ExpertKey) -> Result<u64, StoreError> {
        Ok(SIZE)
    }
    fn read_expert(&self, k: ExpertKey, dst: &mut [u8]) -> Result<(), StoreError> {
        for (i, b) in dst.iter_mut().enumerate() {
            *b = byte(k, i);
        }
        Ok(())
    }
}

fn read(src: &dyn ExpertSource, k: ExpertKey) -> Result<Vec<u8>, StoreError> {
    let mut v = vec![0u8; SIZE as usize];
    src.read_expert(k, &mut v)?;
    Ok(v)
}

fn good(v: &[u8], k: ExpertKey) -> bool {
    v.iter().enumerate().all(|(i, b)| *b == byte(k, i))
}

fn faulty(rules: impl IntoIterator<Item = Rule>) -> Arc<FaultySource> {
    Arc::new(FaultySource::new(Arc::new(Pattern), rules))
}

const QUICK: MirrorPolicy = MirrorPolicy {
    trip: 3,
    cooldown: Duration::from_millis(80),
};

/// A failing first copy: every read is served by the second, the first is skipped
/// after `trip` failures, tried again after the cooldown, and preferred again once it
/// reads well.
#[test]
fn a_failing_copy_is_skipped_then_tried_again() {
    // The first copy fails its first 5 reads, then recovers.
    let a = faulty([Rule {
        target: Target::Any,
        skip: 0,
        times: 5,
        fault: Fault::Eio,
    }]);
    let b = faulty([]);
    let m = MirroredSource::new(
        [
            ("a".to_owned(), a.clone() as Arc<dyn ExpertSource>),
            ("b".to_owned(), b.clone() as _),
        ],
        QUICK,
    );
    for e in 0..10 {
        assert!(good(&read(&m, key(e)).unwrap(), key(e)));
    }
    let h = m.health();
    assert_eq!(
        (h[0].failures, h[0].failing, h[0].skipped),
        (3, 3, true),
        "{h:?}"
    );
    assert_eq!((a.reads(), b.reads()), (3, 10), "a is skipped once tripped");
    assert_eq!(m.failovers(), 3);

    // After the cooldown, `a` is tried again (failing twice more, then recovering).
    std::thread::sleep(QUICK.cooldown + Duration::from_millis(20));
    assert!(good(&read(&m, key(0)).unwrap(), key(0)));
    assert_eq!(a.reads(), 4);
    assert!(m.health()[0].skipped, "one more failure trips it again");
    std::thread::sleep(QUICK.cooldown + Duration::from_millis(20));
    read(&m, key(1)).unwrap();
    std::thread::sleep(QUICK.cooldown + Duration::from_millis(20));
    let before = b.reads();
    for e in 0..4 {
        assert!(good(&read(&m, key(e)).unwrap(), key(e)));
    }
    let h = m.health();
    assert_eq!((h[0].failing, h[0].skipped), (0, false), "{h:?}");
    assert!(b.reads() - before <= 1, "a serves again once it reads well");
}

/// When every copy fails, the error names each copy and its failure; a panicking copy
/// is one failure among others.
#[test]
fn every_copy_failing_is_one_error_naming_them() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let quiet = info
            .payload()
            .downcast_ref::<String>()
            .is_some_and(|m| m.starts_with("injected:"));
        if !quiet {
            prev(info);
        }
    }));
    let m = MirroredSource::new(
        [
            (
                "ssd".to_owned(),
                faulty([Rule::always(Target::Any, Fault::Eio)]) as Arc<dyn ExpertSource>,
            ),
            (
                "usb".to_owned(),
                faulty([Rule::always(Target::Any, Fault::Panic)]) as _,
            ),
        ],
        MirrorPolicy::default(),
    );
    match read(&m, key(3)) {
        Err(StoreError::Load { detail, .. }) => {
            assert!(detail.contains("ssd: injected"), "{detail}");
            assert!(detail.contains("usb: the source panicked"), "{detail}");
        }
        other => panic!("{other:?}"),
    }
    // A panic in the last copy is reported, and the next copy (none) is not needed
    // for a recovered read.
    let m = MirroredSource::new(
        [
            (
                "usb".to_owned(),
                faulty([Rule::once_after(Target::Any, 0, Fault::Panic)]) as Arc<dyn ExpertSource>,
            ),
            ("ssd".to_owned(), faulty([]) as _),
        ],
        MirrorPolicy::default(),
    );
    assert!(good(&read(&m, key(1)).unwrap(), key(1)));
    assert_eq!(m.failovers(), 1);
}

/// Behind checksums, a copy returning corrupt bytes fails over to a good one, and the
/// store sees only good bytes.
#[test]
fn corrupt_bytes_on_one_copy_fail_over() {
    let sums: HashMap<ExpertKey, u64> = (0..8)
        .map(|e| {
            let k = key(e);
            let v: Vec<u8> = (0..SIZE as usize).map(|i| byte(k, i)).collect();
            (k, checksum(&v))
        })
        .collect();
    let bad = faulty([Rule::always(
        Target::Any,
        Fault::Corrupt { offset: 5, xor: 1 },
    )]);
    let copies: Vec<(String, Arc<dyn ExpertSource>)> = vec![
        (
            "bad".to_owned(),
            Arc::new(VerifiedSource::new(bad, sums.clone())),
        ),
        (
            "good".to_owned(),
            Arc::new(VerifiedSource::new(faulty([]), sums)),
        ),
    ];
    let m = Arc::new(MirroredSource::new(copies, MirrorPolicy::default()));
    let pools = [PoolSpec {
        layer: LayerIdx(0),
        slots: 8,
        slot_bytes: SIZE,
    }];
    let s = ExpertStore::with_io_threads(m.clone(), &pools, Policy::LFRU, 2).unwrap();
    let keys: Vec<ExpertKey> = (0..8).map(key).collect();
    for l in s.acquire_many(&keys).unwrap() {
        let l = l.unwrap();
        assert!(good(l.bytes(), l.key()));
    }
    let h = m.health();
    assert_eq!(h[1].reads, 8, "{h:?}");
    assert!(h[0].skipped, "{h:?}");
}
