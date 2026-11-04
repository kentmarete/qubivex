//! The store under injected faults (IMPROVEMENTS_EVALUATION §10): I/O errors, short
//! reads, corrupt bytes, a panicking source, allocation failure and cancellation
//! mid-layer. In every case the failure reaches the caller as an error, nothing panics,
//! no wrong bytes are handed out (corruption needs [`VerifiedSource`]), nothing stays
//! leased, loading or failed, and the next acquire of the same expert succeeds.

#![cfg(not(loom))]
#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use qubivex_core::{ExpertIdx, LayerIdx};
use qubivex_expert_stream::fault::{Fault, FaultySource, Rule, Target, VerifiedSource};
use qubivex_expert_stream::{
    ExpertKey, ExpertSource, ExpertStore, Occupancy, Policy, PoolSpec, StoreError,
};

const SIZE: u64 = 4096;

fn key(l: u32, e: u32) -> ExpertKey {
    (LayerIdx(l), ExpertIdx(e))
}

fn byte(k: ExpertKey, i: usize) -> u8 {
    (k.0.0 as usize * 31 + k.1.0 as usize * 7 + i * 13) as u8
}

/// Experts whose bytes are a function of their key, optionally slow. Layer 9's experts
/// are absurdly large, so allocating a slot for one fails.
#[derive(Default)]
struct Pattern {
    delay: Duration,
}

const HUGE_LAYER: u32 = 9;
/// 4 PiB: more than any address space, less than `isize::MAX`.
const HUGE: u64 = 1 << 52;

impl ExpertSource for Pattern {
    fn expert_size(&self, k: ExpertKey) -> Result<u64, StoreError> {
        Ok(if k.0.0 == HUGE_LAYER { HUGE } else { SIZE })
    }
    fn read_expert(&self, k: ExpertKey, dst: &mut [u8]) -> Result<(), StoreError> {
        std::thread::sleep(self.delay);
        for (i, b) in dst.iter_mut().enumerate() {
            *b = byte(k, i);
        }
        Ok(())
    }
}

fn check(bytes: &[u8], k: ExpertKey) {
    assert_eq!(bytes.len() as u64, SIZE, "{k:?}");
    assert!(
        bytes.iter().enumerate().all(|(i, b)| *b == byte(k, i)),
        "{k:?}: wrong bytes"
    );
}

fn pools(slots: u32) -> Vec<PoolSpec> {
    let mut v: Vec<PoolSpec> = (0..2)
        .map(|l| PoolSpec {
            layer: LayerIdx(l),
            slots,
            slot_bytes: SIZE,
        })
        .collect();
    v.push(PoolSpec {
        layer: LayerIdx(HUGE_LAYER),
        slots: 1,
        slot_bytes: HUGE,
    });
    v
}

fn store(src: Arc<dyn ExpertSource>, slots: u32, threads: usize) -> ExpertStore {
    ExpertStore::with_io_threads(src, &pools(slots), Policy::LFRU, threads).unwrap()
}

fn faulty(delay: Duration, rules: impl IntoIterator<Item = Rule>) -> Arc<FaultySource> {
    Arc::new(FaultySource::new(Arc::new(Pattern { delay }), rules))
}

/// Waits (bounded) until no load is queued or in flight, then checks that nothing is
/// leased or failed and that the bytes accounted are exactly the resident experts'.
fn settled(s: &ExpertStore) -> Occupancy {
    let t = Instant::now();
    let o = loop {
        let o = s.occupancy().unwrap();
        if o.loading == 0 && o.queued == 0 {
            break o;
        }
        assert!(
            t.elapsed() < Duration::from_secs(10),
            "loads never settled: {o:?}"
        );
        std::thread::sleep(Duration::from_millis(1));
    };
    assert_eq!((o.leased, o.failed), (0, 0), "{o:?}");
    assert_eq!(o.used, o.resident as u64 * SIZE, "{o:?}");
    o
}

/// Silences the default panic hook's report of the source's injected panics, which are
/// expected and caught. Other panics (assertions) still print.
fn quiet_injected_panics() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = info
            .payload()
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| info.payload().downcast_ref::<&str>().copied())
            .unwrap_or("");
        if !msg.starts_with("injected:") {
            prev(info);
        }
    }));
}

const THREADS: [usize; 2] = [0, 4];

/// EIO, a short read and a panic each fail one acquire with [`StoreError::Load`]; the
/// next acquire reads again and gets the right bytes.
#[test]
fn a_failed_read_fails_its_acquire_and_the_next_one_succeeds() {
    quiet_injected_panics();
    let faults = [
        (Fault::Eio, "Input/output error"),
        (Fault::ShortRead { keep: 100 }, "short read, 100 of 4096"),
        (Fault::Panic, "the expert source panicked"),
    ];
    for threads in THREADS {
        for (fault, want) in &faults {
            let k = key(1, 3);
            let src = faulty(
                Duration::ZERO,
                [Rule::once_after(Target::Expert(k), 0, fault.clone())],
            );
            let s = store(src.clone(), 4, threads);
            match s.acquire(k) {
                Err(StoreError::Load { key: got, detail }) => {
                    assert_eq!(got, k);
                    assert!(
                        detail.contains(want),
                        "{threads} threads, {fault:?}: {detail}"
                    );
                }
                other => panic!("{threads} threads, {fault:?}: {other:?}"),
            }
            let o = settled(&s);
            assert_eq!(o.resident, 0, "{fault:?}: a failed read was published");
            check(s.acquire(k).unwrap().bytes(), k);
            assert_eq!((src.reads(), src.injected()), (2, 1));
            settled(&s);
        }
    }
}

/// One expert of a layer fails while the others load. Consuming the whole batch gives
/// every other expert with the right bytes; dropping it at the error (as the forward
/// pass does) leaves no lease or load behind once in-flight reads land.
#[test]
fn one_failure_in_a_layer_spares_the_rest() {
    let bad = key(0, 5);
    let keys: Vec<ExpertKey> = (0..8).map(|e| key(0, e)).collect();
    for threads in THREADS {
        for slots in [3, 8] {
            // The whole batch.
            let src = faulty(
                Duration::from_millis(2),
                [Rule::always(Target::Expert(bad), Fault::Eio)],
            );
            let s = store(src, slots, threads);
            let mut errors = 0;
            for r in s.acquire_many(&keys).unwrap() {
                match r {
                    Ok(l) => {
                        assert_ne!(l.key(), bad);
                        check(l.bytes(), l.key());
                    }
                    Err(StoreError::Load { key, .. }) => {
                        assert_eq!(key, bad);
                        errors += 1;
                    }
                    Err(e) => panic!("{threads} threads, {slots} slots: {e}"),
                }
            }
            assert_eq!(errors, 1);
            settled(&s);

            // Stopping at the first error.
            let src = faulty(
                Duration::from_millis(2),
                [Rule::always(Target::Expert(bad), Fault::Eio)],
            );
            let s = store(src, slots, threads);
            {
                let mut b = s.acquire_many(&keys).unwrap();
                loop {
                    match b.next() {
                        Some(Ok(l)) => check(l.bytes(), l.key()),
                        Some(Err(_)) => break,
                        None => panic!("{threads} threads, {slots} slots: no error"),
                    }
                }
            }
            settled(&s);
            for &k in &keys {
                if k != bad {
                    check(s.acquire(k).unwrap().bytes(), k);
                }
            }
            settled(&s);
        }
    }
}

/// Every caller waiting on a failed read gets the error; the entry is then gone and
/// the next acquire reads again.
#[test]
fn every_waiter_on_a_failed_read_gets_the_error() {
    let k = key(0, 2);
    let src = faulty(
        Duration::from_millis(30),
        [Rule::once_after(
            Target::Expert(k),
            0,
            Fault::ShortRead { keep: 0 },
        )],
    );
    let s = store(src.clone(), 4, 4);
    let results: Vec<_> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..4)
            .map(|_| sc.spawn(|| s.acquire(k).map(|l| l.key())))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // Callers that arrived after the failure was taken read again and succeed; at
    // least the first fails, and exactly one read was faulted.
    assert!(results.iter().any(Result::is_err), "{results:?}");
    assert_eq!(src.injected(), 1);
    for r in &results {
        if let Err(e) = r {
            assert!(matches!(e, StoreError::Load { .. }), "{e}");
        }
    }
    settled(&s);
    check(s.acquire(k).unwrap().bytes(), k);
}

/// Corrupt bytes that the source reports as good are handed out by a plain store (it
/// cannot know); behind a [`VerifiedSource`] the read fails instead, and a clean re-read
/// succeeds.
#[test]
fn corruption_is_caught_by_checksums() {
    let k = key(1, 1);
    let corrupt = Fault::Corrupt {
        offset: 1000,
        xor: 0x10,
    };

    let src = faulty(
        Duration::ZERO,
        [Rule::once_after(Target::Expert(k), 0, corrupt.clone())],
    );
    let s = store(src, 4, 0);
    let l = s.acquire(k).unwrap();
    assert_eq!(
        l.bytes()[1000],
        byte(k, 1000) ^ 0x10,
        "unverified: undetectable"
    );
    drop(l);

    for threads in THREADS {
        let pattern = Pattern::default();
        let all = (0..2).flat_map(|l| (0..8).map(move |e| key(l, e)));
        let sums = VerifiedSource::record(&pattern, all).unwrap();
        let inner = faulty(
            Duration::ZERO,
            [Rule::once_after(Target::Expert(k), 0, corrupt.clone())],
        );
        let src = Arc::new(VerifiedSource::new(inner, sums));
        let s = store(src, 4, threads);
        match s.acquire(k) {
            Err(StoreError::Load { detail, .. }) => {
                assert!(detail.contains("checksum mismatch"), "{detail}");
            }
            other => panic!("{threads} threads: {other:?}"),
        }
        settled(&s);
        check(s.acquire(k).unwrap().bytes(), k);
        // An expert with no recorded checksum is refused, not trusted.
        assert!(s.acquire(key(1, 30)).is_err());
        settled(&s);
    }
}

/// A slot the allocator cannot provide fails the acquire with a load error, and the
/// store keeps serving other layers.
#[test]
fn allocation_failure_is_an_error() {
    for threads in THREADS {
        let s = store(Arc::new(Pattern::default()), 4, threads);
        let k = key(HUGE_LAYER, 0);
        match s.acquire(k) {
            Err(StoreError::Load { detail, .. }) => {
                assert!(detail.contains("cannot allocate"), "{detail}");
            }
            other => panic!("{threads} threads: {other:?}"),
        }
        let o = s.occupancy().unwrap();
        assert_eq!((o.loading, o.failed, o.used), (0, 0, 0), "{o:?}");
        check(s.acquire(key(0, 0)).unwrap().bytes(), key(0, 0));
        settled(&s);
    }
}

/// Dropping a batch mid-layer cancels its queued loads, lets started ones finish, and
/// leaves every expert acquirable.
#[test]
fn cancellation_mid_layer_leaks_nothing() {
    let keys: Vec<ExpertKey> = (0..8).map(|e| key(1, e)).collect();
    for threads in [1, 4] {
        let src = faulty(
            Duration::ZERO,
            [Rule::always(
                Target::Layer(LayerIdx(1)),
                Fault::Delay(Duration::from_millis(20)),
            )],
        );
        let s = store(src, 8, threads);
        {
            let mut b = s.acquire_many(&keys).unwrap();
            let first = b.next().unwrap().unwrap();
            check(first.bytes(), first.key());
        }
        if threads == 1 {
            assert!(s.stats().cancelled > 0, "nothing was cancelled");
        }
        settled(&s);
        for r in s.acquire_many(&keys).unwrap() {
            let l = r.unwrap();
            check(l.bytes(), l.key());
        }
        settled(&s);
    }
}

/// A prefetch whose read fails has no waiter to tell; the demand that follows reads
/// again and succeeds.
#[test]
fn a_failed_prefetch_does_not_fail_the_demand() {
    let k = key(0, 4);
    let src = faulty(
        Duration::ZERO,
        [Rule::once_after(Target::Expert(k), 0, Fault::Eio)],
    );
    let s = store(src.clone(), 4, 2);
    let r = s.prefetch(&[k], false).unwrap();
    assert_eq!(r.issued, 1);
    let t = Instant::now();
    while src.injected() == 0 || s.occupancy().unwrap().loading > 0 {
        assert!(t.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(1));
    }
    check(s.acquire(k).unwrap().bytes(), k);
    assert_eq!(src.reads(), 2);
    settled(&s);
}
