//! Prefetch (Phase 7): advisory reads at `Priority::Prefetch`, the eviction guard,
//! stale-speculation cancellation and the counters.

#![cfg(not(loom))]
#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use qubivex_core::{ExpertIdx, LayerIdx};
use qubivex_expert_stream::{
    ExpertKey, ExpertSource, ExpertStore, Policy, PoolSpec, PrefetchReceipt, StoreError,
};

const SIZE: u64 = 64;

/// Experts whose bytes are a function of their key; reads block while the gate is set.
#[derive(Default)]
struct Gated {
    reads: AtomicU64,
    in_flight: AtomicUsize,
    gate: AtomicBool,
}

fn key(l: u32, e: u32) -> ExpertKey {
    (LayerIdx(l), ExpertIdx(e))
}

fn byte(k: ExpertKey, i: usize) -> u8 {
    (k.0.0 as usize * 31 + k.1.0 as usize * 7 + i) as u8
}

impl ExpertSource for Gated {
    fn expert_size(&self, _: ExpertKey) -> Result<u64, StoreError> {
        Ok(SIZE)
    }
    fn read_expert(&self, k: ExpertKey, dst: &mut [u8]) -> Result<(), StoreError> {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        while self.gate.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        for (i, b) in dst.iter_mut().enumerate() {
            *b = byte(k, i);
        }
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }
}

fn check(bytes: &[u8], k: ExpertKey) {
    assert!(
        bytes.iter().enumerate().all(|(i, b)| *b == byte(k, i)),
        "{k:?}"
    );
}

fn store(src: &Arc<Gated>, slots: u32, threads: usize, policy: Policy) -> ExpertStore {
    let pools: Vec<PoolSpec> = (0..2)
        .map(|l| PoolSpec {
            layer: LayerIdx(l),
            slots,
            slot_bytes: SIZE,
        })
        .collect();
    ExpertStore::with_io_threads(src.clone(), &pools, policy, threads).unwrap()
}

/// Waits (bounded) until `f` holds.
fn until(f: impl Fn() -> bool) {
    for _ in 0..5000 {
        if f() {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("condition never held");
}

fn take_all(s: &ExpertStore, keys: &[ExpertKey]) {
    for lease in s.acquire_many(keys).unwrap() {
        let lease = lease.unwrap();
        check(lease.bytes(), lease.key());
    }
}

#[test]
fn a_prefetched_expert_is_a_hit_and_is_read_once() {
    let src = Arc::new(Gated::default());
    let s = store(&src, 4, 2, Policy::LFRU);
    let r = s.prefetch(&[key(0, 1), key(0, 2)], true).unwrap();
    assert_eq!(
        r,
        PrefetchReceipt {
            issued: 2,
            present: 0,
            refused: 0
        }
    );
    until(|| src.reads.load(Ordering::SeqCst) == 2);
    until(|| s.resident().unwrap().len() == 2);
    take_all(&s, &[key(0, 1), key(0, 2), key(0, 3)]);
    let st = s.stats();
    assert_eq!((st.hits, st.loads), (2, 3));
    assert_eq!(src.reads.load(Ordering::SeqCst), 3, "no second read");
    assert_eq!((st.prefetch.issued, st.prefetch.ready), (2, 2));
    assert_eq!(st.prefetch.wasted, 0);
    // Already resident: present, nothing read.
    let r = s.prefetch(&[key(0, 1)], true).unwrap();
    assert_eq!(r.present, 1);
}

#[test]
fn without_io_threads_nothing_is_read_ahead() {
    let src = Arc::new(Gated::default());
    let s = store(&src, 4, 0, Policy::LFRU);
    let r = s.prefetch(&[key(0, 1), key(1, 1)], false).unwrap();
    assert_eq!((r.issued, r.refused), (0, 2));
    assert_eq!(src.reads.load(Ordering::SeqCst), 0);
    assert_eq!(s.used().unwrap(), 0);
}

#[test]
fn unknown_layers_are_refused_not_errors() {
    let src = Arc::new(Gated::default());
    let s = store(&src, 1, 1, Policy::LFRU);
    assert_eq!(s.prefetch(&[key(9, 0)], false).unwrap().refused, 1);
}

#[test]
fn the_guard_keeps_a_hot_resident_from_a_cold_guess() {
    for (guard, issued) in [(true, 0), (false, 1)] {
        let src = Arc::new(Gated::default());
        let s = store(&src, 1, 1, Policy::LFRU);
        // Expert 0 of layer 0 is used often: heat 6.
        for _ in 0..6 {
            take_all(&s, &[key(0, 0)]);
        }
        let r = s.prefetch(&[key(0, 5)], guard).unwrap();
        assert_eq!(r.issued, issued, "guard {guard}");
        assert_eq!(r.refused, 1 - issued);
        if issued == 1 {
            until(|| s.resident().unwrap().contains(&key(0, 5)));
        }
        let resident = s.resident().unwrap();
        assert_eq!(
            resident.contains(&key(0, 0)),
            guard,
            "the hot expert stays only with the guard"
        );
    }
    // A hot prediction passes the guard: expert 5 earns heat while expert 0 decays to
    // a stale, cold resident.
    let src = Arc::new(Gated::default());
    let s = store(&src, 1, 1, Policy::LFRU);
    take_all(&s, &[key(0, 0)]);
    for _ in 0..8 {
        take_all(&s, &[key(0, 5)]);
    }
    take_all(&s, &[key(0, 0)]);
    assert_eq!(s.prefetch(&[key(0, 5)], true).unwrap().issued, 1);
}

#[test]
fn a_prediction_never_evicts_its_own_experts() {
    let src = Arc::new(Gated::default());
    let s = store(&src, 2, 1, Policy::Lru);
    take_all(&s, &[key(0, 1)]);
    take_all(&s, &[key(0, 2)]);
    // Expert 1 is the LRU victim, but the prediction names it: it stays, and the
    // prediction's third expert finds no other victim.
    let r = s
        .prefetch(&[key(0, 1), key(0, 3), key(0, 4)], false)
        .unwrap();
    assert_eq!((r.present, r.issued, r.refused), (1, 1, 1));
    until(|| s.resident().unwrap().contains(&key(0, 3)));
    let res = s.resident().unwrap();
    assert!(res.contains(&key(0, 1)) && !res.contains(&key(0, 2)));
}

#[test]
fn the_guard_is_lfru_only() {
    let src = Arc::new(Gated::default());
    let s = store(&src, 1, 1, Policy::Lru);
    for _ in 0..6 {
        take_all(&s, &[key(0, 0)]);
    }
    assert_eq!(s.prefetch(&[key(0, 5)], true).unwrap().issued, 1);
}

#[test]
fn demand_promotes_its_queued_prefetch_and_drops_the_stale_ones() {
    let src = Arc::new(Gated::default());
    let s = store(&src, 3, 1, Policy::LFRU);
    src.gate.store(true, Ordering::SeqCst);
    // The single I/O thread takes expert 1 and blocks; 2 and 3 stay queued.
    let r = s
        .prefetch(&[key(0, 1), key(0, 2), key(0, 3)], false)
        .unwrap();
    assert_eq!(r.issued, 3);
    until(|| src.in_flight.load(Ordering::SeqCst) == 1);
    let opener = {
        let src = Arc::clone(&src);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            src.gate.store(false, Ordering::SeqCst);
        })
    };
    // Demand 3 (queued: promoted, joined) and 4 (a miss); 2 is stale and dropped.
    take_all(&s, &[key(0, 3), key(0, 4)]);
    opener.join().unwrap();
    let st = s.stats();
    assert_eq!(st.prefetch.cancelled, 1, "expert 2 never read");
    assert_eq!(st.prefetch.late, 1, "expert 3 was demanded while queued");
    until(|| src.in_flight.load(Ordering::SeqCst) == 0);
    assert_eq!(src.reads.load(Ordering::SeqCst), 3, "1, 3 and 4");
    assert!(!s.resident().unwrap().contains(&key(0, 2)));
    assert!(s.used().unwrap() <= s.capacity());
}

#[test]
fn a_layer_full_of_loading_prefetches_waits_instead_of_failing() {
    let src = Arc::new(Gated::default());
    let s = store(&src, 1, 1, Policy::LFRU);
    src.gate.store(true, Ordering::SeqCst);
    s.prefetch(&[key(0, 1)], false).unwrap();
    until(|| src.in_flight.load(Ordering::SeqCst) == 1);
    let opener = {
        let src = Arc::clone(&src);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            src.gate.store(false, Ordering::SeqCst);
        })
    };
    // The only slot is held by a started read of expert 1: expert 2 waits for it.
    take_all(&s, &[key(0, 2)]);
    opener.join().unwrap();
    let st = s.stats();
    assert_eq!((st.prefetch.wasted, st.prefetch.wasted_bytes), (1, SIZE));
}

#[test]
fn an_unused_prefetch_is_counted_as_wasted() {
    let src = Arc::new(Gated::default());
    let s = store(&src, 2, 1, Policy::LFRU);
    s.prefetch(&[key(1, 7)], false).unwrap();
    until(|| s.resident().unwrap().contains(&key(1, 7)));
    take_all(&s, &[key(1, 1)]);
    take_all(&s, &[key(1, 2)]);
    let st = s.stats();
    assert_eq!((st.prefetch.wasted, st.prefetch.wasted_bytes), (1, SIZE));
    assert_eq!(st.prefetch.ready, 0);
}

#[test]
fn cancel_drops_queued_prefetches_of_every_layer() {
    let src = Arc::new(Gated::default());
    let s = store(&src, 4, 1, Policy::LFRU);
    src.gate.store(true, Ordering::SeqCst);
    s.prefetch(&[key(0, 1), key(0, 2), key(1, 3)], false)
        .unwrap();
    until(|| src.in_flight.load(Ordering::SeqCst) == 1);
    assert_eq!(s.cancel_prefetch(Some(LayerIdx(1))).unwrap(), 1);
    assert_eq!(
        s.cancel_prefetch(None).unwrap(),
        1,
        "expert 2; 1 is reading"
    );
    src.gate.store(false, Ordering::SeqCst);
    until(|| src.in_flight.load(Ordering::SeqCst) == 0);
    until(|| s.resident().unwrap().len() == 1);
    assert_eq!(s.used().unwrap(), SIZE);
    assert_eq!(s.stats().prefetch.cancelled, 2);
}

/// Callers demand while a predictor thread prefetches at random: every lease has the
/// right bytes, every batch completes, and the pools never exceed their capacity.
#[test]
fn prefetch_under_contention_never_changes_what_demand_gets() {
    for slots in [1, 2, 4] {
        for policy in [Policy::Lru, Policy::LFRU] {
            let src = Arc::new(Gated::default());
            let s = store(&src, slots, 3, policy);
            let stop = AtomicBool::new(false);
            std::thread::scope(|sc| {
                sc.spawn(|| {
                    let mut x = 7u32;
                    while !stop.load(Ordering::Relaxed) {
                        x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
                        let ks: Vec<ExpertKey> = (0..3)
                            .map(|i| key((x >> 3) % 2, (x >> (5 + i)) % 8))
                            .collect();
                        s.prefetch(&ks, x % 3 == 0).unwrap();
                        if x % 17 == 0 {
                            s.cancel_prefetch(None).unwrap();
                        }
                    }
                });
                for round in 0..300u32 {
                    let layer = round % 2;
                    let mut ks: Vec<ExpertKey> = (0..3)
                        .map(|i| key(layer, (round * 5 + i * 3) % 8))
                        .collect();
                    ks.dedup();
                    take_all(&s, &ks);
                    assert!(s.used().unwrap() <= s.capacity());
                }
                stop.store(true, Ordering::Relaxed);
            });
        }
    }
}
