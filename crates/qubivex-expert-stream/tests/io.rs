//! `acquire_many`, the I/O threads, dedup and cancellation under stress.

#![cfg(not(loom))]
#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use qubivex_core::{ExpertIdx, LayerIdx};
use qubivex_expert_stream::{
    ExpertKey, ExpertSource, ExpertStore, Next, Policy, PoolSpec, StoreError,
};

/// Experts whose bytes are a function of their key; reads can be slowed down, and the
/// source tracks how many run at once.
#[derive(Default)]
struct Slow {
    delay_ms: u64,
    reads: AtomicU64,
    per_key: Mutex<std::collections::HashMap<ExpertKey, u32>>,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
    fail: Option<ExpertKey>,
    /// While set, reads block (to hold loads in the queue).
    gate: AtomicBool,
}

const SIZE: u64 = 64;

fn key(l: u32, e: u32) -> ExpertKey {
    (LayerIdx(l), ExpertIdx(e))
}

fn byte(k: ExpertKey, i: usize) -> u8 {
    (k.0.0 as usize * 31 + k.1.0 as usize * 7 + i) as u8
}

impl ExpertSource for Slow {
    fn expert_size(&self, _: ExpertKey) -> Result<u64, StoreError> {
        Ok(SIZE)
    }
    fn read_expert(&self, k: ExpertKey, dst: &mut [u8]) -> Result<(), StoreError> {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        while self.gate.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        if self.delay_ms > 0 {
            std::thread::sleep(Duration::from_millis(self.delay_ms));
        }
        self.reads.fetch_add(1, Ordering::SeqCst);
        *self.per_key.lock().unwrap().entry(k).or_default() += 1;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        if self.fail == Some(k) {
            return Err(StoreError::Load {
                key: k,
                detail: "boom".into(),
            });
        }
        for (i, b) in dst.iter_mut().enumerate() {
            *b = byte(k, i);
        }
        Ok(())
    }
}

fn check(bytes: &[u8], k: ExpertKey) {
    assert_eq!(bytes.len() as u64, SIZE);
    assert!(
        bytes.iter().enumerate().all(|(i, b)| *b == byte(k, i)),
        "{k:?}"
    );
}

fn pools(layers: u32, slots: u32) -> Vec<PoolSpec> {
    (0..layers)
        .map(|l| PoolSpec {
            layer: LayerIdx(l),
            slots,
            slot_bytes: SIZE,
        })
        .collect()
}

fn store(src: &Arc<Slow>, slots: u32, threads: usize, policy: Policy) -> ExpertStore {
    ExpertStore::with_io_threads(src.clone(), &pools(2, slots), policy, threads).unwrap()
}

/// Every expert of a batch comes out exactly once with the right bytes, whether the
/// layer has fewer slots than the batch (the batch pipelines through them) or more,
/// with or without I/O threads.
#[test]
fn batches_hand_out_every_expert_once() {
    for threads in [0, 1, 4] {
        for slots in [1, 2, 3, 8] {
            for policy in [Policy::Lru, Policy::LFRU] {
                let src = Arc::new(Slow::default());
                let s = store(&src, slots, threads, policy);
                for round in 0..20u32 {
                    let keys: Vec<ExpertKey> = (0..6)
                        .map(|i| key(round % 2, (i * 3 + round) % 10))
                        .collect();
                    let mut seen = HashSet::new();
                    for lease in s.acquire_many(&keys).unwrap() {
                        let lease = lease.unwrap();
                        check(lease.bytes(), lease.key());
                        assert!(seen.insert(lease.key()), "{:?} twice", lease.key());
                    }
                    assert_eq!(seen, keys.iter().copied().collect(), "{threads} {slots}");
                    assert!(s.used().unwrap() <= s.capacity());
                }
                let st = s.stats();
                assert_eq!(st.hits + st.loads, 120, "{threads} {slots} {policy:?}");
                assert_eq!(st.loads, src.reads.load(Ordering::SeqCst));
                assert!(st.slab_allocs <= u64::from(2 * slots));
            }
        }
    }
}

#[test]
fn a_batch_never_evicts_its_own_hits() {
    let src = Arc::new(Slow::default());
    let s = store(&src, 3, 2, Policy::Lru);
    for e in [0, 1] {
        drop(s.acquire(key(0, e)).unwrap());
    }
    // 0 and 1 are hits; 2..5 are misses competing for one free slot, then for the
    // slots 0 and 1 leave behind once used.
    let order: Vec<ExpertKey> = s
        .acquire_many(&[key(0, 2), key(0, 0), key(0, 3), key(0, 1), key(0, 4)])
        .unwrap()
        .map(|l| l.unwrap().key())
        .collect();
    assert_eq!(&order[..2], [key(0, 0), key(0, 1)], "hits come first");
    assert_eq!(s.stats().hits, 2);
    assert_eq!(
        src.reads.load(Ordering::SeqCst),
        2 + 3,
        "each miss read once"
    );
}

/// `try_next` hands out what is ready without waiting, answers `Later` (never `Full`)
/// while the caller's own leases hold every slot, and continues once they are released.
#[test]
fn try_next_lets_a_caller_hold_several_leases() {
    for threads in [0, 2] {
        let src = Arc::new(Slow::default());
        let s = store(&src, 2, threads, Policy::Lru);
        for e in [0, 1] {
            drop(s.acquire(key(0, e)).unwrap());
        }
        let mut b = s.acquire_many(&[key(0, 0), key(0, 1), key(0, 2)]).unwrap();
        let mut held = Vec::new();
        while let Next::Lease(l) = b.try_next() {
            held.push(l.unwrap());
        }
        assert_eq!(held.len(), 2, "threads {threads}: both hits at once");
        // Expert 2 needs a slot that our leases hold: not ready, and not an error.
        assert!(matches!(b.try_next(), Next::Later), "threads {threads}");
        drop(held);
        // With the slots free, the blocking `next` loads it (on this thread without I/O
        // threads); `try_next` then reports the end.
        let l = b.next().unwrap().unwrap();
        assert_eq!(l.key(), key(0, 2));
        check(l.bytes(), key(0, 2));
        drop(l);
        assert!(matches!(b.try_next(), Next::Done), "threads {threads}");
        assert!(b.next().is_none());
    }
}

/// `try_next` reports a load in flight as `Later`, then hands it out once it lands.
#[test]
fn try_next_does_not_wait_for_loads() {
    let src = Arc::new(Slow::default());
    src.gate.store(true, Ordering::SeqCst);
    let s = store(&src, 4, 1, Policy::Lru);
    let mut b = s.acquire_many(&[key(0, 0)]).unwrap();
    assert!(matches!(b.try_next(), Next::Later));
    src.gate.store(false, Ordering::SeqCst);
    let lease = loop {
        match b.try_next() {
            Next::Lease(l) => break l.unwrap(),
            Next::Later => std::thread::sleep(Duration::from_millis(1)),
            Next::Done => panic!("the expert was never handed out"),
        }
    };
    check(lease.bytes(), key(0, 0));
    drop(lease);
    assert!(matches!(b.try_next(), Next::Done));
}

#[test]
fn io_threads_read_a_batch_in_parallel() {
    let src = Arc::new(Slow {
        delay_ms: 20,
        ..Slow::default()
    });
    let s = store(&src, 8, 4, Policy::LFRU);
    let keys: Vec<ExpertKey> = (0..8).map(|e| key(0, e)).collect();
    assert_eq!(s.acquire_many(&keys).unwrap().count(), 8);
    assert!(
        src.max_in_flight.load(Ordering::SeqCst) >= 2,
        "misses are issued before waiting, and read concurrently"
    );
    assert!(s.stats().wait_ns > 0);
}

/// Many threads ask for overlapping experts at once: each expert is read once while
/// it stays resident, and everyone sees the right bytes.
#[test]
fn duplicate_requests_share_one_read() {
    for threads in [0, 3] {
        let src = Arc::new(Slow {
            delay_ms: 2,
            ..Slow::default()
        });
        // Every expert fits: nothing is evicted, so every expert is read exactly once.
        let s = store(&src, 16, threads, Policy::LFRU);
        std::thread::scope(|sc| {
            for t in 0..8u32 {
                let s = &s;
                sc.spawn(move || {
                    for i in 0..30u32 {
                        let keys: Vec<ExpertKey> = (0..4)
                            .map(|j| key((i + j) % 2, (i * 5 + j * 3 + t) % 16))
                            .collect::<HashSet<_>>()
                            .into_iter()
                            .collect();
                        for l in s.acquire_many(&keys).unwrap() {
                            let l = l.unwrap();
                            check(l.bytes(), l.key());
                        }
                    }
                });
            }
        });
        let per_key = src.per_key.lock().unwrap();
        assert!(per_key.values().all(|&n| n == 1), "{threads}: {per_key:?}");
        assert!(s.stats().dedup_waits > 0);
    }
}

/// Under contention for few slots, `Full` is the only error, bytes are always right,
/// and the accounting comes back to the resident experts.
#[test]
fn contention_stress_keeps_the_accounting() {
    let src = Arc::new(Slow::default());
    let s = store(&src, 3, 2, Policy::LFRU);
    std::thread::scope(|sc| {
        for t in 0..6u32 {
            let s = &s;
            sc.spawn(move || {
                for i in 0..200u32 {
                    let keys: Vec<ExpertKey> = (0..3)
                        .map(|j| key((i + t) % 2, (i * 7 + j * 5 + t) % 12))
                        .collect::<HashSet<_>>()
                        .into_iter()
                        .collect();
                    let Ok(b) = s.acquire_many(&keys) else {
                        continue;
                    };
                    for l in b {
                        match l {
                            Ok(l) => check(l.bytes(), l.key()),
                            Err(StoreError::Full { .. }) => break,
                            Err(e) => panic!("{e}"),
                        }
                    }
                }
            });
        }
    });
    assert_eq!(
        s.used().unwrap(),
        s.resident().unwrap().len() as u64 * SIZE,
        "no bytes leak through cancelled or failed loads"
    );
    assert!(s.resident().unwrap().len() <= 6);
}

/// Dropping a batch cancels the loads no thread has started; started ones finish and
/// stay usable.
#[test]
fn dropping_a_batch_cancels_queued_loads() {
    let src = Arc::new(Slow::default());
    src.gate.store(true, Ordering::SeqCst);
    let s = store(&src, 8, 1, Policy::LFRU);
    let keys: Vec<ExpertKey> = (0..6).map(|e| key(0, e)).collect();
    let b = s.acquire_many(&keys).unwrap();
    // The one I/O thread is blocked on the first read; the other five are queued.
    while src.in_flight.load(Ordering::SeqCst) == 0 {
        std::thread::yield_now();
    }
    drop(b);
    src.gate.store(false, Ordering::SeqCst);
    let st = s.stats();
    assert_eq!(st.cancelled, 5);
    // The started read completes and becomes an ordinary resident expert.
    let l = s.acquire(key(0, 0)).unwrap();
    check(l.bytes(), key(0, 0));
    drop(l);
    assert_eq!(src.reads.load(Ordering::SeqCst), 1);
    assert_eq!(s.used().unwrap(), SIZE);
    // Cancelled experts load normally later.
    check(s.acquire(key(0, 5)).unwrap().bytes(), key(0, 5));
}

#[test]
fn a_read_error_reaches_every_waiter_and_frees_its_slot() {
    for threads in [0, 2] {
        let src = Arc::new(Slow {
            delay_ms: 5,
            fail: Some(key(0, 3)),
            ..Slow::default()
        });
        let s = store(&src, 4, threads, Policy::LFRU);
        let errors = AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..4 {
                sc.spawn(|| {
                    for l in s.acquire_many(&[key(0, 3), key(0, 1)]).unwrap() {
                        match l {
                            Ok(l) => check(l.bytes(), key(0, 1)),
                            Err(e) => {
                                assert!(e.to_string().contains("boom"), "{e}");
                                errors.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                    }
                });
            }
        });
        assert_eq!(errors.load(Ordering::SeqCst), 4, "{threads}");
        assert_eq!(s.used().unwrap(), SIZE, "only expert 1 is resident");
        assert!(s.acquire(key(0, 2)).is_ok());
    }
}

#[test]
fn the_store_stops_its_io_threads_on_drop() {
    let src = Arc::new(Slow::default());
    let s = store(&src, 2, 3, Policy::LFRU);
    assert_eq!(s.io_threads(), 3);
    drop(s.acquire(key(0, 0)).unwrap());
    drop(s); // joins; a hang here fails the test by timeout
    assert_eq!(Arc::strong_count(&src), 1, "threads released the source");
}
