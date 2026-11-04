//! Expert store behaviour: correctness at every capacity, leases, errors, threads.

#![cfg(not(loom))]
#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use qubivex_core::{ExpertIdx, LayerIdx};
use qubivex_expert_stream::{
    ExpertKey, ExpertSource, ExpertStore, Held, Policy, PoolSpec, StoreError,
};

/// Experts whose bytes are a function of their key; sizes vary by layer.
struct Source {
    reads: AtomicU64,
    fail: Option<ExpertKey>,
}

fn key(l: u32, e: u32) -> ExpertKey {
    (LayerIdx(l), ExpertIdx(e))
}

fn size_of(k: ExpertKey) -> u64 {
    100 + u64::from(k.0.0) * 10
}

fn byte(k: ExpertKey, i: usize) -> u8 {
    (k.0.0 as usize * 31 + k.1.0 as usize * 7 + i) as u8
}

impl ExpertSource for Source {
    fn expert_size(&self, k: ExpertKey) -> Result<u64, StoreError> {
        Ok(size_of(k))
    }
    fn read_expert(&self, k: ExpertKey, dst: &mut [u8]) -> Result<(), StoreError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
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

fn source() -> Arc<Source> {
    Arc::new(Source {
        reads: AtomicU64::new(0),
        fail: None,
    })
}

fn check(lease: &[u8], k: ExpertKey) {
    assert_eq!(lease.len() as u64, size_of(k));
    assert!(
        lease.iter().enumerate().all(|(i, b)| *b == byte(k, i)),
        "{k:?}"
    );
}

/// `slots` slots for each of layers `0..layers`, sized for that layer's experts.
fn pools(layers: u32, slots: u32) -> Vec<PoolSpec> {
    (0..layers)
        .map(|l| PoolSpec {
            layer: LayerIdx(l),
            slots,
            slot_bytes: size_of(key(l, 0)),
        })
        .collect()
}

#[test]
fn every_capacity_sees_the_same_bytes() {
    // A routing-like access pattern over 3 layers x 8 experts.
    let pattern: Vec<ExpertKey> = (0..200u32)
        .map(|i| key(i % 3, (i * 5 + i / 7) % 8))
        .collect();
    let distinct: std::collections::HashSet<_> = pattern.iter().collect();
    for policy in [Policy::Lru, Policy::LFRU, Policy::Lfru { decay_window: 3 }] {
        for slots in [1, 2, 3, 5, 8] {
            let src = source();
            let store = ExpertStore::new(src.clone(), &pools(3, slots), policy);
            for &k in &pattern {
                let lease = store.acquire(k).unwrap();
                check(lease.bytes(), k);
                assert!(store.used().unwrap() <= store.capacity());
            }
            let st = store.stats();
            assert_eq!(
                st.hits + st.loads,
                pattern.len() as u64,
                "{policy:?} {slots}"
            );
            assert_eq!(st.loads, src.reads.load(Ordering::Relaxed));
            assert!(st.slab_allocs <= u64::from(3 * slots), "slabs are reused");
            if slots == 8 {
                assert_eq!(st.loads, distinct.len() as u64, "all-resident loads once");
                assert_eq!(st.evictions, 0);
            }
        }
    }
}

#[test]
fn pools_compete_only_within_their_layer() {
    let store = ExpertStore::new(source(), &pools(2, 2), Policy::Lru);
    drop(store.acquire(key(0, 0)).unwrap());
    drop(store.acquire(key(0, 1)).unwrap());
    // Layer 1 misses never evict layer 0.
    for e in 0..6 {
        drop(store.acquire(key(1, e)).unwrap());
    }
    let mut r = store.resident().unwrap();
    r.sort();
    assert_eq!(r, [key(0, 0), key(0, 1), key(1, 4), key(1, 5)]);
    assert_eq!(store.capacity(), 2 * 100 + 2 * 110);
}

#[test]
fn lru_skips_leased_experts() {
    let store = ExpertStore::new(source(), &pools(1, 3), Policy::Lru);
    let a = store.acquire(key(0, 0)).unwrap();
    drop(store.acquire(key(0, 1)).unwrap());
    drop(store.acquire(key(0, 2)).unwrap());
    // The LRU unleased one is 1, since 0 is leased.
    drop(store.acquire(key(0, 3)).unwrap());
    let mut r = store.resident().unwrap();
    r.sort();
    assert_eq!(r, vec![key(0, 0), key(0, 2), key(0, 3)]);
    drop(a);
}

#[test]
fn lfru_keeps_hot_experts_through_a_scan() {
    let store = ExpertStore::new(source(), &pools(1, 2), Policy::LFRU);
    for _ in 0..4 {
        drop(store.acquire(key(0, 7)).unwrap());
    }
    // A scan of cold experts churns the other slot but never evicts the hot one.
    for e in 0..6 {
        drop(store.acquire(key(0, e)).unwrap());
    }
    let before = store.stats().hits;
    drop(store.acquire(key(0, 7)).unwrap());
    assert_eq!(store.stats().hits, before + 1);
    // LRU would have lost it.
    let lru = ExpertStore::new(source(), &pools(1, 2), Policy::Lru);
    for _ in 0..4 {
        drop(lru.acquire(key(0, 7)).unwrap());
    }
    for e in 0..6 {
        drop(lru.acquire(key(0, e)).unwrap());
    }
    let before = lru.stats().hits;
    drop(lru.acquire(key(0, 7)).unwrap());
    assert_eq!(lru.stats().hits, before);
}

#[test]
fn full_too_large_and_no_pool_are_errors_not_waits() {
    let store = ExpertStore::new(source(), &pools(1, 2), Policy::LFRU);
    let a = store.acquire(key(0, 0)).unwrap();
    let b = store.acquire(key(0, 1)).unwrap();
    assert!(matches!(
        store.acquire(key(0, 2)),
        Err(StoreError::Full {
            leased: 2,
            slots: 2,
            ..
        })
    ));
    drop(a);
    assert!(
        store.acquire(key(0, 2)).is_ok(),
        "fits once a lease is released"
    );
    drop(b);
    assert!(matches!(
        store.acquire(key(1, 0)),
        Err(StoreError::NoPool(LayerIdx(1)))
    ));
    let small = ExpertStore::new(
        source(),
        &[PoolSpec {
            layer: LayerIdx(0),
            slots: 4,
            slot_bytes: 50,
        }],
        Policy::Lru,
    );
    assert!(matches!(
        small.acquire(key(0, 0)),
        Err(StoreError::TooLarge { size: 100, .. })
    ));
}

#[test]
fn load_errors_release_the_reservation() {
    let src = Arc::new(Source {
        reads: AtomicU64::new(0),
        fail: Some(key(0, 3)),
    });
    let store = ExpertStore::new(src, &pools(1, 4), Policy::LFRU);
    let e = store.acquire(key(0, 3)).unwrap_err();
    assert!(e.to_string().contains("boom"));
    assert_eq!(store.used().unwrap(), 0);
    assert!(store.acquire(key(0, 4)).is_ok());
}

#[test]
fn threads_share_loads_and_respect_capacity() {
    for policy in [Policy::Lru, Policy::LFRU] {
        let src = source();
        let store = ExpertStore::new(src.clone(), &pools(3, 3), policy);
        let cap = store.capacity();
        std::thread::scope(|s| {
            for t in 0..8u32 {
                let store = &store;
                s.spawn(move || {
                    for i in 0..300u32 {
                        let k = key((i + t) % 3, (i * 3 + t) % 6);
                        match store.acquire(k) {
                            Ok(l) => check(l.bytes(), k),
                            // Many threads may hold leases at once; Full is a valid answer.
                            Err(StoreError::Full { .. }) => {}
                            Err(e) => panic!("{e}"),
                        }
                        assert!(store.used().unwrap() <= cap);
                    }
                });
            }
        });
        let st = store.stats();
        assert_eq!(st.loads, src.reads.load(Ordering::Relaxed));
        assert!(st.hits > 0);
    }
}

/// The planner shrinks the pools as the KV grows: unleased experts go first, by the
/// policy; leased ones stay until released; bytes seen are unchanged.
#[test]
fn shrinking_evicts_by_policy_and_keeps_leases() {
    let s = ExpertStore::new(source(), &pools(1, 5), Policy::Lru);
    for e in 0..5 {
        drop(s.acquire(key(0, e)).unwrap()); // LRU order 0..5
    }
    let held = s.acquire(key(0, 0)).unwrap(); // now most recent, and leased
    assert_eq!(s.set_pools(&pools(1, 3)).unwrap(), 300);
    let mut r = s.resident().unwrap();
    r.sort();
    assert_eq!(
        r,
        [key(0, 0), key(0, 3), key(0, 4)],
        "1 and 2 were least recent"
    );
    // Below the leased slots: everything unleased goes, the lease stays.
    let none = |n| {
        vec![PoolSpec {
            layer: LayerIdx(0),
            slots: n,
            slot_bytes: 100,
        }]
    };
    assert_eq!(s.set_pools(&none(0)).unwrap(), 100);
    assert_eq!(s.resident().unwrap(), [key(0, 0)]);
    check(held.bytes(), key(0, 0));
    assert!(matches!(s.acquire(key(0, 1)), Err(StoreError::Full { .. })));
    drop(held);
    // Growing again: misses load normally, and the leased expert is evicted lazily.
    assert_eq!(s.set_pools(&none(1)).unwrap(), 100);
    check(s.acquire(key(0, 1)).unwrap().bytes(), key(0, 1));
    assert_eq!(s.capacity(), 100);
    assert_eq!(s.stats().evictions, 5);
}

#[test]
fn recording_captures_every_acquire() {
    let s = ExpertStore::new(source(), &pools(2, 1), Policy::LFRU);
    drop(s.acquire(key(0, 1)).unwrap());
    s.record(true).unwrap();
    drop(s.acquire(key(1, 2)).unwrap());
    drop(s.acquire(key(0, 1)).unwrap());
    assert_eq!(s.take_trace().unwrap(), [key(1, 2), key(0, 1)]);
    assert!(s.take_trace().unwrap().is_empty());
    s.record(false).unwrap();
    drop(s.acquire(key(0, 1)).unwrap());
    assert!(s.take_trace().unwrap().is_empty());
}

/// `hold_layer` holds a fully resident layer without counting a use; `record_uses`
/// then counts the uses as hits, in the trace and for recency.
#[test]
fn holding_a_layer_is_not_a_use_and_recorded_uses_are() {
    let s = ExpertStore::new(source(), &pools(2, 4), Policy::Lru);
    assert!(matches!(
        s.hold_layer(LayerIdx(0), 5).unwrap(),
        Held::TooSmall
    ));
    drop(s.acquire(key(0, 1)).unwrap());
    let Held::Missing(m) = s.hold_layer(LayerIdx(0), 4).unwrap() else {
        panic!("experts 0, 2 and 3 are not resident");
    };
    assert_eq!(m, [key(0, 0), key(0, 2), key(0, 3)]);
    for e in [3, 2, 0] {
        drop(s.acquire(key(0, e)).unwrap()); // LRU order 1, 3, 2, 0
    }
    let before = s.stats();
    s.record(true).unwrap();
    let Held::All(held) = s.hold_layer(LayerIdx(0), 4).unwrap() else {
        panic!("every expert is resident");
    };
    assert_eq!(held.len(), 4);
    for (e, l) in held.iter().enumerate() {
        assert_eq!(l.key(), key(0, e as u32));
        check(l.bytes(), l.key());
    }
    // Held experts survive a shrink to zero slots.
    s.set_pools(&[PoolSpec {
        layer: LayerIdx(0),
        slots: 0,
        slot_bytes: 100,
    }])
    .unwrap();
    assert_eq!(s.resident().unwrap().len(), 4);
    s.set_pools(&pools(1, 4)).unwrap();
    drop(held);
    assert_eq!(s.stats().hits, before.hits, "holding is not a hit");
    assert!(s.take_trace().unwrap().is_empty(), "nor traced");
    // Recency is unchanged by holding: 1 is still the LRU victim...
    s.record_uses(&[key(0, 3), key(1, 7)]).unwrap(); // (1, 7) is not resident: skipped
    assert_eq!(s.stats().hits, before.hits + 1);
    assert_eq!(s.take_trace().unwrap(), [key(0, 3)]);
    s.set_pools(&pools(1, 3)).unwrap();
    let mut r = s.resident().unwrap();
    r.sort();
    assert_eq!(r, [key(0, 0), key(0, 2), key(0, 3)]);
    // ...and a recorded use makes an expert most recent: order 2, 0, 3 becomes 0, 3, 2.
    s.record_uses(&[key(0, 2)]).unwrap();
    s.set_pools(&pools(1, 2)).unwrap();
    let mut r = s.resident().unwrap();
    r.sort();
    assert_eq!(r, [key(0, 2), key(0, 3)], "0 is now the least recent");
}
