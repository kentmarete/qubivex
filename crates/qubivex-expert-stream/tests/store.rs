//! Expert store behaviour: correctness at every capacity, leases, errors, threads.

#![cfg(not(loom))]
#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use qubivex_core::{ExpertIdx, LayerIdx};
use qubivex_expert_stream::{ExpertKey, ExpertSource, ExpertStore, StoreError};

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

#[test]
fn every_capacity_sees_the_same_bytes() {
    // A routing-like access pattern over 3 layers x 8 experts.
    let pattern: Vec<ExpertKey> = (0..200u32)
        .map(|i| key(i % 3, (i * 5 + i / 7) % 8))
        .collect();
    let distinct: std::collections::HashSet<_> = pattern.iter().collect();
    let total: u64 = distinct.iter().map(|k| size_of(**k)).sum();
    for capacity in [120, 250, 600, 1500, total] {
        let src = source();
        let store = ExpertStore::new(src.clone(), capacity);
        for &k in &pattern {
            let lease = store.acquire(k).unwrap();
            check(lease.bytes(), k);
            assert!(store.used().unwrap() <= capacity);
        }
        let st = store.stats();
        assert_eq!(
            st.hits + st.loads,
            pattern.len() as u64,
            "capacity {capacity}"
        );
        assert_eq!(st.loads, src.reads.load(Ordering::Relaxed));
        if capacity == total {
            assert_eq!(
                st.loads,
                distinct.len() as u64,
                "all-resident loads each expert once"
            );
            assert_eq!(st.evictions, 0);
        }
    }
}

#[test]
fn eviction_is_lru_and_skips_leased_experts() {
    let store = ExpertStore::new(source(), 330); // three layer-0/1 experts (100 + 110 + 110)
    let a = store.acquire(key(0, 0)).unwrap();
    drop(store.acquire(key(1, 0)).unwrap());
    drop(store.acquire(key(1, 1)).unwrap());
    // Full: 100 + 110 + 110 = 320. A layer-1 expert (110) needs an eviction; the LRU
    // unleased one is (1,0), since (0,0) is leased.
    drop(store.acquire(key(1, 2)).unwrap());
    let mut r = store.resident().unwrap();
    r.sort();
    assert_eq!(r, vec![key(0, 0), key(1, 1), key(1, 2)]);
    drop(a);
}

#[test]
fn full_and_too_large_are_errors_not_waits() {
    let store = ExpertStore::new(source(), 220);
    let a = store.acquire(key(0, 0)).unwrap(); // 100
    let b = store.acquire(key(0, 1)).unwrap(); // 100
    assert!(matches!(
        store.acquire(key(0, 2)),
        Err(StoreError::Full { needed: 100, .. })
    ));
    drop(a);
    assert!(
        store.acquire(key(0, 2)).is_ok(),
        "fits once a lease is released"
    );
    drop(b);
    let small = ExpertStore::new(source(), 50);
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
    let store = ExpertStore::new(src, 1000);
    let e = store.acquire(key(0, 3)).unwrap_err();
    assert!(e.to_string().contains("boom"));
    assert_eq!(store.used().unwrap(), 0);
    assert!(store.acquire(key(0, 4)).is_ok());
}

#[test]
fn threads_share_loads_and_respect_capacity() {
    let src = source();
    let store = ExpertStore::new(src.clone(), 700);
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
                    assert!(store.used().unwrap() <= 700);
                }
            });
        }
    });
    let st = store.stats();
    assert_eq!(st.loads, src.reads.load(Ordering::Relaxed));
    assert!(st.hits > 0);
}
