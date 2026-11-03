//! Model checks of lease, in-flight dedup and eviction. Run with
//! `RUSTFLAGS="--cfg loom" cargo test -p qubivex-expert-stream --test loom --release`.

#![cfg(loom)]
#![allow(clippy::unwrap_used)]

use loom::sync::Arc;
use loom::sync::atomic::{AtomicUsize, Ordering};

/// The source is shared immutable state: a plain std `Arc`.
type SrcArc<T> = std::sync::Arc<T>;

use qubivex_core::{ExpertIdx, LayerIdx};
use qubivex_expert_stream::{ExpertKey, ExpertSource, ExpertStore, Policy, PoolSpec, StoreError};

struct Counting(AtomicUsize);

impl ExpertSource for Counting {
    fn expert_size(&self, _: ExpertKey) -> Result<u64, StoreError> {
        Ok(4)
    }
    fn read_expert(&self, k: ExpertKey, dst: &mut [u8]) -> Result<(), StoreError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        dst.fill(u8::try_from(k.1.0).unwrap());
        Ok(())
    }
}

fn key(e: u32) -> ExpertKey {
    (LayerIdx(0), ExpertIdx(e))
}

fn pool(slots: u32) -> [PoolSpec; 1] {
    [PoolSpec {
        layer: LayerIdx(0),
        slots,
        slot_bytes: 4,
    }]
}

#[test]
fn concurrent_misses_of_one_expert_load_it_once() {
    loom::model(|| {
        let src = SrcArc::new(Counting(AtomicUsize::new(0)));
        let store = Arc::new(ExpertStore::new(src.clone(), &pool(2), Policy::LFRU));
        let s2 = store.clone();
        let t = loom::thread::spawn(move || {
            let l = s2.acquire(key(1)).unwrap();
            assert_eq!(l.bytes(), &[1, 1, 1, 1]);
        });
        let l = store.acquire(key(1)).unwrap();
        assert_eq!(l.bytes(), &[1, 1, 1, 1]);
        drop(l);
        t.join().unwrap();
        assert_eq!(src.0.load(Ordering::Relaxed), 1);
        assert_eq!(store.used().unwrap(), 4);
    });
}

#[test]
fn eviction_never_takes_a_leased_expert() {
    loom::model(|| {
        let src = SrcArc::new(Counting(AtomicUsize::new(0)));
        // Room for one expert only.
        let store = Arc::new(ExpertStore::new(src, &pool(1), Policy::LFRU));
        let s2 = store.clone();
        let t = loom::thread::spawn(move || {
            if let Ok(l) = s2.acquire(key(2)) {
                assert_eq!(l.bytes(), &[2, 2, 2, 2]);
            }
        });
        if let Ok(l) = store.acquire(key(3)) {
            assert_eq!(l.bytes(), &[3, 3, 3, 3]);
        }
        t.join().unwrap();
        assert!(store.used().unwrap() <= 4);
    });
}
