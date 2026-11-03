//! The expert store (ADR-006), minimal Phase 4 form.
//!
//! The MoE layer gets expert weights **only** through [`ExpertStore::acquire`], from
//! the first token on. The store keeps experts up to a byte capacity, evicts the least
//! recently used ones that nobody holds, and loads misses through an
//! [`ExpertSource`]. Whatever the capacity (one expert or the whole model), the bytes
//! a caller sees are the same, so the output is the same (invariant 1). Only speed
//! changes.
//!
//! - [`Lease`] keeps an expert resident while it is used and releases it on drop.
//! - Two callers missing the same expert share one load: the second waits on a
//!   condition variable for the first (in-flight dedup).
//! - Loads run outside the lock, so hits proceed while an expert is being read.
//! - When every resident byte is leased and the next expert does not fit,
//!   [`ExpertStore::acquire`] returns [`StoreError::Full`] instead of waiting. A single
//!   generating thread that waited for its own leases would deadlock.
//!
//! Expert bytes live in `Arc<[u8]>`, so the store needs no `unsafe` code. Preallocated
//! slabs, I/O threads, `O_DIRECT` and prefetch come in Phase 6. Lease, dedup and
//! eviction are model-checked with `loom` (`RUSTFLAGS="--cfg loom"`).

use std::collections::HashMap;
use std::fmt;

use qubivex_core::{ExpertIdx, LayerIdx};
use thiserror::Error;

/// The synchronisation the store's logic depends on, swapped for loom's model
/// checker under `cfg(loom)`. Expert bytes are immutable once published and live in
/// a plain `std::sync::Arc`: there is nothing for loom to check there.
mod sync {
    #[cfg(loom)]
    pub(crate) use loom::sync::atomic::{AtomicU64, Ordering};
    #[cfg(loom)]
    pub(crate) use loom::sync::{Condvar, Mutex, MutexGuard};
    #[cfg(not(loom))]
    pub(crate) use std::sync::atomic::{AtomicU64, Ordering};
    #[cfg(not(loom))]
    pub(crate) use std::sync::{Condvar, Mutex, MutexGuard};
}

use std::sync::Arc;

use sync::{AtomicU64, Condvar, Mutex, MutexGuard, Ordering};

/// An expert: its layer and index.
pub type ExpertKey = (LayerIdx, ExpertIdx);

/// Store errors.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StoreError {
    /// The source failed to report or read an expert.
    #[error("loading expert (layer {}, expert {}): {detail}", key.0.0, key.1.0)]
    Load {
        /// The expert.
        key: ExpertKey,
        /// What went wrong.
        detail: String,
    },
    /// One expert is larger than the whole capacity.
    #[error("expert (layer {}, expert {}) is {size} bytes, larger than the {capacity}-byte capacity", key.0.0, key.1.0)]
    TooLarge {
        /// The expert.
        key: ExpertKey,
        /// Its size.
        size: u64,
        /// The capacity.
        capacity: u64,
    },
    /// Every resident byte is leased; the expert does not fit until leases are released.
    #[error("expert store is full: {leased} of {capacity} bytes are leased, {needed} more needed")]
    Full {
        /// Bytes held by leases (and loads in flight).
        leased: u64,
        /// The capacity.
        capacity: u64,
        /// The expert's size.
        needed: u64,
    },
    /// Another thread panicked while holding the store's lock.
    #[error("expert store lock poisoned")]
    Poisoned,
}

/// Where experts come from: knows their exact sizes and how to read them.
pub trait ExpertSource: Send + Sync {
    /// The exact byte size of an expert (all its parts).
    ///
    /// # Errors
    /// [`StoreError::Load`] for an expert the model does not have.
    fn expert_size(&self, key: ExpertKey) -> Result<u64, StoreError>;

    /// Reads an expert into `dst`, whose length is [`ExpertSource::expert_size`].
    ///
    /// # Errors
    /// [`StoreError::Load`] if reading fails.
    fn read_expert(&self, key: ExpertKey, dst: &mut [u8]) -> Result<(), StoreError>;
}

enum Slot {
    /// A load is in flight; its bytes are already reserved.
    Loading,
    Ready {
        data: Arc<[u8]>,
        leases: u32,
        last_use: u64,
    },
}

struct Inner {
    slots: HashMap<ExpertKey, Slot>,
    /// Bytes of ready and loading slots.
    used: u64,
    /// Monotonic use counter for LRU order.
    tick: u64,
    /// The byte capacity. The memory planner lowers it as the KV cache grows.
    capacity: u64,
}

/// Counters, for diagnostics and the capacity-sweep tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoreStats {
    /// Acquires served from memory.
    pub hits: u64,
    /// Acquires that loaded the expert.
    pub loads: u64,
    /// Acquires that waited for another caller's load.
    pub dedup_waits: u64,
    /// Experts evicted.
    pub evictions: u64,
    /// Bytes read from the source.
    pub bytes_read: u64,
}

/// The expert store. Share it by reference (it is `Sync`); leases borrow it.
pub struct ExpertStore {
    source: Arc<dyn ExpertSource>,
    inner: Mutex<Inner>,
    /// Signalled when a load finishes (successfully or not).
    loaded: Condvar,
    hits: AtomicU64,
    loads: AtomicU64,
    dedup_waits: AtomicU64,
    evictions: AtomicU64,
    bytes_read: AtomicU64,
}

impl fmt::Debug for ExpertStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExpertStore")
            .field("capacity", &self.capacity())
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

/// A held expert: its bytes stay resident until the lease is dropped. Not `Clone`.
pub struct Lease<'a> {
    store: &'a ExpertStore,
    key: ExpertKey,
    data: Arc<[u8]>,
}

impl fmt::Debug for Lease<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lease")
            .field("key", &self.key)
            .field("bytes", &self.data.len())
            .finish()
    }
}

impl Lease<'_> {
    /// The expert's bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// Which expert.
    #[must_use]
    pub const fn key(&self) -> ExpertKey {
        self.key
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        // A poisoned lock means another thread panicked; the slot's lease count no
        // longer matters then, and Drop must not panic.
        if let Ok(mut g) = self.store.inner.lock() {
            g.tick += 1;
            let tick = g.tick;
            if let Some(Slot::Ready {
                leases, last_use, ..
            }) = g.slots.get_mut(&self.key)
            {
                *leases = leases.saturating_sub(1);
                *last_use = tick;
            }
        }
    }
}

impl ExpertStore {
    /// A store holding at most `capacity` bytes of experts.
    #[must_use]
    pub fn new(source: Arc<dyn ExpertSource>, capacity: u64) -> Self {
        Self {
            source,
            inner: Mutex::new(Inner {
                slots: HashMap::new(),
                used: 0,
                tick: 0,
                capacity,
            }),
            loaded: Condvar::new(),
            hits: AtomicU64::new(0),
            loads: AtomicU64::new(0),
            dedup_waits: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
        }
    }

    /// The byte capacity (0 if the lock is poisoned).
    #[must_use]
    pub fn capacity(&self) -> u64 {
        self.lock().map_or(0, |g| g.capacity)
    }

    /// Changes the byte capacity, evicting least-recently-used unleased experts until
    /// the resident bytes fit. Leased experts stay until released; the next miss
    /// evicts them. Returns the bytes resident afterwards.
    ///
    /// The memory planner calls this as the KV cache grows (Colibri's `kv_room_fit`):
    /// the two share one room.
    ///
    /// # Errors
    /// [`StoreError::Poisoned`].
    pub fn set_capacity(&self, capacity: u64) -> Result<u64, StoreError> {
        let mut g = self.lock()?;
        g.capacity = capacity;
        while g.used > capacity {
            if !self.evict_one(&mut g) {
                break;
            }
        }
        Ok(g.used)
    }

    /// Counters so far.
    #[must_use]
    pub fn stats(&self) -> StoreStats {
        // Statistics only: Relaxed (RUST_GUIDELINES §4).
        StoreStats {
            hits: self.hits.load(Ordering::Relaxed),
            loads: self.loads.load(Ordering::Relaxed),
            dedup_waits: self.dedup_waits.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
        }
    }

    /// Bytes resident or loading.
    ///
    /// # Errors
    /// [`StoreError::Poisoned`].
    pub fn used(&self) -> Result<u64, StoreError> {
        Ok(self.lock()?.used)
    }

    fn lock(&self) -> Result<MutexGuard<'_, Inner>, StoreError> {
        self.inner.lock().map_err(|_| StoreError::Poisoned)
    }

    /// Holds an expert, loading it if needed.
    ///
    /// # Errors
    /// [`StoreError::TooLarge`], [`StoreError::Full`], [`StoreError::Load`] or
    /// [`StoreError::Poisoned`].
    pub fn acquire(&self, key: ExpertKey) -> Result<Lease<'_>, StoreError> {
        let mut g = self.lock()?;
        let mut waited = false;
        loop {
            g.tick += 1;
            let tick = g.tick;
            match g.slots.get_mut(&key) {
                Some(Slot::Ready {
                    data,
                    leases,
                    last_use,
                }) => {
                    *leases += 1;
                    *last_use = tick;
                    let data = Arc::clone(data);
                    if !waited {
                        self.hits.fetch_add(1, Ordering::Relaxed);
                    }
                    return Ok(Lease {
                        store: self,
                        key,
                        data,
                    });
                }
                Some(Slot::Loading) => {
                    if !waited {
                        self.dedup_waits.fetch_add(1, Ordering::Relaxed);
                        waited = true;
                    }
                    g = self.loaded.wait(g).map_err(|_| StoreError::Poisoned)?;
                }
                None => break,
            }
        }
        // A miss: reserve room, mark it loading, and read outside the lock.
        let size = self.source.expert_size(key)?;
        let capacity = g.capacity;
        if size > capacity {
            return Err(StoreError::TooLarge {
                key,
                size,
                capacity,
            });
        }
        self.make_room(&mut g, size)?;
        g.used += size;
        g.slots.insert(key, Slot::Loading);
        drop(g);

        let mut buf = vec![
            0u8;
            usize::try_from(size).map_err(|_| StoreError::TooLarge {
                key,
                size,
                capacity,
            })?
        ];
        let read = self.source.read_expert(key, &mut buf);

        let mut g = self.lock()?;
        match read {
            Ok(()) => {
                g.tick += 1;
                let data: Arc<[u8]> = buf.into();
                let last_use = g.tick;
                g.slots.insert(
                    key,
                    Slot::Ready {
                        data: Arc::clone(&data),
                        leases: 1,
                        last_use,
                    },
                );
                self.loads.fetch_add(1, Ordering::Relaxed);
                self.bytes_read.fetch_add(size, Ordering::Relaxed);
                drop(g);
                self.loaded.notify_all();
                Ok(Lease {
                    store: self,
                    key,
                    data,
                })
            }
            Err(e) => {
                g.slots.remove(&key);
                g.used -= size;
                drop(g);
                // Waiters retry and see no slot: they load it themselves (and report
                // their own error if the source is broken).
                self.loaded.notify_all();
                Err(e)
            }
        }
    }

    /// Evicts least-recently-used unleased experts until `size` more bytes fit.
    fn make_room(&self, g: &mut Inner, size: u64) -> Result<(), StoreError> {
        while g.used + size > g.capacity {
            if !self.evict_one(g) {
                return Err(StoreError::Full {
                    leased: g.used,
                    capacity: g.capacity,
                    needed: size,
                });
            }
        }
        Ok(())
    }

    /// Evicts the least-recently-used unleased expert; `false` if there is none.
    fn evict_one(&self, g: &mut Inner) -> bool {
        let victim = g
            .slots
            .iter()
            .filter_map(|(k, s)| match s {
                Slot::Ready {
                    leases: 0,
                    last_use,
                    data,
                } => Some((*last_use, *k, data.len() as u64)),
                _ => None,
            })
            .min();
        let Some((_, k, bytes)) = victim else {
            return false;
        };
        g.slots.remove(&k);
        g.used -= bytes;
        self.evictions.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Experts currently resident (ready), in no particular order. Diagnostics only.
    ///
    /// # Errors
    /// [`StoreError::Poisoned`].
    pub fn resident(&self) -> Result<Vec<ExpertKey>, StoreError> {
        Ok(self
            .lock()?
            .slots
            .iter()
            .filter(|(_, s)| matches!(s, Slot::Ready { .. }))
            .map(|(k, _)| *k)
            .collect())
    }
}
