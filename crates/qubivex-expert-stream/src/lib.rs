//! The expert store (ADR-006, ADR-008).
//!
//! The MoE layer gets expert weights **only** through [`ExpertStore::acquire`], from
//! the first token on. Each expert layer has its own pool of slots ([`PoolSpec`]);
//! when a layer's slots are full, a miss evicts an unleased expert *of the same layer*
//! chosen by the [`Policy`] (LRU, or LFRU with decaying heat), and reuses its buffer
//! as the slab for the new expert. Whatever the slot counts (one per layer or every
//! expert), the bytes a caller sees are the same, so the output is the same
//! (invariant 1). Only speed changes.
//!
//! Per-layer pools fix the failure of one global LRU under a small budget: a decode
//! step walks the layers in order, so a global LRU smaller than one step's working set
//! evicts every expert before the next token needs it again (0 hits on the reference
//! model at a 1 GiB budget; ADR-008 amendment 1). A pool only competes with its own
//! layer.
//!
//! - [`Lease`] keeps an expert resident while it is used and releases it on drop.
//! - Two callers missing the same expert share one load: the second waits on a
//!   condition variable for the first (in-flight dedup).
//! - Loads run outside the lock, so hits proceed while an expert is being read.
//! - When every slot of a layer is leased, [`ExpertStore::acquire`] returns
//!   [`StoreError::Full`] instead of waiting. A single generating thread that waited
//!   for its own leases would deadlock.
//! - Slabs: an evicted expert's buffer is reused for the next load of its layer, so a
//!   streaming decode allocates nothing after the pools fill. Buffers are freed only
//!   when a pool shrinks.
//!
//! The store has no `unsafe` code. I/O threads, `acquire_many`, `O_DIRECT` and
//! prefetch come in later Phase 6 slices. Lease, dedup and eviction are model-checked
//! with `loom` (`RUSTFLAGS="--cfg loom"`).

pub mod policy;
pub mod replay;

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
    /// One expert is larger than its layer's slot.
    #[error("expert (layer {}, expert {}) is {size} bytes, larger than its {capacity}-byte slot", key.0.0, key.1.0)]
    TooLarge {
        /// The expert.
        key: ExpertKey,
        /// Its size.
        size: u64,
        /// The slot size.
        capacity: u64,
    },
    /// Every slot of the expert's layer is leased (or loading).
    #[error("expert pool of layer {} is full: {leased} of {slots} slots are leased", key.0.0)]
    Full {
        /// The expert that did not fit.
        key: ExpertKey,
        /// Slots held by leases and loads in flight.
        leased: u32,
        /// The layer's slots.
        slots: u32,
    },
    /// The store has no pool for the expert's layer.
    #[error("no expert pool for layer {}", .0.0)]
    NoPool(LayerIdx),
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

/// Which unleased expert of a full pool to evict (ADR-008).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Policy {
    /// The least recently used.
    Lru,
    /// The lowest [`policy::lfru_score`]: the coldest, recency breaking ties. Heat is
    /// counted per layer over every access (resident or not) and halves every
    /// `decay_window` accesses of the layer.
    Lfru {
        /// Accesses of a layer between halvings of its heat (0: never decay).
        decay_window: u32,
    },
}

impl Policy {
    /// LFRU with the default decay window.
    pub const LFRU: Self = Self::Lfru { decay_window: 256 };
}

impl Default for Policy {
    fn default() -> Self {
        Self::LFRU
    }
}

/// One layer's pool: how many experts it may hold, and the slot size (its largest
/// expert).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolSpec {
    /// The layer.
    pub layer: LayerIdx,
    /// Experts the pool may hold at once (resident or loading).
    pub slots: u32,
    /// Bytes of one slot.
    pub slot_bytes: u64,
}

enum Entry {
    /// A load is in flight; its bytes are already counted.
    Loading { bytes: u64 },
    Ready {
        data: Arc<Vec<u8>>,
        leases: u32,
        last_use: u64,
    },
}

struct Pool {
    slots: u32,
    slot_bytes: u64,
    entries: HashMap<u32, Entry>,
    heat: policy::Heat,
    /// Accesses of this layer: the pool's clock for recency.
    tick: u64,
}

impl Pool {
    fn new(spec: PoolSpec) -> Self {
        Self {
            slots: spec.slots,
            slot_bytes: spec.slot_bytes,
            entries: HashMap::new(),
            heat: policy::Heat::default(),
            tick: 0,
        }
    }

    /// The unleased ready expert to evict next, if any.
    fn victim(&self, policy: Policy) -> Option<u32> {
        let ready = self.entries.iter().filter_map(|(&e, s)| match s {
            Entry::Ready {
                leases: 0,
                last_use,
                ..
            } => Some((e, *last_use)),
            _ => None,
        });
        match policy {
            Policy::Lru => ready.min_by_key(|&(e, t)| (t, e)).map(|(e, _)| e),
            Policy::Lfru { .. } => ready
                .min_by_key(|&(e, t)| (policy::lfru_score(self.heat.get(e), self.tick - t), t, e))
                .map(|(e, _)| e),
        }
    }

    /// Slots held by leases and loads.
    fn busy(&self) -> u32 {
        let n = self
            .entries
            .values()
            .filter(|s| !matches!(s, Entry::Ready { leases: 0, .. }))
            .count();
        u32::try_from(n).unwrap_or(u32::MAX)
    }
}

struct Inner {
    pools: HashMap<LayerIdx, Pool>,
    /// Bytes of ready and loading experts.
    used: u64,
    policy: Policy,
    /// Every acquire, in order, while recording (for trace replay).
    trace: Option<Vec<ExpertKey>>,
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
    /// Slab buffers allocated (a load that reused an evicted buffer allocates none).
    pub slab_allocs: u64,
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
    slab_allocs: AtomicU64,
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
    data: Arc<Vec<u8>>,
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
            let (layer, expert) = self.key;
            if let Some(pool) = g.pools.get_mut(&layer) {
                let now = pool.tick;
                if let Some(Entry::Ready {
                    leases, last_use, ..
                }) = pool.entries.get_mut(&expert.0)
                {
                    *leases = leases.saturating_sub(1);
                    *last_use = now;
                }
            }
        }
    }
}

impl ExpertStore {
    /// A store with one pool per spec, evicting by `policy`.
    #[must_use]
    pub fn new(source: Arc<dyn ExpertSource>, pools: &[PoolSpec], policy: Policy) -> Self {
        Self {
            source,
            inner: Mutex::new(Inner {
                pools: pools.iter().map(|p| (p.layer, Pool::new(*p))).collect(),
                used: 0,
                policy,
                trace: None,
            }),
            loaded: Condvar::new(),
            hits: AtomicU64::new(0),
            loads: AtomicU64::new(0),
            dedup_waits: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            slab_allocs: AtomicU64::new(0),
        }
    }

    /// The bytes the pools may hold: `Σ slots × slot_bytes` (0 if the lock is
    /// poisoned).
    #[must_use]
    pub fn capacity(&self) -> u64 {
        self.lock().map_or(0, |g| {
            g.pools
                .values()
                .map(|p| u64::from(p.slots) * p.slot_bytes)
                .sum()
        })
    }

    /// The pools now, ordered by layer.
    ///
    /// # Errors
    /// [`StoreError::Poisoned`].
    pub fn pools(&self) -> Result<Vec<PoolSpec>, StoreError> {
        let g = self.lock()?;
        let mut v: Vec<PoolSpec> = g
            .pools
            .iter()
            .map(|(&layer, p)| PoolSpec {
                layer,
                slots: p.slots,
                slot_bytes: p.slot_bytes,
            })
            .collect();
        v.sort_by_key(|p| p.layer);
        Ok(v)
    }

    /// Resizes the listed pools (creating any that are missing), evicting unleased
    /// experts by the policy until each fits its new slot count; their buffers are
    /// freed. Leased experts stay until released; the next miss of that layer evicts
    /// them. Pools not listed are unchanged. Returns the bytes resident afterwards.
    ///
    /// The memory planner calls this as the KV cache grows (Colibri's `kv_room_fit`):
    /// the two share one room.
    ///
    /// # Errors
    /// [`StoreError::Poisoned`].
    pub fn set_pools(&self, pools: &[PoolSpec]) -> Result<u64, StoreError> {
        let mut g = self.lock()?;
        let policy = g.policy;
        let mut freed = 0u64;
        let mut evicted = 0u64;
        for spec in pools {
            let pool = g
                .pools
                .entry(spec.layer)
                .or_insert_with(|| Pool::new(*spec));
            pool.slots = spec.slots;
            pool.slot_bytes = spec.slot_bytes;
            while pool.entries.len() > pool.slots as usize {
                let Some(e) = pool.victim(policy) else { break };
                if let Some(Entry::Ready { data, .. }) = pool.entries.remove(&e) {
                    freed += data.len() as u64;
                }
                evicted += 1;
            }
        }
        g.used -= freed;
        self.evictions.fetch_add(evicted, Ordering::Relaxed);
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
            slab_allocs: self.slab_allocs.load(Ordering::Relaxed),
        }
    }

    /// Bytes resident or loading.
    ///
    /// # Errors
    /// [`StoreError::Poisoned`].
    pub fn used(&self) -> Result<u64, StoreError> {
        Ok(self.lock()?.used)
    }

    /// Starts recording every acquire (discarding any earlier recording), or stops.
    ///
    /// # Errors
    /// [`StoreError::Poisoned`].
    pub fn record(&self, on: bool) -> Result<(), StoreError> {
        self.lock()?.trace = on.then(Vec::new);
        Ok(())
    }

    /// The acquires recorded so far, in order; recording continues.
    ///
    /// # Errors
    /// [`StoreError::Poisoned`].
    pub fn take_trace(&self) -> Result<Vec<ExpertKey>, StoreError> {
        Ok(self
            .lock()?
            .trace
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default())
    }

    fn lock(&self) -> Result<MutexGuard<'_, Inner>, StoreError> {
        self.inner.lock().map_err(|_| StoreError::Poisoned)
    }

    /// Holds an expert, loading it if needed.
    ///
    /// # Errors
    /// [`StoreError::NoPool`], [`StoreError::TooLarge`], [`StoreError::Full`],
    /// [`StoreError::Load`] or [`StoreError::Poisoned`].
    #[allow(clippy::too_many_lines)]
    pub fn acquire(&self, key: ExpertKey) -> Result<Lease<'_>, StoreError> {
        let (layer, ExpertIdx(e)) = key;
        let mut g = self.lock()?;
        if let Some(t) = g.trace.as_mut() {
            t.push(key);
        }
        let policy = g.policy;
        {
            let pool = g.pools.get_mut(&layer).ok_or(StoreError::NoPool(layer))?;
            pool.tick += 1;
            if let Policy::Lfru { decay_window } = policy {
                pool.heat.touch(e, decay_window);
            }
        }
        let mut waited = false;
        loop {
            let pool = g.pools.get_mut(&layer).ok_or(StoreError::NoPool(layer))?;
            let now = pool.tick;
            match pool.entries.get_mut(&e) {
                Some(Entry::Ready {
                    data,
                    leases,
                    last_use,
                }) => {
                    *leases += 1;
                    *last_use = now;
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
                Some(Entry::Loading { .. }) => {
                    if !waited {
                        self.dedup_waits.fetch_add(1, Ordering::Relaxed);
                        waited = true;
                    }
                    g = self.loaded.wait(g).map_err(|_| StoreError::Poisoned)?;
                }
                None => break,
            }
        }

        // A miss: free a slot of this layer, mark it loading, and read outside the lock.
        let size = self.source.expert_size(key)?;
        let inner = &mut *g;
        let pool = inner
            .pools
            .get_mut(&layer)
            .ok_or(StoreError::NoPool(layer))?;
        if size > pool.slot_bytes {
            return Err(StoreError::TooLarge {
                key,
                size,
                capacity: pool.slot_bytes,
            });
        }
        let mut slab = None;
        let mut evicted = 0u64;
        while pool.entries.len() >= pool.slots as usize {
            let Some(v) = pool.victim(policy) else {
                self.evictions.fetch_add(evicted, Ordering::Relaxed);
                return Err(StoreError::Full {
                    key,
                    leased: pool.busy(),
                    slots: pool.slots,
                });
            };
            if let Some(Entry::Ready { data, .. }) = pool.entries.remove(&v) {
                inner.used -= data.len() as u64;
                // Keep one buffer to read into; the others are freed.
                slab = Arc::try_unwrap(data).ok().or(slab);
            }
            evicted += 1;
        }
        self.evictions.fetch_add(evicted, Ordering::Relaxed);
        pool.entries.insert(e, Entry::Loading { bytes: size });
        inner.used += size;
        drop(g);

        let len = usize::try_from(size).map_err(|_| StoreError::TooLarge {
            key,
            size,
            capacity: size,
        })?;
        let mut buf = match slab {
            Some(mut b) => {
                b.resize(len, 0);
                b
            }
            None => {
                self.slab_allocs.fetch_add(1, Ordering::Relaxed);
                vec![0u8; len]
            }
        };
        let read = self.source.read_expert(key, &mut buf);

        let mut g = self.lock()?;
        let inner = &mut *g;
        let pool = inner
            .pools
            .get_mut(&layer)
            .ok_or(StoreError::NoPool(layer))?;
        match read {
            Ok(()) => {
                let data = Arc::new(buf);
                let last_use = pool.tick;
                pool.entries.insert(
                    e,
                    Entry::Ready {
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
            Err(err) => {
                if let Some(Entry::Loading { bytes }) = pool.entries.remove(&e) {
                    inner.used -= bytes;
                }
                drop(g);
                // Waiters retry and see no entry: they load it themselves (and report
                // their own error if the source is broken).
                self.loaded.notify_all();
                Err(err)
            }
        }
    }

    /// Experts currently resident (ready), in no particular order. Diagnostics only.
    ///
    /// # Errors
    /// [`StoreError::Poisoned`].
    pub fn resident(&self) -> Result<Vec<ExpertKey>, StoreError> {
        Ok(self
            .lock()?
            .pools
            .iter()
            .flat_map(|(&l, p)| {
                p.entries
                    .iter()
                    .filter(|(_, s)| matches!(s, Entry::Ready { .. }))
                    .map(move |(&e, _)| (l, ExpertIdx(e)))
            })
            .collect())
    }
}
