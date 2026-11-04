//! The expert store (ADR-006, ADR-008).
//!
//! The MoE layer gets expert weights **only** through [`ExpertStore::acquire`] and
//! [`ExpertStore::acquire_many`], from the first token on. Each expert layer has its
//! own pool of slots ([`PoolSpec`]); when a layer's slots are full, a miss evicts an
//! unleased expert *of the same layer* chosen by the [`Policy`] (LRU, or LFRU with
//! decaying heat), and reuses its buffer as the slab for the new expert. Whatever the
//! slot counts (one per layer or every expert) and however many I/O threads read, the
//! bytes a caller sees are the same, so the output is the same (invariant 1). Only
//! speed changes.
//!
//! Per-layer pools fix the failure of one global LRU under a small budget: a decode
//! step walks the layers in order, so a global LRU smaller than one step's working set
//! evicts every expert before the next token needs it again (0 hits on the reference
//! model at a 1 GiB budget; ADR-008 amendment 1). A pool only competes with its own
//! layer.
//!
//! - [`Lease`] keeps an expert resident while it is used and releases it on drop.
//! - [`ExpertStore::acquire_many`] leases a layer's hits at once, queues every miss
//!   that has a free slot before waiting on any, and hands out leases as loads finish
//!   ([`Batch`]). Misses beyond the free slots are issued as earlier leases are
//!   released, so a layer with fewer slots than experts still pipelines.
//! - Loads go through a priority queue (Demand > Prefetch > Warm, FIFO within a
//!   priority) served by a fixed pool of I/O threads ([`ExpertStore::with_io_threads`]).
//!   With no I/O threads, a waiting caller performs queued reads itself. The queue is
//!   bounded by the slots: every queued load holds one.
//! - Two callers missing the same expert share one load (in-flight dedup); a read
//!   error reaches every waiter.
//! - Dropping a [`Batch`] cancels its loads that no thread has started. A started read
//!   always completes.
//! - When every slot of a layer is leased, the miss is [`StoreError::Full`] instead of
//!   a wait. A single generating thread that waited for its own leases would deadlock.
//! - Slabs: an evicted expert's buffer is reused for the next load of its layer, so a
//!   streaming decode allocates nothing after the pools fill. Buffers are freed only
//!   when a pool shrinks.
//!
//! The store has no `unsafe` code. Lease, dedup and eviction are model-checked with
//! `loom` (`RUSTFLAGS="--cfg loom"`), without I/O threads.

pub mod fault;
pub mod policy;
pub mod predict;
pub mod replay;
mod slab;

pub use slab::{Attachment, Slab};

use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::fmt;

use qubivex_compute::host::PageAligned;
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
use std::time::Instant;

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

/// How urgently a load is wanted. The I/O threads serve higher priorities first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Filling the cache ahead of use (a learned hot set).
    Warm,
    /// A predicted expert (Phase 7).
    Prefetch,
    /// An expert the forward pass is waiting for.
    Demand,
}

/// One layer's pool: how many experts it may hold, and the slot size (its largest
/// expert).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolSpec {
    /// The layer.
    pub layer: LayerIdx,
    /// Experts the pool may hold at once (resident, loading or failed but unseen).
    pub slots: u32,
    /// Bytes of one slot.
    pub slot_bytes: u64,
}

enum Entry {
    /// A load is queued or in flight; its bytes are already counted. `waiters` are the
    /// acquirers that will each take one lease when it is ready. `spec`: issued by
    /// [`ExpertStore::prefetch`] and not demanded yet (when it was issued).
    Loading {
        bytes: u64,
        waiters: u32,
        spec: Option<Instant>,
    },
    Ready {
        data: Arc<Slab>,
        leases: u32,
        last_use: u64,
        spec: Option<Instant>,
    },
    /// The read failed; each waiter takes the error, and the last removes the entry.
    Failed { detail: String, waiters: u32 },
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
        self.victim_except(policy, &[])
    }

    /// As [`Pool::victim`], never choosing an expert of `keep`.
    fn victim_except(&self, policy: Policy, keep: &[ExpertKey]) -> Option<u32> {
        let ready = self.entries.iter().filter_map(|(&e, s)| match s {
            Entry::Ready {
                leases: 0,
                last_use,
                ..
            } if !keep.iter().any(|k| k.1.0 == e) => Some((e, *last_use)),
            _ => None,
        });
        match policy {
            Policy::Lru => ready.min_by_key(|&(e, t)| (t, e)).map(|(e, _)| e),
            Policy::Lfru { .. } => ready
                .min_by_key(|&(e, t)| (policy::lfru_score(self.heat.get(e), self.tick - t), t, e))
                .map(|(e, _)| e),
        }
    }

    /// Whether a prefetch nobody waits for is loading: its slot frees up when it lands.
    fn speculating(&self) -> bool {
        self.entries.values().any(|s| {
            matches!(
                s,
                Entry::Loading {
                    waiters: 0,
                    spec: Some(_),
                    ..
                }
            )
        })
    }

    /// Removes an expert, counting a prefetch that was never used as wasted. Returns
    /// its buffer if it was ready.
    fn evict(&mut self, e: u32, pf: &mut PrefetchStats) -> Option<Arc<Slab>> {
        match self.entries.remove(&e) {
            Some(Entry::Ready { data, spec, .. }) => {
                if spec.is_some() {
                    pf.wasted += 1;
                    pf.wasted_bytes += data.len() as u64;
                }
                Some(data)
            }
            _ => None,
        }
    }

    /// Slots held by leases, loads and unseen failures.
    fn busy(&self) -> u32 {
        let n = self
            .entries
            .values()
            .filter(|s| !matches!(s, Entry::Ready { leases: 0, .. }))
            .count();
        u32::try_from(n).unwrap_or(u32::MAX)
    }
}

/// A queued read.
struct Request {
    key: ExpertKey,
    size: u64,
    /// An evicted expert's buffer to read into, if the miss evicted one.
    slab: Option<PageAligned>,
    priority: Priority,
    seq: u64,
}

impl PartialEq for Request {
    fn eq(&self, o: &Self) -> bool {
        (self.priority, self.seq) == (o.priority, o.seq)
    }
}
impl Eq for Request {}
impl PartialOrd for Request {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Request {
    /// The heap's maximum is served first: the highest priority, then the oldest.
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.priority
            .cmp(&o.priority)
            .then_with(|| o.seq.cmp(&self.seq))
    }
}

struct Inner {
    pools: HashMap<LayerIdx, Pool>,
    /// Bytes of ready and loading experts.
    used: u64,
    policy: Policy,
    /// Every acquire, in order, while recording (for trace replay).
    trace: Option<Vec<ExpertKey>>,
    queue: BinaryHeap<Request>,
    seq: u64,
    shutdown: bool,
    prefetch: PrefetchStats,
}

impl Inner {
    /// Drops queued prefetches of `layer` that nobody waits for and that `keep` does
    /// not name; promotes those it names to [`Priority::Demand`]. Returns how many
    /// were dropped. A read already started is left to finish.
    fn settle_prefetches(&mut self, layer: LayerIdx, keep: &[ExpertKey]) -> u64 {
        let mut dropped = 0;
        let mut q = std::mem::take(&mut self.queue).into_vec();
        q.retain_mut(|r| {
            if r.key.0 != layer || r.priority != Priority::Prefetch {
                return true;
            }
            if keep.contains(&r.key) {
                r.priority = Priority::Demand;
                return true;
            }
            let Some(pool) = self.pools.get_mut(&layer) else {
                return true;
            };
            match pool.entries.get(&r.key.1.0) {
                Some(&Entry::Loading {
                    bytes, waiters: 0, ..
                }) => {
                    pool.entries.remove(&r.key.1.0);
                    self.used -= bytes;
                    dropped += 1;
                    false
                }
                _ => true,
            }
        });
        self.queue = q.into();
        self.prefetch.cancelled += dropped;
        dropped
    }
}

/// What prefetching did, for the Phase 7 metrics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefetchStats {
    /// Predicted experts queued for reading.
    pub issued: u64,
    /// Predicted experts already resident or loading.
    pub present: u64,
    /// Predicted experts not read: no free slot, the eviction guard kept the resident
    /// expert, or no I/O threads.
    pub refused: u64,
    /// Prefetched experts demanded after their read finished: the whole read was
    /// hidden.
    pub ready: u64,
    /// Prefetched experts demanded while still loading: part of the read was hidden.
    pub late: u64,
    /// Prefetched experts evicted without being demanded.
    pub wasted: u64,
    /// Bytes of those.
    pub wasted_bytes: u64,
    /// Queued prefetches dropped before any read started.
    pub cancelled: u64,
    /// Nanoseconds from issue to demand, summed over `ready` and `late`.
    pub lead_ns: u64,
}

/// What one [`ExpertStore::prefetch`] call did with each key.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefetchReceipt {
    /// Queued for reading.
    pub issued: u32,
    /// Already resident or loading.
    pub present: u32,
    /// Not read (no free slot, the guard, no I/O threads, or an unknown expert).
    pub refused: u32,
}

/// Counters, for diagnostics and the capacity-sweep tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoreStats {
    /// Acquires served from memory.
    pub hits: u64,
    /// Reads completed.
    pub loads: u64,
    /// Acquires that joined a load already queued or in flight.
    pub dedup_waits: u64,
    /// Experts evicted.
    pub evictions: u64,
    /// Bytes read from the source.
    pub bytes_read: u64,
    /// Slab buffers allocated (a load that reused an evicted buffer allocates none).
    pub slab_allocs: u64,
    /// Queued loads cancelled before any thread started them.
    pub cancelled: u64,
    /// Nanoseconds callers spent blocked waiting for loads.
    pub wait_ns: u64,
    /// Prefetch counters.
    pub prefetch: PrefetchStats,
}

struct Shared {
    source: Arc<dyn ExpertSource>,
    inner: Mutex<Inner>,
    /// Signalled when a load finishes (successfully or not).
    loaded: Condvar,
    /// Signalled when a request is queued, or at shutdown.
    work: Condvar,
    io_threads: usize,
    hits: AtomicU64,
    loads: AtomicU64,
    dedup_waits: AtomicU64,
    evictions: AtomicU64,
    bytes_read: AtomicU64,
    slab_allocs: AtomicU64,
    cancelled: AtomicU64,
    wait_ns: AtomicU64,
}

/// The expert store. Share it by reference (it is `Sync`); leases borrow it.
pub struct ExpertStore {
    shared: Arc<Shared>,
    #[cfg(not(loom))]
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl fmt::Debug for ExpertStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExpertStore")
            .field("capacity", &self.capacity())
            .field("io_threads", &self.shared.io_threads)
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

/// A held expert: its bytes stay resident until the lease is dropped. Not `Clone`.
pub struct Lease<'a> {
    store: &'a ExpertStore,
    key: ExpertKey,
    data: Arc<Slab>,
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
        self.data.bytes()
    }

    /// The slab holding the bytes (page-aligned, with a device attachment slot).
    #[must_use]
    pub fn slab(&self) -> &Slab {
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
        if let Ok(mut g) = self.store.shared.inner.lock() {
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

/// What claiming one expert of a batch gave.
enum Claim {
    Leased(Arc<Slab>),
    /// A load is queued or in flight and this caller is one of its waiters.
    Pending,
    /// No slot of the layer is free now.
    Deferred,
}

/// Leases for a set of experts, handed out as they become ready
/// ([`ExpertStore::acquire_many`]). Dropping it releases unclaimed leases and cancels
/// its queued loads.
pub struct Batch<'a> {
    store: &'a ExpertStore,
    ready: VecDeque<Lease<'a>>,
    pending: Vec<ExpertKey>,
    deferred: Vec<ExpertKey>,
}

impl fmt::Debug for Batch<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Batch")
            .field("ready", &self.ready.len())
            .field("pending", &self.pending)
            .field("deferred", &self.deferred)
            .finish()
    }
}

impl<'a> Iterator for Batch<'a> {
    type Item = Result<Lease<'a>, StoreError>;

    /// The next expert of the batch that is ready, waiting for a load if none is.
    /// Leases come in no particular order: hits first, then loads as they finish.
    /// `None` once every expert has been handed out.
    ///
    /// Release each lease before asking for the next when the layer has fewer slots
    /// than the batch has experts: the deferred ones need those slots.
    ///
    /// # Errors
    /// A read error for one expert ([`StoreError::Load`]; the batch continues with
    /// the others), [`StoreError::Full`] if the deferred experts can get no slot
    /// because other callers hold them all, or [`StoreError::Poisoned`].
    fn next(&mut self) -> Option<Self::Item> {
        let shared = &self.store.shared;
        let mut g = match shared.inner.lock() {
            Ok(g) => g,
            Err(_) => return Some(Err(StoreError::Poisoned)),
        };
        let mut waited: Option<std::time::Instant> = None;
        let out = loop {
            // Issue what can be issued now: released leases may have freed slots. Without
            // I/O threads nothing reads ahead, so a miss is claimed only when the caller
            // has nothing else to take (measured: reading ahead on the caller's own
            // thread was 15% slower).
            let mut err = None;
            let mut budget = if shared.io_threads > 0 {
                usize::MAX
            } else {
                usize::from(self.ready.is_empty() && self.pending.is_empty())
            };
            let deferred = std::mem::take(&mut self.deferred);
            for k in deferred {
                if err.is_some() || budget == 0 {
                    self.deferred.push(k);
                    continue;
                }
                match self.store.claim(&mut g, k, Priority::Demand) {
                    Ok(Claim::Leased(data)) => {
                        budget -= 1;
                        self.ready.push_back(self.store.lease(k, data));
                    }
                    Ok(Claim::Pending) => {
                        budget -= 1;
                        self.pending.push(k);
                    }
                    Ok(Claim::Deferred) => self.deferred.push(k),
                    Err(e) => err = Some(e),
                }
            }
            if let Some(e) = err {
                break Some(Err(e));
            }
            if let Some(l) = self.ready.pop_front() {
                break Some(Ok(l));
            }
            // A finished load of ours?
            if let Some(r) = self.take_finished(&mut g) {
                break Some(r);
            }
            // A deferred expert whose layer is full of prefetches still loading gets a
            // slot when they land: wait for them instead of failing.
            let speculating = self
                .deferred
                .iter()
                .any(|k| g.pools.get(&k.0).is_some_and(Pool::speculating));
            if self.pending.is_empty() && !speculating {
                break self.deferred.first().map(|&key| {
                    let (leased, slots) =
                        g.pools.get(&key.0).map_or((0, 0), |p| (p.busy(), p.slots));
                    Err(StoreError::Full { key, leased, slots })
                });
            }
            waited.get_or_insert_with(std::time::Instant::now);
            // Nothing ready: read something ourselves without I/O threads, else wait.
            if shared.io_threads == 0 {
                if let Some(req) = g.queue.pop() {
                    drop(g);
                    self.store.execute(req);
                    g = match shared.inner.lock() {
                        Ok(g) => g,
                        Err(_) => break Some(Err(StoreError::Poisoned)),
                    };
                    continue;
                }
            }
            g = match shared.loaded.wait(g) {
                Ok(g) => g,
                Err(_) => break Some(Err(StoreError::Poisoned)),
            };
        };
        if let Some(t) = waited {
            let ns = u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX);
            shared.wait_ns.fetch_add(ns, Ordering::Relaxed);
        }
        out
    }
}

impl<'a> Batch<'a> {
    /// Takes a pending expert whose load has finished: a lease (already counted for
    /// this waiter) or its error.
    fn take_finished(&mut self, g: &mut Inner) -> Option<Result<Lease<'a>, StoreError>> {
        for i in 0..self.pending.len() {
            let key = self.pending[i];
            let Some(pool) = g.pools.get_mut(&key.0) else {
                self.pending.swap_remove(i);
                return Some(Err(StoreError::NoPool(key.0)));
            };
            match pool.entries.get_mut(&key.1.0) {
                Some(Entry::Loading { .. }) => {}
                Some(Entry::Ready { data, .. }) => {
                    let data = Arc::clone(data);
                    self.pending.swap_remove(i);
                    return Some(Ok(self.store.lease(key, data)));
                }
                Some(Entry::Failed { detail, waiters }) => {
                    let detail = detail.clone();
                    *waiters -= 1;
                    if *waiters == 0 {
                        pool.entries.remove(&key.1.0);
                    }
                    self.pending.swap_remove(i);
                    return Some(Err(StoreError::Load { key, detail }));
                }
                None => {
                    // Cannot happen while we are a waiter; recover by asking again.
                    self.pending.swap_remove(i);
                    self.deferred.push(key);
                    return None;
                }
            }
        }
        None
    }
}

impl Drop for Batch<'_> {
    fn drop(&mut self) {
        // Leases take the lock when dropped: release them before taking it here.
        self.ready.clear();
        if self.pending.is_empty() {
            return;
        }
        let Ok(mut g) = self.store.shared.inner.lock() else {
            return;
        };
        let inner = &mut *g;
        let mut cancelled = 0;
        for &(layer, ExpertIdx(e)) in &self.pending {
            let Some(pool) = inner.pools.get_mut(&layer) else {
                continue;
            };
            match pool.entries.get_mut(&e) {
                Some(Entry::Loading { bytes, waiters, .. }) => {
                    *waiters -= 1;
                    let bytes = *bytes;
                    let queued = inner.queue.iter().any(|r| r.key == (layer, ExpertIdx(e)));
                    if *waiters == 0 && queued {
                        pool.entries.remove(&e);
                        inner.used -= bytes;
                        inner.queue.retain(|r| r.key != (layer, ExpertIdx(e)));
                        cancelled += 1;
                    }
                }
                Some(Entry::Ready { leases, .. }) => *leases = leases.saturating_sub(1),
                Some(Entry::Failed { waiters, .. }) => {
                    *waiters -= 1;
                    if *waiters == 0 {
                        pool.entries.remove(&e);
                    }
                }
                None => {}
            }
        }
        self.store
            .shared
            .cancelled
            .fetch_add(cancelled, Ordering::Relaxed);
    }
}

impl ExpertStore {
    /// A store with one pool per spec, evicting by `policy`, without I/O threads: a
    /// caller that needs an expert reads it itself.
    #[must_use]
    pub fn new(source: Arc<dyn ExpertSource>, pools: &[PoolSpec], policy: Policy) -> Self {
        Self::build(source, pools, policy, 0)
    }

    /// A store whose reads are served by `threads` dedicated I/O threads (0: none, as
    /// [`ExpertStore::new`]). The threads stop when the store is dropped.
    ///
    /// # Errors
    /// [`StoreError::Load`]-free: only if a thread cannot be started, reported as
    /// [`std::io::Error`].
    #[cfg(not(loom))]
    pub fn with_io_threads(
        source: Arc<dyn ExpertSource>,
        pools: &[PoolSpec],
        policy: Policy,
        threads: usize,
    ) -> std::io::Result<Self> {
        let mut s = Self::build(source, pools, policy, threads);
        for i in 0..threads {
            let shared = Arc::clone(&s.shared);
            let h = std::thread::Builder::new()
                .name(format!("qubivex-io-{i}"))
                .spawn(move || worker(&shared))?;
            s.workers.push(h);
        }
        Ok(s)
    }

    fn build(
        source: Arc<dyn ExpertSource>,
        pools: &[PoolSpec],
        policy: Policy,
        io_threads: usize,
    ) -> Self {
        Self {
            shared: Arc::new(Shared {
                source,
                inner: Mutex::new(Inner {
                    pools: pools.iter().map(|p| (p.layer, Pool::new(*p))).collect(),
                    used: 0,
                    policy,
                    trace: None,
                    queue: BinaryHeap::new(),
                    seq: 0,
                    shutdown: false,
                    prefetch: PrefetchStats::default(),
                }),
                loaded: Condvar::new(),
                work: Condvar::new(),
                io_threads,
                hits: AtomicU64::new(0),
                loads: AtomicU64::new(0),
                dedup_waits: AtomicU64::new(0),
                evictions: AtomicU64::new(0),
                bytes_read: AtomicU64::new(0),
                slab_allocs: AtomicU64::new(0),
                cancelled: AtomicU64::new(0),
                wait_ns: AtomicU64::new(0),
            }),
            #[cfg(not(loom))]
            workers: Vec::new(),
        }
    }

    /// Dedicated I/O threads.
    #[must_use]
    pub fn io_threads(&self) -> usize {
        self.shared.io_threads
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
    /// freed. Leased and loading experts stay; the next miss of that layer evicts
    /// them once released. Pools not listed are unchanged. Returns the bytes resident
    /// afterwards.
    ///
    /// The memory planner calls this as the KV cache grows (Colibri's `kv_room_fit`):
    /// the two share one room.
    ///
    /// # Errors
    /// [`StoreError::Poisoned`].
    pub fn set_pools(&self, pools: &[PoolSpec]) -> Result<u64, StoreError> {
        let mut guard = self.lock()?;
        let g = &mut *guard;
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
                if let Some(data) = pool.evict(e, &mut g.prefetch) {
                    freed += data.len() as u64;
                }
                evicted += 1;
            }
        }
        g.used -= freed;
        self.shared.evictions.fetch_add(evicted, Ordering::Relaxed);
        Ok(g.used)
    }

    /// Counters so far.
    #[must_use]
    pub fn stats(&self) -> StoreStats {
        let s = &self.shared;
        // Statistics only: Relaxed (RUST_GUIDELINES §4).
        StoreStats {
            hits: s.hits.load(Ordering::Relaxed),
            loads: s.loads.load(Ordering::Relaxed),
            dedup_waits: s.dedup_waits.load(Ordering::Relaxed),
            evictions: s.evictions.load(Ordering::Relaxed),
            bytes_read: s.bytes_read.load(Ordering::Relaxed),
            slab_allocs: s.slab_allocs.load(Ordering::Relaxed),
            cancelled: s.cancelled.load(Ordering::Relaxed),
            wait_ns: s.wait_ns.load(Ordering::Relaxed),
            prefetch: self.lock().map(|g| g.prefetch).unwrap_or_default(),
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
        self.shared.inner.lock().map_err(|_| StoreError::Poisoned)
    }

    const fn lease(&self, key: ExpertKey, data: Arc<Slab>) -> Lease<'_> {
        Lease {
            store: self,
            key,
            data,
        }
    }

    /// Holds an expert, loading it if needed.
    ///
    /// # Errors
    /// [`StoreError::NoPool`], [`StoreError::TooLarge`], [`StoreError::Full`],
    /// [`StoreError::Load`] or [`StoreError::Poisoned`].
    pub fn acquire(&self, key: ExpertKey) -> Result<Lease<'_>, StoreError> {
        let mut b = self.acquire_many(&[key])?;
        b.next().unwrap_or(Err(StoreError::Poisoned))
    }

    /// Holds a set of experts (distinct keys): leases every hit now, queues every miss
    /// that has a free slot at [`Priority::Demand`] before waiting on any, and defers
    /// the rest until slots free up. Take the leases with the [`Batch`] iterator.
    ///
    /// # Errors
    /// [`StoreError::NoPool`] or [`StoreError::TooLarge`] for any key (nothing is
    /// held then), or [`StoreError::Poisoned`].
    pub fn acquire_many(&self, keys: &[ExpertKey]) -> Result<Batch<'_>, StoreError> {
        let mut g = self.lock()?;
        let policy = g.policy;
        // Queued prefetches of these layers: the demanded ones become demand reads,
        // the others are stale and give their slots back.
        let mut layers: Vec<LayerIdx> = keys.iter().map(|k| k.0).collect();
        layers.sort_unstable();
        layers.dedup();
        for layer in layers {
            g.settle_prefetches(layer, keys);
        }
        for &key in keys {
            let pool = g.pools.get_mut(&key.0).ok_or(StoreError::NoPool(key.0))?;
            pool.tick += 1;
            if let Policy::Lfru { decay_window } = policy {
                pool.heat.touch(key.1.0, decay_window);
            }
            if let Some(t) = g.trace.as_mut() {
                t.push(key);
            }
        }
        let mut batch = Batch {
            store: self,
            ready: VecDeque::new(),
            pending: Vec::new(),
            deferred: Vec::new(),
        };
        // Hits first, so that misses never evict an expert of this batch.
        let (hits, misses): (Vec<ExpertKey>, Vec<ExpertKey>) = keys.iter().partition(|k| {
            g.pools
                .get(&k.0)
                .is_some_and(|p| matches!(p.entries.get(&k.1.0), Some(Entry::Ready { .. })))
        });
        let mut err = None;
        // Without I/O threads, misses wait in `deferred` until the caller needs them.
        let ahead = self.shared.io_threads > 0;
        for k in hits.into_iter().chain(misses) {
            if err.is_some() {
                break;
            }
            let hit = g
                .pools
                .get(&k.0)
                .is_some_and(|p| matches!(p.entries.get(&k.1.0), Some(Entry::Ready { .. })));
            if !ahead && !hit {
                batch.deferred.push(k);
                continue;
            }
            match self.claim(&mut g, k, Priority::Demand) {
                Ok(Claim::Leased(d)) => batch.ready.push_back(self.lease(k, d)),
                Ok(Claim::Pending) => batch.pending.push(k),
                Ok(Claim::Deferred) => batch.deferred.push(k),
                Err(e) => err = Some(e),
            }
        }
        drop(g);
        match err {
            Some(e) => Err(e), // dropping the batch releases what it holds
            None => Ok(batch),
        }
    }

    /// Leases a ready expert, joins its load, or queues one in a free slot.
    fn claim(
        &self,
        g: &mut Inner,
        key: ExpertKey,
        priority: Priority,
    ) -> Result<Claim, StoreError> {
        let s = &self.shared;
        let (layer, ExpertIdx(e)) = key;
        let policy = g.policy;
        let pool = g.pools.get_mut(&layer).ok_or(StoreError::NoPool(layer))?;
        let now = pool.tick;
        match pool.entries.get_mut(&e) {
            Some(Entry::Ready {
                data,
                leases,
                last_use,
                spec,
            }) => {
                *leases += 1;
                *last_use = now;
                if let Some(t) = spec.take() {
                    g.prefetch.ready += 1;
                    g.prefetch.lead_ns += elapsed_ns(t);
                }
                s.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(Claim::Leased(Arc::clone(data)));
            }
            Some(Entry::Loading { waiters, spec, .. }) => {
                *waiters += 1;
                if let Some(t) = spec.take() {
                    g.prefetch.late += 1;
                    g.prefetch.lead_ns += elapsed_ns(t);
                }
                s.dedup_waits.fetch_add(1, Ordering::Relaxed);
                return Ok(Claim::Pending);
            }
            Some(Entry::Failed { detail, .. }) => {
                return Err(StoreError::Load {
                    key,
                    detail: detail.clone(),
                });
            }
            None => {}
        }
        let size = s.source.expert_size(key)?;
        if size > pool.slot_bytes {
            return Err(StoreError::TooLarge {
                key,
                size,
                capacity: pool.slot_bytes,
            });
        }
        let mut slab = None;
        let mut evicted = 0u64;
        let mut freed = 0u64;
        while pool.entries.len() >= pool.slots as usize {
            let Some(v) = pool.victim(policy) else { break };
            if let Some(data) = pool.evict(v, &mut g.prefetch) {
                freed += data.len() as u64;
                // Keep one buffer to read into; the others are freed.
                slab = Arc::try_unwrap(data)
                    .ok()
                    .and_then(Slab::into_region)
                    .or(slab);
            }
            evicted += 1;
        }
        s.evictions.fetch_add(evicted, Ordering::Relaxed);
        g.used -= freed;
        let pool = g.pools.get_mut(&layer).ok_or(StoreError::NoPool(layer))?;
        if pool.entries.len() >= pool.slots as usize {
            return Ok(Claim::Deferred);
        }
        pool.entries.insert(
            e,
            Entry::Loading {
                bytes: size,
                waiters: 1,
                spec: None,
            },
        );
        g.used += size;
        g.seq += 1;
        let seq = g.seq;
        g.queue.push(Request {
            key,
            size,
            slab,
            priority,
            seq,
        });
        s.work.notify_one();
        Ok(Claim::Pending)
    }

    /// Reads a request and publishes the result.
    fn execute(&self, req: Request) {
        execute(&self.shared, req);
    }

    /// Advisory: queues reads of experts predicted to be needed soon, at
    /// [`Priority::Prefetch`], in the order given (best first: a full layer stops
    /// accepting). Never waits, never takes a leased or loading slot, never fails a
    /// later demand: [`ExpertStore::acquire_many`] promotes a queued prefetch it
    /// demands and drops those of the same layer it does not.
    ///
    /// When the layer's pool is full, the prefetch evicts the policy's victim among the
    /// experts `keys` does not name (a prediction never evicts itself). With
    /// `guard` under [`Policy::Lfru`], only if the prediction's heat beats the victim's
    /// LFRU score by the hysteresis ([`policy::speculation_beats`]; ADR-008): a
    /// speculation must not displace an expert the layer keeps using.
    ///
    /// Without I/O threads every key is refused: nobody would read ahead.
    ///
    /// # Errors
    /// [`StoreError::Poisoned`].
    pub fn prefetch(&self, keys: &[ExpertKey], guard: bool) -> Result<PrefetchReceipt, StoreError> {
        let mut r = PrefetchReceipt::default();
        let mut guard_lock = self.lock()?;
        let g = &mut *guard_lock;
        let s = &self.shared;
        if s.io_threads == 0 {
            r.refused = u32::try_from(keys.len()).unwrap_or(u32::MAX);
            g.prefetch.refused += u64::from(r.refused);
            return Ok(r);
        }
        let policy = g.policy;
        let mut queued = false;
        let mut evicted = 0u64;
        for &key in keys {
            let (layer, ExpertIdx(e)) = key;
            let Some(pool) = g.pools.get_mut(&layer) else {
                r.refused += 1;
                continue;
            };
            if pool.entries.contains_key(&e) {
                r.present += 1;
                continue;
            }
            let size = match s.source.expert_size(key) {
                Ok(size) if size <= pool.slot_bytes => size,
                _ => {
                    r.refused += 1;
                    continue;
                }
            };
            let mut slab = None;
            if pool.entries.len() >= pool.slots as usize {
                // Never evict another expert of the same prediction.
                let same: Vec<ExpertKey> = keys.iter().filter(|k| k.0 == layer).copied().collect();
                let Some(v) = pool.victim_except(policy, &same) else {
                    r.refused += 1;
                    continue;
                };
                if guard {
                    if let Policy::Lfru { .. } = policy {
                        let age = match pool.entries.get(&v) {
                            Some(Entry::Ready { last_use, .. }) => pool.tick - last_use,
                            _ => 0,
                        };
                        let victim = policy::lfru_score(pool.heat.get(v), age);
                        if !policy::speculation_beats(pool.heat.get(e), victim) {
                            r.refused += 1;
                            continue;
                        }
                    }
                }
                if let Some(data) = pool.evict(v, &mut g.prefetch) {
                    g.used -= data.len() as u64;
                    slab = Arc::try_unwrap(data).ok().and_then(Slab::into_region);
                }
                evicted += 1;
            }
            let Some(pool) = g.pools.get_mut(&layer) else {
                continue;
            };
            pool.entries.insert(
                e,
                Entry::Loading {
                    bytes: size,
                    waiters: 0,
                    spec: Some(Instant::now()),
                },
            );
            g.used += size;
            g.seq += 1;
            g.queue.push(Request {
                key,
                size,
                slab,
                priority: Priority::Prefetch,
                seq: g.seq,
            });
            r.issued += 1;
            queued = true;
        }
        g.prefetch.issued += u64::from(r.issued);
        g.prefetch.present += u64::from(r.present);
        g.prefetch.refused += u64::from(r.refused);
        s.evictions.fetch_add(evicted, Ordering::Relaxed);
        drop(guard_lock);
        if queued {
            s.work.notify_all();
        }
        Ok(r)
    }

    /// Drops queued prefetches nobody waits for, of `layer` or of every layer (a turn
    /// ends, or memory pressure stops prefetch). Reads already started finish.
    /// Returns how many were dropped.
    ///
    /// # Errors
    /// [`StoreError::Poisoned`].
    pub fn cancel_prefetch(&self, layer: Option<LayerIdx>) -> Result<u64, StoreError> {
        let mut g = self.lock()?;
        let layers: Vec<LayerIdx> = match layer {
            Some(l) => vec![l],
            None => g.pools.keys().copied().collect(),
        };
        Ok(layers
            .into_iter()
            .map(|l| g.settle_prefetches(l, &[]))
            .sum())
    }
}

/// The message of a caught panic, if it has one.
fn panic_message(p: &(dyn std::any::Any + Send)) -> &str {
    p.downcast_ref::<&str>()
        .copied()
        .or_else(|| p.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("(no message)")
}

/// Nanoseconds since `t`, saturating.
fn elapsed_ns(t: Instant) -> u64 {
    u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Reads a request outside the lock and publishes the result to its waiters.
fn execute(s: &Shared, req: Request) {
    let Request {
        key, size, slab, ..
    } = req;
    let read = usize::try_from(size)
        .map_err(|_| StoreError::TooLarge {
            key,
            size,
            capacity: size,
        })
        .and_then(|len| {
            let buf = if let Some(b) = slab {
                b.try_resized(len)
            } else {
                s.slab_allocs.fetch_add(1, Ordering::Relaxed);
                PageAligned::try_zeroed(len)
            };
            let mut buf = buf.map_err(|e| StoreError::Load {
                key,
                detail: format!("cannot allocate a {len}-byte slab: {e}"),
            })?;
            // A panicking source must not take the I/O thread (or the caller) down
            // with the load unresolved: its waiters would wait forever. The buffer is
            // discarded, so a half-written one is never published.
            let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                s.source.read_expert(key, buf.as_mut_slice())
            }));
            match read {
                Ok(r) => r.map(|()| buf),
                Err(p) => Err(StoreError::Load {
                    key,
                    detail: format!("the expert source panicked: {}", panic_message(&*p)),
                }),
            }
        });
    let Ok(mut g) = s.inner.lock() else {
        return;
    };
    let inner = &mut *g;
    if let Some(pool) = inner.pools.get_mut(&key.0) {
        let last_use = pool.tick;
        if let Some(&Entry::Loading {
            bytes,
            waiters,
            spec,
        }) = pool.entries.get(&key.1.0)
        {
            match read {
                Ok(buf) => {
                    pool.entries.insert(
                        key.1.0,
                        Entry::Ready {
                            data: Arc::new(Slab::new(buf)),
                            leases: waiters,
                            last_use,
                            spec,
                        },
                    );
                    s.loads.fetch_add(1, Ordering::Relaxed);
                    s.bytes_read.fetch_add(size, Ordering::Relaxed);
                }
                Err(e) => {
                    inner.used -= bytes;
                    let detail = match e {
                        StoreError::Load { detail, .. } => detail,
                        other => other.to_string(),
                    };
                    if waiters == 0 {
                        pool.entries.remove(&key.1.0);
                    } else {
                        pool.entries
                            .insert(key.1.0, Entry::Failed { detail, waiters });
                    }
                }
            }
        }
    }
    drop(g);
    s.loaded.notify_all();
}

/// An I/O thread: serves the queue, highest priority first, until shutdown.
#[cfg(not(loom))]
fn worker(s: &Shared) {
    loop {
        let req = {
            let Ok(mut g) = s.inner.lock() else { return };
            loop {
                if g.shutdown {
                    return;
                }
                if let Some(r) = g.queue.pop() {
                    break r;
                }
                g = match s.work.wait(g) {
                    Ok(g) => g,
                    Err(_) => return,
                };
            }
        };
        execute(s, req);
    }
}

impl Drop for ExpertStore {
    fn drop(&mut self) {
        if let Ok(mut g) = self.shared.inner.lock() {
            g.shutdown = true;
        }
        self.shared.work.notify_all();
        #[cfg(not(loom))]
        for h in self.workers.drain(..) {
            // A panicked I/O thread has nothing left to clean up.
            let _ = h.join();
        }
    }
}

impl ExpertStore {
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

    /// What the pools hold now, by state. After every caller has dropped its leases
    /// and batches, and every queued or in-flight read has finished, `leased`,
    /// `loading`, `failed` and `queued` are all 0, whatever the reads did.
    ///
    /// # Errors
    /// [`StoreError::Poisoned`].
    pub fn occupancy(&self) -> Result<Occupancy, StoreError> {
        let g = self.lock()?;
        let mut o = Occupancy {
            queued: g.queue.len(),
            used: g.used,
            ..Occupancy::default()
        };
        for e in g.pools.values().flat_map(|p| p.entries.values()) {
            match e {
                Entry::Ready { leases: 0, .. } => o.resident += 1,
                Entry::Ready { .. } => {
                    o.resident += 1;
                    o.leased += 1;
                }
                Entry::Loading { .. } => o.loading += 1,
                Entry::Failed { .. } => o.failed += 1,
            }
        }
        Ok(o)
    }
}

/// A snapshot of the store's entries ([`ExpertStore::occupancy`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Occupancy {
    /// Experts resident (leased or not).
    pub resident: usize,
    /// Resident experts with at least one lease.
    pub leased: usize,
    /// Loads queued or in flight.
    pub loading: usize,
    /// Failed loads whose error some waiter has not taken yet.
    pub failed: usize,
    /// Requests in the I/O queue.
    pub queued: usize,
    /// Bytes resident or loading ([`ExpertStore::used`]).
    pub used: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_serves_demand_first_then_fifo() {
        let r = |priority, seq| Request {
            key: (LayerIdx(0), ExpertIdx(0)),
            size: 0,
            slab: None,
            priority,
            seq,
        };
        let mut q = BinaryHeap::new();
        q.push(r(Priority::Warm, 1));
        q.push(r(Priority::Demand, 4));
        q.push(r(Priority::Prefetch, 2));
        q.push(r(Priority::Demand, 3));
        let order: Vec<(Priority, u64)> =
            std::iter::from_fn(|| q.pop().map(|r| (r.priority, r.seq))).collect();
        assert_eq!(
            order,
            [
                (Priority::Demand, 3),
                (Priority::Demand, 4),
                (Priority::Prefetch, 2),
                (Priority::Warm, 1)
            ]
        );
    }
}
