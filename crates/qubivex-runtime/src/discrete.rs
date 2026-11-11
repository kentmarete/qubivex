//! Running on a device with its own memory (ADR-018): the dense weights' device copy
//! and the device expert cache.
//!
//! The dense arena is copied to the device once, at load. A routed expert is copied the
//! first time a forward pass uses it while it sits in the RAM cache, and the copy is
//! kept in the device expert cache, evicted least recently used when the cache is over
//! its budget. Copies are tied to the RAM cache's slabs: when the RAM cache reuses a
//! slab for another expert, the device copy of the old one is dropped first, so the
//! device cache only ever holds experts the RAM cache holds (an inclusive cache).
//!
//! Every matmul weight is found on the device by its host address. A weight with no
//! device copy is an error, not a silent CPU fallback: on the simulated device that is
//! how a missing upload is caught.
//!
//! While a forward pass computes an expert it holds a clone of the expert's buffer (a
//! pin), so eviction never frees memory in use (ADR-011 §3). If every cached expert is
//! pinned, a new copy goes over the budget rather than failing; the device's capacity
//! is the hard limit.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};

use qubivex_compute::discrete::{
    DeviceBuffer, DeviceMatMul, DeviceUsage, DeviceWeights, DiscreteDevice, TransferStats,
};
use qubivex_compute::host::PageAligned;
use qubivex_compute::{ComputeError, MatMul, WeightView};
use qubivex_core::Bytes;
use qubivex_expert_stream::{ExpertKey, Slab};
use qubivex_memory::{MemoryLedger, Owner, Tier};

/// The device expert cache's counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceCacheStats {
    /// The cache's budget in bytes.
    pub budget: u64,
    /// Bytes of expert copies held now.
    pub used: u64,
    /// Experts found on the device.
    pub hits: u64,
    /// Experts copied to the device.
    pub misses: u64,
    /// Copies evicted to make room.
    pub evictions: u64,
    /// Copies dropped because the RAM cache dropped the expert.
    pub detached: u64,
    /// Copies made while every cached expert was in use, over the budget.
    pub over_budget: u64,
}

/// What the engine reports about a discrete device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceReport {
    /// The device.
    pub device: String,
    /// Its memory.
    pub usage: DeviceUsage,
    /// Its copies.
    pub transfers: TransferStats,
    /// Its expert cache.
    pub experts: DeviceCacheStats,
}

struct Entry {
    key: ExpertKey,
    len: usize,
    buffer: DeviceBuffer,
    last_use: u64,
}

#[derive(Default)]
struct Cache {
    /// By the host address of the expert's slab.
    entries: BTreeMap<usize, Entry>,
    used: u64,
    clock: u64,
}

struct Shared {
    cache: Mutex<Cache>,
    detached: AtomicU64,
}

impl Shared {
    fn lock(&self) -> std::sync::MutexGuard<'_, Cache> {
        // Every update leaves the map and the byte count consistent.
        self.cache.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn remove(&self, addr: usize, key: ExpertKey) {
        let mut c = self.lock();
        if c.entries.get(&addr).is_some_and(|e| e.key == key) {
            let e = c.entries.remove(&addr).expect("present");
            c.used -= e.len as u64;
            self.detached.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Kept in a RAM slab's attachment: drops the device copy when the slab is reused.
struct Detach {
    addr: usize,
    key: ExpertKey,
    shared: Weak<Shared>,
}

impl Drop for Detach {
    fn drop(&mut self) {
        if let Some(s) = self.shared.upgrade() {
            s.remove(self.addr, self.key);
        }
    }
}

/// The dense weights on a discrete device, and its expert cache.
pub(crate) struct OnDevice {
    device: Arc<dyn DiscreteDevice>,
    ledger: Arc<MemoryLedger>,
    /// The dense arena's device copy, and the arena's host address range.
    dense: DeviceBuffer,
    arena: std::ops::Range<usize>,
    budget: u64,
    shared: Arc<Shared>,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    over_budget: AtomicU64,
}

impl std::fmt::Debug for OnDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnDevice")
            .field("device", &self.device.describe())
            .field("budget", &self.budget)
            .finish_non_exhaustive()
    }
}

fn not_on_device(what: &str) -> ComputeError {
    ComputeError::Device {
        backend: "sim",
        detail: format!("{what} has no device copy (a missing upload)"),
    }
}

impl OnDevice {
    /// Whether the device asks callers to group matmuls.
    pub(crate) fn groups_matmuls(&self) -> bool {
        self.device.groups_matmuls()
    }

    /// Copies `arena` (the dense weights) to `device`, charged to the ledger's
    /// `Vram(0)` tier, with an expert cache of `budget` bytes.
    pub(crate) fn new(
        device: Arc<dyn DiscreteDevice>,
        ledger: Arc<MemoryLedger>,
        arena: &PageAligned,
        budget: u64,
    ) -> Result<Self, ComputeError> {
        let host = arena.as_slice();
        let charge = ledger.charge(Tier::Vram(0), Owner::Dense, Bytes(host.len() as u64));
        let dense = device.upload(host, Some(Box::new(charge)))?;
        let start = host.as_ptr() as usize;
        Ok(Self {
            device,
            ledger,
            dense,
            arena: start..start + host.len(),
            budget,
            shared: Arc::new(Shared {
                cache: Mutex::default(),
                detached: AtomicU64::new(0),
            }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            over_budget: AtomicU64::new(0),
        })
    }

    pub(crate) fn describe(&self) -> String {
        self.device.describe()
    }

    /// The dense copy's bytes.
    pub(crate) fn dense_bytes(&self) -> u64 {
        self.dense.len() as u64
    }

    pub(crate) fn supports(&self, dtype: qubivex_core::DType) -> bool {
        self.device.supports(dtype)
    }

    /// The device copy of the expert `key` held in `slab`, copied now if the device has
    /// none. The returned buffer pins the copy: hold it while computing the expert.
    pub(crate) fn ensure(&self, key: ExpertKey, slab: &Slab) -> Result<DeviceBuffer, ComputeError> {
        let host = slab.bytes();
        let addr = host.as_ptr() as usize;
        {
            let mut c = self.shared.lock();
            c.clock += 1;
            let now = c.clock;
            if let Some(e) = c.entries.get_mut(&addr)
                && e.key == key
                && e.len == host.len()
            {
                e.last_use = now;
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(e.buffer.clone());
            }
            self.make_room(&mut c, host.len() as u64);
            // Reserved now, so concurrent copies cannot all fit into the same room.
            c.used += host.len() as u64;
        }
        // Copy outside the lock, so other experts are found while this one moves.
        let charge =
            self.ledger
                .charge(Tier::Vram(0), Owner::ExpertCache, Bytes(host.len() as u64));
        let uploaded = self.device.upload(host, Some(Box::new(charge)));
        let mut c = self.shared.lock();
        let buffer = match uploaded {
            Ok(b) => b,
            Err(e) => {
                c.used -= host.len() as u64;
                return Err(e);
            }
        };
        self.misses.fetch_add(1, Ordering::Relaxed);
        c.clock += 1;
        let now = c.clock;
        if let Some(e) = c.entries.get_mut(&addr)
            && e.key == key
        {
            // Another thread copied it first; keep theirs and give back the room.
            e.last_use = now;
            let b = e.buffer.clone();
            c.used -= host.len() as u64;
            return Ok(b);
        }
        if let Some(old) = c.entries.remove(&addr) {
            c.used -= old.len as u64;
        }
        c.entries.insert(
            addr,
            Entry {
                key,
                len: host.len(),
                buffer: buffer.clone(),
                last_use: now,
            },
        );
        drop(c);
        // A slab keeps its attachment for as long as it holds this expert; when it is
        // reused, the attachment drops and with it the device copy.
        let _ = slab.attachment().get_or_init(|| {
            Arc::new(Detach {
                addr,
                key,
                shared: Arc::downgrade(&self.shared),
            })
        });
        Ok(buffer)
    }

    /// Evicts unpinned copies, least recently used first, until `len` more bytes fit
    /// the budget or nothing more can go.
    fn make_room(&self, c: &mut Cache, len: u64) {
        while c.used + len > self.budget {
            // A copy is pinned while anyone besides the cache holds its buffer. Pins
            // are only taken under this lock, so a stale count can only overstate them.
            let victim = c
                .entries
                .iter()
                .filter(|(_, e)| !e.buffer.shared())
                .min_by_key(|(_, e)| e.last_use)
                .map(|(&a, _)| a);
            let Some(a) = victim else {
                self.over_budget.fetch_add(1, Ordering::Relaxed);
                return;
            };
            let e = c.entries.remove(&a).expect("present");
            c.used -= e.len as u64;
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The device copy of `w`, found by its host address.
    fn weights(&self, w: &WeightView<'_>) -> Result<DeviceWeights, ComputeError> {
        let p = w.bytes().as_ptr() as usize;
        let len = w.bytes().len();
        if self.arena.contains(&p) && p + len <= self.arena.end {
            return DeviceWeights::like(self.dense.clone(), p - self.arena.start, w);
        }
        let c = self.shared.lock();
        let (&base, e) = c
            .entries
            .range(..=p)
            .next_back()
            .ok_or_else(|| not_on_device("a weight"))?;
        if p + len > base + e.len {
            return Err(not_on_device("a weight"));
        }
        DeviceWeights::like(e.buffer.clone(), p - base, w)
    }

    /// The matmuls as one device submission.
    pub(crate) fn matmul_many(&self, ops: &mut [MatMul<'_>]) -> Result<(), ComputeError> {
        let ws = ops
            .iter()
            .map(|op| self.weights(&op.w))
            .collect::<Result<Vec<_>, _>>()?;
        let mut batch: Vec<DeviceMatMul<'_>> = ops
            .iter_mut()
            .zip(&ws)
            .map(|(op, w)| DeviceMatMul {
                w,
                xs: op.xs,
                s: op.s,
                ys: &mut *op.ys,
            })
            .collect();
        self.device.matmul_many(&mut batch)
    }

    pub(crate) fn report(&self) -> DeviceReport {
        DeviceReport {
            device: self.device.describe(),
            usage: self.device.usage(),
            transfers: self.device.transfers(),
            experts: DeviceCacheStats {
                budget: self.budget,
                used: self.shared.lock().used,
                hits: self.hits.load(Ordering::Relaxed),
                misses: self.misses.load(Ordering::Relaxed),
                evictions: self.evictions.load(Ordering::Relaxed),
                detached: self.shared.detached.load(Ordering::Relaxed),
                over_budget: self.over_budget.load(Ordering::Relaxed),
            },
        }
    }
}
