//! Running on a device with its own memory (ADR-018): the dense weights' device copy
//! and the device expert cache.
//!
//! The dense arena is copied to the device once, at load. A routed expert is copied the
//! first time a forward pass uses it while it sits in the RAM cache, and the copy is
//! kept in the device expert cache, evicted least recently used when the cache is full.
//! Copies are tied to the RAM cache's slabs: when the RAM cache reuses a slab for
//! another expert, the device copy of the old one is dropped first, so the device cache
//! only ever holds experts the RAM cache holds (an inclusive cache).
//!
//! **Fixed-slot arenas** (audit §15.5). Experts of one layer all have the same size,
//! and a model has few distinct sizes (a Q4_K_M file typically two, from Q4_K and Q6_K down
//! projections). The cache keeps one device arena per size class, divided into equal
//! slots: a copy is a write into a free slot, never an allocation, there is no
//! fragmentation, and every expert of a layer sits in one buffer, so a fused expert
//! block can read all of them in one dispatch. The slot counts follow the budget's
//! distribution over layers (the RAM cache's rule), capped by the RAM cache (the device
//! cache is inclusive, so it can never use more) and by the largest arena the device
//! can address at once. An arena is allocated on its class's first copy and charged to
//! the ledger whole. On a device without arenas, or a budget too small for one slot per
//! layer, each copy is its own allocation, as before.
//!
//! Every matmul weight is found on the device by its host address. A weight with no
//! device copy is an error, not a silent CPU fallback: on the simulated device that is
//! how a missing upload is caught.
//!
//! While a forward pass computes an expert it holds a [`Pin`] on its copy, so eviction
//! never frees or overwrites memory in use (ADR-011 §3). If every slot of a class is
//! pinned, a new copy gets its own allocation over the budget rather than failing; the
//! device's capacity is the hard limit.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use qubivex_compute::discrete::{
    DeviceBuffer, DeviceGatedFfn, DeviceMatMul, DeviceMoeAttention, DeviceMoePrelude,
    DeviceRoutedMoe, DeviceUsage, DeviceWeights, DiscreteDevice, KV_BLOCK, TransferStats,
    kv_block_words,
};
use qubivex_compute::host::PageAligned;
use qubivex_compute::{AttnSeq, ComputeError, GatedFfn, MatMul, RoutedMoe, WeightView};
use qubivex_core::Bytes;
use qubivex_expert_stream::{ExpertKey, Slab};
use qubivex_memory::{LayerExperts, MemoryLedger, Owner, Tier};

/// The device expert cache's counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceCacheStats {
    /// The cache's budget in bytes.
    pub budget: u64,
    /// Device bytes the cache occupies: its arenas, whole once allocated, and any
    /// copies with their own allocation. The ledger's VRAM expert cache charge.
    pub used: u64,
    /// Bytes of the expert copies it holds now.
    pub held: u64,
    /// Bytes of its arenas (allocated or not), and their slots.
    pub arena_bytes: u64,
    /// Slots in all arenas.
    pub arena_slots: u64,
    /// Experts found on the device.
    pub hits: u64,
    /// Experts copied to the device.
    pub misses: u64,
    /// Copies evicted to make room.
    pub evictions: u64,
    /// Copies dropped because the RAM cache dropped the expert.
    pub detached: u64,
    /// Copies made while every slot or cached expert was in use, over the budget.
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
    /// MoE layers routed on the device (Step 3b): router, routing, experts and combine
    /// in one submission.
    pub routed_layers: u64,
    /// The KV pool for device attention (audit §21).
    pub kv_pool: KvPoolStats,
}

/// The device KV pool's counters (audit §21).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvPoolStats {
    /// Bytes of the pool (allocated on first use, charged whole).
    pub bytes: u64,
    /// Whether it is allocated.
    pub allocated: bool,
    /// Blocks in all, and free now.
    pub blocks: u64,
    /// Blocks free now.
    pub free: u64,
    /// Layers whose attention ran on the device.
    pub layers: u64,
    /// Rows copied from the host cache into the pool (positions the host computed).
    pub synced_rows: u64,
    /// Layers declined to the host because the pool had no free block.
    pub full: u64,
}

/// Arena slots start at multiples of this many bytes (any storage-buffer offset
/// alignment a driver asks for divides it).
const SLOT_ALIGN: usize = 256;

/// Where a copy lives.
enum Home {
    /// Slot `slot` of arena `class`.
    Slot { class: usize, slot: usize },
    /// Its own allocation, and the pin count for it.
    Own { buffer: DeviceBuffer, pin: Arc<()> },
}

struct Entry {
    key: ExpertKey,
    len: usize,
    home: Home,
    last_use: u64,
}

/// One size class: an arena of equal slots.
struct Arena {
    /// Bytes of one slot (the class's expert size, aligned).
    stride: usize,
    slots: usize,
    /// Allocated on first use; `None` inside when the device refused it.
    buffer: OnceLock<Option<DeviceBuffer>>,
    /// One pin count per slot: held by the cache's entry users and by a copy in
    /// progress, so a slot is reused only when nothing reads or writes it.
    pins: Vec<Arc<()>>,
}

#[derive(Default)]
struct Cache {
    /// By the host address of the expert's slab.
    entries: BTreeMap<usize, Entry>,
    /// Bytes of copies held.
    held: u64,
    /// Bytes of copies with their own allocation.
    own: u64,
    clock: u64,
    /// Per class: slots without a copy.
    free: Vec<Vec<usize>>,
}

struct Shared {
    cache: Mutex<Cache>,
    detached: AtomicU64,
}

impl Shared {
    fn lock(&self) -> std::sync::MutexGuard<'_, Cache> {
        // Every update leaves the map, the free lists and the byte counts consistent.
        self.cache.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn remove(&self, addr: usize, key: ExpertKey) {
        let mut c = self.lock();
        if c.entries.get(&addr).is_some_and(|e| e.key == key) {
            let e = c.entries.remove(&addr).expect("present");
            c.forget(&e);
            self.detached.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl Cache {
    /// Gives back what a removed entry held.
    fn forget(&mut self, e: &Entry) {
        self.held -= e.len as u64;
        match e.home {
            Home::Slot { class, slot } => self.free[class].push(slot),
            Home::Own { .. } => self.own -= e.len as u64,
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

/// A forward pass's hold on an expert's device copy: while it lives, the copy is
/// neither evicted nor overwritten, and its memory is not freed.
#[must_use]
pub(crate) struct Pin {
    _buffer: DeviceBuffer,
    _pin: Arc<()>,
}

/// The free blocks of a KV pool, shared with the mirrors that hold the others.
struct KvFree {
    free: Mutex<Vec<u32>>,
}

impl KvFree {
    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<u32>> {
        // Every update pushes or pops whole block ids.
        self.free.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The device KV pool for attention on the device (audit §21): one arena of
/// [`KV_BLOCK`]-position blocks (keys and values of one layer each), allocated on first
/// use and charged whole to the ledger's KV owner on the VRAM tier. Sequences take
/// blocks as they grow and give them back when their cache is dropped. A pass that
/// cannot get a block runs attention on the host.
struct KvPool {
    /// Words of one block.
    block_words: usize,
    blocks: usize,
    /// Allocated on first use; `None` inside when the device refused it.
    buffer: OnceLock<Option<DeviceBuffer>>,
    free: Arc<KvFree>,
    layers: AtomicU64,
    synced: AtomicU64,
    full: AtomicU64,
}

/// One layer of a sequence's mirror: its blocks in position order, and how many of its
/// positions hold the host cache's keys and values.
#[derive(Default)]
struct LayerMirror {
    blocks: Vec<u32>,
    synced: usize,
}

/// A sequence's keys and values in the device KV pool, kept in its KV cache's device
/// slot. The host cache stays authoritative: positions the device did not compute or
/// copy are copied in before the device reads them, and a mirror of another epoch (the
/// cache was cleared or restored) starts again.
struct Mirror {
    /// The pool's identity: a mirror made for another engine's pool is replaced.
    pool: Weak<KvFree>,
    free: Arc<KvFree>,
    epoch: u64,
    layers: Vec<LayerMirror>,
}

impl Mirror {
    /// Gives every block back.
    fn release(&mut self) {
        let mut f = self.free.lock();
        for l in &mut self.layers {
            f.append(&mut l.blocks);
            l.synced = 0;
        }
    }
}

impl Drop for Mirror {
    fn drop(&mut self) {
        self.release();
    }
}

/// One pass's attention inputs for the device ([`DeviceMoeAttention`]).
#[derive(Default)]
struct AttnPlan {
    pos: Vec<u32>,
    blocks: Vec<u32>,
    table_at: Vec<u32>,
    sync: Vec<f32>,
    sync_to: Vec<u32>,
}

/// The dense weights on a discrete device, and its expert cache.
pub(crate) struct OnDevice {
    device: Arc<dyn DiscreteDevice>,
    ledger: Arc<MemoryLedger>,
    /// The dense arena's device copy, and the arena's host address range.
    dense: DeviceBuffer,
    arena: std::ops::Range<usize>,
    budget: u64,
    /// The expert arenas, one per size class, and each layer's class.
    arenas: Vec<Arena>,
    class_of: Vec<Option<usize>>,
    shared: Arc<Shared>,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    over_budget: AtomicU64,
    /// Whether every arena has a slot for every expert of every layer of its class, so
    /// holding a whole layer on the device never evicts (routing on the device).
    provisioned: bool,
    routed: AtomicU64,
    /// The KV pool, when the device runs attention and the budget leaves room.
    kv: Option<KvPool>,
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

/// The arenas for a device expert cache of `budget` bytes over `layers`, whose RAM
/// cache holds `ram_slots[i]` experts of layer `i`, on a device that addresses at most
/// `max_arena` bytes at once; and each layer's class. No arenas when the budget cannot
/// give every layer a slot.
fn plan_arenas(
    layers: &[LayerExperts],
    ram_slots: &[u32],
    budget: u64,
    max_arena: u64,
) -> (Vec<Arena>, Vec<Option<usize>>) {
    let slots = qubivex_memory::distribute(layers, Bytes(budget), None);
    let need: u64 = layers
        .iter()
        .zip(&slots)
        .map(|(l, &s)| u64::from(s) * l.expert_bytes.0)
        .sum();
    let mut class_of = vec![None; layers.len()];
    if layers.is_empty() || need > budget || slots.contains(&0) {
        return (Vec::new(), class_of);
    }
    // Size classes by expert bytes, in first-seen order.
    let mut sizes: Vec<(u64, u64)> = Vec::new();
    for (i, (l, &s)) in layers.iter().zip(&slots).enumerate() {
        let s = u64::from(s.min(ram_slots.get(i).copied().unwrap_or(u32::MAX)));
        let c = match sizes.iter().position(|&(b, _)| b == l.expert_bytes.0) {
            Some(c) => c,
            None => {
                sizes.push((l.expert_bytes.0, 0));
                sizes.len() - 1
            }
        };
        sizes[c].1 += s;
        class_of[i] = Some(c);
    }
    let arenas = sizes
        .into_iter()
        .map(|(bytes, n)| {
            let stride = usize::try_from(bytes)
                .unwrap_or(usize::MAX)
                .max(1)
                .next_multiple_of(SLOT_ALIGN);
            let fit = max_arena / stride as u64;
            let n = usize::try_from(n.min(fit)).unwrap_or(0);
            Arena {
                stride,
                slots: n,
                buffer: OnceLock::new(),
                pins: (0..n).map(|_| Arc::new(())).collect(),
            }
        })
        .collect();
    (arenas, class_of)
}

impl OnDevice {
    /// Whether the device asks callers to group matmuls.
    pub(crate) fn groups_matmuls(&self) -> bool {
        self.device.groups_matmuls()
    }

    /// Whether the device fuses this many expert blocks of this many rows.
    pub(crate) fn fuses_gated_ffn(&self, blocks: usize, rows: usize) -> bool {
        self.device.fuses_gated_ffn(blocks, rows)
    }

    /// Copies `arena` (the dense weights) to `device`, charged to the ledger's
    /// `Vram(0)` tier, with an expert cache of `budget` bytes for `layers`, whose RAM
    /// cache holds `ram_slots` experts per layer.
    pub(crate) fn new(
        device: Arc<dyn DiscreteDevice>,
        ledger: Arc<MemoryLedger>,
        arena: &PageAligned,
        budget: u64,
        layers: &[LayerExperts],
        ram_slots: &[u32],
        kv_out: usize,
    ) -> Result<Self, ComputeError> {
        let host = arena.as_slice();
        let charge = ledger.charge(Tier::Vram(0), Owner::Dense, Bytes(host.len() as u64));
        let dense = device.upload(host, Some(Box::new(charge)))?;
        let start = host.as_ptr() as usize;
        let (arenas, class_of) = plan_arenas(layers, ram_slots, budget, device.max_arena());
        let provisioned = !arenas.is_empty()
            && arenas.iter().enumerate().all(|(k, a)| {
                let need: u64 = layers
                    .iter()
                    .zip(&class_of)
                    .filter(|(_, c)| **c == Some(k))
                    .map(|(l, _)| u64::from(l.experts))
                    .sum();
                a.slots as u64 >= need
            })
            && class_of.iter().all(Option::is_some);
        let free = arenas
            .iter()
            .map(|a| (0..a.slots).rev().collect())
            .collect();
        // The KV pool: as configured, else what the budget leaves after the expert
        // arenas; never more than one buffer the device addresses at once.
        let expert_bytes: u64 = arenas.iter().map(|a| (a.stride * a.slots) as u64).sum();
        let block_words = kv_block_words(kv_out);
        let kv = (kv_out > 0 && provisioned)
            .then(|| {
                let room = device
                    .kv_pool()
                    .unwrap_or_else(|| budget.saturating_sub(expert_bytes))
                    .min(device.max_arena());
                let blocks = u32::try_from(room / (4 * block_words as u64)).unwrap_or(u32::MAX);
                (blocks > 0).then(|| KvPool {
                    block_words,
                    blocks: blocks as usize,
                    buffer: OnceLock::new(),
                    free: Arc::new(KvFree {
                        free: Mutex::new((0..blocks).rev().collect()),
                    }),
                    layers: AtomicU64::new(0),
                    synced: AtomicU64::new(0),
                    full: AtomicU64::new(0),
                })
            })
            .flatten();
        Ok(Self {
            device,
            ledger,
            dense,
            arena: start..start + host.len(),
            budget,
            arenas,
            class_of,
            shared: Arc::new(Shared {
                cache: Mutex::new(Cache {
                    free,
                    ..Cache::default()
                }),
                detached: AtomicU64::new(0),
            }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            over_budget: AtomicU64::new(0),
            provisioned,
            routed: AtomicU64::new(0),
            kv,
        })
    }

    /// Whether a layer of `experts` experts, top `k`, over `s` tokens should be routed
    /// on the device: the device routes the shape and the cache holds every expert of
    /// every layer, so holding a layer evicts nothing.
    pub(crate) fn routes_moe(&self, experts: usize, k: usize, s: usize) -> bool {
        self.provisioned && self.device.routes_moe(experts, k, s)
    }

    /// Whether such a layer should be routed with its prelude (the O projection,
    /// residual and norm) in the same submission (Step 3c); as [`Self::routes_moe`].
    pub(crate) fn routes_moe_prelude(&self, experts: usize, k: usize, s: usize) -> bool {
        self.provisioned && self.device.routes_moe_prelude(experts, k, s)
    }

    /// Whether such a layer should be routed with decode attention in the same
    /// submission (audit §21): as [`Self::routes_moe_prelude`], with a KV pool.
    pub(crate) fn routes_moe_attention(&self, experts: usize, k: usize, s: usize) -> bool {
        self.provisioned && self.kv.is_some() && self.device.routes_moe_attention(experts, k, s)
    }

    /// The KV pool's buffer, allocated now if this is its first use; `None` when the
    /// device refused the memory.
    fn kv_buffer(&self, pool: &KvPool) -> Option<DeviceBuffer> {
        pool.buffer
            .get_or_init(|| {
                let len = pool.blocks.checked_mul(4 * pool.block_words)?;
                let charge = self
                    .ledger
                    .charge(Tier::Vram(0), Owner::Kv, Bytes(len as u64));
                self.device.alloc(len, Some(Box::new(charge))).ok()
            })
            .clone()
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

    /// Arena `class`'s buffer, allocated now if this is its first use; `None` if the
    /// device has no arenas or refused the memory.
    fn arena(&self, class: usize) -> Option<&DeviceBuffer> {
        let a = &self.arenas[class];
        a.buffer
            .get_or_init(|| {
                let len = a.stride.checked_mul(a.slots)?;
                let charge =
                    self.ledger
                        .charge(Tier::Vram(0), Owner::ExpertCache, Bytes(len as u64));
                self.device.alloc(len, Some(Box::new(charge))).ok()
            })
            .as_ref()
    }

    /// The pin of `home`, for a caller.
    fn pin_of(&self, home: &Home) -> Pin {
        match home {
            Home::Slot { class, slot } => Pin {
                _buffer: self.arenas[*class]
                    .buffer
                    .get()
                    .and_then(Option::as_ref)
                    .expect("a slot's arena is allocated")
                    .clone(),
                _pin: Arc::clone(&self.arenas[*class].pins[*slot]),
            },
            Home::Own { buffer, pin } => Pin {
                _buffer: buffer.clone(),
                _pin: Arc::clone(pin),
            },
        }
    }

    fn pinned(&self, home: &Home) -> bool {
        // A pin is only taken under the cache lock, so a stale count can only overstate.
        match home {
            Home::Slot { class, slot } => Arc::strong_count(&self.arenas[*class].pins[*slot]) > 1,
            Home::Own { pin, .. } => Arc::strong_count(pin) > 1,
        }
    }

    /// A free slot of `class`, evicting the least recently used unpinned copy of the
    /// class if none is free. The slot's pin is taken for the caller's copy.
    fn take_slot(&self, c: &mut Cache, class: usize) -> Option<(usize, Arc<()>)> {
        let pins = &self.arenas[class].pins;
        // A slot freed while still read (dropped with the RAM cache mid-use) waits.
        if let Some(i) = c.free[class]
            .iter()
            .rposition(|&s| Arc::strong_count(&pins[s]) == 1)
        {
            let slot = c.free[class].swap_remove(i);
            return Some((slot, Arc::clone(&pins[slot])));
        }
        let victim = c
            .entries
            .iter()
            .filter(|(_, e)| matches!(e.home, Home::Slot { class: k, .. } if k == class))
            .filter(|(_, e)| !self.pinned(&e.home))
            .min_by_key(|(_, e)| e.last_use)
            .map(|(&a, _)| a)?;
        let e = c.entries.remove(&victim).expect("present");
        c.forget(&e);
        self.evictions.fetch_add(1, Ordering::Relaxed);
        let slot = c.free[class].pop().expect("the victim's slot");
        Some((slot, Arc::clone(&pins[slot])))
    }

    /// The device copy of the expert `key` held in `slab`, copied now if the device has
    /// none. Hold the returned pin while computing the expert.
    pub(crate) fn ensure(&self, key: ExpertKey, slab: &Slab) -> Result<Pin, ComputeError> {
        self.copy_of(key, slab, true)
    }

    /// As [`OnDevice::ensure`], for a layer held whole before the device chooses among
    /// its experts: holding is not a use, so a copy already there is pinned without
    /// counting a hit or refreshing its recency. The pass then calls
    /// [`OnDevice::ensure`] for the experts it used.
    pub(crate) fn hold(&self, key: ExpertKey, slab: &Slab) -> Result<Pin, ComputeError> {
        self.copy_of(key, slab, false)
    }

    fn copy_of(&self, key: ExpertKey, slab: &Slab, used: bool) -> Result<Pin, ComputeError> {
        let host = slab.bytes();
        let addr = host.as_ptr() as usize;
        let class = self
            .class_of
            .get(key.0.0 as usize)
            .copied()
            .flatten()
            .filter(|&k| self.arenas[k].slots > 0 && host.len() <= self.arenas[k].stride);
        // The arena first, outside the cache lock: its allocation can take a while.
        let arena = class.and_then(|k| self.arena(k).cloned());
        let mut c = self.shared.lock();
        c.clock += 1;
        let now = c.clock;
        if let Some(e) = c.entries.get_mut(&addr)
            && e.key == key
            && e.len == host.len()
        {
            if used {
                e.last_use = now;
                self.hits.fetch_add(1, Ordering::Relaxed);
            }
            return Ok(self.pin_of(&e.home));
        }
        let slot = match (class, &arena) {
            (Some(k), Some(_)) => {
                // Copies made over the budget while every slot was pinned go first.
                self.make_room(&mut c, 0);
                let s = self.take_slot(&mut c, k);
                if s.is_none() {
                    self.over_budget.fetch_add(1, Ordering::Relaxed);
                }
                s.map(|(s, pin)| (k, s, pin))
            }
            _ => {
                self.make_room(&mut c, host.len() as u64);
                None
            }
        };
        if slot.is_none() {
            // Reserved now, so concurrent copies cannot all fit into the same room.
            c.own += host.len() as u64;
        }
        drop(c);
        // Copy outside the lock, so other experts are found while this one moves. A
        // slot being written is in no entry and pinned, so nothing reads or takes it.
        let home = match (&slot, &arena) {
            (Some((k, s, pin)), Some(buffer)) => self
                .device
                .write(buffer, s * self.arenas[*k].stride, host)
                .map(|()| Home::Slot {
                    class: *k,
                    slot: *s,
                })
                .map(|h| (h, Arc::clone(pin))),
            _ => {
                let charge =
                    self.ledger
                        .charge(Tier::Vram(0), Owner::ExpertCache, Bytes(host.len() as u64));
                self.device
                    .upload(host, Some(Box::new(charge)))
                    .map(|buffer| {
                        let pin = Arc::new(());
                        (
                            Home::Own {
                                buffer,
                                pin: Arc::clone(&pin),
                            },
                            pin,
                        )
                    })
            }
        };
        let mut c = self.shared.lock();
        let give_back = |c: &mut Cache| match &slot {
            Some((k, s, _)) => c.free[*k].push(*s),
            None => c.own -= host.len() as u64,
        };
        let (home, _copying) = match home {
            Ok(h) => h,
            Err(e) => {
                give_back(&mut c);
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
            let pin = self.pin_of(&e.home);
            give_back(&mut c);
            return Ok(pin);
        }
        if let Some(old) = c.entries.remove(&addr) {
            c.forget(&old);
        }
        let pin = self.pin_of(&home);
        c.held += host.len() as u64;
        c.entries.insert(
            addr,
            Entry {
                key,
                len: host.len(),
                home,
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
        Ok(pin)
    }

    /// Evicts unpinned copies with their own allocation, least recently used first,
    /// until `len` more bytes of them fit the budget left beside the arenas, or
    /// nothing more can go.
    fn make_room(&self, c: &mut Cache, len: u64) {
        let room = self.budget.saturating_sub(self.arena_bytes());
        while c.own + len > room {
            let victim = c
                .entries
                .iter()
                .filter(|(_, e)| matches!(e.home, Home::Own { .. }) && !self.pinned(&e.home))
                .min_by_key(|(_, e)| e.last_use)
                .map(|(&a, _)| a);
            let Some(a) = victim else {
                self.over_budget.fetch_add(1, Ordering::Relaxed);
                return;
            };
            let e = c.entries.remove(&a).expect("present");
            c.forget(&e);
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Bytes of every arena, allocated or not.
    fn arena_bytes(&self) -> u64 {
        self.arenas
            .iter()
            .map(|a| (a.stride * a.slots) as u64)
            .sum()
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
        match &e.home {
            Home::Own { buffer, .. } => DeviceWeights::like(buffer.clone(), p - base, w),
            Home::Slot { class, slot } => {
                let a = &self.arenas[*class];
                let buffer = a
                    .buffer
                    .get()
                    .and_then(Option::as_ref)
                    .ok_or_else(|| not_on_device("a weight"))?;
                DeviceWeights::like(buffer.clone(), slot * a.stride + (p - base), w)
            }
        }
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

    /// The gated feed-forward blocks, as the device runs them.
    pub(crate) fn gated_ffn_many(&self, ops: &mut [GatedFfn<'_>]) -> Result<(), ComputeError> {
        let mut ws = Vec::with_capacity(ops.len());
        for op in ops.iter() {
            op.check()?;
            ws.push([
                self.weights(&op.gate)?,
                self.weights(&op.up)?,
                self.weights(&op.down)?,
            ]);
        }
        let mut batch: Vec<DeviceGatedFfn<'_>> = ops
            .iter_mut()
            .zip(&ws)
            .map(|(op, [g, u, d])| DeviceGatedFfn {
                gate: g,
                up: u,
                down: d,
                xs: op.xs,
                s: op.s,
                ys: &mut *op.ys,
            })
            .collect();
        self.device.gated_ffn_many(&mut batch)
    }

    /// The layer routed on the device, if the device runs it; `Ok(false)` otherwise,
    /// with nothing written. Every expert's copy must be pinned by the caller.
    pub(crate) fn routed_moe(&self, op: &mut RoutedMoe<'_>) -> Result<bool, ComputeError> {
        op.check()?;
        let router = self.weights(&op.router)?;
        let experts = op
            .experts
            .iter()
            .map(|[g, u, d]| Ok([self.weights(g)?, self.weights(u)?, self.weights(d)?]))
            .collect::<Result<Vec<_>, ComputeError>>()?;
        let o = match &op.prelude {
            Some(p) => Some(self.weights(&p.o)?),
            None => None,
        };
        let prelude = match (op.prelude.as_mut(), o.as_ref()) {
            (Some(p), Some(o)) => Some(DeviceMoePrelude {
                o,
                att: p.att,
                hs: &mut *p.hs,
                norm: p.norm,
                eps: p.eps,
            }),
            _ => None,
        };
        // Decode attention: each sequence's blocks for this layer, and the host rows the
        // device does not hold yet.
        let mut plan = None;
        let attn_ws;
        let pool_buf;
        let attention = match op.attention.as_mut() {
            None => None,
            Some(a) => {
                let Some(pool) = self.kv.as_ref() else {
                    return Ok(false);
                };
                let Some(buf) = self.kv_buffer(pool) else {
                    return Ok(false);
                };
                pool_buf = buf;
                attn_ws = [
                    self.weights(&a.q)?,
                    self.weights(&a.k)?,
                    self.weights(&a.v)?,
                ];
                let Some(p) = self.attention_plan(
                    pool,
                    &mut a.seqs,
                    a.kv_heads * a.head_dim,
                    a.head_dim,
                    a.layer,
                ) else {
                    pool.full.fetch_add(1, Ordering::Relaxed);
                    return Ok(false);
                };
                let p = plan.insert(p);
                Some(DeviceMoeAttention {
                    q: &attn_ws[0],
                    k: &attn_ws[1],
                    v: &attn_ws[2],
                    xs: a.xs,
                    heads: a.heads,
                    kv_heads: a.kv_heads,
                    head_dim: a.head_dim,
                    clip: a.clip,
                    qk_norm: a.qk_norm,
                    eps: a.eps,
                    rope: a.rope,
                    pool: &pool_buf,
                    pos: &p.pos,
                    blocks: &p.blocks,
                    table_at: &p.table_at,
                    sync: &p.sync,
                    sync_to: &p.sync_to,
                    k_new: &mut *a.k_new,
                    v_new: &mut *a.v_new,
                })
            }
        };
        let ran = self.device.routed_moe(&mut DeviceRoutedMoe {
            router: &router,
            experts: &experts,
            k: op.k,
            renormalise: op.renormalise,
            xs: op.xs,
            s: op.s,
            ys: &mut *op.ys,
            ids: &mut *op.ids,
            weights: &mut *op.weights,
            prelude,
            attention,
        })?;
        if ran {
            self.routed.fetch_add(1, Ordering::Relaxed);
            if let (Some(a), Some(p), Some(pool)) = (op.attention.as_mut(), plan, self.kv.as_ref())
            {
                // Every sequence's mirror now holds positions 0..=pos of this layer.
                for q in &mut a.seqs {
                    if let Some(m) = q.mirror.as_mut().and_then(|m| m.downcast_mut::<Mirror>()) {
                        m.layers[a.layer].synced = q.pos + 1;
                    }
                }
                pool.layers.fetch_add(1, Ordering::Relaxed);
                pool.synced
                    .fetch_add(p.sync_to.len() as u64, Ordering::Relaxed);
            }
        }
        Ok(ran)
    }

    /// Each sequence's mirror made ready for layer `layer`: in this pool's epoch, with
    /// blocks for positions `0..=pos`, and the host rows it does not hold listed to copy
    /// in. `None` when the pool has no free block (blocks already taken stay with their
    /// sequence).
    fn attention_plan(
        &self,
        pool: &KvPool,
        seqs: &mut [AttnSeq<'_>],
        kv_out: usize,
        d: usize,
        layer: usize,
    ) -> Option<AttnPlan> {
        let mut plan = AttnPlan::default();
        for q in seqs.iter_mut() {
            let mine = q
                .mirror
                .as_ref()
                .and_then(|m| m.downcast_ref::<Mirror>())
                .is_some_and(|m| m.pool.ptr_eq(&Arc::downgrade(&pool.free)));
            if !mine {
                *q.mirror = Some(Box::new(Mirror {
                    pool: Arc::downgrade(&pool.free),
                    free: Arc::clone(&pool.free),
                    epoch: q.epoch,
                    layers: Vec::new(),
                }));
            }
            let m = q.mirror.as_mut()?.downcast_mut::<Mirror>()?;
            if m.epoch != q.epoch {
                m.release();
                m.epoch = q.epoch;
            }
            if m.layers.len() <= layer {
                m.layers.resize_with(layer + 1, LayerMirror::default);
            }
            let lm = &mut m.layers[layer];
            // Rows past the host's count may be from a pass that failed after the device
            // stored them: they are written again.
            lm.synced = lm.synced.min(q.pos);
            let need = q.pos / KV_BLOCK + 1;
            if lm.blocks.len() < need {
                let mut f = pool.free.lock();
                while lm.blocks.len() < need {
                    lm.blocks.push(f.pop()?);
                }
            }
            plan.pos.push(u32::try_from(q.pos).ok()?);
            plan.table_at.push(u32::try_from(plan.blocks.len()).ok()?);
            plan.blocks.extend_from_slice(&lm.blocks[..need]);
            for p in lm.synced..q.pos {
                let b = lm.blocks[p / KV_BLOCK] as usize;
                plan.sync_to
                    .push(u32::try_from(b * KV_BLOCK + p % KV_BLOCK).ok()?);
                for rows in [&q.host_k, &q.host_v] {
                    for r in rows.iter() {
                        plan.sync.extend_from_slice(&r[p * d..(p + 1) * d]);
                    }
                }
            }
            debug_assert_eq!(plan.sync.len(), plan.sync_to.len() * 2 * kv_out);
        }
        Some(plan)
    }

    pub(crate) fn report(&self) -> DeviceReport {
        let (held, own) = {
            let c = self.shared.lock();
            (c.held, c.own)
        };
        let allocated: u64 = self
            .arenas
            .iter()
            .filter(|a| a.buffer.get().is_some_and(Option::is_some))
            .map(|a| (a.stride * a.slots) as u64)
            .sum();
        DeviceReport {
            device: self.device.describe(),
            usage: self.device.usage(),
            transfers: self.device.transfers(),
            experts: DeviceCacheStats {
                budget: self.budget,
                used: allocated + own,
                held,
                arena_bytes: self.arena_bytes(),
                arena_slots: self.arenas.iter().map(|a| a.slots as u64).sum(),
                hits: self.hits.load(Ordering::Relaxed),
                misses: self.misses.load(Ordering::Relaxed),
                evictions: self.evictions.load(Ordering::Relaxed),
                detached: self.shared.detached.load(Ordering::Relaxed),
                over_budget: self.over_budget.load(Ordering::Relaxed),
            },
            routed_layers: self.routed.load(Ordering::Relaxed),
            kv_pool: self
                .kv
                .as_ref()
                .map_or_else(KvPoolStats::default, |p| KvPoolStats {
                    bytes: (p.blocks * 4 * p.block_words) as u64,
                    allocated: p.buffer.get().is_some_and(Option::is_some),
                    blocks: p.blocks as u64,
                    free: p.free.lock().len() as u64,
                    layers: p.layers.load(Ordering::Relaxed),
                    synced_rows: p.synced.load(Ordering::Relaxed),
                    full: p.full.load(Ordering::Relaxed),
                }),
        }
    }
}
