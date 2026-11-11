//! A loaded model ready to generate: resident weights, the expert store, the tokenizer.

use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use qubivex_arch::{
    ArchError, DenseWeights, ExpertProvider, ExpertWeights, KvCache, Logits, Segment, Sources,
    Tracer, Transformer,
};
use qubivex_backend_cpu::sim::{SimConfig, SimDevice};
use qubivex_compute::WeightView;
use qubivex_compute::discrete::DeviceBuffer;
use qubivex_compute::host::PageAligned;
use qubivex_compute::tensor::TensorView;
use qubivex_core::{Bytes, ExpertIdx, LayerIdx, Shape, TensorDesc};
use qubivex_expert_stream::fault::{DiskLimit, SlowDisk, VerifiedSource};
use qubivex_expert_stream::mirror::{MirrorPolicy, MirroredSource};
use qubivex_expert_stream::policy::SaturationGauge;
use qubivex_expert_stream::predict::{Lookahead, LookaheadConfig};
use qubivex_expert_stream::{
    Batch, ExpertKey, ExpertSource, ExpertStore, Lease, Next, Occupancy, Policy, PoolSpec,
    PrefetchStats, ReadTimeout, StoreError, StoreStats,
};
use qubivex_hardware::{Host, Probed, ResourceProbe, SystemProbe};
use qubivex_kv_cache::{PrefixCache, PrefixStats};
use qubivex_memory::{
    Charge, HostMemory, LayerExperts, MemoryLedger, MemoryPlan, ModelFootprint, Owner, PlanRequest,
    PressureConfig, PressureDecision, PressureMonitor, Profile, Tier, VRAM_ACTIVATION_RESERVE,
    VramRequest,
};
use qubivex_model::{ExpertPart, ModelMetadata, TensorRole};
use qubivex_tokenizer::Tokenizer;

use crate::checksums::{ExpertSums, SUMS_FILE, SumsState};
use crate::combine::{BatchStats, Combiner, Outcome, Step};
use crate::discrete::{DeviceReport, OnDevice};
use crate::metal_forward::MetalForward;
use crate::placement::{BackendKind, MetalOps, Placed, PlacementReport};
use crate::{ExpertIo, IoPath, ModelExpertSource, RuntimeError, load_tokenizer};
use qubivex_backend_metal::MetalBackend;
use qubivex_backend_vulkan::{VulkanConfig, VulkanDevice};
use qubivex_compute::discrete::DiscreteDevice;

/// How to load a model.
#[derive(Clone, Copy, Debug, Default)]
pub struct EngineOptions {
    /// A fixed expert-cache capacity in bytes, bypassing the memory planner. `None`:
    /// the planner sizes the cache from available RAM and shrinks it as the KV grows.
    pub expert_cache_bytes: Option<u64>,
    /// Context limit. `None`: the model's maximum (or 4096 if it states none), which
    /// the planner may shorten to what fits.
    pub max_context: Option<usize>,
    /// The memory profile (ignored with `expert_cache_bytes`).
    pub profile: Profile,
    /// An explicit RAM budget in bytes, replacing the profile's (ignored with
    /// `expert_cache_bytes`).
    pub ram_budget: Option<u64>,
    /// How often the pressure monitor samples available RAM, in milliseconds.
    /// `None`: every 500 ms; `Some(0)`: no monitor thread
    /// ([`Engine::observe_pressure`] still works). Planned engines only.
    pub pressure_interval_ms: Option<u64>,
    /// The pressure monitor's thresholds.
    pub pressure: PressureConfig,
    /// How a full expert pool picks its victim (ADR-008).
    pub cache_policy: Policy,
    /// Dedicated threads reading experts (ADR-006). `None`: the default (see
    /// [`DEFAULT_IO_THREADS`]); `Some(0)`: the compute thread reads its own misses.
    pub io_threads: Option<usize>,
    /// How long a forward pass waits for one expert's read before failing the request
    /// (ADR-006 amendment 4). `None`: [`ReadTimeout::DEFAULT`]; [`ReadTimeout::NEVER`]
    /// waits forever. Needs I/O threads: a compute thread reading its own miss cannot
    /// time out.
    pub read_timeout: Option<ReadTimeout>,
    /// Extra attempts after a failed expert read. `None`: [`DEFAULT_READ_RETRIES`].
    pub read_retries: Option<u32>,
    /// Whether expert reads are checked against the model's recorded checksums
    /// ([`crate::checksums`]).
    pub verify: VerifyExperts,
    /// How experts are read: buffered (default) or bypassing the page cache
    /// (ADR-006 amendment 2).
    pub expert_io: ExpertIo,
    /// Compute threads. `None`: the CPU's compute team (its fastest cores: efficiency
    /// cores slow a team that meets at every matmul), else every logical CPU.
    pub threads: Option<usize>,
    /// Router-lookahead prefetch during decode (Phase 7). `None`: off.
    pub prefetch: Option<PrefetchOptions>,
    /// Reads go through a simulated disk with this limit, to measure saturation on a
    /// machine whose warm reads never saturate ([`SlowDisk`]). `None`: the real source.
    pub disk_limit: Option<DiskLimit>,
    /// The compute backend (Phase 8). Placement changes speed, never bits.
    pub backend: BackendKind,
    /// The dense matmuls placed on Metal with [`BackendKind::Metal`].
    pub metal_ops: MetalOps,
    /// The simulated device for [`BackendKind::Sim`]. `None`: [`DEFAULT_SIM_VRAM`]
    /// with PCIe 4.0 transfer costs, counted but not slept.
    pub sim: Option<SimConfig>,
    /// The Vulkan device for [`BackendKind::Vulkan`] (ADR-019): which one, and how
    /// much of its memory to allow.
    pub vulkan: VulkanConfig,
    /// An explicit VRAM budget in bytes for a discrete device, replacing the
    /// profile's fraction of its memory (ADR-018).
    pub vram_budget: Option<u64>,
    /// Most bytes of finished KV to keep for later prompts to reuse (ADR-016). 0: none,
    /// and every prompt runs in full. It counts against the memory plan, and running
    /// requests and memory pressure take it back first.
    pub prefix_cache_bytes: u64,
    /// Most tokens of prefill one shared forward pass takes beside its decode steps
    /// (ADR-017 section 3). `None`: [`DEFAULT_BATCH_TOKENS`].
    pub batch_tokens: Option<usize>,
    /// How long, in microseconds, a shared pass waits for the callers the previous
    /// pass answered to post their next step, so they stay in one pass. `None`:
    /// [`DEFAULT_BATCH_WAIT_US`]; `Some(0)`: no wait. A lone caller never waits.
    pub batch_wait_us: Option<u64>,
}

/// The simulated device's memory when [`EngineOptions::sim`] is `None`: 8 GiB.
pub const DEFAULT_SIM_VRAM: u64 = 8 << 30;

/// The default [`EngineOptions::batch_wait_us`]: sampling a token takes microseconds and
/// a pass tens of milliseconds, so the bound is only reached when a caller has stopped.
pub const DEFAULT_BATCH_WAIT_US: u64 = 1000;

/// The default [`EngineOptions::batch_tokens`]: one prefill chunk, so a long prompt
/// delays the others' next token by at most one chunk.
pub const DEFAULT_BATCH_TOKENS: usize = 256;

/// Router-lookahead prefetch settings (Phase 7).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PrefetchOptions {
    /// How predictions are made.
    pub lookahead: LookaheadConfig,
    /// The LFRU eviction guard: a prediction may evict a resident expert only if it is
    /// clearly hotter (ADR-008).
    pub guard: bool,
    /// Stop speculating while reads hold compute back (ADR-008 amendment 3).
    pub throttle: bool,
}

/// How well the lookahead predicted routing: every decode layer that had a prediction
/// compares it with the experts the router chose.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PredictionStats {
    /// Layers checked.
    pub layers: u64,
    /// Experts predicted, summed over those layers.
    pub predicted: u64,
    /// Experts routed, summed over those layers.
    pub routed: u64,
    /// Routed experts that were predicted.
    pub correct: u64,
}

impl PredictionStats {
    /// The share of routed experts that were predicted.
    #[must_use]
    pub fn recall(&self) -> f64 {
        ratio(self.correct, self.routed)
    }

    /// The share of predicted experts that were routed.
    #[must_use]
    pub fn precision(&self) -> f64 {
        ratio(self.correct, self.predicted)
    }
}

#[allow(clippy::cast_precision_loss)]
fn ratio(a: u64, b: u64) -> f64 {
    if b == 0 { 0.0 } else { a as f64 / b as f64 }
}

/// Prefetch counters: prediction quality and what the store did with predictions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefetchReport {
    /// Prediction quality.
    pub prediction: PredictionStats,
    /// Reads issued, hidden, wasted and cancelled.
    pub store: PrefetchStats,
    /// Forward passes since load.
    pub passes: u64,
    /// Of those, passes the saturation throttle kept from speculating.
    pub held_passes: u64,
}

/// The predictor's state across forward passes.
struct Predictor {
    lookahead: Lookahead,
    guard: bool,
    /// A prediction made during the previous layer, issued after that layer's demand.
    pending: Option<(LayerIdx, Vec<ExpertIdx>)>,
    /// The last prediction issued, checked against the layer's routing.
    issued: Option<(LayerIdx, Vec<ExpertIdx>)>,
    stats: PredictionStats,
    /// `None`: the throttle is off.
    saturation: Option<SaturationGauge>,
    /// The gauge's clock.
    epoch: Instant,
    passes: u64,
    held_passes: u64,
}

/// I/O threads when none are requested.
pub const DEFAULT_IO_THREADS: usize = 4;

/// Whether expert reads are checked against recorded checksums.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VerifyExperts {
    /// When the model has a checksum file that matches its weights (the default).
    #[default]
    Auto,
    /// Never.
    Off,
    /// Always: loading fails without a matching checksum file.
    Require,
}

/// Whether this engine verifies expert reads, and why ([`Engine::verification`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verification {
    /// Every read is checked against this many recorded checksums.
    Verified(usize),
    /// Turned off ([`VerifyExperts::Off`]).
    Off,
    /// The model has no checksum file.
    Missing,
    /// The checksum file does not match the weight files (why).
    Stale(String),
}

/// Extra attempts after a failed expert read: one, for transient errors.
pub const DEFAULT_READ_RETRIES: u32 = 1;

/// The default thread count: the compute team, else every logical CPU.
fn default_threads() -> usize {
    let cpu = qubivex_hardware::probe_cpu(&qubivex_hardware::Host::live());
    cpu.performance_cores()
        .known()
        .and_then(|&n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, std::num::NonZero::get))
}

/// The resident tensors: everything except routed experts. They are stored as read
/// from the file, in one page-aligned arena that a unified-memory GPU uses in place
/// (ADR-011 §4), each described as a matrix (a vector `[n]` is viewed as `[1, n]`).
struct Dense {
    arena: Arc<PageAligned>,
    tensors: HashMap<String, (TensorDesc, std::ops::Range<usize>)>,
}

/// Tensors start at multiples of this many bytes in the arena.
const DENSE_ALIGN: usize = 256;

impl DenseWeights for Dense {
    fn view(&self, name: &str) -> Result<WeightView<'_>, ArchError> {
        let (desc, range) = self
            .tensors
            .get(name)
            .ok_or_else(|| ArchError::Missing(name.to_owned()))?;
        let bytes = &self.arena.as_slice()[range.clone()];
        Ok(WeightView::try_from(TensorView::new(*desc, bytes)?)?)
    }
}

/// Experts from the store, one lease at a time.
struct StoreProvider<'a> {
    store: &'a ExpertStore,
    source: &'a ModelExpertSource,
    /// Set when prefetch is on and memory pressure allows it.
    predictor: Option<&'a Mutex<Predictor>>,
    /// A discrete device: each expert is copied there before it is computed.
    device: Option<&'a OnDevice>,
}

impl StoreProvider<'_> {
    /// After `layer`'s demand is queued: checks the prediction made for `layer`, and
    /// issues the one made for the next layer, so prefetches queue behind demand.
    fn after_demand(&self, layer: LayerIdx, routed: &[ExpertIdx]) {
        let Some(p) = self.predictor else { return };
        let mut p = p.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((l, ids)) = p.issued.take()
            && l == layer
        {
            let correct = routed.iter().filter(|e| ids.contains(e)).count();
            p.stats.layers += 1;
            p.stats.predicted += ids.len() as u64;
            p.stats.routed += routed.len() as u64;
            p.stats.correct += correct as u64;
        }
        if let Some((l, ids)) = p.pending.take()
            && l.0 == layer.0 + 1
        {
            let keys: Vec<ExpertKey> = ids.iter().map(|&e| (l, e)).collect();
            // Advisory: a poisoned store fails the demand that follows anyway.
            let _ = self.store.prefetch(&keys, p.guard);
            p.issued = Some((l, ids));
        }
    }
}

impl StoreProvider<'_> {
    fn weights<'b>(
        &self,
        layer: LayerIdx,
        expert: ExpertIdx,
        bytes: &'b [u8],
    ) -> Result<ExpertWeights<'b>, ArchError> {
        expert_weights(self.source, layer, expert, bytes)
    }

    /// The expert's device copy, if there is a device: hold it while computing.
    fn pin(&self, lease: &Lease<'_>) -> Result<Option<DeviceBuffer>, ArchError> {
        self.device
            .map(|d| d.ensure(lease.key(), lease.slab()))
            .transpose()
            .map_err(ArchError::from)
    }
}

/// An expert's matrices inside its leased bytes.
pub(crate) fn expert_weights<'b>(
    source: &ModelExpertSource,
    layer: LayerIdx,
    expert: ExpertIdx,
    bytes: &'b [u8],
) -> Result<ExpertWeights<'b>, ArchError> {
    let err = |detail: String| ArchError::Expert {
        layer: layer.0,
        expert: expert.0,
        detail,
    };
    let layout = source
        .layout(layer, expert)
        .ok_or_else(|| err("no such expert".into()))?;
    let view = |p: ExpertPart| -> Result<Option<WeightView<'b>>, ArchError> {
        layout
            .part(p)
            .map(|l| {
                WeightView::new(l.dtype, l.rows, l.cols, &bytes[l.offset..l.offset + l.len])
                    .map_err(ArchError::from)
            })
            .transpose()
    };
    let down = view(ExpertPart::Down)?.ok_or_else(|| err("no down projection".into()))?;
    let (gate, up) = match (
        view(ExpertPart::Gate)?,
        view(ExpertPart::Up)?,
        view(ExpertPart::GateUp)?,
    ) {
        (Some(g), Some(u), None) => (g, u),
        (None, None, Some(gu)) => {
            let ffn = gu.rows() / 2;
            (gu.rows_slice(0, ffn)?, gu.rows_slice(ffn, ffn)?)
        }
        _ => return Err(err("expected gate and up, or a stacked gate-up".into())),
    };
    Ok(ExpertWeights { gate, up, down })
}

pub(crate) fn store_err(layer: LayerIdx, expert: ExpertIdx, e: &StoreError) -> ArchError {
    ArchError::Expert {
        layer: layer.0,
        expert: expert.0,
        detail: e.to_string(),
    }
}

/// The experts of one [`StoreProvider::with_experts`] call in flight: the batch they
/// come from, and the first error any of them hit.
struct Wave<'p, 's, 'f> {
    provider: &'p StoreProvider<'s>,
    layer: LayerIdx,
    first: ExpertIdx,
    f: &'f (dyn Fn(ExpertIdx, &ExpertWeights<'_>) -> Result<(), ArchError> + Sync),
    batch: Mutex<Batch<'p>>,
    error: Mutex<Option<ArchError>>,
}

impl<'p> Wave<'p, '_, '_> {
    fn lock_batch(&self) -> std::sync::MutexGuard<'_, Batch<'p>> {
        // The batch's own state is consistent between calls; a panic elsewhere does not
        // corrupt it.
        self.batch.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn lease_err(&self, e: StoreError) -> ArchError {
        let at = match &e {
            StoreError::Load { key, .. } | StoreError::Full { key, .. } => key.1,
            _ => self.first,
        };
        store_err(self.layer, at, &e)
    }

    fn fail(&self, e: ArchError) {
        self.error
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_or_insert(e);
    }

    fn failed(&self) -> bool {
        self.error
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
    }

    fn take_error(&self) -> Option<ArchError> {
        self.error
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }

    /// Computes `held` (leases already taken from the batch) and then every other
    /// expert of the batch as it becomes ready.
    fn run(&self, held: Vec<Lease<'p>>) -> Result<(), ArchError> {
        if !held.is_empty() {
            rayon::scope(|sc| {
                for lease in held {
                    self.spawn(sc, lease);
                }
                self.start_ready(sc);
            });
            if let Some(e) = self.take_error() {
                return Err(e);
            }
        }
        loop {
            // Holding nothing: wait for the next expert (or fail as `next` does).
            let lease = match self.lock_batch().next() {
                None => return Ok(()),
                Some(lease) => lease.map_err(|e| self.lease_err(e))?,
            };
            rayon::scope(|sc| {
                self.spawn(sc, lease);
                self.start_ready(sc);
            });
            if let Some(e) = self.take_error() {
                return Err(e);
            }
        }
    }

    /// Computes `lease`'s expert on the pool, releases it, then starts whatever has
    /// become ready.
    fn spawn<'scope>(&'scope self, sc: &rayon::Scope<'scope>, lease: Lease<'p>) {
        sc.spawn(move |sc| {
            if self.failed() {
                return;
            }
            let expert = lease.key().1;
            let r = self.provider.pin(&lease).and_then(|_pin| {
                self.provider
                    .weights(self.layer, expert, lease.bytes())
                    .and_then(|w| (self.f)(expert, &w))
            });
            drop(lease);
            match r {
                Ok(()) => self.start_ready(sc),
                Err(e) => self.fail(e),
            }
        });
    }

    /// Spawns every expert of the batch that is ready now, without waiting.
    fn start_ready<'scope>(&'scope self, sc: &rayon::Scope<'scope>) {
        loop {
            if self.failed() {
                return;
            }
            let next = self.lock_batch().try_next();
            match next {
                Next::Lease(Ok(lease)) => self.spawn(sc, lease),
                Next::Lease(Err(e)) => return self.fail(self.lease_err(e)),
                Next::Later | Next::Done => return,
            }
        }
    }
}

impl ExpertProvider for StoreProvider<'_> {
    fn with_expert(
        &self,
        layer: LayerIdx,
        expert: ExpertIdx,
        f: &mut dyn FnMut(&ExpertWeights<'_>) -> Result<(), ArchError>,
    ) -> Result<(), ArchError> {
        let lease = self
            .store
            .acquire((layer, expert))
            .map_err(|e| store_err(layer, expert, &e))?;
        let _pin = self.pin(&lease)?;
        f(&self.weights(layer, expert, lease.bytes())?)
    }

    /// Starts every miss of the layer before waiting on any, and computes experts as
    /// they become ready (hits first), several at once on the current rayon pool.
    /// Whenever an expert finishes, its lease is released and whatever has become
    /// ready since is started, so loads into the freed slots begin at once. The
    /// blocking wait happens only while no lease is held, so a layer with fewer slots
    /// than routed experts still pipelines through them.
    fn with_experts(
        &self,
        layer: LayerIdx,
        experts: &[ExpertIdx],
        f: &(dyn Fn(ExpertIdx, &ExpertWeights<'_>) -> Result<(), ArchError> + Sync),
    ) -> Result<(), ArchError> {
        let keys: Vec<_> = experts.iter().map(|&e| (layer, e)).collect();
        let first = experts.first().copied().unwrap_or(ExpertIdx(0));
        let batch = self
            .store
            .acquire_many(&keys)
            .map_err(|e| store_err(layer, first, &e))?;
        self.after_demand(layer, experts);
        Wave {
            provider: self,
            layer,
            first,
            f,
            batch: Mutex::new(batch),
            error: Mutex::new(None),
        }
        .run(Vec::new())
    }

    /// Takes every lease that is ready now; when that is all of `experts`, pins them
    /// and calls `group` once. Otherwise computes the leases already taken and the
    /// rest one by one, exactly as [`StoreProvider::with_experts`]. The store counts
    /// hits, heat and recency at `acquire_many` either way.
    fn with_experts_grouped(
        &self,
        layer: LayerIdx,
        experts: &[ExpertIdx],
        group: &mut dyn FnMut(&[ExpertWeights<'_>]) -> Result<(), ArchError>,
        each: &(dyn Fn(ExpertIdx, &ExpertWeights<'_>) -> Result<(), ArchError> + Sync),
    ) -> Result<(), ArchError> {
        let keys: Vec<_> = experts.iter().map(|&e| (layer, e)).collect();
        let first = experts.first().copied().unwrap_or(ExpertIdx(0));
        let mut batch = self
            .store
            .acquire_many(&keys)
            .map_err(|e| store_err(layer, first, &e))?;
        self.after_demand(layer, experts);
        let mut held = Vec::with_capacity(experts.len());
        let complete = loop {
            match batch.try_next() {
                Next::Lease(Ok(lease)) => held.push(lease),
                Next::Lease(Err(e)) => {
                    let at = match &e {
                        StoreError::Load { key, .. } | StoreError::Full { key, .. } => key.1,
                        _ => first,
                    };
                    return Err(store_err(layer, at, &e));
                }
                Next::Done => break true,
                Next::Later => break false,
            }
        };
        if complete && held.len() == experts.len() {
            drop(batch);
            held.sort_by_key(|l| experts.iter().position(|&e| e == l.key().1));
            let pins = held
                .iter()
                .map(|l| self.pin(l))
                .collect::<Result<Vec<_>, _>>()?;
            let ws = held
                .iter()
                .map(|l| self.weights(layer, l.key().1, l.bytes()))
                .collect::<Result<Vec<_>, _>>()?;
            let r = group(&ws);
            drop(pins);
            return r;
        }
        Wave {
            provider: self,
            layer,
            first,
            f: each,
            batch: Mutex::new(batch),
            error: Mutex::new(None),
        }
        .run(held)
    }

    fn wants_lookahead(&self) -> bool {
        self.predictor.is_some()
    }

    fn lookahead(&self, layer: LayerIdx, router_logits: &[f32]) {
        let Some(p) = self.predictor else { return };
        let mut p = p.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let ids = p.lookahead.predict(layer.0 as usize, router_logits);
        p.pending = Some((layer, ids));
    }
}

/// Sampling settings.
#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    /// 0 for greedy (argmax, lowest id on ties).
    pub temperature: f32,
    /// Keep only the `top_k` likeliest tokens, before `top_p` (0: off).
    pub top_k: u32,
    /// Nucleus mass (1.0: off).
    pub top_p: f32,
    /// RNG seed: the same seed gives the same tokens.
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        }
    }
}

/// A first-come-first-served turn queue (a ticket lock): each caller takes a ticket and
/// waits until it is served. Unlike a plain mutex, no caller can be overtaken
/// indefinitely.
#[derive(Debug, Default)]
struct StepQueue {
    /// (next ticket to hand out, ticket being served).
    state: Mutex<(u64, u64)>,
    turn: std::sync::Condvar,
}

/// The current turn; the next ticket is served when it drops (also on panic).
struct Turn<'q>(&'q StepQueue);

impl StepQueue {
    fn wait(&self) -> Turn<'_> {
        let mut st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let ticket = st.0;
        st.0 += 1;
        while st.1 != ticket {
            st = self.turn.wait(st).unwrap_or_else(PoisonError::into_inner);
        }
        Turn(self)
    }
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        let mut st = self.0.state.lock().unwrap_or_else(PoisonError::into_inner);
        st.1 += 1;
        drop(st);
        self.0.turn.notify_all();
    }
}

/// How [`Engine::generate_with`] generates.
#[derive(Clone, Debug)]
pub struct GenerateOptions {
    /// Most tokens to generate.
    pub max_new: usize,
    /// Sampling.
    pub sampling: Sampling,
    /// Prompt tokens per forward pass. Smaller chunks let a cancellation land sooner;
    /// the bits are the same for every size.
    pub prefill_chunk: usize,
    /// When to stop, checked before every prefill chunk and every token.
    pub deadline: Option<Instant>,
    /// Who may reuse this generation's KV, and whose it may reuse. A reused prefix is
    /// faster, and so tells a caller that another request sent it: requests that must
    /// not learn of each other need different scopes.
    pub cache_scope: Option<String>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new: 128,
            sampling: Sampling::default(),
            prefill_chunk: 256,
            deadline: None,
            cache_scope: None,
        }
    }
}

/// Why a generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// The model produced an end-of-sequence token.
    Eos,
    /// `max_new` tokens were generated.
    Length,
    /// The context is full.
    ContextFull,
    /// The token callback asked to stop (a stop sequence, for example).
    Stopped,
    /// The cancellation flag was set.
    Cancelled,
    /// A stop sequence ended the text ([`Engine::run_request`] only).
    StopSequence,
    /// The deadline passed.
    TimedOut,
}

/// A finished generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generation {
    /// The generated ids (EOS excluded).
    pub tokens: Vec<u32>,
    /// Why it ended.
    pub finish: Finish,
    /// Prompt length in tokens.
    pub prompt_tokens: usize,
    /// Prompt tokens reused from retained KV, which no pass ran.
    pub cached_tokens: usize,
}

/// KV room held for one conversation (see [`Engine::reserve_kv`]).
///
/// It owns its claim on the engine's counter, so it can outlive a borrow of the engine:
/// a scheduler holds it for as long as a request runs.
#[derive(Debug)]
pub struct KvReservation {
    reserved: Arc<AtomicUsize>,
    positions: usize,
}

impl KvReservation {
    /// Positions held (whole KV chunks).
    #[must_use]
    pub const fn positions(&self) -> usize {
        self.positions
    }
}

impl Drop for KvReservation {
    fn drop(&mut self) {
        // Relaxed: see `Engine::reserve_kv`.
        self.reserved.fetch_sub(self.positions, Ordering::Relaxed);
    }
}

/// The memory plan and the ledger of what is actually allocated.
struct Memory {
    /// `None`: a fixed cache size was requested.
    plan: Option<MemoryPlan>,
    ledger: Arc<MemoryLedger>,
    charges: Mutex<Charges>,
}

/// Ledger charges the engine holds. The KV charge is the larger of the most recent
/// sequence's cache and what live [`KvReservation`]s hold across conversations, plus
/// retained KV.
struct Charges {
    _dense: Charge,
    experts: Charge,
    kv: Charge,
}

/// The pressure monitor's state, shared with its sampling thread.
struct Pressure {
    monitor: Mutex<PressureMonitor>,
    ledger: Arc<MemoryLedger>,
    retained: Option<Arc<Retained>>,
}

/// Finished KV kept for later prompts, priced in KV positions (ADR-016).
type Retained = Mutex<PrefixCache<KvCache>>;

fn lock_retained(r: &Retained) -> std::sync::MutexGuard<'_, PrefixCache<KvCache>> {
    // A poisoned lock means a panic mid-update; the index stays usable, and every value
    // in it is immutable.
    r.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Pressure {
    fn lock(&self) -> std::sync::MutexGuard<'_, PressureMonitor> {
        // A poisoned lock means the sampling thread panicked mid-update; the monitor's
        // state is plain data, so keep using it.
        self.monitor
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn observe(&self, available: Bytes) -> PressureDecision {
        let used = self.ledger.snapshot(Tier::Ram).expert_cache;
        let decision = self.lock().observe(available, used);
        // Speculation goes first (ADR-005), and retained KV is speculation. This runs on
        // the sampling thread, so an idle server gives it back too.
        if !decision.prefetch
            && let Some(r) = &self.retained
        {
            lock_retained(r).clear();
        }
        decision
    }
}

/// The sampling thread; stopped and joined on drop.
struct Watcher {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Watcher {
    fn start(pressure: Arc<Pressure>, interval: Duration) -> Result<Self, RuntimeError> {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("qubivex-pressure".into())
            .spawn(move || {
                let probe = SystemProbe::new(Host::live());
                while !flag.load(Ordering::Acquire) {
                    if let Ok(s) = probe.snapshot()
                        && let Probed::Known(a) = s.ram_available
                    {
                        pressure.observe(a);
                    }
                    std::thread::park_timeout(interval);
                }
            })
            .map_err(|e| RuntimeError::Memory(format!("cannot start the pressure monitor: {e}")))?;
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }
}

impl Drop for Watcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            t.thread().unpark();
            // A panicked monitor thread has nothing left to clean up.
            let _ = t.join();
        }
    }
}

/// A model ready to generate.
pub struct Engine {
    /// A discrete device running every matmul ([`BackendKind::Sim`] and
    /// [`BackendKind::Vulkan`], ADR-018).
    device: Option<Arc<OnDevice>>,
    verification: Verification,
    memory: Memory,
    /// `None` without a plan (a fixed cache was requested).
    pressure: Option<Arc<Pressure>>,
    _watcher: Option<Watcher>,
    meta: ModelMetadata,
    dense: Dense,
    source: Arc<ModelExpertSource>,
    store: ExpertStore,
    /// Expert layers, for turning a byte budget into per-layer slots.
    layers: Vec<LayerExperts>,
    /// The byte budget the pools were last sized for.
    cache_target: AtomicU64,
    /// KV positions reserved by live [`KvReservation`]s, across conversations.
    kv_reserved: Arc<AtomicUsize>,
    /// Forward passes run one at a time, first come first served (see
    /// [`Engine::forward`]).
    steps: StepQueue,
    /// Where concurrent [`Engine::forward`] steps meet to share a pass (ADR-017).
    combiner: Combiner,
    batch_tokens: usize,
    batch_wait: Duration,
    /// Finished KV kept for later prompts (`None`: off).
    retained: Option<Arc<Retained>>,
    /// KV bytes per position (keys and values of every layer).
    kv_position_bytes: u64,
    model: Transformer,
    tokenizer: Tokenizer,
    backend: Placed,
    max_context: usize,
    /// The forward pass runs inside this pool, so every matmul starts on a worker that
    /// joins the work instead of a caller blocked on a condition variable.
    pool: rayon::ThreadPool,
    /// `None`: prefetch is off.
    predictor: Option<Mutex<Predictor>>,
    /// The whole-layer Metal pass, with `MetalOps::layers`.
    gpu: Option<MetalForward>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("model", &self.meta.manifest.model.id)
            .field("store", &self.store)
            .finish_non_exhaustive()
    }
}

fn read_into(file: &File, offset: u64, b: &mut [u8]) -> std::io::Result<()> {
    #[cfg(unix)]
    std::os::unix::fs::FileExt::read_exact_at(file, b, offset)?;
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let len = b.len();
        let mut done = 0;
        while done < len {
            let n = file.seek_read(&mut b[done..], offset + done as u64)?;
            if n == 0 {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            done += n;
        }
    }
    Ok(())
}

impl Engine {
    /// Loads a validated model: reads the resident tensors, plans memory with them
    /// resident (ADR-005), sets up the expert store and the tokenizer.
    ///
    /// # Errors
    /// I/O, tokenizer and architecture errors; [`RuntimeError::Memory`] if the model
    /// does not fit the budget or available RAM cannot be measured.
    pub fn load(meta: ModelMetadata, opts: EngineOptions) -> Result<Self, RuntimeError> {
        Self::load_mirrored(meta, opts, &[])
    }

    /// As [`Engine::load`], reading experts from the model's files and, when a read
    /// fails, from copies of them under each of `mirrors` in turn
    /// ([`MirroredSource`], ADR-006 amendment 4). No mirrors: [`Engine::load`].
    ///
    /// # Errors
    /// As [`Engine::load`], and [`ModelExpertSource::mirror_at`]'s errors.
    pub fn load_mirrored(
        meta: ModelMetadata,
        opts: EngineOptions,
        mirrors: &[PathBuf],
    ) -> Result<Self, RuntimeError> {
        let (sums, verification) = match opts.verify {
            VerifyExperts::Off => (None, Verification::Off),
            VerifyExperts::Auto | VerifyExperts::Require => match ExpertSums::load(&meta)? {
                SumsState::Fresh(s) => {
                    let n = s.len();
                    (Some(s.sums().clone()), Verification::Verified(n))
                }
                SumsState::Missing => (None, Verification::Missing),
                SumsState::Stale(why) => (None, Verification::Stale(why)),
            },
        };
        if opts.verify == VerifyExperts::Require && sums.is_none() {
            return Err(RuntimeError::Model(format!(
                "expert checksums are required, but {}: run `qubivex models verify {}`",
                match &verification {
                    Verification::Stale(why) => why.clone(),
                    _ => format!("{} has no {SUMS_FILE}", meta.dir.display()),
                },
                meta.manifest.model.id
            )));
        }
        // Each copy is verified on its own, so a copy with corrupt bytes fails over to
        // the next.
        let verified = |s: Arc<dyn ExpertSource>| -> Arc<dyn ExpertSource> {
            match &sums {
                Some(sums) => Arc::new(VerifiedSource::new(s, sums.clone())),
                None => s,
            }
        };
        let copies = mirrors
            .iter()
            .map(|d| {
                let s: Arc<dyn ExpertSource> =
                    Arc::new(ModelExpertSource::mirror_at(&meta, opts.expert_io, d)?);
                Ok((d.display().to_string(), verified(s)))
            })
            .collect::<Result<Vec<_>, RuntimeError>>()?;
        let primary = meta.dir.display().to_string();
        let disk_limit = opts.disk_limit;
        let mut engine = Self::load_with_source(meta, opts, move |s| {
            let first = verified(s);
            let source: Arc<dyn ExpertSource> = if copies.is_empty() {
                first
            } else {
                Arc::new(MirroredSource::new(
                    std::iter::once((primary, first)).chain(copies),
                    MirrorPolicy::default(),
                ))
            };
            match disk_limit {
                Some(limit) => Arc::new(SlowDisk::new(source, limit)),
                None => source,
            }
        })?;
        engine.verification = verification;
        Ok(engine)
    }

    /// As [`Engine::load`], with the expert store reading through `wrap(source)`
    /// instead of the model's own source: for fault injection
    /// ([`qubivex_expert_stream::fault::FaultySource`]), checksums
    /// ([`qubivex_expert_stream::fault::VerifiedSource`]) or instrumentation. The
    /// wrapper must return the inner source's bytes for a read it reports as good;
    /// expert layouts still come from the model.
    ///
    /// # Errors
    /// As [`Engine::load`].
    pub fn load_with_source(
        meta: ModelMetadata,
        opts: EngineOptions,
        wrap: impl FnOnce(Arc<ModelExpertSource>) -> Arc<dyn ExpertSource>,
    ) -> Result<Self, RuntimeError> {
        let files = meta
            .tensors
            .files()
            .iter()
            .map(|f| {
                File::open(&f.path).map_err(|e| {
                    RuntimeError::Model(format!("cannot open {}: {e}", f.path.display()))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Layout first, then one arena, then the reads.
        let mut layout = Vec::new();
        let mut arena_len = 0usize;
        for (name, role) in &meta.roles {
            if matches!(
                role,
                TensorRole::RoutedExpert { .. } | TensorRole::RoutedExpertStack { .. }
            ) {
                continue;
            }
            let t = meta
                .tensors
                .get(name)
                .ok_or_else(|| RuntimeError::Model(format!("tensor {name:?} not indexed")))?;
            let bad = |e: &dyn std::fmt::Display| RuntimeError::Model(format!("{name:?}: {e}"));
            let mut desc = t.tensor.desc().map_err(|e| bad(&e))?;
            match *desc.dims() {
                [n] => {
                    desc = desc
                        .reshape(&Shape::new(&[1, n]).map_err(|e| bad(&e))?)
                        .map_err(|e| bad(&e))?
                }
                [_, _] => {}
                ref d => {
                    return Err(bad(&format!(
                        "has shape {d:?}, expected a matrix or vector"
                    )));
                }
            }
            let len = usize::try_from(t.tensor.bytes).map_err(|_| bad(&"is too large"))?;
            let at = arena_len.div_ceil(DENSE_ALIGN) * DENSE_ALIGN;
            arena_len = at
                .checked_add(len)
                .ok_or_else(|| bad(&"overflows the dense arena"))?;
            layout.push((name.clone(), desc, at..at + len, t.file, t.tensor.offset));
        }
        let mut arena = PageAligned::zeroed(arena_len);
        let mut tensors = HashMap::with_capacity(layout.len());
        // The ledger is charged the tensor bytes, as the plan prices them. The arena's
        // alignment slack (under 256 bytes per tensor, plus at most two pages) is not.
        let mut dense_bytes = 0u64;
        for (name, desc, range, file, offset) in layout {
            dense_bytes += range.len() as u64;
            let bad = |e: &dyn std::fmt::Display| RuntimeError::Model(format!("{name:?}: {e}"));
            read_into(
                &files[file],
                offset,
                &mut arena.as_mut_slice()[range.clone()],
            )
            .map_err(|e| bad(&format!("cannot read: {e}")))?;
            TensorView::new(desc, &arena.as_slice()[range.clone()]).map_err(|e| bad(&e))?;
            tensors.insert(name, (desc, range));
        }
        let dense = Dense {
            arena: Arc::new(arena),
            tensors,
        };
        let model = Transformer::new(&meta.config, &dense, &Placed::cpu())?;
        // The whole-layer pass is built once the context limit is known (below).
        let mut layers_on = None;
        let backend = match opts.backend {
            // Placed once the ledger exists (below).
            BackendKind::Cpu | BackendKind::Sim | BackendKind::Vulkan => Placed::cpu(),
            // Whole layers exist for routed experts only; a dense model places its
            // matmuls one by one instead (below), with the same bits.
            BackendKind::Metal if opts.metal_ops.layers && model.is_moe() => {
                let be = MetalBackend::new().map_err(|e| metal_err(&e))?;
                let host = be
                    .map_host(Arc::clone(&dense.arena))
                    .map_err(|e| metal_err(&e))?;
                layers_on = Some((be, host));
                Placed::cpu()
            }
            BackendKind::Metal => {
                let weights = model
                    .dense_matmuls()
                    .into_iter()
                    .map(|(kind, name)| Ok((kind, dense.view(&name)?)))
                    .collect::<Result<Vec<_>, ArchError>>()?;
                Placed::metal(&dense.arena, weights, opts.metal_ops)
                    .map_err(|e| RuntimeError::Model(format!("cannot place on Metal: {e}")))?
            }
        };
        let source = Arc::new(ModelExpertSource::with_io(&meta, opts.expert_io)?);
        let footprint = ModelFootprint::from_metadata(&meta)
            .map_err(|e| RuntimeError::Memory(e.to_string()))?;
        let model_ctx = meta
            .manifest
            .context_length
            .or(meta.config.dims.max_context)
            .and_then(|c| usize::try_from(c).ok())
            .unwrap_or(4096);
        let wanted_ctx = opts.max_context.map(|c| c.min(model_ctx));
        // A discrete device is opened before planning: the plan needs its memory.
        let discrete: Option<Arc<dyn DiscreteDevice>> = match opts.backend {
            BackendKind::Sim => Some(Arc::new(SimDevice::new(
                opts.sim.unwrap_or_else(|| SimConfig::new(DEFAULT_SIM_VRAM)),
            ))),
            BackendKind::Vulkan => Some(Arc::new(VulkanDevice::new(opts.vulkan).map_err(|e| {
                RuntimeError::Model(format!("cannot open the Vulkan device: {e}"))
            })?)),
            BackendKind::Cpu | BackendKind::Metal => None,
        };
        let vram = discrete.as_ref().map(|d| d.usage().capacity);
        let (plan, capacity, max_context) = match opts.expert_cache_bytes {
            Some(c) => (None, c, wanted_ctx.unwrap_or(model_ctx)),
            None => {
                let plan = plan_at_load(&footprint, &opts, wanted_ctx, vram)?;
                let capacity = plan
                    .expert_cache_for(0)
                    .map_err(|r| RuntimeError::Memory(r.to_string()))?
                    .0;
                let ctx = usize::try_from(plan.context).unwrap_or(model_ctx);
                (Some(plan), capacity, ctx)
            }
        };
        let top_k = model.moe_config().map_or(0, |c| c.top_k as usize);
        let cpu_pass = cpu_pass_reason(&footprint.layers, capacity, top_k);
        let (backend, gpu) = match layers_on {
            Some((be, _)) if cpu_pass.is_some() => (
                Placed::fallback(be.info().name, opts.metal_ops, cpu_pass.unwrap_or_default()),
                None,
            ),
            Some((be, host)) => {
                let cfg = model.moe_config().ok_or_else(|| {
                    RuntimeError::Model("whole Metal layers need routed experts".into())
                })?;
                let g = MetalForward::new(be, host, cfg, &dense, max_context, opts.metal_ops.token)
                    .map_err(|e| metal_err(&e))?;
                let (tensors, bytes) = g.placed();
                let p = Placed::layers(g.device(), tensors, bytes, opts.metal_ops);
                (p, Some(g))
            }
            None => (backend, None),
        };
        let kv_position_bytes = footprint.kv_bytes_per_token.0;
        let layers = footprint.layers;
        let io_threads = opts.io_threads.unwrap_or(DEFAULT_IO_THREADS);
        let store = ExpertStore::with_io_threads(
            wrap(Arc::clone(&source)),
            &pools_for(&layers, capacity),
            opts.cache_policy,
            io_threads,
        )
        .map_err(|e| RuntimeError::Model(format!("cannot start {io_threads} I/O threads: {e}")))?;
        store
            .set_read_timeout(opts.read_timeout.unwrap_or(ReadTimeout::DEFAULT))
            .map_err(|e| RuntimeError::Model(e.to_string()))?;
        store.set_read_retries(opts.read_retries.unwrap_or(DEFAULT_READ_RETRIES));
        let ledger = MemoryLedger::new(u32::from(discrete.is_some()));
        let device = match discrete {
            None => None,
            Some(dev) => {
                let budget = match plan.as_ref().and_then(|p| p.vram.first()) {
                    Some(v) => v.expert_cache.bytes.0,
                    // A fixed RAM cache: the device's room after the dense copy.
                    None => opts
                        .vram_budget
                        .unwrap_or(dev.usage().capacity)
                        .saturating_sub(dense.arena.len() as u64)
                        .saturating_sub(VRAM_ACTIVATION_RESERVE.0),
                };
                let on = OnDevice::new(dev, Arc::clone(&ledger), &dense.arena, budget)
                    .map_err(|e| RuntimeError::Model(format!("cannot place on the device: {e}")))?;
                Some(Arc::new(on))
            }
        };
        let backend = match &device {
            Some(d) => Placed::discrete(Arc::clone(d), opts.backend, model.dense_matmuls().len()),
            None => backend,
        };
        let charges = Charges {
            _dense: ledger.charge(Tier::Ram, Owner::Dense, Bytes(dense_bytes)),
            experts: ledger.charge(Tier::Ram, Owner::ExpertCache, Bytes(0)),
            kv: ledger.charge(Tier::Ram, Owner::Kv, Bytes(0)),
        };
        let retained = (opts.prefix_cache_bytes > 0).then(|| {
            let positions = opts.prefix_cache_bytes / kv_position_bytes.max(1);
            Arc::new(Mutex::new(PrefixCache::new(positions)))
        });
        let pressure = plan.as_ref().map(|p| {
            Arc::new(Pressure {
                monitor: Mutex::new(PressureMonitor::new(opts.pressure, p.ram.cache_floor.bytes)),
                ledger: Arc::clone(&ledger),
                retained: retained.clone(),
            })
        });
        let interval = opts.pressure_interval_ms.unwrap_or(500);
        let watcher = match &pressure {
            Some(p) if interval > 0 => Some(Watcher::start(
                Arc::clone(p),
                Duration::from_millis(interval),
            )?),
            _ => None,
        };
        let tokenizer = load_tokenizer(&meta)?;
        let threads = opts
            .threads
            .filter(|&t| t > 0)
            .unwrap_or_else(default_threads);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("qubivex-compute-{i}"))
            .build()
            .map_err(|e| {
                RuntimeError::Model(format!("cannot start {threads} compute threads: {e}"))
            })?;
        // Prefetch predicts routed experts; a dense model has none to predict.
        let predictor = opts.prefetch.zip(model.moe_config()).map(|(o, c)| {
            Mutex::new(Predictor {
                lookahead: Lookahead::new(o.lookahead, c.layers as usize, c.top_k as usize),
                guard: o.guard,
                pending: None,
                issued: None,
                stats: PredictionStats::default(),
                saturation: o.throttle.then(SaturationGauge::default),
                epoch: Instant::now(),
                passes: 0,
                held_passes: 0,
            })
        });
        Ok(Self {
            device,
            gpu,
            predictor,
            pressure,
            _watcher: watcher,
            memory: Memory {
                plan,
                ledger,
                charges: Mutex::new(charges),
            },
            pool,
            max_context,
            meta,
            dense,
            source,
            store,
            layers,
            cache_target: AtomicU64::new(capacity),
            kv_reserved: Arc::new(AtomicUsize::new(0)),
            retained,
            steps: StepQueue::default(),
            combiner: Combiner::default(),
            batch_tokens: opts.batch_tokens.unwrap_or(DEFAULT_BATCH_TOKENS).max(1),
            batch_wait: Duration::from_micros(opts.batch_wait_us.unwrap_or(DEFAULT_BATCH_WAIT_US)),
            kv_position_bytes,
            verification: Verification::Off,
            model,
            tokenizer,
            backend,
        })
    }

    /// The discrete device's memory, copies and expert cache, with
    /// [`BackendKind::Sim`] or [`BackendKind::Vulkan`].
    #[must_use]
    pub fn device_report(&self) -> Option<DeviceReport> {
        self.device.as_ref().map(|d| d.report())
    }

    /// Forward passes so far that ran as one Metal submission with routing on the GPU
    /// ([`MetalOps::token`], ADR-011 amendment 4); 0 on any other placement.
    #[must_use]
    pub fn gpu_routed_passes(&self) -> u64 {
        self.gpu.as_ref().map_or(0, MetalForward::routed_passes)
    }

    /// What the execution plan placed on a device.
    #[must_use]
    pub const fn placement(&self) -> &PlacementReport {
        self.backend.report()
    }

    /// Expert I/O threads in use.
    #[must_use]
    pub fn io_threads(&self) -> usize {
        self.store.io_threads()
    }

    /// How experts are read from the weight files.
    #[must_use]
    pub fn expert_io(&self) -> IoPath {
        self.source.io_path()
    }

    /// Compute threads in use.
    #[must_use]
    pub fn threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// The model metadata.
    #[must_use]
    pub const fn metadata(&self) -> &ModelMetadata {
        &self.meta
    }

    /// The tokenizer.
    #[must_use]
    pub const fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Whether expert reads are verified against recorded checksums. An engine made
    /// with [`Engine::load_with_source`] reports [`Verification::Off`]: its wrapper
    /// decides.
    #[must_use]
    pub const fn verification(&self) -> &Verification {
        &self.verification
    }

    /// Expert store counters.
    #[must_use]
    pub fn store_stats(&self) -> StoreStats {
        self.store.stats()
    }

    /// What the expert store holds now, by state (all-zero `leased`, `loading` and
    /// `failed` between requests, whatever the reads did).
    ///
    /// # Errors
    /// [`RuntimeError::Model`] if the store's lock is poisoned.
    pub fn store_occupancy(&self) -> Result<Occupancy, RuntimeError> {
        self.store
            .occupancy()
            .map_err(|e| RuntimeError::Model(e.to_string()))
    }

    /// Prefetch counters, if prefetch is on.
    #[must_use]
    pub fn prefetch_report(&self) -> Option<PrefetchReport> {
        self.predictor.as_ref().map(|p| {
            let p = p.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            PrefetchReport {
                prediction: p.stats,
                store: self.store.stats().prefetch,
                passes: p.passes,
                held_passes: p.held_passes,
            }
        })
    }

    /// Whether prefetch is on (it still pauses under memory pressure).
    #[must_use]
    pub const fn prefetch_enabled(&self) -> bool {
        self.predictor.is_some()
    }

    /// The expert cache capacity in bytes now: the slots of every layer's pool (the
    /// planner lowers it as the KV grows).
    #[must_use]
    pub fn expert_cache_bytes(&self) -> u64 {
        self.store.capacity()
    }

    /// The per-layer expert pools now.
    #[must_use]
    pub fn expert_pools(&self) -> Vec<PoolSpec> {
        self.store.pools().unwrap_or_default()
    }

    /// Starts (or stops) recording every expert acquire, for trace replay.
    pub fn record_routing(&self, on: bool) {
        // A poisoned store fails the next forward pass anyway.
        let _ = self.store.record(on);
    }

    /// The expert acquires recorded since the last call (see
    /// [`Engine::record_routing`]).
    #[must_use]
    pub fn take_routing(&self) -> Vec<ExpertKey> {
        self.store.take_trace().unwrap_or_default()
    }

    /// The memory plan made at load, unless a fixed cache size was requested.
    #[must_use]
    pub const fn memory_plan(&self) -> Option<&MemoryPlan> {
        self.memory.plan.as_ref()
    }

    /// Bytes actually allocated, per tier and owner.
    #[must_use]
    pub const fn ledger(&self) -> &Arc<MemoryLedger> {
        &self.memory.ledger
    }

    /// The pressure monitor's latest decision (`None` without a plan).
    #[must_use]
    pub fn pressure(&self) -> Option<PressureDecision> {
        self.pressure.as_ref().map(|p| p.lock().decision())
    }

    /// Feeds the pressure monitor one observation of available RAM, as its thread
    /// does, and returns the decision. A cache cap applies from the next forward pass.
    pub fn observe_pressure(&self, available: Bytes) -> Option<PressureDecision> {
        self.pressure.as_ref().map(|p| p.observe(available))
    }

    /// Replaces the pressure thresholds.
    pub fn set_pressure_config(&self, config: PressureConfig) {
        if let Some(p) = &self.pressure {
            p.lock().set_config(config);
        }
    }

    /// Refuses a new request under critical memory pressure (ADR-005), or when every
    /// expert I/O thread is blocked in a read past its deadline. Active requests are
    /// never stopped.
    ///
    /// # Errors
    /// [`RuntimeError::Memory`] when the pressure monitor refuses admission;
    /// [`RuntimeError::Storage`] when every I/O thread is stuck.
    pub fn admit(&self) -> Result<(), RuntimeError> {
        // Every I/O thread blocked in a read past its deadline: a new request would
        // only queue behind them and time out.
        let threads = self.store.io_threads();
        if threads > 0 {
            let stuck = self.store_occupancy()?.stuck;
            if stuck >= threads {
                return Err(RuntimeError::Storage(format!(
                    "new request refused: all {threads} expert I/O threads are blocked in \
                     reads past their deadline; the drive may have failed"
                )));
            }
        }
        match self.pressure() {
            Some(d) if !d.admit => Err(RuntimeError::Memory(format!(
                "new request refused: memory pressure is {:?}; the expert cache is at its floor",
                d.level
            ))),
            _ => Ok(()),
        }
    }

    /// Shrinks the expert cache to leave room for `cache` growing by `tokens`
    /// (Colibri's `kv_room_fit`), or refuses if that would cross the floor.
    fn fit_room(&self, cache: &KvCache, tokens: usize) -> Result<(), RuntimeError> {
        self.fit_positions(KvCache::reserved_for(
            cache.len() + tokens,
            cache.capacity(),
        ))
    }

    /// [`Engine::fit_room`] for caches that reserve `positions` together after a pass.
    fn fit_positions(&self, positions: usize) -> Result<(), RuntimeError> {
        let Some(plan) = &self.memory.plan else {
            return Ok(());
        };
        // With concurrent conversations, the plan prices every reservation together.
        let projected = positions.max(self.kv_reserved.load(Ordering::Relaxed));
        let retained = self.make_room(projected);
        let mut cap = plan
            .expert_cache_for(projected as u64 + retained)
            .map_err(|r| RuntimeError::Memory(r.to_string()))?;
        if let Some(limit) = self.pressure().and_then(|d| d.cache_cap) {
            cap = cap.min(limit);
        }
        if self.cache_target.swap(cap.0, Ordering::Relaxed) != cap.0 {
            self.store
                .set_pools(&pools_for(&self.layers, cap.0))
                .map_err(|e| RuntimeError::Memory(e.to_string()))?;
        }
        Ok(())
    }

    /// Evicts retained KV until the plan holds it beside `live` positions, and returns
    /// the positions still retained.
    fn make_room(&self, live: usize) -> u64 {
        let Some(r) = &self.retained else {
            return 0;
        };
        let mut r = lock_retained(r);
        if let Some(plan) = &self.memory.plan {
            let live = live as u64;
            while plan.expert_cache_for(live + r.cost()).is_err() && r.evict_oldest() {}
        }
        r.cost()
    }

    /// Fills the empty `cache` from retained KV sharing a prefix with `prompt`, and
    /// returns how many positions it holds. The request's reservation already prices
    /// the copy, and its first pass sizes the expert cache.
    ///
    /// The copy takes a turn like a pass. No test shows that it must: the device lock
    /// orders submissions. It is kept because device KV has only ever been allocated
    /// under the turn, and the cost is at most one pass of waiting.
    fn restore(
        &self,
        scope: Option<&str>,
        prompt: &[u32],
        cache: &mut KvCache,
    ) -> Result<usize, RuntimeError> {
        let Some(r) = &self.retained else {
            return Ok(0);
        };
        let Some(hit) = lock_retained(r).lookup(scope, prompt) else {
            return Ok(0);
        };
        let _turn = self.steps.wait();
        if let Some(g) = &self.gpu {
            g.restore(&hit.value, cache, hit.reuse)?;
        } else {
            cache.restore_from(&hit.value, hit.reuse)?;
        }
        self.account(cache.allocated_bytes());
        Ok(hit.reuse)
    }

    /// Keeps a finished `cache`, which ran `tokens`, for later prompts. It moves, with no
    /// copy. The positions may still be reserved for the request that just ended, so
    /// that reservation is not counted twice.
    fn retain(&self, scope: Option<&str>, tokens: Vec<u32>, cache: KvCache) {
        let Some(r) = &self.retained else {
            return;
        };
        if self.pressure().is_some_and(|d| !d.prefetch) {
            return;
        }
        let cost = KvCache::reserved_for(cache.len(), cache.capacity());
        lock_retained(r).insert(scope, tokens, cost as u64, cache);
        self.make_room(
            self.kv_reserved
                .load(Ordering::Relaxed)
                .saturating_sub(cost),
        );
        self.account(0);
    }

    /// Prefix reuse so far, and what is retained now. Zeros when it is off.
    #[must_use]
    pub fn prefix_stats(&self) -> PrefixStats {
        self.retained
            .as_ref()
            .map_or_else(PrefixStats::default, |r| lock_retained(r).stats())
    }

    /// Drops all retained KV.
    pub fn clear_prefix_cache(&self) {
        if let Some(r) = &self.retained {
            lock_retained(r).clear();
        }
    }

    /// Updates the ledger from the store, retained KV and `cache_bytes` of a sequence's
    /// KV cache.
    fn account(&self, cache_bytes: u64) {
        // A poisoned lock means another thread panicked mid-update; the counters are
        // diagnostics, so skip rather than propagate.
        if let Ok(mut c) = self.memory.charges.lock() {
            c.experts.resize(Bytes(self.store.used().unwrap_or(0)));
            let reserved = self.kv_reserved.load(Ordering::Relaxed) as u64;
            let live = cache_bytes.max(reserved.saturating_mul(self.kv_position_bytes));
            let retained = self
                .prefix_stats()
                .cost
                .saturating_mul(self.kv_position_bytes);
            c.kv.resize(Bytes(live.saturating_add(retained)));
        }
    }

    /// A fresh KV cache.
    #[must_use]
    pub fn new_cache(&self) -> KvCache {
        self.model.new_cache(self.max_context)
    }

    /// One forward pass (see [`Transformer::forward`]).
    ///
    /// Concurrent callers (conversations of a server) share passes (ADR-017): each
    /// posts its step and waits, and the caller that takes the turn runs every waiting
    /// step in one [`Transformer::forward_batch`], decode steps first, then prefill
    /// within [`EngineOptions::batch_tokens`]. Each caller gets the logits its step
    /// gives alone, with the same bits, and its own error if its step alone is bad
    /// (a full context, a token out of range); an error inside the pass reaches every
    /// step in it, and no cache in it advances. Nothing waits for a batch to fill: a
    /// lone caller runs at once.
    ///
    /// The whole-layer Metal pass combines the same way (ADR-017 §6). A traced pass
    /// runs one caller at a time in arrival order instead; [`Engine::generate_with`]
    /// splits long prompts into chunks so a prompt never holds the turn for long.
    ///
    /// # Errors
    /// Architecture errors; a traced pass with layers on Metal.
    pub fn forward(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        logits: Logits,
        trace: Option<Tracer<'_>>,
    ) -> Result<Vec<f32>, RuntimeError> {
        if trace.is_none() {
            let step = Step {
                tokens: tokens.to_vec(),
                cache: std::mem::take(cache),
                logits,
            };
            let (back, out) =
                self.combiner
                    .run(step, self.batch_tokens, self.batch_wait, |steps| {
                        self.run_steps(steps)
                    });
            *cache = back;
            return out;
        }
        if self.gpu.is_some() {
            return Err(RuntimeError::Model(
                "tracing needs the CPU forward pass (not --metal-ops layers)".into(),
            ));
        }
        let _turn = self.steps.wait();
        let predictor = self.start_pass();
        let provider = StoreProvider {
            store: &self.store,
            source: &self.source,
            predictor,
            device: self.device.as_deref(),
        };
        let src = Sources {
            dense: &self.dense,
            experts: &provider,
            backend: &self.backend,
        };
        self.fit_room(cache, tokens.len())?;
        let out = self
            .pool
            .install(|| self.model.forward(tokens, cache, src, logits, trace));
        // Failed passes too: the ledger follows what the store and cache hold.
        self.account(cache.allocated_bytes());
        Ok(out?)
    }

    /// Runs the steps the combining point gathered as one pass, under the turn.
    fn run_steps(&self, steps: &mut [Step]) -> Vec<Outcome> {
        let _turn = self.steps.wait();
        // Each step's own faults are answered alone (section 4).
        let mut out: Vec<Option<Outcome>> = steps
            .iter()
            .map(|st| {
                self.model
                    .check_segment(&st.tokens, &st.cache, &self.dense)
                    .err()
                    .map(|e| Outcome::Done(Err(e.into())))
            })
            .collect();
        // The memory plan prices the pass's caches together (section 5). What does not
        // fit waits for the next pass, latest first; a step that cannot fit alone gets
        // the plan's refusal.
        let reserved = |out: &[Option<Outcome>]| -> usize {
            steps
                .iter()
                .zip(out)
                .filter(|(_, o)| o.is_none())
                .map(|(st, _)| {
                    KvCache::reserved_for(st.cache.len() + st.tokens.len(), st.cache.capacity())
                })
                .sum()
        };
        loop {
            let live: Vec<usize> = (0..steps.len()).filter(|&i| out[i].is_none()).collect();
            let Some(&last) = live.last() else { break };
            match self.fit_positions(reserved(&out)) {
                Ok(()) => break,
                Err(e) if live.len() == 1 => out[last] = Some(Outcome::Done(Err(e))),
                Err(_) => out[last] = Some(Outcome::Deferred),
            }
        }
        let run: Vec<usize> = (0..steps.len()).filter(|&i| out[i].is_none()).collect();
        if !run.is_empty() {
            let mut segments: Vec<Segment<'_>> = steps
                .iter_mut()
                .zip(&out)
                .filter(|(_, o)| o.is_none())
                .map(|(st, _)| Segment {
                    tokens: &st.tokens,
                    cache: &mut st.cache,
                    logits: st.logits,
                })
                .collect();
            let result = self.pass(&mut segments);
            let bytes = segments.iter().map(|g| g.cache.allocated_bytes()).sum();
            drop(segments);
            // Failed passes too: the ledger follows what the store and caches hold.
            self.account(bytes);
            match result {
                Ok(logits) => {
                    for (i, l) in run.into_iter().zip(logits) {
                        out[i] = Some(Outcome::Done(Ok(l)));
                    }
                }
                Err(e) => {
                    // The model's error, not a sequence's: every step in the pass gets
                    // it, as it would have alone.
                    for &i in &run[1..] {
                        out[i] = Some(Outcome::Done(Err(copy_arch_err(&e).into())));
                    }
                    out[run[0]] = Some(Outcome::Done(Err(e.into())));
                }
            }
        }
        out.into_iter()
            .map(|o| o.unwrap_or(Outcome::Deferred))
            .collect()
    }

    /// How [`Engine::forward`] callers have shared passes so far.
    #[must_use]
    pub fn batch_stats(&self) -> BatchStats {
        self.combiner.stats()
    }

    /// The predictor a pass may use, with its state reset; `None` while memory
    /// pressure or a saturated disk holds speculation, which also drops what is queued
    /// (ADR-005).
    fn start_pass(&self) -> Option<&Mutex<Predictor>> {
        let allowed = self.pressure().is_none_or(|d| d.prefetch) && !self.disk_saturated();
        if !allowed && self.predictor.is_some() {
            let _ = self.store.cancel_prefetch(None);
        }
        let predictor = self.predictor.as_ref().filter(|_| allowed);
        if let Some(p) = predictor {
            let mut p = p.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            // A prediction never crosses a forward pass: the next one starts at layer 0.
            p.pending = None;
            p.issued = None;
        }
        predictor
    }

    /// One forward pass shared by several sequences, each with its own cache (see
    /// [`Transformer::forward_batch`]). Each segment's logits have the bits
    /// [`Engine::forward`] gives that segment alone.
    ///
    /// For a caller driving its own batches; [`Engine::forward`] combines concurrent
    /// callers by itself. Runs on the CPU pass and on whole Metal layers alike.
    ///
    /// # Errors
    /// Architecture errors.
    pub fn forward_batch(
        &self,
        segments: &mut [Segment<'_>],
    ) -> Result<Vec<Vec<f32>>, RuntimeError> {
        let _turn = self.steps.wait();
        for g in segments.iter() {
            self.fit_room(g.cache, g.tokens.len())?;
        }
        let out = self.pass(segments);
        let bytes = segments.iter().map(|g| g.cache.allocated_bytes()).sum();
        self.account(bytes);
        Ok(out?)
    }

    /// One pass over `segments`, on whole Metal layers or the CPU, under the turn.
    fn pass(&self, segments: &mut [Segment<'_>]) -> Result<Vec<Vec<f32>>, ArchError> {
        let predictor = self.start_pass();
        if let Some(g) = &self.gpu {
            return g.forward_batch(segments, &self.dense, &self.store, &self.source);
        }
        let provider = StoreProvider {
            store: &self.store,
            source: &self.source,
            predictor,
            device: self.device.as_deref(),
        };
        let src = Sources {
            dense: &self.dense,
            experts: &provider,
            backend: &self.backend,
        };
        self.pool
            .install(|| self.model.forward_batch(segments, src))
    }

    /// Generates up to `max_new` tokens after `prompt`, calling `on_token` for each;
    /// stops early at an end-of-sequence token or when `on_token` returns `false`.
    /// Returns the generated ids (EOS excluded).
    ///
    /// # Errors
    /// Architecture errors, or an empty prompt.
    pub fn generate(
        &self,
        prompt: &[u32],
        max_new: usize,
        sampling: Sampling,
        on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>, RuntimeError> {
        let opts = GenerateOptions {
            max_new,
            sampling,
            ..GenerateOptions::default()
        };
        Ok(self
            .generate_with(prompt, &opts, &AtomicBool::new(false), on_token)?
            .tokens)
    }

    /// [`Engine::generate`] for a server: the prompt runs in chunks of
    /// `opts.prefill_chunk` tokens, and `cancel` is checked before every chunk and
    /// every generated token, so a cancelled request stops within one chunk or one
    /// token. Chunking never changes bits (a prompt's logits equal token-by-token
    /// decoding, invariant 1). Reports why generation ended.
    ///
    /// # Errors
    /// Architecture errors, or an empty prompt.
    pub fn generate_with(
        &self,
        prompt: &[u32],
        opts: &GenerateOptions,
        cancel: &AtomicBool,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Generation, RuntimeError> {
        if prompt.is_empty() {
            return Err(RuntimeError::Model("the prompt is empty".into()));
        }
        self.admit()?;
        let scope = opts.cache_scope.as_deref();
        let mut cache = self.new_cache();
        if prompt.len() > cache.capacity() {
            return Err(RuntimeError::Model(format!(
                "the prompt has {} tokens; the context holds {}",
                prompt.len(),
                cache.capacity()
            )));
        }
        let cached_tokens = self.restore(scope, prompt, &mut cache)?;
        // However it ended, the cache holds the first `len` tokens of the prompt and the
        // output: the last token generated may not have run.
        let done = |tokens: Vec<u32>, finish, cache: KvCache| {
            self.end_turn();
            if self.retained.is_some() {
                let mut fed = prompt.to_vec();
                fed.extend_from_slice(&tokens);
                fed.truncate(cache.len());
                self.retain(scope, fed, cache);
            }
            Ok(Generation {
                tokens,
                finish,
                prompt_tokens: prompt.len(),
                cached_tokens,
            })
        };
        let mut rng = Rng(opts.sampling.seed ^ 0x9e37_79b9_7f4a_7c15);
        let mut logits = Vec::new();
        let late = || opts.deadline.is_some_and(|d| Instant::now() >= d);
        for chunk in prompt[cached_tokens..].chunks(opts.prefill_chunk.max(1)) {
            // Acquire: pairs with the canceller's Release store.
            if cancel.load(Ordering::Acquire) {
                return done(Vec::new(), Finish::Cancelled, cache);
            }
            if late() {
                return done(Vec::new(), Finish::TimedOut, cache);
            }
            logits = self.forward(chunk, &mut cache, Logits::Last, None)?;
        }
        let mut out = Vec::new();
        loop {
            if out.len() == opts.max_new {
                return done(out, Finish::Length, cache);
            }
            if cancel.load(Ordering::Acquire) {
                return done(out, Finish::Cancelled, cache);
            }
            let next = sample(&logits, opts.sampling, &mut rng);
            if self.tokenizer.eos_ids().contains(&next) {
                return done(out, Finish::Eos, cache);
            }
            out.push(next);
            if !on_token(next) {
                return done(out, Finish::Stopped, cache);
            }
            if out.len() == opts.max_new {
                return done(out, Finish::Length, cache);
            }
            if cache.len() == cache.capacity() {
                return done(out, Finish::ContextFull, cache);
            }
            if late() {
                return done(out, Finish::TimedOut, cache);
            }
            logits = self.forward(&[next], &mut cache, Logits::Last, None)?;
        }
    }

    /// Reserves KV room for a conversation of up to `positions` positions (prompt plus
    /// generated tokens), alongside every other live reservation. Admission control
    /// for concurrent conversations: the memory plan must still keep one expert slot
    /// per layer with all reservations' KV in RAM (ADR-005). Without a plan (a fixed
    /// expert cache) every reservation is granted. Dropping the reservation returns the
    /// room. Retained KV never stands in the way: it is evicted to make room.
    ///
    /// # Errors
    /// [`RuntimeError::Memory`] if the reservations together would not fit.
    pub fn reserve_kv(&self, positions: usize) -> Result<KvReservation, RuntimeError> {
        let want = KvCache::reserved_for(positions.min(self.max_context), self.max_context);
        let mut cur = self.kv_reserved.load(Ordering::Relaxed);
        loop {
            let total = cur.saturating_add(want);
            if let Some(plan) = &self.memory.plan {
                plan.expert_cache_for(total as u64).map_err(|r| {
                    RuntimeError::Memory(format!(
                        "no room for another conversation of {positions} positions while \
                         {cur} are reserved: {r}"
                    ))
                })?;
            }
            // Relaxed: the counter only prices memory; no other data is published.
            match self.kv_reserved.compare_exchange_weak(
                cur,
                total,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.make_room(total);
                    return Ok(KvReservation {
                        reserved: Arc::clone(&self.kv_reserved),
                        positions: want,
                    });
                }
                Err(now) => cur = now,
            }
        }
    }

    /// KV positions reserved across live conversations.
    #[must_use]
    pub fn kv_reserved(&self) -> usize {
        self.kv_reserved.load(Ordering::Relaxed)
    }

    /// The context limit of a conversation, in positions.
    #[must_use]
    pub const fn max_context(&self) -> usize {
        self.max_context
    }

    /// Whether the prefetch throttle holds speculation for this forward pass.
    fn disk_saturated(&self) -> bool {
        let Some(p) = &self.predictor else {
            return false;
        };
        let mut p = p.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let wait_ns = self.store.stats().wait_ns;
        let now_ns = u64::try_from(p.epoch.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let held = p
            .saturation
            .as_mut()
            .is_some_and(|g| g.observe(now_ns, wait_ns));
        p.passes += 1;
        p.held_passes += u64::from(held);
        held
    }

    /// Ends a turn: queued prefetches are dropped, since the next request starts
    /// elsewhere. [`Engine::generate`] calls it; callers driving
    /// [`Engine::forward`] themselves call it between requests.
    pub fn end_turn(&self) {
        if let Some(p) = &self.predictor {
            let mut p = p.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(g) = &mut p.saturation {
                g.pause();
            }
            drop(p);
            // Advisory: a poisoned store fails the next forward pass anyway.
            let _ = self.store.cancel_prefetch(None);
        }
    }
}

/// A copy of a pass's error for each other step in that pass. `ArchError` holds a
/// kernel error that cannot be cloned; it is kept as its message.
fn copy_arch_err(e: &ArchError) -> ArchError {
    match e {
        ArchError::Missing(w) => ArchError::Missing(w.clone()),
        ArchError::Expert {
            layer,
            expert,
            detail,
        } => ArchError::Expert {
            layer: *layer,
            expert: *expert,
            detail: detail.clone(),
        },
        other => ArchError::Input(other.to_string()),
    }
}

fn metal_err(e: &dyn std::fmt::Display) -> RuntimeError {
    RuntimeError::Model(format!("cannot place on Metal: {e}"))
}

/// Plans memory with the dense weights resident: available RAM is measured now
/// (Colibri's after-load measurement).
/// Per-layer pools for an expert cache of `bytes`, spread as the plan spreads them
/// (at least one slot per layer).
fn pools_for(layers: &[LayerExperts], bytes: u64) -> Vec<PoolSpec> {
    let slots = qubivex_memory::distribute(layers, Bytes(bytes), None);
    layers
        .iter()
        .zip(slots)
        .map(|(l, slots)| PoolSpec {
            layer: l.layer,
            slots,
            slot_bytes: l.expert_bytes.0,
        })
        .collect()
}

/// Why a cache of `bytes` should run the CPU pass instead of whole Metal layers, if it
/// should. With fewer slots than a token routes, experts stream through the pool and
/// every layer's GPU round trip costs more than the GPU saves (ADR-011 amendment 3).
/// With fewer than half a layer's experts, decode waits on loads that the CPU pass
/// overlaps with compute (ADR-011 amendment 6).
#[must_use]
pub fn cpu_pass_reason(layers: &[LayerExperts], bytes: u64, top_k: usize) -> Option<String> {
    let slots = qubivex_memory::distribute(layers, Bytes(bytes), None);
    let in_layers = || layers.iter().zip(&slots);
    if let Some((_, s)) = in_layers().find(|(_, s)| (**s as usize) < top_k) {
        return Some(format!(
            "the expert cache holds {s} experts in some layer, fewer than the {top_k} \
             routed per token; the CPU pass is faster then"
        ));
    }
    in_layers()
        .find(|(l, s)| u64::from(**s) * 2 < u64::from(l.experts))
        .map(|(l, s)| {
            format!(
                "the expert cache holds {s} of {} experts in some layer, fewer than half; \
                 the CPU pass decodes faster then",
                l.experts
            )
        })
}

fn plan_at_load(
    fp: &ModelFootprint,
    opts: &EngineOptions,
    context: Option<usize>,
    vram: Option<u64>,
) -> Result<MemoryPlan, RuntimeError> {
    let host = Host::live();
    let memory = qubivex_hardware::probe_memory(&host);
    let now = SystemProbe::new(host)
        .snapshot()
        .map_err(|e| RuntimeError::Memory(format!("cannot measure available RAM: {e}")))?;
    let hm = HostMemory {
        total: memory.effective_total(),
        available: now.ram_available,
        dense_resident: true,
    };
    let req = PlanRequest {
        profile: opts.profile,
        ram_budget: opts.ram_budget.map(Bytes),
        context: context.map(|c| c as u64),
        sequences: 1,
        layer_weights: None,
        vram: vram.map(|available| VramRequest {
            device: 0,
            available: Bytes(available),
            budget: opts.vram_budget.map(Bytes),
        }),
    };
    qubivex_memory::plan(fp, &hm, &req).map_err(|e| RuntimeError::Memory(e.to_string()))
}

/// xorshift64*.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        #[allow(clippy::cast_precision_loss)]
        let v = (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32;
        v
    }
}

/// Greedy (argmax, lowest id on ties), or temperature, then top-k, then top-p: top-p
/// measures its mass within what top-k kept, as `transformers` does.
fn sample(logits: &[f32], s: Sampling, rng: &mut Rng) -> u32 {
    let argmax = || {
        let mut best = 0usize;
        for (i, v) in logits.iter().enumerate() {
            if v.total_cmp(&logits[best]).is_gt() {
                best = i;
            }
        }
        u32::try_from(best).unwrap_or(0)
    };
    if s.temperature <= 0.0 {
        return argmax();
    }
    let mut p: Vec<(usize, f32)> = logits
        .iter()
        .map(|&l| l / s.temperature)
        .enumerate()
        .collect();
    let m = p.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for x in &mut p {
        x.1 = (x.1 - m).exp();
        sum += x.1;
    }
    p.iter_mut().for_each(|x| x.1 /= sum);
    p.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if s.top_k > 0 {
        p.truncate(usize::try_from(s.top_k).unwrap_or(usize::MAX));
    }
    let mut keep = p.len();
    if s.top_p < 1.0 {
        let kept = usize::try_from(s.top_k).is_ok_and(|k| k > 0 && k < logits.len());
        let mass = if kept {
            p.iter().map(|x| x.1).sum()
        } else {
            1.0
        };
        let mut acc = 0f32;
        for (i, x) in p.iter().enumerate() {
            acc += x.1 / mass;
            if acc >= s.top_p {
                keep = i + 1;
                break;
            }
        }
    }
    let p = &p[..keep];
    let total: f32 = p.iter().map(|x| x.1).sum();
    let mut r = rng.next_f32() * total;
    for x in p {
        r -= x.1;
        if r <= 0.0 {
            return u32::try_from(x.0).unwrap_or(0);
        }
    }
    p.last()
        .map_or_else(argmax, |x| u32::try_from(x.0).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling() {
        let l = [0.1f32, 3.0, 3.0, -1.0];
        assert_eq!(
            sample(&l, Sampling::default(), &mut Rng(1)),
            1,
            "greedy, lowest id on ties"
        );
        let s = Sampling {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.5,
            seed: 0,
        };
        for seed in 0..20 {
            let t = sample(&l, s, &mut Rng(seed));
            assert!(t == 1 || t == 2, "top-p keeps the two best");
        }
        let s = Sampling {
            temperature: 0.7,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        };
        let a: Vec<u32> = (0..10).map(|i| sample(&l, s, &mut Rng(i))).collect();
        let b: Vec<u32> = (0..10).map(|i| sample(&l, s, &mut Rng(i))).collect();
        assert_eq!(a, b, "same seed, same tokens");
    }

    #[test]
    fn the_cpu_pass_takes_over_below_half_a_layers_experts() {
        let layers: Vec<LayerExperts> = (0..2)
            .map(|i| LayerExperts {
                layer: qubivex_core::LayerIdx(i),
                experts: 64,
                expert_bytes: Bytes(10),
                total: Bytes(640),
            })
            .collect();
        let reason = |slots: u64| cpu_pass_reason(&layers, slots * 2 * 10, 8);
        assert!(reason(4).is_some_and(|r| r.contains("fewer than the 8")));
        assert!(reason(31).is_some_and(|r| r.contains("31 of 64")));
        assert_eq!(reason(32), None, "half stays on Metal");
        assert_eq!(reason(64), None);
        assert_eq!(cpu_pass_reason(&[], 0, 0), None, "no routed experts");
    }
}
