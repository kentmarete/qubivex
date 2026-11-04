//! A loaded model ready to generate: resident weights, the expert store, the tokenizer.

use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use qubivex_arch::{
    ArchError, DenseWeights, ExpertProvider, ExpertWeights, KvCache, Logits, Sources, Tracer,
    Transformer,
};
use qubivex_compute::WeightView;
use qubivex_compute::host::PageAligned;
use qubivex_compute::tensor::TensorView;
use qubivex_core::{Bytes, ExpertIdx, LayerIdx, Shape, TensorDesc};
use qubivex_expert_stream::fault::VerifiedSource;
use qubivex_expert_stream::mirror::{MirrorPolicy, MirroredSource};
use qubivex_expert_stream::predict::{Lookahead, LookaheadConfig};
use qubivex_expert_stream::{
    Batch, ExpertKey, ExpertSource, ExpertStore, Lease, Next, Occupancy, Policy, PoolSpec,
    PrefetchStats, ReadTimeout, StoreError, StoreStats,
};
use qubivex_hardware::{Host, Probed, ResourceProbe, SystemProbe};
use qubivex_memory::{
    Charge, HostMemory, LayerExperts, MemoryLedger, MemoryPlan, ModelFootprint, Owner, PlanRequest,
    PressureConfig, PressureDecision, PressureMonitor, Profile, Tier,
};
use qubivex_model::{ExpertPart, ModelMetadata, TensorRole};
use qubivex_tokenizer::Tokenizer;

use crate::checksums::{ExpertSums, SUMS_FILE, SumsState};
use crate::metal_forward::MetalForward;
use crate::placement::{BackendKind, MetalOps, Placed, PlacementReport};
use crate::{ExpertIo, IoPath, ModelExpertSource, RuntimeError, load_tokenizer};
use qubivex_backend_metal::MetalBackend;

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
    /// The compute backend (Phase 8). Placement changes speed, never bits.
    pub backend: BackendKind,
    /// The dense matmuls placed on Metal with [`BackendKind::Metal`].
    pub metal_ops: MetalOps,
}

/// Router-lookahead prefetch settings (Phase 7).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PrefetchOptions {
    /// How predictions are made.
    pub lookahead: LookaheadConfig,
    /// The LFRU eviction guard: a prediction may evict a resident expert only if it is
    /// clearly hotter (ADR-008).
    pub guard: bool,
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
}

impl StoreProvider<'_> {
    /// After `layer`'s demand is queued: checks the prediction made for `layer`, and
    /// issues the one made for the next layer, so prefetches queue behind demand.
    fn after_demand(&self, layer: LayerIdx, routed: &[ExpertIdx]) {
        let Some(p) = self.predictor else { return };
        let mut p = p.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((l, ids)) = p.issued.take() {
            if l == layer {
                let correct = routed.iter().filter(|e| ids.contains(e)).count();
                p.stats.layers += 1;
                p.stats.predicted += ids.len() as u64;
                p.stats.routed += routed.len() as u64;
                p.stats.correct += correct as u64;
            }
        }
        if let Some((l, ids)) = p.pending.take() {
            if l.0 == layer.0 + 1 {
                let keys: Vec<ExpertKey> = ids.iter().map(|&e| (l, e)).collect();
                // Advisory: a poisoned store fails the demand that follows anyway.
                let _ = self.store.prefetch(&keys, p.guard);
                p.issued = Some((l, ids));
            }
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

    /// Computes `lease`'s expert on the pool, releases it, then starts whatever has
    /// become ready.
    fn spawn<'scope>(&'scope self, sc: &rayon::Scope<'scope>, lease: Lease<'p>) {
        sc.spawn(move |sc| {
            if self.failed() {
                return;
            }
            let expert = lease.key().1;
            let r = self
                .provider
                .weights(self.layer, expert, lease.bytes())
                .and_then(|w| (self.f)(expert, &w));
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
        let wave = Wave {
            provider: self,
            layer,
            first,
            f,
            batch: Mutex::new(batch),
            error: Mutex::new(None),
        };
        loop {
            // Holding nothing: wait for the next expert (or fail as `next` does).
            let lease = match wave.lock_batch().next() {
                None => return Ok(()),
                Some(lease) => lease.map_err(|e| wave.lease_err(e))?,
            };
            rayon::scope(|sc| {
                wave.spawn(sc, lease);
                wave.start_ready(sc);
            });
            if let Some(e) = wave.take_error() {
                return Err(e);
            }
        }
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
    /// Nucleus mass (1.0: off).
    pub top_p: f32,
    /// RNG seed: the same seed gives the same tokens.
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
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
#[derive(Clone, Copy, Debug)]
pub struct GenerateOptions {
    /// Most tokens to generate.
    pub max_new: usize,
    /// Sampling.
    pub sampling: Sampling,
    /// Prompt tokens per forward pass. Smaller chunks let a cancellation land sooner;
    /// the bits are the same for every size.
    pub prefill_chunk: usize,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new: 128,
            sampling: Sampling::default(),
            prefill_chunk: 256,
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
}

/// KV room held for one conversation (see [`Engine::reserve_kv`]).
#[derive(Debug)]
pub struct KvReservation<'e> {
    engine: &'e Engine,
    positions: usize,
}

impl KvReservation<'_> {
    /// Positions held (whole KV chunks).
    #[must_use]
    pub const fn positions(&self) -> usize {
        self.positions
    }
}

impl Drop for KvReservation<'_> {
    fn drop(&mut self) {
        // Relaxed: see `Engine::reserve_kv`.
        self.engine
            .kv_reserved
            .fetch_sub(self.positions, Ordering::Relaxed);
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
/// sequence's cache and what live [`KvReservation`]s hold across conversations.
struct Charges {
    _dense: Charge,
    experts: Charge,
    kv: Charge,
}

/// The pressure monitor's state, shared with its sampling thread.
struct Pressure {
    monitor: Mutex<PressureMonitor>,
    ledger: Arc<MemoryLedger>,
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
        self.lock().observe(available, used)
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
                    if let Ok(s) = probe.snapshot() {
                        if let Probed::Known(a) = s.ram_available {
                            pressure.observe(a);
                        }
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
    kv_reserved: AtomicUsize,
    /// Forward passes run one at a time, first come first served (see
    /// [`Engine::forward`]).
    steps: StepQueue,
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
        let mut engine = Self::load_with_source(meta, opts, move |s| {
            let first = verified(s);
            if copies.is_empty() {
                return first;
            }
            Arc::new(MirroredSource::new(
                std::iter::once((primary, first)).chain(copies),
                MirrorPolicy::default(),
            ))
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
            BackendKind::Cpu => Placed::cpu(),
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
        let (plan, capacity, max_context) = match opts.expert_cache_bytes {
            Some(c) => (None, c, wanted_ctx.unwrap_or(model_ctx)),
            None => {
                let plan = plan_at_load(&footprint, &opts, wanted_ctx)?;
                let capacity = plan
                    .expert_cache_for(0)
                    .map_err(|r| RuntimeError::Memory(r.to_string()))?
                    .0;
                let ctx = usize::try_from(plan.context).unwrap_or(model_ctx);
                (Some(plan), capacity, ctx)
            }
        };
        // Whole layers beat the CPU only when each layer's pool holds a token's routed
        // experts: with fewer slots, experts stream through the pool and every layer's
        // GPU round trip costs more than the GPU saves (ADR-011 amendment 3). The plan
        // then runs the CPU pass.
        let top_k = model.moe_config().map_or(0, |c| c.top_k as usize);
        let short = pools_for(&footprint.layers, capacity)
            .into_iter()
            .find(|p| (p.slots as usize) < top_k);
        let (backend, gpu) = match layers_on {
            Some((be, _)) if short.is_some() => {
                let p = short.map_or(0, |p| p.slots);
                let reason = format!(
                    "the expert cache holds {p} experts in some layer, fewer than the {top_k} \
                     routed per token; the CPU pass is faster then"
                );
                (
                    Placed::fallback(be.info().name, opts.metal_ops, reason),
                    None,
                )
            }
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
        let ledger = MemoryLedger::new(0);
        let charges = Charges {
            _dense: ledger.charge(Tier::Ram, Owner::Dense, Bytes(dense_bytes)),
            experts: ledger.charge(Tier::Ram, Owner::ExpertCache, Bytes(0)),
            kv: ledger.charge(Tier::Ram, Owner::Kv, Bytes(0)),
        };
        let pressure = plan.as_ref().map(|p| {
            Arc::new(Pressure {
                monitor: Mutex::new(PressureMonitor::new(opts.pressure, p.ram.cache_floor.bytes)),
                ledger: Arc::clone(&ledger),
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
            })
        });
        Ok(Self {
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
            kv_reserved: AtomicUsize::new(0),
            steps: StepQueue::default(),
            kv_position_bytes,
            verification: Verification::Off,
            model,
            tokenizer,
            backend,
        })
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
        self.predictor.as_ref().map(|p| PrefetchReport {
            prediction: p
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .stats,
            store: self.store.stats().prefetch,
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
        let Some(plan) = &self.memory.plan else {
            return Ok(());
        };
        // With concurrent conversations, the plan prices every reservation together.
        let projected = KvCache::reserved_for(cache.len() + tokens, cache.capacity())
            .max(self.kv_reserved.load(Ordering::Relaxed));
        let mut cap = plan
            .expert_cache_for(projected as u64)
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

    /// Updates the ledger from the store and the KV cache.
    fn account(&self, cache: &KvCache) {
        // A poisoned lock means another thread panicked mid-update; the counters are
        // diagnostics, so skip rather than propagate.
        if let Ok(mut c) = self.memory.charges.lock() {
            c.experts.resize(Bytes(self.store.used().unwrap_or(0)));
            let reserved = self.kv_reserved.load(Ordering::Relaxed) as u64;
            let kv = cache
                .allocated_bytes()
                .max(reserved.saturating_mul(self.kv_position_bytes));
            c.kv.resize(Bytes(kv));
        }
    }

    /// A fresh KV cache.
    #[must_use]
    pub fn new_cache(&self) -> KvCache {
        self.model.new_cache(self.max_context)
    }

    /// One forward pass (see [`Transformer::forward`]).
    ///
    /// Concurrent callers (conversations of a server) take turns, first come first
    /// served: each pass already uses every compute thread, and passes running at
    /// once would compete for expert-cache slots (two layers' worth of leases in a
    /// one-slot pool fail instead of waiting). A conversation therefore waits at most
    /// one pass of each conversation ahead of it; [`Engine::generate_with`] splits long
    /// prompts into chunks so a prompt never holds the turn for long.
    ///
    /// # Errors
    /// Architecture errors.
    pub fn forward(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        logits: Logits,
        trace: Option<Tracer<'_>>,
    ) -> Result<Vec<f32>, RuntimeError> {
        let _turn = self.steps.wait();
        // Memory pressure stops prefetch first (ADR-005): drop what is queued.
        let allowed = self.pressure().is_none_or(|d| d.prefetch);
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
        if let Some(g) = &self.gpu {
            if trace.is_some() {
                return Err(RuntimeError::Model(
                    "tracing needs the CPU forward pass (not --metal-ops layers)".into(),
                ));
            }
            self.fit_room(cache, tokens.len())?;
            let out = g.forward(
                tokens,
                cache,
                &self.dense,
                &self.store,
                &self.source,
                logits,
            );
            self.account(cache);
            return Ok(out?);
        }
        let provider = StoreProvider {
            store: &self.store,
            source: &self.source,
            predictor,
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
        self.account(cache);
        Ok(out?)
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
        let done = |tokens, finish| {
            self.end_turn();
            Ok(Generation {
                tokens,
                finish,
                prompt_tokens: prompt.len(),
            })
        };
        let mut cache = self.new_cache();
        if prompt.len() > cache.capacity() {
            return Err(RuntimeError::Model(format!(
                "the prompt has {} tokens; the context holds {}",
                prompt.len(),
                cache.capacity()
            )));
        }
        let mut rng = Rng(opts.sampling.seed ^ 0x9e37_79b9_7f4a_7c15);
        let mut logits = Vec::new();
        for chunk in prompt.chunks(opts.prefill_chunk.max(1)) {
            // Acquire: pairs with the canceller's Release store.
            if cancel.load(Ordering::Acquire) {
                return done(Vec::new(), Finish::Cancelled);
            }
            logits = self.forward(chunk, &mut cache, Logits::Last, None)?;
        }
        let mut out = Vec::new();
        loop {
            if out.len() == opts.max_new {
                return done(out, Finish::Length);
            }
            if cancel.load(Ordering::Acquire) {
                return done(out, Finish::Cancelled);
            }
            let next = sample(&logits, opts.sampling, &mut rng);
            if self.tokenizer.eos_ids().contains(&next) {
                return done(out, Finish::Eos);
            }
            out.push(next);
            if !on_token(next) {
                return done(out, Finish::Stopped);
            }
            if out.len() == opts.max_new {
                return done(out, Finish::Length);
            }
            if cache.len() == cache.capacity() {
                return done(out, Finish::ContextFull);
            }
            logits = self.forward(&[next], &mut cache, Logits::Last, None)?;
        }
    }

    /// Reserves KV room for a conversation of up to `positions` positions (prompt plus
    /// generated tokens), alongside every other live reservation. Admission control
    /// for concurrent conversations: the memory plan must still keep one expert slot
    /// per layer with all reservations' KV in RAM (ADR-005). Without a plan (a fixed
    /// expert cache) every reservation is granted. Dropping the reservation returns the
    /// room.
    ///
    /// # Errors
    /// [`RuntimeError::Memory`] if the reservations together would not fit.
    pub fn reserve_kv(&self, positions: usize) -> Result<KvReservation<'_>, RuntimeError> {
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
                    return Ok(KvReservation {
                        engine: self,
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

    /// Ends a turn: queued prefetches are dropped, since the next request starts
    /// elsewhere. [`Engine::generate`] calls it; callers driving
    /// [`Engine::forward`] themselves call it between requests.
    pub fn end_turn(&self) {
        if self.predictor.is_some() {
            // Advisory: a poisoned store fails the next forward pass anyway.
            let _ = self.store.cancel_prefetch(None);
        }
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

fn plan_at_load(
    fp: &ModelFootprint,
    opts: &EngineOptions,
    context: Option<usize>,
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

/// Greedy (argmax, lowest id on ties) or temperature + top-p sampling.
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
    let mut keep = p.len();
    if s.top_p < 1.0 {
        let mut acc = 0f32;
        for (i, x) in p.iter().enumerate() {
            acc += x.1;
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
            top_p: 0.5,
            seed: 0,
        };
        for seed in 0..20 {
            let t = sample(&l, s, &mut Rng(seed));
            assert!(t == 1 || t == 2, "top-p keeps the two best");
        }
        let s = Sampling {
            temperature: 0.7,
            top_p: 1.0,
            seed: 0,
        };
        let a: Vec<u32> = (0..10).map(|i| sample(&l, s, &mut Rng(i))).collect();
        let b: Vec<u32> = (0..10).map(|i| sample(&l, s, &mut Rng(i))).collect();
        assert_eq!(a, b, "same seed, same tokens");
    }
}
