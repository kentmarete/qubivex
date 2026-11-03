//! A loaded model ready to generate: resident weights, the expert store, the tokenizer.

use std::collections::HashMap;
use std::fs::File;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use qubivex_arch::{
    ArchError, DenseWeights, ExpertProvider, ExpertWeights, KvCache, Logits, Sources, Tracer,
    TransformerMoeModel,
};
use qubivex_backend_cpu::CpuBackend;
use qubivex_compute::WeightView;
use qubivex_compute::tensor::Tensor;
use qubivex_core::{Bytes, ExpertIdx, LayerIdx, Shape};
use qubivex_expert_stream::{ExpertKey, ExpertStore, Policy, PoolSpec, StoreError, StoreStats};
use qubivex_hardware::{Host, Probed, ResourceProbe, SystemProbe};
use qubivex_memory::{
    Charge, HostMemory, LayerExperts, MemoryLedger, MemoryPlan, ModelFootprint, Owner, PlanRequest,
    PressureConfig, PressureDecision, PressureMonitor, Profile, Tier,
};
use qubivex_model::{ExpertPart, ModelMetadata, TensorRole};
use qubivex_tokenizer::Tokenizer;

use crate::{ModelExpertSource, RuntimeError, load_tokenizer};

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
    /// Compute threads. `None`: the CPU's compute team (its fastest cores: efficiency
    /// cores slow a team that meets at every matmul), else every logical CPU.
    pub threads: Option<usize>,
}

/// I/O threads when none are requested.
pub const DEFAULT_IO_THREADS: usize = 4;

/// The default thread count: the compute team, else every logical CPU.
fn default_threads() -> usize {
    let cpu = qubivex_hardware::probe_cpu(&qubivex_hardware::Host::live());
    cpu.performance_cores()
        .known()
        .and_then(|&n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, std::num::NonZero::get))
}

/// The resident tensors: everything except routed experts. Each is stored as read
/// from the file and described as a matrix (a vector `[n]` is viewed as `[1, n]`).
struct Dense {
    tensors: HashMap<String, Tensor<u8>>,
}

impl DenseWeights for Dense {
    fn view(&self, name: &str) -> Result<WeightView<'_>, ArchError> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| ArchError::Missing(name.to_owned()))?;
        Ok(WeightView::try_from(t.view())?)
    }
}

/// Experts from the store, one lease at a time.
struct StoreProvider<'a> {
    store: &'a ExpertStore,
    source: &'a ModelExpertSource,
}

impl StoreProvider<'_> {
    /// An expert's matrices inside its leased bytes.
    fn weights<'b>(
        &self,
        layer: LayerIdx,
        expert: ExpertIdx,
        bytes: &'b [u8],
    ) -> Result<ExpertWeights<'b>, ArchError> {
        let err = |detail: String| ArchError::Expert {
            layer: layer.0,
            expert: expert.0,
            detail,
        };
        let layout = self
            .source
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
}

fn store_err(layer: LayerIdx, expert: ExpertIdx, e: &StoreError) -> ArchError {
    ArchError::Expert {
        layer: layer.0,
        expert: expert.0,
        detail: e.to_string(),
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

    /// Starts every miss of the layer before waiting on any, and computes each expert
    /// as soon as it is ready (hits first). Each lease is released before the next is
    /// taken, so a layer with fewer slots than routed experts pipelines through them.
    fn with_experts(
        &self,
        layer: LayerIdx,
        experts: &[ExpertIdx],
        f: &mut dyn FnMut(ExpertIdx, &ExpertWeights<'_>) -> Result<(), ArchError>,
    ) -> Result<(), ArchError> {
        let keys: Vec<_> = experts.iter().map(|&e| (layer, e)).collect();
        let first = experts.first().copied().unwrap_or(ExpertIdx(0));
        let batch = self
            .store
            .acquire_many(&keys)
            .map_err(|e| store_err(layer, first, &e))?;
        for lease in batch {
            let lease = lease.map_err(|e| {
                let at = match &e {
                    StoreError::Load { key, .. } | StoreError::Full { key, .. } => key.1,
                    _ => first,
                };
                store_err(layer, at, &e)
            })?;
            let expert = lease.key().1;
            f(expert, &self.weights(layer, expert, lease.bytes())?)?;
        }
        Ok(())
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

/// The memory plan and the ledger of what is actually allocated.
struct Memory {
    /// `None`: a fixed cache size was requested.
    plan: Option<MemoryPlan>,
    ledger: Arc<MemoryLedger>,
    charges: Mutex<Charges>,
}

/// Ledger charges the engine holds. The KV charge follows the most recent sequence:
/// the engine runs one at a time until the scheduler (Phase 9).
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
    model: TransformerMoeModel,
    tokenizer: Tokenizer,
    backend: CpuBackend,
    max_context: usize,
    /// The forward pass runs inside this pool, so every matmul starts on a worker that
    /// joins the work instead of a caller blocked on a condition variable.
    pool: rayon::ThreadPool,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("model", &self.meta.manifest.model.id)
            .field("store", &self.store)
            .finish_non_exhaustive()
    }
}

fn read_tensor(file: &File, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
    let mut b = vec![0u8; len];
    #[cfg(unix)]
    std::os::unix::fs::FileExt::read_exact_at(file, &mut b, offset)?;
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < len {
            let n = file.seek_read(&mut b[done..], offset + done as u64)?;
            if n == 0 {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            done += n;
        }
    }
    Ok(b)
}

impl Engine {
    /// Loads a validated model: reads the resident tensors, plans memory with them
    /// resident (ADR-005), sets up the expert store and the tokenizer.
    ///
    /// # Errors
    /// I/O, tokenizer and architecture errors; [`RuntimeError::Memory`] if the model
    /// does not fit the budget or available RAM cannot be measured.
    pub fn load(meta: ModelMetadata, opts: EngineOptions) -> Result<Self, RuntimeError> {
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
        let mut tensors = HashMap::new();
        let mut dense_bytes = 0u64;
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
            let bytes = read_tensor(&files[t.file], t.tensor.offset, len)
                .map_err(|e| bad(&format!("cannot read: {e}")))?;
            dense_bytes += bytes.len() as u64;
            let tensor = Tensor::new(desc, bytes).map_err(|e| bad(&e))?;
            tensors.insert(name.clone(), tensor);
        }
        let dense = Dense { tensors };
        let backend = CpuBackend::new();
        let model = TransformerMoeModel::new(&meta.config, &dense, &backend)?;
        let source = Arc::new(ModelExpertSource::new(&meta)?);
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
        let layers = footprint.layers;
        let io_threads = opts.io_threads.unwrap_or(DEFAULT_IO_THREADS);
        let store = ExpertStore::with_io_threads(
            source.clone(),
            &pools_for(&layers, capacity),
            opts.cache_policy,
            io_threads,
        )
        .map_err(|e| RuntimeError::Model(format!("cannot start {io_threads} I/O threads: {e}")))?;
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
        Ok(Self {
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
            model,
            tokenizer,
            backend,
        })
    }

    /// Expert I/O threads in use.
    #[must_use]
    pub fn io_threads(&self) -> usize {
        self.store.io_threads()
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

    /// Expert store counters.
    #[must_use]
    pub fn store_stats(&self) -> StoreStats {
        self.store.stats()
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

    /// Refuses a new request under critical memory pressure (ADR-005). Active
    /// requests are never stopped.
    ///
    /// # Errors
    /// [`RuntimeError::Memory`] when the pressure monitor refuses admission.
    pub fn admit(&self) -> Result<(), RuntimeError> {
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
        let projected = KvCache::reserved_for(cache.len() + tokens, cache.capacity());
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
            c.kv.resize(Bytes(cache.allocated_bytes()));
        }
    }

    /// A fresh KV cache.
    #[must_use]
    pub fn new_cache(&self) -> KvCache {
        self.model.new_cache(self.max_context)
    }

    /// One forward pass (see [`TransformerMoeModel::forward`]).
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
        let provider = StoreProvider {
            store: &self.store,
            source: &self.source,
        };
        let src = Sources {
            dense: &self.dense,
            experts: &provider,
            backend: &self.backend,
        };
        self.fit_room(cache, tokens.len())?;
        let out = self
            .pool
            .install(|| self.model.forward(tokens, cache, src, logits, trace))?;
        self.account(cache);
        Ok(out)
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
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>, RuntimeError> {
        if prompt.is_empty() {
            return Err(RuntimeError::Model("the prompt is empty".into()));
        }
        self.admit()?;
        let mut cache = self.new_cache();
        let mut rng = Rng(sampling.seed ^ 0x9e37_79b9_7f4a_7c15);
        let mut logits = self.forward(prompt, &mut cache, Logits::Last, None)?;
        let mut out = Vec::new();
        while out.len() < max_new {
            let next = sample(&logits, sampling, &mut rng);
            if self.tokenizer.eos_ids().contains(&next) {
                break;
            }
            out.push(next);
            if !on_token(next) || out.len() == max_new || cache.len() == cache.capacity() {
                break;
            }
            logits = self.forward(&[next], &mut cache, Logits::Last, None)?;
        }
        Ok(out)
    }
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
