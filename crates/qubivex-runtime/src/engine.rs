//! A loaded model ready to generate: resident weights, the expert store, the tokenizer.

use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, Mutex};

use qubivex_arch::{
    ArchError, DenseWeights, ExpertProvider, ExpertWeights, KvCache, Logits, Sources, Tracer,
    TransformerMoeModel,
};
use qubivex_backend_cpu::CpuBackend;
use qubivex_compute::WeightView;
use qubivex_compute::tensor::Tensor;
use qubivex_core::{Bytes, ExpertIdx, LayerIdx, Shape};
use qubivex_expert_stream::{ExpertStore, StoreStats};
use qubivex_hardware::{Host, ResourceProbe, SystemProbe};
use qubivex_memory::{
    Charge, HostMemory, MemoryLedger, MemoryPlan, ModelFootprint, Owner, PlanRequest, Profile, Tier,
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
    /// Compute threads. `None`: the CPU's compute team (its fastest cores: efficiency
    /// cores slow a team that meets at every matmul), else every logical CPU.
    pub threads: Option<usize>,
}

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

impl ExpertProvider for StoreProvider<'_> {
    fn with_expert(
        &self,
        layer: LayerIdx,
        expert: ExpertIdx,
        f: &mut dyn FnMut(&ExpertWeights<'_>) -> Result<(), ArchError>,
    ) -> Result<(), ArchError> {
        let err = |detail: String| ArchError::Expert {
            layer: layer.0,
            expert: expert.0,
            detail,
        };
        let layout = self
            .source
            .layout(layer, expert)
            .ok_or_else(|| err("no such expert".into()))?;
        let lease = self
            .store
            .acquire((layer, expert))
            .map_err(|e| err(e.to_string()))?;
        let bytes = lease.bytes();
        let view = |p: ExpertPart| -> Result<Option<WeightView<'_>>, ArchError> {
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
        f(&ExpertWeights { gate, up, down })
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

/// A model ready to generate.
pub struct Engine {
    memory: Memory,
    meta: ModelMetadata,
    dense: Dense,
    source: Arc<ModelExpertSource>,
    store: ExpertStore,
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
                let plan = plan_at_load(&meta, &opts, wanted_ctx)?;
                let capacity = plan
                    .expert_cache_for(0)
                    .map_err(|r| RuntimeError::Memory(r.to_string()))?
                    .0;
                let ctx = usize::try_from(plan.context).unwrap_or(model_ctx);
                (Some(plan), capacity, ctx)
            }
        };
        let store = ExpertStore::new(source.clone(), capacity);
        let ledger = MemoryLedger::new(0);
        let charges = Charges {
            _dense: ledger.charge(Tier::Ram, Owner::Dense, Bytes(dense_bytes)),
            experts: ledger.charge(Tier::Ram, Owner::ExpertCache, Bytes(0)),
            kv: ledger.charge(Tier::Ram, Owner::Kv, Bytes(0)),
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
            model,
            tokenizer,
            backend,
        })
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

    /// The expert cache capacity in bytes now (the planner lowers it as the KV grows).
    #[must_use]
    pub fn expert_cache_bytes(&self) -> u64 {
        self.store.capacity()
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

    /// Shrinks the expert cache to leave room for `cache` growing by `tokens`
    /// (Colibri's `kv_room_fit`), or refuses if that would cross the floor.
    fn fit_room(&self, cache: &KvCache, tokens: usize) -> Result<(), RuntimeError> {
        let Some(plan) = &self.memory.plan else {
            return Ok(());
        };
        let projected = KvCache::reserved_for(cache.len() + tokens, cache.capacity());
        let cap = plan
            .expert_cache_for(projected as u64)
            .map_err(|r| RuntimeError::Memory(r.to_string()))?;
        if cap.0 != self.store.capacity() {
            self.store
                .set_capacity(cap.0)
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
fn plan_at_load(
    meta: &ModelMetadata,
    opts: &EngineOptions,
    context: Option<usize>,
) -> Result<MemoryPlan, RuntimeError> {
    let fp =
        ModelFootprint::from_metadata(meta).map_err(|e| RuntimeError::Memory(e.to_string()))?;
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
    qubivex_memory::plan(&fp, &hm, &req).map_err(|e| RuntimeError::Memory(e.to_string()))
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
