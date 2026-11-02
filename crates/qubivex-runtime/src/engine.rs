//! A loaded model ready to generate: resident weights, the expert store, the tokenizer.

use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;

use qubivex_arch::{
    ArchError, DenseWeights, ExpertProvider, ExpertWeights, KvCache, Logits, Sources, Tracer,
    TransformerMoeModel,
};
use qubivex_backend_cpu::CpuBackend;
use qubivex_compute::WeightView;
use qubivex_core::{DType, ExpertIdx, LayerIdx};
use qubivex_expert_stream::{ExpertStore, StoreStats};
use qubivex_model::{ExpertPart, ModelMetadata, TensorRole};
use qubivex_tokenizer::Tokenizer;

use crate::{ModelExpertSource, RuntimeError, load_tokenizer};

/// How to load a model.
#[derive(Clone, Copy, Debug, Default)]
pub struct EngineOptions {
    /// Expert cache capacity in bytes. `None`: every expert fits (all resident).
    pub expert_cache_bytes: Option<u64>,
    /// Context limit. `None`: the model's maximum (or 4096 if it states none).
    pub max_context: Option<usize>,
}

struct Tensor {
    dtype: DType,
    rows: usize,
    cols: usize,
    bytes: Vec<u8>,
}

/// The resident tensors: everything except routed experts.
struct Dense {
    tensors: HashMap<String, Tensor>,
}

impl DenseWeights for Dense {
    fn view(&self, name: &str) -> Result<WeightView<'_>, ArchError> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| ArchError::Missing(name.to_owned()))?;
        Ok(WeightView::new(t.dtype, t.rows, t.cols, &t.bytes)?)
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

/// A model ready to generate.
pub struct Engine {
    meta: ModelMetadata,
    dense: Dense,
    source: Arc<ModelExpertSource>,
    store: ExpertStore,
    model: TransformerMoeModel,
    tokenizer: Tokenizer,
    backend: CpuBackend,
    max_context: usize,
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
    /// Loads a validated model: reads the resident tensors, sets up the expert store
    /// and the tokenizer.
    ///
    /// # Errors
    /// I/O, tokenizer and architecture errors.
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
            let (rows, cols) = match t.tensor.shape.dims() {
                [n] => (1, *n),
                [r, c] => (*r, *c),
                d => {
                    return Err(RuntimeError::Model(format!(
                        "tensor {name:?} has shape {d:?}"
                    )));
                }
            };
            let us = |v: u64| {
                usize::try_from(v)
                    .map_err(|_| RuntimeError::Model(format!("{name:?} is too large")))
            };
            let bytes = read_tensor(&files[t.file], t.tensor.offset, us(t.tensor.bytes)?)
                .map_err(|e| RuntimeError::Model(format!("reading {name:?}: {e}")))?;
            tensors.insert(
                name.clone(),
                Tensor {
                    dtype: t.tensor.dtype,
                    rows: us(rows)?,
                    cols: us(cols)?,
                    bytes,
                },
            );
        }
        let dense = Dense { tensors };
        let backend = CpuBackend::new();
        let model = TransformerMoeModel::new(&meta.config, &dense, &backend)?;
        let source = Arc::new(ModelExpertSource::new(&meta)?);
        let capacity = opts
            .expert_cache_bytes
            .unwrap_or_else(|| source.total_bytes());
        let store = ExpertStore::new(source.clone(), capacity);
        let tokenizer = load_tokenizer(&meta)?;
        let model_ctx = meta
            .manifest
            .context_length
            .or(meta.config.dims.max_context)
            .and_then(|c| usize::try_from(c).ok())
            .unwrap_or(4096);
        Ok(Self {
            max_context: opts.max_context.map_or(model_ctx, |c| c.min(model_ctx)),
            meta,
            dense,
            source,
            store,
            model,
            tokenizer,
            backend,
        })
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

    /// The expert cache capacity in bytes.
    #[must_use]
    pub const fn expert_cache_bytes(&self) -> u64 {
        self.store.capacity()
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
        Ok(self.model.forward(tokens, cache, src, logits, trace)?)
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
