//! Architectures: the forward pass of a validated model (QUBIVEX_DESIGN §3).
//!
//! [`Transformer`] runs `transformer_moe`, the architecture whose config
//! [`qubivex_model::arch::TransformerMoe`] parses. Weights arrive through two traits so
//! that this crate knows neither files nor caches:
//!
//! - [`DenseWeights`]: the resident tensors (embedding, attention, norms, routers, head);
//! - [`ExpertProvider`]: routed experts, one at a time, from the expert store. The MoE
//!   layer has no other way to reach expert weights.
//!
//! Each token's computation has one fixed evaluation order: matmuls are batch-invariant
//! (`qubivex-compute`), attention and norms run per token, and expert outputs are
//! combined in router rank order. A prompt processed in one batch therefore gives
//! exactly the same bits as the same tokens fed one at a time, whatever the expert
//! cache holds (invariant 1).

pub mod ops;

use qubivex_compute::tensor::{Tensor, TensorView, TensorViewMut};
use qubivex_compute::{ComputeBackend, ComputeError, GatedFfn, MatMul, WeightView};
use qubivex_core::{DType, Shape};
use qubivex_core::{ExpertIdx, LayerIdx};
use qubivex_model::ArchConfig;
use qubivex_model::arch::{DenseConfig, MoeConfig, QkNorm, RouterScoring};
use rayon::prelude::*;
use thiserror::Error;

use crate::ops::{Rope, attend, rms_norm, route, swiglu};

/// Errors of a forward pass.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ArchError {
    /// A kernel failed.
    #[error(transparent)]
    Compute(#[from] ComputeError),
    /// A weight the architecture needs is not available.
    #[error("weight {0:?} is not loaded")]
    Missing(String),
    /// An expert could not be obtained.
    #[error("expert (layer {layer}, expert {expert}): {detail}")]
    Expert {
        /// Layer.
        layer: u32,
        /// Expert.
        expert: u32,
        /// Why.
        detail: String,
    },
    /// Inputs are inconsistent (token id out of range, context full, wrong config).
    #[error("{0}")]
    Input(String),
}

/// The kinds of dense matmul in the forward pass (see
/// [`Transformer::dense_matmuls`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DenseMatmul {
    /// The q, k, v and output projections of attention.
    Attention,
    /// The MoE router (and the lookahead router, which uses the next layer's weights).
    Router,
    /// The output head (the embedding when tied).
    Head,
    /// A dense FFN's gate, up and down projections.
    Ffn,
}

/// The resident weights, by tensor name.
pub trait DenseWeights: Sync {
    /// The weight as a `rows × cols` view (a 1-D tensor is `1 × n`).
    ///
    /// # Errors
    /// [`ArchError::Missing`] if the tensor is not loaded.
    fn view(&self, name: &str) -> Result<WeightView<'_>, ArchError>;
}

/// One expert's three matrices.
#[derive(Clone, Copy, Debug)]
pub struct ExpertWeights<'a> {
    /// `ffn × hidden`.
    pub gate: WeightView<'a>,
    /// `ffn × hidden`.
    pub up: WeightView<'a>,
    /// `hidden × ffn`.
    pub down: WeightView<'a>,
}

/// Routed experts, lent one at a time.
pub trait ExpertProvider: Sync {
    /// Calls `f` with the expert's weights while they are held.
    ///
    /// # Errors
    /// [`ArchError::Expert`] if the expert cannot be obtained, or whatever `f` returns.
    fn with_expert(
        &self,
        layer: LayerIdx,
        expert: ExpertIdx,
        f: &mut dyn FnMut(&ExpertWeights<'_>) -> Result<(), ArchError>,
    ) -> Result<(), ArchError>;

    /// Calls `f` once for each of `experts` (distinct) of one layer, in any order and
    /// possibly concurrently from several threads. A provider that can read several
    /// experts at once overrides this to start every load before handing out the
    /// first, and may compute ready experts in parallel; callers must not depend on
    /// the order.
    ///
    /// # Errors
    /// As [`ExpertProvider::with_expert`]. When several calls of `f` fail, which error
    /// is returned is unspecified.
    fn with_experts(
        &self,
        layer: LayerIdx,
        experts: &[ExpertIdx],
        f: &(dyn Fn(ExpertIdx, &ExpertWeights<'_>) -> Result<(), ArchError> + Sync),
    ) -> Result<(), ArchError> {
        for &e in experts {
            self.with_expert(layer, e, &mut |w| f(e, w))?;
        }
        Ok(())
    }

    /// As [`ExpertProvider::with_experts`], except that when every one of `experts`
    /// is ready now (resident, nothing to load or wait for), the provider may hold
    /// them all at once and call `group` a single time with their weights, in the
    /// order of `experts`, instead of calling `each` per expert. Callers use it when
    /// the backend groups matmuls ([`ComputeBackend::groups_matmuls`]): one
    /// submission per stage for the whole layer; or when it fuses the expert block
    /// ([`ComputeBackend::fuses_gated_ffn`]): one submission for the whole layer's
    /// experts. The default always calls `each`.
    ///
    /// # Errors
    /// As [`ExpertProvider::with_experts`], or whatever `group` returns.
    fn with_experts_grouped(
        &self,
        layer: LayerIdx,
        experts: &[ExpertIdx],
        group: &mut dyn FnMut(&[ExpertWeights<'_>]) -> Result<(), ArchError>,
        each: &(dyn Fn(ExpertIdx, &ExpertWeights<'_>) -> Result<(), ArchError> + Sync),
    ) -> Result<(), ArchError> {
        let _ = group;
        self.with_experts(layer, experts, each)
    }

    /// Whether the provider wants [`ExpertProvider::lookahead`] calls. When `false`
    /// (the default) the forward pass skips the lookahead router entirely.
    fn wants_lookahead(&self) -> bool {
        false
    }

    /// Advisory, during decode: `layer`'s router logits computed early, from the
    /// hidden state just after the previous layer's attention, normalized with
    /// `layer`'s MoE norm (Colibri's PILOT lookahead). The provider may read predicted
    /// experts ahead. Routing never uses these logits, so nothing a provider does with
    /// them changes the output.
    fn lookahead(&self, layer: LayerIdx, router_logits: &[f32]) {
        let _ = (layer, router_logits);
    }
}

/// Points of the forward pass that can be observed (numerical tests).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TracePoint {
    /// Token embedding.
    Embedding,
    /// Attention block output (after `o_proj`, before the residual).
    Attention,
    /// Router logits, one per expert.
    RouterLogits,
    /// The selected experts' routing weights, in rank order.
    RoutingWeights,
    /// MoE block output (before the residual).
    Moe,
    /// Dense FFN block output (before the residual).
    Ffn,
    /// Hidden state after the layer.
    Hidden,
    /// Output logits.
    Logits,
}

/// One observation: a point, its layer (if any), the token's index in the batch, the
/// values, and for [`TracePoint::RoutingWeights`] the selected expert ids.
#[derive(Debug)]
pub struct TraceEvent<'a> {
    /// The point.
    pub point: TracePoint,
    /// The layer, for per-layer points.
    pub layer: Option<u32>,
    /// Token index within this forward call.
    pub token: usize,
    /// Values.
    pub values: &'a [f32],
    /// Selected experts (routing weights only).
    pub experts: &'a [u32],
}

/// A trace callback. `Send`, because the forward pass may run on a worker thread.
pub type Tracer<'t> = &'t mut (dyn FnMut(TraceEvent<'_>) + Send);

/// Where a forward pass gets its weights and kernels.
#[derive(Clone, Copy)]
pub struct Sources<'a> {
    /// Resident weights.
    pub dense: &'a dyn DenseWeights,
    /// Routed experts.
    pub experts: &'a dyn ExpertProvider,
    /// Kernels.
    pub backend: &'a dyn ComputeBackend,
}

impl std::fmt::Debug for Sources<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sources")
            .field("backend", &self.backend.name())
            .finish_non_exhaustive()
    }
}

/// Which logits a forward pass returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Logits {
    /// Only the last token's (generation).
    Last,
    /// Every token's (tests).
    All,
}

/// One sequence's part of a batched forward pass (see [`Transformer::forward_batch`]).
#[derive(Debug)]
pub struct Segment<'a> {
    /// The sequence's new tokens, at positions `cache.len()..`.
    pub tokens: &'a [u32],
    /// The sequence's own keys and values.
    pub cache: &'a mut KvCache,
    /// Which of its logits to return.
    pub logits: Logits,
}

/// A device backend's keys and values for a cache (see [`KvCache::device_mut`]).
pub type DeviceKv = Box<dyn std::any::Any + Send + Sync>;

/// Keys and values of past positions, per layer and KV head.
///
/// The CPU forward pass keeps them in host tensors, one per layer for keys and one for
/// values, each `[kv_head, position, head_dim]` f32 with room for the reserved
/// positions (ADR-009). A step writes its rows through `narrow` views and attention
/// reads each head's filled positions as one contiguous `narrow` view. A device
/// forward pass (Phase 8)
/// keeps them in its own buffers, in [`KvCache::device_mut`], and advances the
/// position count with [`KvCache::advance`]; one cache is only ever used by one of the
/// two, since an engine's placement is fixed at load.
pub struct KvCache {
    /// Per layer, `[kv_head, reserved positions, head_dim]`; positions `0..len` are
    /// filled.
    k: Vec<Tensor<f32>>,
    v: Vec<Tensor<f32>>,
    kv_heads: u64,
    head_dim: u64,
    len: usize,
    max: usize,
    device: Option<DeviceKv>,
    device_bytes: u64,
}

impl std::fmt::Debug for KvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvCache")
            .field("len", &self.len)
            .field("max", &self.max)
            .field("device", &self.device.is_some())
            .finish_non_exhaustive()
    }
}

/// A cache of no layers and no context, which no model accepts: what a cache moved
/// out of its owner leaves behind.
impl Default for KvCache {
    fn default() -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            kv_heads: 0,
            head_dim: 0,
            len: 0,
            max: 0,
            device: None,
            device_bytes: 0,
        }
    }
}

impl KvCache {
    /// Positions per allocation step. The cache grows by whole chunks (never past its
    /// context limit) instead of doubling, so what it allocates stays within one chunk
    /// of what it holds and the memory planner's KV accounting stays close to reality.
    pub const CHUNK: usize = 256;

    /// The positions a cache of context `max` reserves when it holds `positions`.
    #[must_use]
    pub const fn reserved_for(positions: usize, max: usize) -> usize {
        let r = positions.div_ceil(Self::CHUNK).saturating_mul(Self::CHUNK);
        if r < max { r } else { max }
    }

    /// Bytes allocated for keys and values (f32), including reserved positions, on the
    /// host and (as reported with [`KvCache::set_device_bytes`]) on a device.
    #[must_use]
    pub fn allocated_bytes(&self) -> u64 {
        self.k
            .iter()
            .chain(&self.v)
            .map(|t| t.view().storage().len() as u64 * 4)
            .sum::<u64>()
            + self.device_bytes
    }

    /// A device forward pass's keys and values for this cache.
    pub fn device_mut(&mut self) -> &mut Option<DeviceKv> {
        &mut self.device
    }

    /// A device forward pass's keys and values for this cache, to read.
    #[must_use]
    pub fn device(&self) -> Option<&DeviceKv> {
        self.device.as_ref()
    }

    /// Records the bytes a device forward pass has allocated for this cache.
    pub const fn set_device_bytes(&mut self, bytes: u64) {
        self.device_bytes = bytes;
    }

    /// Counts `s` more positions as filled, for a device forward pass that stored
    /// their keys and values itself.
    ///
    /// # Errors
    /// [`ArchError::Input`] if the context would overflow.
    pub fn advance(&mut self, s: usize) -> Result<(), ArchError> {
        if self.len + s > self.max {
            return Err(ArchError::Input(format!(
                "context full: {} positions + {s} new > {}",
                self.len, self.max
            )));
        }
        self.len += s;
        Ok(())
    }

    /// Positions filled.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether no position is filled.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The context limit.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.max
    }

    /// Fills this empty cache with a copy of `src`'s first `positions` positions. The two
    /// share no memory, so what either writes next the other never sees.
    ///
    /// # Errors
    /// [`ArchError::Input`] if this cache is not empty or does not match `src`, or `src`
    /// holds fewer positions.
    pub fn restore_from(&mut self, src: &Self, positions: usize) -> Result<(), ArchError> {
        let same = self.k.len() == src.k.len()
            && self.kv_heads == src.kv_heads
            && self.head_dim == src.head_dim;
        if !self.is_empty() || !same || positions > src.len || positions > self.max {
            return Err(ArchError::Input(format!(
                "cannot restore {positions} positions of {src:?} into {self:?}"
            )));
        }
        let n = positions as u64;
        for l in 0..self.k.len() {
            self.reserve(l, positions)?;
            for (to, from) in [(&mut self.k[l], &src.k[l]), (&mut self.v[l], &src.v[l])] {
                let mut all = to.view_mut()?;
                for g in 0..self.kv_heads {
                    let dst = contiguous_mut(all.reborrow().select(0, g)?.narrow(0, 0, n)?)?;
                    dst.copy_from_slice(contiguous(from.view().select(0, g)?.narrow(0, 0, n)?)?);
                }
            }
        }
        self.len = positions;
        Ok(())
    }

    /// Forgets every position.
    pub const fn clear(&mut self) {
        self.len = 0;
    }

    /// An empty layer tensor with room for `positions`.
    fn layer_tensor(&self, positions: usize) -> Result<Tensor<f32>, ArchError> {
        let shape = Shape::new(&[self.kv_heads, positions as u64, self.head_dim])
            .map_err(ComputeError::from)?;
        Ok(Tensor::zeros(DType::F32, shape)?)
    }

    /// Positions layer `l` has room for.
    fn room(&self, l: usize) -> usize {
        usize::try_from(self.k[l].desc().dims()[1]).unwrap_or(usize::MAX)
    }

    /// Makes room in layer `l` for `positions`, growing by whole chunks
    /// ([`Self::reserved_for`]) and keeping the filled positions. One layer is copied
    /// at a time, so growing never holds two copies of the whole cache.
    fn reserve(&mut self, l: usize, positions: usize) -> Result<(), ArchError> {
        if self.room(l) >= positions {
            return Ok(());
        }
        let filled = self.len as u64;
        let room = Self::reserved_for(positions, self.max);
        for old in [&mut self.k[l], &mut self.v[l]] {
            let shape = Shape::new(&[self.kv_heads, room as u64, self.head_dim])
                .map_err(ComputeError::from)?;
            let mut new = Tensor::zeros(DType::F32, shape)?;
            let mut dst = new.view_mut()?;
            for g in 0..self.kv_heads {
                let from = old.view().select(0, g)?.narrow(0, 0, filled)?;
                let to = dst.reborrow().select(0, g)?.narrow(0, 0, filled)?;
                contiguous_mut(to)?.copy_from_slice(contiguous(from)?);
            }
            *old = new;
        }
        Ok(())
    }

    /// Writes the keys and values of `s` new positions, from `self.len` on, into layer
    /// `l` (room already reserved): `k` and `v` hold `s` rows of every KV head's
    /// `head_dim` values. One `narrow` view per head covers the whole step.
    fn store(&mut self, l: usize, s: usize, k: &[f32], v: &[f32]) -> Result<(), ArchError> {
        let d = usize::try_from(self.head_dim).map_err(|_| ArchError::Input("head_dim".into()))?;
        let row = k.len() / s.max(1);
        let (p0, n) = (self.len as u64, s as u64);
        for (t, src) in [(&mut self.k[l], k), (&mut self.v[l], v)] {
            let mut all = t.view_mut()?;
            for g in 0..self.kv_heads {
                let dst = contiguous_mut(all.reborrow().select(0, g)?.narrow(0, p0, n)?)?;
                let off = usize::try_from(g).unwrap_or(usize::MAX) * d;
                for (to, from) in dst.chunks_exact_mut(d).zip(src.chunks_exact(row)) {
                    to.copy_from_slice(&from[off..off + d]);
                }
            }
        }
        Ok(())
    }

    /// Per KV head of layer `l`, the keys and the values of positions `0..positions`,
    /// each `positions × head_dim` contiguous floats.
    #[allow(clippy::type_complexity)]
    fn heads(&self, l: usize, positions: usize) -> Result<(Vec<&[f32]>, Vec<&[f32]>), ArchError> {
        let n = positions as u64;
        let rows = |t| head_rows(t, self.kv_heads, n);
        Ok((rows(&self.k[l])?, rows(&self.v[l])?))
    }
}

/// Positions `0..n` of every KV head of one layer tensor.
fn head_rows(t: &Tensor<f32>, kv_heads: u64, n: u64) -> Result<Vec<&[f32]>, ArchError> {
    (0..kv_heads)
        .map(|g| contiguous(t.view().select(0, g)?.narrow(0, 0, n)?))
        .collect()
}

/// The elements of a view the KV cache built as contiguous.
fn contiguous(v: TensorView<'_, f32>) -> Result<&[f32], ArchError> {
    v.contiguous_data()?
        .ok_or_else(|| ArchError::Input(format!("KV cache view {:?} is not contiguous", v.desc())))
}

/// The elements of a mutable view the KV cache built as contiguous.
fn contiguous_mut(v: TensorViewMut<'_, f32>) -> Result<&mut [f32], ArchError> {
    let desc = *v.desc();
    v.contiguous_data_mut()?
        .ok_or_else(|| ArchError::Input(format!("KV cache view {desc:?} is not contiguous")))
}

struct LayerNorms {
    attn: Vec<f32>,
    ffn: Vec<f32>,
    q: Option<Vec<f32>>,
    k: Option<Vec<f32>>,
}

/// Tensor-name patterns the forward pass reads, the same for every FFN kind.
#[derive(Clone, Debug)]
struct Names {
    embedding: String,
    output: String,
    final_norm: String,
    attn_norm: String,
    q: String,
    k: String,
    v: String,
    o: String,
    q_norm: String,
    k_norm: String,
    ffn_norm: String,
}

/// What the forward pass reads from a config, the same for every FFN kind.
#[derive(Clone, Debug)]
struct DecoderConfig {
    hidden: u64,
    layers: u32,
    heads: u64,
    kv_heads: u64,
    head_dim: u64,
    vocab: u64,
    rms_eps: f64,
    rope_theta: f64,
    clip_qkv: Option<f64>,
    tie_embeddings: bool,
    qk_norm: QkNorm,
    names: Names,
}

/// Each layer's FFN.
#[derive(Clone, Debug)]
enum Ffn {
    /// Routed SwiGLU experts (`transformer_moe`).
    Moe {
        experts: u32,
        top_k: u32,
        norm_topk_prob: bool,
        router: String,
    },
    /// One dense SwiGLU (`transformer_dense`).
    Dense {
        gate: String,
        up: String,
        down: String,
    },
}

impl DecoderConfig {
    fn from_moe(c: &MoeConfig) -> (Self, Ffn) {
        let n = &c.names;
        (
            Self {
                hidden: c.hidden,
                layers: c.layers,
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dim: c.head_dim,
                vocab: c.vocab,
                rms_eps: c.rms_eps,
                rope_theta: c.rope_theta,
                clip_qkv: c.clip_qkv,
                tie_embeddings: c.tie_embeddings,
                qk_norm: c.qk_norm,
                names: Names {
                    embedding: n.embedding.clone(),
                    output: n.output.clone(),
                    final_norm: n.final_norm.clone(),
                    attn_norm: n.attn_norm.clone(),
                    q: n.q.clone(),
                    k: n.k.clone(),
                    v: n.v.clone(),
                    o: n.o.clone(),
                    q_norm: n.q_norm.clone(),
                    k_norm: n.k_norm.clone(),
                    ffn_norm: n.ffn_norm.clone(),
                },
            },
            Ffn::Moe {
                experts: c.experts,
                top_k: c.top_k,
                norm_topk_prob: c.norm_topk_prob,
                router: n.router.clone(),
            },
        )
    }

    fn from_dense(c: &DenseConfig) -> (Self, Ffn) {
        let n = &c.names;
        (
            Self {
                hidden: c.hidden,
                layers: c.layers,
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dim: c.head_dim,
                vocab: c.vocab,
                rms_eps: c.rms_eps,
                rope_theta: c.rope_theta,
                clip_qkv: c.clip_qkv,
                tie_embeddings: c.tie_embeddings,
                qk_norm: c.qk_norm,
                names: Names {
                    embedding: n.embedding.clone(),
                    output: n.output.clone(),
                    final_norm: n.final_norm.clone(),
                    attn_norm: n.attn_norm.clone(),
                    q: n.q.clone(),
                    k: n.k.clone(),
                    v: n.v.clone(),
                    o: n.o.clone(),
                    q_norm: n.q_norm.clone(),
                    k_norm: n.k_norm.clone(),
                    ffn_norm: n.ffn_norm.clone(),
                },
            },
            Ffn::Dense {
                gate: n.gate.clone(),
                up: n.up.clone(),
                down: n.down.clone(),
            },
        )
    }
}

/// The forward pass of the built-in decoder-only transformers: `transformer_moe`
/// (routed SwiGLU experts) and `transformer_dense` (one dense SwiGLU per layer). The
/// embedding, attention, KV cache and head are shared; only the FFN differs.
pub struct Transformer {
    cfg: DecoderConfig,
    ffn: Ffn,
    moe: Option<MoeConfig>,
    kv_heads: usize,
    rope: Rope,
    norms: Vec<LayerNorms>,
    final_norm: Vec<f32>,
}

impl std::fmt::Debug for Transformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transformer")
            .field("layers", &self.cfg.layers)
            .field("hidden", &self.cfg.hidden)
            .finish_non_exhaustive()
    }
}

fn usize_of(v: u64) -> Result<usize, ArchError> {
    usize::try_from(v)
        .map_err(|_| ArchError::Input(format!("dimension {v} does not fit in memory")))
}

/// A per-layer tensor name: `pattern` with `{L}` replaced by `l`.
#[must_use]
pub fn layer_name(pattern: &str, l: u32) -> String {
    pattern.replace("{L}", &l.to_string())
}

/// A 1-D weight as f32.
fn vector(
    dense: &dyn DenseWeights,
    be: &dyn ComputeBackend,
    name: &str,
    len: usize,
) -> Result<Vec<f32>, ArchError> {
    let v = dense.view(name)?;
    if v.rows() != 1 || v.cols() != len {
        return Err(ArchError::Input(format!(
            "{name} is {} x {}, expected 1 x {len}",
            v.rows(),
            v.cols()
        )));
    }
    let mut out = vec![0f32; len];
    be.dequant_row(&v, 0, &mut out)?;
    Ok(out)
}

impl Transformer {
    /// Prepares the model: parses nothing (the config is already validated) but
    /// dequantizes the norm weights once.
    ///
    /// # Errors
    /// [`ArchError::Input`] if `cfg` is neither a `transformer_moe` nor a
    /// `transformer_dense` config, and [`ArchError::Missing`] for absent weights.
    pub fn new(
        cfg: &ArchConfig,
        dense: &dyn DenseWeights,
        be: &dyn ComputeBackend,
    ) -> Result<Self, ArchError> {
        let (c, ffn, moe) = if let Some(m) = cfg.specific.downcast_ref::<MoeConfig>() {
            if m.router_scoring != RouterScoring::Softmax {
                return Err(ArchError::Input(
                    "sigmoid router scoring is not implemented yet".into(),
                ));
            }
            let (c, ffn) = DecoderConfig::from_moe(m);
            (c, ffn, Some(m.clone()))
        } else if let Some(d) = cfg.specific.downcast_ref::<DenseConfig>() {
            let (c, ffn) = DecoderConfig::from_dense(d);
            (c, ffn, None)
        } else {
            return Err(ArchError::Input(
                "config is neither a transformer_moe nor a transformer_dense config".into(),
            ));
        };
        let h = usize_of(c.hidden)?;
        let d = usize_of(c.head_dim)?;
        let (q_out, kv_out) = (
            usize_of(c.heads * c.head_dim)?,
            usize_of(c.kv_heads * c.head_dim)?,
        );
        let mut norms = Vec::with_capacity(c.layers as usize);
        for l in 0..c.layers {
            let n = |p: &str| layer_name(p, l);
            let (q, k) = match c.qk_norm {
                QkNorm::None => (None, None),
                QkNorm::Full => (
                    Some(vector(dense, be, &n(&c.names.q_norm), q_out)?),
                    Some(vector(dense, be, &n(&c.names.k_norm), kv_out)?),
                ),
                QkNorm::PerHead => (
                    Some(vector(dense, be, &n(&c.names.q_norm), d)?),
                    Some(vector(dense, be, &n(&c.names.k_norm), d)?),
                ),
            };
            norms.push(LayerNorms {
                attn: vector(dense, be, &n(&c.names.attn_norm), h)?,
                ffn: vector(dense, be, &n(&c.names.ffn_norm), h)?,
                q,
                k,
            });
        }
        let final_norm = vector(dense, be, &c.names.final_norm, h)?;
        Ok(Self {
            kv_heads: usize_of(c.kv_heads)?,
            rope: Rope::new(d, c.rope_theta),
            norms,
            final_norm,
            cfg: c,
            ffn,
            moe,
        })
    }

    /// Every dense matmul weight of the forward pass, by kind: what an execution plan
    /// can place on a device (ADR-007, ADR-011 §5). Routed experts are not listed; a
    /// dense FFN is.
    #[must_use]
    pub fn dense_matmuls(&self) -> Vec<(DenseMatmul, String)> {
        let c = &self.cfg;
        let mut v = Vec::new();
        for l in 0..c.layers {
            let n = |p: &str| layer_name(p, l);
            for p in [&c.names.q, &c.names.k, &c.names.v, &c.names.o] {
                v.push((DenseMatmul::Attention, n(p)));
            }
            match &self.ffn {
                Ffn::Moe { router, .. } => v.push((DenseMatmul::Router, n(router))),
                Ffn::Dense { gate, up, down } => {
                    for p in [gate, up, down] {
                        v.push((DenseMatmul::Ffn, n(p)));
                    }
                }
            }
        }
        let head = if c.tie_embeddings {
            &c.names.embedding
        } else {
            &c.names.output
        };
        v.push((DenseMatmul::Head, head.clone()));
        v
    }

    /// The `transformer_moe` config, if this is a mixture of experts.
    #[must_use]
    pub const fn moe_config(&self) -> Option<&MoeConfig> {
        self.moe.as_ref()
    }

    /// Whether each layer's FFN is routed experts (else one dense SwiGLU).
    #[must_use]
    pub const fn is_moe(&self) -> bool {
        matches!(self.ffn, Ffn::Moe { .. })
    }

    /// An empty KV cache for up to `max_positions` positions.
    #[must_use]
    pub fn new_cache(&self, max_positions: usize) -> KvCache {
        let mut cache = KvCache {
            k: Vec::new(),
            v: Vec::new(),
            kv_heads: self.kv_heads as u64,
            head_dim: self.cfg.head_dim,
            len: 0,
            max: max_positions,
            device: None,
            device_bytes: 0,
        };
        // Zero-sized, so the layers can share it; the first step replaces each.
        if let Ok(empty) = cache.layer_tensor(0) {
            cache.k = vec![empty.clone(); self.norms.len()];
            cache.v = cache.k.clone();
        }
        cache
    }

    /// Runs `tokens` (at positions `cache.len()..`) through the model and appends their
    /// keys and values to `cache`. Returns logits: the last token's, or every token's
    /// (`tokens.len() × vocab`, row-major).
    ///
    /// On an error the cache keeps the positions it held before the call, so the same
    /// tokens can be run again on it (a failed expert read must not leave
    /// half a step of keys and values behind).
    ///
    /// # Errors
    /// [`ArchError::Input`] for an out-of-range token or a full context; any kernel,
    /// weight or expert error.
    pub fn forward(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        src: Sources<'_>,
        logits: Logits,
        trace: Option<Tracer<'_>>,
    ) -> Result<Vec<f32>, ArchError> {
        let mut one = [Segment {
            tokens,
            cache,
            logits,
        }];
        let mut out = self.forward_segments(&mut one, src, trace)?;
        Ok(out.pop().unwrap_or_default())
    }

    /// Runs several sequences through the model in one pass (ADR-017): each segment's
    /// tokens at positions `cache.len()..` of its own cache. Returns each segment's
    /// logits, in segment order, as [`Transformer::forward`] would return them for
    /// that segment alone, with the same bits.
    ///
    /// Every step but attention is per token, and matmuls are batch-invariant, so the
    /// segments share each weight read (and each routed expert's load) while every
    /// token's arithmetic stays what it would be alone. Attention reads only the
    /// token's own cache, at its own position.
    ///
    /// On an error every cache keeps the positions it held before the call.
    ///
    /// # Errors
    /// As [`Transformer::forward`], for any segment.
    pub fn forward_batch(
        &self,
        segments: &mut [Segment<'_>],
        src: Sources<'_>,
    ) -> Result<Vec<Vec<f32>>, ArchError> {
        self.forward_segments(segments, src, None)
    }

    /// Checks one segment on its own, as a pass over it would: its cache has this
    /// model's layers, its tokens fit the context, and every token id is in the
    /// vocabulary. A scheduler combining sequences answers a failing segment alone
    /// with this error instead of failing everything that shares its pass (ADR-017).
    ///
    /// # Errors
    /// The [`ArchError::Input`] a pass over the segment would return.
    pub fn check_segment(
        &self,
        tokens: &[u32],
        cache: &KvCache,
        dense: &dyn DenseWeights,
    ) -> Result<(), ArchError> {
        self.check_room(cache, tokens.len())?;
        let rows = dense.view(&self.cfg.names.embedding)?.rows();
        match tokens.iter().find(|&&id| id as usize >= rows) {
            Some(id) => Err(ArchError::Input(format!(
                "token id {id} >= vocabulary {rows}"
            ))),
            None => Ok(()),
        }
    }

    fn check_room(&self, cache: &KvCache, n: usize) -> Result<(), ArchError> {
        if cache.k.len() != self.norms.len() || cache.v.len() != self.norms.len() {
            return Err(ArchError::Input(format!(
                "KV cache has {} layers, the model {}",
                cache.k.len(),
                self.norms.len()
            )));
        }
        if cache.len + n > cache.max {
            return Err(ArchError::Input(format!(
                "context full: {} positions + {n} new > {}",
                cache.len, cache.max
            )));
        }
        Ok(())
    }

    /// Positions are counted in the caches only once the whole pass has succeeded, so
    /// a failure anywhere leaves every cache as it was.
    #[allow(clippy::too_many_lines)]
    fn forward_segments(
        &self,
        segments: &mut [Segment<'_>],
        src: Sources<'_>,
        mut trace: Option<Tracer<'_>>,
    ) -> Result<Vec<Vec<f32>>, ArchError> {
        let (dense, experts, be) = (src.dense, src.experts, src.backend);
        let c = &self.cfg;
        // Each segment's first token in the batch, token count and first position.
        let mut layout = Vec::with_capacity(segments.len());
        let mut tokens = Vec::new();
        for g in segments.iter() {
            let (p0, n) = (g.cache.len, g.tokens.len());
            self.check_room(g.cache, n)?;
            layout.push((tokens.len(), n, p0));
            tokens.extend_from_slice(g.tokens);
        }
        let s = tokens.len();
        if s == 0 {
            return Ok(vec![Vec::new(); segments.len()]);
        }
        // Per token: its segment and its position in that segment's sequence.
        let mut seg_of = Vec::with_capacity(s);
        let mut pos = Vec::with_capacity(s);
        for (i, &(_, n, p0)) in layout.iter().enumerate() {
            seg_of.extend(std::iter::repeat_n(i, n));
            pos.extend(p0..p0 + n);
        }
        let h = usize_of(c.hidden)?;
        let d = usize_of(c.head_dim)?;
        let heads = usize_of(c.heads)?;
        let kvh = usize_of(c.kv_heads)?;
        let group = heads / kvh;
        let (q_out, kv_out) = (heads * d, kvh * d);
        let (ne, k_top, norm_topk_prob) = match &self.ffn {
            Ffn::Moe {
                experts,
                top_k,
                norm_topk_prob,
                ..
            } => (*experts as usize, *top_k as usize, *norm_topk_prob),
            Ffn::Dense { .. } => (0, 0, false),
        };
        let vocab = usize_of(c.vocab)?;
        let mut emit =
            |point, layer: Option<u32>, token: usize, values: &[f32], experts: &[u32]| {
                if let Some(t) = trace.as_deref_mut() {
                    t(TraceEvent {
                        point,
                        layer,
                        token,
                        values,
                        experts,
                    });
                }
            };

        // Embedding.
        let emb = dense.view(&c.names.embedding)?;
        let mut hs = vec![0f32; s * h];
        for (t, &id) in tokens.iter().enumerate() {
            if id as usize >= emb.rows() {
                return Err(ArchError::Input(format!(
                    "token id {id} >= vocabulary {}",
                    emb.rows()
                )));
            }
            be.dequant_row(&emb, id as usize, &mut hs[t * h..(t + 1) * h])?;
            emit(TracePoint::Embedding, None, t, &hs[t * h..(t + 1) * h], &[]);
        }

        let mut x = vec![0f32; s * h];
        let mut q = vec![0f32; s * q_out];
        let mut k = vec![0f32; s * kv_out];
        let mut v = vec![0f32; s * kv_out];
        let mut att = vec![0f32; s * q_out];
        let mut o = vec![0f32; s * h];
        let mut router = vec![0f32; s * ne];
        let mut rope_cs = vec![0f32; 2 * self.rope.half()];
        // The lookahead router runs during decode only: a prompt batch routes to most
        // experts of every layer, so there is nothing to predict.
        let lookahead = s == 1 && self.is_moe() && experts.wants_lookahead();
        let mut ahead = if lookahead {
            vec![0f32; ne]
        } else {
            Vec::new()
        };
        let mut ahead_x = if lookahead { vec![0f32; h] } else { Vec::new() };

        for (l, norms) in self.norms.iter().enumerate() {
            let li = u32::try_from(l).map_err(|_| ArchError::Input("too many layers".into()))?;
            let n = |p: &str| layer_name(p, li);

            // Attention.
            x.copy_from_slice(&hs);
            for xt in x.chunks_exact_mut(h) {
                rms_norm(xt, &norms.attn, c.rms_eps);
            }
            // One submission: the three projections share their input.
            be.matmul_many(&mut [
                MatMul {
                    w: dense.view(&n(&c.names.q))?,
                    xs: &x,
                    s,
                    ys: &mut q,
                },
                MatMul {
                    w: dense.view(&n(&c.names.k))?,
                    xs: &x,
                    s,
                    ys: &mut k,
                },
                MatMul {
                    w: dense.view(&n(&c.names.v))?,
                    xs: &x,
                    s,
                    ys: &mut v,
                },
            ])?;
            if let Some(clip) = c.clip_qkv {
                #[allow(clippy::cast_possible_truncation)]
                let clip = clip as f32;
                for z in q.iter_mut().chain(k.iter_mut()).chain(v.iter_mut()) {
                    *z = z.clamp(-clip, clip);
                }
            }
            match (c.qk_norm, &norms.q, &norms.k) {
                (QkNorm::Full, Some(qn), Some(kn)) => {
                    q.chunks_exact_mut(q_out)
                        .for_each(|r| rms_norm(r, qn, c.rms_eps));
                    k.chunks_exact_mut(kv_out)
                        .for_each(|r| rms_norm(r, kn, c.rms_eps));
                }
                (QkNorm::PerHead, Some(qn), Some(kn)) => {
                    q.chunks_exact_mut(d)
                        .for_each(|r| rms_norm(r, qn, c.rms_eps));
                    k.chunks_exact_mut(d)
                        .for_each(|r| rms_norm(r, kn, c.rms_eps));
                }
                _ => {}
            }
            for t in 0..s {
                self.rope.table(pos[t], &mut rope_cs);
                q[t * q_out..(t + 1) * q_out]
                    .chunks_exact_mut(d)
                    .for_each(|hd| Rope::rotate(hd, &rope_cs));
                k[t * kv_out..(t + 1) * kv_out]
                    .chunks_exact_mut(d)
                    .for_each(|hd| Rope::rotate(hd, &rope_cs));
            }
            for (g, &(first, n, p0)) in segments.iter_mut().zip(&layout) {
                if n > 0 {
                    let rows = first * kv_out..(first + n) * kv_out;
                    g.cache.reserve(l, p0 + n)?;
                    g.cache.store(l, n, &k[rows.clone()], &v[rows])?;
                }
            }
            // Every (token, head) pair is independent once each cache holds its
            // segment, so they run in parallel; each is the same sequential `attend`
            // over its own sequence's keys and values.
            let kv = segments
                .iter()
                .zip(&layout)
                .map(|(g, &(_, n, p0))| g.cache.heads(l, p0 + n))
                .collect::<Result<Vec<_>, _>>()?;
            att.par_chunks_mut(d)
                .enumerate()
                .for_each_init(Vec::new, |scores, (i, out)| {
                    let (t, hd) = (i / heads, i % heads);
                    let g = hd / group;
                    let (kl, vl) = &kv[seg_of[t]];
                    let qs = &q[t * q_out + hd * d..t * q_out + (hd + 1) * d];
                    attend(qs, kl[g], vl[g], pos[t], out, scores);
                });
            drop(kv);
            be.matmul(&dense.view(&n(&c.names.o))?, &att, s, &mut o)?;
            for t in 0..s {
                emit(
                    TracePoint::Attention,
                    Some(li),
                    t,
                    &o[t * h..(t + 1) * h],
                    &[],
                );
            }
            for (a, b) in hs.iter_mut().zip(&o) {
                *a += b;
            }

            // Lookahead: the next layer's MoE norm and router on this state. Its logits
            // only go to the provider as a hint; routing below uses this layer's own.
            if let (Some(next), Ffn::Moe { router, .. }) =
                (self.norms.get(l + 1).filter(|_| lookahead), &self.ffn)
            {
                ahead_x.copy_from_slice(&hs);
                rms_norm(&mut ahead_x, &next.ffn, c.rms_eps);
                let ln = layer_name(router, li + 1);
                be.matmul(&dense.view(&ln)?, &ahead_x, 1, &mut ahead)?;
                experts.lookahead(LayerIdx(li + 1), &ahead);
            }

            // FFN.
            x.copy_from_slice(&hs);
            for xt in x.chunks_exact_mut(h) {
                rms_norm(xt, &norms.ffn, c.rms_eps);
            }
            match &self.ffn {
                Ffn::Dense { gate, up, down } => {
                    let (wg, wu) = (dense.view(&n(gate))?, dense.view(&n(up))?);
                    let ffn = wg.rows();
                    let mut g = vec![0f32; s * ffn];
                    let mut u = vec![0f32; s * ffn];
                    be.matmul_many(&mut [
                        MatMul {
                            w: wg,
                            xs: &x,
                            s,
                            ys: &mut g,
                        },
                        MatMul {
                            w: wu,
                            xs: &x,
                            s,
                            ys: &mut u,
                        },
                    ])?;
                    swiglu(&mut g, &u);
                    be.matmul(&dense.view(&n(down))?, &g, s, &mut o)?;
                    for t in 0..s {
                        emit(TracePoint::Ffn, Some(li), t, &o[t * h..(t + 1) * h], &[]);
                    }
                }
                Ffn::Moe {
                    router: router_name,
                    ..
                } => {
                    be.matmul(&dense.view(&n(router_name))?, &x, s, &mut router)?;
                    // Per token: top-K experts by probability (ties: lower id), in rank order.
                    let mut picks: Vec<(Vec<u32>, Vec<f32>)> = Vec::with_capacity(s);
                    for t in 0..s {
                        let logits_t = &router[t * ne..(t + 1) * ne];
                        emit(TracePoint::RouterLogits, Some(li), t, logits_t, &[]);
                        let (idx, w) = route(logits_t, k_top, norm_topk_prob);
                        emit(TracePoint::RoutingWeights, Some(li), t, &w, &idx);
                        picks.push((idx, w));
                    }
                    // Expert-major: each expert once, over every token routed to it, then each
                    // token's outputs combined in rank order.
                    let mut contrib = vec![0f32; s * k_top * h];
                    let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ne];
                    for (t, (idx, _)) in picks.iter().enumerate() {
                        for (r, &e) in idx.iter().enumerate() {
                            by_expert[e as usize].push((t, r));
                        }
                    }
                    // Every routed expert of the layer is requested at once, so the provider can
                    // start all its loads first, and may compute several at once. Each writes
                    // only its own rank slots of `contrib`, so neither the order they come back
                    // in nor concurrency can change the result.
                    let used: Vec<ExpertIdx> = by_expert
                        .iter()
                        .enumerate()
                        .filter(|(_, u)| !u.is_empty())
                        .map(|(e, _)| ExpertIdx(u32::try_from(e).unwrap_or(u32::MAX)))
                        .collect();
                    let slots = std::sync::Mutex::new(&mut contrib);
                    // Writes an expert's outputs `y` (one row per user) into its users'
                    // rank slots. A poisoned lock means another expert panicked
                    // mid-copy; its slots are not ours, so ours are still written
                    // correctly.
                    let store = |users: &[(usize, usize)], y: &[f32]| {
                        let mut contrib = slots
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        for (j, &(t, r)) in users.iter().enumerate() {
                            let dst = (t * k_top + r) * h;
                            contrib[dst..dst + h].copy_from_slice(&y[j * h..(j + 1) * h]);
                        }
                    };
                    // An expert's input rows: its users' rows of `x`.
                    let gather = |users: &[(usize, usize)]| {
                        let mut xe = vec![0f32; users.len() * h];
                        for (j, &(t, _)) in users.iter().enumerate() {
                            xe[j * h..(j + 1) * h].copy_from_slice(&x[t * h..(t + 1) * h]);
                        }
                        xe
                    };
                    let each = |ei: ExpertIdx, w: &ExpertWeights<'_>| {
                        let users = &by_expert[ei.0 as usize];
                        let n_t = users.len();
                        let xe = gather(users);
                        let ffn = w.gate.rows();
                        let mut g = vec![0f32; n_t * ffn];
                        let mut u = vec![0f32; n_t * ffn];
                        be.matmul_many(&mut [
                            MatMul {
                                w: w.gate,
                                xs: &xe,
                                s: n_t,
                                ys: &mut g,
                            },
                            MatMul {
                                w: w.up,
                                xs: &xe,
                                s: n_t,
                                ys: &mut u,
                            },
                        ])?;
                        swiglu(&mut g, &u);
                        let mut y = vec![0f32; n_t * h];
                        be.matmul(&w.down, &g, n_t, &mut y)?;
                        store(users, &y);
                        Ok(())
                    };
                    // The largest number of rows any expert computes: a fused block is
                    // chosen by shape (decode, or a few concurrent sequences), not always.
                    let max_rows = used
                        .iter()
                        .map(|ei| by_expert[ei.0 as usize].len())
                        .max()
                        .unwrap_or(0);
                    if be.fuses_gated_ffn(used.len(), max_rows) {
                        // Every expert of the layer as one fused block when all are
                        // resident: gate and up, SwiGLU and down in one submission. Each
                        // output has the separate steps' bits (GatedFfn).
                        let mut group = |ws: &[ExpertWeights<'_>]| -> Result<(), ArchError> {
                            let users: Vec<&[(usize, usize)]> = used
                                .iter()
                                .map(|ei| by_expert[ei.0 as usize].as_slice())
                                .collect();
                            let xes: Vec<Option<Vec<f32>>> = users
                                .iter()
                                .map(|u| {
                                    let all = u.len() == s
                                        && u.iter().enumerate().all(|(j, &(t, _))| j == t);
                                    (!all).then(|| gather(u))
                                })
                                .collect();
                            let mut ys: Vec<Vec<f32>> =
                                users.iter().map(|u| vec![0f32; u.len() * h]).collect();
                            let mut ops: Vec<GatedFfn<'_>> = ws
                                .iter()
                                .zip(&users)
                                .zip(&xes)
                                .zip(ys.iter_mut())
                                .map(|(((w, u), xe), y)| GatedFfn {
                                    gate: w.gate,
                                    up: w.up,
                                    down: w.down,
                                    xs: xe.as_deref().unwrap_or(&x),
                                    s: u.len(),
                                    ys: y,
                                })
                                .collect();
                            be.gated_ffn_many(&mut ops)?;
                            drop(ops);
                            for (u, y) in users.iter().zip(&ys) {
                                store(u, y);
                            }
                            Ok(())
                        };
                        experts.with_experts_grouped(LayerIdx(li), &used, &mut group, &each)?;
                    } else if be.groups_matmuls() {
                        // Every expert at once: one call for every gate and up, then one
                        // for every down. Each matmul has the same weight, input and
                        // kernel as on the per-expert path, so the same bits.
                        let mut group = |ws: &[ExpertWeights<'_>]| -> Result<(), ArchError> {
                            let users: Vec<&[(usize, usize)]> = used
                                .iter()
                                .map(|ei| by_expert[ei.0 as usize].as_slice())
                                .collect();
                            // An expert used by every token in order reads `x` itself, so
                            // the backend sees one shared input.
                            let xes: Vec<Option<Vec<f32>>> = users
                                .iter()
                                .map(|u| {
                                    let all = u.len() == s
                                        && u.iter().enumerate().all(|(j, &(t, _))| j == t);
                                    (!all).then(|| gather(u))
                                })
                                .collect();
                            let mut gs: Vec<Vec<f32>> = Vec::with_capacity(ws.len());
                            let mut us: Vec<Vec<f32>> = Vec::with_capacity(ws.len());
                            for (w, u) in ws.iter().zip(&users) {
                                gs.push(vec![0f32; u.len() * w.gate.rows()]);
                                us.push(vec![0f32; u.len() * w.up.rows()]);
                            }
                            let mut ops = Vec::with_capacity(2 * ws.len());
                            for ((((w, u), xe), g), up) in ws
                                .iter()
                                .zip(&users)
                                .zip(&xes)
                                .zip(gs.iter_mut())
                                .zip(us.iter_mut())
                            {
                                let xs = xe.as_deref().unwrap_or(&x);
                                ops.push(MatMul {
                                    w: w.gate,
                                    xs,
                                    s: u.len(),
                                    ys: g,
                                });
                                ops.push(MatMul {
                                    w: w.up,
                                    xs,
                                    s: u.len(),
                                    ys: up,
                                });
                            }
                            be.matmul_many(&mut ops)?;
                            drop(ops);
                            for (g, u) in gs.iter_mut().zip(&us) {
                                swiglu(g, u);
                            }
                            let mut ys: Vec<Vec<f32>> =
                                users.iter().map(|u| vec![0f32; u.len() * h]).collect();
                            let mut ops: Vec<MatMul<'_>> = ws
                                .iter()
                                .zip(&users)
                                .zip(&gs)
                                .zip(ys.iter_mut())
                                .map(|(((w, u), g), y)| MatMul {
                                    w: w.down,
                                    xs: g,
                                    s: u.len(),
                                    ys: y,
                                })
                                .collect();
                            be.matmul_many(&mut ops)?;
                            drop(ops);
                            for (u, y) in users.iter().zip(&ys) {
                                store(u, y);
                            }
                            Ok(())
                        };
                        experts.with_experts_grouped(LayerIdx(li), &used, &mut group, &each)?;
                    } else {
                        experts.with_experts(LayerIdx(li), &used, &each)?;
                    }
                    for (t, (_, w)) in picks.iter().enumerate() {
                        let out = &mut o[t * h..(t + 1) * h];
                        out.fill(0.0);
                        for (r, wr) in w.iter().enumerate() {
                            let src = &contrib[(t * k_top + r) * h..(t * k_top + r + 1) * h];
                            for (a, b) in out.iter_mut().zip(src) {
                                *a = wr.mul_add(*b, *a);
                            }
                        }
                        emit(TracePoint::Moe, Some(li), t, out, &[]);
                    }
                }
            }
            for (a, b) in hs.iter_mut().zip(&o) {
                *a += b;
            }
            for t in 0..s {
                emit(
                    TracePoint::Hidden,
                    Some(li),
                    t,
                    &hs[t * h..(t + 1) * h],
                    &[],
                );
            }
        }
        // Final norm and head, for the tokens whose logits are wanted: one matmul over
        // every segment's rows.
        let wanted: Vec<std::ops::Range<usize>> = segments
            .iter()
            .zip(&layout)
            .map(|(g, &(first, n, _))| match (g.logits, n) {
                (_, 0) => first..first,
                (Logits::Last, _) => first + n - 1..first + n,
                (Logits::All, _) => first..first + n,
            })
            .collect();
        let rows: Vec<usize> = wanted.iter().flat_map(Clone::clone).collect();
        let m = rows.len();
        let mut xf = vec![0f32; m * h];
        for (dst, &t) in xf.chunks_exact_mut(h).zip(&rows) {
            dst.copy_from_slice(&hs[t * h..(t + 1) * h]);
            rms_norm(dst, &self.final_norm, c.rms_eps);
        }
        let head = if c.tie_embeddings {
            emb
        } else {
            dense.view(&c.names.output)?
        };
        let mut out = vec![0f32; m * vocab];
        be.matmul(&head, &xf, m, &mut out)?;
        for (j, &t) in rows.iter().enumerate() {
            emit(
                TracePoint::Logits,
                None,
                t,
                &out[j * vocab..(j + 1) * vocab],
                &[],
            );
        }
        for (g, &(_, n, _)) in segments.iter_mut().zip(&layout) {
            g.cache.len += n;
        }
        if let [_] = wanted.as_slice() {
            return Ok(vec![out]);
        }
        let mut at = 0;
        Ok(wanted
            .iter()
            .map(|r| {
                let n = r.len() * vocab;
                at += n;
                out[at - n..at].to_vec()
            })
            .collect())
    }
}
