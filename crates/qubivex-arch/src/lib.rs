//! Architectures: the forward pass of a validated model (QUBIVEX_DESIGN §3).
//!
//! [`TransformerMoeModel`] runs `transformer_moe`, the architecture whose config
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

use qubivex_compute::{ComputeBackend, ComputeError, WeightView};
use qubivex_core::{ExpertIdx, LayerIdx};
use qubivex_model::ArchConfig;
use qubivex_model::arch::{MoeConfig, QkNorm, RouterScoring};
use thiserror::Error;

use crate::ops::{Rope, attend, rms_norm, softmax, swiglu};

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

/// The resident weights, by tensor name.
pub trait DenseWeights {
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
pub trait ExpertProvider {
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
}

/// Points of the forward pass that can be observed (numerical tests).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// A trace callback.
pub type Tracer<'t> = &'t mut dyn FnMut(TraceEvent<'_>);

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

/// Keys and values of past positions, per layer and KV head.
#[derive(Clone, Debug)]
pub struct KvCache {
    /// `[layer][kv_head]` → `len × head_dim`.
    k: Vec<Vec<Vec<f32>>>,
    v: Vec<Vec<Vec<f32>>>,
    len: usize,
    max: usize,
}

impl KvCache {
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

    /// Forgets every position.
    pub fn clear(&mut self) {
        for heads in self.k.iter_mut().chain(self.v.iter_mut()) {
            for h in heads {
                h.clear();
            }
        }
        self.len = 0;
    }
}

struct LayerNorms {
    attn: Vec<f32>,
    ffn: Vec<f32>,
    q: Option<Vec<f32>>,
    k: Option<Vec<f32>>,
}

/// The `transformer_moe` forward pass.
pub struct TransformerMoeModel {
    cfg: MoeConfig,
    kv_heads: usize,
    rope: Rope,
    norms: Vec<LayerNorms>,
    final_norm: Vec<f32>,
}

impl std::fmt::Debug for TransformerMoeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformerMoeModel")
            .field("layers", &self.cfg.layers)
            .field("hidden", &self.cfg.hidden)
            .finish_non_exhaustive()
    }
}

fn usize_of(v: u64) -> Result<usize, ArchError> {
    usize::try_from(v)
        .map_err(|_| ArchError::Input(format!("dimension {v} does not fit in memory")))
}

fn layer_name(pattern: &str, l: u32) -> String {
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

impl TransformerMoeModel {
    /// Prepares the model: parses nothing (the config is already validated) but
    /// dequantizes the norm weights once.
    ///
    /// # Errors
    /// [`ArchError::Input`] if `cfg` is not a `transformer_moe` config, and
    /// [`ArchError::Missing`] for absent weights.
    pub fn new(
        cfg: &ArchConfig,
        dense: &dyn DenseWeights,
        be: &dyn ComputeBackend,
    ) -> Result<Self, ArchError> {
        let c = cfg
            .specific
            .downcast_ref::<MoeConfig>()
            .ok_or_else(|| ArchError::Input("config is not a transformer_moe config".into()))?
            .clone();
        if c.router_scoring != RouterScoring::Softmax {
            return Err(ArchError::Input(
                "sigmoid router scoring is not implemented yet".into(),
            ));
        }
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
        })
    }

    /// The config.
    #[must_use]
    pub const fn config(&self) -> &MoeConfig {
        &self.cfg
    }

    /// An empty KV cache for up to `max_positions` positions.
    #[must_use]
    pub fn new_cache(&self, max_positions: usize) -> KvCache {
        let empty = || vec![vec![Vec::new(); self.kv_heads]; self.norms.len()];
        KvCache {
            k: empty(),
            v: empty(),
            len: 0,
            max: max_positions,
        }
    }

    /// Runs `tokens` (at positions `cache.len()..`) through the model and appends their
    /// keys and values to `cache`. Returns logits: the last token's, or every token's
    /// (`tokens.len() × vocab`, row-major).
    ///
    /// # Errors
    /// [`ArchError::Input`] for an out-of-range token or a full context; any kernel,
    /// weight or expert error.
    #[allow(clippy::too_many_lines)]
    pub fn forward(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        src: Sources<'_>,
        logits: Logits,
        mut trace: Option<Tracer<'_>>,
    ) -> Result<Vec<f32>, ArchError> {
        let (dense, experts, be) = (src.dense, src.experts, src.backend);
        let c = &self.cfg;
        let s = tokens.len();
        if s == 0 {
            return Ok(Vec::new());
        }
        let p0 = cache.len;
        if p0 + s > cache.max {
            return Err(ArchError::Input(format!(
                "context full: {} positions + {s} new > {}",
                p0, cache.max
            )));
        }
        let h = usize_of(c.hidden)?;
        let d = usize_of(c.head_dim)?;
        let heads = usize_of(c.heads)?;
        let kvh = usize_of(c.kv_heads)?;
        let group = heads / kvh;
        let (q_out, kv_out) = (heads * d, kvh * d);
        let ne = c.experts as usize;
        let k_top = c.top_k as usize;
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
        let mut scores = Vec::new();

        for (l, norms) in self.norms.iter().enumerate() {
            let li = u32::try_from(l).map_err(|_| ArchError::Input("too many layers".into()))?;
            let n = |p: &str| layer_name(p, li);

            // Attention.
            x.copy_from_slice(&hs);
            for xt in x.chunks_exact_mut(h) {
                rms_norm(xt, &norms.attn, c.rms_eps);
            }
            be.matmul(&dense.view(&n(&c.names.q))?, &x, s, &mut q)?;
            be.matmul(&dense.view(&n(&c.names.k))?, &x, s, &mut k)?;
            be.matmul(&dense.view(&n(&c.names.v))?, &x, s, &mut v)?;
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
                let pos = p0 + t;
                q[t * q_out..(t + 1) * q_out]
                    .chunks_exact_mut(d)
                    .for_each(|hd| self.rope.apply(hd, pos));
                k[t * kv_out..(t + 1) * kv_out]
                    .chunks_exact_mut(d)
                    .for_each(|hd| self.rope.apply(hd, pos));
                for g in 0..kvh {
                    let src = t * kv_out + g * d;
                    cache.k[l][g].extend_from_slice(&k[src..src + d]);
                    cache.v[l][g].extend_from_slice(&v[src..src + d]);
                }
            }
            for t in 0..s {
                let pos = p0 + t;
                for hd in 0..heads {
                    let g = hd / group;
                    let qs = &q[t * q_out + hd * d..t * q_out + (hd + 1) * d];
                    let out = &mut att[t * q_out + hd * d..t * q_out + (hd + 1) * d];
                    attend(qs, &cache.k[l][g], &cache.v[l][g], pos, out, &mut scores);
                }
            }
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

            // MoE.
            x.copy_from_slice(&hs);
            for xt in x.chunks_exact_mut(h) {
                rms_norm(xt, &norms.ffn, c.rms_eps);
            }
            be.matmul(&dense.view(&n(&c.names.router))?, &x, s, &mut router)?;
            // Per token: top-K experts by probability (ties: lower id), in rank order.
            let mut picks: Vec<(Vec<u32>, Vec<f32>)> = Vec::with_capacity(s);
            for t in 0..s {
                let logits_t = &router[t * ne..(t + 1) * ne];
                emit(TracePoint::RouterLogits, Some(li), t, logits_t, &[]);
                let mut p = logits_t.to_vec();
                softmax(&mut p);
                let mut idx: Vec<u32> = (0..u32::try_from(ne).unwrap_or(u32::MAX)).collect();
                idx.sort_by(|a, b| p[*b as usize].total_cmp(&p[*a as usize]).then(a.cmp(b)));
                idx.truncate(k_top);
                let mut w: Vec<f32> = idx.iter().map(|&e| p[e as usize]).collect();
                if c.norm_topk_prob {
                    let sum: f32 = w.iter().sum();
                    w.iter_mut().for_each(|x| *x /= sum);
                }
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
            for (e, users) in by_expert.iter().enumerate() {
                if users.is_empty() {
                    continue;
                }
                let n_t = users.len();
                let mut xe = vec![0f32; n_t * h];
                for (j, &(t, _)) in users.iter().enumerate() {
                    xe[j * h..(j + 1) * h].copy_from_slice(&x[t * h..(t + 1) * h]);
                }
                let ei = ExpertIdx(u32::try_from(e).unwrap_or(u32::MAX));
                experts.with_expert(LayerIdx(li), ei, &mut |w: &ExpertWeights<'_>| {
                    let ffn = w.gate.rows();
                    let mut g = vec![0f32; n_t * ffn];
                    let mut u = vec![0f32; n_t * ffn];
                    be.matmul(&w.gate, &xe, n_t, &mut g)?;
                    be.matmul(&w.up, &xe, n_t, &mut u)?;
                    swiglu(&mut g, &u);
                    let mut y = vec![0f32; n_t * h];
                    be.matmul(&w.down, &g, n_t, &mut y)?;
                    for (j, &(t, r)) in users.iter().enumerate() {
                        let dst = (t * k_top + r) * h;
                        contrib[dst..dst + h].copy_from_slice(&y[j * h..(j + 1) * h]);
                    }
                    Ok(())
                })?;
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
        cache.len += s;

        // Final norm and head, for the tokens whose logits are wanted.
        let first = match logits {
            Logits::Last => s - 1,
            Logits::All => 0,
        };
        let m = s - first;
        let mut xf = hs[first * h..].to_vec();
        for xt in xf.chunks_exact_mut(h) {
            rms_norm(xt, &self.final_norm, c.rms_eps);
        }
        let head = if c.tie_embeddings {
            emb
        } else {
            dense.view(&c.names.output)?
        };
        let mut out = vec![0f32; m * vocab];
        be.matmul(&head, &xf, m, &mut out)?;
        for t in 0..m {
            emit(
                TracePoint::Logits,
                None,
                first + t,
                &out[t * vocab..(t + 1) * vocab],
                &[],
            );
        }
        Ok(out)
    }
}
