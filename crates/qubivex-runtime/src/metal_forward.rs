//! The `transformer_moe` forward pass with whole layers on Metal (Phase 8 slice 3,
//! ADR-011 amendment 2).
//!
//! Each layer is one submission: the previous layer's routed experts (gate, up,
//! SwiGLU, down, the combine and the residual) followed by this layer's attention
//! block, MoE norm and router. Only the router logits come back to the host, where
//! routing runs (the same `ops::route` as the CPU pass) and the store leases the
//! chosen experts; their slabs are wrapped in place once per load (ADR-011 §4) and
//! cached in the slab's attachment. The last submission adds the final norm and the
//! head. Every kernel reproduces the CPU pass's bits (ADR-011 §1), so the logits are
//! bit-identical to [`Transformer::forward`]'s.
//!
//! Keys and values live in device buffers held in the [`KvCache`]'s device slot,
//! position-major and grown by whole chunks like the host cache.
//!
//! [`Transformer::forward`]: qubivex_arch::Transformer::forward

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use qubivex_arch::ops::{Rope, route};
use qubivex_arch::{ArchError, DenseWeights, KvCache, Logits, layer_name};
use qubivex_backend_cpu::CpuBackend;
use qubivex_backend_metal::{
    AttentionShape, Batch, DeviceVec, GROUPED_ENTRIES_PER_EXPERT, MAX_ROUTED_EXPERTS, MetalBackend,
    MetalHost, MetalWeights, RoutedInput, Routing,
};
use qubivex_compute::{ComputeBackend, ComputeError, WeightView};
use qubivex_core::{ExpertIdx, LayerIdx};
use qubivex_expert_stream::{ExpertKey, ExpertStore, Held, Lease};
use qubivex_model::arch::{MoeConfig, QkNorm};

use crate::ModelExpertSource;
use crate::engine::{expert_weights, store_err};

fn dev(e: ComputeError) -> ArchError {
    ArchError::Compute(e)
}

fn usize_of(v: u64) -> Result<usize, ArchError> {
    usize::try_from(v).map_err(|_| ArchError::Input(format!("dimension {v} does not fit")))
}

/// One decoder layer's resident weights on the device.
struct LayerOnMetal {
    attn_norm: DeviceVec,
    ffn_norm: DeviceVec,
    q_norm: Option<DeviceVec>,
    k_norm: Option<DeviceVec>,
    q: MetalWeights,
    k: MetalWeights,
    v: MetalWeights,
    o: MetalWeights,
    router: MetalWeights,
}

/// An expert's matrices on the device: views into its slab, wrapped in place (or
/// copies, for a part the kernels cannot read in place). Kept in the slab's
/// attachment, so it lives exactly as long as the slab holds this expert.
struct ExpertOnMetal {
    gate: MetalWeights,
    up: MetalWeights,
    down: MetalWeights,
}

/// Device keys and values of one cache: per layer, `capacity × kv_heads × head_dim`.
struct GpuKv {
    k: Vec<DeviceVec>,
    v: Vec<DeviceVec>,
    capacity: usize,
}

/// Activation buffers, sized for the largest call so far.
#[derive(Default)]
struct Work {
    tokens: usize,
    span: usize,
    bufs: Option<Bufs>,
}

struct Bufs {
    hs: DeviceVec,
    x: DeviceVec,
    q: DeviceVec,
    k: DeviceVec,
    v: DeviceVec,
    att: DeviceVec,
    o: DeviceVec,
    router: DeviceVec,
    table: DeviceVec,
    scores: DeviceVec,
    xe: DeviceVec,
    g: DeviceVec,
    u: DeviceVec,
    ye: DeviceVec,
    contrib: DeviceVec,
    weights: DeviceVec,
    xf: DeviceVec,
    logits: DeviceVec,
    /// Per layer, the expert ids `moe_route` chose (u32 bits), `s × top_k` each.
    routes: DeviceVec,
    /// The current layer's entries sorted by expert, and the group offsets.
    order: DeviceVec,
    offsets: DeviceVec,
}

/// Model dimensions.
#[derive(Clone, Copy)]
struct Dims {
    h: usize,
    d: usize,
    heads: usize,
    kvh: usize,
    q_out: usize,
    kv_out: usize,
    ne: usize,
    k_top: usize,
    ffn: usize,
    vocab: usize,
}

/// One routed expert's work in a layer: its token rows and where its outputs go.
struct ExpertJob<'s> {
    lease: Lease<'s>,
    /// Token index of each row.
    tokens: Vec<u32>,
    /// `t · k_top + rank` of each row: its slot in `contrib`.
    slots: Vec<u32>,
    /// First row in the expert scratch buffers.
    base: usize,
}

/// The whole-layer Metal forward pass.
pub(crate) struct MetalForward {
    be: MetalBackend,
    _host: MetalHost,
    cfg: MoeConfig,
    dims: Dims,
    rope: Rope,
    layers: Vec<LayerOnMetal>,
    final_norm: DeviceVec,
    head: MetalWeights,
    max_context: usize,
    /// Route on the GPU (one submission per token) when every expert is resident.
    token: bool,
    /// Forward passes run as one submission with routing on the GPU.
    routed_passes: AtomicU64,
    work: Mutex<Work>,
    tensors: usize,
    bytes: u64,
}

impl std::fmt::Debug for MetalForward {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalForward")
            .field("layers", &self.layers.len())
            .finish_non_exhaustive()
    }
}

/// A 1-D weight as f32 on the device.
fn vector(
    be: &MetalBackend,
    dense: &dyn DenseWeights,
    name: &str,
    len: usize,
) -> Result<DeviceVec, ArchError> {
    let v = dense.view(name)?;
    if v.rows() != 1 || v.cols() != len {
        return Err(ArchError::Input(format!(
            "{name} is {} x {}, expected 1 x {len}",
            v.rows(),
            v.cols()
        )));
    }
    let mut out = vec![0f32; len];
    CpuBackend::new().dequant_row(&v, 0, &mut out)?;
    be.vec_from(&out).map_err(dev)
}

/// `w` on the device: in place inside `host` when possible, else a copy.
fn resident(
    be: &MetalBackend,
    host: &MetalHost,
    w: &WeightView<'_>,
    count: &mut (usize, u64),
) -> Result<MetalWeights, ArchError> {
    count.0 += 1;
    count.1 += w.bytes().len() as u64;
    match host.weights(w).map_err(dev)? {
        Some(r) => Ok(r),
        None => be.upload(w).map_err(dev),
    }
}

impl MetalForward {
    /// Places every dense weight of the model on the device (the arena `host` wraps).
    pub(crate) fn new(
        be: MetalBackend,
        host: MetalHost,
        cfg: &MoeConfig,
        dense: &dyn DenseWeights,
        max_context: usize,
        token: bool,
    ) -> Result<Self, ArchError> {
        let c = cfg.clone();
        let dims = Dims {
            h: usize_of(c.hidden)?,
            d: usize_of(c.head_dim)?,
            heads: usize_of(c.heads)?,
            kvh: usize_of(c.kv_heads)?,
            q_out: usize_of(c.heads * c.head_dim)?,
            kv_out: usize_of(c.kv_heads * c.head_dim)?,
            ne: c.experts as usize,
            k_top: c.top_k as usize,
            ffn: usize_of(c.expert_ffn)?,
            vocab: usize_of(c.vocab)?,
        };
        if dims.kvh == 0 || dims.heads % dims.kvh != 0 {
            return Err(ArchError::Input(
                "heads must be a multiple of KV heads".into(),
            ));
        }
        let mut count = (0usize, 0u64);
        let mut layers = Vec::with_capacity(c.layers as usize);
        for l in 0..c.layers {
            let n = |p: &str| layer_name(p, l);
            let m = |p: &str, count: &mut (usize, u64)| -> Result<MetalWeights, ArchError> {
                resident(&be, &host, &dense.view(&n(p))?, count)
            };
            let (q_norm, k_norm) = match c.qk_norm {
                QkNorm::None => (None, None),
                QkNorm::Full => (
                    Some(vector(&be, dense, &n(&c.names.q_norm), dims.q_out)?),
                    Some(vector(&be, dense, &n(&c.names.k_norm), dims.kv_out)?),
                ),
                QkNorm::PerHead => (
                    Some(vector(&be, dense, &n(&c.names.q_norm), dims.d)?),
                    Some(vector(&be, dense, &n(&c.names.k_norm), dims.d)?),
                ),
            };
            layers.push(LayerOnMetal {
                attn_norm: vector(&be, dense, &n(&c.names.attn_norm), dims.h)?,
                ffn_norm: vector(&be, dense, &n(&c.names.ffn_norm), dims.h)?,
                q_norm,
                k_norm,
                q: m(&c.names.q, &mut count)?,
                k: m(&c.names.k, &mut count)?,
                v: m(&c.names.v, &mut count)?,
                o: m(&c.names.o, &mut count)?,
                router: m(&c.names.router, &mut count)?,
            });
        }
        let final_norm = vector(&be, dense, &c.names.final_norm, dims.h)?;
        let head_name = if c.tie_embeddings {
            &c.names.embedding
        } else {
            &c.names.output
        };
        let head = resident(&be, &host, &dense.view(head_name)?, &mut count)?;
        Ok(Self {
            rope: Rope::new(dims.d, c.rope_theta),
            be,
            _host: host,
            cfg: c,
            dims,
            layers,
            final_norm,
            head,
            max_context,
            token,
            routed_passes: AtomicU64::new(0),
            work: Mutex::new(Work::default()),
            tensors: count.0,
            bytes: count.1,
        })
    }

    /// The device name.
    pub(crate) fn device(&self) -> String {
        self.be.info().name
    }

    /// Forward passes so far that ran as one submission, routed on the GPU.
    pub(crate) fn routed_passes(&self) -> u64 {
        self.routed_passes.load(Ordering::Relaxed)
    }

    /// Dense weight matrices on the device, and their bytes.
    pub(crate) const fn placed(&self) -> (usize, u64) {
        (self.tensors, self.bytes)
    }

    /// Makes `cache`'s device keys and values hold at least `positions`, keeping what
    /// they hold. Returns the bytes allocated.
    fn ensure_kv(&self, cache: &mut KvCache, positions: usize) -> Result<u64, ArchError> {
        let want = KvCache::reserved_for(positions, cache.capacity());
        let filled = cache.len();
        let per_layer = |cap: usize| cap * self.dims.kv_out;
        let slot = cache.device_mut();
        let grow = match slot.as_ref().map(|b| b.downcast_ref::<GpuKv>()) {
            None => true,
            Some(None) => {
                return Err(ArchError::Input(
                    "the cache holds another backend's keys and values".into(),
                ));
            }
            Some(Some(kv)) => kv.capacity < want,
        };
        if grow {
            let n = per_layer(want);
            let mut k = Vec::with_capacity(self.layers.len());
            let mut v = Vec::with_capacity(self.layers.len());
            for _ in &self.layers {
                k.push(self.be.vec(n).map_err(dev)?);
                v.push(self.be.vec(n).map_err(dev)?);
            }
            if let Some(old) = slot.as_ref().and_then(|b| b.downcast_ref::<GpuKv>()) {
                let used = per_layer(filled);
                self.be
                    .submit(|b| {
                        for l in 0..self.layers.len() {
                            b.copy(&k[l].all(), &old.k[l].all(), used)?;
                            b.copy(&v[l].all(), &old.v[l].all(), used)?;
                        }
                        Ok(())
                    })
                    .map_err(dev)?;
            }
            *slot = Some(Box::new(GpuKv {
                k,
                v,
                capacity: want,
            }));
        }
        let cap = slot
            .as_ref()
            .and_then(|b| b.downcast_ref::<GpuKv>())
            .map_or(0, |kv| kv.capacity);
        Ok((2 * self.layers.len() * per_layer(cap) * 4) as u64)
    }

    /// Fills the empty `cache` with a copy of `src`'s first `positions` positions.
    /// Caches are position-major, so a prefix of each buffer is a prefix of the positions,
    /// whatever either cache's capacity.
    pub(crate) fn restore(
        &self,
        src: &KvCache,
        cache: &mut KvCache,
        positions: usize,
    ) -> Result<(), ArchError> {
        let bytes = self.ensure_kv(cache, positions)?;
        cache.set_device_bytes(bytes);
        fn gpu(c: &KvCache) -> Result<&GpuKv, ArchError> {
            c.device()
                .and_then(|b| b.downcast_ref::<GpuKv>())
                .ok_or_else(|| ArchError::Input("no device cache".into()))
        }
        let (from, to) = (gpu(src)?, gpu(cache)?);
        let used = positions * self.dims.kv_out;
        self.be
            .submit(|b| {
                for l in 0..self.layers.len() {
                    b.copy(&to.k[l].all(), &from.k[l].all(), used)?;
                    b.copy(&to.v[l].all(), &from.v[l].all(), used)?;
                }
                Ok(())
            })
            .map_err(dev)?;
        cache.advance(positions)
    }

    /// Makes the activation buffers hold `s` tokens attending over `span` positions.
    fn ensure_work(&self, work: &mut Work, s: usize, span: usize) -> Result<(), ArchError> {
        if work.bufs.is_some() && work.tokens >= s && work.span >= span {
            return Ok(());
        }
        let s = s.max(work.tokens);
        // Scores grow with the context; reserve for the whole context at decode sizes.
        let span = span
            .max(work.span)
            .max(if s == 1 { self.max_context } else { 0 });
        let Dims {
            h,
            d,
            heads,
            q_out,
            kv_out,
            ne,
            k_top,
            ffn,
            vocab,
            ..
        } = self.dims;
        let v = |n: usize| self.be.vec(n).map_err(dev);
        work.bufs = Some(Bufs {
            hs: v(s * h)?,
            x: v(s * h)?,
            q: v(s * q_out)?,
            k: v(s * kv_out)?,
            v: v(s * kv_out)?,
            att: v(s * q_out)?,
            o: v(s * h)?,
            router: v(s * ne)?,
            table: v(s * d)?,
            scores: v(s * heads * span)?,
            xe: v(s * k_top * h)?,
            g: v(s * k_top * ffn)?,
            u: v(s * k_top * ffn)?,
            ye: v(s * k_top * h)?,
            contrib: v(s * k_top * h)?,
            weights: v(s * k_top)?,
            xf: v(s * h)?,
            logits: v(s * vocab)?,
            routes: v(self.layers.len() * s * k_top)?,
            order: v(s * k_top)?,
            offsets: v(ne + 1)?,
        });
        work.tokens = s;
        work.span = span;
        Ok(())
    }

    /// The device matrices of a leased expert, wrapping its slab on first use.
    fn expert<'l>(
        &self,
        source: &ModelExpertSource,
        lease: &'l Lease<'_>,
    ) -> Result<&'l ExpertOnMetal, ArchError> {
        let (layer, expert) = lease.key();
        let slab = lease.slab();
        if slab.attachment().get().is_none() {
            let host = self.be.map_host(Arc::clone(slab.region())).map_err(dev)?;
            let w = expert_weights(source, layer, expert, slab.bytes())?;
            let mut count = (0, 0);
            let on = ExpertOnMetal {
                gate: resident(&self.be, &host, &w.gate, &mut count)?,
                up: resident(&self.be, &host, &w.up, &mut count)?,
                down: resident(&self.be, &host, &w.down, &mut count)?,
            };
            // Another thread may have attached first; either attachment is equivalent.
            let _ = slab.attachment().set(Arc::new(on));
        }
        slab.attachment()
            .get()
            .and_then(|a| a.downcast_ref::<ExpertOnMetal>())
            .ok_or_else(|| ArchError::Expert {
                layer: layer.0,
                expert: expert.0,
                detail: "the slab holds another backend's attachment".into(),
            })
    }

    /// Records routed experts: per expert, its rows of `x` through gate, up, SwiGLU and
    /// down into its `contrib` slots.
    fn record_experts(
        &self,
        b: &mut qubivex_backend_metal::Batch<'_, '_>,
        w: &Bufs,
        experts: &[(&ExpertOnMetal, &ExpertJob<'_>)],
    ) -> Result<(), ComputeError> {
        let (h, ffn) = (self.dims.h, self.dims.ffn);
        // Experts own disjoint rows of the scratch buffers and distinct contrib slots,
        // so each stage runs for every expert at once.
        let spans = experts
            .iter()
            .map(|(e, j)| {
                let n = j.tokens.len();
                Ok((
                    *e,
                    *j,
                    n,
                    w.xe.span(j.base * h, n * h)?,
                    w.g.span(j.base * ffn, n * ffn)?,
                    w.u.span(j.base * ffn, n * ffn)?,
                    w.ye.span(j.base * h, n * h)?,
                ))
            })
            .collect::<Result<Vec<_>, ComputeError>>()?;
        b.concurrent(|b| {
            for (_, j, _, xe, ..) in &spans {
                b.gather_rows(xe, &w.x.all(), &j.tokens, h)?;
            }
            Ok(())
        })?;
        b.concurrent(|b| {
            for (e, _, n, xe, g, u, _) in &spans {
                b.matmul(&e.gate, xe, *n, g)?;
                b.matmul(&e.up, xe, *n, u)?;
            }
            Ok(())
        })?;
        b.concurrent(|b| {
            for (_, _, n, _, g, u, _) in &spans {
                b.swiglu(g, u, n * ffn)?;
            }
            Ok(())
        })?;
        b.concurrent(|b| {
            for (e, _, n, _, g, _, ye) in &spans {
                b.matmul(&e.down, g, *n, ye)?;
            }
            Ok(())
        })?;
        b.concurrent(|b| {
            for (_, j, _, _, _, _, ye) in &spans {
                b.scatter_rows(&w.contrib.all(), ye, &j.slots, h)?;
            }
            Ok(())
        })
    }

    /// Records layer `l`'s attention block (into the residual stream `hs`), then its
    /// MoE norm (into `x`) and router (into `router`), for `attn.tokens` tokens.
    fn record_attention(
        &self,
        b: &mut Batch<'_, '_>,
        w: &Bufs,
        kv: &GpuKv,
        l: usize,
        ly: &LayerOnMetal,
        attn: AttentionShape,
    ) -> Result<(), ComputeError> {
        let c = &self.cfg;
        let dm = self.dims;
        let (s, h, d) = (attn.tokens, dm.h, dm.d);
        #[allow(clippy::cast_possible_truncation)]
        let clip = c.clip_qkv.map(|v| v as f32);
        let x = w.x.span(0, s * h)?;
        let (q, k, v) = (
            w.q.span(0, s * dm.q_out)?,
            w.k.span(0, s * dm.kv_out)?,
            w.v.span(0, s * dm.kv_out)?,
        );
        b.copy(&x, &w.hs.all(), s * h)?;
        b.rms_norm(&x, s, &ly.attn_norm.all(), c.rms_eps)?;
        // q, k and v are independent until attention: each step runs as one group.
        b.concurrent(|b| {
            b.matmul(&ly.q, &x, s, &q)?;
            b.matmul(&ly.k, &x, s, &k)?;
            b.matmul(&ly.v, &x, s, &v)
        })?;
        if let Some(cl) = clip {
            b.concurrent(|b| {
                b.clamp(&q, q.len(), cl)?;
                b.clamp(&k, k.len(), cl)?;
                b.clamp(&v, v.len(), cl)
            })?;
        }
        match (c.qk_norm, &ly.q_norm, &ly.k_norm) {
            (QkNorm::Full, Some(qn), Some(kn)) => b.concurrent(|b| {
                b.rms_norm(&q, s, &qn.all(), c.rms_eps)?;
                b.rms_norm(&k, s, &kn.all(), c.rms_eps)
            })?,
            (QkNorm::PerHead, Some(qn), Some(kn)) => b.concurrent(|b| {
                b.rms_norm(&q, s * dm.heads, &qn.all(), c.rms_eps)?;
                b.rms_norm(&k, s * dm.kvh, &kn.all(), c.rms_eps)
            })?,
            _ => {}
        }
        let tb = w.table.span(0, s * d)?;
        b.concurrent(|b| {
            b.rope(&q, s, dm.heads, d / 2, &tb)?;
            b.rope(&k, s, dm.kvh, d / 2, &tb)
        })?;
        b.concurrent(|b| {
            b.kv_store(&kv.k[l].all(), &k, attn)?;
            b.kv_store(&kv.v[l].all(), &v, attn)
        })?;
        let att = w.att.span(0, s * dm.q_out)?;
        b.attention(
            &q,
            &kv.k[l].all(),
            &kv.v[l].all(),
            &w.scores.all(),
            &att,
            attn,
        )?;
        let o = w.o.span(0, s * h)?;
        b.matmul(&ly.o, &att, s, &o)?;
        b.add(&w.hs.all(), &o, s * h)?;
        // MoE norm and router.
        b.copy(&x, &w.hs.all(), s * h)?;
        b.rms_norm(&x, s, &ly.ffn_norm.all(), c.rms_eps)?;
        b.matmul(&ly.router, &x, s, &w.router.span(0, s * dm.ne)?)
    }

    /// Records the final norm and the head on tokens `first..first + m` (into
    /// `logits`).
    fn record_head(
        &self,
        b: &mut Batch<'_, '_>,
        w: &Bufs,
        first: usize,
        m: usize,
    ) -> Result<(), ComputeError> {
        let h = self.dims.h;
        let xf = w.xf.span(0, m * h)?;
        b.copy(&xf, &w.hs.span(first * h, m * h)?, m * h)?;
        b.rms_norm(&xf, m, &self.final_norm.all(), self.cfg.rms_eps)?;
        b.matmul(&self.head, &xf, m, &w.logits.span(0, m * self.dims.vocab)?)
    }

    /// Holds every expert of every layer for a token routed on the GPU, if all are
    /// resident and fit the routed kernels. With I/O threads, missing experts are
    /// queued as prefetches (this token takes the per-layer pass while they load);
    /// without, they are loaded now.
    fn hold_all<'s>(
        &self,
        store: &'s ExpertStore,
    ) -> Result<Option<Vec<Vec<Lease<'s>>>>, ArchError> {
        let dm = self.dims;
        if !self.token || dm.ne > MAX_ROUTED_EXPERTS || dm.k_top == 0 || dm.k_top > dm.ne {
            return Ok(None);
        }
        let ne = u32::try_from(dm.ne).unwrap_or(u32::MAX);
        for attempt in 0..2 {
            let mut held = Vec::with_capacity(self.layers.len());
            let mut missing: Vec<Vec<ExpertKey>> = Vec::new();
            for l in 0..self.layers.len() {
                let li = LayerIdx(u32::try_from(l).unwrap_or(u32::MAX));
                match store
                    .hold_layer(li, ne)
                    .map_err(|e| store_err(li, ExpertIdx(0), &e))?
                {
                    Held::All(v) => held.push(v),
                    Held::TooSmall => return Ok(None),
                    Held::Missing(m) => missing.push(m),
                }
            }
            if missing.is_empty() {
                return Ok(Some(held));
            }
            drop(held);
            if attempt == 1 {
                break;
            }
            if store.io_threads() > 0 {
                let all: Vec<ExpertKey> = missing.concat();
                store
                    .prefetch(&all, false)
                    .map_err(|e| store_err(LayerIdx(0), ExpertIdx(0), &e))?;
                return Ok(None);
            }
            // Nobody reads ahead: load each layer's missing experts now (they count as
            // uses, as any acquire does), then hold again.
            for keys in missing {
                let li = keys.first().map_or(LayerIdx(0), |k| k.0);
                for lease in store
                    .acquire_many(&keys)
                    .map_err(|e| store_err(li, ExpertIdx(0), &e))?
                {
                    drop(lease.map_err(|e| store_err(li, ExpertIdx(0), &e))?);
                }
            }
        }
        Ok(None)
    }

    /// The whole forward pass in one submission, with routing on the GPU (ADR-011
    /// amendment 4): per layer, attention, router, `moe_route`, the routed gate, up,
    /// SwiGLU and down over every expert of the layer (`held`, in expert order), the
    /// combine and the residual; then the final norm and head. Afterwards the store is
    /// told which experts were used. `None` (nothing ran) if some layer's experts differ
    /// in dtype or shape, which the routed kernels need to share.
    #[allow(clippy::too_many_arguments)]
    fn routed(
        &self,
        w: &Bufs,
        kv: &GpuKv,
        held: &[Vec<Lease<'_>>],
        source: &ModelExpertSource,
        store: &ExpertStore,
        attn: AttentionShape,
        first: usize,
        m: usize,
    ) -> Result<Option<Vec<f32>>, ArchError> {
        type Table<'t> = (
            Vec<&'t MetalWeights>,
            Vec<&'t MetalWeights>,
            Vec<&'t MetalWeights>,
        );
        let dm = self.dims;
        let (s, h, ffn, k) = (attn.tokens, dm.h, dm.ffn, dm.k_top);
        let shape = |w: &MetalWeights| (w.dtype(), w.rows(), w.cols(), w.row_bytes());
        let mut tables: Vec<Table<'_>> = Vec::with_capacity(held.len());
        for leases in held {
            let on = leases
                .iter()
                .map(|l| self.expert(source, l))
                .collect::<Result<Vec<_>, _>>()?;
            let Some(e0) = on.first() else {
                return Ok(None);
            };
            if !on.iter().all(|e| {
                shape(&e.gate) == shape(&e0.gate)
                    && shape(&e.up) == shape(&e0.up)
                    && shape(&e.down) == shape(&e0.down)
            }) {
                return Ok(None);
            }
            tables.push((
                on.iter().map(|e| &e.gate).collect(),
                on.iter().map(|e| &e.up).collect(),
                on.iter().map(|e| &e.down).collect(),
            ));
        }
        let entries = s * k;
        self.be
            .submit(|b| {
                let x = w.x.span(0, s * h)?;
                let g = w.g.span(0, entries * ffn)?;
                let u = w.u.span(0, entries * ffn)?;
                let contrib = w.contrib.span(0, entries * h)?;
                let wts = w.weights.span(0, entries)?;
                let o = w.o.span(0, s * h)?;
                let router = w.router.span(0, s * dm.ne)?;
                let order = w.order.span(0, entries)?;
                let offsets = w.offsets.all();
                for (l, ly) in self.layers.iter().enumerate() {
                    self.record_attention(b, w, kv, l, ly, attn)?;
                    let ids = w.routes.span(l * entries, entries)?;
                    b.moe_route(&router, s, dm.ne, k, self.cfg.norm_topk_prob, &ids, &wts)?;
                    let (gate, up, down) = &tables[l];
                    // Enough entries per expert: sort them by expert, so each weight
                    // load serves up to 8 of them (ADR-011 amendment 5).
                    let routing = if entries >= GROUPED_ENTRIES_PER_EXPERT * dm.ne {
                        b.moe_group(&ids, entries, dm.ne, &order, &offsets)?;
                        Routing::Grouped {
                            order: &order,
                            offsets: &offsets,
                        }
                    } else {
                        Routing::Ids(&ids)
                    };
                    b.concurrent(|b| {
                        b.routed_matmul(gate, routing, &x, RoutedInput::Token, s, k, &g)?;
                        b.routed_matmul(up, routing, &x, RoutedInput::Token, s, k, &u)
                    })?;
                    b.swiglu(&g, &u, entries * ffn)?;
                    b.routed_matmul(down, routing, &g, RoutedInput::Entry, s, k, &contrib)?;
                    b.moe_combine(&contrib, &wts, &o, s, k, h)?;
                    b.add(&w.hs.all(), &o, s * h)?;
                }
                self.record_head(b, w, first, m)
            })
            .map_err(dev)?;
        // Tell the store what was used, as the per-layer pass's acquires would: per
        // layer, its distinct experts in ascending order.
        let mut ids = vec![0f32; self.layers.len() * entries];
        self.be
            .read(&w.routes.span(0, ids.len()).map_err(dev)?, &mut ids)
            .map_err(dev)?;
        let mut used = Vec::new();
        for (l, chunk) in ids.chunks_exact(entries.max(1)).enumerate() {
            let li = LayerIdx(u32::try_from(l).unwrap_or(u32::MAX));
            let mut es: Vec<u32> = chunk.iter().map(|v| v.to_bits()).collect();
            es.sort_unstable();
            es.dedup();
            used.extend(es.into_iter().map(|e| (li, ExpertIdx(e))));
        }
        store
            .record_uses(&used)
            .map_err(|e| store_err(LayerIdx(0), ExpertIdx(0), &e))?;
        let mut out = vec![0f32; m * dm.vocab];
        self.be
            .read(&w.logits.span(0, m * dm.vocab).map_err(dev)?, &mut out)
            .map_err(dev)?;
        Ok(Some(out))
    }

    /// The device matrices of every job's expert.
    fn on_device<'j>(
        &self,
        source: &ModelExpertSource,
        jobs: &'j [ExpertJob<'_>],
    ) -> Result<Vec<(&'j ExpertOnMetal, &'j ExpertJob<'j>)>, ArchError> {
        jobs.iter()
            .map(|j| Ok((self.expert(source, &j.lease)?, j)))
            .collect()
    }

    /// Runs `tokens` at positions `cache.len()..` and returns logits as
    /// [`Transformer::forward`] does, with the same bits.
    ///
    /// [`Transformer::forward`]: qubivex_arch::Transformer::forward
    #[allow(clippy::too_many_lines)]
    pub(crate) fn forward(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        dense: &dyn DenseWeights,
        store: &ExpertStore,
        source: &ModelExpertSource,
        logits: Logits,
    ) -> Result<Vec<f32>, ArchError> {
        let s = tokens.len();
        if s == 0 {
            return Ok(Vec::new());
        }
        let p0 = cache.len();
        if p0 + s > cache.capacity() {
            return Err(ArchError::Input(format!(
                "context full: {p0} positions + {s} new > {}",
                cache.capacity()
            )));
        }
        let c = &self.cfg;
        let dm = self.dims;
        let (h, d) = (dm.h, dm.d);
        let kv_bytes = self.ensure_kv(cache, p0 + s)?;
        cache.set_device_bytes(kv_bytes);
        let mut work = self
            .work
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.ensure_work(&mut work, s, p0 + s)?;
        let w = work
            .bufs
            .as_ref()
            .ok_or_else(|| ArchError::Input("no buffers".into()))?;
        let kv = cache
            .device_mut()
            .as_ref()
            .and_then(|b| b.downcast_ref::<GpuKv>())
            .ok_or_else(|| ArchError::Input("no device cache".into()))?;

        // Host: embeddings and the RoPE table.
        let emb = dense.view(&c.names.embedding)?;
        let cpu = CpuBackend::new();
        let mut host = vec![0f32; s * h];
        for (t, &id) in tokens.iter().enumerate() {
            if id as usize >= emb.rows() {
                return Err(ArchError::Input(format!(
                    "token id {id} >= vocabulary {}",
                    emb.rows()
                )));
            }
            cpu.dequant_row(&emb, id as usize, &mut host[t * h..(t + 1) * h])?;
        }
        self.be
            .write(&w.hs.span(0, s * h).map_err(dev)?, &host)
            .map_err(dev)?;
        let mut table = vec![0f32; s * d];
        for t in 0..s {
            self.rope.table(p0 + t, &mut table[t * d..(t + 1) * d]);
        }
        self.be
            .write(&w.table.span(0, s * d).map_err(dev)?, &table)
            .map_err(dev)?;

        let attn = AttentionShape {
            tokens: s,
            heads: dm.heads,
            kv_heads: dm.kvh,
            head_dim: d,
            p0,
            capacity: kv.capacity,
        };
        let first = match logits {
            Logits::Last => s - 1,
            Logits::All => 0,
        };
        let m = s - first;

        // Every expert resident: the whole token is one submission, routed on the GPU.
        if let Some(held) = self.hold_all(store)? {
            if let Some(out) = self.routed(w, kv, &held, source, store, attn, first, m)? {
                self.routed_passes.fetch_add(1, Ordering::Relaxed);
                drop(held);
                drop(work);
                cache.advance(s)?;
                return Ok(out);
            }
        }

        // The previous layer's last wave of experts, recorded at the start of the next
        // submission together with the combine. `Some` (even if empty) once routed.
        let mut pending: Option<Vec<ExpertJob<'_>>> = None;
        let pools = store
            .pools()
            .map_err(|e| store_err(LayerIdx(0), ExpertIdx(0), &e))?;
        let mut router = vec![0f32; s * dm.ne];
        for l in 0..=self.layers.len() {
            let experts = match &pending {
                Some(jobs) => self.on_device(source, jobs)?,
                None => Vec::new(),
            };
            let layer = self.layers.get(l);
            self.be
                .submit(|b| {
                    // MoE of the previous layer: per expert, its rows of x through
                    // gate, up, SwiGLU and down into its contrib slots; then the
                    // combine in rank order and the residual.
                    if pending.is_some() {
                        self.record_experts(b, w, &experts)?;
                        let o = w.o.span(0, s * h)?;
                        b.moe_combine(&w.contrib.all(), &w.weights.all(), &o, s, dm.k_top, h)?;
                        b.add(&w.hs.all(), &o, s * h)?;
                    }
                    match layer {
                        Some(ly) => self.record_attention(b, w, kv, l, ly, attn),
                        None => self.record_head(b, w, first, m),
                    }
                })
                .map_err(dev)?;
            drop(experts);
            // The submission has completed: the previous layer's leases may go.
            pending = None;
            if l == self.layers.len() {
                break;
            }

            // Routing on the host, as the CPU pass does it.
            self.be
                .read(&w.router.span(0, s * dm.ne).map_err(dev)?, &mut router)
                .map_err(dev)?;
            let mut weights = vec![0f32; s * dm.k_top];
            let mut by_expert: Vec<(Vec<u32>, Vec<u32>)> = vec![(Vec::new(), Vec::new()); dm.ne];
            for t in 0..s {
                let (idx, wt) = route(
                    &router[t * dm.ne..(t + 1) * dm.ne],
                    dm.k_top,
                    c.norm_topk_prob,
                );
                for (r, (&e, &p)) in idx.iter().zip(&wt).enumerate() {
                    weights[t * dm.k_top + r] = p;
                    let slot = u32::try_from(t * dm.k_top + r).unwrap_or(u32::MAX);
                    let tok = u32::try_from(t).unwrap_or(u32::MAX);
                    by_expert[e as usize].0.push(tok);
                    by_expert[e as usize].1.push(slot);
                }
            }
            self.be
                .write(&w.weights.span(0, s * dm.k_top).map_err(dev)?, &weights)
                .map_err(dev)?;
            let li = u32::try_from(l).unwrap_or(u32::MAX);
            let used: Vec<(LayerIdx, ExpertIdx)> = by_expert
                .iter()
                .enumerate()
                .filter(|(_, u)| !u.0.is_empty())
                .map(|(e, _)| {
                    (
                        LayerIdx(li),
                        ExpertIdx(u32::try_from(e).unwrap_or(u32::MAX)),
                    )
                })
                .collect();
            let batch = store
                .acquire_many(&used)
                .map_err(|e| store_err(LayerIdx(li), ExpertIdx(0), &e))?;
            // The store hands out more leases only as earlier ones are released, so a
            // layer routed to more experts than its pool holds runs in waves of at most
            // `slots` experts; the last wave joins the next submission.
            let slots = pools
                .iter()
                .find(|p| p.layer == LayerIdx(li))
                .map_or(usize::MAX, |p| (p.slots as usize).max(1));
            let mut wave = Vec::with_capacity(used.len().min(slots));
            let mut base = 0;
            for lease in batch {
                let lease = lease.map_err(|e| store_err(LayerIdx(li), ExpertIdx(0), &e))?;
                let (tokens, slots_of) = by_expert[lease.key().1.0 as usize].clone();
                let n = tokens.len();
                wave.push(ExpertJob {
                    lease,
                    tokens,
                    slots: slots_of,
                    base,
                });
                base += n;
                if wave.len() == slots {
                    let on = self.on_device(source, &wave)?;
                    self.be
                        .submit(|b| self.record_experts(b, w, &on))
                        .map_err(dev)?;
                    drop(on);
                    wave.clear();
                }
            }
            pending = Some(wave);
        }
        drop(pending);
        let mut out = vec![0f32; m * dm.vocab];
        self.be
            .read(&w.logits.span(0, m * dm.vocab).map_err(dev)?, &mut out)
            .map_err(dev)?;
        drop(work);
        cache.advance(s)?;
        Ok(out)
    }
}
