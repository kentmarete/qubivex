//! Recorded submissions over device-resident vectors (ADR-011 §2, amendment 2).
//!
//! A [`Batch`] records many operations (matmuls on resident weights, norms, RoPE,
//! attention, SwiGLU, the MoE combine) into one command buffer, reading and writing
//! [`DeviceVec`]s, so a layer or a whole token costs one submission. Every operation
//! reproduces the bits of its CPU counterpart in `qubivex-arch`'s `ops` (ADR-011 §1).
//!
//! Each method checks that what the kernel will touch lies inside its vectors before
//! recording; nothing runs until [`MetalBackend::submit`] commits the batch.

use qubivex_compute::ComputeError;

use crate::{MetalBackend, MetalWeights, check_dtype};

#[cfg(target_os = "macos")]
use crate::device::{Arg, Buffer, Recorder, bytes_of};

/// The forward-pass kernels, after the six matmul kernels in the pipeline list.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) const OP_KERNELS: [&str; 12] = [
    "exp_in_place",
    "rms_norm",
    "rope",
    "clamp_abs",
    "add_in_place",
    "copy_f32",
    "swiglu",
    "kv_store",
    "attn_scores",
    "attn_softmax",
    "attn_values",
    "moe_combine",
];

#[cfg(target_os = "macos")]
fn op(name: &str) -> usize {
    crate::DTYPES.len()
        + OP_KERNELS
            .iter()
            .position(|k| *k == name)
            .unwrap_or_else(|| unreachable!("unknown kernel {name}"))
}

/// An f32 vector in device memory (shared storage: the host reads and writes it
/// between submissions through [`MetalBackend::write`] and [`MetalBackend::read`]).
#[derive(Debug)]
pub struct DeviceVec {
    #[cfg(target_os = "macos")]
    buf: Buffer,
    len: usize,
}

impl DeviceVec {
    /// Elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether it holds no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The whole vector.
    #[must_use]
    pub const fn all(&self) -> Span<'_> {
        Span {
            v: self,
            at: 0,
            len: self.len,
        }
    }

    /// Elements `at..at + len`.
    ///
    /// # Errors
    /// [`ComputeError::Shape`] if the range is outside the vector.
    pub fn span(&self, at: usize, len: usize) -> Result<Span<'_>, ComputeError> {
        if at.checked_add(len).is_none_or(|e| e > self.len) {
            return Err(shape(
                "span",
                format!("{at}..+{len} of a {}-element vector", self.len),
            ));
        }
        Ok(Span { v: self, at, len })
    }
}

/// A range of a [`DeviceVec`].
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub struct Span<'a> {
    v: &'a DeviceVec,
    at: usize,
    len: usize,
}

impl Span<'_> {
    /// Elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether it holds no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[cfg(target_os = "macos")]
    const fn arg(&self) -> Arg<'_> {
        Arg::Buf(&self.v.buf, self.at * 4)
    }
}

fn shape(op: &'static str, detail: String) -> ComputeError {
    ComputeError::Shape { op, detail }
}

fn need(op: &'static str, what: &str, span: &Span<'_>, n: usize) -> Result<(), ComputeError> {
    if span.len < n {
        return Err(shape(
            op,
            format!("{what} holds {} elements, needs {n}", span.len),
        ));
    }
    Ok(())
}

fn u32_of(v: usize, op: &'static str) -> Result<u32, ComputeError> {
    u32::try_from(v).map_err(|_| shape(op, format!("{v} exceeds 32 bits")))
}

/// Threads per threadgroup for the elementwise kernels.
#[cfg(target_os = "macos")]
const TG: usize = 256;

#[repr(C)]
#[derive(Clone, Copy)]
struct NormParams {
    n: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RopeParams {
    half: u32,
    heads_per_row: u32,
    rows: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EwParams {
    n: u32,
    f: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct KvParams {
    s: u32,
    kv_heads: u32,
    d: u32,
    p0: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttnParams {
    s: u32,
    heads: u32,
    group: u32,
    d: u32,
    p0: u32,
    span: u32,
    stride: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CombineParams {
    s: u32,
    k: u32,
    h: u32,
}

/// The shape of one attention call (see [`Batch::attention`]).
#[derive(Clone, Copy, Debug)]
pub struct AttentionShape {
    /// Tokens (queries).
    pub tokens: usize,
    /// Query heads.
    pub heads: usize,
    /// KV heads (`heads` must be a multiple).
    pub kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// The first token's position; token `t` attends to positions `0..=p0 + t`.
    pub p0: usize,
    /// Positions the cache holds per KV head (its capacity).
    pub capacity: usize,
}

/// Operations recorded into one command buffer.
pub struct Batch<'r, 'e> {
    #[cfg(target_os = "macos")]
    rec: &'r mut Recorder<'e>,
    #[cfg(not(target_os = "macos"))]
    _never: core::marker::PhantomData<(&'r (), &'e ())>,
}

impl std::fmt::Debug for Batch<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Batch").finish_non_exhaustive()
    }
}

impl MetalBackend {
    /// A zeroed device vector of `len` f32.
    ///
    /// # Errors
    /// [`ComputeError::Device`] if the buffer cannot be created.
    pub fn vec(&self, len: usize) -> Result<DeviceVec, ComputeError> {
        #[cfg(target_os = "macos")]
        {
            let buf = self.dev.buffer_zeroed(len.max(1) * 4)?;
            Ok(DeviceVec { buf, len })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = len;
            unreachable!("a MetalBackend cannot be created off macOS")
        }
    }

    /// A device vector holding `data`.
    ///
    /// # Errors
    /// As [`MetalBackend::vec`].
    pub fn vec_from(&self, data: &[f32]) -> Result<DeviceVec, ComputeError> {
        let v = self.vec(data.len())?;
        self.write(&v.all(), data)?;
        Ok(v)
    }

    /// Copies `data` into `dst` (lengths must match). Waits for no GPU work: it takes
    /// the lock submissions hold, so it never overlaps one.
    ///
    /// # Errors
    /// [`ComputeError::Shape`] if the lengths differ.
    pub fn write(&self, dst: &Span<'_>, data: &[f32]) -> Result<(), ComputeError> {
        if dst.len != data.len() {
            return Err(shape(
                "write",
                format!("{} values into {}", data.len(), dst.len),
            ));
        }
        #[cfg(target_os = "macos")]
        {
            self.dev.write_f32(&dst.v.buf, dst.at, data);
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            unreachable!("a MetalBackend cannot be created off macOS")
        }
    }

    /// Copies `src` into `out` (lengths must match).
    ///
    /// # Errors
    /// [`ComputeError::Shape`] if the lengths differ.
    pub fn read(&self, src: &Span<'_>, out: &mut [f32]) -> Result<(), ComputeError> {
        if src.len != out.len() {
            return Err(shape(
                "read",
                format!("{} values into {}", src.len, out.len()),
            ));
        }
        #[cfg(target_os = "macos")]
        {
            self.dev.read_f32(&src.v.buf, src.at, out);
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            unreachable!("a MetalBackend cannot be created off macOS")
        }
    }

    /// Records the operations `f` adds into one command buffer, commits it and waits.
    /// If `f` fails, nothing runs.
    ///
    /// # Errors
    /// `f`'s error, or [`ComputeError::Device`] if the GPU fails.
    pub fn submit(
        &self,
        f: impl FnOnce(&mut Batch<'_, '_>) -> Result<(), ComputeError>,
    ) -> Result<(), ComputeError> {
        #[cfg(target_os = "macos")]
        {
            self.dev.submit(|rec| f(&mut Batch { rec }))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = f;
            unreachable!("a MetalBackend cannot be created off macOS")
        }
    }
}

#[cfg(target_os = "macos")]
fn groups(n: usize) -> [usize; 3] {
    [n.div_ceil(TG).max(1), 1, 1]
}

impl Batch<'_, '_> {
    /// `ys[t] = W · xs[t]` for `s` rows (the bits of [`MetalBackend::matmul_resident`]).
    ///
    /// # Errors
    /// [`ComputeError::Shape`] if a span is too short or the weights overrun their
    /// buffer.
    pub fn matmul(
        &mut self,
        w: &MetalWeights,
        xs: &Span<'_>,
        s: usize,
        ys: &Span<'_>,
    ) -> Result<(), ComputeError> {
        let pipeline = check_dtype(w.dtype, "matmul")?;
        need("matmul", "input", xs, s * w.cols)?;
        need("matmul", "output", ys, s * w.rows)?;
        if s == 0 || w.rows == 0 {
            return Ok(());
        }
        #[cfg(target_os = "macos")]
        {
            let end = w
                .rows
                .checked_mul(w.row_bytes)
                .and_then(|b| b.checked_add(w.offset));
            if end.is_none_or(|e| e > w.buf.len()) {
                return Err(shape("matmul", "weights overrun their buffer".into()));
            }
            if !crate::offset_ok(w.dtype, w.offset) {
                return Err(shape(
                    "matmul",
                    format!("{} weights at unaligned offset {}", w.dtype, w.offset),
                ));
            }
            let params = crate::device::Params::new(w.rows, w.cols, s, w.row_bytes)?;
            let pb = bytes_of(&params);
            self.rec.dispatch(
                pipeline,
                &[
                    Arg::Buf(&w.buf, w.offset),
                    xs.arg(),
                    ys.arg(),
                    Arg::Bytes(&pb),
                ],
                [
                    w.rows.div_ceil(crate::device::rows_per_group(pipeline)),
                    s,
                    1,
                ],
                [crate::device::ROWS * crate::device::SIMD_WIDTH, 1, 1],
            );
        }
        #[cfg(not(target_os = "macos"))]
        let _ = pipeline;
        Ok(())
    }

    /// `ops::rms_norm` on each of `rows` rows of `x` (row length `w.len()`), in place.
    ///
    /// # Errors
    /// [`ComputeError::Shape`] if `x` is too short.
    pub fn rms_norm(
        &mut self,
        x: &Span<'_>,
        rows: usize,
        w: &Span<'_>,
        eps: f64,
    ) -> Result<(), ComputeError> {
        need("rms_norm", "input", x, rows * w.len)?;
        if rows == 0 || w.len == 0 {
            return Ok(());
        }
        #[allow(clippy::cast_possible_truncation)]
        let p = NormParams {
            n: u32_of(w.len, "rms_norm")?,
            eps: eps as f32,
        };
        #[cfg(target_os = "macos")]
        {
            let pb = bytes_of(&p);
            self.rec.dispatch(
                op("rms_norm"),
                &[x.arg(), w.arg(), Arg::Bytes(&pb)],
                [rows, 1, 1],
                [crate::device::SIMD_WIDTH, 1, 1],
            );
        }
        #[cfg(not(target_os = "macos"))]
        let _ = p;
        Ok(())
    }

    /// `Rope::rotate` on `rows` tokens of `heads` heads each (`head_dim = 2 · half`);
    /// token `t` uses `table[t · 2 · half ..]` (one `Rope::table` row per token).
    ///
    /// # Errors
    /// [`ComputeError::Shape`] if a span is too short.
    pub fn rope(
        &mut self,
        x: &Span<'_>,
        rows: usize,
        heads: usize,
        half: usize,
        table: &Span<'_>,
    ) -> Result<(), ComputeError> {
        need("rope", "input", x, rows * heads * 2 * half)?;
        need("rope", "table", table, rows * 2 * half)?;
        let n = rows * heads * half;
        if n == 0 {
            return Ok(());
        }
        let p = RopeParams {
            half: u32_of(half, "rope")?,
            heads_per_row: u32_of(heads, "rope")?,
            rows: u32_of(rows, "rope")?,
        };
        u32_of(n, "rope")?;
        #[cfg(target_os = "macos")]
        {
            let pb = bytes_of(&p);
            self.rec.dispatch(
                op("rope"),
                &[x.arg(), table.arg(), Arg::Bytes(&pb)],
                groups(n),
                [TG, 1, 1],
            );
        }
        #[cfg(not(target_os = "macos"))]
        let _ = p;
        Ok(())
    }

    fn elementwise(
        &mut self,
        name: &'static str,
        spans: &[&Span<'_>],
        n: usize,
        f: f32,
    ) -> Result<(), ComputeError> {
        for s in spans {
            need(name, "operand", s, n)?;
        }
        if n == 0 {
            return Ok(());
        }
        let p = EwParams {
            n: u32_of(n, name)?,
            f,
        };
        #[cfg(target_os = "macos")]
        {
            let pb = bytes_of(&p);
            let mut args: Vec<Arg<'_>> = spans.iter().map(|s| s.arg()).collect();
            args.push(Arg::Bytes(&pb));
            self.rec.dispatch(op(name), &args, groups(n), [TG, 1, 1]);
        }
        #[cfg(not(target_os = "macos"))]
        let _ = p;
        Ok(())
    }

    /// `x = x.clamp(-c, c)` on the first `n` elements.
    ///
    /// # Errors
    /// [`ComputeError::Shape`] if `x` is too short.
    pub fn clamp(&mut self, x: &Span<'_>, n: usize, c: f32) -> Result<(), ComputeError> {
        self.elementwise("clamp_abs", &[x], n, c)
    }

    /// `x = math::exp(x)` (the portable exp of `qubivex_compute::math`) on the first `n`
    /// elements.
    ///
    /// # Errors
    /// [`ComputeError::Shape`] if `x` is too short.
    pub fn exp(&mut self, x: &Span<'_>, n: usize) -> Result<(), ComputeError> {
        self.elementwise("exp_in_place", &[x], n, 0.0)
    }

    /// `a += b` on the first `n` elements.
    ///
    /// # Errors
    /// [`ComputeError::Shape`] if a span is too short.
    pub fn add(&mut self, a: &Span<'_>, b: &Span<'_>, n: usize) -> Result<(), ComputeError> {
        self.elementwise("add_in_place", &[a, b], n, 0.0)
    }

    /// `dst = src` on the first `n` elements.
    ///
    /// # Errors
    /// [`ComputeError::Shape`] if a span is too short.
    pub fn copy(&mut self, dst: &Span<'_>, src: &Span<'_>, n: usize) -> Result<(), ComputeError> {
        self.elementwise("copy_f32", &[dst, src], n, 0.0)
    }

    /// `ops::swiglu`: `g = silu(g) · u` on the first `n` elements.
    ///
    /// # Errors
    /// [`ComputeError::Shape`] if a span is too short.
    pub fn swiglu(&mut self, g: &Span<'_>, u: &Span<'_>, n: usize) -> Result<(), ComputeError> {
        self.elementwise("swiglu", &[g, u], n, 0.0)
    }

    /// Appends `a.tokens` tokens of keys or values (`src`, token-major
    /// `[t][kv_head][d]`) to `cache` (`[kv_head][position][d]`, `a.capacity`
    /// positions per head) at positions `a.p0..`.
    ///
    /// # Errors
    /// [`ComputeError::Shape`] if a span is too short or the positions overrun the
    /// capacity.
    pub fn kv_store(
        &mut self,
        cache: &Span<'_>,
        src: &Span<'_>,
        a: AttentionShape,
    ) -> Result<(), ComputeError> {
        let per_token = a.kv_heads * a.head_dim;
        need("kv_store", "source", src, a.tokens * per_token)?;
        need(
            "kv_store",
            "cache",
            cache,
            a.kv_heads * a.capacity * a.head_dim,
        )?;
        if a.p0 + a.tokens > a.capacity {
            return Err(shape(
                "kv_store",
                format!(
                    "positions {}..{} exceed {}",
                    a.p0,
                    a.p0 + a.tokens,
                    a.capacity
                ),
            ));
        }
        let n = a.tokens * per_token;
        if n == 0 {
            return Ok(());
        }
        let p = KvParams {
            s: u32_of(a.tokens, "kv_store")?,
            kv_heads: u32_of(a.kv_heads, "kv_store")?,
            d: u32_of(a.head_dim, "kv_store")?,
            p0: u32_of(a.p0, "kv_store")?,
            stride: u32_of(a.capacity * a.head_dim, "kv_store")?,
        };
        u32_of(n, "kv_store")?;
        #[cfg(target_os = "macos")]
        {
            let pb = bytes_of(&p);
            self.rec.dispatch(
                op("kv_store"),
                &[cache.arg(), src.arg(), Arg::Bytes(&pb)],
                groups(n),
                [TG, 1, 1],
            );
        }
        #[cfg(not(target_os = "macos"))]
        let _ = p;
        Ok(())
    }

    /// `ops::attend` for every query head of `a.tokens` tokens: `q` is token-major
    /// `[t][head][d]`, `k` and `v` are caches as [`Batch::kv_store`] writes them,
    /// `scores` is scratch of `tokens × heads × (p0 + tokens)` and `out` receives
    /// `[t][head][d]`.
    ///
    /// # Errors
    /// [`ComputeError::Shape`] if a span is too short or the shape is inconsistent.
    pub fn attention(
        &mut self,
        q: &Span<'_>,
        k: &Span<'_>,
        v: &Span<'_>,
        scores: &Span<'_>,
        out: &Span<'_>,
        a: AttentionShape,
    ) -> Result<(), ComputeError> {
        if a.kv_heads == 0 || a.heads % a.kv_heads != 0 {
            return Err(shape(
                "attention",
                format!("{} heads over {} KV heads", a.heads, a.kv_heads),
            ));
        }
        let span = a.p0 + a.tokens;
        if span > a.capacity {
            return Err(shape(
                "attention",
                format!("{span} positions exceed {}", a.capacity),
            ));
        }
        let qn = a.tokens * a.heads * a.head_dim;
        need("attention", "q", q, qn)?;
        need("attention", "out", out, qn)?;
        let cn = a.kv_heads * a.capacity * a.head_dim;
        need("attention", "k", k, cn)?;
        need("attention", "v", v, cn)?;
        need("attention", "scores", scores, a.tokens * a.heads * span)?;
        if qn == 0 {
            return Ok(());
        }
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (a.head_dim as f32).sqrt();
        let p = AttnParams {
            s: u32_of(a.tokens, "attention")?,
            heads: u32_of(a.heads, "attention")?,
            group: u32_of(a.heads / a.kv_heads, "attention")?,
            d: u32_of(a.head_dim, "attention")?,
            p0: u32_of(a.p0, "attention")?,
            span: u32_of(span, "attention")?,
            stride: u32_of(a.capacity * a.head_dim, "attention")?,
            scale,
        };
        u32_of(a.tokens * a.heads * span, "attention")?;
        #[cfg(target_os = "macos")]
        {
            let pb = bytes_of(&p);
            self.rec.dispatch(
                op("attn_scores"),
                &[q.arg(), k.arg(), scores.arg(), Arg::Bytes(&pb)],
                [span.div_ceil(64), a.heads, a.tokens],
                [64, 1, 1],
            );
            self.rec.dispatch(
                op("attn_softmax"),
                &[scores.arg(), Arg::Bytes(&pb)],
                groups(a.tokens * a.heads),
                [TG, 1, 1],
            );
            self.rec.dispatch(
                op("attn_values"),
                &[scores.arg(), v.arg(), out.arg(), Arg::Bytes(&pb)],
                [(a.heads * a.head_dim).div_ceil(64), a.tokens, 1],
                [64, 1, 1],
            );
        }
        #[cfg(not(target_os = "macos"))]
        let _ = p;
        Ok(())
    }

    /// The MoE combine: `out[t] = Σ_r w[t][r] · contrib[t][r]` over `k` ranks, for `s`
    /// tokens of `h` values (sequential fused multiply-adds from zero, in rank order).
    ///
    /// # Errors
    /// [`ComputeError::Shape`] if a span is too short.
    pub fn moe_combine(
        &mut self,
        contrib: &Span<'_>,
        w: &Span<'_>,
        out: &Span<'_>,
        s: usize,
        k: usize,
        h: usize,
    ) -> Result<(), ComputeError> {
        need("moe_combine", "contributions", contrib, s * k * h)?;
        need("moe_combine", "weights", w, s * k)?;
        need("moe_combine", "output", out, s * h)?;
        if s * h == 0 {
            return Ok(());
        }
        let p = CombineParams {
            s: u32_of(s, "moe_combine")?,
            k: u32_of(k, "moe_combine")?,
            h: u32_of(h, "moe_combine")?,
        };
        u32_of(s * k * h, "moe_combine")?;
        #[cfg(target_os = "macos")]
        {
            let pb = bytes_of(&p);
            self.rec.dispatch(
                op("moe_combine"),
                &[contrib.arg(), w.arg(), out.arg(), Arg::Bytes(&pb)],
                [h.div_ceil(64), s, 1],
                [64, 1, 1],
            );
        }
        #[cfg(not(target_os = "macos"))]
        let _ = p;
        Ok(())
    }
}
