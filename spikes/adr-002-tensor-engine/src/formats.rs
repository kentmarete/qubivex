//! Weight formats and the own fused kernels' per-row primitives.
//!
//! Every kernel uses the same reduction shape: 64 f32 lanes (16 independent 4-wide FMA
//! chains, enough to cover FMA latency), one fused multiply-add per lane per 64-element
//! chunk, chunks walked in increasing `k`, then a fixed halving-tree lane reduction
//! ([`reduce`]). The dequantized values a fused kernel
//! multiplies with are exactly what [`Weights::dequant_row`] writes, so the S=1 fused
//! path and the S>1 dequantize-then-dot path produce identical bits.

use half::{bf16, f16};

/// Lanes per accumulator.
pub const LANES: usize = 64;
/// Q8_0 block: 32 weights.
pub const QK8_0: usize = 32;
/// Q8_0 block bytes: f16 scale + 32 × i8.
pub const Q8_0_BYTES: usize = 34;
/// K-quant super-block: 256 weights.
pub const QK_K: usize = 256;
/// Q4_K block bytes: f16 d, f16 dmin, 12 packed scale/min bytes, 128 nibble bytes.
pub const Q4_K_BYTES: usize = 144;

/// A row-major `n × k` weight matrix (`y = W·x`, `W` has `n` rows of `k`).
pub trait Weights: Sync {
    /// Short format label.
    fn label(&self) -> &'static str;
    /// Output rows.
    fn n(&self) -> usize;
    /// Input columns.
    fn k(&self) -> usize;
    /// Bytes of weight storage (what a GEMV must stream).
    fn weight_bytes(&self) -> usize;
    /// Fused dot product of row `r` with `x` (before [`Weights::post_scale`]).
    fn dot_row(&self, r: usize, x: &[f32]) -> f32;
    /// The f32 values [`Weights::dot_row`] multiplies with, for row `r`.
    fn dequant_row(&self, r: usize, out: &mut [f32]);
    /// The row is already f32 in memory (skip the dequantize copy).
    fn row_f32(&self, _r: usize) -> Option<&[f32]> {
        None
    }
    /// The whole matrix is f32 in memory, row-major.
    fn all_f32(&self) -> Option<&[f32]> {
        None
    }
    /// Per-row factor applied after the reduction (1.0 unless row-scaled).
    fn post_scale(&self, _r: usize) -> f32 {
        1.0
    }
    /// Exact effective weights of row `r` in f64 (for the reference).
    fn effective_row_f64(&self, r: usize) -> Vec<f64> {
        let mut tmp = vec![0f32; self.k()];
        self.dequant_row(r, &mut tmp);
        let s = f64::from(self.post_scale(r));
        tmp.iter().map(|&v| f64::from(v) * s).collect()
    }
}

#[inline(always)]
pub fn fma_lanes(acc: &mut [f32; LANES], w: &[f32; LANES], x: &[f32; LANES]) {
    for l in 0..LANES {
        acc[l] = w[l].mul_add(x[l], acc[l]);
    }
}

/// Fixed halving-tree reduction: `a[i] += a[i + w]` for w = 32, 16, 8, 4 (vector adds
/// on a `float32x4 × 16` layout), then lanes as `(l0+l1)+(l2+l3)`.
#[inline(always)]
pub fn reduce(a: &[f32; LANES]) -> f32 {
    let mut a = *a;
    let mut w = LANES / 2;
    while w >= 4 {
        for i in 0..w {
            a[i] += a[i + w];
        }
        w /= 2;
    }
    (a[0] + a[1]) + (a[2] + a[3])
}

#[inline(always)]
fn chunks(x: &[f32]) -> &[[f32; LANES]] {
    let (c, rest) = x.as_chunks::<LANES>();
    debug_assert!(rest.is_empty());
    c
}

/// f32 dot with the shared reduction shape.
#[inline(always)]
pub fn dot_f32(w: &[f32], x: &[f32]) -> f32 {
    let mut acc = [0f32; LANES];
    for (wc, xc) in chunks(w).iter().zip(chunks(x)) {
        fma_lanes(&mut acc, wc, xc);
    }
    reduce(&acc)
}

// ---------------------------------------------------------------------------- f32

pub struct F32Mat {
    pub n: usize,
    pub k: usize,
    pub w: Vec<f32>,
}

impl Weights for F32Mat {
    fn label(&self) -> &'static str {
        "f32"
    }
    fn n(&self) -> usize {
        self.n
    }
    fn k(&self) -> usize {
        self.k
    }
    fn weight_bytes(&self) -> usize {
        self.w.len() * 4
    }
    fn dot_row(&self, r: usize, x: &[f32]) -> f32 {
        dot_f32(&self.w[r * self.k..(r + 1) * self.k], x)
    }
    fn dequant_row(&self, r: usize, out: &mut [f32]) {
        out.copy_from_slice(&self.w[r * self.k..(r + 1) * self.k]);
    }
    fn row_f32(&self, r: usize) -> Option<&[f32]> {
        Some(&self.w[r * self.k..(r + 1) * self.k])
    }
    fn all_f32(&self) -> Option<&[f32]> {
        Some(&self.w)
    }
}

// --------------------------------------------------------------------------- bf16

pub struct Bf16Mat {
    pub n: usize,
    pub k: usize,
    pub w: Vec<u16>,
}

impl Bf16Mat {
    pub fn from_f32(m: &F32Mat) -> Self {
        Self { n: m.n, k: m.k, w: m.w.iter().map(|&v| bf16::from_f32(v).to_bits()).collect() }
    }
}

#[inline(always)]
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

impl Weights for Bf16Mat {
    fn label(&self) -> &'static str {
        "bf16"
    }
    fn n(&self) -> usize {
        self.n
    }
    fn k(&self) -> usize {
        self.k
    }
    fn weight_bytes(&self) -> usize {
        self.w.len() * 2
    }
    fn dot_row(&self, r: usize, x: &[f32]) -> f32 {
        let row = &self.w[r * self.k..(r + 1) * self.k];
        let mut acc = [0f32; LANES];
        for (wc, xc) in row.as_chunks::<LANES>().0.iter().zip(chunks(x)) {
            let w: [f32; LANES] = std::array::from_fn(|l| bf16_to_f32(wc[l]));
            fma_lanes(&mut acc, &w, xc);
        }
        reduce(&acc)
    }
    fn dequant_row(&self, r: usize, out: &mut [f32]) {
        for (o, &b) in out.iter_mut().zip(&self.w[r * self.k..(r + 1) * self.k]) {
            *o = bf16_to_f32(b);
        }
    }
}

// --------------------------------------------------------- int8, per-row symmetric

/// Per-row symmetric int8: `w = q × scale[row]`, f32 activations (lossless path).
pub struct I8RowMat {
    pub n: usize,
    pub k: usize,
    pub q: Vec<i8>,
    pub scale: Vec<f32>,
}

impl I8RowMat {
    pub fn from_f32(m: &F32Mat) -> Self {
        let mut q = Vec::with_capacity(m.w.len());
        let mut scale = Vec::with_capacity(m.n);
        for row in m.w.chunks_exact(m.k) {
            let amax = row.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            let inv = 1.0 / s;
            scale.push(s);
            #[allow(clippy::cast_possible_truncation)]
            q.extend(row.iter().map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8));
        }
        Self { n: m.n, k: m.k, q, scale }
    }
}

impl I8RowMat {
    pub fn dot_row_portable(&self, r: usize, x: &[f32]) -> f32 {
        let row = &self.q[r * self.k..(r + 1) * self.k];
        let mut acc = [0f32; LANES];
        for (wc, xc) in row.as_chunks::<LANES>().0.iter().zip(chunks(x)) {
            let w: [f32; LANES] = std::array::from_fn(|l| f32::from(wc[l]));
            fma_lanes(&mut acc, &w, xc);
        }
        reduce(&acc)
    }
}

impl Weights for I8RowMat {
    fn label(&self) -> &'static str {
        "int8-row"
    }
    fn n(&self) -> usize {
        self.n
    }
    fn k(&self) -> usize {
        self.k
    }
    fn weight_bytes(&self) -> usize {
        self.q.len() + self.scale.len() * 4
    }
    fn dot_row(&self, r: usize, x: &[f32]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        return crate::neon::i8_dot(&self.q[r * self.k..(r + 1) * self.k], x);
        #[cfg(not(target_arch = "aarch64"))]
        self.dot_row_portable(r, x)
    }
    fn dequant_row(&self, r: usize, out: &mut [f32]) {
        for (o, &b) in out.iter_mut().zip(&self.q[r * self.k..(r + 1) * self.k]) {
            *o = f32::from(b);
        }
    }
    fn post_scale(&self, r: usize) -> f32 {
        self.scale[r]
    }
}

// ------------------------------------------------------------- GGML Q8_0 and Q4_K

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlKind {
    Q8_0,
    Q4K,
}

/// Raw GGML blocks, row-major, as produced by the GGML/Candle quantizers.
pub struct GgmlMat {
    pub n: usize,
    pub k: usize,
    pub kind: GgmlKind,
    pub bytes: Vec<u8>,
    pub row_bytes: usize,
}

impl GgmlMat {
    pub fn new(n: usize, k: usize, kind: GgmlKind, bytes: Vec<u8>) -> Self {
        let row_bytes = match kind {
            GgmlKind::Q8_0 => k / QK8_0 * Q8_0_BYTES,
            GgmlKind::Q4K => k / QK_K * Q4_K_BYTES,
        };
        assert_eq!(bytes.len(), row_bytes * n, "ggml byte size mismatch");
        Self { n, k, kind, bytes, row_bytes }
    }

    fn row(&self, r: usize) -> &[u8] {
        &self.bytes[r * self.row_bytes..(r + 1) * self.row_bytes]
    }
}

#[inline(always)]
fn f16_at(b: &[u8], i: usize) -> f32 {
    f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

/// GGML `get_scale_min_k4`.
#[inline(always)]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Visits the dequantized weights of one Q4_K row 64 at a time, in `k` order.
/// `d1 * q - m1` is evaluated as a separate multiply and subtract (no FMA), as GGML's
/// and Candle's `dequantize_row_q4_K` do.
#[inline(always)]
fn q4k_row_chunks(row: &[u8], mut f: impl FnMut(usize, &[f32; LANES])) {
    let mut c = 0;
    for blk in row.chunks_exact(Q4_K_BYTES) {
        let d = f16_at(blk, 0);
        let dmin = f16_at(blk, 2);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        for j in 0..4 {
            let (sc, m) = scale_min_k4(2 * j, scales);
            let (d1, m1) = (d * f32::from(sc), dmin * f32::from(m));
            let (sc, m) = scale_min_k4(2 * j + 1, scales);
            let (d2, m2) = (d * f32::from(sc), dmin * f32::from(m));
            let q = &qs[32 * j..32 * j + 32];
            let mut w = [0f32; LANES];
            let (lo, hi) = w.split_at_mut(32);
            for ((a, b), &v) in lo.iter_mut().zip(hi.iter_mut()).zip(q) {
                *a = d1 * f32::from(v & 0xF) - m1;
                *b = d2 * f32::from(v >> 4) - m2;
            }
            f(c, &w);
            c += 1;
        }
    }
}

#[inline(always)]
fn q8_0_row_chunks(row: &[u8], mut f: impl FnMut(usize, &[f32; LANES])) {
    // Two 32-weight blocks per 64-lane chunk.
    for (c, pair) in row.chunks_exact(2 * Q8_0_BYTES).enumerate() {
        let (b0, b1) = pair.split_at(Q8_0_BYTES);
        let (d0, d1) = (f16_at(b0, 0), f16_at(b1, 0));
        let mut w = [0f32; LANES];
        let (lo, hi) = w.split_at_mut(32);
        #[allow(clippy::cast_possible_wrap)]
        for (((a, b), &u), &v) in lo.iter_mut().zip(hi.iter_mut()).zip(&b0[2..]).zip(&b1[2..]) {
            *a = f32::from(u as i8) * d0;
            *b = f32::from(v as i8) * d1;
        }
        f(c, &w);
    }
}

impl GgmlMat {
    pub fn dot_row_portable(&self, r: usize, x: &[f32]) -> f32 {
        let xs = chunks(x);
        let mut acc = [0f32; LANES];
        match self.kind {
            GgmlKind::Q8_0 => q8_0_row_chunks(self.row(r), |c, w| fma_lanes(&mut acc, w, &xs[c])),
            GgmlKind::Q4K => q4k_row_chunks(self.row(r), |c, w| fma_lanes(&mut acc, w, &xs[c])),
        }
        reduce(&acc)
    }
}

impl Weights for GgmlMat {
    fn label(&self) -> &'static str {
        match self.kind {
            GgmlKind::Q8_0 => "q8_0",
            GgmlKind::Q4K => "q4_k",
        }
    }
    fn n(&self) -> usize {
        self.n
    }
    fn k(&self) -> usize {
        self.k
    }
    fn weight_bytes(&self) -> usize {
        self.bytes.len()
    }
    fn dot_row(&self, r: usize, x: &[f32]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        return match self.kind {
            GgmlKind::Q8_0 => crate::neon::q8_0_dot(self.row(r), x),
            GgmlKind::Q4K => crate::neon::q4k_dot(self.row(r), x),
        };
        #[cfg(not(target_arch = "aarch64"))]
        self.dot_row_portable(r, x)
    }
    fn dequant_row(&self, r: usize, out: &mut [f32]) {
        let (oc, _) = out.as_chunks_mut::<LANES>();
        match self.kind {
            GgmlKind::Q8_0 => q8_0_row_chunks(self.row(r), |c, w| oc[c] = *w),
            GgmlKind::Q4K => q4k_row_chunks(self.row(r), |c, w| oc[c] = *w),
        }
    }
}
