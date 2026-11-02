//! ADR-002 spike: Candle against own kernels on expert-FFN and attention-decode shapes.
//!
//! Not product code. The kernels here are candidates for `qubivex-backend-cpu`.

pub mod attn;
pub mod candle_impl;
pub mod formats;
pub mod kernels;
#[cfg(target_arch = "aarch64")]
pub mod neon;

use formats::{Bf16Mat, F32Mat, GgmlKind, GgmlMat, I8RowMat, Weights};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fmt {
    F32,
    Bf16,
    I8Row,
    Q8_0,
    Q4K,
}

impl Fmt {
    pub const ALL: [Self; 5] = [Self::F32, Self::Bf16, Self::I8Row, Self::Q8_0, Self::Q4K];
    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
            Self::I8Row => "int8-row",
            Self::Q8_0 => "q8_0",
            Self::Q4K => "q4_k",
        }
    }
}

/// One shape's source data: f32 weights plus the GGML blocks Candle's quantizer made.
pub struct Source {
    pub f32: F32Mat,
    pub q8_0: Vec<u8>,
    pub q4k: Vec<u8>,
}

impl Source {
    pub fn new(n: usize, k: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        // Roughly Gaussian (sum of 4 uniforms), std ≈ 0.02, like trained FFN weights.
        let w = (0..n * k)
            .map(|_| (0..4).map(|_| rng.random_range(-1.0f32..1.0)).sum::<f32>() * 0.017)
            .collect();
        let f32 = F32Mat { n, k, w };
        let q8_0 = candle_impl::quantize_bytes(&f32, GgmlKind::Q8_0).expect("q8_0 quantize");
        let q4k = candle_impl::quantize_bytes(&f32, GgmlKind::Q4K).expect("q4_k quantize");
        Self { f32, q8_0, q4k }
    }

    pub fn own(&self, fmt: Fmt) -> Box<dyn Weights> {
        let (n, k) = (self.f32.n, self.f32.k);
        match fmt {
            Fmt::F32 => Box::new(F32Mat { n, k, w: self.f32.w.clone() }),
            Fmt::Bf16 => Box::new(Bf16Mat::from_f32(&self.f32)),
            Fmt::I8Row => Box::new(I8RowMat::from_f32(&self.f32)),
            Fmt::Q8_0 => Box::new(GgmlMat::new(n, k, GgmlKind::Q8_0, self.q8_0.clone())),
            Fmt::Q4K => Box::new(GgmlMat::new(n, k, GgmlKind::Q4K, self.q4k.clone())),
        }
    }

    pub fn candle(&self, fmt: Fmt) -> Option<candle_impl::CandleW> {
        use candle_core::DType;
        use candle_impl::CandleW;
        let (n, k) = (self.f32.n, self.f32.k);
        let r = match fmt {
            Fmt::F32 => CandleW::dense(&self.f32, DType::F32),
            Fmt::Bf16 => CandleW::dense(&self.f32, DType::BF16),
            Fmt::I8Row => return None,
            Fmt::Q8_0 => CandleW::quant_from_bytes(n, k, GgmlKind::Q8_0, &self.q8_0),
            Fmt::Q4K => CandleW::quant_from_bytes(n, k, GgmlKind::Q4K, &self.q4k),
        };
        Some(r.expect("candle weight"))
    }
}

pub fn activations(s: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..s * k).map(|_| rng.random_range(-1.0f32..1.0)).collect()
}

/// `max |y − ref| / max |ref|` over the given outputs.
pub fn rel_err(y: &[f32], r: &[f64]) -> f64 {
    let scale = r.iter().fold(0f64, |a, v| a.max(v.abs()));
    let e = y.iter().zip(r).fold(0f64, |a, (&y, &r)| a.max((f64::from(y) - r).abs()));
    if scale > 0.0 { e / scale } else { e }
}

/// f64 reference outputs for the given token rows of `xs` (`s × k`).
pub fn reference(w: &dyn Weights, xs: &[f32], tokens: &[usize]) -> Vec<Vec<f64>> {
    let (n, k) = (w.n(), w.k());
    let rows: Vec<Vec<f64>> = (0..n).map(|r| w.effective_row_f64(r)).collect();
    tokens
        .iter()
        .map(|&t| {
            let x = &xs[t * k..(t + 1) * k];
            rows.iter().map(|wr| wr.iter().zip(x).map(|(a, &b)| a * f64::from(b)).sum()).collect()
        })
        .collect()
}

/// FNV-1a over the f32 bit patterns.
pub fn bits_hash(y: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in y {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// Attention input with a fixed seed.
pub fn attn_data(h: usize, d: usize, t: usize, seed: u64) -> attn::AttnData {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut v = |len: usize| (0..len).map(|_| rng.random_range(-1.0f32..1.0)).collect::<Vec<_>>();
    attn::AttnData { h, d, t, q: v(h * d), k: v(h * t * d), v: v(h * t * d) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use formats::{QK_K, dot_f32};

    #[test]
    fn own_dequant_matches_candle_bitwise() {
        let src = Source::new(32, 2 * QK_K, 7);
        for kind in [GgmlKind::Q8_0, GgmlKind::Q4K] {
            let bytes = if kind == GgmlKind::Q8_0 { &src.q8_0 } else { &src.q4k };
            let m = GgmlMat::new(32, 2 * QK_K, kind, bytes.clone());
            let theirs = candle_impl::dequantize(32, 2 * QK_K, kind, bytes).expect("dequantize");
            let mut ours = vec![0f32; 2 * QK_K];
            for r in 0..32 {
                m.dequant_row(r, &mut ours);
                let t = &theirs[r * 2 * QK_K..(r + 1) * 2 * QK_K];
                assert!(ours.iter().zip(t).all(|(a, b)| a.to_bits() == b.to_bits()), "{kind:?} row {r}");
            }
        }
    }

    #[test]
    fn fused_equals_dequant_then_dot_bitwise() {
        let src = Source::new(16, 2 * QK_K, 3);
        let x = activations(1, 2 * QK_K, 4);
        for fmt in Fmt::ALL {
            let w = src.own(fmt);
            let mut row = vec![0f32; w.k()];
            for r in 0..w.n() {
                w.dequant_row(r, &mut row);
                assert_eq!(w.dot_row(r, &x).to_bits(), dot_f32(&row, &x).to_bits(), "{fmt:?} row {r}");
            }
        }
    }

    #[test]
    fn simd_twins_equal_portable_bitwise() {
        let (n, k) = (24, 4 * QK_K);
        let src = Source::new(n, k, 11);
        let x = activations(1, k, 12);
        let i8m = I8RowMat::from_f32(&src.f32);
        let q8 = GgmlMat::new(n, k, GgmlKind::Q8_0, src.q8_0.clone());
        let q4 = GgmlMat::new(n, k, GgmlKind::Q4K, src.q4k.clone());
        for r in 0..n {
            assert_eq!(i8m.dot_row(r, &x).to_bits(), i8m.dot_row_portable(r, &x).to_bits(), "int8 row {r}");
            assert_eq!(q8.dot_row(r, &x).to_bits(), q8.dot_row_portable(r, &x).to_bits(), "q8_0 row {r}");
            assert_eq!(q4.dot_row(r, &x).to_bits(), q4.dot_row_portable(r, &x).to_bits(), "q4_k row {r}");
        }
    }

    #[test]
    fn gemm_rows_equal_gemv_bitwise_and_close_to_reference() {
        let (n, k, s) = (48, 2 * QK_K, 11);
        let src = Source::new(n, k, 5);
        let xs = activations(s, k, 6);
        for fmt in Fmt::ALL {
            let w = src.own(fmt);
            let mut ys = vec![0f32; s * n];
            let mut scratch = vec![0f32; s * n];
            kernels::gemm(w.as_ref(), &xs, s, &mut ys, &mut scratch);
            let refs = reference(w.as_ref(), &xs, &(0..s).collect::<Vec<_>>());
            for t in 0..s {
                let mut y = vec![0f32; n];
                kernels::gemv(w.as_ref(), &xs[t * k..(t + 1) * k], &mut y);
                assert_eq!(bits_hash(&y), bits_hash(&ys[t * n..(t + 1) * n]), "{fmt:?} token {t}");
                assert!(rel_err(&y, &refs[t]) < 1e-5, "{fmt:?} token {t}");
            }
        }
    }

    #[test]
    fn attention_close_to_reference() {
        let a = attn_data(4, 128, 97 * 16, 9);
        let mut out = vec![0f32; 4 * 128];
        attn::own(&a, &mut out);
        assert!(rel_err(&out, &attn::reference(&a)) < 1e-5);
        let mut pooled = vec![0f32; 4 * 128];
        attn::own_pool(&a, &mut pooled);
        assert_eq!(bits_hash(&out), bits_hash(&pooled), "pool and rayon attention must match bitwise");
    }
}
