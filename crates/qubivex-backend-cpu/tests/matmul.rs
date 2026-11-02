//! Matmul invariants: batching, thread count and instruction set never change bits;
//! results agree with an f64 reference over the exact dequantized weights.

#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use qubivex_backend_cpu::{CpuBackend, formats, matmul_portable};
use qubivex_compute::{ComputeBackend, WeightView};
use qubivex_core::{DType, GgmlType, Shape};

/// xorshift64*.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Random weight bytes for `dtype`, with sane f16 scales in the GGML blocks.
fn weights(dtype: DType, rows: usize, cols: usize, rng: &mut Rng) -> Vec<u8> {
    let n = dtype
        .storage_bytes(&Shape::new(&[rows as u64, cols as u64]).unwrap())
        .unwrap() as usize;
    let mut b: Vec<u8> = (0..n).map(|_| (rng.next() >> 56) as u8).collect();
    let scale = 0x2000u16.to_le_bytes(); // f16 ≈ 0.0078
    let mut put = |off: usize| b[off..off + 2].copy_from_slice(&scale);
    match dtype {
        DType::F32 => {
            for c in b.chunks_exact_mut(4) {
                c.copy_from_slice(&(rng.f() * 0.05).to_le_bytes());
            }
        }
        DType::F16 => {
            for c in b.chunks_exact_mut(2) {
                c.copy_from_slice(&(0x2000u16 + (rng.next() >> 54) as u16).to_le_bytes());
            }
        }
        DType::BF16 => {
            for c in b.chunks_exact_mut(2) {
                c.copy_from_slice(&(((rng.f() * 0.05).to_bits() >> 16) as u16).to_le_bytes());
            }
        }
        DType::Ggml(GgmlType::Q8_0) => (0..n).step_by(34).for_each(&mut put),
        DType::Ggml(GgmlType::Q4_K) => (0..n).step_by(144).for_each(|o| {
            put(o);
            put(o + 2);
        }),
        DType::Ggml(GgmlType::Q6_K) => (0..n).step_by(210).for_each(|o| put(o + 208)),
        _ => unreachable!(),
    }
    b
}

const DTYPES: [DType; 6] = [
    DType::F32,
    DType::F16,
    DType::BF16,
    DType::Ggml(GgmlType::Q8_0),
    DType::Ggml(GgmlType::Q4_K),
    DType::Ggml(GgmlType::Q6_K),
];

fn shapes(d: DType) -> &'static [(usize, usize)] {
    match d {
        DType::Ggml(GgmlType::Q4_K | GgmlType::Q6_K) => &[(37, 256), (64, 1024)],
        // 32 columns: a tail shorter than the 64 lanes.
        _ => &[(37, 32), (70, 96), (64, 2048)],
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn batching_threads_and_isa_never_change_bits() {
    let be = CpuBackend::new();
    let mut rng = Rng(7);
    for d in DTYPES {
        for &(n, k) in shapes(d) {
            let wb = weights(d, n, k, &mut rng);
            let w = WeightView::new(d, n, k, &wb).unwrap();
            let s = 5;
            let xs: Vec<f32> = (0..s * k).map(|_| rng.f()).collect();
            let mut batch = vec![0f32; s * n];
            be.matmul(&w, &xs, s, &mut batch).unwrap();
            // One row at a time.
            for t in 0..s {
                let mut one = vec![0f32; n];
                be.matmul(&w, &xs[t * k..(t + 1) * k], 1, &mut one).unwrap();
                assert_eq!(
                    bits(&one),
                    bits(&batch[t * n..(t + 1) * n]),
                    "{d} {n}x{k} token {t}"
                );
            }
            // Thread counts.
            for threads in [1, 3, 7] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let mut y = vec![0f32; s * n];
                pool.install(|| be.matmul(&w, &xs, s, &mut y)).unwrap();
                assert_eq!(bits(&y), bits(&batch), "{d} {n}x{k} threads {threads}");
            }
            // Portable path against the dispatched one (AVX2+FMA on x86-64).
            let mut p = vec![0f32; s * n];
            matmul_portable(&w, &xs, s, &mut p).unwrap();
            assert_eq!(bits(&p), bits(&batch), "{d} {n}x{k} portable vs dispatched");

            // f64 reference over the exact dequantized weights.
            let mut row = vec![0f32; k];
            for r in 0..n {
                be.dequant_row(&w, r, &mut row).unwrap();
                for t in 0..s {
                    let x = &xs[t * k..(t + 1) * k];
                    let want: f64 = row
                        .iter()
                        .zip(x)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum();
                    let scale: f64 = row
                        .iter()
                        .zip(x)
                        .map(|(a, b)| (f64::from(*a) * f64::from(*b)).abs())
                        .sum::<f64>()
                        .max(1e-30);
                    let got = f64::from(batch[t * n + r]);
                    assert!(
                        (got - want).abs() / scale < 1e-6,
                        "{d} {n}x{k} r{r} t{t}: {got} vs {want}"
                    );
                }
            }
        }
    }
}

#[test]
fn errors_are_reported_not_panics() {
    let be = CpuBackend::new();
    let wb = vec![0u8; 4 * 8 * 4];
    let w = WeightView::new(DType::F32, 4, 8, &wb).unwrap();
    let mut y = vec![0f32; 4];
    assert!(be.matmul(&w, &[0.0; 7], 1, &mut y).is_err());
    assert!(be.matmul(&w, &[0.0; 8], 1, &mut [0.0; 3]).is_err());
    assert!(be.dequant_row(&w, 4, &mut [0.0; 8]).is_err());
    let q = vec![0u8; 2 * 18];
    let w = WeightView::new(DType::Ggml(GgmlType::Q4_0), 2, 32, &q).unwrap();
    assert!(!be.supports(w.dtype()));
    assert!(
        be.matmul(&w, &[0.0; 32], 1, &mut [0.0; 2])
            .unwrap_err()
            .to_string()
            .contains("no matmul kernel for q4_0")
    );
    // Empty batches are fine.
    let w = WeightView::new(DType::F32, 4, 8, &wb).unwrap();
    be.matmul(&w, &[], 0, &mut []).unwrap();
    assert!(formats::supported(DType::BF16));
}
