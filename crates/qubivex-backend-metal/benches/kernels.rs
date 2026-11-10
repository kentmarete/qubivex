//! Metal kernel benchmarks, on the shapes of the CPU benches (`qubivex-backend-cpu`
//! `benches/kernels.rs`) so the two can be compared directly.
//!
//! Groups:
//! - `metal_decode`: one activation row, weights resident on the device;
//! - `metal_prefill_s128`: 128 activation rows, resident;
//! - `metal_copying`: through `ComputeBackend::matmul`, which uploads the weights on
//!   every call (the slice-1 integration path);
//! - `metal_dispatch`: a 64 × 64 f32 matmul, which is all per-call overhead;
//! - `metal_batch`: the attention projections of one OLMoE layer (q, k, v: three
//!   2048 × 2048 Q4_K matrices, one shared input) as three submissions and as one;
//! - `metal_stream`: kernel throughput without submission cost: 96 distinct expert
//!   matrices (more than the system cache holds) recorded into one submission, as a
//!   token's routed experts would be.
//!
//! Run with `cargo bench -p qubivex-backend-metal --bench kernels`. Skipped without a
//! Metal device.

#![allow(
    missing_docs,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use qubivex_backend_metal::{MetalBackend, ResidentMatMul};
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

/// Random weight bytes for `dtype`, with sane f16 scales in the GGML blocks (the same
/// generator as `tests/matmul.rs`).
fn weights(dtype: DType, rows: usize, cols: usize, rng: &mut Rng) -> Vec<u8> {
    let n = dtype
        .storage_bytes(&Shape::new(&[rows as u64, cols as u64]).unwrap())
        .unwrap() as usize;
    let mut b: Vec<u8> = (0..n).map(|_| (rng.next() >> 56) as u8).collect();
    let scale = 0x2000u16.to_le_bytes(); // f16 ≈ 0.0078
    let mut put = |off: usize| b[off..off + 2].copy_from_slice(&scale);
    match dtype {
        DType::F32 => {
            for c in b.as_chunks_mut::<4>().0 {
                c.copy_from_slice(&(rng.f() * 0.05).to_le_bytes());
            }
        }
        DType::F16 => {
            for c in b.as_chunks_mut::<2>().0 {
                c.copy_from_slice(&(0x2000u16 + (rng.next() >> 54) as u16).to_le_bytes());
            }
        }
        DType::BF16 => {
            for c in b.as_chunks_mut::<2>().0 {
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

const Q4_K: DType = DType::Ggml(GgmlType::Q4_K);
const Q6_K: DType = DType::Ggml(GgmlType::Q6_K);
const Q8_0: DType = DType::Ggml(GgmlType::Q8_0);
const ALL: [DType; 6] = [DType::F32, DType::F16, DType::BF16, Q8_0, Q4_K, Q6_K];

/// `(name, rows, cols)`: OLMoE-1B-7B expert projections.
const EXPERT: [(&str, usize, usize); 2] = [
    ("gate_up_1024x2048", 1024, 2048),
    ("down_2048x1024", 2048, 1024),
];
const HEAD: (&str, usize, usize) = ("head_50304x2048", 50304, 2048);

fn group(
    c: &mut Criterion,
    gpu: &MetalBackend,
    name: &str,
    dtypes: &[DType],
    shapes: &[(&str, usize, usize)],
    s: usize,
    resident: bool,
) {
    let mut g = c.benchmark_group(name);
    let mut rng = Rng(1);
    for &(shape, rows, cols) in shapes {
        for &dtype in dtypes {
            let wb = weights(dtype, rows, cols, &mut rng);
            let w = WeightView::new(dtype, rows, cols, &wb).unwrap();
            let xs: Vec<f32> = (0..s * cols).map(|_| rng.f()).collect();
            let mut ys = vec![0f32; s * rows];
            g.throughput(Throughput::Bytes(wb.len() as u64));
            let id = BenchmarkId::new(dtype.to_string(), shape);
            if resident {
                let m = gpu.upload(&w).unwrap();
                g.bench_function(id, |b| {
                    b.iter(|| gpu.matmul_resident(&m, &xs, s, &mut ys).unwrap());
                });
            } else {
                g.bench_function(id, |b| b.iter(|| gpu.matmul(&w, &xs, s, &mut ys).unwrap()));
            }
        }
    }
    g.finish();
}

fn benches(c: &mut Criterion) {
    let gpu = match MetalBackend::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping Metal benches: {e}");
            return;
        }
    };
    group(c, &gpu, "metal_decode", &ALL, &EXPERT, 1, true);
    group(c, &gpu, "metal_decode", &[Q6_K], &[HEAD], 1, true);
    group(
        c,
        &gpu,
        "metal_prefill_s128",
        &[Q4_K, Q6_K],
        &EXPERT,
        128,
        true,
    );
    group(c, &gpu, "metal_copying", &[Q4_K], &EXPERT, 1, false);
    group(
        c,
        &gpu,
        "metal_dispatch",
        &[DType::F32],
        &[("tiny_64x64", 64, 64)],
        1,
        true,
    );
    batch(c, &gpu);
    stream(c, &gpu);
}

fn stream(c: &mut Criterion, gpu: &MetalBackend) {
    let mut g = c.benchmark_group("metal_stream");
    let mut rng = Rng(3);
    let count = 96;
    for (dtype, (name, n, k)) in [(Q4_K, EXPERT[0]), (Q4_K, EXPERT[1]), (Q6_K, EXPERT[0])] {
        let ws: Vec<_> = (0..count)
            .map(|_| {
                let wb = weights(dtype, n, k, &mut rng);
                gpu.upload(&WeightView::new(dtype, n, k, &wb).unwrap())
                    .unwrap()
            })
            .collect();
        let bytes: usize = ws.iter().map(|w| w.rows() * w.row_bytes()).sum();
        let x = gpu
            .vec_from(&(0..k).map(|_| rng.f()).collect::<Vec<_>>())
            .unwrap();
        let y = gpu.vec(n).unwrap();
        g.throughput(Throughput::Bytes(bytes as u64));
        g.bench_function(BenchmarkId::new(format!("{dtype}"), name), |b| {
            b.iter(|| {
                gpu.submit(|bt| {
                    for w in &ws {
                        bt.matmul(w, &x.all(), 1, &y.all())?;
                    }
                    Ok(())
                })
                .unwrap();
            });
        });
        // The same bytes as one dispatch: what the serial dispatches cost.
        let wb = weights(dtype, n * count, k, &mut rng);
        let big = gpu
            .upload(&WeightView::new(dtype, n * count, k, &wb).unwrap())
            .unwrap();
        let ybig = gpu.vec(n * count).unwrap();
        g.bench_function(
            BenchmarkId::new(format!("{dtype}/one_dispatch"), name),
            |b| {
                b.iter(|| {
                    gpu.submit(|bt| bt.matmul(&big, &x.all(), 1, &ybig.all()))
                        .unwrap();
                });
            },
        );
    }
    g.finish();
}

fn batch(c: &mut Criterion, gpu: &MetalBackend) {
    let mut g = c.benchmark_group("metal_batch");
    let mut rng = Rng(2);
    let (n, k) = (2048, 2048);
    let ws: Vec<_> = (0..3)
        .map(|_| {
            let wb = weights(Q4_K, n, k, &mut rng);
            gpu.upload(&WeightView::new(Q4_K, n, k, &wb).unwrap())
                .unwrap()
        })
        .collect();
    let xs: Vec<f32> = (0..k).map(|_| rng.f()).collect();
    let mut ys = vec![vec![0f32; n]; 3];
    g.bench_function("qkv_2048_q4_k/separate", |b| {
        b.iter(|| {
            for (w, y) in ws.iter().zip(ys.iter_mut()) {
                gpu.matmul_resident(w, &xs, 1, y).unwrap();
            }
        });
    });
    g.bench_function("qkv_2048_q4_k/one_buffer", |b| {
        b.iter(|| {
            let mut ops: Vec<ResidentMatMul<'_>> = ws
                .iter()
                .zip(ys.iter_mut())
                .map(|(w, y)| ResidentMatMul {
                    w,
                    xs: &xs,
                    s: 1,
                    ys: y,
                })
                .collect();
            gpu.matmul_many_resident(&mut ops).unwrap();
        });
    });
    g.finish();
}

criterion_group!(kernels, benches);
criterion_main!(kernels);
