//! Criterion benchmarks for the CPU kernels (MIGRATION_PLAN §3a).
//!
//! Every kernel change is measured before and after against these:
//!
//! ```sh
//! cargo bench -p qubivex-backend-cpu --bench kernels -- --save-baseline before
//! # ... change the kernel ...
//! cargo bench -p qubivex-backend-cpu --bench kernels -- --baseline before
//! ```
//!
//! Shapes are OLMoE-1B-7B's: an expert's gate/up projection (1024 × 2048) and down
//! projection (2048 × 1024), and the output head (50304 × 2048). Throughput is the
//! weight bytes read per call, so Criterion reports GB/s.
//!
//! Groups:
//! - `decode_1t`: one activation row on one thread, the kernel alone (no pool).
//! - `decode_t<N>`: one activation row on the compute team, the matrix in cache.
//! - `decode_stream_t<N>`: the same, cycling through 512 MiB of copies so every call
//!   reads its matrix from DRAM, as decode reads each expert once per token. This is
//!   the group the EIP's "quantised GEMV ≥ 70% of STREAM" target is measured on.
//! - `decode_portable`: the portable path at one row and one thread, for the dtypes
//!   with a fused kernel on this CPU, so the fused gain stays visible.
//! - `prefill`: 128 activation rows on the compute team.
//! - `dequant_row` and `dot`: the building blocks, one 2048-wide row.
//!
//! The team size is `QUBIVEX_BENCH_THREADS`, else the performance cores on macOS
//! (`hw.perflevel0.physicalcpu`), else every logical CPU. It is part of the group
//! name (`decode_t5`), so runs at different team sizes are never compared silently.

#![allow(
    missing_docs,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use qubivex_backend_cpu::{CpuBackend, formats, matmul_portable, reduce};
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

/// The dtypes with a fused one-row kernel on this CPU (see `fused` in the crate).
const FUSED: &[DType] = if cfg!(target_arch = "aarch64") {
    &[DType::BF16, Q8_0, Q4_K, Q6_K]
} else {
    &[]
};

fn short(d: DType) -> String {
    d.to_string().to_lowercase()
}

/// The compute-team size (see the module docs).
fn team_threads() -> usize {
    if let Some(n) = std::env::var("QUBIVEX_BENCH_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return n;
    }
    #[cfg(target_os = "macos")]
    if let Some(n) = std::process::Command::new("sysctl")
        .args(["-n", "hw.perflevel0.physicalcpu"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return n;
    }
    std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

/// One matrix and its activations, ready to multiply.
struct Case {
    dtype: DType,
    rows: usize,
    cols: usize,
    w: Vec<u8>,
    xs: Vec<f32>,
}

impl Case {
    fn new(dtype: DType, rows: usize, cols: usize, s: usize) -> Self {
        let mut rng = Rng(0x5eed ^ (rows * 31 + cols) as u64);
        let w = weights(dtype, rows, cols, &mut rng);
        let xs = (0..s * cols).map(|_| rng.f()).collect();
        Self {
            dtype,
            rows,
            cols,
            w,
            xs,
        }
    }

    fn view(&self) -> WeightView<'_> {
        WeightView::new(self.dtype, self.rows, self.cols, &self.w).unwrap()
    }
}

/// Benchmarks `y = W · x` for `s` rows of each `(dtype, shape)` on `threads` threads.
fn matmul_group(
    c: &mut Criterion,
    group: &str,
    dtypes: &[DType],
    shapes: &[(&str, usize, usize)],
    s: usize,
    threads: usize,
    portable: bool,
) {
    let pool = pool(threads);
    let mut g = c.benchmark_group(group);
    for &(shape, rows, cols) in shapes {
        for &dtype in dtypes {
            let case = Case::new(dtype, rows, cols, s);
            let w = case.view();
            let mut ys = vec![0f32; s * rows];
            g.throughput(Throughput::Bytes(case.w.len() as u64));
            g.bench_function(BenchmarkId::new(short(dtype), shape), |b| {
                pool.install(|| {
                    b.iter(|| {
                        if portable {
                            matmul_portable(black_box(&w), black_box(&case.xs), s, &mut ys)
                                .unwrap();
                        } else {
                            CpuBackend
                                .matmul(black_box(&w), black_box(&case.xs), s, &mut ys)
                                .unwrap();
                        }
                        black_box(&ys);
                    });
                });
            });
        }
    }
    g.finish();
}

fn decode(c: &mut Criterion) {
    matmul_group(c, "decode_1t", &ALL, &EXPERT, 1, 1, false);
    let t = team_threads();
    let name = format!("decode_t{t}");
    matmul_group(c, &name, &ALL, &EXPERT, 1, t, false);
    matmul_group(c, &name, &[Q6_K], &[HEAD], 1, t, false);
    if !FUSED.is_empty() {
        matmul_group(c, "decode_portable_1t", FUSED, &EXPERT, 1, 1, true);
    }
    decode_stream(c, t);
}

/// Bytes of weight copies the `decode_stream` group cycles through: larger than any
/// last-level cache we target (including 256 MiB server L3s), so every call reads its
/// matrix from DRAM, as a decode step reads each expert once per token.
const STREAM_BYTES: usize = 512 << 20;

/// One activation row on the team, each call on the next of many copies of the
/// matrix: the memory-bound case. The other groups reuse one matrix, which stays in
/// cache and so measures the kernel's compute, not DRAM.
fn decode_stream(c: &mut Criterion, threads: usize) {
    let pool = pool(threads);
    let mut g = c.benchmark_group(format!("decode_stream_t{threads}"));
    for &(shape, rows, cols) in &EXPERT {
        for dtype in [DType::BF16, Q8_0, Q4_K, Q6_K] {
            let case = Case::new(dtype, rows, cols, 1);
            let copies = STREAM_BYTES.div_ceil(case.w.len());
            let all: Vec<u8> = case.w.repeat(copies);
            let views: Vec<WeightView<'_>> = all
                .chunks_exact(case.w.len())
                .map(|w| WeightView::new(dtype, rows, cols, w).unwrap())
                .collect();
            let mut ys = vec![0f32; rows];
            let mut i = 0usize;
            g.throughput(Throughput::Bytes(case.w.len() as u64));
            g.bench_function(BenchmarkId::new(short(dtype), shape), |b| {
                pool.install(|| {
                    b.iter(|| {
                        i = (i + 1) % views.len();
                        CpuBackend
                            .matmul(black_box(&views[i]), black_box(&case.xs), 1, &mut ys)
                            .unwrap();
                        black_box(&ys);
                    });
                });
            });
        }
    }
    g.finish();
}

fn prefill(c: &mut Criterion) {
    let t = team_threads();
    matmul_group(
        c,
        &format!("prefill_s128_t{t}"),
        &[DType::F32, DType::BF16, Q8_0, Q4_K, Q6_K],
        &EXPERT,
        128,
        t,
        false,
    );
}

fn blocks(c: &mut Criterion) {
    const K: usize = 2048;
    let mut g = c.benchmark_group("dequant_row");
    for dtype in ALL {
        let case = Case::new(dtype, 1, K, 1);
        let mut out = vec![0f32; K];
        g.throughput(Throughput::Bytes(case.w.len() as u64));
        g.bench_function(BenchmarkId::new(short(dtype), K), |b| {
            b.iter(|| {
                formats::dequant_row(dtype, black_box(&case.w), &mut out);
                black_box(&out);
            });
        });
    }
    g.finish();

    let mut rng = Rng(7);
    let w: Vec<f32> = (0..K).map(|_| rng.f()).collect();
    let x: Vec<f32> = (0..K).map(|_| rng.f()).collect();
    let mut g = c.benchmark_group("dot");
    g.throughput(Throughput::Bytes((2 * K * 4) as u64));
    g.bench_function(BenchmarkId::new("f32", K), |b| {
        b.iter(|| black_box(reduce::dot(black_box(&w), black_box(&x))));
    });
    g.finish();
}

fn config() -> Criterion {
    Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .noise_threshold(0.02)
}

criterion_group! {
    name = benches;
    config = config();
    targets = decode, prefill, blocks
}
criterion_main!(benches);
