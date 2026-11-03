//! Metal against CPU: every output must be bit-identical (ADR-007: the GPU reproduces
//! the CPU's dequantised values and reduction shape, so no drift tolerance is needed).
//!
//! Skipped, with a message, on machines without a Metal device.

#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use qubivex_backend_cpu::CpuBackend;
use std::sync::Arc;

use qubivex_backend_metal::{MetalBackend, ResidentMatMul};
use qubivex_compute::host::PageAligned;
use qubivex_compute::{ComputeBackend, ComputeError, MatMul, WeightView};
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

const DTYPES: [DType; 6] = [
    DType::F32,
    DType::F16,
    DType::BF16,
    DType::Ggml(GgmlType::Q8_0),
    DType::Ggml(GgmlType::Q4_K),
    DType::Ggml(GgmlType::Q6_K),
];

/// How weight bytes are drawn.
#[derive(Clone, Copy, Debug)]
enum Fill {
    /// Values and scales in a typical model range.
    Typical,
    /// Every byte random, except that no weight is NaN or infinite: f16 scales and
    /// elements span the whole finite range (f16 subnormals included, which are normal
    /// in f32). f32 and bf16 values stay in the normal f32 range, and no product
    /// reaches the f32 subnormal range: Apple GPUs flush f32 subnormals to zero (see
    /// `subnormals_are_the_only_divergence`).
    Wild,
}

fn finite_f16(rng: &mut Rng) -> [u8; 2] {
    loop {
        let h = (rng.next() >> 48) as u16;
        if (h >> 10) & 0x1f != 0x1f {
            return h.to_le_bytes();
        }
    }
}

fn weights(dtype: DType, rows: usize, cols: usize, fill: Fill, rng: &mut Rng) -> Vec<u8> {
    let n = dtype
        .storage_bytes(&Shape::new(&[rows as u64, cols as u64]).unwrap())
        .unwrap() as usize;
    let mut b: Vec<u8> = (0..n).map(|_| (rng.next() >> 56) as u8).collect();
    let wild = matches!(fill, Fill::Wild);
    let scale = |b: &mut [u8], off: usize, rng: &mut Rng| {
        let s = if wild {
            finite_f16(rng)
        } else {
            0x2000u16.to_le_bytes()
        };
        b[off..off + 2].copy_from_slice(&s);
    };
    match dtype {
        DType::F32 => {
            for c in b.chunks_exact_mut(4) {
                let v = if wild {
                    // Any exponent from 1e-15 to 1e18.
                    loop {
                        let v = f32::from_bits((rng.next() >> 32) as u32);
                        if v.is_finite() && (1e-15..1e18).contains(&v.abs()) {
                            break v;
                        }
                    }
                } else {
                    rng.f() * 0.05
                };
                c.copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::F16 => {
            for c in b.chunks_exact_mut(2) {
                let h = if wild {
                    finite_f16(rng)
                } else {
                    (0x2000u16 + (rng.next() >> 54) as u16).to_le_bytes()
                };
                c.copy_from_slice(&h);
            }
        }
        DType::BF16 => {
            for c in b.chunks_exact_mut(2) {
                let h = if wild {
                    loop {
                        let h = (rng.next() >> 48) as u16;
                        // Exponents from 2^-50 to 2^60.
                        if (0x4d..0xbb).contains(&((h >> 7) & 0xff)) {
                            break h;
                        }
                    }
                } else {
                    ((rng.f() * 0.05).to_bits() >> 16) as u16
                };
                c.copy_from_slice(&h.to_le_bytes());
            }
        }
        DType::Ggml(GgmlType::Q8_0) => (0..n).step_by(34).for_each(|o| scale(&mut b, o, rng)),
        DType::Ggml(GgmlType::Q4_K) => (0..n).step_by(144).for_each(|o| {
            scale(&mut b, o, rng);
            scale(&mut b, o + 2, rng);
        }),
        DType::Ggml(GgmlType::Q6_K) => (0..n)
            .step_by(210)
            .for_each(|o| scale(&mut b, o + 208, rng)),
        _ => unreachable!(),
    }
    b
}

fn activations(len: usize, fill: Fill, rng: &mut Rng) -> Vec<f32> {
    (0..len)
        .map(|i| match fill {
            Fill::Typical => rng.f(),
            // Mix in tiny (normal) and signed-zero activations.
            Fill::Wild => match i % 7 {
                0 => rng.f() * 1e-6,
                1 => 0.0,
                2 => -0.0,
                _ => rng.f() * 4.0,
            },
        })
        .collect()
}

fn shapes(d: DType) -> &'static [(usize, usize)] {
    match d {
        DType::Ggml(GgmlType::Q4_K | GgmlType::Q6_K) => &[(37, 256), (64, 1024), (9, 2048)],
        // 32 and 96 columns: tails of 32 lanes.
        DType::Ggml(GgmlType::Q8_0) => &[(37, 32), (70, 96), (64, 2048)],
        // 100 and 37: tails of 36 and 37 lanes, so both halves of a thread's lanes
        // are partly used.
        _ => &[(37, 32), (70, 96), (13, 100), (5, 37), (64, 2048)],
    }
}

fn metal() -> Option<MetalBackend> {
    match MetalBackend::new() {
        Ok(m) => Some(m),
        // Only a missing device skips; a shader that fails to compile is a failure.
        Err(qubivex_backend_metal::MetalError::Unavailable(e)) => {
            eprintln!("skipping: {e}");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

/// Bits, with every NaN mapped to one pattern (payloads are not part of the contract).
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter()
        .map(|x| if x.is_nan() { 0x7fc0_0000 } else { x.to_bits() })
        .collect()
}

#[test]
fn metal_matches_cpu_bit_for_bit() {
    let Some(gpu) = metal() else { return };
    let cpu = CpuBackend::new();
    let mut rng = Rng(11);
    for fill in [Fill::Typical, Fill::Wild] {
        for d in DTYPES {
            for &(n, k) in shapes(d) {
                let wb = weights(d, n, k, fill, &mut rng);
                let w = WeightView::new(d, n, k, &wb).unwrap();
                for s in [1, 3, 8] {
                    let xs = activations(s * k, fill, &mut rng);
                    let mut want = vec![0f32; s * n];
                    let mut got = vec![f32::NAN; s * n];
                    cpu.matmul(&w, &xs, s, &mut want).unwrap();
                    gpu.matmul(&w, &xs, s, &mut got).unwrap();
                    let (wb, gb) = (bits(&want), bits(&got));
                    if let Some(i) = (0..wb.len()).find(|&i| wb[i] != gb[i]) {
                        panic!(
                            "{d} {n}x{k} s={s} {fill:?}: output {i} cpu {} ({:#010x}) metal {} ({:#010x})",
                            want[i], wb[i], got[i], gb[i]
                        );
                    }
                }
            }
        }
    }
}

/// `v` with an f32 subnormal replaced by a zero of the same sign.
fn ftz(v: f32) -> f32 {
    if v != 0.0 && v.abs() < f32::MIN_POSITIVE {
        0.0f32.copysign(v)
    } else {
        v
    }
}

/// Apple GPUs flush f32 subnormals to zero, in safe math mode too, so a weight, an
/// activation, a product or an accumulator in the subnormal range is where Metal and
/// the CPU can differ (ADR-011). Each case gives either the CPU's bits or the result
/// with every subnormal input and output flushed.
#[test]
fn subnormals_are_the_only_divergence() {
    let Some(gpu) = metal() else { return };
    let cpu = CpuBackend::new();
    let cases: [(f32, f32); 4] = [(1.0, 1e-39), (1e-39, 1.0), (1e-20, 1e-20), (0.5, 2e-38)];
    let mut flushed = 0;
    for (a, x) in cases {
        let row: Vec<u8> = std::iter::once(a)
            .chain(std::iter::repeat_n(0.0f32, 63))
            .flat_map(f32::to_le_bytes)
            .collect();
        let w = WeightView::new(DType::F32, 1, 64, &row).unwrap();
        let mut xs = vec![0f32; 64];
        xs[0] = x;
        let (mut want, mut got) = ([0f32; 1], [0f32; 1]);
        cpu.matmul(&w, &xs, 1, &mut want).unwrap();
        gpu.matmul(&w, &xs, 1, &mut got).unwrap();
        let ftz_want = ftz(ftz(a) * ftz(x));
        assert!(
            got[0].to_bits() == want[0].to_bits() || got[0].to_bits() == ftz_want.to_bits(),
            "{a:e} * {x:e}: cpu {:e} metal {:e}",
            want[0],
            got[0]
        );
        flushed += usize::from(got[0].to_bits() != want[0].to_bits());
    }
    eprintln!(
        "{flushed} of {} subnormal cases flushed to zero",
        cases.len()
    );
}

#[test]
fn resident_weights_give_the_same_bits() {
    let Some(gpu) = metal() else { return };
    let mut rng = Rng(5);
    let d = DType::Ggml(GgmlType::Q4_K);
    let (n, k) = (64, 1024);
    let wb = weights(d, n, k, Fill::Typical, &mut rng);
    let w = WeightView::new(d, n, k, &wb).unwrap();
    let resident = gpu.upload(&w).unwrap();
    assert_eq!(
        (resident.rows(), resident.cols(), resident.row_bytes()),
        (n, k, w.row_bytes())
    );
    let xs = activations(2 * k, Fill::Typical, &mut rng);
    let (mut a, mut b) = (vec![0f32; 2 * n], vec![0f32; 2 * n]);
    gpu.matmul(&w, &xs, 2, &mut a).unwrap();
    for _ in 0..3 {
        gpu.matmul_resident(&resident, &xs, 2, &mut b).unwrap();
        assert_eq!(bits(&a), bits(&b));
    }
}

#[test]
fn errors_are_reported_not_panics() {
    let Some(gpu) = metal() else { return };
    let wb = vec![0u8; 4 * 64 * 4];
    let w = WeightView::new(DType::F32, 4, 64, &wb).unwrap();
    let mut y = vec![0f32; 4];
    assert!(matches!(
        gpu.matmul(&w, &[0.0; 63], 1, &mut y),
        Err(ComputeError::Shape { .. })
    ));
    assert!(matches!(
        gpu.dequant_row(&w, 0, &mut [0.0; 64]),
        Err(ComputeError::Unsupported { .. })
    ));
    let q = vec![0u8; 4 * 18];
    let w = WeightView::new(DType::Ggml(GgmlType::Q4_0), 4, 32, &q).unwrap();
    assert!(!gpu.supports(DType::Ggml(GgmlType::Q4_0)));
    assert!(matches!(
        gpu.matmul(&w, &[0.0; 32], 1, &mut y),
        Err(ComputeError::Unsupported { .. })
    ));
    // Empty work is a no-op, not a zero-length buffer.
    gpu.matmul(&w.rows_slice(0, 0).unwrap(), &[], 0, &mut [])
        .ok();
    let info = gpu.info();
    eprintln!("{info:?}");
    assert!(!info.name.is_empty());
}

/// Batches in one command buffer, over a wrapped host region at odd offsets, with a
/// shared input, every dtype and shrinking sizes (staging reuse): the CPU's bits.
#[test]
fn batched_and_wrapped_weights_give_the_cpu_bits() {
    let Some(gpu) = metal() else { return };
    let cpu = CpuBackend::new();
    let mut rng = Rng(23);
    let k = 1024;
    // Every dtype as one matrix of a single host region, at offsets that are not page
    // (or even 16-byte) aligned, except Q4_K, whose kernel needs 16-byte alignment.
    let mats: Vec<(DType, usize, Vec<u8>)> = DTYPES
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            let n = 17 + 8 * i;
            (d, n, weights(d, n, k, Fill::Typical, &mut rng))
        })
        .collect();
    let mut at = Vec::new();
    let mut len = 0usize;
    for (d, _, b) in &mats {
        len += 4; // misalign the next matrix
        if *d == DType::Ggml(GgmlType::Q4_K) {
            len = len.next_multiple_of(16);
        }
        at.push(len);
        len += b.len();
    }
    let mut region = PageAligned::zeroed(len);
    for ((_, _, b), &a) in mats.iter().zip(&at) {
        region.as_mut_slice()[a..a + b.len()].copy_from_slice(b);
    }
    let region = Arc::new(region);
    let host = gpu.map_host(Arc::clone(&region)).unwrap();
    assert_eq!(host.len(), len);
    let views: Vec<WeightView<'_>> = mats
        .iter()
        .zip(&at)
        .map(|((d, n, b), &a)| {
            WeightView::new(*d, *n, k, &region.as_slice()[a..a + b.len()]).unwrap()
        })
        .collect();
    let resident: Vec<_> = views
        .iter()
        .map(|w| host.weights(w).unwrap().unwrap())
        .collect();
    let outside = weights(DType::F32, 2, k, Fill::Typical, &mut rng);
    let outside = WeightView::new(DType::F32, 2, k, &outside).unwrap();
    assert!(host.weights(&outside).unwrap().is_none());
    // A misaligned Q4_K matrix is not placed (the caller keeps it on the CPU).
    let q4 = DTYPES
        .iter()
        .position(|d| *d == DType::Ggml(GgmlType::Q4_K))
        .unwrap();
    let (d, n, b) = &mats[q4];
    let a = at[q4];
    let shifted = WeightView::new(
        *d,
        *n - 1,
        k,
        &region.as_slice()[a + 4..a + 4 + b.len() / n * (n - 1)],
    );
    assert!(host.weights(&shifted.unwrap()).unwrap().is_none());

    for s in [8, 3, 1] {
        let shared = activations(s * k, Fill::Typical, &mut rng);
        let own = activations(s * k, Fill::Typical, &mut rng);
        // Even ops read the shared input, odd ones their own.
        let input = |i: usize| if i % 2 == 0 { &shared } else { &own };
        let want: Vec<Vec<f32>> = views
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let mut y = vec![0f32; s * w.rows()];
                cpu.matmul(w, input(i), s, &mut y).unwrap();
                y
            })
            .collect();
        let mut got: Vec<Vec<f32>> = views.iter().map(|w| vec![f32::NAN; s * w.rows()]).collect();
        let mut ops: Vec<ResidentMatMul<'_>> = resident
            .iter()
            .zip(got.iter_mut())
            .enumerate()
            .map(|(i, (w, y))| ResidentMatMul {
                w,
                xs: input(i),
                s,
                ys: y,
            })
            .collect();
        gpu.matmul_many_resident(&mut ops).unwrap();
        for (i, (a, b)) in want.iter().zip(&got).enumerate() {
            assert_eq!(
                bits(a),
                bits(b),
                "resident op {i} ({}) s={s}",
                views[i].dtype()
            );
        }
        // The copying batch path, through the trait.
        let mut got2: Vec<Vec<f32>> = views.iter().map(|w| vec![f32::NAN; s * w.rows()]).collect();
        let mut ops: Vec<MatMul<'_>> = views
            .iter()
            .zip(got2.iter_mut())
            .enumerate()
            .map(|(i, (w, y))| MatMul {
                w: *w,
                xs: input(i),
                s,
                ys: y,
            })
            .collect();
        gpu.matmul_many(&mut ops).unwrap();
        assert_eq!(
            want.iter().map(|v| bits(v)).collect::<Vec<_>>(),
            got2.iter().map(|v| bits(v)).collect::<Vec<_>>()
        );
    }
    // The weights outlive the region's other holders.
    drop(host);
    drop(views);
    let xs = activations(k, Fill::Typical, &mut rng);
    let mut y = vec![0f32; resident[0].rows()];
    gpu.matmul_resident(&resident[0], &xs, 1, &mut y).unwrap();
}

#[test]
fn batch_shape_errors_run_nothing() {
    let Some(gpu) = metal() else { return };
    let wb = vec![0u8; 4 * 64 * 4];
    let w = WeightView::new(DType::F32, 4, 64, &wb).unwrap();
    let r = gpu.upload(&w).unwrap();
    let xs = [1.0f32; 64];
    let (mut a, mut b) = ([7.0f32; 4], [7.0f32; 3]);
    let mut ops = [
        ResidentMatMul {
            w: &r,
            xs: &xs,
            s: 1,
            ys: &mut a,
        },
        ResidentMatMul {
            w: &r,
            xs: &xs,
            s: 1,
            ys: &mut b,
        },
    ];
    assert!(matches!(
        gpu.matmul_many_resident(&mut ops),
        Err(ComputeError::Shape { .. })
    ));
    assert_eq!(a, [7.0; 4], "nothing ran");
    gpu.matmul_many_resident(&mut []).unwrap();
}
