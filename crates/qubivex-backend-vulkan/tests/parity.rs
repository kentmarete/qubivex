//! The Vulkan device against the CPU backend, bit for bit (ADR-019).
//!
//! Each test opens the default Vulkan device and skips (passes, with a note) when there
//! is none, unless `QUBIVEX_REQUIRE_VULKAN=1`, which makes a missing device a failure.
//! Locally that device is MoltenVK on the Apple GPU; in a Linux container, lavapipe.

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    missing_docs
)]

use std::sync::Arc;

use qubivex_backend_cpu::CpuBackend;
use qubivex_backend_vulkan::{VulkanConfig, VulkanDevice};
use qubivex_compute::discrete::{DeviceMatMul, DeviceWeights, DiscreteDevice};
use qubivex_compute::{ComputeBackend, ComputeError, WeightView};
use qubivex_core::{DType, GgmlType, Shape};

fn open(config: VulkanConfig) -> Option<VulkanDevice> {
    match VulkanDevice::new(config) {
        Ok(d) => Some(d),
        Err(e) if std::env::var("QUBIVEX_REQUIRE_VULKAN").as_deref() == Ok("1") => {
            panic!("QUBIVEX_REQUIRE_VULKAN=1 but no device: {e}")
        }
        Err(e) => {
            eprintln!("Vulkan device unavailable, skipped: {e}");
            None
        }
    }
}

fn device() -> Option<VulkanDevice> {
    open(VulkanConfig::default())
}

const DTYPES: [DType; 6] = [
    DType::F32,
    DType::F16,
    DType::BF16,
    DType::Ggml(GgmlType::Q8_0),
    DType::Ggml(GgmlType::Q4_K),
    DType::Ggml(GgmlType::Q6_K),
];

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}

/// Deterministic weight bytes of `dtype` whose values are normal f32 numbers of
/// moderate size (Apple GPUs flush f32 subnormals, ADR-011 §1).
fn weights(dtype: DType, rows: usize, cols: usize, seed: u32) -> Vec<u8> {
    let n = dtype
        .storage_bytes(&Shape::new(&[rows as u64, cols as u64]).unwrap())
        .unwrap() as usize;
    let mut r = Rng(seed.wrapping_mul(2_654_435_761).max(1));
    match dtype {
        DType::F32 => (0..n / 4)
            .flat_map(|_| {
                let v = r.next();
                // Exponent 120..=135: magnitudes about 2^-7 to 2^8.
                let e = 120 + (v >> 24) % 16;
                ((v & 0x807F_FFFF) | (e << 23)).to_le_bytes()
            })
            .collect(),
        DType::F16 => (0..n / 2)
            .flat_map(|_| {
                let v = r.next() as u16;
                let e = 8 + (v >> 10) % 16;
                ((v & 0x83FF) | (e << 10)).to_le_bytes()
            })
            .collect(),
        DType::BF16 => (0..n / 2)
            .flat_map(|_| {
                let v = r.next() as u16;
                let e = 120 + (v >> 7) % 16;
                ((v & 0x807F) | (e << 7)).to_le_bytes()
            })
            .collect(),
        _ => {
            let mut b: Vec<u8> = (0..n).map(|_| (r.next() >> 24) as u8).collect();
            // Keep the f16 scales of the block formats finite and small.
            for v in b.iter_mut().skip(1).step_by(2) {
                *v &= 0x3b;
            }
            b
        }
    }
}

fn acts(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed) * 37 % 101) as f32 - 50.0) / 25.0)
        .collect()
}

/// Asserts the same bits, naming how many outputs differ and the first few.
#[track_caller]
fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    let bad: Vec<_> = (0..got.len())
        .filter(|&i| got[i].to_bits() != want[i].to_bits())
        .collect();
    assert!(
        bad.is_empty(),
        "{what}: {} of {} outputs differ; first {:?}",
        bad.len(),
        got.len(),
        bad.iter()
            .take(4)
            .map(|&i| (i, got[i], want[i]))
            .collect::<Vec<_>>()
    );
}

fn cpu(dtype: DType, rows: usize, cols: usize, host: &[u8], xs: &[f32], s: usize) -> Vec<f32> {
    let wv = WeightView::new(dtype, rows, cols, host).unwrap();
    let mut ys = vec![0.0; s * rows];
    CpuBackend::new().matmul(&wv, xs, s, &mut ys).unwrap();
    ys
}

/// Column counts with tails for the per-element formats; whole blocks for the
/// K-quants.
fn cols_for(dtype: DType) -> &'static [usize] {
    match dtype {
        DType::Ggml(GgmlType::Q4_K | GgmlType::Q6_K) => &[256, 512, 2048],
        DType::Ggml(GgmlType::Q8_0) => &[32, 96, 544, 2048],
        _ => &[1, 7, 63, 64, 65, 130, 2048],
    }
}

#[test]
fn every_dtype_shape_and_batch_matches_the_cpu() {
    let Some(dev) = device() else { return };
    eprintln!("{}", dev.describe());
    for dtype in DTYPES {
        assert!(dev.supports(dtype));
        for &cols in cols_for(dtype) {
            for rows in [1, 9, 33] {
                let host = weights(dtype, rows, cols, (rows * cols) as u32);
                let wv = WeightView::new(dtype, rows, cols, &host).unwrap();
                let dw = DeviceWeights::like(dev.upload(&host, None).unwrap(), 0, &wv).unwrap();
                for s in [1, 3, 8] {
                    let xs = acts(s * cols, rows);
                    let want = cpu(dtype, rows, cols, &host, &xs, s);
                    let mut got = vec![f32::NAN; s * rows];
                    dev.matmul_many(&mut [DeviceMatMul {
                        w: &dw,
                        xs: &xs,
                        s,
                        ys: &mut got,
                    }])
                    .unwrap();
                    assert_bits(&got, &want, &format!("{dtype} {rows} x {cols}, s = {s}"));
                }
            }
        }
    }
}

#[test]
fn many_matrices_at_offsets_in_one_buffer_and_one_submission() {
    let Some(dev) = device() else { return };
    // Every dtype in one buffer at offsets that are not aligned to anything but their
    // own element size, as slabs and arenas lay them out.
    let mut arena = vec![0xAAu8; 6];
    let mut parts = Vec::new();
    for (i, dtype) in DTYPES.iter().enumerate() {
        let (rows, cols) = (5 + i, cols_for(*dtype)[1]);
        let host = weights(*dtype, rows, cols, 99 + i as u32);
        parts.push((*dtype, rows, cols, arena.len(), host.clone()));
        arena.extend_from_slice(&host);
        arena.extend_from_slice(&[0x55; 2]);
    }
    let buf = dev.upload(&arena, None).unwrap();
    let dws: Vec<_> = parts
        .iter()
        .map(|(dt, r, c, off, _)| DeviceWeights::new(buf.clone(), *off, *dt, *r, *c).unwrap())
        .collect();
    let s = 2;
    let xs: Vec<_> = parts.iter().map(|p| acts(s * p.2, p.1)).collect();
    let mut ys: Vec<_> = parts.iter().map(|p| vec![f32::NAN; s * p.1]).collect();
    let mut ops: Vec<_> = dws
        .iter()
        .zip(&xs)
        .zip(ys.iter_mut())
        .map(|((w, x), y)| DeviceMatMul { w, xs: x, s, ys: y })
        .collect();
    dev.matmul_many(&mut ops).unwrap();
    drop(ops);
    for (p, (x, y)) in parts.iter().zip(xs.iter().zip(&ys)) {
        let want = cpu(p.0, p.1, p.2, &p.4, x, s);
        assert_bits(y, &want, &format!("{} at {}", p.0, p.3));
    }
    let t = dev.transfers();
    assert_eq!(t.submissions, 1);
    assert_eq!(t.uploads, 1);
}

#[test]
fn large_submissions_reuse_and_grow_their_buffers() {
    let Some(dev) = device() else { return };
    let dtype = DType::Ggml(GgmlType::Q4_K);
    let (rows, cols) = (1024, 2048);
    let host = weights(dtype, rows, cols, 5);
    let wv = WeightView::new(dtype, rows, cols, &host).unwrap();
    let dw = DeviceWeights::like(dev.upload(&host, None).unwrap(), 0, &wv).unwrap();
    // 100 ops in one submission: more than the first descriptor pool holds.
    let xs = acts(cols, 3);
    let want = cpu(dtype, rows, cols, &host, &xs, 1);
    let mut ys = vec![vec![0.0; rows]; 100];
    let mut ops: Vec<_> = ys
        .iter_mut()
        .map(|y| DeviceMatMul {
            w: &dw,
            xs: &xs,
            s: 1,
            ys: y,
        })
        .collect();
    dev.matmul_many(&mut ops).unwrap();
    drop(ops);
    for (i, y) in ys.iter().enumerate() {
        assert_bits(y, &want, &format!("op {i}"));
    }
}

#[test]
fn capacity_is_enforced_and_freed_on_drop() {
    let Some(dev) = open(VulkanConfig {
        device: None,
        capacity: Some(1000),
    }) else {
        return;
    };
    let a = dev.upload(&[1; 600], None).unwrap();
    let clone = a.clone();
    assert!(matches!(
        dev.upload(&[2; 500], None),
        Err(ComputeError::OutOfDeviceMemory {
            requested: 500,
            used: 600,
            capacity: 1000,
            ..
        })
    ));
    drop(a);
    assert_eq!(dev.usage().used, 600, "a clone keeps the memory");
    drop(clone);
    let b = dev.upload(&[2; 900], None).unwrap();
    let u = dev.usage();
    assert_eq!((u.used, u.peak, u.refused), (900, 900, 1));
    drop(b);
    assert_eq!(dev.usage().used, 0);
}

#[test]
fn guards_live_as_long_as_the_buffer() {
    let Some(dev) = device() else { return };
    let live = Arc::new(());
    let b = dev
        .upload(&[0; 10], Some(Box::new(Arc::clone(&live))))
        .unwrap();
    assert_eq!(Arc::strong_count(&live), 2);
    drop(b);
    assert_eq!(Arc::strong_count(&live), 1);
}

#[test]
fn buffers_of_another_device_are_refused() {
    let (Some(a), Some(b)) = (device(), device()) else {
        return;
    };
    let host = weights(DType::F32, 2, 4, 1);
    let wv = WeightView::new(DType::F32, 2, 4, &host).unwrap();
    let dw = DeviceWeights::like(a.upload(&host, None).unwrap(), 0, &wv).unwrap();
    let mut ys = [0.0; 2];
    let e = b.matmul_many(&mut [DeviceMatMul {
        w: &dw,
        xs: &[1.0; 4],
        s: 1,
        ys: &mut ys,
    }]);
    assert!(matches!(e, Err(ComputeError::Device { .. })), "{e:?}");
}

#[test]
fn transfers_are_counted() {
    let Some(dev) = device() else { return };
    let host = weights(DType::F32, 4, 8, 3);
    let wv = WeightView::new(DType::F32, 4, 8, &host).unwrap();
    let dw = DeviceWeights::like(dev.upload(&host, None).unwrap(), 0, &wv).unwrap();
    let mut ys = [0.0; 8];
    dev.matmul_many(&mut [DeviceMatMul {
        w: &dw,
        xs: &acts(16, 0),
        s: 2,
        ys: &mut ys,
    }])
    .unwrap();
    let t = dev.transfers();
    assert_eq!((t.uploads, t.upload_bytes), (1, 128));
    assert_eq!(
        (t.activations_in, t.activations_out, t.submissions),
        (64, 32, 1)
    );
    assert!(t.transfer_ns > 0);
}

#[test]
fn wrong_activation_lengths_are_refused() {
    let Some(dev) = device() else { return };
    let host = weights(DType::F32, 2, 4, 1);
    let wv = WeightView::new(DType::F32, 2, 4, &host).unwrap();
    let dw = DeviceWeights::like(dev.upload(&host, None).unwrap(), 0, &wv).unwrap();
    let mut ys = [0.0; 2];
    let e = dev.matmul_many(&mut [DeviceMatMul {
        w: &dw,
        xs: &[1.0; 3],
        s: 1,
        ys: &mut ys,
    }]);
    assert!(matches!(e, Err(ComputeError::Shape { .. })), "{e:?}");
}

/// Timing, not a check: `cargo test --release -p qubivex-backend-vulkan --test parity
/// -- --ignored --nocapture speed`. Prints the per-submission overhead and the time of
/// OLMoE-shaped matmuls (a layer's eight experts, three matrices each, in one
/// submission) for each K-quant.
#[test]
#[ignore = "timing; run by hand"]
fn speed() {
    let Some(dev) = device() else { return };
    let time = |ops: &mut [DeviceMatMul<'_>], reps: u32| {
        dev.matmul_many(ops).unwrap();
        let t = std::time::Instant::now();
        for _ in 0..reps {
            dev.matmul_many(ops).unwrap();
        }
        t.elapsed().as_secs_f64() * 1e3 / f64::from(reps)
    };
    // Overhead: one tiny matmul.
    let host = weights(DType::F32, 1, 64, 1);
    let wv = WeightView::new(DType::F32, 1, 64, &host).unwrap();
    let dw = DeviceWeights::like(dev.upload(&host, None).unwrap(), 0, &wv).unwrap();
    let xs = acts(64, 1);
    let mut y = [0f32];
    let ms = time(
        &mut [DeviceMatMul {
            w: &dw,
            xs: &xs,
            s: 1,
            ys: &mut y,
        }],
        200,
    );
    eprintln!("tiny submission: {ms:.3} ms");
    // One large Q4_K matrix: the kernel's rate without per-dispatch costs.
    let q4k = DType::Ggml(GgmlType::Q4_K);
    let (r, c) = (24 * 1024, 2048);
    let host = weights(q4k, r, c, 3);
    let wv = WeightView::new(q4k, r, c, &host).unwrap();
    let big = DeviceWeights::like(dev.upload(&host, None).unwrap(), 0, &wv).unwrap();
    let xs = acts(c, 1);
    let mut ys = vec![0f32; r];
    let ms = time(
        &mut [DeviceMatMul {
            w: &big,
            xs: &xs,
            s: 1,
            ys: &mut ys,
        }],
        20,
    );
    eprintln!(
        "q4_k one {r} x {c}: {ms:.3} ms ({:.1} GB/s)",
        host.len() as f64 / ms / 1e6
    );
    for dtype in [
        DType::Ggml(GgmlType::Q4_K),
        DType::Ggml(GgmlType::Q6_K),
        DType::Ggml(GgmlType::Q8_0),
    ] {
        // gate and up: 1024 x 2048; down: 2048 x 1024.
        let shapes = [(1024, 2048), (1024, 2048), (2048, 1024)];
        let mut ws = Vec::new();
        for (i, &(r, c)) in shapes.iter().enumerate() {
            let host = weights(dtype, r, c, 7 + i as u32);
            let wv = WeightView::new(dtype, r, c, &host).unwrap();
            ws.push(DeviceWeights::like(dev.upload(&host, None).unwrap(), 0, &wv).unwrap());
        }
        for s in [1usize, 8] {
            let xs: Vec<_> = shapes.iter().map(|&(_, c)| acts(s * c, 3)).collect();
            let mut ys: Vec<_> = shapes.iter().map(|&(r, _)| vec![0f32; s * r]).collect();
            let one = {
                let mut op = [DeviceMatMul {
                    w: &ws[0],
                    xs: &xs[0],
                    s,
                    ys: &mut ys[0],
                }];
                time(&mut op, 20)
            };
            // 24 ops (8 experts x 3): each expert's outputs in their own vectors.
            let mut outs: Vec<Vec<f32>> = (0..24).map(|i| vec![0f32; s * shapes[i % 3].0]).collect();
            let mut layer: Vec<_> = outs
                .iter_mut()
                .enumerate()
                .map(|(i, y)| DeviceMatMul {
                    w: &ws[i % 3],
                    xs: &xs[i % 3],
                    s,
                    ys: y,
                })
                .collect();
            let ms = time(&mut layer, 10);
            let bytes: usize = ws.iter().map(DeviceWeights::len).sum::<usize>() * 8;
            eprintln!(
                "{dtype} s = {s}: one 1024 x 2048 {one:.3} ms; 24 expert matmuls {ms:.3} ms \
                 ({:.1} GB/s of weights)",
                bytes as f64 / ms / 1e6
            );
        }
    }
}
