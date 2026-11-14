//! Where a decode token's GPU time goes (audit §22): the matmul kernel's GPU time per
//! dtype and row count on one large matrix, and the LM head's shape. Timing, not a check.
//!
//! `cargo test --release -p qubivex-backend-vulkan --test decode_profile -- --ignored
//! --nocapture`

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    missing_docs
)]

use qubivex_backend_vulkan::{VulkanConfig, VulkanDevice};
use qubivex_compute::WeightView;
use qubivex_compute::discrete::{DeviceMatMul, DeviceWeights, DiscreteDevice};
use qubivex_core::{DType, GgmlType, Shape};

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}

/// As `parity.rs`: normal f32 values; block formats with small finite f16 scales.
fn weights(dtype: DType, rows: usize, cols: usize, seed: u32) -> Vec<u8> {
    let n = dtype
        .storage_bytes(&Shape::new(&[rows as u64, cols as u64]).unwrap())
        .unwrap() as usize;
    let mut r = Rng(seed.wrapping_mul(2_654_435_761).max(1));
    match dtype {
        DType::F32 => (0..n / 4)
            .flat_map(|_| {
                let v = r.next();
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
        _ => {
            let mut b: Vec<u8> = (0..n).map(|_| (r.next() >> 24) as u8).collect();
            for v in b.iter_mut().skip(1).step_by(2) {
                *v &= 0x3b;
            }
            b
        }
    }
}

fn acts(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect()
}

/// Median GPU and wall microseconds of `reps` single-op submissions.
fn time(
    dev: &VulkanDevice,
    w: &DeviceWeights,
    (rows, cols): (usize, usize),
    s: usize,
    reps: usize,
) -> (f64, f64) {
    let xs = acts(s * cols);
    let mut ys = vec![0f32; s * rows];
    let mut gpu = Vec::new();
    let mut wall = Vec::new();
    for i in 0..reps + 3 {
        let before = dev.transfers();
        let t = std::time::Instant::now();
        dev.matmul_many(&mut [DeviceMatMul {
            w,
            xs: &xs,
            s,
            ys: &mut ys,
        }])
        .unwrap();
        let el = t.elapsed().as_secs_f64() * 1e6;
        let after = dev.transfers();
        if i >= 3 {
            gpu.push((after.gpu_ns - before.gpu_ns) as f64 / 1e3);
            wall.push(el);
        }
    }
    gpu.sort_by(f64::total_cmp);
    wall.sort_by(f64::total_cmp);
    (gpu[gpu.len() / 2], wall[wall.len() / 2])
}

#[test]
#[ignore = "timing; run by hand"]
fn kernel_rates() {
    let dev = match VulkanDevice::new(VulkanConfig {
        timestamps: true,
        ..VulkanConfig::default()
    }) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Vulkan device unavailable, skipped: {e}");
            return;
        }
    };
    assert!(dev.timestamps(), "the queue has no timestamps");
    let cols = 2048;
    let shapes: [(&str, DType, usize); 6] = [
        ("F32", DType::F32, 8192),
        ("F16", DType::F16, 16384),
        ("Q8_0", DType::Ggml(GgmlType::Q8_0), 24576),
        ("Q4_K", DType::Ggml(GgmlType::Q4_K), 24576),
        ("Q6_K", DType::Ggml(GgmlType::Q6_K), 24576),
        ("Q6_K head", DType::Ggml(GgmlType::Q6_K), 50304),
    ];
    eprintln!("| matrix | rows x cols | MB | s | GPU µs | wall µs | GB/s (GPU) | GPU µs per row |");
    eprintln!("|---|---|---:|---:|---:|---:|---:|---:|");
    for (name, dtype, rows) in shapes {
        let host = weights(dtype, rows, cols, 3);
        let wv = WeightView::new(dtype, rows, cols, &host).unwrap();
        let w = DeviceWeights::like(dev.upload(&host, None).unwrap(), 0, &wv).unwrap();
        let mb = host.len() as f64 / 1e6;
        for s in [1usize, 2, 4, 8] {
            let (gpu, wall) = time(&dev, &w, (rows, cols), s, 15);
            eprintln!(
                "| {name} | {rows} x {cols} | {mb:.1} | {s} | {gpu:.0} | {wall:.0} | {:.1} | {:.0} |",
                mb * 1e3 / gpu,
                gpu / s as f64
            );
        }
    }
}
