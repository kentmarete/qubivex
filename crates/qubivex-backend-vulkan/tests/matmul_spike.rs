//! Kernel-throughput spike (audit §23; feature `spike`): variants of the production
//! matmul's Q8_0, Q4_K and Q6_K bodies (`spike/mm.comp`) against the CPU backend, bit for
//! bit, and an ignored test that times them on the GPU. No production path uses them.
//!
//! `cargo test --release -p qubivex-backend-vulkan --features spike --test matmul_spike
//! -- --nocapture` (and `--ignored` for the timings).

#![cfg(feature = "spike")]
#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    missing_docs
)]

use qubivex_backend_cpu::CpuBackend;
use qubivex_backend_vulkan::{SpikeKernel, VulkanConfig, VulkanDevice};
use qubivex_compute::{ComputeBackend, WeightView};
use qubivex_core::{DType, GgmlType, Shape};

/// A variant: name, SPIR-V, rows per 32-lane group, invocations per workgroup, and
/// whether it is meant to give the CPU's bits (the `_nd` timing controls are not).
struct Variant {
    name: &'static str,
    spv: &'static [u8],
    nr: u32,
    wg: u32,
    exact: bool,
}

macro_rules! variants {
    ($dt:literal: $(($spec:literal, $nr:literal, $wg:literal, $exact:literal)),* $(,)?) => {
        [$(Variant {
            name: $spec,
            spv: include_bytes!(concat!("../spike/mm_", $dt, "_", $spec, ".spv")),
            nr: $nr,
            wg: $wg,
            exact: $exact,
        }),*]
    };
}

macro_rules! all_variants {
    ($dt:literal) => {
        variants!($dt:
            ("t0_r1_w256", 1, 256, true),
            ("t0_r2_w256", 2, 256, true),
            ("t1_r1_w256", 1, 256, true),
            ("t1_r2_w256", 2, 256, true),
            ("t1_r4_w256", 4, 256, true),
            ("t1_r1_w64", 1, 64, true),
            ("t1_r2_w128", 2, 128, true),
            ("t0_r1_w256_nd", 1, 256, false),
            ("t1_r2_w256_nd", 2, 256, false),
        )
    };
}

const Q8: [Variant; 9] = all_variants!("q8");
const Q4K: [Variant; 9] = all_variants!("q4k");
const Q6K: [Variant; 9] = all_variants!("q6k");
/// Q4_K decode ablations: scales from fixed bytes and the quant byte not loaded (timing
/// only), and no `opaque` (exact on MoltenVK with fast math off, ADR-019).
const Q4K_ABL: [Variant; 3] = variants!("q4k":
    ("t1_r2_w256_ascale", 2, 256, false),
    ("t1_r2_w256_abyte", 2, 256, false),
    ("t1_r2_w256_aopaque", 2, 256, true),
);
/// Q4_K rewrites: the four groups written out (`u`), and the quant bytes loaded one
/// word per lane and handed out by subgroup shuffle (`q`), and the block's scale and min
/// products computed once by lanes 0..7 and shared by shuffle (`s`) or through shared
/// memory (`sh`).
const Q4K_V2: [Variant; 12] = variants!("q4k":
    ("t0_r1_w256_ush", 1, 256, true),
    ("t0_r2_w256_ush", 2, 256, true),
    ("t0_r4_w256_ush", 4, 256, true),
    ("t1_r2_w256_ush", 2, 256, true),
    ("t1_r2_w256_us", 2, 256, true),
    ("t1_r2_w256_uqs", 2, 256, true),
    ("t1_r4_w256_uqs", 4, 256, true),
    ("t1_r1_w256_uqs", 1, 256, true),
    ("t1_r2_w256_u", 2, 256, true),
    ("t1_r2_w256_uq", 2, 256, true),
    ("t1_r1_w256_uq", 1, 256, true),
    ("t1_r4_w256_uq", 4, 256, true),
);

/// Q6_K rewrites: written out (`u`), and the block's 16 scale products computed once and
/// shared by shuffle (`s`) or through shared memory (`sh`).
const Q6K_V2: [Variant; 8] = variants!("q6k":
    ("t0_r1_w256_ush", 1, 256, true),
    ("t0_r2_w256_ush", 2, 256, true),
    ("t0_r4_w256_ush", 4, 256, true),
    ("t1_r2_w256_ush", 2, 256, true),
    ("t1_r2_w256_u", 2, 256, true),
    ("t1_r2_w256_us", 2, 256, true),
    ("t1_r4_w256_us", 4, 256, true),
    ("t1_r1_w256_us", 1, 256, true),
);

fn formats() -> [(&'static str, DType, Vec<&'static Variant>); 3] {
    [
        ("Q8_0", DType::Ggml(GgmlType::Q8_0), Q8.iter().collect()),
        (
            "Q4_K",
            DType::Ggml(GgmlType::Q4_K),
            Q4K.iter().chain(&Q4K_ABL).chain(&Q4K_V2).collect(),
        ),
        (
            "Q6_K",
            DType::Ggml(GgmlType::Q6_K),
            Q6K.iter().chain(&Q6K_V2).collect(),
        ),
    ]
}

/// `QUBIVEX_SPIKE_ONLY=<a>,<b>,...` limits the timings to variants whose
/// "<dtype> <variant>" ends with one of them, and `QUBIVEX_SPIKE_BIG=1` to the large matrix.
fn wanted(fname: &str, v: &Variant) -> bool {
    std::env::var("QUBIVEX_SPIKE_ONLY")
        .map(|f| {
            let name = format!("{fname} {}", v.name);
            f.split(',').any(|f| name.ends_with(f))
        })
        .unwrap_or(true)
}

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}

/// As `parity.rs`: block formats with small finite f16 scales.
fn weights(dtype: DType, rows: usize, cols: usize, seed: u32) -> Vec<u8> {
    let n = dtype
        .storage_bytes(&Shape::new(&[rows as u64, cols as u64]).unwrap())
        .unwrap() as usize;
    let mut r = Rng(seed.wrapping_mul(2_654_435_761).max(1));
    let mut b: Vec<u8> = (0..n).map(|_| (r.next() >> 24) as u8).collect();
    for v in b.iter_mut().skip(1).step_by(2) {
        *v &= 0x3b;
    }
    b
}

fn acts(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed) * 37 % 101) as f32 - 50.0) / 25.0)
        .collect()
}

fn cpu(dtype: DType, rows: usize, cols: usize, host: &[u8], xs: &[f32], s: usize) -> Vec<f32> {
    let wv = WeightView::new(dtype, rows, cols, host).unwrap();
    let mut ys = vec![0.0; s * rows];
    CpuBackend::new().matmul(&wv, xs, s, &mut ys).unwrap();
    ys
}

/// The weight bytes as f32 bit patterns (the spike's buffers are f32), padded to words.
fn as_words(host: &[u8]) -> Vec<f32> {
    host.chunks(4)
        .map(|c| {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            f32::from_bits(u32::from_le_bytes(b))
        })
        .collect()
}

fn open() -> Option<VulkanDevice> {
    match VulkanDevice::new(VulkanConfig {
        timestamps: true,
        ..VulkanConfig::default()
    }) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("Vulkan device unavailable, skipped: {e}");
            None
        }
    }
}

/// Runs `k` `repeat` times in one submission: the outputs, and GPU µs per run.
#[allow(clippy::too_many_arguments)]
fn run(
    dev: &VulkanDevice,
    k: &SpikeKernel,
    v: &Variant,
    w: &[f32],
    row_bytes: usize,
    (rows, cols): (usize, usize),
    xs: &[f32],
    s: usize,
    repeat: u32,
) -> (Vec<f32>, f64) {
    let mut w = w.to_vec();
    let mut x = xs.to_vec();
    let mut ys = vec![0f32; s * rows];
    let mut none = [0f32; 4];
    let per = (v.wg / 32 * v.nr) as usize;
    let push = [rows as u32, cols as u32, row_bytes as u32, 0];
    let t = dev
        .spike_run(
            &[(k, push, [rows.div_ceil(per) as u32, s as u32, 1])],
            [&mut w, &mut x, &mut ys, &mut none],
            repeat,
        )
        .unwrap();
    let us = t.gpu_ns.unwrap_or(t.wait_ns) as f64 / 1e3 / f64::from(repeat);
    (ys, us)
}

fn row_bytes(dtype: DType, cols: usize) -> usize {
    dtype
        .storage_bytes(&Shape::new(&[1, cols as u64]).unwrap())
        .unwrap() as usize
}

#[test]
fn every_exact_variant_gives_the_cpus_bits() {
    let Some(dev) = open() else { return };
    for (fname, dtype, variants) in formats() {
        let cols: &[usize] = if matches!(dtype, DType::Ggml(GgmlType::Q8_0)) {
            &[64, 576, 2048]
        } else {
            &[256, 1024, 2048]
        };
        for v in variants.into_iter().filter(|v| v.exact) {
            let k = dev.spike_kernel(v.spv).unwrap();
            for &c in cols {
                for (rows, s) in [(1, 1), (9, 3), (33, 8), (1027, 2)] {
                    let host = weights(dtype, rows, c, (rows * 7 + c) as u32);
                    let xs = acts(s * c, rows + c);
                    let want = cpu(dtype, rows, c, &host, &xs, s);
                    let (got, _) = run(
                        &dev,
                        &k,
                        v,
                        &as_words(&host),
                        row_bytes(dtype, c),
                        (rows, c),
                        &xs,
                        s,
                        1,
                    );
                    let bad = (0..got.len())
                        .filter(|&i| got[i].to_bits() != want[i].to_bits())
                        .count();
                    assert_eq!(
                        bad,
                        0,
                        "{fname} {} rows {rows} cols {c} s {s}: {bad} of {} differ; first {:?}",
                        v.name,
                        got.len(),
                        (0..got.len())
                            .find(|&i| got[i].to_bits() != want[i].to_bits())
                            .map(|i| (i, got[i], want[i]))
                    );
                }
            }
        }
    }
}

/// Positive control: the test above catches a one-ulp change in one output.
#[test]
fn a_changed_bit_is_caught() {
    let Some(dev) = open() else { return };
    let dtype = DType::Ggml(GgmlType::Q4_K);
    let v = &Q4K[2];
    let k = dev.spike_kernel(v.spv).unwrap();
    let (rows, c, s) = (9, 512, 3);
    let host = weights(dtype, rows, c, 5);
    let xs = acts(s * c, 1);
    let want = cpu(dtype, rows, c, &host, &xs, s);
    let (mut got, _) = run(
        &dev,
        &k,
        v,
        &as_words(&host),
        row_bytes(dtype, c),
        (rows, c),
        &xs,
        s,
        1,
    );
    assert!(
        got.iter()
            .zip(&want)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    got[4] = f32::from_bits(got[4].to_bits() ^ 1);
    assert_eq!(
        got.iter()
            .zip(&want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count(),
        1
    );
    // And the `_nd` control is not exact, so the comparison can fail on real kernels.
    let nd = Q4K.iter().find(|v| !v.exact).unwrap();
    let knd = dev.spike_kernel(nd.spv).unwrap();
    let (got, _) = run(
        &dev,
        &knd,
        nd,
        &as_words(&host),
        row_bytes(dtype, c),
        (rows, c),
        &xs,
        s,
        1,
    );
    assert!(
        got.iter()
            .zip(&want)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    );
}

#[test]
#[ignore = "timing; run by hand"]
fn variant_rates() {
    let Some(dev) = open() else { return };
    assert!(dev.timestamps(), "the queue has no timestamps");
    // The large matrix of `decode_profile.rs`, and OLMoE's expert shapes (gate/up
    // 1024 x 2048, down 2048 x 1024) and Q/K/V/O (2048 x 2048).
    let mut shapes = vec![
        (24576usize, 2048usize),
        (1024, 2048),
        (2048, 1024),
        (2048, 2048),
    ];
    if std::env::var("QUBIVEX_SPIKE_BIG").as_deref() == Ok("1") {
        shapes.truncate(1);
    }
    eprintln!("| dtype | variant | rows x cols | s | GPU µs | GB/s | µs per row-token x1000 |");
    eprintln!("|---|---|---|---:|---:|---:|---:|");
    for (fname, dtype, variants) in formats() {
        for &(rows, cols) in &shapes {
            let host = weights(dtype, rows, cols, 3);
            let words = as_words(&host);
            let rb = row_bytes(dtype, cols);
            let mb = host.len() as f64 / 1e6;
            for v in variants.iter().filter(|v| wanted(fname, v)) {
                let k = dev.spike_kernel(v.spv).unwrap();
                for s in [1usize, 8] {
                    let xs = acts(s * cols, 0);
                    let repeat = if rows >= 16384 { 10 } else { 50 };
                    // Warm once, then the median of 5 submissions.
                    let _ = run(&dev, &k, v, &words, rb, (rows, cols), &xs, s, repeat);
                    let mut us: Vec<f64> = (0..5)
                        .map(|_| run(&dev, &k, v, &words, rb, (rows, cols), &xs, s, repeat).1)
                        .collect();
                    us.sort_by(f64::total_cmp);
                    let m = us[2];
                    eprintln!(
                        "| {fname} | {} | {rows} x {cols} | {s} | {m:.1} | {:.1} | {:.2} |",
                        v.name,
                        mb * 1e3 / m,
                        m * 1e3 / (rows * s) as f64
                    );
                }
            }
        }
    }
}
