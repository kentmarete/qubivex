//! The fused expert block and SwiGLU on the Vulkan device against the CPU, bit for bit
//! (audit §15, Step 3a).
//!
//! Each test opens the default Vulkan device and skips (passes, with a note) when there
//! is none, unless `QUBIVEX_REQUIRE_VULKAN=1`.
//!
//! SwiGLU is compared with these exceptions, counted and printed, never silent:
//! - subnormals, which the GPU flushes (ADR-011 §1): an input, the CPU's intermediate
//!   `silu(g)`, the exact `silu(g)` or the CPU's output below `f32::MIN_POSITIVE`;
//! - NaN payloads: both NaN, different bits. Whether the parity contract counts NaN
//!   payloads is an open owner decision (audit §15.1); NaN-ness itself must match.

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    missing_docs
)]

use qubivex_backend_cpu::CpuBackend;
use qubivex_backend_vulkan::{VulkanConfig, VulkanDevice};
use qubivex_compute::discrete::{DeviceBuffer, DeviceGatedFfn, DeviceWeights, DiscreteDevice};
use qubivex_compute::math::swiglu;
use qubivex_compute::{GatedFfn, WeightView, gated_ffn_steps};
use qubivex_core::{DType, GgmlType, Shape};

fn device() -> Option<VulkanDevice> {
    match VulkanDevice::new(VulkanConfig::default()) {
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

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}

fn bytes_of(dtype: DType, rows: usize, cols: usize) -> usize {
    dtype
        .storage_bytes(&Shape::new(&[rows as u64, cols as u64]).unwrap())
        .unwrap() as usize
}

/// Deterministic weights of moderate, normal values (as in `parity.rs`).
fn weights(dtype: DType, rows: usize, cols: usize, seed: u32) -> Vec<u8> {
    let n = bytes_of(dtype, rows, cols);
    let mut r = Rng(seed.wrapping_mul(2_654_435_761).max(1));
    match dtype {
        DType::F32 => (0..n / 4)
            .flat_map(|_| {
                let v = r.next();
                // Small magnitudes keep SwiGLU's inputs in its interesting range.
                let e = 112 + (v >> 24) % 12;
                ((v & 0x807F_FFFF) | (e << 23)).to_le_bytes()
            })
            .collect(),
        DType::F16 => (0..n / 2)
            .flat_map(|_| {
                let v = r.next() as u16;
                let e = 4 + (v >> 10) % 10;
                ((v & 0x83FF) | (e << 10)).to_le_bytes()
            })
            .collect(),
        _ => {
            let mut b: Vec<u8> = (0..n).map(|_| (r.next() >> 24) as u8).collect();
            for v in b.iter_mut().skip(1).step_by(2) {
                *v &= 0x2b;
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

/// One expert's three matrices, as host bytes.
struct Expert {
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
}

/// Shapes of one test: `h` model width, `ffn` expert width, dtypes of gate/up and down.
#[derive(Clone, Copy, Debug)]
struct Shapes {
    h: usize,
    ffn: usize,
    gu: DType,
    down: DType,
}

fn experts(sh: Shapes, n: usize, seed: u32) -> Vec<Expert> {
    (0..n as u32)
        .map(|e| Expert {
            gate: weights(sh.gu, sh.ffn, sh.h, seed + 3 * e),
            up: weights(sh.gu, sh.ffn, sh.h, seed + 3 * e + 1),
            down: weights(sh.down, sh.h, sh.ffn, seed + 3 * e + 2),
        })
        .collect()
}

/// The CPU's outputs: separate matmuls, `math::swiglu`, matmul.
fn cpu_block(sh: Shapes, ex: &[Expert], xs: &[Vec<f32>], rows: &[usize]) -> Vec<Vec<f32>> {
    let mut ys: Vec<Vec<f32>> = rows.iter().map(|&s| vec![0.0; s * sh.h]).collect();
    let mut ops: Vec<GatedFfn<'_>> = ex
        .iter()
        .zip(xs)
        .zip(rows)
        .zip(ys.iter_mut())
        .map(|(((e, x), &s), y)| GatedFfn {
            gate: WeightView::new(sh.gu, sh.ffn, sh.h, &e.gate).unwrap(),
            up: WeightView::new(sh.gu, sh.ffn, sh.h, &e.up).unwrap(),
            down: WeightView::new(sh.down, sh.h, sh.ffn, &e.down).unwrap(),
            xs: x,
            s,
            ys: y,
        })
        .collect();
    gated_ffn_steps(&CpuBackend::new(), &mut ops).unwrap();
    drop(ops);
    ys
}

/// Where the experts live on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Home {
    /// One arena, each expert in its own slot (the device expert cache's layout).
    Arena,
    /// Each expert its own upload: a dispatch per buffer.
    Own,
}

/// The device's copies: per expert, its three matrices.
fn place(dev: &VulkanDevice, sh: Shapes, ex: &[Expert], home: Home) -> Vec<[DeviceWeights; 3]> {
    let lens = [
        bytes_of(sh.gu, sh.ffn, sh.h),
        bytes_of(sh.gu, sh.ffn, sh.h),
        bytes_of(sh.down, sh.h, sh.ffn),
    ];
    let slot = (lens.iter().sum::<usize>()).next_multiple_of(256);
    let arena = (home == Home::Arena).then(|| dev.alloc(slot * ex.len().max(1), None).unwrap());
    ex.iter()
        .enumerate()
        .map(|(i, e)| {
            let host: Vec<u8> = [&e.gate[..], &e.up[..], &e.down[..]].concat();
            let (buf, base): (DeviceBuffer, usize) = match &arena {
                Some(a) => {
                    dev.write(a, i * slot, &host).unwrap();
                    (a.clone(), i * slot)
                }
                None => (dev.upload(&host, None).unwrap(), 0),
            };
            [
                DeviceWeights::new(buf.clone(), base, sh.gu, sh.ffn, sh.h).unwrap(),
                DeviceWeights::new(buf.clone(), base + lens[0], sh.gu, sh.ffn, sh.h).unwrap(),
                DeviceWeights::new(buf, base + lens[0] + lens[1], sh.down, sh.h, sh.ffn).unwrap(),
            ]
        })
        .collect()
}

fn gpu_block(
    dev: &VulkanDevice,
    sh: Shapes,
    ws: &[[DeviceWeights; 3]],
    xs: &[Vec<f32>],
    rows: &[usize],
) -> Vec<Vec<f32>> {
    let mut ys: Vec<Vec<f32>> = rows.iter().map(|&s| vec![0.0; s * sh.h]).collect();
    let mut ops: Vec<DeviceGatedFfn<'_>> = ws
        .iter()
        .zip(xs)
        .zip(rows)
        .zip(ys.iter_mut())
        .map(|(((w, x), &s), y)| DeviceGatedFfn {
            gate: &w[0],
            up: &w[1],
            down: &w[2],
            xs: x,
            s,
            ys: y,
        })
        .collect();
    dev.gated_ffn_many(&mut ops).unwrap();
    drop(ops);
    ys
}

/// Every dtype pairing the models use (OLMoE Q4_K_M: Q4_K gate/up with Q4_K or Q6_K
/// down), and the per-element formats, at 1, 3 and 8 experts × 1, 2, 4 and 10 rows,
/// in an arena and in separate buffers: one submission, the CPU's bits.
#[test]
fn the_fused_block_matches_the_cpu() {
    let Some(dev) = device() else { return };
    let q4 = DType::Ggml(GgmlType::Q4_K);
    let q6 = DType::Ggml(GgmlType::Q6_K);
    let cases = [
        Shapes {
            h: 512,
            ffn: 256,
            gu: q4,
            down: q4,
        },
        Shapes {
            h: 512,
            ffn: 256,
            gu: q4,
            down: q6,
        },
        Shapes {
            h: 256,
            ffn: 512,
            gu: q6,
            down: q6,
        },
        Shapes {
            h: 64,
            ffn: 96,
            gu: DType::Ggml(GgmlType::Q8_0),
            down: DType::Ggml(GgmlType::Q8_0),
        },
        Shapes {
            h: 65,
            ffn: 130,
            gu: DType::F32,
            down: DType::F16,
        },
        Shapes {
            h: 63,
            ffn: 7,
            gu: DType::BF16,
            down: DType::F32,
        },
    ];
    for (c, &sh) in cases.iter().enumerate() {
        for n in [1, 3, 8] {
            let ex = experts(sh, n, 11 + c as u32);
            for home in [Home::Arena, Home::Own] {
                let ws = place(&dev, sh, &ex, home);
                for s in [1, 2, 4, 10] {
                    // Mixed rows per expert, as routing gives: s, s-1, ... (at least 1).
                    let rows: Vec<usize> = (0..n).map(|i| (s - i.min(s - 1)).max(1)).collect();
                    let xs: Vec<Vec<f32>> = rows
                        .iter()
                        .enumerate()
                        .map(|(i, &r)| acts(r * sh.h, i * 5 + s))
                        .collect();
                    let want = cpu_block(sh, &ex, &xs, &rows);
                    let before = dev.transfers().submissions;
                    let got = gpu_block(&dev, sh, &ws, &xs, &rows);
                    assert_eq!(dev.transfers().submissions - before, 1, "one submission");
                    for (e, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert_bits(
                            g,
                            w,
                            &format!("{sh:?} {home:?} {n} experts, {s} rows, expert {e}"),
                        );
                    }
                }
            }
        }
    }
}

/// OLMoE's real expert shapes (2048 wide, 1024 per expert), Q4_K gate/up with a Q6_K
/// down, 8 experts in an arena: the decode case Step 3a fuses, and the Q6_K layers.
#[test]
fn olmoe_shapes_with_a_q6_k_down_match_the_cpu() {
    let Some(dev) = device() else { return };
    let q4 = DType::Ggml(GgmlType::Q4_K);
    for down in [q4, DType::Ggml(GgmlType::Q6_K)] {
        let sh = Shapes {
            h: 2048,
            ffn: 1024,
            gu: q4,
            down,
        };
        let ex = experts(sh, 8, 101);
        let ws = place(&dev, sh, &ex, Home::Arena);
        for s in [1, 4] {
            let rows = vec![s; 8];
            let xs: Vec<Vec<f32>> = (0..8).map(|i| acts(s * sh.h, i)).collect();
            let want = cpu_block(sh, &ex, &xs, &rows);
            let got = gpu_block(&dev, sh, &ws, &xs, &rows);
            for (e, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_bits(g, w, &format!("{down} down, {s} rows, expert {e}"));
            }
        }
    }
}

/// An expert with no rows, and a shared input slice, in one block.
#[test]
fn empty_blocks_and_shared_inputs() {
    let Some(dev) = device() else { return };
    let sh = Shapes {
        h: 256,
        ffn: 256,
        gu: DType::Ggml(GgmlType::Q4_K),
        down: DType::Ggml(GgmlType::Q6_K),
    };
    let ex = experts(sh, 3, 7);
    let ws = place(&dev, sh, &ex, Home::Arena);
    let x = acts(2 * sh.h, 3);
    let xs = vec![x.clone(), Vec::new(), x];
    let rows = [2, 0, 2];
    let want = cpu_block(sh, &ex, &xs, &rows);
    let got = gpu_block(&dev, sh, &ws, &xs, &rows);
    for (g, w) in got.iter().zip(&want) {
        assert_bits(g, w, "empty and shared");
    }
    // Nothing at all: one (empty) submission, no error.
    let none = gpu_block(&dev, sh, &ws[..1], &[Vec::new()], &[0]);
    assert!(none[0].is_empty());
}

/// Arena writes outside the arena, or into another device's buffer, are refused.
#[test]
fn arena_writes_are_checked() {
    let Some(dev) = device() else { return };
    let a = dev.alloc(1024, None).unwrap();
    dev.write(&a, 1000, &[1; 24]).unwrap();
    assert!(dev.write(&a, 1001, &[1; 24]).is_err());
    assert!(dev.write(&a, usize::MAX, &[1]).is_err());
    let Some(other) = device() else { return };
    assert!(other.write(&a, 0, &[1; 4]).is_err());
    let used = dev.usage().used;
    drop(a);
    assert_eq!(dev.usage().used, used - 1024, "freed on drop");
}

/// How one SwiGLU output compares with the CPU's.
#[derive(Debug, Default)]
struct Classes {
    equal: u64,
    subnormal: u64,
    nan_payload: u64,
    other: Vec<(u32, u32, u32, u32)>,
}

fn is_sub(v: f32) -> bool {
    v != 0.0 && v.abs() < f32::MIN_POSITIVE
}

impl Classes {
    fn add(&mut self, g: f32, u: f32, got: f32, want: f32) {
        if got.to_bits() == want.to_bits() {
            self.equal += 1;
            return;
        }
        if got.is_nan() && want.is_nan() {
            self.nan_payload += 1;
            return;
        }
        // The CPU's intermediate, and the exact one (in f64, from the CPU's
        // denominator, so only the final division's rounding is left out).
        let d = 1.0 + qubivex_compute::math::exp(-g);
        let s = g / d;
        let exact = f64::from(g) / f64::from(d);
        if is_sub(g)
            || is_sub(u)
            || is_sub(s)
            || is_sub(want)
            || (exact != 0.0 && exact.abs() < f64::from(f32::MIN_POSITIVE))
        {
            self.subnormal += 1;
            return;
        }
        if self.other.len() < 1000 {
            self.other
                .push((g.to_bits(), u.to_bits(), got.to_bits(), want.to_bits()));
        } else {
            self.other.push((0, 0, 0, 0));
        }
    }

    #[track_caller]
    fn check(&self, what: &str) {
        eprintln!(
            "{what}: {} equal, {} subnormal (ADR-011), {} NaN payload, {} other",
            self.equal,
            self.subnormal,
            self.nan_payload,
            self.other.len()
        );
        assert!(
            self.other.is_empty(),
            "{what}: {} outputs differ from the CPU outside the subnormal and NaN-payload \
             classes; first (g, u, gpu, cpu) bits: {:x?}",
            self.other.len(),
            &self.other[..self.other.len().min(8)]
        );
    }
}

/// Runs `g` and `u` through the device's SwiGLU and the CPU's, and classifies.
fn compare(dev: &VulkanDevice, g: &[f32], u: &[f32], c: &mut Classes) {
    let mut got = g.to_vec();
    dev.swiglu(&mut got, u).unwrap();
    let mut want = g.to_vec();
    swiglu(&mut want, u);
    for i in 0..g.len() {
        c.add(g[i], u[i], got[i], want[i]);
    }
}

/// SwiGLU at the inputs where it can go wrong: around each rounding boundary of exp's
/// `k = round(x·log2 e)`, the overflow and underflow edges, ±0, the subnormal and
/// `f32::MIN_POSITIVE` boundaries, infinities and NaNs; plus 4 million pseudo-random
/// bit patterns. Built -O, the shader got ordinary values wrong (audit §15.1).
#[test]
fn swiglu_matches_the_cpu_at_its_edges() {
    let Some(dev) = device() else { return };
    let mut g: Vec<f32> = Vec::new();
    let near = |v: f32, out: &mut Vec<f32>| {
        for d in -16i32..=16 {
            let b = v.to_bits() as i64 + i64::from(d);
            if (0..=i64::from(u32::MAX)).contains(&b) {
                out.push(f32::from_bits(b as u32));
            }
        }
    };
    // k ± 0.5 over the whole exp range, both signs (silu evaluates exp(-g)).
    for k in -130i32..=130 {
        let x = (k as f32 + 0.5) / std::f32::consts::LOG2_E;
        near(x, &mut g);
        near(-x, &mut g);
    }
    for v in [
        0.0f32,
        -0.0,
        88.722_84,
        -88.722_84,
        87.336_55,
        -87.336_55,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        2.0 * f32::MIN_POSITIVE,
        -2.0 * f32::MIN_POSITIVE,
        f32::from_bits(0x00ff_ffff),
        f32::from_bits(0x80ff_ffff),
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0,
        -1.0,
        8.94e-8,
    ] {
        near(v, &mut g);
    }
    g.push(f32::NAN);
    g.push(f32::from_bits(0x7fc0_1234));
    g.push(f32::from_bits(0xffa0_0001));
    let mut r = Rng(0x9e37_79b9);
    g.extend((0..1 << 22).map(|_| f32::from_bits(r.next())));
    let mut c = Classes::default();
    let ones = vec![1.0f32; g.len()];
    compare(&dev, &g, &ones, &mut c);
    let u: Vec<f32> = (0..g.len()).map(|_| f32::from_bits(r.next())).collect();
    compare(&dev, &g, &u, &mut c);
    c.check("swiglu edges");
}

/// Every f32 bit pattern of g, with u = 1 and with pseudo-random u: about 3 minutes.
/// `cargo test --release -p qubivex-backend-vulkan --test fused -- --ignored`.
#[test]
#[ignore = "exhaustive: 2 x 2^32 inputs"]
fn swiglu_matches_the_cpu_for_every_input() {
    let Some(dev) = device() else { return };
    const CHUNK: u64 = 1 << 22;
    let mut c = Classes::default();
    let mut r = Rng(0x1234_5678);
    for pass in 0..2 {
        let mut start = 0u64;
        while start < 1 << 32 {
            let g: Vec<f32> = (start..start + CHUNK)
                .map(|b| f32::from_bits(b as u32))
                .collect();
            let u: Vec<f32> = if pass == 0 {
                vec![1.0; g.len()]
            } else {
                (0..g.len()).map(|_| f32::from_bits(r.next())).collect()
            };
            compare(&dev, &g, &u, &mut c);
            start += CHUNK;
        }
    }
    c.check("swiglu, every g");
}
