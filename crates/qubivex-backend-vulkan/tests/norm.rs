//! The RMSNorm scale on the Vulkan device against the CPU, bit for bit (audit §18).
//!
//! `rms.glsl` computes `1 / sqrt(ss / n + eps)` with two divisions and a square root,
//! which must be correctly rounded for the device to give the CPU's bits
//! ([`qubivex_compute::math::rms_scale_of`]). These tests run the shader alone
//! ([`VulkanDevice::rms_scale`]): edge values here, and every f32 input in the ignored
//! exhaustive test.
//!
//! Each test opens the default Vulkan device and skips (passes, with a note) when there
//! is none, unless `QUBIVEX_REQUIRE_VULKAN=1`.
//!
//! Compared with these exceptions, counted and printed, never silent:
//! - subnormals, which the GPU flushes (ADR-011 §1): the input, the CPU's `ss / n`, its
//!   sum with `eps`, or the CPU's output below `f32::MIN_POSITIVE`;
//! - NaN payloads: both NaN, different bits (ADR-011 amendment 7).

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    missing_docs
)]

use qubivex_backend_vulkan::{VulkanConfig, VulkanDevice};
use qubivex_compute::math::rms_scale_of;

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

fn is_sub(v: f32) -> bool {
    v != 0.0 && v.abs() < f32::MIN_POSITIVE
}

#[derive(Default)]
struct Classes {
    equal: u64,
    subnormal: u64,
    nan_payload: u64,
    other: Vec<(u32, u32, u32)>,
}

impl Classes {
    fn add(&mut self, ss: f32, n: f32, eps: f32, got: f32, want: f32) {
        if got.to_bits() == want.to_bits() {
            self.equal += 1;
            return;
        }
        if got.is_nan() && want.is_nan() {
            self.nan_payload += 1;
            return;
        }
        let q = ss / n;
        if is_sub(ss) || is_sub(q) || is_sub(q + eps) || is_sub(want) {
            self.subnormal += 1;
            return;
        }
        if self.other.len() < 1000 {
            self.other
                .push((ss.to_bits(), got.to_bits(), want.to_bits()));
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
             classes; first (ss, gpu, cpu) bits: {:x?}",
            self.other.len(),
            &self.other[..self.other.len().min(8)]
        );
    }
}

fn compare(dev: &VulkanDevice, ss: &[f32], n: f32, eps: f32, c: &mut Classes) {
    let mut y = vec![0f32; ss.len()];
    dev.rms_scale(ss, n, eps, &mut y).unwrap();
    for (&a, &got) in ss.iter().zip(&y) {
        c.add(a, n, eps, got, rms_scale_of(a, n, eps));
    }
}

/// The (n, eps) pairs checked: models' hidden sizes and head sizes with their usual
/// epsilons, and (1, 0), which makes the scale `1 / sqrt(ss)` over every input.
const SHAPES: [(f32, f32); 6] = [
    (1.0, 0.0),
    (2048.0, 1e-5),
    (4096.0, 1e-6),
    (128.0, 1e-6),
    (896.0, 1e-6),
    (7.0, 1e-5),
];

#[test]
fn rms_scale_edges() {
    let Some(dev) = device() else { return };
    let mut ss = vec![
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -1.0,
        1.0,
        2.0,
        4.0,
        0.25,
        1e-30,
        1e30,
    ];
    // Perfect squares and their neighbours, where a square root rounds near a tie.
    for i in 1..4096u32 {
        let v = (i * i) as f32;
        ss.extend([
            v,
            f32::from_bits(v.to_bits() - 1),
            f32::from_bits(v.to_bits() + 1),
        ]);
    }
    let mut r = Rng(0x2545_f491);
    ss.extend((0..1 << 20).map(|_| f32::from_bits(r.next() & 0x7fff_ffff)));
    for (n, eps) in SHAPES {
        let mut c = Classes::default();
        compare(&dev, &ss, n, eps, &mut c);
        c.check(&format!("rms scale edges, n = {n}, eps = {eps:e}"));
    }
}

/// Every f32 bit pattern of `ss` for each (n, eps) in [`SHAPES`]: 6 × 2^32 inputs.
/// `cargo test --release -p qubivex-backend-vulkan --test norm -- --ignored`.
#[test]
#[ignore = "exhaustive: 6 x 2^32 inputs"]
fn rms_scale_matches_the_cpu_for_every_input() {
    let Some(dev) = device() else { return };
    const CHUNK: u64 = 1 << 22;
    for (n, eps) in SHAPES {
        let mut c = Classes::default();
        let mut start = 0u64;
        while start < 1 << 32 {
            let ss: Vec<f32> = (start..start + CHUNK)
                .map(|b| f32::from_bits(b as u32))
                .collect();
            compare(&dev, &ss, n, eps, &mut c);
            start += CHUNK;
        }
        c.check(&format!("rms scale, every ss, n = {n}, eps = {eps:e}"));
    }
}
