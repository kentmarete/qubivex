//! The GPU attention investigation harness (audit §19; feature `spike`). Experimental
//! kernels from `spike/` against the CPU reference in `qubivex_arch::ops`, bit for bit:
//! exact attention (scores in a device scratch buffer, and in shared memory) and RoPE.
//! An online-softmax (tiled) attention is simulated on the CPU to measure how far it is
//! from the reference. An ignored test times the kernels and the CPU.
//!
//! `cargo test --release -p qubivex-backend-vulkan --features spike --test attention_spike
//! -- --nocapture` (and `--ignored` for the timings).
//!
//! Exceptions, as the parity contract states them (ADR-011 amendment 7): NaN payloads.

#![cfg(feature = "spike")]
#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_arguments,
    missing_docs
)]

use qubivex_arch::ops::{Rope, attend};
use qubivex_backend_vulkan::{SpikeKernel, VulkanConfig, VulkanDevice};
use std::time::Instant;

const ATTN: &[u8] = include_bytes!("../spike/attn_exact.spv");
const ATTN_SH: &[u8] = include_bytes!("../spike/attn_exact_sh.spv");
const ROPE: &[u8] = include_bytes!("../spike/rope.spv");
const SCORES_T: &[u8] = include_bytes!("../spike/attn_scores_t.spv");
const SOFTMAX: &[u8] = include_bytes!("../spike/attn_softmax.spv");
const VALUES: &[u8] = include_bytes!("../spike/attn_values.spv");
const VALUES_U: &[u8] = include_bytes!("../spike/attn_values_u.spv");
const SOFTMAX_U: &[u8] = include_bytes!("../spike/attn_softmax_u.spv");

/// The three-dispatch kernel: scores (keys dimension-major), softmax, values.
struct Split {
    scores: SpikeKernel,
    softmax: SpikeKernel,
    values: SpikeKernel,
}

fn split(dev: &VulkanDevice) -> Split {
    Split {
        scores: dev.spike_kernel(SCORES_T).unwrap(),
        softmax: dev.spike_kernel(SOFTMAX).unwrap(),
        values: dev.spike_kernel(VALUES).unwrap(),
    }
}

/// [`split`] with the unrolled softmax sum and values (the same order).
fn split_unrolled(dev: &VulkanDevice) -> Split {
    Split {
        scores: dev.spike_kernel(SCORES_T).unwrap(),
        softmax: dev.spike_kernel(SOFTMAX_U).unwrap(),
        values: dev.spike_kernel(VALUES_U).unwrap(),
    }
}

/// [`gpu`] for the three-dispatch kernel: keys transposed to `[kv_head][d][span]`.
fn gpu_split(dev: &VulkanDevice, k3: &Split, c: &Case, repeat: u32) -> (Vec<f32>, f64) {
    let Shape { heads, kvh, d } = c.sh;
    let n = c.p0 + c.s;
    let mut kt = vec![0f32; kvh * d * n];
    for j in 0..n {
        for g in 0..kvh {
            for i in 0..d {
                kt[(g * d + i) * n + j] = c.k[(j * kvh + g) * d + i];
            }
        }
    }
    let mut q = c.q.clone();
    let mut kv = [kt.as_slice(), c.v.as_slice()].concat();
    let mut out = vec![0f32; c.s * heads * d];
    let mut scratch = vec![0f32; c.s * heads * n];
    let push = [
        (d | (heads / kvh) << 16) as u32,
        (kvh | heads << 16) as u32,
        c.p0 as u32,
        kt.len() as u32,
    ];
    let (h, s) = (heads as u32, c.s as u32);
    let t = dev
        .spike_run(
            &[
                (&k3.scores, push, [(n as u32).div_ceil(256), h, s]),
                (&k3.softmax, push, [h, s, 1]),
                (&k3.values, push, [(d as u32).div_ceil(64), h, s]),
            ],
            [&mut q, &mut kv, &mut out, &mut scratch],
            repeat,
        )
        .unwrap();
    let ns = t.gpu_ns.unwrap_or(t.wait_ns) as f64 / f64::from(repeat);
    (out, ns)
}

fn device() -> Option<VulkanDevice> {
    match VulkanDevice::new(VulkanConfig {
        timestamps: true,
        ..VulkanConfig::default()
    }) {
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

    /// Uniform in (-a, a).
    fn uni(&mut self, a: f32) -> f32 {
        ((self.next() >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * a
    }
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    heads: usize,
    kvh: usize,
    d: usize,
}

/// One sequence's attention inputs: `s` new tokens at positions `p0..p0 + s`, keys and
/// values for positions `0..p0 + s`.
struct Case {
    sh: Shape,
    s: usize,
    p0: usize,
    /// `[token][head][d]`.
    q: Vec<f32>,
    /// Position-major, as the device keeps them: `[pos][kv_head][d]`.
    k: Vec<f32>,
    v: Vec<f32>,
}

fn case(sh: Shape, s: usize, p0: usize, seed: u32, amp: f32) -> Case {
    let n = p0 + s;
    let mut r = Rng(seed.wrapping_mul(2_654_435_761).max(1));
    let q = (0..s * sh.heads * sh.d).map(|_| r.uni(amp)).collect();
    let k = (0..n * sh.kvh * sh.d).map(|_| r.uni(amp)).collect();
    let v = (0..n * sh.kvh * sh.d).map(|_| r.uni(1.0)).collect();
    Case { sh, s, p0, q, k, v }
}

/// The CPU reference: `ops::attend` per (token, head) over the head's own positions,
/// from a head-major copy of the cache (`[kv_head][pos][d]`, the host cache's layout).
fn cpu(c: &Case) -> Vec<f32> {
    let Shape { heads, kvh, d } = c.sh;
    let n = c.p0 + c.s;
    let major = |x: &[f32]| -> Vec<Vec<f32>> {
        (0..kvh)
            .map(|g| {
                (0..n)
                    .flat_map(|j| x[(j * kvh + g) * d..(j * kvh + g + 1) * d].iter().copied())
                    .collect()
            })
            .collect()
    };
    let (k, v) = (major(&c.k), major(&c.v));
    let group = heads / kvh;
    let mut out = vec![0f32; c.s * heads * d];
    let mut scores = Vec::new();
    for t in 0..c.s {
        for h in 0..heads {
            let at = (t * heads + h) * d;
            let g = h / group;
            attend(
                &c.q[at..at + d],
                &k[g],
                &v[g],
                c.p0 + t,
                &mut out[at..at + d],
                &mut scores,
            );
        }
    }
    out
}

/// The kernel on the device: outputs, and GPU nanoseconds per run over `repeat` runs in
/// one submission.
fn gpu(dev: &VulkanDevice, kern: &SpikeKernel, c: &Case, repeat: u32) -> (Vec<f32>, f64) {
    let Shape { heads, kvh, d } = c.sh;
    let n = c.p0 + c.s;
    let mut q = c.q.clone();
    let mut kv = [c.k.as_slice(), c.v.as_slice()].concat();
    let mut out = vec![0f32; c.s * heads * d];
    let mut scratch = vec![0f32; c.s * heads * n];
    let push = [
        (d | (heads / kvh) << 16) as u32,
        (kvh | heads << 16) as u32,
        c.p0 as u32,
        c.k.len() as u32,
    ];
    let t = dev
        .spike_run(
            &[(kern, push, [heads as u32, c.s as u32, 1])],
            [&mut q, &mut kv, &mut out, &mut scratch],
            repeat,
        )
        .unwrap();
    let ns = t.gpu_ns.unwrap_or(t.wait_ns) as f64 / f64::from(repeat);
    (out, ns)
}

#[derive(Default)]
struct Diff {
    equal: u64,
    nan_payload: u64,
    other: Vec<(usize, u32, u32)>,
}

impl Diff {
    fn add(&mut self, got: &[f32], want: &[f32]) {
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            if a.to_bits() == b.to_bits() {
                self.equal += 1;
            } else if a.is_nan() && b.is_nan() {
                self.nan_payload += 1;
            } else if self.other.len() < 1000 {
                self.other.push((i, a.to_bits(), b.to_bits()));
            }
        }
    }

    #[track_caller]
    fn check(&self, what: &str) {
        eprintln!(
            "{what}: {} equal, {} NaN payload, {} other",
            self.equal,
            self.nan_payload,
            self.other.len()
        );
        assert!(
            self.other.is_empty(),
            "{what}: first (index, gpu, cpu) bits {:x?}",
            &self.other[..self.other.len().min(8)]
        );
    }
}

const OLMOE: Shape = Shape {
    heads: 16,
    kvh: 16,
    d: 128,
};

/// Exact attention on the device equals `ops::attend` bit for bit: OLMoE's heads,
/// grouped-query heads, a head size that is not the workgroup's, contexts across chunk
/// edges up to 4096 positions, several new tokens (prefill), and inputs whose scores
/// spread past `exp`'s underflow.
#[test]
fn exact_attention_matches_the_cpu() {
    let Some(dev) = device() else { return };
    let kernels = [
        ("scratch", dev.spike_kernel(ATTN).unwrap()),
        ("shared", dev.spike_kernel(ATTN_SH).unwrap()),
    ];
    let k3 = split(&dev);
    let k3u = split_unrolled(&dev);
    let mut split_diff = Diff::default();
    let mut unrolled_diff = Diff::default();
    let shapes = [
        OLMOE,
        Shape {
            heads: 8,
            kvh: 2,
            d: 64,
        },
        Shape {
            heads: 4,
            kvh: 4,
            d: 96,
        },
        Shape {
            heads: 2,
            kvh: 1,
            d: 200,
        },
    ];
    for (name, kern) in &kernels {
        let mut diff = Diff::default();
        for (si, &sh) in shapes.iter().enumerate() {
            for (s, p0) in [
                (1, 0),
                (1, 1),
                (1, 63),
                (1, 64),
                (1, 137),
                (1, 1000),
                (1, 4095),
                (4, 60),
                (8, 129),
                (10, 0),
                (8, 4088),
            ] {
                for amp in [1.0, 4.0, 30.0] {
                    let c = case(sh, s, p0, (si * 977 + s * 31 + p0) as u32 + 1, amp);
                    let want = cpu(&c);
                    let (got, _) = gpu(&dev, kern, &c, 1);
                    diff.add(&got, &want);
                    if *name == "scratch" {
                        let (got, _) = gpu_split(&dev, &k3, &c, 1);
                        split_diff.add(&got, &want);
                        let (got, _) = gpu_split(&dev, &k3u, &c, 1);
                        unrolled_diff.add(&got, &want);
                    }
                }
            }
        }
        // A NaN in one key and an infinite key: NaN propagation.
        for poison in [f32::NAN, f32::INFINITY] {
            let mut c = case(OLMOE, 2, 40, 7, 1.0);
            c.k[5 * 16 * 128 + 3] = poison;
            let want = cpu(&c);
            let (got, _) = gpu(&dev, kern, &c, 1);
            diff.add(&got, &want);
            if *name == "scratch" {
                let (got, _) = gpu_split(&dev, &k3, &c, 1);
                split_diff.add(&got, &want);
                let (got, _) = gpu_split(&dev, &k3u, &c, 1);
                unrolled_diff.add(&got, &want);
            }
        }
        diff.check(&format!("exact attention ({name})"));
    }
    split_diff.check("exact attention (three dispatches, keys dimension-major)");
    unrolled_diff.check("exact attention (three dispatches, unrolled)");
}

/// RoPE on the device equals `Rope::rotate` bit for bit with the host's tables: OLMoE's
/// 16 heads of 128 at every position up to 4096, and wide input magnitudes.
#[test]
fn rope_matches_the_cpu() {
    let Some(dev) = device() else { return };
    let kern = dev.spike_kernel(ROPE).unwrap();
    let rope = Rope::new(128, 10_000.0);
    let half = rope.half();
    let (heads, rows) = (16usize, 512usize);
    let mut diff = Diff::default();
    let mut r = Rng(99);
    for chunk in 0..8 {
        for amp in [1.0f32, 1e3, 1e-3] {
            let mut x: Vec<f32> = (0..rows * heads * 2 * half).map(|_| r.uni(amp)).collect();
            let mut cs = vec![0f32; rows * 2 * half];
            for t in 0..rows {
                rope.table(chunk * rows + t, &mut cs[t * 2 * half..(t + 1) * 2 * half]);
            }
            let mut want = x.clone();
            for t in 0..rows {
                let tab = &cs[t * 2 * half..(t + 1) * 2 * half];
                want[t * heads * 2 * half..(t + 1) * heads * 2 * half]
                    .chunks_exact_mut(2 * half)
                    .for_each(|h| Rope::rotate(h, tab));
            }
            let mut none = [0f32; 4];
            let mut none2 = [0f32; 4];
            let n = (rows * heads * half) as u32;
            dev.spike_run(
                &[(
                    &kern,
                    [half as u32, heads as u32, rows as u32, 0],
                    [n.div_ceil(256), 1, 1],
                )],
                [&mut x, &mut cs, &mut none, &mut none2],
                1,
            )
            .unwrap();
            diff.add(&x, &want);
        }
    }
    diff.check("rope");
}

/// Tiled attention with an online softmax (FlashAttention's order), simulated on the
/// CPU: scores in tiles of `tile` positions, a running maximum and sum, the output
/// rescaled by `exp(m_old - m_new)` at each tile. The same operations as `attend`
/// otherwise (`math::exp`, fma). Returns the outputs.
fn online(c: &Case, tile: usize) -> Vec<f32> {
    let Shape { heads, kvh, d } = c.sh;
    let group = heads / kvh;
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0f32; c.s * heads * d];
    for t in 0..c.s {
        let n = c.p0 + t + 1;
        for h in 0..heads {
            let g = h / group;
            let q = &c.q[(t * heads + h) * d..(t * heads + h + 1) * d];
            let mut m = f32::NEG_INFINITY;
            let mut l = 0f32;
            let mut acc = vec![0f32; d];
            for start in (0..n).step_by(tile) {
                let end = (start + tile).min(n);
                let sc: Vec<f32> = (start..end)
                    .map(|j| {
                        let k = &c.k[(j * kvh + g) * d..(j * kvh + g + 1) * d];
                        q.iter().zip(k).fold(0f32, |a, (x, y)| x.mul_add(*y, a)) * scale
                    })
                    .collect();
                let mt = sc.iter().copied().fold(m, f32::max);
                let corr = qubivex_compute::math::exp(m - mt);
                l *= corr;
                acc.iter_mut().for_each(|a| *a *= corr);
                for (jj, &sv) in sc.iter().enumerate() {
                    let p = qubivex_compute::math::exp(sv - mt);
                    l += p;
                    let v = &c.v[((start + jj) * kvh + g) * d..((start + jj) * kvh + g + 1) * d];
                    for (a, x) in acc.iter_mut().zip(v) {
                        *a = p.mul_add(*x, *a);
                    }
                }
                m = mt;
            }
            let o = &mut out[(t * heads + h) * d..(t * heads + h + 1) * d];
            for (a, b) in o.iter_mut().zip(&acc) {
                *a = b / l;
            }
        }
    }
    out
}

/// How far an online softmax is from the reference: the share of outputs whose bits
/// differ, and the largest difference in ulps. A measurement for the audit, not a gate;
/// it fails only if the tiled order were ever bit-exact, which would change the design.
#[test]
fn online_softmax_is_not_bit_exact() {
    let mut rows = Vec::new();
    for (p0, tile) in [(137, 64), (1000, 64), (4095, 64), (4095, 256), (63, 64)] {
        let c = case(OLMOE, 1, p0, p0 as u32 + 5, 4.0);
        let (want, got) = (cpu(&c), online(&c, tile));
        let differ = want
            .iter()
            .zip(&got)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let ulp = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs())
            .max()
            .unwrap();
        eprintln!(
            "online softmax, {} positions, tile {tile}: {differ} of {} outputs differ, max {ulp} ulp",
            p0 + 1,
            want.len()
        );
        rows.push((p0, differ));
    }
    assert!(
        rows.iter().filter(|(p0, _)| *p0 >= 64).all(|(_, d)| *d > 0),
        "a multi-tile online softmax matched the reference bit for bit"
    );
}

/// Median of `f` over `n` runs, in nanoseconds.
fn median_ns(n: usize, mut f: impl FnMut()) -> f64 {
    let mut v: Vec<f64> = (0..n)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as f64
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v[n / 2]
}

/// The host's attention block per layer at decode, as `forward_segments` runs it after the
/// QKV submission (one token): Q and K norms over the full width, RoPE on every head, the
/// KV store copy, and attention over the heads on `threads` threads.
fn cpu_block_ns(c: &Case, threads: usize) -> (f64, f64) {
    let Shape { heads, kvh, d } = c.sh;
    let n = c.p0 + c.s;
    let major = |x: &[f32]| -> Vec<Vec<f32>> {
        (0..kvh)
            .map(|g| {
                (0..n)
                    .flat_map(|j| x[(j * kvh + g) * d..(j * kvh + g + 1) * d].iter().copied())
                    .collect()
            })
            .collect()
    };
    let (k, v) = (major(&c.k), major(&c.v));
    let rope = Rope::new(d, 10_000.0);
    let w = vec![1.0f32; heads * d];
    let attn = || {
        let per = heads.div_ceil(threads);
        std::thread::scope(|sc| {
            for part in 0..threads {
                let (k, v) = (&k, &v);
                sc.spawn(move || {
                    let mut out = vec![0f32; d];
                    let mut scores = Vec::new();
                    for h in part * per..((part + 1) * per).min(heads) {
                        attend(
                            &c.q[h * d..(h + 1) * d],
                            &k[h * kvh / heads],
                            &v[h * kvh / heads],
                            c.p0,
                            &mut out,
                            &mut scores,
                        );
                    }
                });
            }
        });
    };
    let attn_ns = median_ns(21, attn);
    let mut q = c.q[..heads * d].to_vec();
    let mut kk = c.k[c.p0 * kvh * d..].to_vec();
    let mut cs = vec![0f32; d];
    let mut store = vec![0f32; kvh * d];
    let rest_ns = median_ns(201, || {
        qubivex_arch::ops::rms_norm(&mut q, &w, 1e-5);
        qubivex_arch::ops::rms_norm(&mut kk, &w, 1e-5);
        rope.table(c.p0, &mut cs);
        q.chunks_exact_mut(d).for_each(|h| Rope::rotate(h, &cs));
        kk.chunks_exact_mut(d).for_each(|h| Rope::rotate(h, &cs));
        store.copy_from_slice(&kk);
    });
    (attn_ns, rest_ns)
}

/// GPU time of exact attention per layer call at OLMoE's shape, 1 and 8 new tokens, from
/// 138 to 4096 positions, both kernels (50 runs in one submission); and the CPU's attention
/// block at decode on 5 threads.
#[test]
#[ignore = "timing"]
fn attention_timings() {
    let Some(dev) = device() else { return };
    let scratch = dev.spike_kernel(ATTN).unwrap();
    let shared = dev.spike_kernel(ATTN_SH).unwrap();
    let k3 = split(&dev);
    let k3u = split_unrolled(&dev);
    eprintln!(
        "| positions | tokens | GPU µs (scratch) | GPU µs (shared) | GPU µs (3 dispatches) | GPU µs (3, unrolled) | CPU attend µs (5 threads) | CPU attend µs (1 thread) | CPU norms+RoPE+store µs |"
    );
    eprintln!("|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for n in [138usize, 512, 1024, 2048, 4096] {
        for s in [1usize, 8] {
            let c = case(OLMOE, s, n - s, n as u32, 4.0);
            let (_, g1) = gpu(&dev, &scratch, &c, 50);
            let (_, g2) = gpu(&dev, &shared, &c, 50);
            let (_, g3) = gpu_split(&dev, &k3, &c, 50);
            let (_, g4) = gpu_split(&dev, &k3u, &c, 50);
            let (cpu_a, cpu_1, cpu_r) = if s == 1 {
                let (a, r) = cpu_block_ns(&c, 5);
                (a, cpu_block_ns(&c, 1).0, r)
            } else {
                (f64::NAN, f64::NAN, f64::NAN)
            };
            eprintln!(
                "| {n} | {s} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |",
                g1 / 1e3,
                g2 / 1e3,
                g3 / 1e3,
                g4 / 1e3,
                cpu_a / 1e3,
                cpu_1 / 1e3,
                cpu_r / 1e3
            );
        }
    }
}

/// GPU time of each of the three dispatches alone at OLMoE's shape, one token.
#[test]
#[ignore = "timing"]
fn attention_phase_timings() {
    let Some(dev) = device() else { return };
    let k3 = split(&dev);
    for n in [512usize, 4096] {
        let c = case(OLMOE, 1, n - 1, n as u32, 4.0);
        let Shape { heads, kvh, d } = c.sh;
        let push = [
            (d | (heads / kvh) << 16) as u32,
            (kvh | heads << 16) as u32,
            c.p0 as u32,
            (kvh * d * n) as u32,
        ];
        let h = heads as u32;
        let steps = [
            ("scores", &k3.scores, [(n as u32).div_ceil(256), h, 1]),
            ("softmax", &k3.softmax, [h, 1, 1]),
            ("values", &k3.values, [(d as u32).div_ceil(64), h, 1]),
        ];
        for (name, k, grid) in steps {
            let mut q = c.q.clone();
            let mut kv = vec![0.5f32; 2 * kvh * d * n];
            let mut out = vec![0f32; heads * d];
            let mut sc = vec![1.0f32 / n as f32; heads * n];
            let t = dev
                .spike_run(&[(k, push, grid)], [&mut q, &mut kv, &mut out, &mut sc], 50)
                .unwrap();
            eprintln!(
                "{n} positions, {name}: {:.1} µs",
                t.gpu_ns.unwrap() as f64 / 50.0 / 1e3
            );
        }
    }
}
