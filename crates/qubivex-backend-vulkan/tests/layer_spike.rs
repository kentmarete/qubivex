//! A whole decode layer's time with attention on the device (audit §20, step 2 of §19.10;
//! feature `spike`). OLMoE's layer shape on synthetic weights, measured four ways:
//!
//! - the QKV submission (`matmul_many`, Q and K Q4_K, V Q6_K);
//! - Step 3c's routed submission with its prelude (O projection, residual, FFN norm,
//!   router, routing, experts, combine);
//! - the same routed submission with the attention prelude (`qk_prelude`) and exact
//!   attention (`attn_one*`) recorded at its head, in the same command buffer
//!   ([`VulkanDevice::spike_arm`]);
//! - the host's attention block between the two, as Step 3c runs it.
//!
//! The armed steps work on their own buffers: the O projection still reads the host's
//! attention outputs, so the GPU does the same work as a chained layer would, with a
//! barrier in between, but without the data dependency. The attention outputs the prefix
//! computed are checked against `ops::attend`, so it ran in that command buffer.
//!
//! `cargo test --release -p qubivex-backend-vulkan --features spike --test layer_spike
//! -- --ignored --nocapture`

#![cfg(feature = "spike")]
#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    missing_docs
)]

use qubivex_arch::ops::{Rope, attend};
use qubivex_backend_vulkan::{SpikeTiming, VulkanConfig, VulkanDevice};
use qubivex_compute::discrete::{
    DeviceMatMul, DeviceMoePrelude, DeviceRoutedMoe, DeviceWeights, DiscreteDevice,
};
use qubivex_core::{DType, GgmlType, Shape};
use std::time::Instant;

const PRELUDE: &[u8] = include_bytes!("../spike/qk_prelude.spv");
const ONE_BATCH: &[u8] = include_bytes!("../spike/attn_oneb_u16.spv");

const H: usize = 2048;
const HEADS: usize = 16;
const D: usize = 128;
const FFN: usize = 1024;
const EXPERTS: usize = 64;
const K: usize = 8;

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    fn uni(&mut self, a: f32) -> f32 {
        ((self.next() >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * a
    }
}

fn bytes_of(dtype: DType, rows: usize, cols: usize) -> usize {
    dtype
        .storage_bytes(&Shape::new(&[rows as u64, cols as u64]).unwrap())
        .unwrap() as usize
}

/// Deterministic weights (as `routed.rs`).
fn weights(dtype: DType, rows: usize, cols: usize, seed: u32) -> Vec<u8> {
    let n = bytes_of(dtype, rows, cols);
    let mut r = Rng(seed.wrapping_mul(2_654_435_761).max(1));
    match dtype {
        DType::F32 => (0..n / 4)
            .flat_map(|_| {
                let v = r.next();
                let e = 112 + (v >> 24) % 12;
                ((v & 0x807F_FFFF) | (e << 23)).to_le_bytes()
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

fn device() -> Option<VulkanDevice> {
    match VulkanDevice::new(VulkanConfig {
        route: true,
        route_o: true,
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

fn upload(dev: &VulkanDevice, dtype: DType, rows: usize, cols: usize, seed: u32) -> DeviceWeights {
    let w = weights(dtype, rows, cols, seed);
    DeviceWeights::new(dev.upload(&w, None).unwrap(), 0, dtype, rows, cols).unwrap()
}

/// The layer's weights on the device: Q, K, V, O, the router and every expert in one
/// arena.
struct Weights {
    q: DeviceWeights,
    k: DeviceWeights,
    v: DeviceWeights,
    o: DeviceWeights,
    router: DeviceWeights,
    experts: Vec<[DeviceWeights; 3]>,
    norm: Vec<f32>,
}

fn layer(dev: &VulkanDevice) -> Weights {
    let q4 = DType::Ggml(GgmlType::Q4_K);
    let q6 = DType::Ggml(GgmlType::Q6_K);
    let lens = [
        bytes_of(q4, FFN, H),
        bytes_of(q4, FFN, H),
        bytes_of(q6, H, FFN),
    ];
    let slot = lens.iter().sum::<usize>().next_multiple_of(256);
    let arena = dev.alloc(slot * EXPERTS, None).unwrap();
    let experts = (0..EXPERTS)
        .map(|e| {
            let s = 100 + 3 * e as u32;
            let bytes = [
                weights(q4, FFN, H, s),
                weights(q4, FFN, H, s + 1),
                weights(q6, H, FFN, s + 2),
            ]
            .concat();
            dev.write(&arena, e * slot, &bytes).unwrap();
            let b = e * slot;
            [
                DeviceWeights::new(arena.clone(), b, q4, FFN, H).unwrap(),
                DeviceWeights::new(arena.clone(), b + lens[0], q4, FFN, H).unwrap(),
                DeviceWeights::new(arena.clone(), b + lens[0] + lens[1], q6, H, FFN).unwrap(),
            ]
        })
        .collect();
    let mut r = Rng(5);
    Weights {
        q: upload(dev, q4, H, H, 1),
        k: upload(dev, q4, H, H, 2),
        v: upload(dev, q6, H, H, 3),
        o: upload(dev, q4, H, H, 4),
        router: upload(dev, DType::F32, EXPERTS, H, 6),
        experts,
        norm: (0..H).map(|_| 0.5 + r.uni(0.4).abs()).collect(),
    }
}

/// Timing of one call: wall nanoseconds around it, and the submission's GPU time.
#[derive(Clone, Copy)]
struct Call {
    wall: f64,
    gpu: f64,
}

fn call(dev: &VulkanDevice, f: impl FnOnce()) -> Call {
    let t = Instant::now();
    f();
    let wall = t.elapsed().as_nanos() as f64;
    let SpikeTiming { gpu_ns, .. } = dev.spike_last().unwrap();
    Call {
        wall,
        gpu: gpu_ns.unwrap() as f64,
    }
}

/// Batched decode's attention inputs: `s` sequences of `n` positions, one new token each
/// at position n - 1. q, k and v region-major (as the QKV submission writes them); the KV
/// buffer as attn_sv.comp's BATCH build reads it: every sequence's K^T `[kv_head][d][n]`,
/// then every sequence's V `[pos][kv_head][d]`; the prelude's table.
struct Att {
    s: usize,
    n: usize,
    qkv: Vec<f32>,
    kv: Vec<f32>,
    table: Vec<f32>,
}

const EPS: f32 = 1e-5;

fn att(s: usize, n: usize, seed: u32) -> Att {
    let mut r = Rng(seed.max(1));
    let qkv: Vec<f32> = (0..s * 3 * H).map(|_| r.uni(2.0)).collect();
    let kv: Vec<f32> = (0..2 * s * H * n).map(|_| r.uni(1.0)).collect();
    // One 16-word header per sequence, then the norm weights and one RoPE row.
    let per = H * n;
    let (qn, kn) = (16 * s, 16 * s + H);
    let cs = kn + H;
    let mut table = vec![0f32; cs + D];
    for q in 0..s {
        let words = [
            D,
            HEADS,
            HEADS,
            n - 1,
            n,
            s * per + q * per,
            EPS.to_bits() as usize,
            0,
            qn,
            kn,
            cs,
            q * per,
        ];
        for (j, w) in words.iter().enumerate() {
            table[16 * q + j] = f32::from_bits(*w as u32);
        }
    }
    let mut r = Rng(seed + 11);
    for w in &mut table[qn..cs] {
        *w = 1.0 + r.uni(0.25);
    }
    Rope::new(D, 10_000.0).table(n - 1, &mut table[cs..cs + D]);
    Att {
        s,
        n,
        qkv,
        kv,
        table,
    }
}

/// The CPU: per sequence, the prelude (Q and K norms, RoPE, the store) then `attend` per
/// head over its own cache. Returns `[sequence][head][d]`.
fn att_cpu(a: &Att) -> Vec<f32> {
    let (s, n) = (a.s, a.n);
    let per = H * n;
    let (qn, kn) = (
        &a.table[16 * s..16 * s + H],
        &a.table[16 * s + H..16 * s + 2 * H],
    );
    let cs = &a.table[16 * s + 2 * H..16 * s + 2 * H + D];
    let mut out = vec![0f32; s * H];
    let mut scores = Vec::new();
    for q in 0..s {
        let mut qq = a.qkv[q * H..(q + 1) * H].to_vec();
        let mut kk = a.qkv[s * H + q * H..s * H + (q + 1) * H].to_vec();
        let vv = &a.qkv[2 * s * H + q * H..2 * s * H + (q + 1) * H];
        qubivex_arch::ops::rms_norm(&mut qq, qn, f64::from(EPS));
        qubivex_arch::ops::rms_norm(&mut kk, kn, f64::from(EPS));
        qq.chunks_exact_mut(D).for_each(|h| Rope::rotate(h, cs));
        kk.chunks_exact_mut(D).for_each(|h| Rope::rotate(h, cs));
        for g in 0..HEADS {
            // The cache, head-major as the host keeps it, with the new token stored.
            let mut kh = vec![0f32; n * D];
            let mut vh = vec![0f32; n * D];
            for j in 0..n {
                for i in 0..D {
                    kh[j * D + i] = a.kv[q * per + (g * D + i) * n + j];
                    vh[j * D + i] = a.kv[s * per + q * per + (j * HEADS + g) * D + i];
                }
            }
            kh[(n - 1) * D..].copy_from_slice(&kk[g * D..(g + 1) * D]);
            vh[(n - 1) * D..].copy_from_slice(&vv[g * D..(g + 1) * D]);
            attend(
                &qq[g * D..(g + 1) * D],
                &kh,
                &vh,
                n - 1,
                &mut out[(q * HEADS + g) * D..(q * HEADS + g + 1) * D],
                &mut scores,
            );
        }
    }
    out
}

struct Kernels {
    pre: qubivex_backend_vulkan::SpikeKernel,
    attn: qubivex_backend_vulkan::SpikeKernel,
}

/// The prefix: the prelude (one dispatch, each token its own sequence's header), then
/// attention (one dispatch). Bindings: 0 q, k, v; 1 the KV buffer; 2 the outputs (the
/// prelude reads its table there first: the table is placed after the outputs); 3 unused.
fn prefix(dev: &VulkanDevice, k: &Kernels, a: &Att) -> qubivex_backend_vulkan::SpikePrefix {
    let (s, n) = (a.s as u32, a.n as u32);
    // Binding 2 holds the outputs, then the table: the prelude's header words are offset
    // by the outputs' length, and every word index in the table is too.
    let off = a.s * H;
    let mut b2 = vec![0f32; off];
    let mut table = a.table.clone();
    for q in 0..a.s {
        for j in [8usize, 9, 10] {
            let w = table[16 * q + j].to_bits() + off as u32;
            table[16 * q + j] = f32::from_bits(w);
        }
    }
    b2.extend_from_slice(&table);
    let per = (H * a.n) as u32;
    let steps = [
        (&k.pre, [off as u32, 16, 0, 0], [3, s, 1]),
        (
            &k.attn,
            [
                (D | 1 << 16) as u32,
                (HEADS | HEADS << 16) as u32,
                n - 1,
                s * per,
            ],
            [HEADS as u32, s, 1],
        ),
    ];
    dev.spike_prefix(&steps, [&a.qkv, &a.kv, &b2, &[0f32; 4]])
        .unwrap()
}

/// The prefix computes exact attention inside a routed submission: its outputs equal the
/// CPU's prelude and `attend`, bit for bit, for 1 and 8 sequences.
#[test]
fn the_prefix_attends_exactly_in_the_routed_submission() {
    let Some(dev) = device() else { return };
    let w = layer(&dev);
    let k = Kernels {
        pre: dev.spike_kernel(PRELUDE).unwrap(),
        attn: dev.spike_kernel(ONE_BATCH).unwrap(),
    };
    let mut equal = 0;
    for (s, n) in [(1, 138), (8, 138), (1, 1024), (8, 300)] {
        let a = att(s, n, (s * 1000 + n) as u32);
        let want = att_cpu(&a);
        dev.spike_arm(prefix(&dev, &k, &a));
        assert!(routed(&dev, &w, s));
        let mut done = dev.spike_done().expect("the prefix ran");
        let got = done.read(2, s * H);
        let bad = got
            .iter()
            .zip(&want)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            bad,
            0,
            "{s} sequences of {n}: {bad} of {} differ",
            want.len()
        );
        equal += want.len();
    }
    eprintln!("prefix attention in the routed submission: {equal} of {equal} equal");
}

/// Step 3c's routed submission with its prelude, for `s` tokens.
fn routed(dev: &VulkanDevice, w: &Weights, s: usize) -> bool {
    let att: Vec<f32> = (0..s * H)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let mut hs: Vec<f32> = (0..s * H)
        .map(|i| ((i * 13 % 89) as f32 - 44.0) / 30.0)
        .collect();
    let (mut ys, mut ids, mut ws) = (vec![0f32; s * H], vec![0u32; s * K], vec![0f32; s * K]);
    dev.routed_moe(&mut DeviceRoutedMoe {
        router: &w.router,
        experts: &w.experts,
        k: K,
        renormalise: false,
        xs: &[],
        s,
        ys: &mut ys,
        ids: &mut ids,
        weights: &mut ws,
        prelude: Some(DeviceMoePrelude {
            o: &w.o,
            att: &att,
            hs: &mut hs,
            norm: &w.norm,
            eps: f64::from(EPS),
        }),
        attention: None,
    })
    .unwrap()
}

/// Step 3c's QKV submission for `s` tokens.
fn qkv(dev: &VulkanDevice, w: &Weights, s: usize) {
    let x: Vec<f32> = (0..s * H)
        .map(|i| ((i * 29 % 97) as f32 - 48.0) / 24.0)
        .collect();
    let (mut q, mut k, mut v) = (vec![0f32; s * H], vec![0f32; s * H], vec![0f32; s * H]);
    dev.matmul_many(&mut [
        DeviceMatMul {
            w: &w.q,
            xs: &x,
            s,
            ys: &mut q,
        },
        DeviceMatMul {
            w: &w.k,
            xs: &x,
            s,
            ys: &mut k,
        },
        DeviceMatMul {
            w: &w.v,
            xs: &x,
            s,
            ys: &mut v,
        },
    ])
    .unwrap();
}

/// The host's attention block per layer as Step 3c runs it for `s` sequences of `n`
/// positions: Q and K norms, RoPE, the store, and `attend` per (sequence, head) on
/// `threads` threads. Median wall nanoseconds.
fn host_attention(a: &Att, threads: usize) -> f64 {
    let (s, n) = (a.s, a.n);
    let kh: Vec<f32> = (0..n * D).map(|i| (i % 7) as f32 / 7.0).collect();
    let vh = kh.clone();
    let rope = Rope::new(D, 10_000.0);
    let w = vec![1.0f32; H];
    let mut v: Vec<f64> = (0..21)
        .map(|_| {
            let t = Instant::now();
            let mut q = a.qkv[..s * H].to_vec();
            let mut k = a.qkv[s * H..2 * s * H].to_vec();
            let mut cs = vec![0f32; D];
            rope.table(n - 1, &mut cs);
            for r in q.chunks_exact_mut(H).chain(k.chunks_exact_mut(H)) {
                qubivex_arch::ops::rms_norm(r, &w, 1e-5);
                r.chunks_exact_mut(D).for_each(|h| Rope::rotate(h, &cs));
            }
            let pairs = s * HEADS;
            let per = pairs.div_ceil(threads);
            std::thread::scope(|sc| {
                for part in 0..threads {
                    let (q, kh, vh) = (&q, &kh, &vh);
                    sc.spawn(move || {
                        let mut out = vec![0f32; D];
                        let mut scores = Vec::new();
                        for i in part * per..((part + 1) * per).min(pairs) {
                            attend(&q[i * D..(i + 1) * D], kh, vh, n - 1, &mut out, &mut scores);
                        }
                    });
                }
            });
            t.elapsed().as_nanos() as f64
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v[10]
}

/// Per layer, OLMoE's shape: Step 3c (QKV submission, host attention, routed submission)
/// against the routed submission with the prelude and attention at its head, at 138,
/// 1024 and 4096 positions, for 1 and 8 sequences. Wall and GPU medians of 21 calls,
/// the three alternating, after two unrecorded.
#[test]
#[ignore = "timing"]
fn layer_timings() {
    let Some(dev) = device() else { return };
    let w = layer(&dev);
    let k = Kernels {
        pre: dev.spike_kernel(PRELUDE).unwrap(),
        attn: dev.spike_kernel(ONE_BATCH).unwrap(),
    };
    // Warm up for a second.
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < 1.0 {
        routed(&dev, &w, 8);
    }
    eprintln!(
        "| sequences | positions | QKV wall / GPU µs | routed (3c) wall / GPU µs | host attention µs (5 thr / 1 thr) | routed + attention wall / GPU µs | added GPU µs |"
    );
    eprintln!("|---:|---:|---:|---:|---:|---:|---:|");
    for s in [1usize, 8] {
        for n in [138usize, 1024, 4096] {
            let a = att(s, n, 7);
            // One prefix, re-armed for every call (its q is normed and rotated again each
            // time, which leaves the work the same).
            let mut p = Some(prefix(&dev, &k, &a));
            let t0 = Instant::now();
            while t0.elapsed().as_secs_f64() < 0.5 {
                routed(&dev, &w, s);
            }
            // The three calls alternate, so drift touches all three alike.
            let (mut q, mut r, mut ra) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..23 {
                q.push(call(&dev, || qkv(&dev, &w, s)));
                r.push(call(&dev, || assert!(routed(&dev, &w, s))));
                dev.spike_arm(p.take().unwrap());
                ra.push(call(&dev, || assert!(routed(&dev, &w, s))));
                p = Some(dev.spike_done().expect("the prefix ran"));
            }
            let med = |v: &[Call]| {
                let mut w: Vec<f64> = v[2..].iter().map(|c| c.wall).collect();
                let mut g: Vec<f64> = v[2..].iter().map(|c| c.gpu).collect();
                w.sort_by(f64::total_cmp);
                g.sort_by(f64::total_cmp);
                Call {
                    wall: w[w.len() / 2],
                    gpu: g[g.len() / 2],
                }
            };
            let (q, r, ra) = (med(&q), med(&r), med(&ra));
            let (h5, h1) = (host_attention(&a, 5), host_attention(&a, 1));
            eprintln!(
                "| {s} | {n} | {:.0} / {:.0} | {:.0} / {:.0} | {:.0} / {:.0} | {:.0} / {:.0} | {:.0} |",
                q.wall / 1e3,
                q.gpu / 1e3,
                r.wall / 1e3,
                r.gpu / 1e3,
                h5 / 1e3,
                h1 / 1e3,
                ra.wall / 1e3,
                ra.gpu / 1e3,
                (ra.gpu - r.gpu) / 1e3
            );
        }
    }
}
