//! Routing on the Vulkan device (audit §17, Step 3b) against the CPU, bit for bit: the
//! route shader alone against `qubivex_compute::math::route`, and the whole routed layer
//! (router, routing, the routed experts' fused block, combine) in one submission against
//! `qubivex_compute::routed_moe_steps` on the CPU; with its prelude (audit §18, Step 3c:
//! the O projection, residual and FFN norm first), the residual stream as well.
//!
//! Exceptions, as the parity contract states them (ADR-011 amendment 7): a token with a
//! NaN logit has NaN probabilities; their payloads are outside the contract, and
//! `total_cmp` orders NaNs by payload, so for such a token only NaN-ness is compared
//! (every chosen weight NaN on both), not which ids were chosen.
//!
//! Each test opens the default Vulkan device and skips (passes, with a note) when there
//! is none, unless `QUBIVEX_REQUIRE_VULKAN=1`.

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    missing_docs
)]

use qubivex_backend_cpu::CpuBackend;
use qubivex_backend_vulkan::{VulkanConfig, VulkanDevice};
use qubivex_compute::discrete::{DeviceMoePrelude, DeviceRoutedMoe, DeviceWeights, DiscreteDevice};
use qubivex_compute::{MoePrelude, RoutedMoe, WeightView, math, routed_moe_steps};
use qubivex_core::{DType, GgmlType, Shape};

fn device(route: bool) -> Option<VulkanDevice> {
    device_with(route, false)
}

fn device_with(route: bool, route_o: bool) -> Option<VulkanDevice> {
    match VulkanDevice::new(VulkanConfig {
        route,
        route_o,
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

    /// Uniform in [-a, a).
    fn uniform(&mut self, a: f32) -> f32 {
        (self.next() >> 8) as f32 / (1 << 24) as f32 * 2.0 * a - a
    }
}

/// The CPU's routing of `s` tokens.
fn cpu_route(logits: &[f32], s: usize, k: usize, norm: bool) -> (Vec<u32>, Vec<f32>) {
    let ne = logits.len() / s;
    let (mut ids, mut ws) = (Vec::new(), Vec::new());
    for t in 0..s {
        let (i, w) = math::route(&logits[t * ne..(t + 1) * ne], k, norm);
        ids.extend(i);
        ws.extend(w);
    }
    (ids, ws)
}

/// Compares the device's routing with the CPU's, token by token.
#[track_caller]
fn check_route(dev: &VulkanDevice, logits: &[f32], s: usize, k: usize, norm: bool, what: &str) {
    let (want_ids, want_w) = cpu_route(logits, s, k, norm);
    let (mut ids, mut ws) = (vec![0u32; s * k], vec![0f32; s * k]);
    dev.route(logits, s, k, norm, &mut ids, &mut ws).unwrap();
    let ne = logits.len() / s;
    let mut nan_tokens = 0;
    for t in 0..s {
        let r = t * k..(t + 1) * k;
        if logits[t * ne..(t + 1) * ne].iter().any(|v| v.is_nan()) {
            nan_tokens += 1;
            assert!(
                ws[r.clone()].iter().all(|v| v.is_nan()) && want_w[r].iter().all(|v| v.is_nan()),
                "{what}: token {t}: a NaN logit must give NaN weights on both"
            );
            continue;
        }
        assert_eq!(ids[r.clone()], want_ids[r.clone()], "{what}: token {t} ids");
        let got: Vec<u32> = ws[r.clone()].iter().map(|v| v.to_bits()).collect();
        let want: Vec<u32> = want_w[r].iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, want, "{what}: token {t} weights");
    }
    if nan_tokens > 0 {
        eprintln!("{what}: {nan_tokens} tokens with NaN logits compared by NaN-ness only");
    }
}

/// Logits chosen to break a careless port: ties (the lower id wins), pairs of ties,
/// infinities, NaNs, a spread past exp's underflow, all equal, signed zeros.
fn adversarial(ne: usize, rng: &mut Rng) -> Vec<Vec<f32>> {
    let mut cases = Vec::new();
    let base: Vec<f32> = (0..ne).map(|_| rng.uniform(4.0)).collect();
    cases.push(base.clone());
    cases.push(vec![0.5; ne]);
    cases.push(vec![0.0; ne]);
    cases.push(
        (0..ne)
            .map(|e| if e % 2 == 0 { 0.0 } else { -0.0 })
            .collect(),
    );
    let mut ties = base.clone();
    for e in (0..ne).step_by(3) {
        ties[e] = 1.25;
    }
    cases.push(ties);
    let mut pairs = base.clone();
    for e in (0..ne.saturating_sub(1)).step_by(2) {
        pairs[e + 1] = pairs[e];
    }
    cases.push(pairs);
    let mut ninf = base.clone();
    ninf[0] = f32::NEG_INFINITY;
    ninf[ne - 1] = f32::NEG_INFINITY;
    cases.push(ninf);
    let mut pinf = base.clone();
    pinf[ne / 2] = f32::INFINITY;
    cases.push(pinf.clone());
    pinf[ne / 3] = f32::INFINITY;
    cases.push(pinf);
    let mut nan = base.clone();
    nan[ne / 2] = f32::NAN;
    cases.push(nan);
    let mut spread = base.clone();
    spread[1] = 120.0;
    spread[2] = 30.0;
    cases.push(spread);
    let mut big = base.clone();
    for v in &mut big {
        *v *= 1.0e30;
    }
    cases.push(big);
    let mut tiny = base;
    for v in &mut tiny {
        *v *= 1.0e-30;
    }
    cases.push(tiny);
    cases
}

/// The route shader alone over adversarial and random logits, at several expert counts
/// and k, with and without renormalisation.
#[test]
fn routing_matches_the_cpu() {
    let Some(dev) = device(false) else { return };
    let mut rng = Rng(0x2545_f491);
    for ne in [8, 60, 64, 128] {
        for k in [1, 2, 8] {
            for norm in [false, true] {
                let cases = adversarial(ne, &mut rng);
                let s = cases.len();
                let logits: Vec<f32> = cases.concat();
                check_route(
                    &dev,
                    &logits,
                    s,
                    k,
                    norm,
                    &format!("adversarial {ne}/{k}/{norm}"),
                );
                // Random tokens at the scales a router produces, and wider.
                let s = 8192;
                let scale = [1.0, 4.0, 16.0, 64.0][(ne + k) % 4];
                let logits: Vec<f32> = (0..s * ne).map(|_| rng.uniform(scale)).collect();
                check_route(
                    &dev,
                    &logits,
                    s,
                    k,
                    norm,
                    &format!("random {ne}/{k}/{norm}"),
                );
            }
        }
    }
}

/// OLMoE's routing shape (64 experts, top 8, no renormalisation) over a million tokens.
#[test]
fn routing_matches_the_cpu_at_scale() {
    let Some(dev) = device(false) else { return };
    let mut rng = Rng(0x9e37_79b9);
    for scale in [0.5f32, 2.0, 8.0, 32.0] {
        let s = 1 << 18;
        let logits: Vec<f32> = (0..s * 64).map(|_| rng.uniform(scale)).collect();
        check_route(&dev, &logits, s, 8, false, &format!("scale {scale}"));
    }
}

fn bytes_of(dtype: DType, rows: usize, cols: usize) -> usize {
    dtype
        .storage_bytes(&Shape::new(&[rows as u64, cols as u64]).unwrap())
        .unwrap() as usize
}

/// Deterministic weights of moderate, normal values (as in `fused.rs`).
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

#[derive(Clone, Copy, Debug)]
struct Layer {
    h: usize,
    ffn: usize,
    experts: usize,
    k: usize,
    router: DType,
    gu: DType,
    down: DType,
}

struct Host {
    router: Vec<u8>,
    experts: Vec<[Vec<u8>; 3]>,
}

fn host(l: Layer, seed: u32, tie: bool) -> Host {
    let mut router = weights(l.router, l.experts, l.h, seed);
    if tie {
        // Expert 3's router row equals expert 1's: equal logits, the lower id first.
        let row = router.len() / l.experts;
        let one = router[row..2 * row].to_vec();
        router[3 * row..4 * row].copy_from_slice(&one);
    }
    let experts = (0..l.experts as u32)
        .map(|e| {
            [
                weights(l.gu, l.ffn, l.h, seed + 7 + 3 * e),
                weights(l.gu, l.ffn, l.h, seed + 8 + 3 * e),
                weights(l.down, l.h, l.ffn, seed + 9 + 3 * e),
            ]
        })
        .collect();
    Host { router, experts }
}

/// The CPU's layer: separate steps.
fn cpu_layer(
    l: Layer,
    h: &Host,
    xs: &[f32],
    s: usize,
    norm: bool,
) -> (Vec<f32>, Vec<u32>, Vec<f32>) {
    let views: Vec<[WeightView<'_>; 3]> = h
        .experts
        .iter()
        .map(|[g, u, d]| {
            [
                WeightView::new(l.gu, l.ffn, l.h, g).unwrap(),
                WeightView::new(l.gu, l.ffn, l.h, u).unwrap(),
                WeightView::new(l.down, l.h, l.ffn, d).unwrap(),
            ]
        })
        .collect();
    let (mut ys, mut ids, mut ws) = (
        vec![0f32; s * l.h],
        vec![0u32; s * l.k],
        vec![0f32; s * l.k],
    );
    routed_moe_steps(
        &CpuBackend::new(),
        &mut RoutedMoe {
            router: WeightView::new(l.router, l.experts, l.h, &h.router).unwrap(),
            experts: &views,
            k: l.k,
            renormalise: norm,
            xs,
            s,
            ys: &mut ys,
            ids: &mut ids,
            weights: &mut ws,
            prelude: None,
        },
    )
    .unwrap();
    (ys, ids, ws)
}

/// The layer's copies on the device: the router uploaded, the experts in one arena
/// (`arena`) or each its own upload.
fn place(
    dev: &VulkanDevice,
    l: Layer,
    h: &Host,
    arena: bool,
) -> (DeviceWeights, Vec<[DeviceWeights; 3]>) {
    let router = DeviceWeights::new(
        dev.upload(&h.router, None).unwrap(),
        0,
        l.router,
        l.experts,
        l.h,
    )
    .unwrap();
    let lens = [
        bytes_of(l.gu, l.ffn, l.h),
        bytes_of(l.gu, l.ffn, l.h),
        bytes_of(l.down, l.h, l.ffn),
    ];
    let slot = lens.iter().sum::<usize>().next_multiple_of(256);
    let a = arena.then(|| dev.alloc(slot * l.experts, None).unwrap());
    let experts = h
        .experts
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let bytes = e.concat();
            let (buf, base) = match &a {
                Some(a) => {
                    dev.write(a, i * slot, &bytes).unwrap();
                    (a.clone(), i * slot)
                }
                None => (dev.upload(&bytes, None).unwrap(), 0),
            };
            [
                DeviceWeights::new(buf.clone(), base, l.gu, l.ffn, l.h).unwrap(),
                DeviceWeights::new(buf.clone(), base + lens[0], l.gu, l.ffn, l.h).unwrap(),
                DeviceWeights::new(buf, base + lens[0] + lens[1], l.down, l.h, l.ffn).unwrap(),
            ]
        })
        .collect();
    (router, experts)
}

/// Runs the routed layer on the device: its outputs, ids, weights, and whether it ran.
fn gpu_layer(
    dev: &VulkanDevice,
    l: Layer,
    router: &DeviceWeights,
    experts: &[[DeviceWeights; 3]],
    xs: &[f32],
    s: usize,
    norm: bool,
) -> Option<(Vec<f32>, Vec<u32>, Vec<f32>)> {
    let (mut ys, mut ids, mut ws) = (
        vec![0f32; s * l.h],
        vec![0u32; s * l.k],
        vec![0f32; s * l.k],
    );
    let ran = dev
        .routed_moe(&mut DeviceRoutedMoe {
            router,
            experts,
            k: l.k,
            renormalise: norm,
            xs,
            s,
            ys: &mut ys,
            ids: &mut ids,
            weights: &mut ws,
            prelude: None,
        })
        .unwrap();
    ran.then_some((ys, ids, ws))
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The whole routed layer in one submission equals the CPU's separate steps, for every
/// dtype pairing, 1 to 8 tokens, with and without renormalisation, with a router tie.
#[test]
fn the_routed_layer_matches_the_cpu() {
    let Some(dev) = device(true) else { return };
    let q4 = DType::Ggml(GgmlType::Q4_K);
    let q6 = DType::Ggml(GgmlType::Q6_K);
    let q8 = DType::Ggml(GgmlType::Q8_0);
    let layers = [
        Layer {
            h: 512,
            ffn: 256,
            experts: 8,
            k: 2,
            router: DType::F32,
            gu: q4,
            down: q4,
        },
        Layer {
            h: 512,
            ffn: 256,
            experts: 16,
            k: 4,
            router: q8,
            gu: q4,
            down: q6,
        },
        Layer {
            h: 256,
            ffn: 512,
            experts: 6,
            k: 6,
            router: DType::F16,
            gu: q6,
            down: q6,
        },
        Layer {
            h: 64,
            ffn: 96,
            experts: 12,
            k: 3,
            router: DType::F32,
            gu: q8,
            down: q8,
        },
        Layer {
            h: 65,
            ffn: 130,
            experts: 5,
            k: 1,
            router: DType::BF16,
            gu: DType::F32,
            down: DType::F16,
        },
    ];
    for (c, &l) in layers.iter().enumerate() {
        for tie in [false, true] {
            let h = host(l, 31 + c as u32, tie);
            let (router, experts) = place(&dev, l, &h, true);
            for s in [1, 2, 4, 8] {
                for norm in [false, true] {
                    let xs = acts(s * l.h, c + s);
                    let what = format!("{l:?} tie {tie}, {s} tokens, renormalise {norm}");
                    let want = cpu_layer(l, &h, &xs, s, norm);
                    let before = dev.transfers().submissions;
                    let got = gpu_layer(&dev, l, &router, &experts, &xs, s, norm)
                        .unwrap_or_else(|| panic!("{what}: not run"));
                    assert_eq!(
                        dev.transfers().submissions - before,
                        1,
                        "{what}: one submission"
                    );
                    assert_eq!(got.1, want.1, "{what}: ids");
                    assert_eq!(bits(&got.2), bits(&want.2), "{what}: weights");
                    assert_eq!(bits(&got.0), bits(&want.0), "{what}: outputs");
                }
            }
        }
    }
}

/// OLMoE's layer shape: 64 experts of 2048 × 1024, top 8, Q4_K gate/up with Q4_K and
/// Q6_K down, at 1 and 4 tokens.
#[test]
fn olmoe_layer_shapes_match_the_cpu() {
    let Some(dev) = device(true) else { return };
    let q4 = DType::Ggml(GgmlType::Q4_K);
    for down in [q4, DType::Ggml(GgmlType::Q6_K)] {
        let l = Layer {
            h: 2048,
            ffn: 1024,
            experts: 64,
            k: 8,
            router: DType::F32,
            gu: q4,
            down,
        };
        let h = host(l, 5, false);
        let (router, experts) = place(&dev, l, &h, true);
        for s in [1, 4] {
            let xs = acts(s * l.h, s);
            let want = cpu_layer(l, &h, &xs, s, false);
            let got = gpu_layer(&dev, l, &router, &experts, &xs, s, false).unwrap();
            assert_eq!(got.1, want.1, "{down}: ids");
            assert_eq!(bits(&got.0), bits(&want.0), "{down}: {s} tokens");
        }
    }
}

/// Declined, with nothing written, when the experts are not in one buffer, when routing
/// is off, or past the token limit; the caller then routes on the host.
#[test]
fn the_routed_layer_is_declined_outside_its_conditions() {
    let Some(dev) = device(true) else { return };
    let l = Layer {
        h: 64,
        ffn: 96,
        experts: 4,
        k: 2,
        router: DType::F32,
        gu: DType::F32,
        down: DType::F32,
    };
    let h = host(l, 3, false);
    let xs = acts(l.h, 1);
    let (router, own) = place(&dev, l, &h, false);
    assert!(
        gpu_layer(&dev, l, &router, &own, &xs, 1, false).is_none(),
        "separate buffers"
    );
    let (router, arena) = place(&dev, l, &h, true);
    assert!(gpu_layer(&dev, l, &router, &arena, &xs, 1, false).is_some());
    let s = dev.fuse_rows() + 1;
    let xs9 = acts(s * l.h, 1);
    assert!(!dev.routes_moe(l.experts, l.k, s));
    assert!(
        gpu_layer(&dev, l, &router, &arena, &xs9, s, false).is_none(),
        "too many tokens"
    );
    let Some(off) = device(false) else { return };
    assert!(!off.routes_moe(l.experts, l.k, 1), "off by default");
}

/// A layer's prelude (Step 3c): the O projection's weights (`h × att`), the attention
/// outputs, the residual stream and the norm weights.
struct Pre {
    dtype: DType,
    att_cols: usize,
    o: Vec<u8>,
    att: Vec<f32>,
    hs: Vec<f32>,
    norm: Vec<f32>,
    eps: f64,
}

fn pre(l: Layer, dtype: DType, att_cols: usize, s: usize, seed: u32, eps: f64) -> Pre {
    let mut r = Rng(seed.wrapping_mul(2_246_822_519).max(1));
    Pre {
        dtype,
        att_cols,
        o: weights(dtype, l.h, att_cols, seed),
        att: acts(s * att_cols, seed as usize + 3),
        // A residual stream of mixed magnitudes, and norm weights near 1.
        hs: (0..s * l.h)
            .map(|_| {
                let v = r.next();
                f32::from_bits((v & 0x807f_ffff) | ((120 + (v >> 24) % 10) << 23))
            })
            .collect(),
        norm: (0..l.h)
            .map(|_| 0.5 + (r.next() >> 8) as f32 / (1u32 << 24) as f32)
            .collect(),
        eps,
    }
}

/// A layer's outputs, ids, weights and residual stream.
type WithPre = (Vec<f32>, Vec<u32>, Vec<f32>, Vec<f32>);

/// The CPU's layer with its prelude: outputs, ids, weights and the residual stream.
fn cpu_layer_pre(l: Layer, h: &Host, p: &Pre, s: usize, norm: bool) -> WithPre {
    let views: Vec<[WeightView<'_>; 3]> = h
        .experts
        .iter()
        .map(|[g, u, d]| {
            [
                WeightView::new(l.gu, l.ffn, l.h, g).unwrap(),
                WeightView::new(l.gu, l.ffn, l.h, u).unwrap(),
                WeightView::new(l.down, l.h, l.ffn, d).unwrap(),
            ]
        })
        .collect();
    let (mut ys, mut ids, mut ws, mut hs) = (
        vec![0f32; s * l.h],
        vec![0u32; s * l.k],
        vec![0f32; s * l.k],
        p.hs.clone(),
    );
    routed_moe_steps(
        &CpuBackend::new(),
        &mut RoutedMoe {
            router: WeightView::new(l.router, l.experts, l.h, &h.router).unwrap(),
            experts: &views,
            k: l.k,
            renormalise: norm,
            xs: &[],
            s,
            ys: &mut ys,
            ids: &mut ids,
            weights: &mut ws,
            prelude: Some(MoePrelude {
                o: WeightView::new(p.dtype, l.h, p.att_cols, &p.o).unwrap(),
                att: &p.att,
                hs: &mut hs,
                norm: &p.norm,
                eps: p.eps,
            }),
        },
    )
    .unwrap();
    (ys, ids, ws, hs)
}

/// The layer with its prelude on the device, if it ran.
fn gpu_layer_pre(
    dev: &VulkanDevice,
    l: Layer,
    router: &DeviceWeights,
    experts: &[[DeviceWeights; 3]],
    p: &Pre,
    s: usize,
    norm: bool,
) -> Option<WithPre> {
    let o =
        DeviceWeights::new(dev.upload(&p.o, None).unwrap(), 0, p.dtype, l.h, p.att_cols).unwrap();
    let (mut ys, mut ids, mut ws, mut hs) = (
        vec![0f32; s * l.h],
        vec![0u32; s * l.k],
        vec![0f32; s * l.k],
        p.hs.clone(),
    );
    let ran = dev
        .routed_moe(&mut DeviceRoutedMoe {
            router,
            experts,
            k: l.k,
            renormalise: norm,
            xs: &[],
            s,
            ys: &mut ys,
            ids: &mut ids,
            weights: &mut ws,
            prelude: Some(DeviceMoePrelude {
                o: &o,
                att: &p.att,
                hs: &mut hs,
                norm: &p.norm,
                eps: p.eps,
            }),
        })
        .unwrap();
    if !ran {
        assert_eq!(
            bits(&hs),
            bits(&p.hs),
            "declined: the residual stream untouched"
        );
    }
    ran.then_some((ys, ids, ws, hs))
}

/// The layer with its prelude in one submission equals the CPU's separate steps (the O
/// matmul, `hs += o`, `rms_norm`, then the layer): outputs, ids, weights and the residual
/// stream, for every dtype pairing, an O projection of each dtype, a width with a short
/// last chunk of the norm's sum (65), 1 to 8 tokens.
#[test]
fn the_routed_layer_with_its_prelude_matches_the_cpu() {
    let Some(dev) = device_with(true, true) else {
        return;
    };
    let q4 = DType::Ggml(GgmlType::Q4_K);
    let q6 = DType::Ggml(GgmlType::Q6_K);
    let q8 = DType::Ggml(GgmlType::Q8_0);
    // (layer, O dtype, attention width)
    let cases = [
        (
            Layer {
                h: 512,
                ffn: 256,
                experts: 8,
                k: 2,
                router: DType::F32,
                gu: q4,
                down: q4,
            },
            q4,
            512,
        ),
        (
            Layer {
                h: 512,
                ffn: 256,
                experts: 16,
                k: 4,
                router: q8,
                gu: q4,
                down: q6,
            },
            q6,
            768,
        ),
        (
            Layer {
                h: 256,
                ffn: 512,
                experts: 6,
                k: 6,
                router: DType::F16,
                gu: q6,
                down: q6,
            },
            q8,
            128,
        ),
        (
            Layer {
                h: 64,
                ffn: 96,
                experts: 12,
                k: 3,
                router: DType::F32,
                gu: q8,
                down: q8,
            },
            DType::F16,
            96,
        ),
        (
            Layer {
                h: 65,
                ffn: 130,
                experts: 5,
                k: 1,
                router: DType::BF16,
                gu: DType::F32,
                down: DType::F16,
            },
            DType::F32,
            70,
        ),
    ];
    for (c, &(l, odt, att)) in cases.iter().enumerate() {
        let h = host(l, 41 + c as u32, c % 2 == 1);
        let (router, experts) = place(&dev, l, &h, true);
        for s in [1, 2, 4, 8] {
            for (norm, eps) in [(false, 1e-5), (true, 1e-6)] {
                let p = pre(l, odt, att, s, 97 + c as u32 + s as u32, eps);
                let what = format!("{l:?} O {odt} x {att}, {s} tokens, renormalise {norm}");
                let want = cpu_layer_pre(l, &h, &p, s, norm);
                let before = dev.transfers().submissions;
                let got = gpu_layer_pre(&dev, l, &router, &experts, &p, s, norm)
                    .unwrap_or_else(|| panic!("{what}: not run"));
                assert_eq!(
                    dev.transfers().submissions - before,
                    1,
                    "{what}: one submission"
                );
                assert_eq!(bits(&got.3), bits(&want.3), "{what}: residual stream");
                assert_eq!(got.1, want.1, "{what}: ids");
                assert_eq!(bits(&got.2), bits(&want.2), "{what}: weights");
                assert_eq!(bits(&got.0), bits(&want.0), "{what}: outputs");
            }
        }
    }
}

/// OLMoE's layer shape with its prelude: a 2048 × 2048 O projection (Q4_K and Q6_K), 64
/// experts of 2048 × 1024, top 8, at 1 and 4 tokens.
#[test]
fn olmoe_layer_shapes_with_the_prelude_match_the_cpu() {
    let Some(dev) = device_with(true, true) else {
        return;
    };
    let q4 = DType::Ggml(GgmlType::Q4_K);
    let q6 = DType::Ggml(GgmlType::Q6_K);
    let l = Layer {
        h: 2048,
        ffn: 1024,
        experts: 64,
        k: 8,
        router: DType::F32,
        gu: q4,
        down: q6,
    };
    let h = host(l, 9, false);
    let (router, experts) = place(&dev, l, &h, true);
    for odt in [q4, q6] {
        for s in [1, 4] {
            let p = pre(l, odt, 2048, s, 13 + s as u32, 1e-5);
            let want = cpu_layer_pre(l, &h, &p, s, false);
            let got = gpu_layer_pre(&dev, l, &router, &experts, &p, s, false).unwrap();
            assert_eq!(bits(&got.3), bits(&want.3), "{odt}, {s} tokens: residual");
            assert_eq!(got.1, want.1, "{odt}, {s} tokens: ids");
            assert_eq!(bits(&got.0), bits(&want.0), "{odt}, {s} tokens: outputs");
        }
    }
}

/// A prelude is declined, with nothing written, when `route_o` is off (Step 3b still
/// runs without one) and when the experts are not in one buffer.
#[test]
fn the_prelude_is_declined_outside_its_conditions() {
    let l = Layer {
        h: 64,
        ffn: 96,
        experts: 4,
        k: 2,
        router: DType::F32,
        gu: DType::F32,
        down: DType::F32,
    };
    let h = host(l, 3, false);
    let p = pre(l, DType::F32, 64, 1, 5, 1e-5);
    let Some(off) = device_with(true, false) else {
        return;
    };
    assert!(off.routes_moe(l.experts, l.k, 1));
    assert!(!off.routes_moe_prelude(l.experts, l.k, 1), "off by default");
    let (router, arena) = place(&off, l, &h, true);
    assert!(gpu_layer_pre(&off, l, &router, &arena, &p, 1, false).is_none());
    let xs = acts(l.h, 1);
    assert!(gpu_layer(&off, l, &router, &arena, &xs, 1, false).is_some());
    let Some(on) = device_with(true, true) else {
        return;
    };
    assert!(on.routes_moe_prelude(l.experts, l.k, 1));
    let (router, own) = place(&on, l, &h, false);
    assert!(
        gpu_layer_pre(&on, l, &router, &own, &p, 1, false).is_none(),
        "separate buffers"
    );
    let Some(no_route) = device_with(false, true) else {
        return;
    };
    assert!(
        !no_route.routes_moe_prelude(l.experts, l.k, 1),
        "needs --vulkan-route"
    );
}
