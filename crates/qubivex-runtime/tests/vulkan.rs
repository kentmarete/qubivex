//! ADR-019 on the fixtures: a Vulkan GPU as the discrete device gives the CPU's bits,
//! with every matmul on the device and its memory in the ledger's VRAM tier.
//!
//! Each test skips (passes, with a note) when no Vulkan device opens, unless
//! `QUBIVEX_REQUIRE_VULKAN=1`. Locally the device is MoltenVK on the Apple GPU.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use qubivex_arch::Logits;
use qubivex_memory::{Tier, VRAM_ACTIVATION_RESERVE};
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{BackendKind, Engine, EngineOptions, Sampling, VulkanConfig};

fn dir(id: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/models")
        .join(id)
}

fn meta(id: &str) -> ModelMetadata {
    ModelMetadata::load(&dir(id), &ArchitectureRegistry::with_builtins()).unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

const MOE: [&str; 3] = ["olmoe-tiny", "olmoe-tiny-gguf", "qwen3-moe-tiny"];
const ALL: [&str; 4] = [
    "olmoe-tiny",
    "olmoe-tiny-gguf",
    "qwen3-moe-tiny",
    "dense-tiny",
];
const PROMPT: &[u32] = &[
    72, 101, 108, 108, 111, 44, 32, 101, 120, 112, 101, 114, 116, 115,
];

/// Prompt logits and 12 greedy tokens.
fn run(e: &Engine) -> (Vec<u32>, Vec<u32>) {
    let mut c = e.new_cache();
    let logits = bits(&e.forward(PROMPT, &mut c, Logits::All, None).unwrap());
    let toks = e
        .generate(PROMPT, 12, Sampling::default(), |_| true)
        .unwrap();
    (logits, toks)
}

fn load(meta: ModelMetadata, opts: EngineOptions) -> Option<Engine> {
    match Engine::load(meta, opts) {
        Ok(e) => Some(e),
        Err(e) if e.to_string().contains("cannot open the Vulkan device") => {
            assert!(
                std::env::var("QUBIVEX_REQUIRE_VULKAN").as_deref() != Ok("1"),
                "QUBIVEX_REQUIRE_VULKAN=1 but: {e}"
            );
            eprintln!("skipped: {e}");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

fn cpu(id: &str, cache: Option<u64>) -> Engine {
    Engine::load(
        meta(id),
        EngineOptions {
            expert_cache_bytes: cache,
            max_context: Some(256),
            ..EngineOptions::default()
        },
    )
    .unwrap()
}

/// A Vulkan engine on a 1 GiB slice of the device. `device_cache`: the device expert
/// cache's bytes, with a fixed RAM cache.
fn vulkan(id: &str, ram_cache: Option<u64>, device_cache: Option<u64>) -> Option<Engine> {
    let vram_budget = device_cache.map(|c| c + VRAM_ACTIVATION_RESERVE.0 + meta(id).bytes().dense);
    load(
        meta(id),
        EngineOptions {
            backend: BackendKind::Vulkan,
            vulkan: VulkanConfig {
                capacity: Some(1 << 30),
                ..VulkanConfig::default()
            },
            vram_budget,
            expert_cache_bytes: ram_cache,
            max_context: Some(256),
            ..EngineOptions::default()
        },
    )
}

#[test]
fn vulkan_matches_the_cpu_bits() {
    for id in ALL {
        let want = run(&cpu(id, None));
        let Some(e) = vulkan(id, None, None) else {
            return;
        };
        let p = e.placement();
        assert_eq!(p.backend, BackendKind::Vulkan);
        assert!(p.device.as_deref().unwrap().contains("Vulkan"), "{p:?}");
        assert_eq!(run(&e), want, "{id}");

        let r = e.device_report().unwrap();
        let v = e.ledger().snapshot(Tier::Vram(0));
        assert_eq!(v.dense.0, p.bytes, "{id}: the dense copy is charged");
        assert_eq!(v.total().0, r.usage.used, "{id}: ledger = device memory");
        assert!(r.transfers.submissions > 0, "{id}");
        assert_eq!(r.usage.refused, 0, "{id}");
        if MOE.contains(&id) {
            assert!(r.experts.misses > 0, "{id}: experts were copied");
            assert_eq!(r.transfers.uploads, 1 + r.experts.misses, "{id}");
        } else {
            assert_eq!(r.transfers.uploads, 1, "{id}: only the dense copy");
        }
        assert_eq!(e.memory_plan().unwrap().vram.len(), 1, "{id}");
    }
}

/// A device cache of about one expert evicts and copies again, with the same bits.
#[test]
fn a_small_device_cache_evicts_without_changing_bits() {
    for id in MOE {
        let want = run(&cpu(id, Some(1 << 30)));
        let one = meta(id).bytes().largest_expert;
        let Some(e) = vulkan(id, Some(1 << 30), Some(one)) else {
            return;
        };
        assert_eq!(run(&e), want, "{id}");
        assert!(e.device_report().unwrap().experts.evictions > 0, "{id}");
    }
}

/// Four requests at once share the device (submissions are serialised on its queue).
#[test]
fn concurrent_requests_match_the_cpu() {
    let id = "olmoe-tiny-gguf";
    let prompts: [&[u32]; 4] = [&[72, 101], &[108, 108, 111], &[44, 32, 101], &[120]];
    let c = cpu(id, None);
    let want: Vec<_> = prompts
        .iter()
        .map(|p| c.generate(p, 8, Sampling::default(), |_| true).unwrap())
        .collect();
    drop(c);
    let Some(e) = vulkan(id, None, None) else {
        return;
    };
    let got: Vec<_> = std::thread::scope(|s| {
        let hs: Vec<_> = prompts
            .iter()
            .map(|p| {
                let e = &e;
                s.spawn(move || e.generate(p, 8, Sampling::default(), |_| true).unwrap())
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(got, want);
}

/// The real OLMoE-1B-7B-0125-Instruct Q4_K_M (set `QUBIVEX_OLMOE_GGUF_DIR`): the
/// Vulkan device gives the CPU's prompt logits and Colibri's greedy tokens, with a
/// device expert cache of 64 experts, so it evicts.
#[test]
fn olmoe_on_vulkan_matches_the_cpu() {
    let Ok(dir) = std::env::var("QUBIVEX_OLMOE_GGUF_DIR") else {
        eprintln!("skipping: QUBIVEX_OLMOE_GGUF_DIR not set");
        return;
    };
    // Colibri `c/ref_olmoe_real.json`, as in real_model.rs.
    const PROMPT: [u32; 5] = [510, 5347, 273, 6181, 310];
    const COLIBRI: [u32; 12] = [
        7785, 15, 187, 187, 510, 3565, 3448, 273, 6181, 310, 5112, 15,
    ];
    let meta =
        || ModelMetadata::load(Path::new(&dir), &ArchitectureRegistry::with_builtins()).unwrap();
    let opts = |backend, vram_budget| EngineOptions {
        backend,
        vram_budget,
        expert_cache_bytes: Some(4 << 30),
        max_context: Some(64),
        ..EngineOptions::default()
    };
    let logits = |e: &Engine| {
        let mut c = e.new_cache();
        bits(&e.forward(&PROMPT, &mut c, Logits::All, None).unwrap())
    };
    let cpu = Engine::load(meta(), opts(BackendKind::Cpu, None)).unwrap();
    let want = logits(&cpu);
    drop(cpu);
    let m = meta();
    let budget = 64 * m.bytes().largest_expert + VRAM_ACTIVATION_RESERVE.0 + m.bytes().dense;
    let Some(e) = load(m, opts(BackendKind::Vulkan, Some(budget))) else {
        return;
    };
    eprintln!("{}", e.placement().device.as_deref().unwrap_or(""));
    assert!(logits(&e) == want, "prompt logits differ from the CPU");
    let toks = e
        .generate(&PROMPT, 12, Sampling::default(), |_| true)
        .unwrap();
    assert_eq!(toks, COLIBRI);
    let r = e.device_report().unwrap();
    eprintln!("{r:?}");
    assert!(r.experts.evictions > 0, "{r:?}");
}

/// Step 3b: a Vulkan engine routing MoE layers on the device, with a device expert
/// cache that holds every expert; with `route_o`, each routed layer's prelude (the O
/// projection, residual and FFN norm) in the same submission (Step 3c).
fn vulkan_routed(id: &str, route_o: bool) -> Option<Engine> {
    load(
        meta(id),
        EngineOptions {
            backend: BackendKind::Vulkan,
            vulkan: VulkanConfig {
                capacity: Some(1 << 30),
                route: true,
                route_o,
                ..VulkanConfig::default()
            },
            expert_cache_bytes: Some(1 << 30),
            max_context: Some(256),
            ..EngineOptions::default()
        },
    )
}

/// Passes `f` until one routes every layer on the device (missing experts are
/// prefetched; until they land a pass routes on the host); returns that pass's value.
fn when_routed<T>(e: &Engine, layers: u64, mut f: impl FnMut() -> T) -> T {
    for _ in 0..100 {
        let before = e.device_report().unwrap().routed_layers;
        let v = f();
        if e.device_report().unwrap().routed_layers - before >= layers {
            return v;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    panic!("never routed every layer on the device");
}

/// Routing on the device (Step 3b), and with each layer's prelude (Step 3c), gives the
/// CPU's prompt logits and tokens on every MoE fixture, alone and with concurrent
/// requests; with the prelude, a pass is two submissions per layer and the head.
#[test]
fn routed_on_vulkan_matches_the_cpu_bits() {
    const SHORT: &[u32] = &[72, 101, 108, 108, 111];
    for (id, route_o) in MOE.iter().flat_map(|id| [(*id, false), (*id, true)]) {
        let c = cpu(id, Some(1 << 30));
        let mut cache = c.new_cache();
        let want = bits(&c.forward(SHORT, &mut cache, Logits::All, None).unwrap());
        let want_run = run(&c);
        drop(c);
        let Some(e) = vulkan_routed(id, route_o) else {
            return;
        };
        let layers = u64::from(meta(id).config.dims.layers);
        let got = when_routed(&e, layers, || {
            let mut cache = e.new_cache();
            bits(&e.forward(SHORT, &mut cache, Logits::All, None).unwrap())
        });
        assert_eq!(got, want, "{id}, prelude {route_o}: prompt logits");
        if route_o {
            let submissions = |e: &Engine| e.device_report().unwrap().transfers.submissions;
            let before = submissions(&e);
            let got = when_routed(&e, layers, || {
                let mut cache = e.new_cache();
                bits(&e.forward(SHORT, &mut cache, Logits::All, None).unwrap())
            });
            assert_eq!(got, want, "{id}: prompt logits");
            assert_eq!(
                submissions(&e) - before,
                2 * layers + 1,
                "{id}: QKV and the routed layer with its prelude per layer, then the head"
            );
            assert!(
                e.placement()
                    .device
                    .as_deref()
                    .unwrap()
                    .contains("O projection"),
                "{id}"
            );
        }
        assert_eq!(run(&e), want_run, "{id}");
        assert!(
            e.placement().device.as_deref().unwrap().contains("routed"),
            "{id}"
        );

        let prompts: [&[u32]; 4] = [&[72, 101], &[108, 108, 111], &[44, 32, 101], &[120]];
        let c = cpu(id, Some(1 << 30));
        let want: Vec<_> = prompts
            .iter()
            .map(|p| c.generate(p, 8, Sampling::default(), |_| true).unwrap())
            .collect();
        drop(c);
        let before = e.device_report().unwrap().routed_layers;
        let got: Vec<_> = std::thread::scope(|s| {
            let hs: Vec<_> = prompts
                .iter()
                .map(|p| {
                    let e = &e;
                    s.spawn(move || e.generate(p, 8, Sampling::default(), |_| true).unwrap())
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(got, want, "{id}: concurrent");
        assert!(e.device_report().unwrap().routed_layers > before, "{id}");
    }
}

/// The real OLMoE (set `QUBIVEX_OLMOE_GGUF_DIR`) routed on the device, every expert
/// held there, without and with the prelude (Step 3c): the CPU's prompt logits and
/// Colibri's greedy tokens.
#[test]
fn olmoe_routed_on_vulkan_matches_the_cpu() {
    let Ok(dir) = std::env::var("QUBIVEX_OLMOE_GGUF_DIR") else {
        eprintln!("skipping: QUBIVEX_OLMOE_GGUF_DIR not set");
        return;
    };
    const PROMPT: [u32; 5] = [510, 5347, 273, 6181, 310];
    const COLIBRI: [u32; 12] = [
        7785, 15, 187, 187, 510, 3565, 3448, 273, 6181, 310, 5112, 15,
    ];
    let meta =
        || ModelMetadata::load(Path::new(&dir), &ArchitectureRegistry::with_builtins()).unwrap();
    let logits = |e: &Engine| {
        let mut c = e.new_cache();
        bits(&e.forward(&PROMPT, &mut c, Logits::All, None).unwrap())
    };
    let cpu = Engine::load(
        meta(),
        EngineOptions {
            expert_cache_bytes: Some(4 << 30),
            max_context: Some(64),
            ..EngineOptions::default()
        },
    )
    .unwrap();
    let want = logits(&cpu);
    drop(cpu);
    for (route_o, attention) in [(false, false), (true, false), (true, true)] {
        let Some(e) = load(
            meta(),
            EngineOptions {
                backend: BackendKind::Vulkan,
                vulkan: VulkanConfig {
                    route: true,
                    route_o,
                    attention,
                    kv_pool: attention.then_some(256 << 20),
                    ..VulkanConfig::default()
                },
                expert_cache_bytes: Some(4 << 30),
                max_context: Some(64),
                ..EngineOptions::default()
            },
        ) else {
            return;
        };
        eprintln!("{}", e.placement().device.as_deref().unwrap_or(""));
        let layers = u64::from(meta().config.dims.layers);
        let got = when_routed(&e, layers, || logits(&e));
        assert!(got == want, "prompt logits differ from the CPU");
        let before = e.device_report().unwrap().routed_layers;
        let toks = e
            .generate(&PROMPT, 12, Sampling::default(), |_| true)
            .unwrap();
        assert_eq!(toks, COLIBRI);
        let r = e.device_report().unwrap();
        eprintln!("{r:?}");
        assert!(r.routed_layers - before >= 12 * layers, "{r:?}");
        if attention {
            assert!(
                r.kv_pool.layers >= 11 * layers,
                "decode attention on the device: {r:?}"
            );
        }
    }
}

/// A prompt short enough to route on the device (at most `fuse_rows` tokens), to warm
/// the device expert cache.
const WARM: &[u32] = &[72, 101, 108, 108, 111];

/// A Vulkan engine routing with the prelude and decode attention on the device (audit
/// §21), with a 64 MiB KV pool and room for `context` positions.
fn vulkan_attention(id: &str, context: usize) -> Option<Engine> {
    load(
        meta(id),
        EngineOptions {
            backend: BackendKind::Vulkan,
            vulkan: VulkanConfig {
                capacity: Some(1 << 30),
                route: true,
                route_o: true,
                attention: true,
                kv_pool: Some(64 << 20),
                ..VulkanConfig::default()
            },
            expert_cache_bytes: Some(1 << 30),
            max_context: Some(context),
            ..EngineOptions::default()
        },
    )
}

fn cpu_context(id: &str, context: usize) -> Engine {
    Engine::load(
        meta(id),
        EngineOptions {
            expert_cache_bytes: Some(1 << 30),
            max_context: Some(context),
            ..EngineOptions::default()
        },
    )
    .unwrap()
}

/// The prompt's last logits, then `steps` greedy decode steps' logits on one cache.
fn decode_logits(e: &Engine, prompt: &[u32], steps: usize) -> Vec<Vec<u32>> {
    let mut c = e.new_cache();
    let mut out = Vec::with_capacity(steps + 1);
    let mut logits = e.forward(prompt, &mut c, Logits::Last, None).unwrap();
    for _ in 0..steps {
        out.push(bits(&logits));
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
            .map(|(i, _)| u32::try_from(i).unwrap())
            .unwrap();
        logits = e.forward(&[next], &mut c, Logits::Last, None).unwrap();
    }
    out.push(bits(&logits));
    out
}

/// Decode attention on Vulkan (audit §21): every decode step's logits are the CPU's on
/// every MoE fixture (Qwen3's per-head Q and K norms decline to the host), a decode
/// layer is one submission, and concurrent requests keep the CPU's tokens.
#[test]
fn decode_attention_on_vulkan_matches_the_cpu_bits() {
    for id in MOE {
        let c = cpu_context(id, 256);
        let want = decode_logits(&c, PROMPT, 12);
        let prompts: [&[u32]; 4] = [&[72, 101], &[108, 108, 111], &[44, 32, 101], &[120]];
        let want_conc: Vec<_> = prompts
            .iter()
            .map(|p| c.generate(p, 12, Sampling::default(), |_| true).unwrap())
            .collect();
        drop(c);
        let Some(e) = vulkan_attention(id, 256) else {
            return;
        };
        let layers = u64::from(meta(id).config.dims.layers);
        let _ = when_routed(&e, layers, || {
            let mut cache = e.new_cache();
            bits(&e.forward(WARM, &mut cache, Logits::All, None).unwrap())
        });
        let before = e.device_report().unwrap();
        assert_eq!(decode_logits(&e, PROMPT, 12), want, "{id}");
        let after = e.device_report().unwrap();
        let ran = after.kv_pool.layers - before.kv_pool.layers;
        if id == "qwen3-moe-tiny" {
            assert_eq!(ran, 0, "{id}: per-head norms run on the host");
        } else {
            assert_eq!(ran, 12 * layers, "{id}: {after:?}");
            assert_eq!(after.kv_pool.free, after.kv_pool.blocks, "{id}");
            // One decode step: one submission per layer, then the head.
            let mut cache = e.new_cache();
            let _ = e.forward(PROMPT, &mut cache, Logits::Last, None).unwrap();
            let n = e.device_report().unwrap().transfers.submissions;
            let _ = e.forward(&[101], &mut cache, Logits::Last, None).unwrap();
            assert_eq!(
                e.device_report().unwrap().transfers.submissions - n,
                layers + 1,
                "{id}"
            );
            assert!(
                e.placement()
                    .device
                    .as_deref()
                    .unwrap()
                    .contains("decode attention"),
                "{id}"
            );
        }
        let got: Vec<_> = std::thread::scope(|s| {
            let hs: Vec<_> = prompts
                .iter()
                .map(|p| {
                    let e = &e;
                    s.spawn(move || e.generate(p, 12, Sampling::default(), |_| true).unwrap())
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(got, want_conc, "{id}: concurrent");
    }
}

/// Decode across KV pool blocks: a 250-token prompt (prefill on the host, its rows copied
/// into the pool) then decode steps past position 256, into a second block, with the
/// CPU's logits at every step.
#[test]
fn decode_attention_on_vulkan_crosses_blocks() {
    let id = "olmoe-tiny";
    let prompt: Vec<u32> = (0..250u32).map(|i| 32 + (i * 7) % 90).collect();
    let want = decode_logits(&cpu_context(id, 600), &prompt, 20);
    let Some(e) = vulkan_attention(id, 600) else {
        return;
    };
    let layers = u64::from(meta(id).config.dims.layers);
    let _ = when_routed(&e, layers, || {
        let mut cache = e.new_cache();
        bits(&e.forward(WARM, &mut cache, Logits::All, None).unwrap())
    });
    let before = e.device_report().unwrap().kv_pool;
    assert_eq!(decode_logits(&e, &prompt, 20), want);
    let after = e.device_report().unwrap().kv_pool;
    assert_eq!(after.layers - before.layers, 20 * layers, "{after:?}");
    assert_eq!(
        after.synced_rows - before.synced_rows,
        250 * layers,
        "{after:?}"
    );
}
