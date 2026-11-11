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
