//! ADR-018 on the fixtures: the simulated discrete GPU gives the CPU's bits, holds its
//! own copy of every weight it computes with, and accounts for its memory in the
//! ledger's VRAM tier, however small its expert cache and the RAM cache it copies from.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use qubivex_arch::Logits;
use qubivex_memory::{Tier, VRAM_ACTIVATION_RESERVE};
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{BackendKind, Engine, EngineOptions, Sampling, SimConfig};

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

/// A sim engine. `ram_cache`: a fixed RAM expert cache (no plan); `device_cache`: the
/// device expert cache's bytes, with a fixed RAM cache only.
fn sim(id: &str, ram_cache: Option<u64>, device_cache: Option<u64>) -> Engine {
    let capacity = 1 << 30;
    let vram_budget = device_cache.map(|c| c + VRAM_ACTIVATION_RESERVE.0 + dense_arena(id));
    Engine::load(
        meta(id),
        EngineOptions {
            backend: BackendKind::Sim,
            sim: Some(SimConfig::new(capacity)),
            vram_budget,
            expert_cache_bytes: ram_cache,
            max_context: Some(256),
            ..EngineOptions::default()
        },
    )
    .unwrap()
}

/// The dense copy's bytes, read from a sim engine's report.
fn dense_arena(id: &str) -> u64 {
    let e = Engine::load(
        meta(id),
        EngineOptions {
            backend: BackendKind::Sim,
            max_context: Some(256),
            expert_cache_bytes: Some(1 << 20),
            ..EngineOptions::default()
        },
    )
    .unwrap();
    e.placement().bytes
}

#[test]
fn sim_matches_the_cpu_bits() {
    for id in ALL {
        let want = run(&cpu(id, None));
        let e = sim(id, None, None);
        let p = e.placement();
        assert_eq!(p.backend, BackendKind::Sim);
        assert!(p.tensors > 0 && p.bytes > 0, "{id}: {p:?}");
        assert_eq!(run(&e), want, "{id}");

        let r = e.device_report().unwrap();
        let v = e.ledger().snapshot(Tier::Vram(0));
        assert_eq!(v.dense.0, p.bytes, "{id}: the dense copy is charged");
        assert_eq!(v.total().0, r.usage.used, "{id}: ledger = device memory");
        assert!(r.transfers.submissions > 0, "{id}");
        assert_eq!(r.usage.refused, 0, "{id}");
        if MOE.contains(&id) {
            assert!(r.experts.misses > 0, "{id}: experts were copied");
            assert!(r.experts.hits > 0, "{id}: and reused");
            assert_eq!(v.expert_cache.0, r.experts.used, "{id}");
            assert_eq!(r.transfers.uploads, 1 + r.experts.misses, "{id}");
        } else {
            assert_eq!(r.transfers.uploads, 1, "{id}: only the dense copy");
        }
        let plan = e.memory_plan().unwrap();
        assert_eq!(plan.vram.len(), 1, "{id}: a VRAM budget was planned");
    }
}

/// A device cache that holds about one expert: copies are evicted and made again, and
/// the bits do not change.
#[test]
fn a_small_device_cache_evicts_without_changing_bits() {
    for id in MOE {
        let want = run(&cpu(id, Some(1 << 30)));
        let one = largest_expert(id);
        let e = sim(id, Some(1 << 30), Some(one));
        assert_eq!(run(&e), want, "{id}");
        let r = e.device_report().unwrap();
        assert_eq!(r.experts.budget, one, "{id}");
        assert!(r.experts.evictions > 0, "{id}: {r:?}");
        assert!(
            r.experts.used <= one || r.experts.over_budget > 0,
            "{id}: {r:?}"
        );
        assert_eq!(
            e.ledger().snapshot(Tier::Vram(0)).expert_cache.0,
            r.experts.used,
            "{id}"
        );
    }
}

/// A RAM cache too small for the routed set: slabs are reused for other experts, which
/// drops their device copies (the device cache stays inclusive), and the bits hold.
#[test]
fn ram_cache_reuse_drops_device_copies() {
    for id in MOE {
        let small = largest_expert(id) * 4 * layers(id);
        let want = run(&cpu(id, Some(small)));
        let e = sim(id, Some(small), Some(1 << 28));
        assert_eq!(run(&e), want, "{id}");
        let r = e.device_report().unwrap();
        assert!(r.experts.detached > 0, "{id}: {r:?}");
        assert!(r.experts.used <= small, "{id}: inclusive: {r:?}");
    }
}

/// With every routed expert resident, a layer's experts are two submissions (every
/// gate and up, then every down): a pass is 5 per layer (q/k/v, output, router, two
/// for the experts) plus the head. With a RAM cache too small to hold a layer's
/// routed set, the experts go one by one, and the bits are the same either way.
#[test]
fn resident_experts_are_grouped_into_two_submissions() {
    for id in MOE {
        let logits = |e: &Engine| {
            let mut c = e.new_cache();
            bits(&e.forward(PROMPT, &mut c, Logits::All, None).unwrap())
        };
        let submissions = |e: &Engine| e.device_report().unwrap().transfers.submissions;
        let pass = |e: &Engine| {
            let before = submissions(e);
            let l = logits(e);
            (l, submissions(e) - before)
        };
        let want = logits(&cpu(id, Some(1 << 30)));

        let e = sim(id, Some(1 << 30), Some(1 << 28));
        let _ = pass(&e); // Loads every expert the prompt routes to.
        let (got, n) = pass(&e);
        assert_eq!(got, want, "{id}");
        assert_eq!(n, 5 * layers(id) + 1, "{id}: grouped");

        let small = largest_expert(id) * 4 * layers(id);
        let e = sim(id, Some(small), Some(1 << 28));
        let _ = pass(&e);
        let (got, n) = pass(&e);
        assert_eq!(got, want, "{id}");
        assert!(n > 5 * layers(id) + 1, "{id}: per expert: {n}");
    }
}

/// Concurrent requests share the device and its cache; each gets the CPU's tokens.
#[test]
fn concurrent_requests_on_one_device() {
    let id = "olmoe-tiny";
    let prompts: Vec<Vec<u32>> = (0..4)
        .map(|i| PROMPT.iter().map(|&t| t + i).collect())
        .collect();
    let c = cpu(id, None);
    let want: Vec<_> = prompts
        .iter()
        .map(|p| c.generate(p, 8, Sampling::default(), |_| true).unwrap())
        .collect();
    let e = sim(id, Some(1 << 30), Some(2 * largest_expert(id)));
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
    let r = e.device_report().unwrap().experts;
    assert!(r.used <= r.budget || r.over_budget > 0, "{r:?}");
}

/// A device too small for the dense copy refuses to load, with the reason.
#[test]
fn a_device_without_room_refuses_to_load() {
    let e = Engine::load(
        meta("olmoe-tiny"),
        EngineOptions {
            backend: BackendKind::Sim,
            sim: Some(SimConfig::new(4096)),
            expert_cache_bytes: Some(1 << 20),
            max_context: Some(256),
            ..EngineOptions::default()
        },
    )
    .unwrap_err();
    assert!(e.to_string().contains("out of memory"), "{e}");

    // With a plan, the planner refuses first.
    let e = Engine::load(
        meta("olmoe-tiny"),
        EngineOptions {
            backend: BackendKind::Sim,
            sim: Some(SimConfig::new(4096)),
            max_context: Some(256),
            ..EngineOptions::default()
        },
    )
    .unwrap_err();
    assert!(e.to_string().contains("device 0"), "{e}");
}

fn largest_expert(id: &str) -> u64 {
    meta(id).bytes().largest_expert
}

fn layers(id: &str) -> u64 {
    u64::from(meta(id).config.dims.layers)
}

/// The real OLMoE-1B-7B-0125-Instruct Q4_K_M (set `QUBIVEX_OLMOE_GGUF_DIR`): the sim
/// gives the CPU's prompt logits and Colibri's greedy tokens, with a device expert
/// cache of 64 experts, far below the 1024 the RAM cache holds, so it evicts.
#[test]
fn olmoe_on_the_sim_matches_the_cpu() {
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
    let load = |backend, vram_budget| {
        Engine::load(
            meta(),
            EngineOptions {
                backend,
                sim: Some(SimConfig::new(8 << 30)),
                vram_budget,
                expert_cache_bytes: Some(4 << 30),
                max_context: Some(64),
                ..EngineOptions::default()
            },
        )
        .unwrap()
    };
    let logits = |e: &Engine| {
        let mut c = e.new_cache();
        bits(&e.forward(&PROMPT, &mut c, Logits::All, None).unwrap())
    };
    let cpu = load(BackendKind::Cpu, None);
    let want = logits(&cpu);
    drop(cpu);
    let expert = meta().bytes().largest_expert;
    let e = load(
        BackendKind::Sim,
        Some(64 * expert + VRAM_ACTIVATION_RESERVE.0 + (400 << 20)),
    );
    assert!(logits(&e) == want, "prompt logits differ from the CPU");
    let toks = e
        .generate(&PROMPT, 12, Sampling::default(), |_| true)
        .unwrap();
    assert_eq!(toks, COLIBRI);
    let r = e.device_report().unwrap();
    eprintln!("{r:?}");
    assert!(r.experts.evictions > 0, "{r:?}");
}
