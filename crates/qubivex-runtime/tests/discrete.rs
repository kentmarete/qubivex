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

/// As [`sim`], with the device fusing expert blocks of up to `fuse_rows` rows.
fn sim_fused(id: &str, fuse_rows: usize) -> Engine {
    let mut config = SimConfig::new(1 << 30);
    config.fuse_rows = fuse_rows;
    Engine::load(
        meta(id),
        EngineOptions {
            backend: BackendKind::Sim,
            sim: Some(config),
            vram_budget: Some(1 << 29),
            expert_cache_bytes: Some(1 << 30),
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
            // The budget holds every expert: copies go into fixed-slot arenas.
            assert!(r.experts.arena_slots > 0, "{id}: {r:?}");
            assert!(r.experts.held <= r.experts.used, "{id}: {r:?}");
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
        assert!(r.experts.held <= small, "{id}: inclusive: {r:?}");
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

/// A device that fuses the expert block: with every routed expert resident, a layer's
/// experts are one submission (4 per layer plus the head), and the bits are the CPU's.
/// Selection is by shape: with blocks of at most 1 row fused, the prompt (several rows
/// per expert) runs unfused and decode fused, and the tokens are the same; so are
/// concurrent sequences, whose decode passes combine into blocks of several rows.
#[test]
fn fused_expert_blocks_are_one_submission() {
    for id in MOE {
        let logits = |e: &Engine| {
            let mut c = e.new_cache();
            bits(&e.forward(PROMPT, &mut c, Logits::All, None).unwrap())
        };
        let submissions = |e: &Engine| e.device_report().unwrap().transfers.submissions;
        let want = logits(&cpu(id, Some(1 << 30)));
        let want_run = run(&cpu(id, Some(1 << 30)));

        let e = sim_fused(id, 64);
        let _ = logits(&e); // Loads every expert the prompt routes to.
        let before = submissions(&e);
        assert_eq!(logits(&e), want, "{id}");
        assert_eq!(submissions(&e) - before, 4 * layers(id) + 1, "{id}: fused");
        assert_eq!(run(&e), want_run, "{id}");

        let e = sim_fused(id, 1);
        let _ = logits(&e);
        let before = submissions(&e);
        assert_eq!(logits(&e), want, "{id}");
        assert!(
            submissions(&e) - before > 4 * layers(id) + 1,
            "{id}: the prompt's blocks have more than 1 row, so they are not fused"
        );
        assert_eq!(run(&e), want_run, "{id}");
    }

    let id = "olmoe-tiny";
    let prompts: Vec<Vec<u32>> = (0..4)
        .map(|i| PROMPT.iter().map(|&t| t + i).collect())
        .collect();
    let c = cpu(id, None);
    let want: Vec<_> = prompts
        .iter()
        .map(|p| c.generate(p, 8, Sampling::default(), |_| true).unwrap())
        .collect();
    let e = sim_fused(id, 4);
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

/// As [`sim_fused`], with MoE layers of up to `route_rows` tokens also routed on the
/// device (Step 3b); `device_cache` limits the device expert cache.
fn sim_routed(id: &str, route_rows: usize, device_cache: Option<u64>) -> Engine {
    sim_routed_with(id, route_rows, false, device_cache)
}

/// As [`sim_routed`], with each routed layer's prelude (the O projection, residual and
/// norm) in the same submission when `prelude` (Step 3c).
fn sim_routed_with(
    id: &str,
    route_rows: usize,
    prelude: bool,
    device_cache: Option<u64>,
) -> Engine {
    let mut config = SimConfig::new(1 << 30);
    config.fuse_rows = 64;
    config.route_rows = route_rows;
    config.route_prelude = prelude;
    Engine::load(
        meta(id),
        EngineOptions {
            backend: BackendKind::Sim,
            sim: Some(config),
            vram_budget: Some(
                device_cache.map_or(1 << 29, |c| c + VRAM_ACTIVATION_RESERVE.0 + dense_arena(id)),
            ),
            expert_cache_bytes: Some(1 << 30),
            max_context: Some(256),
            ..EngineOptions::default()
        },
    )
    .unwrap()
}

/// Prompt logits, after enough passes that every layer is held whole (missing experts
/// are prefetched, and a pass routes on the host until they land).
fn routed_logits(e: &Engine, id: &str) -> Vec<u32> {
    let logits = |e: &Engine| {
        let mut c = e.new_cache();
        bits(&e.forward(PROMPT, &mut c, Logits::All, None).unwrap())
    };
    for _ in 0..50 {
        let before = e.device_report().unwrap().routed_layers;
        let got = logits(e);
        let after = e.device_report().unwrap().routed_layers;
        if after - before == layers(id) {
            return got;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    panic!("the layers were never all routed on the device");
}

/// Step 3b on the sim: a MoE layer's router, routing, experts and combine are one
/// submission with the CPU's bits, and record the same uses as host routing.
#[test]
fn routed_layers_are_one_submission_with_the_cpu_bits() {
    for id in MOE {
        let want = {
            let mut c = cpu(id, Some(1 << 30)).new_cache();
            bits(
                &cpu(id, Some(1 << 30))
                    .forward(PROMPT, &mut c, Logits::All, None)
                    .unwrap(),
            )
        };
        let want_run = run(&cpu(id, Some(1 << 30)));
        let e = sim_routed(id, 64, None);
        assert_eq!(routed_logits(&e, id), want, "{id}");
        let submissions = |e: &Engine| e.device_report().unwrap().transfers.submissions;
        let before = submissions(&e);
        let mut c = e.new_cache();
        assert_eq!(
            bits(&e.forward(PROMPT, &mut c, Logits::All, None).unwrap()),
            want,
            "{id}"
        );
        assert_eq!(
            submissions(&e) - before,
            3 * layers(id) + 1,
            "{id}: QKV, O and the routed layer per layer, then the head"
        );
        assert_eq!(run(&e), want_run, "{id}");

        // The same uses as routing on the host (Step 3a), once both are warm.
        let f = sim_fused(id, 64);
        for _ in 0..3 {
            let _ = run(&f);
        }
        let uses = |e: &Engine| {
            let (h, d) = (
                e.store_stats().hits,
                e.device_report().unwrap().experts.hits,
            );
            let _ = run(e);
            (
                e.store_stats().hits - h,
                e.device_report().unwrap().experts.hits - d,
            )
        };
        assert_eq!(uses(&e), uses(&f), "{id}: store and device-cache hits");

        // Switched off, nothing is routed on the device.
        let off = sim_routed(id, 0, None);
        assert_eq!(run(&off), want_run, "{id}");
        assert_eq!(off.device_report().unwrap().routed_layers, 0, "{id}");
    }
}

/// Step 3c on the sim: the O projection, residual and FFN norm join the routed
/// submission, two submissions per layer, with the CPU's bits and the same store and
/// device-cache uses as Step 3b.
#[test]
fn routed_layers_with_their_prelude_are_two_submissions_per_layer() {
    for id in MOE {
        let cpu_e = cpu(id, Some(1 << 30));
        let want = {
            let mut c = cpu_e.new_cache();
            bits(&cpu_e.forward(PROMPT, &mut c, Logits::All, None).unwrap())
        };
        let want_run = run(&cpu_e);
        let e = sim_routed_with(id, 64, true, None);
        assert_eq!(routed_logits(&e, id), want, "{id}");
        let submissions = |e: &Engine| e.device_report().unwrap().transfers.submissions;
        let before = submissions(&e);
        let mut c = e.new_cache();
        assert_eq!(
            bits(&e.forward(PROMPT, &mut c, Logits::All, None).unwrap()),
            want,
            "{id}"
        );
        assert_eq!(
            submissions(&e) - before,
            2 * layers(id) + 1,
            "{id}: QKV and the routed layer with its prelude per layer, then the head"
        );
        assert_eq!(run(&e), want_run, "{id}");

        // The same uses as Step 3b, once both are warm.
        let b = sim_routed(id, 64, None);
        let _ = routed_logits(&b, id);
        let uses = |e: &Engine| {
            let (h, d) = (
                e.store_stats().hits,
                e.device_report().unwrap().experts.hits,
            );
            let _ = run(e);
            (
                e.store_stats().hits - h,
                e.device_report().unwrap().experts.hits - d,
            )
        };
        assert_eq!(uses(&e), uses(&b), "{id}: store and device-cache hits");

        // A device cache short of the model routes on the host, prelude or not.
        let short = sim_routed_with(id, 64, true, Some(2 * largest_expert(id)));
        assert_eq!(run(&short), want_run, "{id}");
        assert_eq!(short.device_report().unwrap().routed_layers, 0, "{id}");
    }
}

/// Concurrent requests routed with their prelude: each gets the CPU's prompt logits and
/// tokens.
#[test]
fn concurrent_requests_routed_with_their_prelude() {
    let id = "olmoe-tiny";
    let prompts: Vec<Vec<u32>> = (0..4)
        .map(|i| PROMPT.iter().map(|&t| t + i).collect())
        .collect();
    let both = |e: &Engine, p: &[u32]| {
        let mut c = e.new_cache();
        let logits = bits(&e.forward(p, &mut c, Logits::All, None).unwrap());
        (
            logits,
            e.generate(p, 8, Sampling::default(), |_| true).unwrap(),
        )
    };
    let c = cpu(id, Some(1 << 30));
    let want: Vec<_> = prompts.iter().map(|p| both(&c, p)).collect();
    let e = sim_routed_with(id, 64, true, None);
    let _ = routed_logits(&e, id);
    let before = e.device_report().unwrap().routed_layers;
    let got: Vec<_> = std::thread::scope(|s| {
        let hs: Vec<_> = prompts
            .iter()
            .map(|p| {
                let e = &e;
                s.spawn(move || both(e, p))
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(got, want);
    assert!(e.device_report().unwrap().routed_layers > before);
}

/// A device cache that cannot hold every expert never routes on the device (holding a
/// layer must not evict), and keeps the CPU's bits.
#[test]
fn a_device_cache_short_of_the_model_routes_on_the_host() {
    for id in MOE {
        let want_run = run(&cpu(id, Some(1 << 30)));
        let e = sim_routed(id, 64, Some(2 * largest_expert(id)));
        for _ in 0..3 {
            assert_eq!(run(&e), want_run, "{id}");
        }
        assert_eq!(e.device_report().unwrap().routed_layers, 0, "{id}");
    }
}

/// Concurrent requests routed on the device: each gets the CPU's prompt logits and
/// tokens.
#[test]
fn concurrent_requests_routed_on_the_device() {
    let id = "olmoe-tiny";
    let prompts: Vec<Vec<u32>> = (0..4)
        .map(|i| PROMPT.iter().map(|&t| t + i).collect())
        .collect();
    let both = |e: &Engine, p: &[u32]| {
        let mut c = e.new_cache();
        let logits = bits(&e.forward(p, &mut c, Logits::All, None).unwrap());
        (
            logits,
            e.generate(p, 8, Sampling::default(), |_| true).unwrap(),
        )
    };
    let c = cpu(id, Some(1 << 30));
    let want: Vec<_> = prompts.iter().map(|p| both(&c, p)).collect();
    let e = sim_routed(id, 64, None);
    let _ = routed_logits(&e, id);
    let before = e.device_report().unwrap().routed_layers;
    for _ in 0..2 {
        let got: Vec<_> = std::thread::scope(|s| {
            let hs: Vec<_> = prompts
                .iter()
                .map(|p| {
                    let e = &e;
                    s.spawn(move || both(e, p))
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(got, want);
    }
    assert!(e.device_report().unwrap().routed_layers > before);
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
