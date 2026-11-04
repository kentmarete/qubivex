//! Invariant 1 on the fixtures: batching and the expert cache size never change bits.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use qubivex_arch::Logits;
use qubivex_arch::{TraceEvent, TracePoint};
use qubivex_expert_stream::predict::LookaheadConfig;
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{Engine, EngineOptions, ExpertIo, IoPath, PrefetchOptions, Sampling};

fn dir(id: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/models")
        .join(id)
}

fn engine(id: &str, cache: Option<u64>) -> Engine {
    let meta = ModelMetadata::load(&dir(id), &ArchitectureRegistry::with_builtins()).unwrap();
    Engine::load(
        meta,
        EngineOptions {
            expert_cache_bytes: cache,
            max_context: Some(256),
            ..EngineOptions::default()
        },
    )
    .unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

const IDS: [&str; 3] = ["olmoe-tiny", "olmoe-tiny-gguf", "qwen3-moe-tiny"];
/// The mixture-of-experts fixtures and the dense one, for invariants that do not
/// involve routing.
const ALL: [&str; 4] = [
    "olmoe-tiny",
    "olmoe-tiny-gguf",
    "qwen3-moe-tiny",
    "dense-tiny",
];
const PROMPT: &[u32] = &[
    72, 101, 108, 108, 111, 44, 32, 101, 120, 112, 101, 114, 116, 115,
];

#[test]
fn prefill_equals_token_by_token() {
    for id in ALL {
        let e = engine(id, None);
        let mut c = e.new_cache();
        let batch = e.forward(PROMPT, &mut c, Logits::All, None).unwrap();
        let mut c2 = e.new_cache();
        let vocab = batch.len() / PROMPT.len();
        for (t, tok) in PROMPT.iter().enumerate() {
            let one = e.forward(&[*tok], &mut c2, Logits::Last, None).unwrap();
            assert_eq!(
                bits(&one),
                bits(&batch[t * vocab..(t + 1) * vocab]),
                "{id} token {t}"
            );
        }
        assert!(batch.iter().all(|v| v.is_finite()), "{id}");
    }
}

#[test]
fn expert_cache_capacity_never_changes_bits() {
    for id in IDS {
        let full = engine(id, None);
        let largest = full.metadata().bytes().largest_expert;
        let mut want_logits = None;
        let mut want_tokens = None;
        for cap in [Some(largest), Some(2 * largest), Some(5 * largest), None] {
            let e = engine(id, cap);
            let mut c = e.new_cache();
            let logits = e.forward(PROMPT, &mut c, Logits::All, None).unwrap();
            let toks = e
                .generate(PROMPT, 12, Sampling::default(), |_| true)
                .unwrap();
            match (&want_logits, &want_tokens) {
                (None, None) => {
                    want_logits = Some(bits(&logits));
                    want_tokens = Some(toks);
                }
                (Some(l), Some(t)) => {
                    assert_eq!(&bits(&logits), l, "{id} capacity {cap:?}");
                    assert_eq!(&toks, t, "{id} capacity {cap:?}");
                }
                _ => unreachable!(),
            }
            if cap == Some(largest) {
                assert!(
                    e.store_stats().evictions > 0,
                    "{id}: one-expert cache must evict"
                );
            }
        }
    }
}

/// The I/O thread count changes only when experts arrive, never the bits: the MoE
/// layer may compute experts in any order (ADR-006 `acquire_many`). Direct I/O
/// changes only how bytes are read (amendment 2).
#[test]
fn io_threads_never_change_bits() {
    let direct = cfg!(any(target_os = "linux", target_os = "macos"));
    for id in IDS {
        let largest = engine(id, None).metadata().bytes().largest_expert;
        let mut want = None;
        for (io, cap, d) in [
            (0, None, false),
            (0, Some(2), false),
            (1, Some(2), false),
            (4, Some(2), false),
            (4, Some(3), false),
            (8, Some(1), false),
            (0, Some(2), direct),
            (4, Some(1), direct),
        ] {
            let meta =
                ModelMetadata::load(&dir(id), &ArchitectureRegistry::with_builtins()).unwrap();
            let e = Engine::load(
                meta,
                EngineOptions {
                    expert_cache_bytes: cap.map(|n: u64| n * largest),
                    max_context: Some(256),
                    io_threads: Some(io),
                    expert_io: if d {
                        ExpertIo::Direct
                    } else {
                        ExpertIo::Buffered
                    },
                    ..EngineOptions::default()
                },
            )
            .unwrap();
            assert_eq!(e.io_threads(), io);
            assert_eq!(e.expert_io() != IoPath::Buffered, d);
            let mut c = e.new_cache();
            let mut logits = bits(&e.forward(PROMPT, &mut c, Logits::All, None).unwrap());
            for &t in &PROMPT[..6] {
                logits.extend(bits(&e.forward(&[t], &mut c, Logits::Last, None).unwrap()));
            }
            match &want {
                None => want = Some(logits),
                Some(w) => assert_eq!(&logits, w, "{id} io {io} slots {cap:?} direct {d}"),
            }
        }
    }
}

/// Compute threads never change bits, including one thread, where experts computed
/// concurrently (ADR-002 amendment 1) must not wait on each other, and tight caches,
/// where a layer has fewer slots than routed experts.
#[test]
fn compute_threads_never_change_bits() {
    for id in ALL {
        let largest = engine(id, None).metadata().bytes().largest_expert;
        let mut want = None;
        for threads in [1, 3] {
            for (io, cap) in [(0, Some(1)), (2, Some(1)), (2, Some(3)), (2, None)] {
                let meta =
                    ModelMetadata::load(&dir(id), &ArchitectureRegistry::with_builtins()).unwrap();
                let e = Engine::load(
                    meta,
                    EngineOptions {
                        expert_cache_bytes: cap.map(|n: u64| n * largest),
                        max_context: Some(256),
                        io_threads: Some(io),
                        threads: Some(threads),
                        ..EngineOptions::default()
                    },
                )
                .unwrap();
                let mut c = e.new_cache();
                let mut logits = bits(&e.forward(PROMPT, &mut c, Logits::All, None).unwrap());
                for &t in &PROMPT[..4] {
                    logits.extend(bits(&e.forward(&[t], &mut c, Logits::Last, None).unwrap()));
                }
                match &want {
                    None => want = Some(logits),
                    Some(w) => {
                        assert_eq!(&logits, w, "{id} threads {threads} io {io} slots {cap:?}")
                    }
                }
            }
        }
    }
}

#[test]
fn sampling_is_reproducible() {
    let e = engine("qwen3-moe-tiny", None);
    let s = Sampling {
        temperature: 0.8,
        top_p: 0.9,
        seed: 42,
    };
    let a = e.generate(PROMPT, 10, s, |_| true).unwrap();
    let b = e.generate(PROMPT, 10, s, |_| true).unwrap();
    assert_eq!(a, b);
    let c = e
        .generate(PROMPT, 10, Sampling { seed: 43, ..s }, |_| true)
        .unwrap();
    assert_eq!(c.len(), 10);
    // Stopping early from the callback.
    let one = e
        .generate(PROMPT, 10, Sampling::default(), |_| false)
        .unwrap();
    assert_eq!(one.len(), 1);
}

/// The KV cache grows by whole chunks, copying its filled positions into a larger
/// tensor per layer. Growing across chunk boundaries (in one prompt or token by token),
/// a step refused before any layer runs, and reusing a cleared cache never change bits.
/// A step that fails partway through the layers is `faults.rs`'s
/// `a_failed_step_can_be_retried_on_the_same_cache`.
#[test]
fn kv_growth_failure_and_reuse_never_change_bits() {
    let meta =
        ModelMetadata::load(&dir("olmoe-tiny"), &ArchitectureRegistry::with_builtins()).unwrap();
    let e = Engine::load(
        meta,
        EngineOptions {
            max_context: Some(600),
            ..EngineOptions::default()
        },
    )
    .unwrap();
    let toks: Vec<u32> = (0..560u32).map(|i| 32 + (i * 7) % 90).collect();
    let chunk = qubivex_runtime::KvCache::CHUNK;
    // A 300-token prompt (one step crossing the first boundary), then single tokens
    // across the second.
    let mut a = e.new_cache();
    let mut got = vec![e.forward(&toks[..300], &mut a, Logits::Last, None).unwrap()];
    for t in &toks[300..] {
        got.push(e.forward(&[*t], &mut a, Logits::Last, None).unwrap());
    }
    assert_eq!(a.len(), 560);
    assert_eq!(a.allocated_bytes() % (3 * chunk as u64), 0);
    // Token by token from the start, with a failed step at each boundary.
    let mut b = e.new_cache();
    let mut want = Vec::new();
    for (i, t) in toks.iter().enumerate() {
        if i % chunk == chunk - 1 {
            let before = b.len();
            assert!(e.forward(&[*t, 9999], &mut b, Logits::Last, None).is_err());
            assert_eq!(b.len(), before, "a failed step adds no positions");
        }
        let out = e.forward(&[*t], &mut b, Logits::Last, None).unwrap();
        if i >= 299 {
            want.push(out);
        }
    }
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(bits(g), bits(w), "position {}", 299 + i);
    }
    // A cleared cache keeps its room and gives a fresh cache's bits.
    let room = a.allocated_bytes();
    a.clear();
    assert_eq!(a.allocated_bytes(), room);
    let mut fresh = e.new_cache();
    let x = e
        .forward(&toks[100..140], &mut a, Logits::All, None)
        .unwrap();
    let y = e
        .forward(&toks[100..140], &mut fresh, Logits::All, None)
        .unwrap();
    assert_eq!(bits(&x), bits(&y));
}

#[test]
fn errors_are_reported() {
    let e = engine("olmoe-tiny", None);
    let mut c = e.new_cache();
    assert!(
        e.forward(&[9999], &mut c, Logits::Last, None)
            .unwrap_err()
            .to_string()
            .contains("vocabulary")
    );
    let mut c = e.new_cache();
    let long = vec![1u32; 257];
    assert!(
        e.forward(&long, &mut c, Logits::Last, None)
            .unwrap_err()
            .to_string()
            .contains("context full")
    );
    assert!(e.generate(&[], 4, Sampling::default(), |_| true).is_err());
}

/// A planned engine whose KV takes room back from the expert cache as it grows
/// (ADR-005) produces the same tokens as one with every expert resident, and the
/// cache never drops below one slot per layer.
#[test]
fn room_sharing_never_changes_bits() {
    use qubivex_memory::{ACTIVATION_RESERVE, ModelFootprint};
    for id in IDS {
        let meta = ModelMetadata::load(&dir(id), &ArchitectureRegistry::with_builtins()).unwrap();
        let fp = ModelFootprint::from_metadata(&meta).unwrap();
        let kv = fp.kv_bytes_per_token.0;
        // Room for the floor, a 512-position KV, and two more slots per layer; the
        // cache starts above the floor and falls toward it at the second KV chunk.
        let extra = fp.cache_floor().0 * 2;
        let budget =
            fp.dense_bytes().0 + ACTIVATION_RESERVE.0 + fp.cache_floor().0 + 512 * kv + extra;
        let planned = Engine::load(
            meta,
            EngineOptions {
                ram_budget: Some(budget),
                ..EngineOptions::default()
            },
        )
        .unwrap();
        let plan = planned.memory_plan().unwrap();
        assert!(plan.context >= 512, "{id}: context {}", plan.context);
        let start = planned.expert_cache_bytes();
        let mut caps = Vec::new();
        let mut cache = planned.new_cache();
        // Every expert resident, a fixed cache, the same context.
        let meta = ModelMetadata::load(&dir(id), &ArchitectureRegistry::with_builtins()).unwrap();
        let reference = Engine::load(
            meta,
            EngineOptions {
                expert_cache_bytes: Some(fp.expert_bytes().0),
                max_context: Some(512),
                ..EngineOptions::default()
            },
        )
        .unwrap();
        let mut ref_cache = reference.new_cache();
        for (i, &t) in PROMPT.iter().cycle().take(300).enumerate() {
            let a = planned
                .forward(&[t], &mut cache, Logits::Last, None)
                .unwrap();
            let b = reference
                .forward(&[t], &mut ref_cache, Logits::Last, None)
                .unwrap();
            assert_eq!(bits(&a), bits(&b), "{id} position {i}");
            caps.push(planned.expert_cache_bytes());
        }
        let end = *caps.last().unwrap();
        assert!(end < start, "{id}: the cache shrank ({start} -> {end})");
        assert!(
            caps.windows(2).all(|w| w[1] <= w[0]),
            "{id}: monotonic while the KV grows"
        );
        assert!(end >= fp.cache_floor().0, "{id}: never below the floor");
        // One pool per expert layer, each keeping at least one slot of its layer's size.
        let pools = planned.expert_pools();
        assert_eq!(pools.len(), fp.layers.len(), "{id}");
        for (p, l) in pools.iter().zip(&fp.layers) {
            assert_eq!((p.layer, p.slot_bytes), (l.layer, l.expert_bytes.0), "{id}");
            assert!(p.slots >= 1 && p.slots <= l.experts, "{id}: {p:?}");
        }
        assert!(
            pools.iter().any(|p| p.slots < fp.layers[0].experts),
            "{id}: streaming"
        );
        // The KV grows in whole chunks: 300 positions reserve 512, as the plan prices.
        let reserved = qubivex_runtime::KvCache::reserved_for(300, cache.capacity()) as u64;
        assert_eq!(reserved, 512);
        assert_eq!(cache.allocated_bytes(), reserved * kv, "{id}");
        let l = planned.ledger().snapshot(qubivex_memory::Tier::Ram);
        assert_eq!(l.kv.0, reserved * kv, "{id}: the ledger's KV charge");
        assert_eq!(l.dense, fp.dense_bytes(), "{id}: the ledger's dense charge");
    }
}

/// Under pressure (ADR-005): the cache drops to the floor between tokens, an active
/// request runs to the end with the same bits, new requests are refused, and
/// everything recovers when RAM comes back.
#[test]
fn pressure_shrinks_refuses_and_recovers() {
    use qubivex_core::Bytes;
    use qubivex_memory::{ModelFootprint, PressureLevel};
    for id in IDS {
        let meta = ModelMetadata::load(&dir(id), &ArchitectureRegistry::with_builtins()).unwrap();
        let fp = ModelFootprint::from_metadata(&meta).unwrap();
        let e = Engine::load(
            meta,
            EngineOptions {
                ram_budget: Some(fp.dense_bytes().0 + (1 << 30)),
                max_context: Some(256),
                pressure_interval_ms: Some(0), // driven by hand below
                ..EngineOptions::default()
            },
        )
        .unwrap();
        let reference = engine(id, Some(fp.expert_bytes().0));
        let planned_cap = e.expert_cache_bytes();
        let (mut c, mut rc) = (e.new_cache(), reference.new_cache());
        for (i, &t) in PROMPT.iter().cycle().take(40).enumerate() {
            if i == 10 {
                let d = e.observe_pressure(Bytes(1 << 20)).unwrap();
                assert_eq!(
                    (d.level, d.admit, d.prefetch),
                    (PressureLevel::Critical, false, false)
                );
                // A new request is refused; this one continues.
                assert!(
                    e.generate(PROMPT, 1, Sampling::default(), |_| true)
                        .is_err()
                );
            }
            let a = e.forward(&[t], &mut c, Logits::Last, None).unwrap();
            let b = reference
                .forward(&[t], &mut rc, Logits::Last, None)
                .unwrap();
            assert_eq!(bits(&a), bits(&b), "{id} position {i}");
            if i >= 10 {
                assert_eq!(
                    e.expert_cache_bytes(),
                    fp.cache_floor().0,
                    "{id}: at the floor"
                );
            }
        }
        // RAM comes back: admitted again, and the cap lifts.
        let d = e.observe_pressure(Bytes(8 << 30)).unwrap();
        assert_eq!(
            (d.level, d.admit, d.cache_cap),
            (PressureLevel::Normal, true, None)
        );
        e.forward(&[PROMPT[0]], &mut c, Logits::Last, None).unwrap();
        assert_eq!(
            e.expert_cache_bytes(),
            planned_cap,
            "{id}: back to the plan's cache"
        );
        assert!(e.generate(PROMPT, 1, Sampling::default(), |_| true).is_ok());
    }
}

/// The sampling thread starts with the engine and stops when it is dropped.
#[test]
fn pressure_thread_stops_with_the_engine() {
    let meta =
        ModelMetadata::load(&dir("olmoe-tiny"), &ArchitectureRegistry::with_builtins()).unwrap();
    let e = Engine::load(
        meta,
        EngineOptions {
            pressure_interval_ms: Some(5),
            ..EngineOptions::default()
        },
    )
    .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));
    assert!(e.pressure().is_some());
    let t = std::time::Instant::now();
    drop(e);
    assert!(
        t.elapsed() < std::time::Duration::from_secs(2),
        "joined promptly"
    );
}

/// Phase 7 exit test: routing is identical with prediction on and off. The lookahead
/// only chooses what to read ahead, so the router logits, the chosen experts with
/// their weights, and the output logits keep every bit, whatever the cache size, the
/// I/O threads, the guard or the predictor's settings.
#[test]
fn prefetch_never_changes_routing_or_bits() {
    let wide = LookaheadConfig {
        smooth: 0.0,
        confidence: 1.0,
        width: 3,
    };
    for id in IDS {
        let largest = engine(id, None).metadata().bytes().largest_expert;
        let mut want: Option<Vec<u32>> = None;
        let mut hidden = 0;
        for (prefetch, io, cap) in [
            (None, 4, None),
            (None, 4, Some(2)),
            (Some((LookaheadConfig::default(), true)), 4, Some(2)),
            (Some((LookaheadConfig::default(), false)), 4, Some(2)),
            (Some((LookaheadConfig::default(), false)), 1, Some(1)),
            (Some((wide, false)), 8, Some(3)),
            (Some((wide, true)), 4, None),
            (Some((LookaheadConfig::default(), false)), 0, Some(2)),
        ] {
            let meta =
                ModelMetadata::load(&dir(id), &ArchitectureRegistry::with_builtins()).unwrap();
            let e = Engine::load(
                meta,
                EngineOptions {
                    expert_cache_bytes: cap.map(|n: u64| n * largest),
                    max_context: Some(256),
                    io_threads: Some(io),
                    prefetch: prefetch
                        .map(|(lookahead, guard)| PrefetchOptions { lookahead, guard }),
                    ..EngineOptions::default()
                },
            )
            .unwrap();
            let mut seen = Vec::new();
            let mut tracer = |ev: TraceEvent<'_>| {
                if matches!(
                    ev.point,
                    TracePoint::RouterLogits | TracePoint::RoutingWeights
                ) {
                    seen.extend(ev.values.iter().map(|v| v.to_bits()));
                    seen.extend(ev.experts);
                }
            };
            let mut c = e.new_cache();
            let mut out = bits(
                &e.forward(&PROMPT[..4], &mut c, Logits::All, Some(&mut tracer))
                    .unwrap(),
            );
            for &t in &PROMPT[4..] {
                out.extend(bits(
                    &e.forward(&[t], &mut c, Logits::Last, Some(&mut tracer))
                        .unwrap(),
                ));
            }
            e.end_turn();
            seen.extend(out);
            let label = format!("{id} prefetch {prefetch:?} io {io} slots {cap:?}");
            match &want {
                None => want = Some(seen),
                Some(w) => assert!(&seen == w, "{label}: routing or logits changed"),
            }
            match e.prefetch_report() {
                None => assert!(prefetch.is_none()),
                Some(r) => {
                    // Every decode layer after the first had a prediction.
                    let layers = e.metadata().config.dims.layers;
                    assert_eq!(r.prediction.layers, u64::from(layers - 1) * 10, "{label}");
                    assert!(r.prediction.correct <= r.prediction.routed);
                    if io == 0 {
                        assert_eq!(r.store.issued, 0, "{label}: nobody reads ahead");
                    }
                    let st = r.store;
                    assert!(st.ready + st.late <= st.issued, "{label}");
                    if io > 0 {
                        assert!(
                            st.issued + st.present > 0,
                            "{label}: predictions reach the store"
                        );
                        hidden += st.ready + st.late;
                    }
                }
            }
        }
        assert!(hidden > 0, "{id}: some prefetched read was used");
    }
}

/// Phase 8 (ADR-011): the device placement never changes bits. Every Metal placement
/// gives the CPU's logits for a prompt batch and its generated tokens. Skipped without
/// a Metal device.
#[test]
fn metal_placement_never_changes_bits() {
    use qubivex_runtime::{BackendKind, MetalOps};
    match qubivex_backend_metal::MetalBackend::new() {
        Err(qubivex_backend_metal::MetalError::Unavailable(e)) => {
            eprintln!("skipping: {e}");
            return;
        }
        Err(e) => panic!("{e}"),
        Ok(_) => {}
    }
    let all = MetalOps {
        attention: true,
        router: true,
        head: true,
        layers: false,
    };
    let layers = MetalOps::default();
    assert!(layers.layers, "whole layers are the default placement");
    let placements = [
        MetalOps {
            attention: false,
            router: false,
            head: true,
            layers: false,
        },
        MetalOps {
            attention: true,
            router: false,
            head: false,
            layers: false,
        },
        MetalOps {
            attention: false,
            router: true,
            head: false,
            layers: false,
        },
        all,
        layers,
    ];
    for id in ALL {
        let meta =
            || ModelMetadata::load(&dir(id), &ArchitectureRegistry::with_builtins()).unwrap();
        let opts = EngineOptions {
            max_context: Some(256),
            ..EngineOptions::default()
        };
        let cpu = Engine::load(meta(), opts).unwrap();
        assert_eq!(cpu.placement().backend, BackendKind::Cpu);
        let mut c = cpu.new_cache();
        let want = bits(&cpu.forward(PROMPT, &mut c, Logits::All, None).unwrap());
        let want_toks = cpu
            .generate(PROMPT, 12, Sampling::default(), |_| true)
            .unwrap();
        for ops in placements {
            let e = Engine::load(
                meta(),
                EngineOptions {
                    backend: BackendKind::Metal,
                    metal_ops: ops,
                    ..opts
                },
            )
            .unwrap();
            let p = e.placement();
            eprintln!(
                "{id} {}: {} tensors, {} bytes",
                ops.describe(),
                p.tensors,
                p.bytes
            );
            assert!(p.device.is_some(), "{id}");
            let mut c = e.new_cache();
            let got = bits(&e.forward(PROMPT, &mut c, Logits::All, None).unwrap());
            assert_eq!(got, want, "{id} {}: prompt logits", ops.describe());
            let toks = e
                .generate(PROMPT, 12, Sampling::default(), |_| true)
                .unwrap();
            assert_eq!(toks, want_toks, "{id} {}: tokens", ops.describe());
            if ops == all || ops == layers {
                assert!(p.tensors > 0, "{id}: something was placed");
            }
        }
    }
}

/// Phase 8 slices 3–4: the whole-layer Metal pass gives the CPU's bits when experts run
/// in waves (memory pressure shrinks the pools after load, below a prompt's routed set),
/// when the device KV cache grows past a chunk, and when decode steps follow a prompt.
/// A plan whose pools are smaller than top-k from the start runs the CPU pass instead.
#[test]
fn metal_layers_match_the_cpu_in_waves_and_across_kv_growth() {
    use qubivex_runtime::{BackendKind, MetalOps};
    match qubivex_backend_metal::MetalBackend::new() {
        Err(qubivex_backend_metal::MetalError::Unavailable(e)) => {
            eprintln!("skipping: {e}");
            return;
        }
        Err(e) => panic!("{e}"),
        Ok(_) => {}
    }
    for id in IDS {
        let meta =
            || ModelMetadata::load(&dir(id), &ArchitectureRegistry::with_builtins()).unwrap();
        let metal = |opts: EngineOptions| {
            Engine::load(
                meta(),
                EngineOptions {
                    backend: BackendKind::Metal,
                    metal_ops: MetalOps::default(),
                    ..opts
                },
            )
            .unwrap()
        };
        // Pools of one slot from the start: the plan keeps the CPU pass.
        let one_slot = Engine::load(meta(), EngineOptions::default())
            .unwrap()
            .expert_pools()
            .iter()
            .map(|p| p.slot_bytes)
            .sum::<u64>();
        let small = metal(EngineOptions {
            expert_cache_bytes: Some(one_slot),
            ..EngineOptions::default()
        });
        let p = small.placement();
        assert!(p.fallback.is_some() && p.tensors == 0, "{id}: {p:?}");

        // Planned, so pressure can shrink the cache after load.
        let opts = EngineOptions {
            max_context: Some(600),
            pressure_interval_ms: Some(0),
            ..EngineOptions::default()
        };
        let cpu = Engine::load(meta(), opts).unwrap();
        let gpu = metal(opts);
        assert!(gpu.placement().fallback.is_none(), "{id}");
        assert!(gpu.placement().tensors > 0, "{id}");
        let before: Vec<u32> = gpu.expert_pools().iter().map(|p| p.slots).collect();
        // Critical pressure: the cache drops to its floor at the next forward pass.
        gpu.observe_pressure(qubivex_core::Bytes(1 << 20));
        let (mut cc, mut gc) = (cpu.new_cache(), gpu.new_cache());
        let want = bits(&cpu.forward(PROMPT, &mut cc, Logits::All, None).unwrap());
        let got = bits(&gpu.forward(PROMPT, &mut gc, Logits::All, None).unwrap());
        let after: Vec<u32> = gpu.expert_pools().iter().map(|p| p.slots).collect();
        eprintln!("{id}: slots {before:?} -> {after:?}");
        assert!(
            after.iter().zip(&before).all(|(a, b)| a < b),
            "{id}: the pools shrank"
        );
        assert_eq!(got, want, "{id}: prompt logits in waves");
        // Greedy decode across the 256-position chunk boundary, step by step.
        let mut next = PROMPT[PROMPT.len() - 1];
        for step in 0..300 {
            let a = cpu.forward(&[next], &mut cc, Logits::Last, None).unwrap();
            let b = gpu.forward(&[next], &mut gc, Logits::Last, None).unwrap();
            assert_eq!(bits(&b), bits(&a), "{id}: step {step}");
            next = argmax(&a);
        }
        assert_eq!(gc.len(), cc.len());
        assert!(gc.allocated_bytes() > 0, "{id}: device KV is accounted");
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    u32::try_from(best).unwrap()
}
