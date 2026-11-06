//! Planner tests: footprints of the committed fixtures, admission, and property tests
//! that plans never exceed their budget on random machines and models.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::path::Path;

use proptest::prelude::*;
use qubivex_core::{Bytes, LayerIdx};
use qubivex_hardware::Probed;
use qubivex_memory::{
    ACTIVATION_RESERVE, DenseGroup, HostMemory, LayerExperts, ModelFootprint, PlanRequest,
    PlanningError, Profile, RoleGroup, Source, Verdict, assess, distribute, plan,
};
use qubivex_model::{ArchitectureRegistry, ModelMetadata};

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

fn load(id: &str) -> ModelMetadata {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/models")
        .join(id);
    ModelMetadata::load(&dir, &ArchitectureRegistry::with_builtins()).unwrap()
}

fn host(avail: u64, total: u64) -> HostMemory {
    HostMemory {
        total: Probed::Known(Bytes(total)),
        available: Probed::Known(Bytes(avail)),
        dense_resident: false,
    }
}

/// The footprint agrees with the model's byte summary, and its KV cost with the
/// architecture's own head counts (an independent source).
#[test]
fn fixture_footprints() {
    for id in ["olmoe-tiny", "olmoe-tiny-gguf", "qwen3-moe-tiny"] {
        let meta = load(id);
        let fp = ModelFootprint::from_metadata(&meta).unwrap();
        let s = meta.bytes();
        assert_eq!(fp.dense_bytes(), Bytes(s.dense), "{id} dense");
        assert_eq!(fp.expert_bytes(), Bytes(s.routed_experts), "{id} experts");
        let largest = fp.layers.iter().map(|l| l.expert_bytes).max().unwrap();
        assert_eq!(largest, Bytes(s.largest_expert), "{id} largest expert");
        let dims = meta.config.dims;
        assert_eq!(fp.layers.len() as u32, dims.layers, "{id} expert layers");
        assert!(fp.layers.iter().all(|l| l.experts == dims.experts), "{id}");
        let d = &meta.config.describe;
        let kv_heads = d["kv_heads"].as_u64().unwrap();
        let head_dim = d["head_dim"].as_u64().unwrap();
        assert_eq!(
            fp.kv_bytes_per_token,
            Bytes(u64::from(dims.layers) * 2 * kv_heads * head_dim * 4),
            "{id} KV per token"
        );
    }
}

/// A synthetic MoE: `layers` layers of `experts` experts of `eb` bytes.
fn synthetic(layers: u32, experts: u32, eb: u64, dense: u64, kv: u64) -> ModelFootprint {
    ModelFootprint {
        dense: vec![DenseGroup {
            group: RoleGroup::Attention,
            bytes: Bytes(dense),
            tensors: 1,
        }],
        layers: (0..layers)
            .map(|i| LayerExperts {
                layer: LayerIdx(i),
                experts,
                expert_bytes: Bytes(eb),
                total: Bytes(eb * u64::from(experts)),
            })
            .collect(),
        kv_bytes_per_token: Bytes(kv),
        max_context: Some(4096),
    }
}

/// The reference model's shape: 16 layers × 64 experts of 3.4 MiB, 0.4 GiB dense,
/// 256 KiB of KV per token.
fn reference_like() -> ModelFootprint {
    synthetic(16, 64, 3_571_712, 400 * MIB, 256 << 10)
}

#[test]
fn everything_resident_when_ram_allows() {
    let fp = reference_like();
    let p = plan(&fp, &host(12 * GIB, 18 * GIB), &PlanRequest::default()).unwrap();
    assert_eq!(p.context, 4096);
    assert_eq!(p.ram.expert_cache_full_context.bytes, fp.expert_bytes());
    assert!(p.expert_cache.at_full_context.iter().all(|l| l.slots == 64));
    // budget = dense + 0.88 × (12 GiB − dense)
    let expect = 400 * MIB + ((12 * GIB - 400 * MIB) as f64 * 0.88) as u64;
    assert_eq!(p.ram.budget.bytes, Bytes(expect));
    assert_eq!(p.ram.budget.source, Source::Estimated);
}

#[test]
fn kv_takes_room_back_from_the_cache() {
    let fp = reference_like();
    // 2 GiB room: not every expert fits, and a 4096 KV (1 GiB) halves the cache.
    let budget = 400 * MIB + ACTIVATION_RESERVE.0 + 2 * GIB;
    let req = PlanRequest {
        ram_budget: Some(Bytes(budget)),
        ..PlanRequest::default()
    };
    let p = plan(&fp, &host(12 * GIB, 18 * GIB), &req).unwrap();
    assert_eq!(p.ram.room.bytes, Bytes(2 * GIB));
    assert_eq!(p.ram.kv_full_context.bytes, Bytes(GIB));
    assert_eq!(p.ram.expert_cache_start.bytes, Bytes(2 * GIB));
    assert_eq!(p.ram.expert_cache_full_context.bytes, Bytes(GIB));
    assert_eq!(p.expert_cache_for(0).unwrap(), Bytes(2 * GIB));
    assert_eq!(p.expert_cache_for(4096).unwrap(), Bytes(GIB));
    let start: u32 = p.expert_cache.at_start.iter().map(|l| l.slots).sum();
    let full: u32 = p.expert_cache.at_full_context.iter().map(|l| l.slots).sum();
    assert!(full < start, "{full} < {start}");
}

/// MIGRATION_PLAN §3 Phase 5: projected KV must not push the expert cache below the
/// floor (one slot per layer).
#[test]
fn admission_keeps_the_floor() {
    let fp = reference_like();
    let floor = fp.cache_floor().0;
    // Room for the floor plus exactly 100 KV tokens.
    let room = floor + 100 * (256 << 10);
    let req = PlanRequest {
        ram_budget: Some(Bytes(400 * MIB + ACTIVATION_RESERVE.0 + room)),
        ..PlanRequest::default()
    };
    let p = plan(&fp, &host(12 * GIB, 18 * GIB), &req).unwrap();
    assert_eq!(p.context, 100, "the model's 4096 is shortened to what fits");
    assert_eq!(p.expert_cache_for(100).unwrap(), Bytes(floor));
    let r = p.expert_cache_for(101).unwrap_err();
    assert_eq!(r.floor, Bytes(floor));
    assert!(p.expert_cache.at_full_context.iter().all(|l| l.slots == 1));
    // An explicit context that does not fit is refused, with the largest that does.
    let req = PlanRequest {
        context: Some(101),
        ..req
    };
    assert_eq!(
        plan(&fp, &host(12 * GIB, 18 * GIB), &req).unwrap_err(),
        PlanningError::ContextTooLarge {
            requested: 101,
            fits: 100,
            sequences: 1
        }
    );
}

#[test]
fn refusals() {
    let fp = reference_like();
    // dense 400 MiB + reserve 512 MiB + floor 54.5 MiB = 966.5 MiB > 900 MiB.
    let tiny = PlanRequest {
        ram_budget: Some(Bytes(900 * MIB)),
        ..PlanRequest::default()
    };
    assert!(matches!(
        plan(&fp, &host(12 * GIB, 18 * GIB), &tiny),
        Err(PlanningError::DoesNotFit { .. })
    ));
    let unknown = HostMemory {
        total: Probed::unknown("x"),
        available: Probed::unknown("no probe"),
        dense_resident: false,
    };
    assert!(matches!(
        plan(&fp, &unknown, &PlanRequest::default()),
        Err(PlanningError::UnknownAvailable(_))
    ));
    let bad = PlanRequest {
        profile: Profile::Custom(1.5),
        ..PlanRequest::default()
    };
    assert!(matches!(
        plan(&fp, &host(12 * GIB, 18 * GIB), &bad),
        Err(PlanningError::BadRequest(_))
    ));
    // Less available RAM than the dense weights.
    assert!(matches!(
        plan(&fp, &host(100 * MIB, 18 * GIB), &PlanRequest::default()),
        Err(PlanningError::DoesNotFit { .. })
    ));
}

#[test]
fn profiles_order_the_budgets() {
    let fp = reference_like();
    let h = host(8 * GIB, 18 * GIB);
    let b = |profile| {
        plan(
            &fp,
            &h,
            &PlanRequest {
                profile,
                ..PlanRequest::default()
            },
        )
        .unwrap()
        .ram
        .budget
        .bytes
    };
    assert!(b(Profile::Performance) > b(Profile::Balanced));
    assert!(b(Profile::Balanced) > b(Profile::Saver));
}

#[test]
fn plan_serialises() {
    let p = plan(
        &reference_like(),
        &host(12 * GIB, 18 * GIB),
        &PlanRequest::default(),
    )
    .unwrap();
    let j = serde_json::to_value(&p).unwrap();
    assert_eq!(j["ram"]["dense"]["source"], "measured");
    assert_eq!(j["profile"]["profile"], "performance");
}

fn arb_footprint() -> impl Strategy<Value = ModelFootprint> {
    (
        0u32..48,
        1u32..160,
        (1u64 << 10)..(64 * MIB),
        0u64..(24 * GIB),
        0u64..(2 * MIB),
        proptest::option::of(1u64..200_000),
        proptest::bool::ANY,
    )
        .prop_flat_map(|(layers, experts, eb, dense, kv, ctx, mixed)| {
            // Mixed: each layer's experts get their own size (mixed quantisation).
            proptest::collection::vec(
                if mixed {
                    (eb / 2 + 1)..(eb + 1)
                } else {
                    eb..(eb + 1)
                },
                layers as usize,
            )
            .prop_map(move |sizes| {
                let mut fp = synthetic(0, experts, eb, dense, kv);
                fp.max_context = ctx;
                fp.layers = sizes
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| LayerExperts {
                        layer: LayerIdx(i as u32),
                        experts,
                        expert_bytes: Bytes(s),
                        total: Bytes(s * u64::from(experts)),
                    })
                    .collect();
                fp
            })
        })
}

fn arb_request(layers: usize) -> impl Strategy<Value = PlanRequest> {
    (
        prop_oneof![
            Just(Profile::Performance),
            Just(Profile::Balanced),
            Just(Profile::Saver),
            (0.01f64..=1.0).prop_map(Profile::Custom),
        ],
        proptest::option::of((GIB / 4)..(512 * GIB)),
        proptest::option::of(1u64..300_000),
        1u32..5,
        proptest::option::of(proptest::collection::vec(0.0f64..10.0, layers)),
    )
        .prop_map(
            |(profile, ram_budget, context, sequences, layer_weights)| PlanRequest {
                profile,
                ram_budget: ram_budget.map(Bytes),
                context,
                sequences,
                layer_weights,
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// No plan exceeds its budget, at any point of the KV's growth.
    #[test]
    fn plans_never_exceed_the_budget(
        (fp, req) in arb_footprint().prop_flat_map(|fp| {
            let n = fp.layers.len();
            (Just(fp), arb_request(n))
        }),
        avail in 0u64..(512 * GIB),
        total in (GIB / 2)..(1024 * GIB),
        resident in proptest::bool::ANY,
    ) {
        let h = HostMemory {
            total: Probed::Known(Bytes(total)),
            available: Probed::Known(Bytes(avail)),
            dense_resident: resident,
        };
        let Ok(p) = plan(&fp, &h, &req) else { return Ok(()) };
        let r = &p.ram;
        let budget = r.budget.bytes.0;
        // The parts add up to the budget.
        prop_assert_eq!(r.dense.bytes.0 + r.activations.bytes.0 + r.room.bytes.0, budget);
        // Within the budget at both ends of the KV's growth.
        prop_assert!(r.expert_cache_start.bytes.0 <= r.room.bytes.0);
        prop_assert!(r.kv_full_context.bytes.0 + r.expert_cache_full_context.bytes.0 <= r.room.bytes.0);
        // Never below the floor, never above every expert.
        prop_assert!(r.expert_cache_full_context.bytes >= r.cache_floor.bytes);
        prop_assert!(r.expert_cache_start.bytes <= r.experts_total.bytes);
        // The budget respects its source.
        match req.ram_budget {
            Some(b) => prop_assert_eq!(budget, b.0),
            None => {
                prop_assert!(budget <= total);
                let base = if resident { avail } else { avail.saturating_sub(r.dense.bytes.0) };
                prop_assert!(budget <= r.dense.bytes.0 + base);
            }
        }
        if let Some(c) = req.context {
            prop_assert_eq!(p.context, c);
        }
        // Slots: within each layer, within the cache bytes.
        for (cache, slots) in [
            (r.expert_cache_start.bytes.0, &p.expert_cache.at_start),
            (r.expert_cache_full_context.bytes.0, &p.expert_cache.at_full_context),
        ] {
            let mut used = 0u64;
            for (s, l) in slots.iter().zip(&fp.layers) {
                prop_assert!(s.slots >= 1 && s.slots <= l.experts);
                used += u64::from(s.slots) * l.expert_bytes.0;
            }
            prop_assert!(used <= cache, "{} > {}", used, cache);
        }
        // Admission over the whole context: always the floor or more, never past room.
        let tokens = p.context * u64::from(p.sequences);
        for kv in [0, tokens / 3, tokens / 2, tokens] {
            let c = p.expert_cache_for(kv).unwrap();
            prop_assert!(c >= r.cache_floor.bytes);
            prop_assert!(c.0 + kv * fp.kv_bytes_per_token.0 <= r.room.bytes.0);
        }
    }

    /// A refused model really does not fit: dense + reserve + floor + one KV token
    /// exceed whatever budget the request allows.
    #[test]
    fn refusals_are_real(fp in arb_footprint(), b in (GIB / 4)..(64 * GIB)) {
        let req = PlanRequest { ram_budget: Some(Bytes(b)), ..PlanRequest::default() };
        if let Err(PlanningError::DoesNotFit { needed, budget, .. }) =
            plan(&fp, &host(0, 1024 * GIB), &req)
        {
            prop_assert_eq!(budget, b);
            prop_assert_eq!(
                needed,
                fp.dense_bytes().0 + ACTIVATION_RESERVE.0 + fp.cache_floor().0 + fp.kv_bytes_per_token.0
            );
            prop_assert!(needed > b);
        }
    }

    /// Distribution: within bytes, within each layer, everything when it all fits,
    /// and no slot left unspent that would still fit.
    #[test]
    fn distribution_is_tight(
        fp in arb_footprint(),
        frac in 0.0f64..1.2,
    ) {
        let all = fp.expert_bytes().0;
        #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
        let bytes = Bytes((all as f64 * frac) as u64).max(fp.cache_floor());
        let s = distribute(&fp.layers, bytes, None);
        let used: u64 = s.iter().zip(&fp.layers).map(|(&n, l)| u64::from(n) * l.expert_bytes.0).sum();
        prop_assert!(used <= bytes.0);
        for (&n, l) in s.iter().zip(&fp.layers) {
            prop_assert!(n >= 1 && n <= l.experts);
            if n < l.experts {
                prop_assert!(used + l.expert_bytes.0 > bytes.0, "a slot was left unspent");
            }
        }
    }
}

/// Each verdict, with the budget just either side of the figures that decide it.
#[test]
fn assessment_verdicts_turn_on_the_stated_requirements() {
    let fp = ModelFootprint::from_metadata(&load("qwen3-moe-tiny")).unwrap();
    let at = |budget: u64| {
        let req = PlanRequest {
            ram_budget: Some(Bytes(budget)),
            context: Some(64),
            ..PlanRequest::default()
        };
        assess(&fp, &host(GIB, GIB), &req).unwrap()
    };
    let need = at(GIB).requirements;
    assert!(need.minimum <= need.at_context && need.at_context < need.in_memory);
    assert_eq!(need.kv_context, 64);

    let fits = at(need.in_memory.0);
    assert_eq!(fits.verdict, Verdict::Fits);
    assert!(fits.refusal.is_none());

    assert_eq!(at(need.in_memory.0 - 1).verdict, Verdict::Streams);
    assert_eq!(at(need.at_context.0).verdict, Verdict::Streams);

    let refused = at(need.at_context.0 - 1);
    assert_eq!(refused.verdict, Verdict::DoesNotFit);
    assert!(refused.plan.is_none());
    assert!(refused.refusal.unwrap().contains("at most"));

    let too_small = at(need.minimum.0 - 1);
    assert_eq!(too_small.verdict, Verdict::DoesNotFit);
    assert!(too_small.refusal.unwrap().contains("does not fit"));
}

#[test]
fn a_dense_model_has_no_streaming_verdict() {
    let fp = ModelFootprint::from_metadata(&load("dense-tiny")).unwrap();
    let req = PlanRequest {
        context: Some(16),
        ..PlanRequest::default()
    };
    let a = assess(&fp, &host(GIB, GIB), &req).unwrap();
    assert_eq!(a.verdict, Verdict::Fits);
    assert_eq!(a.requirements.at_context, a.requirements.in_memory);
}

#[test]
fn an_unmeasurable_machine_is_an_error_not_a_verdict() {
    let fp = ModelFootprint::from_metadata(&load("dense-tiny")).unwrap();
    let unknown = HostMemory {
        total: Probed::Unknown("no probe".into()),
        available: Probed::Unknown("no probe".into()),
        dense_resident: false,
    };
    assert!(matches!(
        assess(&fp, &unknown, &PlanRequest::default()),
        Err(PlanningError::UnknownAvailable(_))
    ));
}
