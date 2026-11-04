//! QUBIVEX against transformers (f32) on the tiny fixtures, mixture-of-experts and dense
//! (`fixtures/oracle/forward.json`, from `fixtures/oracle/forward.py`).

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::type_complexity
)]

use std::collections::HashMap;
use std::path::Path;

use qubivex_arch::{Logits, TraceEvent, TracePoint};
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{Engine, EngineOptions};
use serde_json::Value;

/// Relative tolerance: max |ours − ref| / max |ref| per tensor.
const TOL: f64 = 1e-5;

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|b| f32::from_bits(b.as_u64().unwrap() as u32))
        .collect()
}

fn rel(ours: &[f32], want: &[f32]) -> f64 {
    assert_eq!(ours.len(), want.len());
    let scale = want
        .iter()
        .fold(0f64, |m, v| m.max(f64::from(v.abs())))
        .max(1e-30);
    ours.iter()
        .zip(want)
        .fold(0f64, |m, (a, b)| m.max(f64::from(a - b).abs()))
        / scale
}

#[test]
fn fixtures_match_transformers() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    let oracle: Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("oracle/forward.json")).unwrap())
            .unwrap();
    let prompt: Vec<u32> = oracle["prompt"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let s = prompt.len();
    for (name, want) in oracle["models"].as_object().unwrap() {
        let meta = ModelMetadata::load(
            &root.join("models").join(name),
            &ArchitectureRegistry::with_builtins(),
        )
        .unwrap();
        let e = Engine::load(meta, EngineOptions::default()).unwrap();
        // Trace: (point, layer, token) → values (and experts for routing weights).
        let mut got: HashMap<(TracePoint, Option<u32>, usize), (Vec<f32>, Vec<u32>)> =
            HashMap::new();
        let mut cache = e.new_cache();
        let mut tracer = |ev: TraceEvent<'_>| {
            got.insert(
                (ev.point, ev.layer, ev.token),
                (ev.values.to_vec(), ev.experts.to_vec()),
            );
        };
        let logits = e
            .forward(&prompt, &mut cache, Logits::All, Some(&mut tracer))
            .unwrap();
        let mut worst = 0f64;
        for (l, lw) in want["layers"].as_array().unwrap().iter().enumerate() {
            let l32 = Some(l as u32);
            // A dense fixture records its FFN output instead of routing.
            let dense = lw.get("ffn").is_some();
            let points: &[(TracePoint, &str)] = if dense {
                &[
                    (TracePoint::Attention, "attention"),
                    (TracePoint::Ffn, "ffn"),
                    (TracePoint::Hidden, "hidden"),
                ]
            } else {
                &[
                    (TracePoint::Attention, "attention"),
                    (TracePoint::RouterLogits, "router_logits"),
                    (TracePoint::Moe, "moe"),
                    (TracePoint::Hidden, "hidden"),
                ]
            };
            for &(point, key) in points {
                let w = floats(&lw[key]);
                let ours: Vec<f32> = (0..s)
                    .flat_map(|t| got[&(point, l32, t)].0.clone())
                    .collect();
                let r = rel(&ours, &w);
                worst = worst.max(r);
                assert!(r < TOL, "{name} layer {l} {key}: relative error {r:.2e}");
            }
            if dense {
                continue;
            }
            let experts = lw["experts"].as_array().unwrap();
            let weights = floats(&lw["routing_weights"]);
            let k = weights.len() / s;
            for t in 0..s {
                let (w, ids) = &got[&(TracePoint::RoutingWeights, l32, t)];
                let want_ids: Vec<u32> = experts[t]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect();
                assert_eq!(
                    ids, &want_ids,
                    "{name} layer {l} token {t}: selected experts"
                );
                let r = rel(w, &weights[t * k..(t + 1) * k]);
                assert!(
                    r < TOL,
                    "{name} layer {l} token {t}: routing weights {r:.2e}"
                );
            }
        }
        let r = rel(&logits, &floats(&want["logits"]));
        worst = worst.max(r);
        assert!(r < TOL, "{name} logits: relative error {r:.2e}");

        // Greedy continuation, decoding token by token after the prompt.
        let mut cache = e.new_cache();
        let mut last = e.forward(&prompt, &mut cache, Logits::Last, None).unwrap();
        let mut toks = Vec::new();
        for _ in 0..want["greedy"].as_array().unwrap().len() {
            let next =
                (0..last.len()).fold(0usize, |b, i| if last[i] > last[b] { i } else { b }) as u32;
            toks.push(next);
            last = e.forward(&[next], &mut cache, Logits::Last, None).unwrap();
        }
        let want_greedy: Vec<u32> = want["greedy"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(toks, want_greedy, "{name} greedy tokens");
        eprintln!("{name}: worst relative error {worst:.2e}");
    }
}
