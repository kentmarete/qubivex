//! A real gpt-oss checkpoint (`transformer_moe` with MXFP4 experts stored as SafeTensors
//! blocks and scales, attention sinks, sliding windows, YaRN, biases, top-K softmax
//! routing and the clamped SwiGLU), against transformers
//! (`fixtures/oracle/gpt_oss_real.py`). Runs when `QUBIVEX_GPT_OSS_DIR` names the
//! imported model directory of the checkpoint in the generator (gpt-oss-20b).
//!
//! transformers cannot hold the whole model in f32 here, so its reference is the model cut
//! to two layers; the full model's layers 0 (sliding window) and 1 (full attention)
//! must match it, over a prompt longer than the window. The tokenizer and chat
//! template are checked too, then a greedy reply, which must be the same on every
//! backend that opens.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::collections::HashMap;
use std::path::Path;

use qubivex_arch::{Logits, TraceEvent, TracePoint};
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{BackendKind, Engine, EngineOptions, Sampling};
use qubivex_tokenizer::ChatMessage;
use serde_json::Value;

/// Relative tolerance per traced tensor: max |ours − ref| / max |ref|.
const TOL: f64 = 1e-5;

fn oracle() -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/oracle/gpt_oss_real.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| u32::try_from(x.as_u64().unwrap()).unwrap())
        .collect()
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| f32::from_bits(u32::try_from(x.as_u64().unwrap()).unwrap()))
        .collect()
}

fn rel(ours: &[f32], want: &[f32]) -> f64 {
    assert_eq!(ours.len(), want.len());
    let scale = want.iter().fold(0f64, |m, x| m.max(f64::from(x.abs())));
    let err = ours.iter().zip(want).fold(0f64, |m, (a, b)| {
        m.max((f64::from(*a) - f64::from(*b)).abs())
    });
    err / scale.max(1e-30)
}

/// The chat text without its date line (`strftime_now`, the day each side ran).
fn undated(s: &str) -> String {
    s.lines()
        .filter(|l| !l.starts_with("Current date: "))
        .collect::<Vec<_>>()
        .join("\n")
}

fn engine(dir: &str, opts: EngineOptions) -> Result<Engine, String> {
    let meta = ModelMetadata::load(Path::new(dir), &ArchitectureRegistry::with_builtins()).unwrap();
    assert_eq!(meta.manifest.architecture.id, "transformer_moe");
    Engine::load(
        meta,
        EngineOptions {
            max_context: Some(512),
            ..opts
        },
    )
    .map_err(|e| e.to_string())
}

#[test]
fn gpt_oss_matches_transformers() {
    let Ok(dir) = std::env::var("QUBIVEX_GPT_OSS_DIR") else {
        eprintln!("skipping: QUBIVEX_GPT_OSS_DIR not set");
        return;
    };
    let o = oracle();
    let e = engine(&dir, EngineOptions::default()).unwrap();
    let tok = e.tokenizer();

    for t in o["texts"].as_array().unwrap() {
        let text = t["text"].as_str().unwrap();
        assert_eq!(tok.encode(text, false).unwrap(), ids(&t["ids"]), "{text:?}");
    }
    let msgs: Vec<ChatMessage> = o["chat"]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| ChatMessage {
            role: m["role"].as_str().unwrap().to_owned(),
            content: m["content"].as_str().unwrap().to_owned(),
        })
        .collect();
    let chat = tok.apply_chat_template(&msgs, true).unwrap();
    assert_eq!(undated(&chat), undated(o["chat"]["text"].as_str().unwrap()));

    // Layers 0 and 1 of the full model, traced over the long prompt.
    let prompt = ids(&o["prompt"]);
    let positions: Vec<usize> = ids(&o["positions"]).iter().map(|&p| p as usize).collect();
    // (point, layer, position) -> (values, selected experts)
    type Traces = HashMap<(TracePoint, u32, usize), (Vec<f32>, Vec<u32>)>;
    let mut got: Traces = HashMap::new();
    let mut tracer = |ev: TraceEvent<'_>| {
        if let Some(l) = ev.layer.filter(|&l| l < 2) {
            got.insert(
                (ev.point, l, ev.token),
                (ev.values.to_vec(), ev.experts.to_vec()),
            );
        }
    };
    let mut cache = e.new_cache();
    e.forward(&prompt, &mut cache, Logits::Last, Some(&mut tracer))
        .unwrap();
    let mut worst = 0f64;
    for (l, lw) in o["layers"].as_array().unwrap().iter().enumerate() {
        let l32 = l as u32;
        let ne = floats(&lw["router_logits"]).len() / prompt.len();
        let ours: Vec<f32> = (0..prompt.len())
            .flat_map(|t| got[&(TracePoint::RouterLogits, l32, t)].0.clone())
            .collect();
        let r = rel(&ours, &floats(&lw["router_logits"]));
        worst = worst.max(r);
        assert!(r < TOL, "layer {l} router logits: {r:.2e} ({ne} experts)");
        let weights = floats(&lw["routing_weights"]);
        let experts = lw["experts"].as_array().unwrap();
        let k = weights.len() / prompt.len();
        for t in 0..prompt.len() {
            let (w, sel) = &got[&(TracePoint::RoutingWeights, l32, t)];
            assert_eq!(sel, &ids(&experts[t]), "layer {l} token {t}: experts");
            let r = rel(w, &weights[t * k..(t + 1) * k]);
            assert!(r < TOL, "layer {l} token {t}: routing weights {r:.2e}");
        }
        for (point, key) in [
            (TracePoint::Attention, "attention"),
            (TracePoint::Moe, "moe"),
            (TracePoint::Hidden, "hidden"),
        ] {
            let ours: Vec<f32> = positions
                .iter()
                .flat_map(|&t| got[&(point, l32, t)].0.clone())
                .collect();
            let r = rel(&ours, &floats(&lw[key]));
            worst = worst.max(r);
            assert!(r < TOL, "layer {l} {key}: relative error {r:.2e}");
        }
    }
    eprintln!("layers 0 and 1: worst relative error {worst:.2e}");

    // A greedy reply to the chat, the same on every backend that opens.
    let chat_ids = tok.encode(&chat, false).unwrap();
    let reply = e
        .generate(&chat_ids, 24, Sampling::default(), |_| true)
        .unwrap();
    eprintln!("cpu reply: {:?}", tok.decode(&reply, false).unwrap());
    for backend in [BackendKind::Metal, BackendKind::Vulkan, BackendKind::Sim] {
        let g = match engine(
            &dir,
            EngineOptions {
                backend,
                ..EngineOptions::default()
            },
        ) {
            Ok(g) => g,
            Err(err) => {
                eprintln!("{}: skipped ({err})", backend.name());
                continue;
            }
        };
        let got = g
            .generate(&chat_ids, 24, Sampling::default(), |_| true)
            .unwrap();
        assert_eq!(got, reply, "{}: tokens", backend.name());
        eprintln!("{}: the CPU's tokens", backend.name());
    }
}
