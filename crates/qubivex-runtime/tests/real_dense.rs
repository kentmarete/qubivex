//! Real dense SafeTensors checkpoints (`transformer_dense`), when available, against
//! transformers (`fixtures/oracle/dense_real.py`). Each test runs when its variable
//! names the imported model directory of the official bf16 checkpoint in the generator:
//!
//! - `QUBIVEX_DENSE_SAFETENSORS_DIR`: Qwen3-0.6B (q/k norm), `dense_real.json`.
//! - `QUBIVEX_QWEN2_SAFETENSORS_DIR`: Qwen2.5-0.5B-Instruct (q/k/v biases, tied head),
//!   `qwen2_real.json`, which also checks the chat template and, where a device is
//!   present, that Metal and Vulkan give the CPU's tokens.
//! - `QUBIVEX_LLAMA3_SAFETENSORS_DIR`: Llama-3.2-3B (llama3 RoPE scaling, tied head),
//!   `llama3_real.json`.

#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]

use std::path::Path;

use qubivex_arch::Logits;
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{BackendKind, Engine, EngineOptions, Sampling};
use qubivex_tokenizer::ChatMessage;
use serde_json::Value;

fn oracle(file: &str) -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/oracle")
        .join(file);
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| u32::try_from(x.as_u64().unwrap()).unwrap())
        .collect()
}

fn load(dir: &str, opts: EngineOptions) -> Result<Engine, String> {
    let meta = ModelMetadata::load(Path::new(dir), &ArchitectureRegistry::with_builtins()).unwrap();
    assert_eq!(meta.manifest.architecture.id, "transformer_dense");
    Engine::load(meta, opts).map_err(|e| e.to_string())
}

fn engine(dir: &str, threads: usize) -> Engine {
    load(
        dir,
        EngineOptions {
            max_context: Some(128),
            threads: Some(threads),
            ..EngineOptions::default()
        },
    )
    .unwrap()
}

/// The tokenizer's ids, the greedy tokens and the largest logits match transformers;
/// the prompt's logits equal token-by-token decoding bit for bit and do not depend on
/// the thread count. With a `chat` section, the chat template renders the same text and
/// ids, and greedy decoding after it gives the same reply.
fn check(dir: &str, file: &str) -> Engine {
    let o = oracle(file);
    let e = engine(dir, 4);
    let prompt = ids(&o["prompt_ids"]);
    let text = o["prompt"].as_str().unwrap();
    assert_eq!(e.tokenizer().encode(text, false).unwrap(), prompt);

    let want = ids(&o["greedy"]);
    let got = e
        .generate(&prompt, want.len(), Sampling::default(), |_| true)
        .unwrap();
    assert_eq!(got, want, "greedy tokens");

    // The largest logits of every prompt position, against transformers' f32 pass.
    // The two sum in different orders, so values agree closely but not bit for bit.
    let mut c = e.new_cache();
    let all = e.forward(&prompt, &mut c, Logits::All, None).unwrap();
    let vocab = usize::try_from(o["vocab"].as_u64().unwrap()).unwrap();
    assert_eq!(all.len(), prompt.len() * vocab);
    let mut worst = 0f64;
    for (t, (tid, tval)) in o["top_ids"]
        .as_array()
        .unwrap()
        .iter()
        .zip(o["top_logits"].as_array().unwrap())
        .enumerate()
    {
        let row = &all[t * vocab..(t + 1) * vocab];
        let top = ids(tid);
        let mut ours: Vec<usize> = (0..vocab).collect();
        ours.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
        assert_eq!(ours[0], top[0] as usize, "position {t}: argmax");
        for (&id, bits) in top.iter().zip(ids(tval)) {
            let want = f64::from(f32::from_bits(bits));
            let got = f64::from(row[id as usize]);
            let err = (got - want).abs() / want.abs().max(1.0);
            worst = worst.max(err);
            assert!(err < 1e-4, "position {t} token {id}: {got} vs {want}");
        }
    }
    eprintln!("{file}: largest relative logit difference from transformers: {worst:.2e}");

    // Invariant 1 and thread independence on the real checkpoint.
    let mut c = e.new_cache();
    for (t, &tok) in prompt.iter().enumerate() {
        let one = e.forward(&[tok], &mut c, Logits::Last, None).unwrap();
        let batch = &all[t * vocab..(t + 1) * vocab];
        assert!(
            one.iter()
                .zip(batch)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "token {t}: prefill differs from token-by-token"
        );
    }
    let e1 = engine(dir, 1);
    let mut c = e1.new_cache();
    let single = e1.forward(&prompt, &mut c, Logits::All, None).unwrap();
    assert!(
        single
            .iter()
            .zip(&all)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "1 thread differs from 4"
    );

    if let Some(chat) = o.get("chat") {
        let msgs: Vec<ChatMessage> = chat["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| ChatMessage::new(m["role"].as_str().unwrap(), m["content"].as_str().unwrap()))
            .collect();
        let rendered = e.tokenizer().apply_chat_template(&msgs, true).unwrap();
        assert_eq!(
            rendered,
            chat["rendered"].as_str().unwrap(),
            "chat template text"
        );
        let chat_ids = ids(&chat["ids"]);
        assert_eq!(
            e.tokenizer().encode(&rendered, false).unwrap(),
            chat_ids,
            "chat ids"
        );
        let reply = ids(&chat["greedy"]);
        let got = e
            .generate(&chat_ids, reply.len(), Sampling::default(), |_| true)
            .unwrap();
        // `generate` stops at an end-of-sequence id and leaves it out, so a reply that
        // ended there comes back one token short: the stop itself is checked too, and
        // the id must be one the tokenizer knows as an end of sequence.
        match reply.split_last() {
            Some((last, text)) if ids(&chat["eos"]).contains(last) => {
                assert!(e.tokenizer().eos_ids().contains(last), "eos id {last}");
                assert_eq!(got, text, "chat reply, stopped at the end of sequence");
            }
            _ => assert_eq!(got, reply, "chat reply"),
        }
    }
    e
}

#[test]
fn dense_checkpoint_matches_transformers() {
    let Ok(dir) = std::env::var("QUBIVEX_DENSE_SAFETENSORS_DIR") else {
        return;
    };
    check(&dir, "dense_real.json");
}

/// Llama 3's RoPE scaling changes the low frequencies' angles; with it switched off
/// the logits leave the tolerance (checked once by hand), so this catches a missing or
/// wrong scaling. Every backend present gives the CPU's tokens.
#[test]
fn llama3_rope_checkpoint_matches_transformers() {
    let Ok(dir) = std::env::var("QUBIVEX_LLAMA3_SAFETENSORS_DIR") else {
        return;
    };
    let cpu = check(&dir, "llama3_real.json");
    same_tokens_on_devices(&dir, &cpu, "llama3_real.json");
}

/// Only a missing device skips a backend; any other load error fails the test.
fn device_missing(backend: BackendKind, err: &str) -> bool {
    match backend {
        BackendKind::Vulkan => {
            err.contains("cannot open the Vulkan device")
                && std::env::var("QUBIVEX_REQUIRE_VULKAN").as_deref() != Ok("1")
        }
        BackendKind::Metal => qubivex_backend_metal::MetalBackend::new().is_err(),
        _ => false,
    }
}

/// A checkpoint with q/k/v biases, its chat template, and the same tokens on every
/// backend present.
#[test]
fn qkv_bias_checkpoint_matches_transformers() {
    let Ok(dir) = std::env::var("QUBIVEX_QWEN2_SAFETENSORS_DIR") else {
        return;
    };
    let cpu = check(&dir, "qwen2_real.json");
    same_tokens_on_devices(&dir, &cpu, "qwen2_real.json");
}

/// Metal, Vulkan and Sim, where present, give the CPU's 16 greedy tokens.
fn same_tokens_on_devices(dir: &str, cpu: &Engine, file: &str) {
    let o = oracle(file);
    let prompt = ids(&o["prompt_ids"]);
    let want = cpu
        .generate(&prompt, 16, Sampling::default(), |_| true)
        .unwrap();
    for backend in [BackendKind::Metal, BackendKind::Vulkan, BackendKind::Sim] {
        let e = match load(
            dir,
            EngineOptions {
                backend,
                max_context: Some(128),
                ..EngineOptions::default()
            },
        ) {
            Ok(e) => e,
            Err(e) if device_missing(backend, &e) => {
                eprintln!("{backend:?} skipped: {e}");
                continue;
            }
            Err(e) => panic!("{backend:?}: {e}"),
        };
        let got = e
            .generate(&prompt, 16, Sampling::default(), |_| true)
            .unwrap();
        assert_eq!(got, want, "{backend:?} tokens");
        eprintln!("{backend:?}: the CPU's tokens");
    }
}
