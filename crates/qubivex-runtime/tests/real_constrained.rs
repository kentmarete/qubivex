//! Constrained decoding (ADR-020) on a real vocabulary and model, Qwen2.5-0.5B-Instruct
//! (151 665 tokens), when `QUBIVEX_QWEN2_SAFETENSORS_DIR` names its imported directory.
//!
//! Checks every token's bytes against the tokenizer's own decoding, times masks, and
//! generates chat replies under schemas at temperature 0 and 0.8: every reply must end
//! at end-of-sequence, parse, and stay in the grammar. With
//! `QUBIVEX_GRAMMAR_SAMPLES=<file>` the replies are written out for
//! `fixtures/oracle/validate_json.py` (Python `jsonschema`).

#![allow(clippy::unwrap_used, clippy::cast_precision_loss, missing_docs)]

use std::io::Write as _;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use qubivex_grammar::{Json, Matcher, json_object, json_schema};
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{Engine, EngineOptions, Finish, GenerateOptions, Sampling};
use qubivex_tokenizer::ChatMessage;

const CASES: &[(&str, &str)] = &[
    (
        "Extract the person: Maria Lopez is 34 years old and lives in Madrid.",
        r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"city":{"type":"string"}},"required":["name","age","city"],"additionalProperties":false}"#,
    ),
    (
        "Classify the sentiment of: \"The food was cold and the waiter was rude.\"",
        r#"{"type":"object","properties":{"sentiment":{"enum":["positive","neutral","negative"]},"confidence":{"type":"number"}},"required":["sentiment","confidence"],"additionalProperties":false}"#,
    ),
    (
        "Name three primary colours.",
        r#"{"type":"array","items":{"type":"string","maxLength":20},"minItems":3,"maxItems":3}"#,
    ),
    (
        "Give a short pancake recipe.",
        r#"{"type":"object","properties":{"title":{"type":"string"},"ingredients":{"type":"array","items":{"type":"object","properties":{"item":{"type":"string"},"quantity":{"type":"string"}},"required":["item","quantity"],"additionalProperties":false},"maxItems":6},"steps":{"type":"array","items":{"type":"string"},"maxItems":5}},"required":["title","ingredients","steps"],"additionalProperties":false}"#,
    ),
    (
        "Is 17 a prime number? Answer with a boolean and a reason.",
        r#"{"type":"object","properties":{"prime":{"type":"boolean"},"reason":{"anyOf":[{"type":"string","maxLength":120},{"type":"null"}]}},"required":["prime","reason"],"additionalProperties":false}"#,
    ),
    ("Describe a cat in JSON.", ""),
];

fn engine(dir: &str) -> Engine {
    let meta = ModelMetadata::load(Path::new(dir), &ArchitectureRegistry::with_builtins()).unwrap();
    Engine::load(
        meta,
        EngineOptions {
            max_context: Some(2048),
            ..EngineOptions::default()
        },
    )
    .unwrap()
}

#[test]
fn qwen2_vocabulary_and_replies() {
    let Ok(dir) = std::env::var("QUBIVEX_QWEN2_SAFETENSORS_DIR") else {
        return;
    };
    let e = engine(&dir);
    let tok = e.tokenizer();

    // Every token's bytes decode to what the tokenizer decodes.
    let bytes = tok.token_bytes().unwrap();
    let (mut checked, mut special) = (0, 0);
    for (id, b) in (0u32..).zip(&bytes) {
        match b {
            None => special += 1,
            Some(b) => {
                if let Ok(s) = std::str::from_utf8(b) {
                    assert_eq!(tok.decode(&[id], false).unwrap(), s, "token {id}");
                    checked += 1;
                }
            }
        }
    }
    eprintln!(
        "vocabulary {}: {checked} tokens decode to their bytes, {special} special",
        bytes.len()
    );
    assert!(checked > 100_000);
    for b in 0..=255u8 {
        assert!(
            bytes.iter().any(|t| t.as_deref() == Some(&[b][..])),
            "byte {b:#x} has no token"
        );
    }

    let t = Instant::now();
    let trie = e.token_trie().unwrap();
    eprintln!("trie: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);

    // Mask cost: cold (first visit of each state) and warm, along a known reply.
    let g = json_schema(&CASES[3].1.parse::<Json>().unwrap()).unwrap();
    let mut m = Matcher::new(Arc::clone(&g)).unwrap();
    let text = br#"{"title": "Pancakes", "ingredients": [{"item": "flour", "quantity": "1 cup"}], "steps": ["Mix", "Cook"]}"#;
    let (mut cold, mut n) = (0f64, 0);
    let mut worst = 0f64;
    for &b in text.iter() {
        let t = Instant::now();
        let mask = m.mask(&trie).unwrap();
        let dt = t.elapsed().as_secs_f64() * 1e3;
        assert!(mask.iter().any(|w| *w != 0));
        cold += dt;
        worst = worst.max(dt);
        n += 1;
        assert!(m.advance(b).unwrap());
    }
    eprintln!(
        "masks over {n} states, each its first visit: mean {:.2} ms, worst {worst:.2} ms",
        cold / f64::from(n)
    );

    // JSON mode on a 0.5B model can loop on repeated keys until max_tokens; that output
    // is cut short (finish `length`), never invalid. Schemas end every reply.
    let mut open_ended = 0;
    let mut dump =
        std::env::var_os("QUBIVEX_GRAMMAR_SAMPLES").map(|p| std::fs::File::create(p).unwrap());
    for (prompt, schema) in CASES {
        let (g, schema_v) = if schema.is_empty() {
            (
                json_object().unwrap(),
                serde_json::json!({"type": "object"}),
            )
        } else {
            (
                json_schema(&schema.parse::<Json>().unwrap()).unwrap(),
                serde_json::from_str(schema).unwrap(),
            )
        };
        let msgs = [
            ChatMessage::new("system", "You are a helpful assistant. Reply in JSON."),
            ChatMessage::new("user", *prompt),
        ];
        let ids = tok
            .encode(&tok.apply_chat_template(&msgs, true).unwrap(), false)
            .unwrap();
        for (temperature, seed) in [(0.0, 0), (0.8, 1), (0.8, 2), (0.8, 3)] {
            let o = GenerateOptions {
                max_new: 400,
                sampling: Sampling {
                    temperature,
                    top_k: 0,
                    top_p: 1.0,
                    seed,
                },
                constraint: Some(Arc::clone(&g)),
                ..GenerateOptions::default()
            };
            let t = Instant::now();
            let out = e
                .generate_with(&ids, &o, &AtomicBool::new(false), |_| true)
                .unwrap();
            let secs = t.elapsed().as_secs_f64();
            let text = tok.decode(&out.tokens, false).unwrap();
            eprintln!(
                "t={temperature} seed={seed} {} tokens {:.1} tok/s: {text}",
                out.tokens.len(),
                out.tokens.len() as f64 / secs
            );
            if schema.is_empty() && out.finish == Finish::Length {
                let mut m = Matcher::new(Arc::clone(&g)).unwrap();
                assert!(m.advance_bytes(text.as_bytes()).unwrap(), "{text:?}");
                open_ended += 1;
                continue;
            }
            assert_eq!(out.finish, Finish::Eos, "{prompt}: {text:?}");
            assert!(Matcher::matches(&g, text.as_bytes()).unwrap(), "{text:?}");
            serde_json::from_str::<serde_json::Value>(&text).unwrap();
            if let Some(f) = dump.as_mut() {
                writeln!(
                    f,
                    "{}",
                    serde_json::json!({"schema": schema_v, "text": text})
                )
                .unwrap();
            }
        }
    }
    assert!(
        open_ended <= 2,
        "JSON mode ran to max_tokens {open_ended} times of 4"
    );
}
