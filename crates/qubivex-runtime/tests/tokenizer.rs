//! Tokenizers of real and fixture models.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::load_tokenizer;
use qubivex_tokenizer::{ChatMessage, Tokenizer};
use serde_json::Value;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

#[test]
fn fixture_tokenizers_load() {
    let reg = ArchitectureRegistry::with_builtins();
    for id in ["olmoe-tiny", "qwen3-moe-tiny", "olmoe-tiny-gguf"] {
        let meta = ModelMetadata::load(&fixtures().join("models").join(id), &reg).unwrap();
        let t = load_tokenizer(&meta).unwrap();
        assert_eq!(t.vocab_size(), 256, "{id}");
        // One token per byte, identical for the SafeTensors and GGUF forms.
        assert_eq!(
            t.encode("hi é", false).unwrap(),
            vec![104, 105, 32, 0xc3, 0xa9],
            "{id}"
        );
        assert_eq!(
            t.decode(&[104, 105, 32, 0xc3, 0xa9], false).unwrap(),
            "hi é",
            "{id}"
        );
        assert_eq!(t.eos_ids(), [1], "{id}");
        let chat = t
            .apply_chat_template(&[ChatMessage::new("user", "hi")], true)
            .unwrap();
        assert_eq!(chat, "<|user|>\nhi\n<|assistant|>\n", "{id}");
    }
}

/// The reference corpus from the official tokenizer via transformers.
fn oracle() -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("oracle/olmoe_tokenizer.json")).unwrap(),
    )
    .unwrap()
}

fn check_against_oracle(t: &Tokenizer, what: &str) {
    let o = oracle();
    for c in o["cases"].as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        let want: Vec<u32> = c["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
            .collect();
        assert_eq!(t.encode(text, false).unwrap(), want, "{what}: {text:?}");
        // Decoding gives the normalized text, exactly as transformers decodes it.
        assert_eq!(
            t.decode(&want, false).unwrap(),
            c["decoded"].as_str().unwrap(),
            "{what}: decode {text:?}"
        );
    }
    let chat = &o["chat"];
    let messages: Vec<ChatMessage> = chat["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| ChatMessage::new(m["role"].as_str().unwrap(), m["content"].as_str().unwrap()))
        .collect();
    let rendered = t.apply_chat_template(&messages, true).unwrap();
    assert_eq!(
        rendered,
        chat["rendered"].as_str().unwrap(),
        "{what}: chat template"
    );
    let ids: Vec<u32> = chat["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    assert_eq!(t.encode(&rendered, false).unwrap(), ids, "{what}: chat ids");
    assert!(
        t.eos_ids().contains(&50279),
        "{what}: eos {:?}",
        t.eos_ids()
    );
}

/// Set QUBIVEX_OLMOE_GGUF_DIR to the imported OLMoE GGUF model directory.
#[test]
fn gguf_tokenizer_matches_the_official_one() {
    let Ok(dir) = std::env::var("QUBIVEX_OLMOE_GGUF_DIR") else {
        return;
    };
    let meta =
        ModelMetadata::load(Path::new(&dir), &ArchitectureRegistry::with_builtins()).unwrap();
    check_against_oracle(&load_tokenizer(&meta).unwrap(), "gguf");
}

/// Set QUBIVEX_OLMOE_TOKENIZER_JSON to the official tokenizer.json (with its configs beside it).
#[test]
fn hf_tokenizer_matches_transformers() {
    let Ok(p) = std::env::var("QUBIVEX_OLMOE_TOKENIZER_JSON") else {
        return;
    };
    check_against_oracle(
        &Tokenizer::from_hf_dir(Path::new(&p)).unwrap(),
        "tokenizer.json",
    );
}
