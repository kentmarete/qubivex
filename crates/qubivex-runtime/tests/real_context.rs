//! Context management on a real model (ADR-021): Qwen2.5-0.5B-Instruct, its own chat
//! template and tokenizer, loaded with a 1024-position context. Runs when
//! `QUBIVEX_QWEN2_SAFETENSORS_DIR` names the imported model directory.
//!
//! A conversation four times longer than the context is prepared, generates, and the
//! next turn reuses the stored note, so the prompt keeps its prefix and the prefix cache
//! skips it (`cached_tokens`).

#![allow(clippy::unwrap_used)]

use std::path::Path;
use std::sync::atomic::AtomicBool;

use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::context::{CheckpointUse, ContextConfig, ContextEngine};
use qubivex_runtime::{
    ChatMessage, Engine, EngineOptions, GenerationEvent, GenerationRequest, ManageOptions, Managed,
    Prompt, Sampling,
};

const CODES: [&str; 6] = ["amber", "basalt", "cobalt", "dune", "ember", "fjord"];

fn engine(dir: &str) -> Engine {
    let meta = ModelMetadata::load(Path::new(dir), &ArchitectureRegistry::with_builtins()).unwrap();
    Engine::load(
        meta,
        EngineOptions {
            max_context: Some(1024),
            prefix_cache_bytes: 64 << 20,
            pressure_interval_ms: Some(0),
            ..EngineOptions::default()
        },
    )
    .unwrap()
}

fn conversation() -> Vec<ChatMessage> {
    let mut m = vec![ChatMessage::new(
        "system",
        "You are a helpful assistant. Answer with the code word only.",
    )];
    for i in 0..60 {
        let code = CODES[i % CODES.len()];
        m.push(ChatMessage::new(
            "user",
            format!(
                "Record {i}: the code word for locker {i} is {code}-{i}. Please remember it \
                 along with the usual paperwork for that locker."
            ),
        ));
        m.push(ChatMessage::new(
            "assistant",
            format!("Recorded: locker {i} has code word {code}-{i}."),
        ));
    }
    m
}

fn run(e: &Engine, m: &Managed) -> (String, usize) {
    let kv = e.reserve_kv(m.prepared.positions()).unwrap();
    let mut text = String::new();
    let o = e
        .run_prepared(
            &m.request,
            &m.prepared,
            &kv,
            &AtomicBool::new(false),
            |ev| {
                if let GenerationEvent::Text(t) = ev {
                    text.push_str(&t);
                }
                true
            },
        )
        .unwrap();
    (text, o.cached_tokens)
}

fn request(msgs: &[ChatMessage]) -> GenerationRequest {
    let mut r = GenerationRequest::new(Prompt::Chat(msgs.to_vec()));
    r.max_tokens = Some(12);
    r.sampling = Sampling::default(); // greedy
    r.cache_scope = Some("tester".into());
    r
}

#[test]
fn a_long_conversation_on_a_real_model_is_prepared_answered_and_reused() {
    let Ok(dir) = std::env::var("QUBIVEX_QWEN2_SAFETENSORS_DIR") else {
        eprintln!("skipped: QUBIVEX_QWEN2_SAFETENSORS_DIR is not set");
        return;
    };
    let e = engine(&dir);
    let ctx = ContextEngine::in_memory(ContextConfig::default());
    let mut conv = conversation();
    conv.push(ChatMessage::new(
        "user",
        "What is the code word for locker 7?",
    ));

    let m1 = e
        .prepare_managed(request(&conv), Some(&ctx), &ManageOptions::default())
        .unwrap();
    let r1 = m1.report.clone().unwrap();
    assert!(r1.original_tokens > 4 * 1024 / 2, "{}", r1.original_tokens);
    assert!(r1.compacted);
    assert_eq!(r1.prepared_tokens, m1.prepared.prompt_tokens());
    assert!(m1.prepared.prompt_tokens() <= r1.budget.input);
    let Prompt::Chat(sent) = &m1.request.prompt else {
        panic!()
    };
    // Locker 7's record is condensed, and relevant, so the note quotes it verbatim.
    assert!(
        sent[1]
            .content
            .contains("the code word for locker 7 is basalt-7"),
        "{}",
        sent[1].content
    );
    let (a1, cached1) = run(&e, &m1);
    eprintln!(
        "turn 1: {} -> {} tokens, {} condensed, {} excerpts, {:?}, answer {a1:?}, cached {cached1}",
        r1.original_tokens, r1.prepared_tokens, r1.messages_condensed, r1.excerpts, r1.total_time
    );
    assert!(a1.contains("basalt-7"), "the model answered {a1:?}");

    conv.push(ChatMessage::new("assistant", a1.trim()));
    conv.push(ChatMessage::new("user", "And for locker 13?"));
    let m2 = e
        .prepare_managed(request(&conv), Some(&ctx), &ManageOptions::default())
        .unwrap();
    let r2 = m2.report.clone().unwrap();
    assert_eq!(r2.checkpoint, CheckpointUse::Hit);
    let (a2, cached2) = run(&e, &m2);
    eprintln!(
        "turn 2: {} -> {} tokens, checkpoint {:?}, {:?}, answer {a2:?}, cached {cached2}",
        r2.original_tokens, r2.prepared_tokens, r2.checkpoint, r2.total_time
    );
    // Locker 13 was not in the stored note; the excerpts chosen for this question are.
    assert!(a2.contains("basalt-13"), "the model answered {a2:?}");
    // The stable prefix (system message and the stored note) came from retained KV.
    assert!(cached2 > 0, "no prefix reuse");
}
