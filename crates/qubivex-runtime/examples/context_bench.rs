//! What context management costs on a real model (ADR-021): the request path when a
//! request fits (management on, off, and the plain `prepare`), and preparations that
//! compact a long conversation and reduce a long document, with and without a stored
//! note. Preparation runs no forward pass; the model's weights are loaded only because
//! an engine needs them.
//!
//! cargo run --release -p qubivex-runtime --example context_bench -- <model dir> [context]

#![allow(clippy::unwrap_used, missing_docs)]

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::context::{ContextConfig, ContextEngine, MemoryStore};
use qubivex_runtime::{
    ChatMessage, Engine, EngineOptions, GenerationRequest, ManageOptions, Prompt,
};

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn time(n: usize, mut f: impl FnMut()) -> Duration {
    f(); // warm
    median(
        (0..n)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed()
            })
            .collect(),
    )
}

fn chat(m: Vec<ChatMessage>) -> GenerationRequest {
    let mut r = GenerationRequest::new(Prompt::Chat(m));
    r.max_tokens = Some(256);
    r
}

fn conversation(turns: usize) -> Vec<ChatMessage> {
    let mut m = vec![ChatMessage::new("system", "You are a helpful assistant.")];
    for i in 0..turns {
        m.push(ChatMessage::new(
            "user",
            format!(
                "Step {i}: we decided the service on host {i} uses port {}. Keep the \
                 deployment notes for host {i} with the others.",
                8000 + i
            ),
        ));
        m.push(ChatMessage::new(
            "assistant",
            format!("Noted: host {i} serves on port {}, as decided.", 8000 + i),
        ));
    }
    m.push(ChatMessage::new("user", "Which port does host 42 use?"));
    m
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .expect("usage: context_bench <model dir> [context]");
    let context: usize = args.get(2).map_or(4096, |c| c.parse().unwrap());
    let meta = ModelMetadata::load(Path::new(dir), &ArchitectureRegistry::with_builtins()).unwrap();
    let e = Engine::load(
        meta,
        EngineOptions {
            max_context: Some(context),
            pressure_interval_ms: Some(0),
            ..EngineOptions::default()
        },
    )
    .unwrap();
    let on = ContextEngine::in_memory(ContextConfig::default());
    let off = ContextEngine::in_memory(ContextConfig {
        enabled: false,
        ..ContextConfig::default()
    });
    let fresh = ContextEngine::new(ContextConfig::default(), Arc::new(MemoryStore::new(0)));
    let opts = ManageOptions::default();
    println!("model {dir}, context {}", e.max_context());

    let fits = chat(conversation(4));
    let n = e.prepare(&fits).unwrap().prompt_tokens();
    let plain = time(300, || {
        e.prepare(&fits).unwrap();
    });
    let managed_on = time(300, || {
        e.prepare_managed(fits.clone(), Some(&on), &opts).unwrap();
    });
    let managed_off = time(300, || {
        e.prepare_managed(fits.clone(), Some(&off), &opts).unwrap();
    });
    println!(
        "fits ({n} tokens): prepare {plain:?}, prepare_managed on {managed_on:?}, off {managed_off:?}"
    );

    let long = chat(conversation(300));
    let detect = time(10, || {
        e.prepare(&long).unwrap_err();
    });
    let m = e
        .prepare_managed(long.clone(), Some(&fresh), &opts)
        .unwrap();
    let r = m.report.unwrap();
    let rebuild = time(10, || {
        e.prepare_managed(long.clone(), Some(&fresh), &opts)
            .unwrap();
    });
    let stored = time(10, || {
        e.prepare_managed(long.clone(), Some(&on), &opts).unwrap();
    });
    println!(
        "conversation: {} -> {} tokens, {} messages condensed, {} excerpts, {} whole counts; \
         overflow detection (prepare) {detect:?}, compaction {rebuild:?} (compaction {:?}), \
         with a stored note {stored:?}",
        r.original_tokens,
        r.prepared_tokens,
        r.messages_condensed,
        r.excerpts,
        r.counts,
        r.compaction_time
    );

    let source =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/engine.rs"))
            .unwrap();
    let doc = chat(vec![
        ChatMessage::new("system", "Answer from the file."),
        ChatMessage::new(
            "user",
            format!("Here is engine.rs:\n\n```rust\n{source}\n```\n\nWhat does reserve_kv return?"),
        ),
    ]);
    let m = e.prepare_managed(doc.clone(), Some(&fresh), &opts).unwrap();
    let r = m.report.unwrap();
    let reduce = time(10, || {
        e.prepare_managed(doc.clone(), Some(&fresh), &opts).unwrap();
    });
    println!(
        "document ({} KiB): {} -> {} tokens, {} blocks kept, {} omitted, {} whole counts; {reduce:?} (ranking and reduction {:?})",
        source.len() / 1024,
        r.original_tokens,
        r.prepared_tokens,
        r.blocks_kept,
        r.blocks_omitted,
        r.counts,
        r.retrieval_time
    );
}
