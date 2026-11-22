//! What a preparation costs (ADR-021), against a byte-per-token stand-in counter so the
//! numbers are the engine's own work, not a tokenizer's. A request that fits never
//! reaches this crate; its cost is measured in the gateway (docs/performance).
//!
//! `cargo bench -p qubivex-context`

#![allow(missing_docs, clippy::unwrap_used)]

use criterion::{Criterion, criterion_group, criterion_main};
use qubivex_context::{
    Budget, ContextConfig, ContextEngine, Input, Interrupt, MemoryStore, Message, Request,
    TokenCounter,
};
use std::hint::black_box;
use std::sync::Arc;

struct Bytes;

impl TokenCounter for Bytes {
    fn count_chat(&self, m: &[Message]) -> Result<usize, String> {
        Ok(m.iter()
            .map(|m| m.role.len() + m.content.len() + 12)
            .sum::<usize>()
            + 14)
    }
    fn count_text(&self, t: &str) -> Result<usize, String> {
        Ok(t.len() + 1)
    }
    fn count_fragment(&self, t: &str) -> Result<usize, String> {
        Ok(t.len())
    }
}

fn conversation(turns: usize) -> Vec<Message> {
    let mut m = vec![Message::new("system", "You are a careful assistant.")];
    for i in 0..turns {
        m.push(Message::new(
            "user",
            format!(
                "Turn {i}: we decided item {i} uses port {}. Some filler text here.",
                8000 + i
            ),
        ));
        m.push(Message::new(
            "assistant",
            format!(
                "Noted: item {i} listens on {}. More filler to make it longer.",
                8000 + i
            ),
        ));
    }
    m.push(Message::new("user", "Which port does item 3 use?"));
    m
}

fn document(paragraphs: usize) -> String {
    let mut d = String::from("Review this file.\n\n");
    for i in 0..paragraphs {
        d.push_str(&format!(
            "Paragraph {i} about subsystem {i} and how it is configured.\n\n"
        ));
    }
    d + "Question: how is subsystem 42 configured?"
}

fn run(c: &mut Criterion, name: &str, input: &Input, ctx: usize, store_bytes: usize) {
    let original_tokens = match input {
        Input::Chat(m) => Bytes.count_chat(m).unwrap(),
        Input::Text(t) => Bytes.count_text(t).unwrap(),
    };
    let req = Request {
        input,
        budget: Budget::new(ctx, Some(256), 512, None),
        original_tokens,
        scope: None,
        namespace: "bench",
    };
    let engine = ContextEngine::new(
        ContextConfig::default(),
        Arc::new(MemoryStore::new(store_bytes)),
    );
    c.bench_function(name, |b| {
        b.iter(|| black_box(engine.prepare(&req, &Bytes, Interrupt::default()).unwrap()));
    });
}

fn benches(c: &mut Criterion) {
    let short = Input::Chat(conversation(100));
    let long = Input::Chat(conversation(1000));
    // No store: every preparation builds its note.
    run(c, "compact_100_turns_into_8k", &short, 8192, 0);
    run(c, "compact_1000_turns_into_8k", &long, 8192, 0);
    // A store: after the first, the note is reused (a repeated request, or the next turn).
    run(
        c,
        "compact_1000_turns_into_8k_checkpointed",
        &long,
        8192,
        MemoryStore::DEFAULT_BYTES,
    );
    let doc = Input::Chat(vec![
        Message::new("system", "Answer from the file."),
        Message::new("user", document(4000)),
    ]);
    run(c, "reduce_250kb_document_into_8k", &doc, 8192, 0);
}

criterion_group!(group, benches);
criterion_main!(group);
