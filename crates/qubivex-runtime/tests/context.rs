//! Context management through the runtime (ADR-021), on the fixture models: their real
//! chat templates and tokenizers, a context of 256 positions.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Barrier};

use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::context::{ContextConfig, ContextEngine, ContextError};
use qubivex_runtime::{
    ChatMessage, Engine, EngineOptions, GenerationEvent, GenerationRequest, ManageOptions,
    Prompt, RequestError,
};

fn engine_with(id: &str, ctx: usize) -> Engine {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/models");
    let meta = ModelMetadata::load(&dir.join(id), &ArchitectureRegistry::with_builtins()).unwrap();
    Engine::load(
        meta,
        EngineOptions {
            max_context: Some(ctx),
            pressure_interval_ms: Some(0),
            ..EngineOptions::default()
        },
    )
    .unwrap()
}

fn engine() -> Engine {
    engine_with("qwen3-moe-tiny", 256)
}

fn on() -> ContextEngine {
    ContextEngine::in_memory(ContextConfig::default())
}

fn chat(msgs: &[(&str, &str)]) -> GenerationRequest {
    let mut r = GenerationRequest::new(Prompt::Chat(
        msgs.iter().map(|(r, c)| ChatMessage::new(*r, *c)).collect(),
    ));
    r.max_tokens = Some(16);
    r
}

/// A conversation that is far longer than 256 positions.
fn long_chat() -> GenerationRequest {
    let mut m: Vec<(String, String)> = vec![("system".into(), "Be brief.".into())];
    for i in 0..12 {
        m.push(("user".into(), format!("Note {i}: the colour of box {i} is blue.")));
        m.push(("assistant".into(), format!("Noted, box {i} is blue.")));
    }
    m.push(("user".into(), "What colour is box 4?".into()));
    let mut r = GenerationRequest::new(Prompt::Chat(
        m.iter().map(|(r, c)| ChatMessage::new(r.as_str(), c.as_str())).collect(),
    ));
    r.max_tokens = Some(16);
    r
}

fn generate(e: &Engine, m: &qubivex_runtime::Managed) -> (usize, String) {
    let kv = e.reserve_kv(m.prepared.positions()).unwrap();
    let mut text = String::new();
    let mut started = 0;
    e.run_prepared(&m.request, &m.prepared, &kv, &AtomicBool::new(false), |ev| {
        match ev {
            GenerationEvent::Started { prompt_tokens } => started = prompt_tokens,
            GenerationEvent::Text(t) => text.push_str(&t),
            GenerationEvent::Token(_) => {}
        }
        true
    })
    .unwrap();
    (started, text)
}

#[test]
fn a_request_that_fits_is_left_exactly_as_it_was() {
    let e = engine();
    let req = chat(&[("system", "Be brief."), ("user", "Hello")]);
    let plain = e.prepare(&req).unwrap();
    let m = e
        .prepare_managed(req.clone(), Some(&on()), &ManageOptions::default())
        .unwrap();
    assert!(m.report.is_none());
    assert_eq!(m.prepared.prompt_tokens(), plain.prompt_tokens());
    assert_eq!(m.prepared.positions(), plain.positions());
    assert_eq!(format!("{:?}", m.request.prompt), format!("{:?}", req.prompt));
    // And it generates the same tokens as the plain path.
    let (_, a) = generate(&e, &m);
    let kv = e.reserve_kv(plain.positions()).unwrap();
    let mut b = String::new();
    e.run_prepared(&req, &plain, &kv, &AtomicBool::new(false), |ev| {
        if let GenerationEvent::Text(t) = ev {
            b.push_str(&t);
        }
        true
    })
    .unwrap();
    assert_eq!(a, b);
}

#[test]
fn a_conversation_longer_than_the_context_is_prepared_and_generates() {
    let e = engine();
    let req = long_chat();
    assert!(matches!(
        e.prepare(&req),
        Err(RequestError::ContextOverflow { .. })
    ));
    let m = e
        .prepare_managed(req, Some(&on()), &ManageOptions::default())
        .unwrap();
    let r = m.report.as_ref().unwrap();
    assert!(r.compacted);
    assert!(r.original_tokens > e.max_context());
    assert_eq!(r.prepared_tokens, m.prepared.prompt_tokens(), "the counts are exact");
    assert!(m.prepared.prompt_tokens() <= r.budget.input);
    assert_eq!(r.budget.context, e.max_context());
    let (started, _) = generate(&e, &m);
    assert_eq!(started, m.prepared.prompt_tokens());
    let Prompt::Chat(msgs) = &m.request.prompt else {
        panic!()
    };
    assert_eq!(msgs[0].content, "Be brief.");
    assert!(msgs.last().unwrap().content.ends_with("What colour is box 4?"));
}

#[test]
fn with_context_management_off_the_old_refusal_stands() {
    let e = engine();
    let off = ContextEngine::in_memory(ContextConfig {
        enabled: false,
        ..ContextConfig::default()
    });
    for ctx in [None, Some(&off)] {
        let err = e
            .prepare_managed(long_chat(), ctx, &ManageOptions::default())
            .unwrap_err();
        assert!(matches!(err, RequestError::ContextOverflow { .. }), "{err:?}");
        assert_eq!(err.to_string(), e.prepare(&long_chat()).unwrap_err().to_string());
    }
}

#[test]
fn instructions_too_long_for_the_context_are_refused_with_the_reason() {
    let e = engine();
    let policy = "Rule: always answer in French. ".repeat(30);
    let req = chat(&[("system", &policy), ("user", "Hello")]);
    let err = e
        .prepare_managed(req, Some(&on()), &ManageOptions::default())
        .unwrap_err();
    assert!(
        matches!(err, RequestError::Context(ContextError::Mandatory { .. })),
        "{err:?}"
    );
}

#[test]
fn a_raw_prompt_longer_than_the_context_is_reduced() {
    let e = engine();
    let text: String = (0..80).map(|i| format!("line {i}\n")).collect();
    let mut req = GenerationRequest::new(Prompt::Text(text + "end"));
    req.max_tokens = Some(8);
    let m = e
        .prepare_managed(req, Some(&on()), &ManageOptions::default())
        .unwrap();
    assert!(m.report.as_ref().unwrap().reduced);
    assert!(m.prepared.prompt_tokens() < e.max_context());
    generate(&e, &m);
}

#[test]
fn a_prompt_cap_below_the_context_bounds_the_prepared_prompt() {
    let e = engine();
    let m = e
        .prepare_managed(
            long_chat(),
            Some(&on()),
            &ManageOptions {
                prompt_cap: Some(150),
                ..ManageOptions::default()
            },
        )
        .unwrap();
    assert!(m.prepared.prompt_tokens() <= 150);
}

#[test]
fn a_cancelled_preparation_stops() {
    let e = engine();
    let cancel = AtomicBool::new(true);
    let err = e
        .prepare_managed(
            long_chat(),
            Some(&on()),
            &ManageOptions {
                cancel: Some(&cancel),
                ..ManageOptions::default()
            },
        )
        .unwrap_err();
    assert!(matches!(err, RequestError::Context(ContextError::Cancelled)));
}

/// Models with other templates and tokenizers (a GGUF vocabulary among them) and other
/// context limits: each budget is its own model's, counted its own way.
#[test]
fn every_fixture_model_gets_a_prompt_that_fits_its_own_context() {
    for (id, ctx) in [
        ("dense-tiny", 256),
        ("qwen2-tiny", 200),
        ("olmoe-tiny", 300),
        ("olmoe-tiny-gguf", 256),
        ("gpt-oss-tiny", 384),
        ("qwen3-moe-tiny", 160),
    ] {
        let e = engine_with(id, ctx);
        if !e.tokenizer().has_chat_template() {
            continue;
        }
        let m = e
            .prepare_managed(long_chat(), Some(&on()), &ManageOptions::default())
            .unwrap_or_else(|err| panic!("{id}: {err}"));
        assert!(m.prepared.prompt_tokens() < ctx, "{id}");
        let limits = e.context_limits();
        assert_eq!(limits.effective, e.max_context(), "{id}");
        assert_eq!(limits.configured, Some(ctx), "{id}");
        assert!(limits.model >= limits.effective, "{id}");
    }
}

/// The same context engine and model shared by many threads, each with its own
/// conversation: every result fits, and the model's KV reservations all return.
#[test]
fn concurrent_requests_prepare_and_run_independently() {
    let e = Arc::new(engine());
    let ctx = Arc::new(on());
    let barrier = Arc::new(Barrier::new(6));
    let handles: Vec<_> = (0..6)
        .map(|t| {
            let (e, ctx, barrier) = (Arc::clone(&e), Arc::clone(&ctx), Arc::clone(&barrier));
            std::thread::spawn(move || {
                let mut req = long_chat();
                if let Prompt::Chat(m) = &mut req.prompt {
                    m[0].content = format!("Be brief, thread {t}.");
                }
                req.cache_scope = Some(format!("client-{t}"));
                barrier.wait();
                let m = e
                    .prepare_managed(req, Some(&ctx), &ManageOptions::default())
                    .unwrap();
                generate(&e, &m).0
            })
        })
        .collect();
    for h in handles {
        assert!(h.join().unwrap() < 256);
    }
    assert_eq!(e.kv_reserved(), 0);
}
