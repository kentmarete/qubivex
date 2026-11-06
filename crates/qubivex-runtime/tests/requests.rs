//! The generation contract, used the way an embedding application uses it: no server,
//! no CLI, no async runtime.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{
    Engine, EngineOptions, Finish, GenerationEvent, GenerationOutcome, GenerationRequest, Prompt,
    RequestError, Sampling,
};

const ID: &str = "qwen3-moe-tiny";

fn engine() -> Engine {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/models");
    let meta = ModelMetadata::load(&dir.join(ID), &ArchitectureRegistry::with_builtins()).unwrap();
    Engine::load(
        meta,
        EngineOptions {
            max_context: Some(64),
            pressure_interval_ms: Some(0),
            ..EngineOptions::default()
        },
    )
    .unwrap()
}

fn request(max: usize) -> GenerationRequest {
    let mut r = GenerationRequest::new(Prompt::Text("Hello".into()));
    r.max_tokens = Some(max);
    r
}

fn run(e: &Engine, r: &GenerationRequest) -> (GenerationOutcome, Vec<u32>, String) {
    let (mut ids, mut text) = (Vec::new(), String::new());
    let o = e
        .run_request(r, &AtomicBool::new(false), |ev| {
            match ev {
                GenerationEvent::Token(id) => ids.push(id),
                GenerationEvent::Text(t) => text.push_str(&t),
                GenerationEvent::Started { .. } => {}
            }
            true
        })
        .unwrap();
    (o, ids, text)
}

#[test]
fn a_request_matches_a_direct_generation() {
    let e = engine();
    let prompt = e.tokenizer().encode("Hello", true).unwrap();
    let want = e
        .generate(&prompt, 12, Sampling::default(), |_| true)
        .unwrap();
    let (o, ids, text) = run(&e, &request(12));
    assert_eq!(ids, want);
    assert_eq!(text, e.tokenizer().decode(&want, false).unwrap());
    assert_eq!(o.prompt_tokens, prompt.len());
    assert_eq!(o.completion_tokens, want.len());
    assert_eq!(e.kv_reserved(), 0, "the reservation returns when it ends");
}

#[test]
fn concurrent_requests_on_a_shared_engine_equal_sequential_ones() {
    let e = Arc::new(engine());
    let prompts = ["Hello", "The quick brown fox", "1 2 3", "Once upon"];
    let req = |p: &str| {
        let mut r = GenerationRequest::new(Prompt::Text(p.to_owned()));
        r.max_tokens = Some(10);
        r
    };
    let want: Vec<_> = prompts.iter().map(|p| run(&e, &req(p)).1).collect();
    let workers: Vec<_> = prompts
        .iter()
        .map(|p| {
            let (e, r) = (Arc::clone(&e), req(p));
            std::thread::spawn(move || run(&e, &r).1)
        })
        .collect();
    let got: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();
    assert_eq!(got, want);
    assert_eq!(e.kv_reserved(), 0);
}

#[test]
fn a_stop_sequence_ends_the_text_before_it() {
    let e = engine();
    let (_, _, full) = run(&e, &request(12));
    let Some(stop) = full.chars().nth(2).filter(|_| full.chars().count() > 3) else {
        panic!("the fixture should generate text, got {full:?}");
    };
    let mut r = request(12);
    r.stops = vec![stop.to_string()];
    let (o, _, text) = run(&e, &r);
    assert_eq!(o.finish, Finish::StopSequence);
    assert_eq!(text, full[..full.find(stop).unwrap()]);
}

#[test]
fn deadlines_callers_and_cancellation_each_end_a_request_their_own_way() {
    let e = engine();
    let mut r = request(12);
    // A deadline that has passed stops the request before its prefill.
    r.deadline = Some(Instant::now());
    let (o, ..) = run(&e, &r);
    assert_eq!((o.finish, o.completion_tokens), (Finish::TimedOut, 0));

    let o = e
        .run_request(&request(12), &AtomicBool::new(false), |_| false)
        .unwrap();
    assert_eq!((o.finish, o.completion_tokens), (Finish::Stopped, 0));

    let cancel = AtomicBool::new(false);
    let o = e
        .run_request(&request(12), &cancel, |ev| {
            if matches!(ev, GenerationEvent::Token(_)) {
                cancel.store(true, Ordering::Release);
            }
            true
        })
        .unwrap();
    assert_eq!((o.finish, o.completion_tokens), (Finish::Cancelled, 1));
    assert_eq!(e.kv_reserved(), 0, "every way out returns the reservation");
}

#[test]
fn requests_the_context_cannot_hold_are_refused_before_any_pass() {
    let e = engine();
    let err = |r: &GenerationRequest| {
        e.run_request(r, &AtomicBool::new(false), |_| panic!("no events expected"))
            .unwrap_err()
    };
    assert!(matches!(
        err(&request(1000)),
        RequestError::Invalid {
            param: Some("max_tokens"),
            ..
        }
    ));
    let long = GenerationRequest::new(Prompt::Text("word ".repeat(500)));
    assert!(matches!(
        err(&long),
        RequestError::Invalid { param: None, .. }
    ));
    assert_eq!(e.store_stats().loads, 0);
}

#[test]
fn top_k_one_is_greedy_and_a_large_top_k_changes_nothing() {
    let e = engine();
    let sampled = |top_k| {
        let mut r = request(10);
        r.sampling = Sampling {
            temperature: 1.5,
            top_k,
            top_p: 1.0,
            seed: 7,
        };
        run(&e, &r).1
    };
    assert_eq!(sampled(1), run(&e, &request(10)).1);
    let off = sampled(0);
    assert_eq!(sampled(u32::MAX), off, "top-k above the vocabulary is off");
}
