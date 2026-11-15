//! Constrained decoding (ADR-020) on the fixtures. Their weights are random, so the
//! model has no idea of JSON: every structural byte comes from the constraint, which
//! makes them the hardest case for "every output parses".

#![allow(clippy::unwrap_used, missing_docs)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use qubivex_grammar::{Grammar, Json, Matcher, json_object, json_schema};
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{Engine, EngineOptions, Finish, GenerateOptions, Sampling};

fn engine(id: &str) -> Engine {
    let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/models")
        .join(id);
    let meta = ModelMetadata::load(&dir, &ArchitectureRegistry::with_builtins()).unwrap();
    Engine::load(
        meta,
        EngineOptions {
            max_context: Some(1024),
            ..EngineOptions::default()
        },
    )
    .unwrap()
}

const SCHEMAS: &[&str] = &[
    r#"{"type":"object","properties":{"name":{"type":"string","maxLength":8},"age":{"type":"integer"},"tags":{"type":"array","items":{"enum":["a","b"]},"maxItems":3}},"required":["name","age"],"additionalProperties":false}"#,
    r#"{"enum":["red","green","blue"]}"#,
    r#"{"type":"array","prefixItems":[{"type":"boolean"},{"type":"null"}],"items":{"type":"number"},"maxItems":4}"#,
    r##"{"$defs":{"n":{"type":"object","properties":{"v":{"type":"integer"},"kids":{"type":"array","items":{"$ref":"#/$defs/n"},"maxItems":2}},"required":["v","kids"],"additionalProperties":false}},"$ref":"#/$defs/n"}"##,
    r#"{"anyOf":[{"type":"string","maxLength":4},{"type":"integer"}]}"#,
];

fn opts(g: &Arc<Grammar>, temperature: f32, seed: u64) -> GenerateOptions {
    GenerateOptions {
        max_new: 600,
        sampling: Sampling {
            temperature,
            top_k: 0,
            top_p: 1.0,
            seed,
        },
        constraint: Some(Arc::clone(g)),
        ..GenerateOptions::default()
    }
}

/// Runs and checks one generation; returns whether it ended at end-of-sequence.
fn check(e: &Engine, g: &Arc<Grammar>, o: &GenerateOptions, what: &str) -> bool {
    let prompt = e.tokenizer().encode("Answer in JSON: ", false).unwrap();
    let out = e
        .generate_with(&prompt, o, &AtomicBool::new(false), |_| true)
        .unwrap();
    let text = e.tokenizer().decode(&out.tokens, false).unwrap();
    // Whatever the finish, the output is a prefix the grammar can still complete.
    let mut m = Matcher::new(Arc::clone(g)).unwrap();
    assert!(
        m.advance_bytes(text.as_bytes()).unwrap(),
        "{what}: {text:?} left the grammar"
    );
    if out.finish != Finish::Eos {
        assert_eq!(out.finish, Finish::Length, "{what}");
        return false;
    }
    assert!(
        m.accepts(),
        "{what}: ended at {text:?}, which is incomplete"
    );
    serde_json::from_str::<serde_json::Value>(&text)
        .unwrap_or_else(|err| panic!("{what}: {text:?} does not parse: {err}"));
    true
}

#[test]
fn random_models_only_write_valid_json() {
    for id in ["olmoe-tiny", "qwen2-tiny"] {
        let e = engine(id);
        for (i, src) in SCHEMAS.iter().enumerate() {
            let g = json_schema(&src.parse::<Json>().unwrap()).unwrap();
            let mut ended = 0;
            for seed in 0..12 {
                ended += usize::from(check(
                    &e,
                    &g,
                    &opts(&g, 1.0, seed),
                    &format!("{id} schema {i} seed {seed}"),
                ));
            }
            assert!(
                ended >= 6,
                "{id} schema {i}: only {ended} of 12 runs finished"
            );
        }
        let g = json_object().unwrap();
        for seed in 0..6 {
            check(
                &e,
                &g,
                &opts(&g, 0.7, seed),
                &format!("{id} json_object seed {seed}"),
            );
        }
        // Greedy: the unconstrained argmax is taken whenever it is allowed.
        let g = json_schema(&SCHEMAS[1].parse::<Json>().unwrap()).unwrap();
        assert!(check(
            &e,
            &g,
            &opts(&g, 0.0, 0),
            &format!("{id} greedy enum")
        ));
    }
}

#[test]
fn a_seed_fixes_the_output_and_batching_does_not_change_it() {
    let e = Arc::new(engine("olmoe-tiny"));
    let g = json_schema(&SCHEMAS[0].parse::<Json>().unwrap()).unwrap();
    let prompt = e.tokenizer().encode("{", false).unwrap();
    let run = |e: &Engine, seed| {
        e.generate_with(
            &prompt,
            &opts(&g, 0.9, seed),
            &AtomicBool::new(false),
            |_| true,
        )
        .unwrap()
        .tokens
    };
    let alone: Vec<Vec<u32>> = (0..4).map(|s| run(&e, s)).collect();
    assert_eq!(alone[1], run(&e, 1));
    // Four constrained requests at once share forward passes (ADR-017); each keeps its
    // own matcher and gets the tokens it got alone.
    let together: Vec<Vec<u32>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..4u64)
            .map(|seed| {
                let e = Arc::clone(&e);
                let run = &run;
                s.spawn(move || run(&e, seed))
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(alone, together);
}

#[test]
fn unconstrained_generation_is_unchanged_by_the_hook() {
    // The same request with `constraint: None` takes the old path; a constraint that
    // allows any JSON value changes what may be sampled, so it must differ somewhere.
    let e = engine("olmoe-tiny");
    let prompt = e.tokenizer().encode("Hello", false).unwrap();
    let plain = GenerateOptions {
        max_new: 32,
        sampling: Sampling {
            temperature: 0.8,
            top_k: 0,
            top_p: 1.0,
            seed: 3,
        },
        ..GenerateOptions::default()
    };
    let a = e
        .generate_with(&prompt, &plain, &AtomicBool::new(false), |_| true)
        .unwrap();
    let b = e.generate(&prompt, 32, plain.sampling, |_| true).unwrap();
    assert_eq!(a.tokens, b);
    assert!(e.token_trie().is_ok());
}
