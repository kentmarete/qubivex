//! The real OLMoE-1B-7B-0125-Instruct Q4_K_M GGUF, when available.
//! Set QUBIVEX_OLMOE_GGUF_DIR to the imported model directory.

#![allow(clippy::unwrap_used)]

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use qubivex_arch::Logits;
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{BackendKind, Engine, EngineOptions, GenerateOptions, Sampling};

/// Colibri `c/ref_olmoe_real.json`: "The capital of France is" and 12 greedy tokens.
const PROMPT: [u32; 5] = [510, 5347, 273, 6181, 310];
const COLIBRI: [u32; 12] = [
    7785, 15, 187, 187, 510, 3565, 3448, 273, 6181, 310, 5112, 15,
];

fn engine(dir: &str, cache_mib: Option<u64>) -> Engine {
    let meta = ModelMetadata::load(Path::new(dir), &ArchitectureRegistry::with_builtins()).unwrap();
    Engine::load(
        meta,
        EngineOptions {
            expert_cache_bytes: cache_mib.map(|m| m << 20),
            max_context: Some(64),
            ..EngineOptions::default()
        },
    )
    .unwrap()
}

#[test]
fn greedy_tokens_and_cache_invariance() {
    let Ok(dir) = std::env::var("QUBIVEX_OLMOE_GGUF_DIR") else {
        return;
    };
    let mut first: Option<Vec<u32>> = None;
    for cache in [None, Some(256), Some(32)] {
        let e = engine(&dir, cache);
        let toks = e
            .generate(&PROMPT, 12, Sampling::default(), |_| true)
            .unwrap();
        assert_eq!(toks, COLIBRI, "cache {cache:?} MiB");
        let mut c = e.new_cache();
        let logits: Vec<u32> = e
            .forward(&PROMPT, &mut c, Logits::All, None)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        match &first {
            None => first = Some(logits),
            Some(f) => assert!(f == &logits, "logits differ at cache {cache:?} MiB"),
        }
    }
}

/// Prefix reuse on the real model: a repeat and a next turn give the tokens of a full
/// run on the CPU and on Metal, and the time saved is printed (ADR-016).
#[test]
fn prefix_reuse_is_bit_identical_on_olmoe() {
    let Ok(dir) = std::env::var("QUBIVEX_OLMOE_GGUF_DIR") else {
        eprintln!("skipping: QUBIVEX_OLMOE_GGUF_DIR not set");
        return;
    };
    let meta =
        || ModelMetadata::load(Path::new(&dir), &ArchitectureRegistry::with_builtins()).unwrap();
    let load = |prefix_cache_bytes, backend| {
        Engine::load(
            meta(),
            EngineOptions {
                max_context: Some(2048),
                prefix_cache_bytes,
                backend,
                ..EngineOptions::default()
            },
        )
        .unwrap()
    };
    let off = load(0, BackendKind::Cpu);
    let text = "The lighthouse keeper climbed the stairs and wrote in the log. ".repeat(50);
    let prompt = off.tokenizer().encode(&text, true).unwrap();
    let opts = GenerateOptions {
        max_new: 32,
        ..GenerateOptions::default()
    };
    let run = |e: &Engine, p: &[u32]| {
        let t = Instant::now();
        let mut first = None;
        let g = e
            .generate_with(p, &opts, &AtomicBool::new(false), |_| {
                first.get_or_insert_with(|| t.elapsed());
                true
            })
            .unwrap();
        (g, first.unwrap())
    };
    let (full, _) = run(&off, &prompt);
    let next = [
        &prompt[..],
        &full.tokens,
        &off.tokenizer().encode(" And then", false).unwrap(),
    ]
    .concat();
    let (next_full, _) = run(&off, &next);
    let mut backends = vec![BackendKind::Cpu];
    if qubivex_backend_metal::MetalBackend::new().is_ok() {
        backends.push(BackendKind::Metal);
    }
    for backend in backends {
        let on = load(1 << 30, backend);
        let (a, cold) = run(&on, &prompt);
        let (b, warm) = run(&on, &prompt);
        let (c, turn) = run(&on, &next);
        eprintln!(
            "{backend:?}: {} tokens: first token {cold:?} cold, {warm:?} repeated; \
             next turn of {} tokens with {} reused: {turn:?}",
            prompt.len(),
            next.len(),
            c.cached_tokens
        );
        assert_eq!(a.tokens, full.tokens, "{backend:?}: first run");
        assert_eq!(b.tokens, full.tokens, "{backend:?}: repeat");
        assert_eq!(b.cached_tokens, prompt.len() - 1);
        assert_eq!(c.tokens, next_full.tokens, "{backend:?}: next turn");
        assert_eq!(c.cached_tokens, prompt.len() + full.tokens.len() - 1);
    }
}
