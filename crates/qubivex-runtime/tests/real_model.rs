//! The real OLMoE-1B-7B-0125-Instruct Q4_K_M GGUF, when available.
//! Set QUBIVEX_OLMOE_GGUF_DIR to the imported model directory.

#![allow(clippy::unwrap_used)]

use std::path::Path;

use qubivex_arch::Logits;
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{Engine, EngineOptions, Sampling};

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
