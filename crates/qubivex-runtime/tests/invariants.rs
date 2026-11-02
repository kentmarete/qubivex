//! Invariant 1 on the fixtures: batching and the expert cache size never change bits.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use qubivex_arch::Logits;
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{Engine, EngineOptions, Sampling};

fn dir(id: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/models")
        .join(id)
}

fn engine(id: &str, cache: Option<u64>) -> Engine {
    let meta = ModelMetadata::load(&dir(id), &ArchitectureRegistry::with_builtins()).unwrap();
    Engine::load(
        meta,
        EngineOptions {
            expert_cache_bytes: cache,
            max_context: Some(256),
        },
    )
    .unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

const IDS: [&str; 3] = ["olmoe-tiny", "olmoe-tiny-gguf", "qwen3-moe-tiny"];
const PROMPT: &[u32] = &[
    72, 101, 108, 108, 111, 44, 32, 101, 120, 112, 101, 114, 116, 115,
];

#[test]
fn prefill_equals_token_by_token() {
    for id in IDS {
        let e = engine(id, None);
        let mut c = e.new_cache();
        let batch = e.forward(PROMPT, &mut c, Logits::All, None).unwrap();
        let mut c2 = e.new_cache();
        let vocab = batch.len() / PROMPT.len();
        for (t, tok) in PROMPT.iter().enumerate() {
            let one = e.forward(&[*tok], &mut c2, Logits::Last, None).unwrap();
            assert_eq!(
                bits(&one),
                bits(&batch[t * vocab..(t + 1) * vocab]),
                "{id} token {t}"
            );
        }
        assert!(batch.iter().all(|v| v.is_finite()), "{id}");
    }
}

#[test]
fn expert_cache_capacity_never_changes_bits() {
    for id in IDS {
        let full = engine(id, None);
        let largest = full.metadata().bytes().largest_expert;
        let mut want_logits = None;
        let mut want_tokens = None;
        for cap in [Some(largest), Some(2 * largest), Some(5 * largest), None] {
            let e = engine(id, cap);
            let mut c = e.new_cache();
            let logits = e.forward(PROMPT, &mut c, Logits::All, None).unwrap();
            let toks = e
                .generate(PROMPT, 12, Sampling::default(), |_| true)
                .unwrap();
            match (&want_logits, &want_tokens) {
                (None, None) => {
                    want_logits = Some(bits(&logits));
                    want_tokens = Some(toks);
                }
                (Some(l), Some(t)) => {
                    assert_eq!(&bits(&logits), l, "{id} capacity {cap:?}");
                    assert_eq!(&toks, t, "{id} capacity {cap:?}");
                }
                _ => unreachable!(),
            }
            if cap == Some(largest) {
                assert!(
                    e.store_stats().evictions > 0,
                    "{id}: one-expert cache must evict"
                );
            }
        }
    }
}

#[test]
fn sampling_is_reproducible() {
    let e = engine("qwen3-moe-tiny", None);
    let s = Sampling {
        temperature: 0.8,
        top_p: 0.9,
        seed: 42,
    };
    let a = e.generate(PROMPT, 10, s, |_| true).unwrap();
    let b = e.generate(PROMPT, 10, s, |_| true).unwrap();
    assert_eq!(a, b);
    let c = e
        .generate(PROMPT, 10, Sampling { seed: 43, ..s }, |_| true)
        .unwrap();
    assert_eq!(c.len(), 10);
    // Stopping early from the callback.
    let one = e
        .generate(PROMPT, 10, Sampling::default(), |_| false)
        .unwrap();
    assert_eq!(one.len(), 1);
}

#[test]
fn errors_are_reported() {
    let e = engine("olmoe-tiny", None);
    let mut c = e.new_cache();
    assert!(
        e.forward(&[9999], &mut c, Logits::Last, None)
            .unwrap_err()
            .to_string()
            .contains("vocabulary")
    );
    let mut c = e.new_cache();
    let long = vec![1u32; 257];
    assert!(
        e.forward(&long, &mut c, Logits::Last, None)
            .unwrap_err()
            .to_string()
            .contains("context full")
    );
    assert!(e.generate(&[], 4, Sampling::default(), |_| true).is_err());
}
