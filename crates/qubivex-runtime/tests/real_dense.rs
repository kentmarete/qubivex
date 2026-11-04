//! A real dense SafeTensors checkpoint (`transformer_dense`), when available, against
//! transformers (`fixtures/oracle/dense_real.json`, from `fixtures/oracle/dense_real.py`).
//! Set QUBIVEX_DENSE_SAFETENSORS_DIR to the imported model directory (the official
//! bf16 checkpoint named in the generator).

#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]

use std::path::Path;

use qubivex_arch::Logits;
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{Engine, EngineOptions, Sampling};
use serde_json::Value;

fn oracle() -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/oracle/dense_real.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| u32::try_from(x.as_u64().unwrap()).unwrap())
        .collect()
}

fn engine(dir: &str, threads: usize) -> Engine {
    let meta = ModelMetadata::load(Path::new(dir), &ArchitectureRegistry::with_builtins()).unwrap();
    assert_eq!(meta.manifest.architecture.id, "transformer_dense");
    Engine::load(
        meta,
        EngineOptions {
            max_context: Some(128),
            threads: Some(threads),
            ..EngineOptions::default()
        },
    )
    .unwrap()
}

/// The tokenizer's ids, the greedy tokens and the largest logits match transformers;
/// the prompt's logits equal token-by-token decoding bit for bit and do not depend on
/// the thread count.
#[test]
fn dense_checkpoint_matches_transformers() {
    let Ok(dir) = std::env::var("QUBIVEX_DENSE_SAFETENSORS_DIR") else {
        return;
    };
    let o = oracle();
    let e = engine(&dir, 4);
    let prompt = ids(&o["prompt_ids"]);
    let text = o["prompt"].as_str().unwrap();
    assert_eq!(e.tokenizer().encode(text, false).unwrap(), prompt);

    let want = ids(&o["greedy"]);
    let got = e
        .generate(&prompt, want.len(), Sampling::default(), |_| true)
        .unwrap();
    assert_eq!(got, want, "greedy tokens");

    // The largest logits of every prompt position, against transformers' f32 pass.
    // The two sum in different orders, so values agree closely but not bit for bit.
    let mut c = e.new_cache();
    let all = e.forward(&prompt, &mut c, Logits::All, None).unwrap();
    let vocab = usize::try_from(o["vocab"].as_u64().unwrap()).unwrap();
    assert_eq!(all.len(), prompt.len() * vocab);
    let mut worst = 0f64;
    for (t, (tid, tval)) in o["top_ids"]
        .as_array()
        .unwrap()
        .iter()
        .zip(o["top_logits"].as_array().unwrap())
        .enumerate()
    {
        let row = &all[t * vocab..(t + 1) * vocab];
        let top = ids(tid);
        let mut ours: Vec<usize> = (0..vocab).collect();
        ours.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
        assert_eq!(ours[0], top[0] as usize, "position {t}: argmax");
        for (&id, bits) in top.iter().zip(ids(tval)) {
            let want = f64::from(f32::from_bits(bits));
            let got = f64::from(row[id as usize]);
            let err = (got - want).abs() / want.abs().max(1.0);
            worst = worst.max(err);
            assert!(err < 1e-4, "position {t} token {id}: {got} vs {want}");
        }
    }
    eprintln!("largest relative logit difference from transformers: {worst:.2e}");

    // Invariant 1 and thread independence on the real checkpoint.
    let mut c = e.new_cache();
    for (t, &tok) in prompt.iter().enumerate() {
        let one = e.forward(&[tok], &mut c, Logits::Last, None).unwrap();
        let batch = &all[t * vocab..(t + 1) * vocab];
        assert!(
            one.iter()
                .zip(batch)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "token {t}: prefill differs from token-by-token"
        );
    }
    let e1 = engine(&dir, 1);
    let mut c = e1.new_cache();
    let single = e1.forward(&prompt, &mut c, Logits::All, None).unwrap();
    assert!(
        single
            .iter()
            .zip(&all)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "1 thread differs from 4"
    );
}
