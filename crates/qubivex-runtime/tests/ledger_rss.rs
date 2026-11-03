//! The memory ledger against the process's real RSS, on the real OLMoE GGUF
//! (MIGRATION_PLAN Phase 5 exit). Set QUBIVEX_OLMOE_GGUF_DIR to the imported model.
//!
//! One test in its own binary: RSS is per process, and other tests in the same
//! process would allocate alongside it.

#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]

use std::path::Path;

use qubivex_core::Bytes;
use qubivex_hardware::Host;
use qubivex_memory::Tier;
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{Engine, EngineOptions, KvCache, Sampling};

fn rss() -> u64 {
    qubivex_hardware::process_rss(&Host::live())
        .known()
        .unwrap()
        .0
}

/// Loads with `budget`, generates `tokens`, and returns (RSS growth, ledger total,
/// plan budget).
fn measure(dir: &str, budget: Option<u64>, tokens: usize) -> (u64, u64, u64) {
    let before = rss();
    let meta = ModelMetadata::load(Path::new(dir), &ArchitectureRegistry::with_builtins()).unwrap();
    let e = Engine::load(
        meta,
        EngineOptions {
            ram_budget: budget,
            ..EngineOptions::default()
        },
    )
    .unwrap();
    let prompt = [510u32, 5347, 273, 6181, 310];
    let got = e
        .generate(&prompt, tokens, Sampling::default(), |_| true)
        .unwrap();
    assert_eq!(got.len(), tokens, "no early end of sequence");
    let l = e.ledger().snapshot(Tier::Ram);
    let positions = KvCache::reserved_for(prompt.len() + tokens - 1, usize::MAX) as u64;
    let plan = e.memory_plan().unwrap();
    assert_eq!(
        l.kv.0,
        positions * plan.ram.kv_per_token.bytes.0,
        "the KV charge is the chunked allocation the plan prices"
    );
    let grown = rss().saturating_sub(before);
    let budget = plan.ram.budget.bytes.0;
    eprintln!(
        "budget {}: ledger {} (dense {}, experts {}, KV {}), RSS grew {}",
        Bytes(budget),
        l.total(),
        l.dense,
        l.expert_cache,
        l.kv,
        Bytes(grown)
    );
    drop(e);
    (grown, l.total().0, budget)
}

#[test]
fn ledger_tracks_rss() {
    let Ok(dir) = std::env::var("QUBIVEX_OLMOE_GGUF_DIR") else {
        return;
    };
    // Every expert resident, then a budget that forces the cache to shrink.
    for budget in [None, Some(1 << 30)] {
        let (grown, ledger, plan) = measure(&dir, budget, 300);
        assert!(ledger <= plan, "the ledger stays within the plan's budget");
        // What the ledger does not count: the tokenizer, metadata, thread stacks,
        // activation buffers and allocator slack. Bounded, not proportional.
        let untracked = grown as i64 - ledger as i64;
        assert!(
            untracked.unsigned_abs() <= TOLERANCE,
            "RSS grew {} but the ledger says {} (difference {} MiB)",
            Bytes(grown),
            Bytes(ledger),
            untracked / (1 << 20)
        );
    }
}

/// The largest allowed gap between RSS growth and the ledger.
const TOLERANCE: u64 = 256 << 20;
