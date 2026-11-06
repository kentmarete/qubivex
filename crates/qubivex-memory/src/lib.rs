//! The QUBIVEX memory planner (ADR-005, QUBIVEX_DESIGN §4.6).
//!
//! - [`ModelFootprint`]: what a model needs, from tensor roles and sizes only.
//! - [`plan`]: one RAM budget per profile, the room the KV and expert caches share,
//!   per-layer expert slots, placements and the reasons for every number. `qubivex
//!   plan` and model loading use the same function.
//! - [`assess`]: whether a model can run at all, and how much RAM each way of running
//!   it takes, on top of [`plan`]. `qubivex models analyse` prints it.
//! - [`MemoryPlan::expert_cache_for`]: the expert-cache capacity for a KV length,
//!   and admission control (a KV that would push the cache below one slot per layer
//!   is refused).
//! - [`MemoryLedger`]: bytes actually allocated, per tier and owner. Plans are
//!   targets; the ledger is what happened.
//! - [`PressureMonitor`]: what to do when RAM runs short, in a fixed order: stop
//!   prefetch, shrink the expert cache, refuse new requests. It never kills an active
//!   request.
//!
//! RAM and VRAM budgets are separate and never summed. There is no VRAM budget until
//! a GPU backend exists (Phase 8).

mod assess;
mod footprint;
mod ledger;
mod plan;
mod pressure;

pub use assess::{Assessment, Requirements, Verdict, assess};
pub use footprint::{DenseGroup, KV_ELEMENT_BYTES, LayerExperts, ModelFootprint, RoleGroup};
pub use ledger::{Charge, LedgerSnapshot, MemoryLedger, Owner, Tier};
pub use plan::{
    ACTIVATION_RESERVE, CacheBudget, DEFAULT_CONTEXT, HostMemory, LayerSlots, Location, MemoryPlan,
    Placement, PlanRequest, Profile, Quantity, RamBudget, Refusal, Source, VramBudget, plan,
};
pub use pressure::{PressureConfig, PressureDecision, PressureLevel, PressureMonitor};
use qubivex_core::Bytes;
use thiserror::Error;

/// Why a plan cannot be made.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanningError {
    /// The request itself is invalid.
    #[error("invalid plan request: {0}")]
    BadRequest(String),
    /// No budget was given and available RAM could not be measured.
    #[error("available RAM is unknown ({0}); pass an explicit RAM budget")]
    UnknownAvailable(String),
    /// The model cannot run within the budget even with the smallest cache.
    #[error("does not fit: {detail} (needs {}, budget {})", Bytes(*needed), Bytes(*budget))]
    DoesNotFit {
        /// The least bytes that would run it.
        needed: u64,
        /// The budget.
        budget: u64,
        /// What does not fit.
        detail: String,
    },
    /// The requested context's KV does not fit beside one expert slot per layer.
    #[error(
        "a context of {requested} tokens{} does not fit; at most {fits} does",
        if *sequences > 1 { format!(" × {sequences} sequences") } else { String::new() }
    )]
    ContextTooLarge {
        /// Requested context per sequence.
        requested: u64,
        /// The largest context that fits.
        fits: u64,
        /// Sequences.
        sequences: u32,
    },
    /// The model's tensors could not be summarised.
    #[error("cannot size the model: {0}")]
    Footprint(String),
}

/// Distributes an expert cache of `bytes` over `layers` as slot counts.
///
/// Every layer gets one slot (the floor) and at most all its experts. Remaining slots
/// go one at a time to the layer with the highest `weight / slots` (ties: the lowest
/// layer) whose slot still fits, so equal weights give an even spread and a heavier
/// layer gets proportionally more (non-uniform budgets). `weights` defaults to equal;
/// a zero-weight layer keeps its floor. Never exceeds `bytes` once the floors fit:
/// `Σ slots × expert_bytes ≤ bytes`. If even the floors exceed `bytes`, every layer
/// still gets 1 (a plan never asks for this; [`plan`] refuses it first).
#[must_use]
pub fn distribute(layers: &[LayerExperts], bytes: Bytes, weights: Option<&[f64]>) -> Vec<u32> {
    let mut slots: Vec<u32> = layers.iter().map(|l| l.experts.min(1)).collect();
    let used: u64 = layers.iter().map(|l| l.expert_bytes.0).sum();
    let Some(mut left) = bytes.0.checked_sub(used) else {
        return slots;
    };
    // All experts fit: no search needed.
    let all: u64 = layers.iter().map(|l| l.total.0).sum();
    if bytes.0 >= all {
        return layers.iter().map(|l| l.experts).collect();
    }
    let weight = |i: usize| weights.map_or(1.0, |w| w[i]);
    loop {
        let mut best: Option<(usize, f64)> = None;
        for (i, l) in layers.iter().enumerate() {
            let w = weight(i);
            if slots[i] >= l.experts || l.expert_bytes.0 > left || w <= 0.0 {
                continue;
            }
            let score = w / f64::from(slots[i]);
            if best.is_none_or(|(_, s)| score > s) {
                best = Some((i, score));
            }
        }
        let Some((i, _)) = best else { break };
        slots[i] += 1;
        left -= layers[i].expert_bytes.0;
    }
    slots
}

#[cfg(test)]
mod tests {
    use super::*;
    use qubivex_core::LayerIdx;

    fn layer(i: u32, experts: u32, eb: u64) -> LayerExperts {
        LayerExperts {
            layer: LayerIdx(i),
            experts,
            expert_bytes: Bytes(eb),
            total: Bytes(eb * u64::from(experts)),
        }
    }

    #[test]
    fn distribution_is_even_then_weighted() {
        let ls = [layer(0, 8, 10), layer(1, 8, 10), layer(2, 8, 10)];
        assert_eq!(distribute(&ls, Bytes(90), None), [3, 3, 3]);
        assert_eq!(distribute(&ls, Bytes(100), None), [4, 3, 3], "ties go low");
        assert_eq!(distribute(&ls, Bytes(29), None), [1, 1, 1], "floors only");
        assert_eq!(distribute(&ls, Bytes(240), None), [8, 8, 8]);
        assert_eq!(distribute(&ls, Bytes(10_000), None), [8, 8, 8], "capped");
        // A layer with twice the demand gets about twice the slots.
        assert_eq!(
            distribute(&ls, Bytes(120), Some(&[2.0, 1.0, 1.0])),
            [6, 3, 3]
        );
        assert_eq!(
            distribute(&ls, Bytes(120), Some(&[0.0, 1.0, 1.0])),
            [1, 6, 5]
        );
    }

    #[test]
    fn distribution_spills_from_full_layers() {
        let ls = [layer(0, 2, 10), layer(1, 8, 10)];
        assert_eq!(distribute(&ls, Bytes(70), None), [2, 5]);
    }

    #[test]
    fn mixed_slot_sizes_fit_the_bytes() {
        let ls = [layer(0, 8, 7), layer(1, 8, 13)];
        let s = distribute(&ls, Bytes(100), None);
        assert!(s[0] * 7 + s[1] * 13 <= 100, "{s:?}");
        assert!(s.iter().all(|&x| x >= 1));
    }
}
