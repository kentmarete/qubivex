//! Can this model run here, and what would it take? [`plan`] answers for one budget and
//! refuses when nothing fits; this keeps a refusal as an answer and adds the RAM each way
//! of running the model needs, so a "no" says how far off it is.

use qubivex_core::Bytes;
use serde::Serialize;

use crate::plan::{ACTIVATION_RESERVE, DEFAULT_CONTEXT, HostMemory, MemoryPlan, PlanRequest, plan};
use crate::{ModelFootprint, PlanningError};

/// Whether a model can run within a budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Everything stays in RAM, even at full context.
    Fits,
    /// It runs, reading experts from disk once the cache is full. Speed then depends on
    /// the disk and on how evenly the model routes.
    Streams,
    /// It cannot run within this budget.
    DoesNotFit,
}

/// RAM a model needs, by how it runs. Each figure holds `kv_context` tokens of context
/// for the requested sequences, except `minimum`, which holds one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Requirements {
    /// The least that starts: dense weights, activations, one expert per layer, and one
    /// token of KV.
    pub minimum: Bytes,
    /// The least that reaches `kv_context` tokens, with one expert per layer.
    pub at_context: Bytes,
    /// What keeps every expert in RAM at `kv_context` tokens.
    pub in_memory: Bytes,
    /// The context the figures assume: the request's, else the model's.
    pub kv_context: u64,
}

/// The answer for one model on one machine.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Assessment {
    /// The answer.
    pub verdict: Verdict,
    /// What it takes.
    pub requirements: Requirements,
    /// The plan, unless the verdict is [`Verdict::DoesNotFit`].
    pub plan: Option<MemoryPlan>,
    /// Why it does not fit, in the planner's words.
    pub refusal: Option<String>,
}

/// The RAM figures of `model` for `req`, which need no machine.
fn requirements(model: &ModelFootprint, req: &PlanRequest) -> Requirements {
    let per_token = model
        .kv_bytes_per_token
        .0
        .saturating_mul(u64::from(req.sequences));
    let kv_context = req.context.or(model.max_context).unwrap_or(DEFAULT_CONTEXT);
    let resident = model.dense_bytes().0.saturating_add(ACTIVATION_RESERVE.0);
    let kv = per_token.saturating_mul(kv_context);
    Requirements {
        minimum: Bytes(
            resident
                .saturating_add(model.cache_floor().0)
                .saturating_add(per_token),
        ),
        at_context: Bytes(
            resident
                .saturating_add(model.cache_floor().0)
                .saturating_add(kv),
        ),
        in_memory: Bytes(
            resident
                .saturating_add(model.expert_bytes().0)
                .saturating_add(kv),
        ),
        kv_context,
    }
}

/// Assesses `model` on a machine with `host` memory.
///
/// # Errors
/// [`PlanningError::BadRequest`], and [`PlanningError::UnknownAvailable`] if no budget
/// is given and available RAM is unknown. A model that does not fit is an answer, not
/// an error.
pub fn assess(
    model: &ModelFootprint,
    host: &HostMemory,
    req: &PlanRequest,
) -> Result<Assessment, PlanningError> {
    let requirements = requirements(model, req);
    match plan(model, host, req) {
        Ok(p) => {
            let cached = p.ram.expert_cache_full_context.bytes;
            let verdict = if cached >= model.expert_bytes() {
                Verdict::Fits
            } else {
                Verdict::Streams
            };
            Ok(Assessment {
                verdict,
                requirements,
                plan: Some(p),
                refusal: None,
            })
        }
        Err(e @ (PlanningError::DoesNotFit { .. } | PlanningError::ContextTooLarge { .. })) => {
            Ok(Assessment {
                verdict: Verdict::DoesNotFit,
                requirements,
                plan: None,
                refusal: Some(e.to_string()),
            })
        }
        Err(e) => Err(e),
    }
}
