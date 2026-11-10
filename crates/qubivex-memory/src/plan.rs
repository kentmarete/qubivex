//! The memory plan: one budget, the room the KV and expert caches share, per-layer
//! cache slots, placements and the reasons for each number (ADR-005).
//!
//! The budget follows Colibri's measured defaults (COLIBRI_ANALYSIS §6):
//!
//! - `budget = dense + f × available`, with `available` measured after the dense
//!   weights are resident; before loading, `available − dense` stands in for it (an
//!   estimate). `f` is the profile's fraction, 0.88 for `performance`;
//! - `room = budget − dense − activation reserve` (0.5 GiB);
//! - the KV cache and the expert cache **share** `room`: the expert cache starts with
//!   all of it (up to every expert) and gives bytes back as the KV grows, never below
//!   one slot per layer. The full-context KV is never reserved up front, but a context
//!   is admitted only if its full KV still leaves that floor.

use qubivex_core::{Bytes, LayerIdx};
use qubivex_hardware::{MemoryInfo, Probed, ResourceSnapshot};
use serde::Serialize;

use crate::footprint::{LayerExperts, ModelFootprint};
use crate::{PlanningError, distribute};

/// Activation and scratch reserve, Colibri's measured default.
pub const ACTIVATION_RESERVE: Bytes = Bytes(512 << 20);

/// The context used when neither the request nor the model states one.
pub const DEFAULT_CONTEXT: u64 = 4096;

/// Where a number comes from (ADR-005: every number is tagged).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    /// Read from the OS or from the model's file headers.
    Measured,
    /// Derived from measurements by a formula or a policy constant.
    Estimated,
    /// Given by the user.
    Requested,
}

/// A byte count and where it comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Quantity {
    /// The bytes.
    pub bytes: Bytes,
    /// Where they come from.
    pub source: Source,
}

impl Quantity {
    const fn measured(b: Bytes) -> Self {
        Self {
            bytes: b,
            source: Source::Measured,
        }
    }
    const fn estimated(b: Bytes) -> Self {
        Self {
            bytes: b,
            source: Source::Estimated,
        }
    }
}

/// How much of the machine a model may take.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "lowercase", tag = "profile", content = "fraction")]
pub enum Profile {
    /// 0.88 of available RAM: Colibri's measured default.
    #[default]
    Performance,
    /// 0.75 of available RAM, leaving room for other programs.
    Balanced,
    /// 0.50 of available RAM.
    Saver,
    /// A chosen fraction of available RAM, in `(0, 1]`.
    Custom(f64),
}

impl Profile {
    /// The fraction of available RAM the budget may take.
    #[must_use]
    pub const fn fraction(self) -> f64 {
        match self {
            Self::Performance => 0.88,
            Self::Balanced => 0.75,
            Self::Saver => 0.50,
            Self::Custom(f) => f,
        }
    }

    /// The profile's name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Performance => "performance",
            Self::Balanced => "balanced",
            Self::Saver => "saver",
            Self::Custom(_) => "custom",
        }
    }
}

/// The machine's RAM, as far as it is known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostMemory {
    /// RAM this process may use at most (the cgroup limit if smaller).
    pub total: Probed<Bytes>,
    /// RAM available now.
    pub available: Probed<Bytes>,
    /// Whether `available` was measured with the dense weights already resident
    /// (at load time), rather than before loading (`qubivex plan`).
    pub dense_resident: bool,
}

impl HostMemory {
    /// From a hardware profile and a snapshot taken before loading.
    #[must_use]
    pub fn before_load(memory: &MemoryInfo, now: &ResourceSnapshot) -> Self {
        Self {
            total: memory.effective_total(),
            available: now.ram_available.clone(),
            dense_resident: false,
        }
    }
}

/// What the plan is for.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanRequest {
    /// The policy profile.
    pub profile: Profile,
    /// An explicit RAM budget, replacing the profile's (Colibri's `RAM_GB`).
    pub ram_budget: Option<Bytes>,
    /// Context per sequence. `None`: the model's limit (or [`DEFAULT_CONTEXT`]),
    /// shortened to what fits.
    pub context: Option<u64>,
    /// Sequences whose KV must fit at once.
    pub sequences: u32,
    /// Relative cache demand per expert layer (non-uniform budgets), one weight per
    /// entry of [`ModelFootprint::layers`]. `None`: uniform.
    pub layer_weights: Option<Vec<f64>>,
}

impl Default for PlanRequest {
    fn default() -> Self {
        Self {
            profile: Profile::Performance,
            ram_budget: None,
            context: None,
            sequences: 1,
            layer_weights: None,
        }
    }
}

/// The RAM side of a plan. VRAM budgets are separate and never summed with these.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RamBudget {
    /// Everything the model may use.
    pub budget: Quantity,
    /// Resident (non-expert) tensors.
    pub dense: Quantity,
    /// Activation and scratch reserve.
    pub activations: Quantity,
    /// `budget − dense − activations`: shared by the KV and expert caches.
    pub room: Quantity,
    /// KV bytes per token, all sequences.
    pub kv_per_token: Quantity,
    /// KV bytes when every sequence is at full context.
    pub kv_full_context: Quantity,
    /// Every routed expert.
    pub experts_total: Quantity,
    /// One slot per layer: the smallest cache that runs the model.
    pub cache_floor: Quantity,
    /// The expert cache with an empty KV.
    pub expert_cache_start: Quantity,
    /// The expert cache when the KV is at full context.
    pub expert_cache_full_context: Quantity,
}

/// One device's VRAM budget. None exist until a discrete-GPU backend (CUDA or Vulkan)
/// is built; Metal shares host memory and is planned as RAM.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VramBudget {
    /// Device index.
    pub device: u32,
    /// Everything the model may use on the device.
    pub budget: Quantity,
}

/// Cache slots of one expert layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct LayerSlots {
    /// The layer.
    pub layer: LayerIdx,
    /// Experts the cache may hold for the layer.
    pub slots: u32,
    /// Experts in the layer.
    pub experts: u32,
}

/// Per-layer expert-cache slots at both ends of the KV's growth.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CacheBudget {
    /// With an empty KV.
    pub at_start: Vec<LayerSlots>,
    /// With the KV at full context.
    pub at_full_context: Vec<LayerSlots>,
}

/// Where a group of tensors lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Location {
    /// Resident in RAM for the model's lifetime.
    Ram,
    /// In the RAM expert cache, loaded on first use and evictable.
    RamCache,
    /// Read from the model file when needed.
    Disk,
}

/// One placement decision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Placement {
    /// What is placed (a role group, or routed experts).
    pub what: String,
    /// Bytes placed there.
    pub bytes: Quantity,
    /// Where.
    pub location: Location,
    /// Why.
    pub reason: String,
}

/// A memory plan (ADR-005).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MemoryPlan {
    /// The profile used.
    pub profile: Profile,
    /// Context per sequence.
    pub context: u64,
    /// Sequences.
    pub sequences: u32,
    /// RAM.
    pub ram: RamBudget,
    /// VRAM, per device (empty: no GPU backend yet).
    pub vram: Vec<VramBudget>,
    /// Per-layer expert-cache slots.
    pub expert_cache: CacheBudget,
    /// Placement by tensor role group.
    pub placements: Vec<Placement>,
    /// Why the numbers are what they are, in plain sentences.
    pub reasons: Vec<String>,
}

/// Why a KV length was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "{kv_tokens} KV tokens need {kv}; with the {floor} expert-cache floor that exceeds the {room} room"
)]
pub struct Refusal {
    /// The KV tokens asked for (all sequences).
    pub kv_tokens: u64,
    /// Their bytes.
    pub kv: Bytes,
    /// The expert-cache floor.
    pub floor: Bytes,
    /// The shared room.
    pub room: Bytes,
}

impl MemoryPlan {
    /// The expert-cache capacity when the KV holds `kv_tokens` tokens in all
    /// (Colibri's `kv_room_fit`): the room left after the KV, up to every expert.
    ///
    /// # Errors
    /// [`Refusal`] if those tokens would push the cache below one slot per layer.
    /// Admission control calls this with the projected KV of a new request.
    pub fn expert_cache_for(&self, kv_tokens: u64) -> Result<Bytes, Refusal> {
        let r = &self.ram;
        let per_token = r.kv_per_token.bytes.0 / u64::from(self.sequences.max(1));
        let kv = Bytes(kv_tokens.saturating_mul(per_token));
        let refuse = Refusal {
            kv_tokens,
            kv,
            floor: r.cache_floor.bytes,
            room: r.room.bytes,
        };
        let left = r.room.bytes.0.checked_sub(kv.0).ok_or(refuse)?;
        if left < r.cache_floor.bytes.0 {
            return Err(refuse);
        }
        Ok(Bytes(left.min(r.experts_total.bytes.0)))
    }
}

/// `f × b`, rounded down.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn scale(b: Bytes, f: f64) -> Bytes {
    Bytes((b.0 as f64 * f).floor() as u64)
}

/// Per-layer slots for a cache of `bytes`.
fn slots(layers: &[LayerExperts], bytes: Bytes, weights: Option<&[f64]>) -> Vec<LayerSlots> {
    let counts = distribute(layers, bytes, weights);
    layers
        .iter()
        .zip(counts)
        .map(|(l, slots)| LayerSlots {
            layer: l.layer,
            slots,
            experts: l.experts,
        })
        .collect()
}

/// Plans `model` on a machine with `host` memory.
///
/// # Errors
/// [`PlanningError::BadRequest`] for an invalid request,
/// [`PlanningError::UnknownAvailable`] if no budget is given and available RAM is
/// unknown, [`PlanningError::DoesNotFit`] if the dense weights, the reserve and one
/// expert slot per layer exceed the budget, and [`PlanningError::ContextTooLarge`]
/// if an explicit context's KV does not fit beside that floor.
#[allow(clippy::too_many_lines)]
pub fn plan(
    model: &ModelFootprint,
    host: &HostMemory,
    req: &PlanRequest,
) -> Result<MemoryPlan, PlanningError> {
    let f = req.profile.fraction();
    if !(f > 0.0 && f <= 1.0) {
        return Err(PlanningError::BadRequest(format!(
            "the RAM fraction must be in (0, 1], got {f}"
        )));
    }
    if req.sequences == 0 {
        return Err(PlanningError::BadRequest("at least one sequence".into()));
    }
    if let Some(w) = &req.layer_weights {
        if w.len() != model.layers.len() {
            return Err(PlanningError::BadRequest(format!(
                "{} layer weights for {} expert layers",
                w.len(),
                model.layers.len()
            )));
        }
        if w.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err(PlanningError::BadRequest(
                "layer weights must be finite and non-negative".into(),
            ));
        }
    }
    if req.context == Some(0) {
        return Err(PlanningError::BadRequest(
            "the context must be at least 1".into(),
        ));
    }

    let mut reasons = Vec::new();
    let dense = model.dense_bytes();
    let experts_total = model.expert_bytes();
    let floor = model.cache_floor();
    let kv_per_token = model
        .kv_bytes_per_token
        .0
        .saturating_mul(u64::from(req.sequences));

    // The budget.
    let budget = if let Some(b) = req.ram_budget {
        reasons.push(format!("RAM budget {b}, as requested."));
        if let Probed::Known(t) = host.total
            && b > t
        {
            reasons.push(format!(
                "Warning: the requested budget exceeds this machine's {t} of RAM; the OS will page."
            ));
        }
        Quantity {
            bytes: b,
            source: Source::Requested,
        }
    } else {
        let Probed::Known(avail) = host.available.clone() else {
            let why = match &host.available {
                Probed::Unknown(r) => r.clone(),
                Probed::Known(_) => String::new(),
            };
            return Err(PlanningError::UnknownAvailable(why));
        };
        let b = if host.dense_resident {
            reasons.push(format!(
                "RAM budget = dense {dense} + {f:.2} × {avail} available, measured with the dense weights resident ({} profile).",
                req.profile.name()
            ));
            Bytes(dense.0.saturating_add(scale(avail, f).0))
        } else {
            let after = avail.saturating_sub(dense);
            reasons.push(format!(
                "RAM budget = dense {dense} + {f:.2} × ({avail} available − dense) ({} profile). Available RAM is measured now, before loading; load time measures it again with the dense weights resident.",
                req.profile.name()
            ));
            if avail < dense {
                return Err(PlanningError::DoesNotFit {
                    needed: dense
                        .0
                        .saturating_add(ACTIVATION_RESERVE.0)
                        .saturating_add(floor.0),
                    budget: avail.0,
                    detail: format!(
                        "the dense weights ({dense}) exceed the {avail} of RAM available"
                    ),
                });
            }
            Bytes(dense.0.saturating_add(scale(after, f).0))
        };
        let b = match host.total {
            Probed::Known(t) if b > t => {
                reasons.push(format!("Capped at this machine's {t} of RAM."));
                t
            }
            _ => b,
        };
        Quantity::estimated(b)
    };

    // The shared room.
    let needed = dense
        .0
        .saturating_add(ACTIVATION_RESERVE.0)
        .saturating_add(floor.0)
        .saturating_add(kv_per_token);
    if budget.bytes.0 < needed {
        return Err(PlanningError::DoesNotFit {
            needed,
            budget: budget.bytes.0,
            detail: format!(
                "dense {dense} + activations {ACTIVATION_RESERVE} + one expert per layer {floor} + one KV token {} exceed the {} budget",
                Bytes(kv_per_token),
                budget.bytes
            ),
        });
    }
    let room = Bytes(budget.bytes.0 - dense.0 - ACTIVATION_RESERVE.0);
    reasons.push(format!(
        "Room = budget − dense {dense} − activation reserve {ACTIVATION_RESERVE} = {room}, shared by the KV cache and the expert cache."
    ));

    // The context: the KV at full context must leave the floor.
    let fits = (room.0 - floor.0) / kv_per_token.max(1);
    let model_ctx = model.max_context.unwrap_or(DEFAULT_CONTEXT);
    let context = match req.context {
        Some(c) if c > fits => {
            return Err(PlanningError::ContextTooLarge {
                requested: c,
                fits,
                sequences: req.sequences,
            });
        }
        Some(c) => {
            if c > model_ctx {
                reasons.push(format!(
                    "Warning: the context {c} exceeds the model's limit of {model_ctx}."
                ));
            }
            c
        }
        None if model_ctx > fits => {
            reasons.push(format!(
                "Context shortened from the model's {model_ctx} to {fits}: the full-context KV must leave one expert slot per layer."
            ));
            fits
        }
        None => model_ctx,
    };
    let kv_full = Bytes(kv_per_token.saturating_mul(context));
    reasons.push(format!(
        "KV: {} per token × {context} context{} = {kv_full} at full context. It is allocated as it grows, never up front.",
        Bytes(kv_per_token),
        if req.sequences > 1 {
            format!(" × {} sequences (included per token)", req.sequences)
        } else {
            String::new()
        }
    ));

    // The expert cache at both ends.
    let cache_start = Bytes(room.0.min(experts_total.0));
    let cache_full = Bytes((room.0 - kv_full.0).min(experts_total.0));
    let weights = req.layer_weights.as_deref();
    let at_start = slots(&model.layers, cache_start, weights);
    let at_full = slots(&model.layers, cache_full, weights);
    if model.layers.is_empty() {
        reasons.push("No routed experts: the model is dense.".to_owned());
    } else if cache_full >= experts_total {
        reasons.push(format!(
            "Every expert ({experts_total}) stays cached, even at full context."
        ));
    } else {
        let min_slots = |s: &[LayerSlots]| s.iter().map(|l| l.slots).min().unwrap_or(0);
        reasons.push(format!(
            "Expert cache {cache_start} with an empty KV, shrinking to {cache_full} at full context (at least {} of {} experts per layer); the rest stream from disk. Each layer's pool holds its slots and evicts only its own experts.",
            min_slots(&at_full),
            model.layers.iter().map(|l| l.experts).max().unwrap_or(0)
        ));
    }

    // Placements.
    let mut placements: Vec<Placement> = model
        .dense
        .iter()
        .map(|g| Placement {
            what: g.group.label().to_owned(),
            bytes: Quantity::measured(g.bytes),
            location: Location::Ram,
            reason: "dense: used by every token".to_owned(),
        })
        .collect();
    if !model.layers.is_empty() {
        placements.push(Placement {
            what: "routed experts".to_owned(),
            bytes: Quantity::estimated(cache_start),
            location: Location::RamCache,
            reason: "loaded on first use; the KV takes bytes back as it grows".to_owned(),
        });
        if cache_full < experts_total {
            placements.push(Placement {
                what: "routed experts".to_owned(),
                bytes: Quantity::estimated(Bytes(experts_total.0 - cache_full.0)),
                location: Location::Disk,
                reason: "beyond the cache at full context: read from the model file on a miss"
                    .to_owned(),
            });
        }
    }
    reasons.push(
        "No VRAM budget: Metal shares host memory, so its buffers count against the RAM budget. Discrete GPUs (CUDA, Vulkan) are not supported yet."
            .to_owned(),
    );

    Ok(MemoryPlan {
        profile: req.profile,
        context,
        sequences: req.sequences,
        ram: RamBudget {
            budget,
            dense: Quantity::measured(dense),
            activations: Quantity::estimated(ACTIVATION_RESERVE),
            room: Quantity::estimated(room),
            kv_per_token: Quantity::measured(Bytes(kv_per_token)),
            kv_full_context: Quantity::measured(kv_full),
            experts_total: Quantity::measured(experts_total),
            cache_floor: Quantity::measured(floor),
            expert_cache_start: Quantity::estimated(cache_start),
            expert_cache_full_context: Quantity::estimated(cache_full),
        },
        vram: Vec::new(),
        expert_cache: CacheBudget {
            at_start,
            at_full_context: at_full,
        },
        placements,
        reasons,
    })
}
