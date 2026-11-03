//! What a model needs in memory, derived from tensor roles and byte sizes only.
//!
//! No architecture or family knowledge (design invariant 3): dense bytes are the sum
//! of every non-expert tensor, expert bytes come from the routed-expert roles, and the
//! KV cost per token is the output width of each layer's K and V projections.

use std::collections::BTreeMap;

use qubivex_core::{Bytes, LayerIdx};
use qubivex_model::{AttentionPart, ModelMetadata, TensorRole};
use serde::Serialize;

use crate::PlanningError;

/// Bytes per cached key or value element: the KV cache holds f32 (lossless).
pub const KV_ELEMENT_BYTES: u64 = 4;

/// A group of resident (non-expert) tensors, for placement and reasons.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RoleGroup {
    /// Token embedding.
    Embedding,
    /// Output head.
    OutputHead,
    /// Per-layer and final norms.
    Norms,
    /// Attention projections and their norms.
    Attention,
    /// MoE routers.
    Router,
}

impl RoleGroup {
    /// A short human-readable name.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::OutputHead => "output head",
            Self::Norms => "norms",
            Self::Attention => "attention",
            Self::Router => "routers",
        }
    }
}

/// The resident tensors of one group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct DenseGroup {
    /// Which group.
    pub group: RoleGroup,
    /// Total bytes.
    pub bytes: Bytes,
    /// Number of tensors.
    pub tensors: u32,
}

/// The routed experts of one layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct LayerExperts {
    /// The layer.
    pub layer: LayerIdx,
    /// Routed experts in the layer.
    pub experts: u32,
    /// Bytes of the largest expert (all parts): one cache slot.
    pub expert_bytes: Bytes,
    /// Bytes of every expert of the layer.
    pub total: Bytes,
}

/// What a model needs in memory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ModelFootprint {
    /// Resident tensors by group (no empty groups), in group order.
    pub dense: Vec<DenseGroup>,
    /// Layers with routed experts, in layer order.
    pub layers: Vec<LayerExperts>,
    /// KV cache bytes per token of one sequence, all layers.
    pub kv_bytes_per_token: Bytes,
    /// The model's context limit, if it states one.
    pub max_context: Option<u64>,
}

impl ModelFootprint {
    /// Bytes of every resident tensor.
    #[must_use]
    pub fn dense_bytes(&self) -> Bytes {
        Bytes(self.dense.iter().map(|g| g.bytes.0).sum())
    }

    /// Bytes of every routed expert.
    #[must_use]
    pub fn expert_bytes(&self) -> Bytes {
        Bytes(self.layers.iter().map(|l| l.total.0).sum())
    }

    /// The smallest expert cache that can run the model: one slot per layer.
    #[must_use]
    pub fn cache_floor(&self) -> Bytes {
        Bytes(self.layers.iter().map(|l| l.expert_bytes.0).sum())
    }

    /// The footprint of a validated model.
    ///
    /// # Errors
    /// [`PlanningError::Footprint`] if a role names a tensor the index lacks, or a
    /// stacked expert tensor does not split evenly.
    pub fn from_metadata(meta: &ModelMetadata) -> Result<Self, PlanningError> {
        let bad = |m: String| PlanningError::Footprint(m);
        let mut dense: BTreeMap<RoleGroup, DenseGroup> = BTreeMap::new();
        // (layer, expert) → bytes of its parts; stacks contribute one share per expert.
        let mut experts: BTreeMap<(LayerIdx, u32), u64> = BTreeMap::new();
        let mut kv = 0u64;
        for (name, role) in &meta.roles {
            let t = meta
                .tensors
                .get(name)
                .ok_or_else(|| bad(format!("tensor {name:?} has a role but is not indexed")))?;
            let bytes = t.tensor.bytes;
            let group = match *role {
                TensorRole::Embedding => RoleGroup::Embedding,
                TensorRole::OutputHead => RoleGroup::OutputHead,
                TensorRole::FinalNorm | TensorRole::Norm { .. } => RoleGroup::Norms,
                TensorRole::Attention { part, .. } => {
                    if matches!(part, AttentionPart::K | AttentionPart::V) {
                        let out = t.tensor.shape.dims().first().copied().unwrap_or(0);
                        kv += out * KV_ELEMENT_BYTES;
                    }
                    RoleGroup::Attention
                }
                TensorRole::Router { .. } => RoleGroup::Router,
                TensorRole::RoutedExpert { layer, expert, .. } => {
                    *experts.entry((layer, expert.0)).or_default() += bytes;
                    continue;
                }
                TensorRole::RoutedExpertStack {
                    layer, experts: n, ..
                } => {
                    if n == 0 || bytes % u64::from(n) != 0 {
                        return Err(bad(format!(
                            "stacked expert tensor {name:?} ({bytes} bytes) does not split into {n} experts"
                        )));
                    }
                    for e in 0..n {
                        *experts.entry((layer, e)).or_default() += bytes / u64::from(n);
                    }
                    continue;
                }
            };
            let g = dense.entry(group).or_insert(DenseGroup {
                group,
                bytes: Bytes(0),
                tensors: 0,
            });
            g.bytes.0 += bytes;
            g.tensors += 1;
        }
        let mut layers: Vec<LayerExperts> = Vec::new();
        for ((layer, _), bytes) in experts {
            match layers.last_mut() {
                Some(l) if l.layer == layer => {
                    l.experts += 1;
                    l.expert_bytes = l.expert_bytes.max(Bytes(bytes));
                    l.total.0 += bytes;
                }
                _ => layers.push(LayerExperts {
                    layer,
                    experts: 1,
                    expert_bytes: Bytes(bytes),
                    total: Bytes(bytes),
                }),
            }
        }
        Ok(Self {
            dense: dense.into_values().collect(),
            layers,
            kv_bytes_per_token: Bytes(kv),
            max_context: meta
                .manifest
                .context_length
                .or(meta.config.dims.max_context),
        })
    }
}
