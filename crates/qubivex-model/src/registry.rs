//! The architecture registry (ADR-003): architecture id → factory.

use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use qubivex_core::{ExpertIdx, LayerIdx};
use serde::Serialize;

use crate::manifest::{ArchRef, Manifest, SemVer};
use crate::{GgufMetadata, ModelError};

/// One part of an expert's SwiGLU FFN.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertPart {
    /// Gate projection.
    Gate,
    /// Up projection.
    Up,
    /// Gate and up stacked in one tensor (gate rows first).
    GateUp,
    /// Down projection.
    Down,
}

/// One part of a dense (always active) SwiGLU FFN.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FfnPart {
    /// Gate projection, `[ffn, hidden]`.
    Gate,
    /// Up projection, `[ffn, hidden]`.
    Up,
    /// Down projection, `[hidden, ffn]`.
    Down,
}

/// Attention tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionPart {
    /// Query projection.
    Q,
    /// Key projection.
    K,
    /// Value projection.
    V,
    /// Output projection.
    O,
    /// Query norm.
    QNorm,
    /// Key norm.
    KNorm,
}

/// What a tensor is for. Validation and the memory planner work on roles only,
/// never on names or model families (ADR-003).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case", tag = "role")]
pub enum TensorRole {
    /// Token embedding.
    Embedding,
    /// Output (LM) head.
    OutputHead,
    /// Final norm before the head.
    FinalNorm,
    /// A per-layer norm (pre-attention or pre-FFN).
    Norm {
        /// Layer.
        layer: LayerIdx,
    },
    /// An attention tensor.
    Attention {
        /// Layer.
        layer: LayerIdx,
        /// Which.
        part: AttentionPart,
    },
    /// A dense FFN tensor (resident, used by every token).
    Ffn {
        /// Layer.
        layer: LayerIdx,
        /// Which part.
        part: FfnPart,
    },
    /// The MoE router (gate) weights.
    Router {
        /// Layer.
        layer: LayerIdx,
    },
    /// One expert's tensor.
    RoutedExpert {
        /// Layer.
        layer: LayerIdx,
        /// Expert.
        expert: ExpertIdx,
        /// Which part.
        part: ExpertPart,
    },
    /// All experts of a layer stacked along the outermost dimension; expert `e` is the
    /// `e`-th contiguous slice of `bytes / experts`.
    RoutedExpertStack {
        /// Layer.
        layer: LayerIdx,
        /// Which part.
        part: ExpertPart,
        /// Number of experts stacked.
        experts: u32,
    },
}

/// One expected tensor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TensorSpec {
    /// Name in the weight files.
    pub name: String,
    /// What it is for.
    #[serde(flatten)]
    pub role: TensorRole,
    /// Expected row-major shape.
    pub shape: Vec<u64>,
}

/// Every tensor an architecture expects, for a given config.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct TensorSchema {
    /// The tensors, in a stable order.
    pub tensors: Vec<TensorSpec>,
}

/// Dimensions common to every architecture, for summaries and planning.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ModelDims {
    /// Transformer layers.
    pub layers: u32,
    /// Hidden size.
    pub hidden: u64,
    /// Vocabulary size.
    pub vocab: u64,
    /// Routed experts per layer (0 for a dense model).
    pub experts: u32,
    /// Experts used per token.
    pub experts_per_token: u32,
    /// Maximum context length the architecture supports, if stated.
    pub max_context: Option<u64>,
}

/// An architecture's parsed config: the common dims plus its own typed part.
#[derive(Clone)]
pub struct ArchConfig {
    /// Common dimensions.
    pub dims: ModelDims,
    /// The architecture's own config; its factory downcasts it.
    pub specific: Arc<dyn Any + Send + Sync>,
    /// A JSON view of `specific`, for `qubivex models inspect`.
    pub describe: serde_json::Value,
}

impl fmt::Debug for ArchConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArchConfig")
            .field("dims", &self.dims)
            .field("describe", &self.describe)
            .finish_non_exhaustive()
    }
}

/// Where a factory reads its config from.
#[derive(Clone, Copy, Debug)]
pub struct ConfigSource<'a> {
    /// The manifest (`architecture.config`, `overrides.config`).
    pub manifest: &'a Manifest,
    /// `config.json`, for SafeTensors models.
    pub hf_config: Option<&'a serde_json::Value>,
    /// GGUF metadata, for GGUF models.
    pub gguf: Option<&'a GgufMetadata>,
}

/// An architecture: parses its config and describes the tensors it needs.
///
/// `build` (constructing an executable model) arrives with the CPU prototype (Phase 4).
pub trait ArchitectureFactory: Send + Sync {
    /// Registry id, for example `transformer_moe`.
    fn id(&self) -> &'static str;
    /// The newest version implemented. Manifests with the same major and a lower or
    /// equal minor are accepted.
    fn version(&self) -> SemVer;
    /// One-line description.
    fn description(&self) -> &'static str;
    /// Parses and checks the architecture config.
    ///
    /// # Errors
    /// [`ModelError::Config`] for missing, malformed or inconsistent values.
    fn parse_config(&self, src: &ConfigSource<'_>) -> Result<ArchConfig, ModelError>;
    /// The tensors this config requires.
    ///
    /// # Errors
    /// [`ModelError::Config`] if `cfg` was not produced by this factory.
    fn tensor_schema(&self, cfg: &ArchConfig) -> Result<TensorSchema, ModelError>;
}

impl fmt::Debug for dyn ArchitectureFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ArchitectureFactory({} {})", self.id(), self.version())
    }
}

/// A registered architecture, for listings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ArchDescriptor {
    /// Id.
    pub id: &'static str,
    /// Newest version implemented.
    pub version: String,
    /// Description.
    pub description: &'static str,
}

/// Architecture id → factory. Built at startup, then shared read-only.
#[derive(Default)]
pub struct ArchitectureRegistry {
    factories: BTreeMap<&'static str, Arc<dyn ArchitectureFactory>>,
}

impl fmt::Debug for ArchitectureRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.factories.keys()).finish()
    }
}

impl ArchitectureRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A registry with every built-in architecture.
    #[must_use]
    pub fn with_builtins() -> Self {
        let mut r = Self::new();
        // Built-in ids are distinct by construction.
        let _ = r.register(Arc::new(crate::arch::TransformerMoe));
        let _ = r.register(Arc::new(crate::arch::TransformerDense));
        r
    }

    /// Adds a factory.
    ///
    /// # Errors
    /// [`ModelError::Registry`] if the id is already registered.
    pub fn register(&mut self, f: Arc<dyn ArchitectureFactory>) -> Result<(), ModelError> {
        let id = f.id();
        if self.factories.contains_key(id) {
            return Err(ModelError::Registry(format!(
                "architecture {id:?} is already registered"
            )));
        }
        self.factories.insert(id, f);
        Ok(())
    }

    /// Removes a factory (tests only; a serving registry is frozen).
    pub fn unregister(&mut self, id: &str) -> Option<Arc<dyn ArchitectureFactory>> {
        self.factories.remove(id)
    }

    /// The factory for a manifest's architecture, checking the version.
    ///
    /// # Errors
    /// [`ModelError::Registry`] for an unknown id or an incompatible version.
    pub fn resolve(&self, r: &ArchRef) -> Result<Arc<dyn ArchitectureFactory>, ModelError> {
        let f = self.factories.get(r.id.as_str()).ok_or_else(|| {
            let known: Vec<&str> = self.factories.keys().copied().collect();
            ModelError::Registry(format!(
                "unknown architecture {:?} (registered: {})",
                r.id,
                known.join(", ")
            ))
        })?;
        let have = f.version();
        if r.version.major != have.major || r.version.minor > have.minor {
            return Err(ModelError::Registry(format!(
                "manifest wants {} {}, this build implements {have}",
                r.id, r.version
            )));
        }
        Ok(Arc::clone(f))
    }

    /// Every registered architecture.
    #[must_use]
    pub fn list_supported(&self) -> Vec<ArchDescriptor> {
        self.factories
            .values()
            .map(|f| ArchDescriptor {
                id: f.id(),
                version: f.version().to_string(),
                description: f.description(),
            })
            .collect()
    }
}
