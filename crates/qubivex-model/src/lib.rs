//! Model files, metadata and the architecture registry (QUBIVEX_DESIGN §4.3).
//!
//! - [`Manifest`] (schema v1, ADR-004) names the files and the architecture.
//! - [`ArchitectureRegistry`] maps an architecture id to an [`ArchitectureFactory`],
//!   which parses its config and describes every tensor it needs as a
//!   [`TensorSchema`] of roles and shapes (ADR-003). [`arch::TransformerMoe`] is
//!   the first architecture.
//! - [`ModelMetadata::load`] cross-checks manifest, config and tensor headers;
//!   [`discover`] and [`find`] locate model directories; [`import`] writes a
//!   manifest for a checkpoint by finding the layout that validates.
//!
//! Weight containers:
//!
//! - [`safetensors`] and [`gguf`] parse headers as untrusted input. Every count,
//!   string, offset and size is bounded and checked against the real file length,
//!   and each tensor's byte length must equal what its dtype and shape require
//!   (Colibri's untrusted-container rule).
//! - [`TensorIndex`] collects the tensors of all of a model's files by name.
//! - [`mod@write`] builds small SafeTensors and GGUF files for tests and fixtures.
//!
//! No code here knows a model family by name: GGUF keys specific to an architecture
//! are looked up through `general.architecture` ([`GgufMetadata::arch_value`]).

pub mod arch;
mod discover;
mod error;
pub mod gguf;
mod import;
mod index;
pub mod manifest;
mod metadata;
mod registry;
pub mod safetensors;
pub mod write;

pub use discover::{Discovered, discover, find};
pub use error::ModelError;
pub use gguf::{GgufHeader, GgufMetadata, GgufValue};
pub use import::{ImportOptions, Imported, import};
pub use index::{RawTensor, TensorEntry, TensorIndex, WeightFile, WeightFormat};
pub use manifest::Manifest;
pub use metadata::{ByteSummary, MANIFEST_FILE, ModelMetadata};
pub use registry::{
    ArchConfig, ArchDescriptor, ArchitectureFactory, ArchitectureRegistry, AttentionPart,
    ConfigSource, ExpertPart, ModelDims, TensorRole, TensorSchema, TensorSpec,
};
pub use safetensors::SafetensorsHeader;
