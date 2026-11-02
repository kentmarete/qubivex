//! Model files, metadata and the architecture registry (QUBIVEX_DESIGN §4.3).
//!
//! This first part reads weight containers:
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

mod error;
pub mod gguf;
mod index;
pub mod safetensors;
pub mod write;

pub use error::ModelError;
pub use gguf::{GgufHeader, GgufMetadata, GgufValue};
pub use index::{RawTensor, TensorEntry, TensorIndex, WeightFile, WeightFormat};
pub use safetensors::SafetensorsHeader;
