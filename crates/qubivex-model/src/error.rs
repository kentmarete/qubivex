use std::path::PathBuf;

use qubivex_core::CoreError;
use thiserror::Error;

/// Errors from reading and validating model files.
///
/// Model files are untrusted input (RUST_GUIDELINES §2): every problem is an error,
/// never a panic, and messages name the file and, where there is one, the tensor.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ModelError {
    /// A file could not be opened or read.
    #[error("cannot read {}", path.display())]
    Io {
        /// The file.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },

    /// A container header is malformed.
    #[error("{}: invalid {format} header: {detail}", path.display())]
    Header {
        /// The file.
        path: PathBuf,
        /// `safetensors` or `gguf`.
        format: &'static str,
        /// What is wrong.
        detail: String,
    },

    /// One tensor's entry is malformed or inconsistent with the file.
    #[error("{}: tensor {name:?}: {detail}", path.display())]
    Tensor {
        /// The file.
        path: PathBuf,
        /// The tensor.
        name: String,
        /// What is wrong.
        detail: String,
    },

    /// Two files (or two entries) define the same tensor.
    #[error("tensor {name:?} is defined in both {} and {}", first.display(), second.display())]
    DuplicateTensor {
        /// The tensor.
        name: String,
        /// Where it was first seen.
        first: PathBuf,
        /// Where it was seen again.
        second: PathBuf,
    },

    /// A size or shape computation failed.
    #[error(transparent)]
    Core(#[from] CoreError),
}
