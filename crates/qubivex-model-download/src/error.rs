//! Why a download failed.

use std::io;
use std::path::PathBuf;
use std::time::Duration;

use qubivex_model::ModelError;
use qubivex_model_registry::RegistryError;
use thiserror::Error;

/// Why a download failed. Messages never contain a token or a redirect URL.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DownloadError {
    /// A local file operation failed.
    #[error("{path}: {source}")]
    Io {
        /// The path involved.
        path: PathBuf,
        /// The cause.
        source: io::Error,
    },
    /// The registry refused or failed.
    #[error(transparent)]
    Registry(#[from] RegistryError),
    /// The files arrived but do not import as a model. They stay in staging.
    #[error("the downloaded files do not import as a model")]
    Import(#[source] ModelError),
    /// The repository id is not valid at this provider.
    #[error("{0:?} is not a repository id: use <owner>/<name>")]
    InvalidRepository(String),
    /// The endpoint cannot be used safely.
    #[error("{0}")]
    InvalidEndpoint(String),
    /// The credential cannot be sent in a header.
    #[error("the access token holds characters a header cannot carry")]
    InvalidToken,
    /// The repository, revision or file does not exist.
    #[error("{0} was not found")]
    NotFound(String),
    /// The server refused access.
    #[error("access to {what} was refused (HTTP {status}): {hint}")]
    Unauthorized {
        /// What was asked for.
        what: String,
        /// The HTTP status.
        status: u16,
        /// What to do about it.
        hint: &'static str,
    },
    /// A failure that may pass: a dropped connection, a timeout, or a server error.
    #[error("{reason}")]
    Transient {
        /// What happened.
        reason: String,
        /// How long the server asked us to wait.
        retry_after: Option<Duration>,
    },
    /// Retries ran out.
    #[error("{what} failed after {attempts} attempts: {last}")]
    Exhausted {
        /// What was being done.
        what: String,
        /// How many attempts ran.
        attempts: u32,
        /// The last failure.
        last: String,
    },
    /// A file does not match its published checksum, twice in a row.
    #[error("{file} does not match its published checksum: expected {expected}, got {actual}")]
    Checksum {
        /// The file.
        file: String,
        /// The published value.
        expected: String,
        /// The value of the bytes that arrived.
        actual: String,
    },
    /// The server sent more than the listing promised.
    #[error("{file} is longer than the {size} bytes its listing gives")]
    TooLarge {
        /// The file.
        file: String,
        /// The listed size.
        size: u64,
    },
    /// The server's reply breaks the protocol or a safety rule.
    #[error("{0}")]
    Protocol(String),
    /// A repository file name cannot be written safely.
    #[error("refusing the repository file name {0:?}")]
    UnsafePath(String),
    /// The repository has no weights this build can use.
    #[error("{0} has no .safetensors or .gguf weights")]
    NoWeights(String),
    /// More than one choice of weights; the caller has to pick.
    #[error("{repository} has several weight files; pick one with --file: {candidates}")]
    Ambiguous {
        /// The repository.
        repository: String,
        /// The choices, comma-separated.
        candidates: String,
    },
    /// A file asked for by name is not in the repository.
    #[error("{repository} has no file {file:?}")]
    FileNotFound {
        /// The repository.
        repository: String,
        /// The file.
        file: String,
    },
    /// The disk cannot hold the download.
    #[error("{path}: needs {needed} more bytes, {available} are free")]
    NoSpace {
        /// The directory.
        path: PathBuf,
        /// Bytes still to arrive.
        needed: u64,
        /// Bytes free.
        available: u64,
    },
    /// The caller cancelled. Files stay in staging, so the same pull continues them.
    #[error("cancelled; run the same command to continue")]
    Cancelled,
}

impl DownloadError {
    pub(crate) fn transient(reason: impl Into<String>) -> Self {
        Self::Transient {
            reason: reason.into(),
            retry_after: None,
        }
    }
}

pub(crate) fn io_err(path: &std::path::Path) -> impl FnOnce(io::Error) -> DownloadError + '_ {
    move |source| DownloadError::Io {
        path: path.to_owned(),
        source,
    }
}
