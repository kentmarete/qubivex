//! Where the admin download routes get models from (ADR-013 amendment 8).
//!
//! The gateway never talks to a model hub itself: it has no network or TLS dependency
//! (ADR-014), so an application that embeds it stays free of one. A [`ModelSource`]
//! supplied in [`ServerConfig::model_source`](crate::ServerConfig::model_source) does the
//! searching and downloading; `qubivex serve` passes one over Hugging Face. Without one,
//! the download routes answer 404 and `/v1/system` says `model_downloads: false`.

use std::fmt;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use serde_json::{Map, Value};

/// Searches and downloads models for the admin routes. Calls block and run on their own
/// threads, never on the server's async workers.
pub trait ModelSource: Send + Sync + fmt::Debug {
    /// Up to `limit` repositories matching `query`, most relevant first.
    ///
    /// # Errors
    /// [`SourceError`] with a message that holds no path, URL or credential.
    fn search(&self, query: &str, limit: usize) -> Result<Vec<SourceHit>, SourceError>;

    /// Downloads and installs a model. Stops with [`SourceErrorKind::Cancelled`] soon
    /// after `cancel` is set, keeping what arrived so a later pull continues.
    ///
    /// # Errors
    /// [`SourceError`] with a message that holds no path, URL or credential.
    fn pull(
        &self,
        spec: &PullSpec,
        cancel: &AtomicBool,
        progress: &(dyn Fn(PullProgress) + Sync),
    ) -> Result<PullOutcome, SourceError>;

    /// Installed models that came from a repository, as `(repository, id)`, so search
    /// results can say what is already here.
    fn installed(&self) -> Vec<(String, String)> {
        Vec::new()
    }
}

/// One search result.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SourceHit {
    /// `owner/name`.
    pub repository: String,
    /// Downloads, as the source counts them.
    pub downloads: u64,
    /// Likes, as the source counts them.
    pub likes: u64,
    /// Weight formats found, such as `safetensors` or `gguf`.
    pub formats: Vec<String>,
    /// The repository needs an access token.
    pub gated: bool,
}

/// What to download.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PullSpec {
    /// `owner/name`.
    pub repository: String,
    /// A branch, tag or commit; `None` for the default branch.
    pub revision: Option<String>,
    /// Files to fetch by repository path; empty lets the source choose.
    pub files: Vec<String>,
    /// The id to install as; `None` lets the source derive one.
    pub id: Option<String>,
    /// A display name.
    pub name: Option<String>,
    /// Config values the checkpoint does not record, by `config.json` name.
    pub set: Map<String, Value>,
}

/// What a pull is doing.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PullProgress {
    /// The revision and files are settled.
    Planned {
        /// The exact revision.
        revision: String,
        /// The files, by local name, with sizes.
        files: Vec<(String, u64)>,
    },
    /// A file reached `done` of `total` bytes. `done` can go down after a restart.
    File {
        /// Its local name.
        file: String,
        /// Bytes so far, including any from an earlier run.
        done: u64,
        /// Its size.
        total: u64,
    },
    /// A transfer failed and will be retried.
    Retrying {
        /// Its local name.
        file: String,
        /// The attempt that failed.
        attempt: u32,
        /// The wait before the next.
        delay: Duration,
        /// What happened, with no path, URL or credential.
        reason: String,
    },
    /// Every file arrived; the model is being imported and checked.
    Validating,
}

/// A finished pull.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PullOutcome {
    /// The installed model's id, which `POST /v1/models/load` takes.
    pub id: String,
    /// Things the operator should know.
    pub warnings: Vec<String>,
    /// Its files, by local name.
    pub files: Vec<String>,
}

/// Why a search or pull failed. The message is shown to clients, so it never holds a
/// path, URL or credential.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceError {
    /// What kind of failure, which picks the HTTP status.
    pub kind: SourceErrorKind,
    /// What happened.
    pub message: String,
}

impl SourceError {
    /// A failure of `kind`.
    pub fn new(kind: SourceErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SourceError {}

/// The kinds of [`SourceError`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SourceErrorKind {
    /// The request cannot work as given: a bad repository id, an ambiguous choice of
    /// files, weights this build cannot run.
    Invalid,
    /// The repository, revision or file does not exist.
    NotFound,
    /// Another install of the same model is running, or it is already installed.
    Conflict,
    /// The source refused access: a gated or private repository.
    Unauthorized,
    /// The source failed or could not be reached.
    Upstream,
    /// The pull stopped because it was cancelled.
    Cancelled,
    /// Something failed on this machine: the disk, the registry.
    Local,
}
