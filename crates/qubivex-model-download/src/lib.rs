//! Model downloads: resumable, verified and atomic installs from model repositories.
//!
//! [`pull`] resolves a repository to an exact revision, chooses the files a model needs,
//! fetches them into the registry's staging directory, imports them, and lets
//! [`qubivex_model_registry`] validate the result and move it into place. A model is
//! complete in the models directory or absent from it.
//!
//! [`ModelProvider`] is the boundary for model sources; [`HuggingFace`] is the first.
//! Nothing here depends on an async runtime, and transfers run on plain threads, so an
//! application can embed this crate with no server and no CLI.
//!
//! # Failure and resumption
//!
//! Each file arrives as `<name>.part` and takes its final name only after its size and
//! published checksum match. A request has a fixed time budget. When it runs out, or a
//! connection drops, the next request continues from the current offset, so stalls,
//! drops and slow links take one path. Files from an interrupted pull stay in staging,
//! and the same command continues them while the revision is unchanged.
//!
//! # Trust
//!
//! Everything a server returns is untrusted. See ADR-014 for the threat review: tokens
//! stay on the configured origin, file names cannot leave the staging directory, and
//! bodies cannot exceed the listed size. Errors never carry redirect URLs, which are
//! signed, or tokens.

mod error;
mod http;
mod huggingface;
mod provider;
mod pull;
mod select;
mod transfer;

pub use error::DownloadError;
pub use huggingface::{DEFAULT_BUDGET, DEFAULT_ENDPOINT, HuggingFace, Token};
pub use provider::{Checksum, ModelProvider, Opened, RemoteFile, RepoListing, SearchHit};
pub use pull::{Progress, PullOptions, PullRequest, Pulled, pull, search};
pub use select::{Selection, select};
