//! The boundary for model sources. Provider metadata stops here: what crosses this
//! trait is normalised, so the registry and runtime never see a provider's formats.

use std::io::Read;

use crate::DownloadError;

/// A checksum a provider publishes for a file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Checksum {
    /// SHA-256 of the content, in lower-case hex.
    Sha256(String),
    /// A git blob id: SHA-1 of `blob <size>\0` and the content, in lower-case hex.
    GitBlobSha1(String),
}

/// One file in a repository.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteFile {
    /// Its path in the repository. Untrusted until selected.
    pub path: String,
    /// Its size in bytes.
    pub size: u64,
    /// Its published checksum, when there is one.
    pub checksum: Option<Checksum>,
}

/// A repository at one exact revision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepoListing {
    /// The repository.
    pub repository: String,
    /// The exact, immutable revision.
    pub revision: String,
    /// Its files.
    pub files: Vec<RemoteFile>,
}

/// One search result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchHit {
    /// The repository.
    pub repository: String,
    /// Downloads, as the provider counts them.
    pub downloads: u64,
    /// Likes, as the provider counts them.
    pub likes: u64,
    /// `gguf` and `safetensors`, where the provider says so.
    pub formats: Vec<String>,
    /// Whether access has to be requested.
    pub gated: bool,
}

/// A response body, positioned at [`Opened::start`].
pub struct Opened {
    /// The file offset of the body's first byte. It is 0 for a server that ignored the
    /// requested offset.
    pub start: u64,
    /// The bytes.
    pub body: Box<dyn Read + Send>,
}

impl std::fmt::Debug for Opened {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Opened")
            .field("start", &self.start)
            .finish()
    }
}

/// A source of models.
pub trait ModelProvider: Sync {
    /// The name recorded in the registry, such as `huggingface`.
    fn name(&self) -> &str;

    /// Resolves `revision` (the default branch if `None`) to an exact revision and
    /// lists its files.
    ///
    /// # Errors
    /// [`DownloadError::Transient`] for failures worth retrying; others are final.
    fn resolve(
        &self,
        repository: &str,
        revision: Option<&str>,
    ) -> Result<RepoListing, DownloadError>;

    /// Finds repositories.
    ///
    /// # Errors
    /// As for [`Self::resolve`].
    fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>, DownloadError>;

    /// Opens a file of a listing at `offset`.
    ///
    /// # Errors
    /// As for [`Self::resolve`].
    fn open(&self, listing: &RepoListing, path: &str, offset: u64)
    -> Result<Opened, DownloadError>;
}
