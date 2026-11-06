//! From a repository name to an installed model.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use qubivex_model::{ArchitectureRegistry, ImportOptions, ModelSource, fill_checksums, import};
use qubivex_model_registry::{Install, ModelRecord, Registry};
use serde_json::{Map, Value};

use crate::error::io_err;
use crate::select::{self, INDEX, default_id, needs_index};
use crate::transfer::{Transfer, retry};
use crate::{Checksum, DownloadError, ModelProvider, RepoListing, SearchHit, Selection};

/// Largest shard index read into memory, which is the importer's limit too.
const MAX_INDEX: u64 = 64 << 20;

/// What to pull.
#[derive(Clone, Debug, Default)]
pub struct PullRequest {
    /// The repository.
    pub repository: String,
    /// A branch, tag or commit; the default branch if `None`.
    pub revision: Option<String>,
    /// Exact repository paths; empty means choose.
    pub files: Vec<String>,
    /// The model id; derived from the repository or file if `None`.
    pub id: Option<String>,
    /// The display name.
    pub name: Option<String>,
    /// `overrides.config` values, for what a checkpoint lacks.
    pub set: Map<String, Value>,
}

/// How to pull.
#[derive(Clone, Debug)]
pub struct PullOptions {
    /// Files fetched at once.
    pub concurrency: usize,
    /// Attempts without progress before giving up. Progress resets the count.
    pub attempts: u32,
    /// The first retry's delay; it doubles each time.
    pub backoff: Duration,
    /// The longest delay, including one a server asks for.
    pub max_backoff: Duration,
}

impl Default for PullOptions {
    fn default() -> Self {
        Self {
            concurrency: 4,
            attempts: 8,
            backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(60),
        }
    }
}

/// What a pull is doing.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Progress {
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
        /// What happened.
        reason: String,
    },
    /// Every file arrived; the model is being imported and validated.
    Validating,
}

/// A finished pull.
#[derive(Clone, Debug)]
pub struct Pulled {
    /// The installed model's record.
    pub record: ModelRecord,
    /// Things the user should know, from the importer.
    pub warnings: Vec<String>,
    /// Its files, by local name.
    pub files: Vec<String>,
}

fn read_index(
    provider: &dyn ModelProvider,
    listing: &RepoListing,
) -> Result<Vec<u8>, DownloadError> {
    let mut bytes = Vec::new();
    provider
        .open(listing, INDEX, 0)?
        .body
        .take(MAX_INDEX)
        .read_to_end(&mut bytes)
        .map_err(|_| DownloadError::transient(format!("{INDEX}: the reply was cut short")))?;
    Ok(bytes)
}

/// Removes staged files this plan does not include: an earlier attempt from the same
/// source may have chosen another file, and the importer reads the whole directory.
fn prune(dir: &Path, files: &[Selection]) -> Result<u64, DownloadError> {
    let mut staged = 0;
    for entry in fs::read_dir(dir).map_err(io_err(dir))? {
        let entry = entry.map_err(io_err(dir))?;
        let path = entry.path();
        let meta = fs::symlink_metadata(&path).map_err(io_err(&path))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let stem = name.strip_suffix(".part").unwrap_or(&name);
        let wanted = files.iter().find(|s| s.name == stem);
        match wanted {
            Some(s) if meta.is_file() => staged += meta.len().min(s.remote.size),
            _ if meta.is_dir() => fs::remove_dir_all(&path).map_err(io_err(&path))?,
            _ => fs::remove_file(&path).map_err(io_err(&path))?,
        }
    }
    Ok(staged)
}

#[cfg(unix)]
fn free_bytes(dir: &Path) -> Result<u64, DownloadError> {
    let s = rustix::fs::statvfs(dir).map_err(|e| io_err(dir)(e.into()))?;
    Ok(s.f_bavail.saturating_mul(s.f_frsize))
}

/// Not measured where `statvfs` does not exist; a full disk then fails while writing.
#[cfg(not(unix))]
#[allow(clippy::unnecessary_wraps)]
fn free_bytes(_dir: &Path) -> Result<u64, DownloadError> {
    Ok(u64::MAX)
}

/// Fetches, imports and writes the manifest. `Err` leaves the install to the caller.
fn fill(
    install: &Install<'_>,
    req: &PullRequest,
    id: &str,
    source: &ModelSource,
    arch: &ArchitectureRegistry,
    transfer: &Transfer<'_>,
    files: &[Selection],
) -> Result<Vec<String>, DownloadError> {
    let dir = install.staging();
    let staged = prune(dir, files)?;
    let total: u64 = files.iter().map(|f| f.remote.size).sum();
    let needed = total.saturating_sub(staged);
    let available = free_bytes(dir)?;
    if needed > available {
        return Err(DownloadError::NoSpace {
            path: dir.to_owned(),
            needed,
            available,
        });
    }
    transfer.run(files)?;
    (transfer.progress)(Progress::Validating);
    let opts = ImportOptions {
        id: Some(id.to_owned()),
        name: req.name.clone(),
        set: req.set.clone(),
        source: Some(source.clone()),
    };
    let mut imported = import(dir, &opts, arch).map_err(DownloadError::Import)?;
    // The transfer checked these against the provider, so large files are not read again.
    let verified: BTreeMap<String, String> = files
        .iter()
        .filter_map(|f| match &f.remote.checksum {
            Some(Checksum::Sha256(h)) => Some((f.name.clone(), h.clone())),
            _ => None,
        })
        .collect();
    fill_checksums(&mut imported.manifest, dir, &verified).map_err(DownloadError::Import)?;
    imported.manifest.save(dir).map_err(DownloadError::Import)?;
    Ok(imported.warnings)
}

/// An error with its causes, for the registry record.
fn chain(e: &DownloadError) -> String {
    let mut out = e.to_string();
    let mut cause = std::error::Error::source(e);
    while let Some(c) = cause {
        out.push_str(": ");
        out.push_str(&c.to_string());
        cause = c.source();
    }
    out
}

/// Downloads and installs a model into `registry`.
///
/// The revision is pinned first, so a branch moving during a pull cannot mix commits.
/// An interrupted pull of the same revision continues; `cancel` stops one, and so does
/// a crash. A failure is recorded in the registry and keeps its files, so a pull that
/// only lacked `--set` values does not download again.
///
/// # Errors
/// Anything in [`DownloadError`]; [`DownloadError::Cancelled`] after `cancel` is set.
pub fn pull(
    registry: &Registry,
    provider: &dyn ModelProvider,
    req: &PullRequest,
    opts: &PullOptions,
    arch: &ArchitectureRegistry,
    cancel: &AtomicBool,
    progress: &(dyn Fn(Progress) + Sync),
) -> Result<Pulled, DownloadError> {
    let cancelled = || cancel.load(Ordering::Relaxed);
    let what = format!("listing {}", req.repository);
    let listing = retry(&what, opts, &cancelled, || {
        provider.resolve(&req.repository, req.revision.as_deref())
    })?;
    let index = if needs_index(&listing, &req.files) {
        Some(retry(&what, opts, &cancelled, || {
            read_index(provider, &listing)
        })?)
    } else {
        None
    };
    let files = select::select(&listing, &req.files, index.as_deref())?;
    let id = req
        .id
        .clone()
        .unwrap_or_else(|| default_id(&req.repository, &files));
    registry.reconcile(arch)?;
    let source = ModelSource {
        provider: provider.name().to_owned(),
        repository: listing.repository.clone(),
        revision: listing.revision.clone(),
    };
    let install = registry.begin_install(&id, Some(source.clone()))?;
    progress(Progress::Planned {
        revision: listing.revision.clone(),
        files: files
            .iter()
            .map(|f| (f.name.clone(), f.remote.size))
            .collect(),
    });
    let transfer = Transfer {
        provider,
        listing: &listing,
        dir: install.staging(),
        opts,
        cancel,
        progress,
        failed: AtomicBool::new(false),
    };
    match fill(&install, req, &id, &source, arch, &transfer, &files) {
        Ok(warnings) => Ok(Pulled {
            record: install.commit(arch)?,
            warnings,
            files: files.into_iter().map(|f| f.name).collect(),
        }),
        Err(e) => {
            install.fail(chain(&e))?;
            Err(e)
        }
    }
}

/// Finds repositories, with retries.
///
/// # Errors
/// As for [`ModelProvider::search`], or [`DownloadError::Exhausted`].
pub fn search(
    provider: &dyn ModelProvider,
    query: &str,
    limit: usize,
    opts: &PullOptions,
    cancel: &AtomicBool,
) -> Result<Vec<SearchHit>, DownloadError> {
    let cancelled = || cancel.load(Ordering::Relaxed);
    retry("search", opts, &cancelled, || provider.search(query, limit))
}
