//! The local model registry: which models are installed under a models directory, where
//! they came from, and how far each install has got.
//!
//! The registry is `registry.json` in the models directory. Models stay plain
//! directories with a `manifest.json`, so `qubivex models list` and the runtime need no
//! registry, and a directory copied in by hand is adopted by [`Registry::reconcile`].
//!
//! # Crash safety
//!
//! An install builds a model in `.staging/<id>`, validates it there, then renames it to
//! `<id>`, so a model directory is complete or absent. Every change to `registry.json`
//! is a write to a temporary file, an fsync and a rename, under an advisory lock on
//! `.registry.lock`. A crash at any step leaves a state [`Registry::reconcile`] or the
//! next [`Registry::begin_install`] can continue from. The locks are `flock`s, which the
//! kernel releases when a process dies, so no crash leaves a stale lock.
//!
//! # Removal safety
//!
//! [`Registry::remove`] deletes only what a record names, and a record's directory must
//! be one normal path component under the models directory. This holds for a tampered
//! `registry.json` too: it fails to load. The directory is first renamed into `.trash`,
//! so a crash cannot leave a half-deleted model that still looks installed.
//!
//! Loaded and unloaded state is not stored here: a file cannot say whether a process is
//! still alive, so it belongs to the process serving the model.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub use qubivex_model::ModelSource;
use qubivex_model::{ArchitectureRegistry, ModelMetadata, discover};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The registry file, in the models directory.
pub const REGISTRY_FILE: &str = "registry.json";
/// The schema this build reads and writes.
pub const SCHEMA_VERSION: u32 = 1;

const LOCK_FILE: &str = ".registry.lock";
const STAGING_DIR: &str = ".staging";
const TRASH_DIR: &str = ".trash";

/// Why a registry operation failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RegistryError {
    /// A file operation failed.
    #[error("{path}: {source}")]
    Io {
        /// The path involved.
        path: PathBuf,
        /// The cause.
        source: io::Error,
    },
    /// `registry.json` cannot be trusted. It is left as it is.
    #[error("{path}: {message}")]
    Corrupt {
        /// The registry file.
        path: PathBuf,
        /// What is wrong.
        message: String,
    },
    /// The id cannot name a directory.
    #[error(
        "{0:?} cannot be a model id: use letters, digits, '.', '_' and '-', not starting with '.'"
    )]
    InvalidId(String),
    /// The model is already installed.
    #[error("the model {0:?} is already installed")]
    AlreadyInstalled(String),
    /// Another install of this model holds its lock.
    #[error("the model {0:?} is being installed by another process")]
    InstallInProgress(String),
    /// The registry has no such model.
    #[error("no model {0:?} in the registry")]
    NotFound(String),
    /// The model failed validation; the record says why too.
    #[error("the model {id:?} failed validation: {reason}")]
    Validation {
        /// The model.
        id: String,
        /// What is wrong with it.
        reason: String,
    },
    /// Removing this would touch something the registry does not own.
    #[error("{0}")]
    Refused(String),
}

/// How far an install has got.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum InstallState {
    /// Files are arriving in the staging directory.
    Downloading,
    /// Every file arrived; the model is being validated.
    Validating,
    /// In place and valid.
    Installed,
    /// The download failed. Its files are kept for a retry from the same source.
    DownloadFailed {
        /// Why.
        reason: String,
    },
    /// The model arrived but is not valid. Its files are discarded on a retry.
    ValidationFailed {
        /// Why.
        reason: String,
    },
}

/// One model in the registry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRecord {
    /// The model id.
    pub id: String,
    /// Its directory name under the models directory.
    pub dir: String,
    /// How far its install has got.
    pub state: InstallState,
    /// Where it came from; `None` for a model adopted from disk with no source in its
    /// manifest.
    pub source: Option<ModelSource>,
    /// When the state last changed, in seconds since the Unix epoch.
    pub updated_at: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Store {
    schema_version: u32,
    models: BTreeMap<String, ModelRecord>,
}

/// What [`Registry::reconcile`] changed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reconciled {
    /// Valid model directories that had no record.
    pub adopted: Vec<String>,
    /// Installs a crash interrupted after the model reached its place.
    pub recovered: Vec<String>,
    /// Installed models whose directories have gone.
    pub dropped: Vec<String>,
    /// Directories holding a manifest that are not usable, with why.
    pub skipped: Vec<(PathBuf, String)>,
}

fn io_err(path: &Path) -> impl FnOnce(io::Error) -> RegistryError + '_ {
    move |source| RegistryError::Io {
        path: path.to_owned(),
        source,
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && !id.starts_with('.')
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// A record's directory must be exactly one normal path component.
fn single_component(dir: &str) -> bool {
    let mut parts = Path::new(dir).components();
    matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none()
}

/// An advisory lock, released when the file closes, including when a process dies.
#[cfg(unix)]
fn lock(file: &File, wait: bool) -> io::Result<bool> {
    use rustix::fs::{FlockOperation, flock};
    let op = if wait {
        FlockOperation::LockExclusive
    } else {
        FlockOperation::NonBlockingLockExclusive
    };
    match flock(file, op) {
        Ok(()) => Ok(true),
        Err(e) if e == rustix::io::Errno::WOULDBLOCK => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// No lock where `flock` does not exist; the supported targets are all Unix.
#[cfg(not(unix))]
#[allow(clippy::unnecessary_wraps)]
fn lock(_file: &File, _wait: bool) -> io::Result<bool> {
    Ok(true)
}

fn open_lock(path: &Path) -> Result<File, RegistryError> {
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(path)
        .map_err(io_err(path))
}

/// Makes a rename in `dir` durable. Directories cannot be opened for sync everywhere,
/// so this is best effort outside Unix.
fn sync_dir(dir: &Path) -> Result<(), RegistryError> {
    if cfg!(unix) {
        File::open(dir)
            .and_then(|d| d.sync_all())
            .map_err(io_err(dir))?;
    }
    Ok(())
}

/// Deletes a directory that may already be gone (another process may be cleaning too).
fn remove_tree(dir: &Path) -> Result<(), RegistryError> {
    match fs::remove_dir_all(dir) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => Err(io_err(dir)(e)),
        _ => Ok(()),
    }
}

/// A models directory's registry.
#[derive(Debug)]
pub struct Registry {
    root: PathBuf,
}

impl Registry {
    /// The registry of the models directory `root`, which is created if missing.
    ///
    /// # Errors
    /// [`RegistryError::Io`] if the directory cannot be created.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, RegistryError> {
        let root = root.into();
        fs::create_dir_all(&root).map_err(io_err(&root))?;
        Ok(Self { root })
    }

    /// The models directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn file(&self) -> PathBuf {
        self.root.join(REGISTRY_FILE)
    }

    fn staging_of(&self, id: &str) -> PathBuf {
        self.root.join(STAGING_DIR).join(id)
    }

    fn read(&self) -> Result<Store, RegistryError> {
        let path = self.file();
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Ok(Store {
                    schema_version: SCHEMA_VERSION,
                    models: BTreeMap::new(),
                });
            }
            Err(e) => return Err(io_err(&path)(e)),
        };
        let corrupt = |message: String| RegistryError::Corrupt {
            path: path.clone(),
            message,
        };
        let store: Store =
            serde_json::from_slice(&bytes).map_err(|e| corrupt(format!("cannot be read: {e}")))?;
        if store.schema_version != SCHEMA_VERSION {
            return Err(corrupt(format!(
                "schema version {}; this build reads {SCHEMA_VERSION}",
                store.schema_version
            )));
        }
        if let Some(r) = store.models.values().find(|r| !single_component(&r.dir)) {
            return Err(corrupt(format!(
                "the model {:?} names the directory {:?}, which is not directly inside the models directory",
                r.id, r.dir
            )));
        }
        Ok(store)
    }

    fn write(&self, store: &Store) -> Result<(), RegistryError> {
        let (path, tmp) = (self.file(), self.root.join(".registry.json.tmp"));
        let bytes = serde_json::to_vec_pretty(store).map_err(|e| RegistryError::Corrupt {
            path: path.clone(),
            message: e.to_string(),
        })?;
        File::create(&tmp)
            .and_then(|mut f| f.write_all(&bytes).and_then(|()| f.sync_all()))
            .map_err(io_err(&tmp))?;
        fs::rename(&tmp, &path).map_err(io_err(&path))?;
        sync_dir(&self.root)
    }

    /// One read, change and atomic write under the registry's lock.
    fn update<R>(
        &self,
        f: impl FnOnce(&mut Store) -> Result<R, RegistryError>,
    ) -> Result<R, RegistryError> {
        let path = self.root.join(LOCK_FILE);
        let guard = open_lock(&path)?;
        lock(&guard, true).map_err(io_err(&path))?;
        let mut store = self.read()?;
        let out = f(&mut store)?;
        self.write(&store)?;
        Ok(out)
    }

    /// Every record, by id.
    ///
    /// # Errors
    /// [`RegistryError::Corrupt`] or [`RegistryError::Io`] if the file cannot be read.
    pub fn list(&self) -> Result<Vec<ModelRecord>, RegistryError> {
        Ok(self.read()?.models.into_values().collect())
    }

    /// The record of `id`.
    ///
    /// # Errors
    /// [`RegistryError::Corrupt`] or [`RegistryError::Io`] if the file cannot be read.
    pub fn get(&self, id: &str) -> Result<Option<ModelRecord>, RegistryError> {
        Ok(self.read()?.models.remove(id))
    }

    /// The directory a record names.
    #[must_use]
    pub fn dir_of(&self, record: &ModelRecord) -> PathBuf {
        self.root.join(&record.dir)
    }

    /// Takes `id`'s install lock without waiting. `Ok(None)`: another install holds it.
    fn try_install_lock(&self, id: &str) -> Result<Option<File>, RegistryError> {
        let dir = self.root.join(STAGING_DIR);
        fs::create_dir_all(&dir).map_err(io_err(&dir))?;
        let path = dir.join(format!("{id}.lock"));
        let file = open_lock(&path)?;
        Ok(lock(&file, false).map_err(io_err(&path))?.then_some(file))
    }

    /// Starts or resumes installing `id` from `source`. The caller fills
    /// [`Install::staging`], then calls [`Install::commit`]. Files from an earlier
    /// attempt are kept only if it was the same source and its download had not failed
    /// validation. Dropping the [`Install`] leaves the record as it is, as a crash does.
    ///
    /// # Errors
    /// [`RegistryError::InvalidId`], [`RegistryError::AlreadyInstalled`],
    /// [`RegistryError::InstallInProgress`], or a file error.
    pub fn begin_install(
        &self,
        id: &str,
        source: Option<ModelSource>,
    ) -> Result<Install<'_>, RegistryError> {
        if !valid_id(id) {
            return Err(RegistryError::InvalidId(id.to_owned()));
        }
        let lock = self
            .try_install_lock(id)?
            .ok_or_else(|| RegistryError::InstallInProgress(id.to_owned()))?;
        let staging = self.staging_of(id);
        self.update(|s| {
            let old = s.models.get(id);
            if old.is_some_and(|r| r.state == InstallState::Installed)
                || self.root.join(id).exists()
            {
                return Err(RegistryError::AlreadyInstalled(id.to_owned()));
            }
            let keep = old.is_some_and(|r| {
                r.source == source && !matches!(r.state, InstallState::ValidationFailed { .. })
            });
            if !keep {
                remove_tree(&staging)?;
            }
            s.models.insert(
                id.to_owned(),
                ModelRecord {
                    id: id.to_owned(),
                    dir: id.to_owned(),
                    state: InstallState::Downloading,
                    source,
                    updated_at: now_secs(),
                },
            );
            Ok(())
        })?;
        fs::create_dir_all(&staging).map_err(io_err(&staging))?;
        Ok(Install {
            registry: self,
            id: id.to_owned(),
            staging,
            _lock: lock,
        })
    }

    fn set_state(&self, id: &str, state: InstallState) -> Result<(), RegistryError> {
        self.update(|s| {
            let r = s
                .models
                .get_mut(id)
                .ok_or_else(|| RegistryError::NotFound(id.to_owned()))?;
            r.state = state;
            r.updated_at = now_secs();
            Ok(())
        })
    }

    /// Brings the registry in line with the directory: finishes installs a crash
    /// interrupted after the rename, adopts valid model directories with no record,
    /// drops installed records whose directories have gone, and empties `.trash`.
    /// Installs another process is running are left alone.
    ///
    /// # Errors
    /// A file error, or [`RegistryError::Corrupt`].
    pub fn reconcile(&self, arch: &ArchitectureRegistry) -> Result<Reconciled, RegistryError> {
        let mut out = Reconciled::default();
        let mut found = BTreeMap::new();
        for d in discover(std::slice::from_ref(&self.root), arch) {
            // A manifest in the models directory itself is not an installed model:
            // removing it would remove every other model too.
            if d.dir == self.root {
                continue;
            }
            match (d.id(), &d.model) {
                (Some(id), Ok(_)) if found.contains_key(&id) => out
                    .skipped
                    .push((d.dir, format!("the id {id:?} is used by another directory"))),
                (Some(id), Ok(m)) => {
                    let source = m.manifest.source.clone();
                    found.insert(id, (d.dir, source));
                }
                (_, Err(e)) => out.skipped.push((d.dir, e.to_string())),
                (None, Ok(_)) => {}
            }
        }
        // Take the locks before the registry's, in the order `begin_install` does.
        let held: Vec<(String, Option<File>)> = self
            .list()?
            .into_iter()
            .filter(|r| r.state != InstallState::Installed)
            .map(|r| Ok((r.id.clone(), self.try_install_lock(&r.id)?)))
            .collect::<Result<_, RegistryError>>()?;
        let busy = |id: &str| held.iter().any(|(i, l)| i == id && l.is_none());
        let in_root = |dir: &Path| {
            dir.strip_prefix(&self.root)
                .ok()
                .and_then(Path::to_str)
                .filter(|d| single_component(d))
                .map(str::to_owned)
        };
        self.update(|s| {
            let gone: Vec<String> = s
                .models
                .values()
                .filter(|r| r.state == InstallState::Installed && !self.dir_of(r).exists())
                .map(|r| r.id.clone())
                .collect();
            for id in gone {
                s.models.remove(&id);
                out.dropped.push(id);
            }
            for (id, (dir, source)) in &found {
                let Some(name) = in_root(dir) else { continue };
                match s.models.get_mut(id) {
                    Some(r) if r.state == InstallState::Installed || busy(id) => {}
                    Some(r) => {
                        r.dir = name;
                        r.state = InstallState::Installed;
                        r.updated_at = now_secs();
                        out.recovered.push(id.clone());
                    }
                    None => {
                        s.models.insert(
                            id.clone(),
                            ModelRecord {
                                id: id.clone(),
                                dir: name,
                                state: InstallState::Installed,
                                source: source.clone(),
                                updated_at: now_secs(),
                            },
                        );
                        out.adopted.push(id.clone());
                    }
                }
            }
            Ok(())
        })?;
        drop(held);
        remove_tree(&self.root.join(TRASH_DIR))?;
        Ok(out)
    }

    /// Deletes `id`'s directory, any staged files and its record. Refuses while an
    /// install of it runs, and refuses a symbolic link. The directory moves to `.trash`
    /// before the record goes, and a crash between them leaves a record with no
    /// directory, which [`Registry::reconcile`] drops, along with anything in `.trash`.
    ///
    /// # Errors
    /// [`RegistryError::NotFound`], [`RegistryError::InstallInProgress`],
    /// [`RegistryError::Refused`], or a file error.
    pub fn remove(&self, id: &str) -> Result<ModelRecord, RegistryError> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let record = self
            .get(id)?
            .ok_or_else(|| RegistryError::NotFound(id.to_owned()))?;
        let _lock = self
            .try_install_lock(id)?
            .ok_or_else(|| RegistryError::InstallInProgress(id.to_owned()))?;
        let dir = self.dir_of(&record);
        let exists = match fs::symlink_metadata(&dir) {
            Ok(m) if m.is_dir() => true,
            Ok(_) => {
                return Err(RegistryError::Refused(format!(
                    "{} is not a plain directory; remove it by hand",
                    dir.display()
                )));
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => false,
            Err(e) => return Err(io_err(&dir)(e)),
        };
        let trash = self.root.join(TRASH_DIR);
        fs::create_dir_all(&trash).map_err(io_err(&trash))?;
        let bin = trash.join(format!(
            "{}-{}-{}",
            record.dir,
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        self.update(|s| {
            s.models.remove(id);
            if exists {
                fs::rename(&dir, &bin).map_err(io_err(&dir))?;
            }
            Ok(())
        })?;
        remove_tree(&bin)?;
        remove_tree(&self.staging_of(id))?;
        Ok(record)
    }
}

/// An install in progress, holding the model's install lock.
#[derive(Debug)]
pub struct Install<'r> {
    registry: &'r Registry,
    id: String,
    staging: PathBuf,
    _lock: File,
}

impl Install<'_> {
    /// Where the model's files go, with its `manifest.json`.
    #[must_use]
    pub fn staging(&self) -> &Path {
        &self.staging
    }

    /// Records a failed download. Its files stay for a retry from the same source.
    ///
    /// # Errors
    /// A file error.
    pub fn fail(self, reason: impl Into<String>) -> Result<(), RegistryError> {
        self.registry.set_state(
            &self.id,
            InstallState::DownloadFailed {
                reason: reason.into(),
            },
        )
    }

    fn validate(&self, arch: &ArchitectureRegistry) -> Result<(), String> {
        let meta = ModelMetadata::load(&self.staging, arch).map_err(|e| e.to_string())?;
        let id = &meta.manifest.model.id;
        if *id == self.id {
            Ok(())
        } else {
            Err(format!("its manifest says it is {id:?}"))
        }
    }

    /// Validates the staged model and moves it into place. A model that fails stays
    /// out of place and is recorded as [`InstallState::ValidationFailed`].
    ///
    /// # Errors
    /// [`RegistryError::Validation`], or a file error.
    pub fn commit(self, arch: &ArchitectureRegistry) -> Result<ModelRecord, RegistryError> {
        let r = self.registry;
        r.set_state(&self.id, InstallState::Validating)?;
        if let Err(reason) = self.validate(arch) {
            r.set_state(
                &self.id,
                InstallState::ValidationFailed {
                    reason: reason.clone(),
                },
            )?;
            return Err(RegistryError::Validation {
                id: self.id,
                reason,
            });
        }
        let target = r.root.join(&self.id);
        fs::rename(&self.staging, &target).map_err(io_err(&target))?;
        sync_dir(&r.root)?;
        r.set_state(&self.id, InstallState::Installed)?;
        r.get(&self.id)?
            .ok_or_else(|| RegistryError::NotFound(self.id.clone()))
    }
}
