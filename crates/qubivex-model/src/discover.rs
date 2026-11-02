//! Finding models: every directory holding a `manifest.json`, in the given roots or
//! one level below them.

use std::path::{Path, PathBuf};

use crate::{ArchitectureRegistry, MANIFEST_FILE, Manifest, ModelError, ModelMetadata};

/// A model directory found by [`discover`].
#[derive(Debug)]
pub struct Discovered {
    /// The directory.
    pub dir: PathBuf,
    /// The validated model, or why it failed.
    pub model: Result<ModelMetadata, ModelError>,
}

impl Discovered {
    /// The model id, from the manifest even when validation failed.
    #[must_use]
    pub fn id(&self) -> Option<String> {
        match &self.model {
            Ok(m) => Some(m.manifest.model.id.clone()),
            Err(_) => Manifest::load(&self.dir.join(MANIFEST_FILE))
                .ok()
                .map(|m| m.model.id),
        }
    }
}

/// Model directories under `roots` (each root itself, and its immediate
/// subdirectories), loaded and validated, sorted by directory. Unreadable roots are
/// skipped: a configured directory that does not exist yet holds no models.
#[must_use]
pub fn discover(roots: &[PathBuf], registry: &ArchitectureRegistry) -> Vec<Discovered> {
    let mut dirs = Vec::new();
    for root in roots {
        if root.join(MANIFEST_FILE).is_file() {
            dirs.push(root.clone());
        }
        let Ok(rd) = std::fs::read_dir(root) else {
            continue;
        };
        for e in rd.filter_map(Result::ok) {
            let p = e.path();
            if p.is_dir() && p.join(MANIFEST_FILE).is_file() {
                dirs.push(p);
            }
        }
    }
    dirs.sort();
    dirs.dedup();
    dirs.into_iter()
        .map(|dir| Discovered {
            model: ModelMetadata::load(&dir, registry),
            dir,
        })
        .collect()
}

/// Resolves `what`: a model directory (or its `manifest.json`), else a model id
/// among the models under `roots`.
///
/// # Errors
/// [`ModelError::Registry`] if `what` is neither a model directory nor a known id,
/// or if the id is ambiguous; any load error of the model found.
pub fn find(
    what: &str,
    roots: &[PathBuf],
    registry: &ArchitectureRegistry,
) -> Result<ModelMetadata, ModelError> {
    let p = Path::new(what);
    let dir = if p.file_name().is_some_and(|n| n == MANIFEST_FILE) {
        p.parent().unwrap_or(p)
    } else {
        p
    };
    if dir.join(MANIFEST_FILE).is_file() {
        return ModelMetadata::load(dir, registry);
    }
    let mut hits: Vec<Discovered> = discover(roots, registry)
        .into_iter()
        .filter(|d| d.id().as_deref() == Some(what))
        .collect();
    match hits.len() {
        0 => Err(ModelError::Registry(format!(
            "no model {what:?}: not a model directory, and no manifest under {} has that id",
            roots
                .iter()
                .map(|r| r.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
        1 => hits.remove(0).model,
        _ => Err(ModelError::Registry(format!(
            "model id {what:?} is used by {}; pass a directory instead",
            hits.iter()
                .map(|d| d.dir.display().to_string())
                .collect::<Vec<_>>()
                .join(" and ")
        ))),
    }
}
