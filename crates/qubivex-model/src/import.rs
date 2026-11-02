//! `qubivex models import`: write a manifest for a checkpoint (ADR-004).
//!
//! The importer finds the weight files, then tries each tensor-name preset and
//! q/k-norm mode against the real tensors, and keeps the first combination that
//! validates. Values the checkpoint does not record come from `set`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use crate::manifest::{
    ArchRef, ManifestFormat, ModelSection, Overrides, SCHEMA_MAJOR, SemVer, Shard,
    TokenizerSection, WeightsSection,
};
use crate::metadata::{check, index_files, read_config};
use crate::{ArchitectureRegistry, Manifest, ModelError};

/// What to import and how.
#[derive(Clone, Debug, Default)]
pub struct ImportOptions {
    /// Model id; defaults to the directory (or file stem) name, sanitised.
    pub id: Option<String>,
    /// Display name; defaults to the directory (or file stem) name.
    pub name: Option<String>,
    /// `overrides.config` values (`config.json` names), for values the checkpoint lacks.
    pub set: Map<String, Value>,
}

/// An imported manifest.
#[derive(Clone, Debug)]
pub struct Imported {
    /// The model directory (where `manifest.json` belongs).
    pub dir: PathBuf,
    /// The manifest, already validated against the files.
    pub manifest: Manifest,
    /// Things the user should know (a missing tokenizer, for example).
    pub warnings: Vec<String>,
}

fn sanitise(s: &str) -> String {
    let mut id: String = s
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "._-".contains(c) {
                c
            } else {
                '-'
            }
        })
        .collect();
    id.truncate(128);
    if id.is_empty() {
        "model".to_owned()
    } else {
        id
    }
}

fn import_err(path: &Path, detail: impl Into<String>) -> ModelError {
    ModelError::Manifest {
        path: path.to_owned(),
        detail: detail.into(),
    }
}

/// The weight files of `path`: one `.gguf` file, or a directory of SafeTensors shards
/// (listed by `model.safetensors.index.json` when present) or one `.gguf`.
fn locate(path: &Path) -> Result<(PathBuf, ManifestFormat, Vec<String>), ModelError> {
    let is_gguf = |p: &Path| {
        p.extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
    };
    if path.is_file() {
        if !is_gguf(path) {
            return Err(import_err(
                path,
                "import a model directory, or a .gguf file",
            ));
        }
        let dir = path
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        return Ok((dir, ManifestFormat::Gguf, vec![name]));
    }
    let entries: Vec<String> = std::fs::read_dir(path)
        .map_err(|source| ModelError::Io {
            path: path.to_owned(),
            source,
        })?
        .filter_map(Result::ok)
        .filter(|e| e.path().is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    let mut ggufs: Vec<&String> = entries.iter().filter(|n| is_gguf(Path::new(n))).collect();
    let mut sts: Vec<String> = entries
        .iter()
        .filter(|n| n.ends_with(".safetensors"))
        .cloned()
        .collect();
    let index = path.join("model.safetensors.index.json");
    if index.is_file() {
        let text = crate::manifest::read_small(&index, 64 << 20)?;
        let v: Value =
            serde_json::from_str(&text).map_err(|e| import_err(&index, e.to_string()))?;
        let files: BTreeSet<String> = v["weight_map"]
            .as_object()
            .ok_or_else(|| import_err(&index, "no weight_map object"))?
            .values()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();
        sts = files.into_iter().collect();
    }
    sts.sort();
    ggufs.sort();
    match (sts.is_empty(), ggufs.len()) {
        (false, 0) => Ok((path.to_owned(), ManifestFormat::Safetensors, sts)),
        (true, 1) => Ok((
            path.to_owned(),
            ManifestFormat::Gguf,
            vec![ggufs[0].clone()],
        )),
        (true, 0) => Err(import_err(path, "no .safetensors or .gguf files")),
        _ => Err(import_err(
            path,
            "several weight formats or GGUF files; import one .gguf file by path",
        )),
    }
}

/// Builds and validates a manifest for the checkpoint at `path`.
///
/// # Errors
/// [`ModelError::Manifest`] if no weights are found; header and config errors; and,
/// if no layout validates, the problems of the closest candidate.
pub fn import(
    path: &Path,
    opts: &ImportOptions,
    registry: &ArchitectureRegistry,
) -> Result<Imported, ModelError> {
    let (dir, format, shard_names) = locate(path)?;
    let stem = if path.is_file() {
        path.file_stem().map(|s| s.to_string_lossy().into_owned())
    } else {
        dir.canonicalize()
            .ok()
            .and_then(|d| d.file_name().map(|s| s.to_string_lossy().into_owned()))
    }
    .unwrap_or_else(|| "model".to_owned());
    let mut warnings = Vec::new();
    let tokenizer = match format {
        ManifestFormat::Gguf => TokenizerSection::Gguf,
        ManifestFormat::Safetensors => {
            if !dir.join("tokenizer.json").is_file() {
                warnings.push(
                    "no tokenizer.json next to the weights; generation will need one".to_owned(),
                );
            }
            TokenizerSection::Huggingface {
                path: "tokenizer.json".to_owned(),
            }
        }
    };
    let shards: Vec<Shard> = shard_names
        .iter()
        .map(|p| {
            let size = std::fs::metadata(dir.join(p)).map(|m| m.len()).ok();
            Shard {
                path: p.clone(),
                size,
                sha256: None,
            }
        })
        .collect();
    let base = Manifest {
        schema_version: SemVer {
            major: SCHEMA_MAJOR,
            minor: 0,
            patch: 0,
        },
        model: ModelSection {
            id: opts.id.clone().unwrap_or_else(|| sanitise(&stem)),
            name: opts.name.clone().unwrap_or_else(|| stem.clone()),
            version: None,
        },
        architecture: ArchRef {
            id: "transformer_moe".to_owned(),
            version: SemVer {
                major: 1,
                minor: 0,
                patch: 0,
            },
            config: Map::new(),
        },
        weights: WeightsSection {
            format,
            shards,
            config: (format == ManifestFormat::Safetensors).then(|| "config.json".to_owned()),
        },
        tokenizer,
        context_length: None,
        capabilities: Vec::new(),
        quantization: None,
        overrides: Overrides {
            config: opts.set.clone(),
            ignore_tensors: Vec::new(),
        },
    };
    // Round-trip through the parser so the checks on ids and paths apply.
    let text = serde_json::to_string(&base).map_err(|e| import_err(&dir, e.to_string()))?;
    let base = Manifest::parse(&text, &dir.join(crate::MANIFEST_FILE))?;

    let tensors = index_files(&dir, &base)?;
    let hf_config = read_config(&dir, &base)?;
    let mut best: Option<(usize, ModelError)> = None;
    for tensor_names in candidates(format) {
        for qk in ["none", "full", "per_head"] {
            let mut m = base.clone();
            m.architecture
                .config
                .insert("tensor_names".into(), tensor_names.clone());
            m.architecture.config.insert("qk_norm".into(), json!(qk));
            match check(&m, &tensors, hf_config.as_ref(), registry) {
                Ok(_) => {
                    let mut m = m;
                    m.quantization = dominant_dtype(&tensors);
                    return Ok(Imported {
                        dir,
                        manifest: m,
                        warnings,
                    });
                }
                // Config errors do not depend on the layout: report them directly.
                Err(e @ (ModelError::Config(_) | ModelError::Registry(_))) => return Err(e),
                Err(ModelError::Validation { model, problems }) => {
                    if best.as_ref().is_none_or(|(n, _)| problems.len() < *n) {
                        best = Some((problems.len(), ModelError::Validation { model, problems }));
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }
    Err(best.map_or_else(|| import_err(&dir, "no layout candidates"), |(_, e)| e))
}

/// Tensor-name presets to try, most common first, with known alternative router names.
fn candidates(format: ManifestFormat) -> Vec<Value> {
    match format {
        ManifestFormat::Gguf => vec![json!("gguf")],
        ManifestFormat::Safetensors => {
            let mut v = Vec::new();
            for base in ["hf", "hf_fused"] {
                v.push(json!(base));
                v.push(json!({ "base": base, "router": "model.layers.{L}.mlp.router.weight" }));
            }
            v
        }
    }
}

/// The dtype holding the most bytes, as a display label.
fn dominant_dtype(t: &crate::TensorIndex) -> Option<String> {
    t.bytes_by_dtype()
        .into_iter()
        .max_by_key(|&(_, b)| b)
        .map(|(d, _)| d.name().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_sanitised() {
        assert_eq!(sanitise("My Model 7B"), "my-model-7b");
        assert_eq!(sanitise(""), "model");
        assert_eq!(sanitise("a/b\\c"), "a-b-c");
    }
}
