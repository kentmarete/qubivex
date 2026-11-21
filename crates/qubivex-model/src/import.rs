//! `qubivex models import`: write a manifest for a checkpoint (ADR-004).
//!
//! The importer finds the weight files, then tries each tensor-name preset and
//! q/k-norm mode against the real tensors, and keeps the first combination that
//! validates. Values the checkpoint does not record come from `set`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use crate::manifest::{
    ArchRef, ManifestFormat, ModelSection, ModelSource, Overrides, SCHEMA_VERSION, SemVer, Shard,
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
    /// Where the checkpoint came from, if a provider supplied it.
    pub source: Option<ModelSource>,
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

/// A model id from a free-form name: lower case, with unsafe characters replaced.
#[must_use]
pub fn sanitise_id(s: &str) -> String {
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
    let index = path.join(crate::SHARD_INDEX_FILE);
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
        schema_version: SCHEMA_VERSION,
        model: ModelSection {
            id: opts.id.clone().unwrap_or_else(|| sanitise_id(&stem)),
            name: opts.name.clone().unwrap_or_else(|| stem.clone()),
            version: None,
        },
        source: opts.source.clone(),
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
        files: Vec::new(),
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
    // A tied output head reads the embedding. Some checkpoints still store a copy
    // under the head's name; it is not needed, so it is ignored rather than loaded.
    let mut base = base;
    let tied = opts
        .set
        .get("tie_word_embeddings")
        .or_else(|| hf_config.as_ref()?.get("tie_word_embeddings"))
        .and_then(Value::as_bool)
        == Some(true);
    let head = match format {
        ManifestFormat::Gguf => "output.weight",
        ManifestFormat::Safetensors => "lm_head.weight",
    };
    if tied && tensors.get(head).is_some() {
        base.overrides.ignore_tensors.push(head.to_owned());
        warnings.push(format!(
            "the output head is tied to the embedding; the stored {head:?} is ignored"
        ));
    }
    // Each built-in architecture in turn; a config it refuses (a dense checkpoint has
    // no experts for `transformer_moe`) moves on to the next.
    let mut best: Option<(usize, ModelError)> = None;
    let mut refused: Vec<String> = Vec::new();
    // Attention options are detected from the tensors: a candidate that leaves a tensor
    // unused, or misses one, fails validation, so the checkpoint's own layout decides.
    'arch: for (arch, names, qkv_biases) in [
        ("transformer_moe", candidates(format), &[false][..]),
        (
            "transformer_dense",
            dense_candidates(format),
            &[false, true][..],
        ),
    ] {
        for tensor_names in names {
            for (qk, &qkv_bias) in ["none", "full", "per_head"]
                .into_iter()
                .flat_map(|qk| qkv_biases.iter().map(move |b| (qk, b)))
            {
                let mut m = base.clone();
                m.architecture.id = arch.to_owned();
                m.architecture
                    .config
                    .insert("tensor_names".into(), tensor_names.clone());
                m.architecture.config.insert("qk_norm".into(), json!(qk));
                if arch == "transformer_moe" {
                    let found = moe_tensor_options(&tensor_names, format, &tensors);
                    if !found.is_empty() {
                        // These options arrived in transformer_moe 1.2.
                        m.architecture.version.minor = m.architecture.version.minor.max(2);
                    }
                    m.architecture.config.extend(found);
                    config_class_defaults(hf_config.as_ref(), &mut m);
                }
                if qkv_bias {
                    m.architecture.config.insert("qkv_bias".into(), json!(true));
                    // `qkv_bias` arrived in transformer_dense 1.1; manifests without it
                    // stay 1.0 so older builds still read them.
                    m.architecture.version = SemVer {
                        major: 1,
                        minor: 1,
                        patch: 0,
                    };
                }
                match check(&m, &tensors, hf_config.as_ref(), registry) {
                    Ok((cfg, _)) => {
                        let mut m = m;
                        m.quantization = dominant_dtype(&tensors);
                        // RoPE scaling arrived in transformer_dense 1.2 and
                        // transformer_moe 1.1: an older build must refuse the manifest
                        // rather than run it with unscaled positions.
                        let d = &cfg.describe;
                        if d["rope_scaling"]["type"] != "none" {
                            let minor = if arch == "transformer_dense" { 2 } else { 1 };
                            m.architecture.version.minor = m.architecture.version.minor.max(minor);
                        }
                        // YaRN and sliding windows arrived in transformer_dense 1.3 and
                        // transformer_moe 1.2, with the MoE's attention and expert
                        // biases, sinks, top-K softmax routing and clamped SwiGLU.
                        let windows = d["attention_windows"]
                            .as_array()
                            .is_some_and(|w| w.iter().any(|x| !x.is_null()));
                        let newer = windows
                            || d["rope_scaling"]["type"] == "yarn"
                            || d["attention_bias"] == true
                            || d["router_scoring"] == "topk_softmax"
                            || d["activation"]["type"] == "clamped_swiglu";
                        if newer {
                            let minor = if arch == "transformer_dense" { 3 } else { 2 };
                            m.architecture.version.minor = m.architecture.version.minor.max(minor);
                        }
                        return Ok(Imported {
                            dir,
                            manifest: m,
                            warnings,
                        });
                    }
                    // Config errors do not depend on the layout: try the next architecture.
                    Err(ModelError::Config(e)) => {
                        refused.push(format!("{arch}: {e}"));
                        continue 'arch;
                    }
                    Err(e @ ModelError::Registry(_)) => return Err(e),
                    Err(ModelError::Validation { model, problems }) => {
                        if best.as_ref().is_none_or(|(n, _)| problems.len() < *n) {
                            best =
                                Some((problems.len(), ModelError::Validation { model, problems }));
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
        }
    }
    match best {
        Some((_, e)) => Err(e),
        None if !refused.is_empty() => Err(ModelError::Config(refused.join("; "))),
        None => Err(import_err(&dir, "no layout candidates")),
    }
}

/// The `transformer_moe` options a checkpoint's own tensors decide: each is on when
/// its tensor exists for layer 0 under the candidate's names (a candidate that is
/// wrong then fails validation on the tensors it misses or leaves unused).
fn moe_tensor_options(
    tensor_names: &Value,
    format: ManifestFormat,
    index: &crate::TensorIndex,
) -> Map<String, Value> {
    let mut out = Map::new();
    let Ok(n) = crate::arch::moe_names(Some(tensor_names), format) else {
        return out;
    };
    let has = |p: &str| index.get(&p.replace("{L}", "0")).is_some();
    for (key, present) in [
        ("attention_sinks", has(&n.sinks)),
        ("router_bias", has(&n.router_bias)),
        // Expert biases exist only beside a stacked gate-up (the config refuses them
        // otherwise, which would end the search for this architecture).
        (
            "expert_bias",
            matches!(n.experts, crate::arch::ExpertLayout::StackedGateUp { .. })
                && (has(&n.expert_gate_up_bias) || has(&n.expert_down_bias)),
        ),
    ] {
        if present {
            out.insert(key.into(), json!(true));
        }
    }
    out
}

/// Applies the values a checkpoint's `config.json` leaves to its transformers config
/// class ([`class_defaults`]), for its `model_type`. Each applies only when the
/// checkpoint omits it, and is written into the manifest, so the model is read the same
/// way without this table: options go to `architecture.config`, values to
/// `overrides.config`.
fn config_class_defaults(hf: Option<&Value>, m: &mut Manifest) {
    let Some(hf) = hf else { return };
    let Some(mt) = hf.get("model_type").and_then(Value::as_str) else {
        return;
    };
    for (key, defaults) in class_defaults() {
        if key != mt {
            continue;
        }
        for (section, k, v) in defaults {
            if hf.get(k).is_some() {
                continue;
            }
            let map = if section == "architecture" {
                &mut m.architecture.config
            } else {
                &mut m.overrides.config
            };
            map.entry(k.to_owned()).or_insert(v);
        }
    }
}

/// The defaults transformers' config classes supply, by `model_type` (built at run
/// time: JSON values are not `const`).
fn class_defaults() -> Vec<(&'static str, Vec<(&'static str, &'static str, Value)>)> {
    vec![(
        "gpt_oss", // model-name-ok: a transformers model_type
        vec![
            // GptOssTopKRouter: top-K logits, then a softmax over them.
            ("architecture", "router_scoring", json!("topk_softmax")),
            // GptOssConfig.swiglu_alpha.
            ("overrides", "swiglu_alpha", json!(1.702)),
        ],
    )]
}

/// Tensor-name presets to try for `transformer_dense`.
fn dense_candidates(format: ManifestFormat) -> Vec<Value> {
    match format {
        ManifestFormat::Gguf => vec![json!("gguf")],
        ManifestFormat::Safetensors => vec![json!("hf")],
    }
}

/// Tensor-name presets to try, most common first, with known alternative router names.
fn candidates(format: ManifestFormat) -> Vec<Value> {
    match format {
        ManifestFormat::Gguf => vec![json!("gguf")],
        ManifestFormat::Safetensors => {
            let mut v = Vec::new();
            for base in ["hf", "hf_fused", "hf_interleaved"] {
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
        assert_eq!(sanitise_id("My Model 7B"), "my-model-7b");
        assert_eq!(sanitise_id(""), "model");
        assert_eq!(sanitise_id("a/b\\c"), "a-b-c");
    }
}
