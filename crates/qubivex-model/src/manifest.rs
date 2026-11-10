//! `manifest.json`, schema 1.1 (ADR-004). JSON Schema: `docs/model-support/manifest.schema.json`.
//!
//! A minor version adds optional fields, so 1.0 still reads as it is; [`Manifest::upgraded`]
//! moves one forward. A manifest from a newer minor is refused by name rather than as an
//! unknown field.

use std::fmt;
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::ModelError;

/// The manifest schema major version this build reads.
pub const SCHEMA_MAJOR: u64 = 1;
/// The newest minor of [`SCHEMA_MAJOR`] this build reads, and the one it writes.
pub const SCHEMA_MINOR: u64 = 1;
/// The schema version this build writes.
pub const SCHEMA_VERSION: SemVer = SemVer {
    major: SCHEMA_MAJOR,
    minor: SCHEMA_MINOR,
    patch: 0,
};

/// A `major.minor.patch` version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SemVer {
    /// Incompatible changes.
    pub major: u64,
    /// Compatible additions.
    pub minor: u64,
    /// Fixes.
    pub patch: u64,
}

impl FromStr for SemVer {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let mut it = s.split('.');
        let mut n = || -> Result<u64, String> {
            let p = it
                .next()
                .ok_or_else(|| format!("{s:?} is not major.minor.patch"))?;
            if p.is_empty()
                || !p.bytes().all(|b| b.is_ascii_digit())
                || (p.len() > 1 && p.starts_with('0'))
            {
                return Err(format!("{s:?} is not major.minor.patch"));
            }
            p.parse().map_err(|_| format!("{s:?}: number too large"))
        };
        let v = Self {
            major: n()?,
            minor: n()?,
            patch: n()?,
        };
        if it.next().is_some() {
            return Err(format!("{s:?} is not major.minor.patch"));
        }
        Ok(v)
    }
}

impl fmt::Display for SemVer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl Serialize for SemVer {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for SemVer {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

/// The `model` section.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSection {
    /// Stable id, used on the command line: lowercase letters, digits, `.`, `_`, `-`.
    pub id: String,
    /// Display name.
    pub name: String,
    /// The model's own version or revision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// Where a model came from. Provider-neutral: repository metadata is normalised to this,
/// and nothing that runs a model reads it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSource {
    /// The provider, such as `huggingface`.
    pub provider: String,
    /// The repository at that provider.
    pub repository: String,
    /// The exact revision.
    pub revision: String,
}

/// The `architecture` section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchRef {
    /// Registered architecture id, for example `transformer_moe`.
    pub id: String,
    /// The architecture version the manifest was written for.
    pub version: SemVer,
    /// Architecture-specific options that the checkpoint does not declare. Each
    /// architecture defines and validates its own keys.
    #[serde(default)]
    pub config: serde_json::Map<String, serde_json::Value>,
}

/// Container format of the weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ManifestFormat {
    /// SafeTensors shards plus `config.json`.
    Safetensors,
    /// One GGUF file.
    Gguf,
}

/// One file of a model: a weight shard, or an entry of [`Manifest::files`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shard {
    /// Path relative to the model directory.
    pub path: String,
    /// Expected size in bytes; checked at load when given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    /// Expected sha256, lowercase hex; checked by [`crate::verify_files`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

/// The `weights` section.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightsSection {
    /// Container format.
    pub format: ManifestFormat,
    /// Weight files.
    pub shards: Vec<Shard>,
    /// `config.json`, relative to the model directory (SafeTensors only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<String>,
}

/// The `tokenizer` section.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "lowercase")]
pub enum TokenizerSection {
    /// A Hugging Face `tokenizer.json`.
    Huggingface {
        /// Path relative to the model directory.
        path: String,
    },
    /// The tokenizer embedded in the GGUF file.
    Gguf,
}

/// The optional `overrides` section.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Overrides {
    /// Config values missing from `config.json` or GGUF metadata. A key that the
    /// checkpoint also states must agree with it.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub config: serde_json::Map<String, serde_json::Value>,
    /// Tensor names present in the files but not used (for example rotary caches).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ignore_tensors: Vec<String>,
}

/// A model manifest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Schema version; this build reads major [`SCHEMA_MAJOR`].
    pub schema_version: SemVer,
    /// Identity.
    pub model: ModelSection,
    /// Where it came from; `None` for a local import. Schema 1.1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<ModelSource>,
    /// Architecture.
    pub architecture: ArchRef,
    /// Weights.
    pub weights: WeightsSection,
    /// Tokenizer.
    pub tokenizer: TokenizerSection,
    /// The model's other files (config, tokenizer and its companions), for their
    /// checksums. Schema 1.1.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<Shard>,
    /// Context length limit, if lower than the architecture's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u64>,
    /// Free-form capability tags (`chat`, `tools`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    /// Quantization label for display (`Q4_K_M`, `bf16`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    /// Overrides.
    #[serde(default)]
    pub overrides: Overrides,
}

/// Largest manifest accepted.
const MAX_MANIFEST_BYTES: u64 = 1 << 20;

impl Manifest {
    /// Parses and checks a manifest.
    ///
    /// # Errors
    /// [`ModelError::Manifest`] for bad JSON, unknown fields, an unsupported schema
    /// major version, a bad id, or unsafe or inconsistent paths.
    pub fn parse(text: &str, path: &Path) -> Result<Self, ModelError> {
        let bad = |detail: String| ModelError::Manifest {
            path: path.to_owned(),
            detail,
        };
        // The version first: a newer manifest's fields would read as unknown ones.
        if let Ok(v) = serde_json::from_str::<Version>(text) {
            check_version(v.schema_version).map_err(bad)?;
        }
        let m: Self = serde_json::from_str(text).map_err(|e| bad(e.to_string()))?;
        if m.schema_version.minor == 0 && (m.source.is_some() || !m.files.is_empty()) {
            return Err(bad(format!(
                "schema_version {} has no source or files; they need 1.1",
                m.schema_version
            )));
        }
        if let Some(s) = &m.source
            && (s.provider.is_empty() || s.repository.is_empty() || s.revision.is_empty())
        {
            return Err(bad(
                "source needs a provider, a repository and a revision".into()
            ));
        }
        let id_ok = !m.model.id.is_empty()
            && m.model.id.len() <= 128
            && m.model
                .id
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"._-".contains(&b));
        if !id_ok {
            return Err(bad(format!(
                "model.id {:?} must be 1-128 of a-z, 0-9, '.', '_', '-'",
                m.model.id
            )));
        }
        if m.weights.shards.is_empty() {
            return Err(bad("weights.shards is empty".into()));
        }
        let mut paths: Vec<&str> = m.weights.shards.iter().map(|s| s.path.as_str()).collect();
        paths.extend(m.weights.config.as_deref());
        paths.extend(m.files.iter().map(|f| f.path.as_str()));
        if let TokenizerSection::Huggingface { path: p } = &m.tokenizer {
            paths.push(p);
        }
        for p in &paths {
            check_relative(p).map_err(bad)?;
        }
        for s in m.weights.shards.iter().chain(&m.files) {
            if let Some(h) = &s.sha256
                && (h.len() != 64
                    || !h
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
            {
                return Err(bad(format!(
                    "file {:?}: sha256 must be 64 lowercase hex digits",
                    s.path
                )));
            }
        }
        match m.weights.format {
            ManifestFormat::Gguf if m.weights.shards.len() != 1 => {
                return Err(bad("a GGUF model has exactly one shard".into()));
            }
            ManifestFormat::Gguf if m.weights.config.is_some() => {
                return Err(bad(
                    "a GGUF model takes its config from the GGUF metadata; remove weights.config"
                        .into(),
                ));
            }
            _ => {}
        }
        Ok(m)
    }

    /// Reads and parses `path`.
    ///
    /// # Errors
    /// [`ModelError::Io`] or [`ModelError::Manifest`].
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        let text = read_small(path, MAX_MANIFEST_BYTES)?;
        Self::parse(&text, path)
    }

    /// This manifest at [`SCHEMA_VERSION`]. Every step from an older minor adds optional
    /// fields, so only the version changes; a later minor that changes more does it here.
    /// What the fields hold comes from [`crate::fill_checksums`] and the registry.
    #[must_use]
    pub fn upgraded(mut self) -> Self {
        self.schema_version = SCHEMA_VERSION;
        self
    }

    /// Writes `manifest.json` in `dir` so that a crash leaves the old file or the new
    /// one, never part of either.
    ///
    /// # Errors
    /// [`ModelError::Io`], or [`ModelError::Manifest`] if it cannot be serialised.
    pub fn save(&self, dir: &Path) -> Result<(), ModelError> {
        use std::io::Write;
        let path = dir.join(crate::MANIFEST_FILE);
        let mut text = serde_json::to_string_pretty(self).map_err(|e| ModelError::Manifest {
            path: path.clone(),
            detail: e.to_string(),
        })?;
        text.push('\n');
        let tmp = dir.join(format!(".manifest.{}.tmp", std::process::id()));
        let io = |path: &Path| {
            let path = path.to_owned();
            move |source| ModelError::Io { path, source }
        };
        let written = std::fs::File::create(&tmp).and_then(|mut f| {
            f.write_all(text.as_bytes())?;
            f.sync_all()
        });
        if let Err(e) = written.and_then(|()| std::fs::rename(&tmp, &path)) {
            let _ = std::fs::remove_file(&tmp);
            return Err(io(&path)(e));
        }
        Ok(())
    }
}

/// Only the version, read before the rest.
#[derive(Deserialize)]
struct Version {
    schema_version: SemVer,
}

fn check_version(v: SemVer) -> Result<(), String> {
    if v.major != SCHEMA_MAJOR {
        return Err(format!(
            "schema_version {v} is not supported (this build reads {SCHEMA_MAJOR}.x)"
        ));
    }
    if v.minor > SCHEMA_MINOR {
        return Err(format!(
            "schema_version {v} is newer than this build, which reads up to \
             {SCHEMA_MAJOR}.{SCHEMA_MINOR}; update qubivex"
        ));
    }
    Ok(())
}

/// Paths inside a model directory: relative, no `..`, no absolute or drive paths.
fn check_relative(p: &str) -> Result<(), String> {
    let path = Path::new(p);
    let ok = !p.is_empty()
        && !p.contains('\\')
        && !p.contains(':')
        && path.is_relative()
        && path
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_)));
    if ok {
        Ok(())
    } else {
        Err(format!(
            "path {p:?} must be relative to the model directory, without '..'"
        ))
    }
}

/// Reads a small text file, refusing anything larger than `max`.
pub(crate) fn read_small(path: &Path, max: u64) -> Result<String, ModelError> {
    use std::io::Read;
    let io = |source| ModelError::Io {
        path: path.to_owned(),
        source,
    };
    let f = std::fs::File::open(path).map_err(io)?;
    let mut s = String::new();
    f.take(max + 1).read_to_string(&mut s).map_err(io)?;
    if s.len() as u64 > max {
        return Err(ModelError::Manifest {
            path: path.to_owned(),
            detail: format!("file is larger than {max} bytes"),
        });
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semver() {
        assert_eq!(
            "1.2.3".parse::<SemVer>().unwrap(),
            SemVer {
                major: 1,
                minor: 2,
                patch: 3
            }
        );
        for bad in ["1.2", "1.2.3.4", "01.2.3", "a.b.c", "", "1..3", "-1.0.0"] {
            assert!(bad.parse::<SemVer>().is_err(), "{bad}");
        }
    }

    #[test]
    fn relative_paths() {
        assert!(check_relative("model-00001.safetensors").is_ok());
        assert!(check_relative("sub/config.json").is_ok());
        for bad in [
            "",
            "/etc/passwd",
            "../x",
            "a/../b",
            "./x",
            "C:\\x",
            "a\\b",
            "c:x",
        ] {
            assert!(check_relative(bad).is_err(), "{bad}");
        }
    }
}
