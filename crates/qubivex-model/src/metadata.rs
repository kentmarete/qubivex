//! What the runtime trusts: manifest ∩ config ∩ tensor headers, cross-checked.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use qubivex_core::{ExpertIdx, LayerIdx};
use serde::Serialize;

use crate::manifest::{Manifest, ManifestFormat, read_small};
use crate::registry::{ArchConfig, ArchitectureRegistry, ConfigSource, TensorRole, TensorSchema};
use crate::{ModelError, TensorIndex};

/// The manifest file name inside a model directory.
pub const MANIFEST_FILE: &str = "manifest.json";
/// Largest `config.json` accepted.
const MAX_CONFIG_BYTES: u64 = 4 << 20;
/// Problems listed in one validation error before the rest are counted.
const MAX_LISTED: usize = 12;

/// Byte totals by what the tensors are for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ByteSummary {
    /// Every tensor.
    pub total: u64,
    /// Routed experts (streamable).
    pub routed_experts: u64,
    /// Everything else (resident: embeddings, attention, norms, routers, head).
    pub dense: u64,
    /// The largest single expert (all its parts), the unit the expert store moves.
    pub largest_expert: u64,
}

/// A validated model.
#[derive(Clone, Debug)]
pub struct ModelMetadata {
    /// The model directory.
    pub dir: PathBuf,
    /// The manifest.
    pub manifest: Manifest,
    /// The architecture's parsed config.
    pub config: ArchConfig,
    /// Every tensor in the weight files.
    pub tensors: TensorIndex,
    /// The role of every tensor the architecture uses, by name.
    pub roles: BTreeMap<String, TensorRole>,
}

impl ModelMetadata {
    /// Loads and validates the model in `dir` (which holds `manifest.json`).
    ///
    /// # Errors
    /// Any manifest, header, config or validation error. Validation errors list
    /// every missing, misshapen and unexpected tensor at once.
    pub fn load(dir: &Path, registry: &ArchitectureRegistry) -> Result<Self, ModelError> {
        let manifest = Manifest::load(&dir.join(MANIFEST_FILE))?;
        Self::load_with(dir, manifest, registry)
    }

    /// Like [`ModelMetadata::load`], with a manifest that is not (yet) on disk.
    ///
    /// # Errors
    /// As [`ModelMetadata::load`].
    pub fn load_with(
        dir: &Path,
        manifest: Manifest,
        registry: &ArchitectureRegistry,
    ) -> Result<Self, ModelError> {
        let tensors = index_files(dir, &manifest)?;
        let hf_config = read_config(dir, &manifest)?;
        let (config, roles) = check(&manifest, &tensors, hf_config.as_ref(), registry)?;
        Ok(Self {
            dir: dir.to_owned(),
            manifest,
            config,
            tensors,
            roles,
        })
    }

    /// Byte totals by role.
    #[must_use]
    pub fn bytes(&self) -> ByteSummary {
        let mut s = ByteSummary::default();
        let mut per_expert: BTreeMap<(LayerIdx, ExpertIdx), u64> = BTreeMap::new();
        for (name, role) in &self.roles {
            let Some(t) = self.tensors.get(name) else {
                continue;
            };
            let b = t.tensor.bytes;
            s.total = s.total.saturating_add(b);
            match *role {
                TensorRole::RoutedExpert { layer, expert, .. } => {
                    s.routed_experts = s.routed_experts.saturating_add(b);
                    let e = per_expert.entry((layer, expert)).or_insert(0);
                    *e = e.saturating_add(b);
                }
                TensorRole::RoutedExpertStack { layer, experts, .. } => {
                    s.routed_experts = s.routed_experts.saturating_add(b);
                    let slice = b / u64::from(experts.max(1));
                    for x in 0..experts {
                        let e = per_expert.entry((layer, ExpertIdx(x))).or_insert(0);
                        *e = e.saturating_add(slice);
                    }
                }
                _ => s.dense = s.dense.saturating_add(b),
            }
        }
        s.largest_expert = per_expert.values().copied().max().unwrap_or(0);
        s
    }
}

/// Indexes the manifest's shards, checking stated sizes.
pub(crate) fn index_files(dir: &Path, manifest: &Manifest) -> Result<TensorIndex, ModelError> {
    let mut tensors = TensorIndex::new();
    for s in &manifest.weights.shards {
        let path = dir.join(&s.path);
        match manifest.weights.format {
            ManifestFormat::Safetensors => {
                tensors.add_safetensors(&path)?;
            }
            ManifestFormat::Gguf => tensors.add_gguf(&path)?,
        }
        let actual = tensors.files().last().map_or(0, |f| f.size);
        if let Some(want) = s.size
            && want != actual
        {
            return Err(ModelError::Manifest {
                path: dir.join(MANIFEST_FILE),
                detail: format!(
                    "shard {:?} is {actual} bytes, the manifest says {want}",
                    s.path
                ),
            });
        }
    }
    Ok(tensors)
}

/// `config.json` for a SafeTensors model; `None` for GGUF.
pub(crate) fn read_config(
    dir: &Path,
    manifest: &Manifest,
) -> Result<Option<serde_json::Value>, ModelError> {
    if manifest.weights.format == ManifestFormat::Gguf {
        return Ok(None);
    }
    let p = dir.join(manifest.weights.config.as_deref().unwrap_or("config.json"));
    let text = read_small(&p, MAX_CONFIG_BYTES)?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| ModelError::Config(format!("{}: {e}", p.display())))?;
    if !v.is_object() {
        return Err(ModelError::Config(format!(
            "{}: not a JSON object",
            p.display()
        )));
    }
    Ok(Some(v))
}

/// Parses the config and validates the tensors against the architecture's schema.
pub(crate) fn check(
    manifest: &Manifest,
    tensors: &TensorIndex,
    hf_config: Option<&serde_json::Value>,
    registry: &ArchitectureRegistry,
) -> Result<(ArchConfig, BTreeMap<String, TensorRole>), ModelError> {
    let factory = registry.resolve(&manifest.architecture)?;
    let config = factory.parse_config(&ConfigSource {
        manifest,
        hf_config,
        gguf: tensors.gguf_metadata(),
    })?;
    let schema = factory.tensor_schema(&config)?;
    let roles = validate(manifest, &schema, tensors)?;
    Ok((config, roles))
}

fn validate(
    manifest: &Manifest,
    schema: &TensorSchema,
    index: &TensorIndex,
) -> Result<BTreeMap<String, TensorRole>, ModelError> {
    let ignore: BTreeSet<&str> = manifest
        .overrides
        .ignore_tensors
        .iter()
        .map(String::as_str)
        .collect();
    let mut problems = Vec::new();
    let mut roles = BTreeMap::new();
    for spec in &schema.tensors {
        match index.get(&spec.name) {
            None => problems.push(format!(
                "missing tensor {:?} (expected shape {:?})",
                spec.name, spec.shape
            )),
            Some(t) if t.tensor.shape.dims() != spec.shape.as_slice() => problems.push(format!(
                "tensor {:?} has shape {:?}, expected {:?}",
                spec.name,
                t.tensor.shape.dims(),
                spec.shape
            )),
            Some(t) => {
                if let TensorRole::RoutedExpertStack { experts, .. } = spec.role
                    && t.tensor.bytes % u64::from(experts.max(1)) != 0
                {
                    problems.push(format!(
                        "tensor {:?} does not split into {experts} equal experts",
                        spec.name
                    ));
                }
                if roles.insert(spec.name.clone(), spec.role).is_some() {
                    problems.push(format!(
                        "tensor {:?} is expected twice; check tensor_names",
                        spec.name
                    ));
                }
            }
        }
    }
    for t in index.iter() {
        let n = t.tensor.name.as_str();
        if !roles.contains_key(n)
            && !ignore.contains(n)
            && !schema.tensors.iter().any(|s| s.name == n)
        {
            problems.push(format!(
                "unexpected tensor {n:?}; the config does not use it (list it in overrides.ignore_tensors if that is intended)"
            ));
        }
    }
    if problems.is_empty() {
        Ok(roles)
    } else {
        Err(ModelError::Validation {
            model: manifest.model.id.clone(),
            problems,
        })
    }
}

impl ModelError {
    /// For [`ModelError::Validation`]: the first problems, and how many more.
    pub(crate) fn listed(problems: &[String]) -> String {
        let mut s = problems
            .iter()
            .take(MAX_LISTED)
            .cloned()
            .collect::<Vec<_>>()
            .join("; ");
        if problems.len() > MAX_LISTED {
            s.push_str(&format!("; and {} more", problems.len() - MAX_LISTED));
        }
        s
    }
}
