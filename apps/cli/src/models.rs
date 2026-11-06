//! `qubivex models`: list, inspect, import, migrate and verify models.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use qubivex_core::Bytes;
use qubivex_model::{
    ArchitectureRegistry, Discovered, ImportOptions, MANIFEST_FILE, Manifest, ModelMetadata,
    ModelSource, TensorRole, discover, fill_checksums, find, import, verify_files,
};
use serde_json::{Value, json};

use qubivex_model_registry::{REGISTRY_FILE, Registry};
use qubivex_runtime::checksums::{ExpertSums, SumsState};

use crate::chain;

/// Model directories: `--dir` values, else `QUBIVEX_MODELS_DIRS` (`:`-separated), else
/// `~/.qubivex/models`.
pub(crate) fn roots(dirs: &[PathBuf]) -> Vec<PathBuf> {
    if !dirs.is_empty() {
        return dirs.to_vec();
    }
    if let Ok(v) = std::env::var("QUBIVEX_MODELS_DIRS") {
        let v: Vec<PathBuf> = v
            .split(':')
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect();
        if !v.is_empty() {
            return v;
        }
    }
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"));
    home.map_or_else(
        || vec![PathBuf::from("models")],
        |h| vec![Path::new(&h).join(".qubivex").join("models")],
    )
}

fn summary(m: &ModelMetadata) -> Value {
    let d = &m.config.dims;
    let b = m.bytes();
    json!({
        "id": m.manifest.model.id,
        "name": m.manifest.model.name,
        "dir": m.dir,
        "architecture": format!("{} {}", m.manifest.architecture.id, m.manifest.architecture.version),
        "format": m.manifest.weights.format,
        "quantization": m.manifest.quantization,
        "source": m.manifest.source,
        "layers": d.layers,
        "hidden": d.hidden,
        "vocab": d.vocab,
        "experts": d.experts,
        "experts_per_token": d.experts_per_token,
        "max_context": m.manifest.context_length.or(d.max_context),
        "bytes": b,
    })
}

pub(crate) fn list(dirs: &[PathBuf], json_out: bool) -> Result<String, String> {
    let roots = roots(dirs);
    let reg = ArchitectureRegistry::with_builtins();
    let found = discover(&roots, &reg);
    if json_out {
        let v: Vec<Value> = found
            .iter()
            .map(|d| match &d.model {
                Ok(m) => json!({ "status": "ok", "model": summary(m) }),
                Err(e) => {
                    json!({ "status": "error", "dir": d.dir, "id": d.id(), "error": chain(e) })
                }
            })
            .collect();
        return crate::to_json(&v);
    }
    if found.is_empty() {
        return Ok(format!(
            "no models under {} (import one with `qubivex models import <dir|file.gguf> --write`)\n",
            roots
                .iter()
                .map(|r| r.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let rows: Vec<[String; 6]> = found.iter().map(row).collect();
    let header = ["ID", "FORMAT", "LAYERS", "EXPERTS", "SIZE", "STATUS"].map(str::to_owned);
    let mut w = [0usize; 6];
    for r in std::iter::once(&header).chain(&rows) {
        for (i, c) in r.iter().enumerate() {
            w[i] = w[i].max(c.len());
        }
    }
    let line = |r: &[String; 6]| {
        let mut s = String::new();
        for (i, c) in r.iter().enumerate() {
            if i + 1 == r.len() {
                s.push_str(c);
            } else {
                s.push_str(&format!("{c:<width$}  ", width = w[i]));
            }
        }
        s.trim_end().to_owned()
    };
    let mut out: Vec<String> = std::iter::once(&header).chain(&rows).map(line).collect();
    out.push(String::new());
    Ok(out.join("\n"))
}

fn row(d: &Discovered) -> [String; 6] {
    match &d.model {
        Ok(m) => {
            let dims = &m.config.dims;
            [
                m.manifest.model.id.clone(),
                format!(
                    "{}{}",
                    serde_json::to_value(m.manifest.weights.format)
                        .ok()
                        .and_then(|v| v.as_str().map(str::to_owned))
                        .unwrap_or_default(),
                    m.manifest
                        .quantization
                        .as_ref()
                        .map(|q| format!(" {q}"))
                        .unwrap_or_default()
                ),
                dims.layers.to_string(),
                format!("{} (top {})", dims.experts, dims.experts_per_token),
                Bytes(m.bytes().total).to_string(),
                "ok".to_owned(),
            ]
        }
        Err(e) => [
            d.id().unwrap_or_else(|| d.dir.display().to_string()),
            "-".into(),
            "-".into(),
            "-".into(),
            "-".into(),
            format!("error: {}", chain(e)),
        ],
    }
}

/// `huggingface Org/Name @ <revision>`, or what stands for a local model.
pub(crate) fn source_line(s: Option<&ModelSource>) -> String {
    s.map_or_else(
        || "local".to_owned(),
        |s| format!("{} {} @ {}", s.provider, s.repository, s.revision),
    )
}

pub(crate) fn inspect(
    what: &str,
    dirs: &[PathBuf],
    json_out: bool,
    tensors: bool,
) -> Result<String, String> {
    let reg = ArchitectureRegistry::with_builtins();
    let m = find(what, &roots(dirs), &reg).map_err(|e| chain(&e))?;
    if json_out {
        let mut v = summary(&m);
        v["manifest"] = serde_json::to_value(&m.manifest).map_err(|e| e.to_string())?;
        v["config"] = m.config.describe.clone();
        v["files"] = serde_json::to_value(m.tensors.files()).map_err(|e| e.to_string())?;
        if tensors {
            v["tensors"] = Value::Array(
                m.tensors
                    .iter()
                    .map(|t| json!({ "tensor": t, "role": m.roles.get(&t.tensor.name) }))
                    .collect(),
            );
        }
        return crate::to_json(&v);
    }
    let d = &m.config.dims;
    let b = m.bytes();
    let mut o = vec![
        format!(
            "Model    {} ({})",
            m.manifest.model.id, m.manifest.model.name
        ),
        format!("Dir      {}", m.dir.display()),
        format!("Source   {}", source_line(m.manifest.source.as_ref())),
        format!(
            "Arch     {} {}",
            m.manifest.architecture.id, m.manifest.architecture.version
        ),
        format!(
            "Shape    {} layers, hidden {}, vocab {}, {} experts (top {}){}",
            d.layers,
            d.hidden,
            d.vocab,
            d.experts,
            d.experts_per_token,
            m.manifest
                .context_length
                .or(d.max_context)
                .map(|c| format!(", context {c}"))
                .unwrap_or_default()
        ),
        format!(
            "Bytes    {} total: {} routed experts, {} dense; largest expert {}",
            Bytes(b.total),
            Bytes(b.routed_experts),
            Bytes(b.dense),
            Bytes(b.largest_expert)
        ),
    ];
    let dtypes: Vec<String> = m
        .tensors
        .bytes_by_dtype()
        .iter()
        .map(|(dt, n)| format!("{dt} {}", Bytes(*n)))
        .collect();
    o.push(format!("Dtypes   {}", dtypes.join(", ")));
    o.push("Files".to_owned());
    for f in m.tensors.files() {
        o.push(format!("  {}  {}", f.path.display(), Bytes(f.size)));
    }
    let mut kinds: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
    for (name, role) in &m.roles {
        let k = match role {
            TensorRole::Embedding => "embedding",
            TensorRole::OutputHead => "output head",
            TensorRole::FinalNorm | TensorRole::Norm { .. } => "norms",
            TensorRole::Attention { .. } => "attention",
            TensorRole::Router { .. } => "router",
            TensorRole::Ffn { .. } => "dense FFN",
            TensorRole::RoutedExpert { .. } | TensorRole::RoutedExpertStack { .. } => {
                "routed experts"
            }
        };
        let e = kinds.entry(k).or_insert((0, 0));
        e.0 += 1;
        e.1 =
            e.1.saturating_add(m.tensors.get(name).map_or(0, |t| t.tensor.bytes));
    }
    o.push(format!("Tensors  {}", m.tensors.len()));
    for (k, (n, bytes)) in &kinds {
        o.push(format!("  {k:<15} {n:>6}  {}", Bytes(*bytes)));
    }
    o.push(format!("Config   {}", m.config.describe));
    if tensors {
        o.push("Tensor list".to_owned());
        for t in m.tensors.iter() {
            o.push(format!(
                "  {}  {} {:?}  {}",
                t.tensor.name,
                t.tensor.dtype,
                t.tensor.shape.dims(),
                m.roles
                    .get(&t.tensor.name)
                    .and_then(|r| serde_json::to_value(r).ok())
                    .map_or_else(|| "ignored".to_owned(), |v| v.to_string())
            ));
        }
    }
    let mut t = o.join("\n");
    t.push('\n');
    Ok(t)
}

/// Parses `key=value`; the value is JSON when it parses as JSON, else a string.
pub(crate) fn parse_set(s: &str) -> Result<(String, Value), String> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| format!("--set {s:?}: expected KEY=VALUE"))?;
    let v = serde_json::from_str(v).unwrap_or_else(|_| Value::String(v.to_owned()));
    Ok((k.trim().to_owned(), v))
}

pub(crate) struct ImportArgs<'a> {
    pub(crate) path: &'a Path,
    pub(crate) id: Option<String>,
    pub(crate) name: Option<String>,
    pub(crate) set: &'a [String],
    pub(crate) write: bool,
    pub(crate) force: bool,
    pub(crate) checksums: bool,
}

pub(crate) fn import_cmd(a: &ImportArgs<'_>) -> Result<String, String> {
    let mut opts = ImportOptions {
        id: a.id.clone(),
        name: a.name.clone(),
        ..ImportOptions::default()
    };
    for s in a.set {
        let (k, v) = parse_set(s)?;
        opts.set.insert(k, v);
    }
    let reg = ArchitectureRegistry::with_builtins();
    let mut imp = import(a.path, &opts, &reg).map_err(|e| with_set_hint(&chain(&e)))?;
    let mut text = serde_json::to_string_pretty(&imp.manifest).map_err(|e| e.to_string())?;
    text.push('\n');
    for w in &imp.warnings {
        eprintln!("qubivex: warning: {w}");
    }
    if !a.write {
        return Ok(text);
    }
    let dest = imp.dir.join(MANIFEST_FILE);
    if dest.exists() && !a.force {
        return Err(format!(
            "{} exists; pass --force to replace it",
            dest.display()
        ));
    }
    if a.checksums {
        fill_checksums(&mut imp.manifest, &imp.dir, &BTreeMap::new()).map_err(|e| chain(&e))?;
    }
    imp.manifest.save(&imp.dir).map_err(|e| chain(&e))?;
    let mut out = format!(
        "wrote {} (model id {})\n",
        dest.display(),
        imp.manifest.model.id
    );
    if a.checksums {
        let m = ModelMetadata::load(&imp.dir, &reg).map_err(|e| chain(&e))?;
        // Checksums cover streamed experts; a dense model streams nothing.
        if m.bytes().routed_experts > 0 {
            out.push_str(&record_sums(&m, 0)?);
        }
    }
    Ok(out)
}

/// The source the registry beside `m` holds for it. Only a registry that already exists
/// is read, since a model can be given by a path outside any models directory.
fn registry_source(m: &ModelMetadata) -> Result<Option<ModelSource>, String> {
    let Some(root) = m.dir.parent().filter(|r| r.join(REGISTRY_FILE).is_file()) else {
        return Ok(None);
    };
    let reg = Registry::open(root).map_err(|e| chain(&e))?;
    let record = reg.get(&m.manifest.model.id).map_err(|e| chain(&e))?;
    Ok(record
        .filter(|r| same_file(&reg.dir_of(r), &m.dir))
        .and_then(|r| r.source))
}

fn same_file(a: &Path, b: &Path) -> bool {
    matches!((a.canonicalize(), b.canonicalize()), (Ok(a), Ok(b)) if a == b)
}

/// `qubivex models migrate`: the manifest at the current schema, printed or written.
/// What it adds comes from the registry and the files as they are now; a checksum already
/// recorded stays, so a file that changed still fails `models verify`.
pub(crate) fn migrate(
    what: &str,
    dirs: &[PathBuf],
    write: bool,
    checksums: bool,
) -> Result<String, String> {
    let reg = ArchitectureRegistry::with_builtins();
    let m = find(what, &roots(dirs), &reg).map_err(|e| chain(&e))?;
    let before = m.manifest.clone();
    let mut after = before.clone().upgraded();
    if after.source.is_none() {
        after.source = registry_source(&m)?;
    }
    if !write {
        return crate::to_json(&after);
    }
    if checksums {
        fill_checksums(&mut after, &m.dir, &BTreeMap::new()).map_err(|e| chain(&e))?;
    }
    if after == before {
        return Ok(format!(
            "{} is already at schema {}\n",
            m.manifest.model.id, after.schema_version
        ));
    }
    // Nothing reaches disk that would not load.
    ModelMetadata::load_with(&m.dir, after.clone(), &reg).map_err(|e| chain(&e))?;
    after.save(&m.dir).map_err(|e| chain(&e))?;
    Ok(format!(
        "migrated {} from schema {} to {}: source {}, {} of {} files with checksums\n",
        after.model.id,
        before.schema_version,
        after.schema_version,
        source_line(after.source.as_ref()),
        checksummed(&after),
        after.weights.shards.len() + after.files.len(),
    ))
}

fn checksummed(m: &Manifest) -> usize {
    m.weights
        .shards
        .iter()
        .chain(&m.files)
        .filter(|f| f.sha256.is_some())
        .count()
}

/// Replaces the manifest's sizes and checksums with those of the files as they are now.
fn record_files(m: &ModelMetadata) -> Result<String, String> {
    let mut manifest = m.manifest.clone().upgraded();
    for f in manifest
        .weights
        .shards
        .iter_mut()
        .chain(&mut manifest.files)
    {
        f.size = None;
        f.sha256 = None;
    }
    fill_checksums(&mut manifest, &m.dir, &BTreeMap::new()).map_err(|e| chain(&e))?;
    manifest.save(&m.dir).map_err(|e| chain(&e))?;
    Ok(format!(
        "recorded checksums of {} files in the manifest\n",
        checksummed(&manifest)
    ))
}

/// Checks each file against the manifest; returns a report line.
fn check_files(m: &ModelMetadata, what: &str) -> Result<String, String> {
    let t = std::time::Instant::now();
    let check = verify_files(&m.manifest, &m.dir);
    if !check.problems.is_empty() {
        let mut msg = format!(
            "{} of {} files do not match the manifest:",
            check.problems.len(),
            check.problems.len() + check.matched
        );
        for p in &check.problems {
            msg.push_str(&format!("\n  {}: {}", p.path, p.detail));
        }
        msg.push_str(&format!(
            "\nIf they were replaced on purpose, run `qubivex models verify {what} --record`"
        ));
        return Err(msg);
    }
    if check.matched == 0 {
        return Ok(format!(
            "no file checksums in the manifest; `qubivex models migrate {what} --write` adds them\n"
        ));
    }
    Ok(format!(
        "ok: {} files match their checksums ({:.1} s)\n",
        check.matched,
        t.elapsed().as_secs_f64()
    ))
}

/// Records and saves the model's per-expert checksums; returns a report line.
pub(crate) fn record_sums(m: &ModelMetadata, threads: usize) -> Result<String, String> {
    let t = std::time::Instant::now();
    let sums = ExpertSums::record(m, threads).map_err(|e| chain(&e))?;
    let path = sums.save(&m.dir).map_err(|e| chain(&e))?;
    Ok(format!(
        "recorded checksums of {} experts ({}) in {:.1} s: {}\n",
        sums.len(),
        Bytes(m.bytes().routed_experts),
        t.elapsed().as_secs_f64(),
        path.display()
    ))
}

fn dir_size(dir: &Path) -> u64 {
    std::fs::read_dir(dir).map_or(0, |entries| {
        entries
            .flatten()
            .filter_map(|e| Some((e.path(), e.file_type().ok()?, e.metadata().ok()?)))
            .map(|(p, t, m)| if t.is_dir() { dir_size(&p) } else { m.len() })
            .sum()
    })
}

/// `qubivex models remove`: deletes a model the registry of one of the model directories
/// knows, adopting what is on disk first. Without `yes`, only says what would go.
pub(crate) fn remove(id: &str, dirs: &[PathBuf], yes: bool) -> Result<String, String> {
    let arch = ArchitectureRegistry::with_builtins();
    let searched = roots(dirs);
    for root in &searched {
        let reg = Registry::open(root).map_err(|e| chain(&e))?;
        reg.reconcile(&arch).map_err(|e| chain(&e))?;
        let Some(record) = reg.get(id).map_err(|e| chain(&e))? else {
            continue;
        };
        let dir = reg.dir_of(&record);
        let size = Bytes(dir_size(&dir));
        if !yes {
            return Err(format!(
                "this would delete {} ({size}); pass --yes to delete it",
                dir.display()
            ));
        }
        reg.remove(id).map_err(|e| chain(&e))?;
        return Ok(format!("Removed {id} ({size}) from {}\n", dir.display()));
    }
    Err(format!(
        "no model {id:?} in {}",
        searched
            .iter()
            .map(|r| r.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// `qubivex models verify`: checks every file against the manifest, then every expert
/// against the recorded checksums, which it records when there are none. With `record`,
/// records both from the files as they are.
pub(crate) fn verify(
    what: &str,
    dirs: &[PathBuf],
    record: bool,
    threads: usize,
) -> Result<String, String> {
    let reg = ArchitectureRegistry::with_builtins();
    let m = find(what, &roots(dirs), &reg).map_err(|e| chain(&e))?;
    let dense = m.bytes().routed_experts == 0;
    if record {
        let mut out = record_files(&m)?;
        if !dense {
            out.push_str(&record_sums(&m, threads)?);
        }
        return Ok(out);
    }
    let files = check_files(&m, what)?;
    // Experts are all a model streams, so a dense model has none to check.
    if dense {
        return Ok(files);
    }
    let experts = expert_check(&m, what, threads)?;
    Ok(files + &experts)
}

fn expert_check(m: &ModelMetadata, what: &str, threads: usize) -> Result<String, String> {
    match ExpertSums::load(m).map_err(|e| chain(&e))? {
        SumsState::Missing => record_sums(m, threads),
        SumsState::Stale(why) => Err(format!(
            "the recorded checksums are stale: {why}. If the weights were replaced on \
             purpose, run `qubivex models verify {what} --record`"
        )),
        SumsState::Fresh(sums) => {
            let t = std::time::Instant::now();
            let bad = sums.check(m, threads).map_err(|e| chain(&e))?;
            if bad.is_empty() {
                return Ok(format!(
                    "ok: {} experts match their checksums ({:.1} s)\n",
                    sums.len(),
                    t.elapsed().as_secs_f64()
                ));
            }
            let mut msg = format!("{} of {} experts do not match:", bad.len(), sums.len());
            for b in bad.iter().take(20) {
                msg.push_str(&format!(
                    "\n  layer {} expert {}: {}",
                    b.key.0.0, b.key.1.0, b.detail
                ));
            }
            if bad.len() > 20 {
                msg.push_str(&format!("\n  ... and {} more", bad.len() - 20));
            }
            Err(msg)
        }
    }
}

/// At import time there is no manifest yet: point "set overrides.config.KEY" at `--set`.
pub(crate) fn with_set_hint(msg: &str) -> String {
    let Some((_, rest)) = msg.split_once("overrides.config.") else {
        return msg.to_owned();
    };
    let key: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if key.is_empty() {
        return msg.to_owned();
    }
    format!(
        "{msg}\n  hint: pass --set {key}=VALUE, with the value from the model's original config.json"
    )
}

pub(crate) fn architectures() -> String {
    let mut o: Vec<String> = ArchitectureRegistry::with_builtins()
        .list_supported()
        .into_iter()
        .map(|a| format!("{} {}  {}", a.id, a.version, a.description))
        .collect();
    o.push(String::new());
    o.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_hint_names_the_key() {
        let m = "config: GGUF metadata has no <architecture>.expert_weights_norm; set overrides.config.norm_topk_prob in the manifest";
        assert!(with_set_hint(m).ends_with("hint: pass --set norm_topk_prob=VALUE, with the value from the model's original config.json"));
        assert_eq!(with_set_hint("other error"), "other error");
    }

    #[test]
    fn set_values() {
        assert_eq!(
            parse_set("norm_topk_prob=true").unwrap(),
            ("norm_topk_prob".into(), json!(true))
        );
        assert_eq!(
            parse_set("rope_theta=1e6").unwrap(),
            ("rope_theta".into(), json!(1e6))
        );
        assert_eq!(parse_set("x=abc").unwrap(), ("x".into(), json!("abc")));
        assert!(parse_set("novalue").is_err());
    }
}
