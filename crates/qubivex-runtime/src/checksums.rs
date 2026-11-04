//! Per-expert checksums kept next to the model (ADR-006 amendment 4).
//!
//! `qubivex models verify` (and `models import --write`) reads every expert once and
//! records its [`checksum`] in [`SUMS_FILE`] in the model's directory, together with
//! each weight file's size and modification time. When the file is there and matches
//! the weight files, the engine verifies every expert read against it
//! ([`VerifiedSource`](qubivex_expert_stream::fault::VerifiedSource)), so corrupt bytes fail the read (and fail over to a mirror)
//! instead of reaching the forward pass. A file that no longer matches the weights is
//! reported as stale and not used: its sums could fail every read of a model that was
//! deliberately replaced.
//!
//! The format is plain text, one record per line:
//!
//! ```text
//! qubivex-expert-sums 1
//! file <size> <mtime ns> <path relative to the model directory>
//! expert <layer> <expert> <checksum, 16 hex digits>
//! ```

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use qubivex_core::{ExpertIdx, LayerIdx};
use qubivex_expert_stream::ExpertKey;
use qubivex_expert_stream::ExpertSource;
use qubivex_expert_stream::fault::checksum;
use qubivex_model::ModelMetadata;
use rayon::prelude::*;

use crate::{ExpertIo, ModelExpertSource, RuntimeError};

/// The checksum file's name, in the model's directory.
pub const SUMS_FILE: &str = "experts.qvxsum";

const HEADER: &str = "qubivex-expert-sums 1";

/// A weight file as the checksums saw it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Stamp {
    path: PathBuf,
    size: u64,
    mtime_ns: u128,
}

fn stamp(meta: &ModelMetadata, path: &Path) -> Result<Stamp, RuntimeError> {
    let m = std::fs::metadata(path)
        .map_err(|e| RuntimeError::Model(format!("cannot stat {}: {e}", path.display())))?;
    let mtime_ns = m
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    let rel = path.strip_prefix(&meta.dir).unwrap_or(path).to_path_buf();
    Ok(Stamp {
        path: rel,
        size: m.len(),
        mtime_ns,
    })
}

fn stamps(meta: &ModelMetadata) -> Result<Vec<Stamp>, RuntimeError> {
    meta.tensors
        .files()
        .iter()
        .map(|f| stamp(meta, &f.path))
        .collect()
}

/// Every expert's checksum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertSums {
    sums: HashMap<ExpertKey, u64>,
    files: Vec<Stamp>,
}

/// What [`ExpertSums::load`] found.
#[derive(Debug)]
pub enum SumsState {
    /// No checksum file.
    Missing,
    /// A checksum file that does not match the weight files now (why).
    Stale(String),
    /// Checksums for these weight files.
    Fresh(ExpertSums),
}

/// An expert whose bytes differ from its recorded checksum ([`ExpertSums::check`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mismatch {
    /// The expert.
    pub key: ExpertKey,
    /// What was wrong.
    pub detail: String,
}

impl ExpertSums {
    /// Reads every expert of the model once, on `threads` threads (0: all CPUs), and
    /// records its checksum.
    ///
    /// # Errors
    /// I/O errors reading the weights.
    pub fn record(meta: &ModelMetadata, threads: usize) -> Result<Self, RuntimeError> {
        let files = stamps(meta)?;
        let src = ModelExpertSource::with_io(meta, ExpertIo::Buffered)?;
        let keys: Vec<ExpertKey> = src.keys().collect();
        let sums = pool(threads)?.install(|| {
            keys.par_iter()
                .map_init(Vec::new, |buf, &key| {
                    let n = usize::try_from(src.expert_size(key).map_err(store)?)
                        .map_err(|e| RuntimeError::Model(e.to_string()))?;
                    buf.clear();
                    buf.resize(n, 0);
                    src.read_expert(key, buf).map_err(store)?;
                    Ok((key, checksum(buf)))
                })
                .collect::<Result<HashMap<_, _>, RuntimeError>>()
        })?;
        Ok(Self { sums, files })
    }

    /// Re-reads every expert and compares it with its recorded checksum (an offline
    /// scrub). Returns the experts that differ or cannot be read.
    ///
    /// # Errors
    /// The weight files cannot be opened.
    pub fn check(
        &self,
        meta: &ModelMetadata,
        threads: usize,
    ) -> Result<Vec<Mismatch>, RuntimeError> {
        let src = ModelExpertSource::with_io(meta, ExpertIo::Buffered)?;
        let mut keys: Vec<ExpertKey> = src.keys().collect();
        keys.sort_unstable();
        let mut out: Vec<Mismatch> = pool(threads)?.install(|| {
            keys.par_iter()
                .map_init(Vec::new, |buf, &key| {
                    let Some(&want) = self.sums.get(&key) else {
                        return Some(Mismatch {
                            key,
                            detail: "no recorded checksum".into(),
                        });
                    };
                    let n = src
                        .expert_size(key)
                        .ok()
                        .and_then(|n| usize::try_from(n).ok())?;
                    buf.clear();
                    buf.resize(n, 0);
                    if let Err(e) = src.read_expert(key, buf) {
                        return Some(Mismatch {
                            key,
                            detail: e.to_string(),
                        });
                    }
                    let got = checksum(buf);
                    (got != want).then(|| Mismatch {
                        key,
                        detail: format!("read {got:016x}, recorded {want:016x}"),
                    })
                })
                .flatten()
                .collect()
        });
        out.sort_by_key(|m| m.key);
        Ok(out)
    }

    /// Experts covered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sums.len()
    }

    /// Whether no expert is covered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sums.is_empty()
    }

    /// The checksums, for a [`VerifiedSource`](qubivex_expert_stream::fault::VerifiedSource).
    #[must_use]
    pub fn sums(&self) -> &HashMap<ExpertKey, u64> {
        &self.sums
    }

    /// Writes [`SUMS_FILE`] into `dir` (through a temporary file, so a crash never
    /// leaves half a file).
    ///
    /// # Errors
    /// I/O errors.
    pub fn save(&self, dir: &Path) -> Result<PathBuf, RuntimeError> {
        let mut s = String::from(HEADER);
        s.push('\n');
        for f in &self.files {
            let _ = writeln!(s, "file {} {} {}", f.size, f.mtime_ns, f.path.display());
        }
        let mut keys: Vec<&ExpertKey> = self.sums.keys().collect();
        keys.sort_unstable();
        for k in keys {
            let _ = writeln!(s, "expert {} {} {:016x}", k.0.0, k.1.0, self.sums[k]);
        }
        let dest = dir.join(SUMS_FILE);
        let tmp = dir.join(format!("{SUMS_FILE}.tmp"));
        let io = |e: std::io::Error| {
            RuntimeError::Model(format!("cannot write {}: {e}", dest.display()))
        };
        std::fs::write(&tmp, s).map_err(io)?;
        std::fs::rename(&tmp, &dest).map_err(io)?;
        Ok(dest)
    }

    /// Reads the model's [`SUMS_FILE`] and checks it against the weight files now.
    ///
    /// # Errors
    /// A checksum file that cannot be read or parsed.
    pub fn load(meta: &ModelMetadata) -> Result<SumsState, RuntimeError> {
        let path = meta.dir.join(SUMS_FILE);
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(SumsState::Missing),
            Err(e) => {
                return Err(RuntimeError::Model(format!(
                    "cannot read {}: {e}",
                    path.display()
                )));
            }
        };
        let parsed =
            parse(&text).map_err(|e| RuntimeError::Model(format!("{}: {e}", path.display())))?;
        let now = stamps(meta)?;
        if now != parsed.files {
            let changed = now.iter().find(|f| !parsed.files.contains(f)).map_or_else(
                || "the set of weight files".to_owned(),
                |f| f.path.display().to_string(),
            );
            return Ok(SumsState::Stale(format!(
                "{changed} changed after {SUMS_FILE} was written"
            )));
        }
        Ok(SumsState::Fresh(parsed))
    }
}

fn parse(text: &str) -> Result<ExpertSums, String> {
    let mut lines = text.lines().enumerate();
    match lines.next() {
        Some((_, HEADER)) => {}
        other => return Err(format!("not a checksum file (first line {other:?})")),
    }
    let mut files = Vec::new();
    let mut sums = HashMap::new();
    for (i, line) in lines {
        let bad = || format!("line {}: cannot parse {line:?}", i + 1);
        let mut w = line.splitn(4, ' ');
        match w.next() {
            Some("file") => {
                let size = w.next().and_then(|v| v.parse().ok()).ok_or_else(bad)?;
                let mtime_ns = w.next().and_then(|v| v.parse().ok()).ok_or_else(bad)?;
                let path = PathBuf::from(w.next().ok_or_else(bad)?);
                files.push(Stamp {
                    path,
                    size,
                    mtime_ns,
                });
            }
            Some("expert") => {
                let layer = w.next().and_then(|v| v.parse().ok()).ok_or_else(bad)?;
                let expert = w.next().and_then(|v| v.parse().ok()).ok_or_else(bad)?;
                let sum = w
                    .next()
                    .and_then(|v| u64::from_str_radix(v, 16).ok())
                    .ok_or_else(bad)?;
                sums.insert((LayerIdx(layer), ExpertIdx(expert)), sum);
            }
            Some("") => {}
            _ => return Err(bad()),
        }
    }
    Ok(ExpertSums { sums, files })
}

fn pool(threads: usize) -> Result<rayon::ThreadPool, RuntimeError> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("qubivex-sum-{i}"))
        .build()
        .map_err(|e| RuntimeError::Model(format!("cannot start checksum threads: {e}")))
}

fn store(e: qubivex_expert_stream::StoreError) -> RuntimeError {
    RuntimeError::Model(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_text_format_round_trips() {
        let s = ExpertSums {
            sums: [
                ((LayerIdx(0), ExpertIdx(3)), 0xdead_beef_u64),
                ((LayerIdx(2), ExpertIdx(0)), u64::MAX),
            ]
            .into_iter()
            .collect(),
            files: vec![Stamp {
                path: PathBuf::from("sub dir/model.gguf"),
                size: 42,
                mtime_ns: 1_700_000_000_123_456_789,
            }],
        };
        let mut text = String::from(HEADER);
        text.push('\n');
        text.push_str("file 42 1700000000123456789 sub dir/model.gguf\n");
        text.push_str("expert 0 3 00000000deadbeef\nexpert 2 0 ffffffffffffffff\n");
        assert_eq!(parse(&text).unwrap(), s);
        assert!(parse("something else\n").is_err());
        assert!(parse(&format!("{HEADER}\nexpert 0 x 00\n")).is_err());
    }
}
