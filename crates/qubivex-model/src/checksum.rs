//! SHA-256 of a model's files, recorded in the manifest and checked by
//! `qubivex models verify`. `experts.qvxsum` is the finer check: it locates damage to
//! one expert, and it is what the engine uses on reads. This one covers every file,
//! including the tokenizer and config, against what was recorded when the model arrived.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::ModelError;
use crate::manifest::{Manifest, Shard, TokenizerSection};

/// Files that travel with a model's weights, when there are any. A pull fetches them
/// and the manifest records their checksums.
pub const COMPANION_FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "NOTICE",
];

/// The SafeTensors shard index, which the importer reads.
pub const SHARD_INDEX_FILE: &str = "model.safetensors.index.json";

/// Read size. Large enough that hashing is limited by the disk, not by system calls.
const BLOCK: usize = 1 << 20;

/// The SHA-256 of a file, in lowercase hex.
///
/// # Errors
/// [`ModelError::Io`].
pub fn sha256_file(path: &Path) -> Result<String, ModelError> {
    hash(path).map_err(|source| ModelError::Io {
        path: path.to_owned(),
        source,
    })
}

fn hash(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut block = vec![0u8; BLOCK];
    loop {
        match file.read(&mut block)? {
            0 => break,
            n => hasher.update(&block[..n]),
        }
    }
    let mut out = String::with_capacity(64);
    for b in hasher.finalize() {
        out.push(char::from(HEX[usize::from(b >> 4)]));
        out.push(char::from(HEX[usize::from(b & 0xf)]));
    }
    Ok(out)
}

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Fills in what `manifest` lacks: a size and a checksum for every shard, and an entry in
/// [`Manifest::files`] for each file beside the weights that is present.
///
/// A checksum already there stays, so this never hides a file that has changed since it
/// was recorded. `known` gives checksums already verified, by path, so a download's large
/// files are not read a second time.
///
/// # Errors
/// [`ModelError::Io`] if a file cannot be read.
pub fn fill_checksums(
    manifest: &mut Manifest,
    dir: &Path,
    known: &BTreeMap<String, String>,
) -> Result<(), ModelError> {
    let mut others: Vec<String> = COMPANION_FILES
        .iter()
        .chain(&[SHARD_INDEX_FILE])
        .map(|&f| f.to_owned())
        .collect();
    others.extend(manifest.weights.config.clone());
    if let TokenizerSection::Huggingface { path } = &manifest.tokenizer {
        others.push(path.clone());
    }
    others.sort();
    others.dedup();
    for path in others {
        let listed = manifest.files.iter().any(|f| f.path == path);
        if !listed && dir.join(&path).is_file() {
            manifest.files.push(Shard {
                path,
                size: None,
                sha256: None,
            });
        }
    }
    for f in manifest
        .weights
        .shards
        .iter_mut()
        .chain(&mut manifest.files)
    {
        let full = dir.join(&f.path);
        if f.size.is_none() {
            let size = std::fs::metadata(&full).map_err(|source| ModelError::Io {
                path: full.clone(),
                source,
            })?;
            f.size = Some(size.len());
        }
        if f.sha256.is_none() {
            f.sha256 = Some(match known.get(&f.path) {
                Some(h) => h.clone(),
                None => sha256_file(&full)?,
            });
        }
    }
    Ok(())
}

/// A file that does not match its manifest entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileProblem {
    /// Relative to the model directory.
    pub path: String,
    /// What is wrong.
    pub detail: String,
}

/// What [`verify_files`] found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FileCheck {
    /// Files that match their checksums.
    pub matched: usize,
    /// Files with no recorded checksum, which were not read.
    pub unrecorded: usize,
    /// Files that do not match.
    pub problems: Vec<FileProblem>,
}

/// Checks every file with a recorded checksum. A missing or changed file is a problem
/// to report, not an error, so one report covers all of them.
#[must_use]
pub fn verify_files(manifest: &Manifest, dir: &Path) -> FileCheck {
    let mut out = FileCheck::default();
    for f in manifest.weights.shards.iter().chain(&manifest.files) {
        let Some(want) = &f.sha256 else {
            out.unrecorded += 1;
            continue;
        };
        let problem = |detail: String| FileProblem {
            path: f.path.clone(),
            detail,
        };
        match hash(&dir.join(&f.path)) {
            Ok(got) if got == *want => out.matched += 1,
            Ok(got) => out
                .problems
                .push(problem(format!("sha256 is {got}, expected {want}"))),
            Err(e) => out.problems.push(problem(e.to_string())),
        }
    }
    out
}
