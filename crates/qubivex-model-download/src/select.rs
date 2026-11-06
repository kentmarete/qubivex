//! Which files of a repository make up a model, and what they are called locally.
//!
//! Selection is an allow-list, so a repository's other files (PyTorch weights, ONNX
//! exports, images) are never fetched. Each selected file gets a flat local name that
//! is checked here, because repository paths are untrusted.

use std::collections::BTreeSet;

use serde_json::Value;

use crate::{DownloadError, RemoteFile, RepoListing};

use qubivex_model::COMPANION_FILES as COMPANIONS;
pub(crate) use qubivex_model::SHARD_INDEX_FILE as INDEX;

/// One file to fetch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    /// Its flat name in the model directory.
    pub name: String,
    /// The file in the repository.
    pub remote: RemoteFile,
}

fn base(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn in_root(f: &&RemoteFile) -> bool {
    !f.path.contains('/')
}

fn ends_with(path: &str, suffix: &str) -> bool {
    path.len() >= suffix.len()
        && path.as_bytes()[path.len() - suffix.len()..].eq_ignore_ascii_case(suffix.as_bytes())
}

/// The path must be relative, with only normal parts, and its name must be safe to
/// create: no separators, no control characters, not hidden, and not a name this crate
/// or the registry uses.
fn safe(path: &str) -> Result<&str, DownloadError> {
    let unsafe_path = || DownloadError::UnsafePath(path.to_owned());
    if path.is_empty()
        || path
            .chars()
            .any(|c| c == '\\' || c == ':' || c.is_control())
        || path
            .split('/')
            .any(|p| p.is_empty() || p == "." || p == "..")
    {
        return Err(unsafe_path());
    }
    let name = base(path);
    if name.starts_with('.') || ends_with(name, ".part") || name == "manifest.json" {
        return Err(unsafe_path());
    }
    Ok(name)
}

fn find<'l>(listing: &'l RepoListing, path: &str) -> Result<&'l RemoteFile, DownloadError> {
    listing
        .files
        .iter()
        .find(|f| f.path == path)
        .ok_or_else(|| DownloadError::FileNotFound {
            repository: listing.repository.clone(),
            file: path.to_owned(),
        })
}

/// The shard names an index file lists.
pub(crate) fn shards(index: &[u8]) -> Result<Vec<String>, DownloadError> {
    let bad = || DownloadError::Protocol(format!("{INDEX} has no usable weight_map"));
    let v: Value = serde_json::from_slice(index).map_err(|_| bad())?;
    let names: BTreeSet<&str> = v["weight_map"]
        .as_object()
        .ok_or_else(bad)?
        .values()
        .filter_map(Value::as_str)
        .collect();
    Ok(names.into_iter().map(str::to_owned).collect())
}

/// True if the listing's SafeTensors weights are named by an index, which then has to
/// be read before [`select`] runs.
pub(crate) fn needs_index(listing: &RepoListing, wanted: &[String]) -> bool {
    wanted.is_empty() && listing.files.iter().any(|f| f.path == INDEX)
}

fn with_companions<'l>(
    listing: &'l RepoListing,
    weights: Vec<&'l RemoteFile>,
    index: bool,
) -> Vec<&'l RemoteFile> {
    let mut out = weights;
    out.extend(
        listing
            .files
            .iter()
            .filter(in_root)
            .filter(|f| COMPANIONS.contains(&f.path.as_str()) || (index && f.path == INDEX)),
    );
    out
}

fn ambiguous(listing: &RepoListing, files: &[&RemoteFile]) -> DownloadError {
    DownloadError::Ambiguous {
        repository: listing.repository.clone(),
        candidates: files
            .iter()
            .map(|f| f.path.as_str())
            .collect::<Vec<_>>()
            .join(", "),
    }
}

/// The files to fetch. `wanted` names files exactly; empty means work it out. `index`
/// is the content of `model.safetensors.index.json`, when the repository has one.
///
/// With no names: SafeTensors at the repository root win, as they come with a
/// `config.json`; failing that, a single GGUF file. Several candidates are an error
/// rather than a guess, as a guess can be a 40 GiB download of the wrong quantisation.
///
/// # Errors
/// [`DownloadError::FileNotFound`], [`DownloadError::Ambiguous`],
/// [`DownloadError::NoWeights`], [`DownloadError::UnsafePath`].
pub fn select(
    listing: &RepoListing,
    wanted: &[String],
    index: Option<&[u8]>,
) -> Result<Vec<Selection>, DownloadError> {
    let chosen = if wanted.is_empty() {
        choose(listing, index)?
    } else {
        let named: Vec<&RemoteFile> = wanted
            .iter()
            .map(|w| find(listing, w))
            .collect::<Result<_, _>>()?;
        let ggufs: Vec<&RemoteFile> = named
            .iter()
            .copied()
            .filter(|f| ends_with(&f.path, ".gguf"))
            .collect();
        match ggufs.len() {
            // A SafeTensors file picked by name still needs its config and tokenizer, but
            // not the index, which may name other shards.
            0 if named.iter().any(|f| ends_with(&f.path, ".safetensors")) => {
                with_companions(listing, named, false)
            }
            0 | 1 => named,
            _ => return Err(ambiguous(listing, &ggufs)),
        }
    };
    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(chosen.len());
    for f in chosen {
        let name = safe(&f.path)?.to_owned();
        if !seen.insert(name.clone()) {
            continue;
        }
        out.push(Selection {
            name,
            remote: f.clone(),
        });
    }
    Ok(out)
}

fn choose<'l>(
    listing: &'l RepoListing,
    index: Option<&[u8]>,
) -> Result<Vec<&'l RemoteFile>, DownloadError> {
    if let Some(index) = index {
        let weights = shards(index)?
            .iter()
            .map(|s| find(listing, s))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(with_companions(listing, weights, true));
    }
    let safetensors: Vec<&RemoteFile> = listing
        .files
        .iter()
        .filter(in_root)
        .filter(|f| ends_with(&f.path, ".safetensors"))
        .collect();
    if let Some(one) = safetensors.iter().find(|f| f.path == "model.safetensors") {
        return Ok(with_companions(listing, vec![*one], false));
    }
    match safetensors.as_slice() {
        [one] => return Ok(with_companions(listing, vec![*one], false)),
        [] => {}
        many => return Err(ambiguous(listing, many)),
    }
    let ggufs: Vec<&RemoteFile> = listing
        .files
        .iter()
        .filter(|f| ends_with(&f.path, ".gguf"))
        .collect();
    match ggufs.as_slice() {
        [one] => Ok(vec![*one]),
        [] => Err(DownloadError::NoWeights(listing.repository.clone())),
        many => Err(ambiguous(listing, many)),
    }
}

/// The default model id: the repository's name, or for GGUF the file's stem, so that
/// two quantisations of one repository are two models.
pub(crate) fn default_id(repository: &str, selection: &[Selection]) -> String {
    let gguf = selection.iter().find(|s| ends_with(&s.name, ".gguf"));
    let name = gguf.map_or_else(
        || base(repository),
        |s| &s.name[..s.name.len() - ".gguf".len()],
    );
    qubivex_model::sanitise_id(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn listing(paths: &[&str]) -> RepoListing {
        RepoListing {
            repository: "o/r".to_owned(),
            revision: "0".repeat(40),
            files: paths
                .iter()
                .map(|p| RemoteFile {
                    path: (*p).to_owned(),
                    size: 1,
                    checksum: None,
                })
                .collect(),
        }
    }

    fn names(l: &RepoListing, wanted: &[&str], index: Option<&[u8]>) -> Vec<String> {
        let wanted: Vec<String> = wanted.iter().map(|s| (*s).to_owned()).collect();
        select(l, &wanted, index)
            .unwrap_or_else(|e| panic!("{e}"))
            .into_iter()
            .map(|s| s.name)
            .collect()
    }

    #[test]
    fn safetensors_take_only_what_a_model_needs() {
        let l = listing(&[
            ".gitattributes",
            "README.md",
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "onnx/model.onnx",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "LICENSE",
        ]);
        assert_eq!(
            names(&l, &[], None),
            [
                "model.safetensors",
                "config.json",
                "tokenizer.json",
                "LICENSE"
            ]
        );
    }

    #[test]
    fn an_index_names_the_shards() {
        let l = listing(&[
            "config.json",
            INDEX,
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "consolidated.safetensors",
        ]);
        let index = br#"{"weight_map":{"a":"model-00001-of-00002.safetensors",
            "b":"model-00002-of-00002.safetensors","c":"model-00001-of-00002.safetensors"}}"#;
        assert!(needs_index(&l, &[]));
        assert_eq!(
            names(&l, &[], Some(index)),
            [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "config.json",
                INDEX
            ]
        );
    }

    #[test]
    fn an_index_cannot_name_a_file_outside_the_listing() {
        let l = listing(&["config.json", INDEX]);
        let index = br#"{"weight_map":{"a":"../../etc/passwd"}}"#;
        assert!(matches!(
            select(&l, &[], Some(index)),
            Err(DownloadError::FileNotFound { .. })
        ));
    }

    #[test]
    fn several_ggufs_are_a_question_not_a_guess() {
        let l = listing(&["a-Q4_K_M.gguf", "sub/a-Q8_0.gguf", "README.md"]);
        assert!(matches!(
            select(&l, &[], None),
            Err(DownloadError::Ambiguous { .. })
        ));
        assert_eq!(names(&l, &["sub/a-Q8_0.gguf"], None), ["a-Q8_0.gguf"]);
        assert!(matches!(
            select(
                &l,
                &["a-Q4_K_M.gguf".to_owned(), "sub/a-Q8_0.gguf".to_owned()],
                None
            ),
            Err(DownloadError::Ambiguous { .. })
        ));
    }

    #[test]
    fn no_weights_and_unknown_names_are_refused() {
        let l = listing(&["README.md", "pytorch_model.bin"]);
        assert!(matches!(
            select(&l, &[], None),
            Err(DownloadError::NoWeights(_))
        ));
        assert!(matches!(
            select(&l, &["x.gguf".to_owned()], None),
            Err(DownloadError::FileNotFound { .. })
        ));
    }

    #[test]
    fn hostile_paths_are_refused() {
        for bad in [
            "../x.gguf",
            "a/../../x.gguf",
            "/etc/x.gguf",
            "a//x.gguf",
            "a\\..\\x.gguf",
            "C:x.gguf",
            ".hidden.gguf",
            "x\n.gguf",
            "x.gguf.part",
            "manifest.json",
        ] {
            let l = listing(&[bad]);
            assert!(
                matches!(
                    select(&l, &[bad.to_owned()], None),
                    Err(DownloadError::UnsafePath(_))
                ),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn ids_separate_quantisations() {
        let l = listing(&["sub/Model-Q4_K_M.gguf"]);
        let s = select(&l, &["sub/Model-Q4_K_M.gguf".to_owned()], None)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(default_id("o/Repo-GGUF", &s), "model-q4_k_m");
        let l = listing(&["model.safetensors"]);
        let s = select(&l, &[], None).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(default_id("Acme/Model-0.6B", &s), "model-0.6b");
    }
}
