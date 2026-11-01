//! Guard against model-name literals in shared code (design invariant 3).
//!
//! Scans `crates/` and `apps/` for identifiers or words that start with a known model
//! family name. Paths under a `tests`, `fixtures` or `benches` directory are exempt,
//! as are `models/`, `docs/` and `xtask/`, which this scan does not visit.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Lowercase prefixes of model family names. A token matches if it starts with one.
const MODEL_NAMES: &[&str] = &[
    "olmoe", "olmo", "qwen", "deepseek", "mixtral", "mistral", "llama", "gemma", "glm", "phi3",
    "phi4", "granite", "jamba", "dbrx", "grok", "kimi",
];

const SCANNED_ROOTS: &[&str] = &["crates", "apps"];
const EXEMPT_DIRS: &[&str] = &["tests", "fixtures", "benches", "target"];
const SCANNED_EXTENSIONS: &[&str] = &["rs", "toml", "json"];

/// Returns `path:line: token` for every match.
pub(crate) fn scan(root: &Path) -> io::Result<Vec<String>> {
    let mut files = Vec::new();
    for dir in SCANNED_ROOTS {
        let dir = root.join(dir);
        if dir.is_dir() {
            collect(&dir, &mut files)?;
        }
    }
    files.sort();
    let mut hits = Vec::new();
    for file in files {
        let text = fs::read_to_string(&file)?;
        let rel = file
            .strip_prefix(root)
            .unwrap_or(&file)
            .display()
            .to_string();
        for (n, line) in text.lines().enumerate() {
            if let Some(tok) = find_model_name(line) {
                hits.push(format!("{rel}:{}: {tok}", n + 1));
            }
        }
    }
    Ok(hits)
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        if entry.file_type()?.is_dir() {
            if !EXEMPT_DIRS.iter().any(|d| name == *d) {
                collect(&path, out)?;
            }
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| SCANNED_EXTENSIONS.contains(&e))
        {
            out.push(path);
        }
    }
    Ok(())
}

/// Returns the first token in `line` that starts with a model name.
///
/// Tokens are maximal runs of ASCII letters and digits, compared case-insensitively
/// from their start. So `OlmoeConfig`, `qwen3_moe` and `"deepseek-v2"` all match.
/// A name inside a camel-case identifier (`ExpertGlmFoo`) does not.
fn find_model_name(line: &str) -> Option<&str> {
    line.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .find(|t| {
            let lower = t.to_ascii_lowercase();
            MODEL_NAMES.iter().any(|m| lower.starts_with(m))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_names_in_identifiers_and_strings() {
        assert_eq!(find_model_name("struct OlmoeConfig {"), Some("OlmoeConfig"));
        assert_eq!(find_model_name(r#"if arch == "qwen3_moe""#), Some("qwen3"));
        assert_eq!(find_model_name("// DeepSeek-V2 uses MLA"), Some("DeepSeek"));
        assert_eq!(find_model_name("let glm4 = 1;"), Some("glm4"));
    }

    #[test]
    fn ignores_ordinary_words() {
        for line in [
            "let experts_per_token = 8;",
            "graphics philosophy alphabet",
            "use std::sync::Mutex;",
            "struct ExpertGlmFoo;", // the token starts with "expert", not a model name
        ] {
            assert_eq!(find_model_name(line), None, "{line}");
        }
    }

    #[test]
    fn prefix_match_is_deliberately_broad() {
        // False positives are cheap to rename; false negatives leak model names.
        assert_eq!(find_model_name("fn llamafy() {}"), Some("llamafy"));
    }

    #[test]
    fn scan_skips_exempt_directories() {
        let root = std::env::temp_dir().join(format!("qbx-names-{}", std::process::id()));
        let src = root.join("crates/demo/src");
        let tests = root.join("crates/demo/tests");
        fs::create_dir_all(&src).unwrap();
        fs::create_dir_all(&tests).unwrap();
        fs::write(
            src.join("lib.rs"),
            "fn ok() {}\nconst A: &str = \"mixtral\";\n",
        )
        .unwrap();
        fs::write(tests.join("t.rs"), "const A: &str = \"mixtral\";\n").unwrap();
        fs::write(src.join("notes.txt"), "mixtral\n").unwrap();
        let hits = scan(&root).unwrap();
        fs::remove_dir_all(&root).unwrap();
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert!(hits[0].ends_with("lib.rs:2: mixtral"), "{hits:?}");
    }
}
