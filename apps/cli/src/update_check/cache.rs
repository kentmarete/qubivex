//! `~/.qubivex/update-check.json`: when the latest version was last looked up, where,
//! and what it was. Anything unreadable counts as no cache.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// The cache's file name in `~/.qubivex`.
pub(crate) const FILE: &str = "update-check.json";

/// What the last lookup found.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Cache {
    /// Unix time of the last lookup, successful or not.
    pub(crate) checked_at: u64,
    /// The URL looked up; a cache for another branch or repository does not count.
    pub(crate) source: String,
    /// The latest version found, if a lookup has succeeded.
    pub(crate) latest: Option<String>,
}

impl Cache {
    /// Whether the cache was made from `source` less than `ttl_secs` before `now`. A
    /// time in the future (a clock moved back) counts as stale.
    pub(crate) fn is_fresh(&self, source: &str, now: u64, ttl_secs: u64) -> bool {
        self.source == source && self.checked_at <= now && now - self.checked_at < ttl_secs
    }
}

/// The cache at `path`, or `None` if it is missing or not a valid cache.
pub(crate) fn load(path: &Path) -> Option<Cache> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

/// Writes `cache` to `path` through a temporary file and a rename, so a reader never
/// sees half a file.
pub(crate) fn save(path: &Path, cache: &Cache) -> std::io::Result<()> {
    let text = serde_json::to_string(cache).map_err(std::io::Error::other)?;
    let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
    fs::write(&tmp, text + "\n")?;
    fs::rename(&tmp, path).inspect_err(|_| {
        let _ = fs::remove_file(&tmp);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("qubivex-cache-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap_or_else(|e| panic!("{e}"));
        d
    }

    #[test]
    fn round_trips() {
        let d = dir("round");
        let p = d.join(FILE);
        assert_eq!(load(&p), None);
        let c = Cache {
            checked_at: 7,
            source: "u".into(),
            latest: Some("0.1.2".into()),
        };
        save(&p, &c).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(load(&p), Some(c));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn corrupted_or_foreign_files_count_as_missing() {
        let d = dir("corrupt");
        let p = d.join(FILE);
        for junk in [
            &b""[..],
            b"{",
            b"null",
            b"[1,2]",
            b"\xff\xfe",
            b"{\"checked_at\":\"x\"}",
        ] {
            fs::write(&p, junk).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(load(&p), None, "{junk:?}");
        }
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn freshness_expires_after_the_ttl() {
        let c = Cache {
            checked_at: 1000,
            source: "a".into(),
            latest: None,
        };
        assert!(c.is_fresh("a", 1000, 10));
        assert!(c.is_fresh("a", 1009, 10));
        assert!(!c.is_fresh("a", 1010, 10));
        assert!(!c.is_fresh("b", 1001, 10), "another source");
        assert!(!c.is_fresh("a", 999, 10), "a clock moved back");
    }
}
