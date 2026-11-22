//! Where the latest version comes from: the workspace `Cargo.toml` of the branch the
//! installation follows. That file is the project's one version source (`qubivex
//! --version` is compiled from it), and that branch is what `qubivex update` installs.

use std::time::Duration;

use super::version::Version;

/// The most of the manifest read; the real one is a few KiB.
const BODY_LIMIT: u64 = 256 * 1024;

/// The raw URL of `Cargo.toml` on `branch` of `repo`, for a GitHub repository given as
/// `https://github.com/O/R(.git)`, `ssh://git@github.com/O/R(.git)` or
/// `git@github.com:O/R(.git)`. `None` for any other host, or a branch name that would
/// need escaping in a URL.
pub(crate) fn manifest_url(repo: &str, branch: &str) -> Option<String> {
    let path = if let Some(rest) = repo.strip_prefix("git@github.com:") {
        rest
    } else {
        let rest = repo
            .strip_prefix("https://")
            .or_else(|| repo.strip_prefix("ssh://"))?;
        let (authority, path) = rest.split_once('/')?;
        // Drop any user or token before the host.
        let host = authority.rsplit('@').next()?;
        (host.eq_ignore_ascii_case("github.com")).then_some(path)?
    };
    let path = path.trim_end_matches('/');
    let path = path.strip_suffix(".git").unwrap_or(path);
    let (owner, name) = path.split_once('/')?;
    let safe = |s: &str, slash: bool| {
        !s.is_empty()
            && !s.starts_with('.')
            && s.bytes().all(|b| {
                b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.') || (slash && b == b'/')
            })
            && !s.contains("..")
    };
    (safe(owner, false) && safe(name, false) && safe(branch, true))
        .then(|| format!("https://raw.githubusercontent.com/{owner}/{name}/{branch}/Cargo.toml"))
}

/// Whether `url` may be fetched: https, or plain http to this machine (for tests and
/// local mirrors).
fn allowed(url: &str) -> bool {
    if url.starts_with("https://") {
        return true;
    }
    let Some(rest) = url.strip_prefix("http://") else {
        return false;
    };
    let host = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let host = host.rsplit_once(':').map_or(host, |(h, port)| {
        if port.bytes().all(|b| b.is_ascii_digit()) {
            h
        } else {
            host
        }
    });
    matches!(host, "127.0.0.1" | "localhost" | "[::1]")
}

/// The version in a `Cargo.toml`: `[workspace.package] version`, else `[package]
/// version`.
pub(crate) fn version_in_manifest(text: &str) -> Result<Version, String> {
    let table: toml::Table = toml::from_str(text).map_err(|_| "not a Cargo.toml".to_owned())?;
    let from = |t: Option<&toml::Value>| {
        t.and_then(|p| p.get("version"))
            .and_then(toml::Value::as_str)
            .map(str::to_owned)
    };
    let workspace = table.get("workspace").and_then(|w| w.get("package"));
    let text = from(workspace)
        .or_else(|| from(table.get("package")))
        .ok_or_else(|| "the Cargo.toml has no version".to_owned())?;
    Version::parse(&text).ok_or_else(|| format!("{text:?} is not a semantic version"))
}

/// Fetches `url` and reads the version from it, giving up after `timeout` overall.
pub(crate) fn fetch_latest(url: &str, timeout: Duration) -> Result<Version, String> {
    if !allowed(url) {
        return Err(format!("refusing to fetch {url}: not https"));
    }
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .user_agent(concat!("qubivex/", env!("CARGO_PKG_VERSION")))
        .build()
        .into();
    let mut response = agent.get(url).call().map_err(|e| e.to_string())?;
    let body = response
        .body_mut()
        .with_config()
        .limit(BODY_LIMIT)
        .read_to_string()
        .map_err(|e| e.to_string())?;
    version_in_manifest(&body)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAW: &str = "https://raw.githubusercontent.com/kentmarete/qubivex/main/Cargo.toml";

    #[test]
    fn github_urls_in_every_spelling() {
        for repo in [
            "https://github.com/kentmarete/qubivex.git",
            "https://github.com/kentmarete/qubivex",
            "https://github.com/kentmarete/qubivex/",
            "https://x-access-token:secret@github.com/kentmarete/qubivex.git",
            "ssh://git@github.com/kentmarete/qubivex.git",
            "git@github.com:kentmarete/qubivex.git",
        ] {
            assert_eq!(manifest_url(repo, "main").as_deref(), Some(RAW), "{repo}");
        }
        assert_eq!(
            manifest_url("https://github.com/o/r.git", "feature/x-1").as_deref(),
            Some("https://raw.githubusercontent.com/o/r/feature/x-1/Cargo.toml")
        );
    }

    #[test]
    fn other_hosts_and_odd_names_have_no_url() {
        for (repo, branch) in [
            ("https://gitlab.com/o/r.git", "main"),
            ("https://github.com.evil.example/o/r", "main"),
            ("/home/me/qubivex", "main"),
            ("file:///home/me/qubivex", "main"),
            ("https://github.com/o", "main"),
            ("https://github.com/o/r", "a b"),
            ("https://github.com/o/r", "../x"),
            ("https://github.com/o/r", "x?y"),
            ("https://github.com/o/r", ""),
        ] {
            assert_eq!(manifest_url(repo, branch), None, "{repo} {branch}");
        }
    }

    #[test]
    fn only_https_or_this_machine() {
        assert!(allowed(RAW));
        assert!(allowed("http://127.0.0.1:8080/Cargo.toml"));
        assert!(allowed("http://localhost/x"));
        assert!(!allowed("http://example.com/Cargo.toml"));
        assert!(!allowed("http://127.0.0.1.example.com/x"));
        assert!(!allowed("file:///etc/passwd"));
    }

    #[test]
    fn reads_the_workspace_version_then_the_package_version() {
        let ws = "[workspace.package]\nversion = \"0.1.2\"\n[package]\nversion = \"9.9.9\"\n";
        assert_eq!(
            version_in_manifest(ws).map(|v| v.to_string()),
            Ok("0.1.2".into())
        );
        let pkg = "[package]\nname = \"x\"\nversion = \"0.3.0-rc.1\"\n";
        assert_eq!(
            version_in_manifest(pkg).map(|v| v.to_string()),
            Ok("0.3.0-rc.1".into())
        );
    }

    #[test]
    fn the_real_workspace_manifest_has_the_compiled_version() {
        let text = include_str!("../../../../Cargo.toml");
        assert_eq!(
            version_in_manifest(text).map(|v| v.to_string()),
            Ok(qubivex_core::VERSION.to_owned())
        );
    }

    #[test]
    fn bad_manifests_are_errors() {
        for text in [
            "<html>404</html>",
            "[package]\nname = \"x\"\n",
            "[workspace.package]\nversion = \"latest\"\n",
            "[workspace.package]\nversion = 3\n",
        ] {
            assert!(version_in_manifest(text).is_err(), "{text}");
        }
    }
}
