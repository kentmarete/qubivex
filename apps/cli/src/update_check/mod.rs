//! The update notice: after a command, tell someone at a terminal that the branch their
//! installation follows has a newer version, and that `qubivex update` installs it.
//!
//! - The installed version is the compiled one (`qubivex_core::VERSION`, which
//!   `--version` prints). The latest is the workspace `Cargo.toml` version on the branch
//!   `install.sh` recorded in `~/.qubivex/install-receipt` ([`source`]), which is what
//!   `qubivex update` would install. Builds without a receipt (a checkout, `cargo
//!   install`) and tag-pinned installations get no notice: `qubivex update` would not
//!   move them.
//! - The lookup never runs in the command's own process: when the cache
//!   (`~/.qubivex/update-check.json`) is older than a day, the command starts
//!   `qubivex refresh-update-check` in the background and shows what the cache already
//!   holds. That process gives up after a few seconds, and failures only leave the cache
//!   as it was until the next day.
//! - `--version` is the exception ([`after_version`]): whoever runs it is asking which
//!   version they have, so it looks the latest up then and there (2 s at most), whatever
//!   the cache holds, and falls back to the cache when the lookup fails. A release made
//!   since the last daily lookup is therefore seen at once.
//! - The notice goes to stderr, after the command's output, and only when stdout and
//!   stderr are both terminals, the command is not one whose output must stay clean
//!   (`run`, `serve`, `bench`, `update`, `--json`), and neither
//!   `QUBIVEX_NO_UPDATE_CHECK` nor `CI` is set. `--version` keeps its one stdout line.

mod cache;
mod source;
mod version;

use std::ffi::OsStr;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cache::Cache;
use version::Version;

/// Set (to anything but `0`) to turn the update check off.
pub(crate) const DISABLE_ENV: &str = "QUBIVEX_NO_UPDATE_CHECK";
/// Replaces the looked-up URL (a mirror, or a test server on this machine).
pub(crate) const URL_ENV: &str = "QUBIVEX_UPDATE_CHECK_URL";
/// The hidden subcommand that refreshes the cache.
pub(crate) const REFRESH_COMMAND: &str = "refresh-update-check";
/// How long a lookup stays good.
const TTL_SECS: u64 = 24 * 60 * 60;
/// How long the background lookup may take in all.
const TIMEOUT: Duration = Duration::from_secs(3);
/// How long `--version` waits for its lookup before using the cache.
const VERSION_TIMEOUT: Duration = Duration::from_secs(2);

/// Whether an environment value turns something on: set, not empty, not `0` or `false`.
fn truthy(value: Option<&OsStr>) -> bool {
    value.is_some_and(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
}

/// Everything that decides whether the check runs at all.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Gate {
    /// The command finished and its output may be followed by a notice.
    pub(crate) eligible: bool,
    /// `QUBIVEX_NO_UPDATE_CHECK` is set.
    pub(crate) disabled: bool,
    /// `CI` is set.
    pub(crate) ci: bool,
    /// stdout and stderr are both terminals.
    pub(crate) interactive: bool,
}

impl Gate {
    /// The gate for this process, given whether the command is eligible.
    pub(crate) fn from_env(eligible: bool) -> Self {
        Self {
            eligible,
            disabled: truthy(std::env::var_os(DISABLE_ENV).as_deref()),
            ci: truthy(std::env::var_os("CI").as_deref()),
            interactive: std::io::stdout().is_terminal() && std::io::stderr().is_terminal(),
        }
    }

    /// Whether to show a notice (and refresh the cache).
    pub(crate) fn open(self) -> bool {
        self.eligible && !self.disabled && !self.ci && self.interactive
    }
}

/// `~/.qubivex`, without creating it.
fn data_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .map(|h| Path::new(&h).join(".qubivex"))
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The URL holding the latest version for the installation in `data`, or `None` when
/// `qubivex update` could not update it.
fn lookup_url(data: &Path) -> Option<String> {
    let (repo, branch) = crate::update::followed_branch(data)?;
    match std::env::var(URL_ENV) {
        Ok(url) if !url.is_empty() => Some(url),
        _ => source::manifest_url(&repo, &branch),
    }
}

/// The cached latest version for `source`, if it is newer than `installed`.
fn newer(cached: Option<&Cache>, source: &str, installed: &Version) -> Option<Version> {
    let c = cached.filter(|c| c.source == source)?;
    Version::parse(c.latest.as_deref()?).filter(|latest| latest > installed)
}

/// The notice for `installed` and `latest`.
pub(crate) fn notice(installed: &Version, latest: &Version) -> String {
    let rule = "-".repeat(40);
    format!(
        "\n{rule}\nQubivex Update Available\n\nCurrent version: {installed}\nLatest version:  \
         {latest}\n\nRun `qubivex update` to upgrade.\n{rule}\n"
    )
}

/// What to do after a command, given the cache at the time: the notice to show, and
/// whether to start a refresh (with the cache claimed first).
#[derive(Debug, PartialEq)]
struct Plan {
    notice: Option<String>,
    claim: Option<Cache>,
}

fn plan(cached: Option<&Cache>, source: &str, installed: &Version, now: u64) -> Plan {
    let fresh = cached.is_some_and(|c| c.is_fresh(source, now, TTL_SECS));
    let claim = (!fresh).then(|| Cache {
        checked_at: now,
        source: source.to_owned(),
        latest: cached
            .filter(|c| c.source == source)
            .and_then(|c| c.latest.clone()),
    });
    Plan {
        notice: newer(cached, source, installed).map(|latest| notice(installed, &latest)),
        claim,
    }
}

/// Runs after a command. Never fails and never waits on the network.
pub(crate) fn after_command(eligible: bool) {
    if !Gate::from_env(eligible).open() {
        return;
    }
    let Some(data) = data_dir() else { return };
    let Some(source) = lookup_url(&data) else {
        return;
    };
    let Some(installed) = Version::parse(qubivex_core::VERSION) else {
        return;
    };
    let path = data.join(cache::FILE);
    let p = plan(cache::load(&path).as_ref(), &source, &installed, now());
    // Claiming first stops every command started meanwhile from starting a lookup too;
    // if the cache cannot be written, no lookup is started at all.
    if let Some(claim) = p.claim
        && cache::save(&path, &claim).is_ok()
    {
        spawn_refresh(&source);
    }
    if let Some(text) = p.notice {
        let _ = std::io::stderr().lock().write_all(text.as_bytes());
    }
}

/// Starts `qubivex refresh-update-check --url SOURCE`, detached, with no stdio.
fn spawn_refresh(source: &str) {
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let mut c = Command::new(exe);
    c.args([REFRESH_COMMAND, "--url", source])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // Its own process group: Ctrl-C in the terminal does not reach it.
        c.process_group(0);
    }
    let _ = c.spawn();
}

/// Runs after `--version`. The person asked which version they have, so the latest is
/// looked up now (at most [`VERSION_TIMEOUT`]) and recorded in the cache; if that fails,
/// the cache answers. Shown under the same conditions as [`after_command`].
pub(crate) fn after_version() {
    if !Gate::from_env(true).open() {
        return;
    }
    let Some(data) = data_dir() else { return };
    let Some(source) = lookup_url(&data) else {
        return;
    };
    let Some(installed) = Version::parse(qubivex_core::VERSION) else {
        return;
    };
    let latest = record_lookup(&data, &source, VERSION_TIMEOUT)
        .ok()
        .or_else(|| {
            cache::load(&data.join(cache::FILE))
                .filter(|c| c.source == source)
                .and_then(|c| Version::parse(c.latest.as_deref()?))
        });
    if let Some(latest) = latest.filter(|l| l > &installed) {
        let _ = std::io::stderr()
            .lock()
            .write_all(notice(&installed, &latest).as_bytes());
    }
}

/// Looks up the latest version at `url` (giving up after `timeout`) and records it in the
/// cache under `data`. On a failure the cache keeps the last version found, and the
/// error is returned.
fn record_lookup(data: &Path, url: &str, timeout: Duration) -> Result<Version, String> {
    std::fs::create_dir_all(data).map_err(|e| format!("cannot create {}: {e}", data.display()))?;
    let path = data.join(cache::FILE);
    let previous = cache::load(&path)
        .filter(|c| c.source == url)
        .and_then(|c| c.latest);
    let found = source::fetch_latest(url, timeout);
    let entry = Cache {
        checked_at: now(),
        source: url.to_owned(),
        latest: match &found {
            Ok(v) => Some(v.to_string()),
            Err(_) => previous,
        },
    };
    cache::save(&path, &entry).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    found
}

/// `qubivex refresh-update-check --url URL`: looks up the latest version at `url` and
/// records it in the cache. On a failure the cache keeps the last version found, and the
/// error is returned (for whoever runs it by hand; in the background nobody reads it).
pub(crate) fn refresh(url: &str) -> Result<(), String> {
    let data = data_dir().ok_or_else(|| "HOME is not set".to_owned())?;
    let latest = record_lookup(&data, url, TIMEOUT)?;
    println!("latest version at {url}: {latest}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> Version {
        Version::parse(s).unwrap_or_else(|| panic!("{s}"))
    }

    fn cache(at: u64, source: &str, latest: Option<&str>) -> Cache {
        Cache {
            checked_at: at,
            source: source.into(),
            latest: latest.map(str::to_owned),
        }
    }

    const NOW: u64 = 1_000_000;

    #[test]
    fn the_gate_needs_every_condition() {
        let open = Gate {
            eligible: true,
            disabled: false,
            ci: false,
            interactive: true,
        };
        assert!(open.open());
        assert!(
            !Gate {
                eligible: false,
                ..open
            }
            .open()
        );
        assert!(
            !Gate {
                disabled: true,
                ..open
            }
            .open()
        );
        assert!(!Gate { ci: true, ..open }.open());
        assert!(
            !Gate {
                interactive: false,
                ..open
            }
            .open()
        );
    }

    #[test]
    fn environment_switches() {
        assert!(truthy(Some(OsStr::new("1"))));
        assert!(truthy(Some(OsStr::new("yes"))));
        assert!(truthy(Some(OsStr::new("true"))));
        assert!(!truthy(None));
        assert!(!truthy(Some(OsStr::new(""))));
        assert!(!truthy(Some(OsStr::new("0"))));
        assert!(!truthy(Some(OsStr::new("FALSE"))));
    }

    #[test]
    fn notice_only_for_a_newer_version() {
        let fresh = |latest| cache(NOW - 10, "u", Some(latest));
        for (installed, latest, shown) in [
            ("0.1.1", "0.1.2", true),
            ("0.1.1", "0.2.0", true),
            ("0.1.1", "1.0.0", true),
            ("0.1.2", "0.1.2", false),
            ("0.2.0", "0.1.2", false),
            ("0.2.0-rc.1", "0.2.0", true),
            ("0.2.0", "0.2.0-rc.1", false),
        ] {
            let p = plan(Some(&fresh(latest)), "u", &v(installed), NOW);
            assert_eq!(p.notice.is_some(), shown, "{installed} -> {latest}");
            assert_eq!(p.claim, None, "a fresh cache needs no lookup");
        }
    }

    #[test]
    fn notice_text_names_both_versions_and_the_command() {
        let text = notice(&v("0.1.1"), &v("0.1.2"));
        assert_eq!(
            text,
            "\n----------------------------------------\nQubivex Update Available\n\n\
             Current version: 0.1.1\nLatest version:  0.1.2\n\nRun `qubivex update` to \
             upgrade.\n----------------------------------------\n"
        );
    }

    #[test]
    fn no_cache_starts_a_lookup_and_shows_nothing() {
        let p = plan(None, "u", &v("0.1.1"), NOW);
        assert_eq!(p.notice, None);
        assert_eq!(p.claim, Some(cache(NOW, "u", None)));
    }

    #[test]
    fn an_expired_cache_still_shows_its_version_and_keeps_it_while_refreshing() {
        let old = cache(NOW - TTL_SECS, "u", Some("0.9.0"));
        let p = plan(Some(&old), "u", &v("0.1.1"), NOW);
        assert!(p.notice.is_some());
        assert_eq!(p.claim, Some(cache(NOW, "u", Some("0.9.0"))));
    }

    #[test]
    fn a_cache_for_another_branch_counts_for_nothing() {
        let other = cache(NOW - 10, "other", Some("9.0.0"));
        let p = plan(Some(&other), "u", &v("0.1.1"), NOW);
        assert_eq!(p.notice, None);
        assert_eq!(p.claim, Some(cache(NOW, "u", None)));
    }

    #[test]
    fn an_invalid_cached_version_is_ignored() {
        let bad = cache(NOW - 10, "u", Some("not-a-version"));
        assert_eq!(plan(Some(&bad), "u", &v("0.1.1"), NOW).notice, None);
    }

    #[test]
    fn after_an_update_the_notice_stops() {
        // The cache still says 0.1.2; the binary now is 0.1.2.
        let c = cache(NOW - 10, "u", Some("0.1.2"));
        assert!(plan(Some(&c), "u", &v("0.1.1"), NOW).notice.is_some());
        assert!(plan(Some(&c), "u", &v("0.1.2"), NOW).notice.is_none());
    }
}
