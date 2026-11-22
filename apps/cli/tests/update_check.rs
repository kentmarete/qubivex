//! The update notice end to end: the background lookup against a local HTTP server (no
//! GitHub), the cache it keeps, and where the notice may and may not appear.
#![cfg(unix)]
#![allow(clippy::unwrap_used)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The URL the notice looks up for the receipt below.
const SOURCE: &str = "https://raw.githubusercontent.com/o/r/main/Cargo.toml";
const RULE: &str = "----------------------------------------";

struct Env {
    root: PathBuf,
}

impl Env {
    /// A HOME whose ~/.qubivex has an install.sh receipt following branch main of o/r.
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "qubivex-cli-update-check-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join(".qubivex")).unwrap();
        std::fs::write(
            root.join(".qubivex/install-receipt"),
            "ref_kind=branch\nref=main\ncommit=abc\nrepo=https://github.com/o/r.git\n",
        )
        .unwrap();
        Self { root }
    }

    fn cache_path(&self) -> PathBuf {
        self.root.join(".qubivex/update-check.json")
    }

    fn write_cache(&self, checked_at: u64, source: &str, latest: Option<&str>) {
        let v = serde_json::json!({"checked_at": checked_at, "source": source, "latest": latest});
        std::fs::write(self.cache_path(), v.to_string()).unwrap();
    }

    fn cache(&self) -> Option<serde_json::Value> {
        serde_json::from_slice(&std::fs::read(self.cache_path()).ok()?).ok()
    }

    /// `qubivex` with this HOME and none of the switches the outer environment may set.
    fn qubivex(&self) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_qubivex"));
        c.env("HOME", &self.root)
            .env_remove("CI")
            .env_remove("QUBIVEX_NO_UPDATE_CHECK")
            .env_remove("QUBIVEX_UPDATE_CHECK_URL");
        c
    }

    /// Runs `args` in a pseudo-terminal (so stdout and stderr are terminals), returning
    /// what the terminal showed.
    fn in_terminal(&self, args: &[&str], extra_env: &[(&str, &str)]) -> String {
        let exe = env!("CARGO_BIN_EXE_qubivex");
        let mut c = Command::new("script");
        if cfg!(target_os = "macos") {
            c.args(["-q", "/dev/null", exe]).args(args);
        } else {
            let line = std::iter::once(exe)
                .chain(args.iter().copied())
                .collect::<Vec<_>>()
                .join(" ");
            c.args(["-qec", &line, "/dev/null"]);
        }
        c.env("HOME", &self.root)
            .env_remove("CI")
            .env_remove("QUBIVEX_NO_UPDATE_CHECK")
            .env_remove("QUBIVEX_UPDATE_CHECK_URL")
            .stdin(std::process::Stdio::null());
        for (k, v) in extra_env {
            c.env(k, v);
        }
        let out = c.output().unwrap();
        String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n")
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// A server on 127.0.0.1 that answers each of `responses` connections in turn: `Some`
/// with that status and body, `None` by reading the request and never answering.
fn serve(responses: Vec<Option<(u16, String)>>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/Cargo.toml", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        let mut held: Vec<TcpStream> = Vec::new();
        for response in responses {
            let Ok((mut s, _)) = listener.accept() else {
                return;
            };
            let mut buf = [0u8; 4096];
            let _ = s.read(&mut buf);
            match response {
                Some((status, body)) => {
                    let _ = write!(
                        s,
                        "HTTP/1.1 {status} X\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                }
                None => held.push(s),
            }
        }
        std::thread::sleep(Duration::from_secs(30));
    });
    url
}

fn manifest(version: &str) -> String {
    format!("[workspace]\nmembers = []\n\n[workspace.package]\nversion = \"{version}\"\n")
}

fn refresh(env: &Env, url: &str) -> Output {
    env.qubivex()
        .args(["refresh-update-check", "--url", url])
        .output()
        .unwrap()
}

#[test]
fn refresh_records_the_latest_version() {
    let env = Env::new("ok");
    let url = serve(vec![Some((200, manifest("9.9.9")))]);
    let out = refresh(&env, &url);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let c = env.cache().unwrap();
    assert_eq!(c["latest"], "9.9.9");
    assert_eq!(c["source"], url.as_str());
    assert!(c["checked_at"].as_u64().unwrap() + 5 >= now());
}

#[test]
fn a_server_error_keeps_the_last_version_and_waits_a_day() {
    let env = Env::new("500");
    let url = serve(vec![Some((500, "oops".into()))]);
    env.write_cache(1, &url, Some("0.5.0"));
    let out = refresh(&env, &url);
    assert_eq!(out.status.code(), Some(1));
    let c = env.cache().unwrap();
    assert_eq!(c["latest"], "0.5.0");
    assert!(
        c["checked_at"].as_u64().unwrap() > 1,
        "the failed lookup is recorded"
    );
}

#[test]
fn an_invalid_remote_version_is_not_recorded() {
    let env = Env::new("invalid");
    let url = serve(vec![
        Some((200, manifest("latest"))),
        Some((200, "<html>not toml</html>".into())),
    ]);
    for _ in 0..2 {
        let out = refresh(&env, &url);
        assert_eq!(out.status.code(), Some(1));
        assert_eq!(env.cache().unwrap()["latest"], serde_json::Value::Null);
    }
}

#[test]
fn a_server_that_never_answers_times_out() {
    let env = Env::new("timeout");
    let url = serve(vec![None]);
    let start = Instant::now();
    let out = refresh(&env, &url);
    let took = start.elapsed();
    assert_eq!(out.status.code(), Some(1));
    assert!(took < Duration::from_secs(10), "took {took:?}");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("timeout")
            || String::from_utf8_lossy(&out.stderr).contains("timed out"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn no_server_at_all_fails_quietly() {
    let env = Env::new("refused");
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let out = refresh(&env, &format!("http://127.0.0.1:{port}/Cargo.toml"));
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
}

#[test]
fn plain_http_elsewhere_is_refused() {
    let env = Env::new("http");
    let out = refresh(&env, "http://example.invalid/Cargo.toml");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("not https"));
}

#[test]
fn a_corrupted_cache_is_replaced() {
    let env = Env::new("corrupt");
    std::fs::write(env.cache_path(), b"{\"checked_at\": tru").unwrap();
    let url = serve(vec![Some((200, manifest("0.3.0")))]);
    assert!(refresh(&env, &url).status.success());
    assert_eq!(env.cache().unwrap()["latest"], "0.3.0");
}

/// Piped output (as scripts and the tests see it) never gets a notice and never starts
/// a lookup, even with a newer version cached and with the cache expired.
#[test]
fn piped_output_is_untouched() {
    let env = Env::new("piped");
    for checked_at in [now(), 1] {
        env.write_cache(checked_at, SOURCE, Some("99.0.0"));
        let version = env.qubivex().arg("--version").output().unwrap();
        assert_eq!(
            String::from_utf8(version.stdout).unwrap(),
            format!("qubivex {}\n", env!("CARGO_PKG_VERSION"))
        );
        assert!(version.stderr.is_empty());

        let help = env.qubivex().arg("--help").output().unwrap();
        let text = String::from_utf8(help.stdout).unwrap();
        assert!(text.contains("Usage: qubivex"), "{text}");
        assert!(!text.contains("Update Available"), "{text}");
        assert!(help.stderr.is_empty());

        let json = env.qubivex().args(["system", "--json"]).output().unwrap();
        assert!(json.status.success());
        serde_json::from_slice::<serde_json::Value>(&json.stdout).unwrap();
        assert!(json.stderr.is_empty());

        let system = env.qubivex().arg("system").output().unwrap();
        assert!(system.stderr.is_empty());
        assert!(!String::from_utf8(system.stdout).unwrap().contains(RULE));

        // Nothing was looked up: the cache is as written.
        assert_eq!(env.cache().unwrap()["checked_at"], checked_at);
    }
}

#[test]
fn the_notice_follows_help_in_a_terminal() {
    let env = Env::new("tty-help");
    env.write_cache(now(), SOURCE, Some("99.0.0"));
    let shown = env.in_terminal(&["--help"], &[]);
    let help_end = shown.find("Print version").expect(&shown);
    let notice = shown.find("Qubivex Update Available").expect(&shown);
    assert!(notice > help_end, "{shown}");
    assert!(
        shown.contains(&format!(
            "Current version: {}\nLatest version:  99.0.0\n\nRun `qubivex update` to upgrade.",
            env!("CARGO_PKG_VERSION")
        )),
        "{shown}"
    );
    assert_eq!(shown.matches("Update Available").count(), 1, "{shown}");
}

/// Arguments, and extra environment variables.
type Case = (
    &'static [&'static str],
    &'static [(&'static str, &'static str)],
);

#[test]
fn no_notice_in_a_terminal_when_not_due() {
    let env = Env::new("tty-quiet");
    env.write_cache(now(), SOURCE, Some("99.0.0"));
    let cases: [Case; 6] = [
        (&["--version"], &[("QUBIVEX_NO_UPDATE_CHECK", "1")]),
        (&["--version"], &[("CI", "true")]),
        (&["system", "--json"], &[]),
        (&["--help"], &[("QUBIVEX_NO_UPDATE_CHECK", "1")]),
        (&["--help"], &[("CI", "true")]),
        (&["update", "--status"], &[]),
    ];
    for (args, extra) in cases {
        let shown = env.in_terminal(args, extra);
        assert!(
            !shown.contains("Update Available"),
            "{args:?} {extra:?}: {shown}"
        );
    }
    // The current version: nothing to say.
    env.write_cache(now(), SOURCE, Some(env!("CARGO_PKG_VERSION")));
    assert!(
        !env.in_terminal(&["--help"], &[])
            .contains("Update Available")
    );
    // An older one on the branch (a newer build installed): nothing either.
    env.write_cache(now(), SOURCE, Some("0.0.1"));
    assert!(
        !env.in_terminal(&["--help"], &[])
            .contains("Update Available")
    );
}

#[test]
fn without_a_receipt_nothing_happens() {
    let env = Env::new("no-receipt");
    std::fs::remove_file(env.root.join(".qubivex/install-receipt")).unwrap();
    env.write_cache(now(), SOURCE, Some("99.0.0"));
    assert!(
        !env.in_terminal(&["--help"], &[])
            .contains("Update Available")
    );
    // A tag-pinned installation is not moved by `qubivex update` either.
    std::fs::write(
        env.root.join(".qubivex/install-receipt"),
        "ref_kind=tag\nref=v0.1.0\ncommit=abc\nrepo=https://github.com/o/r.git\n",
    )
    .unwrap();
    assert!(
        !env.in_terminal(&["--help"], &[])
            .contains("Update Available")
    );
}

/// The first command in a terminal with no cache shows nothing, starts the lookup in the
/// background and returns at once; the next one shows what it found.
#[test]
fn an_expired_cache_is_refreshed_in_the_background() {
    let env = Env::new("tty-refresh");
    let url = serve(vec![Some((200, manifest("42.0.0")))]);
    let start = Instant::now();
    let first = env.in_terminal(&["--help"], &[("QUBIVEX_UPDATE_CHECK_URL", &url)]);
    assert!(start.elapsed() < Duration::from_secs(3));
    assert!(!first.contains("Update Available"), "{first}");
    let deadline = Instant::now() + Duration::from_secs(10);
    while env.cache().is_none_or(|c| c["latest"] != "42.0.0") {
        assert!(Instant::now() < deadline, "no lookup: {:?}", env.cache());
        std::thread::sleep(Duration::from_millis(50));
    }
    let second = env.in_terminal(&["--help"], &[("QUBIVEX_UPDATE_CHECK_URL", &url)]);
    assert!(second.contains("Latest version:  42.0.0"), "{second}");
}

/// A URL on this machine where nothing listens: a lookup there is refused at once.
fn refused_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/Cargo.toml", listener.local_addr().unwrap());
    drop(listener);
    url
}

/// `--version` in a terminal looks the latest up at once, even when the daily lookup has
/// just said there is nothing newer: a release made since then is shown straight away,
/// after the version line, and recorded in the cache.
#[test]
fn version_looks_up_the_latest_now() {
    let env = Env::new("tty-version");
    let url = serve(vec![Some((200, manifest("42.0.0")))]);
    env.write_cache(now(), &url, Some(env!("CARGO_PKG_VERSION")));
    let shown = env.in_terminal(&["--version"], &[("QUBIVEX_UPDATE_CHECK_URL", &url)]);
    let line = shown
        .find(&format!("qubivex {}\n", env!("CARGO_PKG_VERSION")))
        .expect(&shown);
    let notice = shown.find("Qubivex Update Available").expect(&shown);
    assert!(line < notice, "{shown}");
    assert!(
        shown.contains(&format!(
            "{RULE}\nQubivex Update Available\n\nCurrent version: {}\nLatest version:  42.0.0\n\n\
             Run `qubivex update` to upgrade.\n{RULE}\n",
            env!("CARGO_PKG_VERSION")
        )),
        "{shown}"
    );
    assert_eq!(env.cache().unwrap()["latest"], "42.0.0");
}

/// When the lookup fails, `--version` answers from the cache, and never waits more than
/// its few seconds.
#[test]
fn version_falls_back_to_the_cache() {
    let env = Env::new("tty-version-offline");
    let refused = refused_url();
    env.write_cache(now(), &refused, Some("99.0.0"));
    let shown = env.in_terminal(&["--version"], &[("QUBIVEX_UPDATE_CHECK_URL", &refused)]);
    assert!(shown.contains("Latest version:  99.0.0"), "{shown}");

    let silent = serve(vec![None]);
    env.write_cache(now(), &silent, Some("98.0.0"));
    let start = Instant::now();
    let shown = env.in_terminal(&["--version"], &[("QUBIVEX_UPDATE_CHECK_URL", &silent)]);
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "{:?}",
        start.elapsed()
    );
    assert!(shown.contains("Latest version:  98.0.0"), "{shown}");

    // Nothing cached and nothing found: only the version line.
    std::fs::remove_file(env.cache_path()).unwrap();
    let shown = env.in_terminal(&["--version"], &[("QUBIVEX_UPDATE_CHECK_URL", &refused)]);
    assert!(
        shown.contains(&format!("qubivex {}\n", env!("CARGO_PKG_VERSION"))),
        "{shown}"
    );
    assert!(!shown.contains(RULE), "{shown}");
}
