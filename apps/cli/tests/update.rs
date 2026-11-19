//! `qubivex update` against a local git repository whose install.sh is a stub that
//! records how it was called. No network, no build.
#![cfg(unix)]
#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

struct Env {
    root: PathBuf,
}

impl Env {
    fn new(name: &str) -> Self {
        let root =
            std::env::temp_dir().join(format!("qubivex-cli-update-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("home/.qubivex")).unwrap();
        std::fs::create_dir_all(root.join("repo")).unwrap();
        std::fs::create_dir_all(root.join("tmp")).unwrap();
        Self { root }
    }

    fn home(&self) -> PathBuf {
        self.root.join("home")
    }

    fn repo(&self) -> PathBuf {
        self.root.join("repo")
    }

    /// A repository on branch main whose install.sh writes its arguments and
    /// QUBIVEX_REPO_URL to $HOME/installer-args. Returns the commit.
    fn init_repo(&self) -> String {
        let script = "#!/usr/bin/env bash\n\
                      printf '%s\\n' \"$*\" \"$QUBIVEX_REPO_URL\" \"${QUBIVEX_NO_PROMPT:-}\" \
                      > \"$HOME/installer-args\"\necho stub installer ran\n";
        std::fs::write(self.repo().join("install.sh"), script).unwrap();
        self.git(&["init", "-q", "-b", "main"]);
        self.git(&["add", "install.sh"]);
        self.git(&[
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            "commit",
            "-q",
            "-m",
            "stub",
        ]);
        let out = self.git(&["rev-parse", "HEAD"]);
        String::from_utf8(out.stdout).unwrap().trim().to_owned()
    }

    fn git(&self, args: &[&str]) -> Output {
        let out = Command::new("git")
            .args(args)
            .current_dir(self.repo())
            .env("HOME", self.home())
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?}: {out:?}");
        out
    }

    fn receipt(&self, ref_kind: &str, reference: &str, commit: &str) {
        std::fs::write(
            self.home().join(".qubivex/install-receipt"),
            format!(
                "# Written by qubivex install.sh.\nref_kind={ref_kind}\nref={reference}\n\
                 commit={commit}\nrepo={}\n",
                self.repo().display()
            ),
        )
        .unwrap();
    }

    fn qubivex(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_qubivex"))
            .arg("update")
            .args(args)
            .env("HOME", self.home())
            .env("TMPDIR", self.root.join("tmp"))
            .output()
            .unwrap()
    }

    fn installer_args(&self) -> Option<String> {
        std::fs::read_to_string(self.home().join("installer-args")).ok()
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

fn wait_for(path: &Path, needle: &str) -> String {
    let start = Instant::now();
    loop {
        let t = std::fs::read_to_string(path).unwrap_or_default();
        if t.contains(needle) || start.elapsed() > Duration::from_secs(30) {
            return t;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[test]
fn without_a_receipt_it_says_how_to_update() {
    let env = Env::new("noreceipt");
    let out = env.qubivex(&[]);
    assert_eq!(out.status.code(), Some(1));
    let err = text(&out.stderr);
    assert!(
        err.contains("install-receipt") && err.contains("cargo install"),
        "{err}"
    );
}

#[test]
fn a_tag_pinned_install_needs_branch() {
    let env = Env::new("tag");
    env.init_repo();
    env.receipt("tag", "v0.1.0", "0000");
    let out = env.qubivex(&["--check"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(text(&out.stderr).contains("--branch"), "{out:?}");
    let out = env.qubivex(&["--check", "--branch", "main"]);
    assert!(out.status.success(), "{out:?}");
    assert!(text(&out.stdout).contains("update available"), "{out:?}");
}

#[test]
fn up_to_date_does_not_run_the_installer() {
    let env = Env::new("current");
    let head = env.init_repo();
    env.receipt("branch", "main", &head);
    let out = env.qubivex(&[]);
    assert!(out.status.success(), "{out:?}");
    assert!(text(&out.stdout).contains("up to date"), "{out:?}");
    assert_eq!(env.installer_args(), None);
}

#[test]
fn check_reports_without_installing() {
    let env = Env::new("check");
    env.init_repo();
    env.receipt("branch", "main", "1111111111111111111111111111111111111111");
    let out = env.qubivex(&["--check"]);
    assert!(out.status.success(), "{out:?}");
    assert!(text(&out.stdout).contains("111111111111 -> "), "{out:?}");
    assert_eq!(env.installer_args(), None);
}

#[test]
fn update_runs_the_branch_installer() {
    let env = Env::new("run");
    env.init_repo();
    env.receipt("branch", "main", "1111111111111111111111111111111111111111");
    let out = env.qubivex(&["--jobs", "2", "--no-test"]);
    assert!(out.status.success(), "{out:?}");
    let args = env.installer_args().expect("installer ran");
    let lines: Vec<&str> = args.lines().collect();
    assert_eq!(lines[0], "--update --branch main --jobs 2 --no-test");
    assert_eq!(lines[1], env.repo().display().to_string());
    assert_eq!(lines[2], "", "a foreground update may prompt");
}

#[test]
fn unknown_branch_is_an_error() {
    let env = Env::new("nobranch");
    env.init_repo();
    env.receipt("branch", "main", "1111");
    let out = env.qubivex(&["--branch", "nope"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(text(&out.stderr).contains("does not exist"), "{out:?}");
}

#[test]
fn detach_runs_in_the_background_and_status_reports_it() {
    let env = Env::new("detach");
    env.init_repo();
    env.receipt("branch", "main", "1111111111111111111111111111111111111111");
    let out = env.qubivex(&["--detach"]);
    assert!(out.status.success(), "{out:?}");
    assert!(text(&out.stdout).contains("in the background"), "{out:?}");
    let log = env.home().join(".qubivex/update.log");
    let t = wait_for(&log, "==> qubivex update f"); // finished or failed
    assert!(t.contains("stub installer ran"), "{t}");
    assert!(t.contains("==> qubivex update finished"), "{t}");
    let args = env.installer_args().expect("installer ran");
    assert_eq!(
        args.lines().nth(2),
        Some("1"),
        "background runs never prompt"
    );
    // The pid file goes once the child ends; give it a moment after the last log line.
    let start = Instant::now();
    while env.home().join(".qubivex/update.pid").exists()
        && start.elapsed() < Duration::from_secs(10)
    {
        std::thread::sleep(Duration::from_millis(50));
    }
    let out = env.qubivex(&["--status"]);
    assert!(out.status.success(), "{out:?}");
    let s = text(&out.stdout);
    assert!(
        s.contains("no update is running") && s.contains("update finished"),
        "{s}"
    );
}

#[test]
fn status_before_any_update() {
    let env = Env::new("status");
    let out = env.qubivex(&["--status"]);
    assert!(out.status.success(), "{out:?}");
    assert!(
        text(&out.stdout).contains("no background update has run"),
        "{out:?}"
    );
}

#[test]
fn check_and_detach_conflict() {
    let env = Env::new("conflict");
    let out = env.qubivex(&["--check", "--detach"]);
    assert_eq!(out.status.code(), Some(2));
}
