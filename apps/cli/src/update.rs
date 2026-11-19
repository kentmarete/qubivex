//! `qubivex update`: rebuilds the installed binary from the branch `install.sh` recorded.
//!
//! The installer stays the one place that knows how to build and swap the binary. This
//! command reads `~/.qubivex/install-receipt`, asks the repository for the branch's
//! newest commit (`git ls-remote`), and when it differs, fetches that commit's
//! `install.sh` and runs it with `--update`. `--detach` runs the same thing in the
//! background with its output in `~/.qubivex/update.log`; `--status` reports on it.

use std::ffi::OsString;
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::background;

const DEFAULT_REPO: &str = "https://github.com/kentmarete/qubivex.git";
const RECEIPT: &str = "install-receipt";
const LOG: &str = "update.log";
const PID: &str = "update.pid";
/// Lines of the log `--status` shows.
const STATUS_LINES: usize = 15;

/// The flags of `qubivex update`.
#[derive(Clone, Debug, Default, clap::Args)]
pub(crate) struct UpdateFlags {
    /// Only report whether the installed branch has a newer commit.
    #[arg(long, conflicts_with_all = ["detach", "status"])]
    check: bool,
    /// Follow this branch instead of the installed one (also moves a tag-pinned
    /// install onto a branch).
    #[arg(long)]
    branch: Option<String>,
    /// Parallel build jobs (default: cargo's, the number of CPUs).
    #[arg(long)]
    jobs: Option<u32>,
    /// Skip the installer's post-install checks (qubivex system, doctor --storage).
    #[arg(long)]
    no_test: bool,
    /// Run the update in the background and return at once. Output goes to
    /// ~/.qubivex/update.log; see `qubivex update --status`.
    #[arg(long, conflicts_with = "status")]
    detach: bool,
    /// Show whether a background update is running, and the end of its log.
    #[arg(long, conflicts_with_all = ["branch", "jobs", "no_test"])]
    status: bool,
    /// Set on the background process `--detach` starts.
    #[arg(long, hide = true)]
    detached_child: bool,
}

/// Runs `qubivex update` with `flags`.
pub(crate) fn update(flags: &UpdateFlags) -> Result<(), String> {
    let data = background::data_dir()?;
    if flags.status {
        return status(&data);
    }
    if flags.detach {
        return detach(flags, &data);
    }
    if !flags.detached_child {
        return run_update(flags, &data);
    }
    // The background process: record how it ended in the log, and release the pid file.
    let result = run_update(flags, &data);
    match &result {
        Ok(()) => println!("==> qubivex update finished"),
        Err(e) => println!("==> qubivex update failed: {e}"),
    }
    background::release(&data.join(PID));
    result
}

/// What `install.sh` recorded about the installation.
#[derive(Debug, Default)]
struct Receipt {
    ref_kind: String,
    reference: String,
    commit: String,
    repo: String,
}

fn read_receipt(data: &Path) -> Result<Receipt, String> {
    let path = data.join(RECEIPT);
    let text = fs::read_to_string(&path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            format!(
                "no {} found, so this qubivex was not installed with install.sh. Update it the \
                 way you installed it, e.g. from a checkout: git pull && cargo install --path \
                 apps/cli --locked --force",
                path.display()
            )
        } else {
            format!("cannot read {}: {e}", path.display())
        }
    })?;
    let mut r = Receipt::default();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.to_owned();
        match key {
            "ref_kind" => r.ref_kind = value,
            "ref" => r.reference = value,
            "commit" => r.commit = value,
            "repo" => r.repo = value,
            _ => {}
        }
    }
    if r.repo.is_empty() {
        r.repo = DEFAULT_REPO.to_owned();
    }
    Ok(r)
}

/// The branch to follow: `--branch`, else the installed one. A tag-pinned install has none.
fn branch_of(flags: &UpdateFlags, r: &Receipt) -> Result<String, String> {
    if let Some(b) = &flags.branch {
        return Ok(b.clone());
    }
    if r.ref_kind == "tag" {
        return Err(format!(
            "this installation is pinned to tag {:?}; update follows branches. Pass --branch \
             main to follow a branch, or install another tag with install.sh --version VERSION",
            r.reference
        ));
    }
    Ok(if r.reference.is_empty() {
        "main".to_owned()
    } else {
        r.reference.clone()
    })
}

fn git() -> Command {
    let mut c = Command::new("git");
    // Never stop at a credentials prompt, least of all in the background.
    c.env("GIT_TERMINAL_PROMPT", "0");
    c
}

/// The newest commit of `branch` in `repo`.
fn remote_head(repo: &str, branch: &str) -> Result<String, String> {
    let out = git()
        .args(["ls-remote", "--exit-code", "--", repo])
        .arg(format!("refs/heads/{branch}"))
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("cannot run git (is it installed?): {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(if out.status.code() == Some(2) {
            format!("branch {branch:?} does not exist in {repo}")
        } else {
            format!("git ls-remote {repo} failed: {}", err.trim())
        });
    }
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .map(str::to_owned)
        .ok_or_else(|| format!("git ls-remote {repo} returned no commit for {branch:?}"))
}

fn short(commit: &str) -> &str {
    commit.get(..12).unwrap_or(commit)
}

fn run_update(flags: &UpdateFlags, data: &Path) -> Result<(), String> {
    let receipt = read_receipt(data)?;
    let branch = branch_of(flags, &receipt)?;
    let head = remote_head(&receipt.repo, &branch)?;
    if head == receipt.commit {
        println!(
            "qubivex {} is up to date (branch {branch}, {})",
            qubivex_core::VERSION,
            short(&head)
        );
        return Ok(());
    }
    let from = if receipt.commit.is_empty() {
        String::new()
    } else {
        format!("{} -> ", short(&receipt.commit))
    };
    if flags.check {
        println!(
            "update available on branch {branch}: {from}{}",
            short(&head)
        );
        println!("run `qubivex update` (or `qubivex update --detach`) to install it");
        return Ok(());
    }
    println!("updating on branch {branch}: {from}{}", short(&head));
    let src = std::env::temp_dir().join(format!("qubivex-update-{}", std::process::id()));
    // A directory left by an earlier process with the same pid is ours to replace.
    let _ = fs::remove_dir_all(&src);
    let result = clone_and_install(flags, &receipt.repo, &branch, &src);
    // Leaving the temporary clone behind wastes only disk.
    let _ = fs::remove_dir_all(&src);
    result
}

fn clone_and_install(
    flags: &UpdateFlags,
    repo: &str,
    branch: &str,
    src: &Path,
) -> Result<(), String> {
    let status = git()
        .args([
            "clone", "--quiet", "--depth", "1", "--branch", branch, "--", repo,
        ])
        .arg(src)
        .stdin(Stdio::null())
        .status()
        .map_err(|e| format!("cannot run git: {e}"))?;
    if !status.success() {
        return Err(format!("git clone of {repo} (branch {branch}) failed"));
    }
    let script = src.join("install.sh");
    if !script.is_file() {
        return Err(format!("branch {branch} of {repo} has no install.sh"));
    }
    let mut c = Command::new("bash");
    c.arg(&script)
        .args(["--update", "--branch", branch])
        .env("QUBIVEX_REPO_URL", repo);
    if let Some(j) = flags.jobs {
        c.arg("--jobs").arg(j.to_string());
    }
    if flags.no_test {
        c.arg("--no-test");
    }
    if flags.detached_child {
        // Nobody can answer a prompt, and the log is not a terminal.
        c.env("QUBIVEX_NO_PROMPT", "1").env("NO_COLOR", "1");
        c.stdin(Stdio::null());
    }
    let status = c.status().map_err(|e| format!("cannot run bash: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(match status.code() {
            Some(code) => format!("install.sh failed (exit {code})"),
            None => "install.sh was stopped by a signal".to_owned(),
        })
    }
}

fn detach(flags: &UpdateFlags, data: &Path) -> Result<(), String> {
    // Fail here, where someone reads the message, rather than in the log.
    let receipt = read_receipt(data)?;
    branch_of(flags, &receipt)?;
    let pid_path = data.join(PID);
    if let Some(pid) = background::running(&pid_path) {
        return Err(format!(
            "an update is already running (pid {pid}); see `qubivex update --status`"
        ));
    }
    let mut args: Vec<OsString> = vec!["update".into(), "--detached-child".into()];
    if let Some(b) = &flags.branch {
        args.extend(["--branch".into(), b.into()]);
    }
    if let Some(j) = flags.jobs {
        args.extend(["--jobs".into(), j.to_string().into()]);
    }
    if flags.no_test {
        args.push("--no-test".into());
    }
    let log_path = data.join(LOG);
    let child = background::spawn(&args, &log_path, "qubivex update started", &pid_path)?;
    println!("updating in the background (pid {})", child.id());
    println!("  log:    {}", log_path.display());
    println!("  status: qubivex update --status");
    Ok(())
}

fn status(data: &Path) -> Result<(), String> {
    let log_path = data.join(LOG);
    match background::running(&data.join(PID)) {
        Some(pid) => println!("an update is running (pid {pid})"),
        None => println!("no update is running"),
    }
    let Some(lines) = background::tail(&log_path, STATUS_LINES)? else {
        println!(
            "no background update has run ({} does not exist)",
            log_path.display()
        );
        return Ok(());
    };
    println!("log: {} (last {STATUS_LINES} lines)", log_path.display());
    for line in &lines {
        println!("  {line}");
    }
    Ok(())
}
