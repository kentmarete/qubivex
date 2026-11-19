//! Running a `qubivex` command in the background (`update --detach`, `serve --detach`):
//! the state lives in `~/.qubivex` as a pid file and a log per job.

use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

/// `~/.qubivex`, created if missing.
pub(crate) fn data_dir() -> Result<PathBuf, String> {
    let dir = std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .map(|h| Path::new(&h).join(".qubivex"))
        .ok_or_else(|| "HOME is not set".to_owned())?;
    fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    Ok(dir)
}

pub(crate) fn read_pid(path: &Path) -> Option<u32> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// Whether process `pid` exists (`kill -0`, which sends nothing).
pub(crate) fn alive(pid: u32) -> bool {
    signal(pid, "-0")
}

/// Sends `sig` (as `kill` spells it, e.g. `-TERM`) to `pid`; false if it could not.
pub(crate) fn signal(pid: u32, sig: &str) -> bool {
    Command::new("kill")
        .args([sig, &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// The pid recorded in `pid_file`, if that process is still running.
pub(crate) fn running(pid_file: &Path) -> Option<u32> {
    read_pid(pid_file).filter(|&p| alive(p))
}

/// Removes `pid_file` if it names this process. Called by a background job as it ends.
pub(crate) fn release(pid_file: &Path) {
    if read_pid(pid_file) == Some(std::process::id()) {
        // A pid file left behind only makes `--status` check a dead pid.
        let _ = fs::remove_file(pid_file);
    }
}

/// The last `n` lines of `path`, or `None` if it does not exist.
pub(crate) fn tail(path: &Path, n: usize) -> Result<Option<Vec<String>>, String> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    let mut lines: Vec<String> = BufReader::new(file).lines().map_while(Result::ok).collect();
    let start = lines.len().saturating_sub(n);
    Ok(Some(lines.split_off(start)))
}

/// Starts this binary with `args` in its own process group, so Ctrl-C and the terminal
/// closing do not reach it, with stdin closed and stdout and stderr in `log` (replaced),
/// whose first line is `header`. Records its pid in `pid_file`.
#[cfg(unix)]
pub(crate) fn spawn(
    args: &[OsString],
    log: &Path,
    header: &str,
    pid_file: &Path,
) -> Result<Child, String> {
    use std::os::unix::process::CommandExt;

    let mut out = File::create(log).map_err(|e| format!("cannot create {}: {e}", log.display()))?;
    let started = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    writeln!(out, "==> {header} (unix time {started})")
        .map_err(|e| format!("cannot write {}: {e}", log.display()))?;
    let err = out
        .try_clone()
        .map_err(|e| format!("cannot open {}: {e}", log.display()))?;
    let exe = std::env::current_exe().map_err(|e| format!("cannot find this binary: {e}"))?;
    let child = Command::new(exe)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err))
        .process_group(0)
        .spawn()
        .map_err(|e| format!("cannot start the background process: {e}"))?;
    fs::write(pid_file, format!("{}\n", child.id()))
        .map_err(|e| format!("cannot write {}: {e}", pid_file.display()))?;
    Ok(child)
}

#[cfg(not(unix))]
pub(crate) fn spawn(_: &[OsString], _: &Path, _: &str, _: &Path) -> Result<Child, String> {
    Err("--detach needs Linux or macOS".to_owned())
}
