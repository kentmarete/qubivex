//! Where probes read from: the live system, or a recorded fixture.
//!
//! Probes never touch `std::fs` or `std::process` directly. They go through a
//! [`Host`], so the same code runs against the real machine and against a recorded
//! tree of `/proc`, `/sys` and command outputs in tests.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::HardwareError;

/// Default time limit for helper programs.
pub(crate) const COMMAND_TIMEOUT: Duration = Duration::from_secs(5);
/// GPU probes may load drivers; a broken driver must not hang QUBIVEX.
pub(crate) const GPU_TIMEOUT: Duration = Duration::from_secs(10);
/// Files read by probes are small; anything larger is not what we expect.
const MAX_FILE_BYTES: u64 = 4 << 20;

/// The operating-system family a probe set targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OsFamily {
    /// Linux: `/proc` and `/sys`.
    Linux,
    /// macOS: `sysctl`, `vm_stat` and friends.
    MacOs,
    /// Anything else: every probe reports unknown.
    Other,
}

/// A helper-program invocation. `key` names its recorded output in a fixture.
#[derive(Clone, Debug)]
pub(crate) struct Cmd<'a> {
    pub(crate) key: String,
    pub(crate) program: &'a str,
    pub(crate) args: Vec<String>,
    pub(crate) timeout: Duration,
    /// Accept output even on a non-zero exit (`sysctl` with an unknown key).
    pub(crate) lenient: bool,
}

impl<'a> Cmd<'a> {
    pub(crate) fn new(program: &'a str, args: &[&str]) -> Self {
        Self {
            key: program.to_owned(),
            program,
            args: args.iter().map(|&a| a.to_owned()).collect(),
            timeout: COMMAND_TIMEOUT,
            lenient: false,
        }
    }
    pub(crate) fn key(mut self, key: impl Into<String>) -> Self {
        self.key = key.into();
        self
    }
    pub(crate) const fn timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }
    pub(crate) const fn lenient(mut self) -> Self {
        self.lenient = true;
        self
    }
}

#[derive(Clone, Debug)]
enum Source {
    Live,
    Fixture(PathBuf),
}

/// The system being probed.
#[derive(Clone, Debug)]
pub struct Host {
    source: Source,
    os: OsFamily,
    arch: String,
    simd: Vec<String>,
    pid: u32,
}

impl Host {
    /// The machine this process runs on.
    #[must_use]
    pub fn live() -> Self {
        let os = if cfg!(target_os = "linux") {
            OsFamily::Linux
        } else if cfg!(target_os = "macos") {
            OsFamily::MacOs
        } else {
            OsFamily::Other
        };
        Self {
            source: Source::Live,
            os,
            arch: std::env::consts::ARCH.to_owned(),
            simd: crate::cpu::detect_simd(),
            pid: std::process::id(),
        }
    }

    /// A recorded host: files are read under `root` (so `/proc/meminfo` is
    /// `root/proc/meminfo`), and command `key`s from `root/cmd/<key>`. A missing
    /// command file behaves like a program that is not installed.
    ///
    /// Used by tests and by bug reports that attach a recorded tree.
    #[must_use]
    pub fn fixture(root: impl Into<PathBuf>, os: OsFamily, arch: &str, simd: &[&str]) -> Self {
        Self {
            source: Source::Fixture(root.into()),
            os,
            arch: arch.to_owned(),
            simd: simd.iter().map(|&s| s.to_owned()).collect(),
            pid: 4242,
        }
    }

    /// The operating-system family.
    #[must_use]
    pub const fn os(&self) -> OsFamily {
        self.os
    }

    pub(crate) fn arch(&self) -> &str {
        &self.arch
    }

    pub(crate) fn simd(&self) -> &[String] {
        &self.simd
    }

    pub(crate) const fn pid(&self) -> u32 {
        self.pid
    }

    fn path(&self, abs: &str) -> PathBuf {
        match &self.source {
            Source::Live => PathBuf::from(abs),
            Source::Fixture(root) => root.join(abs.trim_start_matches('/')),
        }
    }

    /// Reads a small text file such as `/proc/meminfo`.
    pub(crate) fn read(&self, abs: &str) -> Result<String, HardwareError> {
        let path = self.path(abs);
        let io = |source| HardwareError::Io {
            path: PathBuf::from(abs),
            source,
        };
        let f = std::fs::File::open(&path).map_err(io)?;
        let mut s = String::new();
        f.take(MAX_FILE_BYTES).read_to_string(&mut s).map_err(io)?;
        Ok(s)
    }

    /// Whether a path exists.
    pub(crate) fn exists(&self, abs: &str) -> bool {
        self.path(abs).exists()
    }

    /// Runs a helper program and returns its standard output.
    pub(crate) fn run(&self, cmd: &Cmd<'_>) -> Result<String, HardwareError> {
        match &self.source {
            Source::Live => run_with_timeout(cmd),
            Source::Fixture(root) => {
                let p = root.join("cmd").join(&cmd.key);
                if !p.exists() {
                    return Err(HardwareError::NotFound {
                        program: cmd.program.to_owned(),
                    });
                }
                std::fs::read_to_string(&p).map_err(|source| HardwareError::Io {
                    path: Path::new("cmd").join(&cmd.key),
                    source,
                })
            }
        }
    }
}

/// Runs `cmd` in a child process, killing it after `cmd.timeout`.
///
/// Output is drained on reader threads, so a chatty child cannot block on a full pipe.
fn run_with_timeout(cmd: &Cmd<'_>) -> Result<String, HardwareError> {
    let program = cmd.program.to_owned();
    let mut child = Command::new(cmd.program)
        .args(&cmd.args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("LC_ALL", "C")
        .spawn()
        .map_err(|source| {
            if source.kind() == std::io::ErrorKind::NotFound {
                HardwareError::NotFound {
                    program: program.clone(),
                }
            } else {
                HardwareError::Spawn {
                    program: program.clone(),
                    source,
                }
            }
        })?;

    let drain = |r: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut r) = r {
                // A read error leaves what was read; the exit status decides.
                let _ = r.by_ref().take(MAX_FILE_BYTES).read_to_end(&mut buf);
            }
            buf
        })
    };
    let out = drain(
        child
            .stdout
            .take()
            .map(|s| Box::new(s) as Box<dyn Read + Send>),
    );
    let err = drain(
        child
            .stderr
            .take()
            .map(|s| Box::new(s) as Box<dyn Read + Send>),
    );

    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) if start.elapsed() >= cmd.timeout => {
                // Already-exited races are harmless: kill then reap.
                let _ = child.kill();
                let _ = child.wait();
                return Err(HardwareError::Timeout {
                    program,
                    timeout: cmd.timeout,
                });
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(5)),
            Err(source) => return Err(HardwareError::Spawn { program, source }),
        }
    };
    let stdout = String::from_utf8_lossy(&out.join().unwrap_or_default()).into_owned();
    let stderr = String::from_utf8_lossy(&err.join().unwrap_or_default()).into_owned();
    if status.success() || (cmd.lenient && !stdout.trim().is_empty()) {
        Ok(stdout)
    } else {
        Err(HardwareError::CommandFailed {
            program,
            status: status.to_string(),
            stderr: stderr.lines().next().unwrap_or("").to_owned(),
        })
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn missing_program_is_not_found() {
        let e = run_with_timeout(&Cmd::new("qubivex-no-such-program", &[])).unwrap_err();
        assert!(matches!(e, HardwareError::NotFound { .. }), "{e}");
    }

    #[test]
    fn slow_program_is_killed() {
        let c = Cmd::new("sleep", &["5"]).timeout(Duration::from_millis(100));
        let t = Instant::now();
        let e = run_with_timeout(&c).unwrap_err();
        assert!(matches!(e, HardwareError::Timeout { .. }), "{e}");
        assert!(t.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn failure_reports_stderr() {
        let c = Cmd::new("sh", &["-c", "echo boom >&2; exit 3"]);
        let e = run_with_timeout(&c).unwrap_err();
        assert!(e.to_string().contains("boom"), "{e}");
        let ok = run_with_timeout(&Cmd::new("sh", &["-c", "echo hi; exit 1"]).lenient()).unwrap();
        assert_eq!(ok.trim(), "hi");
    }
}
