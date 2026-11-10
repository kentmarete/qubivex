//! `qubivex serve` settings through the real binary: the config file is read, checked and
//! refused when it is wrong, all before any model loads or any port opens.

#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn qubivex(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_qubivex"))
        .args(args)
        .env_remove("QUBIVEX_API_KEY")
        .env_remove("QUBIVEX_ADMIN_TOKEN")
        .env_remove("QUBIVEX_CONFIG")
        .output()
        .unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qubivex-cli-serve-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn a_misspelt_key_is_an_error_naming_it() {
    let f = scratch("typo.toml");
    fs::write(&f, "[serve]\nmax_concurent = 3\n").unwrap();
    let o = qubivex(&["serve", "--config", f.to_str().unwrap()]);
    assert!(!o.status.success());
    assert!(stderr(&o).contains("max_concurent"), "{}", stderr(&o));
}

#[test]
fn a_remote_listener_without_a_key_is_refused() {
    let f = scratch("open.toml");
    fs::write(
        &f,
        "[serve]\nlisten = \"0.0.0.0:8080\"\nmodels = [\"nonexistent\"]\n",
    )
    .unwrap();
    let o = qubivex(&["serve", "--config", f.to_str().unwrap()]);
    assert!(!o.status.success());
    // Refused before the model is looked for.
    assert!(stderr(&o).contains("without an API key"), "{}", stderr(&o));
}

#[cfg(unix)]
#[test]
fn a_config_file_with_a_key_must_be_private() {
    use std::os::unix::fs::PermissionsExt;
    let f = scratch("key.toml");
    fs::write(
        &f,
        "[serve]\napi_key = \"secret\"\nmodels = [\"nonexistent\"]\n",
    )
    .unwrap();
    fs::set_permissions(&f, fs::Permissions::from_mode(0o644)).unwrap();
    let o = qubivex(&["serve", "--config", f.to_str().unwrap()]);
    assert!(!o.status.success());
    let e = stderr(&o);
    assert!(e.contains("chmod 600"), "{e}");
    assert!(!e.contains("secret"), "the key is never printed");

    // Private, it is read: the run gets as far as looking for the model.
    fs::set_permissions(&f, fs::Permissions::from_mode(0o600)).unwrap();
    let o = qubivex(&["serve", "--config", f.to_str().unwrap()]);
    assert!(!o.status.success());
    let e = stderr(&o);
    assert!(!e.contains("chmod"), "{e}");
    assert!(e.contains("nonexistent"), "{e}");
}

/// `docker stop`, systemd and Kubernetes stop a service with SIGTERM. It must start the
/// same drain as Ctrl-C and end with status 0, not kill the process mid-request.
#[cfg(unix)]
#[test]
fn sigterm_drains_and_exits_zero() {
    use std::io::{BufRead, BufReader, Read};
    use std::process::Stdio;

    let models = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/models");
    let mut child = Command::new(env!("CARGO_BIN_EXE_qubivex"))
        .args(["serve", "--model", "olmoe-tiny", "--listen", "127.0.0.1:0"])
        .arg("--dir")
        .arg(&models)
        .env_remove("QUBIVEX_API_KEY")
        .env_remove("QUBIVEX_ADMIN_TOKEN")
        .env_remove("QUBIVEX_CONFIG")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut err = BufReader::new(child.stderr.take().unwrap());
    let mut seen = String::new();
    // The "serving" line is printed once the port is open and the handler can be installed.
    loop {
        let mut line = String::new();
        if err.read_line(&mut line).unwrap() == 0 {
            let _ = child.kill();
            panic!("the server ended before listening: {seen}");
        }
        seen.push_str(&line);
        if line.contains("serving") {
            break;
        }
    }
    // The handler is installed just after that line; a moment covers the gap.
    std::thread::sleep(std::time::Duration::from_millis(300));
    let killed = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(killed.success());
    let status = child.wait().unwrap();
    err.read_to_string(&mut seen).unwrap();
    assert!(status.success(), "{status:?}: {seen}");
    assert!(seen.contains("shutting down"), "{seen}");
}
