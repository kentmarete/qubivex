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
