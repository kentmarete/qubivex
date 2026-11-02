//! End-to-end tests of the `qubivex` binary.

use std::process::Command;

fn qubivex() -> Command {
    Command::new(env!("CARGO_BIN_EXE_qubivex"))
}

#[test]
fn version_prints_the_workspace_version() {
    let out = qubivex().arg("--version").output().unwrap();
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        format!("qubivex {}\n", env!("CARGO_PKG_VERSION"))
    );
}

#[test]
fn no_arguments_prints_help_and_fails() {
    let out = qubivex().output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(
        String::from_utf8(out.stderr)
            .unwrap()
            .contains("Usage: qubivex")
    );
}

#[test]
fn unknown_argument_is_a_usage_error() {
    let out = qubivex().arg("--frobnicate").output().unwrap();
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn system_prints_a_profile() {
    let out = qubivex().arg("system").output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.starts_with("OS "), "{text}");
    assert!(text.contains("\nCPU "), "{text}");
}

#[test]
fn system_json_is_valid_json() {
    let out = qubivex().args(["system", "--json"]).output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["profile"]["cpu"]["arch"].is_string());
    assert!(v["snapshot"]["ram_available"].is_object());
}
