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
