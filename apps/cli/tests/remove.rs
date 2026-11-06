//! `qubivex models remove` and `validate` through the real binary, on a copy of a fixture.

#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

const ID: &str = "qwen3-moe-tiny";

fn qubivex(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_qubivex"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn remove_needs_yes_and_validate_is_verify() {
    let root = std::env::temp_dir().join(format!("qubivex-cli-remove-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let model = root.join(ID);
    fs::create_dir_all(&model).unwrap();
    let from = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/models")
        .join(ID);
    for e in fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        fs::copy(e.path(), model.join(e.file_name())).unwrap();
    }
    let dir = root.to_str().unwrap();

    assert!(
        qubivex(&["models", "validate", ID, "--dir", dir])
            .status
            .success()
    );

    let dry = qubivex(&["models", "remove", ID, "--dir", dir]);
    assert!(!dry.status.success());
    assert!(String::from_utf8_lossy(&dry.stderr).contains("--yes"));
    assert!(
        model.join("manifest.json").is_file(),
        "no --yes, no deletion"
    );

    let done = qubivex(&["models", "remove", ID, "--dir", dir, "--yes"]);
    assert!(done.status.success(), "{done:?}");
    assert!(!model.exists());

    let again = qubivex(&["models", "remove", ID, "--dir", dir, "--yes"]);
    assert!(!again.status.success());
    fs::remove_dir_all(&root).unwrap();
}
