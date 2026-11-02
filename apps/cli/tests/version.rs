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

#[test]
fn doctor_without_a_check_is_an_error() {
    let out = qubivex().arg("doctor").output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("pass --storage"));
}

#[cfg(unix)]
#[test]
fn doctor_storage_json() {
    let dir = std::env::temp_dir().join(format!("qubivex-cli-doctor-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = qubivex()
        .args([
            "doctor",
            "--storage",
            "--size-mib",
            "64",
            "--json",
            "--path",
        ])
        .arg(&dir)
        .output()
        .unwrap();
    let leftover = std::fs::read_dir(&dir).unwrap().count();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(leftover, 0);
    let stderr = String::from_utf8_lossy(&out.stderr);
    if out.status.success() {
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert!(v["seq_read_bps"].as_f64().unwrap() > 0.0);
    } else {
        // A runner without 1 GiB headroom free must say so.
        assert!(stderr.contains("free space"), "{stderr}");
    }
}

fn fixtures() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/models")
}

#[test]
fn models_list_and_inspect_fixtures() {
    let out = qubivex()
        .args(["models", "list", "--dir"])
        .arg(fixtures())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let t = String::from_utf8(out.stdout).unwrap();
    for id in ["olmoe-tiny ", "olmoe-tiny-gguf ", "qwen3-moe-tiny "] {
        assert!(
            t.lines().any(|l| l.starts_with(id) && l.ends_with("ok")),
            "{id}: {t}"
        );
    }
    let out = qubivex()
        .args([
            "models",
            "inspect",
            "qwen3-moe-tiny",
            "--json",
            "--tensors",
            "--dir",
        ])
        .arg(fixtures())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["config"]["head_dim"], 32);
    assert_eq!(v["config"]["qk_norm"], "per_head");
    let roles: Vec<&str> = v["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t["role"]["role"].as_str())
        .collect();
    assert!(roles.contains(&"routed_expert_stack"));
    let out = qubivex()
        .args(["models", "inspect", "missing-model", "--dir"])
        .arg(fixtures())
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn models_import_prints_a_manifest() {
    let out = qubivex()
        .args(["models", "import", "--set", "norm_topk_prob=false"])
        .arg(fixtures().join("olmoe-tiny-gguf/olmoe-tiny-q8_0.gguf"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["architecture"]["config"]["tensor_names"], "gguf");
    assert_eq!(v["model"]["id"], "olmoe-tiny-q8_0");
    // Without --set, the error says which --set to pass.
    let out = qubivex()
        .args(["models", "import"])
        .arg(fixtures().join("olmoe-tiny-gguf/olmoe-tiny-q8_0.gguf"))
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--set norm_topk_prob=VALUE"));
    // --write refuses to replace an existing manifest without --force.
    let out = qubivex()
        .args(["models", "import", "--write"])
        .arg(fixtures().join("olmoe-tiny"))
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--force"));
}

#[test]
fn run_generates_on_a_fixture() {
    let out = qubivex()
        .args([
            "run",
            "--model",
            "olmoe-tiny-gguf",
            "--prompt",
            "hi",
            "--max-tokens",
            "4",
            "--dir",
        ])
        .arg(fixtures())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).starts_with("hi"));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("prompt tokens") && err.contains("experts:"),
        "{err}"
    );
    let chat = qubivex()
        .args([
            "run",
            "--model",
            "olmoe-tiny",
            "--chat",
            "--prompt",
            "hi",
            "--max-tokens",
            "2",
            "--quiet",
            "--dir",
        ])
        .arg(fixtures())
        .output()
        .unwrap();
    assert!(
        chat.status.success(),
        "{}",
        String::from_utf8_lossy(&chat.stderr)
    );
}
