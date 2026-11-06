//! `qubivex models pull` and `search` through the real binary, against a local hub.

#![allow(clippy::unwrap_used)]

#[path = "../../../crates/qubivex-model-download/tests/support/mod.rs"]
mod support;

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use support::{Fault, Hub, REPO, fixture};

fn qubivex(hub: &Hub, home: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_qubivex"))
        .args(args)
        .env("HF_ENDPOINT", &hub.url)
        .env("HOME", home)
        .env_remove("HF_TOKEN")
        .env_remove("HUGGING_FACE_HUB_TOKEN")
        .env_remove("HF_HOME")
        .env_remove("HF_TOKEN_PATH")
        .env_remove("QUBIVEX_MODELS_DIRS")
        .output()
        .unwrap()
}

fn text(o: &[u8]) -> String {
    String::from_utf8_lossy(o).into_owned()
}

fn temp(name: &str) -> std::path::PathBuf {
    let root = std::env::temp_dir().join(format!("qubivex-cli-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    root
}

#[test]
fn pull_list_validate_and_remove() {
    let root = temp("pull");
    let hub = Hub::start(fixture("dense-tiny"));
    let dir = root.to_str().unwrap();
    // One dropped connection, to see it survive the whole path.
    hub.fault("model.safetensors", vec![Fault::Cut(50_000)]);

    let pulled = qubivex(&hub, &root, &["models", "pull", REPO, "--dir", dir]);
    assert!(pulled.status.success(), "{pulled:?}");
    assert!(text(&pulled.stdout).contains("Installed dense-tiny"));

    let listed = qubivex(&hub, &root, &["models", "list", "--dir", dir]);
    assert!(text(&listed.stdout).contains("dense-tiny"), "{listed:?}");
    let registry = fs::read_to_string(root.join("registry.json")).unwrap();
    assert!(registry.contains(REPO));

    let again = qubivex(&hub, &root, &["models", "pull", REPO, "--dir", dir]);
    assert!(!again.status.success());
    assert!(
        text(&again.stderr).contains("already installed"),
        "{again:?}"
    );

    let inspect = qubivex(
        &hub,
        &root,
        &["models", "inspect", "dense-tiny", "--dir", dir],
    );
    assert!(inspect.status.success(), "{inspect:?}");
    let gone = qubivex(
        &hub,
        &root,
        &["models", "remove", "dense-tiny", "--dir", dir, "--yes"],
    );
    assert!(gone.status.success(), "{gone:?}");
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn a_gguf_pull_says_how_to_supply_what_the_file_lacks() {
    let root = temp("pull-gguf");
    let hub = Hub::start(fixture("olmoe-tiny-gguf"));
    let dir = root.to_str().unwrap();
    let file = "olmoe-tiny-q8_0.gguf";

    let bare = qubivex(
        &hub,
        &root,
        &["models", "pull", REPO, "--file", file, "--dir", dir],
    );
    assert!(!bare.status.success());
    assert!(
        text(&bare.stderr).contains("--set norm_topk_prob=VALUE"),
        "{bare:?}"
    );

    let args = [
        "models",
        "pull",
        REPO,
        "--file",
        file,
        "--dir",
        dir,
        "--set",
        "norm_topk_prob=false",
    ];
    let set = qubivex(&hub, &root, &args);
    assert!(set.status.success(), "{set:?}");
    assert_eq!(hub.downloads(file).len(), 1, "the retry reused the file");
    assert!(root.join("olmoe-tiny-q8_0/experts.qvxsum").is_file());
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn search_prints_a_table_and_json() {
    let root = temp("search");
    let hub = Hub::start(Vec::new());
    let table = qubivex(&hub, &root, &["models", "search", "tiny", "--limit", "2"]);
    assert!(table.status.success(), "{table:?}");
    let out = text(&table.stdout);
    assert!(
        out.contains("acme/a-GGUF") && out.contains("safetensors, gated"),
        "{out}"
    );
    let json = qubivex(&hub, &root, &["models", "search", "tiny", "--json"]);
    let v: serde_json::Value = serde_json::from_slice(&json.stdout).unwrap();
    assert_eq!(v[0]["repository"], "acme/a-GGUF");
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn a_refused_endpoint_fails_before_any_network() {
    let root = temp("endpoint");
    let hub = Hub::start(Vec::new());
    let o = Command::new(env!("CARGO_BIN_EXE_qubivex"))
        .args(["models", "search", "x"])
        .env("HF_ENDPOINT", "http://example.com")
        .env("HOME", &root)
        .output()
        .unwrap();
    assert!(!o.status.success());
    assert!(text(&o.stderr).contains("plain http"), "{o:?}");
    assert!(hub.seen().is_empty());
    fs::remove_dir_all(&root).unwrap();
}
