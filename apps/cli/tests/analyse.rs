//! `qubivex models analyse`, `migrate` and file checks in `verify`, through the real
//! binary, on copies of fixtures.

#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{Value, json};

fn qubivex(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_qubivex"))
        .args(args)
        .output()
        .unwrap()
}

fn stdout(o: &Output) -> String {
    assert!(o.status.success(), "{o:?}");
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    assert!(!o.status.success(), "{o:?}");
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// A models directory holding a copy of one fixture.
fn models_dir(id: &str, test: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("qubivex-cli-{test}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let model = root.join(id);
    fs::create_dir_all(&model).unwrap();
    let from = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/models")
        .join(id);
    for e in fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        fs::copy(e.path(), model.join(e.file_name())).unwrap();
    }
    root
}

fn manifest(root: &Path, id: &str) -> Value {
    serde_json::from_str(&fs::read_to_string(root.join(id).join("manifest.json")).unwrap()).unwrap()
}

fn analyse(id: &str, dir: &str, budget: &str) -> Value {
    let o = qubivex(&[
        "models",
        "analyse",
        id,
        "--dir",
        dir,
        "--ram-budget",
        budget,
        "--ctx",
        "64",
        "--json",
    ]);
    serde_json::from_str(&stdout(&o)).unwrap()
}

#[test]
fn analyse_answers_each_way_and_a_no_is_not_a_failure() {
    const ID: &str = "qwen3-moe-tiny";
    let root = models_dir(ID, "analyse");
    let dir = root.to_str().unwrap();

    let big = analyse(ID, dir, "8GiB");
    assert_eq!(big["assessment"]["verdict"], "fits");
    let need = &big["assessment"]["requirements"];
    let (floor, all) = (
        need["at_context"].as_u64().unwrap(),
        need["in_memory"].as_u64().unwrap(),
    );
    assert!(floor < all);
    assert!(big["disk"].as_u64().unwrap() > 0);
    assert_eq!(big["backends"][0]["name"], "cpu");

    let some = analyse(ID, dir, &(all - 1).to_string());
    assert_eq!(some["assessment"]["verdict"], "streams");

    let none = analyse(ID, dir, &(floor - 1).to_string());
    assert_eq!(none["assessment"]["verdict"], "does_not_fit");
    assert_eq!(none["assessment"]["plan"], Value::Null);

    let text = stdout(&qubivex(&[
        "models",
        "analyse",
        ID,
        "--dir",
        dir,
        "--ram-budget",
        "1",
        "--ctx",
        "64",
    ]));
    assert!(text.contains("Verdict  does not fit"), "{text}");
    assert!(text.contains("RAM to start"), "{text}");
    assert!(!text.contains("runs this model"), "{text}");
    assert_eq!(text.matches("does not fit").count(), 1, "{text}");
    assert!(!text.contains("shows the placement"), "{text}");

    let text = stdout(&qubivex(&[
        "models",
        "analyze",
        ID,
        "--dir",
        dir,
        "--ram-budget",
        "8GiB",
        "--ctx",
        "64",
    ]));
    assert!(
        text.contains("Verdict  runs here, everything in RAM\n"),
        "{text}"
    );
    assert!(text.contains("Source   local"), "{text}");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn analyse_says_when_it_shortened_the_context_itself() {
    const ID: &str = "dense-tiny";
    let root = models_dir(ID, "shortened");
    let dir = root.to_str().unwrap();
    // A budget that starts the model but cannot hold its whole context.
    let probe = analyse(ID, dir, "8GiB");
    let start = probe["assessment"]["requirements"]["minimum"]
        .as_u64()
        .unwrap();
    let budget = (start + 1024).to_string();
    let text = stdout(&qubivex(&[
        "models",
        "analyse",
        ID,
        "--dir",
        dir,
        "--ram-budget",
        &budget,
    ]));
    assert!(text.contains("with the context shortened from"), "{text}");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn analyse_changes_nothing_on_disk() {
    const ID: &str = "dense-tiny";
    let root = models_dir(ID, "readonly");
    let dir = root.to_str().unwrap();
    let before = fs::read(root.join(ID).join("manifest.json")).unwrap();
    analyse(ID, dir, "8GiB");
    assert_eq!(
        fs::read(root.join(ID).join("manifest.json")).unwrap(),
        before
    );
    assert_eq!(
        fs::read_dir(&root).unwrap().count(),
        1,
        "no registry appears"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn analyse_reports_a_model_it_cannot_read_as_an_error() {
    let root = models_dir("dense-tiny", "unknown");
    let e = stderr(&qubivex(&[
        "models",
        "analyse",
        "nothing",
        "--dir",
        root.to_str().unwrap(),
    ]));
    assert!(e.contains("nothing"), "{e}");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn migrate_previews_then_writes_then_has_nothing_to_do() {
    const ID: &str = "dense-tiny";
    let root = models_dir(ID, "migrate");
    let dir = root.to_str().unwrap();
    let path = root.join(ID).join("manifest.json");
    let before = fs::read(&path).unwrap();

    let preview: Value =
        serde_json::from_str(&stdout(&qubivex(&["models", "migrate", ID, "--dir", dir]))).unwrap();
    assert_eq!(preview["schema_version"], "1.1.0");
    assert_eq!(fs::read(&path).unwrap(), before, "a preview writes nothing");

    let out = stdout(&qubivex(&[
        "models", "migrate", ID, "--dir", dir, "--write",
    ]));
    assert!(out.contains("from schema 1.0.0 to 1.1.0"), "{out}");
    assert!(out.contains("4 of 4 files"), "{out}");
    let m = manifest(&root, ID);
    assert_eq!(m["schema_version"], "1.1.0");
    assert_eq!(
        m["weights"]["shards"][0]["sha256"].as_str().unwrap().len(),
        64
    );
    assert_eq!(m["files"].as_array().unwrap().len(), 3);

    let again = stdout(&qubivex(&[
        "models", "migrate", ID, "--dir", dir, "--write",
    ]));
    assert!(again.contains("already at schema 1.1.0"), "{again}");

    let v = stdout(&qubivex(&["models", "verify", ID, "--dir", dir]));
    assert!(v.contains("ok: 4 files match"), "{v}");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn migrate_takes_the_source_from_the_registry() {
    const ID: &str = "dense-tiny";
    let root = models_dir(ID, "source");
    let dir = root.to_str().unwrap();
    let source = json!({ "provider": "huggingface", "repository": "acme/tiny", "revision": "abc" });
    fs::write(
        root.join("registry.json"),
        json!({
            "schema_version": 1,
            "models": { ID: {
                "id": ID, "dir": ID, "state": { "kind": "installed" },
                "source": source, "updated_at": 1
            }}
        })
        .to_string(),
    )
    .unwrap();
    let out = stdout(&qubivex(&[
        "models", "migrate", ID, "--dir", dir, "--write",
    ]));
    assert!(out.contains("huggingface acme/tiny @ abc"), "{out}");
    assert_eq!(manifest(&root, ID)["source"], source);
    let shown = stdout(&qubivex(&["models", "inspect", ID, "--dir", dir]));
    assert!(
        shown.contains("Source   huggingface acme/tiny @ abc"),
        "{shown}"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn migrate_ignores_a_registry_record_of_another_directory() {
    const ID: &str = "dense-tiny";
    let root = models_dir(ID, "otherdir");
    let dir = root.to_str().unwrap();
    fs::create_dir(root.join("elsewhere")).unwrap();
    fs::write(
        root.join("registry.json"),
        json!({
            "schema_version": 1,
            "models": { ID: {
                "id": ID, "dir": "elsewhere", "state": { "kind": "installed" },
                "source": { "provider": "p", "repository": "r", "revision": "v" },
                "updated_at": 1
            }}
        })
        .to_string(),
    )
    .unwrap();
    stdout(&qubivex(&[
        "models", "migrate", ID, "--dir", dir, "--write",
    ]));
    assert!(manifest(&root, ID).get("source").is_none());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn migrating_keeps_a_models_expert_checksums_fresh() {
    const ID: &str = "qwen3-moe-tiny";
    let root = models_dir(ID, "sums");
    let dir = root.to_str().unwrap();
    let first = stdout(&qubivex(&["models", "verify", ID, "--dir", dir]));
    assert!(first.contains("no file checksums"), "{first}");
    assert!(first.contains("recorded checksums"), "{first}");

    stdout(&qubivex(&[
        "models", "migrate", ID, "--dir", dir, "--write",
    ]));
    let v = stdout(&qubivex(&["models", "verify", ID, "--dir", dir]));
    assert!(v.contains("files match"), "{v}");
    assert!(v.contains("experts match"), "{v}");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn verify_names_a_changed_file_and_record_accepts_it() {
    const ID: &str = "dense-tiny";
    let root = models_dir(ID, "verify");
    let dir = root.to_str().unwrap();
    stdout(&qubivex(&[
        "models", "migrate", ID, "--dir", dir, "--write",
    ]));

    let tokenizer = root.join(ID).join("tokenizer_config.json");
    let mut bytes = fs::read(&tokenizer).unwrap();
    bytes.push(b'\n');
    fs::write(&tokenizer, bytes).unwrap();

    let e = stderr(&qubivex(&["models", "verify", ID, "--dir", dir]));
    assert!(e.contains("1 of 4 files do not match"), "{e}");
    assert!(e.contains("tokenizer_config.json"), "{e}");
    assert!(e.contains("--record"), "{e}");

    let migrate = stdout(&qubivex(&[
        "models", "migrate", ID, "--dir", dir, "--write",
    ]));
    assert!(
        migrate.contains("already"),
        "migrating must not accept the change: {migrate}"
    );
    assert!(
        !qubivex(&["models", "verify", ID, "--dir", dir])
            .status
            .success()
    );

    stdout(&qubivex(&[
        "models", "verify", ID, "--dir", dir, "--record",
    ]));
    stdout(&qubivex(&["models", "verify", ID, "--dir", dir]));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn verify_finds_a_flipped_weight_byte() {
    const ID: &str = "dense-tiny";
    let root = models_dir(ID, "weights");
    let dir = root.to_str().unwrap();
    stdout(&qubivex(&[
        "models", "migrate", ID, "--dir", dir, "--write",
    ]));

    let weights = root.join(ID).join("model.safetensors");
    let mut bytes = fs::read(&weights).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(&weights, bytes).unwrap();

    let e = stderr(&qubivex(&["models", "verify", ID, "--dir", dir]));
    assert!(e.contains("model.safetensors"), "{e}");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_fresh_import_writes_the_current_schema_with_checksums() {
    const ID: &str = "dense-tiny";
    let root = models_dir(ID, "import");
    let model = root.join(ID);
    fs::remove_file(model.join("manifest.json")).unwrap();
    let dir = model.to_str().unwrap();
    stdout(&qubivex(&["models", "import", dir, "--id", ID, "--write"]));
    let m = manifest(&root, ID);
    assert_eq!(m["schema_version"], "1.1.0");
    assert!(m["weights"]["shards"][0]["sha256"].is_string());

    stdout(&qubivex(&[
        "models",
        "import",
        dir,
        "--id",
        ID,
        "--write",
        "--force",
        "--no-checksums",
    ]));
    assert!(manifest(&root, ID)["weights"]["shards"][0]["sha256"].is_null());
    fs::remove_dir_all(root).unwrap();
}
