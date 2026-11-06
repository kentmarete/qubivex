//! Manifest 1.1: every 1.0 manifest committed here is a migration test, since fixtures
//! stay at 1.0 on purpose.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use qubivex_model::manifest::{SCHEMA_VERSION, SemVer, Shard};
use qubivex_model::{
    ArchitectureRegistry, MANIFEST_FILE, Manifest, ModelMetadata, ModelSource, fill_checksums,
    sha256_file, verify_files,
};
use serde_json::{Value, json};

const FIXTURES: [&str; 4] = [
    "dense-tiny",
    "olmoe-tiny",
    "olmoe-tiny-gguf",
    "qwen3-moe-tiny",
];

fn fixture(id: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/models")
        .join(id)
}

fn parse(v: &Value) -> Result<Manifest, String> {
    Manifest::parse(&v.to_string(), Path::new("manifest.json")).map_err(|e| e.to_string())
}

fn manifest_json(id: &str) -> Value {
    serde_json::from_str(&fs::read_to_string(fixture(id).join(MANIFEST_FILE)).unwrap()).unwrap()
}

/// A private copy of a fixture.
fn copy(id: &str, test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qubivex-manifest-{test}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    for e in fs::read_dir(fixture(id)).unwrap() {
        let e = e.unwrap();
        fs::copy(e.path(), dir.join(e.file_name())).unwrap();
    }
    dir
}

fn source() -> ModelSource {
    ModelSource {
        provider: "huggingface".into(),
        repository: "acme/tiny".into(),
        revision: "0123456789abcdef".into(),
    }
}

#[test]
fn every_1_0_fixture_upgrades_and_still_validates() {
    let arch = ArchitectureRegistry::with_builtins();
    for id in FIXTURES {
        let dir = fixture(id);
        let old = Manifest::load(&dir.join(MANIFEST_FILE)).unwrap();
        assert_eq!(old.schema_version.minor, 0, "{id}: fixtures stay at 1.0");

        let new = old.clone().upgraded();
        assert_eq!(new.schema_version, SCHEMA_VERSION, "{id}");
        // Only the version moves: nothing the 1.0 file said is lost.
        let mut back = new.clone();
        back.schema_version = old.schema_version;
        assert_eq!(back, old, "{id}");

        let text = serde_json::to_string(&new).unwrap();
        let reread = Manifest::parse(&text, Path::new("m")).unwrap();
        assert_eq!(reread, new, "{id}: round trip");
        ModelMetadata::load_with(&dir, reread, &arch).unwrap();
    }
}

#[test]
fn unset_source_and_files_stay_out_of_the_json() {
    let old = Manifest::load(&fixture("dense-tiny").join(MANIFEST_FILE)).unwrap();
    let v = serde_json::to_value(&old).unwrap();
    assert!(v.get("source").is_none());
    assert!(v.get("files").is_none());
}

#[test]
fn source_and_files_need_1_1() {
    let mut m = manifest_json("dense-tiny");
    m["source"] = serde_json::to_value(source()).unwrap();
    assert!(parse(&m).unwrap_err().contains("need 1.1"));
    m["schema_version"] = json!("1.1.0");
    assert_eq!(parse(&m).unwrap().source, Some(source()));

    let mut m = manifest_json("dense-tiny");
    m["files"] = json!([{ "path": "tokenizer.json" }]);
    assert!(parse(&m).unwrap_err().contains("need 1.1"));
}

#[test]
fn a_newer_manifest_is_refused_by_version_not_by_its_fields() {
    let mut m = manifest_json("dense-tiny");
    m["schema_version"] = json!("1.2.0");
    m["some_field_from_the_future"] = json!(1);
    let e = parse(&m).unwrap_err();
    assert!(e.contains("newer than this build"), "{e}");

    m["schema_version"] = json!("2.0.0");
    let e = parse(&m).unwrap_err();
    assert!(e.contains("not supported"), "{e}");
}

#[test]
fn a_later_patch_of_a_known_minor_reads() {
    let mut m = manifest_json("dense-tiny");
    m["schema_version"] = json!("1.1.7");
    assert_eq!(
        parse(&m).unwrap().schema_version,
        SemVer {
            major: 1,
            minor: 1,
            patch: 7
        }
    );
}

#[test]
fn a_source_cannot_have_empty_fields() {
    let mut m = manifest_json("dense-tiny");
    m["schema_version"] = json!("1.1.0");
    for field in ["provider", "repository", "revision"] {
        let mut s = serde_json::to_value(source()).unwrap();
        s[field] = json!("");
        m["source"] = s;
        assert!(parse(&m).unwrap_err().contains("source needs"), "{field}");
    }
    m["source"] = json!({ "provider": "a", "repository": "b", "revision": "c", "url": "d" });
    assert!(parse(&m).unwrap_err().contains("unknown field"));
}

#[test]
fn listed_files_get_the_checks_shards_get() {
    let mut m = manifest_json("dense-tiny");
    m["schema_version"] = json!("1.1.0");
    for path in ["../secret", "/etc/passwd", "a/../../b"] {
        m["files"] = json!([{ "path": path }]);
        assert!(parse(&m).unwrap_err().contains("relative"), "{path}");
    }
    m["files"] = json!([{ "path": "tokenizer.json", "sha256": "ABC" }]);
    assert!(parse(&m).unwrap_err().contains("sha256"));
}

#[test]
fn checksums_cover_weights_config_and_tokenizer() {
    let dir = copy("dense-tiny", "fill");
    let mut m = Manifest::load(&dir.join(MANIFEST_FILE)).unwrap().upgraded();
    fill_checksums(&mut m, &dir, &BTreeMap::new()).unwrap();

    let listed: Vec<&str> = m.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(
        listed,
        ["config.json", "tokenizer.json", "tokenizer_config.json"]
    );
    for f in m.weights.shards.iter().chain(&m.files) {
        let path = dir.join(&f.path);
        assert_eq!(
            f.size,
            Some(fs::metadata(&path).unwrap().len()),
            "{}",
            f.path
        );
        assert_eq!(
            f.sha256.as_ref(),
            Some(&sha256_file(&path).unwrap()),
            "{}",
            f.path
        );
    }
    let check = verify_files(&m, &dir);
    assert_eq!((check.matched, check.unrecorded), (4, 0));
    assert!(check.problems.is_empty());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_known_checksum_saves_reading_the_file() {
    let dir = copy("dense-tiny", "known");
    let mut m = Manifest::load(&dir.join(MANIFEST_FILE)).unwrap().upgraded();
    let shard = m.weights.shards[0].path.clone();
    let known = BTreeMap::from([(shard, "ab".repeat(32))]);
    fill_checksums(&mut m, &dir, &known).unwrap();
    assert_eq!(m.weights.shards[0].sha256, Some("ab".repeat(32)));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn filling_never_replaces_a_recorded_checksum() {
    let dir = copy("dense-tiny", "keep");
    let mut m = Manifest::load(&dir.join(MANIFEST_FILE)).unwrap().upgraded();
    fill_checksums(&mut m, &dir, &BTreeMap::new()).unwrap();
    let recorded = m.clone();

    fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    fill_checksums(&mut m, &dir, &BTreeMap::new()).unwrap();
    assert_eq!(m, recorded, "the changed file must still fail verification");

    let check = verify_files(&m, &dir);
    assert_eq!(check.problems.len(), 1);
    assert_eq!(check.problems[0].path, "tokenizer.json");
    assert_eq!(check.matched, 3);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_missing_file_is_reported_with_the_rest() {
    let dir = copy("dense-tiny", "missing");
    let mut m = Manifest::load(&dir.join(MANIFEST_FILE)).unwrap().upgraded();
    fill_checksums(&mut m, &dir, &BTreeMap::new()).unwrap();
    fs::remove_file(dir.join("tokenizer.json")).unwrap();
    fs::write(dir.join("config.json"), b"{}").unwrap();

    let check = verify_files(&m, &dir);
    let paths: Vec<&str> = check.problems.iter().map(|p| p.path.as_str()).collect();
    assert_eq!(paths, ["config.json", "tokenizer.json"]);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn files_without_a_checksum_count_as_unrecorded_not_as_passing() {
    let dir = fixture("dense-tiny");
    let m = Manifest::load(&dir.join(MANIFEST_FILE)).unwrap();
    let check = verify_files(&m, &dir);
    assert_eq!((check.matched, check.unrecorded), (0, 1));
}

#[test]
fn saving_replaces_the_file_and_leaves_nothing_behind() {
    let dir = copy("dense-tiny", "save");
    let path = dir.join(MANIFEST_FILE);
    let mut m = Manifest::load(&path).unwrap().upgraded();
    m.source = Some(source());
    m.files.push(Shard {
        path: "tokenizer.json".into(),
        size: None,
        sha256: None,
    });
    m.save(&dir).unwrap();
    assert_eq!(Manifest::load(&path).unwrap(), m);
    let strays: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains("tmp"))
        .collect();
    assert!(strays.is_empty(), "{strays:?}");
    fs::remove_dir_all(dir).unwrap();
}
