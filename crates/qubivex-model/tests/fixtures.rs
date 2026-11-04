//! The committed tiny fixtures (`fixtures/models/`, from `cargo xtask gen-fixtures`)
//! load and validate, each in its own tensor layout and architecture.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use qubivex_model::{ArchitectureRegistry, ImportOptions, ModelMetadata, discover, find, import};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/models")
}

#[test]
fn every_fixture_validates() {
    let reg = ArchitectureRegistry::with_builtins();
    let found = discover(&[root()], &reg);
    let ids: Vec<String> = found.iter().map(|d| d.id().unwrap()).collect();
    assert_eq!(
        ids,
        [
            "dense-tiny",
            "olmoe-tiny",
            "olmoe-tiny-gguf",
            "qwen3-moe-tiny"
        ]
    );
    for d in &found {
        let m = d.model.as_ref().unwrap();
        let b = m.bytes();
        assert_eq!(
            b.total,
            m.tensors.total_bytes(),
            "{}: every tensor has a role",
            d.dir.display()
        );
        let dense = m.manifest.architecture.id == "transformer_dense";
        assert_eq!(
            (
                m.config.dims.layers,
                m.config.dims.experts,
                m.config.dims.experts_per_token
            ),
            if dense { (2, 0, 0) } else { (2, 8, 2) }
        );
        if dense {
            assert_eq!(b.routed_experts, 0, "{}", d.dir.display());
        }
    }
    // One expert: 3 matrices of 32 x 64. bf16: 2 bytes each; Q8_0: 34 bytes per 32.
    let b = |id: &str| find(id, &[root()], &reg).unwrap().bytes().largest_expert;
    assert_eq!(b("olmoe-tiny"), 3 * 32 * 64 * 2);
    assert_eq!(b("qwen3-moe-tiny"), 3 * 32 * 64 * 2);
    assert_eq!(b("olmoe-tiny-gguf"), 3 * 32 * 64 / 32 * 34);
}

#[test]
fn import_reproduces_the_committed_manifests() {
    let reg = ArchitectureRegistry::with_builtins();
    for id in ["olmoe-tiny", "qwen3-moe-tiny", "dense-tiny"] {
        let dir = root().join(id);
        let committed = ModelMetadata::load(&dir, &reg).unwrap().manifest;
        let opts = ImportOptions {
            id: Some(id.to_owned()),
            name: Some(committed.model.name.clone()),
            ..Default::default()
        };
        let imported = import(&dir, &opts, &reg).unwrap().manifest;
        assert_eq!(imported, committed, "{id}");
    }
    // GGUF does not record top-K renormalisation: without --set, import says so.
    let e = import(
        &root().join("olmoe-tiny-gguf"),
        &ImportOptions::default(),
        &reg,
    )
    .unwrap_err();
    assert!(
        e.to_string().contains("overrides.config.norm_topk_prob"),
        "{e}"
    );
}

/// A copy of a fixture directory, removed on drop.
struct Copy(PathBuf);
impl Drop for Copy {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn copy(id: &str, tag: &str) -> Copy {
    let d = std::env::temp_dir().join(format!("qubivex-fixture-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    for e in std::fs::read_dir(root().join(id)).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), d.join(e.file_name())).unwrap();
    }
    Copy(d)
}

/// `transformer_dense` refuses what it does not implement, and a mixture of experts,
/// with a message that names the config key.
#[test]
fn the_dense_architecture_refuses_what_it_does_not_implement() {
    let reg = ArchitectureRegistry::with_builtins();
    for (key, value, want) in [
        (
            "num_experts",
            serde_json::json!(8),
            "transformer_moe, not transformer_dense",
        ),
        ("mlp_bias", serde_json::json!(true), "mlp_bias"),
        (
            "use_sliding_window",
            serde_json::json!(true),
            "sliding-window",
        ),
        ("attention_bias", serde_json::json!(true), "attention_bias"),
    ] {
        let d = copy("dense-tiny", key);
        let path = d.0.join("config.json");
        let mut cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        cfg[key] = value;
        std::fs::write(&path, cfg.to_string()).unwrap();
        let e = ModelMetadata::load(&d.0, &reg).unwrap_err().to_string();
        assert!(e.contains(want), "{key}: {e}");
    }
    // The importer tries the mixture-of-experts architecture first and moves on.
    let imported = import(&root().join("dense-tiny"), &ImportOptions::default(), &reg).unwrap();
    assert_eq!(imported.manifest.architecture.id, "transformer_dense");
}
