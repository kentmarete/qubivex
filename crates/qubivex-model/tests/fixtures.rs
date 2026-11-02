//! The committed tiny fixtures (`fixtures/models/`, from `cargo xtask gen-fixtures`)
//! load and validate, each in its own tensor layout.

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
    assert_eq!(ids, ["olmoe-tiny", "olmoe-tiny-gguf", "qwen3-moe-tiny"]);
    for d in &found {
        let m = d.model.as_ref().unwrap();
        let b = m.bytes();
        assert_eq!(
            b.total,
            m.tensors.total_bytes(),
            "{}: every tensor has a role",
            d.dir.display()
        );
        assert_eq!(
            (
                m.config.dims.layers,
                m.config.dims.experts,
                m.config.dims.experts_per_token
            ),
            (2, 8, 2)
        );
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
    for id in ["olmoe-tiny", "qwen3-moe-tiny"] {
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
