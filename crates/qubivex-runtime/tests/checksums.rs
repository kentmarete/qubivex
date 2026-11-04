//! Per-expert checksums next to the model (ADR-006 amendment 4): recorded once, checked
//! by `ExpertSums::check`, and used by the engine to verify every expert read.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use qubivex_core::{ExpertIdx, LayerIdx};
use qubivex_expert_stream::ExpertSource;
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::checksums::{ExpertSums, SumsState};
use qubivex_runtime::{
    Engine, EngineOptions, ModelExpertSource, Sampling, Verification, VerifyExperts,
};

const IDS: [&str; 3] = ["olmoe-tiny", "olmoe-tiny-gguf", "qwen3-moe-tiny"];
const PROMPT: &[u32] = &[
    72, 101, 108, 108, 111, 44, 32, 101, 120, 112, 101, 114, 116, 115,
];

fn fixture(id: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/models")
        .join(id)
}

/// A copy of a fixture in a fresh temporary directory.
fn copy_model(id: &str, tag: &str) -> PathBuf {
    let dst = std::env::temp_dir().join(format!("qubivex-sums-{}-{id}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dst);
    std::fs::create_dir_all(&dst).unwrap();
    for f in std::fs::read_dir(fixture(id)).unwrap() {
        let f = f.unwrap();
        if f.file_type().unwrap().is_file() {
            std::fs::copy(f.path(), dst.join(f.file_name())).unwrap();
        }
    }
    dst
}

fn meta(dir: &Path) -> ModelMetadata {
    ModelMetadata::load(dir, &ArchitectureRegistry::with_builtins()).unwrap()
}

/// One slot per layer, so every request reads experts.
fn options(m: &ModelMetadata) -> EngineOptions {
    EngineOptions {
        expert_cache_bytes: Some(m.bytes().largest_expert * u64::from(m.config.dims.layers)),
        max_context: Some(64),
        threads: Some(2),
        ..EngineOptions::default()
    }
}

fn tokens(e: &Engine) -> Result<Vec<u32>, String> {
    e.generate(PROMPT, 8, Sampling::default(), |_| true)
        .map_err(|e| e.to_string())
}

/// Flips one byte inside expert (0, 0) in the weight file, keeping the file's size and
/// modification time, as silent corruption on a drive would.
fn corrupt_expert(dir: &Path) {
    let m = meta(dir);
    let src = ModelExpertSource::new(&m).unwrap();
    let key = (LayerIdx(0), ExpertIdx(0));
    let mut bytes = vec![0u8; usize::try_from(src.expert_size(key).unwrap()).unwrap()];
    src.read_expert(key, &mut bytes).unwrap();
    let mid = bytes.len() / 2;
    let window = &bytes[mid..mid + 64];
    for f in m.tensors.files() {
        let mut data = std::fs::read(&f.path).unwrap();
        let hits: Vec<usize> = data
            .windows(64)
            .enumerate()
            .filter(|(_, w)| *w == window)
            .map(|(i, _)| i)
            .collect();
        if let [at] = hits[..] {
            data[at + 7] ^= 0x20;
            let mtime = std::fs::metadata(&f.path).unwrap().modified().unwrap();
            std::fs::write(&f.path, &data).unwrap();
            let file = std::fs::OpenOptions::new()
                .write(true)
                .open(&f.path)
                .unwrap();
            file.set_modified(mtime).unwrap();
            return;
        }
    }
    panic!("expert (0, 0) not found in a single place");
}

#[test]
fn recorded_checksums_verify_reads_and_catch_corruption() {
    for id in IDS {
        let want = tokens(&Engine::load(meta(&fixture(id)), options(&meta(&fixture(id)))).unwrap())
            .unwrap();
        let dir = copy_model(id, "sums");
        let m = meta(&dir);
        assert!(matches!(ExpertSums::load(&m).unwrap(), SumsState::Missing));
        let e = Engine::load(meta(&dir), options(&m)).unwrap();
        assert_eq!(*e.verification(), Verification::Missing, "{id}");
        drop(e);

        let sums = ExpertSums::record(&m, 2).unwrap();
        let n = sums.len();
        assert_eq!(
            n,
            m.config.dims.layers as usize * m.config.dims.experts as usize
        );
        sums.save(&dir).unwrap();
        let SumsState::Fresh(loaded) = ExpertSums::load(&m).unwrap() else {
            panic!("{id}: not fresh");
        };
        assert_eq!(loaded, sums, "{id}: round trip");
        assert!(loaded.check(&m, 2).unwrap().is_empty(), "{id}: clean model");

        // Verified, and the tokens are unchanged.
        let e = Engine::load(meta(&dir), options(&m)).unwrap();
        assert_eq!(*e.verification(), Verification::Verified(n), "{id}");
        assert_eq!(tokens(&e).unwrap(), want, "{id}");
        drop(e);

        // Silent corruption: the file looks unchanged, so the checksums stay fresh; the
        // scrub names the expert, and a verified engine refuses to compute with it.
        corrupt_expert(&dir);
        let SumsState::Fresh(loaded) = ExpertSums::load(&m).unwrap() else {
            panic!("{id}: corruption must not make the sums stale");
        };
        let bad = loaded.check(&m, 2).unwrap();
        assert_eq!(bad.len(), 1, "{id}: {bad:?}");
        assert_eq!(bad[0].key, (LayerIdx(0), ExpertIdx(0)));
        let e = Engine::load(meta(&dir), options(&m)).unwrap();
        let err = tokens(&e).expect_err(id);
        assert!(err.contains("checksum mismatch"), "{id}: {err}");
        drop(e);

        // With a good mirror, the corrupt copy fails over and the tokens are right.
        let mirror = copy_model(id, "mirror");
        let e =
            Engine::load_mirrored(meta(&dir), options(&m), std::slice::from_ref(&mirror)).unwrap();
        assert_eq!(tokens(&e).unwrap(), want, "{id}: through the mirror");
        drop(e);

        // Unverified, the corruption goes unnoticed (it changes the computation, not
        // necessarily the tokens): verification is what makes it visible.
        let opts = EngineOptions {
            verify: VerifyExperts::Off,
            ..options(&m)
        };
        let e = Engine::load(meta(&dir), opts).unwrap();
        assert_eq!(*e.verification(), Verification::Off);
        assert!(tokens(&e).is_ok(), "{id}: unverified reads do not fail");
        for d in [dir, mirror] {
            let _ = std::fs::remove_dir_all(d);
        }
    }
}

/// Weight files that changed after the checksums were recorded make them stale: they
/// are reported and not used, and `Require` refuses to load.
#[test]
fn stale_checksums_are_reported_not_used() {
    let id = "olmoe-tiny";
    let dir = copy_model(id, "stale");
    let m = meta(&dir);
    ExpertSums::record(&m, 1).unwrap().save(&dir).unwrap();
    let weights = &m.tensors.files()[0].path;
    let later = SystemTime::now() + std::time::Duration::from_secs(60);
    std::fs::OpenOptions::new()
        .write(true)
        .open(weights)
        .unwrap()
        .set_modified(later)
        .unwrap();
    let SumsState::Stale(why) = ExpertSums::load(&m).unwrap() else {
        panic!("not stale");
    };
    assert!(why.contains("changed after"), "{why}");
    let e = Engine::load(meta(&dir), options(&m)).unwrap();
    assert!(matches!(e.verification(), Verification::Stale(_)));
    assert!(tokens(&e).is_ok());
    drop(e);
    let opts = EngineOptions {
        verify: VerifyExperts::Require,
        ..options(&m)
    };
    let err = Engine::load(meta(&dir), opts)
        .expect_err("stale must be refused")
        .to_string();
    assert!(err.contains("qubivex models verify"), "{err}");
    let _ = std::fs::remove_dir_all(dir);
}
