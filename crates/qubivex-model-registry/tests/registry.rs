//! Installs, recovery from a crash at each step, and removal, on copies of a fixture.

#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use qubivex_model::ArchitectureRegistry;
use qubivex_model_registry::{InstallState, ModelSource, REGISTRY_FILE, Registry, RegistryError};

const FIXTURE: &str = "qwen3-moe-tiny";

struct Tmp(PathBuf);

impl Tmp {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let d = std::env::temp_dir().join(format!(
            "qubivex-registry-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&d).unwrap();
        Self(d)
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn arch() -> ArchitectureRegistry {
    ArchitectureRegistry::with_builtins()
}

fn copy_fixture(to: &Path) {
    let from = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/models")
        .join(FIXTURE);
    fs::create_dir_all(to).unwrap();
    for e in fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        fs::copy(e.path(), to.join(e.file_name())).unwrap();
    }
}

fn source(revision: &str) -> Option<ModelSource> {
    Some(ModelSource {
        provider: "huggingface".into(),
        repository: "org/model".into(),
        revision: revision.into(),
    })
}

fn state(r: &Registry, id: &str) -> InstallState {
    r.get(id).unwrap().unwrap().state
}

#[test]
fn an_install_moves_a_valid_model_into_place() {
    let t = Tmp::new();
    let r = Registry::open(&t.0).unwrap();
    let i = r.begin_install(FIXTURE, source("abc")).unwrap();
    assert_eq!(state(&r, FIXTURE), InstallState::Downloading);
    copy_fixture(i.staging());
    let rec = i.commit(&arch()).unwrap();
    assert_eq!(rec.state, InstallState::Installed);
    assert_eq!(rec.source, source("abc"));
    assert!(t.0.join(FIXTURE).join("manifest.json").is_file());
    assert!(!t.0.join(".staging").join(FIXTURE).exists());
    // Installed once only.
    assert!(matches!(
        r.begin_install(FIXTURE, source("abc")),
        Err(RegistryError::AlreadyInstalled(_))
    ));
}

#[test]
fn a_model_that_fails_validation_stays_out_of_place() {
    let t = Tmp::new();
    let r = Registry::open(&t.0).unwrap();
    let i = r.begin_install(FIXTURE, None).unwrap();
    fs::write(i.staging().join("manifest.json"), b"{}").unwrap();
    assert!(matches!(
        i.commit(&arch()),
        Err(RegistryError::Validation { .. })
    ));
    assert!(matches!(
        state(&r, FIXTURE),
        InstallState::ValidationFailed { .. }
    ));
    assert!(!t.0.join(FIXTURE).exists());
    // A retry starts clean: bad files are not resumed.
    let i = r.begin_install(FIXTURE, None).unwrap();
    assert_eq!(fs::read_dir(i.staging()).unwrap().count(), 0);
}

#[test]
fn a_model_whose_manifest_names_another_id_is_refused() {
    let t = Tmp::new();
    let r = Registry::open(&t.0).unwrap();
    let i = r.begin_install("something-else", None).unwrap();
    copy_fixture(i.staging());
    let e = i.commit(&arch()).unwrap_err();
    assert!(e.to_string().contains("manifest says"), "{e}");
    assert!(!t.0.join("something-else").exists());
}

#[test]
fn a_crash_while_downloading_keeps_the_files_for_the_same_source_only() {
    let t = Tmp::new();
    let r = Registry::open(&t.0).unwrap();
    let part = |r: &Registry, s| {
        let i = r.begin_install(FIXTURE, source(s)).unwrap();
        let exists = i.staging().join("part").exists();
        fs::write(i.staging().join("part"), b"x").unwrap();
        drop(i); // what a crash leaves: the lock released, the record as it was
        exists
    };
    assert!(!part(&r, "abc"));
    assert_eq!(state(&r, FIXTURE), InstallState::Downloading);
    assert!(part(&r, "abc"), "the same source resumes");
    assert!(!part(&r, "def"), "another revision starts again");
}

#[test]
fn a_failed_download_is_recorded_and_can_be_retried() {
    let t = Tmp::new();
    let r = Registry::open(&t.0).unwrap();
    let i = r.begin_install(FIXTURE, source("abc")).unwrap();
    fs::write(i.staging().join("part"), b"x").unwrap();
    i.fail("connection reset").unwrap();
    assert_eq!(
        state(&r, FIXTURE),
        InstallState::DownloadFailed {
            reason: "connection reset".into()
        }
    );
    let i = r.begin_install(FIXTURE, source("abc")).unwrap();
    assert!(i.staging().join("part").exists());
}

#[test]
fn a_crash_after_the_rename_is_recovered_with_its_source() {
    let t = Tmp::new();
    let r = Registry::open(&t.0).unwrap();
    let i = r.begin_install(FIXTURE, source("abc")).unwrap();
    copy_fixture(i.staging());
    fs::rename(i.staging(), t.0.join(FIXTURE)).unwrap();
    drop(i);
    let done = r.reconcile(&arch()).unwrap();
    assert_eq!(done.recovered, [FIXTURE]);
    let rec = r.get(FIXTURE).unwrap().unwrap();
    assert_eq!(rec.state, InstallState::Installed);
    assert_eq!(rec.source, source("abc"));
}

#[test]
fn one_install_at_a_time_and_reconcile_leaves_a_running_one_alone() {
    let t = Tmp::new();
    let r = Registry::open(&t.0).unwrap();
    let i = r.begin_install(FIXTURE, None).unwrap();
    assert!(matches!(
        r.begin_install(FIXTURE, None),
        Err(RegistryError::InstallInProgress(_))
    ));
    assert!(matches!(
        r.remove(FIXTURE),
        Err(RegistryError::InstallInProgress(_))
    ));
    copy_fixture(i.staging());
    fs::rename(i.staging(), t.0.join(FIXTURE)).unwrap();
    assert!(r.reconcile(&arch()).unwrap().recovered.is_empty());
    assert_eq!(state(&r, FIXTURE), InstallState::Downloading);
}

#[test]
fn reconcile_adopts_drops_and_skips() {
    let t = Tmp::new();
    copy_fixture(&t.0.join("by-hand"));
    fs::create_dir(t.0.join("broken")).unwrap();
    fs::write(t.0.join("broken/manifest.json"), b"{}").unwrap();
    let r = Registry::open(&t.0).unwrap();
    let done = r.reconcile(&arch()).unwrap();
    assert_eq!(done.adopted, [FIXTURE]);
    assert_eq!(done.skipped.len(), 1);
    assert_eq!(r.get(FIXTURE).unwrap().unwrap().dir, "by-hand");

    // An installed model that turned invalid keeps its record, so it can be removed.
    fs::write(t.0.join("by-hand/manifest.json"), b"{}").unwrap();
    assert!(r.reconcile(&arch()).unwrap().dropped.is_empty());
    assert!(r.get(FIXTURE).unwrap().is_some());

    fs::remove_dir_all(t.0.join("by-hand")).unwrap();
    assert_eq!(r.reconcile(&arch()).unwrap().dropped, [FIXTURE]);
    assert!(r.list().unwrap().is_empty());
}

#[test]
fn a_manifest_in_the_models_directory_itself_is_never_adopted() {
    let t = Tmp::new();
    copy_fixture(&t.0);
    let r = Registry::open(&t.0).unwrap();
    assert!(r.reconcile(&arch()).unwrap().adopted.is_empty());
    assert!(r.list().unwrap().is_empty());
}

#[test]
fn removal_deletes_the_model_and_nothing_else() {
    let t = Tmp::new();
    copy_fixture(&t.0.join("keep"));
    let r = Registry::open(&t.0).unwrap();
    r.reconcile(&arch()).unwrap();
    let i = r.begin_install("other", None).unwrap();
    fs::write(i.staging().join("part"), b"x").unwrap();
    i.fail("stopped").unwrap();
    r.remove("other").unwrap();
    assert!(!t.0.join(".staging/other").exists());
    assert!(r.get("other").unwrap().is_none());
    assert!(t.0.join("keep/manifest.json").is_file());
    r.remove(FIXTURE).unwrap();
    assert!(!t.0.join("keep").exists());
    assert!(matches!(r.remove(FIXTURE), Err(RegistryError::NotFound(_))));
}

#[test]
fn a_crash_during_removal_never_leaves_a_half_deleted_model_installed() {
    let t = Tmp::new();
    copy_fixture(&t.0.join("m"));
    let r = Registry::open(&t.0).unwrap();
    r.reconcile(&arch()).unwrap();
    // The state between the move to .trash and the record's removal.
    fs::create_dir(t.0.join(".trash")).unwrap();
    fs::rename(t.0.join("m"), t.0.join(".trash/m-1")).unwrap();
    assert_eq!(r.reconcile(&arch()).unwrap().dropped, [FIXTURE]);
    assert!(!t.0.join(".trash").exists());
}

#[cfg(unix)]
#[test]
fn removal_refuses_a_symbolic_link() {
    let t = Tmp::new();
    let elsewhere = Tmp::new();
    copy_fixture(&elsewhere.0);
    std::os::unix::fs::symlink(&elsewhere.0, t.0.join("link")).unwrap();
    let r = Registry::open(&t.0).unwrap();
    r.reconcile(&arch()).unwrap();
    assert!(matches!(r.remove(FIXTURE), Err(RegistryError::Refused(_))));
    assert!(elsewhere.0.join("manifest.json").is_file());
}

#[test]
fn ids_and_records_cannot_escape_the_models_directory() {
    let t = Tmp::new();
    let r = Registry::open(&t.0).unwrap();
    for id in ["", ".", "..", "../x", "a/b", ".hidden", "a b"] {
        assert!(
            matches!(r.begin_install(id, None), Err(RegistryError::InvalidId(_))),
            "{id:?}"
        );
    }
    for dir in ["../victim", "/etc", "a/b", "."] {
        let body = serde_json::json!({
            "schema_version": 1,
            "models": { "m": {
                "id": "m", "dir": dir, "state": { "kind": "installed" },
                "source": null, "updated_at": 0
            }}
        });
        fs::write(t.0.join(REGISTRY_FILE), body.to_string()).unwrap();
        assert!(
            matches!(r.remove("m"), Err(RegistryError::Corrupt { .. })),
            "{dir:?}"
        );
    }
}

#[test]
fn a_corrupt_or_newer_registry_is_refused_and_left_as_it_is() {
    let t = Tmp::new();
    let r = Registry::open(&t.0).unwrap();
    for body in [
        "not json".to_owned(),
        r#"{"schema_version": 2, "models": {}}"#.to_owned(),
    ] {
        fs::write(t.0.join(REGISTRY_FILE), &body).unwrap();
        assert!(matches!(r.list(), Err(RegistryError::Corrupt { .. })));
        assert!(r.begin_install("m", None).is_err());
        assert_eq!(fs::read_to_string(t.0.join(REGISTRY_FILE)).unwrap(), body);
    }
}

#[test]
fn concurrent_updates_lose_nothing() {
    let t = Tmp::new();
    let root = t.0.clone();
    let workers: Vec<_> = (0..16)
        .map(|n| {
            let root = root.clone();
            std::thread::spawn(move || {
                let r = Registry::open(root).unwrap();
                r.begin_install(&format!("m{n}"), None)
                    .unwrap()
                    .fail("stopped")
                    .unwrap();
            })
        })
        .collect();
    for w in workers {
        w.join().unwrap();
    }
    assert_eq!(Registry::open(root).unwrap().list().unwrap().len(), 16);
}
