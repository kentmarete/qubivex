//! Pulls against a local hub that misbehaves on purpose. Each test ends by asserting
//! what ADR-014 promises: a model is complete in place or absent.

#![allow(clippy::unwrap_used)]

mod support;

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_model_download::{
    DownloadError, HuggingFace, ModelProvider, Opened, Progress, PullOptions, PullRequest, Pulled,
    RepoListing, SearchHit, Token, pull, search,
};
use qubivex_model_registry::{InstallState, Registry, RegistryError};
use support::{Fault, File, Hub, REPO, SHA, fixture};

const WEIGHTS: &str = "model.safetensors";
const ID: &str = "dense-tiny";

struct Tmp(PathBuf);

impl Tmp {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let d = std::env::temp_dir().join(format!(
            "qubivex-download-{}-{}",
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

fn fast() -> PullOptions {
    PullOptions {
        concurrency: 2,
        attempts: 4,
        backoff: Duration::from_millis(1),
        max_backoff: Duration::from_millis(20),
    }
}

fn provider(hub: &Hub) -> HuggingFace {
    HuggingFace::new(&hub.url, None, Duration::from_secs(10)).unwrap()
}

fn request() -> PullRequest {
    PullRequest {
        repository: REPO.to_owned(),
        ..PullRequest::default()
    }
}

fn run(root: &Path, hub: &Hub, req: &PullRequest) -> Result<Pulled, DownloadError> {
    let reg = Registry::open(root).unwrap();
    let arch = ArchitectureRegistry::with_builtins();
    let cancel = AtomicBool::new(false);
    pull(&reg, &provider(hub), req, &fast(), &arch, &cancel, &|_| {})
}

fn state(root: &Path, id: &str) -> Option<InstallState> {
    let reg = Registry::open(root).unwrap();
    reg.get(id).unwrap().map(|r| r.state)
}

fn staged(root: &Path, id: &str) -> PathBuf {
    root.join(".staging").join(id)
}

/// The model loads, and its weights are the hub's bytes.
fn assert_installed(root: &Path, hub: &Hub, id: &str) {
    let dir = root.join(id);
    ModelMetadata::load(&dir, &ArchitectureRegistry::with_builtins()).unwrap();
    assert_eq!(fs::read(dir.join(WEIGHTS)).unwrap(), hub.file(WEIGHTS));
    assert_eq!(state(root, id), Some(InstallState::Installed));
    assert!(!staged(root, id).exists());
}

fn assert_absent(root: &Path, id: &str) {
    assert!(!root.join(id).exists(), "no partial model in place");
}

#[test]
fn a_clean_pull_installs_exactly_what_a_model_needs() {
    let tmp = Tmp::new();
    let mut files = fixture(ID);
    for path in ["README.md", "pytorch_model.bin", "onnx/model.onnx"] {
        files.push(File {
            path: path.to_owned(),
            bytes: vec![1; 8],
            lfs: false,
        });
    }
    let hub = Hub::start(files);
    let pulled = run(&tmp.0, &hub, &request()).unwrap();
    assert_installed(&tmp.0, &hub, ID);

    let source = pulled.record.source.unwrap();
    assert_eq!(source.provider, "huggingface");
    assert_eq!(source.repository, REPO);
    assert_eq!(source.revision, SHA, "the exact commit, not a branch name");

    for skipped in ["README.md", "pytorch_model.bin", "onnx/model.onnx"] {
        assert!(hub.downloads(skipped).is_empty(), "{skipped} was fetched");
    }
    assert!(!tmp.0.join(ID).join("README.md").exists());
}

#[test]
fn a_dropped_connection_resumes_where_it_stopped() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    hub.fault(WEIGHTS, vec![Fault::Cut(50_000), Fault::Cut(40_000)]);
    run(&tmp.0, &hub, &request()).unwrap();
    assert_installed(&tmp.0, &hub, ID);
    let ranges: Vec<_> = hub.downloads(WEIGHTS).iter().map(|s| s.range).collect();
    assert_eq!(ranges, [None, Some(50_000), Some(90_000)]);
}

#[test]
fn a_stalled_connection_is_ended_by_the_budget_and_continues() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    hub.fault(WEIGHTS, vec![Fault::Stall(60_000, Duration::from_secs(30))]);
    let reg = Registry::open(&tmp.0).unwrap();
    let p = HuggingFace::new(&hub.url, None, Duration::from_millis(300)).unwrap();
    let arch = ArchitectureRegistry::with_builtins();
    let cancel = AtomicBool::new(false);
    pull(&reg, &p, &request(), &fast(), &arch, &cancel, &|_| {}).unwrap();
    assert_installed(&tmp.0, &hub, ID);
    assert_eq!(hub.downloads(WEIGHTS)[1].range, Some(60_000));
}

#[test]
fn server_errors_back_off_and_recover() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    hub.fault(
        WEIGHTS,
        vec![
            Fault::Status(503),
            Fault::Limited(429, 1),
            Fault::Status(500),
        ],
    );
    let reg = Registry::open(&tmp.0).unwrap();
    let arch = ArchitectureRegistry::with_builtins();
    let cancel = AtomicBool::new(false);
    let delays = Mutex::new(Vec::new());
    let report = |p: Progress| {
        if let Progress::Retrying { delay, .. } = p {
            delays.lock().unwrap().push(delay);
        }
    };
    let mut opts = fast();
    opts.max_backoff = Duration::from_millis(50);
    pull(
        &reg,
        &provider(&hub),
        &request(),
        &opts,
        &arch,
        &cancel,
        &report,
    )
    .unwrap();
    assert_installed(&tmp.0, &hub, ID);
    let delays = delays.into_inner().unwrap();
    assert_eq!(delays.len(), 3);
    assert_eq!(delays[1], opts.max_backoff, "Retry-After, within the cap");
}

#[test]
fn listing_failures_are_retried() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    hub.state.api_faults.lock().unwrap().extend([503, 502]);
    run(&tmp.0, &hub, &request()).unwrap();
    assert_installed(&tmp.0, &hub, ID);
}

#[test]
fn final_statuses_are_not_retried() {
    for (status, attempts) in [(404, 1), (401, 1), (403, 1)] {
        let tmp = Tmp::new();
        let hub = Hub::start(fixture(ID));
        hub.fault(WEIGHTS, vec![Fault::Status(status)]);
        let e = run(&tmp.0, &hub, &request()).unwrap_err();
        assert!(
            matches!(
                e,
                DownloadError::NotFound(_) | DownloadError::Unauthorized { .. }
            ),
            "{e:?}"
        );
        assert_eq!(hub.downloads(WEIGHTS).len(), attempts);
        assert_absent(&tmp.0, ID);
    }
}

#[test]
fn running_out_of_attempts_is_recorded_and_can_be_retried() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    hub.fault(WEIGHTS, vec![Fault::Status(500); 4]);
    let e = run(&tmp.0, &hub, &request()).unwrap_err();
    assert!(
        matches!(e, DownloadError::Exhausted { attempts: 4, .. }),
        "{e:?}"
    );
    assert_absent(&tmp.0, ID);
    assert!(matches!(
        state(&tmp.0, ID),
        Some(InstallState::DownloadFailed { .. })
    ));
    run(&tmp.0, &hub, &request()).unwrap();
    assert_installed(&tmp.0, &hub, ID);
}

#[test]
fn a_server_that_ignores_ranges_restarts_the_file() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    hub.fault(WEIGHTS, vec![Fault::Cut(50_000), Fault::IgnoreRange]);
    run(&tmp.0, &hub, &request()).unwrap();
    assert_installed(&tmp.0, &hub, ID);
}

#[test]
fn a_range_that_was_not_asked_for_is_refused() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    hub.fault(WEIGHTS, vec![Fault::Cut(50_000), Fault::WrongRange]);
    let e = run(&tmp.0, &hub, &request()).unwrap_err();
    assert!(matches!(e, DownloadError::Protocol(_)), "{e:?}");
    assert_absent(&tmp.0, ID);
}

#[test]
fn corrupt_bytes_are_downloaded_again() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    hub.fault(WEIGHTS, vec![Fault::Corrupt]);
    run(&tmp.0, &hub, &request()).unwrap();
    assert_installed(&tmp.0, &hub, ID);
    assert_eq!(hub.downloads(WEIGHTS).len(), 2);
}

#[test]
fn corrupt_small_files_fail_their_git_checksum_too() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    hub.fault("config.json", vec![Fault::Corrupt]);
    run(&tmp.0, &hub, &request()).unwrap();
    assert_eq!(hub.downloads("config.json").len(), 2);
    assert_installed(&tmp.0, &hub, ID);
}

#[test]
fn bytes_that_stay_corrupt_never_install() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    hub.fault(WEIGHTS, vec![Fault::Corrupt, Fault::Corrupt]);
    let e = run(&tmp.0, &hub, &request()).unwrap_err();
    assert!(matches!(e, DownloadError::Checksum { .. }), "{e:?}");
    assert_absent(&tmp.0, ID);
    assert!(!staged(&tmp.0, ID).join(WEIGHTS).exists());
    assert!(!staged(&tmp.0, ID).join("model.safetensors.part").exists());
}

#[test]
fn a_body_longer_than_its_listing_stops_at_the_listed_size() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    hub.fault(WEIGHTS, vec![Fault::Oversize]);
    let e = run(&tmp.0, &hub, &request()).unwrap_err();
    assert!(matches!(e, DownloadError::TooLarge { .. }), "{e:?}");
    assert_absent(&tmp.0, ID);
    assert!(!staged(&tmp.0, ID).join("model.safetensors.part").exists());
}

#[test]
fn cancelling_keeps_the_files_and_the_same_pull_continues() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    *hub.state.pace.lock().unwrap() = Duration::from_millis(5);
    let reg = Registry::open(&tmp.0).unwrap();
    let arch = ArchitectureRegistry::with_builtins();
    let cancel = AtomicBool::new(false);
    let report = |p: Progress| {
        if matches!(&p, Progress::File { file, done, .. } if file == WEIGHTS && *done > 40_000) {
            cancel.store(true, Ordering::Relaxed);
        }
    };
    let e = pull(
        &reg,
        &provider(&hub),
        &request(),
        &fast(),
        &arch,
        &cancel,
        &report,
    )
    .unwrap_err();
    assert!(matches!(e, DownloadError::Cancelled), "{e:?}");
    assert_absent(&tmp.0, ID);
    assert!(matches!(
        state(&tmp.0, ID),
        Some(InstallState::DownloadFailed { .. })
    ));
    let kept = fs::metadata(staged(&tmp.0, ID).join("model.safetensors.part"))
        .unwrap()
        .len();
    assert!(kept > 0);

    *hub.state.pace.lock().unwrap() = Duration::ZERO;
    run(&tmp.0, &hub, &request()).unwrap();
    assert_installed(&tmp.0, &hub, ID);
    assert_eq!(hub.downloads(WEIGHTS).last().unwrap().range, Some(kept));
}

/// A crash can leave anything at the end of a `.part` file.
#[test]
fn a_bad_partial_file_is_caught_by_the_final_check() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    hub.fault(WEIGHTS, vec![Fault::Status(404)]);
    run(&tmp.0, &hub, &request()).unwrap_err();
    fs::write(
        staged(&tmp.0, ID).join("model.safetensors.part"),
        vec![9; 1000],
    )
    .unwrap();
    run(&tmp.0, &hub, &request()).unwrap();
    assert_installed(&tmp.0, &hub, ID);
}

#[test]
fn a_moved_branch_discards_files_from_the_old_commit() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    hub.fault(WEIGHTS, vec![Fault::Cut(50_000), Fault::Status(404)]);
    run(&tmp.0, &hub, &request()).unwrap_err();
    assert!(staged(&tmp.0, ID).join("model.safetensors.part").exists());
    *hub.state.sha.lock().unwrap() = "f".repeat(40);
    run(&tmp.0, &hub, &request()).unwrap();
    assert_installed(&tmp.0, &hub, ID);
    assert_eq!(hub.downloads(WEIGHTS).last().unwrap().range, None);
}

#[test]
fn a_second_pull_of_an_installed_model_fetches_nothing() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    run(&tmp.0, &hub, &request()).unwrap();
    let before = hub.downloads(WEIGHTS).len();
    let e = run(&tmp.0, &hub, &request()).unwrap_err();
    assert!(
        matches!(
            e,
            DownloadError::Registry(RegistryError::AlreadyInstalled(_))
        ),
        "{e:?}"
    );
    assert_eq!(hub.downloads(WEIGHTS).len(), before);
    assert_installed(&tmp.0, &hub, ID);
}

#[test]
fn a_concurrent_pull_of_one_model_is_refused() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    let reg = Registry::open(&tmp.0).unwrap();
    let _held = reg.begin_install(ID, None).unwrap();
    let e = run(&tmp.0, &hub, &request()).unwrap_err();
    assert!(
        matches!(
            e,
            DownloadError::Registry(RegistryError::InstallInProgress(_))
        ),
        "{e:?}"
    );
}

/// Counts bodies open at once, on the client, which is what the limit is about. The
/// server cannot see this: it may still be finishing a body the client has moved past.
struct Counting {
    inner: HuggingFace,
    open: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
    /// Each open waits for this many, so overlap does not depend on timing.
    expect: usize,
}

struct Counted {
    body: Box<dyn Read + Send>,
    open: Arc<AtomicUsize>,
}

impl Read for Counted {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.body.read(buf)
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        self.open.fetch_sub(1, Ordering::SeqCst);
    }
}

impl ModelProvider for Counting {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn resolve(&self, repo: &str, revision: Option<&str>) -> Result<RepoListing, DownloadError> {
        self.inner.resolve(repo, revision)
    }

    fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>, DownloadError> {
        self.inner.search(query, limit)
    }

    fn open(&self, list: &RepoListing, path: &str, at: u64) -> Result<Opened, DownloadError> {
        let opened = self.inner.open(list, path, at)?;
        let now = self.open.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        let waited = Instant::now();
        while self.peak.load(Ordering::SeqCst) < self.expect
            && waited.elapsed() < Duration::from_secs(5)
        {
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(Opened {
            start: opened.start,
            body: Box::new(Counted {
                body: opened.body,
                open: self.open.clone(),
            }),
        })
    }
}

fn peak_transfers(concurrency: usize) -> usize {
    let tmp = Tmp::new();
    let names: Vec<String> = (0..6).map(|i| format!("data-{i}.bin")).collect();
    let files = names
        .iter()
        .map(|path| File {
            path: path.clone(),
            bytes: vec![3; 4096],
            lfs: true,
        })
        .collect();
    let hub = Hub::start(files);
    let peak = Arc::new(AtomicUsize::new(0));
    let counting = Counting {
        inner: provider(&hub),
        open: Arc::new(AtomicUsize::new(0)),
        peak: peak.clone(),
        expect: concurrency,
    };
    let mut req = request();
    req.files = names;
    let mut opts = fast();
    opts.concurrency = concurrency;
    let reg = Registry::open(&tmp.0).unwrap();
    let arch = ArchitectureRegistry::with_builtins();
    let cancel = AtomicBool::new(false);
    // These are not a model, so the import fails after every file has arrived.
    let e = pull(&reg, &counting, &req, &opts, &arch, &cancel, &|_| {}).unwrap_err();
    assert!(matches!(e, DownloadError::Import(_)), "{e:?}");
    peak.load(Ordering::SeqCst)
}

#[test]
fn concurrency_is_bounded_and_real() {
    assert_eq!(peak_transfers(1), 1);
    assert_eq!(peak_transfers(3), 3);
}

#[test]
fn the_token_reaches_the_hub_and_never_another_host() {
    let tmp = Tmp::new();
    let cdn = Hub::start(fixture(ID));
    let hub = Hub::start(fixture(ID));
    // `localhost` and 127.0.0.1 are the same machine but different origins.
    let signed = format!(
        "http://localhost:{}/{REPO}/resolve/{SHA}/{WEIGHTS}?Signature=abc",
        cdn.port
    );
    hub.fault(WEIGHTS, vec![Fault::Redirect(signed)]);
    let reg = Registry::open(&tmp.0).unwrap();
    let token = Token::new("hf_secret").unwrap();
    let p = HuggingFace::new(&hub.url, Some(token), Duration::from_secs(10)).unwrap();
    let arch = ArchitectureRegistry::with_builtins();
    let cancel = AtomicBool::new(false);
    pull(&reg, &p, &request(), &fast(), &arch, &cancel, &|_| {}).unwrap();
    assert_installed(&tmp.0, &hub, ID);

    assert!(
        hub.seen()
            .iter()
            .all(|s| s.auth.as_deref() == Some("Bearer hf_secret"))
    );
    let at_cdn = cdn.seen();
    assert!(!at_cdn.is_empty());
    assert!(at_cdn.iter().all(|s| s.auth.is_none()), "{at_cdn:?}");
}

#[test]
fn a_listing_cannot_page_the_token_to_another_host() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    let other = Hub::start(fixture(ID));
    *hub.state.next_link.lock().unwrap() = Some(format!("{}/steal", other.url));
    let token = Token::new("hf_secret").unwrap();
    let p = HuggingFace::new(&hub.url, Some(token), Duration::from_secs(10)).unwrap();
    let reg = Registry::open(&tmp.0).unwrap();
    let arch = ArchitectureRegistry::with_builtins();
    let cancel = AtomicBool::new(false);
    let e = pull(&reg, &p, &request(), &fast(), &arch, &cancel, &|_| {}).unwrap_err();
    assert!(matches!(e, DownloadError::Protocol(_)), "{e:?}");
    assert!(other.seen().is_empty());
}

#[test]
fn redirects_to_other_schemes_are_refused_and_stay_out_of_errors() {
    for target in [
        "file:///etc/passwd?Signature=abc",
        "ftp://example.com/x?Signature=abc",
        "gopher://example.com/?Signature=abc",
    ] {
        let tmp = Tmp::new();
        let hub = Hub::start(fixture(ID));
        hub.fault(WEIGHTS, vec![Fault::Redirect(target.to_owned())]);
        let e = run(&tmp.0, &hub, &request()).unwrap_err();
        assert!(matches!(e, DownloadError::Protocol(_)), "{e:?}");
        assert!(!e.to_string().contains("Signature"), "{e}");
        assert_absent(&tmp.0, ID);
    }
}

#[test]
fn a_redirect_loop_ends() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    let me = format!("/{REPO}/resolve/{SHA}/{WEIGHTS}");
    hub.fault(WEIGHTS, vec![Fault::Redirect(me); 32]);
    let e = run(&tmp.0, &hub, &request()).unwrap_err();
    assert!(matches!(e, DownloadError::Protocol(_)), "{e:?}");
}

#[test]
fn file_names_cannot_leave_the_model_directory() {
    for evil in ["../../evil.gguf", "/tmp/evil.gguf", "..\\evil.gguf"] {
        let tmp = Tmp::new();
        let inner = tmp.0.join("models");
        let hub = Hub::start(vec![File {
            path: evil.to_owned(),
            bytes: vec![1; 8],
            lfs: true,
        }]);
        let mut req = request();
        req.files = vec![evil.to_owned()];
        let e = run(&inner, &hub, &req).unwrap_err();
        assert!(matches!(e, DownloadError::UnsafePath(_)), "{e:?}");
        assert!(hub.downloads("evil.gguf").is_empty());
        assert!(!tmp.0.join("evil.gguf").exists());
    }
}

#[test]
fn a_bad_repository_or_endpoint_sends_nothing() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture(ID));
    let mut req = request();
    req.repository = "../../api/secret".to_owned();
    assert!(matches!(
        run(&tmp.0, &hub, &req),
        Err(DownloadError::InvalidRepository(_))
    ));
    assert!(hub.seen().is_empty());
    for bad in ["http://example.com", "ftp://127.0.0.1", "not a url"] {
        assert!(matches!(
            HuggingFace::new(bad, None, Duration::from_secs(1)),
            Err(DownloadError::InvalidEndpoint(_))
        ));
    }
}

#[test]
fn files_that_do_not_import_stay_for_the_next_try() {
    let tmp = Tmp::new();
    let mut files = fixture(ID);
    let config = files.iter_mut().find(|f| f.path == "config.json").unwrap();
    let mut v: serde_json::Value = serde_json::from_slice(&config.bytes).unwrap();
    let hidden = v["hidden_size"].take();
    v.as_object_mut().unwrap().remove("hidden_size");
    config.bytes = serde_json::to_vec(&v).unwrap();
    let hub = Hub::start(files);

    let e = run(&tmp.0, &hub, &request()).unwrap_err();
    assert!(matches!(e, DownloadError::Import(_)), "{e:?}");
    assert_absent(&tmp.0, ID);
    let reason = match state(&tmp.0, ID) {
        Some(InstallState::DownloadFailed { reason }) => reason,
        other => panic!("{other:?}"),
    };
    assert!(reason.contains("hidden_size"), "{reason}");

    let mut req = request();
    req.set.insert("hidden_size".to_owned(), hidden);
    run(&tmp.0, &hub, &req).unwrap();
    assert_installed(&tmp.0, &hub, ID);
    assert_eq!(hub.downloads(WEIGHTS).len(), 1, "not downloaded twice");
}

/// GGUF metadata lacks this value, as it does for real OLMoE files.
fn gguf_request(file: &str) -> PullRequest {
    let mut req = request();
    req.files = vec![file.to_owned()];
    req.set
        .insert("norm_topk_prob".to_owned(), serde_json::json!(false));
    req
}

#[test]
fn a_gguf_is_pulled_alone_under_its_own_name() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture("olmoe-tiny-gguf"));
    let file = "olmoe-tiny-q8_0.gguf";
    let pulled = run(&tmp.0, &hub, &gguf_request(file)).unwrap();
    assert_eq!(pulled.record.id, "olmoe-tiny-q8_0");
    assert_eq!(pulled.files, [file]);
    let dir = tmp.0.join("olmoe-tiny-q8_0");
    ModelMetadata::load(&dir, &ArchitectureRegistry::with_builtins()).unwrap();
}

#[test]
fn a_changed_choice_leaves_no_stale_file_in_staging() {
    let tmp = Tmp::new();
    let hub = Hub::start(fixture("olmoe-tiny-gguf"));
    let file = "olmoe-tiny-q8_0.gguf";
    let id = "olmoe-tiny-q8_0";
    hub.fault(file, vec![Fault::Status(404)]);
    run(&tmp.0, &hub, &gguf_request(file)).unwrap_err();
    fs::write(
        staged(&tmp.0, id).join("other.gguf"),
        b"from an earlier choice",
    )
    .unwrap();
    fs::write(staged(&tmp.0, id).join("other.gguf.part"), b"and this").unwrap();
    run(&tmp.0, &hub, &gguf_request(file)).unwrap();
    assert!(!tmp.0.join(id).join("other.gguf").exists());
    assert_eq!(state(&tmp.0, id), Some(InstallState::Installed));
}

#[test]
fn an_index_chooses_the_weights() {
    let tmp = Tmp::new();
    let mut files = fixture(ID);
    files.push(File {
        path: "model.safetensors.index.json".to_owned(),
        bytes: br#"{"weight_map":{"a":"model.safetensors"}}"#.to_vec(),
        lfs: false,
    });
    files.push(File {
        path: "consolidated.safetensors".to_owned(),
        bytes: vec![0; 64],
        lfs: true,
    });
    let hub = Hub::start(files);
    run(&tmp.0, &hub, &request()).unwrap();
    assert_installed(&tmp.0, &hub, ID);
    assert!(hub.downloads("consolidated.safetensors").is_empty());
}

#[test]
fn search_returns_normalised_hits() {
    let hub = Hub::start(Vec::new());
    let cancel = AtomicBool::new(false);
    let hits = search(&provider(&hub), "a b&c", 5, &fast(), &cancel).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].repository, "acme/a-GGUF");
    assert_eq!(hits[0].formats, ["gguf"]);
    assert!(hits[1].gated);
    assert!(hub.seen()[0].path.contains("search=a%20b%26c&limit=5"));
}
