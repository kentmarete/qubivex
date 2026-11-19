//! `qubivex models pull` and `search`.

use std::collections::BTreeMap;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use qubivex_api::{
    ModelSource, PullOutcome, PullProgress, PullSpec, SourceError, SourceErrorKind, SourceHit,
};
use qubivex_core::Bytes;
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_model_download::{
    DEFAULT_BUDGET, DEFAULT_ENDPOINT, DownloadError, HuggingFace, Progress, PullOptions,
    PullRequest, Token, pull, search,
};
use qubivex_model_registry::Registry;
use serde_json::json;

use crate::chain;
use crate::models::{parse_set, record_sums, roots, with_set_hint};

/// How often progress prints when no one is watching a terminal.
const QUIET_INTERVAL: Duration = Duration::from_secs(15);
const TERMINAL_INTERVAL: Duration = Duration::from_millis(250);

fn provider() -> Result<HuggingFace, String> {
    let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.to_owned());
    let token = Token::from_environment().map_err(|e| chain(&e))?;
    HuggingFace::new(&endpoint, token, DEFAULT_BUDGET).map_err(|e| chain(&e))
}

/// A flag the first Ctrl-C sets, so a pull can stop cleanly. A second one exits at
/// once, which is safe: a killed pull leaves what a cancelled one does.
fn cancel_on_interrupt() -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    let set = flag.clone();
    std::thread::spawn(move || {
        let Ok(rt) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        else {
            return;
        };
        rt.block_on(async {
            if tokio::signal::ctrl_c().await.is_ok() {
                set.store(true, Ordering::Relaxed);
                eprintln!("\nqubivex: stopping; press Ctrl-C again to exit now");
            }
            if tokio::signal::ctrl_c().await.is_ok() {
                std::process::exit(130);
            }
        });
    });
    flag
}

struct Meter {
    files: BTreeMap<String, (u64, u64)>,
    started: Instant,
    last: Instant,
    /// Bytes present when the transfer began, left out of the rate.
    resumed: Option<u64>,
    terminal: bool,
}

impl Meter {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            files: BTreeMap::new(),
            started: now,
            last: now,
            resumed: None,
            terminal: std::io::stderr().is_terminal(),
        }
    }

    fn line(&self) -> String {
        let done: u64 = self.files.values().map(|f| f.0).sum();
        let total: u64 = self.files.values().map(|f| f.1).sum();
        let new = done.saturating_sub(self.resumed.unwrap_or(0));
        let secs = self.started.elapsed().as_secs_f64().max(0.001);
        // Display only, so the rounding of a float is harmless.
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let (percent, rate) = (
            done as f64 * 100.0 / (total.max(1) as f64),
            (new as f64 / secs) as u64,
        );
        format!(
            "{percent:5.1}%  {} of {}  {}/s",
            Bytes(done),
            Bytes(total),
            Bytes(rate)
        )
    }

    fn update(&mut self, file: String, done: u64, total: u64) {
        self.files.insert(file, (done, total));
        let interval = if self.terminal {
            TERMINAL_INTERVAL
        } else {
            QUIET_INTERVAL
        };
        if self.last.elapsed() < interval {
            return;
        }
        self.last = Instant::now();
        if self.resumed.is_none() {
            self.resumed = Some(self.files.values().map(|f| f.0).sum());
            self.started = Instant::now();
        }
        if self.terminal {
            eprint!("\r{}\x1b[K", self.line());
            let _ = std::io::stderr().flush();
        } else {
            eprintln!("{}", self.line());
        }
    }

    fn event(&mut self, p: Progress) {
        match p {
            Progress::Planned { revision, files } => {
                let total: u64 = files.iter().map(|f| f.1).sum();
                eprintln!(
                    "revision {revision}: {} files, {}",
                    files.len(),
                    Bytes(total)
                );
                for (name, size) in files {
                    eprintln!("  {name}  {}", Bytes(size));
                    self.files.insert(name, (0, size));
                }
            }
            Progress::File { file, done, total } => self.update(file, done, total),
            Progress::Retrying {
                file,
                attempt,
                delay,
                reason,
            } => eprintln!(
                "\r{file}: attempt {attempt} failed ({reason}); retrying in {:.1} s\x1b[K",
                delay.as_secs_f64()
            ),
            Progress::Validating => {
                if self.terminal {
                    eprintln!();
                }
                eprintln!("validating");
            }
            _ => {}
        }
    }
}

pub(crate) struct PullArgs<'a> {
    pub(crate) repository: &'a str,
    pub(crate) files: &'a [String],
    pub(crate) revision: Option<&'a str>,
    pub(crate) id: Option<&'a str>,
    pub(crate) name: Option<&'a str>,
    pub(crate) set: &'a [String],
    pub(crate) dir: Option<&'a Path>,
    pub(crate) concurrency: usize,
}

pub(crate) fn pull_cmd(a: &PullArgs<'_>) -> Result<String, String> {
    let mut set = serde_json::Map::new();
    for s in a.set {
        let (k, v) = parse_set(s)?;
        set.insert(k, v);
    }
    let req = PullRequest {
        repository: a.repository.to_owned(),
        revision: a.revision.map(str::to_owned),
        files: a.files.to_vec(),
        id: a.id.map(str::to_owned),
        name: a.name.map(str::to_owned),
        set,
    };
    let root = a
        .dir
        .map_or_else(|| roots(&[]).swap_remove(0), Path::to_path_buf);
    let opts = PullOptions {
        concurrency: a.concurrency.max(1),
        ..PullOptions::default()
    };
    let registry = Registry::open(&root).map_err(|e| chain(&e))?;
    let arch = ArchitectureRegistry::with_builtins();
    let cancel = cancel_on_interrupt();
    let meter = Mutex::new(Meter::new());
    let report = |p: Progress| {
        if let Ok(mut m) = meter.lock() {
            m.event(p);
        }
    };
    let provider = provider()?;
    let pulled = pull(&registry, &provider, &req, &opts, &arch, &cancel, &report)
        .map_err(|e| with_set_hint(&chain(&e)))?;
    for w in &pulled.warnings {
        eprintln!("qubivex: warning: {w}");
    }
    let dir = registry.dir_of(&pulled.record);
    let mut out = format!("Installed {} in {}\n", pulled.record.id, dir.display());
    let model = ModelMetadata::load(&dir, &arch).map_err(|e| chain(&e))?;
    // Checksums cover streamed experts; a dense model streams nothing.
    if model.bytes().routed_experts > 0 {
        out.push_str(&record_sums(&model, 0)?);
    }
    Ok(out)
}

pub(crate) fn search_cmd(query: &str, limit: usize, json_out: bool) -> Result<String, String> {
    let cancel = AtomicBool::new(false);
    let hits = search(&provider()?, query, limit, &PullOptions::default(), &cancel)
        .map_err(|e| chain(&e))?;
    if json_out {
        let rows: Vec<_> = hits
            .iter()
            .map(|h| {
                json!({
                    "repository": h.repository,
                    "downloads": h.downloads,
                    "likes": h.likes,
                    "formats": h.formats,
                    "gated": h.gated,
                })
            })
            .collect();
        return crate::to_json(&rows);
    }
    if hits.is_empty() {
        return Ok(format!("nothing found for {query:?}\n"));
    }
    let width = hits
        .iter()
        .map(|h| h.repository.len())
        .max()
        .unwrap_or(0)
        .max(10);
    let mut out = format!(
        "{:<width$}  {:>10}  {:>6}  FORMATS\n",
        "REPOSITORY", "DOWNLOADS", "LIKES"
    );
    for h in &hits {
        let mut formats = h.formats.join(", ");
        if h.gated {
            formats.push_str(if formats.is_empty() {
                "gated"
            } else {
                ", gated"
            });
        }
        let line = format!(
            "{:<width$}  {:>10}  {:>6}  {formats}",
            h.repository, h.downloads, h.likes
        );
        out.push_str(line.trim_end());
        out.push('\n');
    }
    Ok(out)
}

/// What `qubivex serve` downloads with, for the admin download routes: Hugging Face,
/// installing into the first models directory. The gateway itself has no network code
/// (ADR-014); this is where it gets some.
#[derive(Debug)]
pub(crate) struct HubSource {
    root: PathBuf,
}

impl HubSource {
    pub(crate) fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// `e` as a client may see it: no path, URL or credential, and wording for the API
    /// rather than the command line.
    fn error(&self, e: &DownloadError) -> SourceError {
        use SourceErrorKind as K;
        use qubivex_model_registry::RegistryError as R;
        let (kind, message) = match e {
            DownloadError::Cancelled => (
                K::Cancelled,
                "cancelled; retry the job to continue from what arrived".to_owned(),
            ),
            DownloadError::Io { source, .. } => (
                K::Local,
                format!("a file operation in the models directory failed: {source}"),
            ),
            DownloadError::NoSpace {
                needed, available, ..
            } => (
                K::Local,
                format!(
                    "the models directory needs {} more, and {} are free",
                    Bytes(*needed),
                    Bytes(*available)
                ),
            ),
            DownloadError::Registry(r) => match r {
                R::AlreadyInstalled(_) | R::InstallInProgress(_) => (K::Conflict, r.to_string()),
                R::InvalidId(_) | R::Validation { .. } => (K::Invalid, r.to_string()),
                R::Io { source, .. } => (
                    K::Local,
                    format!("the model registry could not be updated: {source}"),
                ),
                _ => (K::Local, "the model registry could not be read".to_owned()),
            },
            DownloadError::Import(_) => (K::Invalid, set_hint(&chain(e))),
            DownloadError::NotFound(_) | DownloadError::FileNotFound { .. } => {
                (K::NotFound, e.to_string())
            }
            DownloadError::Unauthorized { .. } => (K::Unauthorized, e.to_string()),
            DownloadError::InvalidRepository(_)
            | DownloadError::Ambiguous { .. }
            | DownloadError::NoWeights(_)
            | DownloadError::UnsafePath(_) => (K::Invalid, e.to_string()),
            DownloadError::InvalidEndpoint(_) | DownloadError::InvalidToken => {
                (K::Local, format!("the server's download settings: {e}"))
            }
            _ => (K::Upstream, e.to_string()),
        };
        SourceError::new(kind, self.scrub(&message))
    }

    /// Takes local paths out of `message` and swaps command-line wording for the API's.
    fn scrub(&self, message: &str) -> String {
        let mut m = message.replace(&*self.root.to_string_lossy(), "the models directory");
        if let Some(home) = std::env::var_os("HOME").filter(|h| h.len() > 1) {
            m = m.replace(&*home.to_string_lossy(), "~");
        }
        m.replace("pick one with --file", "pick one in files")
    }
}

/// Points a missing config value at the pull's `set`.
fn set_hint(msg: &str) -> String {
    let Some((_, rest)) = msg.split_once("overrides.config.") else {
        return msg.to_owned();
    };
    let key: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if key.is_empty() {
        return msg.to_owned();
    }
    format!(
        "{msg}. Retry with set: {{\"{key}\": VALUE}}, with the value from the model's original config.json"
    )
}

impl ModelSource for HubSource {
    fn search(&self, query: &str, limit: usize) -> Result<Vec<SourceHit>, SourceError> {
        let provider =
            provider().map_err(|e| SourceError::new(SourceErrorKind::Local, self.scrub(&e)))?;
        let cancel = AtomicBool::new(false);
        let hits = search(&provider, query, limit, &PullOptions::default(), &cancel)
            .map_err(|e| self.error(&e))?;
        Ok(hits
            .into_iter()
            .map(|h| SourceHit {
                repository: h.repository,
                downloads: h.downloads,
                likes: h.likes,
                formats: h.formats,
                gated: h.gated,
            })
            .collect())
    }

    fn pull(
        &self,
        spec: &PullSpec,
        cancel: &AtomicBool,
        progress: &(dyn Fn(PullProgress) + Sync),
    ) -> Result<PullOutcome, SourceError> {
        let provider =
            provider().map_err(|e| SourceError::new(SourceErrorKind::Local, self.scrub(&e)))?;
        let registry =
            Registry::open(&self.root).map_err(|e| self.error(&DownloadError::Registry(e)))?;
        let arch = ArchitectureRegistry::with_builtins();
        let req = PullRequest {
            repository: spec.repository.clone(),
            revision: spec.revision.clone(),
            files: spec.files.clone(),
            id: spec.id.clone(),
            name: spec.name.clone(),
            set: spec.set.clone(),
        };
        let report = |p: Progress| {
            let p = match p {
                Progress::Planned { revision, files } => PullProgress::Planned { revision, files },
                Progress::File { file, done, total } => PullProgress::File { file, done, total },
                Progress::Retrying {
                    file,
                    attempt,
                    delay,
                    reason,
                } => PullProgress::Retrying {
                    file,
                    attempt,
                    delay,
                    reason: self.scrub(&reason),
                },
                Progress::Validating => PullProgress::Validating,
                _ => return,
            };
            progress(p);
        };
        let pulled = pull(
            &registry,
            &provider,
            &req,
            &PullOptions::default(),
            &arch,
            cancel,
            &report,
        )
        .map_err(|e| self.error(&e))?;
        let mut warnings: Vec<String> = pulled.warnings.iter().map(|w| self.scrub(w)).collect();
        // As `models pull` does: checksums cover streamed experts; a dense model streams
        // nothing.
        let dir = registry.dir_of(&pulled.record);
        match ModelMetadata::load(&dir, &arch) {
            Ok(m) if m.bytes().routed_experts > 0 => {
                if let Err(e) = record_sums(&m, 0) {
                    warnings.push(format!(
                        "expert checksums were not recorded: {}",
                        self.scrub(&e)
                    ));
                }
            }
            Ok(_) => {}
            Err(e) => warnings.push(format!(
                "the installed model does not read back: {}",
                self.scrub(&chain(&e))
            )),
        }
        Ok(PullOutcome {
            id: pulled.record.id,
            warnings,
            files: pulled.files,
        })
    }

    fn installed(&self) -> Vec<(String, String)> {
        Registry::open(&self.root)
            .and_then(|r| r.list())
            .map(|records| {
                records
                    .into_iter()
                    .filter_map(|r| r.source.map(|s| (s.repository, r.id)))
                    .collect()
            })
            .unwrap_or_default()
    }
}
