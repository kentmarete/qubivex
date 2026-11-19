//! Model downloads as server-owned jobs (ADR-013 amendment 8): search, pull, and the
//! jobs that pulls become. A job runs on its own thread; a client starts it and then
//! reads its state, and closing the client does not stop it.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use kerniq_server::axum::Json;
use kerniq_server::axum::extract::rejection::{JsonRejection, QueryRejection};
use kerniq_server::axum::extract::{Path, Query, State};
use kerniq_server::axum::http::{HeaderMap, HeaderValue, StatusCode};
use kerniq_server::axum::response::{IntoResponse, Response};
use serde_json::{Map, Value, json};

use crate::source::{
    ModelSource, PullOutcome, PullProgress, PullSpec, SourceError, SourceErrorKind,
};
use crate::{ApiError, AppState, json_body, now_secs};

/// Pulls that may run at once; more get a 429.
const MAX_RUNNING: usize = 2;
/// How long a finished job stays listed.
const RETAIN: Duration = Duration::from_secs(3600);
/// Most jobs kept; the oldest finished ones go first.
const MAX_KEPT: usize = 64;
/// Search results when the request does not say, and the most it may ask for.
const SEARCH_DEFAULT: usize = 20;
const SEARCH_MAX: usize = 100;
/// The shortest span a transfer rate is measured over.
const RATE_SPAN: Duration = Duration::from_secs(1);

/// Where a job is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Running,
    Validating,
    Succeeded,
    Failed,
    Cancelled,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Validating => "validating",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    fn finished(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }
}

/// A job's state, as clients see it.
struct View {
    status: Status,
    /// The id the model installs as, once known.
    model: Option<String>,
    revision: Option<String>,
    created_at: u64,
    finished_at: Option<u64>,
    /// Per file: bytes done and total.
    files: BTreeMap<String, (u64, u64)>,
    /// Bytes already present when this run's transfer began; kept out of the rate.
    resumed: Option<u64>,
    /// The last rate sample: when, and bytes done then.
    sample: Option<(Instant, u64)>,
    rate_bps: Option<u64>,
    retrying: Option<Value>,
    result: Option<PullOutcome>,
    error: Option<(&'static str, String)>,
    /// Finished, for pruning.
    ended: Option<Instant>,
}

impl View {
    fn done_total(&self) -> (u64, u64) {
        self.files
            .values()
            .fold((0, 0), |(d, t), &(fd, ft)| (d + fd, t + ft))
    }

    fn on_progress(&mut self, p: PullProgress) {
        match p {
            PullProgress::Planned { revision, files } => {
                self.revision = Some(revision);
                for (name, size) in files {
                    self.files.insert(name, (0, size));
                }
            }
            PullProgress::File { file, done, total } => {
                self.files.insert(file, (done, total));
                self.retrying = None;
                let (done, _) = self.done_total();
                let now = Instant::now();
                match self.sample {
                    None => {
                        self.resumed = Some(done);
                        self.sample = Some((now, done));
                    }
                    Some((at, then)) if now.duration_since(at) >= RATE_SPAN => {
                        let secs = now.duration_since(at).as_secs_f64();
                        // Display only, so the rounding of a float is harmless.
                        #[allow(
                            clippy::cast_precision_loss,
                            clippy::cast_possible_truncation,
                            clippy::cast_sign_loss
                        )]
                        let fresh = (done.saturating_sub(then) as f64 / secs) as u64;
                        // Smoothed over a few samples, so one slow second does not swing it.
                        self.rate_bps = Some(match self.rate_bps {
                            None => fresh,
                            Some(old) => (old.saturating_mul(3).saturating_add(fresh)) / 4,
                        });
                        self.sample = Some((now, done));
                    }
                    Some(_) => {}
                }
            }
            PullProgress::Retrying {
                file,
                attempt,
                delay,
                reason,
            } => {
                self.retrying = Some(json!({
                    "file": file,
                    "attempt": attempt,
                    "delay_s": delay.as_secs_f64(),
                    "reason": reason,
                }));
            }
            PullProgress::Validating => {
                self.status = Status::Validating;
                self.retrying = None;
            }
        }
    }

    fn finish(&mut self, status: Status) {
        self.status = status;
        self.finished_at = Some(now_secs());
        self.ended = Some(Instant::now());
        self.retrying = None;
    }
}

/// One job.
struct Job {
    id: String,
    spec: PullSpec,
    view: Mutex<View>,
    cancel: AtomicBool,
}

impl Job {
    fn status(&self) -> Status {
        self.view.lock().map_or(Status::Failed, |v| v.status)
    }

    fn to_json(&self) -> Value {
        let Ok(v) = self.view.lock() else {
            return json!({ "object": "job", "id": self.id });
        };
        let (done, total) = v.done_total();
        let eta = match (v.rate_bps, v.status) {
            (Some(r), Status::Running) if r > 0 && total > 0 => {
                Some(total.saturating_sub(done) / r)
            }
            _ => None,
        };
        let files: Vec<Value> = v
            .files
            .iter()
            .map(|(name, (d, t))| json!({ "name": name, "done": d, "total": t }))
            .collect();
        json!({
            "object": "job",
            "id": self.id,
            "kind": "pull",
            "status": v.status.as_str(),
            "model": v.model,
            "repository": self.spec.repository,
            "revision": v.revision,
            "created_at": v.created_at,
            "started_at": v.created_at,
            "finished_at": v.finished_at,
            "progress": {
                "bytes_done": done,
                "bytes_total": total,
                "bytes_resumed": v.resumed.unwrap_or(0),
                "rate_bps": v.rate_bps,
                "eta_seconds": eta,
                "files": files,
                "retrying": v.retrying,
            },
            "result": v.result.as_ref().map(|r| json!({
                "id": r.id,
                "warnings": r.warnings,
                "files": r.files,
            })),
            "error": v.error.as_ref().map(|(code, message)| json!({ "code": code, "message": message })),
            "cancellable": !v.status.finished(),
        })
    }
}

/// The server's jobs. The lock is never held across an `.await`.
#[derive(Default)]
pub(crate) struct Jobs {
    inner: Mutex<BTreeMap<String, Arc<Job>>>,
    next: AtomicU64,
}

impl Jobs {
    fn get(&self, id: &str) -> Result<Arc<Job>, ApiError> {
        self.inner
            .lock()
            .ok()
            .and_then(|m| m.get(id).cloned())
            .ok_or_else(|| ApiError::job_not_found(id))
    }

    fn list(&self) -> Vec<Arc<Job>> {
        self.prune();
        self.inner
            .lock()
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Drops finished jobs older than [`RETAIN`], and the oldest finished ones past
    /// [`MAX_KEPT`].
    fn prune(&self) {
        let Ok(mut m) = self.inner.lock() else {
            return;
        };
        m.retain(|_, j| {
            j.view
                .lock()
                .map_or(true, |v| v.ended.is_none_or(|e| e.elapsed() < RETAIN))
        });
        // Ids sort by creation, so the first finished ones are the oldest.
        let excess = m.len().saturating_sub(MAX_KEPT);
        let old: Vec<String> = m
            .iter()
            .filter(|(_, j)| j.status().finished())
            .take(excess)
            .map(|(k, _)| k.clone())
            .collect();
        for k in old {
            m.remove(&k);
        }
    }

    /// Stops every running job (at shutdown).
    pub(crate) fn cancel_all(&self) {
        if let Ok(m) = self.inner.lock() {
            for j in m.values() {
                j.cancel.store(true, Ordering::Relaxed);
            }
        }
    }

    /// Registers a pull of `spec`, refusing a second pull of the same repository or id
    /// and more than [`MAX_RUNNING`] at once.
    fn admit(&self, spec: PullSpec) -> Result<Arc<Job>, ApiError> {
        self.prune();
        let mut m = self
            .inner
            .lock()
            .map_err(|_| ApiError::overloaded("the job list is unavailable"))?;
        let running: Vec<&Arc<Job>> = m.values().filter(|j| !j.status().finished()).collect();
        if let Some(j) = running.iter().find(|j| {
            j.spec.repository.eq_ignore_ascii_case(&spec.repository)
                || (spec.id.is_some() && j.spec.id == spec.id)
        }) {
            return Err(ApiError::conflict(format!(
                "a pull of {} is already running as job {}",
                spec.repository, j.id
            )));
        }
        if running.len() >= MAX_RUNNING {
            return Err(ApiError::too_many_jobs(MAX_RUNNING));
        }
        let id = format!(
            "job-{:x}-{:x}",
            now_secs(),
            self.next.fetch_add(1, Ordering::Relaxed) + 1
        );
        let job = Arc::new(Job {
            id: id.clone(),
            view: Mutex::new(View {
                status: Status::Running,
                model: spec.id.clone(),
                revision: None,
                created_at: now_secs(),
                finished_at: None,
                files: BTreeMap::new(),
                resumed: None,
                sample: None,
                rate_bps: None,
                retrying: None,
                result: None,
                error: None,
                ended: None,
            }),
            spec,
            cancel: AtomicBool::new(false),
        });
        m.insert(id, Arc::clone(&job));
        Ok(job)
    }
}

impl ApiError {
    /// 404: no job with this id (it may have been pruned).
    fn job_not_found(id: &str) -> Self {
        Self {
            code: "job_not_found",
            ..Self::not_found_message(format!("no job {id:?}; finished jobs are kept for an hour"))
        }
    }

    /// 404: the server was started without a model source.
    fn downloads_disabled() -> Self {
        Self {
            code: "downloads_disabled",
            ..Self::not_found_message(
                "this server cannot download models: it was started without a model source".into(),
            )
        }
    }

    /// 429: too many pulls are running.
    fn too_many_jobs(max: usize) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            kind: "rate_limit_error",
            code: "too_many_requests",
            ..Self::invalid(format!(
                "{max} pulls are already running; wait for one to finish"
            ))
        }
    }

    /// A source's failure as the client should see it. Its message is path-free by the
    /// [`ModelSource`] contract.
    fn from_source(e: SourceError) -> Self {
        let message = e.message;
        match e.kind {
            SourceErrorKind::NotFound => Self {
                code: "not_found",
                ..Self::not_found_message(message)
            },
            SourceErrorKind::Conflict => Self::conflict(message),
            SourceErrorKind::Unauthorized => Self {
                status: StatusCode::FORBIDDEN,
                kind: "permission_error",
                code: "upstream_unauthorized",
                ..Self::invalid(message)
            },
            SourceErrorKind::Upstream => Self::upstream(message),
            SourceErrorKind::Cancelled => Self {
                code: "cancelled",
                ..Self::conflict(message)
            },
            SourceErrorKind::Local => Self::overloaded(message),
            _ => Self::invalid(message),
        }
    }
}

/// The error code a failed job records.
fn error_code(kind: SourceErrorKind) -> &'static str {
    match kind {
        SourceErrorKind::Invalid => "invalid_request",
        SourceErrorKind::NotFound => "not_found",
        SourceErrorKind::Conflict => "conflict",
        SourceErrorKind::Unauthorized => "upstream_unauthorized",
        SourceErrorKind::Upstream => "upstream_error",
        SourceErrorKind::Cancelled => "cancelled",
        _ => "local_error",
    }
}

fn source(st: &AppState) -> Result<Arc<dyn ModelSource>, ApiError> {
    st.cfg
        .model_source
        .clone()
        .ok_or_else(ApiError::downloads_disabled)
}

/// Runs `job` on its own thread.
fn spawn(st: &Arc<AppState>, src: Arc<dyn ModelSource>, job: &Arc<Job>) -> Result<(), ApiError> {
    let (st2, job2) = (Arc::clone(st), Arc::clone(job));
    st.log.info(
        "job_started",
        &[
            ("job_id", job.id.as_str().into()),
            ("kind", "pull".into()),
            ("repository", job.spec.repository.as_str().into()),
        ],
    );
    let spawned = std::thread::Builder::new()
        .name("qubivex-job".into())
        .spawn(move || run(&st2, src.as_ref(), &job2));
    if spawned.is_err() {
        if let Ok(mut v) = job.view.lock() {
            v.error = Some(("local_error", "could not start the job's thread".into()));
            v.finish(Status::Failed);
        }
        return Err(ApiError::overloaded("could not start the job's thread"));
    }
    Ok(())
}

fn run(st: &AppState, src: &dyn ModelSource, job: &Job) {
    let progress = |p: PullProgress| {
        if let Ok(mut v) = job.view.lock() {
            v.on_progress(p);
        }
    };
    let r = src.pull(&job.spec, &job.cancel, &progress);
    let status = if let Ok(mut v) = job.view.lock() {
        match r {
            Ok(out) => {
                v.model = Some(out.id.clone());
                v.result = Some(out);
                v.finish(Status::Succeeded);
            }
            Err(e)
                if e.kind == SourceErrorKind::Cancelled || job.cancel.load(Ordering::Relaxed) =>
            {
                v.error = Some(("cancelled", e.message));
                v.finish(Status::Cancelled);
            }
            Err(e) => {
                v.error = Some((error_code(e.kind), e.message));
                v.finish(Status::Failed);
            }
        }
        v.status
    } else {
        Status::Failed
    };
    let model = job
        .view
        .lock()
        .ok()
        .and_then(|v| v.model.clone())
        .unwrap_or_default();
    let fields = [
        ("job_id", job.id.as_str().into()),
        ("kind", "pull".into()),
        ("model", model.into()),
        ("status", status.as_str().into()),
    ];
    if status == Status::Failed {
        st.log.warn("job_finished", &fields);
    } else {
        st.log.info("job_finished", &fields);
    }
}

fn accepted(job: &Job) -> Response {
    let mut resp = (StatusCode::ACCEPTED, Json(job.to_json())).into_response();
    if let Ok(v) = HeaderValue::from_str(&format!("/v1/admin/jobs/{}", job.id)) {
        resp.headers_mut().insert("location", v);
    }
    resp
}

fn opt_string(body: &Value, key: &'static str) -> Result<Option<String>, ApiError> {
    match body.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if !s.trim().is_empty() => Ok(Some(s.trim().to_owned())),
        Some(Value::String(_)) => Ok(None),
        Some(_) => Err(ApiError::invalid_param(
            key,
            format!("{key} must be a string"),
        )),
    }
}

/// The pull a request body asks for.
fn pull_spec(body: &Value) -> Result<PullSpec, ApiError> {
    let repository = opt_string(body, "repository")?.ok_or_else(|| {
        ApiError::invalid_param("repository", "give the repository to pull, as owner/name")
    })?;
    let mut parts = repository.split('/');
    let well_formed = matches!(
        (parts.next(), parts.next(), parts.next()),
        (Some(o), Some(n), None) if !o.is_empty() && !n.is_empty()
    );
    if !well_formed {
        return Err(ApiError::invalid_param(
            "repository",
            format!("{repository:?} is not a repository id: use owner/name"),
        ));
    }
    let files = match body.get("files") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => a
            .iter()
            .map(|f| {
                f.as_str().map(str::to_owned).ok_or_else(|| {
                    ApiError::invalid_param("files", "files must be a list of file paths")
                })
            })
            .collect::<Result<_, _>>()?,
        Some(_) => {
            return Err(ApiError::invalid_param(
                "files",
                "files must be a list of file paths",
            ));
        }
    };
    let set = match body.get("set") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(m)) => m.clone(),
        Some(_) => {
            return Err(ApiError::invalid_param(
                "set",
                "set must be an object of config values",
            ));
        }
    };
    Ok(PullSpec {
        repository,
        revision: opt_string(body, "revision")?,
        files,
        id: opt_string(body, "id")?,
        name: opt_string(body, "name")?,
        set,
    })
}

pub(crate) async fn search(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    query: Result<Query<HashMap<String, String>>, QueryRejection>,
) -> Response {
    let began = Instant::now();
    let rid = st.request_id(&headers);
    let r = async {
        st.check_admin(&headers)?;
        let src = source(&st)?;
        let Query(q) = query.map_err(|_| ApiError::invalid("invalid query string"))?;
        let text = q
            .get("q")
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| ApiError::invalid_param("q", "give something to search for in q"))?;
        let limit = match q.get("limit") {
            None => SEARCH_DEFAULT,
            Some(l) => l
                .parse::<usize>()
                .ok()
                .filter(|l| (1..=SEARCH_MAX).contains(l))
                .ok_or_else(|| {
                    ApiError::invalid_param("limit", format!("limit must be 1 to {SEARCH_MAX}"))
                })?,
        };
        let q2 = text.clone();
        let (hits, installed) =
            tokio::task::spawn_blocking(move || (src.search(&q2, limit), src.installed()))
                .await
                .map_err(|_| ApiError::internal(&rid))?;
        let hits = hits.map_err(ApiError::from_source)?;
        let data: Vec<Value> = hits
            .iter()
            .map(|h| {
                let installed_as: Vec<&str> = installed
                    .iter()
                    .filter(|(repo, _)| repo.eq_ignore_ascii_case(&h.repository))
                    .map(|(_, id)| id.as_str())
                    .collect();
                json!({
                    "repository": h.repository,
                    "downloads": h.downloads,
                    "likes": h.likes,
                    "formats": h.formats,
                    "gated": h.gated,
                    "installed_as": installed_as,
                })
            })
            .collect();
        Ok(Json(json!({ "object": "list", "query": text, "data": data })).into_response())
    }
    .await;
    st.finish("admin_models_search", &rid, began, r)
}

pub(crate) async fn pull(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    let began = Instant::now();
    let rid = st.request_id(&headers);
    let r = (|| {
        st.check_admin(&headers)?;
        let src = source(&st)?;
        let spec = pull_spec(&json_body(&st, body)?)?;
        let job = st.jobs.admit(spec)?;
        spawn(&st, src, &job)?;
        Ok(accepted(&job))
    })();
    st.finish("admin_models_pull", &rid, began, r)
}

pub(crate) async fn list(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let began = Instant::now();
    let rid = st.request_id(&headers);
    let r = st.check_admin(&headers).map(|()| {
        // Newest first.
        let data: Vec<Value> = st.jobs.list().iter().rev().map(|j| j.to_json()).collect();
        Json(json!({ "object": "list", "data": data })).into_response()
    });
    st.finish("admin_jobs", &rid, began, r)
}

pub(crate) async fn get(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let began = Instant::now();
    let rid = st.request_id(&headers);
    let r = st
        .check_admin(&headers)
        .and_then(|()| st.jobs.get(&id))
        .map(|j| Json(j.to_json()).into_response());
    st.finish("admin_jobs", &rid, began, r)
}

/// Asks a running job to stop. It answers at once; the job reads `cancelled` when its
/// thread has stopped. What arrived is kept, so a retry continues.
pub(crate) async fn cancel(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let began = Instant::now();
    let rid = st.request_id(&headers);
    let r = (|| {
        st.check_admin(&headers)?;
        let job = st.jobs.get(&id)?;
        if job.status().finished() {
            return Err(ApiError::conflict(format!(
                "job {id} has already {}",
                job.status().as_str()
            )));
        }
        job.cancel.store(true, Ordering::Relaxed);
        Ok(accepted(&job))
    })();
    st.finish("admin_jobs", &rid, began, r)
}

/// Starts a failed or cancelled pull again, as a new job. A pull continues from the
/// files that already arrived.
pub(crate) async fn retry(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let began = Instant::now();
    let rid = st.request_id(&headers);
    let r = (|| {
        st.check_admin(&headers)?;
        let src = source(&st)?;
        let old = st.jobs.get(&id)?;
        if !matches!(old.status(), Status::Failed | Status::Cancelled) {
            return Err(ApiError::conflict(format!(
                "job {id} is {}; only a failed or cancelled job can be retried",
                old.status().as_str()
            )));
        }
        let job = st.jobs.admit(old.spec.clone())?;
        spawn(&st, src, &job)?;
        Ok(accepted(&job))
    })();
    st.finish("admin_jobs", &rid, began, r)
}
