//! The QUBIVEX HTTP gateway (Phase 9): OpenAI-compatible endpoints over the runtime.
//!
//! - `GET /health`, `GET /metrics` (Prometheus text), `GET /v1/system`
//! - `GET /v1/models`, `GET /v1/models/{id}`
//! - `POST /v1/chat/completions` and `POST /v1/completions`, streaming (SSE) or not
//! - `POST /v1/models/load` and `POST /v1/models/unload`, behind the admin token, with
//!   `POST /v1/admin/models` and `DELETE /v1/admin/models/{id}` as aliases
//!
//! Tokio and Kerniq Server (over Axum and Hyper) stay at the edge: every generation runs on its own dedicated thread
//! (`worker`) and reaches the request over a bounded channel (QUBIVEX_DESIGN §4.7,
//! RUST_GUIDELINES §4). Each model has a [`Scheduler`](qubivex_runtime::scheduler) that
//! decides when a request runs: its queue is bounded, and memory admission holds each
//! conversation's KV against the plan with every other live one. A request whose client
//! goes away is cancelled: generation stops within one prefill chunk or one token, and
//! a stream that stops being read ends after [`ServerConfig::stall_timeout`]. Every
//! response carries an `x-request-id`; errors are OpenAI-shaped and never contain file
//! paths.
//!
//! Spec §8: `/metrics` reports the inference, memory and serving numbers (`metrics`),
//! and the log is one line per event, as text or JSON, with request, model and
//! sequence ids and never a prompt or generated text (`log`).
//!
//! Spec §11: an optional per-client rate limit, limits on the body, the prompt and
//! `max_tokens`, CORS for configured origins, and no listening beyond loopback without
//! an API key unless that is asked for explicitly. Named keys ([`ClientKey`]) tell
//! clients apart: a named client's requests are rate-limited, scheduled fairly and
//! scoped in the prefix cache as that client, whatever `user` they send.

mod error;
mod keys;
mod limits;
mod log;
mod metrics;
mod models;
mod openai;
mod worker;

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::future::Future;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub use error::ApiError;
use kerniq_server::App;
use kerniq_server::axum::extract::rejection::JsonRejection;
use kerniq_server::axum::extract::{DefaultBodyLimit, Path, State};
use kerniq_server::axum::http::{HeaderMap, HeaderValue, StatusCode};
use kerniq_server::axum::response::sse::{Event as SseEvent, Sse};
use kerniq_server::axum::response::{IntoResponse, Response};
use kerniq_server::axum::{Json, Router};
use kerniq_server::config::ServerConfig as ServeConfig;
use kerniq_server::routing::Router as Routes;
pub use keys::{ClientKey, hash_key};
pub use limits::RateLimit;
pub use log::LogFormat;
pub use metrics::Metrics;
use qubivex_runtime::scheduler::{EnterError, Outcome, RequestInfo};
pub use qubivex_runtime::scheduler::{MemoryPolicy, Stats};
use qubivex_runtime::{Engine, EngineOptions};
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot, watch};
use tokio_stream::wrappers::ReceiverStream;

use crate::keys::Caller;
use crate::limits::RateLimiter;
use crate::log::{Log, millis};
use crate::models::Entry;
use crate::openai::{ChatRequest, CompletionRequest, Params};
use crate::worker::{CancelOnDrop, Event, FinishReason, Input, Job};

/// How long the server gives a stopped generation's thread and connection to end, after
/// it has told them to.
const STOP_MARGIN: Duration = Duration::from_secs(5);

/// Server settings.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// A key shared by every client, accepted as `Authorization: Bearer <key>` on
    /// `/v1/*`. With it or [`client_keys`](Self::client_keys) set, `/v1/*` needs one of
    /// them.
    pub api_key: Option<String>,
    /// Named keys, one per client (ADR-013 amendment 7).
    pub client_keys: Vec<ClientKey>,
    /// Required on the admin endpoints, which are disabled without it.
    pub admin_token: Option<String>,
    /// Serve beyond loopback without `api_key`. Off: [`Gateway::serve`] refuses such a
    /// listener ([`ServerConfig::check_listen`]).
    pub allow_unauthenticated_remote: bool,
    /// Requests per client on `/v1/*`: per named key, else per peer address. A named key
    /// may set its own. `None`: unlimited.
    pub rate_limit: Option<RateLimit>,
    /// Origins browsers may call the API from (`scheme://host[:port]`, or `*`). Empty:
    /// no CORS headers.
    pub cors_origins: Vec<String>,
    /// Largest request body, in bytes; a larger one gets a 413.
    pub max_body_bytes: usize,
    /// Longest prompt, in tokens after the chat template; a longer one gets a 400.
    /// `None`: only the context limits it.
    pub max_prompt_tokens: Option<usize>,
    /// Largest `max_tokens` a request may ask for; more gets a 400. `None`: only the
    /// context limits it.
    pub max_generation_tokens: Option<usize>,
    /// Generations that run at once per model; more wait for a slot.
    pub max_concurrent: usize,
    /// Requests per model that may wait beyond those running. More get a 429 at once.
    pub max_queue: usize,
    /// How long a request waits to start before a 429.
    pub queue_timeout: Duration,
    /// What a request does when its memory is not available yet.
    pub memory_policy: MemoryPolicy,
    /// How long a waiting request takes to gain one priority level. `None`: never.
    pub priority_aging: Option<Duration>,
    /// Longest a generation runs; it then ends as if it reached `max_tokens`.
    pub request_timeout: Duration,
    /// How long a stream may go without the client reading before it ends. A client
    /// that keeps its connection open but stops reading would otherwise hold a slot and
    /// its memory for as long as it liked.
    pub stall_timeout: Duration,
    /// How long running requests get to finish at shutdown before they are stopped.
    pub shutdown_grace: Duration,
    /// `max_tokens` when a request does not say (capped by the context).
    pub default_max_tokens: usize,
    /// Prompt tokens per forward pass (how soon a cancelled prefill stops).
    pub prefill_chunk: usize,
    /// Directories the admin load endpoint searches for model ids.
    pub model_dirs: Vec<PathBuf>,
    /// How the admin load endpoint loads models.
    pub engine: EngineOptions,
    /// How the log is written: text, JSON lines, or not at all.
    pub log_format: LogFormat,
    /// Serve `GET /metrics`. Off: it answers 404.
    pub metrics_enabled: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            client_keys: Vec::new(),
            admin_token: None,
            allow_unauthenticated_remote: false,
            rate_limit: None,
            cors_origins: Vec::new(),
            max_body_bytes: 4 << 20,
            max_prompt_tokens: None,
            max_generation_tokens: None,
            max_concurrent: 4,
            max_queue: 32,
            queue_timeout: Duration::from_secs(30),
            memory_policy: MemoryPolicy::Queue,
            priority_aging: Some(Duration::from_secs(10)),
            request_timeout: Duration::from_secs(600),
            stall_timeout: Duration::from_secs(30),
            shutdown_grace: Duration::from_secs(30),
            default_max_tokens: 512,
            prefill_chunk: 256,
            model_dirs: Vec::new(),
            engine: EngineOptions::default(),
            log_format: LogFormat::default(),
            metrics_enabled: true,
        }
    }
}

struct AppState {
    cfg: ServerConfig,
    limiter: Option<RateLimiter>,
    /// Never held across an `.await`.
    models: RwLock<BTreeMap<String, Entry>>,
    /// Unloaded engines, to see when running requests have let go of them.
    unloaded: Mutex<Vec<Weak<Engine>>>,
    metrics: Arc<Metrics>,
    log: Arc<Log>,
    next_id: AtomicU64,
    /// The next sequence id: one per admitted generation, in order of admission.
    next_sequence: AtomicU64,
    /// Set when the shutdown grace ends: running requests stop.
    stop_now: watch::Sender<bool>,
}

/// The gateway: models plus settings. Clones share state.
#[derive(Clone)]
pub struct Gateway {
    state: Arc<AppState>,
}

impl std::fmt::Debug for Gateway {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gateway")
            .field("models", &self.model_ids())
            .finish_non_exhaustive()
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Compares secrets in time independent of where they differ.
fn same_secret(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

pub(crate) fn bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
}

impl Gateway {
    /// A gateway with no models, logging to standard error.
    #[must_use]
    pub fn new(cfg: ServerConfig) -> Self {
        let log = Log::stderr(cfg.log_format);
        Self::with(cfg, log)
    }

    /// A gateway with no models that writes its log lines to `log` instead of standard
    /// error, in [`ServerConfig::log_format`].
    #[must_use]
    pub fn with_log_writer(cfg: ServerConfig, log: Box<dyn std::io::Write + Send>) -> Self {
        let log = Log::to(cfg.log_format, log);
        Self::with(cfg, log)
    }

    fn with(cfg: ServerConfig, log: Log) -> Self {
        Self {
            state: Arc::new(AppState {
                limiter: RateLimiter::with_keys(
                    cfg.rate_limit,
                    cfg.client_keys.iter().map(|k| k.rate_limit).collect(),
                ),
                cfg,
                models: RwLock::new(BTreeMap::new()),
                unloaded: Mutex::new(Vec::new()),
                metrics: Arc::new(Metrics::default()),
                log: Arc::new(log),
                next_id: AtomicU64::new(1),
                next_sequence: AtomicU64::new(1),
                stop_now: watch::Sender::new(false),
            }),
        }
    }

    /// Serves `engine` under `id`.
    ///
    /// # Errors
    /// [`ApiError`] (409) if `id` is already served.
    pub fn add_model(&self, id: &str, engine: Engine) -> Result<(), ApiError> {
        self.state.insert(id, engine)
    }

    /// The ids served.
    #[must_use]
    pub fn model_ids(&self) -> Vec<String> {
        self.state.ready().into_iter().map(|(id, _)| id).collect()
    }

    /// Unloaded models whose weights are still held by running requests.
    #[must_use]
    pub fn draining_models(&self) -> usize {
        self.state.draining()
    }

    /// The queue of model `id`, if it is loaded.
    #[must_use]
    pub fn queue_stats(&self, id: &str) -> Option<Stats> {
        self.state.model(id).ok().map(|l| l.scheduler.stats())
    }

    /// KV positions reserved on model `id`, if it is loaded.
    #[must_use]
    pub fn kv_reserved(&self, id: &str) -> Option<usize> {
        self.state.model(id).ok().map(|l| l.engine.kv_reserved())
    }

    /// The gateway's counters.
    #[must_use]
    pub fn metrics(&self) -> Arc<Metrics> {
        Arc::clone(&self.state.metrics)
    }

    /// The routes, middleware and state, as a Kerniq application.
    fn app(&self) -> App<Arc<AppState>> {
        let routes = Routes::new()
            .get("/health", health)
            .get("/metrics", metrics_text)
            .get("/v1/system", system)
            .get("/v1/models", list_models)
            .post("/v1/models/load", admin_load)
            .post("/v1/models/unload", models_unload)
            .get("/v1/models/{id}", get_model)
            .post("/v1/chat/completions", chat)
            .post("/v1/completions", completions)
            .post("/v1/admin/models", admin_load)
            .delete("/v1/admin/models/{id}", admin_unload);
        // Kerniq's defaults add no request timeout (a stream may run for minutes), no
        // concurrency limit (the schedulers queue) and no body limit of its own (ours
        // below answers in the OpenAI shape).
        let serve =
            ServeConfig::default().shutdown_timeout(self.state.cfg.shutdown_grace + STOP_MARGIN);
        App::new()
            .state(Arc::clone(&self.state))
            .config(serve)
            .merge(routes)
            .fallback(|| async { ApiError::not_found() })
            // The first declared is the outermost: CORS answers preflights before the
            // rate limit, and adds its headers to the rate limit's 429s.
            .layer(kerniq_server::axum::middleware::from_fn_with_state(
                Arc::clone(&self.state),
                limits::cors,
            ))
            .layer(kerniq_server::axum::middleware::from_fn_with_state(
                Arc::clone(&self.state),
                limits::rate_limit,
            ))
            .layer(DefaultBodyLimit::max(self.state.cfg.max_body_bytes))
    }

    /// The gateway as an Axum router, for tests and for embedding in another server.
    pub fn router(&self) -> Router {
        self.app().into_axum_router()
    }

    /// Serves on `listener` until `shutdown` completes. New requests are then refused
    /// and waiting ones end with a 503. Running requests get
    /// [`ServerConfig::shutdown_grace`] to finish, and are stopped after that. Returns
    /// when they have ended; a connection that will not close cannot hold it up.
    ///
    /// Refuses (`InvalidInput`) settings that [`ServerConfig::validate`] rejects, and a
    /// listener beyond loopback without an API key ([`ServerConfig::check_listen`]).
    ///
    /// # Errors
    /// Those refusals, and I/O errors from the listener.
    pub async fn serve(
        self,
        listener: tokio::net::TcpListener,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> std::io::Result<()> {
        let st = Arc::clone(&self.state);
        let refuse = |m: String| std::io::Error::new(std::io::ErrorKind::InvalidInput, m);
        st.cfg.validate().map_err(refuse)?;
        st.cfg
            .check_listen(listener.local_addr()?)
            .map_err(refuse)?;
        let grace = st.cfg.shutdown_grace;
        if !st.cfg.needs_key() && !listener.local_addr()?.ip().is_loopback() {
            st.log.warn(
                "serving_without_api_key",
                &[("listen", listener.local_addr()?.to_string().into())],
            );
        }
        st.log.info(
            "server_started",
            &[
                ("listen", listener.local_addr()?.to_string().into()),
                ("models", st.ready().len().into()),
            ],
        );
        let (began_tx, began_rx) = oneshot::channel::<()>();
        let begin = Arc::clone(&st);
        let signal = async move {
            shutdown.await;
            begin.log.info(
                "server_stopping",
                &[("generations_active", begin.metrics.active().into())],
            );
            begin.begin_shutdown();
            let _ = began_tx.send(());
        };
        // Kerniq inserts the peer address, which the rate limit counts by, and closes
        // connections still open after the grace and the stop margin.
        let server = self.app().shutdown_signal(signal).serve(listener);
        let stopper = async {
            if began_rx.await.is_err() {
                std::future::pending::<()>().await;
            }
            tokio::time::sleep(grace).await;
            st.stop_now.send_replace(true);
            tokio::time::sleep(STOP_MARGIN).await;
        };
        let served = tokio::select! {
            r = server => r.map_err(|e| std::io::Error::other(e.to_string())),
            () = stopper => Ok(()),
        };
        // A server that drained early never reached the grace's end.
        st.stop_now.send_replace(true);
        let waited = Instant::now();
        while st.metrics.active() > 0 && waited.elapsed() < STOP_MARGIN {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        st.log.info(
            "server_stopped",
            &[("generations_active", st.metrics.active().into())],
        );
        served
    }
}

impl AppState {
    fn request_id(&self, headers: &HeaderMap) -> String {
        let given = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .filter(|s| {
                !s.is_empty()
                    && s.len() <= 64
                    && s.bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
            });
        match given {
            Some(s) => s.to_owned(),
            None => format!(
                "req-{:x}-{:x}",
                now_secs(),
                self.next_id.fetch_add(1, Ordering::Relaxed)
            ),
        }
    }

    /// Checks the request's key, and says whose it is.
    fn check_key(&self, headers: &HeaderMap) -> Result<Caller, ApiError> {
        let given = bearer(headers);
        if let Some(i) = keys::identify(&self.cfg.client_keys, given) {
            return Ok(Caller::Client(i));
        }
        let shared = self
            .cfg
            .api_key
            .as_deref()
            .is_some_and(|k| given.is_some_and(|b| same_secret(b, k)));
        if self.cfg.needs_key() && !shared {
            return Err(ApiError::unauthorized());
        }
        Ok(Caller::Unnamed)
    }

    /// Sets who `params` counts as, from the caller and its `user`.
    fn identify(&self, caller: Caller, params: &mut Params) {
        let (client, scope) =
            keys::identity(&self.cfg.client_keys, caller, params.user.as_deref());
        params.client = client;
        params.cache_scope = scope;
        params.client_name = match caller {
            Caller::Client(i) => Some(self.cfg.client_keys[i].client.clone()),
            Caller::Unnamed => None,
        };
    }

    fn check_admin(&self, headers: &HeaderMap) -> Result<(), ApiError> {
        match &self.cfg.admin_token {
            None => Err(ApiError::admin_disabled()),
            Some(t) if bearer(headers).is_some_and(|b| same_secret(b, t)) => Ok(()),
            Some(_) => Err(ApiError::unauthorized()),
        }
    }

    /// Refuses new requests and ends the waiting ones.
    fn begin_shutdown(&self) {
        for (_, l) in self.ready() {
            l.scheduler.close();
        }
    }

    /// Adds the request id header, and counts and logs the request. `began` is when it
    /// arrived.
    fn finish(
        &self,
        endpoint: &'static str,
        rid: &str,
        began: Instant,
        r: Result<Response, ApiError>,
    ) -> Response {
        let code = r.as_ref().err().map(|e| e.code);
        let mut resp = r.unwrap_or_else(IntoResponse::into_response);
        let status = resp.status().as_u16();
        let took = began.elapsed();
        self.metrics.request(endpoint, status);
        self.metrics.answered(endpoint, took);
        let mut fields = vec![
            ("request_id", rid.into()),
            ("endpoint", endpoint.into()),
            ("status", status.into()),
            ("duration_ms", millis(took)),
        ];
        if let Some(code) = code {
            fields.push(("error", code.into()));
        }
        if status >= 500 {
            self.log.warn("request", &fields);
        } else {
            self.log.info("request", &fields);
        }
        if let Ok(v) = HeaderValue::from_str(rid) {
            resp.headers_mut().insert("x-request-id", v);
        }
        resp
    }
}

async fn health(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({ "status": "ok", "models": st.ready().len() }))
}

/// What `/metrics` reports for one loaded model, by the names in
/// [`metrics::MODEL_VALUES`].
fn model_values(cfg: &ServerConfig, l: &models::Loaded) -> metrics::ModelValues {
    let e = &l.engine;
    let q = l.scheduler.stats();
    let p = e.prefix_stats();
    let s = e.store_stats();
    let b = e.batch_stats();
    let ram = e.ledger().snapshot(qubivex_memory::Tier::Ram);
    let budget = e.memory_plan().map_or(0, |p| p.ram.budget.bytes.0);
    vec![
        ("qubivex_kv_reserved_positions", e.kv_reserved() as f64),
        ("qubivex_expert_cache_bytes", e.expert_cache_bytes() as f64),
        (
            "qubivex_generation_slots_free",
            cfg.max_concurrent.saturating_sub(q.running) as f64,
        ),
        ("qubivex_queue_depth", q.queued as f64),
        ("qubivex_queue_rejected_total", q.rejected as f64),
        ("qubivex_queue_granted_total", q.granted as f64),
        ("qubivex_prefix_cache_lookups_total", p.lookups as f64),
        ("qubivex_prefix_cache_hits_total", p.hits as f64),
        (
            "qubivex_prefix_cache_tokens_reused_total",
            p.tokens_reused as f64,
        ),
        ("qubivex_prefix_cache_evictions_total", p.evicted as f64),
        ("qubivex_prefix_cache_positions", p.cost as f64),
        ("qubivex_expert_cache_hits_total", s.hits as f64),
        ("qubivex_expert_cache_loads_total", s.loads as f64),
        ("qubivex_expert_cache_evictions_total", s.evictions as f64),
        ("qubivex_expert_read_bytes_total", s.bytes_read as f64),
        (
            "qubivex_expert_load_wait_seconds_total",
            Duration::from_nanos(s.wait_ns).as_secs_f64(),
        ),
        ("qubivex_expert_read_retries_total", s.retried as f64),
        ("qubivex_expert_read_timeouts_total", s.timeouts as f64),
        ("qubivex_batch_passes_total", b.passes as f64),
        ("qubivex_batch_steps_total", b.steps as f64),
        ("qubivex_memory_dense_bytes", ram.dense.0 as f64),
        (
            "qubivex_memory_expert_cache_bytes",
            ram.expert_cache.0 as f64,
        ),
        ("qubivex_memory_kv_bytes", ram.kv.0 as f64),
        ("qubivex_memory_activations_bytes", ram.activations.0 as f64),
        ("qubivex_memory_budget_bytes", budget as f64),
    ]
}

/// This process's resident set size and its peak, where the OS says.
fn process_memory() -> (Option<u64>, Option<u64>) {
    let host = qubivex_hardware::Host::live();
    let known = |p: qubivex_hardware::Probed<_>| match p {
        qubivex_hardware::Probed::Known(bytes) => Some(bytes),
        qubivex_hardware::Probed::Unknown(_) => None,
    };
    (
        known(qubivex_hardware::process_rss(&host)).map(|b| b.0),
        known(qubivex_hardware::process_peak_rss(&host)).map(|b| b.0),
    )
}

async fn metrics_text(State(st): State<Arc<AppState>>) -> Response {
    if !st.cfg.metrics_enabled {
        return ApiError::not_found().into_response();
    }
    let mut out = String::new();
    st.metrics.render(&mut out);
    let models: Vec<(String, metrics::ModelValues)> = st
        .ready()
        .iter()
        .map(|(id, l)| (id.clone(), model_values(&st.cfg, l)))
        .collect();
    metrics::render_models(&mut out, &models);
    // Reading the process's memory can run a command (macOS): off the Tokio workers.
    let (rss, peak) = tokio::task::spawn_blocking(process_memory)
        .await
        .unwrap_or((None, None));
    if let Some(rss) = rss {
        metrics::render_value(
            &mut out,
            "qubivex_process_resident_bytes",
            "gauge",
            "Resident set size of the serving process.",
            rss as f64,
        );
        // Where the OS keeps no high-water mark we can read, the highest scrape stands in.
        let seen = st.metrics.note_rss(rss);
        metrics::render_value(
            &mut out,
            "qubivex_process_peak_resident_bytes",
            "gauge",
            "Peak resident set size: the kernel's high-water mark on Linux, elsewhere the highest value a scrape has read.",
            peak.unwrap_or(seen).max(rss) as f64,
        );
    }
    (
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        out,
    )
        .into_response()
}

async fn system(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let began = Instant::now();
    let rid = st.request_id(&headers);
    let r = async {
        st.check_key(&headers)?;
        let hardware = tokio::task::spawn_blocking(|| {
            serde_json::to_value(qubivex_hardware::probe(&qubivex_hardware::Host::live()))
                .unwrap_or(Value::Null)
        })
        .await
        .unwrap_or(Value::Null);
        let models: Vec<Value> = st
            .ready()
            .iter()
            .map(|(id, l)| {
                let e = &l.engine;
                let queue = l.scheduler.stats();
                let prefix = e.prefix_stats();
                let ledger = e.ledger().snapshot(qubivex_memory::Tier::Ram);
                json!({
                    "id": id,
                    "architecture": e.metadata().manifest.architecture.id,
                    "max_context": e.max_context(),
                    "generation_slots": st.cfg.max_concurrent,
                    "generation_slots_free": st.cfg.max_concurrent.saturating_sub(queue.running),
                    "queue_depth": queue.queued,
                    "queue_limit": st.cfg.max_queue,
                    "kv_reserved_positions": e.kv_reserved(),
                    "prefix_cache": {
                        "entries": prefix.entries,
                        "positions": prefix.cost,
                        "lookups": prefix.lookups,
                        "hits": prefix.hits,
                        "tokens_reused": prefix.tokens_reused,
                        "evicted": prefix.evicted,
                    },
                    "expert_cache_bytes": e.expert_cache_bytes(),
                    "plan_budget_bytes": e.memory_plan().map(|p| p.ram.budget.bytes.0),
                    "ledger_bytes": {
                        "dense": ledger.dense.0,
                        "expert_cache": ledger.expert_cache.0,
                        "kv": ledger.kv.0,
                    },
                    "placement": format!("{:?}", e.placement().backend).to_lowercase(),
                })
            })
            .collect();
        Ok(Json(json!({
            "object": "system",
            "hardware": hardware,
            "models": models,
            "model_states": st.states(),
            "models_draining": st.draining(),
            "generations_active": st.metrics.active(),
        }))
        .into_response())
    }
    .await;
    st.finish("system", &rid, began, r)
}

fn model_card(id: &str, created: u64) -> Value {
    json!({ "id": id, "object": "model", "created": created, "owned_by": "qubivex" })
}

async fn list_models(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let began = Instant::now();
    let rid = st.request_id(&headers);
    let r = st.check_key(&headers).map(|_| {
        let data: Vec<Value> = st
            .ready()
            .iter()
            .map(|(id, l)| model_card(id, l.created))
            .collect();
        Json(json!({ "object": "list", "data": data })).into_response()
    });
    st.finish("models", &rid, began, r)
}

async fn get_model(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let began = Instant::now();
    let rid = st.request_id(&headers);
    let r = st
        .check_key(&headers)
        .and_then(|_| st.model(&id))
        .map(|l| Json(model_card(&id, l.created)).into_response());
    st.finish("models", &rid, began, r)
}

fn json_body<T>(st: &AppState, body: Result<Json<T>, JsonRejection>) -> Result<T, ApiError> {
    body.map(|Json(b)| b).map_err(|e| {
        if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
            return ApiError::too_large(st.cfg.max_body_bytes);
        }
        // The rejection text names the field and the problem, never a path.
        ApiError::invalid(format!("invalid request body: {}", e.body_text()))
    })
}

/// What a completion endpoint returns, by kind.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Chat,
    Text,
}

async fn chat(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<ChatRequest>, JsonRejection>,
) -> Response {
    let began = Instant::now();
    let rid = st.request_id(&headers);
    let r = async {
        let caller = st.check_key(&headers)?;
        let req = json_body(&st, body)?;
        let messages = req.messages()?;
        let max = req.max_completion_tokens.or(req.common.max_tokens);
        let mut params = req.common.params(max)?;
        st.identify(caller, &mut params);
        generate(
            &st,
            &rid,
            &req.model,
            Input::Chat(messages),
            params,
            Kind::Chat,
        )
        .await
    }
    .await;
    st.finish("chat_completions", &rid, began, r)
}

async fn completions(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let began = Instant::now();
    let rid = st.request_id(&headers);
    let r = async {
        let caller = st.check_key(&headers)?;
        let req = json_body(&st, body)?;
        let prompt = req.prompt()?;
        let max = req.common.max_tokens;
        let mut params = req.common.params(max)?;
        st.identify(caller, &mut params);
        generate(
            &st,
            &rid,
            &req.model,
            Input::Text(prompt),
            params,
            Kind::Text,
        )
        .await
    }
    .await;
    st.finish("completions", &rid, began, r)
}

/// Completes when the server has told running requests to stop.
async fn stopped(stop: &mut watch::Receiver<bool>) {
    if stop.wait_for(|s| *s).await.is_err() {
        std::future::pending::<()>().await;
    }
}

/// Queues a generation, starts it, and answers with its result or its stream.
async fn generate(
    st: &Arc<AppState>,
    rid: &str,
    model: &str,
    input: Input,
    params: Params,
    kind: Kind,
) -> Result<Response, ApiError> {
    if let (Some(asked), Some(cap)) = (params.max_tokens, st.cfg.max_generation_tokens)
        && asked > cap
    {
        return Err(ApiError::invalid_param(
            "max_tokens",
            format!("max_tokens is {asked}; this server allows at most {cap}"),
        ));
    }
    let arrived = Instant::now();
    let limit = params.timeout.map(|t| arrived + t);
    let queue_until = arrived + st.cfg.queue_timeout;
    let wait_until = limit.map_or(queue_until, |l| l.min(queue_until));
    let loaded = st.model_ready(model, wait_until).await?;
    let timings = st.metrics.model(model);
    // The place is taken before any work, so a burst past the limit costs nothing.
    let ticket = loaded
        .scheduler
        .enter(RequestInfo {
            priority: params.priority,
            client: params.client.clone(),
        })
        .map_err(|e| match e {
            EnterError::Full => ApiError::queue_full(st.cfg.max_concurrent, st.cfg.max_queue),
            EnterError::Closed => ApiError::model_closing(),
        })?;
    let mut request = worker::request(
        input,
        &params,
        st.cfg.default_max_tokens,
        st.cfg.prefill_chunk,
    );
    let engine = Arc::clone(&loaded.engine);
    // Tokenizing a long prompt is CPU work: off the Tokio workers.
    let (returned, prepared) = tokio::task::spawn_blocking(move || {
        let prepared = engine.prepare(&request);
        (request, prepared)
    })
    .await
    .map_err(|_| ApiError::internal(rid))?;
    request = returned;
    let prepared = prepared.map_err(|e| ApiError::from_request(&e, rid, &st.log))?;
    if let Some(cap) = st.cfg.max_prompt_tokens {
        let n = prepared.prompt_tokens();
        if n > cap {
            let param = match kind {
                Kind::Chat => "messages",
                Kind::Text => "prompt",
            };
            return Err(ApiError::prompt_too_long(param, n, cap));
        }
    }
    let (grant_tx, grant_rx) = oneshot::channel();
    ticket.ready(prepared.positions() as u64, move |o| {
        let _ = grant_tx.send(o);
    });
    let grant = match tokio::time::timeout_at(wait_until.into(), grant_rx).await {
        Ok(Ok(Outcome::Granted(g))) => g,
        Ok(Ok(Outcome::Refused(e))) => return Err(ApiError::from_runtime(&e, rid, &st.log)),
        Ok(Ok(Outcome::Closed) | Err(_)) => return Err(ApiError::model_closing()),
        // Dropping the ticket on the way out leaves the queue.
        Err(_) if limit.is_some_and(|l| l <= queue_until) => {
            return Err(ApiError::deadline_exceeded());
        }
        Err(_) => return Err(ApiError::busy(st.cfg.max_concurrent)),
    };
    let now = Instant::now();
    let waited = now - arrived;
    timings.queue_wait.observe_duration(waited);
    let sequence = st.next_sequence.fetch_add(1, Ordering::Relaxed);
    let mut fields = vec![
        ("request_id", rid.into()),
        ("model", model.into()),
        ("sequence_id", sequence.into()),
        ("prompt_tokens", prepared.prompt_tokens().into()),
        ("stream", params.stream.into()),
        ("queue_ms", millis(waited)),
    ];
    if let Some(c) = &params.client_name {
        fields.push(("client", c.as_str().into()));
    }
    st.log.info("generation_started", &fields);
    let run_limit = now + st.cfg.request_timeout;
    request.deadline = Some(limit.map_or(run_limit, |l| l.min(run_limit)));
    let cancel = Arc::new(AtomicBool::new(false));
    // Dropped (cancelling the generation) when the request is abandoned: the handler
    // future for a non-streaming request, the forwarding task for a stream.
    let guard = CancelOnDrop(Arc::clone(&cancel));
    let (tx, mut rx) = mpsc::channel(64);
    let mut stop = st.stop_now.subscribe();
    worker::spawn(
        Job {
            engine: Arc::clone(&loaded.engine),
            request,
            prepared,
            grant,
            cancel,
            request_id: rid.to_owned(),
            model: model.to_owned(),
            sequence,
            arrived,
            metrics: Arc::clone(&st.metrics),
            timings,
            log: Arc::clone(&st.log),
        },
        tx,
    );
    // The first event decides the status: admission and prompt errors are HTTP errors
    // even for a stream.
    let prompt_tokens = match rx.recv().await {
        Some(Event::Started { prompt_tokens }) => prompt_tokens,
        Some(Event::Failed(e)) => return Err(e),
        _ => return Err(ApiError::internal(rid)),
    };
    let reply = Reply {
        id: match kind {
            Kind::Chat => format!("chatcmpl-{rid}"),
            Kind::Text => format!("cmpl-{rid}"),
        },
        kind,
        created: now_secs(),
        model: model.to_owned(),
        prompt_tokens,
        include_usage: params.include_usage,
    };
    if !params.stream {
        let mut text = String::new();
        loop {
            let event = tokio::select! {
                e = rx.recv() => e,
                () = stopped(&mut stop) => return Err(ApiError::shutting_down()),
            };
            match event {
                Some(Event::Text(t)) => text.push_str(&t),
                Some(Event::Done {
                    finish,
                    completion_tokens,
                    cached_tokens,
                }) => {
                    drop(guard);
                    let tokens = usage(prompt_tokens, completion_tokens, cached_tokens);
                    return Ok(reply.whole(&text, finish, tokens));
                }
                Some(Event::Failed(e)) => return Err(e),
                Some(Event::Started { .. }) | None => return Err(ApiError::internal(rid)),
            }
        }
    }
    // Streaming: a task forwards worker events as SSE events. When the client goes
    // away the response body (and with it `out_rx`) is dropped, the next send fails,
    // the task ends and drops `guard`, which cancels the generation.
    let (out_tx, out_rx) = mpsc::channel::<Result<SseEvent, Infallible>>(64);
    let stall = st.cfg.stall_timeout;
    tokio::spawn(async move {
        let _guard = guard;
        reply.forward(rx, out_tx, stall, stop).await;
    });
    Ok(Sse::new(ReceiverStream::new(out_rx)).into_response())
}

/// What every part of one response repeats.
struct Reply {
    id: String,
    kind: Kind,
    created: u64,
    model: String,
    prompt_tokens: usize,
    include_usage: bool,
}

impl Reply {
    /// The whole answer of a non-streaming request.
    fn whole(&self, text: &str, finish: FinishReason, usage: Value) -> Response {
        let choice = match self.kind {
            Kind::Chat => json!({
                "index": 0,
                "message": { "role": "assistant", "content": text },
                "logprobs": null,
                "finish_reason": finish.as_str(),
            }),
            Kind::Text => json!({
                "index": 0,
                "text": text,
                "logprobs": null,
                "finish_reason": finish.as_str(),
            }),
        };
        let object = match self.kind {
            Kind::Chat => "chat.completion",
            Kind::Text => "text_completion",
        };
        Json(json!({
            "id": self.id,
            "object": object,
            "created": self.created,
            "model": self.model,
            "choices": [choice],
            "usage": usage,
        }))
        .into_response()
    }

    fn object(&self) -> &'static str {
        match self.kind {
            Kind::Chat => "chat.completion.chunk",
            Kind::Text => "text_completion",
        }
    }

    fn chunk(&self, delta: &Value, finish: Option<&str>) -> Value {
        let choice = match self.kind {
            Kind::Chat => json!({
                "index": 0, "delta": delta, "logprobs": null, "finish_reason": finish,
            }),
            Kind::Text => json!({
                "index": 0,
                "text": delta.get("content").and_then(Value::as_str).unwrap_or(""),
                "logprobs": null,
                "finish_reason": finish,
            }),
        };
        json!({
            "id": self.id, "object": self.object(), "created": self.created,
            "model": self.model, "choices": [choice],
        })
    }

    /// Forwards worker events to the client until the generation ends, the client goes
    /// away or stops reading for `stall`, or the server stops. Returning drops `rx`, so
    /// a worker stuck sending to a client that stopped reading is released.
    async fn forward(
        &self,
        mut rx: mpsc::Receiver<Event>,
        out: mpsc::Sender<Result<SseEvent, Infallible>>,
        stall: Duration,
        mut stop: watch::Receiver<bool>,
    ) {
        let send = |text: String| {
            let out = out.clone();
            async move {
                let sent = out.send(Ok(SseEvent::default().data(text)));
                matches!(tokio::time::timeout(stall, sent).await, Ok(Ok(())))
            }
        };
        if self.kind == Kind::Chat {
            let first = self.chunk(&json!({ "role": "assistant", "content": "" }), None);
            if !send(first.to_string()).await {
                return;
            }
        }
        loop {
            let event = tokio::select! {
                e = rx.recv() => e,
                () = stopped(&mut stop) => {
                    let _ = send(ApiError::shutting_down().body().to_string()).await;
                    break;
                }
            };
            match event {
                Some(Event::Text(t)) => {
                    if !send(self.chunk(&json!({ "content": t }), None).to_string()).await {
                        return;
                    }
                }
                Some(Event::Done {
                    finish,
                    completion_tokens,
                    cached_tokens,
                }) => {
                    let last = self.chunk(&json!({}), Some(finish.as_str()));
                    let _ = send(last.to_string()).await;
                    if self.include_usage {
                        let totals = json!({
                            "id": self.id, "object": self.object(), "created": self.created,
                            "model": self.model, "choices": [],
                            "usage": usage(self.prompt_tokens, completion_tokens, cached_tokens),
                        });
                        let _ = send(totals.to_string()).await;
                    }
                    break;
                }
                Some(Event::Failed(e)) => {
                    // Mid-stream: the status is already sent, so the error is an event.
                    let _ = send(e.body().to_string()).await;
                    break;
                }
                Some(Event::Started { .. }) => {}
                None => break,
            }
        }
        let _ = send("[DONE]".to_owned()).await;
    }
}

fn usage(prompt: usize, completion: usize, cached: usize) -> Value {
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": { "cached_tokens": cached },
    })
}

async fn admin_load(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    let began = Instant::now();
    let rid = st.request_id(&headers);
    let r = async {
        st.check_admin(&headers)?;
        let body = json_body(&st, body)?;
        let what = body
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError::invalid_param("model", "give the model id to load"))?
            .to_owned();
        let id = body
            .get("id")
            .and_then(Value::as_str)
            .map_or_else(|| what.clone(), str::to_owned);
        let (l, started) = st.load(&rid, &what, &id).await?;
        let status = if started {
            StatusCode::CREATED
        } else {
            StatusCode::OK
        };
        Ok((status, Json(model_card(&id, l.created))).into_response())
    }
    .await;
    st.finish("admin", &rid, began, r)
}

async fn admin_unload(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let began = Instant::now();
    let rid = st.request_id(&headers);
    let r = st.check_admin(&headers).and_then(|()| unload(&st, &id));
    st.finish("admin", &rid, began, r)
}

async fn models_unload(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    let began = Instant::now();
    let rid = st.request_id(&headers);
    let r = st.check_admin(&headers).and_then(|()| {
        let body = json_body(&st, body)?;
        let id = body
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError::invalid_param("model", "give the model id to unload"))?;
        unload(&st, id)
    });
    st.finish("admin", &rid, began, r)
}

/// Requests already running hold the model and finish; new ones get 404. The weights are
/// freed when the last of them ends.
fn unload(st: &AppState, id: &str) -> Result<Response, ApiError> {
    let draining = st.unload(id)?;
    st.log.info(
        "model_unloaded",
        &[("model", id.into()), ("draining", draining.into())],
    );
    Ok(
        Json(json!({ "id": id, "object": "model", "deleted": true, "draining": draining }))
            .into_response(),
    )
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use tokio::sync::{mpsc, watch};

    use super::{Kind, Reply};
    use crate::worker::{Event, FinishReason};

    fn reply() -> Reply {
        Reply {
            id: "cmpl-x".into(),
            kind: Kind::Text,
            created: 0,
            model: "m".into(),
            prompt_tokens: 1,
            include_usage: false,
        }
    }

    /// A client that keeps its connection open and stops reading: the forwarder gives up
    /// after the stall limit and drops the worker's channel, even with events to send.
    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_stream_nobody_reads_ends_at_the_stall_limit() {
        let (tx, rx) = mpsc::channel(64);
        // Nobody reads the far end of this, and it holds one event.
        let (out, unread) = mpsc::channel(1);
        let (_stop_tx, stop) = watch::channel(false);
        for _ in 0..8 {
            tx.send(Event::Text("x".into())).await.unwrap();
        }
        let start = tokio::time::Instant::now();
        reply()
            .forward(rx, out, Duration::from_secs(30), stop)
            .await;
        assert_eq!(start.elapsed(), Duration::from_secs(30));
        assert!(tx.is_closed(), "the worker's sends now fail");
        drop(unread);
    }

    /// A slow reader is not a stalled one.
    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_slow_reader_gets_everything() {
        let (tx, rx) = mpsc::channel(64);
        let (out, mut read) = mpsc::channel(1);
        let (_stop_tx, stop) = watch::channel(false);
        tokio::spawn(async move {
            for _ in 0..4 {
                tx.send(Event::Text("x".into())).await.unwrap();
            }
            tx.send(Event::Done {
                finish: FinishReason::Stop,
                completion_tokens: 4,
                cached_tokens: 0,
            })
            .await
            .unwrap();
        });
        let forward = tokio::spawn(async move {
            reply()
                .forward(rx, out, Duration::from_secs(30), stop)
                .await;
        });
        let mut events = 0;
        while read.recv().await.is_some() {
            events += 1;
            tokio::time::sleep(Duration::from_secs(20)).await;
        }
        forward.await.unwrap();
        // Four pieces, the finish and `[DONE]`.
        assert_eq!(events, 6);
    }

    /// Stopping the server ends a stream with an error, not a silent cut.
    #[tokio::test]
    async fn stopping_the_server_ends_a_stream_with_an_error() {
        let (_tx, rx) = mpsc::channel::<Event>(64);
        let (out, mut read) = mpsc::channel(4);
        let (stop_tx, stop) = watch::channel(false);
        let started = Instant::now();
        let forward = tokio::spawn(async move {
            reply()
                .forward(rx, out, Duration::from_secs(30), stop)
                .await;
        });
        stop_tx.send_replace(true);
        forward.await.unwrap();
        let mut seen = 0;
        while read.recv().await.is_some() {
            seen += 1;
        }
        assert_eq!(seen, 2, "the error, then [DONE]");
        assert!(started.elapsed() < Duration::from_secs(5));
    }
}
