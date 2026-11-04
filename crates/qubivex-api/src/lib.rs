//! The QUBIVEX HTTP gateway (Phase 9): OpenAI-compatible endpoints over the runtime.
//!
//! - `GET /health`, `GET /metrics` (Prometheus text), `GET /v1/system`
//! - `GET /v1/models`, `GET /v1/models/{id}`
//! - `POST /v1/chat/completions` and `POST /v1/completions`, streaming (SSE) or not
//! - `POST /v1/admin/models` (load) and `DELETE /v1/admin/models/{id}` (unload), behind
//!   the admin token
//!
//! Tokio and Axum stay at the edge: every generation runs on its own dedicated thread
//! (`worker`) and reaches the request over a bounded channel (QUBIVEX_DESIGN §4.7,
//! RUST_GUIDELINES §4). Each conversation gets its own KV cache, and is admitted against
//! the memory plan together with every other live one ([`Engine::reserve_kv`]). A
//! request whose client goes away is cancelled: generation stops within one prefill
//! chunk or one token. Every response carries an `x-request-id`; errors are
//! OpenAI-shaped and never contain file paths.

mod error;
mod metrics;
mod openai;
mod worker;

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::future::Future;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::rejection::JsonRejection;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event as SseEvent, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
pub use error::ApiError;
pub use metrics::Metrics;
use qubivex_model::{ArchitectureRegistry, find};
use qubivex_runtime::{Engine, EngineOptions};
use serde_json::{Value, json};
use tokio::sync::{Semaphore, mpsc};
use tokio_stream::wrappers::ReceiverStream;

use crate::openai::{ChatRequest, CompletionRequest, Params};
use crate::worker::{CancelOnDrop, Event, Input, Job};

/// Server settings.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// Required as `Authorization: Bearer <key>` on `/v1/*` when set.
    pub api_key: Option<String>,
    /// Required on the admin endpoints, which are disabled without it.
    pub admin_token: Option<String>,
    /// Generations that run at once per model; more wait for a slot.
    pub max_concurrent: usize,
    /// How long a request waits for a generation slot before a 429.
    pub queue_timeout: Duration,
    /// Longest a generation runs; it then ends as if it reached `max_tokens`.
    pub request_timeout: Duration,
    /// `max_tokens` when a request does not say (capped by the context).
    pub default_max_tokens: usize,
    /// Prompt tokens per forward pass (how soon a cancelled prefill stops).
    pub prefill_chunk: usize,
    /// Directories the admin load endpoint searches for model ids.
    pub model_dirs: Vec<PathBuf>,
    /// How the admin load endpoint loads models.
    pub engine: EngineOptions,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            admin_token: None,
            max_concurrent: 4,
            queue_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(600),
            default_max_tokens: 512,
            prefill_chunk: 256,
            model_dirs: Vec::new(),
            engine: EngineOptions::default(),
        }
    }
}

/// A loaded model.
struct Loaded {
    engine: Arc<Engine>,
    slots: Arc<Semaphore>,
    created: u64,
}

struct AppState {
    cfg: ServerConfig,
    /// Never held across an `.await`.
    models: RwLock<BTreeMap<String, Arc<Loaded>>>,
    metrics: Arc<Metrics>,
    next_id: AtomicU64,
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

fn bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
}

impl Gateway {
    /// A gateway with no models.
    #[must_use]
    pub fn new(cfg: ServerConfig) -> Self {
        Self {
            state: Arc::new(AppState {
                cfg,
                models: RwLock::new(BTreeMap::new()),
                metrics: Arc::new(Metrics::default()),
                next_id: AtomicU64::new(1),
            }),
        }
    }

    /// Serves `engine` under `id`.
    ///
    /// # Errors
    /// [`ApiError`] (409) if `id` is already served.
    pub fn add_model(&self, id: &str, engine: Engine) -> Result<(), ApiError> {
        let mut m = self
            .state
            .models
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if m.contains_key(id) {
            return Err(ApiError::conflict(format!(
                "the model {id:?} is already loaded"
            )));
        }
        m.insert(
            id.to_owned(),
            Arc::new(Loaded {
                engine: Arc::new(engine),
                slots: Arc::new(Semaphore::new(self.state.cfg.max_concurrent.max(1))),
                created: now_secs(),
            }),
        );
        Ok(())
    }

    /// The ids served.
    #[must_use]
    pub fn model_ids(&self) -> Vec<String> {
        self.state
            .models
            .read()
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// The gateway's counters.
    #[must_use]
    pub fn metrics(&self) -> Arc<Metrics> {
        Arc::clone(&self.state.metrics)
    }

    /// The Axum router.
    pub fn router(&self) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/metrics", get(metrics_text))
            .route("/v1/system", get(system))
            .route("/v1/models", get(list_models))
            .route("/v1/models/{id}", get(get_model))
            .route("/v1/chat/completions", post(chat))
            .route("/v1/completions", post(completions))
            .route("/v1/admin/models", post(admin_load))
            .route("/v1/admin/models/{id}", delete(admin_unload))
            .fallback(|| async { ApiError::not_found() })
            .with_state(Arc::clone(&self.state))
    }

    /// Serves on `listener` until `shutdown` completes, then lets open requests finish.
    ///
    /// # Errors
    /// I/O errors from the listener.
    pub async fn serve(
        self,
        listener: tokio::net::TcpListener,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> std::io::Result<()> {
        axum::serve(listener, self.router())
            .with_graceful_shutdown(shutdown)
            .await
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

    fn check_key(&self, headers: &HeaderMap) -> Result<(), ApiError> {
        match &self.cfg.api_key {
            Some(k) if !bearer(headers).is_some_and(|b| same_secret(b, k)) => {
                Err(ApiError::unauthorized())
            }
            _ => Ok(()),
        }
    }

    fn check_admin(&self, headers: &HeaderMap) -> Result<(), ApiError> {
        match &self.cfg.admin_token {
            None => Err(ApiError::admin_disabled()),
            Some(t) if bearer(headers).is_some_and(|b| same_secret(b, t)) => Ok(()),
            Some(_) => Err(ApiError::unauthorized()),
        }
    }

    fn model(&self, id: &str) -> Result<Arc<Loaded>, ApiError> {
        self.models
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(id)
            .cloned()
            .ok_or_else(|| ApiError::model_not_found(id))
    }

    /// Adds the request id header and counts the request.
    fn finish(&self, endpoint: &'static str, rid: &str, r: Result<Response, ApiError>) -> Response {
        let mut resp = r.unwrap_or_else(IntoResponse::into_response);
        self.metrics.request(endpoint, resp.status().as_u16());
        if let Ok(v) = HeaderValue::from_str(rid) {
            resp.headers_mut().insert("x-request-id", v);
        }
        resp
    }
}

async fn health(State(st): State<Arc<AppState>>) -> Json<Value> {
    let n = st.models.read().map_or(0, |m| m.len());
    Json(json!({ "status": "ok", "models": n }))
}

async fn metrics_text(State(st): State<Arc<AppState>>) -> Response {
    let mut out = String::new();
    st.metrics.render(&mut out);
    let models: Vec<(String, Arc<Loaded>)> = st
        .models
        .read()
        .map(|m| m.iter().map(|(k, v)| (k.clone(), Arc::clone(v))).collect())
        .unwrap_or_default();
    for (name, help) in [
        ("qubivex_kv_reserved_positions", "gauge"),
        ("qubivex_expert_cache_bytes", "gauge"),
        ("qubivex_generation_slots_free", "gauge"),
    ] {
        out.push_str(&format!("# TYPE {name} {help}\n"));
        for (id, l) in &models {
            let v = match name {
                "qubivex_kv_reserved_positions" => l.engine.kv_reserved() as u64,
                "qubivex_expert_cache_bytes" => l.engine.expert_cache_bytes(),
                _ => l.slots.available_permits() as u64,
            };
            out.push_str(&format!("{name}{{model=\"{id}\"}} {v}\n"));
        }
    }
    (
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        out,
    )
        .into_response()
}

async fn system(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let rid = st.request_id(&headers);
    let r = async {
        st.check_key(&headers)?;
        let hardware = tokio::task::spawn_blocking(|| {
            serde_json::to_value(qubivex_hardware::probe(&qubivex_hardware::Host::live()))
                .unwrap_or(Value::Null)
        })
        .await
        .unwrap_or(Value::Null);
        let models: Vec<(String, Arc<Loaded>)> = st
            .models
            .read()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), Arc::clone(v))).collect())
            .unwrap_or_default();
        let models: Vec<Value> = models
            .iter()
            .map(|(id, l)| {
                let e = &l.engine;
                let ledger = e.ledger().snapshot(qubivex_memory::Tier::Ram);
                json!({
                    "id": id,
                    "architecture": e.metadata().manifest.architecture.id,
                    "max_context": e.max_context(),
                    "generation_slots": st.cfg.max_concurrent,
                    "generation_slots_free": l.slots.available_permits(),
                    "kv_reserved_positions": e.kv_reserved(),
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
            "generations_active": st.metrics.active(),
        }))
        .into_response())
    }
    .await;
    st.finish("system", &rid, r)
}

fn model_card(id: &str, created: u64) -> Value {
    json!({ "id": id, "object": "model", "created": created, "owned_by": "qubivex" })
}

async fn list_models(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let rid = st.request_id(&headers);
    let r = st.check_key(&headers).map(|()| {
        let data: Vec<Value> = st
            .models
            .read()
            .map(|m| m.iter().map(|(id, l)| model_card(id, l.created)).collect())
            .unwrap_or_default();
        Json(json!({ "object": "list", "data": data })).into_response()
    });
    st.finish("models", &rid, r)
}

async fn get_model(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let rid = st.request_id(&headers);
    let r = st
        .check_key(&headers)
        .and_then(|()| st.model(&id))
        .map(|l| Json(model_card(&id, l.created)).into_response());
    st.finish("models", &rid, r)
}

fn json_body<T>(body: Result<Json<T>, JsonRejection>) -> Result<T, ApiError> {
    body.map(|Json(b)| b).map_err(|e| {
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
    let rid = st.request_id(&headers);
    let r = async {
        st.check_key(&headers)?;
        let req = json_body(body)?;
        let messages = req.messages()?;
        let max = req.max_completion_tokens.or(req.common.max_tokens);
        let params = req.common.params(max)?;
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
    st.finish("chat_completions", &rid, r)
}

async fn completions(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let rid = st.request_id(&headers);
    let r = async {
        st.check_key(&headers)?;
        let req = json_body(body)?;
        let prompt = req.prompt()?;
        let max = req.common.max_tokens;
        let params = req.common.params(max)?;
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
    st.finish("completions", &rid, r)
}

/// Starts a generation and answers with its result or its stream.
async fn generate(
    st: &Arc<AppState>,
    rid: &str,
    model: &str,
    input: Input,
    params: Params,
    kind: Kind,
) -> Result<Response, ApiError> {
    let loaded = st.model(model)?;
    let permit = tokio::time::timeout(
        st.cfg.queue_timeout,
        Arc::clone(&loaded.slots).acquire_owned(),
    )
    .await
    .map_err(|_| ApiError::busy(st.cfg.max_concurrent))?
    .map_err(|_| ApiError::overloaded("the model is being unloaded"))?;
    let cancel = Arc::new(AtomicBool::new(false));
    // Dropped (cancelling the generation) when the request is abandoned: the handler
    // future for a non-streaming request, the forwarding task for a stream.
    let guard = CancelOnDrop(Arc::clone(&cancel));
    let (tx, mut rx) = mpsc::channel(64);
    let stream = params.stream;
    let include_usage = params.include_usage;
    worker::spawn(
        Job {
            engine: Arc::clone(&loaded.engine),
            input,
            params,
            default_max_tokens: st.cfg.default_max_tokens,
            prefill_chunk: st.cfg.prefill_chunk,
            deadline: Instant::now() + st.cfg.request_timeout,
            cancel,
            request_id: rid.to_owned(),
            metrics: Arc::clone(&st.metrics),
        },
        tx,
        permit,
    );
    // The first event decides the status: admission and prompt errors are HTTP errors
    // even for a stream.
    let prompt_tokens = match rx.recv().await {
        Some(Event::Started { prompt_tokens }) => prompt_tokens,
        Some(Event::Failed(e)) => return Err(e),
        _ => return Err(ApiError::internal(rid)),
    };
    let id = match kind {
        Kind::Chat => format!("chatcmpl-{rid}"),
        Kind::Text => format!("cmpl-{rid}"),
    };
    let created = now_secs();
    let model = model.to_owned();
    if !stream {
        let mut text = String::new();
        loop {
            match rx.recv().await {
                Some(Event::Text(t)) => text.push_str(&t),
                Some(Event::Done {
                    finish,
                    completion_tokens,
                }) => {
                    drop(guard);
                    let usage = usage(prompt_tokens, completion_tokens);
                    let choice = match kind {
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
                    let object = match kind {
                        Kind::Chat => "chat.completion",
                        Kind::Text => "text_completion",
                    };
                    return Ok(Json(json!({
                        "id": id,
                        "object": object,
                        "created": created,
                        "model": model,
                        "choices": [choice],
                        "usage": usage,
                    }))
                    .into_response());
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
    tokio::spawn(async move {
        let _guard = guard;
        let object = match kind {
            Kind::Chat => "chat.completion.chunk",
            Kind::Text => "text_completion",
        };
        let chunk = |delta: Value, finish: Option<&str>| {
            let choice = match kind {
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
                "id": id, "object": object, "created": created, "model": model,
                "choices": [choice],
            })
        };
        let send = |v: Value| {
            let out_tx = out_tx.clone();
            async move {
                out_tx
                    .send(Ok(SseEvent::default().data(v.to_string())))
                    .await
                    .is_ok()
            }
        };
        if kind == Kind::Chat
            && !send(chunk(json!({ "role": "assistant", "content": "" }), None)).await
        {
            return;
        }
        while let Some(ev) = rx.recv().await {
            match ev {
                Event::Text(t) => {
                    if !send(chunk(json!({ "content": t }), None)).await {
                        return;
                    }
                }
                Event::Done {
                    finish,
                    completion_tokens,
                } => {
                    let _ = send(chunk(json!({}), Some(finish.as_str()))).await;
                    if include_usage {
                        let _ = send(json!({
                            "id": id, "object": object, "created": created, "model": model,
                            "choices": [], "usage": usage(prompt_tokens, completion_tokens),
                        }))
                        .await;
                    }
                    break;
                }
                Event::Failed(e) => {
                    // Mid-stream: the status is already sent, so the error is an event.
                    let _ = send(e.body()).await;
                    break;
                }
                Event::Started { .. } => {}
            }
        }
        let _ = out_tx.send(Ok(SseEvent::default().data("[DONE]"))).await;
    });
    Ok(Sse::new(ReceiverStream::new(out_rx)).into_response())
}

fn usage(prompt: usize, completion: usize) -> Value {
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    })
}

async fn admin_load(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    let rid = st.request_id(&headers);
    let r = async {
        st.check_admin(&headers)?;
        let body = json_body(body)?;
        let what = body
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError::invalid_param("model", "give the model id to load"))?
            .to_owned();
        let id = body
            .get("id")
            .and_then(Value::as_str)
            .map_or_else(|| what.clone(), str::to_owned);
        if st.model(&id).is_ok() {
            return Err(ApiError::conflict(format!(
                "the model {id:?} is already loaded"
            )));
        }
        let (dirs, opts, rid2) = (st.cfg.model_dirs.clone(), st.cfg.engine, rid.clone());
        let what2 = what.clone();
        // Loading reads weights for seconds: off the Tokio workers.
        let engine = tokio::task::spawn_blocking(move || {
            let meta =
                find(&what2, &dirs, &ArchitectureRegistry::with_builtins()).map_err(|e| {
                    eprintln!("qubivex-api: request {rid2}: cannot find {what2:?}: {e}");
                    ApiError::model_not_found(&what2)
                })?;
            Engine::load(meta, opts).map_err(|e| ApiError::from_runtime(&e, &rid2))
        })
        .await
        .map_err(|_| ApiError::internal(&rid))??;
        let gw = Gateway {
            state: Arc::clone(&st),
        };
        gw.add_model(&id, engine)?;
        Ok((StatusCode::CREATED, Json(model_card(&id, now_secs()))).into_response())
    }
    .await;
    st.finish("admin", &rid, r)
}

async fn admin_unload(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let rid = st.request_id(&headers);
    let r = (|| {
        st.check_admin(&headers)?;
        let removed = st
            .models
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&id)
            .ok_or_else(|| ApiError::model_not_found(&id))?;
        // Requests already running hold the model and finish; new ones get 404. The
        // weights are freed when the last of them ends.
        removed.slots.close();
        let draining = st
            .cfg
            .max_concurrent
            .saturating_sub(removed.slots.available_permits());
        Ok(
            Json(json!({ "id": id, "object": "model", "deleted": true, "draining": draining }))
                .into_response(),
        )
    })();
    st.finish("admin", &rid, r)
}
