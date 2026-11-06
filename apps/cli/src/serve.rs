//! `qubivex serve`: the OpenAI-compatible HTTP gateway (Phase 9, `qubivex-api`).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use qubivex_api::{Gateway, MemoryPolicy, ServerConfig};
use qubivex_model::{ArchitectureRegistry, find};
use qubivex_runtime::{Engine, EngineOptions};

use crate::chain;
use crate::models::roots;

pub(crate) struct ServeArgs<'a> {
    pub(crate) models: &'a [String],
    pub(crate) dirs: &'a [PathBuf],
    pub(crate) addr: SocketAddr,
    pub(crate) api_key: Option<String>,
    pub(crate) admin_token: Option<String>,
    pub(crate) max_concurrent: usize,
    pub(crate) default_max_tokens: usize,
    pub(crate) max_queue: usize,
    pub(crate) queue_timeout: Duration,
    pub(crate) memory_policy: MemoryPolicy,
    pub(crate) request_timeout: Duration,
    pub(crate) stall_timeout: Duration,
    pub(crate) shutdown_grace: Duration,
    pub(crate) engine: EngineOptions,
}

pub(crate) fn memory_policy(s: &str) -> Result<MemoryPolicy, String> {
    match s {
        "queue" => Ok(MemoryPolicy::Queue),
        "refuse" => Ok(MemoryPolicy::Refuse),
        _ => Err(format!("--memory-policy takes queue or refuse, not {s:?}")),
    }
}

pub(crate) fn serve(a: &ServeArgs<'_>) -> Result<(), String> {
    let dirs = roots(a.dirs);
    let gw = Gateway::new(ServerConfig {
        api_key: a.api_key.clone(),
        admin_token: a.admin_token.clone(),
        max_concurrent: a.max_concurrent,
        max_queue: a.max_queue,
        queue_timeout: a.queue_timeout,
        memory_policy: a.memory_policy,
        request_timeout: a.request_timeout,
        stall_timeout: a.stall_timeout,
        shutdown_grace: a.shutdown_grace,
        default_max_tokens: a.default_max_tokens,
        model_dirs: dirs.clone(),
        engine: a.engine,
        ..ServerConfig::default()
    });
    let reg = ArchitectureRegistry::with_builtins();
    for m in a.models {
        let meta = find(m, &dirs, &reg).map_err(|e| chain(&e))?;
        let id = meta.manifest.model.id.clone();
        let engine = Engine::load(meta, a.engine).map_err(|e| chain(&e))?;
        eprintln!(
            "qubivex: loaded {id} ({}; context {})",
            engine.metadata().manifest.architecture.id,
            engine.max_context()
        );
        gw.add_model(&id, engine).map_err(|e| e.message)?;
    }
    if a.admin_token.is_none() && a.models.is_empty() {
        return Err("no --model to serve and no --admin-token to load one later".into());
    }
    if a.api_key.is_none() && !a.addr.ip().is_loopback() {
        eprintln!(
            "qubivex: warning: serving on {} without --api-key; anyone who can reach it can use it",
            a.addr
        );
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("qubivex-http")
        .enable_all()
        .build()
        .map_err(|e| format!("cannot start the HTTP runtime: {e}"))?;
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(a.addr)
            .await
            .map_err(|e| format!("cannot listen on {}: {e}", a.addr))?;
        let local = listener.local_addr().map_err(|e| e.to_string())?;
        eprintln!(
            "qubivex: serving {:?} on http://{local} (OpenAI API under /v1)",
            gw.model_ids()
        );
        gw.serve(listener, async {
            let _ = tokio::signal::ctrl_c().await;
            eprintln!("qubivex: shutting down; letting open requests finish");
        })
        .await
        .map_err(|e| format!("server error: {e}"))
    })
}
