//! `qubivex serve`: the OpenAI-compatible HTTP gateway (Phase 9, `qubivex-api`).
//!
//! Settings come from, highest first: a flag, its environment variable, the `[serve]`
//! table of the config file (`--config` or `$QUBIVEX_CONFIG`), then the built-in
//! default. Each key in the file is the flag's name with `_` for `-`; repeatable flags
//! (`--model`, `--dir`, `--cors-origin`) are lists named `models`, `dirs` and
//! `cors_origins`, and flags given on the command line replace the file's list.
//!
//! Named client keys (ADR-013 amendment 7) are only in the file, as
//! `[[serve.client_keys]]` tables (`qubivex keys new` prints one), so no key hash or
//! client list shows in `ps` or the environment.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use qubivex_api::{ClientKey, Gateway, LogFormat, MemoryPolicy, RateLimit, ServerConfig};
use qubivex_model::{ArchitectureRegistry, find};
use qubivex_runtime::Engine;
use serde::Deserialize;

use crate::models::roots;
use crate::{DeviceArgs, chain, memory_options};

/// The flags of `qubivex serve`. Every setting is optional here, so a flag that is not
/// given leaves the config file's value or the default in force.
#[derive(Clone, Debug, Default, clap::Args)]
pub(crate) struct ServeFlags {
    /// A TOML file whose [serve] table holds any of these settings, named as the flags
    /// are with _ for - (default: $QUBIVEX_CONFIG). Flags and environment variables
    /// override it.
    #[arg(long, env = "QUBIVEX_CONFIG")]
    config: Option<PathBuf>,
    /// A model id or directory to load at start (repeatable).
    #[arg(long = "model")]
    models: Vec<String>,
    /// A model directory to search for ids (repeatable); also where the admin load
    /// endpoint looks.
    #[arg(long = "dir")]
    dirs: Vec<PathBuf>,
    /// Address to listen on (default: 127.0.0.1:8080). Anything but loopback needs an
    /// API key, or --allow-unauthenticated.
    #[arg(long)]
    listen: Option<SocketAddr>,
    /// A key shared by every client, required as `Authorization: Bearer <key>` on /v1
    /// (default: $QUBIVEX_API_KEY). Named per-client keys go in the config file
    /// (`qubivex keys new`).
    #[arg(long, env = "QUBIVEX_API_KEY", hide_env_values = true)]
    api_key: Option<String>,
    /// Enable the admin endpoints with this bearer token (default:
    /// $QUBIVEX_ADMIN_TOKEN; without one they are off).
    #[arg(long, env = "QUBIVEX_ADMIN_TOKEN", hide_env_values = true)]
    admin_token: Option<String>,
    /// Serve beyond loopback with no API key: anyone who can reach the port can use it.
    #[arg(long)]
    allow_unauthenticated: bool,
    /// Requests per minute each client address may make to /v1 (default: unlimited).
    #[arg(long)]
    rate_limit: Option<u32>,
    /// Requests a client may make at once after being idle (default: --rate-limit).
    #[arg(long)]
    rate_limit_burst: Option<u32>,
    /// An origin browsers may call the API from, such as https://app.example.com, or *
    /// (repeatable; default: none, so no CORS).
    #[arg(long = "cors-origin")]
    cors_origins: Vec<String>,
    /// Largest request body in bytes (default: 4194304).
    #[arg(long)]
    max_body_bytes: Option<usize>,
    /// Longest prompt in tokens, after the chat template (default: the context).
    #[arg(long)]
    max_prompt_tokens: Option<usize>,
    /// Largest max_tokens a request may ask for (default: the context).
    #[arg(long)]
    max_generation_tokens: Option<usize>,
    /// Generations that run at once per model; more wait for a slot (default: 4).
    #[arg(long)]
    max_concurrent: Option<usize>,
    /// max_tokens when a request does not set it (default: 512).
    #[arg(long)]
    default_max_tokens: Option<usize>,
    /// Requests per model that may wait beyond those running; more get a 429 at once
    /// (default: 32).
    #[arg(long)]
    max_queue: Option<usize>,
    /// Seconds a request waits to start before a 429 (default: 30).
    #[arg(long)]
    queue_timeout: Option<u64>,
    /// When a request's memory is not free yet: queue (wait for it) or refuse (503)
    /// (default: queue).
    #[arg(long)]
    memory_policy: Option<String>,
    /// Seconds a generation may run (default: 600).
    #[arg(long)]
    request_timeout: Option<u64>,
    /// Seconds a stream may go without the client reading before it ends (default: 30).
    #[arg(long)]
    stall_timeout: Option<u64>,
    /// Seconds running requests get to finish at shutdown before they are stopped
    /// (default: 30).
    #[arg(long)]
    shutdown_grace: Option<u64>,
    /// MiB per model of finished KV to keep, so a prompt that repeats a prefix skips
    /// it. Counts against the memory plan. Requests with different `user` values or
    /// named keys never share it (default: 0, none).
    #[arg(long)]
    prefix_cache_mb: Option<u64>,
    /// A fixed expert cache in MiB per model, bypassing the memory planner.
    #[arg(long)]
    expert_cache_mb: Option<u64>,
    /// Memory profile: performance (0.88 of available RAM), balanced (0.75) or saver
    /// (0.50) (default: performance).
    #[arg(long)]
    profile: Option<String>,
    /// A custom fraction of available RAM, in (0, 1].
    #[arg(long)]
    ram_fraction: Option<f64>,
    /// An explicit RAM budget per model, for example 8GiB.
    #[arg(long)]
    ram_budget: Option<String>,
    /// Threads reading experts from disk. Default: 4.
    #[arg(long)]
    io_threads: Option<usize>,
    /// Compute backend: cpu or metal (default: cpu). See `qubivex run --help`.
    #[arg(long)]
    backend: Option<String>,
    /// With --backend metal: what runs on the GPU. See `qubivex run --help`.
    #[arg(long)]
    metal_ops: Option<String>,
    /// Context limit per conversation, in tokens (default: the model's).
    #[arg(long)]
    context: Option<usize>,
    /// Compute threads (default: the CPU's performance cores).
    #[arg(long)]
    threads: Option<usize>,
    /// How the server logs to standard error: text, json (one object per line) or off
    /// (default: text). No format logs prompts or generated text.
    #[arg(long)]
    log_format: Option<String>,
    /// Do not serve GET /metrics.
    #[arg(long)]
    no_metrics: bool,
}

/// The config file. Unknown keys are errors, so a misspelt setting is not ignored.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigFile {
    #[serde(default)]
    serve: FileServe,
}

/// One `[[serve.client_keys]]` table.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileKey {
    client: String,
    key_sha256: String,
    rate_limit: Option<u32>,
    rate_limit_burst: Option<u32>,
}

impl FileKey {
    fn key(self) -> Result<ClientKey, String> {
        let rate_limit = match (self.rate_limit, self.rate_limit_burst) {
            (Some(per_minute), burst) => Some(RateLimit {
                per_minute,
                burst: burst.unwrap_or(per_minute),
            }),
            (None, Some(_)) => {
                return Err(format!(
                    "client {:?}: rate_limit_burst needs rate_limit",
                    self.client
                ));
            }
            (None, None) => None,
        };
        ClientKey::from_hex(self.client, &self.key_sha256, rate_limit)
    }
}

/// The `[serve]` table: the flags, by the same names, plus the client keys.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileServe {
    client_keys: Option<Vec<FileKey>>,
    models: Option<Vec<String>>,
    dirs: Option<Vec<PathBuf>>,
    listen: Option<SocketAddr>,
    api_key: Option<String>,
    admin_token: Option<String>,
    allow_unauthenticated: Option<bool>,
    rate_limit: Option<u32>,
    rate_limit_burst: Option<u32>,
    cors_origins: Option<Vec<String>>,
    max_body_bytes: Option<usize>,
    max_prompt_tokens: Option<usize>,
    max_generation_tokens: Option<usize>,
    max_concurrent: Option<usize>,
    default_max_tokens: Option<usize>,
    max_queue: Option<usize>,
    queue_timeout: Option<u64>,
    memory_policy: Option<String>,
    request_timeout: Option<u64>,
    stall_timeout: Option<u64>,
    shutdown_grace: Option<u64>,
    prefix_cache_mb: Option<u64>,
    expert_cache_mb: Option<u64>,
    profile: Option<String>,
    ram_fraction: Option<f64>,
    ram_budget: Option<String>,
    io_threads: Option<usize>,
    backend: Option<String>,
    metal_ops: Option<String>,
    context: Option<usize>,
    threads: Option<usize>,
    log_format: Option<String>,
    no_metrics: Option<bool>,
}

/// Refuses a config file holding a secret that other users can read (Unix).
#[cfg(unix)]
fn check_private(path: &Path, meta: &std::fs::Metadata) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let mode = meta.permissions().mode() & 0o777;
    if mode & 0o077 != 0 {
        return Err(format!(
            "{} holds API keys or an admin token but other users can read it (mode {mode:o}); run chmod 600 on it",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn check_private(_: &Path, _: &std::fs::Metadata) -> Result<(), String> {
    Ok(())
}

fn read_config(path: &Path) -> Result<FileServe, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read config file {}: {e}", path.display()))?;
    let file: ConfigFile =
        toml::from_str(&text).map_err(|e| format!("config file {}: {e}", path.display()))?;
    if file.serve.api_key.is_some()
        || file.serve.admin_token.is_some()
        || file.serve.client_keys.is_some()
    {
        let meta = std::fs::metadata(path)
            .map_err(|e| format!("cannot read config file {}: {e}", path.display()))?;
        check_private(path, &meta)?;
    }
    Ok(file.serve)
}

fn memory_policy(s: &str) -> Result<MemoryPolicy, String> {
    match s {
        "queue" => Ok(MemoryPolicy::Queue),
        "refuse" => Ok(MemoryPolicy::Refuse),
        _ => Err(format!("memory_policy takes queue or refuse, not {s:?}")),
    }
}

fn log_format(s: &str) -> Result<LogFormat, String> {
    LogFormat::parse(s).ok_or_else(|| format!("log_format takes text, json or off, not {s:?}"))
}

/// Everything `serve` needs, with flags, environment, file and defaults merged.
#[derive(Debug)]
pub(crate) struct Plan {
    models: Vec<String>,
    addr: SocketAddr,
    cfg: ServerConfig,
}

/// A list from the command line, else the file's, else empty.
fn list<T>(flags: Vec<T>, file: Option<Vec<T>>) -> Vec<T> {
    if flags.is_empty() {
        file.unwrap_or_default()
    } else {
        flags
    }
}

impl ServeFlags {
    /// Merges these flags over `file` and the defaults, and checks the result.
    fn plan(self, file: FileServe) -> Result<Plan, String> {
        let d = ServerConfig::default();
        let secs = |flag: Option<u64>, file: Option<u64>, default: Duration| {
            flag.or(file).map_or(default, Duration::from_secs)
        };
        let per_minute = self.rate_limit.or(file.rate_limit);
        let burst = self.rate_limit_burst.or(file.rate_limit_burst);
        let rate_limit = match (per_minute, burst) {
            (Some(per_minute), burst) => Some(RateLimit {
                per_minute,
                burst: burst.unwrap_or(per_minute),
            }),
            (None, Some(_)) => return Err("rate_limit_burst needs rate_limit".into()),
            (None, None) => None,
        };
        let device = DeviceArgs {
            backend: self
                .backend
                .or(file.backend)
                .unwrap_or_else(|| "cpu".to_owned()),
            metal_ops: self.metal_ops.or(file.metal_ops),
            sim_vram: None,
            sim_bandwidth: None,
            sim_delay: false,
            vulkan_device: None,
            vulkan_vram: None,
            vulkan_timestamps: false,
            vulkan_fuse_rows: None,
            vulkan_route: None,
            vulkan_route_o: None,
            vram_budget: None,
        };
        let profile = self
            .profile
            .or(file.profile)
            .unwrap_or_else(|| "performance".to_owned());
        let ram_budget = self.ram_budget.or(file.ram_budget);
        let mut engine = memory_options(
            self.expert_cache_mb.or(file.expert_cache_mb),
            &profile,
            self.ram_fraction.or(file.ram_fraction),
            ram_budget.as_deref(),
            "lfru",
            self.io_threads.or(file.io_threads),
            false,
        )?
        .with_device(device.options()?)
        .engine(self.context.or(file.context), self.threads.or(file.threads));
        engine.prefix_cache_bytes = self
            .prefix_cache_mb
            .or(file.prefix_cache_mb)
            .unwrap_or(0)
            .saturating_mul(1 << 20);
        let dirs = roots(&list(self.dirs, file.dirs));
        let client_keys = file
            .client_keys
            .unwrap_or_default()
            .into_iter()
            .map(FileKey::key)
            .collect::<Result<Vec<_>, _>>()?;
        let cfg = ServerConfig {
            api_key: self.api_key.or(file.api_key),
            client_keys,
            admin_token: self.admin_token.or(file.admin_token),
            allow_unauthenticated_remote: self.allow_unauthenticated
                || file.allow_unauthenticated.unwrap_or(false),
            rate_limit,
            cors_origins: list(self.cors_origins, file.cors_origins),
            max_body_bytes: self
                .max_body_bytes
                .or(file.max_body_bytes)
                .unwrap_or(d.max_body_bytes),
            max_prompt_tokens: self.max_prompt_tokens.or(file.max_prompt_tokens),
            max_generation_tokens: self.max_generation_tokens.or(file.max_generation_tokens),
            max_concurrent: self
                .max_concurrent
                .or(file.max_concurrent)
                .unwrap_or(d.max_concurrent),
            max_queue: self.max_queue.or(file.max_queue).unwrap_or(d.max_queue),
            queue_timeout: secs(self.queue_timeout, file.queue_timeout, d.queue_timeout),
            memory_policy: self
                .memory_policy
                .or(file.memory_policy)
                .as_deref()
                .map_or(Ok(d.memory_policy), memory_policy)?,
            request_timeout: secs(
                self.request_timeout,
                file.request_timeout,
                d.request_timeout,
            ),
            stall_timeout: secs(self.stall_timeout, file.stall_timeout, d.stall_timeout),
            shutdown_grace: secs(self.shutdown_grace, file.shutdown_grace, d.shutdown_grace),
            default_max_tokens: self
                .default_max_tokens
                .or(file.default_max_tokens)
                .unwrap_or(d.default_max_tokens),
            model_dirs: dirs,
            engine,
            log_format: self
                .log_format
                .or(file.log_format)
                .as_deref()
                .map_or(Ok(d.log_format), log_format)?,
            metrics_enabled: !(self.no_metrics || file.no_metrics.unwrap_or(false)),
            ..d
        };
        cfg.validate()?;
        let addr = self
            .listen
            .or(file.listen)
            .unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], 8080)));
        cfg.check_listen(addr).map_err(|e| {
            format!("{e} (--api-key, client_keys in --config, --listen or --allow-unauthenticated)")
        })?;
        let models = list(self.models, file.models);
        if cfg.admin_token.is_none() && models.is_empty() {
            return Err("no model to serve and no admin token to load one later".into());
        }
        Ok(Plan { models, addr, cfg })
    }
}

/// Reads the config file, if any, and merges `flags` over it.
pub(crate) fn plan(mut flags: ServeFlags) -> Result<Plan, String> {
    let file = match flags.config.take() {
        Some(path) => read_config(&path)?,
        None => FileServe::default(),
    };
    flags.plan(file)
}

pub(crate) fn serve(p: Plan) -> Result<(), String> {
    let Plan { models, addr, cfg } = p;
    let dirs = cfg.model_dirs.clone();
    let opts = cfg.engine;
    let open = !cfg.needs_key() && !addr.ip().is_loopback();
    // The human-readable lines below belong to the text log; in the other formats the
    // gateway's own events say the same (`model_added`, `server_started`, ...).
    let text = cfg.log_format == LogFormat::Text;
    let gw = Gateway::new(cfg);
    let reg = ArchitectureRegistry::with_builtins();
    for m in &models {
        let meta = find(m, &dirs, &reg).map_err(|e| chain(&e))?;
        let id = meta.manifest.model.id.clone();
        let engine = Engine::load(meta, opts).map_err(|e| chain(&e))?;
        if text {
            eprintln!(
                "qubivex: loaded {id} ({}; context {})",
                engine.metadata().manifest.architecture.id,
                engine.max_context()
            );
        }
        gw.add_model(&id, engine).map_err(|e| e.message)?;
    }
    if open && text {
        eprintln!(
            "qubivex: warning: serving on {addr} without an API key, as allowed; anyone who can reach it can use it"
        );
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("qubivex-http")
        .enable_all()
        .build()
        .map_err(|e| format!("cannot start the HTTP runtime: {e}"))?;
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| format!("cannot listen on {addr}: {e}"))?;
        let local = listener.local_addr().map_err(|e| e.to_string())?;
        if text {
            eprintln!(
                "qubivex: serving {:?} on http://{local} (OpenAI API under /v1)",
                gw.model_ids()
            );
        }
        gw.serve(listener, async move {
            stop_signal().await;
            if text {
                eprintln!("qubivex: shutting down; letting open requests finish");
            }
        })
        .await
        .map_err(|e| format!("server error: {e}"))
    })
}

/// Resolves when the process is asked to stop: Ctrl-C (SIGINT) or, on Unix, SIGTERM,
/// which is what `docker stop`, systemd and Kubernetes send. Both start the same drain.
async fn stop_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        // If the handler cannot be installed, Ctrl-C still works.
        if let Ok(mut term) = signal(SignalKind::terminate()) {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
            return;
        }
    }
    let _ = tokio::signal::ctrl_c().await;
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::time::Duration;

    use clap::Parser;
    use qubivex_api::{LogFormat, MemoryPolicy, RateLimit};

    use super::{ConfigFile, FileServe, ServeFlags};

    #[derive(Parser)]
    struct T {
        #[command(flatten)]
        f: ServeFlags,
    }

    fn flags(args: &[&str]) -> ServeFlags {
        T::try_parse_from(std::iter::once("serve").chain(args.iter().copied()))
            .unwrap()
            .f
    }

    fn file(text: &str) -> FileServe {
        toml::from_str::<ConfigFile>(text).unwrap().serve
    }

    #[test]
    fn flags_beat_the_file_and_the_file_beats_the_defaults() {
        let f = file(
            r#"
            [serve]
            models = ["from-file"]
            listen = "127.0.0.1:9000"
            max_concurrent = 8
            max_queue = 5
            queue_timeout = 7
            memory_policy = "refuse"
            cors_origins = ["https://a.example"]
            rate_limit = 120
            max_prompt_tokens = 1000
            "#,
        );
        let p = flags(&[
            "--model",
            "from-flag",
            "--max-concurrent",
            "2",
            "--cors-origin",
            "https://b.example",
        ])
        .plan(f)
        .unwrap();
        assert_eq!(p.models, ["from-flag"]);
        assert_eq!(p.addr.to_string(), "127.0.0.1:9000");
        assert_eq!(p.cfg.max_concurrent, 2, "the flag");
        assert_eq!(p.cfg.max_queue, 5, "the file");
        assert_eq!(p.cfg.queue_timeout, Duration::from_secs(7));
        assert_eq!(p.cfg.memory_policy, MemoryPolicy::Refuse);
        assert_eq!(
            p.cfg.cors_origins,
            ["https://b.example"],
            "flags replace the list"
        );
        assert_eq!(
            p.cfg.rate_limit,
            Some(RateLimit {
                per_minute: 120,
                burst: 120
            })
        );
        assert_eq!(p.cfg.max_prompt_tokens, Some(1000));
        assert_eq!(p.cfg.stall_timeout, Duration::from_secs(30), "the default");
        assert_eq!(p.cfg.default_max_tokens, 512, "the default");
        assert_eq!(p.cfg.log_format, LogFormat::Text, "the default");
        assert!(p.cfg.metrics_enabled, "the default");
    }

    #[test]
    fn the_log_format_and_metrics_follow_the_same_precedence() {
        let f = || file("[serve]\nlog_format = \"json\"\nno_metrics = true\n");
        let p = flags(&["--model", "m"]).plan(f()).unwrap();
        assert_eq!(p.cfg.log_format, LogFormat::Json, "the file");
        assert!(!p.cfg.metrics_enabled, "the file");
        let p = flags(&["--model", "m", "--log-format", "off"])
            .plan(f())
            .unwrap();
        assert_eq!(p.cfg.log_format, LogFormat::Off, "the flag");
        let p = flags(&["--model", "m", "--no-metrics"])
            .plan(FileServe::default())
            .unwrap();
        assert!(!p.cfg.metrics_enabled);
        assert_eq!(
            p.cfg.prefill_chunk, 256,
            "untouched settings keep their defaults"
        );
    }

    #[test]
    fn no_file_means_the_defaults() {
        let p = flags(&["--model", "m"]).plan(FileServe::default()).unwrap();
        assert_eq!(p.addr.to_string(), "127.0.0.1:8080");
        assert_eq!(p.cfg.max_body_bytes, 4 << 20);
        assert_eq!(p.cfg.rate_limit, None);
        assert!(p.cfg.cors_origins.is_empty());
        assert!(!p.cfg.allow_unauthenticated_remote);
    }

    #[test]
    fn unknown_keys_are_errors() {
        let e = toml::from_str::<ConfigFile>("[serve]\nmax_concurent = 3\n").unwrap_err();
        assert!(e.to_string().contains("max_concurent"), "{e}");
        assert!(toml::from_str::<ConfigFile>("[srve]\n").is_err());
    }

    #[test]
    fn a_remote_listener_without_a_key_is_refused_before_any_model_loads() {
        let e = flags(&["--model", "m", "--listen", "0.0.0.0:8080"])
            .plan(FileServe::default())
            .unwrap_err();
        assert!(e.contains("without an API key"), "{e}");
        let ok = flags(&["--model", "m", "--listen", "0.0.0.0:8080", "--api-key", "k"])
            .plan(FileServe::default());
        assert!(ok.is_ok());
        let allowed = file("[serve]\nallow_unauthenticated = true\nlisten = \"0.0.0.0:8080\"\n");
        assert!(flags(&["--model", "m"]).plan(allowed).is_ok());
    }

    #[test]
    fn client_keys_come_from_the_file_and_count_as_keys() {
        let h = qubivex_api::hash_key("secret");
        let f = file(&format!(
            "[serve]\nlisten = \"0.0.0.0:8080\"\n\
             [[serve.client_keys]]\nclient = \"a\"\nkey_sha256 = \"{h}\"\nrate_limit = 10\n\
             [[serve.client_keys]]\nclient = \"b\"\nkey_sha256 = \"{}\"\n",
            qubivex_api::hash_key("other")
        ));
        let p = flags(&["--model", "m"]).plan(f).unwrap();
        assert_eq!(p.cfg.client_keys.len(), 2);
        assert_eq!(
            p.cfg.client_keys[0],
            qubivex_api::ClientKey::new(
                "a",
                "secret",
                Some(RateLimit {
                    per_minute: 10,
                    burst: 10
                })
            )
        );
        assert_eq!(p.cfg.client_keys[1].rate_limit, None);
        assert!(
            p.cfg.needs_key(),
            "a remote listener with named keys only is allowed"
        );

        let bad = |t: &str| {
            flags(&["--model", "m"])
                .plan(file(&format!("[serve]\n{t}")))
                .unwrap_err()
        };
        let e = bad("[[serve.client_keys]]\nclient = \"a\"\nkey_sha256 = \"nothex\"\n");
        assert!(e.contains("64 hex"), "{e}");
        let e = bad(&format!(
            "[[serve.client_keys]]\nclient = \"a b\"\nkey_sha256 = \"{h}\"\n"
        ));
        assert!(e.contains("client name"), "{e}");
        let e = bad(&format!(
            "[[serve.client_keys]]\nclient = \"a\"\nkey_sha256 = \"{h}\"\nrate_limit_burst = 2\n"
        ));
        assert!(e.contains("rate_limit_burst"), "{e}");
        assert!(
            toml::from_str::<ConfigFile>(&format!(
                "[[serve.client_keys]]\nclient = \"a\"\nkey = \"{h}\"\n"
            ))
            .is_err(),
            "a plain key is an unknown field"
        );
    }

    #[test]
    fn bad_settings_are_refused_with_their_name() {
        let e = flags(&["--model", "m", "--cors-origin", "app.example"])
            .plan(FileServe::default())
            .unwrap_err();
        assert!(e.contains("app.example"), "{e}");
        let e = flags(&["--model", "m", "--rate-limit-burst", "5"])
            .plan(FileServe::default())
            .unwrap_err();
        assert!(e.contains("rate_limit"), "{e}");
        let e = flags(&["--model", "m", "--memory-policy", "maybe"])
            .plan(FileServe::default())
            .unwrap_err();
        assert!(e.contains("memory_policy"), "{e}");
        let e = flags(&["--model", "m", "--log-format", "xml"])
            .plan(FileServe::default())
            .unwrap_err();
        assert!(e.contains("log_format"), "{e}");
        let e = flags(&[]).plan(FileServe::default()).unwrap_err();
        assert!(e.contains("no model"), "{e}");
    }
}
