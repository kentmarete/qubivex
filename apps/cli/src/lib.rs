//! The `qubivex` command-line interface.
//!
//! Subcommands arrive with the phases that implement them (MIGRATION_PLAN §3):
//! `system` and `doctor` (Phase 2), `models` (Phase 3), `run` and `bench` (Phase 4),
//! `plan` (Phase 5); `serve` later.

mod bench;
mod doctor;
mod models;
mod plan;
mod run;
mod serve;
mod system;

use std::ffi::OsString;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use qubivex_hardware::{Host, ResourceProbe, SystemProbe};

/// Run Mixture-of-Experts models with expert streaming.
#[derive(Debug, Parser)]
#[command(name = "qubivex", version = qubivex_core::VERSION, about, long_about = None)]
#[command(arg_required_else_help = true)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Show the hardware profile and the resources available now.
    System {
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Measure things `system` cannot see by probing.
    Doctor {
        /// Measure read throughput and 4 MiB latency of the filesystem holding --path.
        /// Writes and removes a scratch file there.
        #[arg(long)]
        storage: bool,
        /// Directory on the filesystem to measure.
        #[arg(long, default_value = ".")]
        path: PathBuf,
        /// Scratch file size in MiB (at least 64).
        #[arg(long, default_value_t = 1024)]
        size_mib: u64,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Generate text with a model (CPU).
    Run {
        /// A model id, or a model directory.
        #[arg(long)]
        model: String,
        /// A model directory to search for the id (repeatable).
        #[arg(long = "dir")]
        dirs: Vec<PathBuf>,
        /// The prompt.
        #[arg(long)]
        prompt: String,
        /// Wrap the prompt as a user message with the model's chat template.
        #[arg(long)]
        chat: bool,
        /// Most tokens to generate.
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// Sampling temperature; 0 is greedy.
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Nucleus sampling mass.
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        /// Random seed for sampling.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// A fixed expert cache in MiB, bypassing the memory planner (default: planned
        /// from available RAM, shrinking as the KV grows).
        #[arg(long)]
        expert_cache_mb: Option<u64>,
        /// Memory profile: performance (0.88 of available RAM), balanced (0.75) or saver (0.50).
        #[arg(long, default_value = "performance")]
        profile: String,
        /// A custom fraction of available RAM, in (0, 1].
        #[arg(long)]
        ram_fraction: Option<f64>,
        /// An explicit RAM budget, for example 8GiB; replaces the profile's.
        #[arg(long)]
        ram_budget: Option<String>,
        /// How a full per-layer expert pool picks its victim: lfru (the coldest, by
        /// decaying heat) or lru.
        #[arg(long, default_value = "lfru")]
        cache_policy: String,
        /// Threads reading experts from disk (0: the compute thread reads its own
        /// misses). Default: 4.
        #[arg(long)]
        io_threads: Option<usize>,
        /// Read experts bypassing the OS page cache (F_NOCACHE on macOS, O_DIRECT on
        /// Linux), so streaming does not fill the file cache.
        #[arg(long)]
        direct_io: bool,
        #[command(flatten)]
        prefetch: PrefetchArgs,
        #[command(flatten)]
        storage: StorageArgs,
        #[command(flatten)]
        device: DeviceArgs,
        /// Context limit in tokens (default: the model's).
        #[arg(long)]
        context: Option<usize>,
        /// Compute threads (default: the CPU's performance cores).
        #[arg(long)]
        threads: Option<usize>,
        /// Do not print timing and cache statistics.
        #[arg(long)]
        quiet: bool,
    },
    /// Serve models over an OpenAI-compatible HTTP API (/v1/chat/completions,
    /// /v1/completions, /v1/models, /health, /metrics).
    Serve {
        /// A model id or directory to load at start (repeatable).
        #[arg(long = "model")]
        models: Vec<String>,
        /// A model directory to search for ids (repeatable); also where the admin load
        /// endpoint looks.
        #[arg(long = "dir")]
        dirs: Vec<PathBuf>,
        /// Address to listen on.
        #[arg(long, default_value = "127.0.0.1:8080")]
        listen: std::net::SocketAddr,
        /// Require `Authorization: Bearer <key>` on /v1 (default: $QUBIVEX_API_KEY).
        #[arg(long, env = "QUBIVEX_API_KEY", hide_env_values = true)]
        api_key: Option<String>,
        /// Enable the admin endpoints with this bearer token (default:
        /// $QUBIVEX_ADMIN_TOKEN; without one they are off).
        #[arg(long, env = "QUBIVEX_ADMIN_TOKEN", hide_env_values = true)]
        admin_token: Option<String>,
        /// Generations that run at once per model; more wait for a slot.
        #[arg(long, default_value_t = 4)]
        max_concurrent: usize,
        /// max_tokens when a request does not set it.
        #[arg(long, default_value_t = 512)]
        default_max_tokens: usize,
        /// Seconds a request waits for a generation slot before a 429.
        #[arg(long, default_value_t = 30)]
        queue_timeout: u64,
        /// Seconds a generation may run.
        #[arg(long, default_value_t = 600)]
        request_timeout: u64,
        /// A fixed expert cache in MiB per model, bypassing the memory planner.
        #[arg(long)]
        expert_cache_mb: Option<u64>,
        /// Memory profile: performance (0.88 of available RAM), balanced (0.75) or saver (0.50).
        #[arg(long, default_value = "performance")]
        profile: String,
        /// A custom fraction of available RAM, in (0, 1].
        #[arg(long)]
        ram_fraction: Option<f64>,
        /// An explicit RAM budget per model, for example 8GiB.
        #[arg(long)]
        ram_budget: Option<String>,
        /// Threads reading experts from disk. Default: 4.
        #[arg(long)]
        io_threads: Option<usize>,
        #[command(flatten)]
        device: DeviceArgs,
        /// Context limit per conversation, in tokens (default: the model's).
        #[arg(long)]
        context: Option<usize>,
        /// Compute threads (default: the CPU's performance cores).
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Time prefill and decode, and record them with hardware, build flags, peak
    /// memory and a token check (docs/performance/BENCHMARKING.md).
    Bench {
        /// A model id, or a model directory.
        #[arg(long)]
        model: String,
        /// A model directory to search for the id (repeatable).
        #[arg(long = "dir")]
        dirs: Vec<PathBuf>,
        /// The prompt (encoded as `run` encodes it, without a chat template).
        #[arg(long, default_value = bench::DEFAULT_PROMPT)]
        prompt: String,
        /// Prompt length in tokens: the encoded prompt repeated or cut to this length.
        /// Default: the encoded prompt as is.
        #[arg(long)]
        prompt_tokens: Option<usize>,
        /// Tokens generated per run (the first comes from prefill).
        #[arg(long, default_value_t = 128)]
        decode_tokens: usize,
        /// Unmeasured runs first (they also fill the expert cache).
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        /// Measured runs.
        #[arg(long, default_value_t = 5)]
        runs: usize,
        /// Compute threads (default: the CPU's performance cores).
        #[arg(long)]
        threads: Option<usize>,
        /// A fixed expert cache in MiB, bypassing the memory planner (default: planned
        /// from available RAM, shrinking as the KV grows).
        #[arg(long)]
        expert_cache_mb: Option<u64>,
        /// Memory profile: performance (0.88 of available RAM), balanced (0.75) or saver (0.50).
        #[arg(long, default_value = "performance")]
        profile: String,
        /// A custom fraction of available RAM, in (0, 1].
        #[arg(long)]
        ram_fraction: Option<f64>,
        /// An explicit RAM budget, for example 8GiB; replaces the profile's.
        #[arg(long)]
        ram_budget: Option<String>,
        /// How a full per-layer expert pool picks its victim: lfru (the coldest, by
        /// decaying heat) or lru.
        #[arg(long, default_value = "lfru")]
        cache_policy: String,
        /// Threads reading experts from disk (0: the compute thread reads its own
        /// misses). Default: 4.
        #[arg(long)]
        io_threads: Option<usize>,
        /// Read experts bypassing the OS page cache (F_NOCACHE on macOS, O_DIRECT on
        /// Linux), so streaming does not fill the file cache.
        #[arg(long)]
        direct_io: bool,
        #[command(flatten)]
        prefetch: PrefetchArgs,
        #[command(flatten)]
        storage: StorageArgs,
        #[command(flatten)]
        device: DeviceArgs,
        /// Context limit in tokens (default: the model's).
        #[arg(long)]
        context: Option<usize>,
        /// A label stored in the record, for example `before` or `after`.
        #[arg(long)]
        label: Option<String>,
        /// A previous record (or a JSON array of token ids) whose tokens must match.
        #[arg(long)]
        reference: Option<PathBuf>,
        /// Also write the JSON record to this file.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Write every expert acquire of the measured runs to this file, one
        /// `layer expert` pair per line, for trace replay
        /// (`cargo run -p qubivex-expert-stream --example replay`).
        #[arg(long)]
        routing_trace: Option<PathBuf>,
        /// Print the JSON record instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Explain how a model would be placed in memory on this machine.
    Plan {
        /// A model id, or a model directory.
        model: String,
        /// A model directory to search for the id (repeatable).
        #[arg(long = "dir")]
        dirs: Vec<PathBuf>,
        /// performance (0.88 of available RAM), balanced (0.75) or saver (0.50).
        #[arg(long, default_value = "performance")]
        profile: String,
        /// A custom fraction of available RAM, in (0, 1] (the custom profile).
        #[arg(long)]
        ram_fraction: Option<f64>,
        /// An explicit RAM budget, for example 8GiB; replaces the profile's.
        #[arg(long)]
        ram_budget: Option<String>,
        /// Context per sequence in tokens (default: the model's, shortened to fit).
        #[arg(long = "ctx")]
        context: Option<u64>,
        /// Sequences whose KV must fit at once.
        #[arg(long, default_value_t = 1)]
        sequences: u32,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// List, inspect and import models.
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ModelsCommand {
    /// List the models in the model directories.
    List {
        /// A model directory to scan (repeatable). Default: $QUBIVEX_MODELS_DIRS, else ~/.qubivex/models.
        #[arg(long = "dir")]
        dirs: Vec<PathBuf>,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Show a model's dimensions, bytes, files and tensors.
    Inspect {
        /// A model id, or a model directory.
        model: String,
        /// A model directory to search for the id (repeatable).
        #[arg(long = "dir")]
        dirs: Vec<PathBuf>,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
        /// List every tensor with its dtype, shape and role.
        #[arg(long)]
        tensors: bool,
    },
    /// Write a manifest for a checkpoint directory or a .gguf file.
    Import {
        /// The checkpoint directory, or a .gguf file.
        path: PathBuf,
        /// Model id (default: from the directory or file name).
        #[arg(long)]
        id: Option<String>,
        /// Display name.
        #[arg(long)]
        name: Option<String>,
        /// A config value the checkpoint does not record, as KEY=VALUE (repeatable).
        #[arg(long = "set")]
        set: Vec<String>,
        /// Write manifest.json next to the weights instead of printing it.
        #[arg(long)]
        write: bool,
        /// Replace an existing manifest.json.
        #[arg(long)]
        force: bool,
        /// With --write, do not record per-expert checksums (experts.qvxsum). Recording
        /// reads every expert once.
        #[arg(long)]
        no_checksums: bool,
    },
    /// Check every expert against the model's recorded checksums (experts.qvxsum), or
    /// record them if there are none. The engine verifies expert reads against them.
    Verify {
        /// A model id, or a model directory.
        model: String,
        /// A model directory to search for the id (repeatable).
        #[arg(long = "dir")]
        dirs: Vec<PathBuf>,
        /// Record the checksums again from the files as they are now (after replacing
        /// the weights on purpose).
        #[arg(long)]
        record: bool,
        /// Threads reading experts (default: every CPU).
        #[arg(long)]
        threads: Option<usize>,
    },
    /// List the architectures this build supports.
    Architectures,
}

/// An error and its chain of causes, `: `-separated.
fn chain(e: &dyn std::error::Error) -> String {
    let mut s = e.to_string();
    let mut cur = e.source();
    while let Some(c) = cur {
        s.push_str(": ");
        s.push_str(&c.to_string());
        cur = c.source();
    }
    s
}

/// How `run` and `bench` size memory.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MemoryOptions {
    expert_cache_bytes: Option<u64>,
    profile: qubivex_memory::Profile,
    ram_budget: Option<u64>,
    cache_policy: qubivex_expert_stream::Policy,
    io_threads: Option<usize>,
    expert_io: qubivex_runtime::ExpertIo,
    prefetch: Option<qubivex_runtime::PrefetchOptions>,
    read_timeout: Option<qubivex_expert_stream::ReadTimeout>,
    read_retries: Option<u32>,
    verify: qubivex_runtime::VerifyExperts,
    backend: qubivex_runtime::BackendKind,
    metal_ops: qubivex_runtime::MetalOps,
}

/// The compute backend and device placement (Phase 8), shared by `run` and `bench`.
#[derive(clap::Args, Clone, Debug)]
pub(crate) struct DeviceArgs {
    /// Compute backend: cpu, or metal (the Apple GPU runs what --metal-ops chooses, the
    /// CPU the rest). Output bits are the same on both.
    #[arg(long, default_value = "cpu")]
    backend: String,
    /// With --backend metal: what runs on the GPU. layers (the default): whole layers,
    /// experts included, one submission per layer. Or a comma-separated list of single
    /// dense matmuls from attention, router and head (slower; kept for comparison).
    #[arg(long)]
    metal_ops: Option<String>,
}

impl DeviceArgs {
    fn options(&self) -> Result<(qubivex_runtime::BackendKind, qubivex_runtime::MetalOps), String> {
        let backend = match self.backend.as_str() {
            "cpu" => qubivex_runtime::BackendKind::Cpu,
            "metal" => qubivex_runtime::BackendKind::Metal,
            other => return Err(format!("unknown --backend {other:?}; use cpu or metal")),
        };
        let Some(list) = &self.metal_ops else {
            return Ok((backend, qubivex_runtime::MetalOps::default()));
        };
        if backend != qubivex_runtime::BackendKind::Metal {
            return Err("--metal-ops needs --backend metal".to_owned());
        }
        let mut ops = qubivex_runtime::MetalOps {
            attention: false,
            router: false,
            head: false,
            layers: false,
        };
        for item in list.split(',').map(str::trim).filter(|i| !i.is_empty()) {
            match item {
                "attention" => ops.attention = true,
                "router" => ops.router = true,
                "head" => ops.head = true,
                "layers" => ops.layers = true,
                other => {
                    return Err(format!(
                        "unknown --metal-ops item {other:?}; use attention, router, head or layers"
                    ));
                }
            }
        }
        Ok((backend, ops))
    }
}

/// Expert-read failure handling (ADR-006 amendment 4), shared by `run` and `bench`.
#[derive(clap::Args, Clone, Debug)]
pub(crate) struct StorageArgs {
    /// Seconds to wait for one expert's read before failing the request (plus the
    /// expert's size at 10 MB/s), or off. Default: 5.
    #[arg(long)]
    read_timeout: Option<String>,
    /// Extra attempts after a failed expert read. Default: 1.
    #[arg(long)]
    read_retries: Option<u32>,
    /// Check expert reads against the model's recorded checksums (experts.qvxsum, from
    /// `qubivex models verify`): auto (when they match the weights), off, or require.
    #[arg(long, default_value = "auto")]
    verify_experts: String,
    /// A directory holding a copy of the model's weight files (same names), read when
    /// a read from the model's own files fails. Repeatable; tried in order.
    #[arg(long = "mirror")]
    mirrors: Vec<PathBuf>,
}

impl StorageArgs {
    fn options(
        &self,
    ) -> Result<
        (
            Option<qubivex_expert_stream::ReadTimeout>,
            Option<u32>,
            qubivex_runtime::VerifyExperts,
        ),
        String,
    > {
        let verify = match self.verify_experts.as_str() {
            "auto" => qubivex_runtime::VerifyExperts::Auto,
            "off" => qubivex_runtime::VerifyExperts::Off,
            "require" => qubivex_runtime::VerifyExperts::Require,
            other => {
                return Err(format!(
                    "unknown --verify-experts {other:?}; use auto, off or require"
                ));
            }
        };
        let timeout = match self.read_timeout.as_deref() {
            None => None,
            Some("off") => Some(qubivex_expert_stream::ReadTimeout::NEVER),
            Some(v) => {
                let secs: f64 = v.parse().map_err(|_| {
                    format!("--read-timeout {v:?} is not a number of seconds or off")
                })?;
                let floor = std::time::Duration::try_from_secs_f64(secs)
                    .ok()
                    .filter(|d| !d.is_zero())
                    .ok_or_else(|| format!("--read-timeout {v} must be positive"))?;
                Some(qubivex_expert_stream::ReadTimeout {
                    floor,
                    ..qubivex_expert_stream::ReadTimeout::DEFAULT
                })
            }
        };
        Ok((timeout, self.read_retries, verify))
    }
}

/// Router-lookahead prefetch (Phase 7), shared by `run` and `bench`.
#[derive(clap::Args, Clone, Debug)]
pub(crate) struct PrefetchArgs {
    /// Read predicted experts ahead during decode: the next layer's router runs early
    /// on the current hidden state. Routing and output never change.
    #[arg(long)]
    prefetch: bool,
    /// The prefetch eviction guard: on (a prediction evicts a resident expert only if
    /// clearly hotter) or off.
    #[arg(long, default_value = "on")]
    prefetch_guard: String,
    /// Predict at most top-K × this many experts per layer (default 1).
    #[arg(long)]
    prefetch_width: Option<u32>,
    /// Stop predicting once this probability mass is covered, in (0, 1] (default 0.92).
    #[arg(long)]
    prefetch_confidence: Option<f32>,
    /// Weight of the moving average of earlier predictions, in [0, 1) (default 0.3).
    #[arg(long)]
    prefetch_smooth: Option<f32>,
}

impl PrefetchArgs {
    fn options(&self) -> Result<Option<qubivex_runtime::PrefetchOptions>, String> {
        let guard = match self.prefetch_guard.as_str() {
            "on" => true,
            "off" => false,
            other => return Err(format!("unknown --prefetch-guard {other:?}; use on or off")),
        };
        if !self.prefetch {
            return Ok(None);
        }
        let mut l = qubivex_expert_stream::predict::LookaheadConfig::default();
        if let Some(w) = self.prefetch_width {
            if w == 0 {
                return Err("--prefetch-width must be at least 1".to_owned());
            }
            l.width = w;
        }
        if let Some(c) = self.prefetch_confidence {
            if !(c > 0.0 && c <= 1.0) {
                return Err(format!("--prefetch-confidence {c} is not in (0, 1]"));
            }
            l.confidence = c;
        }
        if let Some(sm) = self.prefetch_smooth {
            if !(0.0..1.0).contains(&sm) {
                return Err(format!("--prefetch-smooth {sm} is not in [0, 1)"));
            }
            l.smooth = sm;
        }
        Ok(Some(qubivex_runtime::PrefetchOptions {
            lookahead: l,
            guard,
        }))
    }
}

impl MemoryOptions {
    /// The expert pools' replacement policy.
    pub(crate) const fn cache_policy(self) -> qubivex_expert_stream::Policy {
        self.cache_policy
    }

    /// The same options with these prefetch settings.
    const fn with_prefetch(mut self, p: Option<qubivex_runtime::PrefetchOptions>) -> Self {
        self.prefetch = p;
        self
    }

    /// The same options with these expert-read settings.
    const fn with_storage(
        mut self,
        (timeout, retries, verify): (
            Option<qubivex_expert_stream::ReadTimeout>,
            Option<u32>,
            qubivex_runtime::VerifyExperts,
        ),
    ) -> Self {
        self.read_timeout = timeout;
        self.read_retries = retries;
        self.verify = verify;
        self
    }

    /// The same options with this backend and placement.
    const fn with_device(
        mut self,
        (backend, ops): (qubivex_runtime::BackendKind, qubivex_runtime::MetalOps),
    ) -> Self {
        self.backend = backend;
        self.metal_ops = ops;
        self
    }

    /// Prefetch settings (`None`: off).
    pub(crate) const fn prefetch(self) -> Option<qubivex_runtime::PrefetchOptions> {
        self.prefetch
    }

    /// Engine options with these memory settings.
    pub(crate) fn engine(
        self,
        context: Option<usize>,
        threads: Option<usize>,
    ) -> qubivex_runtime::EngineOptions {
        qubivex_runtime::EngineOptions {
            expert_cache_bytes: self.expert_cache_bytes,
            max_context: context,
            threads,
            profile: self.profile,
            ram_budget: self.ram_budget,
            cache_policy: self.cache_policy,
            io_threads: self.io_threads,
            expert_io: self.expert_io,
            prefetch: self.prefetch,
            read_timeout: self.read_timeout,
            read_retries: self.read_retries,
            verify: self.verify,
            backend: self.backend,
            metal_ops: self.metal_ops,
            ..qubivex_runtime::EngineOptions::default()
        }
    }
}

fn memory_options(
    expert_cache_mb: Option<u64>,
    profile: &str,
    ram_fraction: Option<f64>,
    ram_budget: Option<&str>,
    cache_policy: &str,
    io_threads: Option<usize>,
    direct_io: bool,
) -> Result<MemoryOptions, String> {
    Ok(MemoryOptions {
        prefetch: None,
        read_timeout: None,
        read_retries: None,
        verify: qubivex_runtime::VerifyExperts::Auto,
        backend: qubivex_runtime::BackendKind::Cpu,
        metal_ops: qubivex_runtime::MetalOps::default(),
        io_threads,
        expert_io: if direct_io {
            qubivex_runtime::ExpertIo::Direct
        } else {
            qubivex_runtime::ExpertIo::Buffered
        },
        cache_policy: match cache_policy {
            "lfru" => qubivex_expert_stream::Policy::LFRU,
            "lru" => qubivex_expert_stream::Policy::Lru,
            other => return Err(format!("unknown cache policy {other:?}; use lfru or lru")),
        },
        expert_cache_bytes: expert_cache_mb.map(|mb| mb.saturating_mul(1 << 20)),
        profile: plan::profile(profile, ram_fraction)?,
        ram_budget: ram_budget.map(plan::parse_bytes).transpose()?.map(|b| b.0),
    })
}

fn emit(text: &str) -> Result<(), String> {
    std::io::stdout()
        .lock()
        .write_all(text.as_bytes())
        .map_err(|e| format!("cannot write output: {e}"))
}

fn to_json(v: &impl serde::Serialize) -> Result<String, String> {
    let mut t = serde_json::to_string_pretty(v).map_err(|e| e.to_string())?;
    t.push('\n');
    Ok(t)
}

/// Parses `args` (including the program name) and runs the command.
///
/// Returns the process exit code: 0 on success, 1 on a runtime error, 2 on a usage
/// error.
pub fn run<I, T>(args: I) -> ExitCode
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    match Cli::try_parse_from(args) {
        Ok(cli) => match execute(&cli.command) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("qubivex: {e}");
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            // `--help` and `--version` arrive here as "errors" with exit code 0.
            let code = e.exit_code();
            // Printing can fail only on a closed stdout/stderr, where nothing useful remains to do.
            let _ = e.print();
            ExitCode::from(u8::try_from(code).unwrap_or(2))
        }
    }
}

fn execute(cmd: &Command) -> Result<(), String> {
    match cmd {
        Command::System { json } => {
            let host = Host::live();
            let profile = qubivex_hardware::probe(&host);
            let snapshot = SystemProbe::new(host).snapshot().map_err(|e| chain(&e))?;
            let text = if *json {
                to_json(&system::Report {
                    profile: &profile,
                    snapshot: &snapshot,
                })?
            } else {
                system::render(&profile, &snapshot)
            };
            emit(&text)
        }
        Command::Doctor {
            storage,
            path,
            size_mib,
            json,
        } => {
            if !storage {
                return Err("nothing to check; pass --storage".to_owned());
            }
            emit(&doctor::storage(path, *size_mib, *json)?)
        }
        Command::Run {
            model,
            dirs,
            prompt,
            chat,
            max_tokens,
            temperature,
            top_p,
            seed,
            expert_cache_mb,
            profile,
            ram_fraction,
            ram_budget,
            cache_policy,
            io_threads,
            direct_io,
            prefetch,
            storage,
            device,
            context,
            threads,
            quiet,
        } => run::run(&run::RunArgs {
            model,
            dirs,
            prompt,
            chat: *chat,
            max_tokens: *max_tokens,
            sampling: qubivex_runtime::Sampling {
                temperature: *temperature,
                top_p: *top_p,
                seed: *seed,
            },
            memory: memory_options(
                *expert_cache_mb,
                profile,
                *ram_fraction,
                ram_budget.as_deref(),
                cache_policy,
                *io_threads,
                *direct_io,
            )?
            .with_prefetch(prefetch.options()?)
            .with_storage(storage.options()?)
            .with_device(device.options()?),
            mirrors: &storage.mirrors,
            context: *context,
            threads: *threads,
            quiet: *quiet,
        }),
        Command::Serve {
            models,
            dirs,
            listen,
            api_key,
            admin_token,
            max_concurrent,
            default_max_tokens,
            queue_timeout,
            request_timeout,
            expert_cache_mb,
            profile,
            ram_fraction,
            ram_budget,
            io_threads,
            device,
            context,
            threads,
        } => serve::serve(&serve::ServeArgs {
            models,
            dirs,
            addr: *listen,
            api_key: api_key.clone(),
            admin_token: admin_token.clone(),
            max_concurrent: *max_concurrent,
            default_max_tokens: *default_max_tokens,
            queue_timeout: std::time::Duration::from_secs(*queue_timeout),
            request_timeout: std::time::Duration::from_secs(*request_timeout),
            engine: memory_options(
                *expert_cache_mb,
                profile,
                *ram_fraction,
                ram_budget.as_deref(),
                "lfru",
                *io_threads,
                false,
            )?
            .with_device(device.options()?)
            .engine(*context, *threads),
        }),
        Command::Bench {
            model,
            dirs,
            prompt,
            prompt_tokens,
            decode_tokens,
            warmup,
            runs,
            threads,
            expert_cache_mb,
            profile,
            ram_fraction,
            ram_budget,
            cache_policy,
            io_threads,
            direct_io,
            prefetch,
            storage,
            device,
            context,
            label,
            reference,
            out,
            routing_trace,
            json,
        } => emit(&bench::bench(&bench::BenchArgs {
            model,
            dirs,
            prompt,
            prompt_tokens: *prompt_tokens,
            decode_tokens: *decode_tokens,
            warmup: *warmup,
            runs: *runs,
            threads: *threads,
            memory: memory_options(
                *expert_cache_mb,
                profile,
                *ram_fraction,
                ram_budget.as_deref(),
                cache_policy,
                *io_threads,
                *direct_io,
            )?
            .with_prefetch(prefetch.options()?)
            .with_storage(storage.options()?)
            .with_device(device.options()?),
            mirrors: &storage.mirrors,
            context: *context,
            label: label.as_deref(),
            reference: reference.as_deref(),
            out: out.as_deref(),
            routing_trace: routing_trace.as_deref(),
            json: *json,
        })?),
        Command::Plan {
            model,
            dirs,
            profile,
            ram_fraction,
            ram_budget,
            context,
            sequences,
            json,
        } => emit(&plan::run(&plan::PlanArgs {
            model,
            dirs,
            profile,
            ram_fraction: *ram_fraction,
            ram_budget: ram_budget.as_deref(),
            context: *context,
            sequences: *sequences,
            json: *json,
        })?),
        Command::Models { command } => emit(&match command {
            ModelsCommand::List { dirs, json } => models::list(dirs, *json)?,
            ModelsCommand::Inspect {
                model,
                dirs,
                json,
                tensors,
            } => models::inspect(model, dirs, *json, *tensors)?,
            ModelsCommand::Import {
                path,
                id,
                name,
                set,
                write,
                force,
                no_checksums,
            } => models::import_cmd(&models::ImportArgs {
                path,
                id: id.clone(),
                name: name.clone(),
                set,
                write: *write,
                force: *force,
                checksums: !*no_checksums,
            })?,
            ModelsCommand::Verify {
                model,
                dirs,
                record,
                threads,
            } => models::verify(model, dirs, *record, threads.unwrap_or(0))?,
            ModelsCommand::Architectures => models::architectures(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }
}
