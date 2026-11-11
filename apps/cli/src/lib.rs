//! The `qubivex` command-line interface.
//!
//! Subcommands arrive with the phases that implement them (MIGRATION_PLAN §3):
//! `system` and `doctor` (Phase 2), `models` (Phase 3), `run` and `bench` (Phase 4),
//! `plan` (Phase 5); `serve` later.

mod analyse;
mod bench;
mod doctor;
mod keys;
mod models;
mod plan;
mod pull;
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
        /// Keep only the K likeliest tokens before top-p (0: off).
        #[arg(long, default_value_t = 0)]
        top_k: u32,
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
    Serve(serve::ServeFlags),
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
        /// Simulate a disk of this many MB/s (decimal), shared by all I/O threads:
        /// warm reads otherwise come from the page cache and never saturate.
        #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
        disk_mbps: Option<u64>,
        /// Time added to every read of the simulated disk, in microseconds.
        #[arg(long, default_value_t = 0, requires = "disk_mbps")]
        disk_latency_us: u64,
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
        /// Also plan a discrete GPU with this much free memory, for example 16GiB
        /// (CUDA and Vulkan are not built yet; `run --backend sim` simulates one).
        #[arg(long)]
        vram: Option<String>,
        /// With --vram: an explicit VRAM budget; replaces the profile's fraction.
        #[arg(long)]
        vram_budget: Option<String>,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// List, inspect and import models.
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
    /// Make named API keys for `serve`: each key belongs to a client, which then has its
    /// own rate limit and prefix cache.
    Keys {
        #[command(subcommand)]
        command: KeysCommand,
    },
}

#[derive(Debug, Subcommand)]
enum KeysCommand {
    /// Print a new key for CLIENT once, and the [[serve.client_keys]] table holding its
    /// hash for the serve config file.
    New {
        /// The client's name: letters, digits, '.', '_' or '-' (shown in the log).
        client: String,
        /// Requests per minute this client may make (default: the server's --rate-limit).
        #[arg(long)]
        rate_limit: Option<u32>,
        /// Requests this client may make at once after being idle (default: its
        /// --rate-limit).
        #[arg(long)]
        rate_limit_burst: Option<u32>,
    },
    /// Print the SHA-256 of a key read from standard input, for key_sha256.
    Hash,
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
        /// With --write, do not record checksums (in the manifest and experts.qvxsum).
        /// Recording reads every file once.
        #[arg(long)]
        no_checksums: bool,
    },
    /// Estimate the memory a model needs and say whether it can run on this machine.
    #[command(visible_alias = "analyze")]
    Analyse {
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
        /// Context per sequence in tokens (default: the model's).
        #[arg(long = "ctx")]
        context: Option<u64>,
        /// Sequences whose KV must fit at once.
        #[arg(long, default_value_t = 1)]
        sequences: u32,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Bring a model's manifest up to the current schema, adding its source from the
    /// registry and the checksum of every file. Prints the result unless --write is given.
    Migrate {
        /// A model id, or a model directory.
        model: String,
        /// A model directory to search for the id (repeatable).
        #[arg(long = "dir")]
        dirs: Vec<PathBuf>,
        /// Replace manifest.json instead of printing it. Without --no-checksums this reads
        /// every file once.
        #[arg(long)]
        write: bool,
        /// Do not add file checksums.
        #[arg(long)]
        no_checksums: bool,
    },
    /// Check every file against the checksums in the manifest and every expert against
    /// experts.qvxsum, recording the latter if there are none. The engine verifies expert
    /// reads against them.
    #[command(visible_alias = "validate")]
    Verify {
        /// A model id, or a model directory.
        model: String,
        /// A model directory to search for the id (repeatable).
        #[arg(long = "dir")]
        dirs: Vec<PathBuf>,
        /// Record both sets of checksums again from the files as they are now (after
        /// replacing the weights on purpose).
        #[arg(long)]
        record: bool,
        /// Threads reading experts (default: every CPU).
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Find models on Hugging Face.
    Search {
        /// What to look for.
        query: String,
        /// How many results to show.
        #[arg(long, default_value_t = 20)]
        limit: usize,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Download a model from Hugging Face and install it. An interrupted pull continues
    /// when the same command runs again. Private and gated models read HF_TOKEN, or the
    /// token `hf auth login` saved. HF_ENDPOINT selects a mirror.
    Pull {
        /// The repository, as owner/name.
        repository: String,
        /// A file to fetch, by its path in the repository (repeatable). Needed when a
        /// repository has several weight files, as GGUF repositories usually do.
        #[arg(long = "file")]
        files: Vec<String>,
        /// A branch, tag or commit (default: the default branch).
        #[arg(long)]
        revision: Option<String>,
        /// Model id (default: from the repository, or the GGUF file's name).
        #[arg(long)]
        id: Option<String>,
        /// Display name.
        #[arg(long)]
        name: Option<String>,
        /// A config value the checkpoint does not record, as KEY=VALUE (repeatable).
        #[arg(long = "set")]
        set: Vec<String>,
        /// The models directory to install into. Default: the first of $QUBIVEX_MODELS_DIRS,
        /// else ~/.qubivex/models.
        #[arg(long)]
        dir: Option<PathBuf>,
        /// Files to fetch at once.
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
    },
    /// Delete an installed model from a models directory. Without --yes, only says what
    /// would be deleted.
    Remove {
        /// A model id.
        model: String,
        /// A models directory to search for the id (repeatable).
        #[arg(long = "dir")]
        dirs: Vec<PathBuf>,
        /// Delete it.
        #[arg(long)]
        yes: bool,
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
    disk_limit: Option<qubivex_runtime::DiskLimit>,
    read_timeout: Option<qubivex_expert_stream::ReadTimeout>,
    read_retries: Option<u32>,
    verify: qubivex_runtime::VerifyExperts,
    backend: qubivex_runtime::BackendKind,
    metal_ops: qubivex_runtime::MetalOps,
    sim: Option<qubivex_runtime::SimConfig>,
    vulkan: qubivex_runtime::VulkanConfig,
    vram_budget: Option<u64>,
}

/// What [`DeviceArgs`] chose.
pub(crate) struct DeviceChoice {
    backend: qubivex_runtime::BackendKind,
    metal_ops: qubivex_runtime::MetalOps,
    sim: Option<qubivex_runtime::SimConfig>,
    vulkan: qubivex_runtime::VulkanConfig,
    vram_budget: Option<u64>,
}

/// The compute backend and device placement (Phase 8), shared by `run` and `bench`.
#[derive(clap::Args, Clone, Debug)]
pub(crate) struct DeviceArgs {
    /// Compute backend: cpu, metal (the Apple GPU runs what --metal-ops chooses, the
    /// CPU the rest), vulkan (a Vulkan GPU runs every matmul from its own memory), or
    /// sim (a simulated discrete GPU with its own memory runs every matmul; for testing
    /// VRAM planning without one). Output bits are the same on all four.
    #[arg(long, default_value = "cpu")]
    backend: String,
    /// With --backend sim: the simulated device's memory, for example 8GiB.
    #[arg(long)]
    sim_vram: Option<String>,
    /// With --backend sim: host-to-device bandwidth in GB/s (default 25, PCIe 4.0).
    #[arg(long)]
    sim_bandwidth: Option<f64>,
    /// With --backend sim: sleep for each copy's modelled time instead of only
    /// counting it.
    #[arg(long)]
    sim_delay: bool,
    /// With --backend vulkan: the device's index in `qubivex system` (default: the
    /// first discrete GPU, else the first integrated one).
    #[arg(long)]
    vulkan_device: Option<usize>,
    /// With --backend vulkan: the device memory to allow, for example 4GiB (default:
    /// its largest device-local heap).
    #[arg(long)]
    vulkan_vram: Option<String>,
    /// With --backend vulkan: time every submission on the GPU with timestamp
    /// queries (for `bench`'s device timings; adds two timestamps per submission).
    #[arg(long)]
    vulkan_timestamps: bool,
    /// With --backend vulkan: run a layer's experts as one fused submission when no
    /// expert computes more than N rows (default 4; 0: never, each expert's matmuls
    /// submitted on their own as before).
    #[arg(long, value_name = "N")]
    vulkan_fuse_rows: Option<usize>,
    /// With a discrete device: an explicit VRAM budget, for example 6GiB; replaces the
    /// profile's fraction of the device's memory.
    #[arg(long)]
    vram_budget: Option<String>,
    /// With --backend metal: what runs on the GPU. token (the default): whole layers,
    /// experts included, with routing on the GPU so a token is one submission while
    /// every expert is resident. layers: whole layers with routing on the host, one
    /// submission per layer. Or a comma-separated list of single dense matmuls from
    /// attention, router and head (slower; kept for comparison).
    #[arg(long)]
    metal_ops: Option<String>,
}

impl DeviceArgs {
    fn options(&self) -> Result<DeviceChoice, String> {
        let backend = match self.backend.as_str() {
            "cpu" => qubivex_runtime::BackendKind::Cpu,
            "metal" => qubivex_runtime::BackendKind::Metal,
            "sim" => qubivex_runtime::BackendKind::Sim,
            "vulkan" => qubivex_runtime::BackendKind::Vulkan,
            other => {
                return Err(format!(
                    "unknown --backend {other:?}; use cpu, metal, vulkan or sim"
                ));
            }
        };
        let vulkan = backend == qubivex_runtime::BackendKind::Vulkan;
        if !vulkan
            && (self.vulkan_device.is_some()
                || self.vulkan_vram.is_some()
                || self.vulkan_timestamps
                || self.vulkan_fuse_rows.is_some())
        {
            return Err(
                "--vulkan-device, --vulkan-vram, --vulkan-timestamps and --vulkan-fuse-rows \
                 need --backend vulkan"
                    .to_owned(),
            );
        }
        let vulkan = qubivex_runtime::VulkanConfig {
            device: self.vulkan_device,
            capacity: self
                .vulkan_vram
                .as_deref()
                .map(plan::parse_bytes)
                .transpose()?
                .map(|b| b.0),
            timestamps: self.vulkan_timestamps,
            fuse_rows: self.vulkan_fuse_rows,
        };
        let sim = backend == qubivex_runtime::BackendKind::Sim;
        if !sim && (self.sim_vram.is_some() || self.sim_bandwidth.is_some() || self.sim_delay) {
            return Err(
                "--sim-vram, --sim-bandwidth and --sim-delay need --backend sim".to_owned(),
            );
        }
        if !sim && backend != qubivex_runtime::BackendKind::Vulkan && self.vram_budget.is_some() {
            return Err(
                "--vram-budget needs a discrete device (--backend vulkan or sim)".to_owned(),
            );
        }
        let vram_budget = self
            .vram_budget
            .as_deref()
            .map(plan::parse_bytes)
            .transpose()?
            .map(|b| b.0);
        let sim = if sim {
            let capacity = self
                .sim_vram
                .as_deref()
                .map(plan::parse_bytes)
                .transpose()?
                .map_or(qubivex_runtime::DEFAULT_SIM_VRAM, |b| b.0);
            let mut c = qubivex_runtime::SimConfig::new(capacity);
            if let Some(gbs) = self.sim_bandwidth {
                if !(gbs.is_finite() && gbs >= 0.0) {
                    return Err(format!(
                        "--sim-bandwidth must be a non-negative number, got {gbs}"
                    ));
                }
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    c.bandwidth = (gbs * 1e9) as u64;
                }
            }
            c.delay = self.sim_delay;
            Some(c)
        } else {
            None
        };
        let choice = |metal_ops| DeviceChoice {
            backend,
            metal_ops,
            sim,
            vulkan,
            vram_budget,
        };
        let Some(list) = &self.metal_ops else {
            return Ok(choice(qubivex_runtime::MetalOps::default()));
        };
        if backend != qubivex_runtime::BackendKind::Metal {
            return Err("--metal-ops needs --backend metal".to_owned());
        }
        let mut ops = qubivex_runtime::MetalOps {
            attention: false,
            router: false,
            head: false,
            layers: false,
            token: false,
        };
        for item in list.split(',').map(str::trim).filter(|i| !i.is_empty()) {
            match item {
                "attention" => ops.attention = true,
                "router" => ops.router = true,
                "head" => ops.head = true,
                "layers" => ops.layers = true,
                "token" => {
                    ops.layers = true;
                    ops.token = true;
                }
                other => {
                    return Err(format!(
                        "unknown --metal-ops item {other:?}; use attention, router, head, layers or token"
                    ));
                }
            }
        }
        Ok(choice(ops))
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
    /// Stop speculating while reads hold compute back: on or off. On a saturated disk
    /// speed follows the bytes read, so unused speculation costs throughput.
    #[arg(long, default_value = "on")]
    prefetch_throttle: String,
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
        let throttle = match self.prefetch_throttle.as_str() {
            "on" => true,
            "off" => false,
            other => {
                return Err(format!(
                    "unknown --prefetch-throttle {other:?}; use on or off"
                ));
            }
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
            throttle,
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
    const fn with_device(mut self, d: DeviceChoice) -> Self {
        self.backend = d.backend;
        self.metal_ops = d.metal_ops;
        self.sim = d.sim;
        self.vulkan = d.vulkan;
        self.vram_budget = d.vram_budget;
        self
    }

    /// The same options with reads going through a simulated disk.
    const fn with_disk(mut self, disk_limit: Option<qubivex_runtime::DiskLimit>) -> Self {
        self.disk_limit = disk_limit;
        self
    }

    /// The simulated disk (`None`: the real one).
    pub(crate) const fn disk_limit(self) -> Option<qubivex_runtime::DiskLimit> {
        self.disk_limit
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
            disk_limit: self.disk_limit,
            read_timeout: self.read_timeout,
            read_retries: self.read_retries,
            verify: self.verify,
            backend: self.backend,
            metal_ops: self.metal_ops,
            sim: self.sim,
            vulkan: self.vulkan,
            vram_budget: self.vram_budget,
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
        disk_limit: None,
        read_timeout: None,
        read_retries: None,
        verify: qubivex_runtime::VerifyExperts::Auto,
        backend: qubivex_runtime::BackendKind::Cpu,
        metal_ops: qubivex_runtime::MetalOps::default(),
        sim: None,
        vulkan: qubivex_runtime::VulkanConfig::default(),
        vram_budget: None,
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
            top_k,
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
                top_k: *top_k,
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
        Command::Serve(flags) => serve::serve(serve::plan(flags.clone())?),
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
            disk_mbps,
            disk_latency_us,
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
            .with_disk(disk_mbps.map(|mbps| qubivex_runtime::DiskLimit {
                bytes_per_sec: mbps.saturating_mul(1_000_000),
                latency: std::time::Duration::from_micros(*disk_latency_us),
            }))
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
            vram,
            vram_budget,
            json,
        } => emit(&plan::run(&plan::PlanArgs {
            model,
            dirs,
            profile,
            ram_fraction: *ram_fraction,
            ram_budget: ram_budget.as_deref(),
            context: *context,
            sequences: *sequences,
            vram: vram.as_deref(),
            vram_budget: vram_budget.as_deref(),
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
            ModelsCommand::Analyse {
                model,
                dirs,
                profile,
                ram_fraction,
                ram_budget,
                context,
                sequences,
                json,
            } => analyse::run(&plan::PlanArgs {
                model,
                dirs,
                profile,
                ram_fraction: *ram_fraction,
                ram_budget: ram_budget.as_deref(),
                context: *context,
                sequences: *sequences,
                vram: None,
                vram_budget: None,
                json: *json,
            })?,
            ModelsCommand::Migrate {
                model,
                dirs,
                write,
                no_checksums,
            } => models::migrate(model, dirs, *write, !*no_checksums)?,
            ModelsCommand::Search { query, limit, json } => pull::search_cmd(query, *limit, *json)?,
            ModelsCommand::Pull {
                repository,
                files,
                revision,
                id,
                name,
                set,
                dir,
                concurrency,
            } => pull::pull_cmd(&pull::PullArgs {
                repository,
                files,
                revision: revision.as_deref(),
                id: id.as_deref(),
                name: name.as_deref(),
                set,
                dir: dir.as_deref(),
                concurrency: *concurrency,
            })?,
            ModelsCommand::Remove { model, dirs, yes } => models::remove(model, dirs, *yes)?,
            ModelsCommand::Architectures => models::architectures(),
        }),
        Command::Keys { command } => emit(&match command {
            KeysCommand::New {
                client,
                rate_limit,
                rate_limit_burst,
            } => keys::new(client, *rate_limit, *rate_limit_burst)?,
            KeysCommand::Hash => keys::hash()?,
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
