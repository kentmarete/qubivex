//! `qubivex bench`: decode and prefill timings with everything needed to compare runs.
//!
//! One record holds the hardware, the build (commit, profile, flags), the operating
//! point (threads, expert cache, prompt and decode lengths, warm-up and measured
//! runs), the timings (median and P95), peak memory and a correctness check: greedy
//! tokens must be identical across runs, and equal to a reference record if one is
//! given (MIGRATION_PLAN §3a; docs/performance/BENCHMARKING.md).
//!
//! Each run starts from an empty KV cache. Prefill is one forward pass over the whole
//! prompt; each decode step is one forward pass over one token, timed on its own.
//! Greedy decoding (argmax, lowest id on ties, as `qubivex run`), and an
//! end-of-sequence token does not stop the run, so every run does the same work.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use qubivex_core::Bytes;
use qubivex_hardware::{CpuInfo, Host, MemoryInfo, OperatingSystem, Probed};
use qubivex_model::{ArchitectureRegistry, find};
use qubivex_runtime::{Engine, Logits};
use serde::{Deserialize, Serialize};

use crate::models::roots;
use crate::{MemoryOptions, chain};

/// The record format version; bump it when a field changes meaning.
const SCHEMA: u32 = 1;

/// The prompt of the recorded Colibri comparison (in docs/performance/).
pub(crate) const DEFAULT_PROMPT: &str = "Write a long story about a lighthouse keeper.";

pub(crate) struct BenchArgs<'a> {
    pub(crate) model: &'a str,
    pub(crate) dirs: &'a [PathBuf],
    pub(crate) prompt: &'a str,
    pub(crate) prompt_tokens: Option<usize>,
    pub(crate) decode_tokens: usize,
    pub(crate) warmup: usize,
    pub(crate) runs: usize,
    pub(crate) threads: Option<usize>,
    pub(crate) memory: MemoryOptions,
    /// Copies of the weight files to fail over to (`--mirror`).
    pub(crate) mirrors: &'a [PathBuf],
    pub(crate) context: Option<usize>,
    pub(crate) label: Option<&'a str>,
    pub(crate) reference: Option<&'a Path>,
    pub(crate) out: Option<&'a Path>,
    pub(crate) routing_trace: Option<&'a Path>,
    pub(crate) json: bool,
}

/// One benchmark record.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Record {
    schema: u32,
    /// Free-form label (`--label`), for example `before` or `after`.
    label: Option<String>,
    /// UTC start time, RFC 3339.
    started: String,
    build: Build,
    hardware: Hardware,
    model: ModelInfo,
    config: Config,
    prefill: Prefill,
    decode: Decode,
    memory: Memory,
    experts: Experts,
    check: Check,
    runs: Vec<RunSummary>,
    /// Greedy tokens of the first measured run (prefill's token first); a later
    /// record may use this one as its `--reference`.
    tokens: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Build {
    version: String,
    git: String,
    rustc: String,
    profile: String,
    opt_level: String,
    debug_assertions: bool,
    target: String,
    /// Target features enabled at compile time (run-time SIMD is in `hardware.cpu.simd`).
    target_features: String,
    rustflags: String,
}

impl Build {
    fn current() -> Self {
        Self {
            version: qubivex_core::VERSION.to_owned(),
            git: env!("QUBIVEX_BUILD_GIT").to_owned(),
            rustc: env!("QUBIVEX_BUILD_RUSTC").to_owned(),
            profile: env!("QUBIVEX_BUILD_PROFILE").to_owned(),
            opt_level: env!("QUBIVEX_BUILD_OPT_LEVEL").to_owned(),
            debug_assertions: cfg!(debug_assertions),
            target: env!("QUBIVEX_BUILD_TARGET").to_owned(),
            target_features: env!("QUBIVEX_BUILD_TARGET_FEATURES").to_owned(),
            rustflags: env!("QUBIVEX_BUILD_RUSTFLAGS").to_owned(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Hardware {
    cpu: CpuInfo,
    memory: MemoryInfo,
    os: OperatingSystem,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelInfo {
    id: String,
    name: String,
    architecture: String,
    dir: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    threads: usize,
    expert_cache: Bytes,
    /// Experts each layer's pool may hold, layer by layer (absent before Phase 6).
    #[serde(default)]
    expert_slots: Vec<u32>,
    /// The expert pools' replacement policy (absent before Phase 6).
    #[serde(default)]
    cache_policy: Option<String>,
    /// Expert I/O threads (absent before Phase 6 slice 2).
    #[serde(default)]
    io_threads: Option<usize>,
    /// How experts were read: `buffered`, `f_nocache` or `o_direct` (absent before
    /// Phase 6 slice 3).
    #[serde(default)]
    expert_io: Option<String>,
    /// Whether expert reads were verified against recorded checksums: `verified`,
    /// `off`, `missing` or `stale` (absent before ADR-006 amendment 4).
    #[serde(default)]
    verification: Option<String>,
    /// Router-lookahead prefetch settings; absent when prefetch is off (and before
    /// Phase 7).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prefetch: Option<PrefetchConfig>,
    /// The compute backend and device placement; absent on the CPU (and before
    /// Phase 8 slice 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    device: Option<DeviceConfig>,
    context: usize,
    prompt: String,
    prompt_tokens: usize,
    /// Generated tokens per run: one from prefill, the rest from decode steps.
    generated_tokens: usize,
    warmup_runs: usize,
    measured_runs: usize,
}

/// Order statistics of a sample, in milliseconds.
#[derive(Debug, Serialize, Deserialize)]
struct Stats {
    n: usize,
    min: f64,
    median: f64,
    p95: f64,
    max: f64,
    mean: f64,
}

impl Stats {
    /// Nearest-rank percentiles. `None` for an empty sample.
    fn of(ms: &[f64]) -> Option<Self> {
        if ms.is_empty() {
            return None;
        }
        let mut v = ms.to_vec();
        v.sort_by(f64::total_cmp);
        let rank = |p: f64| {
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let r = (p * v.len() as f64).ceil() as usize;
            v[r.clamp(1, v.len()) - 1]
        };
        #[allow(clippy::cast_precision_loss)]
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        Some(Self {
            n: v.len(),
            min: v[0],
            median: rank(0.5),
            p95: rank(0.95),
            max: v[v.len() - 1],
            mean,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Prefill {
    /// Prefill time per run.
    ms: Stats,
    /// Prompt tokens over the median prefill time.
    tok_s_at_median: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct Decode {
    /// Every measured decode step of every run.
    step_ms: Option<Stats>,
    /// One over the median step time.
    tok_s_at_median: f64,
    /// Decode tokens per second of each run (steps over their summed time).
    run_tok_s: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Memory {
    rss_after_load: Probed<Bytes>,
    /// The peak resident set size of the whole process.
    peak_rss: Probed<Bytes>,
    /// `VmHWM` (the kernel's high-water mark), or `sampled`: the largest RSS seen
    /// after load and after each prefill and run (macOS, where the kernel's peak needs
    /// `getrusage`); a sampled peak misses transient scratch inside a forward pass.
    peak_source: String,
    /// The memory plan's RAM budget (`None`: a fixed expert cache was requested).
    plan_budget: Option<Bytes>,
    /// The ledger after the last measured run: bytes the engine allocated.
    ledger: Ledger,
}

/// The engine's RAM ledger.
#[derive(Debug, Serialize, Deserialize)]
struct Ledger {
    dense: Bytes,
    expert_cache: Bytes,
    kv: Bytes,
    total: Bytes,
}

#[derive(Debug, Serialize, Deserialize)]
struct Experts {
    /// Expert store counters over the measured runs only.
    hits: u64,
    loads: u64,
    bytes_read: Bytes,
    evictions: u64,
    /// `hits / (hits + loads)` (absent before Phase 6).
    #[serde(default)]
    hit_rate: f64,
    /// Bytes read per generated token (absent before Phase 6).
    #[serde(default)]
    bytes_per_token: Bytes,
    /// Slab buffers allocated during the measured runs (absent before Phase 6).
    #[serde(default)]
    slab_allocs: u64,
    /// Time the forward pass spent blocked on expert loads, per generated token
    /// (absent before Phase 6 slice 2).
    #[serde(default)]
    wait_ms_per_token: f64,
    /// Failed reads attempted again, and waits that timed out, during the measured
    /// runs (absent before ADR-006 amendment 4). Non-zero means the drive misbehaved.
    #[serde(default)]
    read_retries: u64,
    #[serde(default)]
    read_timeouts: u64,
    /// What prefetch did during the measured runs; absent when it is off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prefetch: Option<PrefetchRecord>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DeviceConfig {
    backend: String,
    device: Option<String>,
    /// The dense matmul kinds on the device (`head`, `attention,router,head`, ...).
    ops: Option<String>,
    tensors: usize,
    bytes: Bytes,
    /// Why the plan ran the CPU pass instead, if it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fallback: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PrefetchConfig {
    guard: bool,
    smooth: f32,
    confidence: f32,
    width: u32,
}

/// Prefetch counters over the measured runs (Phase 7 metrics).
#[derive(Debug, Serialize, Deserialize)]
struct PrefetchRecord {
    /// Routed experts that were predicted, over decode layers with a prediction.
    recall: f64,
    /// Predicted experts that were routed.
    precision: f64,
    /// Experts predicted per layer, on average.
    predicted_per_layer: f64,
    /// Predicted experts queued for reading.
    issued: u64,
    /// Predicted experts already resident or loading.
    present: u64,
    /// Not read: no free slot, or the guard kept the resident expert.
    refused: u64,
    /// Prefetched experts demanded after their read finished.
    ready: u64,
    /// Prefetched experts demanded while still loading.
    late: u64,
    /// `(ready + late) / issued`.
    useful_rate: f64,
    /// Prefetched experts evicted unused, and their bytes.
    wasted: u64,
    wasted_bytes: Bytes,
    /// Queued prefetches dropped before reading (stale, or at the end of a turn).
    cancelled: u64,
    /// Mean time from issue to demand of a used prefetch.
    lead_ms: f64,
}

#[allow(clippy::cast_precision_loss)]
fn prefetch_record(
    before: qubivex_runtime::PrefetchReport,
    after: qubivex_runtime::PrefetchReport,
) -> PrefetchRecord {
    let (p0, p1) = (before.prediction, after.prediction);
    let p = qubivex_runtime::PredictionStats {
        layers: p1.layers - p0.layers,
        predicted: p1.predicted - p0.predicted,
        routed: p1.routed - p0.routed,
        correct: p1.correct - p0.correct,
    };
    let (s0, s1) = (before.store, after.store);
    let d = |f: fn(&qubivex_expert_stream::PrefetchStats) -> u64| f(&s1) - f(&s0);
    let (issued, ready, late) = (d(|s| s.issued), d(|s| s.ready), d(|s| s.late));
    PrefetchRecord {
        recall: p.recall(),
        precision: p.precision(),
        predicted_per_layer: p.predicted as f64 / (p.layers as f64).max(1.0),
        issued,
        present: d(|s| s.present),
        refused: d(|s| s.refused),
        ready,
        late,
        useful_rate: (ready + late) as f64 / (issued as f64).max(1.0),
        wasted: d(|s| s.wasted),
        wasted_bytes: Bytes(d(|s| s.wasted_bytes)),
        cancelled: d(|s| s.cancelled),
        lead_ms: d(|s| s.lead_ns) as f64 / 1e6 / ((ready + late) as f64).max(1.0),
    }
}

/// The correctness part of the record.
#[derive(Debug, Serialize, Deserialize)]
struct Check {
    /// Every warm-up and measured run produced the same tokens (invariant 1).
    deterministic: bool,
    /// The run and token index of the first divergence from the first run.
    first_divergence: Option<(usize, usize)>,
    /// The reference record or token file, if one was given.
    reference: Option<PathBuf>,
    /// Whether the tokens equal the reference's (over the shorter of the two).
    reference_match: Option<bool>,
    /// The first token index that differs from the reference.
    reference_divergence: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RunSummary {
    prefill_ms: f64,
    decode_step_ms: Option<Stats>,
    decode_tok_s: f64,
}

/// One run's raw timings and tokens.
struct Sample {
    prefill: Duration,
    steps: Vec<Duration>,
    tokens: Vec<u32>,
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Argmax, lowest id on ties (the engine's greedy rule).
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, v) in logits.iter().enumerate() {
        if v.total_cmp(&logits[best]).is_gt() {
            best = i;
        }
    }
    u32::try_from(best).unwrap_or(0)
}

fn one_run(engine: &Engine, prompt: &[u32], generated: usize) -> Result<Sample, String> {
    let mut cache = engine.new_cache();
    let t = Instant::now();
    let logits = engine
        .forward(prompt, &mut cache, Logits::Last, None)
        .map_err(|e| chain(&e))?;
    let prefill = t.elapsed();
    let mut next = argmax(&logits);
    let mut tokens = Vec::with_capacity(generated);
    let mut steps = Vec::with_capacity(generated.saturating_sub(1));
    tokens.push(next);
    while tokens.len() < generated {
        let t = Instant::now();
        let logits = engine
            .forward(&[next], &mut cache, Logits::Last, None)
            .map_err(|e| chain(&e))?;
        steps.push(t.elapsed());
        next = argmax(&logits);
        tokens.push(next);
    }
    engine.end_turn();
    Ok(Sample {
        prefill,
        steps,
        tokens,
    })
}

/// `n` prompt tokens: the encoded prompt, repeated (or cut) to length.
fn prompt_ids(encoded: &[u32], n: Option<usize>) -> Vec<u32> {
    match n {
        None => encoded.to_vec(),
        Some(n) => encoded.iter().copied().cycle().take(n).collect(),
    }
}

/// Reads a reference: a bench record (its `tokens`) or a JSON array of token ids.
fn read_reference(path: &Path) -> Result<Vec<u32>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Reference {
        Record { tokens: Vec<u32> },
        Tokens(Vec<u32>),
    }
    match serde_json::from_str::<Reference>(&text) {
        Ok(Reference::Record { tokens } | Reference::Tokens(tokens)) => Ok(tokens),
        Err(e) => Err(format!(
            "{}: not a bench record or a token array: {e}",
            path.display()
        )),
    }
}

/// The first index where `a` and `b` differ, over the shorter of the two.
fn divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// `path` with the home directory written as `~`, so committed records don't carry
/// a user name.
fn home_relative(path: &Path) -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .and_then(|home| {
            path.strip_prefix(home)
                .ok()
                .map(|rest| Path::new("~").join(rest))
        })
        .unwrap_or_else(|| path.to_path_buf())
}

/// Whole seconds since the epoch as RFC 3339 UTC.
fn rfc3339(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Civil from days (Howard Hinnant's algorithm), for days since 1970-01-01.
    #[allow(clippy::cast_possible_wrap)]
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// Tracks peak RSS: the kernel's high-water mark where there is one, else samples.
struct PeakRss {
    host: Host,
    sampled: Option<Bytes>,
}

impl PeakRss {
    const fn new(host: Host) -> Self {
        Self {
            host,
            sampled: None,
        }
    }

    /// Samples the current RSS (outside timed regions only: on macOS this runs `ps`).
    fn sample(&mut self) -> Probed<Bytes> {
        let now = qubivex_hardware::process_rss(&self.host);
        if let Some(&b) = now.known() {
            self.sampled = Some(self.sampled.map_or(b, |s| s.max(b)));
        }
        now
    }

    fn finish(&mut self) -> (Probed<Bytes>, String) {
        match qubivex_hardware::process_peak_rss(&self.host) {
            Probed::Known(b) => (Probed::Known(b), "VmHWM".to_owned()),
            Probed::Unknown(why) => {
                self.sample();
                (
                    self.sampled
                        .map_or_else(|| Probed::Unknown(why), Probed::Known),
                    "sampled".to_owned(),
                )
            }
        }
    }
}

pub(crate) fn bench(a: &BenchArgs<'_>) -> Result<String, String> {
    if a.runs == 0 {
        return Err("--runs must be at least 1".into());
    }
    if a.decode_tokens == 0 {
        return Err("--decode-tokens must be at least 1".into());
    }
    let started = rfc3339(SystemTime::now());
    let host = Host::live();
    let hardware = Hardware {
        cpu: qubivex_hardware::probe_cpu(&host),
        memory: qubivex_hardware::probe_memory(&host),
        os: qubivex_hardware::probe_os(&host),
    };
    let reference = a.reference.map(read_reference).transpose()?;

    let meta = find(
        a.model,
        &roots(a.dirs),
        &ArchitectureRegistry::with_builtins(),
    )
    .map_err(|e| chain(&e))?;
    let model = ModelInfo {
        id: meta.manifest.model.id.clone(),
        name: meta.manifest.model.name.clone(),
        architecture: meta.manifest.architecture.id.clone(),
        dir: home_relative(&meta.dir),
    };
    let engine = Engine::load_mirrored(meta, a.memory.engine(a.context, a.threads), a.mirrors)
        .map_err(|e| chain(&e))?;
    let mut peak = PeakRss::new(host);
    let rss_after_load = peak.sample();

    let encoded = engine
        .tokenizer()
        .encode(a.prompt, true)
        .map_err(|e| chain(&e))?;
    if encoded.is_empty() {
        return Err("the prompt encodes to no tokens".into());
    }
    let prompt = prompt_ids(&encoded, a.prompt_tokens);
    if prompt.is_empty() {
        return Err("--prompt-tokens must be at least 1".into());
    }
    let context = engine.new_cache().capacity();
    if prompt.len() + a.decode_tokens > context {
        return Err(format!(
            "{} prompt + {} generated tokens exceed the context of {context}; pass --context",
            prompt.len(),
            a.decode_tokens
        ));
    }

    let progress = |what: &str| {
        if !a.json {
            eprintln!("qubivex bench: {what}");
        }
    };
    let mut first: Option<Vec<u32>> = None;
    let mut first_divergence = None;
    let mut check = |run: usize, tokens: &[u32]| match &first {
        None => first = Some(tokens.to_vec()),
        Some(f) => {
            if first_divergence.is_none() {
                first_divergence = divergence(f, tokens)
                    .or_else(|| (f.len() != tokens.len()).then_some(f.len().min(tokens.len())))
                    .map(|i| (run, i));
            }
        }
    };
    for i in 0..a.warmup {
        progress(&format!("warm-up {}/{}", i + 1, a.warmup));
        let s = one_run(&engine, &prompt, a.decode_tokens)?;
        check(i, &s.tokens);
        peak.sample();
    }
    let before = engine.store_stats();
    let prefetch_before = engine.prefetch_report();
    if a.routing_trace.is_some() {
        engine.record_routing(true);
    }
    let mut samples = Vec::with_capacity(a.runs);
    for i in 0..a.runs {
        progress(&format!("run {}/{}", i + 1, a.runs));
        let s = one_run(&engine, &prompt, a.decode_tokens)?;
        check(a.warmup + i, &s.tokens);
        peak.sample();
        samples.push(s);
    }
    let after = engine.store_stats();
    let prefetch_after = engine.prefetch_report();
    if let Some(path) = a.routing_trace {
        let trace = engine.take_routing();
        engine.record_routing(false);
        let f = std::fs::File::create(path)
            .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
        qubivex_expert_stream::replay::write_trace(&trace, std::io::BufWriter::new(f))
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }
    let (peak_rss, peak_source) = peak.finish();

    let prefill_ms: Vec<f64> = samples.iter().map(|s| ms(s.prefill)).collect();
    let all_steps: Vec<f64> = samples
        .iter()
        .flat_map(|s| s.steps.iter().copied().map(ms))
        .collect();
    let runs: Vec<RunSummary> = samples
        .iter()
        .map(|s| {
            let total: Duration = s.steps.iter().sum();
            #[allow(clippy::cast_precision_loss)]
            let tok_s = if s.steps.is_empty() {
                0.0
            } else {
                s.steps.len() as f64 / total.as_secs_f64().max(1e-12)
            };
            RunSummary {
                prefill_ms: ms(s.prefill),
                decode_step_ms: Stats::of(&s.steps.iter().copied().map(ms).collect::<Vec<_>>()),
                decode_tok_s: tok_s,
            }
        })
        .collect();
    let prefill = Stats::of(&prefill_ms).ok_or("no measured runs")?;
    let step_ms = Stats::of(&all_steps);
    let tokens = first.unwrap_or_default();
    let reference_divergence = reference.as_ref().and_then(|r| divergence(r, &tokens));
    #[allow(clippy::cast_precision_loss)]
    let record = Record {
        schema: SCHEMA,
        label: a.label.map(str::to_owned),
        started,
        build: Build::current(),
        hardware,
        model,
        config: Config {
            threads: engine.threads(),
            expert_cache: Bytes(engine.expert_cache_bytes()),
            expert_slots: engine.expert_pools().iter().map(|p| p.slots).collect(),
            cache_policy: Some(format!("{:?}", a.memory.cache_policy())),
            io_threads: Some(engine.io_threads()),
            expert_io: Some(engine.expert_io().name().to_owned()),
            verification: Some(
                match engine.verification() {
                    qubivex_runtime::Verification::Verified(_) => "verified",
                    qubivex_runtime::Verification::Off => "off",
                    qubivex_runtime::Verification::Missing => "missing",
                    qubivex_runtime::Verification::Stale(_) => "stale",
                }
                .to_owned(),
            ),
            prefetch: a.memory.prefetch().map(|p| PrefetchConfig {
                guard: p.guard,
                smooth: p.lookahead.smooth,
                confidence: p.lookahead.confidence,
                width: p.lookahead.width,
            }),
            device: {
                let p = engine.placement();
                (p.backend != qubivex_runtime::BackendKind::Cpu).then(|| DeviceConfig {
                    backend: p.backend.name().to_owned(),
                    device: p.device.clone(),
                    ops: p.ops.map(qubivex_runtime::MetalOps::describe),
                    tensors: p.tensors,
                    bytes: Bytes(p.bytes),
                    fallback: p.fallback.clone(),
                })
            },
            context,
            prompt: a.prompt.to_owned(),
            prompt_tokens: prompt.len(),
            generated_tokens: a.decode_tokens,
            warmup_runs: a.warmup,
            measured_runs: a.runs,
        },
        prefill: Prefill {
            tok_s_at_median: prompt.len() as f64 / (prefill.median / 1e3).max(1e-12),
            ms: prefill,
        },
        decode: Decode {
            tok_s_at_median: step_ms.as_ref().map_or(0.0, |s| 1e3 / s.median.max(1e-12)),
            step_ms,
            run_tok_s: runs.iter().map(|r| r.decode_tok_s).collect(),
        },
        memory: Memory {
            rss_after_load,
            peak_rss,
            peak_source,
            plan_budget: engine.memory_plan().map(|p| p.ram.budget.bytes),
            ledger: {
                let l = engine.ledger().snapshot(qubivex_memory::Tier::Ram);
                Ledger {
                    dense: l.dense,
                    expert_cache: l.expert_cache,
                    kv: l.kv,
                    total: l.total(),
                }
            },
        },
        experts: Experts {
            hits: after.hits - before.hits,
            loads: after.loads - before.loads,
            bytes_read: Bytes(after.bytes_read - before.bytes_read),
            evictions: after.evictions - before.evictions,
            hit_rate: {
                let (h, l) = (after.hits - before.hits, after.loads - before.loads);
                h as f64 / ((h + l) as f64).max(1.0)
            },
            bytes_per_token: Bytes(
                (after.bytes_read - before.bytes_read) / (a.runs * a.decode_tokens).max(1) as u64,
            ),
            slab_allocs: after.slab_allocs - before.slab_allocs,
            wait_ms_per_token: (after.wait_ns - before.wait_ns) as f64
                / 1e6
                / (a.runs * a.decode_tokens).max(1) as f64,
            read_retries: after.retried - before.retried,
            read_timeouts: after.timeouts - before.timeouts,
            prefetch: prefetch_before
                .zip(prefetch_after)
                .map(|(b, a)| prefetch_record(b, a)),
        },
        check: Check {
            deterministic: first_divergence.is_none(),
            first_divergence,
            reference: a.reference.map(Path::to_path_buf),
            reference_match: reference.as_ref().map(|_| reference_divergence.is_none()),
            reference_divergence,
        },
        runs,
        tokens,
    };

    let json = crate::to_json(&record)?;
    if let Some(out) = a.out {
        std::fs::write(out, &json).map_err(|e| format!("cannot write {}: {e}", out.display()))?;
    }
    if !record.check.deterministic {
        // Still report, but fail: a nondeterministic engine is a bug, not a timing.
        let text = if a.json { json } else { render(&record) };
        crate::emit(&text)?;
        return Err(format!(
            "tokens differ between runs (run, token) = {:?}",
            record.check.first_divergence
        ));
    }
    Ok(if a.json { json } else { render(&record) })
}

fn known<T: std::fmt::Display>(p: &Probed<T>) -> String {
    p.known()
        .map_or_else(|| "unknown".to_owned(), ToString::to_string)
}

fn render(r: &Record) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let b = &r.build;
    let c = &r.config;
    let _ = writeln!(
        s,
        "qubivex bench {}{}",
        r.started,
        r.label
            .as_ref()
            .map_or(String::new(), |l| format!(" [{l}]"))
    );
    let _ = writeln!(
        s,
        "  build     {} {} ({}, opt-level {}{}), {}",
        b.version,
        b.git,
        b.profile,
        b.opt_level,
        if b.debug_assertions {
            ", debug assertions"
        } else {
            ""
        },
        b.rustc
    );
    if !b.rustflags.is_empty() {
        let _ = writeln!(s, "  rustflags {}", b.rustflags);
    }
    let _ = writeln!(
        s,
        "  hardware  {}, {} RAM, {}",
        known(&r.hardware.cpu.model),
        known(&r.hardware.memory.total),
        known(&r.hardware.os.name)
    );
    let _ = writeln!(s, "  model     {} ({})", r.model.id, r.model.architecture);
    let _ = writeln!(
        s,
        "  config    {} threads, expert cache {}, {} prompt + {} generated tokens, {} warm-up + {} measured runs",
        c.threads,
        c.expert_cache,
        c.prompt_tokens,
        c.generated_tokens,
        c.warmup_runs,
        c.measured_runs
    );
    let p = &r.prefill.ms;
    let _ = writeln!(
        s,
        "  prefill   median {:.1} ms, P95 {:.1} ms ({:.1} tok/s at median)",
        p.median, p.p95, r.prefill.tok_s_at_median
    );
    if let Some(d) = &r.decode.step_ms {
        let worst = r
            .decode
            .run_tok_s
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let _ = writeln!(
            s,
            "  decode    step median {:.2} ms, P95 {:.2} ms ({:.2} tok/s at median; slowest run {worst:.2} tok/s)",
            d.median, d.p95, r.decode.tok_s_at_median
        );
    }
    let _ = writeln!(
        s,
        "  memory    RSS after load {}, peak {} ({})",
        known(&r.memory.rss_after_load),
        known(&r.memory.peak_rss),
        r.memory.peak_source
    );
    let l = &r.memory.ledger;
    let _ = writeln!(
        s,
        "            ledger {} (dense {}, experts {}, KV {}), plan budget {}",
        l.total,
        l.dense,
        l.expert_cache,
        l.kv,
        r.memory
            .plan_budget
            .map_or_else(|| "none (fixed cache)".to_owned(), |b| b.to_string())
    );
    let e = &r.experts;
    let _ = writeln!(
        s,
        "  experts   {} hits, {} loads ({}), {} evictions during measured runs",
        e.hits, e.loads, e.bytes_read, e.evictions
    );
    let _ = writeln!(
        s,
        "            hit rate {:.1}%, {} read and {:.2} ms blocked per token, {} slab allocations; {} pools of {}, {} I/O threads, {} reads",
        e.hit_rate * 100.0,
        e.bytes_per_token,
        e.wait_ms_per_token,
        e.slab_allocs,
        r.config.cache_policy.as_deref().unwrap_or("?"),
        slot_runs(&r.config.expert_slots),
        r.config
            .io_threads
            .map_or_else(|| "?".to_owned(), |n| n.to_string()),
        r.config.expert_io.as_deref().unwrap_or("buffered")
    );
    if let Some(p) = &e.prefetch {
        let _ = writeln!(
            s,
            "  prefetch  recall {:.1}%, precision {:.1}% ({:.1} predicted per layer); {} issued, {} present, {} refused; {} ready + {} late = {:.1}% useful, lead {:.2} ms; {} wasted ({}), {} cancelled",
            p.recall * 100.0,
            p.precision * 100.0,
            p.predicted_per_layer,
            p.issued,
            p.present,
            p.refused,
            p.ready,
            p.late,
            p.useful_rate * 100.0,
            p.lead_ms,
            p.wasted,
            p.wasted_bytes,
            p.cancelled
        );
    }
    let k = &r.check;
    let reference = match (k.reference_match, k.reference_divergence) {
        (None, _) => String::new(),
        (Some(true), _) => ", matches the reference".to_owned(),
        (Some(false), i) => format!(
            ", DIFFERS from the reference at token {}",
            i.unwrap_or_default()
        ),
    };
    let _ = writeln!(
        s,
        "  check     tokens {}{reference}",
        if k.deterministic {
            "identical across runs"
        } else {
            "DIFFER between runs"
        }
    );
    s
}

/// Slot counts as runs of layers, for example `5 layers × 3, 11 layers × 2`.
fn slot_runs(slots: &[u32]) -> String {
    let mut runs: Vec<(usize, u32)> = Vec::new();
    for &v in slots {
        match runs.last_mut() {
            Some((n, x)) if *x == v => *n += 1,
            _ => runs.push((1, v)),
        }
    }
    runs.iter()
        .map(|(n, v)| format!("{n} layer{} × {v}", if *n == 1 { "" } else { "s" }))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_percentiles() {
        let v: Vec<f64> = (1..=20).map(f64::from).collect();
        let s = Stats::of(&v).unwrap();
        assert_eq!(
            (s.n, s.min.to_bits(), s.max.to_bits()),
            (20, 1f64.to_bits(), 20f64.to_bits())
        );
        assert_eq!(s.median.to_bits(), 10f64.to_bits());
        assert_eq!(s.p95.to_bits(), 19f64.to_bits());
        let one = Stats::of(&[3.0]).unwrap();
        assert_eq!(
            (one.median.to_bits(), one.p95.to_bits()),
            (3f64.to_bits(), 3f64.to_bits())
        );
        assert!(Stats::of(&[]).is_none());
    }

    #[test]
    fn prompt_repeats_to_length() {
        assert_eq!(prompt_ids(&[1, 2, 3], Some(7)), [1, 2, 3, 1, 2, 3, 1]);
        assert_eq!(prompt_ids(&[1, 2, 3], Some(2)), [1, 2]);
        assert_eq!(prompt_ids(&[1, 2, 3], None), [1, 2, 3]);
    }

    #[test]
    fn divergence_and_greedy() {
        assert_eq!(divergence(&[1, 2, 3], &[1, 2, 3, 4]), None);
        assert_eq!(divergence(&[1, 2, 3], &[1, 5, 3]), Some(1));
        assert_eq!(argmax(&[0.0, 2.0, 2.0, -1.0]), 1, "lowest id on ties");
    }

    #[test]
    fn utc_timestamps() {
        let t = |s| rfc3339(UNIX_EPOCH + Duration::from_secs(s));
        assert_eq!(t(0), "1970-01-01T00:00:00Z");
        assert_eq!(t(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(t(1_793_712_245), "2026-11-03T13:24:05Z");
    }

    #[test]
    fn references_read_records_and_arrays() {
        let dir = std::env::temp_dir().join(format!("qubivex-bench-ref-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.json");
        std::fs::write(&a, "[5, 6]").unwrap();
        assert_eq!(read_reference(&a).unwrap(), [5, 6]);
        let b = dir.join("b.json");
        std::fs::write(&b, r#"{"schema": 1, "tokens": [7]}"#).unwrap();
        assert_eq!(read_reference(&b).unwrap(), [7]);
        std::fs::write(&b, r#"{"x": 1}"#).unwrap();
        assert!(read_reference(&b).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
