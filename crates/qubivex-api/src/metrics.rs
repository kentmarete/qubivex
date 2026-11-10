//! What `/metrics` reports, in the Prometheus text format (spec §8). Plain atomics: the
//! gateway counts requests and tokens and times each generation into fixed-bucket
//! histograms, and reads the engines' own counters when scraped.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Bucket bounds in seconds for waits and whole requests: 5 ms to 10 minutes.
const SECONDS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
];
/// Bucket bounds in seconds for the gap between two tokens: 1 ms to 5 s.
const TOKEN_GAP: &[f64] = &[
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
];
/// Bucket bounds in tokens per second.
const RATE: &[f64] = &[
    1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0,
    2000.0, 5000.0,
];

/// A label value as the text format wants it.
fn escape(v: &str) -> String {
    v.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// A histogram with fixed bucket bounds. Observations are `Relaxed`: statistics only.
#[derive(Debug)]
pub(crate) struct Histogram {
    bounds: &'static [f64],
    /// One count per bound, then one for everything above the last.
    counts: Box<[AtomicU64]>,
    /// The sum of what was observed, as `f64` bits.
    sum: AtomicU64,
}

impl Histogram {
    fn new(bounds: &'static [f64]) -> Self {
        Self {
            bounds,
            counts: (0..=bounds.len()).map(|_| AtomicU64::new(0)).collect(),
            sum: AtomicU64::new(0f64.to_bits()),
        }
    }

    /// Counts `v`. A value that is not a finite, non-negative number is dropped.
    pub(crate) fn observe(&self, v: f64) {
        if !v.is_finite() || v < 0.0 {
            return;
        }
        let i = self.bounds.partition_point(|b| *b < v);
        if let Some(c) = self.counts.get(i) {
            c.fetch_add(1, Ordering::Relaxed);
        }
        let mut sum = self.sum.load(Ordering::Relaxed);
        loop {
            let next = (f64::from_bits(sum) + v).to_bits();
            match self
                .sum
                .compare_exchange_weak(sum, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(now) => sum = now,
            }
        }
    }

    /// Counts a duration, in seconds.
    pub(crate) fn observe_duration(&self, d: Duration) {
        self.observe(d.as_secs_f64());
    }

    /// Observations so far.
    pub(crate) fn count(&self) -> u64 {
        self.counts.iter().map(|c| c.load(Ordering::Relaxed)).sum()
    }

    /// The series of this histogram under `name`, with `labels` (`key="value"`, already
    /// escaped) on each.
    fn render(&self, out: &mut String, name: &str, labels: &str) {
        let mut seen = 0;
        for (bound, c) in self.bounds.iter().zip(&self.counts) {
            seen += c.load(Ordering::Relaxed);
            let _ = writeln!(out, "{name}_bucket{{{labels},le=\"{bound}\"}} {seen}");
        }
        let total = self.count();
        let sum = f64::from_bits(self.sum.load(Ordering::Relaxed));
        let _ = writeln!(out, "{name}_bucket{{{labels},le=\"+Inf\"}} {total}");
        let _ = writeln!(out, "{name}_sum{{{labels}}} {sum}");
        let _ = writeln!(out, "{name}_count{{{labels}}} {total}");
    }
}

/// One model's timings.
#[derive(Debug)]
pub(crate) struct ModelMetrics {
    /// From a request's arrival to its grant of a slot and memory.
    pub(crate) queue_wait: Histogram,
    /// From a request's arrival to its first generated token.
    pub(crate) first_token: Histogram,
    /// From a request's arrival to the end of its generation.
    pub(crate) duration: Histogram,
    /// Between one generated token and the next.
    pub(crate) token_gap: Histogram,
    /// Prompt tokens run per second of prefill.
    pub(crate) prompt_rate: Histogram,
    /// Tokens generated per second, after the first.
    pub(crate) generation_rate: Histogram,
    /// How long a load took.
    pub(crate) load: Histogram,
}

impl ModelMetrics {
    fn new() -> Self {
        Self {
            queue_wait: Histogram::new(SECONDS),
            first_token: Histogram::new(SECONDS),
            duration: Histogram::new(SECONDS),
            token_gap: Histogram::new(TOKEN_GAP),
            prompt_rate: Histogram::new(RATE),
            generation_rate: Histogram::new(RATE),
            load: Histogram::new(SECONDS),
        }
    }
}

type Pick = fn(&ModelMetrics) -> &Histogram;

/// The per-model histograms: name, help and which one.
const MODEL_HISTOGRAMS: &[(&str, &str, Pick)] = &[
    (
        "qubivex_queue_wait_seconds",
        "Scheduler wait: from a request's arrival to its grant of a slot and memory.",
        |m| &m.queue_wait,
    ),
    (
        "qubivex_time_to_first_token_seconds",
        "From a request's arrival to its first generated token.",
        |m| &m.first_token,
    ),
    (
        "qubivex_generation_duration_seconds",
        "Total request latency: from arrival to the end of the generation, however it ended.",
        |m| &m.duration,
    ),
    (
        "qubivex_inter_token_seconds",
        "Time between one generated token and the next.",
        |m| &m.token_gap,
    ),
    (
        "qubivex_prompt_tokens_per_second",
        "Prompt tokens run (not reused from retained KV) per second, up to the first token.",
        |m| &m.prompt_rate,
    ),
    (
        "qubivex_generation_tokens_per_second",
        "Tokens generated per second after the first, per generation of two tokens or more.",
        |m| &m.generation_rate,
    ),
    (
        "qubivex_model_load_seconds",
        "How long a model load through the admin endpoint took.",
        |m| &m.load,
    ),
];

/// What `/metrics` reads from each loaded model's engine and scheduler when scraped:
/// name, type and help. The gateway supplies a value per name ([`render_models`]).
pub(crate) const MODEL_VALUES: &[(&str, &str, &str)] = &[
    (
        "qubivex_kv_reserved_positions",
        "gauge",
        "KV positions reserved by live conversations.",
    ),
    (
        "qubivex_expert_cache_bytes",
        "gauge",
        "Expert cache capacity (the planner lowers it as the KV grows).",
    ),
    (
        "qubivex_generation_slots_free",
        "gauge",
        "Generations that could start now.",
    ),
    ("qubivex_queue_depth", "gauge", "Requests waiting to start."),
    (
        "qubivex_queue_rejected_total",
        "counter",
        "Requests refused because the queue was full.",
    ),
    (
        "qubivex_queue_granted_total",
        "counter",
        "Requests granted a slot and memory.",
    ),
    (
        "qubivex_prefix_cache_lookups_total",
        "counter",
        "Retained-KV lookups.",
    ),
    (
        "qubivex_prefix_cache_hits_total",
        "counter",
        "Retained-KV lookups that found a prefix.",
    ),
    (
        "qubivex_prefix_cache_tokens_reused_total",
        "counter",
        "Prompt tokens that did not have to run.",
    ),
    (
        "qubivex_prefix_cache_evictions_total",
        "counter",
        "Retained KV dropped for room.",
    ),
    (
        "qubivex_prefix_cache_positions",
        "gauge",
        "KV positions retained now.",
    ),
    (
        "qubivex_expert_cache_hits_total",
        "counter",
        "Expert acquires served from memory.",
    ),
    (
        "qubivex_expert_cache_loads_total",
        "counter",
        "Expert reads completed (cache misses and prefetches).",
    ),
    (
        "qubivex_expert_cache_evictions_total",
        "counter",
        "Experts evicted.",
    ),
    (
        "qubivex_expert_read_bytes_total",
        "counter",
        "Bytes of experts read from disk; its rate is the disk read throughput.",
    ),
    (
        "qubivex_expert_load_wait_seconds_total",
        "counter",
        "Time generations spent blocked on expert reads; over loads, the expert load latency.",
    ),
    (
        "qubivex_expert_read_retries_total",
        "counter",
        "Failed expert reads attempted again.",
    ),
    (
        "qubivex_expert_read_timeouts_total",
        "counter",
        "Waits for an expert that timed out.",
    ),
    (
        "qubivex_batch_passes_total",
        "counter",
        "Forward passes run through the combining point.",
    ),
    (
        "qubivex_batch_steps_total",
        "counter",
        "Sequence steps those passes ran; over passes, the mean batch width.",
    ),
    (
        "qubivex_memory_dense_bytes",
        "gauge",
        "Host memory of the weights that stay resident.",
    ),
    (
        "qubivex_memory_expert_cache_bytes",
        "gauge",
        "Host memory of the experts cached now.",
    ),
    (
        "qubivex_memory_kv_bytes",
        "gauge",
        "Host memory of KV caches, live and retained.",
    ),
    (
        "qubivex_memory_activations_bytes",
        "gauge",
        "Host memory of activations and scratch.",
    ),
    (
        "qubivex_memory_budget_bytes",
        "gauge",
        "The memory plan's host budget (0 without a plan).",
    ),
];

/// One loaded model's values for [`MODEL_VALUES`], by name.
pub(crate) type ModelValues = Vec<(&'static str, f64)>;

/// Writes [`MODEL_VALUES`] for `models`, one series per model.
pub(crate) fn render_models(out: &mut String, models: &[(String, ModelValues)]) {
    for (name, kind, help) in MODEL_VALUES {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} {kind}");
        for (id, values) in models {
            if let Some((_, v)) = values.iter().find(|(n, _)| n == name) {
                let _ = writeln!(out, "{name}{{model=\"{}\"}} {v}", escape(id));
            }
        }
    }
}

/// Writes one unlabelled value.
pub(crate) fn render_value(out: &mut String, name: &str, kind: &str, help: &str, v: f64) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} {kind}");
    let _ = writeln!(out, "{name} {v}");
}

/// Gateway counters. Every counter is `Relaxed`: they are statistics only.
#[derive(Debug, Default)]
pub struct Metrics {
    /// Requests by (endpoint, status).
    requests: Mutex<BTreeMap<(&'static str, u16), u64>>,
    /// How long requests took to answer, by endpoint.
    answered: Mutex<BTreeMap<&'static str, Histogram>>,
    /// Timings by model id. An entry is made when a model is first loaded or used, so
    /// only ids the server has served appear.
    models: Mutex<BTreeMap<String, Arc<ModelMetrics>>>,
    /// The highest resident set size a scrape has seen.
    peak_rss: AtomicU64,
    /// Generations running now.
    pub(crate) active: AtomicUsize,
    /// Prompt tokens of admitted generations.
    pub(crate) prompt_tokens: AtomicU64,
    /// Tokens generated.
    pub(crate) tokens: AtomicU64,
    /// Generations stopped because the client went away or the request was abandoned.
    pub(crate) cancelled: AtomicU64,
    /// Generations cut short by the request timeout.
    pub(crate) timeouts: AtomicU64,
    /// Generations that failed after admission.
    pub(crate) errors: AtomicU64,
    /// Model loads started.
    model_loads: AtomicU64,
}

impl Metrics {
    /// Counts one finished request.
    pub(crate) fn request(&self, endpoint: &'static str, status: u16) {
        // A poisoned lock only loses a statistic.
        if let Ok(mut m) = self.requests.lock() {
            *m.entry((endpoint, status)).or_default() += 1;
        }
    }

    /// Counts how long a request to `endpoint` took to answer.
    pub(crate) fn answered(&self, endpoint: &'static str, took: Duration) {
        if let Ok(mut m) = self.answered.lock() {
            m.entry(endpoint)
                .or_insert_with(|| Histogram::new(SECONDS))
                .observe_duration(took);
        }
    }

    /// The timings of model `id`.
    pub(crate) fn model(&self, id: &str) -> Arc<ModelMetrics> {
        let mut m = self
            .models
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(found) = m.get(id) {
            return Arc::clone(found);
        }
        let made = Arc::new(ModelMetrics::new());
        m.insert(id.to_owned(), Arc::clone(&made));
        made
    }

    /// Generations of model `id` that have ended, however they ended.
    #[must_use]
    pub fn generations(&self, id: &str) -> u64 {
        self.models
            .lock()
            .ok()
            .and_then(|m| m.get(id).map(|m| m.duration.count()))
            .unwrap_or(0)
    }

    /// Records the resident set size a scrape read, and returns the highest so far.
    pub(crate) fn note_rss(&self, bytes: u64) -> u64 {
        self.peak_rss.fetch_max(bytes, Ordering::Relaxed).max(bytes)
    }

    /// Generations running now.
    #[must_use]
    pub fn active(&self) -> usize {
        self.active.load(Ordering::Relaxed)
    }

    /// Generations cancelled so far.
    #[must_use]
    pub fn cancelled(&self) -> u64 {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Counts a model load starting.
    pub(crate) fn count_load(&self) {
        self.model_loads.fetch_add(1, Ordering::Relaxed);
    }

    /// Model loads started so far.
    #[must_use]
    pub fn model_loads(&self) -> u64 {
        self.model_loads.load(Ordering::Relaxed)
    }

    /// Requests counted for `endpoint` with `status`.
    #[must_use]
    pub fn requests(&self, endpoint: &str, status: u16) -> u64 {
        self.requests.lock().map_or(0, |m| {
            m.iter()
                .filter(|((e, s), _)| *e == endpoint && *s == status)
                .map(|(_, n)| *n)
                .sum()
        })
    }

    /// The gateway's own counters and histograms in the Prometheus text format.
    pub(crate) fn render(&self, out: &mut String) {
        let _ = writeln!(
            out,
            "# HELP qubivex_requests_total Requests answered, by endpoint and status."
        );
        let _ = writeln!(out, "# TYPE qubivex_requests_total counter");
        if let Ok(m) = self.requests.lock() {
            for ((endpoint, status), n) in m.iter() {
                let _ = writeln!(
                    out,
                    "qubivex_requests_total{{endpoint=\"{endpoint}\",status=\"{status}\"}} {n}"
                );
            }
        }
        let name = "qubivex_http_request_duration_seconds";
        let _ = writeln!(
            out,
            "# HELP {name} Time to answer a request (for a stream, to its first byte)."
        );
        let _ = writeln!(out, "# TYPE {name} histogram");
        if let Ok(m) = self.answered.lock() {
            for (endpoint, h) in m.iter() {
                h.render(out, name, &format!("endpoint=\"{endpoint}\""));
            }
        }
        let r = |c: &AtomicU64| c.load(Ordering::Relaxed);
        for (name, kind, help, v) in [
            (
                "qubivex_generations_active",
                "gauge",
                "Generations running now.",
                self.active() as u64,
            ),
            (
                "qubivex_prompt_tokens_total",
                "counter",
                "Prompt tokens of admitted generations.",
                r(&self.prompt_tokens),
            ),
            (
                "qubivex_generated_tokens_total",
                "counter",
                "Tokens generated; its rate is the aggregate tokens per second.",
                r(&self.tokens),
            ),
            (
                "qubivex_generations_cancelled_total",
                "counter",
                "Generations stopped because the client went away or the request was abandoned.",
                r(&self.cancelled),
            ),
            (
                "qubivex_generations_timed_out_total",
                "counter",
                "Generations cut short by the request timeout.",
                r(&self.timeouts),
            ),
            (
                "qubivex_generations_failed_total",
                "counter",
                "Generations that failed after admission.",
                r(&self.errors),
            ),
            (
                "qubivex_model_loads_total",
                "counter",
                "Model loads started through the admin endpoint.",
                r(&self.model_loads),
            ),
        ] {
            let _ = writeln!(out, "# HELP {name} {help}");
            let _ = writeln!(out, "# TYPE {name} {kind}");
            let _ = writeln!(out, "{name} {v}");
        }
        let models: Vec<(String, Arc<ModelMetrics>)> = self.models.lock().map_or_else(
            |_| Vec::new(),
            |m| {
                m.iter()
                    .map(|(id, m)| (id.clone(), Arc::clone(m)))
                    .collect()
            },
        );
        for (name, help, pick) in MODEL_HISTOGRAMS {
            let _ = writeln!(out, "# HELP {name} {help}");
            let _ = writeln!(out, "# TYPE {name} histogram");
            for (id, m) in &models {
                pick(m).render(out, name, &format!("model=\"{}\"", escape(id)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{Histogram, Metrics, SECONDS, escape};

    #[test]
    fn a_histogram_renders_cumulative_buckets_a_sum_and_a_count() {
        let h = Histogram::new(&[0.5, 1.0]);
        for v in [0.25, 0.5, 0.75, 3.0, f64::NAN, -1.0] {
            h.observe(v);
        }
        let mut out = String::new();
        h.render(&mut out, "x_seconds", "model=\"m\"");
        assert_eq!(
            out,
            "x_seconds_bucket{model=\"m\",le=\"0.5\"} 2\n\
             x_seconds_bucket{model=\"m\",le=\"1\"} 3\n\
             x_seconds_bucket{model=\"m\",le=\"+Inf\"} 4\n\
             x_seconds_sum{model=\"m\"} 4.5\n\
             x_seconds_count{model=\"m\"} 4\n"
        );
    }

    #[test]
    fn bounds_rise() {
        for b in [SECONDS, super::TOKEN_GAP, super::RATE] {
            assert!(b.windows(2).all(|w| w[0] < w[1]));
        }
    }

    #[test]
    fn a_model_id_cannot_break_out_of_its_label() {
        assert_eq!(escape("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
        let m = Metrics::default();
        m.model("odd\"id")
            .duration
            .observe_duration(Duration::from_millis(20));
        let mut out = String::new();
        m.render(&mut out);
        assert!(out.contains("qubivex_generation_duration_seconds_count{model=\"odd\\\"id\"} 1"));
        assert_eq!(m.generations("odd\"id"), 1);
    }
}
