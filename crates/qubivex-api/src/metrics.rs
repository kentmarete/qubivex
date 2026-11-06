//! Counters for `/metrics`, in the Prometheus text format. Plain atomics: the gateway
//! counts requests and tokens, and reads the engines' own state when scraped.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Gateway counters. Every counter is `Relaxed`: they are statistics only.
#[derive(Debug, Default)]
pub struct Metrics {
    /// Requests by (endpoint, status).
    requests: Mutex<BTreeMap<(&'static str, u16), u64>>,
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

    /// The gateway counters in the Prometheus text format.
    pub(crate) fn render(&self, out: &mut String) {
        let _ = writeln!(out, "# TYPE qubivex_requests_total counter");
        if let Ok(m) = self.requests.lock() {
            for ((endpoint, status), n) in m.iter() {
                let _ = writeln!(
                    out,
                    "qubivex_requests_total{{endpoint=\"{endpoint}\",status=\"{status}\"}} {n}"
                );
            }
        }
        let gauge = |out: &mut String, name: &str, kind: &str, v: u64| {
            let _ = writeln!(out, "# TYPE {name} {kind}");
            let _ = writeln!(out, "{name} {v}");
        };
        gauge(
            out,
            "qubivex_generations_active",
            "gauge",
            self.active() as u64,
        );
        let r = |c: &AtomicU64| c.load(Ordering::Relaxed);
        gauge(
            out,
            "qubivex_prompt_tokens_total",
            "counter",
            r(&self.prompt_tokens),
        );
        gauge(
            out,
            "qubivex_generated_tokens_total",
            "counter",
            r(&self.tokens),
        );
        gauge(
            out,
            "qubivex_generations_cancelled_total",
            "counter",
            r(&self.cancelled),
        );
        gauge(
            out,
            "qubivex_generations_timed_out_total",
            "counter",
            r(&self.timeouts),
        );
        gauge(
            out,
            "qubivex_generations_failed_total",
            "counter",
            r(&self.errors),
        );
    }
}
