//! Spec §8 over real HTTP, on the tiny fixtures: `/metrics` reports the inference,
//! memory and serving numbers for a model that has generated, and the log is one JSON
//! object per event with request, model and sequence ids and no prompt or generated
//! text.

#![allow(clippy::unwrap_used)]

mod common;

use std::io::Write;
use std::sync::{Arc, Mutex};

use common::{engine, http, post, start, text_of, wait_until};
use qubivex_api::{Gateway, LogFormat, ServerConfig};
use serde_json::{Value, json};

/// A log sink the test can read back.
#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl Write for Captured {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Captured {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }
    fn events(&self) -> Vec<Value> {
        self.text()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{e}: {l}")))
            .collect()
    }
}

/// The value of the one series of `name` whose labels contain `labels`.
fn series(metrics: &str, name: &str, labels: &str) -> f64 {
    let found: Vec<&str> = metrics
        .lines()
        .filter(|l| {
            l.split_once(['{', ' '])
                .is_some_and(|(n, rest)| n == name && rest.contains(labels))
        })
        .collect();
    assert_eq!(found.len(), 1, "{name} with {labels}: {found:?}");
    found[0].rsplit_once(' ').unwrap().1.parse().unwrap()
}

const PROMPT: &str = "zebra quokka xylophone";
const SECRET: &str = "marmalade narwhal";

#[test]
fn metrics_cover_a_generation_and_logs_carry_ids_but_no_text() {
    let log = Captured::default();
    let cfg = ServerConfig {
        log_format: LogFormat::Json,
        ..ServerConfig::default()
    };
    let gw = Gateway::with_log_writer(cfg, Box::new(log.clone()));
    gw.add_model("olmoe-tiny", engine("olmoe-tiny")).unwrap();
    let s = start(&gw);

    let body =
        json!({ "model": "olmoe-tiny", "prompt": PROMPT, "max_tokens": 12, "temperature": 0 });
    let r = http(
        s.addr,
        "POST",
        "/v1/completions",
        &[("x-request-id", "obs-1")],
        Some(&body),
    );
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    let generated = text_of(&r.json());
    let completion = r.json()["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(completion >= 2, "the fixture generates several tokens");

    let chat = json!({
        "model": "olmoe-tiny", "max_tokens": 8, "temperature": 0, "stream": true,
        "messages": [{ "role": "user", "content": SECRET }],
    });
    let r = http(
        s.addr,
        "POST",
        "/v1/chat/completions",
        &[("x-request-id", "obs-2")],
        Some(&chat),
    );
    assert_eq!(r.status, 200);
    // A request that fails is logged with its error code and no more.
    let missing = json!({ "model": "nope", "prompt": SECRET });
    assert_eq!(post(s.addr, "/v1/completions", &missing).status, 404);
    wait_until("both generations to end", || {
        gw.metrics().generations("olmoe-tiny") == 2
    });

    let m = String::from_utf8(http(s.addr, "GET", "/metrics", &[], None).body).unwrap();
    let model = "model=\"olmoe-tiny\"";
    // Inference: one observation per generation, and one gap per token after the first.
    for name in [
        "qubivex_queue_wait_seconds_count",
        "qubivex_time_to_first_token_seconds_count",
        "qubivex_generation_duration_seconds_count",
        "qubivex_prompt_tokens_per_second_count",
        "qubivex_generation_tokens_per_second_count",
    ] {
        assert!((series(&m, name, model) - 2.0).abs() < 0.5, "{name}\n{m}");
    }
    let generated_tokens = series(&m, "qubivex_generated_tokens_total", "");
    assert!(
        (series(&m, "qubivex_inter_token_seconds_count", model) - (generated_tokens - 2.0)).abs()
            < 0.5
    );
    assert!(
        (series(
            &m,
            "qubivex_generation_duration_seconds_bucket",
            "model=\"olmoe-tiny\",le=\"+Inf\""
        ) - 2.0)
            .abs()
            < 0.5
    );
    assert!(series(&m, "qubivex_generation_duration_seconds_sum", model) > 0.0);
    assert!(series(&m, "qubivex_model_load_seconds_count", model) < 0.5);
    // Memory and serving: every per-model value is there, and the ones a generation
    // must have moved have moved.
    for name in [
        "qubivex_kv_reserved_positions",
        "qubivex_expert_cache_bytes",
        "qubivex_generation_slots_free",
        "qubivex_queue_depth",
        "qubivex_queue_rejected_total",
        "qubivex_prefix_cache_lookups_total",
        "qubivex_expert_cache_hits_total",
        "qubivex_expert_cache_evictions_total",
        "qubivex_expert_load_wait_seconds_total",
        "qubivex_batch_steps_total",
        "qubivex_memory_kv_bytes",
        "qubivex_memory_activations_bytes",
        "qubivex_memory_budget_bytes",
    ] {
        assert!(series(&m, name, model) >= 0.0, "{name}");
    }
    for name in [
        "qubivex_queue_granted_total",
        "qubivex_expert_cache_loads_total",
        "qubivex_expert_read_bytes_total",
        "qubivex_batch_passes_total",
        "qubivex_memory_dense_bytes",
        "qubivex_memory_expert_cache_bytes",
    ] {
        assert!(series(&m, name, model) > 0.0, "{name}");
    }
    assert!(
        series(
            &m,
            "qubivex_http_request_duration_seconds_count",
            "endpoint=\"completions\""
        ) >= 2.0
    );
    assert!(
        series(&m, "qubivex_process_peak_resident_bytes", "")
            >= series(&m, "qubivex_process_resident_bytes", "")
    );
    assert!(series(&m, "qubivex_process_resident_bytes", "") > 0.0);

    // The log: JSON lines, ids on everything about a generation, and no text.
    let text = log.text();
    for private in [PROMPT, "quokka", SECRET, "narwhal"] {
        assert!(
            !text.contains(private),
            "the log holds {private:?}:\n{text}"
        );
    }
    if generated.trim().len() >= 4 {
        assert!(!text.contains(generated.trim()), "the log holds the output");
    }
    let events = log.events();
    let of = |event: &str, rid: &str| -> Value {
        let found: Vec<&Value> = events
            .iter()
            .filter(|e| e["event"] == event && e["request_id"] == rid)
            .collect();
        assert_eq!(found.len(), 1, "{event} for {rid}:\n{text}");
        found[0].clone()
    };
    for e in &events {
        assert!(e["ts"].as_str().is_some_and(|t| t.ends_with('Z')), "{e}");
        assert!(e["level"].is_string() && e["event"].is_string(), "{e}");
    }
    assert_eq!(events[0]["event"], "model_added");
    assert_eq!(events[0]["model"], "olmoe-tiny");
    let started = of("generation_started", "obs-1");
    let finished = of("generation_finished", "obs-1");
    assert_eq!(started["model"], "olmoe-tiny");
    assert_eq!(started["stream"], false);
    assert_eq!(started["sequence_id"], finished["sequence_id"]);
    assert_eq!(finished["completion_tokens"], completion);
    assert!(finished["ttft_ms"].as_f64().unwrap() > 0.0);
    assert!(finished["duration_ms"].as_f64().unwrap() >= finished["ttft_ms"].as_f64().unwrap());
    assert!(["length", "eos"].contains(&finished["finish"].as_str().unwrap()));
    let request = of("request", "obs-1");
    assert_eq!(
        (&request["endpoint"], &request["status"]),
        (&json!("completions"), &json!(200))
    );
    let second = of("generation_finished", "obs-2");
    assert_eq!(second["model"], "olmoe-tiny");
    assert_ne!(second["sequence_id"], finished["sequence_id"]);
    assert_eq!(of("generation_started", "obs-2")["stream"], true);
    let refused: Vec<&Value> = events
        .iter()
        .filter(|e| e["event"] == "request" && e["status"] == 404)
        .collect();
    assert_eq!(refused.len(), 1);
    assert_eq!(refused[0]["error"], "model_not_found");
}

#[test]
fn metrics_can_be_switched_off_and_the_log_too() {
    let log = Captured::default();
    let cfg = ServerConfig {
        log_format: LogFormat::Off,
        metrics_enabled: false,
        ..ServerConfig::default()
    };
    let gw = Gateway::with_log_writer(cfg, Box::new(log.clone()));
    gw.add_model("dense-tiny", engine("dense-tiny")).unwrap();
    let s = start(&gw);
    assert_eq!(http(s.addr, "GET", "/metrics", &[], None).status, 404);
    assert_eq!(http(s.addr, "GET", "/health", &[], None).status, 200);
    let body = json!({ "model": "dense-tiny", "prompt": "Hello", "max_tokens": 2 });
    assert_eq!(post(s.addr, "/v1/completions", &body).status, 200);
    drop(s);
    assert_eq!(log.text(), "");
}
