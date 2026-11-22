//! Context management over real HTTP (ADR-021), on a fixture model with a context of
//! 256 positions: requests that fit are answered exactly as without it; longer ones are
//! prepared to fit, with headers saying so, or refused with the reason.

#![allow(clippy::unwrap_used)]

mod common;

use std::io::Write;
use std::sync::{Arc, Mutex};

use common::{engine, http, post, start, text_of};
use qubivex_api::{ContextConfig, ContextSettings, Gateway, LogFormat, ServerConfig};
use serde_json::{Value, json};

const MODEL: &str = "olmoe-tiny";
const SECRET: &str = "marmalade narwhal";

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

fn gateway(cfg: ServerConfig) -> Gateway {
    let gw = Gateway::new(ServerConfig {
        log_format: LogFormat::Off,
        ..cfg
    });
    gw.add_model(MODEL, engine(MODEL)).unwrap();
    gw
}

fn off() -> ServerConfig {
    ServerConfig {
        context: ContextSettings {
            config: ContextConfig {
                enabled: false,
                ..ContextConfig::default()
            },
            ..ContextSettings::default()
        },
        ..ServerConfig::default()
    }
}

fn short_chat() -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello"}],
        "max_tokens": 8, "temperature": 0,
    })
}

/// A conversation several times longer than the context.
fn long_messages(user: &str) -> Vec<Value> {
    let mut m = vec![json!({"role": "system", "content": "Be brief."})];
    for i in 0..12 {
        m.push(json!({"role": "user", "content": format!("Note {i}: box {i} is blue. {SECRET}")}));
        m.push(json!({"role": "assistant", "content": format!("Noted, box {i} is blue.")}));
    }
    m.push(json!({"role": "user", "content": user}));
    m
}

fn long_chat(stream: bool) -> Value {
    json!({
        "model": MODEL, "messages": long_messages("What colour is box 4?"),
        "max_tokens": 8, "temperature": 0, "stream": stream,
    })
}

/// The parts of a completion that do not change between two runs.
fn stable(v: &Value) -> Value {
    json!({ "choices": v["choices"], "usage": v["usage"], "model": v["model"] })
}

#[test]
fn a_request_that_fits_gets_the_same_answer_with_management_on_or_off() {
    let on = gateway(ServerConfig::default());
    let off = gateway(off());
    let (a, b) = (start(&on), start(&off));
    for body in [
        short_chat(),
        json!({"model": MODEL, "prompt": "Once upon", "max_tokens": 8, "temperature": 0}),
    ] {
        let path = if body.get("messages").is_some() {
            "/v1/chat/completions"
        } else {
            "/v1/completions"
        };
        let (x, y) = (post(a.addr, path, &body), post(b.addr, path, &body));
        assert_eq!((x.status, y.status), (200, 200));
        assert_eq!(stable(&x.json()), stable(&y.json()));
        assert!(x.header("x-qubivex-context").is_none());
    }
    assert_eq!(on.metrics().context_overflows(), 0);
}

#[test]
fn a_conversation_longer_than_the_context_is_answered_and_says_so() {
    let gw = gateway(ServerConfig::default());
    let s = start(&gw);
    let r = post(s.addr, "/v1/chat/completions", &long_chat(false));
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    assert_eq!(r.header("x-qubivex-context"), Some("compacted"));
    let (orig, prep) = r
        .header("x-qubivex-context-tokens")
        .unwrap()
        .split_once(';')
        .unwrap();
    let (orig, prep): (usize, usize) = (orig.parse().unwrap(), prep.parse().unwrap());
    assert!(orig > 256 && prep < 256);
    let v = r.json();
    assert_eq!(
        v["usage"]["prompt_tokens"], prep,
        "usage is what the model ran"
    );
    // The body is the usual completion: no summary or metadata in the message.
    assert_eq!(v["object"], "chat.completion");
    assert!(!text_of(&v).contains("Earlier conversation"));
    assert_eq!(gw.metrics().context_overflows(), 1);
    assert_eq!(gw.metrics().context_managed(), (1, 0));
}

#[test]
fn a_stream_after_preparation_streams_as_usual() {
    let gw = gateway(ServerConfig::default());
    let s = start(&gw);
    let r = post(s.addr, "/v1/chat/completions", &long_chat(true));
    assert_eq!(r.status, 200);
    assert_eq!(r.header("x-qubivex-context"), Some("compacted"));
    let ev = r.events();
    assert_eq!(ev.last().map(String::as_str), Some("[DONE]"));
    let first: Value = serde_json::from_str(&ev[0]).unwrap();
    assert_eq!(first["object"], "chat.completion.chunk");
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
}

#[test]
fn with_management_off_the_old_refusal_is_unchanged() {
    let gw = gateway(off());
    let s = start(&gw);
    let r = post(s.addr, "/v1/chat/completions", &long_chat(false));
    assert_eq!(r.status, 400);
    let e = &r.json()["error"];
    assert!(
        e["message"]
            .as_str()
            .unwrap()
            .ends_with("leaving no room to generate"),
        "{e}"
    );
    assert_eq!(e["type"], "invalid_request_error");
    assert!(e["param"].is_null());
    assert_eq!(gw.metrics().context_overflows(), 1);
}

#[test]
fn instructions_longer_than_the_context_are_refused_with_the_reason() {
    let gw = gateway(ServerConfig::default());
    let s = start(&gw);
    let body = json!({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Policy: never do X. ".repeat(40)},
            {"role": "user", "content": "hi"},
        ],
    });
    let r = post(s.addr, "/v1/chat/completions", &body);
    assert_eq!(r.status, 400);
    let e = &r.json()["error"];
    assert_eq!(e["code"], "context_length_exceeded");
    assert_eq!(e["param"], "messages");
    assert!(
        e["message"]
            .as_str()
            .unwrap()
            .contains("system and developer messages")
    );
}

#[test]
fn a_long_raw_prompt_is_reduced() {
    let gw = gateway(ServerConfig::default());
    let s = start(&gw);
    let prompt: String = (0..120).map(|i| format!("row {i}\n")).collect::<String>() + "end";
    let body = json!({"model": MODEL, "prompt": prompt, "max_tokens": 4, "temperature": 0});
    let r = post(s.addr, "/v1/completions", &body);
    assert_eq!(r.status, 200);
    assert_eq!(r.header("x-qubivex-context"), Some("reduced"));
    assert_eq!(gw.metrics().context_managed(), (0, 1));
}

#[test]
fn inspect_explains_without_generating() {
    let gw = gateway(ServerConfig::default());
    let s = start(&gw);
    let fits = post(s.addr, "/v1/context/inspect", &short_chat());
    assert_eq!(fits.status, 200);
    let v = fits.json();
    assert_eq!(v["managed"], false);
    assert!(v["report"].is_null() && v.get("prepared").is_none());
    assert_eq!(v["context_limits"]["effective"], 256);
    assert_eq!(v["context_limits"]["configured"], 256);

    let mut body = long_chat(false);
    let r = post(s.addr, "/v1/context/inspect", &body);
    let v = r.json();
    assert_eq!(v["managed"], true);
    let rep = &v["report"];
    assert_eq!(rep["compacted"], true);
    assert!(rep["prepared_tokens"].as_u64().unwrap() <= v["budget"]["input"].as_u64().unwrap());
    let fates: Vec<&str> = rep["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["fate"].as_str().unwrap())
        .collect();
    assert_eq!(fates[0], "kept");
    assert!(fates.contains(&"condensed"));
    // No text unless asked, and then only this request's own.
    assert!(v.get("prepared").is_none());
    assert!(!r.body.windows(SECRET.len()).any(|w| w == SECRET.as_bytes()));
    body["include_text"] = json!(true);
    let v = post(s.addr, "/v1/context/inspect", &body).json();
    let prepared = v["prepared"].as_array().unwrap();
    assert_eq!(prepared[0]["content"], "Be brief.");
    assert!(
        prepared.last().unwrap()["content"]
            .as_str()
            .unwrap()
            .ends_with("What colour is box 4?")
    );
    // Nothing generated.
    assert_eq!(gw.metrics().generations(MODEL), 0);
}

#[test]
fn metrics_and_logs_count_what_happened_without_text() {
    let log = Captured::default();
    let gw = Gateway::with_log_writer(
        ServerConfig {
            log_format: LogFormat::Json,
            ..ServerConfig::default()
        },
        Box::new(log.clone()),
    );
    gw.add_model(MODEL, engine(MODEL)).unwrap();
    let s = start(&gw);
    assert_eq!(
        post(s.addr, "/v1/chat/completions", &long_chat(false)).status,
        200
    );
    let m = http(s.addr, "GET", "/metrics", &[], None);
    let text = String::from_utf8_lossy(&m.body).to_string();
    assert!(text.contains("qubivex_context_overflows_total 1"), "{text}");
    assert!(text.contains("qubivex_context_managed_total{strategy=\"compacted\"} 1"));
    assert!(text.contains("qubivex_context_prepare_seconds_count{stage=\"total\"} 1"));
    let logged = String::from_utf8(log.0.lock().unwrap().clone()).unwrap();
    let ev: Vec<Value> = logged
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let managed = ev.iter().find(|e| e["event"] == "context_managed").unwrap();
    assert_eq!(managed["strategy"], "compacted");
    assert!(managed["original_tokens"].as_u64().unwrap() > 256);
    assert!(!logged.contains(SECRET), "no conversation text in the log");
    // `/v1/system` says how it is set up.
    let sys = http(s.addr, "GET", "/v1/system", &[], None).json();
    assert_eq!(sys["context_management"]["enabled"], true);
    assert_eq!(sys["context_management"]["store"], "memory");
    assert_eq!(sys["models"][0]["context_limits"]["effective"], 256);
}

#[test]
fn concurrent_long_requests_from_different_users_all_fit() {
    let gw = gateway(ServerConfig::default());
    let s = start(&gw);
    let addr = s.addr;
    let handles: Vec<_> = (0..6)
        .map(|i| {
            std::thread::spawn(move || {
                let mut body = long_chat(i % 2 == 0);
                body["user"] = json!(format!("user-{i}"));
                body["messages"] =
                    json!(long_messages(&format!("Which box did user {i} ask about?")));
                post(addr, "/v1/chat/completions", &body)
            })
        })
        .collect();
    for h in handles {
        let r = h.join().unwrap();
        assert_eq!(r.status, 200);
        assert_eq!(r.header("x-qubivex-context"), Some("compacted"));
    }
    assert_eq!(gw.metrics().context_managed().0, 6);
}

/// Checkpoints in files: the next turn after a restart reuses the note. A context of
/// 1024, so the note has room for excerpts.
#[test]
fn a_restarted_server_reuses_its_notes() {
    let big = || {
        qubivex_runtime::Engine::load(
            common::meta(MODEL),
            qubivex_runtime::EngineOptions {
                max_context: Some(1024),
                ..common::opts()
            },
        )
        .unwrap()
    };
    let served = |cfg: ServerConfig| {
        let gw = Gateway::new(ServerConfig {
            log_format: LogFormat::Off,
            ..cfg
        });
        gw.add_model(MODEL, big()).unwrap();
        gw
    };
    let dir = std::env::temp_dir().join(format!("qubivex-api-context-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = || ServerConfig {
        context: ContextSettings {
            store_dir: Some(dir.clone()),
            ..ContextSettings::default()
        },
        ..ServerConfig::default()
    };
    let mut messages = long_messages("What colour is box 4?");
    let more = long_messages("x");
    messages.splice(1..1, more[1..more.len() - 1].iter().cloned());
    messages.splice(1..1, more[1..more.len() - 1].iter().cloned());
    {
        let gw = served(cfg());
        let s = start(&gw);
        let body = json!({"model": MODEL, "messages": messages, "max_tokens": 8, "user": "u1"});
        let v = post(s.addr, "/v1/context/inspect", &body).json();
        assert_eq!(v["report"]["checkpoint"], "miss");
    }
    messages.push(json!({"role": "assistant", "content": "Blue."}));
    messages.push(json!({"role": "user", "content": "And box 9?"}));
    let gw = served(cfg());
    let s = start(&gw);
    let body = json!({"model": MODEL, "messages": messages, "max_tokens": 8, "user": "u1"});
    let v = post(s.addr, "/v1/context/inspect", &body).json();
    assert_eq!(v["report"]["checkpoint"], "hit", "{v}");
    // Another user never sees it.
    let other = json!({"model": MODEL, "messages": messages, "max_tokens": 8, "user": "u2"});
    let v = post(s.addr, "/v1/context/inspect", &other).json();
    assert_eq!(v["report"]["checkpoint"], "miss");
    drop(s);
    std::fs::remove_dir_all(&dir).unwrap();
}
