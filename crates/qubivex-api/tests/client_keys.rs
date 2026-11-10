//! Named API keys over real HTTP, on the tiny fixtures (ADR-013 amendment 7): each key
//! is its client's for authentication, the rate limit, prefix-cache reuse and the log.

#![allow(clippy::unwrap_used)]

mod common;

use std::io::Write;
use std::sync::{Arc, Mutex};

use common::{engine, http, meta, opts, start};
use qubivex_api::{ClientKey, Gateway, LogFormat, RateLimit, ServerConfig, hash_key};
use qubivex_runtime::{Engine, EngineOptions};
use serde_json::{Value, json};

fn bearer(key: &str) -> String {
    format!("Bearer {key}")
}

fn keys() -> Vec<ClientKey> {
    vec![
        ClientKey::new("alpha", "alpha-secret", None),
        ClientKey::from_hex("beta", &hash_key("beta-secret"), None).unwrap(),
    ]
}

#[test]
fn each_named_key_and_the_shared_key_are_accepted_and_nothing_else() {
    let gw = Gateway::new(ServerConfig {
        api_key: Some("shared".into()),
        client_keys: keys(),
        ..ServerConfig::default()
    });
    gw.add_model("olmoe-tiny", engine("olmoe-tiny")).unwrap();
    let s = start(&gw);
    let get = |auth: Option<&str>| {
        let h = auth.map(bearer);
        let headers: Vec<(&str, &str)> = h.iter().map(|v| ("authorization", v.as_str())).collect();
        http(s.addr, "GET", "/v1/models", &headers, None).status
    };
    assert_eq!(get(Some("alpha-secret")), 200);
    assert_eq!(get(Some("beta-secret")), 200);
    assert_eq!(get(Some("shared")), 200);
    assert_eq!(get(Some("gamma-secret")), 401);
    // The hash is not the key.
    assert_eq!(get(Some(&hash_key("alpha-secret"))), 401);
    assert_eq!(get(None), 401);

    // Named keys alone also close the API.
    let gw = Gateway::new(ServerConfig {
        client_keys: keys(),
        ..ServerConfig::default()
    });
    let s2 = start(&gw);
    assert_eq!(http(s2.addr, "GET", "/v1/models", &[], None).status, 401);
    let a = bearer("alpha-secret");
    assert_eq!(
        http(s2.addr, "GET", "/v1/models", &[("authorization", &a)], None).status,
        200
    );
}

/// All requests come from 127.0.0.1, as they would from behind a proxy: keys still get
/// their own allowances.
#[test]
fn each_key_has_its_own_allowance_and_unknown_keys_share_the_address() {
    let mut k = keys();
    k[0].rate_limit = Some(RateLimit {
        per_minute: 1,
        burst: 1,
    });
    let gw = Gateway::new(ServerConfig {
        client_keys: k,
        rate_limit: Some(RateLimit {
            per_minute: 1,
            burst: 3,
        }),
        ..ServerConfig::default()
    });
    let s = start(&gw);
    let get = |key: &str| {
        let a = bearer(key);
        http(s.addr, "GET", "/v1/models", &[("authorization", &a)], None).status
    };
    // alpha's own limit: one request.
    assert_eq!(get("alpha-secret"), 200);
    assert_eq!(get("alpha-secret"), 429);
    // beta has the server's: three, untouched by alpha.
    for _ in 0..3 {
        assert_eq!(get("beta-secret"), 200);
    }
    assert_eq!(get("beta-secret"), 429);
    // Made-up keys count by address: three between them, each a 401, then 429s.
    for n in 0..3 {
        assert_eq!(get(&format!("made-up-{n}")), 401);
    }
    assert_eq!(get("made-up-3"), 429);
    assert_eq!(http(s.addr, "GET", "/v1/models", &[], None).status, 429);
}

#[test]
fn clients_never_share_retained_kv_whatever_user_they_send() {
    let gw = Gateway::new(ServerConfig {
        client_keys: keys(),
        ..ServerConfig::default()
    });
    let on = EngineOptions {
        prefix_cache_bytes: 64 << 20,
        ..opts()
    };
    gw.add_model("on", Engine::load(meta("olmoe-tiny"), on).unwrap())
        .unwrap();
    let s = start(&gw);
    let ask = |key: &str, user: Option<&str>| {
        let mut b = json!({
            "model": "on", "temperature": 0, "max_tokens": 4,
            "prompt": "the quick brown fox jumps over the lazy dog ".repeat(4),
        });
        if let Some(u) = user {
            b["user"] = json!(u);
        }
        let a = bearer(key);
        let r = http(
            s.addr,
            "POST",
            "/v1/completions",
            &[("authorization", &a)],
            Some(&b),
        );
        assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
        r.json()
    };
    let cached = |r: &Value| {
        r["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
    };
    let first = ask("alpha-secret", Some("u"));
    assert_eq!(cached(&first), 0);
    // beta claims alpha's user, and alpha's name as its user: no reuse either way.
    assert_eq!(cached(&ask("beta-secret", Some("u"))), 0);
    assert_eq!(cached(&ask("beta-secret", Some("alpha"))), 0);
    assert_eq!(cached(&ask("beta-secret", None)), 0);
    // alpha's own users stay apart, and the same one reuses its prefix.
    assert_eq!(cached(&ask("alpha-secret", Some("v"))), 0);
    let again = ask("alpha-secret", Some("u"));
    assert!(cached(&again) > 0);
    assert_eq!(again["choices"], first["choices"]);
}

#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl Write for Captured {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn the_log_names_the_client_and_never_the_key() {
    let log = Captured::default();
    let cfg = ServerConfig {
        client_keys: keys(),
        log_format: LogFormat::Json,
        ..ServerConfig::default()
    };
    let gw = Gateway::with_log_writer(cfg, Box::new(log.clone()));
    gw.add_model("olmoe-tiny", engine("olmoe-tiny")).unwrap();
    let s = start(&gw);
    let body = json!({ "model": "olmoe-tiny", "prompt": "hi", "max_tokens": 2, "user": "u" });
    let a = bearer("beta-secret");
    let r = http(
        s.addr,
        "POST",
        "/v1/completions",
        &[("authorization", &a)],
        Some(&body),
    );
    assert_eq!(r.status, 200);
    drop(s);
    let text = String::from_utf8(log.0.lock().unwrap().clone()).unwrap();
    let started: Value = text
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .find(|l| l["event"] == "generation_started")
        .unwrap();
    assert_eq!(started["client"], "beta");
    assert!(!text.contains("beta-secret"), "{text}");
    assert!(!text.contains(&hash_key("beta-secret")), "{text}");
}
