//! Spec §11 over real HTTP, on the tiny fixtures: the per-client rate limit, request
//! size, prompt and generation limits, and CORS.

#![allow(clippy::unwrap_used)]

mod common;

use common::{engine, http, post, start};
use qubivex_api::{Gateway, RateLimit, ServerConfig};
use serde_json::json;

fn gateway(cfg: ServerConfig) -> Gateway {
    let gw = Gateway::new(cfg);
    gw.add_model("olmoe-tiny", engine("olmoe-tiny")).unwrap();
    gw
}

#[test]
fn a_client_past_its_rate_limit_gets_a_429_with_retry_after() {
    let gw = gateway(ServerConfig {
        rate_limit: Some(RateLimit {
            per_minute: 1,
            burst: 2,
        }),
        ..ServerConfig::default()
    });
    let s = start(&gw);
    for _ in 0..2 {
        assert_eq!(http(s.addr, "GET", "/v1/models", &[], None).status, 200);
    }
    let r = http(
        s.addr,
        "GET",
        "/v1/models",
        &[("x-request-id", "rl-1")],
        None,
    );
    assert_eq!(r.status, 429);
    assert_eq!(r.json()["error"]["code"], "rate_limit_exceeded");
    assert_eq!(r.header("x-request-id"), Some("rl-1"));
    let wait: u64 = r.header("retry-after").unwrap().parse().unwrap();
    assert!((1..=60).contains(&wait), "{wait}");
    // Generations count against the same allowance.
    let body = json!({ "model": "olmoe-tiny", "prompt": "hi", "max_tokens": 2 });
    assert_eq!(post(s.addr, "/v1/completions", &body).status, 429);
    // Monitoring is never limited.
    assert_eq!(http(s.addr, "GET", "/health", &[], None).status, 200);
    assert_eq!(http(s.addr, "GET", "/metrics", &[], None).status, 200);
    assert_eq!(gw.metrics().requests("rate_limit", 429), 2);
}

#[test]
fn bodies_prompts_and_max_tokens_past_the_limits_are_refused() {
    let gw = gateway(ServerConfig {
        max_body_bytes: 1024,
        max_prompt_tokens: Some(6),
        max_generation_tokens: Some(8),
        default_max_tokens: 4,
        ..ServerConfig::default()
    });
    let s = start(&gw);

    let big = json!({ "model": "olmoe-tiny", "prompt": "x".repeat(2000), "max_tokens": 2 });
    let r = post(s.addr, "/v1/completions", &big);
    assert_eq!(r.status, 413, "{}", String::from_utf8_lossy(&r.body));
    assert_eq!(r.json()["error"]["code"], "request_too_large");

    let long = json!({
        "model": "olmoe-tiny",
        "prompt": "one two three four five six seven eight nine ten",
        "max_tokens": 2,
    });
    let r = post(s.addr, "/v1/completions", &long);
    assert_eq!(r.status, 400);
    let e = &r.json()["error"];
    assert_eq!(
        (e["code"].as_str(), e["param"].as_str()),
        (Some("prompt_too_long"), Some("prompt"))
    );
    let chat = json!({
        "model": "olmoe-tiny",
        "messages": [{ "role": "user", "content": "one two three four five six seven eight" }],
    });
    let r = post(s.addr, "/v1/chat/completions", &chat);
    assert_eq!(r.status, 400);
    assert_eq!(r.json()["error"]["param"], "messages");

    let ask = |n: u64| json!({ "model": "olmoe-tiny", "prompt": "hi", "max_tokens": n });
    let r = post(s.addr, "/v1/completions", &ask(9));
    assert_eq!(r.status, 400);
    assert_eq!(r.json()["error"]["param"], "max_tokens");
    let r = post(s.addr, "/v1/completions", &ask(8));
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    // No max_tokens: the default, which validation keeps under the cap.
    let r = post(
        s.addr,
        "/v1/completions",
        &json!({ "model": "olmoe-tiny", "prompt": "hi" }),
    );
    assert_eq!(r.status, 200);
}

#[test]
fn cors_allows_the_configured_origins_only() {
    let gw = gateway(ServerConfig {
        api_key: Some("k".into()),
        cors_origins: vec!["https://app.example".into()],
        rate_limit: Some(RateLimit {
            per_minute: 60,
            burst: 100,
        }),
        ..ServerConfig::default()
    });
    let s = start(&gw);
    let preflight = |origin: &str| {
        http(
            s.addr,
            "OPTIONS",
            "/v1/chat/completions",
            &[
                ("origin", origin),
                ("access-control-request-method", "POST"),
                (
                    "access-control-request-headers",
                    "authorization, content-type",
                ),
            ],
            None,
        )
    };
    let r = preflight("https://app.example");
    assert_eq!(r.status, 204);
    assert_eq!(
        r.header("access-control-allow-origin"),
        Some("https://app.example")
    );
    assert!(
        r.header("access-control-allow-headers")
            .unwrap()
            .contains("authorization")
    );
    assert!(
        r.header("access-control-allow-methods")
            .unwrap()
            .contains("POST")
    );
    assert_eq!(r.header("access-control-allow-credentials"), None);

    let r = preflight("https://evil.example");
    assert_eq!(r.status, 403);
    assert_eq!(r.header("access-control-allow-origin"), None);

    // An allowed origin can read errors as well as answers, and the request id.
    let r = http(
        s.addr,
        "GET",
        "/v1/models",
        &[("origin", "https://app.example")],
        None,
    );
    assert_eq!(r.status, 401);
    assert_eq!(
        r.header("access-control-allow-origin"),
        Some("https://app.example")
    );
    assert!(
        r.header("access-control-expose-headers")
            .unwrap()
            .contains("x-request-id")
    );
    assert_eq!(r.header("vary"), Some("origin"));
    let r = http(
        s.addr,
        "GET",
        "/v1/models",
        &[
            ("origin", "https://evil.example"),
            ("authorization", "Bearer k"),
        ],
        None,
    );
    assert_eq!(r.status, 200);
    assert_eq!(r.header("access-control-allow-origin"), None);
}

#[test]
fn without_cors_origins_no_cors_headers_are_sent() {
    let gw = gateway(ServerConfig::default());
    let s = start(&gw);
    let r = http(
        s.addr,
        "GET",
        "/v1/models",
        &[("origin", "https://app.example")],
        None,
    );
    assert_eq!(r.status, 200);
    assert_eq!(r.header("access-control-allow-origin"), None);
}

#[test]
fn a_wildcard_origin_is_answered_with_a_star() {
    let gw = gateway(ServerConfig {
        cors_origins: vec!["*".into()],
        ..ServerConfig::default()
    });
    let s = start(&gw);
    let r = http(
        s.addr,
        "GET",
        "/health",
        &[("origin", "http://localhost:3000")],
        None,
    );
    assert_eq!(r.header("access-control-allow-origin"), Some("*"));
    assert_eq!(r.header("vary"), None);
}
