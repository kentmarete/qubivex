//! The gateway end to end over real HTTP, on the tiny fixtures: OpenAI response shapes,
//! streaming equal to non-streaming, concurrent conversations with independent KV,
//! cancellation when the client goes away, KV admission, auth and admin, and errors
//! without paths. The client is a few lines of blocking HTTP/1.1 over `TcpStream`.

#![allow(clippy::unwrap_used, clippy::too_many_lines)]

mod common;

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use common::{
    Resp, engine, http, meta, post, reference, request, root, slow_engine, start, text_of,
    wait_until,
};
use qubivex_api::{Gateway, MemoryPolicy, ServerConfig};
use serde_json::{Value, json};

#[test]
fn openai_shapes_and_streaming_equal_non_streaming() {
    let gw = Gateway::new(ServerConfig::default());
    for id in ["olmoe-tiny", "dense-tiny"] {
        gw.add_model(id, engine(id)).unwrap();
    }
    let s = start(&gw);
    let r = http(s.addr, "GET", "/health", &[], None);
    assert_eq!((r.status, r.json()["models"].clone()), (200, json!(2)));
    let r = http(
        s.addr,
        "GET",
        "/v1/models",
        &[("x-request-id", "abc-123")],
        None,
    );
    assert_eq!(r.status, 200);
    assert_eq!(r.header("x-request-id"), Some("abc-123"));
    let ids: Vec<&str> = r.json()["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_owned())
        .collect::<Vec<_>>()
        .leak()
        .iter()
        .map(String::as_str)
        .collect();
    assert_eq!(ids, ["dense-tiny", "olmoe-tiny"]);
    assert!(r.json()["data"][0]["object"] == "model");

    for id in ["olmoe-tiny", "dense-tiny"] {
        let want = reference(&engine(id), "Hello, experts", 12);
        let body =
            json!({ "model": id, "prompt": "Hello, experts", "max_tokens": 12, "temperature": 0 });
        let r = post(s.addr, "/v1/completions", &body);
        assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
        assert!(
            r.header("x-request-id")
                .is_some_and(|v| v.starts_with("req-"))
        );
        let j = r.json();
        assert_eq!(j["object"], "text_completion");
        assert_eq!(text_of(&j), want, "{id}");
        let u = &j["usage"];
        assert_eq!(
            u["total_tokens"],
            u["prompt_tokens"].as_u64().unwrap() + u["completion_tokens"].as_u64().unwrap()
        );
        assert!(["stop", "length"].contains(&j["choices"][0]["finish_reason"].as_str().unwrap()));

        // The same request streamed: the deltas join to the same text.
        let mut body = body.clone();
        body["stream"] = json!(true);
        body["stream_options"] = json!({ "include_usage": true });
        let r = post(s.addr, "/v1/completions", &body);
        assert_eq!(r.status, 200);
        assert!(
            r.header("content-type")
                .is_some_and(|c| c.starts_with("text/event-stream"))
        );
        let ev = r.events();
        assert_eq!(ev.last().map(String::as_str), Some("[DONE]"));
        let chunks: Vec<Value> = ev[..ev.len() - 1]
            .iter()
            .map(|e| serde_json::from_str(e).unwrap())
            .collect();
        let streamed: String = chunks
            .iter()
            .filter_map(|c| c["choices"].get(0).and_then(|ch| ch["text"].as_str()))
            .collect();
        assert_eq!(streamed, want, "{id} streamed");
        assert_eq!(
            chunks.last().unwrap()["usage"],
            j["usage"],
            "{id} usage chunk"
        );

        // Chat, streamed and not.
        let body = json!({
            "model": id,
            "messages": [{ "role": "system", "content": "be brief" }, { "role": "user", "content": "hi" }],
            "max_tokens": 8, "temperature": 0,
        });
        let r = post(s.addr, "/v1/chat/completions", &body);
        assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
        let j = r.json();
        assert_eq!(j["object"], "chat.completion");
        assert_eq!(j["choices"][0]["message"]["role"], "assistant");
        let mut sbody = body.clone();
        sbody["stream"] = json!(true);
        let ev = post(s.addr, "/v1/chat/completions", &sbody).events();
        let chunks: Vec<Value> = ev[..ev.len() - 1]
            .iter()
            .map(|e| serde_json::from_str(e).unwrap())
            .collect();
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunks[0]["object"], "chat.completion.chunk");
        let streamed: String = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(streamed, text_of(&j), "{id} chat streamed");
        assert!(chunks.last().unwrap()["choices"][0]["finish_reason"].is_string());
    }
}

#[test]
fn concurrent_conversations_get_independent_kv() {
    let gw = Gateway::new(ServerConfig {
        max_concurrent: 4,
        ..ServerConfig::default()
    });
    gw.add_model("olmoe-tiny", engine("olmoe-tiny")).unwrap();
    let s = start(&gw);
    let prompts = [
        "Hello, experts",
        "The quick brown fox",
        "abc abc abc",
        "Once upon a time",
    ];
    let e = engine("olmoe-tiny");
    let want: Vec<String> = prompts.iter().map(|p| reference(&e, p, 24)).collect();
    for _round in 0..3 {
        let handles: Vec<_> = prompts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let (addr, p) = (s.addr, (*p).to_owned());
                std::thread::spawn(move || {
                    let stream = i % 2 == 0;
                    let r = post(
                        addr,
                        "/v1/completions",
                        &json!({ "model": "olmoe-tiny", "prompt": p, "max_tokens": 24, "temperature": 0, "stream": stream }),
                    );
                    assert_eq!(r.status, 200);
                    if stream {
                        r.events()
                            .iter()
                            .filter(|e| *e != "[DONE]")
                            .filter_map(|e| serde_json::from_str::<Value>(e).ok())
                            .filter_map(|c| c["choices"].get(0).and_then(|ch| ch["text"].as_str().map(str::to_owned)))
                            .collect::<String>()
                    } else {
                        text_of(&r.json())
                    }
                })
            })
            .collect();
        let got: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(got, want);
    }
    assert_eq!(gw.metrics().active(), 0);
}

#[test]
fn a_client_that_goes_away_cancels_its_generation() {
    let gw = Gateway::new(ServerConfig {
        max_concurrent: 1,
        queue_timeout: Duration::from_secs(5),
        ..ServerConfig::default()
    });
    gw.add_model(
        "olmoe-tiny",
        slow_engine("olmoe-tiny", Duration::from_millis(5), None),
    )
    .unwrap();
    let m = gw.metrics();
    let s = start(&gw);
    // Streaming: read until the first token arrives, then hang up.
    let body = json!({ "model": "olmoe-tiny", "prompt": "Hello, experts", "max_tokens": 200, "temperature": 0, "stream": true });
    let mut c = TcpStream::connect(s.addr).unwrap();
    c.write_all(request(s.addr, "POST", "/v1/completions", &[], Some(&body)).as_bytes())
        .unwrap();
    let mut seen = Vec::new();
    let mut buf = [0u8; 4096];
    while !String::from_utf8_lossy(&seen).contains("\"text\":\"") {
        let n = c.read(&mut buf).unwrap();
        assert!(n > 0, "the stream ended early");
        seen.extend_from_slice(&buf[..n]);
    }
    assert_eq!(m.active(), 1);
    drop(c);
    wait_until("the streamed generation to be cancelled", || {
        m.cancelled() == 1 && m.active() == 0
    });

    // Not streaming: send the request and hang up before the answer.
    let mut c = TcpStream::connect(s.addr).unwrap();
    let mut body = body.clone();
    body["stream"] = json!(false);
    c.write_all(request(s.addr, "POST", "/v1/completions", &[], Some(&body)).as_bytes())
        .unwrap();
    wait_until("the generation to start", || m.active() == 1);
    drop(c);
    wait_until("the non-streamed generation to be cancelled", || {
        m.cancelled() == 2 && m.active() == 0
    });

    // The only slot is free again: a new request runs to the end.
    let r = post(
        s.addr,
        "/v1/completions",
        &json!({ "model": "olmoe-tiny", "prompt": "hi", "max_tokens": 3, "temperature": 0 }),
    );
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    assert_eq!(r.json()["usage"]["completion_tokens"], 3);
}

#[test]
fn kv_admission_refuses_what_the_plan_cannot_hold() {
    let m = meta("olmoe-tiny");
    let fp = qubivex_memory::ModelFootprint::from_metadata(&m).unwrap();
    // Room for four 256-position conversations' KV, not five (as in the runtime's
    // reservation test).
    let budget = fp.dense_bytes().0
        + qubivex_memory::ACTIVATION_RESERVE.0
        + fp.cache_floor().0
        + 1024 * fp.kv_bytes_per_token.0;
    let gw = Gateway::new(ServerConfig {
        max_concurrent: 8,
        memory_policy: MemoryPolicy::Refuse,
        ..ServerConfig::default()
    });
    gw.add_model(
        "olmoe-tiny",
        slow_engine("olmoe-tiny", Duration::from_millis(20), Some(budget)),
    )
    .unwrap();
    let metrics = gw.metrics();
    let s = start(&gw);
    // Four long conversations, each reserving the whole 256-position context.
    let body = json!({ "model": "olmoe-tiny", "prompt": "Hello", "max_tokens": 250, "temperature": 0, "stream": true });
    let mut held = Vec::new();
    for _ in 0..4 {
        let mut c = TcpStream::connect(s.addr).unwrap();
        c.write_all(request(s.addr, "POST", "/v1/completions", &[], Some(&body)).as_bytes())
            .unwrap();
        let mut buf = [0u8; 1024];
        let n = c.read(&mut buf).unwrap();
        assert!(String::from_utf8_lossy(&buf[..n]).starts_with("HTTP/1.1 200"));
        held.push(c);
    }
    let mut fifth = body.clone();
    fifth["stream"] = json!(false);
    let r = post(s.addr, "/v1/completions", &fifth);
    assert_eq!(r.status, 503, "{}", String::from_utf8_lossy(&r.body));
    let j = r.json();
    assert_eq!(j["error"]["code"], "overloaded");
    assert!(
        j["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no room for another conversation")
    );
    // A short one still fits once a long one ends.
    drop(held.pop());
    wait_until("the dropped conversation to end", || metrics.active() == 3);
    let r = post(
        s.addr,
        "/v1/completions",
        &json!({ "model": "olmoe-tiny", "prompt": "hi", "max_tokens": 2, "temperature": 0 }),
    );
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    drop(held);
    wait_until("every conversation to end", || metrics.active() == 0);
}

#[test]
fn auth_admin_and_errors_without_paths() {
    let gw = Gateway::new(ServerConfig {
        api_key: Some("sk-test".into()),
        admin_token: Some("admin-secret".into()),
        model_dirs: vec![root()],
        ..ServerConfig::default()
    });
    gw.add_model("olmoe-tiny", engine("olmoe-tiny")).unwrap();
    let s = start(&gw);
    let key = [("authorization", "Bearer sk-test")];
    let admin = [("authorization", "Bearer admin-secret")];
    let fixtures = root().to_string_lossy().into_owned();
    let no_paths = |r: &Resp| {
        let b = String::from_utf8_lossy(&r.body);
        assert!(
            !b.contains(&fixtures) && !b.contains("/Users/") && !b.contains(".safetensors"),
            "{b}"
        );
    };

    // Keys.
    let r = http(s.addr, "GET", "/v1/models", &[], None);
    assert_eq!(
        (r.status, r.json()["error"]["type"].clone()),
        (401, json!("authentication_error"))
    );
    let r = http(
        s.addr,
        "GET",
        "/v1/models",
        &[("authorization", "Bearer nope")],
        None,
    );
    assert_eq!(r.status, 401);
    assert_eq!(http(s.addr, "GET", "/v1/models", &key, None).status, 200);
    assert_eq!(http(s.addr, "GET", "/health", &[], None).status, 200);

    // Request errors.
    let chat = |body: &Value| http(s.addr, "POST", "/v1/chat/completions", &key, Some(body));
    let msgs = json!([{ "role": "user", "content": "hi" }]);
    for (body, status, param) in [
        (json!({ "model": "nope", "messages": msgs }), 404, "model"),
        (
            json!({ "model": "olmoe-tiny", "messages": msgs, "n": 2 }),
            400,
            "n",
        ),
        (
            json!({ "model": "olmoe-tiny", "messages": msgs, "max_tokens": 100000 }),
            400,
            "max_tokens",
        ),
        (
            json!({ "model": "olmoe-tiny", "messages": msgs, "temperature": 7 }),
            400,
            "temperature",
        ),
        (
            json!({ "model": "olmoe-tiny", "messages": msgs, "tools": [{}] }),
            400,
            "tools",
        ),
        (
            json!({ "model": "olmoe-tiny", "messages": [] }),
            400,
            "messages",
        ),
    ] {
        let r = chat(&body);
        assert_eq!(
            r.status,
            status,
            "{body}: {}",
            String::from_utf8_lossy(&r.body)
        );
        assert_eq!(r.json()["error"]["param"], param, "{body}");
        no_paths(&r);
    }
    let mut raw = TcpStream::connect(s.addr).unwrap();
    let bad = "{not json";
    raw.write_all(
        format!(
            "POST /v1/completions HTTP/1.1\r\nhost: x\r\nconnection: close\r\nauthorization: Bearer sk-test\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{bad}",
            bad.len()
        )
        .as_bytes(),
    )
    .unwrap();
    let mut out = String::new();
    raw.read_to_string(&mut out).unwrap();
    assert!(out.starts_with("HTTP/1.1 400"), "{out}");
    assert!(out.contains("invalid_request_error"), "{out}");
    let r = http(s.addr, "GET", "/nope", &[], None);
    assert_eq!(
        (r.status, r.json()["error"]["code"].clone()),
        (404, json!("not_found"))
    );

    // Admin: refused without the token; load, duplicate, unload.
    let load = json!({ "model": "dense-tiny" });
    assert_eq!(
        http(s.addr, "POST", "/v1/admin/models", &key, Some(&load)).status,
        401
    );
    let r = http(s.addr, "POST", "/v1/admin/models", &admin, Some(&load));
    assert_eq!(r.status, 201, "{}", String::from_utf8_lossy(&r.body));
    assert_eq!(r.json()["id"], "dense-tiny");
    assert_eq!(gw.model_ids(), ["dense-tiny", "olmoe-tiny"]);
    // Loading it again changes nothing.
    assert_eq!(
        http(s.addr, "POST", "/v1/admin/models", &admin, Some(&load)).status,
        200
    );
    // One model cannot be served twice, and one id cannot mean two models.
    let again = json!({ "model": "dense-tiny", "id": "other-name" });
    assert_eq!(
        http(s.addr, "POST", "/v1/admin/models", &admin, Some(&again)).status,
        409
    );
    let clash = json!({ "model": "olmoe-tiny", "id": "dense-tiny" });
    assert_eq!(
        http(s.addr, "POST", "/v1/admin/models", &admin, Some(&clash)).status,
        409
    );
    let r = http(
        s.addr,
        "POST",
        "/v1/admin/models",
        &admin,
        Some(&json!({ "model": "no-such-model" })),
    );
    assert_eq!(r.status, 404);
    no_paths(&r);
    let r = chat(
        &json!({ "model": "dense-tiny", "messages": msgs, "max_tokens": 4, "temperature": 0 }),
    );
    assert_eq!(r.status, 200);
    let r = http(
        s.addr,
        "DELETE",
        "/v1/admin/models/dense-tiny",
        &admin,
        None,
    );
    assert_eq!((r.status, r.json()["deleted"].clone()), (200, json!(true)));
    assert_eq!(
        chat(&json!({ "model": "dense-tiny", "messages": msgs })).status,
        404
    );
    assert_eq!(
        http(
            s.addr,
            "DELETE",
            "/v1/admin/models/dense-tiny",
            &admin,
            None
        )
        .status,
        404
    );

    // Without an admin token the endpoints are off.
    let off = Gateway::new(ServerConfig::default());
    let s2 = start(&off);
    let r = http(s2.addr, "POST", "/v1/admin/models", &admin, Some(&load));
    assert_eq!(
        (r.status, r.json()["error"]["code"].clone()),
        (403, json!("admin_disabled"))
    );

    // Metrics and system.
    let r = http(s.addr, "GET", "/metrics", &[], None);
    let text = String::from_utf8_lossy(&r.body);
    assert!(
        text.contains("qubivex_requests_total{endpoint=\"chat_completions\",status=\"404\"}"),
        "{text}"
    );
    assert!(text.contains("qubivex_generated_tokens_total"), "{text}");
    assert!(
        text.contains("qubivex_kv_reserved_positions{model=\"olmoe-tiny\"} 0"),
        "{text}"
    );
    let r = http(s.addr, "GET", "/v1/system", &key, None);
    assert_eq!(r.status, 200);
    assert_eq!(r.json()["models"][0]["id"], "olmoe-tiny");
    no_paths(&r);
}
