//! PLATFORM_SPEC §12.3 over real HTTP on the tiny fixtures: bursts, the bounded queue,
//! priorities, deadlines, memory admission, load dedup, unloading with requests running,
//! shutdown with requests running, mixed context lengths and several models at once.
//! Every test also checks nothing is left behind: slots, the queue and KV reservations.
//! Tests 3, 4, 5, 7, 8 and 10 of §12.3 are in the store's tests and in `gateway.rs`.

#![allow(clippy::unwrap_used, clippy::too_many_lines)]

mod common;

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::{Arc, Barrier, Mutex};
use std::time::{Duration, Instant};

use common::{
    Resp, Server, engine, http, meta, post, reference, request, root, slow_engine, start, text_of,
    wait_until,
};
use qubivex_api::{Gateway, ServerConfig};
use serde_json::{Value, json};

const M: &str = "olmoe-tiny";

fn completion(prompt: &str, max_tokens: usize) -> Value {
    json!({ "model": M, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0 })
}

/// Text of a streamed completion.
fn streamed_text(r: &Resp) -> String {
    r.events()
        .iter()
        .filter(|e| *e != "[DONE]")
        .filter_map(|e| serde_json::from_str::<Value>(e).ok())
        .filter_map(|c| c["choices"][0]["text"].as_str().map(str::to_owned))
        .collect()
}

/// Nothing is left running, queued or reserved.
fn all_clear(gw: &Gateway, id: &str) {
    wait_until("the last generation to end", || gw.metrics().active() == 0);
    let q = gw.queue_stats(id).unwrap();
    assert_eq!((q.running, q.queued), (0, 0), "{q:?}");
    assert_eq!(gw.kv_reserved(id), Some(0));
}

/// Opens a stream and returns once its first token arrived.
fn open_stream(addr: SocketAddr, body: &Value) -> (TcpStream, Vec<u8>) {
    let mut c = TcpStream::connect(addr).unwrap();
    c.write_all(request(addr, "POST", "/v1/completions", &[], Some(body)).as_bytes())
        .unwrap();
    let mut seen = Vec::new();
    let mut buf = [0u8; 4096];
    while !String::from_utf8_lossy(&seen).contains("\"text\":\"") {
        let n = c.read(&mut buf).unwrap();
        assert!(n > 0, "{}", String::from_utf8_lossy(&seen));
        seen.extend_from_slice(&buf[..n]);
    }
    (c, seen)
}

/// Reads a stream to its end and returns the data lines.
fn read_to_end(mut c: TcpStream, mut seen: Vec<u8>) -> Vec<String> {
    c.read_to_end(&mut seen).unwrap();
    String::from_utf8_lossy(&seen)
        .lines()
        .filter_map(|l| l.strip_prefix("data: ").map(str::to_owned))
        .collect()
}

/// §12.3 test 1. More requests than slots, all answered, each with its own result.
#[test]
fn ten_simultaneous_requests_on_one_model() {
    let gw = Gateway::new(ServerConfig {
        max_concurrent: 4,
        ..ServerConfig::default()
    });
    gw.add_model(M, engine(M)).unwrap();
    let s = start(&gw);
    let prompts: Vec<String> = (0..10).map(|i| format!("Prompt number {i}:")).collect();
    let e = engine(M);
    let want: Vec<String> = prompts.iter().map(|p| reference(&e, p, 16)).collect();
    let go = Arc::new(Barrier::new(10));
    let handles: Vec<_> = prompts
        .iter()
        .map(|p| {
            let (addr, body, go) = (s.addr, completion(p, 16), Arc::clone(&go));
            std::thread::spawn(move || {
                go.wait();
                post(addr, "/v1/completions", &body)
            })
        })
        .collect();
    let got: Vec<String> = handles
        .into_iter()
        .map(|h| {
            let r = h.join().unwrap();
            assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
            text_of(&r.json())
        })
        .collect();
    assert_eq!(got, want);
    assert!(gw.queue_stats(M).unwrap().peak_running <= 4);
    all_clear(&gw, M);
}

/// §12.3 test 2. A burst past the limit is refused at once, what is accepted is
/// correct, and the queue never grows past its bound.
#[test]
fn fifty_simultaneous_requests_stay_bounded() {
    let gw = Gateway::new(ServerConfig {
        max_concurrent: 4,
        max_queue: 6,
        ..ServerConfig::default()
    });
    gw.add_model(M, slow_engine(M, Duration::from_millis(2), None))
        .unwrap();
    let s = start(&gw);
    let prompt = "The quick brown fox";
    let want = reference(&engine(M), prompt, 8);
    let go = Arc::new(Barrier::new(50));
    let handles: Vec<_> = (0..50)
        .map(|_| {
            let (addr, body, go) = (s.addr, completion(prompt, 8), Arc::clone(&go));
            std::thread::spawn(move || {
                go.wait();
                post(addr, "/v1/completions", &body)
            })
        })
        .collect();
    let (mut ok, mut refused) = (0, 0);
    for h in handles {
        let r = h.join().unwrap();
        match r.status {
            200 => {
                assert_eq!(text_of(&r.json()), want);
                ok += 1;
            }
            429 => {
                assert_eq!(r.json()["error"]["code"], "too_many_requests");
                refused += 1;
            }
            other => panic!("{other}: {}", String::from_utf8_lossy(&r.body)),
        }
    }
    assert_eq!(ok + refused, 50);
    assert!(ok >= 10, "{ok}");
    assert!(refused > 0);
    let q = gw.queue_stats(M).unwrap();
    assert!(q.peak_running <= 4, "{q:?}");
    assert!(q.peak_queued <= 10, "{q:?}");
    assert_eq!(q.rejected, refused);
    all_clear(&gw, M);
}

/// A model that holds its one slot until told to stop, so tests can line up a queue.
fn occupied(cfg: ServerConfig) -> (Gateway, Server, TcpStream) {
    let gw = Gateway::new(ServerConfig {
        max_concurrent: 1,
        ..cfg
    });
    gw.add_model(M, slow_engine(M, Duration::from_millis(5), None))
        .unwrap();
    let s = start(&gw);
    let mut body = completion("Hello, experts", 200);
    body["stream"] = json!(true);
    let (c, _) = open_stream(s.addr, &body);
    (gw, s, c)
}

fn wait_queued(gw: &Gateway, n: usize) {
    wait_until("the queue to fill", || {
        gw.queue_stats(M).unwrap().queued == n
    });
}

#[test]
fn a_full_queue_refuses_at_once_and_a_timeout_leaves_it() {
    let (gw, s, holder) = occupied(ServerConfig {
        max_queue: 1,
        queue_timeout: Duration::from_millis(300),
        ..ServerConfig::default()
    });
    let addr = s.addr;
    let waiter = std::thread::spawn(move || post(addr, "/v1/completions", &completion("hi", 2)));
    wait_queued(&gw, 1);
    let t = Instant::now();
    let r = post(addr, "/v1/completions", &completion("hi", 2));
    assert_eq!(r.status, 429);
    assert!(
        t.elapsed() < Duration::from_millis(250),
        "{:?}",
        t.elapsed()
    );
    assert!(
        r.json()["error"]["message"]
            .as_str()
            .unwrap()
            .contains("1 requests are waiting")
    );
    // The waiter's queue timeout passes.
    assert_eq!(waiter.join().unwrap().status, 429);
    assert_eq!(gw.queue_stats(M).unwrap().queued, 0);
    drop(holder);
    all_clear(&gw, M);
}

#[test]
fn higher_priorities_start_first() {
    let (gw, s, holder) = occupied(ServerConfig::default());
    let order = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();
    for (n, priority) in ["low", "normal", "high"].into_iter().enumerate() {
        let (addr, order) = (s.addr, Arc::clone(&order));
        handles.push(std::thread::spawn(move || {
            let mut body = completion("hi", 1);
            body["priority"] = json!(priority);
            let r = post(addr, "/v1/completions", &body);
            assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
            order.lock().unwrap().push(priority);
        }));
        wait_queued(&gw, n + 1);
    }
    drop(holder);
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(*order.lock().unwrap(), ["high", "normal", "low"]);
    all_clear(&gw, M);
}

#[test]
fn a_request_deadline_ends_it_in_the_queue_or_while_running() {
    let (gw, s, holder) = occupied(ServerConfig::default());
    // In the queue: nothing started, so it is an error.
    let mut body = completion("hi", 2);
    body["timeout"] = json!(0.1);
    let t = Instant::now();
    let r = post(s.addr, "/v1/completions", &body);
    assert_eq!(r.status, 504, "{}", String::from_utf8_lossy(&r.body));
    assert_eq!(r.json()["error"]["code"], "deadline_exceeded");
    assert!(t.elapsed() < Duration::from_secs(5));
    assert_eq!(gw.queue_stats(M).unwrap().queued, 0);
    drop(holder);
    all_clear(&gw, M);
    // Running: it ends as if it reached max_tokens.
    let mut body = completion("Hello, experts", 200);
    body["timeout"] = json!(0.2);
    let r = post(s.addr, "/v1/completions", &body);
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    let j = r.json();
    assert_eq!(j["choices"][0]["finish_reason"], "length");
    assert!(j["usage"]["completion_tokens"].as_u64().unwrap() < 200);
    all_clear(&gw, M);
    // Nonsense is refused.
    for bad in [json!(0), json!(-1), json!(1e300)] {
        let mut body = completion("hi", 2);
        body["timeout"] = bad;
        assert_eq!(post(s.addr, "/v1/completions", &body).status, 400);
    }
    let mut body = completion("hi", 2);
    body["priority"] = json!("urgent");
    let r = post(s.addr, "/v1/completions", &body);
    assert_eq!(
        (r.status, r.json()["error"]["param"].clone()),
        (400, json!("priority"))
    );
}

#[test]
fn a_client_that_goes_away_leaves_the_queue() {
    let (gw, s, holder) = occupied(ServerConfig::default());
    let body = completion("hi", 2);
    let mut c = TcpStream::connect(s.addr).unwrap();
    c.write_all(request(s.addr, "POST", "/v1/completions", &[], Some(&body)).as_bytes())
        .unwrap();
    wait_queued(&gw, 1);
    drop(c);
    wait_queued(&gw, 0);
    assert_eq!(gw.metrics().cancelled(), 0, "it never ran");
    drop(holder);
    all_clear(&gw, M);
}

/// The slot is free by the time the client hears the end, so a client that sends its
/// next request straight away never finds its own last request in the way.
#[test]
fn the_next_request_finds_the_slot_free() {
    let gw = Gateway::new(ServerConfig {
        max_concurrent: 1,
        max_queue: 0,
        ..ServerConfig::default()
    });
    gw.add_model(M, engine(M)).unwrap();
    let s = start(&gw);
    for i in 0..100 {
        let mut body = completion("hi", 1);
        body["stream"] = json!(i % 2 == 0);
        let r = post(s.addr, "/v1/completions", &body);
        assert_eq!(r.status, 200, "{i}: {}", String::from_utf8_lossy(&r.body));
    }
    all_clear(&gw, M);
}

/// Room for four whole-context conversations' KV, as in `gateway.rs`.
fn four_conversations_budget() -> u64 {
    let fp = qubivex_memory::ModelFootprint::from_metadata(&meta(M)).unwrap();
    fp.dense_bytes().0
        + qubivex_memory::ACTIVATION_RESERVE.0
        + fp.cache_floor().0
        + 1024 * fp.kv_bytes_per_token.0
}

/// §12.3 test 10, with the queue policy: a request waits for memory, then runs.
#[test]
fn a_request_waits_for_memory_held_by_others() {
    let gw = Gateway::new(ServerConfig {
        max_concurrent: 8,
        ..ServerConfig::default()
    });
    let budget = four_conversations_budget();
    gw.add_model(M, slow_engine(M, Duration::from_millis(20), Some(budget)))
        .unwrap();
    let s = start(&gw);
    let mut body = completion("Hello", 250);
    body["stream"] = json!(true);
    let mut held: Vec<_> = (0..4).map(|_| open_stream(s.addr, &body).0).collect();
    let addr = s.addr;
    let fifth = std::thread::spawn(move || post(addr, "/v1/completions", &completion("Hello", 2)));
    wait_queued(&gw, 1);
    assert_eq!(gw.queue_stats(M).unwrap().running, 4);
    drop(held.pop());
    let r = fifth.join().unwrap();
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    drop(held);
    all_clear(&gw, M);
}

/// What can never run is refused without waiting for a slot it would not use.
#[test]
fn a_request_past_the_context_does_not_wait() {
    let (gw, s, holder) = occupied(ServerConfig::default());
    let t = Instant::now();
    let r = post(
        s.addr,
        "/v1/completions",
        &completion(&"word ".repeat(100), 2),
    );
    assert_eq!(r.status, 400, "{}", String::from_utf8_lossy(&r.body));
    assert!(t.elapsed() < Duration::from_secs(5));
    drop(holder);
    all_clear(&gw, M);
}

/// §12.3 test 6. Concurrent loads share one load, and a generation that arrives during
/// it waits rather than failing.
#[test]
fn concurrent_loads_of_a_model_share_one_load() {
    let gw = Gateway::new(ServerConfig {
        admin_token: Some("t".into()),
        model_dirs: vec![root()],
        ..ServerConfig::default()
    });
    let s = start(&gw);
    let admin = [("authorization", "Bearer t")];
    let go = Arc::new(Barrier::new(8));
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let (addr, go) = (s.addr, Arc::clone(&go));
            std::thread::spawn(move || {
                go.wait();
                let load = json!({ "model": M });
                let admin = [("authorization", "Bearer t")];
                http(addr, "POST", "/v1/models/load", &admin, Some(&load)).status
            })
        })
        .collect();
    let chat = {
        let addr = s.addr;
        let gw = gw.clone();
        std::thread::spawn(move || {
            wait_until("the load to start", || gw.metrics().model_loads() > 0);
            post(addr, "/v1/completions", &completion("hi", 2))
        })
    };
    let mut statuses: Vec<u16> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    statuses.sort_unstable();
    assert_eq!(statuses, [200, 200, 200, 200, 200, 200, 200, 201]);
    assert_eq!(gw.metrics().model_loads(), 1);
    assert_eq!(gw.model_ids(), [M]);
    let r = chat.join().unwrap();
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    // Two ids for one model load it once.
    let alias = json!({ "model": M, "id": "another-name" });
    let r = http(s.addr, "POST", "/v1/models/load", &admin, Some(&alias));
    assert_eq!(r.status, 409);
    assert_eq!(gw.model_ids(), [M]);
    // A load that fails tells every waiter and leaves no state behind.
    let missing = json!({ "model": "no-such-model" });
    let r = http(s.addr, "POST", "/v1/models/load", &admin, Some(&missing));
    assert_eq!(r.status, 404);
    let r = http(s.addr, "GET", "/v1/system", &admin, None);
    assert_eq!(r.status, 200);
    assert_eq!(
        r.json()["model_states"],
        json!([{ "id": M, "state": "loaded" }])
    );
}

/// §12.3 test 9, with a live stream and a waiter.
#[test]
fn unloading_lets_running_requests_finish() {
    let gw = Gateway::new(ServerConfig {
        admin_token: Some("t".into()),
        max_concurrent: 1,
        ..ServerConfig::default()
    });
    gw.add_model(M, slow_engine(M, Duration::from_millis(5), None))
        .unwrap();
    let s = start(&gw);
    let admin = [("authorization", "Bearer t")];
    let prompt = "Hello, experts";
    let want = reference(&engine(M), prompt, 40);
    let mut body = completion(prompt, 40);
    body["stream"] = json!(true);
    let (c, seen) = open_stream(s.addr, &body);
    let addr = s.addr;
    let waiter = std::thread::spawn(move || post(addr, "/v1/completions", &completion("hi", 2)));
    wait_queued(&gw, 1);
    let unload = json!({ "model": M });
    let r = http(s.addr, "POST", "/v1/models/unload", &admin, Some(&unload));
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    assert_eq!(r.json()["draining"], 1);
    assert_eq!(gw.draining_models(), 1);
    assert!(gw.model_ids().is_empty());
    // The waiter ends now, and new requests find no model.
    assert_eq!(waiter.join().unwrap().status, 503);
    assert_eq!(
        post(s.addr, "/v1/completions", &completion("hi", 2)).status,
        404
    );
    // The running request finishes, whole and correct.
    let events = read_to_end(c, seen);
    assert_eq!(events.last().unwrap(), "[DONE]");
    let text: String = events
        .iter()
        .filter(|e| *e != "[DONE]")
        .filter_map(|e| serde_json::from_str::<Value>(e).ok())
        .filter_map(|c| c["choices"][0]["text"].as_str().map(str::to_owned))
        .collect();
    assert_eq!(text, want);
    // Then the weights go.
    wait_until("the weights to be freed", || gw.draining_models() == 0);
}

fn stop(mut s: Server) -> Duration {
    let t = Instant::now();
    let _ = s.stop.take().unwrap().send(());
    s.thread.take().unwrap().join().unwrap();
    t.elapsed()
}

/// §12.3 test 11. Within the grace period, running requests finish and waiters are told.
#[test]
fn shutdown_lets_running_requests_finish_and_ends_the_queue() {
    let (gw, s, holder) = occupied(ServerConfig {
        shutdown_grace: Duration::from_secs(60),
        ..ServerConfig::default()
    });
    let addr = s.addr;
    let waiter = std::thread::spawn(move || post(addr, "/v1/completions", &completion("hi", 2)));
    wait_queued(&gw, 1);
    let reader = std::thread::spawn(move || read_to_end(holder, Vec::new()));
    let stopped = std::thread::spawn(move || stop(s));
    assert_eq!(waiter.join().unwrap().status, 503);
    let events = reader.join().unwrap();
    assert_eq!(events.last().unwrap(), "[DONE]");
    let finish = events
        .iter()
        .rev()
        .nth(1)
        .and_then(|e| serde_json::from_str::<Value>(e).ok())
        .unwrap();
    assert_eq!(finish["choices"][0]["finish_reason"], "length");
    assert!(stopped.join().unwrap() < Duration::from_secs(55));
    assert_eq!(
        gw.metrics().active(),
        0,
        "serve returned before its workers"
    );
    all_clear(&gw, M);
    // The port is closed.
    assert!(TcpStream::connect(addr).is_err());
}

/// §12.3 test 11. Past the grace period, running requests stop and say why.
#[test]
fn shutdown_stops_what_outlasts_the_grace() {
    let gw = Gateway::new(ServerConfig {
        max_concurrent: 2,
        shutdown_grace: Duration::from_millis(100),
        ..ServerConfig::default()
    });
    gw.add_model(M, slow_engine(M, Duration::from_millis(20), None))
        .unwrap();
    let s = start(&gw);
    let mut body = completion("Hello, experts", 240);
    body["stream"] = json!(true);
    let (c, seen) = open_stream(s.addr, &body);
    let addr = s.addr;
    let whole = std::thread::spawn(move || {
        post(addr, "/v1/completions", &completion("Hello, experts", 240))
    });
    wait_until("both to run", || gw.metrics().active() == 2);
    let took = stop(s);
    assert!(took < Duration::from_secs(4), "{took:?}");
    assert_eq!(
        gw.metrics().active(),
        0,
        "serve returned before its workers"
    );
    let events = read_to_end(c, seen);
    assert_eq!(events.last().unwrap(), "[DONE]");
    let error: Value = serde_json::from_str(&events[events.len() - 2]).unwrap();
    assert_eq!(error["error"]["code"], "overloaded");
    let r = whole.join().unwrap();
    assert_eq!(r.status, 503, "{}", String::from_utf8_lossy(&r.body));
    all_clear(&gw, M);
}

/// §12.3 test 12. Short and long conversations together, each keeping its own KV.
#[test]
fn concurrent_requests_with_different_context_lengths() {
    let gw = Gateway::new(ServerConfig {
        max_concurrent: 4,
        ..ServerConfig::default()
    });
    gw.add_model(M, engine(M)).unwrap();
    let s = start(&gw);
    let e = engine(M);
    let long = "word ".repeat(30);
    let cases = [
        ("a", 1),
        ("Hello, experts", 30),
        (long.as_str(), 5),
        ("The quick brown fox", 120),
        (long.as_str(), 60),
        ("abc", 2),
    ];
    let want: Vec<String> = cases.iter().map(|(p, n)| reference(&e, p, *n)).collect();
    for _round in 0..3 {
        let handles: Vec<_> = cases
            .iter()
            .enumerate()
            .map(|(i, (p, n))| {
                let (addr, mut body) = (s.addr, completion(p, *n));
                body["stream"] = json!(i % 2 == 0);
                std::thread::spawn(move || {
                    let r = post(addr, "/v1/completions", &body);
                    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
                    if body["stream"] == json!(true) {
                        streamed_text(&r)
                    } else {
                        text_of(&r.json())
                    }
                })
            })
            .collect();
        let got: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(got, want);
    }
    all_clear(&gw, M);
}

/// §12.3 test 13. Two models at once: right results, and one model's full queue does
/// not touch the other.
#[test]
fn several_models_share_the_machine() {
    let gw = Gateway::new(ServerConfig {
        max_concurrent: 1,
        max_queue: 0,
        ..ServerConfig::default()
    });
    let dense = "dense-tiny";
    gw.add_model(M, slow_engine(M, Duration::from_millis(5), None))
        .unwrap();
    gw.add_model(dense, engine(dense)).unwrap();
    let s = start(&gw);
    let mut body = completion("Hello, experts", 200);
    body["stream"] = json!(true);
    let (holder, _) = open_stream(s.addr, &body);
    // The first model is full.
    assert_eq!(
        post(s.addr, "/v1/completions", &completion("hi", 2)).status,
        429
    );
    // The second is not, and answers as it would alone.
    let want = reference(&engine(dense), "Hello", 12);
    let mut other = completion("Hello", 12);
    other["model"] = json!(dense);
    for _ in 0..3 {
        let r = post(s.addr, "/v1/completions", &other);
        assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
        assert_eq!(text_of(&r.json()), want);
    }
    drop(holder);
    all_clear(&gw, M);
    all_clear(&gw, dense);
    // Both at once, side by side.
    let want_olmoe = reference(&engine(M), "abc abc abc", 12);
    let mut handles = Vec::new();
    for id in [M, dense, M, dense] {
        let addr = s.addr;
        let mut body = completion(if id == M { "abc abc abc" } else { "Hello" }, 12);
        body["model"] = json!(id);
        handles.push(std::thread::spawn(move || {
            (
                id,
                loop {
                    let r = post(addr, "/v1/completions", &body);
                    if r.status != 429 {
                        break r;
                    }
                },
            )
        }));
    }
    for h in handles {
        let (id, r) = h.join().unwrap();
        assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
        let expected = if id == M { &want_olmoe } else { &want };
        assert_eq!(&text_of(&r.json()), expected);
    }
    all_clear(&gw, M);
    all_clear(&gw, dense);
}
