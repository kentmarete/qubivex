//! The gateway's test harness: tiny fixture engines, a real server on a local port and a
//! few lines of blocking HTTP/1.1 over `TcpStream`.

#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use qubivex_api::Gateway;
use qubivex_expert_stream::ExpertSource;
use qubivex_expert_stream::fault::{Fault, FaultySource, Rule, Target};
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{Engine, EngineOptions, Sampling};
use serde_json::Value;

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/models")
}

pub fn meta(id: &str) -> ModelMetadata {
    ModelMetadata::load(&root().join(id), &ArchitectureRegistry::with_builtins()).unwrap()
}

pub fn opts() -> EngineOptions {
    EngineOptions {
        max_context: Some(256),
        pressure_interval_ms: Some(0),
        ..EngineOptions::default()
    }
}

pub fn engine(id: &str) -> Engine {
    Engine::load(meta(id), opts()).unwrap()
}

/// An engine whose expert reads each take `delay` and whose cache holds one expert per
/// layer, so every token reads from the (slow) source.
pub fn slow_engine(id: &str, delay: Duration, ram_budget: Option<u64>) -> Engine {
    let m = meta(id);
    let largest = m.bytes().largest_expert;
    let o = EngineOptions {
        expert_cache_bytes: ram_budget.is_none().then_some(largest),
        ram_budget,
        ..opts()
    };
    Engine::load_with_source(m, o, move |src| {
        let inner: Arc<dyn ExpertSource> = src;
        Arc::new(FaultySource::new(
            inner,
            [Rule::always(Target::Any, Fault::Delay(delay))],
        ))
    })
    .unwrap()
}

pub struct Server {
    pub addr: SocketAddr,
    pub stop: Option<tokio::sync::oneshot::Sender<()>>,
    pub thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(s) = self.stop.take() {
            let _ = s.send(());
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

pub fn start(gw: &Gateway) -> Server {
    let gw = gw.clone();
    let (addr_tx, addr_rx) = std::sync::mpsc::sync_channel(1);
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let thread = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            addr_tx.send(l.local_addr().unwrap()).unwrap();
            gw.serve(l, async {
                let _ = stop_rx.await;
            })
            .await
            .unwrap();
        });
    });
    Server {
        addr: addr_rx.recv().unwrap(),
        stop: Some(stop_tx),
        thread: Some(thread),
    }
}

pub struct Resp {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Resp {
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body)
            .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&self.body)))
    }
    pub fn header(&self, k: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(k))
            .map(|(_, v)| v.as_str())
    }
    /// The `data:` payloads of an SSE body.
    pub fn events(&self) -> Vec<String> {
        String::from_utf8_lossy(&self.body)
            .lines()
            .filter_map(|l| l.strip_prefix("data: ").map(str::to_owned))
            .collect()
    }
}

pub fn request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&Value>,
) -> String {
    let body = body.map(Value::to_string).unwrap_or_default();
    let mut req = format!(
        "{method} {path} HTTP/1.1\r\nhost: {addr}\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n",
        body.len()
    );
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");
    req.push_str(&body);
    req
}

pub fn dechunk(mut b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let Some(nl) = b.windows(2).position(|w| w == b"\r\n") else {
            return out;
        };
        let size =
            usize::from_str_radix(std::str::from_utf8(&b[..nl]).unwrap().trim(), 16).unwrap();
        if size == 0 {
            return out;
        }
        out.extend_from_slice(&b[nl + 2..nl + 2 + size]);
        b = &b[nl + 2 + size + 2..];
    }
}

pub fn http(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&Value>,
) -> Resp {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
    s.write_all(request(addr, method, path, headers, body).as_bytes())
        .unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let mut lines = head.lines();
    let status = lines
        .next()
        .unwrap()
        .split(' ')
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
        .collect();
    let rest = &raw[split + 4..];
    let chunked = headers
        .iter()
        .any(|(k, v)| k.eq_ignore_ascii_case("transfer-encoding") && v.contains("chunked"));
    let body = if chunked {
        dechunk(rest)
    } else {
        rest.to_vec()
    };
    Resp {
        status,
        headers,
        body,
    }
}

pub fn post(addr: SocketAddr, path: &str, body: &Value) -> Resp {
    http(addr, "POST", path, &[], Some(body))
}

pub fn text_of(r: &Value) -> String {
    let c = &r["choices"][0];
    c["message"]["content"]
        .as_str()
        .or_else(|| c["text"].as_str())
        .unwrap()
        .to_owned()
}

/// The text the engine generates greedily for `prompt` (as `/v1/completions` encodes it).
pub fn reference(e: &Engine, prompt: &str, n: usize) -> String {
    let ids = e.tokenizer().encode(prompt, true).unwrap();
    let toks = e.generate(&ids, n, Sampling::default(), |_| true).unwrap();
    e.tokenizer().decode(&toks, true).unwrap()
}

pub fn wait_until(what: &str, mut f: impl FnMut() -> bool) {
    let t = Instant::now();
    while !f() {
        assert!(
            t.elapsed() < Duration::from_secs(20),
            "timed out waiting for {what}"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}
