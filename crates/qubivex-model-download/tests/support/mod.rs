//! A fake model hub on a local socket, written against raw TCP so tests control every
//! byte: where a body is cut, whether a range is honoured, and where a redirect goes.

#![allow(dead_code, clippy::unwrap_used)]

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sha1::Sha1;
use sha2::{Digest, Sha256};

pub const SHA: &str = "0123456789abcdef0123456789abcdef01234567";
pub const REPO: &str = "acme/dense-tiny";

/// One request as the server saw it.
#[derive(Clone, Debug)]
pub struct Seen {
    pub path: String,
    pub host: String,
    pub range: Option<u64>,
    pub auth: Option<String>,
}

/// What to do to the next download of a file.
#[derive(Clone, Debug)]
pub enum Fault {
    /// Reply with this status.
    Status(u16),
    /// Reply with this status and a `Retry-After` of this many seconds.
    Limited(u16, u64),
    /// Send this many bytes of the body, then close.
    Cut(usize),
    /// Send a 200 from the first byte, whatever was asked for.
    IgnoreRange,
    /// Flip a byte of the body.
    Corrupt,
    /// Send more bytes than the listing gives.
    Oversize,
    /// Send this many bytes, then hold the connection open for this long.
    Stall(usize, Duration),
    /// Redirect to this URL.
    Redirect(String),
    /// Send a 206 for an offset that was not asked for.
    WrongRange,
}

#[derive(Clone)]
pub struct File {
    pub path: String,
    pub bytes: Vec<u8>,
    pub lfs: bool,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn git_oid(bytes: &[u8]) -> String {
    let mut h = Sha1::new();
    h.update(format!("blob {}\0", bytes.len()));
    h.update(bytes);
    hex(&h.finalize())
}

/// Reads a fixture model's files; weights are LFS, as on a real hub.
pub fn fixture(name: &str) -> Vec<File> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/models")
        .join(name);
    let mut files: Vec<File> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| e.file_name() != "manifest.json")
        .map(|e| {
            let path = e.file_name().to_string_lossy().into_owned();
            File {
                lfs: path.ends_with(".safetensors") || path.ends_with(".gguf"),
                bytes: std::fs::read(e.path()).unwrap(),
                path,
            }
        })
        .collect();
    files.sort_by(|a, b| a.path.cmp(&b.path));
    files
}

pub struct State {
    pub repo: String,
    pub sha: Mutex<String>,
    pub files: Mutex<Vec<File>>,
    pub faults: Mutex<Vec<(String, VecDeque<Fault>)>>,
    pub api_faults: Mutex<VecDeque<u16>>,
    pub seen: Mutex<Vec<Seen>>,
    pub next_link: Mutex<Option<String>>,
    pub active: AtomicUsize,
    pub peak: AtomicUsize,
    /// Pacing per 16 KiB of a body, so concurrency can overlap.
    pub pace: Mutex<Duration>,
}

pub struct Hub {
    pub url: String,
    pub port: u16,
    pub state: Arc<State>,
}

impl Hub {
    pub fn start(files: Vec<File>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let state = Arc::new(State {
            repo: REPO.to_owned(),
            sha: Mutex::new(SHA.to_owned()),
            files: Mutex::new(files),
            faults: Mutex::new(Vec::new()),
            api_faults: Mutex::new(VecDeque::new()),
            seen: Mutex::new(Vec::new()),
            next_link: Mutex::new(None),
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            pace: Mutex::new(Duration::ZERO),
        });
        let shared = state.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let (stream, state) = (stream.unwrap(), shared.clone());
                std::thread::spawn(move || {
                    let _ = serve(stream, &state);
                });
            }
        });
        Self {
            url: format!("http://127.0.0.1:{port}"),
            port,
            state,
        }
    }

    pub fn fault(&self, path: &str, faults: Vec<Fault>) {
        self.state
            .faults
            .lock()
            .unwrap()
            .push((path.to_owned(), faults.into()));
    }

    pub fn seen(&self) -> Vec<Seen> {
        self.state.seen.lock().unwrap().clone()
    }

    pub fn downloads(&self, path: &str) -> Vec<Seen> {
        let tail = format!("/{path}");
        self.seen()
            .into_iter()
            .filter(|s| {
                s.path.contains("/resolve/")
                    && s.path.split('?').next().is_some_and(|p| p.ends_with(&tail))
            })
            .collect()
    }

    pub fn file(&self, path: &str) -> Vec<u8> {
        let files = self.state.files.lock().unwrap();
        files.iter().find(|f| f.path == path).unwrap().bytes.clone()
    }
}

fn respond(s: &mut TcpStream, status: u16, headers: &str, body: &[u8]) -> std::io::Result<()> {
    write!(
        s,
        "HTTP/1.1 {status} X\r\nConnection: close\r\nContent-Length: {}\r\n{headers}\r\n",
        body.len()
    )?;
    s.write_all(body)
}

fn listing(state: &State) -> Vec<u8> {
    let files = state.files.lock().unwrap();
    let entries: Vec<serde_json::Value> = files
        .iter()
        .map(|f| {
            if f.lfs {
                let oid = hex(&Sha256::digest(&f.bytes));
                serde_json::json!({"type":"file","path":f.path,"size":f.bytes.len(),
                    "oid":"f".repeat(40),"lfs":{"oid":oid,"size":f.bytes.len(),"pointerSize":134}})
            } else {
                serde_json::json!({"type":"file","path":f.path,"size":f.bytes.len(),
                    "oid":git_oid(&f.bytes)})
            }
        })
        .collect();
    serde_json::to_vec(&entries).unwrap()
}

fn body(s: &mut TcpStream, state: &State, bytes: &[u8]) -> std::io::Result<()> {
    let pace = *state.pace.lock().unwrap();
    if pace.is_zero() {
        return s.write_all(bytes);
    }
    for chunk in bytes.chunks(16 << 10) {
        s.write_all(chunk)?;
        std::thread::sleep(pace);
    }
    Ok(())
}

fn download(
    s: &mut TcpStream,
    state: &State,
    path: &str,
    range: Option<u64>,
) -> std::io::Result<()> {
    let file = state
        .files
        .lock()
        .unwrap()
        .iter()
        .find(|f| path.ends_with(&format!("/{}", f.path)))
        .cloned();
    let Some(file) = file else {
        return respond(s, 404, "", b"");
    };
    let fault = state
        .faults
        .lock()
        .unwrap()
        .iter_mut()
        .find(|(p, _)| *p == file.path)
        .and_then(|(_, q)| q.pop_front());
    let start = usize::try_from(range.unwrap_or(0)).unwrap();
    let total = file.bytes.len();
    let headers = |from: usize| {
        if range.is_some() {
            format!("Content-Range: bytes {from}-{}/{total}\r\n", total - 1)
        } else {
            String::new()
        }
    };
    let status = if range.is_some() { 206 } else { 200 };
    let rest = &file.bytes[start.min(total)..];
    match fault {
        Some(Fault::Status(c)) => respond(s, c, "", b"nope"),
        Some(Fault::Limited(c, secs)) => {
            respond(s, c, &format!("Retry-After: {secs}\r\n"), b"slow down")
        }
        Some(Fault::Redirect(to)) => respond(s, 302, &format!("Location: {to}\r\n"), b""),
        Some(Fault::IgnoreRange) => respond(s, 200, "", &file.bytes),
        Some(Fault::WrongRange) => {
            respond(s, 206, "Content-Range: bytes 1-2/9\r\n", &file.bytes[1..3])
        }
        Some(Fault::Corrupt) => {
            let mut b = rest.to_vec();
            let i = b.len() / 2;
            b[i] ^= 0xff;
            respond(s, status, &headers(start), &b)
        }
        Some(Fault::Oversize) => {
            let mut b = rest.to_vec();
            b.extend_from_slice(&[7; 4096]);
            respond(s, status, &headers(start), &b)
        }
        Some(Fault::Cut(n)) => {
            write!(
                s,
                "HTTP/1.1 {status} X\r\nConnection: close\r\nContent-Length: {}\r\n{}\r\n",
                rest.len(),
                headers(start)
            )?;
            s.write_all(&rest[..n.min(rest.len())])
        }
        Some(Fault::Stall(n, d)) => {
            write!(
                s,
                "HTTP/1.1 {status} X\r\nConnection: close\r\nContent-Length: {}\r\n{}\r\n",
                rest.len(),
                headers(start)
            )?;
            s.write_all(&rest[..n.min(rest.len())])?;
            s.flush()?;
            std::thread::sleep(d);
            Ok(())
        }
        None => {
            write!(
                s,
                "HTTP/1.1 {status} X\r\nConnection: close\r\nContent-Length: {}\r\n{}\r\n",
                rest.len(),
                headers(start)
            )?;
            let n = state.active.fetch_add(1, Ordering::SeqCst) + 1;
            state.peak.fetch_max(n, Ordering::SeqCst);
            let out = body(s, state, rest);
            state.active.fetch_sub(1, Ordering::SeqCst);
            out
        }
    }
}

fn serve(mut stream: TcpStream, state: &State) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let path = line.split(' ').nth(1).unwrap_or("").to_owned();
    let (mut range, mut auth, mut host) = (None, None, String::new());
    loop {
        let mut h = String::new();
        reader.read_line(&mut h)?;
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        let (k, v) = h.split_once(':').unwrap_or((h, ""));
        match k.to_ascii_lowercase().as_str() {
            "range" => {
                range = v
                    .trim()
                    .strip_prefix("bytes=")
                    .and_then(|r| r.trim_end_matches('-').parse().ok());
            }
            "authorization" => auth = Some(v.trim().to_owned()),
            "host" => host = v.trim().to_owned(),
            _ => {}
        }
    }
    state.seen.lock().unwrap().push(Seen {
        path: path.clone(),
        host,
        range,
        auth,
    });
    if path.starts_with("/api/") {
        if let Some(code) = state.api_faults.lock().unwrap().pop_front() {
            return respond(&mut stream, code, "", b"");
        }
    }
    let api = format!("/api/models/{}", state.repo);
    if path.starts_with(&format!("{api}/tree/")) {
        let link = state.next_link.lock().unwrap().take();
        let headers = link
            .map(|l| format!("Link: <{l}>; rel=\"next\"\r\n"))
            .unwrap_or_default();
        return respond(&mut stream, 200, &headers, &listing(state));
    }
    if path.starts_with(&format!("{api}/revision/")) {
        let sha = state.sha.lock().unwrap().clone();
        let info = serde_json::json!({ "id": state.repo, "sha": sha });
        return respond(&mut stream, 200, "", &serde_json::to_vec(&info).unwrap());
    }
    if path.starts_with("/api/models?") {
        let hits = serde_json::json!([
            {"id":"acme/a-GGUF","downloads":5,"likes":2,"tags":["gguf"],"gated":false},
            {"id":"acme/b","downloads":3,"likes":1,"tags":["safetensors"],"gated":"manual"},
        ]);
        return respond(&mut stream, 200, "", &serde_json::to_vec(&hits).unwrap());
    }
    if path.contains("/resolve/") {
        let file = path.split('?').next().unwrap_or(&path);
        return download(&mut stream, state, file, range);
    }
    respond(&mut stream, 404, "", b"")
}
