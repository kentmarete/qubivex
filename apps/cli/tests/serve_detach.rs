//! `qubivex serve --detach`, `--status` and `--stop` through the real binary, serving a
//! fixture model on a free loopback port with a scratch HOME.
#![cfg(unix)]
#![allow(clippy::unwrap_used)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Output};

struct Home(PathBuf);

impl Home {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "qubivex-cli-serve-detach-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn qubivex(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_qubivex"))
            .arg("serve")
            .args(args)
            .env("HOME", &self.0)
            .env_remove("QUBIVEX_API_KEY")
            .env_remove("QUBIVEX_ADMIN_TOKEN")
            .env_remove("QUBIVEX_CONFIG")
            .output()
            .unwrap()
    }
}

impl Drop for Home {
    fn drop(&mut self) {
        // Never leave a server behind, whatever the test did.
        let _ = self.qubivex(&["--stop"]);
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixtures() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/models")
        .display()
        .to_string()
}

fn free_addr() -> String {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().to_string()
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

fn health(addr: &str) -> String {
    let mut s = TcpStream::connect(addr).unwrap();
    write!(
        s,
        "GET /health HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    out
}

#[test]
fn detach_status_and_stop() {
    let home = Home::new("cycle");
    let addr = free_addr();
    let dir = fixtures();
    let serve = ["--model", "olmoe-tiny", "--dir", &dir, "--listen", &addr];
    let out = home.qubivex(&[&serve[..], &["--detach"]].concat());
    assert!(out.status.success(), "{out:?}");
    assert!(
        text(&out.stdout).contains("serving in the background"),
        "{out:?}"
    );
    // It answers as soon as --detach returns.
    assert!(
        health(&addr).starts_with("HTTP/1.1 200"),
        "{}",
        health(&addr)
    );

    // A second server on the same port is refused before it starts.
    let again = home.qubivex(&[&serve[..], &["--detach"]].concat());
    assert_eq!(again.status.code(), Some(1));
    assert!(text(&again.stderr).contains("already running"), "{again:?}");

    let st = home.qubivex(&["--status"]);
    assert!(st.status.success(), "{st:?}");
    let s = text(&st.stdout);
    assert!(s.contains("running (pid") && s.contains("serving"), "{s}");

    let stop = home.qubivex(&["--stop", "--listen", &addr]);
    assert!(stop.status.success(), "{stop:?}");
    assert!(text(&stop.stdout).contains("stopped 1 server"), "{stop:?}");
    assert!(TcpStream::connect(&addr).is_err(), "the port is closed");
    let st = home.qubivex(&["--status"]);
    assert!(text(&st.stdout).contains("no background server"), "{st:?}");
}

#[test]
fn a_server_that_fails_to_start_reports_its_log() {
    let home = Home::new("fail");
    let addr = free_addr();
    let dir = fixtures();
    let out = home.qubivex(&[
        "--model",
        "no-such-model",
        "--dir",
        &dir,
        "--listen",
        &addr,
        "--detach",
    ]);
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    assert!(
        text(&out.stderr).contains("exited while starting"),
        "{out:?}"
    );
    assert!(
        text(&out.stdout).contains("no-such-model"),
        "the log is shown: {out:?}"
    );
}

#[test]
fn a_taken_port_is_refused_before_starting() {
    let home = Home::new("taken");
    let held = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = held.local_addr().unwrap().to_string();
    let dir = fixtures();
    let out = home.qubivex(&[
        "--model",
        "olmoe-tiny",
        "--dir",
        &dir,
        "--listen",
        &addr,
        "--detach",
    ]);
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    assert!(text(&out.stderr).contains("cannot listen"), "{out:?}");
}

#[test]
fn stop_and_status_with_nothing_running() {
    let home = Home::new("none");
    let out = home.qubivex(&["--stop"]);
    assert!(out.status.success(), "{out:?}");
    assert!(
        text(&out.stdout).contains("no background server"),
        "{out:?}"
    );
    let out = home.qubivex(&["--status"]);
    assert!(out.status.success(), "{out:?}");
    assert!(
        text(&out.stdout).contains("no background server"),
        "{out:?}"
    );
}

#[test]
fn detach_and_stop_conflict() {
    let home = Home::new("conflict");
    let out = home.qubivex(&["--detach", "--stop"]);
    assert_eq!(out.status.code(), Some(2));
}
