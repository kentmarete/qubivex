//! An external process takes RAM while the real model generates (MIGRATION_PLAN
//! Phase 5 exit). The pressure monitor must refuse new requests and drop the cache to
//! its floor, while the active request runs to the end with unchanged tokens; when the
//! RAM comes back, so does normal service.
//!
//! Opt-in: it puts the machine under real memory pressure for a few seconds. Set
//! QUBIVEX_OLMOE_GGUF_DIR and QUBIVEX_PRESSURE_TEST=1. The hog takes half of the RAM
//! available after loading, and the thresholds are set relative to that, so the machine
//! never gets near running out.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use qubivex_core::Bytes;
use qubivex_hardware::{Host, ResourceProbe, SystemProbe};
use qubivex_memory::{PressureConfig, PressureLevel};
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{Engine, EngineOptions, Sampling};

const PROMPT: [u32; 5] = [510, 5347, 273, 6181, 310];
const TOKENS: usize = 40;

fn available() -> u64 {
    SystemProbe::new(Host::live())
        .snapshot()
        .unwrap()
        .ram_available
        .known()
        .unwrap()
        .0
}

/// The hog process: killed and reaped on drop, so a failing assertion never leaves
/// it holding RAM.
struct Hog(Child);

impl Drop for Hog {
    fn drop(&mut self) {
        // Already exited is fine; there is nothing else to do in a test teardown.
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Runs this test binary's `hog_helper` in a new process holding `bytes`.
fn spawn_hog(bytes: u64) -> Hog {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "hog_helper",
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("QUBIVEX_HOG_BYTES", bytes.to_string())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let out = BufReader::new(child.stdout.take().unwrap());
    let hog = Hog(child);
    for line in out.lines() {
        if line.unwrap().contains("HOG READY") {
            return hog;
        }
    }
    panic!("the hog exited before allocating");
}

/// Waits up to 20 s for `pred` on the engine's pressure decision.
fn wait_for(e: &Engine, pred: impl Fn(PressureLevel) -> bool) -> PressureLevel {
    let t = Instant::now();
    loop {
        let level = e.pressure().unwrap().level;
        if pred(level) || t.elapsed() > Duration::from_secs(20) {
            return level;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn ram_hog_triggers_the_responses_in_order() {
    let (Ok(dir), Ok(_)) = (
        std::env::var("QUBIVEX_OLMOE_GGUF_DIR"),
        std::env::var("QUBIVEX_PRESSURE_TEST"),
    ) else {
        return;
    };
    let meta =
        ModelMetadata::load(Path::new(&dir), &ArchitectureRegistry::with_builtins()).unwrap();
    let e = Engine::load(
        meta,
        EngineOptions {
            max_context: Some(512),
            pressure_interval_ms: Some(100),
            ..EngineOptions::default()
        },
    )
    .unwrap();
    // Reference tokens, which also bring every used expert into the cache.
    let reference = e
        .generate(&PROMPT, TOKENS, Sampling::default(), |_| true)
        .unwrap();
    let planned = e.expert_cache_bytes();
    let floor = e.memory_plan().unwrap().ram.cache_floor.bytes.0;

    let a = available();
    let hog = a / 2;
    assert!(
        hog >= 256 << 20,
        "too little RAM available ({}) to test",
        Bytes(a)
    );
    // After the hog, about a/2 is available: below `critical`.
    e.set_pressure_config(PressureConfig {
        elevated: Bytes(a - hog / 8),
        high: Bytes(a - hog / 4),
        critical: Bytes(a - hog / 2),
        hysteresis: Bytes(hog / 16),
    });
    eprintln!("available {}, hog {}", Bytes(a), Bytes(hog));

    let mut child: Option<Hog> = None;
    let mut seen_critical = false;
    let mut refused = false;
    let mut caps = Vec::new();
    let got = e
        .generate(&PROMPT, TOKENS, Sampling::default(), |_| {
            caps.push(e.expert_cache_bytes());
            if caps.len() == 10 {
                child = Some(spawn_hog(hog));

                seen_critical =
                    wait_for(&e, |l| l == PressureLevel::Critical) == PressureLevel::Critical;
                // While critical, a new request is refused; this one goes on.
                refused = e.admit().is_err();
                eprintln!("with the hog: available {}", Bytes(available()));
            }
            true
        })
        .unwrap();
    let child = child.unwrap();
    assert!(seen_critical, "the monitor saw the hog");
    assert!(refused, "a new request was refused under critical pressure");
    assert_eq!(
        got, reference,
        "the active request finished with the same tokens"
    );
    let min = *caps.iter().min().unwrap();
    assert_eq!(
        min,
        floor,
        "the cache dropped to its floor ({} vs {})",
        Bytes(min),
        Bytes(floor)
    );

    drop(child); // kills and reaps the hog
    let level = wait_for(&e, |l| l == PressureLevel::Normal);
    assert_eq!(
        level,
        PressureLevel::Normal,
        "pressure cleared after the hog exited"
    );
    let again = e
        .generate(&PROMPT, TOKENS, Sampling::default(), |_| true)
        .unwrap();
    assert_eq!(again, reference);
    assert_eq!(e.expert_cache_bytes(), planned, "the plan's cache is back");
}

/// Not a test: the RAM hog `ram_hog_triggers_the_responses_in_order` runs in a child
/// process. Holds QUBIVEX_HOG_BYTES of touched memory until killed (or 120 s).
#[test]
#[ignore = "helper process for the RAM-hog test"]
fn hog_helper() {
    let Ok(n) = std::env::var("QUBIVEX_HOG_BYTES") else {
        return;
    };
    let n: usize = n.parse().unwrap();
    // Pseudo-random bytes: uniform pages would be squeezed to nothing by the macOS
    // memory compressor and barely lower available RAM.
    let mut hog = vec![0u64; n / 8];
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    for w in &mut hog {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *w = x;
    }
    println!("HOG READY {}", hog.len() * 8);
    std::thread::sleep(Duration::from_secs(120));
    drop(hog);
}
