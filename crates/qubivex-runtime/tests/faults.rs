//! The engine under injected expert-read faults (IMPROVEMENTS_EVALUATION §10), on the
//! fixtures: EIO, short reads, a panicking source and corrupt bytes (caught by
//! checksums), during prefill and mid-decode, with and without I/O threads, with the
//! whole model resident and with experts streaming through one slot per layer.
//!
//! In every case:
//! - the request fails with an error naming the expert; nothing panics;
//! - every token emitted before the fault is the fault-free token (no wrong tokens);
//! - nothing stays leased, loading or failed, and the ledger matches the store;
//! - the next request gives exactly the fault-free tokens;
//! - a failed decode step leaves the KV cache as it was, so retrying the step on the
//!   same cache gives the fault-free logits, bit for bit.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use qubivex_arch::Logits;
use qubivex_expert_stream::ExpertSource;
use qubivex_expert_stream::fault::{Fault, FaultySource, Rule, Target, VerifiedSource};
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{BackendKind, Engine, EngineOptions, ModelExpertSource, Sampling};

const IDS: [&str; 3] = ["olmoe-tiny", "olmoe-tiny-gguf", "qwen3-moe-tiny"];
const PROMPT: &[u32] = &[
    72, 101, 108, 108, 111, 44, 32, 101, 120, 112, 101, 114, 116, 115,
];
const NEW: usize = 10;
/// The token of a decode step that fails, and of the step run after it.
const FAILED: u32 = 101;
const RETRY: u32 = 44;

fn meta(id: &str) -> ModelMetadata {
    let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/models")
        .join(id);
    ModelMetadata::load(&dir, &ArchitectureRegistry::with_builtins()).unwrap()
}

/// The two cache shapes: everything resident, or one slot per layer (streaming).
#[derive(Clone, Copy, Debug)]
enum Cache {
    Resident,
    Streaming,
}

fn options(m: &ModelMetadata, cache: Cache, io_threads: usize) -> EngineOptions {
    options_on(m, cache, io_threads, false)
}

fn options_on(m: &ModelMetadata, cache: Cache, io_threads: usize, metal: bool) -> EngineOptions {
    let b = m.bytes();
    EngineOptions {
        backend: if metal {
            BackendKind::Metal
        } else {
            BackendKind::Cpu
        },
        expert_cache_bytes: Some(match cache {
            Cache::Resident => b.routed_experts,
            Cache::Streaming => b.largest_expert * u64::from(m.config.dims.layers),
        }),
        max_context: Some(64),
        io_threads: Some(io_threads),
        threads: Some(2),
        ..EngineOptions::default()
    }
}

fn engine(
    id: &str,
    cache: Cache,
    io_threads: usize,
    wrap: impl FnOnce(Arc<ModelExpertSource>) -> Arc<dyn ExpertSource>,
) -> Engine {
    engine_on(id, cache, io_threads, false, wrap)
}

fn engine_on(
    id: &str,
    cache: Cache,
    io_threads: usize,
    metal: bool,
    wrap: impl FnOnce(Arc<ModelExpertSource>) -> Arc<dyn ExpertSource>,
) -> Engine {
    let m = meta(id);
    let opts = options_on(&m, cache, io_threads, metal);
    let e = Engine::load_with_source(m, opts, wrap).unwrap();
    if metal {
        let p = e.placement();
        assert!(
            p.fallback.is_none() && p.tensors > 0,
            "{id}: not on Metal: {p:?}"
        );
    }
    e
}

/// The fault-free tokens, and how many expert reads the prefill and the whole request
/// take.
fn reference(id: &str, cache: Cache, io_threads: usize) -> (Vec<u32>, u64, u64) {
    reference_on(id, cache, io_threads, false)
}

fn reference_on(id: &str, cache: Cache, io_threads: usize, metal: bool) -> (Vec<u32>, u64, u64) {
    let mut probe = None;
    let e = engine_on(id, cache, io_threads, metal, |s| {
        let f = Arc::new(FaultySource::new(s, []));
        probe = Some(Arc::clone(&f));
        f
    });
    let probe = probe.unwrap();
    let mut prefill = None;
    let tokens = e
        .generate(PROMPT, NEW, Sampling::default(), |_| {
            prefill.get_or_insert(probe.reads());
            true
        })
        .unwrap();
    assert_eq!(tokens.len(), NEW, "{id}: the reference must not stop early");
    (tokens, prefill.unwrap(), probe.reads())
}

/// Waits for loads still in flight after a failure, then checks nothing is held.
fn settled(e: &Engine, what: &str) {
    let t = Instant::now();
    loop {
        let o = e.store_occupancy().unwrap();
        if o.loading == 0 && o.queued == 0 {
            assert_eq!((o.leased, o.failed), (0, 0), "{what}: {o:?}");
            return;
        }
        assert!(
            t.elapsed() < Duration::from_secs(20),
            "{what}: never settled: {o:?}"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn ledger_matches_store(e: &Engine, what: &str) {
    let l = e.ledger().snapshot(qubivex_memory::Tier::Ram);
    let used = e.store_occupancy().unwrap().used;
    assert_eq!(l.expert_cache.0, used, "{what}: ledger vs store");
}

/// The default panic hook would print each injected (and caught) panic.
fn quiet_injected_panics() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = info
            .payload()
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| info.payload().downcast_ref::<&str>().copied())
            .unwrap_or("");
        if !msg.starts_with("injected:") {
            prev(info);
        }
    }));
}

/// The source the store reads through: the model's, with `fault` on read `at` (counted
/// from the first), behind checksums when the fault is corruption.
fn wrapped(fault: &Fault, at: u64) -> impl FnOnce(Arc<ModelExpertSource>) -> Arc<dyn ExpertSource> {
    let fault = fault.clone();
    move |src| {
        let verify = matches!(fault, Fault::Corrupt { .. });
        let sums = verify.then(|| VerifiedSource::record(&*src, src.keys()).unwrap());
        let inner: Arc<dyn ExpertSource> = src;
        let faulty: Arc<dyn ExpertSource> = Arc::new(FaultySource::new(
            inner,
            [Rule::once_after(Target::Any, at, fault)],
        ));
        match sums {
            Some(s) => Arc::new(VerifiedSource::new(faulty, s)),
            None => faulty,
        }
    }
}

fn faults() -> [Fault; 4] {
    [
        Fault::Eio,
        Fault::ShortRead { keep: 17 },
        Fault::Panic,
        Fault::Corrupt {
            offset: 333,
            xor: 0x40,
        },
    ]
}

#[test]
fn a_failed_read_fails_the_request_and_the_next_succeeds() {
    quiet_injected_panics();
    for id in IDS {
        for cache in [Cache::Resident, Cache::Streaming] {
            for io in [0, 4] {
                let (want, prefill, total) = reference(id, cache, io);
                // A read in the middle of the prefill, and (when decode reads at all)
                // one in the middle of the decode.
                let mut points = vec![prefill / 2];
                if total > prefill + 1 {
                    points.push(prefill + (total - prefill) / 2);
                }
                if matches!(cache, Cache::Streaming) {
                    assert!(points.len() == 2, "{id}: streaming must read during decode");
                }
                for fault in faults() {
                    for &at in &points {
                        let what = format!("{id} {cache:?} io {io} {fault:?} at read {at}");
                        let e = engine(id, cache, io, wrapped(&fault, at));
                        let mut got = Vec::new();
                        let r = e.generate(PROMPT, NEW, Sampling::default(), |t| {
                            got.push(t);
                            true
                        });
                        let err = r.expect_err(&what).to_string();
                        assert!(err.contains("expert"), "{what}: {err}");
                        assert_eq!(got[..], want[..got.len()], "{what}: wrong tokens");
                        if at >= prefill {
                            assert!(!got.is_empty(), "{what}: the fault came too early");
                        }
                        settled(&e, &what);
                        let again = e.generate(PROMPT, NEW, Sampling::default(), |_| true);
                        assert_eq!(again.unwrap(), want, "{what}: the next request");
                        settled(&e, &what);
                        ledger_matches_store(&e, &what);
                    }
                }
            }
        }
    }
}

/// With the whole model resident, the ledger after the recovered request equals a
/// fault-free engine's after the same request: the failure left nothing behind.
#[test]
fn the_ledger_returns_to_the_fault_free_baseline() {
    for id in IDS {
        let (_, prefill, _) = reference(id, Cache::Resident, 4);
        let clean = engine(id, Cache::Resident, 4, |s| s);
        clean
            .generate(PROMPT, NEW, Sampling::default(), |_| true)
            .unwrap();
        let e = engine(id, Cache::Resident, 4, wrapped(&Fault::Eio, prefill / 2));
        assert!(
            e.generate(PROMPT, NEW, Sampling::default(), |_| true)
                .is_err()
        );
        settled(&e, id);
        e.generate(PROMPT, NEW, Sampling::default(), |_| true)
            .unwrap();
        let tier = qubivex_memory::Tier::Ram;
        assert_eq!(
            e.ledger().snapshot(tier),
            clean.ledger().snapshot(tier),
            "{id}"
        );
        assert_eq!(
            e.store_occupancy().unwrap(),
            clean.store_occupancy().unwrap(),
            "{id}"
        );
    }
}

/// A decode step that fails on an expert read leaves the KV cache as it was: the next
/// step on the same cache gives the fault-free logits. The failed step and the retry
/// run different tokens, so keys and values the failed step left behind would show.
#[test]
fn a_failed_step_can_be_retried_on_the_same_cache() {
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for id in IDS {
        for io in [0, 4] {
            let clean = engine(id, Cache::Streaming, io, |s| s);
            let mut c = clean.new_cache();
            clean.forward(PROMPT, &mut c, Logits::Last, None).unwrap();
            let want = clean.forward(&[RETRY], &mut c, Logits::Last, None).unwrap();

            // A source that fails the third read after the prefill.
            let (_, prefill, _) = reference(id, Cache::Streaming, io);
            let m = meta(id);
            let opts = options(&m, Cache::Streaming, io);
            let e = Engine::load_with_source(m, opts, wrapped(&Fault::Eio, prefill + 2)).unwrap();
            let mut c = e.new_cache();
            e.forward(PROMPT, &mut c, Logits::Last, None).unwrap();
            let len = c.len();
            let err = e.forward(&[FAILED], &mut c, Logits::Last, None);
            assert!(err.is_err(), "{id} io {io}: the step should fail");
            assert_eq!(c.len(), len, "{id} io {io}: the cache advanced");
            settled(&e, id);
            let got = e.forward(&[RETRY], &mut c, Logits::Last, None).unwrap();
            assert_eq!(bits(&got), bits(&want), "{id} io {io}: retried step");
        }
    }
}

/// The same on Metal whole layers (the model resident, so the plan keeps the GPU pass):
/// a failed read fails the request, and the next one gives the fault-free tokens.
#[test]
fn metal_layers_survive_a_failed_read() {
    match qubivex_backend_metal::MetalBackend::new() {
        Err(qubivex_backend_metal::MetalError::Unavailable(e)) => {
            eprintln!("skipping: {e}");
            return;
        }
        Err(e) => panic!("{e}"),
        Ok(_) => {}
    }
    quiet_injected_panics();
    for id in IDS {
        for io in [0, 4] {
            let (want, prefill, total) = reference_on(id, Cache::Resident, io, true);
            let mut points = vec![prefill / 2];
            if total > prefill + 1 {
                points.push(prefill + (total - prefill) / 2);
            }
            for fault in faults() {
                for &at in &points {
                    let what = format!("{id} metal io {io} {fault:?} at read {at}");
                    let e = engine_on(id, Cache::Resident, io, true, wrapped(&fault, at));
                    let mut got = Vec::new();
                    let r = e.generate(PROMPT, NEW, Sampling::default(), |t| {
                        got.push(t);
                        true
                    });
                    assert!(r.is_err(), "{what}");
                    assert_eq!(got[..], want[..got.len()], "{what}: wrong tokens");
                    settled(&e, &what);
                    let again = e.generate(PROMPT, NEW, Sampling::default(), |_| true);
                    assert_eq!(again.unwrap(), want, "{what}: the next request");
                    settled(&e, &what);
                    ledger_matches_store(&e, &what);
                }
            }
        }
    }
}
