//! Prefix reuse (ADR-016): reusing retained KV never changes bits, sequences stay
//! isolated, and retained KV is bounded and gives way to everything else.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use qubivex_core::Bytes;
use qubivex_expert_stream::ExpertSource;
use qubivex_expert_stream::fault::{Fault, FaultySource, Rule, Target};
use qubivex_kv_cache::MIN_REUSE;
use qubivex_memory::{ACTIVATION_RESERVE, ModelFootprint, Tier};
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{
    BackendKind, Engine, EngineOptions, Finish, GenerateOptions, Generation, KvCache, Logits,
    MetalOps, Sampling,
};

/// Plenty for every test that is not about the budget.
const LARGE: u64 = 64 << 20;
const IDS: [&str; 4] = [
    "olmoe-tiny",
    "olmoe-tiny-gguf",
    "qwen3-moe-tiny",
    "dense-tiny",
];

fn meta(id: &str) -> ModelMetadata {
    let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/models");
    ModelMetadata::load(&dir.join(id), &ArchitectureRegistry::with_builtins()).unwrap()
}

fn opts(prefix_cache_bytes: u64) -> EngineOptions {
    EngineOptions {
        max_context: Some(600),
        pressure_interval_ms: Some(0),
        prefix_cache_bytes,
        ..EngineOptions::default()
    }
}

fn engine(id: &str, prefix_cache_bytes: u64) -> Engine {
    Engine::load(meta(id), opts(prefix_cache_bytes)).unwrap()
}

/// `n` tokens from a pattern that `seed` picks.
fn toks(seed: u32, n: usize) -> Vec<u32> {
    (0..n as u32)
        .map(|i| 32 + (i * 7 + seed * 13 + i * i * seed) % 90)
        .collect()
}

fn gen_opts(max_new: usize) -> GenerateOptions {
    GenerateOptions {
        max_new,
        sampling: Sampling {
            temperature: 0.8,
            seed: 7,
            ..Sampling::default()
        },
        ..GenerateOptions::default()
    }
}

fn generate(e: &Engine, prompt: &[u32], o: &GenerateOptions) -> Generation {
    e.generate_with(prompt, o, &AtomicBool::new(false), |_| true)
        .unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn concat(a: &[u32], b: &[u32]) -> Vec<u32> {
    [a, b].concat()
}

/// A restored cache gives the logits of the whole prompt, bit for bit, for prefixes on
/// both sides of a KV chunk boundary, and it shares no memory with its source.
#[test]
fn a_restored_cache_gives_identical_logits_and_stays_isolated() {
    let chunk = KvCache::CHUNK;
    for id in IDS {
        let e = engine(id, 0);
        let t = toks(1, 400);
        let mut whole = e.new_cache();
        let want = bits(&e.forward(&t, &mut whole, Logits::All, None).unwrap());
        let vocab = want.len() / t.len();
        for n in [1, 40, chunk - 1, chunk, chunk + 1, 399] {
            let mut c = e.new_cache();
            c.restore_from(&whole, n).unwrap();
            assert_eq!(c.len(), n);
            let got = bits(&e.forward(&t[n..], &mut c, Logits::All, None).unwrap());
            assert_eq!(got, want[n * vocab..], "{id} prefix {n}");
        }
        // Writing past the prefix of a copy leaves the source's rows alone.
        let mut other = e.new_cache();
        other.restore_from(&whole, 100).unwrap();
        e.forward(&toks(2, 300), &mut other, Logits::Last, None)
            .unwrap();
        let mut again = e.new_cache();
        again.restore_from(&whole, 100).unwrap();
        let got = bits(&e.forward(&t[100..], &mut again, Logits::All, None).unwrap());
        assert_eq!(got, want[100 * vocab..], "{id}: the source changed");
    }
}

#[test]
fn a_restore_refuses_what_does_not_fit() {
    let e = engine("olmoe-tiny", 0);
    let mut src = e.new_cache();
    e.forward(&toks(1, 40), &mut src, Logits::Last, None)
        .unwrap();
    let mut c = e.new_cache();
    assert!(c.restore_from(&src, 41).is_err(), "more than it holds");
    c.restore_from(&src, 40).unwrap();
    assert!(c.restore_from(&src, 40).is_err(), "not empty");
    let mut other = engine("dense-tiny", 0).new_cache();
    assert!(
        other.restore_from(&src, 40).is_err(),
        "another model's shape"
    );
}

/// The prompts of a session, each with the tokens it should reuse.
fn session(a: &[u32], out: &[u32]) -> Vec<(Vec<u32>, usize)> {
    let turn = concat(a, out);
    let fed = turn.len() - 1;
    vec![
        (a.to_vec(), 0),
        // The same prompt: all but its last token.
        (a.to_vec(), a.len() - 1),
        // The next turn of a conversation: what ran, and the last output token runs.
        (concat(&turn, &toks(5, 20)), fed),
        // Sharing only the start.
        (concat(&a[..50], &toks(6, 30)), 50),
        // Shorter than what is held.
        (a[..60].to_vec(), 59),
        // Nothing in common.
        (toks(9, 80), 0),
    ]
}

fn reuse_matches_a_full_run(on: &Engine, off: &Engine, label: &str) {
    let a = toks(3, 120);
    for prefill_chunk in [7, 256] {
        let o = GenerateOptions {
            prefill_chunk,
            ..gen_opts(16)
        };
        on.clear_prefix_cache();
        let out = generate(off, &a, &o).tokens;
        for (i, (prompt, reuse)) in session(&a, &out).iter().enumerate() {
            let got = generate(on, prompt, &o);
            assert_eq!(
                got.tokens,
                generate(off, prompt, &o).tokens,
                "{label} step {i} chunk {prefill_chunk}"
            );
            assert_eq!(got.cached_tokens, *reuse, "{label} step {i}");
        }
    }
}

#[test]
fn reuse_never_changes_tokens() {
    for id in IDS {
        reuse_matches_a_full_run(&engine(id, LARGE), &engine(id, 0), id);
    }
}

/// The same session with KV in device buffers, for each Metal placement.
#[test]
fn reuse_never_changes_tokens_on_metal() {
    if qubivex_backend_metal::MetalBackend::new().is_err() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let token = MetalOps::default();
    let layers = MetalOps {
        token: false,
        ..token
    };
    for id in IDS {
        let off = engine(id, 0);
        for ops in [layers, token] {
            let on = Engine::load(
                meta(id),
                EngineOptions {
                    backend: BackendKind::Metal,
                    metal_ops: ops,
                    ..opts(LARGE)
                },
            )
            .unwrap();
            assert!(on.placement().device.is_some());
            reuse_matches_a_full_run(&on, &off, &format!("{id} {}", ops.describe()));
        }
    }
}

#[test]
fn a_prompt_held_whole_still_produces_its_logits() {
    let (on, off) = (engine("olmoe-tiny", LARGE), engine("olmoe-tiny", 0));
    let a = toks(3, 60);
    let o = gen_opts(1);
    generate(&on, &a, &o);
    for _ in 0..3 {
        let got = generate(&on, &a, &o);
        assert_eq!(got.cached_tokens, 59);
        assert_eq!(got.tokens, generate(&off, &a, &o).tokens);
    }
}

#[test]
fn off_by_default_nothing_is_kept() {
    let e = Engine::load(meta("olmoe-tiny"), EngineOptions::default()).unwrap();
    let a = toks(3, 60);
    for _ in 0..2 {
        assert_eq!(generate(&e, &a, &gen_opts(2)).cached_tokens, 0);
    }
    assert_eq!(e.prefix_stats(), qubivex_runtime::PrefixStats::default());
}

#[test]
fn scopes_do_not_share() {
    let e = engine("olmoe-tiny", LARGE);
    let a = toks(3, 60);
    let scoped = |s: &str| GenerateOptions {
        cache_scope: Some(s.to_owned()),
        ..gen_opts(2)
    };
    assert_eq!(generate(&e, &a, &scoped("a")).cached_tokens, 0);
    assert_eq!(generate(&e, &a, &scoped("b")).cached_tokens, 0);
    assert_eq!(generate(&e, &a, &gen_opts(2)).cached_tokens, 0);
    assert_eq!(generate(&e, &a, &scoped("a")).cached_tokens, 59);
}

#[test]
fn retained_kv_stays_within_its_budget() {
    let m = meta("olmoe-tiny");
    let per = ModelFootprint::from_metadata(&m)
        .unwrap()
        .kv_bytes_per_token
        .0;
    // Room for two 256-position entries, not three.
    let e = Engine::load(m, opts(per * 512 + per)).unwrap();
    let prompts: Vec<_> = (1..=3).map(|s| toks(s, 100)).collect();
    for p in &prompts {
        generate(&e, p, &gen_opts(2));
        assert!(e.prefix_stats().cost <= 512);
    }
    let s = e.prefix_stats();
    assert_eq!((s.entries, s.cost, s.evicted), (2, 512, 1));
    assert_eq!(generate(&e, &prompts[0], &gen_opts(2)).cached_tokens, 0);
    assert_eq!(generate(&e, &prompts[2], &gen_opts(2)).cached_tokens, 99);

    // A conversation longer than the whole budget is not kept, and takes nothing.
    let before = e.prefix_stats();
    let long = toks(4, 520);
    generate(&e, &long, &gen_opts(2));
    let after = e.prefix_stats();
    assert_eq!(
        (after.entries, after.evicted),
        (before.entries, before.evicted)
    );
}

#[test]
fn the_ledger_counts_retained_kv() {
    let m = meta("olmoe-tiny");
    let per = ModelFootprint::from_metadata(&m)
        .unwrap()
        .kv_bytes_per_token
        .0;
    let e = Engine::load(m, opts(LARGE)).unwrap();
    generate(&e, &toks(3, 100), &gen_opts(2));
    assert_eq!(e.prefix_stats().cost, 256);
    assert_eq!(e.ledger().snapshot(Tier::Ram).kv.0, 256 * per);
    // What the index calls 256 positions is what the cache allocated.
    let mut c = e.new_cache();
    e.forward(&toks(3, 100), &mut c, Logits::Last, None)
        .unwrap();
    assert_eq!(c.allocated_bytes(), 256 * per);
}

/// A planned engine with KV room for `conversations` of 256 positions, and retained KV
/// allowed to use all of it.
fn tight(id: &str, conversations: u64) -> Engine {
    let m = meta(id);
    let fp = ModelFootprint::from_metadata(&m).unwrap();
    let room = conversations * 256 * fp.kv_bytes_per_token.0;
    let o = EngineOptions {
        ram_budget: Some(fp.dense_bytes().0 + ACTIVATION_RESERVE.0 + fp.cache_floor().0 + room),
        max_context: Some(512),
        ..opts(room)
    };
    Engine::load(m, o).unwrap()
}

#[test]
fn running_requests_take_room_from_retained_kv() {
    let e = tight("olmoe-tiny", 3);
    for s in 1..=3 {
        generate(&e, &toks(s, 100), &gen_opts(2));
    }
    assert_eq!(e.prefix_stats().cost, 3 * 256, "fills the plan's KV room");
    // Every reservation the plan allows is granted, oldest entries going first.
    let one = e.reserve_kv(256).unwrap();
    assert_eq!(e.prefix_stats().cost, 2 * 256);
    assert_eq!(generate(&e, &toks(1, 100), &gen_opts(2)).cached_tokens, 0);
    let rest = [e.reserve_kv(256).unwrap(), e.reserve_kv(256).unwrap()];
    assert_eq!(e.prefix_stats().cost, 0);
    drop((one, rest));
}

#[test]
fn a_refused_reservation_evicts_nothing() {
    let e = tight("olmoe-tiny", 3);
    generate(&e, &toks(1, 100), &gen_opts(2));
    let held = [e.reserve_kv(256).unwrap(), e.reserve_kv(256).unwrap()];
    assert_eq!(e.prefix_stats().cost, 256, "two live and one retained fit");
    assert!(e.reserve_kv(512).is_err());
    assert_eq!(
        e.prefix_stats().entries,
        1,
        "evicting would not have helped"
    );
    drop(held);
}

#[test]
fn retained_kv_is_priced_in_the_plan_and_gives_back_the_expert_cache() {
    let e = tight("olmoe-tiny", 3);
    let full = {
        generate(&e, &toks(1, 40), &gen_opts(1));
        e.clear_prefix_cache();
        generate(&e, &[40, 41], &gen_opts(1));
        e.expert_cache_bytes()
    };
    for s in 1..=3 {
        generate(&e, &toks(s, 100), &gen_opts(2));
    }
    // A pass's own KV takes one chunk of the plan's three, so one entry goes.
    generate(&e, &[40, 41], &gen_opts(1));
    assert_eq!(e.prefix_stats().cost, 2 * 256);
    assert!(
        e.expert_cache_bytes() < full,
        "the plan counts what is held"
    );
    e.clear_prefix_cache();
    generate(&e, &[40, 41], &gen_opts(1));
    assert_eq!(e.expert_cache_bytes(), full);
}

#[test]
fn memory_pressure_drops_retained_kv_and_stops_keeping_it() {
    let m = meta("olmoe-tiny");
    let fp = ModelFootprint::from_metadata(&m).unwrap();
    let o = EngineOptions {
        ram_budget: Some(fp.dense_bytes().0 + (1 << 30)),
        ..opts(LARGE)
    };
    let e = Engine::load(m, o).unwrap();
    let a = toks(3, 60);
    generate(&e, &a, &gen_opts(2));
    assert_eq!(e.prefix_stats().entries, 1);
    // With no request running: an idle server gives it back too.
    let d = e.observe_pressure(Bytes(768 << 20)).unwrap();
    assert!(!d.prefetch && d.admit);
    assert_eq!(e.prefix_stats().entries, 0);
    assert_eq!(generate(&e, &a, &gen_opts(2)).cached_tokens, 0);
    assert_eq!(e.prefix_stats().entries, 0, "nothing kept under pressure");
    e.observe_pressure(Bytes(8 << 30)).unwrap();
    generate(&e, &a, &gen_opts(2));
    assert_eq!(generate(&e, &a, &gen_opts(2)).cached_tokens, 59);
}

/// Reads into the prompt of [`a_cancelled_prefill_keeps_what_ran`].
const CANCEL_AFTER_LOADS: u64 = 60;

/// What a cancelled prefill ran is good KV: the retry starts from there, and gets the
/// bits of a run that was never interrupted.
#[test]
fn a_cancelled_prefill_keeps_what_ran() {
    let off = engine("olmoe-tiny", 0);
    let m = meta("olmoe-tiny");
    let delay = Fault::Delay(Duration::from_millis(2));
    let e = Arc::new(
        Engine::load_with_source(
            m.clone(),
            EngineOptions {
                expert_cache_bytes: Some(m.bytes().largest_expert),
                ..opts(LARGE)
            },
            move |src| {
                let inner: Arc<dyn ExpertSource> = src;
                Arc::new(FaultySource::new(
                    inner,
                    [Rule::always(Target::Any, delay.clone())],
                ))
            },
        )
        .unwrap(),
    );
    let a = toks(3, 300);
    let o = GenerateOptions {
        prefill_chunk: 14,
        ..gen_opts(4)
    };
    // Cancelled by progress, not by time: several chunks in, well short of the prompt.
    let cancel = Arc::new(AtomicBool::new(false));
    let ended = Arc::new(AtomicBool::new(false));
    let watcher = {
        let (e, cancel, ended) = (Arc::clone(&e), Arc::clone(&cancel), Arc::clone(&ended));
        std::thread::spawn(move || {
            while e.store_stats().loads < CANCEL_AFTER_LOADS && !ended.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(1));
            }
            cancel.store(true, Ordering::Release);
        })
    };
    let g = e.generate_with(&a, &o, &cancel, |_| true).unwrap();
    ended.store(true, Ordering::Release);
    watcher.join().unwrap();
    eprintln!("cancelled at {} loads", e.store_stats().loads);
    assert_eq!((g.finish, g.tokens.len()), (Finish::Cancelled, 0));
    assert_eq!(e.kv_reserved(), 0);
    let retry = generate(&e, &a, &o);
    eprintln!("the retry reused {} tokens", retry.cached_tokens);
    assert!(retry.cached_tokens >= MIN_REUSE, "{}", retry.cached_tokens);
    assert!(retry.cached_tokens < a.len());
    assert_eq!(retry.cached_tokens % 14, 0, "whole chunks ran");
    assert_eq!(retry.tokens, generate(&off, &a, &o).tokens);
}

#[test]
fn a_deadline_already_gone_keeps_nothing() {
    let e = engine("olmoe-tiny", LARGE);
    let late = GenerateOptions {
        deadline: Some(Instant::now()),
        ..gen_opts(2)
    };
    let g = generate(&e, &toks(3, 60), &late);
    assert_eq!(g.finish, Finish::TimedOut);
    assert_eq!(e.prefix_stats().entries, 0);
}

/// The last token generated was never fed back in, so it is not in the entry.
#[test]
fn a_stopped_generation_keeps_what_ran() {
    let (e, off) = (engine("olmoe-tiny", LARGE), engine("olmoe-tiny", 0));
    let a = toks(3, 120);
    let mut seen = 0;
    let g = e
        .generate_with(&a, &gen_opts(8), &AtomicBool::new(false), |_| {
            seen += 1;
            seen < 3
        })
        .unwrap();
    assert_eq!((g.finish, g.tokens.len()), (Finish::Stopped, 3));
    let next = concat(&concat(&a, &g.tokens), &toks(4, 4));
    let got = generate(&e, &next, &gen_opts(4));
    assert_eq!(got.cached_tokens, a.len() + 2);
    assert_eq!(got.tokens, generate(&off, &next, &gen_opts(4)).tokens);
}

#[test]
fn a_failed_generation_keeps_nothing() {
    let e = engine("olmoe-tiny", LARGE);
    let mut bad = toks(3, 60);
    bad.push(1 << 30);
    let o = GenerateOptions {
        prefill_chunk: 20,
        ..gen_opts(2)
    };
    assert!(
        e.generate_with(&bad, &o, &AtomicBool::new(false), |_| true)
            .is_err()
    );
    assert_eq!(e.prefix_stats().entries, 0);
}

/// Many sequences on one engine, overlapping on purpose so hits, inserts, replacements
/// and evictions all race. Each must still get what it gets alone, with no reuse.
#[test]
fn concurrent_sequences_sharing_prefixes_stay_isolated() {
    let off = engine("olmoe-tiny", 0);
    let m = meta("olmoe-tiny");
    let per = ModelFootprint::from_metadata(&m)
        .unwrap()
        .kv_bytes_per_token
        .0;
    // Room for two entries, so evictions race with the copies.
    let on = Arc::new(Engine::load(m, opts(per * 512)).unwrap());
    let stem = toks(3, 50);
    let prompts: Vec<_> = (1..=6).map(|s| concat(&stem, &toks(s + 10, 30))).collect();
    let want: Vec<_> = prompts
        .iter()
        .map(|p| generate(&off, p, &gen_opts(6)).tokens)
        .collect();
    let workers: Vec<_> = (0..4)
        .map(|w| {
            let (on, prompts, want) = (Arc::clone(&on), prompts.clone(), want.clone());
            std::thread::spawn(move || {
                for round in 0..3 {
                    for i in 0..prompts.len() {
                        let i = (i + w + round) % prompts.len();
                        let got = generate(&on, &prompts[i], &gen_opts(6)).tokens;
                        assert_eq!(got, want[i], "worker {w} round {round} prompt {i}");
                    }
                }
            })
        })
        .collect();
    for w in workers {
        w.join().unwrap();
    }
    let s = on.prefix_stats();
    assert!(s.hits > 0 && s.evicted > 0, "{s:?}");
    assert!(s.cost <= 512);
    assert_eq!(on.kv_reserved(), 0);
}
