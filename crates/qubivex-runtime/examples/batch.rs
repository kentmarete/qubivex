//! ADR-017 measurement: greedy decoding of N sequences three ways: one pass per sequence
//! per step on one thread (taking turns), one shared pass per step driven by the caller
//! ([`Engine::forward_batch`]), and N threads each calling [`Engine::forward`], which
//! combine at the engine's turn (how a server runs concurrent requests). Checks the
//! tokens are identical and prints decode throughput. Prefill runs per sequence and is
//! not timed.
//!
//! ```text
//! cargo run --release -p qubivex-runtime --example batch -- \
//!     <MODEL_DIR> <TOKENIZER_JSON> [--cache-mb N] [--steps N] [--runs N] [--sizes 1,2,4,8] \
//!     [--metal layers|token] [--vulkan on] [--threads N]
//! ```
//!
//! `--metal` runs whole layers on Metal, routed on the host (`layers`) or on the GPU
//! (`token`). `--vulkan on` runs on the default Vulkan device, with the device expert
//! cache sized like `qubivex bench`'s (every expert resident when the RAM cache is).

use std::error::Error;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use qubivex_expert_stream::predict::LookaheadConfig;
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{
    BackendKind, Engine, EngineOptions, KvCache, Logits, MetalOps, PrefetchOptions, Segment,
    VulkanConfig,
};
use qubivex_tokenizer::Tokenizer;

const PROMPTS: [&str; 8] = [
    "Explain how a mixture-of-experts language model routes each token.",
    "Write a short poem about the sea at night.",
    "What is the capital of Kenya, and what is it known for?",
    "List three ways to make a Rust program faster.",
    "Translate 'good morning, how are you?' into French and Spanish.",
    "Summarise the plot of a heist film in two sentences.",
    "Why does ice float on water?",
    "Give me a recipe for a simple vegetable soup.",
];

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    u32::try_from(best).unwrap_or(u32::MAX)
}

/// Prefills each prompt in its own cache; returns the caches and each first token.
fn prefill(e: &Engine, prompts: &[Vec<u32>]) -> Result<(Vec<KvCache>, Vec<u32>), Box<dyn Error>> {
    let mut caches = Vec::new();
    let mut next = Vec::new();
    for p in prompts {
        let mut c = e.new_cache();
        next.push(argmax(&e.forward(p, &mut c, Logits::Last, None)?));
        caches.push(c);
    }
    Ok((caches, next))
}

/// Each sequence's tokens, the decode time, and the store's blocked time and loads
/// during it.
type Run = (Vec<Vec<u32>>, Duration, Duration, u64);

/// The store's blocked time and loads so far.
fn reads(e: &Engine) -> (u64, u64) {
    let st = e.store_stats();
    (st.wait_ns, st.loads)
}

fn run_since(e: &Engine, out: Vec<Vec<u32>>, start: Instant, r0: (u64, u64)) -> Run {
    let took = start.elapsed();
    let r1 = reads(e);
    (out, took, Duration::from_nanos(r1.0 - r0.0), r1.1 - r0.1)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Sequential,
    Batched,
    Concurrent,
}

/// Decodes `steps` tokens per sequence.
fn decode(
    e: &Engine,
    prompts: &[Vec<u32>],
    steps: usize,
    mode: Mode,
) -> Result<Run, Box<dyn Error>> {
    let (mut caches, mut next) = prefill(e, prompts)?;
    let mut out: Vec<Vec<u32>> = next.iter().map(|&t| vec![t]).collect();
    if mode == Mode::Concurrent {
        let r0 = reads(e);
        let start = Instant::now();
        let done: Vec<Result<(), String>> = std::thread::scope(|s| {
            let handles: Vec<_> = caches
                .iter_mut()
                .zip(&mut out)
                .map(|(c, o)| {
                    s.spawn(move || {
                        for _ in 1..steps {
                            let t = o[o.len() - 1];
                            let l = e
                                .forward(&[t], c, Logits::Last, None)
                                .map_err(|x| x.to_string())?;
                            o.push(argmax(&l));
                        }
                        Ok(())
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err("panicked".into())))
                .collect()
        });
        let run = run_since(e, out, start, r0);
        for d in done {
            d?;
        }
        return Ok(run);
    }
    let batched = mode == Mode::Batched;
    let r0 = reads(e);
    let start = Instant::now();
    for _ in 1..steps {
        let feed: Vec<[u32; 1]> = next.iter().map(|&t| [t]).collect();
        let logits = if batched {
            let mut segments: Vec<Segment<'_>> = caches
                .iter_mut()
                .zip(&feed)
                .map(|(cache, tokens)| Segment {
                    tokens,
                    cache,
                    logits: Logits::Last,
                })
                .collect();
            e.forward_batch(&mut segments)?
        } else {
            caches
                .iter_mut()
                .zip(&feed)
                .map(|(c, t)| e.forward(t, c, Logits::Last, None))
                .collect::<Result<_, _>>()?
        };
        for ((n, o), l) in next.iter_mut().zip(&mut out).zip(&logits) {
            *n = argmax(l);
            o.push(*n);
        }
    }
    Ok(run_since(e, out, start, r0))
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn Error>> {
    let usage = "usage: batch <MODEL_DIR> <TOKENIZER_JSON> [--cache-mb N] [--steps N] [--runs N] [--sizes 1,2,4,8] [--metal layers|token] [--vulkan on] [--threads N] [--prefetch on|off]";
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().ok_or(usage)?);
    let tok = Tokenizer::from_hf_dir(&PathBuf::from(args.next().ok_or(usage)?))?;
    let (mut cache_mb, mut steps, mut runs) = (None, 32usize, 3usize);
    let mut sizes = vec![1usize, 2, 4, 8];
    let mut metal = None;
    let vulkan = VulkanConfig::default();
    let mut on_vulkan = false;
    let mut threads = None;
    let mut prefetch = None;
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(usage)?;
        match flag.as_str() {
            "--cache-mb" => cache_mb = Some(value.parse::<u64>()?),
            "--steps" => steps = value.parse()?,
            "--runs" => runs = value.parse()?,
            "--threads" => threads = Some(value.parse()?),
            "--vulkan" => {
                on_vulkan = match value.as_str() {
                    "on" => true,
                    "off" => false,
                    _ => return Err(usage.into()),
                };
            }
            "--prefetch" => {
                prefetch = match value.as_str() {
                    "on" => Some(PrefetchOptions {
                        lookahead: LookaheadConfig::default(),
                        guard: true,
                        throttle: true,
                    }),
                    "off" => None,
                    _ => return Err(usage.into()),
                };
            }
            "--metal" => {
                let token = MetalOps::default();
                metal = Some(match value.as_str() {
                    "layers" => MetalOps {
                        token: false,
                        ..token
                    },
                    "token" => token,
                    _ => return Err(usage.into()),
                });
            }
            "--sizes" => {
                sizes = value.split(',').map(str::parse).collect::<Result<_, _>>()?;
            }
            _ => return Err(usage.into()),
        }
    }
    let meta = ModelMetadata::load(&dir, &ArchitectureRegistry::with_builtins())?;
    let e = Engine::load(
        meta,
        EngineOptions {
            expert_cache_bytes: cache_mb.map(|m| m << 20),
            max_context: Some(512),
            backend: if metal.is_some() {
                BackendKind::Metal
            } else if on_vulkan {
                BackendKind::Vulkan
            } else {
                BackendKind::Cpu
            },
            vulkan,
            threads,
            metal_ops: metal.unwrap_or_default(),
            prefetch,
            ..EngineOptions::default()
        },
    )?;
    let p = e.placement();
    if let Some(why) = &p.fallback {
        println!("not on Metal: {why}");
    }
    let prompts: Vec<Vec<u32>> = PROMPTS
        .iter()
        .map(|p| tok.encode(p, true))
        .collect::<Result<_, _>>()?;
    println!(
        "{}; prefetch {}; expert cache: {}; {steps} tokens per sequence, median of {runs} runs",
        match (metal, &p.device) {
            (Some(o), _) => format!("Metal {}", o.describe()),
            (None, Some(d)) if on_vulkan => d.clone(),
            _ => "CPU".into(),
        },
        if prefetch.is_some() { "on" } else { "off" },
        cache_mb.map_or("all resident".into(), |m| format!("{m} MiB"))
    );
    println!(
        "| N | sequential tok/s | batched tok/s | speed-up | concurrent tok/s | speed-up | widest pass | tokens identical | read wait seq / conc | loads per token seq / conc |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|");
    for n in sizes {
        let ps = &prompts[..n.min(prompts.len())];
        // Warm-up: fills the expert cache the way the measured runs will use it.
        decode(&e, ps, steps.min(4), Mode::Batched)?;
        let mut times: [Vec<Duration>; 3] = Default::default();
        let mut waits: [Vec<Duration>; 3] = Default::default();
        let mut loads: [Vec<u64>; 3] = Default::default();
        let mut same = true;
        let mut widest = 0;
        let modes = [Mode::Sequential, Mode::Batched, Mode::Concurrent];
        for r in 0..runs {
            // Rotate which goes first, so drift does not favour any.
            let mut tokens = Vec::new();
            for k in 0..3 {
                let i = (k + r) % 3;
                let before = e.batch_stats().widest;
                let (t, d, w, l) = decode(&e, ps, steps, modes[i])?;
                waits[i].push(w);
                loads[i].push(l);
                if modes[i] == Mode::Concurrent {
                    // The engine keeps its widest pass; reset per size is not
                    // possible, so this is the widest so far.
                    widest = widest.max(e.batch_stats().widest.max(before));
                }
                times[i].push(d);
                tokens.push(t);
            }
            same &= tokens.windows(2).all(|w| w[0] == w[1]);
        }
        #[allow(clippy::cast_precision_loss)]
        let generated = (ps.len() * (steps - 1)) as f64;
        let [s, b, c] = times.map(median);
        let [ws, _, wc] = waits.map(median);
        let [ls, _, lc] = loads.map(|mut v| {
            v.sort_unstable();
            v[v.len() / 2]
        });
        #[allow(clippy::cast_precision_loss)]
        let per_token = |l: u64| l as f64 / generated;
        println!(
            "| {} | {:.2} | {:.2} | {:.2}x | {:.2} | {:.2}x | {} | {} | {:.0}% / {:.0}% | {:.1} / {:.1} |",
            ps.len(),
            generated / s.as_secs_f64(),
            generated / b.as_secs_f64(),
            s.as_secs_f64() / b.as_secs_f64(),
            generated / c.as_secs_f64(),
            s.as_secs_f64() / c.as_secs_f64(),
            widest,
            if same { "yes" } else { "NO" },
            100.0 * ws.as_secs_f64() / s.as_secs_f64(),
            100.0 * wc.as_secs_f64() / c.as_secs_f64(),
            per_token(ls),
            per_token(lc),
        );
    }
    Ok(())
}
