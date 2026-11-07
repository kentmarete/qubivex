//! ADR-017 measurement: greedy decoding of N sequences, one pass per sequence per step
//! (as the engine runs concurrent requests today) against one shared pass per step
//! ([`Engine::forward_batch`]). Checks the tokens are identical and prints decode
//! throughput. Prefill runs per sequence and is not timed.
//!
//! ```text
//! cargo run --release -p qubivex-runtime --example batch -- \
//!     <MODEL_DIR> <TOKENIZER_JSON> [--cache-mb N] [--steps N] [--runs N] [--sizes 1,2,4,8]
//! ```

use std::error::Error;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{Engine, EngineOptions, KvCache, Logits, Segment};
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

/// Each sequence's tokens and the decode time.
type Run = (Vec<Vec<u32>>, Duration);

/// Decodes `steps` tokens per sequence.
fn decode(
    e: &Engine,
    prompts: &[Vec<u32>],
    steps: usize,
    batched: bool,
) -> Result<Run, Box<dyn Error>> {
    let (mut caches, mut next) = prefill(e, prompts)?;
    let mut out: Vec<Vec<u32>> = next.iter().map(|&t| vec![t]).collect();
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
    Ok((out, start.elapsed()))
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn Error>> {
    let usage = "usage: batch <MODEL_DIR> <TOKENIZER_JSON> [--cache-mb N] [--steps N] [--runs N] [--sizes 1,2,4,8]";
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().ok_or(usage)?);
    let tok = Tokenizer::from_hf_dir(&PathBuf::from(args.next().ok_or(usage)?))?;
    let (mut cache_mb, mut steps, mut runs) = (None, 32usize, 3usize);
    let mut sizes = vec![1usize, 2, 4, 8];
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(usage)?;
        match flag.as_str() {
            "--cache-mb" => cache_mb = Some(value.parse::<u64>()?),
            "--steps" => steps = value.parse()?,
            "--runs" => runs = value.parse()?,
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
            ..EngineOptions::default()
        },
    )?;
    let prompts: Vec<Vec<u32>> = PROMPTS
        .iter()
        .map(|p| tok.encode(p, true))
        .collect::<Result<_, _>>()?;
    println!(
        "expert cache: {}; {steps} tokens per sequence, median of {runs} runs",
        cache_mb.map_or("all resident".into(), |m| format!("{m} MiB"))
    );
    println!("| N | sequential tok/s | batched tok/s | speed-up | tokens identical |");
    println!("|---:|---:|---:|---:|---|");
    for n in sizes {
        let ps = &prompts[..n.min(prompts.len())];
        // Warm-up: fills the expert cache the way the measured runs will use it.
        decode(&e, ps, steps.min(4), true)?;
        let (mut seq_t, mut bat_t) = (Vec::new(), Vec::new());
        let mut same = true;
        for r in 0..runs {
            // Alternate which goes first, so drift does not favour either.
            let order = if r % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            let mut tokens = Vec::new();
            for batched in order {
                let (t, d) = decode(&e, ps, steps, batched)?;
                if batched { &mut bat_t } else { &mut seq_t }.push(d);
                tokens.push(t);
            }
            same &= tokens[0] == tokens[1];
        }
        #[allow(clippy::cast_precision_loss)]
        let generated = (ps.len() * (steps - 1)) as f64;
        let (s, b) = (median(seq_t), median(bat_t));
        println!(
            "| {} | {:.2} | {:.2} | {:.2}x | {} |",
            ps.len(),
            generated / s.as_secs_f64(),
            generated / b.as_secs_f64(),
            s.as_secs_f64() / b.as_secs_f64(),
            if same { "yes" } else { "NO" }
        );
    }
    Ok(())
}
