//! `qubivex run`: generate text with a model.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use qubivex_core::Bytes;
use qubivex_model::{ArchitectureRegistry, find};
use qubivex_runtime::{Engine, EngineOptions, Sampling};
use qubivex_tokenizer::ChatMessage;

use crate::chain;
use crate::models::roots;

pub(crate) struct RunArgs<'a> {
    pub(crate) model: &'a str,
    pub(crate) dirs: &'a [PathBuf],
    pub(crate) prompt: &'a str,
    pub(crate) chat: bool,
    pub(crate) max_tokens: usize,
    pub(crate) sampling: Sampling,
    pub(crate) expert_cache_mb: Option<u64>,
    pub(crate) context: Option<usize>,
    pub(crate) threads: Option<usize>,
    pub(crate) quiet: bool,
}

pub(crate) fn run(a: &RunArgs<'_>) -> Result<(), String> {
    let t0 = Instant::now();
    let meta = find(
        a.model,
        &roots(a.dirs),
        &ArchitectureRegistry::with_builtins(),
    )
    .map_err(|e| chain(&e))?;
    let opts = EngineOptions {
        expert_cache_bytes: a.expert_cache_mb.map(|mb| mb.saturating_mul(1 << 20)),
        max_context: a.context,
        threads: a.threads,
    };
    let engine = Engine::load(meta, opts).map_err(|e| chain(&e))?;
    let load = t0.elapsed();
    let tok = engine.tokenizer();
    let text = if a.chat {
        tok.apply_chat_template(&[ChatMessage::new("user", a.prompt)], true)
            .map_err(|e| chain(&e))?
    } else {
        a.prompt.to_owned()
    };
    let prompt = tok.encode(&text, !a.chat).map_err(|e| chain(&e))?;
    if prompt.is_empty() {
        return Err("the prompt encodes to no tokens".into());
    }
    let mut stream = tok.stream();
    let mut out = std::io::stdout().lock();
    if !a.chat {
        let _ = write!(out, "{}", a.prompt);
        let _ = out.flush();
    }
    let t1 = Instant::now();
    let mut first: Option<f64> = None;
    let mut write_err = None;
    let generated = engine
        .generate(&prompt, a.max_tokens, a.sampling, |id| {
            first.get_or_insert_with(|| t1.elapsed().as_secs_f64());
            match stream.push(id) {
                Ok(piece) => {
                    if let Err(e) = write!(out, "{piece}").and_then(|()| out.flush()) {
                        write_err = Some(e);
                        return false;
                    }
                    true
                }
                Err(_) => true,
            }
        })
        .map_err(|e| chain(&e))?;
    let _ = writeln!(out);
    if let Some(e) = write_err {
        return Err(format!("cannot write output: {e}"));
    }
    let total = t1.elapsed().as_secs_f64();
    if !a.quiet {
        let st = engine.store_stats();
        let ttft = first.unwrap_or(total);
        #[allow(clippy::cast_precision_loss)]
        let decode = if generated.len() > 1 {
            (generated.len() - 1) as f64 / (total - ttft).max(1e-9)
        } else {
            0.0
        };
        #[allow(clippy::cast_precision_loss)]
        let prefill = prompt.len() as f64 / ttft.max(1e-9);
        eprintln!(
            "\n[{} prompt tokens, {} generated | {} threads | load {:.2}s | prefill {:.1} tok/s | decode {:.2} tok/s | experts: cache {}, {} hits, {} loads ({}), {} evictions]",
            prompt.len(),
            generated.len(),
            engine.threads(),
            load.as_secs_f64(),
            prefill,
            decode,
            Bytes(engine.expert_cache_bytes()),
            st.hits,
            st.loads,
            Bytes(st.bytes_read),
            st.evictions
        );
    }
    Ok(())
}
