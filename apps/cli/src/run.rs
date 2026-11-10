//! `qubivex run`: generate text with a model.

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use qubivex_core::Bytes;
use qubivex_model::{ArchitectureRegistry, find};
use qubivex_runtime::{ChatMessage, Engine, GenerationEvent, GenerationRequest, Prompt, Sampling};

use crate::models::roots;
use crate::{MemoryOptions, chain};

pub(crate) struct RunArgs<'a> {
    pub(crate) model: &'a str,
    pub(crate) dirs: &'a [PathBuf],
    pub(crate) prompt: &'a str,
    pub(crate) chat: bool,
    pub(crate) max_tokens: usize,
    pub(crate) sampling: Sampling,
    pub(crate) memory: MemoryOptions,
    /// Copies of the weight files to fail over to (`--mirror`).
    pub(crate) mirrors: &'a [PathBuf],
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
    let engine = Engine::load_mirrored(meta, a.memory.engine(a.context, a.threads), a.mirrors)
        .map_err(|e| chain(&e))?;
    if let qubivex_runtime::Verification::Stale(why) = engine.verification() {
        eprintln!(
            "qubivex: warning: expert reads are not verified: {why}; run `qubivex models verify {}`",
            a.model
        );
    }
    let load = t0.elapsed();
    let mut request = GenerationRequest::new(if a.chat {
        Prompt::Chat(vec![ChatMessage::new("user", a.prompt)])
    } else {
        Prompt::Text(a.prompt.to_owned())
    });
    // As many as the context holds, up to --max-tokens.
    request.default_max_tokens = a.max_tokens;
    request.sampling = a.sampling;
    let mut out = std::io::stdout().lock();
    let t1 = Instant::now();
    let mut first: Option<f64> = None;
    let mut write_err = None;
    let outcome = engine
        .run_request(&request, &AtomicBool::new(false), |event| {
            let written = match event {
                GenerationEvent::Started { .. } if a.chat => Ok(()),
                GenerationEvent::Started { .. } => {
                    write!(out, "{}", a.prompt).and_then(|()| out.flush())
                }
                GenerationEvent::Token(_) => {
                    first.get_or_insert_with(|| t1.elapsed().as_secs_f64());
                    Ok(())
                }
                GenerationEvent::Text(piece) => write!(out, "{piece}").and_then(|()| out.flush()),
            };
            written.map_err(|e| write_err = Some(e)).is_ok()
        })
        .map_err(|e| chain(&e))?;
    let _ = writeln!(out);
    if let Some(e) = write_err {
        return Err(format!("cannot write output: {e}"));
    }
    let (prompt_tokens, generated) = (outcome.prompt_tokens, outcome.completion_tokens);
    let total = t1.elapsed().as_secs_f64();
    if !a.quiet {
        let st = engine.store_stats();
        let ttft = first.unwrap_or(total);
        #[allow(clippy::cast_precision_loss)]
        let decode = if generated > 1 {
            (generated - 1) as f64 / (total - ttft).max(1e-9)
        } else {
            0.0
        };
        #[allow(clippy::cast_precision_loss)]
        let prefill = prompt_tokens as f64 / ttft.max(1e-9);
        eprintln!(
            "\n[{} prompt tokens, {} generated | {} threads | load {:.2}s | prefill {:.1} tok/s | decode {:.2} tok/s | experts: cache {}, {} hits, {} loads ({}), {} evictions | {}]",
            prompt_tokens,
            generated,
            engine.threads(),
            load.as_secs_f64(),
            prefill,
            decode,
            Bytes(engine.expert_cache_bytes()),
            st.hits,
            st.loads,
            Bytes(st.bytes_read),
            st.evictions,
            memory_line(&engine)
        );
        let place = engine.placement();
        if let Some(why) = &place.fallback {
            eprintln!("[backend: cpu, not {}: {why}]", place.backend.name());
        } else if let (Some(dev), Some(ops)) = (&place.device, place.ops) {
            eprintln!(
                "[backend: {} on {dev} | {}: {} tensors, {}]",
                place.backend.name(),
                ops.describe(),
                place.tensors,
                Bytes(place.bytes)
            );
        }
        if let Some(line) = device_line(&engine) {
            eprintln!("{line}");
        }
        if let Some(p) = engine.prefetch_report() {
            let f = p.store;
            eprintln!(
                "[prefetch: recall {:.1}%, precision {:.1}% | {} issued, {} ready, {} late, {} wasted, {} cancelled]",
                p.prediction.recall() * 100.0,
                p.prediction.precision() * 100.0,
                f.issued,
                f.ready,
                f.late,
                f.wasted,
                f.cancelled
            );
        }
    }
    Ok(())
}

/// A discrete device's memory, expert cache and copies, for the stats line.
pub(crate) fn device_line(engine: &Engine) -> Option<String> {
    let r = engine.device_report()?;
    let (u, t, x) = (r.usage, r.transfers, r.experts);
    #[allow(clippy::cast_precision_loss)]
    let ms = t.transfer_ns as f64 / 1e6;
    Some(format!(
        "[backend: {} on {} | VRAM {} used, peak {} of {} | device experts: cache {} of {}, {} hits, {} copies, {} evictions, {} dropped with the RAM cache{} | copies: {} weights ({}), {} activations in, {} out, {} submissions, {ms:.1} ms modelled]",
        engine.placement().backend.name(),
        r.device,
        Bytes(u.used),
        Bytes(u.peak),
        Bytes(u.capacity),
        Bytes(x.used),
        Bytes(x.budget),
        x.hits,
        x.misses,
        x.evictions,
        x.detached,
        if x.over_budget > 0 {
            format!(", {} over budget", x.over_budget)
        } else {
            String::new()
        },
        t.uploads,
        Bytes(t.upload_bytes),
        Bytes(t.activations_in),
        Bytes(t.activations_out),
        t.submissions,
    ))
}

/// The plan's budget and the ledger's totals, for the stats line.
pub(crate) fn memory_line(engine: &Engine) -> String {
    let l = engine.ledger().snapshot(qubivex_memory::Tier::Ram);
    let plan = engine.memory_plan().map_or_else(
        || "fixed cache".to_owned(),
        |p| format!("{} budget {}", p.profile.name(), p.ram.budget.bytes),
    );
    let pressure = engine.pressure().map_or_else(String::new, |d| {
        format!("; pressure {:?}", d.level).to_lowercase()
    });
    format!(
        "memory: {plan}; ledger {} (dense {}, experts {}, KV {}){pressure}",
        l.total(),
        l.dense,
        l.expert_cache,
        l.kv
    )
}
