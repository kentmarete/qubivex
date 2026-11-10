//! One generation, on a dedicated thread, never on a Tokio worker (RUST_GUIDELINES §4):
//! [`Engine::run_request`] does the work, and this module turns its events into bounded
//! channel sends, metrics and the log line that says how the generation went. Cancellation is a flag the request side sets when the
//! client goes away or the request is abandoned; the engine checks it between prefill
//! chunks and tokens.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use qubivex_runtime::scheduler::Grant;
use qubivex_runtime::{
    ChatMessage, Engine, Finish, GenerationEvent, GenerationRequest, KvAdmission, PreparedRequest,
    Prompt, RequestError, Sampling,
};
use tokio::sync::mpsc;

use crate::error::ApiError;
use crate::log::{Field, Log, millis};
use crate::metrics::{Metrics, ModelMetrics};
use crate::openai::Params;

/// What to generate from.
#[derive(Clone, Debug)]
pub(crate) enum Input {
    /// Chat messages, rendered with the model's chat template.
    Chat(Vec<(String, String)>),
    /// A raw prompt.
    Text(String),
}

/// Why generation ended, as OpenAI reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

/// Events from the worker, in order: `Started`, any number of `Text`, then `Done`; or
/// `Failed` at any point.
#[derive(Debug)]
pub(crate) enum Event {
    Started {
        prompt_tokens: usize,
    },
    Text(String),
    Done {
        finish: FinishReason,
        completion_tokens: usize,
        cached_tokens: usize,
    },
    Failed(ApiError),
}

/// Sets the cancellation flag when dropped: the request side holds one for as long as
/// someone wants the output.
pub(crate) struct CancelOnDrop(pub(crate) Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        // Release: pairs with the engine's Acquire load of the flag.
        self.0.store(true, Ordering::Release);
    }
}

/// The runtime's request for what an HTTP request asked. The deadline is set when it is
/// granted.
pub(crate) fn request(
    input: Input,
    params: &Params,
    default_max_tokens: usize,
    prefill_chunk: usize,
) -> GenerationRequest {
    let prompt = match input {
        Input::Text(t) => Prompt::Text(t),
        Input::Chat(msgs) => Prompt::Chat(
            msgs.iter()
                .map(|(r, c)| ChatMessage::new(r.as_str(), c.as_str()))
                .collect(),
        ),
    };
    GenerationRequest {
        prompt,
        max_tokens: params.max_tokens,
        default_max_tokens,
        sampling: Sampling {
            temperature: params.temperature,
            top_k: params.top_k,
            top_p: params.top_p,
            seed: params.seed,
        },
        stops: params.stops.clone(),
        prefill_chunk,
        deadline: None,
        // Retained KV is shared within a client and never across clients, because a
        // reused prefix is faster and so shows that someone already sent it (ADR-016).
        cache_scope: params.client.clone(),
    }
}

/// A generation request handed to the worker.
pub(crate) struct Job {
    pub(crate) engine: Arc<Engine>,
    pub(crate) request: GenerationRequest,
    pub(crate) prepared: PreparedRequest,
    /// The generation's slot and KV reservation.
    pub(crate) grant: Grant<KvAdmission>,
    pub(crate) cancel: Arc<AtomicBool>,
    pub(crate) request_id: String,
    /// The model's id, for the log.
    pub(crate) model: String,
    /// This generation's number among those the server has admitted.
    pub(crate) sequence: u64,
    /// When the request arrived.
    pub(crate) arrived: Instant,
    pub(crate) metrics: Arc<Metrics>,
    /// The model's timings.
    pub(crate) timings: Arc<ModelMetrics>,
    pub(crate) log: Arc<Log>,
}

impl Job {
    /// The fields every log line about this generation starts with.
    fn ids(&self) -> Vec<Field> {
        vec![
            ("request_id", self.request_id.as_str().into()),
            ("model", self.model.as_str().into()),
            ("sequence_id", self.sequence.into()),
        ]
    }
}

/// How a generation ended, for the log.
const fn finish_name(f: Finish) -> &'static str {
    match f {
        Finish::Eos => "eos",
        Finish::StopSequence => "stop_sequence",
        Finish::Length => "length",
        Finish::ContextFull => "context_full",
        Finish::TimedOut => "timed_out",
        // `Stopped` is the event callback refusing more: the client went away.
        Finish::Cancelled | Finish::Stopped => "cancelled",
    }
}

/// Starts `job` on its own thread.
pub(crate) fn spawn(job: Job, tx: mpsc::Sender<Event>) {
    let name = format!("qubivex-req-{}", job.request_id);
    let metrics = Arc::clone(&job.metrics);
    let log = Arc::clone(&job.log);
    let ids = job.ids();
    // Counted here, not on the thread: shutdown must not see a gap between the grant
    // and the thread's start.
    metrics.active.fetch_add(1, Ordering::Relaxed);
    let on_thread = (Arc::clone(&metrics), Arc::clone(&log), ids.clone());
    let spawned = std::thread::Builder::new().name(name).spawn(move || {
        let (metrics, log, ids) = on_thread;
        let rid = job.request_id.clone();
        let fail_tx = tx.clone();
        let r = std::panic::catch_unwind(AssertUnwindSafe(|| run(job, &tx)));
        if r.is_err() {
            log.error("generation_panicked", &ids);
            metrics.errors.fetch_add(1, Ordering::Relaxed);
            let _ = fail_tx.blocking_send(Event::Failed(ApiError::internal(&rid)));
        }
        metrics.active.fetch_sub(1, Ordering::Relaxed);
    });
    if let Err(e) = spawned {
        let mut fields = ids;
        fields.push(("detail", e.to_string().into()));
        log.error("generation_thread_failed", &fields);
        metrics.active.fetch_sub(1, Ordering::Relaxed);
    }
}

/// When the tokens of one generation came.
struct Clock {
    /// When the prompt was admitted and prefill began.
    prefill: Instant,
    first: Option<Instant>,
    last: Instant,
    tokens: u64,
}

fn run(job: Job, tx: &mpsc::Sender<Event>) {
    let send = |ev| tx.blocking_send(ev).is_ok();
    let fail = |e: ApiError| {
        let _ = tx.blocking_send(Event::Failed(e));
    };
    let m = &job.metrics;
    let t = &job.timings;
    let began = Instant::now();
    let mut clock = Clock {
        prefill: began,
        first: None,
        last: began,
        tokens: 0,
    };
    let result = job.engine.run_prepared(
        &job.request,
        &job.prepared,
        job.grant.hold(),
        &job.cancel,
        |event| match event {
            GenerationEvent::Started { prompt_tokens } => {
                clock.prefill = Instant::now();
                let open = send(Event::Started { prompt_tokens });
                if open {
                    m.prompt_tokens
                        .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
                }
                open
            }
            GenerationEvent::Token(_) => {
                let now = Instant::now();
                if clock.first.is_none() {
                    clock.first = Some(now);
                    t.first_token.observe_duration(now - job.arrived);
                } else {
                    t.token_gap.observe_duration(now - clock.last);
                }
                clock.last = now;
                clock.tokens += 1;
                m.tokens.fetch_add(1, Ordering::Relaxed);
                true
            }
            GenerationEvent::Text(text) => send(Event::Text(text)),
        },
    );
    let took = job.arrived.elapsed();
    t.duration.observe_duration(took);
    let mut fields = job.ids();
    // The slot and the memory go back before the client hears the end, so a client that
    // sends its next request straight away finds them free.
    let Job {
        grant,
        request_id,
        arrived,
        metrics: m,
        timings: t,
        log,
        ..
    } = job;
    drop(grant);
    match result {
        Ok(o) => {
            fields.push(("finish", finish_name(o.finish).into()));
            fields.push(("prompt_tokens", o.prompt_tokens.into()));
            fields.push(("cached_tokens", o.cached_tokens.into()));
            fields.push(("completion_tokens", o.completion_tokens.into()));
            if let Some(first) = clock.first {
                fields.push(("ttft_ms", millis(first - arrived)));
                // Up to the first token: the prompt's passes and one sampling step.
                let ran = o.prompt_tokens.saturating_sub(o.cached_tokens);
                let prefill = (first - clock.prefill).as_secs_f64();
                if ran > 0 && prefill > 0.0 {
                    t.prompt_rate.observe(ran as f64 / prefill);
                }
                let decode = (clock.last - first).as_secs_f64();
                if clock.tokens >= 2 && decode > 0.0 {
                    let rate = (clock.tokens - 1) as f64 / decode;
                    t.generation_rate.observe(rate);
                    fields.push(("tokens_per_second", ((rate * 100.0).round() / 100.0).into()));
                }
            }
            fields.push(("duration_ms", millis(took)));
            log.info("generation_finished", &fields);
            let finish = match o.finish {
                Finish::Cancelled | Finish::Stopped => {
                    m.cancelled.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                Finish::TimedOut => {
                    m.timeouts.fetch_add(1, Ordering::Relaxed);
                    FinishReason::Length
                }
                Finish::Eos | Finish::StopSequence => FinishReason::Stop,
                Finish::Length | Finish::ContextFull => FinishReason::Length,
            };
            let _ = send(Event::Done {
                finish,
                completion_tokens: o.completion_tokens,
                cached_tokens: o.cached_tokens,
            });
        }
        Err(e) => {
            if matches!(e, RequestError::Generation(_)) {
                m.errors.fetch_add(1, Ordering::Relaxed);
            }
            let error = ApiError::from_request(&e, &request_id, &log);
            fields.push(("error", error.code.into()));
            fields.push(("completion_tokens", clock.tokens.into()));
            fields.push(("duration_ms", millis(took)));
            log.warn("generation_failed", &fields);
            fail(error);
        }
    }
}
