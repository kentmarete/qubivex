//! One generation, on a dedicated thread, never on a Tokio worker (RUST_GUIDELINES §4):
//! [`Engine::run_request`] does the work, and this module turns its events into bounded
//! channel sends and metrics. Cancellation is a flag the request side sets when the
//! client goes away or the request is abandoned; the engine checks it between prefill
//! chunks and tokens.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use qubivex_runtime::scheduler::Grant;
use qubivex_runtime::{
    ChatMessage, Engine, Finish, GenerationEvent, GenerationRequest, KvAdmission, PreparedRequest,
    Prompt, RequestError, Sampling,
};
use tokio::sync::mpsc;

use crate::error::ApiError;
use crate::metrics::Metrics;
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
    pub(crate) metrics: Arc<Metrics>,
}

/// Starts `job` on its own thread.
pub(crate) fn spawn(job: Job, tx: mpsc::Sender<Event>) {
    let name = format!("qubivex-req-{}", job.request_id);
    let metrics = Arc::clone(&job.metrics);
    // Counted here, not on the thread: shutdown must not see a gap between the grant
    // and the thread's start.
    metrics.active.fetch_add(1, Ordering::Relaxed);
    let on_thread = Arc::clone(&metrics);
    let spawned = std::thread::Builder::new().name(name).spawn(move || {
        let metrics = on_thread;
        let rid = job.request_id.clone();
        let fail_tx = tx.clone();
        let r = std::panic::catch_unwind(AssertUnwindSafe(|| run(job, &tx)));
        if r.is_err() {
            eprintln!("qubivex-api: request {rid}: generation panicked");
            metrics.errors.fetch_add(1, Ordering::Relaxed);
            let _ = fail_tx.blocking_send(Event::Failed(ApiError::internal(&rid)));
        }
        metrics.active.fetch_sub(1, Ordering::Relaxed);
    });
    if let Err(e) = spawned {
        eprintln!("qubivex-api: cannot start a generation thread: {e}");
        metrics.active.fetch_sub(1, Ordering::Relaxed);
    }
}

fn run(job: Job, tx: &mpsc::Sender<Event>) {
    let send = |ev| tx.blocking_send(ev).is_ok();
    let fail = |e: ApiError| {
        let _ = tx.blocking_send(Event::Failed(e));
    };
    let m = &job.metrics;
    let result = job.engine.run_prepared(
        &job.request,
        &job.prepared,
        job.grant.hold(),
        &job.cancel,
        |event| match event {
            GenerationEvent::Started { prompt_tokens } => {
                let open = send(Event::Started { prompt_tokens });
                if open {
                    m.prompt_tokens
                        .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
                }
                open
            }
            GenerationEvent::Token(_) => {
                m.tokens.fetch_add(1, Ordering::Relaxed);
                true
            }
            GenerationEvent::Text(text) => send(Event::Text(text)),
        },
    );
    // The slot and the memory go back before the client hears the end, so a client that
    // sends its next request straight away finds them free.
    let Job {
        grant,
        request_id,
        metrics: m,
        ..
    } = job;
    drop(grant);
    let m = &m;
    match result {
        Ok(o) => {
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
            });
        }
        Err(e) => {
            if matches!(e, RequestError::Generation(_)) {
                m.errors.fetch_add(1, Ordering::Relaxed);
            }
            fail(ApiError::from_request(&e, &request_id));
        }
    }
}
