//! One generation, on a dedicated thread: the chat template, tokenization, the KV
//! reservation and the decode loop all run here, never on a Tokio worker
//! (RUST_GUIDELINES §4). Events go back over a bounded channel. Cancellation is a flag
//! the request side sets when the client goes away or the request is abandoned; the
//! engine checks it between prefill chunks and tokens.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use qubivex_runtime::{Engine, Finish, GenerateOptions, Sampling};
use qubivex_tokenizer::ChatMessage;
use tokio::sync::{OwnedSemaphorePermit, mpsc};

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

/// A generation request handed to the worker.
pub(crate) struct Job {
    pub(crate) engine: Arc<Engine>,
    pub(crate) input: Input,
    pub(crate) params: Params,
    pub(crate) default_max_tokens: usize,
    pub(crate) prefill_chunk: usize,
    pub(crate) deadline: Instant,
    pub(crate) cancel: Arc<AtomicBool>,
    pub(crate) request_id: String,
    pub(crate) metrics: Arc<Metrics>,
}

/// Holds back the tail of the text that could still become a stop sequence, and ends
/// the text at the first stop sequence.
struct Stops {
    stops: Vec<String>,
    /// Longest stop sequence, in bytes.
    longest: usize,
    text: String,
    sent: usize,
}

impl Stops {
    fn new(stops: Vec<String>) -> Self {
        let longest = stops.iter().map(String::len).max().unwrap_or(0);
        Self {
            stops,
            longest,
            text: String::new(),
            sent: 0,
        }
    }

    fn floor(&self, mut i: usize) -> usize {
        while !self.text.is_char_boundary(i) {
            i -= 1;
        }
        i
    }

    /// Adds a piece; returns the text that is safe to send, and whether a stop
    /// sequence was found (the text then ends before it).
    fn push(&mut self, piece: &str) -> (String, bool) {
        self.text.push_str(piece);
        if !self.stops.is_empty() {
            let from = self.floor(self.sent.saturating_sub(self.longest));
            let hit = self
                .stops
                .iter()
                .filter_map(|s| self.text[from..].find(s.as_str()).map(|p| from + p))
                .min();
            if let Some(at) = hit {
                let out = self.text[self.sent.min(at)..at].to_owned();
                self.sent = at;
                return (out, true);
            }
        }
        let keep = self.longest.saturating_sub(1);
        let safe = self
            .floor(self.text.len().saturating_sub(keep))
            .max(self.sent);
        let out = self.text[self.sent..safe].to_owned();
        self.sent = safe;
        (out, false)
    }

    /// Whatever was held back.
    fn flush(&mut self) -> String {
        let out = self.text[self.sent..].to_owned();
        self.sent = self.text.len();
        out
    }
}

/// Starts `job` on its own thread. The permit (one of the model's generation slots) is
/// held until the thread ends.
pub(crate) fn spawn(job: Job, tx: mpsc::Sender<Event>, permit: OwnedSemaphorePermit) {
    let name = format!("qubivex-req-{}", job.request_id);
    let spawned = std::thread::Builder::new().name(name).spawn(move || {
        let _permit = permit;
        let metrics = Arc::clone(&job.metrics);
        let rid = job.request_id.clone();
        metrics.active.fetch_add(1, Ordering::Relaxed);
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
    }
}

fn run(job: Job, tx: &mpsc::Sender<Event>) {
    let send = |ev| tx.blocking_send(ev).is_ok();
    let fail = |e: ApiError| {
        let _ = tx.blocking_send(Event::Failed(e));
    };
    let e = &job.engine;
    let tok = e.tokenizer();
    let (text, special) = match &job.input {
        Input::Text(t) => (t.clone(), true),
        Input::Chat(msgs) => {
            if !tok.has_chat_template() {
                return fail(ApiError::invalid(
                    "this model has no chat template; use /v1/completions",
                ));
            }
            let msgs: Vec<ChatMessage> = msgs
                .iter()
                .map(|(r, c)| ChatMessage::new(r.as_str(), c.as_str()))
                .collect();
            match tok.apply_chat_template(&msgs, true) {
                // The template writes any special tokens itself.
                Ok(t) => (t, false),
                Err(err) => {
                    return fail(ApiError::invalid(format!(
                        "the chat template failed on these messages: {err}"
                    )));
                }
            }
        }
    };
    let prompt = match tok.encode(&text, special) {
        Ok(p) if !p.is_empty() => p,
        Ok(_) => {
            return fail(ApiError::invalid_param(
                "prompt",
                "the prompt encodes to no tokens",
            ));
        }
        Err(err) => {
            return fail(ApiError::invalid(format!(
                "cannot tokenize the prompt: {err}"
            )));
        }
    };
    let ctx = e.max_context();
    if prompt.len() >= ctx {
        return fail(ApiError::invalid(format!(
            "the prompt has {} tokens; the context holds {ctx}, leaving no room to generate",
            prompt.len()
        )));
    }
    let room = ctx - prompt.len();
    let max_new = match job.params.max_tokens {
        Some(m) if m > room => {
            return fail(ApiError::invalid_param(
                "max_tokens",
                format!(
                    "the prompt has {} tokens and max_tokens is {m}; the context holds {ctx}",
                    prompt.len()
                ),
            ));
        }
        Some(m) => m,
        None => job.default_max_tokens.min(room),
    };
    // Admission: this conversation's KV, priced with every other live one.
    let _kv = match e.reserve_kv(prompt.len() + max_new) {
        Ok(r) => r,
        Err(err) => return fail(ApiError::from_runtime(&err, &job.request_id)),
    };
    if !send(Event::Started {
        prompt_tokens: prompt.len(),
    }) {
        job.metrics.cancelled.fetch_add(1, Ordering::Relaxed);
        return;
    }
    job.metrics
        .prompt_tokens
        .fetch_add(prompt.len() as u64, Ordering::Relaxed);

    let opts = GenerateOptions {
        max_new,
        sampling: Sampling {
            temperature: job.params.temperature,
            top_p: job.params.top_p,
            seed: job.params.seed,
        },
        prefill_chunk: job.prefill_chunk,
    };
    let mut decoder = tok.stream();
    let mut stops = Stops::new(job.params.stops.clone());
    let (mut hit_stop, mut gone, mut timed_out) = (false, false, false);
    let result = e.generate_with(&prompt, &opts, &job.cancel, |id| {
        job.metrics.tokens.fetch_add(1, Ordering::Relaxed);
        let piece = decoder.push(id).unwrap_or_default();
        let (out, hit) = stops.push(&piece);
        if !out.is_empty() && !send(Event::Text(out)) {
            gone = true;
            return false;
        }
        if hit {
            hit_stop = true;
            return false;
        }
        if Instant::now() >= job.deadline {
            timed_out = true;
            return false;
        }
        true
    });
    match result {
        Ok(g) => {
            if g.finish == Finish::Cancelled || gone {
                job.metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                return;
            }
            if timed_out {
                job.metrics.timeouts.fetch_add(1, Ordering::Relaxed);
            }
            if !hit_stop {
                // A trailing incomplete character, as `decode` of every id shows it.
                let tail = decoder.finish().unwrap_or_default();
                let (out, hit) = stops.push(&tail);
                if hit {
                    if !out.is_empty() {
                        let _ = send(Event::Text(out));
                    }
                    let _ = send(Event::Done {
                        finish: FinishReason::Stop,
                        completion_tokens: g.tokens.len(),
                    });
                    return;
                }
                if !out.is_empty() && !send(Event::Text(out)) {
                    job.metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                let rest = stops.flush();
                if !rest.is_empty() && !send(Event::Text(rest)) {
                    job.metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            }
            let finish = match g.finish {
                Finish::Eos => FinishReason::Stop,
                Finish::Stopped if hit_stop => FinishReason::Stop,
                _ => FinishReason::Length,
            };
            let _ = send(Event::Done {
                finish,
                completion_tokens: g.tokens.len(),
            });
        }
        Err(err) => {
            job.metrics.errors.fetch_add(1, Ordering::Relaxed);
            fail(ApiError::from_runtime(&err, &job.request_id));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Stops;

    fn run(stops: &[&str], pieces: &[&str]) -> (String, bool) {
        let mut s = Stops::new(stops.iter().map(|&s| s.to_owned()).collect());
        let mut out = String::new();
        for p in pieces {
            let (o, hit) = s.push(p);
            out.push_str(&o);
            if hit {
                return (out, true);
            }
        }
        out.push_str(&s.flush());
        (out, false)
    }

    #[test]
    fn stop_sequences_end_the_text_even_across_pieces() {
        assert_eq!(run(&[], &["a", "b"]), ("ab".into(), false));
        assert_eq!(
            run(&["\n\n"], &["hi", "\n", "\nthere"]),
            ("hi".into(), true)
        );
        assert_eq!(run(&["END"], &["xxE", "N", "Dyy"]), ("xx".into(), true));
        assert_eq!(run(&["END"], &["xxE", "N", "o"]), ("xxENo".into(), false));
        assert_eq!(run(&["b", "abc"], &["zab"]), ("za".into(), true));
        // Multi-byte text is held back on character boundaries.
        assert_eq!(
            run(&["日本"], &["こん", "にち日", "本"]),
            ("こんにち".into(), true)
        );
    }
}
