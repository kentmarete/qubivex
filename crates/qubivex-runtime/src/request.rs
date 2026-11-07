//! One generation from text to text, for any host: the chat template, tokenization,
//! context checks, the KV reservation, the decode loop, streaming detokenization and
//! stop sequences. `qubivex run` and the gateway both come through here, and so does an
//! application embedding the runtime, which needs neither a server nor Tokio.
//!
//! [`Engine::run_request`] blocks its thread. It reports progress through a callback,
//! and a host can run as many as it likes on an `Arc<Engine>`, one thread each. Weights
//! and the expert store are shared; each request owns its KV cache and reservation, and
//! both are released on every way out, including a panic in the callback.
//!
//! A host with a queue splits that call in two: [`Engine::prepare`] says what a request
//! will cost, [`KvAdmission`] lets a [`qubivex_scheduler::Scheduler`] decide when it
//! runs, and [`Engine::run_prepared`] runs it under the reservation that decision took.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use qubivex_scheduler::Admission;
use qubivex_tokenizer::ChatMessage;
use thiserror::Error;

use crate::RuntimeError;
use crate::engine::{Engine, Finish, GenerateOptions, KvReservation, Sampling};

/// What to generate from.
#[derive(Clone, Debug)]
pub enum Prompt {
    /// Messages, rendered with the model's chat template.
    Chat(Vec<ChatMessage>),
    /// Raw text, tokenized with the model's special tokens.
    Text(String),
}

/// A generation.
#[derive(Clone, Debug)]
pub struct GenerationRequest {
    /// What to generate from.
    pub prompt: Prompt,
    /// Most tokens to generate. A value the context cannot hold is refused.
    pub max_tokens: Option<usize>,
    /// Most tokens when `max_tokens` is `None`, capped by the room left in the context.
    pub default_max_tokens: usize,
    /// Sampling.
    pub sampling: Sampling,
    /// Text that ends the generation. The output stops before the earliest match.
    pub stops: Vec<String>,
    /// Prompt tokens per forward pass (how soon a cancelled prefill stops).
    pub prefill_chunk: usize,
    /// When to stop generating, checked before each prefill chunk and each token.
    pub deadline: Option<Instant>,
    /// See [`GenerateOptions::cache_scope`].
    pub cache_scope: Option<String>,
}

impl GenerationRequest {
    /// A request with default settings, ending at `default_max_tokens` tokens.
    #[must_use]
    pub fn new(prompt: Prompt) -> Self {
        let d = GenerateOptions::default();
        Self {
            prompt,
            max_tokens: None,
            default_max_tokens: d.max_new,
            sampling: d.sampling,
            stops: Vec::new(),
            prefill_chunk: d.prefill_chunk,
            deadline: None,
            cache_scope: None,
        }
    }
}

/// Progress of a request, in order: one `Started`, then for each token a `Token`
/// followed by any `Text` it released. The callback returns `false` to stop, and the
/// request ends with [`Finish::Stopped`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GenerationEvent {
    /// The prompt is tokenized and admitted; no forward pass has run.
    Started {
        /// Prompt length in tokens.
        prompt_tokens: usize,
    },
    /// A generated token id.
    Token(u32),
    /// Text that is safe to show. Text that could still become a stop sequence is held
    /// back, so one piece is not always one token.
    Text(String),
}

/// How a request ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenerationOutcome {
    /// Why it ended.
    pub finish: Finish,
    /// Prompt length in tokens.
    pub prompt_tokens: usize,
    /// Tokens generated.
    pub completion_tokens: usize,
    /// Prompt tokens reused from retained KV, which no pass ran.
    pub cached_tokens: usize,
}

/// A request tokenized and checked against the context, not yet admitted.
#[derive(Clone, Debug)]
pub struct PreparedRequest {
    prompt: Vec<u32>,
    max_new: usize,
}

impl PreparedRequest {
    /// Prompt length in tokens.
    #[must_use]
    pub fn prompt_tokens(&self) -> usize {
        self.prompt.len()
    }

    /// Most KV positions the conversation can use: the request's cost.
    #[must_use]
    pub fn positions(&self) -> usize {
        self.prompt.len() + self.max_new
    }
}

/// Admits requests against an engine's memory plan, in KV positions, for a
/// [`qubivex_scheduler::Scheduler`].
#[derive(Clone, Debug)]
pub struct KvAdmission(pub Arc<Engine>);

impl Admission for KvAdmission {
    type Hold = KvReservation;
    type Refusal = RuntimeError;

    /// Loading refuses a context the plan cannot hold for one conversation, and
    /// [`Engine::prepare`] refuses what the context cannot hold, so a prepared request
    /// always fits alone.
    fn check(&self, _cost: u64) -> Result<(), RuntimeError> {
        Ok(())
    }

    fn try_hold(&self, cost: u64) -> Result<KvReservation, RuntimeError> {
        self.0
            .reserve_kv(usize::try_from(cost).unwrap_or(usize::MAX))
    }
}

/// Why a request failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RequestError {
    /// The request cannot run on this model.
    #[error("{message}")]
    Invalid {
        /// The request field at fault, if one is.
        param: Option<&'static str>,
        /// What is wrong.
        message: String,
    },
    /// The memory plan has no room for this conversation now.
    #[error(transparent)]
    Admission(RuntimeError),
    /// The forward pass failed.
    #[error(transparent)]
    Generation(RuntimeError),
}

impl RequestError {
    fn invalid(message: String) -> Self {
        Self::Invalid {
            param: None,
            message,
        }
    }

    fn invalid_param(param: &'static str, message: String) -> Self {
        Self::Invalid {
            param: Some(param),
            message,
        }
    }
}

/// Holds back the tail of the text that could still become a stop sequence, and ends
/// the text at the first stop sequence.
struct Stops<'a> {
    stops: &'a [String],
    /// Longest stop sequence, in bytes.
    longest: usize,
    text: String,
    sent: usize,
}

impl<'a> Stops<'a> {
    fn new(stops: &'a [String]) -> Self {
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

impl Engine {
    /// The prompt's tokens.
    fn prompt_ids(&self, prompt: &Prompt) -> Result<Vec<u32>, RequestError> {
        let tok = self.tokenizer();
        let (text, special) = match prompt {
            Prompt::Text(t) => (t.clone(), true),
            Prompt::Chat(msgs) => {
                if !tok.has_chat_template() {
                    return Err(RequestError::invalid(
                        "this model has no chat template; send a raw prompt".into(),
                    ));
                }
                let t = tok.apply_chat_template(msgs, true).map_err(|e| {
                    RequestError::invalid(format!(
                        "the chat template failed on these messages: {e}"
                    ))
                })?;
                // The template writes any special tokens itself.
                (t, false)
            }
        };
        let ids = tok
            .encode(&text, special)
            .map_err(|e| RequestError::invalid(format!("cannot tokenize the prompt: {e}")))?;
        if ids.is_empty() {
            return Err(RequestError::invalid_param(
                "prompt",
                "the prompt encodes to no tokens".into(),
            ));
        }
        Ok(ids)
    }

    /// Tokenizes `req` and checks it against the context. Costs no memory and runs no
    /// pass.
    ///
    /// # Errors
    /// [`RequestError::Invalid`] if the request cannot run on this model.
    pub fn prepare(&self, req: &GenerationRequest) -> Result<PreparedRequest, RequestError> {
        let prompt = self.prompt_ids(&req.prompt)?;
        let prompt_tokens = prompt.len();
        let ctx = self.max_context();
        if prompt_tokens >= ctx {
            return Err(RequestError::invalid(format!(
                "the prompt has {prompt_tokens} tokens; the context holds {ctx}, leaving no room to generate"
            )));
        }
        let room = ctx - prompt_tokens;
        let max_new = match req.max_tokens {
            Some(m) if m > room => {
                return Err(RequestError::invalid_param(
                    "max_tokens",
                    format!(
                        "the prompt has {prompt_tokens} tokens and max_tokens is {m}; the context holds {ctx}"
                    ),
                ));
            }
            Some(m) => m,
            None => req.default_max_tokens.min(room),
        };
        Ok(PreparedRequest { prompt, max_new })
    }

    /// Runs `req` on this thread, reporting through `on_event`. Reserves KV room with
    /// every other live conversation, which is what lets several requests share an
    /// engine, and returns it when the request ends. `cancel` is checked before every
    /// prefill chunk and every token.
    ///
    /// # Errors
    /// [`RequestError::Invalid`] before any pass runs, [`RequestError::Admission`] if
    /// the plan has no room, [`RequestError::Generation`] if a pass fails.
    pub fn run_request(
        &self,
        req: &GenerationRequest,
        cancel: &AtomicBool,
        on_event: impl FnMut(GenerationEvent) -> bool,
    ) -> Result<GenerationOutcome, RequestError> {
        let prepared = self.prepare(req)?;
        let kv = self
            .reserve_kv(prepared.positions())
            .map_err(RequestError::Admission)?;
        self.run_prepared(req, &prepared, &kv, cancel, on_event)
    }

    /// Runs a request [`Engine::prepare`] checked, under a reservation for its
    /// [`PreparedRequest::positions`]. `req` must be the request that was prepared.
    ///
    /// # Errors
    /// [`RequestError::Generation`] if a pass fails.
    pub fn run_prepared(
        &self,
        req: &GenerationRequest,
        prepared: &PreparedRequest,
        _kv: &KvReservation,
        cancel: &AtomicBool,
        mut on_event: impl FnMut(GenerationEvent) -> bool,
    ) -> Result<GenerationOutcome, RequestError> {
        let (prompt, max_new) = (&prepared.prompt, prepared.max_new);
        let prompt_tokens = prepared.prompt_tokens();
        if !on_event(GenerationEvent::Started { prompt_tokens }) {
            return Ok(GenerationOutcome {
                finish: Finish::Stopped,
                prompt_tokens,
                completion_tokens: 0,
                cached_tokens: 0,
            });
        }

        let opts = GenerateOptions {
            max_new,
            sampling: req.sampling,
            prefill_chunk: req.prefill_chunk,
            deadline: req.deadline,
            cache_scope: req.cache_scope.clone(),
        };
        let mut decoder = self.tokenizer().stream();
        let mut stops = Stops::new(&req.stops);
        let (mut hit_stop, mut gone) = (false, false);
        let g = self
            .generate_with(prompt, &opts, cancel, |id| {
                if !on_event(GenerationEvent::Token(id)) {
                    gone = true;
                    return false;
                }
                let piece = decoder.push(id).unwrap_or_default();
                let (out, hit) = stops.push(&piece);
                if !out.is_empty() && !on_event(GenerationEvent::Text(out)) {
                    gone = true;
                    return false;
                }
                if hit {
                    hit_stop = true;
                    return false;
                }
                true
            })
            .map_err(RequestError::Generation)?;
        let ended = |finish| GenerationOutcome {
            finish,
            prompt_tokens,
            completion_tokens: g.tokens.len(),
            cached_tokens: g.cached_tokens,
        };
        if g.finish == Finish::Cancelled {
            return Ok(ended(Finish::Cancelled));
        }
        if gone {
            return Ok(ended(Finish::Stopped));
        }
        if !hit_stop {
            // A trailing incomplete character, as `decode` of every id shows it.
            let tail = decoder.finish().unwrap_or_default();
            let (out, hit) = stops.push(&tail);
            if !out.is_empty() && !on_event(GenerationEvent::Text(out)) {
                return Ok(ended(Finish::Stopped));
            }
            if hit {
                hit_stop = true;
            } else {
                let rest = stops.flush();
                if !rest.is_empty() && !on_event(GenerationEvent::Text(rest)) {
                    return Ok(ended(Finish::Stopped));
                }
            }
        }
        let finish = if hit_stop {
            Finish::StopSequence
        } else {
            g.finish
        };
        Ok(ended(finish))
    }
}

#[cfg(test)]
mod tests {
    use super::Stops;

    fn run(stops: &[&str], pieces: &[&str]) -> (String, bool) {
        let stops: Vec<String> = stops.iter().map(|&s| s.to_owned()).collect();
        let mut s = Stops::new(&stops);
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
