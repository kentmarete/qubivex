//! Context management (ADR-021): a request longer than the model's context is prepared
//! to fit instead of refused, and a request that fits is never touched.
//!
//! The host calls [`ContextEngine::prepare`] only after its own exact context check
//! refused a request for length. The engine then
//!
//! 1. keeps every system and developer message verbatim, and refuses
//!    ([`ContextError::Mandatory`]) when they alone exceed the budget;
//! 2. reduces a message that alone is too long (a pasted document or source file) to its
//!    most relevant blocks, verbatim, with markers naming the lines left out;
//! 3. keeps the most recent turns verbatim and condenses older ones into a note of quoted
//!    excerpts (never paraphrase), with each excerpt's message number and role;
//! 4. counts the result exactly with the model's own template and tokenizer
//!    ([`TokenCounter`]) after every change, and refuses ([`ContextError::Unfit`]) rather
//!    than send something that does not fit.
//!
//! Checkpoints ([`store`]) keep a condensed note stable from one turn to the next, so
//! the prompt keeps its prefix and the host's prefix cache can reuse its KV.
//!
//! The crate knows nothing of models, tensors, HTTP or threads. Everything it reads is
//! the request's own text, so nothing can cross from one user's request to another's.

mod compact;
mod rank;
pub mod store;
mod text;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::Serialize;
use thiserror::Error;

pub use store::{Checkpoint, CheckpointStore, FileStore, MemoryStore, StoreError};

/// One chat message, as the template sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    /// `system`, `developer`, `user` or `assistant`.
    pub role: String,
    /// The text.
    pub content: String,
}

impl Message {
    /// A message.
    #[must_use]
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    /// System and developer messages: instructions, never compacted.
    #[must_use]
    pub fn is_instruction(&self) -> bool {
        self.role == "system" || self.role == "developer"
    }
}

/// What a request generates from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Input {
    /// Messages, rendered with the model's chat template.
    Chat(Vec<Message>),
    /// A raw prompt.
    Text(String),
}

/// Exact token counts from the model's own template and tokenizer. Implemented by the
/// host; the counts must be the ones its context check uses.
pub trait TokenCounter {
    /// Tokens of the prompt these messages render to.
    ///
    /// # Errors
    /// The template or tokenizer failed.
    fn count_chat(&self, messages: &[Message]) -> Result<usize, String>;
    /// Tokens of a raw prompt.
    ///
    /// # Errors
    /// The tokenizer failed.
    fn count_text(&self, text: &str) -> Result<usize, String>;
    /// Tokens of a piece of text on its own, without special tokens. Used to choose what
    /// to keep; whether the result fits is always decided by a whole count.
    ///
    /// # Errors
    /// The tokenizer failed.
    fn count_fragment(&self, text: &str) -> Result<usize, String>;
}

/// The token budget of one request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Budget {
    /// The context of one conversation: the most restrictive of the model's limit, the
    /// configured limit and what the memory plan holds.
    pub context: usize,
    /// Tokens kept free for the answer.
    pub reserve: usize,
    /// Most tokens the prompt may have.
    pub input: usize,
}

impl Budget {
    /// The budget for a request asking for `max_tokens` (else `default_max_tokens`) on a
    /// context of `context`, under a server cap on prompts of `prompt_cap`. The reserve
    /// is at most half the context, so a large `max_tokens` cannot leave no room for
    /// the prompt; it is a bound, as `max_tokens` is.
    #[must_use]
    pub fn new(
        context: usize,
        max_tokens: Option<usize>,
        default_max_tokens: usize,
        prompt_cap: Option<usize>,
    ) -> Self {
        let reserve = max_tokens
            .unwrap_or(default_max_tokens)
            .clamp(1, (context / 2).max(1));
        let input = context
            .saturating_sub(reserve)
            .min(prompt_cap.unwrap_or(usize::MAX));
        Self {
            context,
            reserve,
            input,
        }
    }
}

/// Settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContextConfig {
    /// Prepare requests that do not fit. Off: they are refused, as without this crate.
    pub enabled: bool,
    /// The share of the room left after instructions and the current message that is
    /// kept for the note on older turns, in [0, 1]; recent turns get the rest first.
    pub note_share: f64,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            note_share: 0.25,
        }
    }
}

/// Lets a host stop a preparation: a cancellation flag and a deadline, checked between
/// steps.
#[derive(Clone, Copy, Debug, Default)]
pub struct Interrupt<'a> {
    /// Set when nobody wants the result any more.
    pub cancel: Option<&'a AtomicBool>,
    /// When to give up.
    pub deadline: Option<Instant>,
}

impl Interrupt<'_> {
    pub(crate) fn check(&self) -> Result<(), ContextError> {
        if self.cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err(ContextError::Cancelled);
        }
        if self.deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(ContextError::DeadlineExceeded);
        }
        Ok(())
    }
}

/// Why a request could not be prepared.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContextError {
    /// The system and developer messages alone exceed the budget. They are never
    /// shortened.
    #[error(
        "the system and developer messages alone take {tokens} tokens; the context of {} \
         tokens, with {} kept for the answer, leaves {} for the prompt. Use a model or a \
         --context with a longer context, or shorten the instructions",
        budget.context, budget.reserve, budget.input
    )]
    Mandatory {
        /// Their tokens.
        tokens: usize,
        /// The budget.
        budget: Budget,
    },
    /// The request could not be reduced to the budget without losing what must stay.
    #[error(
        "the prompt has {tokens} tokens and cannot be reduced to the {} the context of {} \
         leaves after keeping {} for the answer without losing the current message",
        budget.input, budget.context, budget.reserve
    )]
    Unfit {
        /// The prompt's tokens as sent.
        tokens: usize,
        /// The budget.
        budget: Budget,
    },
    /// The host cancelled.
    #[error("context preparation was cancelled")]
    Cancelled,
    /// The deadline passed.
    #[error("context preparation ran past the request's deadline")]
    DeadlineExceeded,
    /// The template or tokenizer failed.
    #[error("cannot count the prompt's tokens: {0}")]
    Count(String),
}

/// What happened to one message of the request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MessageFate {
    /// Its index in the request's messages (0 for a raw prompt).
    pub index: usize,
    /// Its role (`prompt` for a raw prompt).
    pub role: String,
    /// What was done with it.
    #[serde(flatten)]
    pub fate: Fate,
}

/// What was done with a message.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "fate", rename_all = "snake_case")]
pub enum Fate {
    /// Sent verbatim.
    Kept,
    /// Quoted in the note on earlier turns: `excerpts` of its `units` sentences and code
    /// blocks; none means it is only counted there.
    Condensed {
        /// Sentences and code blocks quoted.
        excerpts: usize,
        /// Sentences and code blocks it has.
        units: usize,
    },
    /// Sent with some lines left out, each range named in the text by a marker.
    Reduced {
        /// Lines left out, as `[first, last]`, counting from 1.
        omitted_lines: Vec<[usize; 2]>,
    },
}

/// How a checkpoint was used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointUse {
    /// No older turns were condensed.
    None,
    /// None matched; the note was built from scratch.
    Miss,
    /// A stored note was reused as it was, and extended when more turns were condensed.
    Hit,
}

/// What a preparation did, for logs, metrics and diagnostics. Holds no text.
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    /// The budget.
    pub budget: Budget,
    /// The prompt's tokens as sent.
    pub original_tokens: usize,
    /// The prompt's tokens as prepared.
    pub prepared_tokens: usize,
    /// Older turns were condensed.
    pub compacted: bool,
    /// A long message was reduced to its relevant blocks.
    pub reduced: bool,
    /// Messages condensed into the note.
    pub messages_condensed: usize,
    /// Sentences and code blocks the note quotes.
    pub excerpts: usize,
    /// Blocks left out of reduced messages.
    pub blocks_omitted: usize,
    /// Blocks kept in reduced messages: the retrieved ones.
    pub blocks_kept: usize,
    /// How a checkpoint was used.
    pub checkpoint: CheckpointUse,
    /// Whole counts it took to fit.
    pub counts: usize,
    /// Each message's fate, in order.
    pub messages: Vec<MessageFate>,
    /// Time spent ranking and reducing long messages.
    #[serde(serialize_with = "micros")]
    pub retrieval_time: Duration,
    /// Time spent condensing older turns.
    #[serde(serialize_with = "micros")]
    pub compaction_time: Duration,
    /// Time spent in all.
    #[serde(serialize_with = "micros")]
    pub total_time: Duration,
}

fn micros<S: serde::Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_u64(u64::try_from(d.as_micros()).unwrap_or(u64::MAX))
}

impl Report {
    /// Tokens the preparation took out.
    #[must_use]
    pub const fn tokens_saved(&self) -> usize {
        self.original_tokens.saturating_sub(self.prepared_tokens)
    }

    /// What was done, as the `x-qubivex-context` header says it: `compacted`,
    /// `reduced`, or both.
    #[must_use]
    pub fn strategies(&self) -> String {
        match (self.compacted, self.reduced) {
            (true, true) => "compacted,reduced".into(),
            (true, false) => "compacted".into(),
            (false, true) => "reduced".into(),
            (false, false) => "none".into(),
        }
    }
}

/// A prepared request: what to send instead, and what was done.
#[derive(Clone, Debug)]
pub struct Prepared {
    /// The input to send.
    pub input: Input,
    /// What was done.
    pub report: Report,
}

/// One request to prepare.
#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    /// What the client sent.
    pub input: &'a Input,
    /// Its budget.
    pub budget: Budget,
    /// Its prompt's tokens as sent, from the host's check.
    pub original_tokens: usize,
    /// Who may share its checkpoints: the host's cache scope. Requests of different
    /// scopes never see each other's checkpoints.
    pub scope: Option<&'a str>,
    /// What the checkpoints are for: the model (its tokenizer and context), so a note
    /// sized for one model is not reused on another.
    pub namespace: &'a str,
}

/// Prepares requests that do not fit. Shared by every request of a host.
#[derive(Debug)]
pub struct ContextEngine {
    config: ContextConfig,
    store: Arc<dyn CheckpointStore>,
}

impl ContextEngine {
    /// An engine keeping checkpoints in `store`.
    #[must_use]
    pub fn new(config: ContextConfig, store: Arc<dyn CheckpointStore>) -> Self {
        Self { config, store }
    }

    /// An engine keeping checkpoints in memory, within [`MemoryStore::DEFAULT_BYTES`].
    #[must_use]
    pub fn in_memory(config: ContextConfig) -> Self {
        Self::new(
            config,
            Arc::new(MemoryStore::new(MemoryStore::DEFAULT_BYTES)),
        )
    }

    /// Its settings.
    #[must_use]
    pub const fn config(&self) -> ContextConfig {
        self.config
    }

    /// Whether it prepares requests at all.
    #[must_use]
    pub const fn enabled(&self) -> bool {
        self.config.enabled
    }

    /// Its checkpoint store.
    #[must_use]
    pub fn store(&self) -> &dyn CheckpointStore {
        &*self.store
    }

    /// Prepares `req`, which the host found too long, to fit `req.budget`.
    ///
    /// # Errors
    /// [`ContextError::Mandatory`] or [`ContextError::Unfit`] if it cannot fit without
    /// losing what must stay; [`ContextError::Cancelled`] or
    /// [`ContextError::DeadlineExceeded`] if `interrupt` says so;
    /// [`ContextError::Count`] if counting failed.
    pub fn prepare(
        &self,
        req: &Request<'_>,
        counter: &dyn TokenCounter,
        interrupt: Interrupt<'_>,
    ) -> Result<Prepared, ContextError> {
        compact::prepare(self, req, counter, interrupt)
    }
}

#[cfg(test)]
mod tests {
    use super::Budget;

    #[test]
    fn the_reserve_follows_max_tokens_and_keeps_half_the_context() {
        assert_eq!(
            Budget::new(4096, Some(256), 512, None),
            Budget {
                context: 4096,
                reserve: 256,
                input: 3840
            }
        );
        assert_eq!(Budget::new(4096, None, 512, None).reserve, 512);
        assert_eq!(Budget::new(4096, Some(100_000), 512, None).input, 2048);
        assert_eq!(Budget::new(4096, Some(256), 512, Some(1000)).input, 1000);
        // A tiny context still leaves the answer one token.
        assert_eq!(Budget::new(1, None, 512, None).reserve, 1);
    }
}
