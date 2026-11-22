//! Requests longer than the context (ADR-021). [`Engine::prepare_managed`] is
//! [`Engine::prepare`] with one more way through: a request the context refuses is
//! handed to a [`ContextEngine`], and what it returns is checked by `prepare` again,
//! exactly like any other request. A request that fits takes the same single call it
//! always did.

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use qubivex_context::{
    Budget, ContextEngine, Input, Interrupt, Message, Report, Request, TokenCounter,
};
use qubivex_tokenizer::ChatMessage;

use crate::engine::Engine;
use crate::request::{GenerationRequest, PreparedRequest, Prompt, RequestError};

/// Where an engine's context limit comes from: the least of these is
/// [`Engine::max_context`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextLimits {
    /// The model's own limit (its manifest or `config.json`, else 4096).
    pub model: usize,
    /// The limit the engine was loaded with, if one was asked for.
    pub configured: Option<usize>,
    /// What the memory plan holds for one conversation, when there is a plan.
    pub memory: Option<usize>,
    /// The context a conversation gets: the least of the above.
    pub effective: usize,
}

/// A request ready to run: the request to run (rewritten only if it was managed), its
/// tokens, and what context management did.
#[derive(Clone, Debug)]
pub struct Managed {
    /// The request to run: the one given, or its prepared version.
    pub request: GenerationRequest,
    /// Its tokens, checked against the context.
    pub prepared: PreparedRequest,
    /// What context management did; `None` when the request fitted as sent.
    pub report: Option<Report>,
}

/// How [`Engine::prepare_managed`] manages a request that does not fit.
#[derive(Clone, Copy, Debug, Default)]
pub struct ManageOptions<'a> {
    /// The host's cap on prompt tokens, which the prepared prompt must also respect.
    pub prompt_cap: Option<usize>,
    /// Set when nobody wants the result any more.
    pub cancel: Option<&'a AtomicBool>,
    /// When to give up preparing.
    pub deadline: Option<Instant>,
}

/// Exact counts: the same chat template and tokenizer `prepare` uses.
struct Counter<'a>(&'a Engine);

impl TokenCounter for Counter<'_> {
    fn count_chat(&self, m: &[Message]) -> Result<usize, String> {
        let msgs = m
            .iter()
            .map(|m| ChatMessage::new(m.role.as_str(), m.content.as_str()))
            .collect();
        self.0
            .prompt_ids(&Prompt::Chat(msgs))
            .map(|ids| ids.len())
            .map_err(|e| e.to_string())
    }

    fn count_text(&self, t: &str) -> Result<usize, String> {
        self.0
            .prompt_ids(&Prompt::Text(t.to_owned()))
            .map(|ids| ids.len())
            .map_err(|e| e.to_string())
    }

    fn count_fragment(&self, t: &str) -> Result<usize, String> {
        self.0
            .tokenizer()
            .encode(t, false)
            .map(|ids| ids.len())
            .map_err(|e| e.to_string())
    }
}

impl Engine {
    /// What checkpoints made on this engine are for: the model and its context, so a
    /// note sized for one is never reused on another.
    #[must_use]
    pub fn context_namespace(&self) -> String {
        format!(
            "{}@{}",
            self.metadata().manifest.model.id,
            self.max_context()
        )
    }

    /// [`Engine::prepare`], and when it refuses the request for length and `ctx` is on,
    /// the request prepared by `ctx` to fit, checked by `prepare` again.
    ///
    /// # Errors
    /// What `prepare` returns, and [`RequestError::Context`] when the request cannot be
    /// prepared to fit. With `ctx` off or `None`, a request too long for the context
    /// is [`RequestError::ContextOverflow`], as from `prepare`.
    pub fn prepare_managed(
        &self,
        req: GenerationRequest,
        ctx: Option<&ContextEngine>,
        opts: &ManageOptions<'_>,
    ) -> Result<Managed, RequestError> {
        let original_tokens = match self.prepare(&req) {
            Ok(prepared) => {
                return Ok(Managed {
                    request: req,
                    prepared,
                    report: None,
                });
            }
            Err(RequestError::ContextOverflow { prompt_tokens, .. }) => prompt_tokens,
            Err(e) => return Err(e),
        };
        let Some(ctx) = ctx.filter(|c| c.enabled()) else {
            return Err(RequestError::ContextOverflow {
                prompt_tokens: original_tokens,
                context: self.max_context(),
            });
        };
        let input = match &req.prompt {
            Prompt::Chat(m) => Input::Chat(
                m.iter()
                    .map(|m| Message::new(m.role.as_str(), m.content.as_str()))
                    .collect(),
            ),
            Prompt::Text(t) => Input::Text(t.clone()),
        };
        let namespace = self.context_namespace();
        let r = Request {
            input: &input,
            budget: Budget::new(
                self.max_context(),
                req.max_tokens,
                req.default_max_tokens,
                opts.prompt_cap,
            ),
            original_tokens,
            scope: req.cache_scope.as_deref(),
            namespace: &namespace,
        };
        let interrupt = Interrupt {
            cancel: opts.cancel,
            deadline: opts.deadline,
        };
        let p = ctx
            .prepare(&r, &Counter(self), interrupt)
            .map_err(RequestError::Context)?;
        let mut request = req;
        request.prompt = match p.input {
            Input::Chat(m) => Prompt::Chat(
                m.into_iter()
                    .map(|m| ChatMessage::new(m.role, m.content))
                    .collect(),
            ),
            Input::Text(t) => Prompt::Text(t),
        };
        // The final check, as for any request: the prepared prompt goes through the same
        // template, tokenizer and context check.
        let prepared = self.prepare(&request)?;
        Ok(Managed {
            request,
            prepared,
            report: Some(p.report),
        })
    }
}
