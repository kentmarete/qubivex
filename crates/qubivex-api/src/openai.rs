//! The OpenAI request schemas this gateway accepts. Fields it does not implement are
//! refused with a clear message rather than silently ignored when they would change
//! the output (`n > 1`, tools, logprobs); fields that would not (`user`, penalties of
//! zero) are accepted and ignored.

use std::time::Duration;

use qubivex_runtime::scheduler::Priority;
use serde::Deserialize;
use serde_json::Value;

use crate::error::ApiError;

/// A string, or a list of strings.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

impl OneOrMany {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(s) => vec![s],
            Self::Many(v) => v,
        }
    }
}

/// Message content: a string or text parts.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum Content {
    Text(String),
    Parts(Vec<Part>),
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Part {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Message {
    pub(crate) role: String,
    #[serde(default)]
    pub(crate) content: Option<Content>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct StreamOptions {
    #[serde(default)]
    pub(crate) include_usage: bool,
}

/// Sampling and length fields shared by both endpoints.
#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct Common {
    #[serde(default)]
    pub(crate) stream: bool,
    pub(crate) stream_options: Option<StreamOptions>,
    pub(crate) max_tokens: Option<u64>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_k: Option<u32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) seed: Option<u64>,
    pub(crate) n: Option<u32>,
    pub(crate) stop: Option<OneOrMany>,
    pub(crate) logprobs: Option<Value>,
    pub(crate) presence_penalty: Option<f32>,
    pub(crate) frequency_penalty: Option<f32>,
    pub(crate) logit_bias: Option<Value>,
    /// A QUBIVEX field: `low`, `normal` or `high`.
    pub(crate) priority: Option<String>,
    /// A QUBIVEX field: seconds this request may take, queue included.
    pub(crate) timeout: Option<f64>,
    /// Who the request is for: the client, for fairness.
    pub(crate) user: Option<String>,
}

/// `POST /v1/chat/completions`.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ChatRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<Message>,
    #[serde(flatten)]
    pub(crate) common: Common,
    pub(crate) max_completion_tokens: Option<u64>,
    pub(crate) tools: Option<Value>,
    pub(crate) tool_choice: Option<Value>,
    pub(crate) response_format: Option<Value>,
}

/// `POST /v1/completions`.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct CompletionRequest {
    pub(crate) model: String,
    pub(crate) prompt: OneOrMany,
    #[serde(flatten)]
    pub(crate) common: Common,
    #[serde(default)]
    pub(crate) echo: bool,
    pub(crate) suffix: Option<String>,
    pub(crate) best_of: Option<u32>,
}

/// Validated generation parameters.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Params {
    pub(crate) max_tokens: Option<usize>,
    pub(crate) temperature: f32,
    pub(crate) top_k: u32,
    pub(crate) top_p: f32,
    pub(crate) seed: u64,
    pub(crate) stops: Vec<String>,
    pub(crate) stream: bool,
    pub(crate) include_usage: bool,
    pub(crate) priority: Priority,
    pub(crate) timeout: Option<Duration>,
    pub(crate) client: Option<String>,
}

fn zero_or_absent(v: Option<f32>) -> bool {
    v.is_none_or(|x| x == 0.0)
}

impl Common {
    pub(crate) fn params(self, max_tokens: Option<u64>) -> Result<Params, ApiError> {
        if self.n.is_some_and(|n| n != 1) {
            return Err(ApiError::invalid_param("n", "only n = 1 is supported"));
        }
        if self
            .logprobs
            .as_ref()
            .is_some_and(|v| !v.is_null() && v != &Value::Bool(false))
        {
            return Err(ApiError::invalid_param(
                "logprobs",
                "logprobs are not supported",
            ));
        }
        if !zero_or_absent(self.presence_penalty) || !zero_or_absent(self.frequency_penalty) {
            return Err(ApiError::invalid(
                "presence and frequency penalties are not supported",
            ));
        }
        if self
            .logit_bias
            .as_ref()
            .is_some_and(|v| v.as_object().is_none_or(|o| !o.is_empty()))
        {
            return Err(ApiError::invalid_param(
                "logit_bias",
                "logit_bias is not supported",
            ));
        }
        let temperature = self.temperature.unwrap_or(1.0);
        if !(0.0..=2.0).contains(&temperature) {
            return Err(ApiError::invalid_param(
                "temperature",
                "temperature must be between 0 and 2",
            ));
        }
        let top_p = self.top_p.unwrap_or(1.0);
        if !(top_p > 0.0 && top_p <= 1.0) {
            return Err(ApiError::invalid_param("top_p", "top_p must be in (0, 1]"));
        }
        let max_tokens = max_tokens
            .map(|m| {
                usize::try_from(m).ok().filter(|&m| m > 0).ok_or_else(|| {
                    ApiError::invalid_param("max_tokens", "max_tokens must be positive")
                })
            })
            .transpose()?;
        let priority = match self.priority.as_deref() {
            None | Some("normal") => Priority::Normal,
            Some("low") => Priority::Low,
            Some("high") => Priority::High,
            Some(_) => {
                return Err(ApiError::invalid_param(
                    "priority",
                    "priority must be low, normal or high",
                ));
            }
        };
        let timeout = self
            .timeout
            .map(|t| {
                Duration::try_from_secs_f64(t)
                    .ok()
                    .filter(|d| !d.is_zero())
                    .ok_or_else(|| {
                        ApiError::invalid_param("timeout", "timeout must be positive seconds")
                    })
            })
            .transpose()?;
        let stops = self.stop.map(OneOrMany::into_vec).unwrap_or_default();
        if stops.len() > 4 || stops.iter().any(String::is_empty) {
            return Err(ApiError::invalid_param(
                "stop",
                "stop takes up to 4 non-empty strings",
            ));
        }
        Ok(Params {
            max_tokens,
            temperature,
            top_k: self.top_k.unwrap_or(0),
            top_p,
            seed: self.seed.unwrap_or(0),
            stops,
            stream: self.stream,
            include_usage: self.stream_options.is_some_and(|o| o.include_usage),
            priority,
            timeout,
            client: self.user,
        })
    }
}

impl ChatRequest {
    /// The messages as (role, text), checked.
    pub(crate) fn messages(&self) -> Result<Vec<(String, String)>, ApiError> {
        if self.tools.as_ref().is_some_and(|t| !t.is_null())
            || self
                .tool_choice
                .as_ref()
                .is_some_and(|t| !t.is_null() && t != "none")
        {
            return Err(ApiError::invalid_param(
                "tools",
                "tool calls are not supported",
            ));
        }
        if let Some(f) = &self.response_format
            && f.get("type")
                .and_then(Value::as_str)
                .is_some_and(|t| t != "text")
        {
            return Err(ApiError::invalid_param(
                "response_format",
                "only the text response format is supported",
            ));
        }
        if self.messages.is_empty() {
            return Err(ApiError::invalid_param("messages", "messages is empty"));
        }
        self.messages
            .iter()
            .map(|m| {
                if !["system", "developer", "user", "assistant"].contains(&m.role.as_str()) {
                    return Err(ApiError::invalid_param(
                        "messages",
                        format!("unsupported message role {:?}", m.role),
                    ));
                }
                let role = if m.role == "developer" {
                    "system".to_owned()
                } else {
                    m.role.clone()
                };
                let text = match &m.content {
                    None => String::new(),
                    Some(Content::Text(t)) => t.clone(),
                    Some(Content::Parts(parts)) => {
                        let mut t = String::new();
                        for p in parts {
                            if p.kind != "text" {
                                return Err(ApiError::invalid_param(
                                    "messages",
                                    format!("unsupported content part {:?}; only text", p.kind),
                                ));
                            }
                            t.push_str(p.text.as_deref().unwrap_or_default());
                        }
                        t
                    }
                };
                Ok((role, text))
            })
            .collect()
    }
}

impl CompletionRequest {
    /// The prompt text, checked.
    pub(crate) fn prompt(&self) -> Result<String, ApiError> {
        if self.echo {
            return Err(ApiError::invalid_param("echo", "echo is not supported"));
        }
        if self.suffix.as_ref().is_some_and(|s| !s.is_empty()) {
            return Err(ApiError::invalid_param("suffix", "suffix is not supported"));
        }
        if self.best_of.is_some_and(|b| b != 1) {
            return Err(ApiError::invalid_param(
                "best_of",
                "only best_of = 1 is supported",
            ));
        }
        match self.prompt.clone().into_vec().as_slice() {
            [one] => Ok(one.clone()),
            _ => Err(ApiError::invalid_param("prompt", "send exactly one prompt")),
        }
    }
}
