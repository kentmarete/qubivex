//! Tokenizers and chat templates (QUBIVEX_DESIGN §3, ADR-004).
//!
//! - [`Tokenizer::from_hf_dir`] loads a Hugging Face `tokenizer.json`, with special
//!   tokens and the chat template from `tokenizer_config.json` and end-of-sequence ids
//!   from `generation_config.json` / `config.json`.
//! - [`Tokenizer::from_gguf`] rebuilds a byte-level BPE tokenizer from the vocabulary
//!   embedded in a GGUF file ([`GgufVocab`]). Only pre-tokenizer types whose result has
//!   been checked against the official `tokenizer.json` are accepted; anything else is
//!   refused with a message to supply `tokenizer.json` instead.
//! - Chat templates run in a sandboxed MiniJinja environment: no file system access,
//!   a fuel limit against runaway loops, and Python-compatible string methods. The
//!   settings match the Jinja environment `transformers` uses.
//!
//! The tokenizer is pure Rust (`tokenizers` with `fancy-regex`, no C or C++ code).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::Serialize;
use thiserror::Error;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

/// Largest tokenizer file read.
const MAX_TOKENIZER_BYTES: u64 = 64 << 20;
/// Largest config file read.
const MAX_CONFIG_BYTES: u64 = 4 << 20;
/// Template instructions allowed per render (a guard against runaway templates).
const TEMPLATE_FUEL: u64 = 1_000_000;

/// Tokenizer errors.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TokenizerError {
    /// A file could not be read.
    #[error("cannot read {}", path.display())]
    Io {
        /// The file.
        path: PathBuf,
        /// The cause.
        #[source]
        source: std::io::Error,
    },
    /// A tokenizer or config file is malformed.
    #[error("{}: {detail}", path.display())]
    Invalid {
        /// The file.
        path: PathBuf,
        /// What is wrong.
        detail: String,
    },
    /// The GGUF vocabulary cannot be rebuilt faithfully.
    #[error("GGUF tokenizer: {0}")]
    Gguf(String),
    /// Encoding or decoding failed.
    #[error("tokenizer: {0}")]
    Codec(String),
    /// The chat template is missing, invalid or failed to render.
    #[error("chat template: {0}")]
    Template(String),
}

/// One chat message.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ChatMessage {
    /// `system`, `user` or `assistant`.
    pub role: String,
    /// The text.
    pub content: String,
}

impl ChatMessage {
    /// A message.
    #[must_use]
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// A tokenizer vocabulary as GGUF stores it (`tokenizer.ggml.*`).
#[derive(Clone, Debug, Default)]
pub struct GgufVocab {
    /// `tokenizer.ggml.model`, for example `gpt2`.
    pub model: String,
    /// `tokenizer.ggml.pre`, the pre-tokenizer type.
    pub pre: Option<String>,
    /// Token strings, by id.
    pub tokens: Vec<String>,
    /// Token types, by id (1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte).
    pub token_types: Vec<i32>,
    /// BPE merges, `"left right"`.
    pub merges: Vec<String>,
    /// Beginning-of-sequence id.
    pub bos: Option<u32>,
    /// End-of-sequence id.
    pub eos: Option<u32>,
    /// `tokenizer.chat_template`.
    pub chat_template: Option<String>,
}

/// Pre-tokenizer types whose rebuilt tokenizer was checked against the official
/// `tokenizer.json`, and whether they normalize to NFC.
fn gguf_pre(pre: &str) -> Option<bool> {
    match pre {
        "olmo" => Some(true), // model-name-ok: GGUF tokenizer.ggml.pre format identifier
        "gpt-2" | "default" => Some(false),
        _ => None,
    }
}

const TYPE_CONTROL: i32 = 3;
const TYPE_USER_DEFINED: i32 = 4;

/// A tokenizer with its special tokens and chat template.
pub struct Tokenizer {
    inner: HfTokenizer,
    bos: Option<(u32, String)>,
    eos: Vec<u32>,
    eos_text: Option<String>,
    template: Option<String>,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab", &self.vocab_size())
            .field("bos", &self.bos)
            .field("eos", &self.eos)
            .field("has_template", &self.template.is_some())
            .finish()
    }
}

fn read_small(path: &Path, max: u64) -> Result<Vec<u8>, TokenizerError> {
    use std::io::Read;
    let io = |source| TokenizerError::Io {
        path: path.to_owned(),
        source,
    };
    let f = std::fs::File::open(path).map_err(io)?;
    let mut v = Vec::new();
    f.take(max + 1).read_to_end(&mut v).map_err(io)?;
    if v.len() as u64 > max {
        return Err(TokenizerError::Invalid {
            path: path.to_owned(),
            detail: format!("larger than {max} bytes"),
        });
    }
    Ok(v)
}

fn read_json(path: &Path) -> Result<Option<serde_json::Value>, TokenizerError> {
    if !path.is_file() {
        return Ok(None);
    }
    let b = read_small(path, MAX_CONFIG_BYTES)?;
    serde_json::from_slice(&b)
        .map(Some)
        .map_err(|e| TokenizerError::Invalid {
            path: path.to_owned(),
            detail: e.to_string(),
        })
}

/// A special token in `tokenizer_config.json`: a string, or `{"content": ...}`.
fn token_text(v: Option<&serde_json::Value>) -> Option<String> {
    match v? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => o.get("content")?.as_str().map(str::to_owned),
        _ => None,
    }
}

/// `eos_token_id` as one id or a list.
fn ids(v: Option<&serde_json::Value>) -> Vec<u32> {
    match v {
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .and_then(|x| u32::try_from(x).ok())
            .into_iter()
            .collect(),
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_u64().and_then(|x| u32::try_from(x).ok()))
            .collect(),
        _ => Vec::new(),
    }
}

impl Tokenizer {
    /// Loads `tokenizer_json`, and special tokens and the chat template from the
    /// `tokenizer_config.json`, `generation_config.json` and `config.json` next to it.
    ///
    /// # Errors
    /// [`TokenizerError::Io`] or [`TokenizerError::Invalid`].
    pub fn from_hf_dir(tokenizer_json: &Path) -> Result<Self, TokenizerError> {
        let bytes = read_small(tokenizer_json, MAX_TOKENIZER_BYTES)?;
        let inner = HfTokenizer::from_bytes(&bytes).map_err(|e| TokenizerError::Invalid {
            path: tokenizer_json.to_owned(),
            detail: e.to_string(),
        })?;
        let dir = tokenizer_json.parent().unwrap_or_else(|| Path::new("."));
        let tc = read_json(&dir.join("tokenizer_config.json"))?;
        let gc = read_json(&dir.join("generation_config.json"))?;
        let mc = read_json(&dir.join("config.json"))?;
        let id_of = |t: &str| inner.token_to_id(t);

        let bos_text = token_text(tc.as_ref().and_then(|c| c.get("bos_token")));
        let bos = bos_text.and_then(|t| id_of(&t).map(|i| (i, t)));
        let eos_text = token_text(tc.as_ref().and_then(|c| c.get("eos_token")));
        let mut eos: Vec<u32> = Vec::new();
        eos.extend(ids(gc.as_ref().and_then(|c| c.get("eos_token_id"))));
        eos.extend(ids(mc.as_ref().and_then(|c| c.get("eos_token_id"))));
        eos.extend(eos_text.as_deref().and_then(id_of));
        let mut seen = HashSet::new();
        eos.retain(|i| seen.insert(*i));
        let template = tc
            .as_ref()
            .and_then(|c| c.get("chat_template"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        Ok(Self {
            inner,
            bos,
            eos,
            eos_text,
            template,
        })
    }

    /// Rebuilds a byte-level BPE tokenizer from a GGUF vocabulary.
    ///
    /// # Errors
    /// [`TokenizerError::Gguf`] for a model or pre-tokenizer type that cannot be
    /// rebuilt faithfully, or an inconsistent vocabulary.
    pub fn from_gguf(v: &GgufVocab) -> Result<Self, TokenizerError> {
        let g = |m: String| TokenizerError::Gguf(m);
        if v.model != "gpt2" {
            return Err(g(format!(
                "tokenizer.ggml.model {:?} is not supported; supply tokenizer.json in the manifest",
                v.model
            )));
        }
        let pre = v.pre.as_deref().unwrap_or("default");
        let nfc = gguf_pre(pre).ok_or_else(|| {
            g(format!(
                "pre-tokenizer {pre:?} has not been verified against an official tokenizer.json; supply tokenizer.json in the manifest"
            ))
        })?;
        if v.tokens.len() != v.token_types.len() {
            return Err(g(format!(
                "{} tokens but {} token types",
                v.tokens.len(),
                v.token_types.len()
            )));
        }
        let n = u32::try_from(v.tokens.len()).map_err(|_| g("too many tokens".into()))?;
        let mut vocab = tokenizers::models::bpe::Vocab::with_capacity(v.tokens.len());
        for (i, t) in (0..n).zip(&v.tokens) {
            if vocab.insert(t.clone(), i).is_some() {
                return Err(g(format!("duplicate token {t:?}")));
            }
        }
        let merges: Vec<(String, String)> = v
            .merges
            .iter()
            .map(|m| {
                m.split_once(' ')
                    .map(|(a, b)| (a.to_owned(), b.to_owned()))
                    .ok_or_else(|| g(format!("merge {m:?} is not \"left right\"")))
            })
            .collect::<Result<_, _>>()?;
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .map_err(|e| g(e.to_string()))?;
        let mut inner = HfTokenizer::new(bpe);
        if nfc {
            inner
                .with_normalizer(Some(NFC))
                .map_err(|e| g(e.to_string()))?;
        }
        inner.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
        inner.with_decoder(Some(ByteLevelDecoder::new(false, true, true)));
        let mut special = Vec::new();
        let mut user = Vec::new();
        for (t, &ty) in v.tokens.iter().zip(&v.token_types) {
            match ty {
                TYPE_CONTROL => special.push(AddedToken::from(t.clone(), true).normalized(false)),
                TYPE_USER_DEFINED => user.push(AddedToken::from(t.clone(), false).normalized(true)),
                _ => {}
            }
        }
        inner
            .add_special_tokens(special)
            .map_err(|e| g(e.to_string()))?;
        inner.add_tokens(user).map_err(|e| g(e.to_string()))?;
        let text = |id: u32| v.tokens.get(id as usize).cloned();
        Ok(Self {
            bos: v.bos.and_then(|b| text(b).map(|t| (b, t))),
            eos: v.eos.into_iter().collect(),
            eos_text: v.eos.and_then(text),
            template: v.chat_template.clone(),
            inner,
        })
    }

    /// Token ids of `text`. `add_special_tokens` applies the tokenizer's own
    /// post-processing (prompts already rendered by a chat template pass `false`).
    ///
    /// # Errors
    /// [`TokenizerError::Codec`].
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        self.inner
            .encode(text, add_special_tokens)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| TokenizerError::Codec(e.to_string()))
    }

    /// Text of `ids`.
    ///
    /// # Errors
    /// [`TokenizerError::Codec`].
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| TokenizerError::Codec(e.to_string()))
    }

    /// Vocabulary size, added tokens included.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// End-of-sequence ids: generation stops on any of them.
    #[must_use]
    pub fn eos_ids(&self) -> &[u32] {
        &self.eos
    }

    /// The beginning-of-sequence id, if the model has one.
    #[must_use]
    pub fn bos_id(&self) -> Option<u32> {
        self.bos.as_ref().map(|b| b.0)
    }

    /// Whether a chat template is available.
    #[must_use]
    pub fn has_chat_template(&self) -> bool {
        self.template.is_some()
    }

    /// Renders the chat template, as `transformers`' `apply_chat_template(...,
    /// tokenize=False)` does.
    ///
    /// # Errors
    /// [`TokenizerError::Template`] if there is no template or rendering fails.
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        let src = self
            .template
            .as_deref()
            .ok_or_else(|| TokenizerError::Template("the model has no chat template".into()))?;
        render_template(
            src,
            messages,
            add_generation_prompt,
            self.bos.as_ref().map_or("", |b| b.1.as_str()),
            self.eos_text.as_deref().unwrap_or(""),
        )
    }

    /// An incremental decoder for streaming output.
    #[must_use]
    pub fn stream(&self) -> StreamDecoder<'_> {
        StreamDecoder {
            tok: self,
            ids: Vec::new(),
            emitted: 0,
        }
    }
}

/// Renders a chat template in a sandbox matching `transformers`' Jinja settings
/// (`trim_blocks`, `lstrip_blocks`), with a fuel limit and no template loader.
fn render_template(
    src: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    bos: &str,
    eos: &str,
) -> Result<String, TokenizerError> {
    let t = |e: minijinja::Error| TokenizerError::Template(e.to_string());
    let mut env = minijinja::Environment::new();
    env.set_syntax(
        minijinja::syntax::SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .map_err(t)?,
    );
    env.set_fuel(Some(TEMPLATE_FUEL));
    minijinja_contrib::add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    env.add_template("chat", src).map_err(t)?;
    env.get_template("chat")
        .map_err(t)?
        .render(minijinja::context! {
            messages => messages
                .iter()
                .map(|m| minijinja::Value::from_pairs([("role", m.role.as_str()), ("content", m.content.as_str())]))
                .collect::<Vec<_>>(),
            add_generation_prompt => add_generation_prompt,
            bos_token => bos,
            eos_token => eos,
        })
        .map_err(t)
}

/// Turns a growing id sequence into text pieces, holding back incomplete UTF-8.
///
/// Byte-level BPE tokens can split a character; the decoder only emits text once the
/// decoded prefix ends on a complete character.
pub struct StreamDecoder<'a> {
    tok: &'a Tokenizer,
    ids: Vec<u32>,
    emitted: usize,
}

impl StreamDecoder<'_> {
    /// Adds a token and returns any newly completed text.
    ///
    /// # Errors
    /// [`TokenizerError::Codec`].
    pub fn push(&mut self, id: u32) -> Result<String, TokenizerError> {
        self.ids.push(id);
        let text = self.tok.decode(&self.ids, true)?;
        if text.ends_with('\u{fffd}')
            || text.len() < self.emitted
            || !text.is_char_boundary(self.emitted)
        {
            return Ok(String::new());
        }
        let piece = text[self.emitted..].to_owned();
        self.emitted = text.len();
        Ok(piece)
    }

    /// Ends the stream: the text not yet returned, including a trailing incomplete
    /// UTF-8 sequence (as U+FFFD), so the pieces join to `decode` of every id.
    ///
    /// # Errors
    /// As [`Tokenizer::decode`].
    pub fn finish(&mut self) -> Result<String, TokenizerError> {
        let text = self.tok.decode(&self.ids, true)?;
        let piece = text
            .get(self.emitted..)
            .map(str::to_owned)
            .unwrap_or_default();
        self.emitted = text.len();
        Ok(piece)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_matches_transformers_settings() {
        let src = "{{ bos_token }}{% for m in messages %}\n<{{ m.role }}>{{ m.content.strip() }}\n{% endfor %}{% if add_generation_prompt %}<a>{% endif %}";
        let out = render_template(
            src,
            &[ChatMessage::new("user", " hi ")],
            true,
            "<s>",
            "</s>",
        )
        .unwrap();
        // trim_blocks drops the newline after a block tag; lstrip_blocks the indent before it.
        assert_eq!(out, "<s><user>hi\n<a>");
        let e = render_template(
            "{{ raise_exception('no system role') }}",
            &[],
            false,
            "",
            "",
        )
        .unwrap_err();
        assert!(e.to_string().contains("no system role"));
        // A huge range is refused outright; nested loops run until the fuel is gone.
        let e = render_template(
            "{% for i in range(100000000) %}x{% endfor %}",
            &[],
            false,
            "",
            "",
        )
        .unwrap_err();
        assert!(e.to_string().contains("too many elements"), "{e}");
        let nested =
            "{% for i in range(10000) %}{% for j in range(10000) %}x{% endfor %}{% endfor %}";
        let e = render_template(nested, &[], false, "", "").unwrap_err();
        assert!(e.to_string().contains("fuel"), "{e}");
        let e = render_template("{% include 'x' %}", &[], false, "", "").unwrap_err();
        assert!(e.to_string().contains("template"), "{e}");
    }

    #[test]
    fn gguf_rejects_what_it_cannot_verify() {
        let v = GgufVocab {
            model: "bert".into(),
            ..GgufVocab::default()
        };
        assert!(
            Tokenizer::from_gguf(&v)
                .unwrap_err()
                .to_string()
                .contains("supply tokenizer.json")
        );
        let v = GgufVocab {
            model: "gpt2".into(),
            pre: Some("exotic".into()),
            ..GgufVocab::default()
        };
        assert!(
            Tokenizer::from_gguf(&v)
                .unwrap_err()
                .to_string()
                .contains("not been verified")
        );
        let v = GgufVocab {
            model: "gpt2".into(),
            pre: Some("gpt-2".into()),
            tokens: vec!["a".into(), "b".into(), "ab".into()],
            token_types: vec![1, 1, 1],
            merges: vec!["a b".into()],
            ..GgufVocab::default()
        };
        let t = Tokenizer::from_gguf(&v).unwrap();
        assert_eq!(t.encode("ab", false).unwrap(), vec![2]);
        assert_eq!(t.decode(&[0, 1], false).unwrap(), "ab");
        let bad = GgufVocab {
            merges: vec!["nospace".into()],
            ..v
        };
        assert!(Tokenizer::from_gguf(&bad).is_err());
    }
}
