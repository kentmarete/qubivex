//! Config parsing shared by the built-in architectures: reading values from
//! `config.json` or GGUF metadata with the manifest's overrides, and the attention
//! dimensions every decoder-only transformer has.

use qubivex_core::LayerIdx;
use serde_json::Value;

use crate::registry::{AttentionPart, TensorRole, TensorSpec};
use crate::{GgufMetadata, GgufValue, ModelError};

pub(crate) fn cfg_err(detail: impl Into<String>) -> ModelError {
    ModelError::Config(detail.into())
}

fn gguf_json(v: &GgufValue) -> Option<Value> {
    Some(match v {
        GgufValue::Bool(b) => Value::Bool(*b),
        GgufValue::String(s) => Value::String(s.clone()),
        GgufValue::F32(_) | GgufValue::F64(_) => serde_json::Number::from_f64(v.as_f64()?)?.into(),
        GgufValue::Array(_) => return None,
        _ => match v.as_u64() {
            Some(u) => u.into(),
            None => v.as_f64().and_then(serde_json::Number::from_f64)?.into(),
        },
    })
}

/// Config values from the checkpoint, with the manifest's `overrides.config` merged in.
pub(crate) struct Values<'a> {
    pub(crate) hf: Option<&'a Value>,
    pub(crate) gguf: Option<&'a GgufMetadata>,
    pub(crate) overrides: &'a serde_json::Map<String, Value>,
}

fn same(a: &Value, b: &Value) -> bool {
    match (a.as_f64(), b.as_f64()) {
        // Relative 1e-6: a GGUF value stored as f32 must still match the same decimal.
        (Some(x), Some(y)) => (x - y).abs() <= 1e-6 * x.abs().max(y.abs()),
        _ => a == b,
    }
}

impl Values<'_> {
    /// `key` is the `config.json` name; `gguf` the `<arch>.` suffix, if GGUF has one.
    pub(crate) fn get(&self, key: &str, gguf: Option<&str>) -> Result<Option<Value>, ModelError> {
        let from_src = match (self.hf, self.gguf) {
            (Some(h), _) => h.get(key).filter(|v| !v.is_null()).cloned(),
            (None, Some(g)) => gguf.and_then(|s| g.arch_value(s)).and_then(gguf_json),
            (None, None) => None,
        };
        match (from_src, self.overrides.get(key)) {
            (Some(s), Some(o)) if !same(&s, o) => Err(cfg_err(format!(
                "overrides.config.{key} = {o} contradicts the checkpoint's {s}"
            ))),
            (Some(s), _) => Ok(Some(s)),
            (None, o) => Ok(o.cloned()),
        }
    }

    pub(crate) fn u64(&self, key: &str, gguf: Option<&str>) -> Result<Option<u64>, ModelError> {
        self.get(key, gguf)?
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| cfg_err(format!("{key} = {v} is not a non-negative integer")))
            })
            .transpose()
    }

    pub(crate) fn req_u64(&self, key: &str, gguf: Option<&str>) -> Result<u64, ModelError> {
        self.u64(key, gguf)?.ok_or_else(|| self.missing(key, gguf))
    }

    pub(crate) fn f64(&self, key: &str, gguf: Option<&str>) -> Result<Option<f64>, ModelError> {
        self.get(key, gguf)?
            .map(|v| {
                v.as_f64()
                    .ok_or_else(|| cfg_err(format!("{key} = {v} is not a number")))
            })
            .transpose()
    }

    pub(crate) fn bool(&self, key: &str, gguf: Option<&str>) -> Result<Option<bool>, ModelError> {
        self.get(key, gguf)?
            .map(|v| {
                v.as_bool()
                    .ok_or_else(|| cfg_err(format!("{key} = {v} is not a boolean")))
            })
            .transpose()
    }

    pub(crate) fn missing(&self, key: &str, gguf: Option<&str>) -> ModelError {
        match (self.gguf, gguf) {
            (Some(_), Some(s)) => cfg_err(format!(
                "GGUF metadata has no <architecture>.{s}; set overrides.config.{key} in the manifest"
            )),
            (Some(_), None) => cfg_err(format!(
                "GGUF has no such value; set overrides.config.{key} in the manifest"
            )),
            _ => cfg_err(format!(
                "config.json has no {key}; set overrides.config.{key} in the manifest"
            )),
        }
    }
}

pub(crate) fn nonzero(what: &str, v: u64) -> Result<u64, ModelError> {
    if v == 0 {
        Err(cfg_err(format!("{what} must be positive")))
    } else {
        Ok(v)
    }
}

pub(crate) fn small(what: &str, v: u64, max: u64) -> Result<u32, ModelError> {
    u32::try_from(v)
        .ok()
        .filter(|&x| u64::from(x) <= max && x > 0)
        .ok_or_else(|| cfg_err(format!("{what} = {v} is out of range 1..={max}")))
}

/// Query/key normalisation before RoPE.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QkNorm {
    /// None.
    None,
    /// One RMSNorm over all heads (weights `heads * head_dim` wide).
    Full,
    /// RMSNorm per head (weights `head_dim` wide, shared by the heads).
    PerHead,
}

/// Dimensions and constants every decoder-only transformer here reads from its
/// checkpoint.
pub(crate) struct DecoderDims {
    pub(crate) hidden: u64,
    pub(crate) layers: u32,
    pub(crate) heads: u64,
    pub(crate) kv_heads: u64,
    pub(crate) head_dim: u64,
    pub(crate) vocab: u64,
    pub(crate) rms_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) clip_qkv: Option<f64>,
    pub(crate) max_context: Option<u64>,
    pub(crate) tie_embeddings: bool,
}

/// Reads [`DecoderDims`], checking them.
pub(crate) fn decoder_dims(
    v: &Values<'_>,
    gguf: Option<&GgufMetadata>,
) -> Result<DecoderDims, ModelError> {
    let hidden = nonzero(
        "hidden_size",
        v.req_u64("hidden_size", Some("embedding_length"))?,
    )?;
    let layers = small(
        "num_hidden_layers",
        v.req_u64("num_hidden_layers", Some("block_count"))?,
        4096,
    )?;
    let heads = nonzero(
        "num_attention_heads",
        v.req_u64("num_attention_heads", Some("attention.head_count"))?,
    )?;
    let kv_heads = v
        .u64("num_key_value_heads", Some("attention.head_count_kv"))?
        .unwrap_or(heads);
    if kv_heads == 0 || heads % kv_heads != 0 {
        return Err(cfg_err(format!(
            "{heads} query heads are not a multiple of {kv_heads} key/value heads"
        )));
    }
    let head_dim = match v.u64("head_dim", Some("attention.key_length"))? {
        Some(d) => nonzero("head_dim", d)?,
        None if hidden % heads == 0 => hidden / heads,
        None => {
            return Err(cfg_err(
                "hidden_size is not a multiple of num_attention_heads; set head_dim",
            ));
        }
    };
    let vocab = match v.u64("vocab_size", Some("vocab_size"))? {
        Some(n) => n,
        None => match gguf.and_then(|g| g.get("tokenizer.ggml.tokens")) {
            Some(GgufValue::Array(t)) => t.len() as u64,
            _ => return Err(v.missing("vocab_size", Some("vocab_size"))),
        },
    };
    let vocab = nonzero("vocab_size", vocab)?;
    let rms_eps = v
        .f64("rms_norm_eps", Some("attention.layer_norm_rms_epsilon"))?
        .ok_or_else(|| v.missing("rms_norm_eps", Some("attention.layer_norm_rms_epsilon")))?;
    let rope_theta = match v.f64("rope_theta", Some("rope.freq_base"))? {
        Some(t) => t,
        None => match v.get("rope_parameters", None)? {
            Some(Value::Object(p)) => p
                .get("rope_theta")
                .and_then(Value::as_f64)
                .ok_or_else(|| v.missing("rope_theta", None))?,
            _ => return Err(v.missing("rope_theta", Some("rope.freq_base"))),
        },
    };
    if !(rms_eps.is_finite() && rms_eps > 0.0 && rope_theta.is_finite() && rope_theta > 0.0) {
        return Err(cfg_err(
            "rms_norm_eps and rope_theta must be positive and finite",
        ));
    }
    let clip_qkv = v.f64("clip_qkv", None)?;
    let max_context = v.u64("max_position_embeddings", Some("context_length"))?;
    let tie_embeddings = v.bool("tie_word_embeddings", None)?.unwrap_or(false);
    Ok(DecoderDims {
        hidden,
        layers,
        heads,
        kv_heads,
        head_dim,
        vocab,
        rms_eps,
        rope_theta,
        clip_qkv,
        max_context,
        tie_embeddings,
    })
}

/// Rejects config features no architecture here implements yet.
pub(crate) fn unsupported_common(v: &Values<'_>) -> Result<(), ModelError> {
    if v.bool("attention_bias", None)? == Some(true) {
        return Err(cfg_err(
            "attention_bias = true: attention biases are not supported yet",
        ));
    }
    if let Some(act) = v.get("hidden_act", None)? {
        if act != "silu" {
            return Err(cfg_err(format!(
                "hidden_act = {act}: only silu (SwiGLU) is supported"
            )));
        }
    }
    if v.bool("mlp_bias", None)? == Some(true) {
        return Err(cfg_err("mlp_bias = true: FFN biases are not supported yet"));
    }
    if v.bool("use_sliding_window", None)? == Some(true) {
        return Err(cfg_err(
            "use_sliding_window = true: sliding-window attention is not supported yet",
        ));
    }
    if let Some(Value::Object(rs)) = v.get("rope_scaling", None)? {
        let kind = rs
            .get("rope_type")
            .or_else(|| rs.get("type"))
            .and_then(Value::as_str);
        if kind != Some("default") {
            return Err(cfg_err(format!(
                "rope_scaling {kind:?} is not supported yet"
            )));
        }
    }
    Ok(())
}

/// Tensor-name patterns of one attention block (`{L}` is the layer).
pub(crate) struct AttentionNames<'a> {
    pub(crate) attn_norm: &'a str,
    pub(crate) q: &'a str,
    pub(crate) k: &'a str,
    pub(crate) v: &'a str,
    pub(crate) o: &'a str,
    pub(crate) q_norm: &'a str,
    pub(crate) k_norm: &'a str,
}

/// Layer `l`'s pre-attention norm and attention tensors.
pub(crate) fn attention_specs(
    l: u32,
    n: &AttentionNames<'_>,
    qk_norm: QkNorm,
    h: u64,
    q_out: u64,
    kv_out: u64,
    head_dim: u64,
) -> Vec<TensorSpec> {
    let layer = LayerIdx(l);
    let at = |p: &str| p.replace("{L}", &l.to_string());
    let attn = |part| TensorRole::Attention { layer, part };
    let mut t = vec![
        (at(n.attn_norm), TensorRole::Norm { layer }, vec![h]),
        (at(n.q), attn(AttentionPart::Q), vec![q_out, h]),
        (at(n.k), attn(AttentionPart::K), vec![kv_out, h]),
        (at(n.v), attn(AttentionPart::V), vec![kv_out, h]),
        (at(n.o), attn(AttentionPart::O), vec![h, q_out]),
    ];
    match qk_norm {
        QkNorm::None => {}
        QkNorm::Full => {
            t.push((at(n.q_norm), attn(AttentionPart::QNorm), vec![q_out]));
            t.push((at(n.k_norm), attn(AttentionPart::KNorm), vec![kv_out]));
        }
        QkNorm::PerHead => {
            t.push((at(n.q_norm), attn(AttentionPart::QNorm), vec![head_dim]));
            t.push((at(n.k_norm), attn(AttentionPart::KNorm), vec![head_dim]));
        }
    }
    t.into_iter()
        .map(|(name, role, shape)| TensorSpec { name, role, shape })
        .collect()
}
