//! `transformer_moe`: decoder-only transformer, RMSNorm, RoPE attention (MHA/GQA/MQA,
//! optional q/k norm), and a routed SwiGLU mixture of experts in every layer.
//!
//! Every variation is configuration (ADR-003): dimensions come from `config.json` or
//! GGUF metadata, the rest from the manifest's `architecture.config`, and tensor
//! names from a preset (`hf`, `hf_fused`, `gguf`) with optional per-name overrides.
//! No code path depends on which model family a checkpoint belongs to.

use std::sync::Arc;

use qubivex_core::{ExpertIdx, LayerIdx};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::manifest::{ManifestFormat, SemVer};
use crate::registry::{
    ArchConfig, ArchitectureFactory, AttentionPart, ConfigSource, ExpertPart, ModelDims,
    TensorRole, TensorSchema, TensorSpec,
};
use crate::{GgufMetadata, GgufValue, ModelError};

/// The factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct TransformerMoe;

/// Query/key normalisation before RoPE.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QkNorm {
    /// None.
    None,
    /// One RMSNorm over all heads (weights `heads * head_dim` wide).
    Full,
    /// RMSNorm per head (weights `head_dim` wide, shared by the heads).
    PerHead,
}

/// How router logits become expert weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouterScoring {
    /// Softmax over all experts.
    Softmax,
    /// Independent sigmoid per expert.
    Sigmoid,
}

/// How expert weights are stored.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "layout")]
pub enum ExpertLayout {
    /// One tensor per expert and part: `[ffn, hidden]` gate and up, `[hidden, ffn]` down.
    PerExpert {
        /// Gate pattern (`{L}` layer, `{E}` expert).
        gate: String,
        /// Up pattern.
        up: String,
        /// Down pattern.
        down: String,
    },
    /// Per layer: gate and up stacked `[experts, 2 * ffn, hidden]`, down `[experts, hidden, ffn]`.
    StackedGateUp {
        /// Gate-up pattern (`{L}`).
        gate_up: String,
        /// Down pattern.
        down: String,
    },
    /// Per layer: `[experts, ffn, hidden]` gate and up, `[experts, hidden, ffn]` down.
    Stacked {
        /// Gate pattern (`{L}`).
        gate: String,
        /// Up pattern.
        up: String,
        /// Down pattern.
        down: String,
    },
}

/// Tensor-name patterns. `{L}` is the layer, `{E}` the expert.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TensorNames {
    /// Token embedding.
    pub embedding: String,
    /// Output head.
    pub output: String,
    /// Final norm.
    pub final_norm: String,
    /// Pre-attention norm.
    pub attn_norm: String,
    /// Query projection.
    pub q: String,
    /// Key projection.
    pub k: String,
    /// Value projection.
    pub v: String,
    /// Output projection.
    pub o: String,
    /// Query norm.
    pub q_norm: String,
    /// Key norm.
    pub k_norm: String,
    /// Pre-FFN norm.
    pub ffn_norm: String,
    /// Router.
    pub router: String,
    /// Experts.
    pub experts: ExpertLayout,
}

impl TensorNames {
    /// A built-in preset: `hf` (per-expert tensors), `hf_fused` (stacked gate-up,
    /// as newer `transformers` writes), or `gguf` (the standard GGUF tensor names, stacked).
    #[must_use]
    pub fn preset(name: &str) -> Option<Self> {
        let hf = |experts| Self {
            embedding: "model.embed_tokens.weight".into(),
            output: "lm_head.weight".into(),
            final_norm: "model.norm.weight".into(),
            attn_norm: "model.layers.{L}.input_layernorm.weight".into(),
            q: "model.layers.{L}.self_attn.q_proj.weight".into(),
            k: "model.layers.{L}.self_attn.k_proj.weight".into(),
            v: "model.layers.{L}.self_attn.v_proj.weight".into(),
            o: "model.layers.{L}.self_attn.o_proj.weight".into(),
            q_norm: "model.layers.{L}.self_attn.q_norm.weight".into(),
            k_norm: "model.layers.{L}.self_attn.k_norm.weight".into(),
            ffn_norm: "model.layers.{L}.post_attention_layernorm.weight".into(),
            router: "model.layers.{L}.mlp.gate.weight".into(),
            experts,
        };
        match name {
            "hf" => Some(hf(ExpertLayout::PerExpert {
                gate: "model.layers.{L}.mlp.experts.{E}.gate_proj.weight".into(),
                up: "model.layers.{L}.mlp.experts.{E}.up_proj.weight".into(),
                down: "model.layers.{L}.mlp.experts.{E}.down_proj.weight".into(),
            })),
            "hf_fused" => Some(hf(ExpertLayout::StackedGateUp {
                gate_up: "model.layers.{L}.mlp.experts.gate_up_proj".into(),
                down: "model.layers.{L}.mlp.experts.down_proj".into(),
            })),
            "gguf" => Some(Self {
                embedding: "token_embd.weight".into(),
                output: "output.weight".into(),
                final_norm: "output_norm.weight".into(),
                attn_norm: "blk.{L}.attn_norm.weight".into(),
                q: "blk.{L}.attn_q.weight".into(),
                k: "blk.{L}.attn_k.weight".into(),
                v: "blk.{L}.attn_v.weight".into(),
                o: "blk.{L}.attn_output.weight".into(),
                q_norm: "blk.{L}.attn_q_norm.weight".into(),
                k_norm: "blk.{L}.attn_k_norm.weight".into(),
                ffn_norm: "blk.{L}.ffn_norm.weight".into(),
                router: "blk.{L}.ffn_gate_inp.weight".into(),
                experts: ExpertLayout::Stacked {
                    gate: "blk.{L}.ffn_gate_exps.weight".into(),
                    up: "blk.{L}.ffn_up_exps.weight".into(),
                    down: "blk.{L}.ffn_down_exps.weight".into(),
                },
            }),
            _ => None,
        }
    }

    fn check(&self) -> Result<(), ModelError> {
        let need = |what: &str, p: &str, l: bool, e: bool| {
            let ok = p.matches("{L}").count() == usize::from(l)
                && p.matches("{E}").count() == usize::from(e);
            if ok {
                Ok(())
            } else {
                Err(cfg_err(format!(
                    "tensor name {what} {p:?} must contain {}{{L}} and {}{{E}}",
                    if l { "one " } else { "no " },
                    if e { "one " } else { "no " }
                )))
            }
        };
        need("embedding", &self.embedding, false, false)?;
        need("output", &self.output, false, false)?;
        need("final_norm", &self.final_norm, false, false)?;
        for (w, p) in [
            ("attn_norm", &self.attn_norm),
            ("q", &self.q),
            ("k", &self.k),
            ("v", &self.v),
            ("o", &self.o),
            ("q_norm", &self.q_norm),
            ("k_norm", &self.k_norm),
            ("ffn_norm", &self.ffn_norm),
            ("router", &self.router),
        ] {
            need(w, p, true, false)?;
        }
        match &self.experts {
            ExpertLayout::PerExpert { gate, up, down } => {
                for (w, p) in [("gate", gate), ("up", up), ("down", down)] {
                    need(w, p, true, true)?;
                }
            }
            ExpertLayout::StackedGateUp { gate_up, down } => {
                need("gate_up", gate_up, true, false)?;
                need("down", down, true, false)?;
            }
            ExpertLayout::Stacked { gate, up, down } => {
                for (w, p) in [("gate", gate), ("up", up), ("down", down)] {
                    need(w, p, true, false)?;
                }
            }
        }
        Ok(())
    }
}

/// The parsed config.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MoeConfig {
    /// Hidden size.
    pub hidden: u64,
    /// Layers.
    pub layers: u32,
    /// Query heads.
    pub heads: u64,
    /// Key/value heads.
    pub kv_heads: u64,
    /// Per-head dimension.
    pub head_dim: u64,
    /// Vocabulary size.
    pub vocab: u64,
    /// Routed experts per layer.
    pub experts: u32,
    /// Experts per token.
    pub top_k: u32,
    /// Expert FFN width.
    pub expert_ffn: u64,
    /// Renormalise the top-K weights to sum to 1.
    pub norm_topk_prob: bool,
    /// RMSNorm epsilon.
    pub rms_eps: f64,
    /// RoPE base.
    pub rope_theta: f64,
    /// Clamp q, k and v to `[-clip, clip]` after projection, if set.
    pub clip_qkv: Option<f64>,
    /// Maximum context, if stated.
    pub max_context: Option<u64>,
    /// The output head reuses the embedding.
    pub tie_embeddings: bool,
    /// Q/K norm.
    pub qk_norm: QkNorm,
    /// Router scoring.
    pub router_scoring: RouterScoring,
    /// Tensor names.
    pub names: TensorNames,
}

fn cfg_err(detail: impl Into<String>) -> ModelError {
    ModelError::Config(detail.into())
}

/// `architecture.config` keys for this architecture.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestOptions {
    #[serde(default)]
    tensor_names: Option<Value>,
    #[serde(default)]
    qk_norm: Option<QkNorm>,
    #[serde(default)]
    router_scoring: Option<RouterScoring>,
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
struct Values<'a> {
    hf: Option<&'a Value>,
    gguf: Option<&'a GgufMetadata>,
    overrides: &'a serde_json::Map<String, Value>,
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
    fn get(&self, key: &str, gguf: Option<&str>) -> Result<Option<Value>, ModelError> {
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

    fn u64(&self, key: &str, gguf: Option<&str>) -> Result<Option<u64>, ModelError> {
        self.get(key, gguf)?
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| cfg_err(format!("{key} = {v} is not a non-negative integer")))
            })
            .transpose()
    }

    fn req_u64(&self, key: &str, gguf: Option<&str>) -> Result<u64, ModelError> {
        self.u64(key, gguf)?.ok_or_else(|| self.missing(key, gguf))
    }

    fn f64(&self, key: &str, gguf: Option<&str>) -> Result<Option<f64>, ModelError> {
        self.get(key, gguf)?
            .map(|v| {
                v.as_f64()
                    .ok_or_else(|| cfg_err(format!("{key} = {v} is not a number")))
            })
            .transpose()
    }

    fn bool(&self, key: &str, gguf: Option<&str>) -> Result<Option<bool>, ModelError> {
        self.get(key, gguf)?
            .map(|v| {
                v.as_bool()
                    .ok_or_else(|| cfg_err(format!("{key} = {v} is not a boolean")))
            })
            .transpose()
    }

    fn missing(&self, key: &str, gguf: Option<&str>) -> ModelError {
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

fn nonzero(what: &str, v: u64) -> Result<u64, ModelError> {
    if v == 0 {
        Err(cfg_err(format!("{what} must be positive")))
    } else {
        Ok(v)
    }
}

fn small(what: &str, v: u64, max: u64) -> Result<u32, ModelError> {
    u32::try_from(v)
        .ok()
        .filter(|&x| u64::from(x) <= max && x > 0)
        .ok_or_else(|| cfg_err(format!("{what} = {v} is out of range 1..={max}")))
}

/// Rejects config features this architecture does not implement yet.
fn unsupported(v: &Values<'_>) -> Result<(), ModelError> {
    if let Some(s) = v.u64("decoder_sparse_step", None)? {
        if s != 1 {
            return Err(cfg_err(format!(
                "decoder_sparse_step = {s}: dense FFN layers are not supported yet"
            )));
        }
    }
    if let Some(Value::Array(a)) = v.get("mlp_only_layers", None)? {
        if !a.is_empty() {
            return Err(cfg_err(
                "mlp_only_layers is not empty: dense FFN layers are not supported yet",
            ));
        }
    }
    for key in [
        "first_k_dense_replace",
        "n_shared_experts",
        "shared_expert_intermediate_size",
    ] {
        if v.u64(key, None)?.is_some_and(|n| n > 0) {
            return Err(cfg_err(format!(
                "{key} > 0: shared or dense experts are not supported yet"
            )));
        }
    }
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

fn names(opt: Option<&Value>, format: ManifestFormat) -> Result<TensorNames, ModelError> {
    let default = match format {
        ManifestFormat::Gguf => "gguf",
        ManifestFormat::Safetensors => "hf",
    };
    let preset = |p: &str| {
        TensorNames::preset(p).ok_or_else(|| cfg_err(format!("unknown tensor_names preset {p:?}")))
    };
    let n = match opt {
        None => preset(default)?,
        Some(Value::String(p)) => preset(p)?,
        Some(Value::Object(o)) => {
            let base = o.get("base").and_then(Value::as_str).unwrap_or(default);
            let mut n = preset(base)?;
            for (k, val) in o {
                if k == "base" {
                    continue;
                }
                let s = val
                    .as_str()
                    .ok_or_else(|| cfg_err(format!("tensor_names.{k} must be a string")))?
                    .to_owned();
                let slot = match (k.as_str(), &mut n.experts) {
                    ("embedding", _) => &mut n.embedding,
                    ("output", _) => &mut n.output,
                    ("final_norm", _) => &mut n.final_norm,
                    ("attn_norm", _) => &mut n.attn_norm,
                    ("q", _) => &mut n.q,
                    ("k", _) => &mut n.k,
                    ("v", _) => &mut n.v,
                    ("o", _) => &mut n.o,
                    ("q_norm", _) => &mut n.q_norm,
                    ("k_norm", _) => &mut n.k_norm,
                    ("ffn_norm", _) => &mut n.ffn_norm,
                    ("router", _) => &mut n.router,
                    (
                        "expert_gate",
                        ExpertLayout::PerExpert { gate, .. } | ExpertLayout::Stacked { gate, .. },
                    ) => gate,
                    (
                        "expert_up",
                        ExpertLayout::PerExpert { up, .. } | ExpertLayout::Stacked { up, .. },
                    ) => up,
                    ("expert_gate_up", ExpertLayout::StackedGateUp { gate_up, .. }) => gate_up,
                    (
                        "expert_down",
                        ExpertLayout::PerExpert { down, .. }
                        | ExpertLayout::Stacked { down, .. }
                        | ExpertLayout::StackedGateUp { down, .. },
                    ) => down,
                    _ => {
                        return Err(cfg_err(format!(
                            "tensor_names.{k} is not a name of the {base:?} layout"
                        )));
                    }
                };
                *slot = s;
            }
            n
        }
        Some(v) => {
            return Err(cfg_err(format!(
                "tensor_names must be a preset name or an object, got {v}"
            )));
        }
    };
    n.check()?;
    Ok(n)
}

impl ArchitectureFactory for TransformerMoe {
    fn id(&self) -> &'static str {
        "transformer_moe"
    }

    fn version(&self) -> SemVer {
        SemVer {
            major: 1,
            minor: 0,
            patch: 0,
        }
    }

    fn description(&self) -> &'static str {
        "decoder-only transformer with RoPE attention and routed SwiGLU experts in every layer"
    }

    fn parse_config(&self, src: &ConfigSource<'_>) -> Result<ArchConfig, ModelError> {
        let m = src.manifest;
        let opts: ManifestOptions =
            serde_json::from_value(Value::Object(m.architecture.config.clone()))
                .map_err(|e| cfg_err(format!("architecture.config: {e}")))?;
        let v = Values {
            hf: src.hf_config,
            gguf: src.gguf,
            overrides: &m.overrides.config,
        };
        if v.hf.is_none() && v.gguf.is_none() {
            return Err(cfg_err("no config.json or GGUF metadata"));
        }
        unsupported(&v)?;

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
            None => match src.gguf.and_then(|g| g.get("tokenizer.ggml.tokens")) {
                Some(GgufValue::Array(t)) => t.len() as u64,
                _ => return Err(v.missing("vocab_size", Some("vocab_size"))),
            },
        };
        let vocab = nonzero("vocab_size", vocab)?;
        let experts = small(
            "num_experts",
            v.req_u64("num_experts", Some("expert_count"))?,
            1 << 16,
        )?;
        let top_k = small(
            "num_experts_per_tok",
            v.req_u64("num_experts_per_tok", Some("expert_used_count"))?,
            u64::from(experts),
        )?;
        let expert_ffn = match v.u64("moe_intermediate_size", Some("expert_feed_forward_length"))? {
            Some(n) => n,
            None => v.req_u64("intermediate_size", Some("feed_forward_length"))?,
        };
        let expert_ffn = nonzero("expert FFN width", expert_ffn)?;
        let norm_topk_prob = v
            .bool("norm_topk_prob", Some("expert_weights_norm"))?
            .ok_or_else(|| v.missing("norm_topk_prob", Some("expert_weights_norm")))?;
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
        let cfg = MoeConfig {
            hidden,
            layers,
            heads,
            kv_heads,
            head_dim,
            vocab,
            experts,
            top_k,
            expert_ffn,
            norm_topk_prob,
            rms_eps,
            rope_theta,
            clip_qkv,
            max_context,
            tie_embeddings,
            qk_norm: opts.qk_norm.unwrap_or(QkNorm::None),
            router_scoring: opts.router_scoring.unwrap_or(RouterScoring::Softmax),
            names: names(opts.tensor_names.as_ref(), m.weights.format)?,
        };
        Ok(ArchConfig {
            dims: ModelDims {
                layers,
                hidden,
                vocab,
                experts,
                experts_per_token: top_k,
                max_context,
            },
            describe: serde_json::to_value(&cfg).map_err(|e| cfg_err(e.to_string()))?,
            specific: Arc::new(cfg),
        })
    }

    fn tensor_schema(&self, cfg: &ArchConfig) -> Result<TensorSchema, ModelError> {
        let c = cfg
            .specific
            .downcast_ref::<MoeConfig>()
            .ok_or_else(|| cfg_err("config was not parsed by transformer_moe"))?;
        let n = &c.names;
        let (h, ffn, e) = (c.hidden, c.expert_ffn, u64::from(c.experts));
        let mul = |a: u64, b: u64, what: &str| {
            a.checked_mul(b)
                .ok_or_else(|| cfg_err(format!("{what} overflows")))
        };
        let q_out = mul(c.heads, c.head_dim, "heads * head_dim")?;
        let kv_out = mul(c.kv_heads, c.head_dim, "kv_heads * head_dim")?;
        let ffn2 = mul(2, ffn, "2 * expert FFN width")?;
        let mut t = Vec::new();
        let mut push =
            |name: String, role, shape: Vec<u64>| t.push(TensorSpec { name, role, shape });

        push(n.embedding.clone(), TensorRole::Embedding, vec![c.vocab, h]);
        for l in 0..c.layers {
            let layer = LayerIdx(l);
            let at = |p: &str| p.replace("{L}", &l.to_string());
            let attn = |part| TensorRole::Attention { layer, part };
            push(at(&n.attn_norm), TensorRole::Norm { layer }, vec![h]);
            push(at(&n.q), attn(AttentionPart::Q), vec![q_out, h]);
            push(at(&n.k), attn(AttentionPart::K), vec![kv_out, h]);
            push(at(&n.v), attn(AttentionPart::V), vec![kv_out, h]);
            push(at(&n.o), attn(AttentionPart::O), vec![h, q_out]);
            match c.qk_norm {
                QkNorm::None => {}
                QkNorm::Full => {
                    push(at(&n.q_norm), attn(AttentionPart::QNorm), vec![q_out]);
                    push(at(&n.k_norm), attn(AttentionPart::KNorm), vec![kv_out]);
                }
                QkNorm::PerHead => {
                    push(at(&n.q_norm), attn(AttentionPart::QNorm), vec![c.head_dim]);
                    push(at(&n.k_norm), attn(AttentionPart::KNorm), vec![c.head_dim]);
                }
            }
            push(at(&n.ffn_norm), TensorRole::Norm { layer }, vec![h]);
            push(at(&n.router), TensorRole::Router { layer }, vec![e, h]);
            let stack = |part| TensorRole::RoutedExpertStack {
                layer,
                part,
                experts: c.experts,
            };
            match &n.experts {
                ExpertLayout::PerExpert { gate, up, down } => {
                    for x in 0..c.experts {
                        let ex = |p: &str| at(p).replace("{E}", &x.to_string());
                        let role = |part| TensorRole::RoutedExpert {
                            layer,
                            expert: ExpertIdx(x),
                            part,
                        };
                        push(ex(gate), role(ExpertPart::Gate), vec![ffn, h]);
                        push(ex(up), role(ExpertPart::Up), vec![ffn, h]);
                        push(ex(down), role(ExpertPart::Down), vec![h, ffn]);
                    }
                }
                ExpertLayout::StackedGateUp { gate_up, down } => {
                    push(at(gate_up), stack(ExpertPart::GateUp), vec![e, ffn2, h]);
                    push(at(down), stack(ExpertPart::Down), vec![e, h, ffn]);
                }
                ExpertLayout::Stacked { gate, up, down } => {
                    push(at(gate), stack(ExpertPart::Gate), vec![e, ffn, h]);
                    push(at(up), stack(ExpertPart::Up), vec![e, ffn, h]);
                    push(at(down), stack(ExpertPart::Down), vec![e, h, ffn]);
                }
            }
        }
        push(n.final_norm.clone(), TensorRole::FinalNorm, vec![h]);
        if !c.tie_embeddings {
            push(n.output.clone(), TensorRole::OutputHead, vec![c.vocab, h]);
        }
        Ok(TensorSchema { tensors: t })
    }
}
