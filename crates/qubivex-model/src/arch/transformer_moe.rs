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

pub use super::common::QkNorm;
use super::common::{
    AttentionNames, DecoderDims, Values, attention_specs, cfg_err, decoder_dims, nonzero, small,
    unsupported_common,
};
use crate::ModelError;
use crate::manifest::{ManifestFormat, SemVer};
use crate::registry::{
    ArchConfig, ArchitectureFactory, ConfigSource, ExpertPart, ModelDims, TensorRole, TensorSchema,
    TensorSpec,
};

/// The factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct TransformerMoe;

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
    unsupported_common(v)
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
        let DecoderDims {
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
        } = decoder_dims(&v, src.gguf)?;

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
            let names = AttentionNames {
                attn_norm: &n.attn_norm,
                q: &n.q,
                k: &n.k,
                v: &n.v,
                o: &n.o,
                q_norm: &n.q_norm,
                k_norm: &n.k_norm,
            };
            for spec in attention_specs(l, &names, c.qk_norm, h, q_out, kv_out, c.head_dim) {
                push(spec.name, spec.role, spec.shape);
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
