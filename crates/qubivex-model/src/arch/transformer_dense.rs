//! `transformer_dense`: decoder-only transformer, RMSNorm, RoPE attention (MHA/GQA/MQA,
//! optional q/k norm), and a dense SwiGLU FFN in every layer.
//!
//! The attention block and its configuration are the same as `transformer_moe`'s
//! (shared in `common`); the FFN is one always-active SwiGLU instead of routed experts.
//! Every variation is configuration (ADR-003): dimensions come from `config.json` or
//! GGUF metadata, the rest from the manifest's `architecture.config`, and tensor names
//! from a preset (`hf`, `gguf`) with optional per-name overrides. No code path depends
//! on which model family a checkpoint belongs to.

use std::sync::Arc;

use qubivex_core::LayerIdx;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::common::{
    AttentionNames, DecoderDims, QkNorm, Values, attention_specs, cfg_err, decoder_dims, nonzero,
    unsupported_common,
};
use crate::ModelError;
use crate::manifest::{ManifestFormat, SemVer};
use crate::registry::{
    ArchConfig, ArchitectureFactory, ConfigSource, FfnPart, ModelDims, TensorRole, TensorSchema,
    TensorSpec,
};

/// The factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct TransformerDense;

/// Tensor-name patterns. `{L}` is the layer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DenseTensorNames {
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
    /// FFN gate projection.
    pub gate: String,
    /// FFN up projection.
    pub up: String,
    /// FFN down projection.
    pub down: String,
}

impl DenseTensorNames {
    /// A built-in preset: `hf` (the `transformers` names) or `gguf` (the standard GGUF
    /// tensor names).
    #[must_use]
    pub fn preset(name: &str) -> Option<Self> {
        match name {
            "hf" => Some(Self {
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
                gate: "model.layers.{L}.mlp.gate_proj.weight".into(),
                up: "model.layers.{L}.mlp.up_proj.weight".into(),
                down: "model.layers.{L}.mlp.down_proj.weight".into(),
            }),
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
                gate: "blk.{L}.ffn_gate.weight".into(),
                up: "blk.{L}.ffn_up.weight".into(),
                down: "blk.{L}.ffn_down.weight".into(),
            }),
            _ => None,
        }
    }

    fn slot(&mut self, key: &str) -> Option<&mut String> {
        Some(match key {
            "embedding" => &mut self.embedding,
            "output" => &mut self.output,
            "final_norm" => &mut self.final_norm,
            "attn_norm" => &mut self.attn_norm,
            "q" => &mut self.q,
            "k" => &mut self.k,
            "v" => &mut self.v,
            "o" => &mut self.o,
            "q_norm" => &mut self.q_norm,
            "k_norm" => &mut self.k_norm,
            "ffn_norm" => &mut self.ffn_norm,
            "gate" => &mut self.gate,
            "up" => &mut self.up,
            "down" => &mut self.down,
            _ => return None,
        })
    }

    fn check(&self) -> Result<(), ModelError> {
        let global = [
            ("embedding", &self.embedding),
            ("output", &self.output),
            ("final_norm", &self.final_norm),
        ];
        let per_layer = [
            ("attn_norm", &self.attn_norm),
            ("q", &self.q),
            ("k", &self.k),
            ("v", &self.v),
            ("o", &self.o),
            ("q_norm", &self.q_norm),
            ("k_norm", &self.k_norm),
            ("ffn_norm", &self.ffn_norm),
            ("gate", &self.gate),
            ("up", &self.up),
            ("down", &self.down),
        ];
        for (layers, list) in [(0, &global[..]), (1, &per_layer[..])] {
            for (what, p) in list {
                if p.matches("{L}").count() != layers || p.contains("{E}") {
                    return Err(cfg_err(format!(
                        "tensor name {what} {p:?} must contain {} and no {{E}}",
                        if layers == 1 { "one {L}" } else { "no {L}" }
                    )));
                }
            }
        }
        Ok(())
    }

    fn attention(&self) -> AttentionNames<'_> {
        AttentionNames {
            attn_norm: &self.attn_norm,
            q: &self.q,
            k: &self.k,
            v: &self.v,
            o: &self.o,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
        }
    }
}

/// The parsed config.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DenseConfig {
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
    /// FFN width.
    pub ffn: u64,
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
    /// Tensor names.
    pub names: DenseTensorNames,
}

/// `architecture.config` keys for this architecture.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestOptions {
    #[serde(default)]
    tensor_names: Option<Value>,
    #[serde(default)]
    qk_norm: Option<QkNorm>,
}

/// Refuses a mixture-of-experts checkpoint, which `transformer_moe` describes.
fn not_moe(v: &Values<'_>) -> Result<(), ModelError> {
    for (key, gguf) in [
        ("num_experts", Some("expert_count")),
        ("num_local_experts", None),
        ("n_routed_experts", None),
    ] {
        if v.u64(key, gguf)?.is_some_and(|n| n > 0) {
            return Err(cfg_err(format!(
                "{key} > 0: a mixture of experts is transformer_moe, not transformer_dense"
            )));
        }
    }
    Ok(())
}

fn names(opt: Option<&Value>, format: ManifestFormat) -> Result<DenseTensorNames, ModelError> {
    let default = match format {
        ManifestFormat::Gguf => "gguf",
        ManifestFormat::Safetensors => "hf",
    };
    let preset = |p: &str| {
        DenseTensorNames::preset(p)
            .ok_or_else(|| cfg_err(format!("unknown tensor_names preset {p:?}")))
    };
    let n = match opt {
        None => preset(default)?,
        Some(Value::String(p)) => preset(p)?,
        Some(Value::Object(o)) => {
            let base = o.get("base").and_then(Value::as_str).unwrap_or(default);
            let mut n = preset(base)?;
            for (k, val) in o.iter().filter(|(k, _)| k.as_str() != "base") {
                let s = val
                    .as_str()
                    .ok_or_else(|| cfg_err(format!("tensor_names.{k} must be a string")))?;
                *n.slot(k)
                    .ok_or_else(|| cfg_err(format!("tensor_names.{k} is not a tensor name")))? =
                    s.to_owned();
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

impl ArchitectureFactory for TransformerDense {
    fn id(&self) -> &'static str {
        "transformer_dense"
    }

    fn version(&self) -> SemVer {
        SemVer {
            major: 1,
            minor: 0,
            patch: 0,
        }
    }

    fn description(&self) -> &'static str {
        "decoder-only transformer with RoPE attention and a dense SwiGLU FFN in every layer"
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
        not_moe(&v)?;
        unsupported_common(&v)?;
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
        let ffn = nonzero(
            "intermediate_size",
            v.req_u64("intermediate_size", Some("feed_forward_length"))?,
        )?;
        let cfg = DenseConfig {
            hidden,
            layers,
            heads,
            kv_heads,
            head_dim,
            vocab,
            ffn,
            rms_eps,
            rope_theta,
            clip_qkv,
            max_context,
            tie_embeddings,
            qk_norm: opts.qk_norm.unwrap_or(QkNorm::None),
            names: names(opts.tensor_names.as_ref(), m.weights.format)?,
        };
        Ok(ArchConfig {
            dims: ModelDims {
                layers,
                hidden,
                vocab,
                experts: 0,
                experts_per_token: 0,
                max_context,
            },
            describe: serde_json::to_value(&cfg).map_err(|e| cfg_err(e.to_string()))?,
            specific: Arc::new(cfg),
        })
    }

    fn tensor_schema(&self, cfg: &ArchConfig) -> Result<TensorSchema, ModelError> {
        let c = cfg
            .specific
            .downcast_ref::<DenseConfig>()
            .ok_or_else(|| cfg_err("config was not parsed by transformer_dense"))?;
        let n = &c.names;
        let (h, ffn) = (c.hidden, c.ffn);
        let mul = |a: u64, b: u64, what: &str| {
            a.checked_mul(b)
                .ok_or_else(|| cfg_err(format!("{what} overflows")))
        };
        let q_out = mul(c.heads, c.head_dim, "heads * head_dim")?;
        let kv_out = mul(c.kv_heads, c.head_dim, "kv_heads * head_dim")?;
        let mut t = vec![TensorSpec {
            name: n.embedding.clone(),
            role: TensorRole::Embedding,
            shape: vec![c.vocab, h],
        }];
        for l in 0..c.layers {
            let layer = LayerIdx(l);
            let at = |p: &str| p.replace("{L}", &l.to_string());
            t.extend(attention_specs(
                l,
                &n.attention(),
                c.qk_norm,
                h,
                q_out,
                kv_out,
                c.head_dim,
            ));
            let ffn_role = |part| TensorRole::Ffn { layer, part };
            for (name, role, shape) in [
                (at(&n.ffn_norm), TensorRole::Norm { layer }, vec![h]),
                (at(&n.gate), ffn_role(FfnPart::Gate), vec![ffn, h]),
                (at(&n.up), ffn_role(FfnPart::Up), vec![ffn, h]),
                (at(&n.down), ffn_role(FfnPart::Down), vec![h, ffn]),
            ] {
                t.push(TensorSpec { name, role, shape });
            }
        }
        t.push(TensorSpec {
            name: n.final_norm.clone(),
            role: TensorRole::FinalNorm,
            shape: vec![h],
        });
        if !c.tie_embeddings {
            t.push(TensorSpec {
                name: n.output.clone(),
                role: TensorRole::OutputHead,
                shape: vec![c.vocab, h],
            });
        }
        Ok(TensorSchema { tensors: t })
    }
}
