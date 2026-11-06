//! Tiny model fixtures under `fixtures/models/`, generated deterministically.
//!
//! The configs mirror the key sets of the released checkpoints they stand in for
//! (OLMoE-1B-7B-0125 and Qwen3-30B-A3B), shrunk to a few hundred KB. Weights are
//! fixed pseudo-random values, so `cargo xtask check` can regenerate the fixtures in
//! memory and fail if the committed files drift. Manifests come from
//! `qubivex_model::import`, which also tests the importer.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use qubivex_core::{DType, GgmlType};
use qubivex_model::manifest::SemVer;
use qubivex_model::write::{self, TensorData};
use qubivex_model::{ArchitectureRegistry, GgufValue, ImportOptions, import};
use serde_json::{Value, json};

/// Root of the fixtures, relative to the workspace.
pub(crate) const DIR: &str = "fixtures/models";

/// xorshift64*.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Roughly N(0, std): a sum of four uniforms.
    #[allow(clippy::cast_precision_loss)]
    fn normal(&mut self, std: f32) -> f32 {
        let u = |r: &mut Self| (r.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        (u(self) + u(self) + u(self) + u(self)) * std * 1.732
    }
}

#[allow(clippy::cast_possible_truncation)] // the high half of a u32
fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    let r = b + 0x7fff + ((b >> 16) & 1);
    (r >> 16) as u16
}

/// f32 → f16 bits, round to nearest even (the fixture values are all in normal range).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp <= 0 {
        return sign; // fixtures never reach f16 subnormals
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    let mut h = u32::from(sign) | ((exp as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    h as u16
}

/// Q8_0: per 32 values, an f16 scale then 32 int8.
fn q8_0(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() / 32 * 34);
    for blk in v.chunks_exact(32) {
        let amax = blk.iter().fold(0f32, |a, x| a.max(x.abs()));
        let d = amax / 127.0;
        out.extend_from_slice(&f16(d).to_le_bytes());
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        #[allow(clippy::cast_possible_truncation)]
        out.extend(
            blk.iter()
                .map(|x| ((x * inv).round().clamp(-127.0, 127.0) as i8) as u8),
        );
    }
    out
}

struct T {
    name: String,
    dims: Vec<u64>,
    values: Vec<f32>,
}

fn tensor(rng: &mut Rng, name: String, dims: Vec<u64>) -> T {
    let n: u64 = dims.iter().product();
    let norm = dims.len() == 1;
    let values = (0..n)
        .map(|_| if norm { 1.0 } else { rng.normal(0.02) })
        .collect();
    T { name, dims, values }
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&x| bf16(x).to_le_bytes()).collect()
}

/// GPT-2's byte-to-symbol table: the symbol of byte `b` is at index `b`. A byte-level
/// BPE with only these 256 tokens and no merges encodes any text, one token per byte.
fn byte_symbols() -> Vec<String> {
    let printable = |b: u32| {
        (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b)
    };
    let mut next = 256u32;
    (0u32..256)
        .map(|b| {
            let c = if printable(b) {
                b
            } else {
                next += 1;
                next - 1
            };
            char::from_u32(c).map(String::from).unwrap_or_default()
        })
        .collect()
}

/// The fixtures' chat template (rendered with transformers' trim/lstrip settings).
const CHAT_TEMPLATE: &str = "{% for m in messages %}<|{{ m.role }}|>\n{{ m.content }}\n{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}";

/// End-of-sequence: byte 0x01 (never produced by ordinary text).
const EOS_ID: u32 = 1;

/// A byte-level BPE tokenizer.json over [`byte_symbols`], shared by every fixture so
/// the SafeTensors and GGUF forms of a model tokenize identically.
fn tokenizer(vocab: u64) -> Value {
    assert_eq!(vocab, 256, "the fixture tokenizer has one token per byte");
    let v: serde_json::Map<String, Value> = byte_symbols()
        .into_iter()
        .enumerate()
        .map(|(i, s)| (s, json!(i)))
        .collect();
    let byte_level = json!({ "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true });
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": byte_level,
        "post_processor": null,
        "decoder": byte_level,
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
            "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
            "vocab": v, "merges": []
        }
    })
}

fn tokenizer_config() -> Value {
    json!({
        "add_bos_token": false,
        "add_eos_token": false,
        "bos_token": null,
        "eos_token": byte_symbols()[EOS_ID as usize],
        "chat_template": CHAT_TEMPLATE
    })
}

/// Every fixture: directory name → files (name → bytes), manifests excluded.
fn weight_files() -> BTreeMap<&'static str, BTreeMap<String, Vec<u8>>> {
    let mut all = BTreeMap::new();
    let (h, ffn, experts, vocab, layers) = (64u64, 32u64, 8u64, 256u64, 2u32);

    // OLMoE-style: per-expert tensors (as released), full q/k norm, kv heads = heads.
    {
        let mut rng = Rng(0x0123_4567_89ab_cdef);
        let cfg = json!({
            "architectures": ["OlmoeForCausalLM"], "model_type": "olmoe",
            "hidden_size": h, "intermediate_size": ffn, "num_hidden_layers": layers,
            "num_attention_heads": 4, "num_key_value_heads": 4, "num_experts": experts,
            "num_experts_per_tok": 2, "norm_topk_prob": false, "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0, "rope_scaling": null, "clip_qkv": null, "hidden_act": "silu",
            "attention_bias": false, "attention_dropout": 0.0, "max_position_embeddings": 4096,
            "vocab_size": vocab, "tie_word_embeddings": false, "torch_dtype": "bfloat16",
            "bos_token_id": null, "eos_token_id": 1, "pad_token_id": 0
        });
        let mut t = vec![tensor(
            &mut rng,
            "model.embed_tokens.weight".into(),
            vec![vocab, h],
        )];
        for l in 0..layers {
            let p = |s: &str| format!("model.layers.{l}.{s}");
            for (n, d) in [
                ("input_layernorm.weight", vec![h]),
                ("self_attn.q_proj.weight", vec![h, h]),
                ("self_attn.k_proj.weight", vec![h, h]),
                ("self_attn.v_proj.weight", vec![h, h]),
                ("self_attn.o_proj.weight", vec![h, h]),
                ("self_attn.q_norm.weight", vec![h]),
                ("self_attn.k_norm.weight", vec![h]),
                ("post_attention_layernorm.weight", vec![h]),
                ("mlp.gate.weight", vec![experts, h]),
            ] {
                t.push(tensor(&mut rng, p(n), d));
            }
            for e in 0..experts {
                t.push(tensor(
                    &mut rng,
                    p(&format!("mlp.experts.{e}.gate_proj.weight")),
                    vec![ffn, h],
                ));
                t.push(tensor(
                    &mut rng,
                    p(&format!("mlp.experts.{e}.up_proj.weight")),
                    vec![ffn, h],
                ));
                t.push(tensor(
                    &mut rng,
                    p(&format!("mlp.experts.{e}.down_proj.weight")),
                    vec![h, ffn],
                ));
            }
        }
        t.push(tensor(&mut rng, "model.norm.weight".into(), vec![h]));
        t.push(tensor(&mut rng, "lm_head.weight".into(), vec![vocab, h]));
        all.insert("olmoe-tiny", safetensors_files(&t, &cfg, vocab));

        // The same weights as GGUF: Q8_0 matrices, F32 vectors, stacked experts.
        let get = |n: &str| {
            t.iter()
                .find(|x| x.name == n)
                .map(|x| x.values.clone())
                .unwrap_or_default()
        };
        let mut g: Vec<(String, Vec<u64>, Vec<f32>)> = vec![(
            "token_embd.weight".into(),
            vec![vocab, h],
            get("model.embed_tokens.weight"),
        )];
        for l in 0..layers {
            let hf = |s: &str| get(&format!("model.layers.{l}.{s}"));
            let b = |s: &str| format!("blk.{l}.{s}");
            g.push((b("attn_norm.weight"), vec![h], hf("input_layernorm.weight")));
            g.push((
                b("attn_q.weight"),
                vec![h, h],
                hf("self_attn.q_proj.weight"),
            ));
            g.push((
                b("attn_k.weight"),
                vec![h, h],
                hf("self_attn.k_proj.weight"),
            ));
            g.push((
                b("attn_v.weight"),
                vec![h, h],
                hf("self_attn.v_proj.weight"),
            ));
            g.push((
                b("attn_output.weight"),
                vec![h, h],
                hf("self_attn.o_proj.weight"),
            ));
            g.push((
                b("attn_q_norm.weight"),
                vec![h],
                hf("self_attn.q_norm.weight"),
            ));
            g.push((
                b("attn_k_norm.weight"),
                vec![h],
                hf("self_attn.k_norm.weight"),
            ));
            g.push((
                b("ffn_norm.weight"),
                vec![h],
                hf("post_attention_layernorm.weight"),
            ));
            g.push((
                b("ffn_gate_inp.weight"),
                vec![experts, h],
                hf("mlp.gate.weight"),
            ));
            for (part, gname, d) in [
                ("gate_proj", "ffn_gate_exps.weight", vec![experts, ffn, h]),
                ("up_proj", "ffn_up_exps.weight", vec![experts, ffn, h]),
                ("down_proj", "ffn_down_exps.weight", vec![experts, h, ffn]),
            ] {
                let v: Vec<f32> = (0..experts)
                    .flat_map(|e| hf(&format!("mlp.experts.{e}.{part}.weight")))
                    .collect();
                g.push((b(gname), d, v));
            }
        }
        g.push((
            "output_norm.weight".into(),
            vec![h],
            get("model.norm.weight"),
        ));
        g.push((
            "output.weight".into(),
            vec![vocab, h],
            get("lm_head.weight"),
        ));
        let data: Vec<(DType, Vec<u8>)> = g
            .iter()
            .map(|(_, d, v)| {
                if d.len() == 1 {
                    (DType::F32, v.iter().flat_map(|x| x.to_le_bytes()).collect())
                } else {
                    (DType::Ggml(GgmlType::Q8_0), q8_0(v))
                }
            })
            .collect();
        let td: Vec<TensorData<'_>> = g
            .iter()
            .zip(&data)
            .map(|((n, d, _), (dt, b))| TensorData {
                name: n,
                dtype: *dt,
                dims: d,
                data: b,
            })
            .collect();
        let tokens = GgufValue::Array(byte_symbols().into_iter().map(GgufValue::String).collect());
        let token_types = GgufValue::Array((0..vocab).map(|_| GgufValue::I32(1)).collect());
        let meta = [
            ("general.architecture", GgufValue::String("olmoe".into())),
            (
                "general.name",
                GgufValue::String("olmoe tiny fixture".into()),
            ),
            ("olmoe.context_length", GgufValue::U32(4096)),
            ("olmoe.embedding_length", GgufValue::U32(64)),
            ("olmoe.feed_forward_length", GgufValue::U32(32)),
            ("olmoe.block_count", GgufValue::U32(layers)),
            ("olmoe.attention.head_count", GgufValue::U32(4)),
            ("olmoe.attention.head_count_kv", GgufValue::U32(4)),
            (
                "olmoe.attention.layer_norm_rms_epsilon",
                GgufValue::F32(1e-5),
            ),
            ("olmoe.rope.freq_base", GgufValue::F32(10000.0)),
            ("olmoe.expert_count", GgufValue::U32(8)),
            ("olmoe.expert_used_count", GgufValue::U32(2)),
            ("tokenizer.ggml.model", GgufValue::String("gpt2".into())),
            ("tokenizer.ggml.pre", GgufValue::String("gpt-2".into())),
            ("tokenizer.ggml.tokens", tokens),
            ("tokenizer.ggml.token_type", token_types),
            ("tokenizer.ggml.merges", GgufValue::Array(Vec::new())),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(EOS_ID)),
            (
                "tokenizer.chat_template",
                GgufValue::String(CHAT_TEMPLATE.into()),
            ),
        ];
        let mut files = BTreeMap::new();
        files.insert("olmoe-tiny-q8_0.gguf".to_owned(), write::gguf(&meta, &td));
        all.insert("olmoe-tiny-gguf", files);
    }

    // Qwen3-MoE-style: stacked gate-up experts (as newer transformers saves), per-head
    // q/k norm, head_dim != hidden / heads.
    {
        let mut rng = Rng(0xfedc_ba98_7654_3210);
        let hd = 32u64;
        let (heads, kv) = (4u64, 2u64);
        let cfg = json!({
            "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
            "hidden_size": h, "intermediate_size": 128, "moe_intermediate_size": ffn,
            "num_hidden_layers": layers, "num_attention_heads": heads, "num_key_value_heads": kv,
            "head_dim": hd, "num_experts": experts, "num_experts_per_tok": 2, "norm_topk_prob": true,
            "decoder_sparse_step": 1, "mlp_only_layers": [], "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0, "rope_scaling": null, "hidden_act": "silu",
            "attention_bias": false, "attention_dropout": 0.0, "max_position_embeddings": 40960,
            "sliding_window": null, "use_sliding_window": false, "vocab_size": vocab,
            "tie_word_embeddings": false, "torch_dtype": "bfloat16", "bos_token_id": 0, "eos_token_id": 1
        });
        let mut t = vec![tensor(
            &mut rng,
            "model.embed_tokens.weight".into(),
            vec![vocab, h],
        )];
        for l in 0..layers {
            let p = |s: &str| format!("model.layers.{l}.{s}");
            for (n, d) in [
                ("input_layernorm.weight", vec![h]),
                ("self_attn.q_proj.weight", vec![heads * hd, h]),
                ("self_attn.k_proj.weight", vec![kv * hd, h]),
                ("self_attn.v_proj.weight", vec![kv * hd, h]),
                ("self_attn.o_proj.weight", vec![h, heads * hd]),
                ("self_attn.q_norm.weight", vec![hd]),
                ("self_attn.k_norm.weight", vec![hd]),
                ("post_attention_layernorm.weight", vec![h]),
                ("mlp.gate.weight", vec![experts, h]),
                ("mlp.experts.gate_up_proj", vec![experts, 2 * ffn, h]),
                ("mlp.experts.down_proj", vec![experts, h, ffn]),
            ] {
                t.push(tensor(&mut rng, p(n), d));
            }
        }
        t.push(tensor(&mut rng, "model.norm.weight".into(), vec![h]));
        t.push(tensor(&mut rng, "lm_head.weight".into(), vec![vocab, h]));
        all.insert("qwen3-moe-tiny", safetensors_files(&t, &cfg, vocab));
    }

    // Dense (transformer_dense), Llama-style: no q/k norm, grouped-query attention,
    // the output head tied to the embedding.
    {
        let mut rng = Rng(0x0f1e_2d3c_4b5a_6978);
        let hd = 16u64;
        let (heads, kv, dffn) = (4u64, 2u64, 96u64);
        let cfg = json!({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": h, "intermediate_size": dffn, "num_hidden_layers": layers,
            "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": hd,
            "rms_norm_eps": 1e-5, "rope_theta": 500000.0, "rope_scaling": null,
            "hidden_act": "silu", "attention_bias": false, "mlp_bias": false,
            "attention_dropout": 0.0, "max_position_embeddings": 2048, "vocab_size": vocab,
            "tie_word_embeddings": true, "torch_dtype": "bfloat16", "bos_token_id": 0,
            "eos_token_id": 1
        });
        let mut t = vec![tensor(
            &mut rng,
            "model.embed_tokens.weight".into(),
            vec![vocab, h],
        )];
        for l in 0..layers {
            let p = |s: &str| format!("model.layers.{l}.{s}");
            for (n, d) in [
                ("input_layernorm.weight", vec![h]),
                ("self_attn.q_proj.weight", vec![heads * hd, h]),
                ("self_attn.k_proj.weight", vec![kv * hd, h]),
                ("self_attn.v_proj.weight", vec![kv * hd, h]),
                ("self_attn.o_proj.weight", vec![h, heads * hd]),
                ("post_attention_layernorm.weight", vec![h]),
                ("mlp.gate_proj.weight", vec![dffn, h]),
                ("mlp.up_proj.weight", vec![dffn, h]),
                ("mlp.down_proj.weight", vec![h, dffn]),
            ] {
                t.push(tensor(&mut rng, p(n), d));
            }
        }
        t.push(tensor(&mut rng, "model.norm.weight".into(), vec![h]));
        all.insert("dense-tiny", safetensors_files(&t, &cfg, vocab));
    }
    all
}

fn safetensors_files(t: &[T], cfg: &Value, vocab: u64) -> BTreeMap<String, Vec<u8>> {
    let data: Vec<Vec<u8>> = t.iter().map(|x| bf16_bytes(&x.values)).collect();
    let td: Vec<TensorData<'_>> = t
        .iter()
        .zip(&data)
        .map(|(x, b)| TensorData {
            name: &x.name,
            dtype: DType::BF16,
            dims: &x.dims,
            data: b,
        })
        .collect();
    let mut meta = BTreeMap::new();
    meta.insert("format".to_owned(), "pt".to_owned());
    let mut files = BTreeMap::new();
    files.insert(
        "model.safetensors".to_owned(),
        write::safetensors(&td, &meta),
    );
    files.insert("config.json".to_owned(), pretty(cfg));
    files.insert("tokenizer.json".to_owned(), pretty(&tokenizer(vocab)));
    files.insert(
        "tokenizer_config.json".to_owned(),
        pretty(&tokenizer_config()),
    );
    files
}

fn pretty(v: &Value) -> Vec<u8> {
    let mut s = serde_json::to_string_pretty(v)
        .unwrap_or_default()
        .into_bytes();
    s.push(b'\n');
    s
}

/// The schema version written to the fixtures' manifests.
const OLDEST_SCHEMA: SemVer = SemVer {
    major: 1,
    minor: 0,
    patch: 0,
};

/// Values the GGUF fixture's metadata does not record.
fn import_options(dir: &str) -> ImportOptions {
    let mut o = ImportOptions {
        id: Some(dir.to_owned()),
        ..ImportOptions::default()
    };
    if dir == "olmoe-tiny-gguf" {
        o.set.insert("norm_topk_prob".into(), json!(false));
    }
    o
}

/// Writes every fixture under `out` and returns the files written.
fn generate(out: &Path) -> Result<Vec<PathBuf>, String> {
    let reg = ArchitectureRegistry::with_builtins();
    let mut written = Vec::new();
    for (dir, files) in weight_files() {
        let d = out.join(dir);
        std::fs::create_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
        for (name, bytes) in files {
            let p = d.join(&name);
            std::fs::write(&p, bytes).map_err(|e| format!("{}: {e}", p.display()))?;
            written.push(p);
        }
        let imp =
            import(&d, &import_options(dir), &reg).map_err(|e| format!("import {dir}: {e}"))?;
        let mut m = imp.manifest;
        // Fixtures stay at the oldest schema a build reads: they are the migration
        // tests' input, and the proof that such a manifest still loads.
        m.schema_version = OLDEST_SCHEMA;
        m.model.name = match dir {
            "olmoe-tiny" => "Tiny OLMoE-style fixture",
            "olmoe-tiny-gguf" => "Tiny OLMoE-style fixture, GGUF Q8_0",
            "qwen3-moe-tiny" => "Tiny Qwen3-MoE-style fixture",
            _ => dir,
        }
        .to_owned();
        let p = d.join(qubivex_model::MANIFEST_FILE);
        std::fs::write(
            &p,
            pretty(&serde_json::to_value(&m).map_err(|e| e.to_string())?),
        )
        .map_err(|e| format!("{}: {e}", p.display()))?;
        written.push(p);
    }
    Ok(written)
}

/// `cargo xtask gen-fixtures`.
pub(crate) fn write_fixtures(root: &Path) -> Result<(), String> {
    let files = generate(&root.join(DIR))?;
    println!("gen-fixtures: wrote {} files under {DIR}", files.len());
    Ok(())
}

/// `cargo xtask check-fixtures`: regenerate into a temporary directory and compare.
pub(crate) fn check_fixtures(root: &Path) -> Result<(), String> {
    let tmp = std::env::temp_dir().join(format!("qubivex-fixtures-{}", std::process::id()));
    let files = generate(&tmp);
    let result = files.and_then(|files| {
        let mut diff = Vec::new();
        for f in &files {
            let rel = f.strip_prefix(&tmp).map_err(|e| e.to_string())?;
            let committed = root.join(DIR).join(rel);
            let want = std::fs::read(f).map_err(|e| e.to_string())?;
            if std::fs::read(&committed).ok().as_deref() != Some(want.as_slice()) {
                diff.push(format!("{DIR}/{}", rel.display()));
            }
        }
        if diff.is_empty() {
            println!("check-fixtures: ok ({} files)", files.len());
            Ok(())
        } else {
            Err(format!(
                "check-fixtures: these differ from the generator (run `cargo xtask gen-fixtures`):\n  {}",
                diff.join("\n  ")
            ))
        }
    });
    let _ = std::fs::remove_dir_all(&tmp);
    result
}
