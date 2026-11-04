//! Manifests, the registry, `transformer_moe` and cross-validation, on tiny models
//! written in each tensor layout.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use qubivex_core::{DType, ExpertIdx, GgmlType, LayerIdx, Shape};
use qubivex_model::arch::TransformerMoe;
use qubivex_model::manifest::SemVer;
use qubivex_model::write::{self, TensorData};
use qubivex_model::{
    ArchitectureFactory, ArchitectureRegistry, ConfigSource, ExpertPart, GgufMetadata, GgufValue,
    Manifest, ModelError, ModelMetadata, TensorRole,
};
use serde_json::{Value, json};

struct Dir(PathBuf);
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn tmp(tag: &str) -> Dir {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    let d = std::env::temp_dir().join(format!(
        "qubivex-meta-{tag}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&d).unwrap();
    Dir(d)
}

fn hf_config() -> Value {
    json!({
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "intermediate_size": 32, "num_experts": 4, "num_experts_per_tok": 2, "norm_topk_prob": false,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "vocab_size": 128, "max_position_embeddings": 4096,
        "hidden_act": "silu", "attention_bias": false, "tie_word_embeddings": false
    })
}

fn manifest(format: &str, arch_cfg: Value, overrides: Value) -> Value {
    let (shards, tok, config) = if format == "gguf" {
        (
            json!([{ "path": "model.gguf" }]),
            json!({ "kind": "gguf" }),
            Value::Null,
        )
    } else {
        (
            json!([{ "path": "model.safetensors" }]),
            json!({ "kind": "huggingface", "path": "tokenizer.json" }),
            json!("config.json"),
        )
    };
    let mut weights = json!({ "format": format, "shards": shards });
    if !config.is_null() {
        weights["config"] = config;
    }
    json!({
        "schema_version": "1.0.0",
        "model": { "id": "toy-moe", "name": "Toy MoE" },
        "architecture": { "id": "transformer_moe", "version": "1.0.0", "config": arch_cfg },
        "weights": weights,
        "tokenizer": tok,
        "overrides": overrides
    })
}

/// The tensors the factory expects for this manifest and config, by name and shape.
fn expected(m: &Value, hf: Option<&Value>, gguf: Option<&GgufMetadata>) -> Vec<(String, Vec<u64>)> {
    let man = Manifest::parse(&m.to_string(), Path::new("manifest.json")).unwrap();
    let f = TransformerMoe;
    let cfg = f
        .parse_config(&ConfigSource {
            manifest: &man,
            hf_config: hf,
            gguf,
        })
        .unwrap();
    f.tensor_schema(&cfg)
        .unwrap()
        .tensors
        .into_iter()
        .map(|t| (t.name, t.shape))
        .collect()
}

fn dtype_for(dims: &[u64], gguf: bool) -> DType {
    match (gguf, dims.len()) {
        (true, 1) => DType::F32,
        (true, _) => DType::Ggml(GgmlType::Q8_0),
        (false, _) => DType::BF16,
    }
}

/// Writes a SafeTensors model; `edit` may change the tensor list first.
fn write_st(dir: &Path, m: &Value, cfg: &Value, edit: impl FnOnce(&mut Vec<(String, Vec<u64>)>)) {
    let mut t = expected(m, Some(cfg), None);
    edit(&mut t);
    let data: Vec<Vec<u8>> = t
        .iter()
        .map(|(_, d)| {
            vec![
                0u8;
                usize::try_from(DType::BF16.storage_bytes(&Shape::new(d).unwrap()).unwrap())
                    .unwrap()
            ]
        })
        .collect();
    let td: Vec<TensorData<'_>> = t
        .iter()
        .zip(&data)
        .map(|((n, d), b)| TensorData {
            name: n,
            dtype: dtype_for(d, false),
            dims: d,
            data: b,
        })
        .collect();
    std::fs::write(
        dir.join("model.safetensors"),
        write::safetensors(&td, &BTreeMap::new()),
    )
    .unwrap();
    std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
    std::fs::write(dir.join("manifest.json"), m.to_string()).unwrap();
}

fn gguf_meta() -> Vec<(&'static str, GgufValue)> {
    vec![
        ("general.architecture", GgufValue::String("toyarch".into())),
        ("toyarch.embedding_length", GgufValue::U32(64)),
        ("toyarch.block_count", GgufValue::U32(2)),
        ("toyarch.attention.head_count", GgufValue::U32(4)),
        ("toyarch.attention.head_count_kv", GgufValue::U32(2)),
        ("toyarch.expert_feed_forward_length", GgufValue::U32(32)),
        ("toyarch.expert_count", GgufValue::U32(4)),
        ("toyarch.expert_used_count", GgufValue::U32(2)),
        (
            "toyarch.attention.layer_norm_rms_epsilon",
            GgufValue::F32(1e-6),
        ),
        ("toyarch.rope.freq_base", GgufValue::F32(1e6)),
        ("toyarch.context_length", GgufValue::U32(4096)),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(
                (0..128)
                    .map(|i| GgufValue::String(format!("t{i}")))
                    .collect(),
            ),
        ),
    ]
}

fn write_gguf(dir: &Path, m: &Value) {
    let meta = gguf_meta();
    let gm = GgufMetadata(
        meta.iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect(),
    );
    let t = expected(m, None, Some(&gm));
    let data: Vec<Vec<u8>> = t
        .iter()
        .map(|(_, d)| {
            let dt = dtype_for(d, true);
            vec![0u8; usize::try_from(dt.storage_bytes(&Shape::new(d).unwrap()).unwrap()).unwrap()]
        })
        .collect();
    let td: Vec<TensorData<'_>> = t
        .iter()
        .zip(&data)
        .map(|((n, d), b)| TensorData {
            name: n,
            dtype: dtype_for(d, true),
            dims: d,
            data: b,
        })
        .collect();
    std::fs::write(dir.join("model.gguf"), write::gguf(&meta, &td)).unwrap();
    std::fs::write(dir.join("manifest.json"), m.to_string()).unwrap();
}

fn load(dir: &Path) -> Result<ModelMetadata, ModelError> {
    ModelMetadata::load(dir, &ArchitectureRegistry::with_builtins())
}

fn problems(r: Result<ModelMetadata, ModelError>) -> Vec<String> {
    match r {
        Err(ModelError::Validation { problems, .. }) => problems,
        other => panic!("expected a validation error, got {other:?}"),
    }
}

fn err(r: Result<ModelMetadata, ModelError>) -> String {
    r.expect_err("must fail").to_string()
}

#[test]
fn per_expert_layout_with_full_qk_norm() {
    let d = tmp("hf");
    let m = manifest("safetensors", json!({ "qk_norm": "full" }), json!({}));
    write_st(&d.0, &m, &hf_config(), |_| {});
    let md = load(&d.0).unwrap();
    assert_eq!(
        (
            md.config.dims.layers,
            md.config.dims.experts,
            md.config.dims.experts_per_token
        ),
        (2, 4, 2)
    );
    let r = |n: &str| md.roles.get(n).copied();
    assert_eq!(
        r("model.layers.1.mlp.experts.3.down_proj.weight"),
        Some(TensorRole::RoutedExpert {
            layer: LayerIdx(1),
            expert: ExpertIdx(3),
            part: ExpertPart::Down
        })
    );
    assert_eq!(
        md.tensors
            .get("model.layers.0.self_attn.q_norm.weight")
            .unwrap()
            .tensor
            .shape
            .dims(),
        &[64]
    );
    assert_eq!(
        md.tensors
            .get("model.layers.0.self_attn.k_norm.weight")
            .unwrap()
            .tensor
            .shape
            .dims(),
        &[32]
    );
    let b = md.bytes();
    // One expert: gate and up [32, 64] + down [64, 32], bf16.
    assert_eq!(b.largest_expert, 3 * 32 * 64 * 2);
    assert_eq!(b.routed_experts, 2 * 4 * b.largest_expert);
    assert_eq!(b.total, b.routed_experts + b.dense);
    assert_eq!(b.total, md.tensors.total_bytes());
}

#[test]
fn stacked_gate_up_layout_with_name_override_and_per_head_norm() {
    let d = tmp("fused");
    let m = manifest(
        "safetensors",
        json!({
            "qk_norm": "per_head",
            "tensor_names": { "base": "hf_fused", "router": "model.layers.{L}.mlp.router.weight" }
        }),
        json!({}),
    );
    let mut cfg = hf_config();
    cfg["head_dim"] = json!(16);
    cfg["moe_intermediate_size"] = json!(32);
    cfg["intermediate_size"] = json!(9999); // dense width, unused: the expert width wins
    write_st(&d.0, &m, &cfg, |_| {});
    let md = load(&d.0).unwrap();
    let gu = md
        .tensors
        .get("model.layers.0.mlp.experts.gate_up_proj")
        .unwrap();
    assert_eq!(gu.tensor.shape.dims(), &[4, 64, 64]);
    assert!(md.roles.contains_key("model.layers.1.mlp.router.weight"));
    assert_eq!(
        md.tensors
            .get("model.layers.0.self_attn.q_norm.weight")
            .unwrap()
            .tensor
            .shape
            .dims(),
        &[16]
    );
    assert_eq!(
        md.bytes().largest_expert,
        3 * 32 * 64 * 2,
        "stack slices sum to the same expert size"
    );
}

#[test]
fn gguf_layout_takes_dims_from_metadata() {
    let d = tmp("gguf");
    // GGUF does not record top-K renormalisation for every architecture: the manifest supplies it.
    let m = manifest(
        "gguf",
        json!({ "qk_norm": "per_head" }),
        json!({ "config": { "norm_topk_prob": true } }),
    );
    write_gguf(&d.0, &m);
    let md = load(&d.0).unwrap();
    assert_eq!(
        md.config.dims.vocab, 128,
        "vocab from the tokenizer token list"
    );
    assert_eq!(md.config.dims.max_context, Some(4096));
    assert_eq!(md.config.describe["norm_topk_prob"], json!(true));
    assert_eq!(md.config.describe["head_dim"], json!(16));
    let down = md.tensors.get("blk.1.ffn_down_exps.weight").unwrap();
    assert_eq!(
        (down.tensor.shape.dims(), down.tensor.dtype),
        (&[4u64, 64, 32][..], DType::Ggml(GgmlType::Q8_0))
    );
    // Q8_0: 34 bytes per 32 elements. One expert = 3 matrices of 2048 elements.
    assert_eq!(md.bytes().largest_expert, 3 * 2048 / 32 * 34);

    // Without the override, the missing value is named, with the fix.
    let m2 = manifest("gguf", json!({ "qk_norm": "per_head" }), json!({}));
    std::fs::write(d.0.join("manifest.json"), m2.to_string()).unwrap();
    let e = err(load(&d.0));
    assert!(
        e.contains("expert_weights_norm") && e.contains("overrides.config.norm_topk_prob"),
        "{e}"
    );
}

#[test]
fn validation_lists_every_problem() {
    let d = tmp("bad");
    let m = manifest("safetensors", json!({ "qk_norm": "full" }), json!({}));
    write_st(&d.0, &m, &hf_config(), |t| {
        t.retain(|(n, _)| n != "model.layers.0.mlp.gate.weight");
        for (n, s) in t.iter_mut() {
            if n == "model.layers.1.self_attn.k_proj.weight" {
                *s = vec![64, 64];
            }
        }
        t.push((
            "model.layers.0.self_attn.rotary_emb.inv_freq".into(),
            vec![8],
        ));
    });
    let p = problems(load(&d.0));
    assert_eq!(p.len(), 3, "{p:#?}");
    assert!(
        p.iter()
            .any(|x| x.contains("missing tensor \"model.layers.0.mlp.gate.weight\""))
    );
    assert!(
        p.iter()
            .any(|x| x.contains("has shape [64, 64], expected [32, 64]"))
    );
    assert!(
        p.iter()
            .any(|x| x.contains("unexpected tensor") && x.contains("inv_freq"))
    );

    // qk_norm omitted: the norm tensors are unexpected, which points at the setting.
    let d2 = tmp("noqk");
    write_st(&d2.0, &m, &hf_config(), |_| {});
    std::fs::write(
        d2.0.join("manifest.json"),
        manifest("safetensors", json!({}), json!({})).to_string(),
    )
    .unwrap();
    assert_eq!(problems(load(&d2.0)).len(), 4); // q_norm and k_norm in 2 layers

    // ignore_tensors accepts a deliberate extra tensor.
    let d3 = tmp("ignore");
    let m3 = manifest(
        "safetensors",
        json!({ "qk_norm": "full" }),
        json!({ "ignore_tensors": ["model.rotary.inv_freq"] }),
    );
    write_st(&d3.0, &m3, &hf_config(), |t| {
        t.push(("model.rotary.inv_freq".into(), vec![8]))
    });
    assert!(load(&d3.0).is_ok());
}

#[test]
fn config_rules() {
    let d = tmp("cfg");
    let m = manifest("safetensors", json!({ "qk_norm": "full" }), json!({}));
    write_st(&d.0, &m, &hf_config(), |_| {});
    let with_cfg = |c: Value| {
        std::fs::write(d.0.join("config.json"), c.to_string()).unwrap();
        err(load(&d.0))
    };
    let set = |k: &str, v: Value| {
        let mut c = hf_config();
        c[k] = v;
        c
    };
    assert!(with_cfg(set("attention_bias", json!(true))).contains("attention biases"));
    assert!(with_cfg(set("mlp_only_layers", json!([0]))).contains("dense FFN layers"));
    assert!(with_cfg(set("decoder_sparse_step", json!(2))).contains("dense FFN layers"));
    assert!(with_cfg(set("hidden_act", json!("gelu"))).contains("only silu"));
    assert!(with_cfg(set("num_key_value_heads", json!(3))).contains("not a multiple"));
    assert!(with_cfg(set("num_experts_per_tok", json!(5))).contains("out of range"));
    assert!(with_cfg(set("hidden_size", json!(-1))).contains("not a non-negative integer"));
    assert!(with_cfg(set("rope_scaling", json!({ "rope_type": "yarn" }))).contains("rope_scaling"));
    let mut c = hf_config();
    c.as_object_mut().unwrap().remove("rms_norm_eps");
    assert!(with_cfg(c).contains("set overrides.config.rms_norm_eps"));
    let mut c = hf_config();
    c.as_object_mut().unwrap().remove("rope_theta");
    c["rope_parameters"] = json!({ "rope_theta": 10000.0, "rope_type": "default" });
    std::fs::write(d.0.join("config.json"), c.to_string()).unwrap();
    assert!(load(&d.0).is_ok(), "rope_theta read from rope_parameters");

    // Overrides fill gaps but may not contradict the checkpoint.
    std::fs::write(d.0.join("config.json"), hf_config().to_string()).unwrap();
    let mo = manifest(
        "safetensors",
        json!({ "qk_norm": "full" }),
        json!({ "config": { "hidden_size": 128 } }),
    );
    std::fs::write(d.0.join("manifest.json"), mo.to_string()).unwrap();
    assert!(err(load(&d.0)).contains("contradicts"));
    let mo = manifest(
        "safetensors",
        json!({ "qk_norm": "full" }),
        json!({ "config": { "rms_norm_eps": 1e-5 } }),
    );
    std::fs::write(d.0.join("manifest.json"), mo.to_string()).unwrap();
    assert!(load(&d.0).is_ok(), "an agreeing override is fine");

    for (cfg, want) in [
        (json!({ "qk_norm": "sideways" }), "architecture.config"),
        (json!({ "unknown_key": 1 }), "architecture.config"),
        (
            json!({ "tensor_names": "nope" }),
            "unknown tensor_names preset",
        ),
        (
            json!({ "tensor_names": { "router": "fixed.name" } }),
            "must contain one {L}",
        ),
        (
            json!({ "tensor_names": { "expert_gate_up": "x.{L}" } }),
            "not a name of",
        ),
    ] {
        std::fs::write(
            d.0.join("manifest.json"),
            manifest("safetensors", cfg, json!({})).to_string(),
        )
        .unwrap();
        let e = err(load(&d.0));
        assert!(e.contains(want), "{e}");
    }
}

#[test]
fn manifest_rules() {
    let p = Path::new("manifest.json");
    let base = manifest("safetensors", json!({}), json!({}));
    assert!(Manifest::parse(&base.to_string(), p).is_ok());
    let bad = |edit: &dyn Fn(&mut Value), want: &str| {
        let mut m = base.clone();
        edit(&mut m);
        let e = Manifest::parse(&m.to_string(), p).unwrap_err().to_string();
        assert!(e.contains(want), "{want}: {e}");
    };
    bad(&|m| m["schema_version"] = json!("2.0.0"), "not supported");
    bad(&|m| m["schema_version"] = json!("1.0"), "major.minor.patch");
    bad(&|m| m["model"]["id"] = json!("Toy MoE"), "model.id");
    bad(
        &|m| m["weights"]["shards"][0]["path"] = json!("../../etc/passwd"),
        "relative",
    );
    bad(
        &|m| m["weights"]["shards"][0]["path"] = json!("/abs/model.safetensors"),
        "relative",
    );
    bad(&|m| m["weights"]["shards"] = json!([]), "empty");
    bad(
        &|m| m["weights"]["shards"][0]["sha256"] = json!("ABC"),
        "sha256",
    );
    bad(&|m| m["surprise"] = json!(1), "unknown field");
    bad(
        &|m| m["tokenizer"] = json!({ "kind": "huggingface", "path": "../t.json" }),
        "relative",
    );
    let g = manifest("gguf", json!({}), json!({}));
    let mut two = g.clone();
    two["weights"]["shards"] = json!([{ "path": "a.gguf" }, { "path": "b.gguf" }]);
    assert!(
        Manifest::parse(&two.to_string(), p)
            .unwrap_err()
            .to_string()
            .contains("exactly one shard")
    );

    // A shard size that disagrees with the file.
    let d = tmp("size");
    let mut m = manifest("safetensors", json!({ "qk_norm": "full" }), json!({}));
    write_st(&d.0, &m, &hf_config(), |_| {});
    m["weights"]["shards"][0]["size"] = json!(1);
    std::fs::write(d.0.join("manifest.json"), m.to_string()).unwrap();
    assert!(err(load(&d.0)).contains("the manifest says 1"));
}

#[test]
fn registry_rules() {
    let mut r = ArchitectureRegistry::with_builtins();
    assert_eq!(r.list_supported().len(), 2);
    assert!(
        r.register(Arc::new(TransformerMoe))
            .unwrap_err()
            .to_string()
            .contains("already registered")
    );
    let m = |id: &str, v: &str| {
        let mut j = manifest("safetensors", json!({}), json!({}));
        j["architecture"]["id"] = json!(id);
        j["architecture"]["version"] = json!(v);
        Manifest::parse(&j.to_string(), Path::new("m"))
            .unwrap()
            .architecture
    };
    assert!(r.resolve(&m("transformer_moe", "1.0.7")).is_ok());
    assert!(
        r.resolve(&m("transformer_moe", "1.1.0"))
            .unwrap_err()
            .to_string()
            .contains("implements 1.0.0")
    );
    assert!(r.resolve(&m("transformer_moe", "2.0.0")).is_err());
    assert!(
        r.resolve(&m("nonesuch", "1.0.0"))
            .unwrap_err()
            .to_string()
            .contains("registered: transformer_dense, transformer_moe")
    );
    assert!(r.unregister("transformer_moe").is_some());
    assert!(r.resolve(&m("transformer_moe", "1.0.0")).is_err());
    assert_eq!(
        TransformerMoe.version(),
        SemVer {
            major: 1,
            minor: 0,
            patch: 0
        }
    );
}

/// The published JSON Schema names the same fields the parser accepts.
#[test]
fn json_schema_matches_the_manifest_type() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/model-support/manifest.schema.json");
    let s: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let keys = |v: &Value| -> Vec<String> {
        let mut k: Vec<String> = v["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        k.sort();
        k
    };
    let mut m = manifest(
        "safetensors",
        json!({}),
        json!({ "config": { "a": 1 }, "ignore_tensors": ["x"] }),
    );
    m["context_length"] = json!(2048);
    m["capabilities"] = json!(["chat"]);
    m["quantization"] = json!("bf16");
    m["model"]["version"] = json!("1");
    m["weights"]["shards"][0]["size"] = json!(1);
    m["weights"]["shards"][0]["sha256"] = json!("0".repeat(64));
    // Round-trip through the type, so only fields it knows survive.
    let parsed = Manifest::parse(&m.to_string(), Path::new("m")).unwrap();
    let back = serde_json::to_value(&parsed).unwrap();
    let mut top: Vec<String> = back.as_object().unwrap().keys().cloned().collect();
    top.sort();
    assert_eq!(keys(&s), top);
    for sec in ["model", "architecture", "weights", "overrides"] {
        let mut k: Vec<String> = back[sec].as_object().unwrap().keys().cloned().collect();
        k.sort();
        assert_eq!(keys(&s["properties"][sec]), k, "{sec}");
    }
    let mut k: Vec<String> = back["weights"]["shards"][0]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    k.sort();
    assert_eq!(
        keys(&s["properties"]["weights"]["properties"]["shards"]["items"]),
        k
    );
}
