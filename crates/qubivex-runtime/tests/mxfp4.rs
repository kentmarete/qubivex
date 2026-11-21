//! MXFP4 end to end, on a model made at test time from the GGUF fixture: every
//! quantized matrix (embedding, attention, routers, stacked experts, head) re-encoded
//! as MXFP4 by gguf-py's quantizer, ported below.
//!
//! - On the CPU it gives, bit for bit, the logits and tokens of the same model with
//!   those matrices stored as f32 holding each MXFP4 weight's exact value: the packed
//!   kernels compute what the dequantised weights would, without ever storing them.
//! - Metal (dense matmuls, whole layers, routed tokens), Vulkan and the simulated
//!   discrete device give the CPU's bits. A GPU backend that cannot open is skipped.

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use std::path::{Path, PathBuf};

use qubivex_arch::Logits;
use qubivex_backend_cpu::formats::{MXFP4_VALUES, dequant_row, e8m0_to_f32_half};
use qubivex_core::{DType, GgmlType, Shape};
use qubivex_model::write::{TensorData, gguf};
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{BackendKind, Engine, EngineOptions, MetalOps, Sampling};

const MXFP4: DType = DType::Ggml(GgmlType::MXFP4);
const PROMPT: &[u32] = &[
    72, 101, 108, 108, 111, 44, 32, 101, 120, 112, 101, 114, 116, 115,
];

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/models/olmoe-tiny-gguf")
}

/// gguf-py `MXFP4.quantize_blocks` for one block of 32: the scale exponent
/// `floor(log2(amax)) − 2 + 127` (0 for an all-zero block), then per value the first
/// nibble whose doubled E2M1 value times the halved scale is nearest.
fn quantize_block(v: &[f32]) -> [u8; 17] {
    let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    let e = if amax > 0.0 {
        (f64::from(amax).log2().floor() - 2.0 + 127.0) as u8
    } else {
        0
    };
    let d = e8m0_to_f32_half(e);
    let nib = |x: f32| {
        (0..16u8)
            .min_by(|&a, &b| {
                let da = (d * f32::from(MXFP4_VALUES[a as usize]) - x).abs();
                let db = (d * f32::from(MXFP4_VALUES[b as usize]) - x).abs();
                da.total_cmp(&db)
            })
            .unwrap()
    };
    let mut out = [0u8; 17];
    out[0] = e;
    for i in 0..16 {
        out[1 + i] = nib(v[i]) | (nib(v[i + 16]) << 4);
    }
    out
}

/// Writes the fixture re-encoded into `dir`: each quantized matrix as MXFP4 (`packed`)
/// or as f32 with the MXFP4 values. Returns the MXFP4 bytes written.
fn write_model(dir: &Path, packed: bool) -> u64 {
    let src = fixture();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(src.join("manifest.json")).unwrap()).unwrap();
    let shard = manifest["weights"]["shards"][0]["path"].as_str().unwrap();
    let file = std::fs::read(src.join(shard)).unwrap();
    let header =
        qubivex_model::gguf::parse(file.as_slice(), file.len() as u64, &src.join(shard)).unwrap();
    let mut tensors: Vec<(String, DType, Vec<u64>, Vec<u8>)> = Vec::new();
    let mut mx_bytes = 0;
    for t in &header.tensors {
        let bytes = &file[t.offset as usize..(t.offset + t.bytes) as usize];
        let dims = t.shape.dims().to_vec();
        if !matches!(t.dtype, DType::Ggml(_)) {
            tensors.push((t.name.clone(), t.dtype, dims, bytes.to_vec()));
            continue;
        }
        let cols = *dims.last().unwrap() as usize;
        let row_bytes = t
            .dtype
            .storage_bytes(&Shape::new(&[1, cols as u64]).unwrap())
            .unwrap() as usize;
        let mut row = vec![0f32; cols];
        let mut mx = Vec::new();
        for r in bytes.chunks_exact(row_bytes) {
            dequant_row(t.dtype, r, &mut row);
            for b in row.chunks_exact(32) {
                mx.extend_from_slice(&quantize_block(b));
            }
        }
        mx_bytes += mx.len() as u64;
        if packed {
            tensors.push((t.name.clone(), MXFP4, dims, mx));
        } else {
            let mx_row = cols / 32 * 17;
            let mut f = Vec::with_capacity(mx.len() / 17 * 32 * 4);
            for r in mx.chunks_exact(mx_row) {
                dequant_row(MXFP4, r, &mut row);
                f.extend(row.iter().flat_map(|x| x.to_le_bytes()));
            }
            tensors.push((t.name.clone(), DType::F32, dims, f));
        }
    }
    let meta: Vec<(&str, qubivex_model::GgufValue)> = header
        .metadata
        .0
        .iter()
        .map(|(k, v)| (k.as_str(), v.clone()))
        .collect();
    let td: Vec<TensorData<'_>> = tensors
        .iter()
        .map(|(n, d, dims, b)| TensorData {
            name: n,
            dtype: *d,
            dims,
            data: b,
        })
        .collect();
    let out = gguf(&meta, &td);
    std::fs::create_dir_all(dir).unwrap();
    let name = if packed {
        "tiny-mxfp4.gguf"
    } else {
        "tiny-f32.gguf"
    };
    std::fs::write(dir.join(name), &out).unwrap();
    let mut m = manifest.clone();
    m["weights"]["shards"][0]["path"] = name.into();
    m["weights"]["shards"][0]["size"] = (out.len() as u64).into();
    m["quantization"] = if packed { "mxfp4" } else { "f32" }.into();
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec_pretty(&m).unwrap(),
    )
    .unwrap();
    mx_bytes
}

fn dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("qubivex-mxfp4-{}-{tag}", std::process::id()))
}

fn meta(dir: &Path) -> ModelMetadata {
    ModelMetadata::load(dir, &ArchitectureRegistry::with_builtins()).unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Prompt logits and 12 greedy tokens.
fn run(e: &Engine) -> (Vec<u32>, Vec<u32>) {
    let mut c = e.new_cache();
    let logits = bits(&e.forward(PROMPT, &mut c, Logits::All, None).unwrap());
    let toks = e
        .generate(PROMPT, 12, Sampling::default(), |_| true)
        .unwrap();
    (logits, toks)
}

fn opts() -> EngineOptions {
    EngineOptions {
        max_context: Some(256),
        ..EngineOptions::default()
    }
}

/// A GPU engine, or `None` when the device cannot open.
fn gpu(meta: ModelMetadata, opts: EngineOptions) -> Option<Engine> {
    match Engine::load(meta, opts) {
        Ok(e) => Some(e),
        Err(e) => {
            let s = e.to_string();
            assert!(
                s.contains("cannot open the Vulkan device") || s.contains("Metal unavailable"),
                "{s}"
            );
            eprintln!("skipped: {s}");
            None
        }
    }
}

#[test]
fn mxfp4_model_runs_end_to_end_with_the_bits_of_its_decoded_values() {
    let (pd, fd) = (dir("packed"), dir("f32"));
    let mx_bytes = write_model(&pd, true);
    write_model(&fd, false);
    let m = meta(&pd);
    let quantized = m
        .roles
        .keys()
        .filter(|n| m.tensors.get(n).is_some_and(|t| t.tensor.dtype == MXFP4))
        .count();
    assert!(quantized >= 10, "{quantized} MXFP4 tensors");

    let cpu = Engine::load(m, opts()).unwrap();
    let want = run(&cpu);
    let reference = Engine::load(meta(&fd), opts()).unwrap();
    assert_eq!(
        run(&reference),
        want,
        "the f32 model with the MXFP4 values gives other bits"
    );
    eprintln!(
        "{quantized} MXFP4 tensors ({mx_bytes} bytes): {} prompt logits and {} tokens as the f32 model",
        want.0.len(),
        want.1.len()
    );

    let mut backends = vec![
        (
            "metal dense matmuls",
            EngineOptions {
                backend: BackendKind::Metal,
                metal_ops: MetalOps {
                    attention: true,
                    router: true,
                    head: true,
                    layers: false,
                    token: false,
                },
                ..opts()
            },
        ),
        (
            "metal layers",
            EngineOptions {
                backend: BackendKind::Metal,
                metal_ops: MetalOps {
                    token: false,
                    ..MetalOps::default()
                },
                ..opts()
            },
        ),
        (
            "metal tokens",
            EngineOptions {
                backend: BackendKind::Metal,
                ..opts()
            },
        ),
        (
            "sim",
            EngineOptions {
                backend: BackendKind::Sim,
                ..opts()
            },
        ),
    ];
    backends.push((
        "vulkan",
        EngineOptions {
            backend: BackendKind::Vulkan,
            ..opts()
        },
    ));
    for (name, o) in backends {
        let Some(e) = gpu(meta(&pd), o) else { continue };
        let p = e.placement();
        assert!(p.fallback.is_none(), "{name}: fell back: {:?}", p.fallback);
        assert_eq!(run(&e), want, "{name}");
        eprintln!(
            "{name}: {} tensors on the device, the CPU's bits",
            p.tensors
        );
    }
    let _ = std::fs::remove_dir_all(&pd);
    let _ = std::fs::remove_dir_all(&fd);
}
