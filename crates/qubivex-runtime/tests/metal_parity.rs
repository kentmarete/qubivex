//! Metal against CPU on the real model (env-gated): with activations in the range a
//! forward pass produces, every weight matrix's output is bit-identical; and the engine
//! with every dense matmul on Metal gives the CPU engine's logits and tokens.
//!
//! Set QUBIVEX_OLMOE_GGUF_DIR to the imported model directory. Skipped without it or
//! without a Metal device.

#![allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use qubivex_backend_cpu::CpuBackend;
use qubivex_backend_metal::MetalBackend;
use qubivex_compute::{ComputeBackend, WeightView};
use qubivex_model::{ArchitectureRegistry, ModelMetadata};

/// xorshift64*, then a sum of four uniforms (roughly normal, like normed activations).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn normalish(&mut self) -> f32 {
        (0..4)
            .map(|_| (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
            .sum::<f32>()
            * 1.7
    }
}

#[test]
fn every_real_weight_matrix_is_bit_identical() {
    let Ok(dir) = std::env::var("QUBIVEX_OLMOE_GGUF_DIR") else {
        eprintln!("skipping: QUBIVEX_OLMOE_GGUF_DIR not set");
        return;
    };
    let gpu = match MetalBackend::new() {
        Ok(g) => g,
        Err(qubivex_backend_metal::MetalError::Unavailable(e)) => {
            eprintln!("skipping: {e}");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    let cpu = CpuBackend::new();
    let meta =
        ModelMetadata::load(Path::new(&dir), &ArchitectureRegistry::with_builtins()).unwrap();
    let mut files: Vec<std::fs::File> = meta
        .tensors
        .files()
        .iter()
        .map(|f| std::fs::File::open(&f.path).unwrap())
        .collect();
    let mut rng = Rng(3);
    let (mut matrices, mut outputs, mut bytes) = (0usize, 0usize, 0u64);
    for name in meta.roles.keys() {
        let e = meta.tensors.get(name).unwrap();
        let t = &e.tensor;
        if t.shape.rank() < 2 || !gpu.supports(t.dtype) {
            continue;
        }
        let cols = t.shape.innermost().unwrap() as usize;
        let rows = t.shape.rows().unwrap() as usize;
        let mut data = vec![0u8; t.bytes as usize];
        let f = &mut files[e.file];
        f.seek(SeekFrom::Start(t.offset)).unwrap();
        f.read_exact(&mut data).unwrap();
        let w = WeightView::new(t.dtype, rows, cols, &data).unwrap();
        let s = if matrices % 4 == 0 { 4 } else { 1 };
        let xs: Vec<f32> = (0..s * cols).map(|_| rng.normalish()).collect();
        let (mut want, mut got) = (vec![0f32; s * rows], vec![0f32; s * rows]);
        cpu.matmul(&w, &xs, s, &mut want).unwrap();
        gpu.matmul(&w, &xs, s, &mut got).unwrap();
        for i in 0..want.len() {
            assert_eq!(
                want[i].to_bits(),
                got[i].to_bits(),
                "{name} ({} {rows}x{cols}) s={s} output {i}: cpu {} metal {}",
                t.dtype,
                want[i],
                got[i]
            );
        }
        matrices += 1;
        outputs += want.len();
        bytes += t.bytes;
    }
    eprintln!(
        "{matrices} tensors, {outputs} outputs, {:.2} GiB of weights: bit-identical",
        bytes as f64 / (1u64 << 30) as f64
    );
    assert!(matrices > 0);
}

/// End-to-end token parity (Phase 8 slice 2): the engine with attention, routers and
/// the head on Metal gives the CPU engine's prompt logits bit for bit, and the same
/// generated tokens.
#[test]
fn engine_on_metal_gives_the_cpu_tokens() {
    use qubivex_runtime::{BackendKind, Engine, EngineOptions, Logits, MetalOps, Sampling};
    let Ok(dir) = std::env::var("QUBIVEX_OLMOE_GGUF_DIR") else {
        eprintln!("skipping: QUBIVEX_OLMOE_GGUF_DIR not set");
        return;
    };
    match MetalBackend::new() {
        Err(qubivex_backend_metal::MetalError::Unavailable(e)) => {
            eprintln!("skipping: {e}");
            return;
        }
        Err(e) => panic!("{e}"),
        Ok(_) => {}
    }
    let meta =
        || ModelMetadata::load(Path::new(&dir), &ArchitectureRegistry::with_builtins()).unwrap();
    let opts = EngineOptions {
        max_context: Some(512),
        ..EngineOptions::default()
    };
    let cpu = Engine::load(meta(), opts).unwrap();
    let gpu = Engine::load(
        meta(),
        EngineOptions {
            backend: BackendKind::Metal,
            metal_ops: MetalOps {
                attention: true,
                router: true,
                head: true,
            },
            ..opts
        },
    )
    .unwrap();
    let p = gpu.placement();
    eprintln!(
        "{:?}: {} tensors, {} bytes on the device",
        p.device, p.tensors, p.bytes
    );
    assert_eq!(
        p.tensors,
        16 * 5 + 1,
        "q, k, v, o and router per layer, and the head"
    );
    let prompt = cpu
        .tokenizer()
        .encode("Write a long story about a lighthouse keeper.", true)
        .unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let (mut a, mut b) = (cpu.new_cache(), gpu.new_cache());
    let la = cpu.forward(&prompt, &mut a, Logits::All, None).unwrap();
    let lb = gpu.forward(&prompt, &mut b, Logits::All, None).unwrap();
    assert!(bits(&la) == bits(&lb), "prompt logits differ");
    let ta = cpu
        .generate(&prompt, 48, Sampling::default(), |_| true)
        .unwrap();
    let tb = gpu
        .generate(&prompt, 48, Sampling::default(), |_| true)
        .unwrap();
    assert_eq!(ta, tb);
    eprintln!(
        "{} prompt tokens, {} generated tokens identical",
        prompt.len(),
        ta.len()
    );
}
