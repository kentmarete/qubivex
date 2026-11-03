//! Metal against CPU on every weight matrix of the real model (env-gated): with
//! activations in the range a forward pass produces, every output is bit-identical.
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
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
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
