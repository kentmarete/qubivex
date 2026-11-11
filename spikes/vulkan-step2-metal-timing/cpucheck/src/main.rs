//! Reads the harness's `check` dump and compares every output, fused and unfused, with
//! `CpuBackend::matmul` bit for bit: the reference the Vulkan parity tests use.

use qubivex_backend_cpu::CpuBackend;
use qubivex_compute::{ComputeBackend, WeightView};
use qubivex_core::{DType, GgmlType};

// Must match main.m.
const H: usize = 2048;
const F: usize = 1024;
const MAXE: usize = 8;
const MAXT: usize = 32;

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let dir = std::env::args().nth(1).expect("dump directory");
    let read = |n: &str| std::fs::read(format!("{dir}/{n}")).unwrap_or_else(|e| panic!("{n}: {e}"));
    let w = read("weights.bin");
    let x = floats(&read("x.bin"));
    let q4k = DType::Ggml(GgmlType::Q4_K);
    let rb = |k: usize| k / 256 * 144;
    let expert_bytes = 2 * F * rb(H) + H * rb(F);
    let cpu = CpuBackend::new();
    let (mut configs, mut bad_configs) = (0, 0);
    for e in [1usize, 3, 8] {
        for s in [1usize, 2, 4, 10, 32] {
            // Expected, in the harness's output order: per expert gate, up, down.
            let mut want = Vec::new();
            for ex in 0..e {
                let base = ex * expert_bytes;
                let mat = |off: usize, n: usize, k: usize, xs: &[f32]| {
                    let wv = WeightView::new(q4k, n, k, &w[base + off..base + off + n * rb(k)]).unwrap();
                    let mut ys = vec![0f32; s * n];
                    cpu.matmul(&wv, xs, s, &mut ys).unwrap();
                    ys
                };
                let xg = &x[..s * H];
                let gbase = MAXT * H + ex * MAXT * F;
                let xd = &x[gbase..gbase + s * F];
                want.extend(mat(0, F, H, xg));
                want.extend(mat(F * rb(H), F, H, xg));
                want.extend(mat(2 * F * rb(H), H, F, xd));
            }
            for kind in ["fused", "unfused"] {
                let got = floats(&read(&format!("y-{kind}-e{e}-s{s}.bin")));
                assert_eq!(got.len(), want.len());
                let bad = got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                println!("cpu check e{e} s{s:<2} {kind:<7}: {} outputs, {bad} differ from the CPU", got.len());
                configs += 1;
                bad_configs += usize::from(bad != 0);
            }
        }
    }
    println!("cpu check: {bad_configs} of {configs} differ");
    let _ = MAXE;
    std::process::exit(i32::from(bad_configs != 0));
}
