//! Checks the binding harness's SE dumps (audit §15): for every routed slot, the gate
//! after SwiGLU, the up, and the down outputs against `CpuBackend::matmul`,
//! `qubivex_arch::ops::swiglu` and `CpuBackend::matmul` on the SwiGLU output, bit for bit.

use qubivex_arch::ops::swiglu;
use qubivex_backend_cpu::CpuBackend;
use qubivex_compute::{ComputeBackend, WeightView};
use qubivex_core::{DType, GgmlType};

// Must match binding.m.
const H: usize = 2048;
const F: usize = 1024;
const MAXE: usize = 8;

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let dir = std::env::args().nth(1).expect("dump directory");
    let read = |n: &str| std::fs::read(format!("{dir}/{n}")).unwrap_or_else(|e| panic!("{n}: {e}"));
    let pool = read("pool.bin");
    let x = floats(&read("x.bin"));
    let rb = |k: usize| k / 256 * 144;
    let eb = 2 * F * rb(H) + H * rb(F);
    let q4k = DType::Ggml(GgmlType::Q4_K);
    let cpu = CpuBackend::new();
    let mm = |w: &[u8], n: usize, k: usize, xs: &[f32], s: usize| {
        let wv = WeightView::new(q4k, n, k, w).unwrap();
        let mut ys = vec![0f32; s * n];
        cpu.matmul(&wv, xs, s, &mut ys).unwrap();
        ys
    };
    let (mut bad_cfg, mut cfgs) = (0, 0);
    for e in [1usize, 3, 8] {
        for s in [1usize, 4, 10] {
            let raw = read(&format!("se-e{e}-s{s}.bin"));
            let sel: Vec<usize> = raw[..4 * MAXE].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize).collect();
            let got = floats(&raw[4 * MAXE..]);
            let (gg, rest) = got.split_at(e * s * F);
            let (gu, gd) = rest.split_at(e * s * F);
            let mut bad = [0usize; 3];
            for (i, &xi) in sel[..e].iter().enumerate() {
                let w = &pool[xi * eb..(xi + 1) * eb];
                let xs = &x[..s * H];
                let mut g = mm(&w[..F * rb(H)], F, H, xs, s);
                let u = mm(&w[F * rb(H)..2 * F * rb(H)], F, H, xs, s);
                swiglu(&mut g, &u);
                let d = mm(&w[2 * F * rb(H)..], H, F, &g, s);
                let cmp = |a: &[f32], b: &[f32]| a.iter().zip(b).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
                bad[0] += cmp(&gg[i * s * F..(i + 1) * s * F], &g);
                bad[1] += cmp(&gu[i * s * F..(i + 1) * s * F], &u);
                bad[2] += cmp(&gd[i * s * H..(i + 1) * s * H], &d);
            }
            println!("block e{e} s{s:<2} experts {:?}: swiglu(gate) {} / up {} / down {} outputs differ from the CPU (of {}, {}, {})",
                     &sel[..e], bad[0], bad[1], bad[2], e * s * F, e * s * F, e * s * H);
            cfgs += 1;
            bad_cfg += usize::from(bad.iter().any(|&b| b != 0));
        }
    }
    println!("block check: {bad_cfg} of {cfgs} configurations differ");
    std::process::exit(i32::from(bad_cfg != 0));
}
