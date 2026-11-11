//! Host time of `qubivex_arch::ops::swiglu` at a layer's sizes (audit §15): what the
//! device SwiGLU would take off the critical path between the two expert stages.
use qubivex_arch::ops::swiglu;

fn main() {
    for n in [1024usize, 3 * 1024, 8 * 1024, 8 * 4 * 1024, 8 * 10 * 1024] {
        let u: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
        let g0: Vec<f32> = (0..n).map(|i| ((i * 53 % 97) as f32 - 48.0) / 16.0).collect();
        let mut g = g0.clone();
        let mut t = Vec::new();
        for _ in 0..2000 {
            g.copy_from_slice(&g0);
            let s = std::time::Instant::now();
            swiglu(&mut g, &u);
            t.push(s.elapsed().as_secs_f64() * 1e6);
            std::hint::black_box(&g);
        }
        t.sort_by(f64::total_cmp);
        println!("host ops::swiglu, n = {n:6}: {:.1} us (median, one thread)", t[t.len() / 2]);
    }
}
