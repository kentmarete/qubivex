//! The SwiGLU check's reference hash (audit §15): runs `qubivex_arch::ops::swiglu` on every
//! f32 bit pattern of g, with u as in `swiglu/check.m` (pass 1: u = 1; pass 2: u from
//! mix64), and prints the same order-independent hash of the outputs. Equal hashes mean
//! the harness's C port computes exactly the CPU's bits over all 2^32 inputs.

use qubivex_arch::ops::swiglu;

fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn u_bits(pass: u32, i: u64) -> u32 {
    if pass == 1 { 0x3f80_0000 } else { mix64(i ^ 0x5851_f42d_4c95_7f2d) as u32 }
}

fn main() {
    let passes = std::env::args().nth(1).unwrap_or_else(|| "12".into());
    let threads = std::thread::available_parallelism().map_or(8, usize::from) as u64;
    for pass in passes.chars().map(|c| c.to_digit(10).unwrap()) {
        let per = (1u64 << 32) / threads;
        let hash = std::thread::scope(|s| {
            (0..threads)
                .map(|t| {
                    s.spawn(move || {
                        let end = if t + 1 == threads { 1u64 << 32 } else { (t + 1) * per };
                        let (mut g, mut u) = (vec![0f32; 4096], vec![0f32; 4096]);
                        let mut h = 0u64;
                        let mut i = t * per;
                        while i < end {
                            let n = (end - i).min(4096) as usize;
                            for j in 0..n {
                                g[j] = f32::from_bits((i + j as u64) as u32);
                                u[j] = f32::from_bits(u_bits(pass, i + j as u64));
                            }
                            swiglu(&mut g[..n], &u[..n]);
                            for j in 0..n {
                                h ^= mix64(((i + j as u64) << 32) | u64::from(g[j].to_bits()));
                            }
                            i += n as u64;
                        }
                        h
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .fold(0, |a, j| a ^ j.join().unwrap())
        });
        println!("pass {pass}: Rust ops::swiglu hash {hash:016x}");
    }
}
