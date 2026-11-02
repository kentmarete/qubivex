//! ADR-002 spike harness.
//!
//! `adr-002-tensor-engine`        run every benchmark, print markdown tables
//! `adr-002-tensor-engine det`    print output bit hashes (run by the parent per thread count)

use adr_002_tensor_engine::{
    Fmt, Source, activations, attn, attn_data, bits_hash, candle_impl::CandleW, kernels, reference,
    rel_err,
};
use candle_core::{Device, Tensor};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::process::Command;
use std::time::{Duration, Instant};

/// Expert FFN shapes: (label, n, k). gate/up project 2048 → 1024, down 1024 → 2048.
const SHAPES: [(&str, usize, usize); 2] = [("gate/up 2048→1024", 1024, 2048), ("down 1024→2048", 2048, 1024)];
const SEQS: [usize; 2] = [1, 128];
/// Weight bytes rotated through in cold mode, well above the M3 Pro's caches.
const COLD_POOL_BYTES: usize = 512 << 20;
const DET_THREADS: [usize; 4] = [1, 2, 4, 11];

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("det") => det_child(),
        Some("dump") => dump(&std::env::args().nth(2).expect("dump <dir>")),
        Some("attn") => attn_bench(),
        _ => bench(),
    }
}

/// Attention decode only (own kernel), for repeated A/B runs against the Colibri harness.
fn attn_bench() {
    for t in [1024, 4096] {
        let a = attn_data(16, 128, t, 300 + t as u64);
        let mut out = vec![0f32; 16 * 128];
        let ns = time_ns(|_| attn::own(&a, &mut out), 10, Duration::from_millis(400));
        println!("| {t} | own (rayon) | {:.1} | {:.1} |", ns / 1e3, a.kv_bytes() as f64 / ns);
        let ns = time_ns(|_| attn::own_pool(&a, &mut out), 10, Duration::from_millis(400));
        println!("| {t} | own (spinning pool) | {:.1} | {:.1} |", ns / 1e3, a.kv_bytes() as f64 / ns);
    }
}

fn write_raw<T: Copy>(path: &std::path::Path, v: &[T]) {
    // SAFETY: T is a plain-old-data numeric type (f32/f64/i8/u8); we only read its bytes.
    let b = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) };
    std::fs::write(path, b).expect("write dump file");
}

/// Write the exact inputs and f64 references the benchmarks use, so an external
/// engine (the Colibri baseline harness) runs on identical bytes.
///
/// Per shape `n{n}_k{k}`: `w_f32`, `w_i8`, `s_i8`, `w_q8_0`, `w_q4_k`, `x_s{s}`,
/// `ref_{fmt}_s{s}` (f64, the reference token rows concatenated, n each).
/// Attention: `attn_t{t}_{q,k,v}` (f32) and `attn_t{t}_ref` (f64).
fn dump(dir: &str) {
    use adr_002_tensor_engine::formats::I8RowMat;
    let d = std::path::Path::new(dir);
    std::fs::create_dir_all(d).expect("mkdir");
    for (si, &(_, n, k)) in SHAPES.iter().enumerate() {
        let src = Source::new(n, k, 100 + si as u64);
        let p = |name: &str| d.join(format!("n{n}_k{k}_{name}"));
        write_raw(&p("w_f32"), &src.f32.w);
        let i8m = I8RowMat::from_f32(&src.f32);
        write_raw(&p("w_i8"), &i8m.q);
        write_raw(&p("s_i8"), &i8m.scale);
        write_raw(&p("w_q8_0"), &src.q8_0);
        write_raw(&p("w_q4_k"), &src.q4k);
        for &s in &SEQS {
            let xs = activations(s, k, 200 + s as u64);
            write_raw(&p(&format!("x_s{s}")), &xs);
            let toks: Vec<usize> = if s == 1 { vec![0] } else { vec![0, 37, s - 1] };
            for fmt in Fmt::ALL {
                let refs: Vec<f64> = reference(src.own(fmt).as_ref(), &xs, &toks).concat();
                write_raw(&p(&format!("ref_{}_s{s}", fmt.label())), &refs);
            }
        }
    }
    for t in [1024usize, 4096] {
        let a = attn_data(16, 128, t, 300 + t as u64);
        write_raw(&d.join(format!("attn_t{t}_q")), &a.q);
        write_raw(&d.join(format!("attn_t{t}_k")), &a.k);
        write_raw(&d.join(format!("attn_t{t}_v")), &a.v);
        write_raw(&d.join(format!("attn_t{t}_ref")), &attn::reference(&a));
    }
}

/// Median wall time of `f(i)` in ns; `f` gets the iteration index for pool rotation.
fn time_ns(mut f: impl FnMut(usize), warm: usize, budget: Duration) -> f64 {
    for i in 0..warm {
        f(i);
    }
    let mut v = Vec::new();
    let start = Instant::now();
    let mut i = warm;
    while (start.elapsed() < budget || v.len() < 15) && v.len() < 2000 {
        let t = Instant::now();
        f(i);
        v.push(t.elapsed().as_nanos() as f64);
        i += 1;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn stream_read_gbs(threads_all: bool) -> f64 {
    let n = 256 << 20; // 1 GiB of f32
    let buf: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    // 64 independent lanes so the adds vectorize and memory, not FP latency, binds.
    let sum = |c: &[f32]| {
        let mut acc = [0f32; 64];
        for ch in c.as_chunks::<64>().0 {
            for l in 0..64 {
                acc[l] += ch[l];
            }
        }
        acc.iter().sum::<f32>()
    };
    let run = || -> f32 {
        if threads_all { buf.par_chunks(1 << 16).map(sum).sum() } else { buf.chunks(1 << 16).map(sum).sum() }
    };
    let ns = time_ns(|_| { std::hint::black_box(run()); }, 2, Duration::from_secs(2));
    (n * 4) as f64 / ns
}

struct Row {
    shape: &'static str,
    s: usize,
    fmt: Fmt,
    engine: &'static str,
    hot_us: f64,
    cold_us: f64,
    gbs_cold: f64,
    gflops_hot: f64,
    err: f64,
    s1_eq_sn: Option<bool>,
}

fn bench() {
    let threads = rayon::current_num_threads();
    println!("# ADR-002 spike results\n");
    println!("- rayon threads: {threads} (Candle reads RAYON_NUM_THREADS too)");
    println!("- candle accelerate feature: {}", cfg!(feature = "accelerate"));
    let st1 = stream_read_gbs(false);
    let stn = stream_read_gbs(true);
    println!("- STREAM-style read bandwidth: {st1:.1} GB/s (1 thread), {stn:.1} GB/s (all threads)\n");

    let mut rows = Vec::new();
    for (si, &(label, n, k)) in SHAPES.iter().enumerate() {
        let src = Source::new(n, k, 100 + si as u64);
        for &s in &SEQS {
            let xs = activations(s, k, 200 + s as u64);
            let ref_tokens: Vec<usize> = if s == 1 { vec![0] } else { vec![0, 37, s - 1] };
            let x_t = Tensor::from_slice(&xs, (s, k), &Device::Cpu).expect("x");
            for fmt in Fmt::ALL {
                let own = src.own(fmt);
                let refs = reference(own.as_ref(), &xs, &ref_tokens);
                let wbytes = own.weight_bytes();
                let pool = (COLD_POOL_BYTES / wbytes).max(2);
                let flops = (2 * n * k * s) as f64;
                let budget = Duration::from_millis(if s == 1 { 400 } else { 800 });

                // Own kernels.
                let own_pool: Vec<_> = (0..pool).map(|_| src.own(fmt)).collect();
                let mut ys = vec![0f32; s * n];
                let mut scratch = vec![0f32; s * n];
                let mut run_own = |w: &dyn adr_002_tensor_engine::formats::Weights, ys: &mut [f32]| {
                    if s == 1 { kernels::gemv(w, &xs, ys) } else { kernels::gemm(w, &xs, s, ys, &mut scratch) }
                };
                let hot = time_ns(|_| run_own(own.as_ref(), &mut ys), 5, budget);
                let cold = time_ns(|i| run_own(own_pool[i % pool].as_ref(), &mut ys), pool, budget);
                run_own(own.as_ref(), &mut ys);
                let err = ref_tokens
                    .iter()
                    .zip(&refs)
                    .map(|(&t, r)| rel_err(&ys[t * n..(t + 1) * n], r))
                    .fold(0.0, f64::max);
                // Token 0 of the batched path against the single-token path.
                if s == 1 {
                    let hot = time_ns(|_| kernels::gemv_rayon(own.as_ref(), &xs, &mut ys), 5, budget);
                    let cold = time_ns(|i| kernels::gemv_rayon(own_pool[i % pool].as_ref(), &xs, &mut ys), pool, budget);
                    rows.push(Row {
                        shape: label, s, fmt, engine: "own, rayon pool", hot_us: hot / 1e3, cold_us: cold / 1e3,
                        gbs_cold: wbytes as f64 / cold, gflops_hot: flops / hot, err, s1_eq_sn: None,
                    });
                }
                let mut y1 = vec![0f32; n];
                kernels::gemv(own.as_ref(), &xs[..k], &mut y1);
                let eq = bits_hash(&y1) == bits_hash(&ys[..n]);
                rows.push(Row {
                    shape: label, s, fmt, engine: "own", hot_us: hot / 1e3, cold_us: cold / 1e3,
                    gbs_cold: wbytes as f64 / cold, gflops_hot: flops / hot, err, s1_eq_sn: Some(eq),
                });
                if s > 1 {
                    let mut deq = vec![0f32; n * k];
                    let hot = time_ns(|_| kernels::gemm_crate(own.as_ref(), &xs, s, &mut ys, &mut deq), 5, budget);
                    let cold = time_ns(|i| kernels::gemm_crate(own_pool[i % pool].as_ref(), &xs, s, &mut ys, &mut deq), pool, budget);
                    kernels::gemm_crate(own.as_ref(), &xs, s, &mut ys, &mut deq);
                    let err = ref_tokens
                        .iter()
                        .zip(&refs)
                        .map(|(&t, r)| rel_err(&ys[t * n..(t + 1) * n], r))
                        .fold(0.0, f64::max);
                    let eq = bits_hash(&y1) == bits_hash(&ys[..n]);
                    rows.push(Row {
                        shape: label, s, fmt, engine: "own deq + gemm crate", hot_us: hot / 1e3, cold_us: cold / 1e3,
                        gbs_cold: wbytes as f64 / cold, gflops_hot: flops / hot, err, s1_eq_sn: Some(eq),
                    });
                }
                drop(own_pool);

                // Candle.
                if let Some(cw) = src.candle(fmt) {
                    let c_pool: Vec<_> = (0..pool).map(|_| src.candle(fmt).expect("candle")).collect();
                    let run_c = |w: &CandleW| w.forward(&x_t).expect("candle forward");
                    let hot = time_ns(|_| { std::hint::black_box(run_c(&cw)); }, 5, budget);
                    let cold = time_ns(|i| { std::hint::black_box(run_c(&c_pool[i % pool])); }, pool, budget);
                    let y: Vec<f32> = run_c(&cw).flatten_all().expect("flat").to_vec1().expect("vec");
                    let err = ref_tokens
                        .iter()
                        .zip(&refs)
                        .map(|(&t, r)| rel_err(&y[t * n..(t + 1) * n], r))
                        .fold(0.0, f64::max);
                    rows.push(Row {
                        shape: label, s, fmt, engine: "candle", hot_us: hot / 1e3, cold_us: cold / 1e3,
                        gbs_cold: wbytes as f64 / cold, gflops_hot: flops / hot, err, s1_eq_sn: None,
                    });
                }
            }
        }
    }

    println!("## Expert-FFN matrices\n");
    println!("Hot: one matrix, cache-resident. Cold: rotating over ≥ 512 MiB of copies (DRAM-bound).");
    println!("GB/s is weight bytes over cold time. Error is max|y−ref|/max|ref| against f64 on the same");
    println!("dequantized weights and f32 activations.\n");
    println!("| shape | S | dtype | engine | hot µs | cold µs | cold GB/s | % STREAM | hot GFLOP/s | rel err | S=1 ≡ S=128 bits |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for r in &rows {
        println!(
            "| {} | {} | {} | {} | {:.1} | {:.1} | {:.1} | {:.0}% | {:.1} | {:.1e} | {} |",
            r.shape, r.s, r.fmt.label(), r.engine, r.hot_us, r.cold_us, r.gbs_cold,
            100.0 * r.gbs_cold / stn, r.gflops_hot, r.err,
            r.s1_eq_sn.map_or("–", |b| if b { "yes" } else { "NO" }),
        );
    }

    println!("\n## Attention decode (16 heads × 128, f32 KV)\n");
    println!("| context | engine | µs | KV GB/s | rel err |");
    println!("|---|---|---|---|---|");
    for t in [1024, 4096] {
        let a = attn_data(16, 128, t, 300 + t as u64);
        let r = attn::reference(&a);
        let mut out = vec![0f32; 16 * 128];
        let ns = time_ns(|_| attn::own(&a, &mut out), 10, Duration::from_millis(400));
        attn::own(&a, &mut out);
        println!("| {t} | own | {:.1} | {:.1} | {:.1e} |", ns / 1e3, a.kv_bytes() as f64 / ns, rel_err(&out, &r));
        let c = attn::CandleAttn::new(&a).expect("candle attn");
        let ns = time_ns(|_| { std::hint::black_box(c.forward().expect("fwd")); }, 10, Duration::from_millis(400));
        let y: Vec<f32> = c.forward().expect("fwd").flatten_all().expect("flat").to_vec1().expect("vec");
        println!("| {t} | candle | {:.1} | {:.1} | {:.1e} |", ns / 1e3, a.kv_bytes() as f64 / ns, rel_err(&y, &r));
    }

    determinism();
}

/// Every case's output hash, computed twice in-process (run-to-run check).
fn det_hashes() -> BTreeMap<String, u64> {
    let mut m = BTreeMap::new();
    for (si, &(label, n, k)) in SHAPES.iter().enumerate() {
        let src = Source::new(n, k, 100 + si as u64);
        for &s in &SEQS {
            let xs = activations(s, k, 200 + s as u64);
            let x_t = Tensor::from_slice(&xs, (s, k), &Device::Cpu).expect("x");
            for fmt in Fmt::ALL {
                let own = src.own(fmt);
                let mut ys = vec![0f32; s * n];
                let mut scratch = vec![0f32; s * n];
                let mut h = Vec::new();
                for _ in 0..2 {
                    if s == 1 { kernels::gemv(own.as_ref(), &xs, &mut ys) } else { kernels::gemm(own.as_ref(), &xs, s, &mut ys, &mut scratch) }
                    h.push(bits_hash(&ys));
                }
                m.insert(format!("{label} S={s} {} own", fmt.label()), if h[0] == h[1] { h[0] } else { 0 });
                if s > 1 {
                    let mut deq = vec![0f32; n * k];
                    let h: Vec<u64> = (0..2)
                        .map(|_| {
                            kernels::gemm_crate(own.as_ref(), &xs, s, &mut ys, &mut deq);
                            bits_hash(&ys)
                        })
                        .collect();
                    m.insert(format!("{label} S={s} {} own deq + gemm crate", fmt.label()), if h[0] == h[1] { h[0] } else { 0 });
                }
                if let Some(cw) = src.candle(fmt) {
                    let h: Vec<u64> = (0..2)
                        .map(|_| bits_hash(&cw.forward(&x_t).expect("fwd").flatten_all().expect("f").to_vec1::<f32>().expect("v")))
                        .collect();
                    m.insert(format!("{label} S={s} {} candle", fmt.label()), if h[0] == h[1] { h[0] } else { 0 });
                }
            }
        }
    }
    for t in [1024, 4096] {
        let a = attn_data(16, 128, t, 300 + t as u64);
        let mut out = vec![0f32; 16 * 128];
        attn::own(&a, &mut out);
        m.insert(format!("attn ctx={t} own"), bits_hash(&out));
        let c = attn::CandleAttn::new(&a).expect("candle attn");
        let y: Vec<f32> = c.forward().expect("fwd").flatten_all().expect("flat").to_vec1().expect("vec");
        m.insert(format!("attn ctx={t} candle"), bits_hash(&y));
    }
    m
}

fn det_child() {
    for (k, v) in det_hashes() {
        println!("{k}\t{v:016x}");
    }
}

fn determinism() {
    println!("\n## Bitwise determinism\n");
    println!("Output hashes from fresh processes with RAYON_NUM_THREADS = {DET_THREADS:?}; each process also");
    println!("runs every case twice (a run-to-run mismatch shows as `0000000000000000`).\n");
    let exe = std::env::current_exe().expect("exe");
    let mut per_thread: Vec<BTreeMap<String, String>> = Vec::new();
    for t in DET_THREADS {
        let out = Command::new(&exe).arg("det").env("RAYON_NUM_THREADS", t.to_string()).output().expect("child");
        assert!(out.status.success(), "det child failed: {}", String::from_utf8_lossy(&out.stderr));
        per_thread.push(
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter_map(|l| l.split_once('\t').map(|(a, b)| (a.to_owned(), b.to_owned())))
                .collect(),
        );
    }
    println!("| case | run-to-run | across thread counts |");
    println!("|---|---|---|");
    for (case, h0) in &per_thread[0] {
        let rr = per_thread.iter().all(|m| m.get(case).is_some_and(|h| h != "0000000000000000"));
        let tc = per_thread.iter().all(|m| m.get(case) == Some(h0));
        println!("| {case} | {} | {} |", if rr { "same" } else { "DIFFERS" }, if tc { "same" } else { "DIFFERS" });
    }
}
