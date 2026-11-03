//! Metal forward-pass operations against `qubivex-arch`'s CPU ops: every output must be
//! bit-identical (ADR-011 §1, amendment 2).
//!
//! Skipped, with a message, on machines without a Metal device.

#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use qubivex_arch::ops::{Rope, attend, rms_norm, softmax, swiglu};
use qubivex_backend_metal::{AttentionShape, MetalBackend};
use qubivex_compute::math;

/// xorshift64*.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in [-1, 1).
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f() * scale).collect()
    }
}

fn backend() -> Option<MetalBackend> {
    match MetalBackend::new() {
        Ok(b) => Some(b),
        // Only a missing device skips; a shader that fails to compile is a failure.
        Err(qubivex_backend_metal::MetalError::Unavailable(e)) => {
            eprintln!("skipping: {e}");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

fn same_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "{what}: element {i}: metal {g:e} ({:#010x}) vs cpu {w:e} ({:#010x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

/// Runs one elementwise op on `x` (and `y`) and returns `x` afterwards.
fn run(
    be: &MetalBackend,
    x: &[f32],
    y: Option<&[f32]>,
    f: impl FnOnce(
        &mut qubivex_backend_metal::Batch<'_, '_>,
        &qubivex_backend_metal::Span<'_>,
        Option<&qubivex_backend_metal::Span<'_>>,
    ) -> Result<(), qubivex_compute::ComputeError>,
) -> Vec<f32> {
    let dx = be.vec_from(x).unwrap();
    let dy = y.map(|y| be.vec_from(y).unwrap());
    be.submit(|b| f(b, &dx.all(), dy.as_ref().map(|v| v.all()).as_ref()))
        .unwrap();
    let mut out = vec![0f32; x.len()];
    be.read(&dx.all(), &mut out).unwrap();
    out
}

#[test]
fn exp_matches_the_portable_exp() {
    let Some(be) = backend() else { return };
    let mut x: Vec<f32> = Vec::new();
    // A dense sweep over the whole range, plus edges.
    let mut v = -110.0f32;
    while v < 110.0 {
        x.push(v);
        v += 0.000_731;
    }
    let mut rng = Rng(7);
    x.extend(rng.vec(100_000, 1.0));
    x.extend(rng.vec(100_000, 1e-6));
    x.extend([
        0.0,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        88.72284,
        88.722_85,
        -87.336_55,
        -87.3365,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ]);
    let got = run(&be, &x, None, |b, dx, _| b.exp(dx, dx.len()));
    let want: Vec<f32> = x.iter().map(|v| math::exp(*v)).collect();
    // NaN payloads may differ; compare NaN-ness for them.
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        if w.is_nan() {
            assert!(g.is_nan(), "exp({}) should be NaN", x[i]);
        } else {
            assert_eq!(g.to_bits(), w.to_bits(), "exp({:e}): {g:e} vs {w:e}", x[i]);
        }
    }
}

#[test]
fn rms_norm_matches() {
    let Some(be) = backend() else { return };
    let mut rng = Rng(11);
    for n in [1, 31, 32, 63, 64, 65, 100, 128, 2047, 2048] {
        for rows in [1, 3] {
            for eps in [1e-5, 1e-6] {
                let x = rng.vec(rows * n, 3.0);
                let w = rng.vec(n, 1.5);
                let dw = be.vec_from(&w).unwrap();
                let got = run(&be, &x, None, |b, dx, _| {
                    b.rms_norm(dx, rows, &dw.all(), eps)
                });
                let mut want = x.clone();
                want.chunks_exact_mut(n).for_each(|r| rms_norm(r, &w, eps));
                same_bits(&format!("rms_norm n={n} rows={rows}"), &got, &want);
            }
        }
    }
}

#[test]
fn rope_matches() {
    let Some(be) = backend() else { return };
    let mut rng = Rng(13);
    for (d, heads, theta) in [(16, 3, 10_000.0), (128, 16, 10_000.0), (64, 4, 1e6)] {
        let rope = Rope::new(d, theta);
        let half = d / 2;
        for (p0, rows) in [(0, 1), (7, 1), (4095, 1), (3, 5)] {
            let x = rng.vec(rows * heads * d, 2.0);
            let mut table = vec![0f32; rows * d];
            for t in 0..rows {
                rope.table(p0 + t, &mut table[t * d..(t + 1) * d]);
            }
            let dt = be.vec_from(&table).unwrap();
            let got = run(&be, &x, None, |b, dx, _| {
                b.rope(dx, rows, heads, half, &dt.all())
            });
            let mut want = x.clone();
            for t in 0..rows {
                want[t * heads * d..(t + 1) * heads * d]
                    .chunks_exact_mut(d)
                    .for_each(|h| rope.apply(h, p0 + t));
            }
            same_bits(&format!("rope d={d} p0={p0} rows={rows}"), &got, &want);
        }
    }
}

#[test]
fn elementwise_ops_match() {
    let Some(be) = backend() else { return };
    let mut rng = Rng(17);
    let n = 10_000;
    let g = rng.vec(n, 20.0);
    let u = rng.vec(n, 3.0);

    let got = run(&be, &g, Some(&u), |b, dg, du| b.swiglu(dg, du.unwrap(), n));
    let mut want = g.clone();
    swiglu(&mut want, &u);
    same_bits("swiglu", &got, &want);

    let got = run(&be, &g, Some(&u), |b, dg, du| b.add(dg, du.unwrap(), n));
    let want: Vec<f32> = g.iter().zip(&u).map(|(a, b)| a + b).collect();
    same_bits("add", &got, &want);

    let got = run(&be, &g, Some(&u), |b, dg, du| b.copy(dg, du.unwrap(), n));
    same_bits("copy", &got, &u);

    let mut c = g.clone();
    c.extend([f32::INFINITY, f32::NEG_INFINITY, -0.0, 8.0, -8.0]);
    let got = run(&be, &c, None, |b, dc, _| b.clamp(dc, dc.len(), 8.0));
    let want: Vec<f32> = c.iter().map(|v| v.clamp(-8.0, 8.0)).collect();
    same_bits("clamp", &got, &want);
}

/// The CPU reference: the forward pass's KV layout (`[kv_head]` → `[pos][d]`) and
/// `ops::attend` per query head.
fn cpu_attention(q: &[f32], kc: &[Vec<f32>], vc: &[Vec<f32>], a: AttentionShape) -> Vec<f32> {
    let (h, d) = (a.heads, a.head_dim);
    let group = a.heads / a.kv_heads;
    let mut out = vec![0f32; a.tokens * h * d];
    let mut scores = Vec::new();
    for t in 0..a.tokens {
        for hd in 0..h {
            let at = (t * h + hd) * d;
            attend(
                &q[at..at + d],
                &kc[hd / group],
                &vc[hd / group],
                a.p0 + t,
                &mut out[at..at + d],
                &mut scores,
            );
        }
    }
    out
}

#[test]
fn kv_store_and_attention_match() {
    let Some(be) = backend() else { return };
    let mut rng = Rng(19);
    for (heads, kv_heads, d) in [(4, 2, 16), (16, 16, 128), (6, 1, 8)] {
        for (p0, tokens) in [(0, 1), (0, 6), (9, 1), (40, 3), (200, 1)] {
            let a = AttentionShape {
                tokens,
                heads,
                kv_heads,
                head_dim: d,
                p0,
                capacity: 256,
            };
            let kv_out = kv_heads * d;
            let n_pos = p0 + tokens;
            // Past keys and values as the CPU cache holds them, plus this call's new ones.
            let k_all = rng.vec(n_pos * kv_out, 2.0);
            let v_all = rng.vec(n_pos * kv_out, 2.0);
            let q = rng.vec(tokens * heads * d, 2.0);
            let mut kc = vec![Vec::new(); kv_heads];
            let mut vc = vec![Vec::new(); kv_heads];
            for pos in 0..n_pos {
                for g in 0..kv_heads {
                    let s = pos * kv_out + g * d;
                    kc[g].extend_from_slice(&k_all[s..s + d]);
                    vc[g].extend_from_slice(&v_all[s..s + d]);
                }
            }
            let want = cpu_attention(&q, &kc, &vc, a);

            // Device: store the past in one call, then this call's tokens in another,
            // inside the same submission as the attention.
            let cap = kv_heads * a.capacity * d;
            let (dk, dv) = (be.vec(cap).unwrap(), be.vec(cap).unwrap());
            let (sk, sv) = (be.vec_from(&k_all).unwrap(), be.vec_from(&v_all).unwrap());
            let dq = be.vec_from(&q).unwrap();
            let scores = be.vec(tokens * heads * n_pos).unwrap();
            let out = be.vec(q.len()).unwrap();
            be.submit(|b| {
                let past = AttentionShape {
                    tokens: p0,
                    p0: 0,
                    ..a
                };
                if p0 > 0 {
                    b.kv_store(&dk.all(), &sk.span(0, p0 * kv_out)?, past)?;
                    b.kv_store(&dv.all(), &sv.span(0, p0 * kv_out)?, past)?;
                }
                b.kv_store(&dk.all(), &sk.span(p0 * kv_out, tokens * kv_out)?, a)?;
                b.kv_store(&dv.all(), &sv.span(p0 * kv_out, tokens * kv_out)?, a)?;
                b.attention(
                    &dq.all(),
                    &dk.all(),
                    &dv.all(),
                    &scores.all(),
                    &out.all(),
                    a,
                )
            })
            .unwrap();
            let mut got = vec![0f32; q.len()];
            be.read(&out.all(), &mut got).unwrap();
            same_bits(
                &format!("attention h={heads}/{kv_heads} d={d} p0={p0} s={tokens}"),
                &got,
                &want,
            );
        }
    }
}

#[test]
fn moe_combine_matches() {
    let Some(be) = backend() else { return };
    let mut rng = Rng(23);
    let (s, k, h) = (3, 8, 2048);
    let contrib = rng.vec(s * k * h, 4.0);
    let mut w = rng.vec(s * k, 1.0);
    for row in w.chunks_exact_mut(k) {
        row.iter_mut().for_each(|x| *x = x.abs());
        softmax(row);
    }
    let (dc, dw, out) = (
        be.vec_from(&contrib).unwrap(),
        be.vec_from(&w).unwrap(),
        be.vec(s * h).unwrap(),
    );
    be.submit(|b| b.moe_combine(&dc.all(), &dw.all(), &out.all(), s, k, h))
        .unwrap();
    let mut got = vec![0f32; s * h];
    be.read(&out.all(), &mut got).unwrap();
    let mut want = vec![0f32; s * h];
    for t in 0..s {
        for r in 0..k {
            let wr = w[t * k + r];
            for i in 0..h {
                let o = &mut want[t * h + i];
                *o = wr.mul_add(contrib[(t * k + r) * h + i], *o);
            }
        }
    }
    same_bits("moe_combine", &got, &want);
}

#[test]
fn gather_and_scatter_rows() {
    let Some(be) = backend() else { return };
    let mut rng = Rng(29);
    let h = 70;
    let src = rng.vec(6 * h, 1.0);
    let rows = [4u32, 0, 5];
    let (ds, dg, dd) = (
        be.vec_from(&src).unwrap(),
        be.vec(3 * h).unwrap(),
        be.vec(6 * h).unwrap(),
    );
    be.submit(|b| {
        b.gather_rows(&dg.all(), &ds.all(), &rows, h)?;
        b.scatter_rows(&dd.all(), &dg.all(), &rows, h)
    })
    .unwrap();
    let (mut g, mut d) = (vec![0f32; 3 * h], vec![0f32; 6 * h]);
    be.read(&dg.all(), &mut g).unwrap();
    be.read(&dd.all(), &mut d).unwrap();
    for (j, &r) in rows.iter().enumerate() {
        let r = r as usize;
        same_bits("gather", &g[j * h..(j + 1) * h], &src[r * h..(r + 1) * h]);
        same_bits("scatter", &d[r * h..(r + 1) * h], &src[r * h..(r + 1) * h]);
    }
    assert!(
        d[h..4 * h].iter().all(|v| *v == 0.0),
        "untouched rows stay zero"
    );
    // Out-of-range indices are refused before anything is recorded.
    assert!(
        be.submit(|b| b.gather_rows(&dg.all(), &ds.all(), &[6], h))
            .is_err()
    );
}

/// Asynchronous submissions run in commit order, host reads wait for them, and a
/// wrapped host region stays alive until the GPU is done even when the caller drops
/// every handle to it first.
#[test]
fn async_submissions_are_ordered_and_keep_host_memory_alive() {
    use qubivex_compute::WeightView;
    use qubivex_compute::host::PageAligned;
    use qubivex_core::DType;
    use std::sync::Arc;
    let Some(be) = backend() else { return };
    let mut rng = Rng(31);
    let n = 50_000;
    let (a, b) = (rng.vec(n, 1.0), rng.vec(n, 1.0));
    let (da, db) = (be.vec_from(&a).unwrap(), be.vec_from(&b).unwrap());
    let p1 = be
        .submit_async(|bt| bt.add(&da.all(), &db.all(), n))
        .unwrap();
    let p2 = be
        .submit_async(|bt| bt.add(&da.all(), &db.all(), n))
        .unwrap();
    // The read drains both before copying.
    let mut got = vec![0f32; n];
    be.read(&da.all(), &mut got).unwrap();
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| (x + y) + y).collect();
    same_bits("two ordered adds", &got, &want);
    assert!(p1.is_complete() && p2.is_complete());

    // A matmul over a wrapped region whose every handle is dropped before waiting.
    let (rows, cols) = (512, 1024);
    let w: Vec<f32> = rng.vec(rows * cols, 0.05);
    let x = rng.vec(cols, 1.0);
    let mut region = PageAligned::zeroed(rows * cols * 4);
    for (dst, v) in region.as_mut_slice().chunks_exact_mut(4).zip(&w) {
        dst.copy_from_slice(&v.to_le_bytes());
    }
    let region = Arc::new(region);
    let host = be.map_host(Arc::clone(&region)).unwrap();
    let view = WeightView::new(DType::F32, rows, cols, region.as_slice()).unwrap();
    let mw = host.weights(&view).unwrap().unwrap();
    let (dx, dy) = (be.vec_from(&x).unwrap(), be.vec(rows).unwrap());
    let mut p = be
        .submit_async(|bt| bt.matmul(&mw, &dx.all(), 1, &dy.all()))
        .unwrap();
    drop((mw, host, region));
    p.wait().unwrap();
    let mut y = vec![0f32; rows];
    be.read(&dy.all(), &mut y).unwrap();
    let want: Vec<f32> = w
        .chunks_exact(cols)
        .map(|r| qubivex_compute::reduce::dot(r, &x))
        .collect();
    same_bits("matmul over a released region", &y, &want);
}
