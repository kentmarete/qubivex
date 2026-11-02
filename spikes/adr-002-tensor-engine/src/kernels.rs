//! Own GEMV (S=1) and GEMM (S>1) drivers over [`Weights`].
//!
//! Work is split across rows only, never across `k`, so every output element's
//! reduction order is independent of the thread count.

use crate::formats::{Weights, dot_f32};
use rayon::prelude::*;
use std::cell::RefCell;

const GEMV_ROWS_PER_TASK: usize = 16;
const GEMM_ROWS_PER_TASK: usize = 16;
const GEMM_TOKENS_PER_BLOCK: usize = 8;

/// Raw output pointer shared across pool workers that write disjoint rows.
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: workers only write disjoint indices (one row range each), and the pool joins
// before the borrow of `y` ends.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

/// `y = W·x` on Candle's persistent spinning barrier pool (static row ranges).
pub fn gemv<W: Weights + ?Sized>(w: &W, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), w.k());
    debug_assert_eq!(y.len(), w.n());
    let out = OutPtr(y.as_mut_ptr());
    candle_core::utils::barrier_pool().execute_chunked(w.n(), |rows| {
        // Rebind so the closure captures the whole `OutPtr` (Sync), not its raw-pointer field.
        #[allow(clippy::redundant_locals)]
        let out = out;
        for r in rows {
            let v = w.dot_row(r, x) * w.post_scale(r);
            // SAFETY: r < n = y.len(); each r is visited by exactly one worker.
            unsafe { *out.0.add(r) = v };
        }
    });
}

/// `y = W·x` on the rayon global pool (to measure dispatch cost).
pub fn gemv_rayon<W: Weights + ?Sized>(w: &W, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), w.k());
    debug_assert_eq!(y.len(), w.n());
    y.par_chunks_mut(GEMV_ROWS_PER_TASK).enumerate().for_each(|(i, yc)| {
        for (j, yo) in yc.iter_mut().enumerate() {
            let r = i * GEMV_ROWS_PER_TASK + j;
            *yo = w.dot_row(r, x) * w.post_scale(r);
        }
    });
}

thread_local! {
    static DEQ: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// `Y = X·Wᵀ` for `s` activation rows (`xs` is `s × k`, `ys` is `s × n`).
/// `scratch` must hold `n × s` floats; it receives the row-major transpose.
pub fn gemm<W: Weights + ?Sized>(w: &W, xs: &[f32], s: usize, ys: &mut [f32], scratch: &mut [f32]) {
    let (n, k) = (w.n(), w.k());
    debug_assert_eq!(xs.len(), s * k);
    debug_assert_eq!(ys.len(), s * n);
    let yt = &mut scratch[..n * s];
    yt.par_chunks_mut(GEMM_ROWS_PER_TASK * s).enumerate().for_each(|(i, ytc)| {
        let r0 = i * GEMM_ROWS_PER_TASK;
        let rows = ytc.len() / s;
        DEQ.with(|cell| {
            let mut buf = cell.borrow_mut();
            let direct = w.row_f32(r0).is_some();
            if !direct {
                buf.resize(rows * k, 0.0);
                for j in 0..rows {
                    w.dequant_row(r0 + j, &mut buf[j * k..(j + 1) * k]);
                }
            }
            for t0 in (0..s).step_by(GEMM_TOKENS_PER_BLOCK) {
                let t1 = (t0 + GEMM_TOKENS_PER_BLOCK).min(s);
                for j in 0..rows {
                    let wr: &[f32] = match w.row_f32(r0 + j) {
                        Some(r) => r,
                        None => &buf[j * k..(j + 1) * k],
                    };
                    let sc = w.post_scale(r0 + j);
                    let out = &mut ytc[j * s..(j + 1) * s];
                    for t in t0..t1 {
                        out[t] = dot_f32(wr, &xs[t * k..(t + 1) * k]) * sc;
                    }
                }
            }
        });
    });
    // Transpose n × s → s × n.
    ys.par_chunks_mut(n).enumerate().for_each(|(t, yrow)| {
        for (r, yo) in yrow.iter_mut().enumerate() {
            *yo = yt[r * s + t];
        }
    });
}

/// Prefill through the `gemm` crate: dequantize the whole matrix to f32 (row scale
/// folded in), then one `Y = X·Wᵀ` GEMM. `deq` must hold `n × k` floats.
pub fn gemm_crate<W: Weights + ?Sized>(w: &W, xs: &[f32], s: usize, ys: &mut [f32], deq: &mut [f32]) {
    let (n, k) = (w.n(), w.k());
    let wf: &[f32] = match w.all_f32() {
        Some(all) => all,
        None => {
            deq[..n * k].par_chunks_mut(k).enumerate().for_each(|(r, row)| {
                w.dequant_row(r, row);
                let sc = w.post_scale(r);
                if sc != 1.0 {
                    row.iter_mut().for_each(|v| *v *= sc);
                }
            });
            &deq[..n * k]
        }
    };
    let threads = rayon::current_num_threads();
    let par = if threads > 1 { gemm::Parallelism::Rayon(threads) } else { gemm::Parallelism::None };
    // SAFETY: dst is s × n row-major (rs = n, cs = 1); lhs X is s × k row-major; rhs is
    // Wᵀ, i.e. k × n read from W (n × k row-major) with rs = 1, cs = k. All in bounds.
    unsafe {
        gemm::gemm(
            s, n, k,
            ys.as_mut_ptr(), 1, n as isize, false,
            xs.as_ptr(), 1, k as isize,
            wf.as_ptr(), k as isize, 1,
            0.0f32, 1.0f32, false, false, false, par,
        );
    }
}
