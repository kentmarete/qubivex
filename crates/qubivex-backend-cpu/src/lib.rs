//! The CPU compute backend.
//!
//! Kernels are lossless (f32 activations times the exact dequantized weights) and
//! deterministic: work is split across output rows only, never across the reduction,
//! and every dot product uses the one fixed reduction shape in [`reduce`]. The bits
//! of an output therefore depend only on its weight row and activation row, never on
//! the batch size, the thread count or the instruction set used (ADR-002).
//!
//! Supported dtypes: f32, f16, bf16, Q8_0, Q4_K and Q6_K.
//!
//! Paths, all with the same bits: one activation row (decode) uses a fused NEON row dot
//! where there is one, and two to five rows (a decode step over a few sequences) its
//! multi-row form, which dequantises each weight once into every row; six or more rows
//! (prefill) use the register-blocked kernel in `tile`, which computes the same 64
//! lanes in a different grouping; everything else dequantises a row and takes one
//! portable dot per output.
//!
//! On x86-64 the same portable source is also compiled with `avx2,fma` and selected
//! at run time; `f32::mul_add` is correctly rounded either way, so both give the
//! same bits. Hand-written NEON and AVX2 kernels come later, as a measured
//! optimisation that must reproduce these bits.

// The only unsafe code is the call into the `avx2,fma`-compiled copy of the portable
// kernel after run-time feature detection (RUST_GUIDELINES §3: this crate may opt in).
#![allow(unsafe_code)]

pub mod formats;
#[cfg(target_arch = "aarch64")]
mod neon;
pub mod reduce;
mod tile;

use qubivex_compute::{ComputeBackend, ComputeError, WeightView};
use qubivex_core::DType;
use rayon::prelude::*;

/// Output rows per parallel task.
const ROWS_PER_TASK: usize = 16;

/// A row-block kernel: `(weights, activations, s, first row, out, scratch row)`.
type RowsKernel = fn(&WeightView<'_>, &[f32], usize, usize, &mut [f32], &mut [f32]);

/// The CPU backend. Parallel work runs on the current rayon pool.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

impl CpuBackend {
    /// A CPU backend.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

fn check(w: &WeightView<'_>, op: &'static str) -> Result<(), ComputeError> {
    if formats::supported(w.dtype()) {
        Ok(())
    } else {
        Err(ComputeError::Unsupported {
            backend: "cpu",
            op,
            dtype: w.dtype(),
        })
    }
}

/// Computes rows `r0 .. r0 + out.len() / s` into `out` (row-major, `s` values per row):
/// dequantize the row once, then one dot product per activation row.
#[inline(always)]
fn rows_portable(
    w: &WeightView<'_>,
    xs: &[f32],
    s: usize,
    r0: usize,
    out: &mut [f32],
    buf: &mut [f32],
) {
    let k = w.cols();
    for (j, o) in out.chunks_exact_mut(s).enumerate() {
        formats::dequant_row(w.dtype(), w.row(r0 + j), buf);
        for (t, y) in o.iter_mut().enumerate() {
            *y = reduce::dot(buf, &xs[t * k..(t + 1) * k]);
        }
    }
}

/// The same code compiled with AVX2 and hardware FMA.
///
/// # Safety
/// The CPU must support `avx2` and `fma`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rows_avx2(
    w: &WeightView<'_>,
    xs: &[f32],
    s: usize,
    r0: usize,
    out: &mut [f32],
    buf: &mut [f32],
) {
    rows_portable(w, xs, s, r0, out, buf);
}

/// A fused row dot: `(row bytes, activations) -> dot product`.
#[cfg(target_arch = "aarch64")]
type RowDot = fn(&[u8], &[f32]) -> f32;

/// A fused row dot for `dtype`, if this CPU has one.
#[cfg(target_arch = "aarch64")]
fn fused(dtype: DType) -> Option<RowDot> {
    use qubivex_core::GgmlType;
    match dtype {
        DType::Ggml(GgmlType::Q4_K) => Some(neon::q4_k),
        DType::Ggml(GgmlType::Q6_K) => Some(neon::q6_k),
        DType::Ggml(GgmlType::Q8_0) => Some(neon::q8_0),
        DType::BF16 => Some(neon::bf16),
        _ => None,
    }
}

/// A fused multi-row dot for `dtype` (2 to `neon::MAX_ROWS` activation rows), if this
/// CPU has one.
#[cfg(target_arch = "aarch64")]
fn fused_rows(dtype: DType) -> Option<neon::RowsDot> {
    use qubivex_core::GgmlType;
    match dtype {
        DType::Ggml(GgmlType::Q4_K) => Some(neon::q4_k_rows),
        DType::Ggml(GgmlType::Q6_K) => Some(neon::q6_k_rows),
        DType::Ggml(GgmlType::Q8_0) => Some(neon::q8_0_rows),
        DType::BF16 => Some(neon::bf16_rows),
        _ => None,
    }
}

fn rows(w: &WeightView<'_>, xs: &[f32], s: usize, r0: usize, out: &mut [f32], buf: &mut [f32]) {
    // One activation row (decode): fuse dequantization into the dot, without the
    // scratch row. The bits equal the portable path's (see `neon`).
    #[cfg(target_arch = "aarch64")]
    if s == 1 {
        if let Some(dot) = fused(w.dtype()) {
            for (j, y) in out.iter_mut().enumerate() {
                *y = dot(w.row(r0 + j), xs);
            }
            return;
        }
    }
    // Two to five rows (a decode step over a few sequences): dequantise each weight
    // once, in registers, into every row, with each row's lanes unchanged (`neon`).
    #[cfg(target_arch = "aarch64")]
    if (2..=neon::MAX_ROWS).contains(&s) {
        if let Some(dot) = fused_rows(w.dtype()) {
            for (j, o) in out.chunks_exact_mut(s).enumerate() {
                dot(w.row(r0 + j), xs, o);
            }
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: both target features were detected on this CPU just above.
        unsafe { rows_avx2(w, xs, s, r0, out, buf) };
        return;
    }
    rows_portable(w, xs, s, r0, out, buf);
}

/// The tiled kernel compiled with AVX2 and hardware FMA.
///
/// # Safety
/// The CPU must support `avx2` and `fma`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tiled_avx2(
    w: &WeightView<'_>,
    xp: &tile::PackedX,
    r0: usize,
    out: &mut [f32],
    sc: &mut tile::Scratch,
) {
    tile::rows(w, xp, r0, out, sc);
}

fn tiled(
    w: &WeightView<'_>,
    xp: &tile::PackedX,
    r0: usize,
    out: &mut [f32],
    sc: &mut tile::Scratch,
) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: both target features were detected on this CPU just above.
        unsafe { tiled_avx2(w, xp, r0, out, sc) };
        return;
    }
    tile::rows(w, xp, r0, out, sc);
}

/// The reference path: one dequantized row and one portable dot product per output,
/// without the tiled kernel or run-time dispatch, for tests that compare the paths.
#[doc(hidden)]
pub fn matmul_portable(
    w: &WeightView<'_>,
    xs: &[f32],
    s: usize,
    ys: &mut [f32],
) -> Result<(), ComputeError> {
    matmul_with(w, xs, s, ys, Path::Rows(rows_portable))
}

/// How [`matmul_with`] computes a block of rows.
#[derive(Clone, Copy)]
enum Path {
    /// One dot product per output.
    Rows(RowsKernel),
    /// The register-blocked kernel for several activation rows ([`tile`]), else
    /// the given row kernel.
    Tiled(RowsKernel),
}

fn matmul_with(
    w: &WeightView<'_>,
    xs: &[f32],
    s: usize,
    ys: &mut [f32],
    path: Path,
) -> Result<(), ComputeError> {
    check(w, "matmul")?;
    let (n, k) = (w.rows(), w.cols());
    if xs.len() != s * k || ys.len() != s * n {
        return Err(ComputeError::Shape {
            op: "matmul",
            detail: format!(
                "{s} activation rows of {k} and {s} output rows of {n} need {} and {} values, got {} and {}",
                s * k,
                s * n,
                xs.len(),
                ys.len()
            ),
        });
    }
    if s == 0 || n == 0 {
        return Ok(());
    }
    // Rows are independent: compute them row-major (n × s), then transpose for s > 1.
    let run = |yt: &mut [f32]| match path {
        Path::Tiled(_) if tile::applies(s) => {
            let xp = tile::PackedX::new(xs, s, k);
            yt.par_chunks_mut(ROWS_PER_TASK * s)
                .enumerate()
                .for_each_init(
                    || tile::Scratch::new(k),
                    |sc, (i, out)| tiled(w, &xp, i * ROWS_PER_TASK, out, sc),
                );
        }
        Path::Rows(kernel) | Path::Tiled(kernel) => {
            yt.par_chunks_mut(ROWS_PER_TASK * s)
                .enumerate()
                .for_each_init(
                    || vec![0f32; k],
                    |buf, (i, out)| kernel(w, xs, s, i * ROWS_PER_TASK, out, buf),
                );
        }
    };
    if s == 1 {
        run(ys);
    } else {
        let mut yt = vec![0f32; n * s];
        run(&mut yt);
        ys.par_chunks_mut(n).enumerate().for_each(|(t, yrow)| {
            for (r, y) in yrow.iter_mut().enumerate() {
                *y = yt[r * s + t];
            }
        });
    }
    Ok(())
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn supports(&self, dtype: DType) -> bool {
        formats::supported(dtype)
    }

    fn matmul(
        &self,
        w: &WeightView<'_>,
        xs: &[f32],
        s: usize,
        ys: &mut [f32],
    ) -> Result<(), ComputeError> {
        matmul_with(w, xs, s, ys, Path::Tiled(rows))
    }

    fn dequant_row(
        &self,
        w: &WeightView<'_>,
        r: usize,
        out: &mut [f32],
    ) -> Result<(), ComputeError> {
        check(w, "dequant_row")?;
        if r >= w.rows() || out.len() != w.cols() {
            return Err(ComputeError::Shape {
                op: "dequant_row",
                detail: format!(
                    "row {r} of {} into {} values (cols {})",
                    w.rows(),
                    out.len(),
                    w.cols()
                ),
            });
        }
        formats::dequant_row(w.dtype(), w.row(r), out);
        Ok(())
    }
}
