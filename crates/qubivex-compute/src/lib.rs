//! The compute backend interface (QUBIVEX_DESIGN §4.4).
//!
//! A backend multiplies weights in their stored dtype with f32 activations, without
//! changing the weights' numerics: kernels are lossless (the dequantized value of every
//! weight is used exactly) and deterministic (results do not depend on the thread
//! count). Placement may change speed, never bits (invariant 1).
//!
//! Phase 4 defines the host-memory form used by the CPU backend. Unified-memory GPUs
//! (Metal) use host memory in place; devices with their own memory implement
//! [`discrete::DiscreteDevice`] (ADR-018).
//!
//! [`tensor`] holds host tensors and views described by a
//! [`TensorDesc`](qubivex_core::TensorDesc); kernels take a [`WeightView`], which a
//! contiguous rank-2 byte view converts into.

pub mod discrete;
pub mod host;
pub mod math;
pub mod reduce;
pub mod tensor;

use qubivex_core::{CoreError, DType, Shape};
use thiserror::Error;

/// Errors from compute operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ComputeError {
    /// Operand sizes do not match.
    #[error("shape mismatch in {op}: {detail}")]
    Shape {
        /// The operation.
        op: &'static str,
        /// What does not match.
        detail: String,
    },
    /// The backend has no kernel for this dtype.
    #[error("{backend} backend has no {op} kernel for {dtype}")]
    Unsupported {
        /// Backend name.
        backend: &'static str,
        /// The operation.
        op: &'static str,
        /// The dtype.
        dtype: DType,
    },
    /// A size computation failed.
    #[error(transparent)]
    Core(#[from] CoreError),
    /// The storage element type cannot hold or address this dtype.
    #[error("{element} storage cannot hold or address {dtype} elements")]
    ElementType {
        /// The dtype.
        dtype: DType,
        /// The storage element type.
        element: &'static str,
    },
    /// Mutation needs unique storage, but another tensor shares it.
    #[error("tensor storage is shared; mutation needs a unique owner")]
    Shared,
    /// A device does not have the memory for an allocation.
    #[error(
        "{backend} device is out of memory: {requested} bytes requested, {used} of {capacity} in use"
    )]
    OutOfDeviceMemory {
        /// Backend name.
        backend: &'static str,
        /// Bytes asked for.
        requested: u64,
        /// Bytes in use.
        used: u64,
        /// The device's capacity.
        capacity: u64,
    },
    /// A device backend failed (driver, allocation or execution).
    #[error("{backend} device error: {detail}")]
    Device {
        /// Backend name.
        backend: &'static str,
        /// What failed.
        detail: String,
    },
}

/// A row-major `rows × cols` weight matrix in its stored dtype.
///
/// The byte length is checked against the dtype and shape at construction, so kernels
/// may index rows without further checks.
#[derive(Clone, Copy, Debug)]
pub struct WeightView<'a> {
    dtype: DType,
    rows: usize,
    cols: usize,
    row_bytes: usize,
    data: &'a [u8],
}

impl<'a> WeightView<'a> {
    /// A view of `data` as `rows × cols` elements of `dtype`.
    ///
    /// # Errors
    /// [`ComputeError::Shape`] if `data` is not exactly the storage size, and
    /// [`ComputeError::Core`] if the shape is not valid for the dtype (for example a
    /// row length that is not a multiple of the block size).
    pub fn new(
        dtype: DType,
        rows: usize,
        cols: usize,
        data: &'a [u8],
    ) -> Result<Self, ComputeError> {
        let r = rows as u64;
        let c = cols as u64;
        let total = dtype.storage_bytes(&Shape::new(&[r, c])?)?;
        if total != data.len() as u64 {
            return Err(ComputeError::Shape {
                op: "weight view",
                detail: format!(
                    "{dtype} {rows} x {cols} needs {total} bytes, got {}",
                    data.len()
                ),
            });
        }
        let row_bytes = if rows == 0 {
            0
        } else {
            usize::try_from(total / r).map_err(|_| CoreError::Overflow { what: "row bytes" })?
        };
        Ok(Self {
            dtype,
            rows,
            cols,
            row_bytes,
            data,
        })
    }

    /// Element type.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Output rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Input columns.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Bytes per row.
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// The bytes of row `r`. Panics if `r >= rows` (an internal indexing bug).
    #[must_use]
    pub fn row(&self, r: usize) -> &'a [u8] {
        &self.data[r * self.row_bytes..(r + 1) * self.row_bytes]
    }

    /// All bytes.
    #[must_use]
    pub const fn bytes(&self) -> &'a [u8] {
        self.data
    }

    /// The view of `count` consecutive rows starting at `first` (for example one expert
    /// of a stacked expert tensor).
    ///
    /// # Errors
    /// [`ComputeError::Shape`] if the range is outside the matrix.
    pub fn rows_slice(&self, first: usize, count: usize) -> Result<Self, ComputeError> {
        let end = first
            .checked_add(count)
            .filter(|&e| e <= self.rows)
            .ok_or_else(|| ComputeError::Shape {
                op: "rows slice",
                detail: format!("rows {first}..+{count} of {}", self.rows),
            })?;
        Ok(Self {
            rows: count,
            data: &self.data[first * self.row_bytes..end * self.row_bytes],
            ..*self
        })
    }
}

/// One matmul of a batch: `ys = W · xs` for `s` activation rows (as
/// [`ComputeBackend::matmul`]).
#[derive(Debug)]
pub struct MatMul<'a> {
    /// The weights.
    pub w: WeightView<'a>,
    /// `s × cols` activations.
    pub xs: &'a [f32],
    /// Activation rows.
    pub s: usize,
    /// `s × rows` outputs.
    pub ys: &'a mut [f32],
}

/// One gated feed-forward block (SwiGLU), as a mixture-of-experts layer runs per
/// expert: `ys = down · swiglu(gate · xs, up · xs)` for `s` activation rows.
///
/// Every output has the bits of the three matmuls and [`math::swiglu`] run in that
/// order: fusing changes where the intermediates live, never the arithmetic.
#[derive(Debug)]
pub struct GatedFfn<'a> {
    /// `ffn × cols` gate projection.
    pub gate: WeightView<'a>,
    /// `ffn × cols` up projection.
    pub up: WeightView<'a>,
    /// `rows × ffn` down projection.
    pub down: WeightView<'a>,
    /// `s × cols` activations.
    pub xs: &'a [f32],
    /// Activation rows.
    pub s: usize,
    /// `s × rows` outputs.
    pub ys: &'a mut [f32],
}

impl GatedFfn<'_> {
    /// Checks that the three matrices and both buffers fit together.
    ///
    /// # Errors
    /// [`ComputeError::Shape`] when they do not.
    pub fn check(&self) -> Result<(), ComputeError> {
        let (g, u, d) = (&self.gate, &self.up, &self.down);
        if g.rows() != u.rows()
            || g.cols() != u.cols()
            || d.cols() != g.rows()
            || self.xs.len() != self.s * g.cols()
            || self.ys.len() != self.s * d.rows()
        {
            return Err(ComputeError::Shape {
                op: "gated ffn",
                detail: format!(
                    "gate {} x {}, up {} x {}, down {} x {}, {} rows: {} inputs, {} outputs",
                    g.rows(),
                    g.cols(),
                    u.rows(),
                    u.cols(),
                    d.rows(),
                    d.cols(),
                    self.s,
                    self.xs.len(),
                    self.ys.len()
                ),
            });
        }
        Ok(())
    }
}

/// [`ComputeBackend::gated_ffn_many`] as the separate steps: one
/// [`ComputeBackend::matmul_many`] for every gate and up, [`math::swiglu`] on the host,
/// then one for every down. Backends without a fused block use this.
///
/// # Errors
/// As [`ComputeBackend::matmul_many`].
pub fn gated_ffn_steps(
    be: &(impl ComputeBackend + ?Sized),
    ops: &mut [GatedFfn<'_>],
) -> Result<(), ComputeError> {
    for op in ops.iter() {
        op.check()?;
    }
    let mut gs: Vec<Vec<f32>> = ops
        .iter()
        .map(|op| vec![0f32; op.s * op.gate.rows()])
        .collect();
    let mut us: Vec<Vec<f32>> = ops
        .iter()
        .map(|op| vec![0f32; op.s * op.up.rows()])
        .collect();
    let mut first = Vec::with_capacity(2 * ops.len());
    for ((op, g), u) in ops.iter().zip(gs.iter_mut()).zip(us.iter_mut()) {
        first.push(MatMul {
            w: op.gate,
            xs: op.xs,
            s: op.s,
            ys: g,
        });
        first.push(MatMul {
            w: op.up,
            xs: op.xs,
            s: op.s,
            ys: u,
        });
    }
    be.matmul_many(&mut first)?;
    drop(first);
    for (g, u) in gs.iter_mut().zip(&us) {
        math::swiglu(g, u);
    }
    let mut second: Vec<MatMul<'_>> = ops
        .iter_mut()
        .zip(&gs)
        .map(|(op, g)| MatMul {
            w: op.down,
            xs: g,
            s: op.s,
            ys: &mut *op.ys,
        })
        .collect();
    be.matmul_many(&mut second)
}

/// A compute backend.
pub trait ComputeBackend: Send + Sync {
    /// Short name (`cpu`).
    fn name(&self) -> &'static str;

    /// Whether [`ComputeBackend::matmul`] and [`ComputeBackend::dequant_row`] support `dtype`.
    fn supports(&self, dtype: DType) -> bool;

    /// `ys[t] = W · xs[t]` for `s` activation rows: `xs` is `s × cols`, `ys` is
    /// `s × rows`, both row-major f32.
    ///
    /// Every output element is the same function of its weight row and activation row
    /// whatever `s` and the thread count are, so batching never changes bits.
    ///
    /// # Errors
    /// [`ComputeError::Shape`] for mismatched lengths, [`ComputeError::Unsupported`]
    /// for a dtype without a kernel.
    fn matmul(
        &self,
        w: &WeightView<'_>,
        xs: &[f32],
        s: usize,
        ys: &mut [f32],
    ) -> Result<(), ComputeError>;

    /// Independent matmuls (no output feeds another's input), as one submission where
    /// the backend can batch them (ADR-011: one command buffer on a GPU). Each output
    /// has the bits [`ComputeBackend::matmul`] gives it.
    ///
    /// # Errors
    /// As [`ComputeBackend::matmul`]; on an error, outputs may be partly written.
    fn matmul_many(&self, ops: &mut [MatMul<'_>]) -> Result<(), ComputeError> {
        for op in ops {
            self.matmul(&op.w, op.xs, op.s, op.ys)?;
        }
        Ok(())
    }

    /// Whether callers should put independent matmuls into one
    /// [`ComputeBackend::matmul_many`] wherever they can, rather than issuing them
    /// from several threads at once: `true` when the backend's discrete device asks
    /// for it ([`discrete::DiscreteDevice::groups_matmuls`]). Grouping never changes
    /// bits.
    fn groups_matmuls(&self) -> bool {
        false
    }

    /// Whether callers should run `blocks` gated feed-forward blocks of at most `rows`
    /// activation rows each as one [`ComputeBackend::gated_ffn_many`] (a fused expert
    /// block), rather than as separate matmuls. The answer may depend on the shape: a
    /// fused block can win at decode and lose at prefill. The default is `false`.
    fn fuses_gated_ffn(&self, blocks: usize, rows: usize) -> bool {
        let _ = (blocks, rows);
        false
    }

    /// Independent gated feed-forward blocks ([`GatedFfn`]). Each output has the bits
    /// of the separate steps ([`gated_ffn_steps`], the default).
    ///
    /// # Errors
    /// As [`ComputeBackend::matmul`]; on an error, outputs may be partly written.
    fn gated_ffn_many(&self, ops: &mut [GatedFfn<'_>]) -> Result<(), ComputeError> {
        gated_ffn_steps(self, ops)
    }

    /// Writes the exact f32 values of row `r` (embedding lookup).
    ///
    /// # Errors
    /// As [`ComputeBackend::matmul`], and [`ComputeError::Shape`] if `r` is out of range.
    fn dequant_row(
        &self,
        w: &WeightView<'_>,
        r: usize,
        out: &mut [f32],
    ) -> Result<(), ComputeError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use qubivex_core::GgmlType;

    #[test]
    fn views_check_sizes() {
        let d = vec![0u8; 2 * 3 * 4];
        let v = WeightView::new(DType::F32, 2, 3, &d).unwrap();
        assert_eq!(
            (v.rows(), v.cols(), v.row_bytes(), v.row(1).len()),
            (2, 3, 12, 12)
        );
        assert!(WeightView::new(DType::F32, 2, 4, &d).is_err());
        let q = vec![0u8; 2 * 144];
        assert!(WeightView::new(DType::Ggml(GgmlType::Q4_K), 2, 256, &q).is_ok());
        assert!(
            WeightView::new(DType::Ggml(GgmlType::Q4_K), 2, 128, &q).is_err(),
            "misaligned block"
        );
        let s = v.rows_slice(1, 1).unwrap();
        assert_eq!(s.bytes().as_ptr(), d[12..].as_ptr());
        assert!(v.rows_slice(1, 2).is_err());
        assert!(v.rows_slice(usize::MAX, 2).is_err());
    }
}
