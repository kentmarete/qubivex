//! The compute backend interface (QUBIVEX_DESIGN §4.4).
//!
//! A backend multiplies weights in their stored dtype with f32 activations, without
//! changing the weights' numerics: kernels are lossless (the dequantized value of every
//! weight is used exactly) and deterministic (results do not depend on the thread
//! count). Placement may change speed, never bits (invariant 1).
//!
//! Phase 4 defines the host-memory form used by the CPU backend. Device buffers,
//! uploads and events join the trait with the first GPU backend (Phase 8).
//!
//! [`tensor`] holds host tensors and views described by a
//! [`TensorDesc`](qubivex_core::TensorDesc); kernels take a [`WeightView`], which a
//! contiguous rank-2 byte view converts into.

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
