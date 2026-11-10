//! The Metal compute backend (ADR-007, ADR-011).
//!
//! Kernels reproduce the CPU backend's bits exactly: each weight has the value
//! `qubivex-backend-cpu`'s `formats::dequant_row` gives it, and each dot product uses
//! the CPU's fixed 64-lane reduction shape (see `kernels.metal`). Shaders are compiled
//! from source when the backend is created (no offline Metal toolchain is needed), with
//! fast math off.
//!
//! Supported dtypes: f32, f16, bf16, Q8_0, Q4_K and Q6_K, the same set as the CPU.
//!
//! [`MetalBackend`] implements [`ComputeBackend`] by copying host weights into a
//! shared buffer per call (correct, not fast). The fast path keeps weights on the
//! device: [`MetalBackend::map_host`] wraps a page-aligned host region in place
//! (unified memory, no copy; ADR-011 §4), [`MetalHost::weights`] names a matrix inside
//! it, and [`MetalBackend::matmul_many_resident`] runs a batch of matmuls in one
//! command buffer (ADR-011 §2).
//!
//! On targets other than macOS the crate builds, and [`MetalBackend::new`] returns
//! [`MetalError::Unavailable`].

// FFI into Metal through objc2 (RUST_GUIDELINES §3: GPU backends may opt in). The
// unsafe code is confined to `device.rs`; `program.rs` records through its safe API.
#![allow(unsafe_code)]

#[cfg(target_os = "macos")]
mod device;
mod program;

pub use program::{
    Attend, AttentionShape, Batch, DeviceVec, GROUPED_ENTRIES_PER_EXPERT, MAX_ROUTED_EXPERTS,
    Pending, RoutedInput, Routing, Span,
};

use std::sync::Arc;

use qubivex_compute::host::PageAligned;
use qubivex_compute::{ComputeBackend, ComputeError, MatMul, WeightView};
use qubivex_core::{DType, GgmlType};
use thiserror::Error;

/// The kernel source, compiled at backend creation.
pub const KERNEL_SOURCE: &str = include_str!("kernels.metal");

/// Activation rows per threadgroup of the token-tiled kernels; must equal `TILE` in
/// `kernels.metal`.
pub(crate) const TILE: usize = 8;

/// Errors from creating or driving the Metal backend.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MetalError {
    /// No Metal device on this machine or target.
    #[error("Metal unavailable: {0}")]
    Unavailable(String),
    /// The shader library failed to compile.
    #[error("Metal shader compilation failed: {0}")]
    Compile(String),
    /// A pipeline, buffer, queue or command buffer could not be created.
    #[error("Metal {what} creation failed{}", detail.as_deref().map(|d| format!(": {d}")).unwrap_or_default())]
    Create {
        /// What could not be created.
        what: &'static str,
        /// The driver's message, if any.
        detail: Option<String>,
    },
    /// A command buffer finished with an error.
    #[error("Metal command buffer failed: {0}")]
    Execution(String),
}

impl From<MetalError> for ComputeError {
    fn from(e: MetalError) -> Self {
        Self::Device {
            backend: "metal",
            detail: e.to_string(),
        }
    }
}

/// Whether the Metal kernels support `dtype`.
#[must_use]
pub const fn supported(dtype: DType) -> bool {
    kernel_name(dtype).is_some()
}

/// The kernel function for `dtype`.
const fn kernel_name(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::F32 => Some("matmul_f32"),
        DType::F16 => Some("matmul_f16"),
        DType::BF16 => Some("matmul_bf16"),
        DType::Ggml(GgmlType::Q8_0) => Some("matmul_q8_0"),
        DType::Ggml(GgmlType::Q4_K) => Some("matmul_q4_k"),
        DType::Ggml(GgmlType::Q6_K) => Some("matmul_q6_k"),
        _ => None,
    }
}

/// Every dtype with a kernel, in pipeline order.
pub(crate) const DTYPES: [DType; 6] = [
    DType::F32,
    DType::F16,
    DType::BF16,
    DType::Ggml(GgmlType::Q8_0),
    DType::Ggml(GgmlType::Q4_K),
    DType::Ggml(GgmlType::Q6_K),
];

fn pipeline_index(dtype: DType) -> Option<usize> {
    DTYPES.iter().position(|d| *d == dtype)
}

/// What the device reports about itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Device name, for example "Apple M3 Pro".
    pub name: String,
    /// Whether the GPU shares system RAM.
    pub unified_memory: bool,
    /// Metal's recommended working-set limit, in bytes.
    pub recommended_working_set: u64,
}

fn check_shapes(w: &WeightView<'_>, xs: &[f32], s: usize, ys: &[f32]) -> Result<(), ComputeError> {
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
    Ok(())
}

/// The Q4_K kernel loads each block's 16-byte header as one `uint4`, so its weights must
/// start 16-byte aligned (every Q4_K row is 144 · n bytes, so rows stay aligned).
pub(crate) const Q4_K_ALIGN: usize = 16;

/// Whether weights of `dtype` may start at byte `offset` of a device buffer.
pub(crate) const fn offset_ok(dtype: DType, offset: usize) -> bool {
    !matches!(dtype, DType::Ggml(GgmlType::Q4_K)) || offset.is_multiple_of(Q4_K_ALIGN)
}

pub(crate) fn check_dtype(dtype: DType, op: &'static str) -> Result<usize, ComputeError> {
    pipeline_index(dtype).ok_or(ComputeError::Unsupported {
        backend: "metal",
        op,
        dtype,
    })
}

/// The Metal backend: one device, one command queue and a pipeline per dtype.
#[derive(Debug)]
pub struct MetalBackend {
    #[cfg(target_os = "macos")]
    dev: device::Device,
}

/// A weight matrix resident in a device buffer (a copy, or a range of a wrapped host
/// region). It holds the buffer, and so the host memory, alive.
#[derive(Clone, Debug)]
pub struct MetalWeights {
    #[cfg(target_os = "macos")]
    buf: device::Buffer,
    offset: usize,
    dtype: DType,
    rows: usize,
    cols: usize,
    row_bytes: usize,
}

impl MetalWeights {
    /// Byte offset of the matrix in its device buffer (0 for an upload).
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
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
}

/// A host region wrapped as one device buffer without a copy (ADR-011 §4).
#[derive(Debug)]
pub struct MetalHost {
    #[cfg(target_os = "macos")]
    buf: device::Buffer,
    host: Arc<PageAligned>,
}

impl MetalHost {
    /// The bytes wrapped (the region's unpadded length).
    #[must_use]
    pub fn len(&self) -> usize {
        self.host.len()
    }

    /// Whether the region is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.host.is_empty()
    }

    /// `w` as resident weights, if its bytes lie inside the region; `None` otherwise.
    ///
    /// # Errors
    /// [`ComputeError::Unsupported`] for a dtype without a kernel.
    pub fn weights(&self, w: &WeightView<'_>) -> Result<Option<MetalWeights>, ComputeError> {
        check_dtype(w.dtype(), "matmul")?;
        let Some(offset) = self.host.offset_of(w.bytes()) else {
            return Ok(None);
        };
        if !offset_ok(w.dtype(), offset) {
            return Ok(None);
        }
        #[cfg(target_os = "macos")]
        {
            Ok(Some(MetalWeights {
                buf: self.buf.clone(),
                offset,
                dtype: w.dtype(),
                rows: w.rows(),
                cols: w.cols(),
                row_bytes: w.row_bytes(),
            }))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = offset;
            unreachable!("a MetalHost cannot be created off macOS")
        }
    }
}

/// One matmul of a resident batch (see [`MetalBackend::matmul_many_resident`]).
#[derive(Debug)]
pub struct ResidentMatMul<'a> {
    /// The weights.
    pub w: &'a MetalWeights,
    /// `s × cols` activations.
    pub xs: &'a [f32],
    /// Activation rows.
    pub s: usize,
    /// `s × rows` outputs.
    pub ys: &'a mut [f32],
}

impl MetalBackend {
    /// The system's default Metal device, with every kernel compiled.
    ///
    /// # Errors
    /// [`MetalError::Unavailable`] without a Metal device (or off macOS), and
    /// [`MetalError::Compile`] or [`MetalError::Create`] if setup fails.
    pub fn new() -> Result<Self, MetalError> {
        #[cfg(target_os = "macos")]
        {
            let mut names: Vec<&str> = DTYPES
                .iter()
                .map(|d| kernel_name(*d).unwrap_or_default())
                .collect();
            names.extend(program::OP_KERNELS);
            Ok(Self {
                dev: device::Device::new(KERNEL_SOURCE, &names)?,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(MetalError::Unavailable("Metal is macOS only".into()))
        }
    }

    /// What the device reports about itself.
    #[must_use]
    pub fn info(&self) -> DeviceInfo {
        #[cfg(target_os = "macos")]
        {
            self.dev.info()
        }
        #[cfg(not(target_os = "macos"))]
        {
            unreachable!("a MetalBackend cannot be created off macOS")
        }
    }

    /// Copies `w` into a device buffer.
    ///
    /// # Errors
    /// [`ComputeError::Unsupported`] for a dtype without a kernel, and
    /// [`ComputeError::Device`] if the buffer cannot be created.
    pub fn upload(&self, w: &WeightView<'_>) -> Result<MetalWeights, ComputeError> {
        check_dtype(w.dtype(), "upload")?;
        #[cfg(target_os = "macos")]
        {
            Ok(MetalWeights {
                buf: self.dev.buffer_from(w.bytes())?,
                offset: 0,
                dtype: w.dtype(),
                rows: w.rows(),
                cols: w.cols(),
                row_bytes: w.row_bytes(),
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            unreachable!("a MetalBackend cannot be created off macOS")
        }
    }

    /// Wraps `host` as one device buffer without copying it. The buffer, and every
    /// [`MetalWeights`] taken from it, hold `host` alive.
    ///
    /// # Errors
    /// [`ComputeError::Device`] if Metal refuses the region.
    pub fn map_host(&self, host: Arc<PageAligned>) -> Result<MetalHost, ComputeError> {
        #[cfg(target_os = "macos")]
        {
            Ok(MetalHost {
                buf: self.dev.wrap(Arc::clone(&host))?,
                host,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = host;
            unreachable!("a MetalBackend cannot be created off macOS")
        }
    }

    /// `ys[t] = W · xs[t]` with `W` resident on the device; the same contract as
    /// [`ComputeBackend::matmul`].
    ///
    /// # Errors
    /// [`ComputeError::Shape`] for mismatched lengths and [`ComputeError::Device`] if
    /// the GPU fails.
    pub fn matmul_resident(
        &self,
        w: &MetalWeights,
        xs: &[f32],
        s: usize,
        ys: &mut [f32],
    ) -> Result<(), ComputeError> {
        self.matmul_many_resident(&mut [ResidentMatMul { w, xs, s, ys }])
    }

    /// Runs independent resident matmuls in one command buffer and waits for it. Each
    /// output has the bits [`MetalBackend::matmul_resident`] gives it.
    ///
    /// # Errors
    /// [`ComputeError::Shape`] for mismatched lengths (nothing runs) and
    /// [`ComputeError::Device`] if the GPU fails.
    pub fn matmul_many_resident(&self, ops: &mut [ResidentMatMul<'_>]) -> Result<(), ComputeError> {
        let mut pipelines = Vec::with_capacity(ops.len());
        for op in ops.iter() {
            let w = op.w;
            let (n, k, s) = (w.rows, w.cols, op.s);
            if op.xs.len() != s * k || op.ys.len() != s * n {
                return Err(ComputeError::Shape {
                    op: "matmul",
                    detail: format!(
                        "{s} x {k} into {s} x {n}, got {} and {}",
                        op.xs.len(),
                        op.ys.len()
                    ),
                });
            }
            pipelines.push(check_dtype(w.dtype, "matmul")?);
        }
        #[cfg(target_os = "macos")]
        {
            let mut ds = Vec::with_capacity(ops.len());
            for (op, p) in ops.iter_mut().zip(pipelines) {
                if op.s == 0 || op.w.rows == 0 {
                    continue;
                }
                let w = op.w;
                let end = w
                    .rows
                    .checked_mul(w.row_bytes)
                    .and_then(|b| b.checked_add(w.offset));
                if end.is_none_or(|e| e > w.buf.len()) || !offset_ok(w.dtype, w.offset) {
                    return Err(ComputeError::Shape {
                        op: "matmul",
                        detail: format!(
                            "{} rows of {} bytes at {} exceed a {}-byte buffer or are misaligned",
                            w.rows,
                            w.row_bytes,
                            w.offset,
                            w.buf.len()
                        ),
                    });
                }
                ds.push(device::Dispatch {
                    pipeline: p,
                    w: &w.buf,
                    w_offset: w.offset,
                    xs: op.xs,
                    params: device::Params::new(w.rows, w.cols, op.s, w.row_bytes)?,
                    ys: &mut *op.ys,
                });
            }
            self.dev.run(&mut ds)?;
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = pipelines;
            unreachable!("a MetalBackend cannot be created off macOS")
        }
    }
}

impl ComputeBackend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn supports(&self, dtype: DType) -> bool {
        supported(dtype)
    }

    fn matmul(
        &self,
        w: &WeightView<'_>,
        xs: &[f32],
        s: usize,
        ys: &mut [f32],
    ) -> Result<(), ComputeError> {
        check_dtype(w.dtype(), "matmul")?;
        check_shapes(w, xs, s, ys)?;
        if s == 0 || w.rows() == 0 {
            return Ok(());
        }
        let resident = self.upload(w)?;
        self.matmul_resident(&resident, xs, s, ys)
    }

    /// Copies each op's weights to the device, then runs them in one command buffer.
    fn matmul_many(&self, ops: &mut [MatMul<'_>]) -> Result<(), ComputeError> {
        let mut resident = Vec::with_capacity(ops.len());
        for op in ops.iter() {
            check_dtype(op.w.dtype(), "matmul")?;
            check_shapes(&op.w, op.xs, op.s, op.ys)?;
            resident.push(if op.s == 0 || op.w.rows() == 0 {
                None
            } else {
                Some(self.upload(&op.w)?)
            });
        }
        let mut batch: Vec<ResidentMatMul<'_>> = ops
            .iter_mut()
            .zip(&resident)
            .filter_map(|(op, w)| {
                w.as_ref().map(|w| ResidentMatMul {
                    w,
                    xs: op.xs,
                    s: op.s,
                    ys: &mut *op.ys,
                })
            })
            .collect();
        self.matmul_many_resident(&mut batch)
    }

    /// Dequantisation of one row is a host operation (embedding lookup); it is not
    /// offloaded, so it returns [`ComputeError::Unsupported`]. The runtime uses the CPU
    /// backend for it.
    fn dequant_row(
        &self,
        w: &WeightView<'_>,
        _r: usize,
        _out: &mut [f32],
    ) -> Result<(), ComputeError> {
        Err(ComputeError::Unsupported {
            backend: "metal",
            op: "dequant_row",
            dtype: w.dtype(),
        })
    }
}
