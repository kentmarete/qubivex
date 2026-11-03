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
//! Slice 1 of Phase 8 provides the kernels and their parity tests.
//! [`MetalBackend`] implements [`ComputeBackend`] by copying host weights into a
//! shared buffer per call (correct, not fast), and [`MetalWeights`] keeps a matrix
//! resident on the device. Engine integration and zero-copy weights come next.
//!
//! On targets other than macOS the crate builds, and [`MetalBackend::new`] returns
//! [`MetalError::Unavailable`].

// FFI into Metal through objc2 (RUST_GUIDELINES §3: GPU backends may opt in). The
// unsafe code is confined to `device.rs`.
#![allow(unsafe_code)]

#[cfg(target_os = "macos")]
mod device;

use qubivex_compute::{ComputeBackend, ComputeError, WeightView};
use qubivex_core::{DType, GgmlType};
use thiserror::Error;

/// The kernel source, compiled at backend creation.
pub const KERNEL_SOURCE: &str = include_str!("kernels.metal");

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
const DTYPES: [DType; 6] = [
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

fn check_dtype(dtype: DType, op: &'static str) -> Result<usize, ComputeError> {
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

/// A weight matrix resident in a device buffer.
#[derive(Debug)]
pub struct MetalWeights {
    #[cfg(target_os = "macos")]
    buf: device::Buffer,
    dtype: DType,
    rows: usize,
    cols: usize,
    row_bytes: usize,
}

impl MetalWeights {
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

impl MetalBackend {
    /// The system's default Metal device, with every kernel compiled.
    ///
    /// # Errors
    /// [`MetalError::Unavailable`] without a Metal device (or off macOS), and
    /// [`MetalError::Compile`] or [`MetalError::Create`] if setup fails.
    pub fn new() -> Result<Self, MetalError> {
        #[cfg(target_os = "macos")]
        {
            let names = DTYPES.map(|d| kernel_name(d).unwrap_or_default());
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
        let (n, k) = (w.rows, w.cols);
        if xs.len() != s * k || ys.len() != s * n {
            return Err(ComputeError::Shape {
                op: "matmul",
                detail: format!(
                    "{s} x {k} into {s} x {n}, got {} and {}",
                    xs.len(),
                    ys.len()
                ),
            });
        }
        if s == 0 || n == 0 {
            return Ok(());
        }
        let p = check_dtype(w.dtype, "matmul")?;
        #[cfg(target_os = "macos")]
        {
            let params = device::Params::new(n, k, s, w.row_bytes)?;
            self.dev.matmul(p, &w.buf, xs, ys, params)?;
            Ok(())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = p;
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
