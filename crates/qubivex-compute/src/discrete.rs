//! Devices with their own memory (ADR-018): CUDA and Vulkan GPUs, and the simulated
//! device that stands in for them on machines without one.
//!
//! A discrete device computes only on bytes in its own memory. Weights get there by an
//! explicit, counted copy ([`DiscreteDevice::upload`]) into a [`DeviceBuffer`], which
//! holds the device allocation until its last clone drops. An operation in flight owns
//! a clone of every buffer it reads (ADR-011 §3), so evicting a buffer from a cache
//! never frees memory a kernel is still reading.
//!
//! Activations stay in host memory at this interface: each matmul copies its inputs to
//! the device and its outputs back, and the device counts those bytes too
//! ([`TransferStats`]). Keeping whole layers on the device is a later step.
//!
//! Unified-memory devices (Metal) do not use this interface: they wrap host memory in
//! place and are planned as RAM (ADR-011 §4).

use std::any::Any;
use std::sync::Arc;

use qubivex_core::{DType, Shape};

use crate::{ComputeError, WeightView};

/// Something to keep alive exactly as long as a device allocation, such as the memory
/// ledger's charge for its bytes.
pub type Guard = Box<dyn Any + Send + Sync>;

struct BufferInner {
    // Declared first so the device memory is released before the guard (the charge
    // never shows less than what is allocated).
    state: Box<dyn Any + Send + Sync>,
    len: usize,
    _guard: Option<Guard>,
}

/// An allocation in a discrete device's memory. Clones share it; the memory is freed
/// when the last clone drops.
#[derive(Clone)]
pub struct DeviceBuffer {
    inner: Arc<BufferInner>,
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("len", &self.inner.len)
            .finish_non_exhaustive()
    }
}

impl DeviceBuffer {
    /// For backends: wraps the backend's `state` for an allocation of `len` bytes.
    /// `state` releases the device memory when dropped; `guard` is dropped after it.
    #[must_use]
    pub fn new(len: usize, state: impl Any + Send + Sync, guard: Option<Guard>) -> Self {
        Self {
            inner: Arc::new(BufferInner {
                state: Box::new(state),
                len,
                _guard: guard,
            }),
        }
    }

    /// Bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.len == 0
    }

    /// For backends: the state, if it is a `T`.
    #[must_use]
    pub fn state<T: Any>(&self) -> Option<&T> {
        self.inner.state.downcast_ref()
    }

    /// Whether another clone of this buffer exists (for example an operation in flight
    /// that reads it). Racy unless the caller controls where clones are made.
    #[must_use]
    pub fn shared(&self) -> bool {
        Arc::strong_count(&self.inner) > 1
    }

    /// Whether `self` and `other` are the same allocation.
    #[must_use]
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

/// A weight matrix inside a [`DeviceBuffer`], as [`WeightView`] is inside host bytes.
#[derive(Clone, Debug)]
pub struct DeviceWeights {
    buffer: DeviceBuffer,
    offset: usize,
    dtype: DType,
    rows: usize,
    cols: usize,
    len: usize,
}

impl DeviceWeights {
    /// The `rows × cols` matrix of `dtype` at `offset` in `buffer`.
    ///
    /// # Errors
    /// [`ComputeError::Shape`] if the matrix does not fit inside the buffer, and the
    /// errors of [`WeightView::new`] for an invalid shape.
    pub fn new(
        buffer: DeviceBuffer,
        offset: usize,
        dtype: DType,
        rows: usize,
        cols: usize,
    ) -> Result<Self, ComputeError> {
        let len = dtype.storage_bytes(&Shape::new(&[rows as u64, cols as u64])?)?;
        let len = usize::try_from(len).map_err(|_| ComputeError::Shape {
            op: "device weights",
            detail: format!("{len} bytes do not fit in usize"),
        })?;
        if offset.checked_add(len).is_none_or(|end| end > buffer.len()) {
            return Err(ComputeError::Shape {
                op: "device weights",
                detail: format!(
                    "{dtype} {rows} x {cols} ({len} bytes) at {offset} of a {}-byte buffer",
                    buffer.len()
                ),
            });
        }
        Ok(Self {
            buffer,
            offset,
            dtype,
            rows,
            cols,
            len,
        })
    }

    /// The device copy of `host`, which starts `offset` bytes into `buffer`.
    ///
    /// # Errors
    /// As [`DeviceWeights::new`].
    pub fn like(
        buffer: DeviceBuffer,
        offset: usize,
        host: &WeightView<'_>,
    ) -> Result<Self, ComputeError> {
        Self::new(buffer, offset, host.dtype(), host.rows(), host.cols())
    }

    /// The buffer.
    #[must_use]
    pub const fn buffer(&self) -> &DeviceBuffer {
        &self.buffer
    }

    /// Byte offset in the buffer.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the matrix has no bytes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
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
}

/// One matmul of a submission: `ys = W · xs` for `s` activation rows, with `W` in
/// device memory and `xs`, `ys` in host memory (as [`crate::ComputeBackend::matmul`]).
#[derive(Debug)]
pub struct DeviceMatMul<'a> {
    /// The weights.
    pub w: &'a DeviceWeights,
    /// `s × cols` activations.
    pub xs: &'a [f32],
    /// Activation rows.
    pub s: usize,
    /// `s × rows` outputs.
    pub ys: &'a mut [f32],
}

/// A device's memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceUsage {
    /// Bytes the device can allocate in all.
    pub capacity: u64,
    /// Bytes allocated now.
    pub used: u64,
    /// The most bytes allocated at once.
    pub peak: u64,
    /// Allocations refused for lack of memory.
    pub refused: u64,
}

/// Copies between host and device so far.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransferStats {
    /// Host-to-device weight copies.
    pub uploads: u64,
    /// Their bytes.
    pub upload_bytes: u64,
    /// Activation bytes copied to the device.
    pub activations_in: u64,
    /// Output bytes copied back.
    pub activations_out: u64,
    /// Submissions (calls of [`DiscreteDevice::matmul_many`]).
    pub submissions: u64,
    /// The time the copies took, in nanoseconds: measured on a real device, modelled
    /// from the configured bandwidth on a simulated one.
    pub transfer_ns: u64,
    /// Waits for a submission to complete (fence waits), counted for matmul
    /// submissions that reached the device. 0 on a device without them.
    pub fence_waits: u64,
    /// Host time preparing and making those submissions (descriptors, recording, the
    /// submit call), in nanoseconds; activation copies are in `transfer_ns`.
    pub submit_ns: u64,
    /// Host time blocked waiting for them to complete, in nanoseconds.
    pub wait_ns: u64,
    /// Submissions whose GPU time was measured (timestamp queries, when enabled).
    pub timed: u64,
    /// GPU time of the timed submissions, summed, in nanoseconds. Concurrent
    /// submissions overlap, so this can exceed wall time.
    pub gpu_ns: u64,
    /// The union of the timed submissions' GPU intervals, in nanoseconds: the time the
    /// GPU was busy with at least one of them.
    pub gpu_busy_ns: u64,
}

impl TransferStats {
    /// What happened between `before` and `self` (counters only grow).
    #[must_use]
    pub const fn since(&self, before: &Self) -> Self {
        Self {
            uploads: self.uploads.saturating_sub(before.uploads),
            upload_bytes: self.upload_bytes.saturating_sub(before.upload_bytes),
            activations_in: self.activations_in.saturating_sub(before.activations_in),
            activations_out: self.activations_out.saturating_sub(before.activations_out),
            submissions: self.submissions.saturating_sub(before.submissions),
            transfer_ns: self.transfer_ns.saturating_sub(before.transfer_ns),
            fence_waits: self.fence_waits.saturating_sub(before.fence_waits),
            submit_ns: self.submit_ns.saturating_sub(before.submit_ns),
            wait_ns: self.wait_ns.saturating_sub(before.wait_ns),
            timed: self.timed.saturating_sub(before.timed),
            gpu_ns: self.gpu_ns.saturating_sub(before.gpu_ns),
            gpu_busy_ns: self.gpu_busy_ns.saturating_sub(before.gpu_busy_ns),
        }
    }
}

/// A device with its own memory (ADR-018).
///
/// Every output has the bits [`crate::ComputeBackend::matmul`] on the CPU gives it,
/// under the parity contract of ADR-011 §1.
pub trait DiscreteDevice: Send + Sync {
    /// Short backend name (`sim`, `cuda`, `vulkan`).
    fn name(&self) -> &'static str;

    /// A description of the device for reports.
    fn describe(&self) -> String;

    /// Whether [`DiscreteDevice::matmul_many`] supports `dtype`.
    fn supports(&self, dtype: DType) -> bool;

    /// Copies `host` into new device memory. `guard` lives as long as the allocation.
    ///
    /// # Errors
    /// [`ComputeError::OutOfDeviceMemory`] when the device cannot hold `host.len()`
    /// more bytes; [`ComputeError::Device`] for a driver failure.
    fn upload(&self, host: &[u8], guard: Option<Guard>) -> Result<DeviceBuffer, ComputeError>;

    /// Independent matmuls (no output feeds another's input) as one submission. Each
    /// output has the CPU's bits.
    ///
    /// # Errors
    /// As [`crate::ComputeBackend::matmul`], and [`ComputeError::Device`] for a buffer
    /// of another device. On an error, outputs may be partly written.
    fn matmul_many(&self, ops: &mut [DeviceMatMul<'_>]) -> Result<(), ComputeError>;

    /// Memory now.
    fn usage(&self) -> DeviceUsage;

    /// Copies so far.
    fn transfers(&self) -> TransferStats;

    /// Whether callers should group a layer's independent matmuls into one
    /// [`DiscreteDevice::matmul_many`] ([`crate::ComputeBackend::groups_matmuls`]).
    /// The default is `false`: grouping saves host submission work but leaves only
    /// one submission in flight, which exposes every round trip that concurrent
    /// submissions hid. On MoltenVK that is slower end to end
    /// (docs/engineering/VULKAN_EXECUTION_AUDIT.md §13). Grouping never changes bits.
    fn groups_matmuls(&self) -> bool {
        false
    }
}
