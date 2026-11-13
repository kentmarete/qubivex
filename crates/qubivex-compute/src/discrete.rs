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

/// One gated feed-forward block of a submission, as [`crate::GatedFfn`] with the
/// weights in device memory: `ys = down · swiglu(gate · xs, up · xs)`.
#[derive(Debug)]
pub struct DeviceGatedFfn<'a> {
    /// `ffn × cols` gate projection.
    pub gate: &'a DeviceWeights,
    /// `ffn × cols` up projection.
    pub up: &'a DeviceWeights,
    /// `rows × ffn` down projection.
    pub down: &'a DeviceWeights,
    /// `s × cols` activations.
    pub xs: &'a [f32],
    /// Activation rows.
    pub s: usize,
    /// `s × rows` outputs.
    pub ys: &'a mut [f32],
}

/// [`crate::MoePrelude`] with the O projection in device memory.
#[derive(Debug)]
pub struct DeviceMoePrelude<'a> {
    /// `cols × att_cols` output projection.
    pub o: &'a DeviceWeights,
    /// `s × att_cols` attention outputs.
    pub att: &'a [f32],
    /// `s × cols` residual stream: read, and written back with the projection added.
    pub hs: &'a mut [f32],
    /// `cols` norm weights.
    pub norm: &'a [f32],
    /// The norm's epsilon.
    pub eps: f64,
}

/// Positions per block of a device KV pool (audit §21).
pub const KV_BLOCK: usize = 256;

/// The f32 words of one KV pool block for `kv_out = kv_heads · head_dim`: the keys
/// dimension-major (`[kv_head][d][KV_BLOCK]`), then the values position-major
/// (`[KV_BLOCK][kv_head][d]`). Block `b` starts at word `b · kv_block_words(kv_out)`.
#[must_use]
pub const fn kv_block_words(kv_out: usize) -> usize {
    2 * KV_BLOCK * kv_out
}

/// [`crate::MoeAttention`] on a device (audit §21): the projections in device memory,
/// each sequence's keys and values in blocks of one KV pool. Token `t` is its own
/// sequence's token at position `pos[t]`, whose blocks are
/// `blocks[table_at[t]..table_at[t] + pos[t] / KV_BLOCK + 1]` in position order.
///
/// Before anything else the submission writes the `sync` rows into the pool: the host
/// cache's keys and values of positions the device did not hold. Then the new token's
/// key and value are stored at `pos[t]`, and attention reads positions `0..=pos[t]`.
#[derive(Debug)]
pub struct DeviceMoeAttention<'a> {
    /// `heads·head_dim × cols` query projection.
    pub q: &'a DeviceWeights,
    /// `kv_heads·head_dim × cols` key projection.
    pub k: &'a DeviceWeights,
    /// `kv_heads·head_dim × cols` value projection.
    pub v: &'a DeviceWeights,
    /// `s × cols` attention-normed activations.
    pub xs: &'a [f32],
    /// Query heads.
    pub heads: usize,
    /// KV heads.
    pub kv_heads: usize,
    /// Values per head.
    pub head_dim: usize,
    /// `z.clamp(-c, c)` on q, k and v.
    pub clip: Option<f32>,
    /// Full-width q and k norm weights.
    pub qk_norm: Option<(&'a [f32], &'a [f32])>,
    /// The norms' epsilon.
    pub eps: f64,
    /// `s × head_dim` RoPE rows.
    pub rope: &'a [f32],
    /// The KV pool (an arena from [`DiscreteDevice::alloc`]).
    pub pool: &'a DeviceBuffer,
    /// `s` positions.
    pub pos: &'a [u32],
    /// Block tables, concatenated.
    pub blocks: &'a [u32],
    /// `s` starts in `blocks`.
    pub table_at: &'a [u32],
    /// Rows to write first: each `2 · kv_out` words, the key row then the value row,
    /// as the host cache holds them (rotated, clipped).
    pub sync: &'a [f32],
    /// Each sync row's pool position: `block · KV_BLOCK + position in the block`.
    pub sync_to: &'a [u32],
    /// `s × kv_out` new keys, written.
    pub k_new: &'a mut [f32],
    /// `s × kv_out` new values, written.
    pub v_new: &'a mut [f32],
}

impl DeviceMoeAttention<'_> {
    /// `kv_heads · head_dim`.
    #[must_use]
    pub const fn kv_out(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// The pool word of key `(g, i)` at pool position `at` (`block · KV_BLOCK + slot`).
#[must_use]
pub const fn kv_key_word(kv_out: usize, d: usize, at: usize, g: usize, i: usize) -> usize {
    let (b, slot) = (at / KV_BLOCK, at % KV_BLOCK);
    b * kv_block_words(kv_out) + (g * d + i) * KV_BLOCK + slot
}

/// The pool word of value `(g, i)` at pool position `at`.
#[must_use]
pub const fn kv_value_word(kv_out: usize, d: usize, at: usize, g: usize, i: usize) -> usize {
    let (b, slot) = (at / KV_BLOCK, at % KV_BLOCK);
    b * kv_block_words(kv_out) + KV_BLOCK * kv_out + slot * kv_out + g * d + i
}

/// A mixture-of-experts layer as one submission, as [`crate::RoutedMoe`] with the
/// weights in device memory. `experts` holds every expert of the layer by id; the
/// caller keeps all of them on the device until the call returns.
#[derive(Debug)]
pub struct DeviceRoutedMoe<'a> {
    /// `experts × cols` router.
    pub router: &'a DeviceWeights,
    /// Every expert's gate, up and down, indexed by expert id.
    pub experts: &'a [[DeviceWeights; 3]],
    /// Experts per token.
    pub k: usize,
    /// Whether the chosen probabilities are renormalised to sum to 1.
    pub renormalise: bool,
    /// `s × cols` activations; empty with a prelude, which computes them.
    pub xs: &'a [f32],
    /// Tokens.
    pub s: usize,
    /// `s × rows` combined outputs.
    pub ys: &'a mut [f32],
    /// `s × k` chosen expert ids, in rank order per token.
    pub ids: &'a mut [u32],
    /// `s × k` routing weights.
    pub weights: &'a mut [f32],
    /// The O projection, residual and norm before the layer (Step 3c).
    pub prelude: Option<DeviceMoePrelude<'a>>,
    /// Decode attention before the prelude (audit §21); the prelude's `att` is then
    /// empty.
    pub attention: Option<DeviceMoeAttention<'a>>,
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

    /// Whether callers should run `blocks` gated feed-forward blocks of at most `rows`
    /// activation rows each as one [`DiscreteDevice::gated_ffn_many`]
    /// ([`crate::ComputeBackend::fuses_gated_ffn`]). The default is `false`.
    fn fuses_gated_ffn(&self, blocks: usize, rows: usize) -> bool {
        let _ = (blocks, rows);
        false
    }

    /// Independent gated feed-forward blocks. Each output has the bits of the separate
    /// steps: the default runs them as two [`DiscreteDevice::matmul_many`] submissions
    /// with [`crate::math::swiglu`] on the host between them. A device with a fused
    /// block runs them as one submission, the intermediates staying on the device.
    ///
    /// # Errors
    /// As [`DiscreteDevice::matmul_many`].
    fn gated_ffn_many(&self, ops: &mut [DeviceGatedFfn<'_>]) -> Result<(), ComputeError> {
        gated_ffn_steps(self, ops)
    }

    /// New device memory of `len` bytes, contents unspecified, filled later with
    /// [`DiscreteDevice::write`]: an arena that holds many copies at fixed offsets.
    /// `guard` lives as long as the allocation. The default refuses
    /// ([`ComputeError::Device`]); callers then upload each copy on its own.
    ///
    /// # Errors
    /// [`ComputeError::OutOfDeviceMemory`] when the device cannot hold `len` more
    /// bytes; [`ComputeError::Device`] when the device has no arenas or the driver
    /// fails.
    fn alloc(&self, len: usize, guard: Option<Guard>) -> Result<DeviceBuffer, ComputeError> {
        let _ = (len, guard);
        Err(ComputeError::Device {
            backend: self.name(),
            detail: "this device has no writable arenas".into(),
        })
    }

    /// Copies `host` into `buffer` at byte `offset` (an arena from
    /// [`DiscreteDevice::alloc`]), counted as an upload. The caller makes sure no
    /// submission reads those bytes meanwhile.
    ///
    /// # Errors
    /// [`ComputeError::Device`] for a buffer of another device, a range outside it, or
    /// a driver failure.
    fn write(&self, buffer: &DeviceBuffer, offset: usize, host: &[u8]) -> Result<(), ComputeError> {
        let _ = (buffer, offset, host);
        Err(ComputeError::Device {
            backend: self.name(),
            detail: "this device has no writable arenas".into(),
        })
    }

    /// The largest buffer one submission can address as a whole, in bytes: an arena
    /// should not be larger, so a fused block can read every copy in it.
    fn max_arena(&self) -> u64 {
        u64::MAX
    }

    /// Whether callers should offer a layer of `experts` experts, top `k`, over `s`
    /// tokens as one [`DiscreteDevice::routed_moe`]. The default is `false`.
    fn routes_moe(&self, experts: usize, k: usize, s: usize) -> bool {
        let _ = (experts, k, s);
        false
    }

    /// Whether callers should offer such a layer with its prelude (the O projection,
    /// residual and norm, Step 3c) as one [`DiscreteDevice::routed_moe`]. The default is
    /// `false`.
    fn routes_moe_prelude(&self, experts: usize, k: usize, s: usize) -> bool {
        let _ = (experts, k, s);
        false
    }

    /// Whether callers should offer such a layer with its prelude and decode attention
    /// ([`DeviceMoeAttention`]) as one [`DiscreteDevice::routed_moe`]. The default is
    /// `false`.
    fn routes_moe_attention(&self, experts: usize, k: usize, s: usize) -> bool {
        let _ = (experts, k, s);
        false
    }

    /// Bytes the KV pool for device attention may take, when configured; `None`: what
    /// the device budget leaves after the expert cache (audit §21).
    fn kv_pool(&self) -> Option<u64> {
        None
    }

    /// Runs the layer as one submission if this device can (routing on the device), and
    /// says whether it did; `Ok(false)` writes nothing. The default declines.
    ///
    /// # Errors
    /// As [`DiscreteDevice::matmul_many`].
    fn routed_moe(&self, op: &mut DeviceRoutedMoe<'_>) -> Result<bool, ComputeError> {
        let _ = op;
        Ok(false)
    }
}

/// [`DiscreteDevice::gated_ffn_many`] as separate steps (the default): one
/// [`DiscreteDevice::matmul_many`] for every gate and up, [`crate::math::swiglu`] on the
/// host, then one for every down.
///
/// # Errors
/// As [`DiscreteDevice::matmul_many`].
pub fn gated_ffn_steps(
    dev: &(impl DiscreteDevice + ?Sized),
    ops: &mut [DeviceGatedFfn<'_>],
) -> Result<(), ComputeError> {
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
        first.push(DeviceMatMul {
            w: op.gate,
            xs: op.xs,
            s: op.s,
            ys: g,
        });
        first.push(DeviceMatMul {
            w: op.up,
            xs: op.xs,
            s: op.s,
            ys: u,
        });
    }
    dev.matmul_many(&mut first)?;
    drop(first);
    for (g, u) in gs.iter_mut().zip(&us) {
        crate::math::swiglu(g, u);
    }
    let mut second: Vec<DeviceMatMul<'_>> = ops
        .iter_mut()
        .zip(&gs)
        .map(|(op, g)| DeviceMatMul {
            w: op.down,
            xs: g,
            s: op.s,
            ys: &mut *op.ys,
        })
        .collect();
    dev.matmul_many(&mut second)
}

/// [`DiscreteDevice::routed_moe`] as separate steps on `dev`: the router as one
/// [`DiscreteDevice::matmul_many`], routing on the host ([`crate::route_tokens`]), every
/// used expert's block as one [`DiscreteDevice::gated_ffn_many`], and the combine on the
/// host ([`crate::combine_ranks`]). Every output has the bits of
/// [`crate::routed_moe_steps`]; a simulated device uses it.
///
/// # Errors
/// [`ComputeError::Shape`] for inconsistent shapes; otherwise as the device's calls.
pub fn routed_moe_steps(
    dev: &(impl DiscreteDevice + ?Sized),
    op: &mut DeviceRoutedMoe<'_>,
) -> Result<(), ComputeError> {
    let (ne, cols, s, k) = (op.router.rows(), op.router.cols(), op.s, op.k);
    let rows = op.ys.len().checked_div(s).unwrap_or(0);
    let xs_len = if op.prelude.is_some() { 0 } else { s * cols };
    if op.experts.len() != ne
        || k == 0
        || k > ne
        || op.xs.len() != xs_len
        || op.prelude.as_ref().is_some_and(|p| {
            p.o.rows() != cols
                || p.att.len() != s * p.o.cols()
                || p.hs.len() != s * cols
                || p.norm.len() != cols
        })
        || op.ids.len() != s * k
        || op.weights.len() != s * k
    {
        return Err(ComputeError::Shape {
            op: "routed moe",
            detail: format!(
                "{ne}-row router, {} experts, top {k}, {s} tokens",
                op.experts.len()
            ),
        });
    }
    let normed = match op.prelude.as_mut() {
        Some(p) => {
            let mut o = vec![0f32; s * cols];
            dev.matmul_many(&mut [DeviceMatMul {
                w: p.o,
                xs: p.att,
                s,
                ys: &mut o,
            }])?;
            Some(crate::residual_norm(&o, p.hs, p.norm, p.eps))
        }
        None => None,
    };
    let xs: &[f32] = normed.as_deref().unwrap_or(op.xs);
    let mut logits = vec![0f32; s * ne];
    dev.matmul_many(&mut [DeviceMatMul {
        w: op.router,
        xs,
        s,
        ys: &mut logits,
    }])?;
    crate::route_tokens(&logits, s, k, op.renormalise, op.ids, op.weights);
    let users = crate::users_by_expert(op.ids, k);
    let xes: Vec<Vec<f32>> = users
        .iter()
        .map(|(_, u)| {
            u.iter()
                .flat_map(|&(t, _)| xs[t * cols..(t + 1) * cols].iter().copied())
                .collect()
        })
        .collect();
    let mut outs: Vec<Vec<f32>> = users
        .iter()
        .map(|(_, u)| vec![0f32; u.len() * rows])
        .collect();
    let mut blocks: Vec<DeviceGatedFfn<'_>> = users
        .iter()
        .zip(&xes)
        .zip(outs.iter_mut())
        .map(|(((e, u), xe), y)| {
            let [gate, up, down] = &op.experts[*e as usize];
            DeviceGatedFfn {
                gate,
                up,
                down,
                xs: xe,
                s: u.len(),
                ys: y,
            }
        })
        .collect();
    dev.gated_ffn_many(&mut blocks)?;
    drop(blocks);
    let mut contrib = vec![0f32; s * k * rows];
    for ((_, u), y) in users.iter().zip(&outs) {
        for (i, &(t, r)) in u.iter().enumerate() {
            let j = t * k + r;
            contrib[j * rows..(j + 1) * rows].copy_from_slice(&y[i * rows..(i + 1) * rows]);
        }
    }
    crate::combine_ranks(&contrib, op.weights, k, op.ys);
    Ok(())
}
