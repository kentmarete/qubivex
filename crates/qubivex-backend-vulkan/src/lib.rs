//! The Vulkan compute backend (ADR-019): a [`DiscreteDevice`] (ADR-018) on any Vulkan
//! 1.1 driver.
//!
//! [`VulkanDevice`] copies weights into device-local memory ([`DiscreteDevice::upload`])
//! and runs each [`DiscreteDevice::matmul_many`] as one command buffer. Activations
//! stay in host memory at this interface: each submission copies its inputs into a
//! host-visible buffer and reads its outputs back.
//!
//! Every output has the CPU backend's bits: the shader (`shaders/matmul.comp`)
//! dequantises each weight with the CPU's operations (block-wise for the K-quants and
//! Q8_0) and sums with the CPU's 64-lane reduction shape, using shared memory rather
//! than subgroup operations so the result does not depend on the driver's subgroup
//! size. Every add, subtract and `fma` is `NoContraction`, so no multiply is fused
//! into what consumes it (ADR-019 amendment 1). On MoltenVK the backend turns Metal's
//! fast math off at instance creation (`MVK_CONFIG_FAST_MATH_ENABLED`). Apple GPUs
//! flush f32 subnormals to zero, as under Metal (ADR-011 §1).
//!
//! A device must compute `fma` with one rounding. [`VulkanDevice::new`] checks it with
//! one probe and refuses a device that splits it into a multiply and an add, as Mesa's
//! lavapipe does: its outputs would differ from the CPU's in the last bits.
//!
//! Supported dtypes: f32, f16, bf16, Q8_0, Q4_K and Q6_K, the same set as the CPU.
//!
//! The Vulkan loader is opened at run time, so the crate builds everywhere. Without
//! one, [`VulkanDevice::new`] returns [`ComputeError::Device`]. `QUBIVEX_VULKAN_LIBRARY`
//! names the loader explicitly.
//!
//! Up to 8 submissions are in flight at once, one per concurrent caller; each still
//! completes before its call returns.
//!
//! [`TransferStats`] counts each matmul submission's host time (preparing and
//! submitting, then blocked on its fence) and, with [`VulkanConfig::timestamps`], its
//! GPU time from timestamp queries: summed, and as the union of busy intervals.
//!
//! **The fused expert block** (audit §14–15, Step 3a). [`DiscreteDevice::gated_ffn_many`]
//! runs a layer's experts as one submission: one dispatch for every gate and up matrix
//! (`shaders/matmul_fused.comp`: `matmul.comp`'s body, each workgroup finding its
//! matrix in a table), SwiGLU on the device (`shaders/swiglu.comp`, built -O0), then one
//! dispatch for every down matrix, the intermediates staying in device memory. Fusing
//! changes which workgroup computes what, never the arithmetic, so every output keeps
//! the CPU's bits. A dispatch reads every matrix from one buffer, so the experts should
//! sit in an arena ([`DiscreteDevice::alloc`]); matrices in different buffers or of
//! different dtypes get separate dispatches in the same submission. Fusion pays when
//! each expert has few rows (decode) and not at prefill sizes, so it is chosen by shape:
//! [`VulkanConfig::fuse_rows`].
//!
//! Not built: activations kept on the device across submissions (whole layers on the
//! device), several queues, asynchronous copies, and token tiles (each activation row
//! reads the weights again).

// FFI into Vulkan through ash (RUST_GUIDELINES §3: GPU backends may opt in). The
// unsafe code is confined to `device.rs`.
#![allow(unsafe_code)]

mod device;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

pub use device::{PhysicalInfo, list};
#[cfg(feature = "spike")]
pub use device::{SpikeKernel, SpikeStep, SpikeTiming};
use qubivex_compute::ComputeError;
use qubivex_compute::discrete::{
    DeviceBuffer, DeviceGatedFfn, DeviceMatMul, DeviceRoutedMoe, DeviceUsage, DiscreteDevice,
    Guard, TransferStats,
};
use qubivex_core::DType;

pub(crate) const BACKEND: &str = "vulkan";

/// Submissions that can be in flight at once: one per concurrent caller, such as the
/// compute threads running a layer's experts.
const SUBMITTERS: usize = 8;

/// [`VulkanConfig::fuse_rows`]'s default: fuse expert blocks of up to 8 rows. In the
/// harness (audit §15.2) fusion saved 15% of GPU time at 1 row per expert and 3% at 4,
/// and was neutral or slightly slower at 10. End to end (audit §16) 8 beat 4 by 1% with
/// 8 concurrent sequences and equalled it below; above 8 nothing was measured to gain.
pub const FUSE_ROWS: usize = 8;

/// Which device a [`VulkanDevice`] opens, and how much of its memory it may use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VulkanConfig {
    /// The device's index in [`list`]. `None`: the first discrete GPU, else the first
    /// integrated one, else any device with a compute queue.
    pub device: Option<usize>,
    /// Bytes of device memory to allow. `None`: the largest device-local heap.
    pub capacity: Option<u64>,
    /// Measure each matmul submission's GPU time with timestamp queries (into
    /// [`TransferStats`]). Off by default: it adds a query reset and two timestamps to
    /// every command buffer. Ignored on a queue without timestamps.
    pub timestamps: bool,
    /// Run a layer's expert blocks as one fused submission when no expert has more
    /// than this many rows ([`DiscreteDevice::fuses_gated_ffn`]). `None`: [`FUSE_ROWS`];
    /// `Some(0)`: never (each expert's matmuls are submitted on their own, as before).
    pub fuse_rows: Option<usize>,
    /// Route MoE layers on the device (Step 3b, audit §17): the router, softmax top-k,
    /// the routed experts' fused block and the combine in one submission, for layers
    /// of at most [`VulkanConfig::fuse_rows`] tokens whose experts are all on the
    /// device. Off by default; Step 3a (routing on the host) is the fallback.
    pub route: bool,
    /// With [`VulkanConfig::route`], put a routed layer's prelude in the same
    /// submission (Step 3c, audit §18): the attention output projection, the residual
    /// and the FFN RMSNorm. Off by default; Step 3b is the fallback.
    pub route_o: bool,
}

/// A Vulkan GPU as a discrete device. See the crate documentation.
pub struct VulkanDevice {
    /// [`SUBMITTERS`] submitters: a caller takes a free one, so up to that many
    /// submissions are recorded, in flight and waited for at once.
    submitters: Vec<Mutex<device::Submitter>>,
    /// Where the next caller starts looking for a free submitter.
    next: AtomicUsize,
    counters: Arc<device::Counters>,
    capacity: u64,
    info: PhysicalInfo,
    /// Whether the driver built the fused pipelines.
    fused: bool,
    /// Fuse expert blocks of up to this many rows; 0 when off or not available.
    fuse_rows: usize,
    /// The largest arena a fused dispatch can address.
    max_arena: u64,
    /// Route MoE layers of up to this many tokens on the device; 0 when off or not
    /// available.
    route_rows: usize,
    /// Route them with their prelude (the O projection, residual and norm).
    route_o: bool,
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("info", &self.info)
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}

impl VulkanDevice {
    /// Opens the device `config` names.
    ///
    /// # Errors
    /// [`ComputeError::Device`] when there is no Vulkan loader, no device, or the
    /// driver fails to create the device or its pipelines.
    pub fn new(config: VulkanConfig) -> Result<Self, ComputeError> {
        let opened = device::Shared::open(&config)?;
        let timestamps = config.timestamps;
        let mut submitter = device::Submitter::new(&opened.shared, timestamps)?;
        fused_fma(&mut submitter, &opened.info)?;
        let mut submitters = vec![Mutex::new(submitter)];
        for _ in 1..SUBMITTERS {
            submitters.push(Mutex::new(device::Submitter::new(
                &opened.shared,
                timestamps,
            )?));
        }
        let heap = opened.info.device_local;
        let capacity = config.capacity.unwrap_or(heap);
        let fused = opened.shared.fused.is_some();
        let fuse_rows = if fused {
            config.fuse_rows.unwrap_or(FUSE_ROWS)
        } else {
            0
        };
        let route_rows = if config.route && opened.shared.routed.is_some() {
            fuse_rows
        } else {
            0
        };
        // Offsets inside a bound range are 32-bit in the shaders.
        let max_arena =
            u64::from(opened.shared.limits.max_storage_buffer_range).min(u64::from(u32::MAX)) / 256
                * 256;
        Ok(Self {
            submitters,
            next: AtomicUsize::new(0),
            counters: Arc::default(),
            capacity,
            info: opened.info,
            fused,
            fuse_rows,
            max_arena,
            route_rows,
            route_o: config.route_o && route_rows > 0,
        })
    }

    /// A free submitter if there is one, else the next in turn once it is free.
    fn submitter(&self) -> MutexGuard<'_, device::Submitter> {
        let n = self.submitters.len();
        let start = self.next.fetch_add(1, Ordering::Relaxed);
        for i in 0..n {
            match self.submitters[(start + i) % n].try_lock() {
                Ok(g) => return g,
                Err(TryLockError::Poisoned(p)) => return p.into_inner(),
                Err(TryLockError::WouldBlock) => {}
            }
        }
        device::lock(&self.submitters[start % n])
    }

    /// The physical device.
    #[must_use]
    pub const fn info(&self) -> &PhysicalInfo {
        &self.info
    }

    /// The most rows per expert the fused expert block is used for; 0 when it is off
    /// or the driver could not build it.
    #[must_use]
    pub const fn fuse_rows(&self) -> usize {
        self.fuse_rows
    }

    /// The SwiGLU shader alone, `g = silu(g) · u` in place: for the numerical
    /// conformance tests, which check it against [`qubivex_compute::math::swiglu`].
    ///
    /// # Errors
    /// [`ComputeError::Shape`] when the lengths differ; [`ComputeError::Device`] when
    /// the device has no fused pipelines or the driver fails.
    pub fn swiglu(&self, g: &mut [f32], u: &[f32]) -> Result<(), ComputeError> {
        self.submitter().swiglu(g, u)
    }

    /// The routing shader alone: `s` tokens' router logits (`s × experts`) to their top
    /// `k` ids and weights (`s × k` each), for the numerical conformance tests, which
    /// check it against [`qubivex_compute::math::route`].
    ///
    /// # Errors
    /// [`ComputeError::Shape`] for inconsistent lengths; [`ComputeError::Device`] when
    /// the device has no routing pipelines or the driver fails.
    pub fn route(
        &self,
        logits: &[f32],
        s: usize,
        k: usize,
        renormalise: bool,
        ids: &mut [u32],
        weights: &mut [f32],
    ) -> Result<(), ComputeError> {
        self.submitter()
            .route(logits, s, k, renormalise, ids, weights)
    }

    /// The RMSNorm scale shader alone: `y[i] = 1 / sqrt(a[i] / n + eps)`, for the
    /// numerical conformance tests, which check it against the CPU's arithmetic
    /// ([`qubivex_compute::math::rms_scale_of`]).
    ///
    /// # Errors
    /// [`ComputeError::Shape`] when the lengths differ or are zero;
    /// [`ComputeError::Device`] when the device has no routing pipelines or the driver
    /// fails.
    pub fn rms_scale(
        &self,
        a: &[f32],
        n: f32,
        eps: f32,
        y: &mut [f32],
    ) -> Result<(), ComputeError> {
        self.submitter().rms_scale(a, n, eps, y)
    }

    /// Experimental (feature `spike`, audit §19): builds `spv` as a pipeline on the
    /// fused layout (4 storage bindings, 16 bytes of push constants).
    ///
    /// # Errors
    /// [`ComputeError::Device`] when the driver fails or there is no fused layout.
    #[cfg(feature = "spike")]
    pub fn spike_kernel(&self, spv: &[u8]) -> Result<SpikeKernel, ComputeError> {
        self.submitter().spike_kernel(spv)
    }

    /// Experimental (feature `spike`): runs `steps` `repeat` times in one submission
    /// over four host buffers (bindings 0 to 3), copied in and back.
    ///
    /// # Errors
    /// [`ComputeError::Device`] when the driver fails.
    #[cfg(feature = "spike")]
    pub fn spike_run(
        &self,
        steps: &[SpikeStep<'_>],
        bufs: [&mut [f32]; 4],
        repeat: u32,
    ) -> Result<SpikeTiming, ComputeError> {
        self.submitter().spike_run(steps, bufs, repeat)
    }

    /// Whether submissions are timed on the GPU (asked for, and the queue supports it).
    #[must_use]
    pub fn timestamps(&self) -> bool {
        device::lock(&self.submitters[0]).timed()
    }
}

/// Checks that the device computes `fma` with one rounding, as the parity contract
/// needs. One output, `fma(1 + 2^-12, 1 + 2^-12, -1)` in lane 0, differs if the driver
/// splits it into a multiply and an add (lavapipe does: ADR-019). The parity tests are
/// the full check; this one stops such a device from passing silently at run time.
fn fused_fma(sub: &mut device::Submitter, info: &PhysicalInfo) -> Result<(), ComputeError> {
    let v = 1.0f32 + 2f32.powi(-12);
    // k = 65: element 0 (w = -1, x = 1) and element 64 (w = x = v) share lane 0.
    let mut w = vec![0f32; 65];
    let mut x = vec![0f32; 65];
    (w[0], x[0], w[64], x[64]) = (-1.0, 1.0, v, v);
    let host: Vec<u8> = w.iter().flat_map(|f| f.to_le_bytes()).collect();
    let scratch = Arc::default();
    let buf = sub.upload(&host, &scratch, u64::MAX, None)?;
    let dw = qubivex_compute::discrete::DeviceWeights::new(buf, 0, DType::F32, 1, 65)?;
    let mut y = [0f32];
    sub.matmul_many(
        &mut [DeviceMatMul {
            w: &dw,
            xs: &x,
            s: 1,
            ys: &mut y,
        }],
        &scratch,
    )?;
    let want = v.mul_add(v, -1.0);
    if y[0].to_bits() == want.to_bits() {
        return Ok(());
    }
    Err(ComputeError::Device {
        backend: BACKEND,
        detail: format!(
            "{} ({}) splits fma into a multiply and an add ({:e} where one rounding gives \
             {want:e}), so it cannot give the CPU's bits (ADR-019)",
            info.name, info.driver, y[0]
        ),
    })
}

impl DiscreteDevice for VulkanDevice {
    fn name(&self) -> &'static str {
        BACKEND
    }

    fn describe(&self) -> String {
        let i = &self.info;
        let driver = if i.driver.is_empty() {
            String::new()
        } else {
            format!(", {}", i.driver)
        };
        let experts = match (self.fuse_rows, self.route_rows) {
            (0, _) => "experts unfused".to_owned(),
            (n, 0) => format!("experts fused up to {n} rows"),
            (n, _) if self.route_o => format!(
                "experts fused and routed on the device with the O projection and FFN norm \
                 up to {n} rows"
            ),
            (n, _) => format!("experts fused and routed on the device up to {n} rows"),
        };
        format!(
            "Vulkan device {} ({}, {}{driver}), {} usable, {experts}",
            i.index,
            i.name,
            i.kind,
            qubivex_core::Bytes(self.capacity)
        )
    }

    fn supports(&self, dtype: DType) -> bool {
        device::dtype_index(dtype).is_some()
    }

    fn upload(&self, host: &[u8], guard: Option<Guard>) -> Result<DeviceBuffer, ComputeError> {
        self.submitter()
            .upload(host, &self.counters, self.capacity, guard)
    }

    fn matmul_many(&self, ops: &mut [DeviceMatMul<'_>]) -> Result<(), ComputeError> {
        self.submitter().matmul_many(ops, &self.counters)
    }

    fn fuses_gated_ffn(&self, blocks: usize, rows: usize) -> bool {
        blocks > 0 && rows > 0 && rows <= self.fuse_rows
    }

    /// One submission when the driver built the fused pipelines (whatever
    /// `fuse_rows` says: it only steers callers); otherwise the separate steps.
    fn gated_ffn_many(&self, ops: &mut [DeviceGatedFfn<'_>]) -> Result<(), ComputeError> {
        if !self.fused {
            return qubivex_compute::discrete::gated_ffn_steps(self, ops);
        }
        self.submitter().gated_ffn_many(ops, &self.counters)
    }

    fn alloc(&self, len: usize, guard: Option<Guard>) -> Result<DeviceBuffer, ComputeError> {
        self.submitter()
            .alloc(len, &self.counters, self.capacity, guard)
    }

    fn write(&self, buffer: &DeviceBuffer, offset: usize, host: &[u8]) -> Result<(), ComputeError> {
        self.submitter().write(buffer, offset, host, &self.counters)
    }

    fn max_arena(&self) -> u64 {
        self.max_arena
    }

    fn routes_moe(&self, experts: usize, k: usize, s: usize) -> bool {
        k > 0 && k <= experts && experts <= device::ROUTED_EXPERTS && s > 0 && s <= self.route_rows
    }

    fn routes_moe_prelude(&self, experts: usize, k: usize, s: usize) -> bool {
        self.route_o && self.routes_moe(experts, k, s)
    }

    /// One submission when the layer's experts sit in one buffer per matrix kind (the
    /// arenas); otherwise `Ok(false)`, and the caller routes on the host. A prelude is
    /// run only with [`VulkanConfig::route_o`].
    fn routed_moe(&self, op: &mut DeviceRoutedMoe<'_>) -> Result<bool, ComputeError> {
        if !self.routes_moe(op.experts.len(), op.k, op.s) || (op.prelude.is_some() && !self.route_o)
        {
            return Ok(false);
        }
        self.submitter().routed_moe(op, &self.counters)
    }

    fn usage(&self) -> DeviceUsage {
        self.counters.usage(self.capacity)
    }

    fn transfers(&self) -> TransferStats {
        self.counters.transfers()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[allow(clippy::unwrap_used)]
    fn spirv_is_current() {
        // Rebuild the shader with shaders/build.sh if its tools are installed, and
        // compare with the checked-in SPIR-V; skip otherwise.
        let have = |t: &str| {
            std::process::Command::new(t)
                .arg("--version")
                .output()
                .is_ok_and(|o| o.status.success())
        };
        if !["glslc", "spirv-dis", "spirv-as", "spirv-val"]
            .into_iter()
            .all(have)
        {
            eprintln!("glslc or SPIRV-Tools not found: skipped");
            return;
        }
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let out = std::env::temp_dir().join(format!("qubivex-spv-{}", std::process::id()));
        std::fs::create_dir_all(&out).unwrap();
        let status = std::process::Command::new("sh")
            .arg(dir.join("shaders/build.sh"))
            .arg(&out)
            .status()
            .unwrap();
        assert!(status.success(), "shaders/build.sh failed");
        let built = |name: &str| std::fs::read(out.join(name)).unwrap();
        let pairs = [
            ("matmul.spv", built("matmul.spv"), crate::device::MATMUL_SPV),
            (
                "matmul_fused.spv",
                built("matmul_fused.spv"),
                crate::device::MATMUL_FUSED_SPV,
            ),
            ("swiglu.spv", built("swiglu.spv"), crate::device::SWIGLU_SPV),
            ("route.spv", built("route.spv"), crate::device::ROUTE_SPV),
            (
                "combine.spv",
                built("combine.spv"),
                crate::device::COMBINE_SPV,
            ),
            (
                "resnorm.spv",
                built("resnorm.spv"),
                crate::device::RESNORM_SPV,
            ),
            (
                "rms_scale.spv",
                built("rms_scale.spv"),
                crate::device::RMS_SCALE_SPV,
            ),
        ];
        let _ = std::fs::remove_dir_all(&out);
        for (name, built, checked_in) in pairs {
            assert!(
                built == checked_in,
                "src/{name} is stale: run crates/qubivex-backend-vulkan/shaders/build.sh"
            );
        }
    }
}
