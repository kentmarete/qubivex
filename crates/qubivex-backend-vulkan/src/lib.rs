//! The Vulkan compute backend (ADR-019): a [`DiscreteDevice`] (ADR-018) on any Vulkan
//! 1.1 driver.
//!
//! [`VulkanDevice`] copies weights into device-local memory ([`DiscreteDevice::upload`])
//! and runs each [`DiscreteDevice::matmul_many`] as one command buffer. Activations
//! stay in host memory at this interface: each submission copies its inputs into a
//! host-visible buffer and reads its outputs back.
//!
//! Every output has the CPU backend's bits: the shader (`shaders/matmul.comp`)
//! dequantises each weight with the CPU's operations and sums with the CPU's 64-lane
//! reduction shape, using shared memory rather than subgroup operations so the result
//! does not depend on the driver's subgroup size. On MoltenVK the backend turns Metal's
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
//! Not built: weights or activations kept on the device across submissions, several
//! queues, asynchronous copies, and kernels tuned for speed (each weight is
//! dequantised per element, and each activation row reads the weights again).

// FFI into Vulkan through ash (RUST_GUIDELINES §3: GPU backends may opt in). The
// unsafe code is confined to `device.rs`.
#![allow(unsafe_code)]

mod device;

use std::sync::{Arc, Mutex};

pub use device::{PhysicalInfo, list};
use qubivex_compute::ComputeError;
use qubivex_compute::discrete::{
    DeviceBuffer, DeviceMatMul, DeviceUsage, DiscreteDevice, Guard, TransferStats,
};
use qubivex_core::DType;

pub(crate) const BACKEND: &str = "vulkan";

/// Which device a [`VulkanDevice`] opens, and how much of its memory it may use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VulkanConfig {
    /// The device's index in [`list`]. `None`: the first discrete GPU, else the first
    /// integrated one, else any device with a compute queue.
    pub device: Option<usize>,
    /// Bytes of device memory to allow. `None`: the largest device-local heap.
    pub capacity: Option<u64>,
}

/// A Vulkan GPU as a discrete device. See the crate documentation.
pub struct VulkanDevice {
    submitter: Mutex<device::Submitter>,
    counters: Arc<device::Counters>,
    capacity: u64,
    info: PhysicalInfo,
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
        let mut submitter = device::Submitter::new(&opened.shared)?;
        fused_fma(&mut submitter, &opened.info)?;
        let heap = opened.info.device_local;
        let capacity = config.capacity.unwrap_or(heap);
        Ok(Self {
            submitter: Mutex::new(submitter),
            counters: Arc::default(),
            capacity,
            info: opened.info,
        })
    }

    /// The physical device.
    #[must_use]
    pub const fn info(&self) -> &PhysicalInfo {
        &self.info
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
        format!(
            "Vulkan device {} ({}, {}{driver}), {} usable",
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
        device::lock(&self.submitter).upload(host, &self.counters, self.capacity, guard)
    }

    fn matmul_many(&self, ops: &mut [DeviceMatMul<'_>]) -> Result<(), ComputeError> {
        device::lock(&self.submitter).matmul_many(ops, &self.counters)
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
        let built = std::fs::read(out.join("matmul.spv")).unwrap();
        let _ = std::fs::remove_dir_all(&out);
        assert!(
            built == crate::device::MATMUL_SPV,
            "src/matmul.spv is stale: run crates/qubivex-backend-vulkan/shaders/build.sh"
        );
    }
}
