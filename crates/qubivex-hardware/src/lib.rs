//! Hardware profile and resource snapshots (QUBIVEX_DESIGN §4.2).
//!
//! - [`HardwareProfile`]: static facts, probed once ([`probe`]): CPU and core tiers,
//!   RAM and the cgroup limit, GPUs per backend, storage, OS.
//! - [`ResourceSnapshot`]: dynamic, cheap facts sampled by the pressure monitor
//!   ([`SystemProbe`]): available RAM (capped by the cgroup's room) and process RSS.
//!
//! Every value is [`Probed::Known`] or [`Probed::Unknown`] with a reason, never a
//! silent zero. GPU probes and every other helper program run in a child process
//! with a timeout, so a broken driver cannot hang QUBIVEX. The crate has no `unsafe`
//! code: macOS facts come from `sysctl`, `vm_stat`, `mount`, `df`, `diskutil` and
//! `system_profiler`, as Colibri's `setup_hw.py` reads them.
//!
//! Probes read through a [`Host`], so they run unchanged against a recorded
//! fixture tree ([`Host::fixture`]).

mod cpu;
mod error;
mod gpu;
mod host;
mod memory;
mod os;
mod parse;
mod probed;
mod snapshot;
mod storage;

use serde::{Deserialize, Serialize};

pub use cpu::{CoreTier, CpuInfo};
pub use error::HardwareError;
pub use gpu::{BackendStatus, GpuDevice, GpuInfo};
pub use host::{Host, OsFamily};
pub use memory::{CgroupLimit, MemoryInfo};
pub use os::OperatingSystem;
pub use probed::Probed;
pub use snapshot::{DeviceMemory, ResourceProbe, ResourceSnapshot, SystemProbe};
pub use storage::{StorageInfo, StorageKind};

/// Static hardware facts, probed once and cached by the caller.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// The CPU.
    pub cpu: CpuInfo,
    /// RAM.
    pub memory: MemoryInfo,
    /// One entry per GPU backend, available or not.
    pub gpus: Vec<GpuInfo>,
    /// Mounted filesystems that can hold model files.
    pub storage: Probed<Vec<StorageInfo>>,
    /// The operating system.
    pub os: OperatingSystem,
}

/// Probes `host`. Never fails: whatever cannot be determined is [`Probed::Unknown`].
#[must_use]
pub fn probe(host: &Host) -> HardwareProfile {
    HardwareProfile {
        cpu: cpu::probe(host),
        memory: memory::probe(host),
        gpus: gpu::probe(host),
        storage: storage::probe(host).into(),
        os: os::probe(host),
    }
}
