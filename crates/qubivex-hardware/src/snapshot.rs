//! Dynamic resource snapshots for the pressure monitor.

use std::time::Instant;

use serde::Serialize;

use qubivex_core::{Bytes, DeviceId};

use crate::host::Host;
use crate::{HardwareError, Probed, memory};

/// Free memory on one device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct DeviceMemory {
    /// The device.
    pub device: DeviceId,
    /// Memory it can still allocate.
    pub available: Bytes,
}

/// Resources available right now.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ResourceSnapshot {
    /// When the snapshot was taken.
    #[serde(skip)]
    pub taken_at: Instant,
    /// RAM this process could still allocate: the machine's available memory, capped
    /// by the cgroup's room (limit minus non-reclaimable usage).
    pub ram_available: Probed<Bytes>,
    /// Free memory per GPU. Filled by the GPU backends (Phase 8); empty until then.
    pub vram_available: Vec<DeviceMemory>,
    /// Resident set size of this process.
    pub process_rss: Probed<Bytes>,
}

/// A source of [`ResourceSnapshot`]s.
pub trait ResourceProbe: Send + Sync {
    /// Takes a snapshot.
    ///
    /// # Errors
    /// Implementations return an error only when nothing at all could be measured.
    /// Individual values that fail are [`Probed::Unknown`].
    fn snapshot(&self) -> Result<ResourceSnapshot, HardwareError>;
}

/// Snapshots of a [`Host`].
///
/// On Linux a snapshot reads only `/proc` and cgroup files. On macOS it runs `vm_stat`
/// and `ps` (a few milliseconds), because the crate has no FFI; sample accordingly.
#[derive(Clone, Debug)]
pub struct SystemProbe {
    host: Host,
}

impl SystemProbe {
    /// Snapshots of `host`.
    #[must_use]
    pub const fn new(host: Host) -> Self {
        Self { host }
    }
}

impl ResourceProbe for SystemProbe {
    fn snapshot(&self) -> Result<ResourceSnapshot, HardwareError> {
        Ok(ResourceSnapshot {
            taken_at: Instant::now(),
            ram_available: memory::available(&self.host),
            vram_available: Vec::new(),
            process_rss: memory::process_rss(&self.host),
        })
    }
}
