//! GPUs, per backend. Every probe runs in a child process with a timeout, so a broken
//! driver cannot hang QUBIVEX (Colibri `setup_hw.py`).

use serde::{Deserialize, Serialize};

use qubivex_core::{Bytes, DeviceKind};

use crate::host::{Cmd, GPU_TIMEOUT, Host, OsFamily};
use crate::{HardwareError, Probed};

/// One GPU as a backend sees it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuDevice {
    /// The backend's ordinal for this device.
    pub ordinal: u32,
    /// Device name.
    pub name: String,
    /// Dedicated memory. Unknown for unified-memory devices, whose memory is system RAM.
    pub memory: Probed<Bytes>,
    /// Whether the device shares system RAM (Apple Silicon, integrated GPUs).
    pub unified_memory: bool,
    /// Compute units (GPU cores, SMs) where the probe reports them.
    pub compute_units: Probed<u32>,
    /// Driver version where the probe reports it.
    pub driver: Probed<String>,
}

/// The result of one backend's probe.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "status")]
pub enum BackendStatus {
    /// The backend found these devices (possibly none).
    Available {
        /// The devices.
        devices: Vec<GpuDevice>,
    },
    /// The backend cannot be used, and why.
    Unavailable {
        /// Why, for example "nvidia-smi not found".
        reason: String,
    },
}

/// One backend's probe.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuInfo {
    /// The backend.
    pub backend: DeviceKind,
    /// What its probe found.
    pub status: BackendStatus,
}

pub(crate) fn probe(host: &Host) -> Vec<GpuInfo> {
    let cuda = if host.os() == OsFamily::MacOs {
        unavailable("CUDA is not supported on macOS")
    } else {
        status(nvidia(host))
    };
    let vulkan = status(vulkan(host));
    let metal = if host.os() == OsFamily::MacOs {
        status(metal(host))
    } else {
        unavailable("Metal is macOS only")
    };
    vec![
        GpuInfo {
            backend: DeviceKind::Cuda,
            status: cuda,
        },
        GpuInfo {
            backend: DeviceKind::Vulkan,
            status: vulkan,
        },
        GpuInfo {
            backend: DeviceKind::Metal,
            status: metal,
        },
    ]
}

fn unavailable(r: &str) -> BackendStatus {
    BackendStatus::Unavailable {
        reason: r.to_owned(),
    }
}

fn status(r: Result<Vec<GpuDevice>, HardwareError>) -> BackendStatus {
    match r {
        Ok(devices) => BackendStatus::Available { devices },
        Err(e) => BackendStatus::Unavailable {
            reason: e.to_string(),
        },
    }
}

// ---- CUDA via nvidia-smi (NVML) -----------------------------------------------

fn nvidia(host: &Host) -> Result<Vec<GpuDevice>, HardwareError> {
    let out = host.run(
        &Cmd::new(
            "nvidia-smi",
            &[
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
        )
        .timeout(GPU_TIMEOUT),
    )?;
    parse_nvidia(&out)
}

pub(crate) fn parse_nvidia(out: &str) -> Result<Vec<GpuDevice>, HardwareError> {
    const W: &str = "nvidia-smi output";
    out.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            let [idx, name, mib, driver] = f.as_slice() else {
                return Err(HardwareError::parse(
                    W,
                    format!("expected 4 fields: {line:?}"),
                ));
            };
            let ordinal = idx
                .parse()
                .map_err(|_| HardwareError::parse(W, format!("bad index {idx:?}")))?;
            let memory = mib
                .parse::<u64>()
                .ok()
                .and_then(|m| m.checked_mul(1 << 20))
                .map_or_else(
                    || Probed::unknown(format!("nvidia-smi memory {mib:?}")),
                    |b| Probed::Known(Bytes(b)),
                );
            Ok(GpuDevice {
                ordinal,
                name: (*name).to_owned(),
                memory,
                unified_memory: false,
                compute_units: Probed::unknown("not reported by nvidia-smi"),
                driver: Probed::Known((*driver).to_owned()),
            })
        })
        .collect()
}

// ---- Vulkan via vulkaninfo -----------------------------------------------------

fn vulkan(host: &Host) -> Result<Vec<GpuDevice>, HardwareError> {
    let out = host.run(&Cmd::new("vulkaninfo", &["--summary"]).timeout(GPU_TIMEOUT))?;
    Ok(parse_vulkan(&out))
}

/// `vulkaninfo --summary`: `GPUn:` blocks with `key = value` lines. CPU (software)
/// devices such as llvmpipe are skipped.
pub(crate) fn parse_vulkan(out: &str) -> Vec<GpuDevice> {
    let mut devs = Vec::new();
    let mut cur: Option<(u32, Vec<(String, String)>)> = None;
    let flush = |c: Option<(u32, Vec<(String, String)>)>, devs: &mut Vec<GpuDevice>| {
        let Some((ordinal, kv)) = c else { return };
        let get = |k: &str| kv.iter().find(|(a, _)| a == k).map(|(_, b)| b.clone());
        let ty = get("deviceType").unwrap_or_default();
        if ty.contains("CPU") {
            return;
        }
        devs.push(GpuDevice {
            ordinal,
            name: get("deviceName").unwrap_or_else(|| format!("GPU{ordinal}")),
            memory: Probed::unknown("vulkaninfo --summary reports no heap sizes"),
            unified_memory: ty.contains("INTEGRATED"),
            compute_units: Probed::unknown("not reported by vulkaninfo --summary"),
            driver: get("driverInfo")
                .or_else(|| get("driverVersion"))
                .map_or_else(|| Probed::unknown("no driver version"), Probed::Known),
        });
    };
    for line in out.lines() {
        let t = line.trim();
        if let Some(n) = t.strip_prefix("GPU").and_then(|r| r.strip_suffix(':'))
            && let Ok(i) = n.parse()
        {
            flush(cur.take(), &mut devs);
            cur = Some((i, Vec::new()));
            continue;
        }
        if let (Some((_, kv)), Some((k, v))) = (cur.as_mut(), t.split_once('=')) {
            kv.push((k.trim().to_owned(), v.trim().to_owned()));
        }
    }
    flush(cur, &mut devs);
    devs
}

// ---- Metal via system_profiler ---------------------------------------------------

fn metal(host: &Host) -> Result<Vec<GpuDevice>, HardwareError> {
    let out = host.run(
        &Cmd::new("system_profiler", &["SPDisplaysDataType", "-json"])
            .key("system_profiler-displays")
            .timeout(GPU_TIMEOUT),
    )?;
    parse_metal(&out)
}

pub(crate) fn parse_metal(out: &str) -> Result<Vec<GpuDevice>, HardwareError> {
    const W: &str = "system_profiler SPDisplaysDataType";
    let v: serde_json::Value =
        serde_json::from_str(out).map_err(|e| HardwareError::parse(W, e.to_string()))?;
    let list = v
        .get("SPDisplaysDataType")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| HardwareError::parse(W, "no SPDisplaysDataType array"))?;
    let mut devs = Vec::new();
    for (i, g) in list.iter().enumerate() {
        let s = |k: &str| g.get(k).and_then(serde_json::Value::as_str);
        // Only devices that support Metal.
        if s("spdisplays_mtlgpufamilysupport").is_none() && s("spdisplays_metal").is_none() {
            continue;
        }
        let name = s("sppci_model")
            .or_else(|| s("_name"))
            .unwrap_or("GPU")
            .to_owned();
        let builtin = s("sppci_bus") == Some("spdisplays_builtin");
        // Discrete GPUs on Intel Macs report "spdisplays_vram": "8 GB".
        let memory = match s("spdisplays_vram").or_else(|| s("spdisplays_vram_shared")) {
            Some(v) if !builtin => {
                parse_size(v).map_or_else(|| Probed::unknown(format!("vram {v:?}")), Probed::Known)
            }
            _ => Probed::unknown("unified memory: shares system RAM"),
        };
        devs.push(GpuDevice {
            ordinal: u32::try_from(devs.len()).unwrap_or(u32::MAX),
            name,
            memory,
            unified_memory: builtin,
            compute_units: s("sppci_cores").and_then(|c| c.parse().ok()).map_or_else(
                || Probed::unknown(format!("no core count for device {i}")),
                Probed::Known,
            ),
            driver: Probed::unknown("not reported by system_profiler"),
        });
    }
    Ok(devs)
}

/// "8 GB", "1536 MB".
fn parse_size(s: &str) -> Option<Bytes> {
    let (n, unit) = s.trim().split_once(' ')?;
    let n: u64 = n.parse().ok()?;
    let mul: u64 = match unit {
        "GB" => 1 << 30,
        "MB" => 1 << 20,
        _ => return None,
    };
    n.checked_mul(mul).map(Bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_rows() {
        let d = parse_nvidia("0, NVIDIA RTX PRO 4000 Blackwell, 24467, 580.142\n").unwrap();
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].name, "NVIDIA RTX PRO 4000 Blackwell");
        assert_eq!(d[0].memory, Probed::Known(Bytes(24_467 << 20)));
        assert!(parse_nvidia("garbage\n").is_err());
        assert!(parse_nvidia("").unwrap().is_empty());
    }

    #[test]
    fn sizes() {
        assert_eq!(parse_size("8 GB"), Some(Bytes(8 << 30)));
        assert_eq!(parse_size("1536 MB"), Some(Bytes(1536 << 20)));
        assert_eq!(parse_size("lots"), None);
    }
}
