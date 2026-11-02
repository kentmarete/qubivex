//! `qubivex system`: the hardware profile and current resources.

use qubivex_core::Bytes;
use qubivex_hardware::{
    BackendStatus, CgroupLimit, GpuDevice, HardwareProfile, Probed, ResourceSnapshot, StorageKind,
};
use serde::Serialize;

/// What `qubivex system --json` prints.
#[derive(Serialize)]
pub(crate) struct Report<'a> {
    pub(crate) profile: &'a HardwareProfile,
    pub(crate) snapshot: &'a ResourceSnapshot,
}

fn or_unknown<T>(p: &Probed<T>, f: impl FnOnce(&T) -> String) -> String {
    match p {
        Probed::Known(v) => f(v),
        Probed::Unknown(r) => format!("unknown ({r})"),
    }
}

fn bytes(p: &Probed<Bytes>) -> String {
    or_unknown(p, ToString::to_string)
}

/// The human-readable report.
pub(crate) fn render(p: &HardwareProfile, s: &ResourceSnapshot) -> String {
    let mut o: Vec<String> = Vec::new();
    let c = &p.cpu;

    o.push(format!(
        "OS       {}, kernel {}",
        or_unknown(&p.os.name, Clone::clone),
        or_unknown(&p.os.kernel, Clone::clone)
    ));
    o.push(format!(
        "CPU      {} ({}, {})",
        or_unknown(&c.model, Clone::clone),
        or_unknown(&c.vendor, Clone::clone),
        c.arch
    ));
    o.push(format!(
        "  cores  {} logical, {} physical",
        or_unknown(&c.logical_cores, ToString::to_string),
        or_unknown(&c.physical_cores, ToString::to_string)
    ));
    o.push(format!(
        "  tiers  {}; compute team {}",
        or_unknown(&c.tiers, |t| t
            .iter()
            .map(|t| format!("{} {}", t.name, t.physical))
            .collect::<Vec<_>>()
            .join(", ")),
        or_unknown(&c.performance_cores(), ToString::to_string)
    ));
    o.push(format!(
        "  SIMD   {}",
        if c.simd.is_empty() {
            "none detected".to_owned()
        } else {
            c.simd.join(", ")
        }
    ));
    o.push(format!(
        "  NUMA   {}",
        or_unknown(&c.numa_nodes, |n| format!("{n} node(s)"))
    ));

    o.push(format!("Memory   {} total", bytes(&p.memory.total)));
    o.push(format!(
        "  cgroup {}",
        or_unknown(&p.memory.cgroup_limit, |l| match l {
            CgroupLimit::None => "no limit".to_owned(),
            CgroupLimit::Limited(b) => format!("limited to {b}"),
        })
    ));
    o.push(format!("  usable {}", bytes(&p.memory.effective_total())));
    o.push(format!("  free   {} now", bytes(&s.ram_available)));
    o.push(format!("  RSS    {} (this process)", bytes(&s.process_rss)));

    o.push("GPUs".to_owned());
    for g in &p.gpus {
        let name = g.backend.as_str();
        match &g.status {
            BackendStatus::Unavailable { reason } => {
                o.push(format!("  {name:<7}unavailable: {reason}"))
            }
            BackendStatus::Available { devices } if devices.is_empty() => {
                o.push(format!("  {name:<7}no devices"));
            }
            BackendStatus::Available { devices } => {
                for d in devices {
                    o.push(format!("  {name:<7}{}", gpu(d)));
                }
            }
        }
    }

    o.push("Storage".to_owned());
    match &p.storage {
        Probed::Unknown(r) => o.push(format!("  unknown ({r})")),
        Probed::Known(list) if list.is_empty() => o.push("  none found".to_owned()),
        Probed::Known(list) => {
            let w = list.iter().map(|s| s.mount_point.len()).max().unwrap_or(0);
            for st in list {
                // Column output: the reason is in --json.
                let kind = match st.kind {
                    Probed::Known(StorageKind::Ssd) => "ssd",
                    Probed::Known(StorageKind::Hdd) => "hdd",
                    Probed::Known(StorageKind::Network) => "network",
                    Probed::Unknown(_) => "?",
                };
                o.push(format!(
                    "  {:<w$}  {:<8} {:<8} {} total, {} free",
                    st.mount_point,
                    st.filesystem,
                    kind,
                    bytes(&st.capacity),
                    bytes(&st.available),
                ));
            }
        }
    }
    let mut out = o.join("\n");
    out.push('\n');
    out
}

fn gpu(d: &GpuDevice) -> String {
    let mut parts = Vec::new();
    if let Probed::Known(n) = d.compute_units {
        parts.push(format!("{n} cores"));
    }
    if d.unified_memory {
        parts.push("unified memory".to_owned());
    } else if let Probed::Known(b) = d.memory {
        parts.push(format!("{b} memory"));
    } else {
        parts.push("memory size unknown".to_owned());
    }
    if let Probed::Known(v) = &d.driver {
        parts.push(format!("driver {v}"));
    }
    format!("#{} {} ({})", d.ordinal, d.name, parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use qubivex_hardware::{Host, OsFamily, ResourceProbe, SystemProbe, probe};

    fn fixture(name: &str, os: OsFamily) -> String {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/qubivex-hardware/tests/fixtures")
            .join(name);
        let h = Host::fixture(root, os, "x86_64", &["avx2", "fma"]);
        let p = probe(&h);
        let s = SystemProbe::new(h).snapshot().unwrap();
        render(&p, &s)
    }

    #[test]
    fn renders_a_container_host() {
        let t = fixture("linux-i5-13500-cgroup2", OsFamily::Linux);
        assert!(
            t.contains("OS       Ubuntu 24.04.1 LTS, kernel 6.8.0-110-generic"),
            "{t}"
        );
        assert!(
            t.contains("tiers  cpu_core 6, cpu_atom 8; compute team 6"),
            "{t}"
        );
        assert!(t.contains("cgroup limited to 120.70 GiB"), "{t}");
        assert!(
            t.contains(
                "cuda   #0 NVIDIA RTX PRO 4000 Blackwell (23.89 GiB memory, driver 580.142)"
            ),
            "{t}"
        );
        assert!(t.contains("metal  unavailable: Metal is macOS only"), "{t}");
        assert!(t.contains("/models  ext4     ssd"), "{t}");
    }

    #[test]
    fn renders_unknowns_with_reasons() {
        let t = fixture("linux-arm-biglittle-outside-cgroup", OsFamily::Linux);
        assert!(
            t.contains("cgroup unknown (cgroup is outside this namespace"),
            "{t}"
        );
        assert!(t.contains("free   unknown ("), "{t}");
    }
}
