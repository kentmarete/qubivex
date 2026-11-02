//! CPU: model, core tiers (P/E), SIMD set, NUMA nodes.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::host::{Cmd, Host, OsFamily};
use crate::parse::{colon_map, cpu_list};
use crate::{HardwareError, Probed};

/// A group of cores of one performance class, fastest group first.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreTier {
    /// The platform's name for the tier ("Performance", "Efficiency", "cpu_core", ...).
    pub name: String,
    /// Physical cores in the tier.
    pub physical: u32,
    /// Logical CPUs (hardware threads) in the tier.
    pub logical: u32,
}

/// The host CPU.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuInfo {
    /// Architecture, as `std::env::consts::ARCH` names it (`x86_64`, `aarch64`).
    pub arch: String,
    /// Vendor ("GenuineIntel", "AuthenticAMD", "Apple", ...).
    pub vendor: Probed<String>,
    /// Marketing model name.
    pub model: Probed<String>,
    /// Logical CPUs online.
    pub logical_cores: Probed<u32>,
    /// Physical cores (SMT siblings counted once).
    pub physical_cores: Probed<u32>,
    /// Core tiers, fastest first. One tier on a CPU with a single kind of core.
    pub tiers: Probed<Vec<CoreTier>>,
    /// SIMD features detected at run time on this CPU (for example `avx2`, `neon`).
    pub simd: Vec<String>,
    /// NUMA nodes online.
    pub numa_nodes: Probed<u32>,
}

impl CpuInfo {
    /// Physical cores outside the slowest tier: the size of a compute thread team.
    ///
    /// With a barrier after every matmul, the slowest thread sets the pace, so
    /// efficiency cores slow a team down (Colibri #707). On a CPU with one tier this is
    /// every physical core.
    #[must_use]
    pub fn performance_cores(&self) -> Probed<u32> {
        self.tiers.clone().and_then(|t| match t.as_slice() {
            [] => Probed::unknown("no core tiers"),
            [only] => Probed::Known(only.physical),
            [fast @ .., _slowest] => Probed::Known(fast.iter().map(|t| t.physical).sum()),
        })
    }
}

/// SIMD features of the running CPU, detected at run time.
pub(crate) fn detect_simd() -> Vec<String> {
    let mut v: Vec<&str> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        macro_rules! x86 {
            ($($f:tt),*) => { $( if std::arch::is_x86_feature_detected!($f) { v.push($f); } )* };
        }
        x86!(
            "sse4.1",
            "sse4.2",
            "avx",
            "avx2",
            "fma",
            "f16c",
            "avx512f",
            "avx512bw",
            "avx512vl",
            "avx512vnni",
            "avx512bf16",
            "avxvnni"
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        macro_rules! arm {
            ($($f:tt),*) => { $( if std::arch::is_aarch64_feature_detected!($f) { v.push($f); } )* };
        }
        arm!("neon", "fp16", "dotprod", "i8mm", "bf16", "sve", "sve2");
    }
    v.into_iter().map(str::to_owned).collect()
}

pub(crate) fn probe(host: &Host) -> CpuInfo {
    let mut info = CpuInfo {
        arch: host.arch().to_owned(),
        vendor: Probed::unknown("not probed on this OS"),
        model: Probed::unknown("not probed on this OS"),
        logical_cores: Probed::unknown("not probed on this OS"),
        physical_cores: Probed::unknown("not probed on this OS"),
        tiers: Probed::unknown("not probed on this OS"),
        simd: host.simd().to_vec(),
        numa_nodes: Probed::unknown("not probed on this OS"),
    };
    match host.os() {
        OsFamily::Linux => linux(host, &mut info),
        OsFamily::MacOs => macos(host, &mut info),
        OsFamily::Other => {}
    }
    info
}

// ---- macOS ------------------------------------------------------------------

const MAC_KEYS: &[&str] = &[
    "machdep.cpu.brand_string",
    "machdep.cpu.vendor",
    "hw.physicalcpu",
    "hw.logicalcpu",
    "hw.nperflevels",
    "hw.perflevel0.name",
    "hw.perflevel0.physicalcpu",
    "hw.perflevel0.logicalcpu",
    "hw.perflevel1.name",
    "hw.perflevel1.physicalcpu",
    "hw.perflevel1.logicalcpu",
    "hw.perflevel2.name",
    "hw.perflevel2.physicalcpu",
    "hw.perflevel2.logicalcpu",
    "hw.perflevel3.name",
    "hw.perflevel3.physicalcpu",
    "hw.perflevel3.logicalcpu",
];

fn macos(host: &Host, info: &mut CpuInfo) {
    // Unknown keys (an Intel Mac has no perflevel*) make sysctl exit 1 but print the rest.
    let out = match host.run(&Cmd::new("sysctl", MAC_KEYS).key("sysctl-cpu").lenient()) {
        Ok(o) => o,
        Err(e) => {
            let r = e.to_string();
            info.vendor = Probed::unknown(&r);
            info.model = Probed::unknown(&r);
            info.logical_cores = Probed::unknown(&r);
            info.physical_cores = Probed::unknown(&r);
            info.tiers = Probed::unknown(&r);
            info.numa_nodes = Probed::unknown(r);
            return;
        }
    };
    let m = colon_map(&out);
    let get = |k: &str| {
        m.get(k).filter(|v| !v.is_empty()).cloned().map_or_else(
            || Probed::unknown(format!("sysctl {k} missing")),
            Probed::Known,
        )
    };
    let get_u32 = |k: &str| {
        get(k).and_then(|v| Probed::from(crate::parse::u64_value(k, &v)).and_then(|n| to_u32(k, n)))
    };
    info.model = get("machdep.cpu.brand_string");
    info.vendor = match get("machdep.cpu.vendor") {
        Probed::Known(v) => Probed::Known(v),
        Probed::Unknown(_) => match &info.model {
            Probed::Known(b) if b.starts_with("Apple") => Probed::Known("Apple".to_owned()),
            _ => Probed::unknown("sysctl machdep.cpu.vendor missing"),
        },
    };
    info.logical_cores = get_u32("hw.logicalcpu");
    info.physical_cores = get_u32("hw.physicalcpu");
    info.tiers = match get_u32("hw.nperflevels") {
        Probed::Known(n) if (1..=4).contains(&n) => (0..n)
            .map(|i| {
                let p = |f: &str| format!("hw.perflevel{i}.{f}");
                Ok(CoreTier {
                    name: get(&p("name"))
                        .into_known()
                        .unwrap_or_else(|| format!("perflevel{i}")),
                    physical: get_u32(&p("physicalcpu"))
                        .into_known()
                        .ok_or_else(|| format!("sysctl {} missing", p("physicalcpu")))?,
                    logical: get_u32(&p("logicalcpu"))
                        .into_known()
                        .ok_or_else(|| format!("sysctl {} missing", p("logicalcpu")))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()
            .into(),
        // An Intel Mac: one kind of core.
        _ => match (&info.physical_cores, &info.logical_cores) {
            (Probed::Known(p), Probed::Known(l)) => Probed::Known(vec![CoreTier {
                name: "all".to_owned(),
                physical: *p,
                logical: *l,
            }]),
            _ => Probed::unknown("core counts unknown"),
        },
    };
    // macOS exposes no NUMA topology; Apple Silicon and Intel Macs are single-node.
    info.numa_nodes = Probed::Known(1);
}

fn to_u32(what: &str, n: u64) -> Probed<u32> {
    u32::try_from(n).map_or_else(
        |_| Probed::unknown(format!("{what} out of range: {n}")),
        Probed::Known,
    )
}

// ---- Linux ------------------------------------------------------------------

fn linux(host: &Host, info: &mut CpuInfo) {
    match host.read("/proc/cpuinfo") {
        Ok(s) => {
            let m = colon_map(&s);
            info.vendor = first_of(&m, &["vendor_id", "CPU implementer"], "vendor");
            info.model = first_of(&m, &["model name", "Model", "cpu model"], "model name");
        }
        Err(e) => {
            info.vendor = Probed::unknown(e.to_string());
            info.model = Probed::unknown(e.to_string());
        }
    }

    let online = host
        .read("/sys/devices/system/cpu/online")
        .and_then(|s| cpu_list("/sys/devices/system/cpu/online", &s));
    let online = match online {
        Ok(v) if !v.is_empty() => v,
        Ok(_) => {
            set_counts_unknown(info, "no CPUs online");
            return;
        }
        Err(e) => {
            set_counts_unknown(info, &e.to_string());
            return;
        }
    };
    info.logical_cores = to_u32("logical cores", online.len() as u64);

    // A physical core is one distinct thread_siblings_list.
    let mut siblings: BTreeMap<u32, String> = BTreeMap::new();
    for &c in &online {
        let p = format!("/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list");
        match host.read(&p) {
            Ok(s) => {
                siblings.insert(c, s.trim().to_owned());
            }
            Err(e) => {
                info.physical_cores = Probed::unknown(e.to_string());
                info.tiers = Probed::unknown(e.to_string());
                info.numa_nodes = numa(host);
                return;
            }
        }
    }
    let physical = |cpus: &[u32]| -> u32 {
        let set: BTreeSet<&str> = cpus
            .iter()
            .filter_map(|c| siblings.get(c).map(String::as_str))
            .collect();
        u32::try_from(set.len()).unwrap_or(u32::MAX)
    };
    let all_physical = physical(&online);
    info.physical_cores = Probed::Known(all_physical);
    let one_tier = || {
        Probed::Known(vec![CoreTier {
            name: "all".to_owned(),
            physical: all_physical,
            logical: u32::try_from(online.len()).unwrap_or(u32::MAX),
        }])
    };
    info.tiers = hybrid_x86(host, &online, &physical)
        .or_else(|| capacity_tiers(host, &online, &physical))
        .unwrap_or_else(one_tier);
    info.numa_nodes = numa(host);
}

fn set_counts_unknown(info: &mut CpuInfo, r: &str) {
    info.logical_cores = Probed::unknown(r);
    info.physical_cores = Probed::unknown(r);
    info.tiers = Probed::unknown(r);
}

fn first_of(m: &BTreeMap<String, String>, keys: &[&str], what: &str) -> Probed<String> {
    keys.iter()
        .find_map(|k| m.get(*k).filter(|v| !v.is_empty()).cloned())
        .map_or_else(
            || Probed::unknown(format!("/proc/cpuinfo has no {what}")),
            Probed::Known,
        )
}

/// A hybrid Intel CPU registers two core PMUs: `cpu_core` (P) and `cpu_atom` (E).
fn hybrid_x86(
    host: &Host,
    online: &[u32],
    physical: &dyn Fn(&[u32]) -> u32,
) -> Option<Probed<Vec<CoreTier>>> {
    let p = host.read("/sys/devices/cpu_core/cpus").ok()?;
    let e = host.read("/sys/devices/cpu_atom/cpus").ok()?;
    let tier = |name: &str, list: &str| -> Result<Option<CoreTier>, HardwareError> {
        let cpus: Vec<u32> = cpu_list(name, list)?
            .into_iter()
            .filter(|c| online.contains(c))
            .collect();
        Ok((!cpus.is_empty()).then(|| CoreTier {
            name: name.to_owned(),
            physical: physical(&cpus),
            logical: u32::try_from(cpus.len()).unwrap_or(u32::MAX),
        }))
    };
    Some(
        match (tier("cpu_core", &p), tier("cpu_atom", &e)) {
            (Ok(a), Ok(b)) => Ok(a.into_iter().chain(b).collect::<Vec<_>>()),
            (Err(x), _) | (_, Err(x)) => Err(x),
        }
        .into(),
    )
}

/// big.LITTLE and similar: `cpu_capacity` differs between cores. Faster tiers first.
fn capacity_tiers(
    host: &Host,
    online: &[u32],
    physical: &dyn Fn(&[u32]) -> u32,
) -> Option<Probed<Vec<CoreTier>>> {
    let mut by_cap: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
    for &c in online {
        let p = format!("/sys/devices/system/cpu/cpu{c}/cpu_capacity");
        let cap = crate::parse::u64_value(&p, &host.read(&p).ok()?).ok()?;
        by_cap.entry(cap).or_default().push(c);
    }
    if by_cap.len() < 2 {
        return None;
    }
    Some(Probed::Known(
        by_cap
            .iter()
            .rev()
            .map(|(cap, cpus)| CoreTier {
                name: format!("capacity {cap}"),
                physical: physical(cpus),
                logical: u32::try_from(cpus.len()).unwrap_or(u32::MAX),
            })
            .collect(),
    ))
}

fn numa(host: &Host) -> Probed<u32> {
    const P: &str = "/sys/devices/system/node/online";
    if !host.exists("/sys/devices/system/node") {
        // A kernel built without NUMA support has one node.
        return Probed::Known(1);
    }
    host.read(P)
        .and_then(|s| cpu_list(P, &s))
        .map(|v| u32::try_from(v.len()).unwrap_or(u32::MAX))
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiers(t: &[(u32, u32)]) -> CpuInfo {
        CpuInfo {
            arch: "x".into(),
            vendor: Probed::unknown(""),
            model: Probed::unknown(""),
            logical_cores: Probed::unknown(""),
            physical_cores: Probed::unknown(""),
            tiers: Probed::Known(
                t.iter()
                    .map(|&(physical, logical)| CoreTier {
                        name: "t".into(),
                        physical,
                        logical,
                    })
                    .collect(),
            ),
            simd: vec![],
            numa_nodes: Probed::Known(1),
        }
    }

    #[test]
    fn performance_cores_drop_the_slowest_tier() {
        assert_eq!(tiers(&[(8, 16)]).performance_cores(), Probed::Known(8));
        assert_eq!(
            tiers(&[(5, 5), (6, 6)]).performance_cores(),
            Probed::Known(5)
        );
        // Three tiers (2 super, 4 performance, 6 efficiency): 6, not perflevel0's 2.
        assert_eq!(
            tiers(&[(2, 2), (4, 4), (6, 6)]).performance_cores(),
            Probed::Known(6)
        );
    }

    #[test]
    fn live_simd_is_consistent_with_arch() {
        let s = detect_simd();
        if cfg!(target_arch = "aarch64") {
            assert!(s.iter().any(|f| f == "neon"));
        }
    }
}
