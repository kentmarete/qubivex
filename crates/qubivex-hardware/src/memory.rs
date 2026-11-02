//! RAM: machine total and available, the cgroup limit and room, process RSS.

use serde::{Deserialize, Serialize};

use qubivex_core::Bytes;

use crate::host::{Cmd, Host, OsFamily};
use crate::parse::{colon_map, u64_value};
use crate::{HardwareError, Probed};

/// The memory limit of the cgroup this process runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CgroupLimit {
    /// No cgroup limit applies (or the OS has no cgroups).
    None,
    /// The tightest limit from this cgroup up to the root.
    Limited(Bytes),
}

/// Static memory facts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Physical RAM of the machine.
    pub total: Probed<Bytes>,
    /// The cgroup memory limit. In a container this, not `total`, is the RAM there is.
    pub cgroup_limit: Probed<CgroupLimit>,
}

impl MemoryInfo {
    /// The RAM this process may use at most: the smaller of `total` and the cgroup limit.
    #[must_use]
    pub fn effective_total(&self) -> Probed<Bytes> {
        match (&self.total, &self.cgroup_limit) {
            (Probed::Known(t), Probed::Known(CgroupLimit::Limited(l))) => Probed::Known(*t.min(l)),
            (Probed::Known(t), Probed::Known(CgroupLimit::None)) => Probed::Known(*t),
            (Probed::Unknown(r), _) | (_, Probed::Unknown(r)) => Probed::unknown(r.clone()),
        }
    }
}

pub(crate) fn probe(host: &Host) -> MemoryInfo {
    match host.os() {
        OsFamily::Linux => MemoryInfo {
            total: meminfo_field(host, "MemTotal"),
            cgroup_limit: cgroup(host)
                .map(|c| c.map_or(CgroupLimit::None, |c| CgroupLimit::Limited(c.limit))),
        },
        OsFamily::MacOs => MemoryInfo {
            total: host
                .run(&Cmd::new("sysctl", &["-n", "hw.memsize"]).key("sysctl-memsize"))
                .and_then(|s| u64_value("sysctl hw.memsize", &s))
                .map(Bytes)
                .into(),
            cgroup_limit: Probed::Known(CgroupLimit::None),
        },
        OsFamily::Other => MemoryInfo {
            total: Probed::unknown("not probed on this OS"),
            cgroup_limit: Probed::unknown("not probed on this OS"),
        },
    }
}

/// RAM this process could still allocate: the machine's available memory, capped by
/// the cgroup's room.
pub(crate) fn available(host: &Host) -> Probed<Bytes> {
    match host.os() {
        OsFamily::Linux => {
            let machine = meminfo_field(host, "MemAvailable");
            match (machine, cgroup(host)) {
                (Probed::Known(m), Probed::Known(Some(c))) => Probed::Known(m.min(c.room)),
                (Probed::Known(m), Probed::Known(None)) => Probed::Known(m),
                (Probed::Unknown(r), _) | (_, Probed::Unknown(r)) => Probed::Unknown(r),
            }
        }
        OsFamily::MacOs => host
            .run(&Cmd::new("vm_stat", &[]))
            .and_then(|s| darwin_available(&s))
            .into(),
        OsFamily::Other => Probed::unknown("not probed on this OS"),
    }
}

/// Resident set size of this process.
pub(crate) fn process_rss(host: &Host) -> Probed<Bytes> {
    match host.os() {
        OsFamily::Linux => host
            .read("/proc/self/status")
            .and_then(|s| kib_field("/proc/self/status", &s, "VmRSS"))
            .into(),
        OsFamily::MacOs => {
            let pid = host.pid().to_string();
            host.run(&Cmd::new("ps", &["-o", "rss=", "-p", &pid]).key("ps-rss"))
                .and_then(|s| u64_value("ps rss", &s))
                .and_then(|k| kib("ps rss", k))
                .into()
        }
        OsFamily::Other => Probed::unknown("not probed on this OS"),
    }
}

fn kib(what: &str, k: u64) -> Result<Bytes, HardwareError> {
    k.checked_mul(1024)
        .map(Bytes)
        .ok_or_else(|| HardwareError::parse(what, "value overflows"))
}

/// A `Name:   123 kB` field of `/proc/meminfo` or `/proc/self/status`.
fn kib_field(what: &str, text: &str, key: &str) -> Result<Bytes, HardwareError> {
    let m = colon_map(text);
    let v = m
        .get(key)
        .ok_or_else(|| HardwareError::parse(what, format!("no {key}")))?;
    let n = v
        .strip_suffix("kB")
        .ok_or_else(|| HardwareError::parse(what, format!("{key} is not in kB: {v:?}")))?;
    kib(what, u64_value(what, n)?)
}

fn meminfo_field(host: &Host, key: &str) -> Probed<Bytes> {
    host.read("/proc/meminfo")
        .and_then(|s| kib_field("/proc/meminfo", &s, key))
        .into()
}

/// `vm_stat`: free + inactive + speculative + purgeable pages, Colibri's definition
/// of available memory on macOS (`setup_hw.py memory_from_darwin`).
pub(crate) fn darwin_available(vm_stat: &str) -> Result<Bytes, HardwareError> {
    const W: &str = "vm_stat";
    let page = vm_stat
        .split_once("page size of ")
        .and_then(|(_, r)| r.split_whitespace().next())
        .ok_or_else(|| HardwareError::parse(W, "no page size"))
        .and_then(|p| u64_value(W, p))?;
    let m = colon_map(vm_stat);
    let mut pages = 0u64;
    let mut found = 0;
    for key in [
        "Pages free",
        "Pages inactive",
        "Pages speculative",
        "Pages purgeable",
    ] {
        if let Some(v) = m.get(key) {
            pages = pages
                .checked_add(u64_value(W, v.trim_end_matches('.'))?)
                .ok_or_else(|| HardwareError::parse(W, "page count overflows"))?;
            found += 1;
        }
    }
    if found == 0 {
        return Err(HardwareError::parse(W, "no page counts"));
    }
    pages
        .checked_mul(page)
        .map(Bytes)
        .ok_or_else(|| HardwareError::parse(W, "byte count overflows"))
}

// ---- cgroups (port of Colibri compat.h compat_cgroup_room) -------------------

/// The tightest cgroup limit and the room left under it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CgroupRoom {
    pub(crate) limit: Bytes,
    /// Limit minus what the kernel cannot reclaim (usage minus file cache).
    pub(crate) room: Bytes,
}

/// v1 reports "unlimited" as a value near 2^63.
const V1_UNLIMITED: u64 = 1 << 60;
/// Depth limit for the walk to the root.
const MAX_DEPTH: usize = 64;

/// `Ok(None)` when no limit applies. `Unknown` when the cgroup is outside this
/// namespace (its ancestors are invisible, so a limit could exist that we cannot see).
pub(crate) fn cgroup(host: &Host) -> Probed<Option<CgroupRoom>> {
    let text = match host.read("/proc/self/cgroup") {
        Ok(t) => t,
        // No cgroup file: not in a cgroup hierarchy at all.
        Err(_) if !host.exists("/proc/self") => return Probed::Known(None),
        Err(e) => return Probed::unknown(e.to_string()),
    };
    let mut v2: Option<&str> = None;
    let mut v1: Option<&str> = None;
    for line in text.lines() {
        let mut it = line.splitn(3, ':');
        let (Some(h), Some(ctl), Some(path)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        if !path.starts_with('/') {
            continue;
        }
        if h == "0" && ctl.is_empty() {
            v2 = Some(path);
        }
        if ctl.split(',').any(|c| c == "memory") {
            v1 = Some(path);
        }
    }
    let outside = |p: &str| p.split('/').any(|seg| seg == "..");
    if v2.is_some_and(outside) || v1.is_some_and(outside) {
        return Probed::unknown("cgroup is outside this namespace; its limit cannot be read");
    }
    let mut acc = Acc::default();
    let seen_v2 = v2.is_some_and(|p| walk(host, "/sys/fs/cgroup", p, false, &mut acc));
    if !seen_v2 {
        if let Some(p) = v1 {
            walk(host, "/sys/fs/cgroup/memory", p, true, &mut acc);
        }
    }
    Probed::Known(acc.limit.map(|limit| CgroupRoom {
        limit: Bytes(limit),
        room: Bytes(acc.room.unwrap_or(0)),
    }))
}

#[derive(Default)]
struct Acc {
    limit: Option<u64>,
    room: Option<u64>,
}

/// Walks from the leaf to the mount root. True if any level had the memory controls.
fn walk(host: &Host, root: &str, member: &str, v1: bool, acc: &mut Acc) -> bool {
    let member = member.trim_end_matches('/');
    let mut dir = format!("{root}{member}");
    let mut seen = false;
    for _ in 0..MAX_DEPTH {
        seen |= pair(host, &dir, v1, acc);
        if dir.len() <= root.len() {
            break;
        }
        match dir.rfind('/') {
            Some(i) if i >= root.len() => dir.truncate(i),
            _ => break,
        }
    }
    seen
}

fn pair(host: &Host, dir: &str, v1: bool, acc: &mut Acc) -> bool {
    let (lf, uf) = if v1 {
        ("memory.limit_in_bytes", "memory.usage_in_bytes")
    } else {
        ("memory.max", "memory.current")
    };
    let (Ok(l), Ok(u)) = (
        host.read(&format!("{dir}/{lf}")),
        host.read(&format!("{dir}/{uf}")),
    ) else {
        return false;
    };
    let Ok(cur) = u64_value(uf, &u) else {
        return true;
    };
    let Ok(limit) = u64_value(lf, &l) else {
        return true; // v2 "max": no limit at this level
    };
    if v1 && limit >= V1_UNLIMITED {
        return true;
    }
    // File cache is reclaimable before the OOM killer acts.
    let (ki, ka) = if v1 {
        ("total_inactive_file", "total_active_file")
    } else {
        ("inactive_file", "active_file")
    };
    let (mut inactive, mut active) = (0u64, 0u64);
    if let Ok(stat) = host.read(&format!("{dir}/memory.stat")) {
        for line in stat.lines() {
            match line.split_once(' ') {
                Some((k, v)) if k == ki => inactive = u64_value(k, v).unwrap_or(0),
                Some((k, v)) if k == ka => active = u64_value(k, v).unwrap_or(0),
                _ => {}
            }
        }
    }
    let working = cur.saturating_sub(inactive).saturating_sub(active);
    let room = limit.saturating_sub(working);
    acc.limit = Some(acc.limit.map_or(limit, |x| x.min(limit)));
    acc.room = Some(acc.room.map_or(room, |x| x.min(room)));
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    const VM_STAT: &str = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
        Pages free:                                    29773.\n\
        Pages active:                                 322627.\n\
        Pages inactive:                               318257.\n\
        Pages speculative:                              4743.\n\
        Pages purgeable:                               11312.\n";

    #[test]
    fn darwin_available_counts_reclaimable_pages() {
        let b = darwin_available(VM_STAT).unwrap();
        assert_eq!(b, Bytes((29_773 + 318_257 + 4_743 + 11_312) * 16_384));
        assert!(darwin_available("garbage").is_err());
    }

    #[test]
    fn kib_fields() {
        let s = "MemTotal:       65536 kB\nMemAvailable:   1024 kB\nHugePages_Total: 0\n";
        assert_eq!(
            kib_field("m", s, "MemAvailable").unwrap(),
            Bytes(1024 * 1024)
        );
        assert!(kib_field("m", s, "HugePages_Total").is_err());
        assert!(kib_field("m", s, "Nope").is_err());
    }

    #[test]
    fn effective_total_takes_the_cgroup_limit() {
        let m = MemoryInfo {
            total: Probed::Known(Bytes(100)),
            cgroup_limit: Probed::Known(CgroupLimit::Limited(Bytes(40))),
        };
        assert_eq!(m.effective_total(), Probed::Known(Bytes(40)));
    }
}
