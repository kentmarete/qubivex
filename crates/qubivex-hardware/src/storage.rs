//! Mounted filesystems that can hold model files: mount, fs type, device kind, capacity.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use qubivex_core::Bytes;

use crate::host::{Cmd, Host, OsFamily};
use crate::parse::u64_value;
use crate::{HardwareError, Probed};

/// What backs a filesystem.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StorageKind {
    /// Solid-state (NVMe, SATA SSD).
    Ssd,
    /// Rotational disk.
    Hdd,
    /// A network or host-shared filesystem (NFS, SMB, 9p): expect high, variable latency.
    Network,
}

/// One mounted filesystem.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageInfo {
    /// Where it is mounted.
    pub mount_point: String,
    /// Filesystem type (`ext4`, `apfs`, `nfs4`, `overlay`, ...).
    pub filesystem: String,
    /// The mounted source (`/dev/nvme0n1p2`, `server:/export`).
    pub source: String,
    /// What backs it.
    pub kind: Probed<StorageKind>,
    /// Size of the filesystem.
    pub capacity: Probed<Bytes>,
    /// Space free for unprivileged users.
    pub available: Probed<Bytes>,
}

struct Mount {
    point: String,
    fstype: String,
    source: String,
    /// Linux `major:minor`.
    dev: Option<String>,
}

pub(crate) fn probe(host: &Host) -> Result<Vec<StorageInfo>, HardwareError> {
    let mounts = match host.os() {
        OsFamily::Linux => linux_mounts(&host.read("/proc/self/mountinfo")?)?,
        OsFamily::MacOs => mac_mounts(&host.run(&Cmd::new("mount", &[]))?),
        OsFamily::Other => return Ok(Vec::new()),
    };
    let df = host.run(&Cmd::new("df", &["-kP"])).map(|s| parse_df(&s));
    Ok(mounts
        .into_iter()
        .map(|m| {
            let (capacity, available) = match &df {
                Ok(map) => match map.get(&m.point) {
                    Some(&(c, a)) => (Probed::Known(c), Probed::Known(a)),
                    None => (
                        Probed::unknown("not listed by df"),
                        Probed::unknown("not listed by df"),
                    ),
                },
                Err(e) => (
                    Probed::unknown(e.to_string()),
                    Probed::unknown(e.to_string()),
                ),
            };
            StorageInfo {
                kind: kind(host, &m),
                mount_point: m.point,
                filesystem: m.fstype,
                source: m.source,
                capacity,
                available,
            }
        })
        .collect())
}

const NETWORK_FS: &[&str] = &[
    "nfs",
    "nfs4",
    "cifs",
    "smb3",
    "smbfs",
    "afpfs",
    "webdav",
    "9p",
    "fuse.sshfs",
    "ceph",
    "glusterfs",
    "fuse.rclone",
];

fn kind(host: &Host, m: &Mount) -> Probed<StorageKind> {
    if NETWORK_FS.contains(&m.fstype.as_str()) {
        return Probed::Known(StorageKind::Network);
    }
    match host.os() {
        OsFamily::Linux => match &m.dev {
            Some(dev) if m.source.starts_with("/dev/") => linux_rotational(host, dev),
            _ => Probed::unknown(format!("{} has no block device", m.fstype)),
        },
        OsFamily::MacOs => mac_solid_state(host, &m.point),
        OsFamily::Other => Probed::unknown("not probed on this OS"),
    }
}

// ---- Linux ------------------------------------------------------------------

/// Pseudo and memory filesystems: never where model files live.
const SKIP_FS: &[&str] = &[
    "proc",
    "sysfs",
    "cgroup",
    "cgroup2",
    "devtmpfs",
    "devpts",
    "tmpfs",
    "mqueue",
    "debugfs",
    "tracefs",
    "securityfs",
    "pstore",
    "bpf",
    "configfs",
    "fusectl",
    "hugetlbfs",
    "autofs",
    "binfmt_misc",
    "efivarfs",
    "nsfs",
    "rpc_pipefs",
    "ramfs",
    "squashfs",
    "fuse.gvfsd-fuse",
    "fuse.portal",
    "selinuxfs",
];
/// Mount points under these are system plumbing (a container's /etc/hosts bind mount).
const SKIP_PREFIX: &[&str] = &[
    "/proc",
    "/sys",
    "/dev",
    "/run",
    "/etc/",
    "/boot/efi",
    "/snap/",
];

fn linux_mounts(text: &str) -> Result<Vec<Mount>, HardwareError> {
    const W: &str = "/proc/self/mountinfo";
    let mut v = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let (pre, post) = line
            .split_once(" - ")
            .ok_or_else(|| HardwareError::parse(W, format!("no separator: {line:?}")))?;
        let pre: Vec<&str> = pre.split_whitespace().collect();
        let mut post = post.split_whitespace();
        let (Some(dev), Some(point), Some(fstype), Some(source)) =
            (pre.get(2), pre.get(4), post.next(), post.next())
        else {
            return Err(HardwareError::parse(W, format!("short line: {line:?}")));
        };
        let point = unescape(point);
        if SKIP_FS.contains(&fstype) || SKIP_PREFIX.iter().any(|p| point.starts_with(p)) {
            continue;
        }
        v.push(Mount {
            point,
            fstype: fstype.to_owned(),
            source: unescape(source),
            dev: Some((*dev).to_owned()),
        });
    }
    Ok(v)
}

/// mountinfo escapes space, tab, newline and backslash as `\ooo`.
fn unescape(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\\'
            && i + 3 < b.len()
            && b[i + 1..i + 4].iter().all(|c| (b'0'..=b'7').contains(c))
        {
            let v = b[i + 1..i + 4]
                .iter()
                .fold(0u32, |a, c| a * 8 + u32::from(c - b'0'));
            out.push(u8::try_from(v).unwrap_or(b'?'));
            i += 4;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn linux_rotational(host: &Host, dev: &str) -> Probed<StorageKind> {
    // A whole disk (or dm device) has queue/ itself; a partition's disk is its parent.
    let paths = [
        format!("/sys/dev/block/{dev}/queue/rotational"),
        format!("/sys/dev/block/{dev}/../queue/rotational"),
    ];
    for p in &paths {
        if let Ok(s) = host.read(p) {
            return match s.trim() {
                "0" => Probed::Known(StorageKind::Ssd),
                "1" => Probed::Known(StorageKind::Hdd),
                o => Probed::unknown(format!("rotational = {o:?}")),
            };
        }
    }
    Probed::unknown(format!("no queue/rotational for block device {dev}"))
}

// ---- macOS ------------------------------------------------------------------

fn mac_mounts(text: &str) -> Vec<Mount> {
    let mut v = Vec::new();
    for line in text.lines() {
        // "<source> on <mount point> (<fstype>, <option>, ...)"
        let Some((source, rest)) = line.split_once(" on ") else {
            continue;
        };
        let Some((point, opts)) = rest.rsplit_once(" (") else {
            continue;
        };
        let opts: Vec<&str> = opts.trim_end_matches(')').split(", ").collect();
        let Some((&fstype, flags)) = opts.split_first() else {
            continue;
        };
        let network = NETWORK_FS.contains(&fstype);
        let local = flags.contains(&"local");
        // Hidden system volumes (Preboot, VM, Recovery) are skipped; the Data volume holds
        // every user file (/Users is firmlinked there), so it stays.
        let hidden = flags.contains(&"nobrowse") && point != "/System/Volumes/Data";
        if fstype == "devfs" || fstype == "autofs" || hidden || !(local || network) {
            continue;
        }
        v.push(Mount {
            point: point.to_owned(),
            fstype: fstype.to_owned(),
            source: source.to_owned(),
            dev: None,
        });
    }
    v
}

fn mac_solid_state(host: &Host, point: &str) -> Probed<StorageKind> {
    let key = format!("diskutil-{}", point.replace('/', "_"));
    let out = match host.run(&Cmd::new("diskutil", &["info", "-plist", point]).key(key)) {
        Ok(o) => o,
        Err(e) => return Probed::unknown(e.to_string()),
    };
    // <key>SolidState</key> followed by <true/> or <false/>.
    let Some((_, after)) = out.split_once("<key>SolidState</key>") else {
        return Probed::unknown("diskutil reports no SolidState");
    };
    match after.trim_start() {
        s if s.starts_with("<true/>") => Probed::Known(StorageKind::Ssd),
        s if s.starts_with("<false/>") => Probed::Known(StorageKind::Hdd),
        _ => Probed::unknown("diskutil SolidState is not a boolean"),
    }
}

// ---- df -----------------------------------------------------------------------

/// `df -kP`: mount point → (capacity, available). Lines that do not parse are skipped.
fn parse_df(text: &str) -> BTreeMap<String, (Bytes, Bytes)> {
    let mut m = BTreeMap::new();
    for line in text.lines().skip(1) {
        let t: Vec<&str> = line.split_whitespace().collect();
        // Source (may contain spaces), 1024-blocks, Used, Available, Capacity%, Mount (may contain spaces).
        let Some(pct) = t
            .iter()
            .skip(4)
            .position(|s| s.ends_with('%'))
            .map(|i| i + 4)
        else {
            continue;
        };
        let (Ok(total), Ok(avail)) = (u64_value("df", t[pct - 3]), u64_value("df", t[pct - 1]))
        else {
            continue;
        };
        let (Some(total), Some(avail)) = (total.checked_mul(1024), avail.checked_mul(1024)) else {
            continue;
        };
        m.insert(t[pct + 1..].join(" "), (Bytes(total), Bytes(avail)));
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unescapes_mountinfo() {
        assert_eq!(unescape(r"/mnt/my\040models"), "/mnt/my models");
        assert_eq!(unescape(r"/a\\b"), r"/a\\b");
        assert_eq!(unescape(r"/x\04"), r"/x\04");
    }

    #[test]
    fn df_with_spaces() {
        let m = parse_df(
            "Filesystem 1024-blocks Used Available Capacity Mounted on\n\
             /dev/disk3s2 482797652 880 343727556 1% /Volumes/My Models\n\
             map auto_home 0 0 0 100% /System/Volumes/Data/home\n\
             bad line\n",
        );
        assert_eq!(
            m["/Volumes/My Models"],
            (Bytes(482_797_652 * 1024), Bytes(343_727_556 * 1024))
        );
        assert!(m.contains_key("/System/Volumes/Data/home"));
        assert_eq!(m.len(), 2);
    }
}
