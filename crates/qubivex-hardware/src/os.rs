//! Operating system name, version and kernel.

use serde::{Deserialize, Serialize};

use crate::Probed;
use crate::host::{Cmd, Host, OsFamily};
use crate::parse::{colon_map, env_map};

/// The operating system.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatingSystem {
    /// `linux`, `macos` or `other`.
    pub family: String,
    /// Distribution or product name with version ("Ubuntu 24.04.1 LTS", "macOS 27.2").
    pub name: Probed<String>,
    /// Kernel release ("6.8.0-110-generic", "27.2.0").
    pub kernel: Probed<String>,
}

pub(crate) fn probe(host: &Host) -> OperatingSystem {
    match host.os() {
        OsFamily::Linux => OperatingSystem {
            family: "linux".to_owned(),
            name: host
                .read("/etc/os-release")
                .map(|s| {
                    let m = env_map(&s);
                    m.get("PRETTY_NAME")
                        .or_else(|| m.get("NAME"))
                        .cloned()
                        .unwrap_or_else(|| "Linux".to_owned())
                })
                .into(),
            kernel: host
                .read("/proc/sys/kernel/osrelease")
                .map(|s| s.trim().to_owned())
                .into(),
        },
        OsFamily::MacOs => OperatingSystem {
            family: "macos".to_owned(),
            name: host
                .run(&Cmd::new("sw_vers", &[]))
                .map(|s| {
                    let m = colon_map(&s);
                    let name = m.get("ProductName").map_or("macOS", String::as_str);
                    match (m.get("ProductVersion"), m.get("BuildVersion")) {
                        (Some(v), Some(b)) => format!("{name} {v} ({b})"),
                        (Some(v), None) => format!("{name} {v}"),
                        _ => name.to_owned(),
                    }
                })
                .into(),
            kernel: host
                .run(&Cmd::new("sysctl", &["-n", "kern.osrelease"]).key("sysctl-osrelease"))
                .map(|s| s.trim().to_owned())
                .into(),
        },
        OsFamily::Other => OperatingSystem {
            family: "other".to_owned(),
            name: Probed::unknown("not probed on this OS"),
            kernel: Probed::unknown("not probed on this OS"),
        },
    }
}
