//! Probes against recorded hosts (`tests/fixtures/<host>/`): `/proc` and `/sys` trees
//! for Linux, command outputs under `cmd/` for both families.

// Test helpers outside `#[test]` fns are not covered by clippy.toml's test exemption.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use qubivex_core::{Bytes, DeviceKind};
use qubivex_hardware::{
    BackendStatus, CgroupLimit, CoreTier, GpuInfo, HardwareProfile, Host, OsFamily, Probed,
    ResourceProbe, StorageKind, SystemProbe, probe,
};

fn host(name: &str, os: OsFamily, arch: &str) -> Host {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    Host::fixture(root, os, arch, &["neon"])
}

fn tier(name: &str, physical: u32, logical: u32) -> CoreTier {
    CoreTier {
        name: name.into(),
        physical,
        logical,
    }
}

fn gpu(p: &HardwareProfile, k: DeviceKind) -> &BackendStatus {
    &p.gpus
        .iter()
        .find(|g: &&GpuInfo| g.backend == k)
        .unwrap()
        .status
}

fn storage(p: &HardwareProfile) -> Vec<(String, String, Probed<StorageKind>, Probed<Bytes>)> {
    p.storage
        .known()
        .unwrap()
        .iter()
        .map(|s| {
            (
                s.mount_point.clone(),
                s.filesystem.clone(),
                s.kind.clone(),
                s.capacity.clone(),
            )
        })
        .collect()
}

const GIB: u64 = 1 << 30;

#[test]
fn hybrid_intel_in_a_cgroup2_container() {
    let h = host("linux-i5-13500-cgroup2", OsFamily::Linux, "x86_64");
    let p = probe(&h);

    assert_eq!(p.cpu.vendor, Probed::Known("GenuineIntel".into()));
    assert_eq!(
        p.cpu.model,
        Probed::Known("13th Gen Intel(R) Core(TM) i5-13500".into())
    );
    assert_eq!(p.cpu.logical_cores, Probed::Known(20));
    assert_eq!(p.cpu.physical_cores, Probed::Known(14));
    assert_eq!(
        p.cpu.tiers,
        Probed::Known(vec![tier("cpu_core", 6, 12), tier("cpu_atom", 8, 8)])
    );
    assert_eq!(p.cpu.performance_cores(), Probed::Known(6));
    assert_eq!(p.cpu.numa_nodes, Probed::Known(1));

    // The container's limit, not the machine's RAM, is what there is.
    assert_eq!(p.memory.total, Probed::Known(Bytes(263_856_212 * 1024)));
    assert_eq!(
        p.memory.cgroup_limit,
        Probed::Known(CgroupLimit::Limited(Bytes(129_601_297_408)))
    );
    assert_eq!(
        p.memory.effective_total(),
        Probed::Known(Bytes(129_601_297_408))
    );
    let s = SystemProbe::new(h).snapshot().unwrap();
    // room = limit − (current − inactive_file − active_file) = 120.7 GiB − 4 GiB
    assert_eq!(
        s.ram_available,
        Probed::Known(Bytes(129_601_297_408 - 4 * GIB))
    );
    assert_eq!(s.process_rss, Probed::Known(Bytes(524_288 * 1024)));

    let st = storage(&p);
    assert_eq!(st.len(), 3, "{st:?}"); // /etc/hosts bind mount, /proc, /dev, cgroup and shm skipped
    assert_eq!(st[0].0, "/");
    assert_eq!(st[0].1, "overlay");
    assert!(matches!(st[0].2, Probed::Unknown(_)));
    assert_eq!(
        (st[1].0.as_str(), st[1].2.clone()),
        ("/models", Probed::Known(StorageKind::Ssd))
    );
    assert_eq!(st[1].3, Probed::Known(Bytes(960_302_804 * 1024)));
    assert_eq!(
        (st[2].0.as_str(), st[2].2.clone()),
        ("/data", Probed::Known(StorageKind::Network))
    );

    let BackendStatus::Available { devices } = gpu(&p, DeviceKind::Cuda) else {
        panic!("cuda")
    };
    assert_eq!(devices.len(), 1);
    assert_eq!(devices[0].memory, Probed::Known(Bytes(24_467 << 20)));
    let BackendStatus::Available { devices } = gpu(&p, DeviceKind::Vulkan) else {
        panic!("vulkan")
    };
    assert_eq!(devices.len(), 1, "llvmpipe is skipped");
    assert_eq!(devices[0].name, "NVIDIA RTX PRO 4000 Blackwell");
    assert!(!devices[0].unified_memory);
    assert!(matches!(
        gpu(&p, DeviceKind::Metal),
        BackendStatus::Unavailable { .. }
    ));

    assert_eq!(p.os.name, Probed::Known("Ubuntu 24.04.1 LTS".into()));
    assert_eq!(p.os.kernel, Probed::Known("6.8.0-110-generic".into()));
}

#[test]
fn epyc_numa_cgroup1_hdd_root() {
    let h = host("linux-epyc-cgroup1", OsFamily::Linux, "x86_64");
    let p = probe(&h);

    assert_eq!(p.cpu.vendor, Probed::Known("AuthenticAMD".into()));
    assert_eq!(p.cpu.logical_cores, Probed::Known(32));
    assert_eq!(p.cpu.physical_cores, Probed::Known(16));
    assert_eq!(p.cpu.tiers, Probed::Known(vec![tier("all", 16, 32)]));
    assert_eq!(p.cpu.performance_cores(), Probed::Known(16));
    assert_eq!(p.cpu.numa_nodes, Probed::Known(2));

    // The leaf is unlimited; its parent user.slice is limited to 32 GiB.
    assert_eq!(
        p.memory.cgroup_limit,
        Probed::Known(CgroupLimit::Limited(Bytes(32 * GIB)))
    );
    let s = SystemProbe::new(h).snapshot().unwrap();
    assert_eq!(s.ram_available, Probed::Known(Bytes(26 * GIB)));

    let st = storage(&p);
    assert_eq!(st.len(), 2, "{st:?}");
    assert_eq!(
        (st[0].0.as_str(), st[0].2.clone()),
        ("/", Probed::Known(StorageKind::Hdd))
    );
    assert_eq!(
        (st[1].0.as_str(), st[1].2.clone()),
        ("/mnt/my models", Probed::Known(StorageKind::Ssd))
    );
    assert_eq!(st[1].3, Probed::Known(Bytes(3_750_000_000 * 1024)));

    assert_eq!(
        gpu(&p, DeviceKind::Cuda),
        &BackendStatus::Unavailable {
            reason: "nvidia-smi not found".into()
        }
    );
    assert_eq!(
        gpu(&p, DeviceKind::Vulkan),
        &BackendStatus::Unavailable {
            reason: "vulkaninfo not found".into()
        }
    );
    assert_eq!(
        p.os.name,
        Probed::Known("Debian GNU/Linux 12 (bookworm)".into())
    );
}

#[test]
fn arm_big_little_with_unreadable_cgroup_reports_unknowns() {
    let h = host(
        "linux-arm-biglittle-outside-cgroup",
        OsFamily::Linux,
        "aarch64",
    );
    let p = probe(&h);

    assert_eq!(p.cpu.arch, "aarch64");
    assert_eq!(p.cpu.vendor, Probed::Known("0x41".into()));
    assert!(matches!(p.cpu.model, Probed::Unknown(_)));
    assert_eq!(
        p.cpu.tiers,
        Probed::Known(vec![
            tier("capacity 1024", 4, 4),
            tier("capacity 446", 4, 4)
        ])
    );
    assert_eq!(p.cpu.performance_cores(), Probed::Known(4));
    assert_eq!(p.cpu.numa_nodes, Probed::Known(1));

    // Unknown is not "unlimited": a limit may exist above the namespace.
    assert!(matches!(p.memory.cgroup_limit, Probed::Unknown(ref r) if r.contains("outside")));
    assert!(matches!(p.memory.effective_total(), Probed::Unknown(_)));
    let s = SystemProbe::new(h).snapshot().unwrap();
    assert!(matches!(s.ram_available, Probed::Unknown(_)));

    let st = storage(&p);
    assert_eq!(st.len(), 1);
    assert!(matches!(st[0].2, Probed::Unknown(_)));
    assert!(matches!(st[0].3, Probed::Unknown(_)), "no df recorded");
    assert!(matches!(p.os.name, Probed::Unknown(_)));
}

#[test]
fn apple_m3_pro() {
    let h = host("macos-m3pro", OsFamily::MacOs, "aarch64");
    let p = probe(&h);

    assert_eq!(p.cpu.vendor, Probed::Known("Apple".into()));
    assert_eq!(p.cpu.model, Probed::Known("Apple M3 Pro".into()));
    assert_eq!(p.cpu.logical_cores, Probed::Known(11));
    assert_eq!(
        p.cpu.tiers,
        Probed::Known(vec![tier("Performance", 5, 5), tier("Efficiency", 6, 6)])
    );
    assert_eq!(p.cpu.performance_cores(), Probed::Known(5));
    assert_eq!(p.cpu.simd, vec!["neon".to_owned()]);

    assert_eq!(p.memory.total, Probed::Known(Bytes(19_327_352_832)));
    assert_eq!(
        p.memory.effective_total(),
        Probed::Known(Bytes(19_327_352_832))
    );
    let s = SystemProbe::new(h).snapshot().unwrap();
    assert!(s.ram_available.known().is_some_and(|b| b.get() > GIB));
    assert_eq!(s.process_rss, Probed::Known(Bytes(2960 * 1024)));

    let st = storage(&p);
    let points: Vec<&str> = st.iter().map(|s| s.0.as_str()).collect();
    assert_eq!(points, ["/", "/System/Volumes/Data", "/Volumes/Models"]);
    assert!(
        st.iter()
            .all(|s| s.1 == "apfs" && s.2 == Probed::Known(StorageKind::Ssd))
    );
    assert_eq!(st[0].3, Probed::Known(Bytes(482_797_652 * 1024)));

    assert!(matches!(
        gpu(&p, DeviceKind::Cuda),
        BackendStatus::Unavailable { .. }
    ));
    let BackendStatus::Available { devices } = gpu(&p, DeviceKind::Metal) else {
        panic!("metal")
    };
    assert_eq!(devices.len(), 1);
    assert_eq!(devices[0].name, "Apple M3 Pro");
    assert_eq!(devices[0].compute_units, Probed::Known(14));
    assert!(devices[0].unified_memory);
    assert!(matches!(devices[0].memory, Probed::Unknown(_)));

    assert_eq!(p.os.name, Probed::Known("macOS 27.2 (26B123)".into()));
    assert_eq!(p.os.kernel, Probed::Known("27.2.0".into()));
}

#[test]
fn profile_serializes_with_explicit_unknowns() {
    let p = probe(&host(
        "linux-arm-biglittle-outside-cgroup",
        OsFamily::Linux,
        "aarch64",
    ));
    let j = serde_json::to_value(&p).unwrap();
    assert!(j["memory"]["cgroup_limit"]["unknown"].is_string());
    assert_eq!(j["cpu"]["logical_cores"]["known"], 8);
    let back: HardwareProfile = serde_json::from_value(j).unwrap();
    assert_eq!(back, p);
}

/// The live probe must never panic and must find the basics on a supported OS.
#[test]
fn live_host_smoke() {
    let h = Host::live();
    let p = probe(&h);
    if h.os() != OsFamily::Other {
        assert!(
            p.cpu.logical_cores.known().is_some(),
            "{:?}",
            p.cpu.logical_cores
        );
        assert!(p.memory.total.known().is_some(), "{:?}", p.memory.total);
        let s = SystemProbe::new(h).snapshot().unwrap();
        assert!(s.process_rss.known().is_some(), "{:?}", s.process_rss);
    }
}
