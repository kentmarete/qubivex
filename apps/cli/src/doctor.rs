//! `qubivex doctor`: measurements that probing cannot provide.

use std::path::Path;

/// Runs the storage measurement and returns the report text.
#[cfg(unix)]
pub(crate) fn storage(path: &Path, size_mib: u64, json: bool) -> Result<String, String> {
    use qubivex_core::Bytes;
    use qubivex_hardware::{CacheBypass, StorageBenchOptions, measure_storage};

    let file_size = size_mib
        .checked_mul(1 << 20)
        .map(Bytes)
        .ok_or("--size-mib is too large")?;
    let opts = StorageBenchOptions {
        file_size,
        ..StorageBenchOptions::default()
    };
    let m = measure_storage(path, opts).map_err(|e| crate::chain(&e))?;
    if json {
        return crate::to_json(&m);
    }
    let gbs = |bps: f64| format!("{:.2} GB/s", bps / 1e9);
    let ms = |d: std::time::Duration| format!("{:.2} ms", d.as_secs_f64() * 1e3);
    let bypass = match m.bypass {
        CacheBypass::ODirect => "O_DIRECT",
        CacheBypass::FNocache => "F_NOCACHE",
        CacheBypass::None => "unsupported here; all passes buffered",
    };
    let mut o = vec![
        format!(
            "Storage  {} ({} scratch file)",
            m.path.display(),
            m.file_size
        ),
        format!("  bypass      {bypass}"),
        format!(
            "  sequential  {} uncached, {} buffered cold",
            gbs(m.seq_read_bps),
            gbs(m.buffered_seq_read_bps)
        ),
        format!(
            "  random 4MiB {} median, {} p90",
            ms(m.rand_read_4m_latency),
            ms(m.rand_read_4m_p90)
        ),
    ];
    if let Some(g) = m.direct_io_gain {
        o.push(format!("  direct I/O  {g:.2}x the buffered rate"));
    }
    let mut t = o.join("\n");
    t.push('\n');
    Ok(t)
}

#[cfg(not(unix))]
pub(crate) fn storage(_: &Path, _: u64, _: bool) -> Result<String, String> {
    Err("doctor --storage is supported on Linux and macOS only".to_owned())
}
