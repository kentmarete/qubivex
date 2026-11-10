//! `qubivex doctor --storage`: read throughput and latency of one filesystem.
//!
//! Writes a scratch file of incompressible data, then measures, in this order:
//! 1. a sequential read that bypasses the page cache (`O_DIRECT` on Linux,
//!    `F_NOCACHE` on macOS), in 8 MiB requests: the device's streaming rate;
//! 2. random 4 MiB reads, also uncached: what one expert load costs;
//! 3. a buffered sequential read after evicting the file (Linux `POSIX_FADV_DONTNEED`;
//!    on macOS the file was written with `F_NOCACHE`): the rate without direct I/O.
//!
//! Uncached reads come first because a buffered read would fill the cache, and an
//! `F_NOCACHE` read on macOS may still be served from pages already cached.
//! The scratch file is removed on every exit path.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use qubivex_core::Bytes;

use crate::HardwareError;

/// Request size of the sequential passes.
const SEQ_BLOCK: usize = 8 << 20;
/// Random-read size: about one expert's weights.
const RAND_BLOCK: usize = 4 << 20;
/// Buffer and offset alignment for direct I/O (covers 512 B and 4 KiB sectors).
const ALIGN: usize = 4096;
/// Space left free on the filesystem after the scratch file is written.
const HEADROOM: u64 = 1 << 30;

/// Options for [`measure_storage`].
#[derive(Clone, Copy, Debug)]
pub struct StorageBenchOptions {
    /// Scratch file size. Rounded up to 8 MiB, at least 64 MiB.
    pub file_size: Bytes,
    /// Number of random 4 MiB reads.
    pub random_reads: u32,
}

impl Default for StorageBenchOptions {
    fn default() -> Self {
        Self {
            file_size: Bytes::GIB,
            random_reads: 64,
        }
    }
}

/// How the page cache was bypassed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheBypass {
    /// Linux `O_DIRECT`.
    ODirect,
    /// macOS `F_NOCACHE`.
    FNocache,
    /// Not supported on this filesystem: every pass is buffered.
    None,
}

/// One filesystem's measured read performance (QUBIVEX_DESIGN §4.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StorageMeasurement {
    /// The directory measured.
    pub path: PathBuf,
    /// Scratch file size.
    pub file_size: Bytes,
    /// How the uncached passes bypassed the page cache.
    pub bypass: CacheBypass,
    /// Sequential read rate, bytes per second, uncached when `bypass` is not `None`.
    pub seq_read_bps: f64,
    /// Median latency of a random 4 MiB read.
    pub rand_read_4m_latency: Duration,
    /// 90th-percentile latency of a random 4 MiB read.
    pub rand_read_4m_p90: Duration,
    /// Buffered cold sequential rate, bytes per second.
    pub buffered_seq_read_bps: f64,
    /// `seq_read_bps / buffered_seq_read_bps`: above 1, direct I/O is faster than
    /// buffered I/O on this filesystem. `None` without a cache bypass.
    pub direct_io_gain: Option<f64>,
}

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        // Best effort: the measurement result (or error) matters more than cleanup.
        let _ = std::fs::remove_file(&self.0);
    }
}

fn io(path: &Path) -> impl Fn(std::io::Error) -> HardwareError + '_ {
    move |source| HardwareError::Io {
        path: path.to_owned(),
        source,
    }
}

/// A zeroed buffer of `len` bytes whose start is `ALIGN`-aligned.
struct Aligned {
    buf: Vec<u8>,
    off: usize,
    len: usize,
}

impl Aligned {
    fn new(len: usize) -> Self {
        let buf = vec![0u8; len + ALIGN];
        let off = (ALIGN - buf.as_ptr().addr() % ALIGN) % ALIGN;
        Self { buf, off, len }
    }
    fn get_mut(&mut self) -> &mut [u8] {
        &mut self.buf[self.off..self.off + self.len]
    }
}

/// xorshift64*: a fixed-seed stream, so runs are comparable.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

/// Measures the filesystem holding `dir`. Needs `file_size` + 1 GiB free.
///
/// # Errors
/// [`HardwareError::Io`] if the scratch file cannot be written or read, and
/// [`HardwareError::Parse`] (with what = "free space") if there is not enough room.
pub fn measure_storage(
    dir: &Path,
    opts: StorageBenchOptions,
) -> Result<StorageMeasurement, HardwareError> {
    let block = SEQ_BLOCK as u64;
    let size = opts.file_size.get().max(64 << 20).div_ceil(block) * block;
    let blocks = size / block;

    let vfs = rustix::fs::statvfs(dir).map_err(|e| io(dir)(e.into()))?;
    let free = vfs.f_bavail.saturating_mul(vfs.f_frsize);
    if free < size.saturating_add(HEADROOM) {
        return Err(HardwareError::parse(
            "free space",
            format!(
                "{} needs {} plus {} headroom, {} free",
                dir.display(),
                Bytes(size),
                Bytes(HEADROOM),
                Bytes(free)
            ),
        ));
    }

    let path = dir.join(format!(".qubivex-doctor-{}.tmp", std::process::id()));
    let scratch = Scratch(path.clone());
    write_scratch(&path, blocks)?;

    // 1 and 2: uncached.
    let (direct, bypass) = open_uncached(&path)?;
    let mut buf = Aligned::new(SEQ_BLOCK);
    let t = Instant::now();
    for b in 0..blocks {
        read_exact_at(&direct, buf.get_mut(), b * block, &path)?;
    }
    let seq = t.elapsed();

    let slots = size / RAND_BLOCK as u64;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut rbuf = Aligned::new(RAND_BLOCK);
    let mut lat: Vec<Duration> = Vec::new();
    for _ in 0..opts.random_reads.max(1) {
        let off = (rng.next() % slots) * RAND_BLOCK as u64;
        let t = Instant::now();
        read_exact_at(&direct, rbuf.get_mut(), off, &path)?;
        lat.push(t.elapsed());
    }
    drop(direct);
    lat.sort_unstable();
    let pct = |p: usize| lat[(lat.len() - 1) * p / 100];

    // 3: buffered, cold.
    let buffered = File::open(&path).map_err(io(&path))?;
    evict(&buffered);
    let t = Instant::now();
    for b in 0..blocks {
        read_exact_at(&buffered, buf.get_mut(), b * block, &path)?;
    }
    let buffered_time = t.elapsed();
    drop(buffered);
    drop(scratch);

    #[allow(clippy::cast_precision_loss)] // rates are reported, never accounted
    let rate = |d: Duration| size as f64 / d.as_secs_f64().max(1e-9);
    let seq_read_bps = rate(seq);
    let buffered_seq_read_bps = rate(buffered_time);
    Ok(StorageMeasurement {
        path: dir.to_owned(),
        file_size: Bytes(size),
        bypass,
        seq_read_bps,
        rand_read_4m_latency: pct(50),
        rand_read_4m_p90: pct(90),
        buffered_seq_read_bps,
        direct_io_gain: (bypass != CacheBypass::None).then(|| seq_read_bps / buffered_seq_read_bps),
    })
}

fn write_scratch(path: &Path, blocks: u64) -> Result<(), HardwareError> {
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(io(path))?;
    // Keep the written pages out of the cache where the OS allows it.
    #[cfg(target_os = "macos")]
    let _ = rustix::fs::fcntl_nocache(&f, true);
    let mut rng = Rng(0x5851_f42d_4c95_7f2d);
    let mut data = vec![0u8; SEQ_BLOCK];
    for chunk in data.as_chunks_mut::<8>().0 {
        chunk.copy_from_slice(&rng.next().to_le_bytes());
    }
    for b in 0..blocks {
        // Make every block distinct, so deduplicating filesystems store them all.
        data[..8].copy_from_slice(&b.to_le_bytes());
        f.write_all(&data).map_err(io(path))?;
    }
    f.sync_all().map_err(io(path))?;
    evict(&f);
    Ok(())
}

/// Drops the file's clean pages from the page cache, where the OS has a call for it.
fn evict(f: &File) {
    #[cfg(target_os = "linux")]
    let _ = rustix::fs::fadvise(f, 0, None, rustix::fs::Advice::DontNeed);
    #[cfg(not(target_os = "linux"))]
    let _ = f;
}

fn open_uncached(path: &Path) -> Result<(File, CacheBypass), HardwareError> {
    #[cfg(target_os = "linux")]
    {
        let flags = i32::try_from(rustix::fs::OFlags::DIRECT.bits()).unwrap_or(0);
        match OpenOptions::new().read(true).custom_flags(flags).open(path) {
            Ok(f) => return Ok((f, CacheBypass::ODirect)),
            // EINVAL: the filesystem (tmpfs, some FUSE) does not support O_DIRECT.
            Err(e) if e.raw_os_error() == Some(rustix::io::Errno::INVAL.raw_os_error()) => {}
            Err(e) => return Err(io(path)(e)),
        }
    }
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(0)
        .open(path)
        .map_err(io(path))?;
    #[cfg(target_os = "macos")]
    if rustix::fs::fcntl_nocache(&f, true).is_ok() {
        return Ok((f, CacheBypass::FNocache));
    }
    Ok((f, CacheBypass::None))
}

fn read_exact_at(f: &File, buf: &mut [u8], off: u64, path: &Path) -> Result<(), HardwareError> {
    f.read_exact_at(buf, off).map_err(io(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_buffers_are_aligned() {
        for len in [4096, SEQ_BLOCK] {
            let mut a = Aligned::new(len);
            let s = a.get_mut();
            assert_eq!(s.len(), len);
            assert_eq!(s.as_ptr().addr() % ALIGN, 0);
        }
    }

    #[test]
    fn small_run_measures_and_cleans_up() {
        let dir = std::env::temp_dir().join(format!("qubivex-bench-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let opts = StorageBenchOptions {
            file_size: Bytes(64 << 20),
            random_reads: 4,
        };
        let r = measure_storage(&dir, opts);
        let leftover = std::fs::read_dir(&dir).unwrap().count();
        std::fs::remove_dir_all(&dir).unwrap();
        // A CI temp dir may lack the free space; then the error must say so.
        match r {
            Ok(m) => {
                assert_eq!(m.file_size, Bytes(64 << 20));
                assert!(m.seq_read_bps > 0.0 && m.buffered_seq_read_bps > 0.0);
                assert!(m.rand_read_4m_latency <= m.rand_read_4m_p90);
            }
            Err(e) => assert!(e.to_string().contains("free space"), "{e}"),
        }
        assert_eq!(leftover, 0, "scratch file must be removed");
    }
}
