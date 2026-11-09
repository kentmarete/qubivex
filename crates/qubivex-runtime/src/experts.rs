//! Routed experts as the store sees them: byte extents in the weight files.
//!
//! Reads are buffered `pread`s by default. [`ExpertIo::Direct`] bypasses the page
//! cache (ADR-006 amendment 2): `F_NOCACHE` on macOS, which takes any offset and
//! length, and `O_DIRECT` on Linux, which needs 4 KiB-aligned offsets, lengths and
//! buffers. Expert extents in a GGUF file are not aligned that far, so on Linux each
//! extent is read as the aligned window around it into a per-thread bounce buffer
//! and copied out.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use qubivex_core::{DType, ExpertIdx, LayerIdx};
use qubivex_expert_stream::{ExpertKey, ExpertSource, StoreError};
use qubivex_model::{ExpertPart, ModelMetadata, TensorRole};

use crate::RuntimeError;

/// One part of an expert inside the bytes the store returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartLayout {
    /// Which part.
    pub part: ExpertPart,
    /// Stored dtype.
    pub dtype: DType,
    /// Rows (output features).
    pub rows: usize,
    /// Columns (input features).
    pub cols: usize,
    /// Offset of the part within the expert's bytes.
    pub offset: usize,
    /// Byte length.
    pub len: usize,
    /// Weight file index.
    file: usize,
    /// Absolute offset in that file.
    file_offset: u64,
}

/// Every part of one expert, in a fixed order (gate, up, gate-up, down).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertLayout {
    /// The parts.
    pub parts: Vec<PartLayout>,
    /// Total bytes.
    pub size: usize,
    /// The reads that fill the expert: parts adjacent both in the file and in the
    /// expert's bytes are coalesced into one.
    extents: Vec<Extent>,
}

/// One read: `len` bytes at `file_offset` of `file` into `offset` of the expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Extent {
    file: usize,
    file_offset: u64,
    offset: usize,
    len: usize,
}

/// The parts' reads, merging each part into the previous read when it continues it
/// in both the file and the expert's bytes.
fn coalesce(parts: &[PartLayout]) -> Vec<Extent> {
    let mut out: Vec<Extent> = Vec::new();
    for p in parts {
        if let Some(last) = out.last_mut()
            && last.file == p.file
            && last.file_offset + last.len as u64 == p.file_offset
            && last.offset + last.len == p.offset
        {
            last.len += p.len;
            continue;
        }
        out.push(Extent {
            file: p.file,
            file_offset: p.file_offset,
            offset: p.offset,
            len: p.len,
        });
    }
    out
}

impl ExpertLayout {
    /// The part of the given kind, if present.
    #[must_use]
    pub fn part(&self, p: ExpertPart) -> Option<&PartLayout> {
        self.parts.iter().find(|x| x.part == p)
    }

    /// Reads needed to fill the expert, after coalescing.
    #[must_use]
    pub fn reads(&self) -> usize {
        self.extents.len()
    }
}

/// How the model source reads experts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExpertIo {
    /// Buffered `pread`: the OS keeps what was read in its page cache.
    #[default]
    Buffered,
    /// Bypass the page cache: `F_NOCACHE` on macOS, `O_DIRECT` on Linux. Other
    /// systems, and filesystems without `O_DIRECT`, refuse it.
    Direct,
}

/// How the weight files were actually opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoPath {
    /// Buffered reads.
    Buffered,
    /// macOS `F_NOCACHE`: reads straight into the expert's buffer.
    FNocache,
    /// Linux `O_DIRECT`: aligned windows through a bounce buffer.
    ODirect,
}

impl IoPath {
    /// A short name for records: `buffered`, `f_nocache` or `o_direct`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Buffered => "buffered",
            Self::FNocache => "f_nocache",
            Self::ODirect => "o_direct",
        }
    }
}

/// Reads routed experts straight from the model's weight files (`pread`).
pub struct ModelExpertSource {
    files: Vec<File>,
    path: IoPath,
    experts: BTreeMap<ExpertKey, ExpertLayout>,
}

impl std::fmt::Debug for ModelExpertSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelExpertSource")
            .field("files", &self.files.len())
            .field("path", &self.path)
            .field("experts", &self.experts.len())
            .finish()
    }
}

fn usize_of(v: u64, what: &str) -> Result<usize, RuntimeError> {
    usize::try_from(v)
        .map_err(|_| RuntimeError::Model(format!("{what} {v} does not fit in memory")))
}

impl ModelExpertSource {
    /// The experts of a validated model, from its tensor roles, read buffered.
    ///
    /// # Errors
    /// [`RuntimeError::Model`] for an expert tensor that is not a matrix (or a stack of
    /// matrices), and I/O errors opening the weight files.
    pub fn new(meta: &ModelMetadata) -> Result<Self, RuntimeError> {
        Self::with_io(meta, ExpertIo::Buffered)
    }

    /// The experts of a validated model, read as `io` asks.
    ///
    /// # Errors
    /// As [`ModelExpertSource::new`], and [`RuntimeError::Model`] when direct I/O is
    /// asked for on a system or filesystem without it.
    pub fn with_io(meta: &ModelMetadata, io: ExpertIo) -> Result<Self, RuntimeError> {
        let paths: Vec<PathBuf> = meta
            .tensors
            .files()
            .iter()
            .map(|f| f.path.clone())
            .collect();
        Self::open_files(meta, io, &paths)
    }

    /// The same experts, read from a copy of the model's weight files under `dir` (a
    /// mirror on another drive): each file at the same path relative to the model's
    /// directory. Each copy must have the indexed size; its bytes are not checked here
    /// (wrap the source in a `VerifiedSource` for that).
    ///
    /// # Errors
    /// [`RuntimeError::Model`] for a missing file or one of another size, and as
    /// [`ModelExpertSource::with_io`].
    pub fn mirror_at(meta: &ModelMetadata, io: ExpertIo, dir: &Path) -> Result<Self, RuntimeError> {
        let mut paths = Vec::new();
        for f in meta.tensors.files() {
            let rel = f
                .path
                .strip_prefix(&meta.dir)
                .ok()
                .map(Path::to_path_buf)
                .or_else(|| f.path.file_name().map(PathBuf::from))
                .ok_or_else(|| {
                    RuntimeError::Model(format!("{} has no file name", f.path.display()))
                })?;
            let p = dir.join(rel);
            let len = std::fs::metadata(&p)
                .map_err(|e| RuntimeError::Model(format!("mirror {}: {e}", p.display())))?
                .len();
            if len != f.size {
                return Err(RuntimeError::Model(format!(
                    "mirror {} has {len} bytes; the model's copy has {}",
                    p.display(),
                    f.size
                )));
            }
            paths.push(p);
        }
        Self::open_files(meta, io, &paths)
    }

    fn open_files(
        meta: &ModelMetadata,
        io: ExpertIo,
        paths: &[PathBuf],
    ) -> Result<Self, RuntimeError> {
        let mut path = IoPath::Buffered;
        let mut files = Vec::new();
        for f in paths {
            let (file, p) = match io {
                ExpertIo::Buffered => (open(f)?, IoPath::Buffered),
                ExpertIo::Direct => open_direct(f)?,
            };
            path = p;
            files.push(file);
        }
        let mut parts: BTreeMap<ExpertKey, Vec<PartLayout>> = BTreeMap::new();
        for (name, role) in &meta.roles {
            let Some(t) = meta.tensors.get(name) else {
                continue;
            };
            let dims = t.tensor.shape.dims();
            let bad = || RuntimeError::Model(format!("expert tensor {name:?} has shape {dims:?}"));
            match *role {
                TensorRole::RoutedExpert {
                    layer,
                    expert,
                    part,
                } => {
                    let [rows, cols] = dims else {
                        return Err(bad());
                    };
                    parts.entry((layer, expert)).or_default().push(PartLayout {
                        part,
                        dtype: t.tensor.dtype,
                        rows: usize_of(*rows, "rows")?,
                        cols: usize_of(*cols, "cols")?,
                        offset: 0,
                        len: usize_of(t.tensor.bytes, "expert bytes")?,
                        file: t.file,
                        file_offset: t.tensor.offset,
                    });
                }
                TensorRole::RoutedExpertStack {
                    layer,
                    part,
                    experts,
                } => {
                    let [n, rows, cols] = dims else {
                        return Err(bad());
                    };
                    if *n != u64::from(experts) || experts == 0 {
                        return Err(bad());
                    }
                    let slice = t.tensor.bytes / u64::from(experts);
                    for e in 0..experts {
                        parts
                            .entry((layer, ExpertIdx(e)))
                            .or_default()
                            .push(PartLayout {
                                part,
                                dtype: t.tensor.dtype,
                                rows: usize_of(*rows, "rows")?,
                                cols: usize_of(*cols, "cols")?,
                                offset: 0,
                                len: usize_of(slice, "expert bytes")?,
                                file: t.file,
                                file_offset: t.tensor.offset + u64::from(e) * slice,
                            });
                    }
                }
                _ => {}
            }
        }
        let experts = parts
            .into_iter()
            .map(|(k, mut ps)| {
                ps.sort_by_key(|p| p.part);
                let mut off = 0;
                for p in &mut ps {
                    p.offset = off;
                    off += p.len;
                }
                let extents = coalesce(&ps);
                (
                    k,
                    ExpertLayout {
                        parts: ps,
                        size: off,
                        extents,
                    },
                )
            })
            .collect();
        Ok(Self {
            files,
            path,
            experts,
        })
    }

    /// How the weight files are read.
    #[must_use]
    pub const fn io_path(&self) -> IoPath {
        self.path
    }

    /// The layout of an expert, if the model has it.
    #[must_use]
    pub fn layout(&self, layer: LayerIdx, expert: ExpertIdx) -> Option<&ExpertLayout> {
        self.experts.get(&(layer, expert))
    }

    /// Every expert, in key order.
    pub fn keys(&self) -> impl Iterator<Item = ExpertKey> + '_ {
        self.experts.keys().copied()
    }

    /// Number of experts.
    #[must_use]
    pub fn len(&self) -> usize {
        self.experts.len()
    }

    /// Whether the model has no routed experts.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.experts.is_empty()
    }

    /// Bytes of all experts together (the all-resident capacity).
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.experts.values().map(|e| e.size as u64).sum()
    }
}

fn open(p: &Path) -> Result<File, RuntimeError> {
    File::open(p).map_err(|e| RuntimeError::Model(format!("cannot open {}: {e}", p.display())))
}

/// Opens a weight file so reads bypass the page cache.
#[cfg(target_os = "linux")]
fn open_direct(p: &Path) -> Result<(File, IoPath), RuntimeError> {
    use std::os::unix::fs::OpenOptionsExt;
    let flags = i32::try_from(rustix::fs::OFlags::DIRECT.bits()).unwrap_or(0);
    match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(flags)
        .open(p)
    {
        Ok(f) => Ok((f, IoPath::ODirect)),
        Err(e) if e.raw_os_error() == Some(rustix::io::Errno::INVAL.raw_os_error()) => {
            Err(RuntimeError::Model(format!(
                "the filesystem holding {} does not support O_DIRECT; drop --direct-io",
                p.display()
            )))
        }
        Err(e) => Err(RuntimeError::Model(format!(
            "cannot open {}: {e}",
            p.display()
        ))),
    }
}

/// Opens a weight file so reads bypass the page cache.
#[cfg(target_os = "macos")]
fn open_direct(p: &Path) -> Result<(File, IoPath), RuntimeError> {
    let f = open(p)?;
    rustix::fs::fcntl_nocache(&f, true).map_err(|e| {
        RuntimeError::Model(format!("cannot set F_NOCACHE on {}: {e}", p.display()))
    })?;
    Ok((f, IoPath::FNocache))
}

/// Opens a weight file so reads bypass the page cache.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn open_direct(_: &Path) -> Result<(File, IoPath), RuntimeError> {
    Err(RuntimeError::Model(
        "direct I/O is supported on Linux and macOS only; drop --direct-io".into(),
    ))
}

/// Reads at `off` into `buf`, returning the bytes read (0 at the end of the file).
#[cfg(unix)]
fn pread(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(f, buf, off)
}

/// Reads at `off` into `buf`, returning the bytes read (0 at the end of the file).
#[cfg(windows)]
fn pread(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(f, buf, off)
}

/// Fills at least `need` bytes of `buf` from `off`; the end of the file before
/// `need` bytes is an error. With `aligned`, every read starts on a block boundary.
fn read_full(
    read: &mut impl FnMut(&mut [u8], u64) -> std::io::Result<usize>,
    buf: &mut [u8],
    off: u64,
    need: usize,
    aligned: bool,
) -> std::io::Result<()> {
    let mut done = 0;
    while done < need {
        match read(&mut buf[done..], off + done as u64) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                done += n;
                // Under O_DIRECT only the end of the file ends a read off a block
                // boundary, and the next read would be misaligned.
                if aligned && done < need && done % DIRECT_ALIGN != 0 {
                    return Err(std::io::ErrorKind::UnexpectedEof.into());
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// `O_DIRECT` alignment of offsets, lengths and buffers (covers 512 B and 4 KiB
/// logical blocks).
const DIRECT_ALIGN: usize = 4096;

/// A buffer whose start is `DIRECT_ALIGN`-aligned, without `unsafe`: an
/// over-allocated vector and an offset into it.
#[derive(Default)]
struct Bounce {
    buf: Vec<u8>,
    off: usize,
    len: usize,
}

impl Bounce {
    /// The first `len` aligned bytes, growing the buffer if needed.
    fn get(&mut self, len: usize) -> &mut [u8] {
        if len > self.len {
            self.buf = vec![0u8; len + DIRECT_ALIGN];
            self.off = (DIRECT_ALIGN - self.buf.as_ptr().addr() % DIRECT_ALIGN) % DIRECT_ALIGN;
            self.len = len;
        }
        &mut self.buf[self.off..self.off + len]
    }
}

thread_local! {
    /// Each I/O thread's bounce buffer, as large as the largest window it has read.
    static BOUNCE: RefCell<Bounce> = RefCell::new(Bounce::default());
}

/// Reads `dst.len()` bytes at `off` using only aligned reads: the aligned window
/// around the extent goes into `bounce`, and the extent is copied out. The window may
/// run past the end of the file; only the extent's bytes must exist.
fn read_window(
    read: &mut impl FnMut(&mut [u8], u64) -> std::io::Result<usize>,
    bounce: &mut Bounce,
    dst: &mut [u8],
    off: u64,
) -> std::io::Result<()> {
    let align = DIRECT_ALIGN as u64;
    let start = off - off % align;
    let head = usize::try_from(off - start).map_err(std::io::Error::other)?;
    let need = head + dst.len();
    let window = need.next_multiple_of(DIRECT_ALIGN);
    let buf = bounce.get(window);
    read_full(read, buf, start, need, true)?;
    dst.copy_from_slice(&buf[head..need]);
    Ok(())
}

impl ModelExpertSource {
    fn read_extent(&self, x: &Extent, dst: &mut [u8]) -> std::io::Result<()> {
        let f = &self.files[x.file];
        let mut read = |b: &mut [u8], o: u64| pread(f, b, o);
        match self.path {
            IoPath::Buffered | IoPath::FNocache => {
                read_full(&mut read, dst, x.file_offset, x.len, false)
            }
            IoPath::ODirect => {
                BOUNCE.with(|b| read_window(&mut read, &mut b.borrow_mut(), dst, x.file_offset))
            }
        }
    }
}

impl ExpertSource for ModelExpertSource {
    fn expert_size(&self, key: ExpertKey) -> Result<u64, StoreError> {
        self.experts
            .get(&key)
            .map(|e| e.size as u64)
            .ok_or_else(|| StoreError::Load {
                key,
                detail: "the model has no such expert".into(),
            })
    }

    fn read_expert(&self, key: ExpertKey, dst: &mut [u8]) -> Result<(), StoreError> {
        let e = self.experts.get(&key).ok_or_else(|| StoreError::Load {
            key,
            detail: "the model has no such expert".into(),
        })?;
        if dst.len() != e.size {
            return Err(StoreError::Load {
                key,
                detail: format!("buffer of {} bytes for a {}-byte expert", dst.len(), e.size),
            });
        }
        for x in &e.extents {
            self.read_extent(x, &mut dst[x.offset..x.offset + x.len])
                .map_err(|err| StoreError::Load {
                    key,
                    detail: format!("{} bytes at {}: {err}", x.len, x.file_offset),
                })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(
        part: ExpertPart,
        file: usize,
        file_offset: u64,
        offset: usize,
        len: usize,
    ) -> PartLayout {
        PartLayout {
            part,
            dtype: DType::F32,
            rows: 1,
            cols: len / 4,
            offset,
            len,
            file,
            file_offset,
        }
    }

    /// A file of `len` bytes (byte `i` is `i * 7 + 3`) read as `O_DIRECT` would allow:
    /// a misaligned offset, length or buffer fails the test.
    fn strict_reader(data: &[u8]) -> impl FnMut(&mut [u8], u64) -> std::io::Result<usize> + '_ {
        move |buf: &mut [u8], off: u64| {
            assert_eq!(off % DIRECT_ALIGN as u64, 0, "offset {off}");
            assert_eq!(buf.len() % DIRECT_ALIGN, 0, "length {}", buf.len());
            assert_eq!(buf.as_ptr().addr() % DIRECT_ALIGN, 0, "buffer");
            let off = usize::try_from(off).unwrap();
            if off >= data.len() {
                return Ok(0);
            }
            // Short reads, as a kernel may return: at most 3 blocks at a time.
            let n = buf.len().min(data.len() - off).min(3 * DIRECT_ALIGN);
            buf[..n].copy_from_slice(&data[off..off + n]);
            Ok(n)
        }
    }

    #[test]
    fn aligned_windows_read_any_extent() {
        #[allow(clippy::cast_possible_truncation)]
        let data: Vec<u8> = (0..5 * DIRECT_ALIGN + 123)
            .map(|i| (i * 7 + 3) as u8)
            .collect();
        let mut bounce = Bounce::default();
        let a = DIRECT_ALIGN;
        let cases = [
            (0, 1),
            (0, a),
            (1, a),
            (a - 1, 2),
            (a, 3 * a),
            (37, 4 * a + 5),
            // Extents that end exactly at the end of an unaligned file.
            (data.len() - 10, 10),
            (100, data.len() - 100),
            (0, data.len()),
        ];
        for (off, len) in cases {
            let mut dst = vec![0u8; len];
            let mut read = strict_reader(&data);
            read_window(&mut read, &mut bounce, &mut dst, off as u64).unwrap();
            assert_eq!(dst, data[off..off + len], "extent {off}+{len}");
        }
        // Past the end of the file: an error, not zeros.
        let mut dst = vec![0u8; 20];
        let mut read = strict_reader(&data);
        let err =
            read_window(&mut read, &mut bounce, &mut dst, (data.len() - 10) as u64).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn bounce_buffers_stay_aligned_as_they_grow() {
        let mut b = Bounce::default();
        for len in [4096, 100, 3 * 4096, 4096, 10 * 4096] {
            let s = b.get(len);
            assert_eq!(s.len(), len);
            assert_eq!(s.as_ptr().addr() % DIRECT_ALIGN, 0);
        }
    }

    #[test]
    fn contiguous_parts_become_one_read() {
        use ExpertPart::{Down, Gate, Up};
        // gate, up, down back to back in one file: one read.
        let ps = [
            part(Gate, 0, 100, 0, 8),
            part(Up, 0, 108, 8, 8),
            part(Down, 0, 116, 16, 8),
        ];
        assert_eq!(
            coalesce(&ps),
            [Extent {
                file: 0,
                file_offset: 100,
                offset: 0,
                len: 24
            }]
        );
        // A gap, a different file, or a file order unlike the byte order: separate reads.
        let gap = [part(Gate, 0, 100, 0, 8), part(Up, 0, 112, 8, 8)];
        assert_eq!(coalesce(&gap).len(), 2);
        let files = [part(Gate, 0, 100, 0, 8), part(Up, 1, 108, 8, 8)];
        assert_eq!(coalesce(&files).len(), 2);
        let swapped = [part(Gate, 0, 108, 0, 8), part(Up, 0, 100, 8, 8)];
        assert_eq!(coalesce(&swapped).len(), 2);
    }
}
