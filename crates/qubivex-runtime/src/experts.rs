//! Routed experts as the store sees them: byte extents in the weight files.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

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
}

impl ExpertLayout {
    /// The part of the given kind, if present.
    #[must_use]
    pub fn part(&self, p: ExpertPart) -> Option<&PartLayout> {
        self.parts.iter().find(|x| x.part == p)
    }
}

/// Reads routed experts straight from the model's weight files (`pread`).
pub struct ModelExpertSource {
    files: Vec<File>,
    experts: BTreeMap<ExpertKey, ExpertLayout>,
}

impl std::fmt::Debug for ModelExpertSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelExpertSource")
            .field("files", &self.files.len())
            .field("experts", &self.experts.len())
            .finish()
    }
}

fn usize_of(v: u64, what: &str) -> Result<usize, RuntimeError> {
    usize::try_from(v)
        .map_err(|_| RuntimeError::Model(format!("{what} {v} does not fit in memory")))
}

impl ModelExpertSource {
    /// The experts of a validated model, from its tensor roles.
    ///
    /// # Errors
    /// [`RuntimeError::Model`] for an expert tensor that is not a matrix (or a stack of
    /// matrices), and I/O errors opening the weight files.
    pub fn new(meta: &ModelMetadata) -> Result<Self, RuntimeError> {
        let files = meta
            .tensors
            .files()
            .iter()
            .map(|f| open(&f.path))
            .collect::<Result<Vec<_>, _>>()?;
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
                (
                    k,
                    ExpertLayout {
                        parts: ps,
                        size: off,
                    },
                )
            })
            .collect();
        Ok(Self { files, experts })
    }

    /// The layout of an expert, if the model has it.
    #[must_use]
    pub fn layout(&self, layer: LayerIdx, expert: ExpertIdx) -> Option<&ExpertLayout> {
        self.experts.get(&(layer, expert))
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

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(f, buf, off)
}

#[cfg(windows)]
fn read_at(f: &File, mut buf: &mut [u8], mut off: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match f.seek_read(buf, off)? {
            0 => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            n => {
                buf = &mut buf[n..];
                off += n as u64;
            }
        }
    }
    Ok(())
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
        for p in &e.parts {
            read_at(
                &self.files[p.file],
                &mut dst[p.offset..p.offset + p.len],
                p.file_offset,
            )
            .map_err(|err| StoreError::Load {
                key,
                detail: format!("{:?}: {err}", p.part),
            })?;
        }
        Ok(())
    }
}
