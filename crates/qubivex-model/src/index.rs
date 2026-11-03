//! The tensor index: every tensor of a model, across its weight files.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use qubivex_core::{CoreError, DType, Shape, TensorDesc};
use serde::Serialize;

use crate::gguf::{self, GgufMetadata};
use crate::{ModelError, safetensors};

/// A tensor as a container header describes it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RawTensor {
    /// Name in the file.
    pub name: String,
    /// Element type.
    pub dtype: DType,
    /// Row-major shape (last dimension contiguous).
    pub shape: Shape,
    /// Absolute byte offset of the data in the file.
    pub offset: u64,
    /// Exact byte length, from `dtype` and `shape`.
    pub bytes: u64,
}

impl RawTensor {
    /// The contiguous descriptor of this tensor's bytes, relative to its first byte
    /// (the file offset is [`RawTensor::offset`]).
    ///
    /// # Errors
    /// As [`TensorDesc::contiguous`]; never for a tensor that passed the index's checks.
    pub fn desc(&self) -> Result<TensorDesc, CoreError> {
        TensorDesc::contiguous(self.dtype, self.shape)
    }
}

/// A weight file's container format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WeightFormat {
    /// SafeTensors.
    Safetensors,
    /// GGUF.
    Gguf,
}

/// One weight file of a model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct WeightFile {
    /// Path as given to the index.
    pub path: PathBuf,
    /// Container format.
    pub format: WeightFormat,
    /// File size in bytes, at the time it was indexed.
    pub size: u64,
}

/// One indexed tensor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TensorEntry {
    /// The tensor.
    #[serde(flatten)]
    pub tensor: RawTensor,
    /// Index into [`TensorIndex::files`].
    pub file: usize,
}

/// Every tensor of a model, by name, with the file and byte range holding it.
///
/// Byte ranges are validated against the real file sizes when files are added.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct TensorIndex {
    files: Vec<WeightFile>,
    tensors: BTreeMap<String, TensorEntry>,
    /// GGUF metadata, if a GGUF file was added.
    gguf: Option<GgufMetadata>,
}

fn open(path: &Path) -> Result<(BufReader<File>, u64), ModelError> {
    let io = |source| ModelError::Io {
        path: path.to_owned(),
        source,
    };
    let f = File::open(path).map_err(io)?;
    let len = f.metadata().map_err(io)?.len();
    Ok((BufReader::with_capacity(1 << 20, f), len))
}

impl TensorIndex {
    /// An empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Indexes a SafeTensors file. Returns its `__metadata__`.
    ///
    /// # Errors
    /// Any header problem, or a tensor already defined by another file.
    pub fn add_safetensors(&mut self, path: &Path) -> Result<BTreeMap<String, String>, ModelError> {
        let (r, len) = open(path)?;
        let h = safetensors::parse(r, len, path)?;
        self.add(path, WeightFormat::Safetensors, len, h.tensors)?;
        Ok(h.metadata)
    }

    /// Indexes a GGUF file and keeps its metadata ([`TensorIndex::gguf_metadata`]).
    ///
    /// # Errors
    /// Any header problem, a tensor already defined by another file, or a second
    /// GGUF file (split GGUF is not supported yet).
    pub fn add_gguf(&mut self, path: &Path) -> Result<(), ModelError> {
        if self.gguf.is_some() {
            return Err(ModelError::Header {
                path: path.to_owned(),
                format: "gguf",
                detail: "a model may have only one GGUF file (split GGUF is not supported)"
                    .to_owned(),
            });
        }
        let (r, len) = open(path)?;
        let h = gguf::parse(r, len, path)?;
        self.add(path, WeightFormat::Gguf, len, h.tensors)?;
        self.gguf = Some(h.metadata);
        Ok(())
    }

    fn add(
        &mut self,
        path: &Path,
        format: WeightFormat,
        size: u64,
        tensors: Vec<RawTensor>,
    ) -> Result<(), ModelError> {
        let file = self.files.len();
        // Check every name before inserting any, so a failed add leaves the index as it was.
        for t in &tensors {
            if let Some(prev) = self.tensors.get(&t.name) {
                return Err(ModelError::DuplicateTensor {
                    name: t.name.clone(),
                    first: self.files[prev.file].path.clone(),
                    second: path.to_owned(),
                });
            }
        }
        self.files.push(WeightFile {
            path: path.to_owned(),
            format,
            size,
        });
        for t in tensors {
            self.tensors
                .insert(t.name.clone(), TensorEntry { tensor: t, file });
        }
        Ok(())
    }

    /// The indexed files.
    #[must_use]
    pub fn files(&self) -> &[WeightFile] {
        &self.files
    }

    /// A tensor by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.get(name)
    }

    /// All tensors, by name.
    pub fn iter(&self) -> impl Iterator<Item = &TensorEntry> {
        self.tensors.values()
    }

    /// Number of tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Sum of all tensor sizes.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.tensors
            .values()
            .map(|t| t.tensor.bytes)
            .fold(0, u64::saturating_add)
    }

    /// GGUF metadata, if the model is a GGUF file.
    #[must_use]
    pub fn gguf_metadata(&self) -> Option<&GgufMetadata> {
        self.gguf.as_ref()
    }

    /// Bytes per dtype, for summaries.
    #[must_use]
    pub fn bytes_by_dtype(&self) -> BTreeMap<DType, u64> {
        let mut m = BTreeMap::new();
        for t in self.tensors.values() {
            let e = m.entry(t.tensor.dtype).or_insert(0u64);
            *e = e.saturating_add(t.tensor.bytes);
        }
        m
    }
}
