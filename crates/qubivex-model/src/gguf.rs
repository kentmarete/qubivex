//! GGUF header reader (versions 2 and 3, little-endian).
//!
//! Layout: `GGUF`, version (u32), tensor count (u64), metadata count (u64), the
//! metadata key/value pairs, the tensor infos (name, dims innermost first, type,
//! offset), padding to `general.alignment` (default 32), then the data region.
//! Tensor offsets are relative to the data region and aligned.
//!
//! Every length and count is bounded by hard caps and by the bytes left in the file,
//! so a crafted header cannot trigger a large allocation.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use qubivex_core::{DType, MAX_RANK, Shape};
use serde::Serialize;

use crate::ModelError;
use crate::index::RawTensor;

/// Longest metadata key.
const MAX_KEY: u64 = 64 << 10;
/// Longest string value or tensor name.
const MAX_STRING: u64 = 16 << 20;
/// Most elements in one metadata array (a 256k-entry vocabulary fits).
const MAX_ARRAY: u64 = 64 << 20;
/// Most metadata entries.
const MAX_KV: u64 = 1 << 20;
/// Most tensors.
const MAX_TENSORS: u64 = 1 << 22;
/// Deepest array nesting.
const MAX_DEPTH: u32 = 2;
/// Default `general.alignment`.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// A GGUF metadata value.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum GgufValue {
    /// Type 0.
    U8(u8),
    /// Type 1.
    I8(i8),
    /// Type 2.
    U16(u16),
    /// Type 3.
    I16(i16),
    /// Type 4.
    U32(u32),
    /// Type 5.
    I32(i32),
    /// Type 6.
    F32(f32),
    /// Type 7.
    Bool(bool),
    /// Type 8.
    String(String),
    /// Type 9: elements share one type.
    Array(Vec<GgufValue>),
    /// Type 10.
    U64(u64),
    /// Type 11.
    I64(i64),
    /// Type 12.
    F64(f64),
}

impl GgufValue {
    /// The value as an unsigned integer, if it is a non-negative integer of any width.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(u64::from(v)),
            Self::U16(v) => Some(u64::from(v)),
            Self::U32(v) => Some(u64::from(v)),
            Self::U64(v) => Some(v),
            Self::I8(v) => u64::try_from(v).ok(),
            Self::I16(v) => u64::try_from(v).ok(),
            Self::I32(v) => u64::try_from(v).ok(),
            Self::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    /// The value as a float, if it is numeric.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // config values, far below 2^53
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::F32(v) => Some(f64::from(v)),
            Self::F64(v) => Some(v),
            Self::I8(v) => Some(f64::from(v)),
            Self::I16(v) => Some(f64::from(v)),
            Self::I32(v) => Some(f64::from(v)),
            Self::I64(v) => Some(v as f64),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    /// The value as a string slice, if it is a string.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// The value as a bool, if it is one.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(b) => Some(b),
            _ => None,
        }
    }
}

/// GGUF metadata, by key.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(transparent)]
pub struct GgufMetadata(pub BTreeMap<String, GgufValue>);

impl GgufMetadata {
    /// A value by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.0.get(key)
    }

    /// `general.architecture`, the prefix of architecture-specific keys.
    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        self.get("general.architecture").and_then(GgufValue::as_str)
    }

    /// An architecture-specific value, `<general.architecture>.<suffix>`.
    #[must_use]
    pub fn arch_value(&self, suffix: &str) -> Option<&GgufValue> {
        let arch = self.architecture()?;
        self.get(&format!("{arch}.{suffix}"))
    }
}

/// A parsed GGUF header.
#[derive(Clone, Debug, PartialEq)]
pub struct GgufHeader {
    /// Format version (2 or 3).
    pub version: u32,
    /// Metadata.
    pub metadata: GgufMetadata,
    /// Tensors, sorted by offset. `offset` is absolute within the file.
    pub tensors: Vec<RawTensor>,
    /// Data alignment.
    pub alignment: u64,
    /// Where the data region starts.
    pub data_start: u64,
}

struct Reader<'p, R> {
    r: R,
    pos: u64,
    len: u64,
    path: &'p Path,
}

impl<R: Read> Reader<'_, R> {
    fn err(&self, detail: impl Into<String>) -> ModelError {
        ModelError::Header {
            path: self.path.to_owned(),
            format: "gguf",
            detail: format!("at byte {}: {}", self.pos, detail.into()),
        }
    }

    fn remaining(&self) -> u64 {
        self.len - self.pos
    }

    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], ModelError> {
        if (N as u64) > self.remaining() {
            return Err(self.err("unexpected end of file"));
        }
        let mut b = [0u8; N];
        self.r.read_exact(&mut b).map_err(|source| ModelError::Io {
            path: self.path.to_owned(),
            source,
        })?;
        self.pos += N as u64;
        Ok(b)
    }

    fn u32(&mut self) -> Result<u32, ModelError> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }

    fn u64(&mut self) -> Result<u64, ModelError> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }

    /// A count of items each at least `min_item` bytes long, capped at `cap`.
    fn count(&mut self, what: &str, cap: u64, min_item: u64) -> Result<u64, ModelError> {
        let n = self.u64()?;
        if n > cap || n.saturating_mul(min_item) > self.remaining() {
            return Err(self.err(format!("{what} {n} is larger than the file allows")));
        }
        Ok(n)
    }

    fn string(&mut self, cap: u64) -> Result<String, ModelError> {
        let n = self.count("string length", cap, 1)?;
        let n = usize::try_from(n).map_err(|_| self.err("string does not fit in memory"))?;
        let mut b = vec![0u8; n];
        self.r.read_exact(&mut b).map_err(|source| ModelError::Io {
            path: self.path.to_owned(),
            source,
        })?;
        self.pos += n as u64;
        String::from_utf8(b).map_err(|_| self.err("string is not UTF-8"))
    }

    fn value(&mut self, ty: u32, depth: u32) -> Result<GgufValue, ModelError> {
        Ok(match ty {
            0 => GgufValue::U8(u8::from_le_bytes(self.bytes()?)),
            1 => GgufValue::I8(i8::from_le_bytes(self.bytes()?)),
            2 => GgufValue::U16(u16::from_le_bytes(self.bytes()?)),
            3 => GgufValue::I16(i16::from_le_bytes(self.bytes()?)),
            4 => GgufValue::U32(self.u32()?),
            5 => GgufValue::I32(i32::from_le_bytes(self.bytes()?)),
            6 => GgufValue::F32(f32::from_le_bytes(self.bytes()?)),
            7 => match u8::from_le_bytes(self.bytes()?) {
                0 => GgufValue::Bool(false),
                1 => GgufValue::Bool(true),
                b => return Err(self.err(format!("bool byte {b}"))),
            },
            8 => GgufValue::String(self.string(MAX_STRING)?),
            9 => {
                if depth >= MAX_DEPTH {
                    return Err(self.err("arrays nested too deeply"));
                }
                let et = self.u32()?;
                let min = match et {
                    0 | 1 | 7 => 1,
                    2 | 3 => 2,
                    4..=6 => 4,
                    8..=12 => 8,
                    t => return Err(self.err(format!("unknown array element type {t}"))),
                };
                let n = self.count("array length", MAX_ARRAY, min)?;
                // Grow as elements arrive; the count was bounded by the file above.
                let mut v = Vec::with_capacity(usize::try_from(n.min(1 << 16)).unwrap_or(0));
                for _ in 0..n {
                    v.push(self.value(et, depth + 1)?);
                }
                GgufValue::Array(v)
            }
            10 => GgufValue::U64(self.u64()?),
            11 => GgufValue::I64(i64::from_le_bytes(self.bytes()?)),
            12 => GgufValue::F64(f64::from_le_bytes(self.bytes()?)),
            t => return Err(self.err(format!("unknown value type {t}"))),
        })
    }
}

/// Reads the header of a GGUF file of `file_len` bytes from `r`.
///
/// `path` is used only in error messages.
///
/// # Errors
/// [`ModelError::Header`] for a malformed header (bad magic, unsupported version,
/// limits exceeded, bad alignment), [`ModelError::Tensor`] for a bad tensor (unknown
/// type, misaligned or out-of-bounds data, overlap), and [`ModelError::Io`].
pub fn parse(r: impl Read, file_len: u64, path: &Path) -> Result<GgufHeader, ModelError> {
    let mut rd = Reader {
        r,
        pos: 0,
        len: file_len,
        path,
    };
    if &rd.bytes::<4>()? != b"GGUF" {
        return Err(rd.err("not a GGUF file (bad magic)"));
    }
    let version = rd.u32()?;
    match version {
        2 | 3 => {}
        v if matches!(v.swap_bytes(), 2 | 3) => {
            return Err(rd.err("big-endian GGUF is not supported"));
        }
        v => return Err(rd.err(format!("unsupported GGUF version {v}"))),
    }
    // Minimal sizes: a tensor info is >= 8 (name len) + 4 + 4 + 8 bytes; a kv >= 8 + 4 + 1.
    let n_tensors = rd.count("tensor count", MAX_TENSORS, 24)?;
    let n_kv = rd.count("metadata count", MAX_KV, 13)?;

    let mut meta = BTreeMap::new();
    for _ in 0..n_kv {
        let key = rd.string(MAX_KEY)?;
        let ty = rd.u32()?;
        let v = rd.value(ty, 0)?;
        if meta.insert(key.clone(), v).is_some() {
            return Err(rd.err(format!("duplicate metadata key {key:?}")));
        }
    }
    let metadata = GgufMetadata(meta);

    struct Info {
        name: String,
        dims: Vec<u64>,
        ty: u32,
        offset: u64,
    }
    let mut infos = Vec::new();
    for _ in 0..n_tensors {
        let name = rd.string(MAX_STRING)?;
        let n_dims = rd.u32()?;
        if n_dims as usize > MAX_RANK {
            return Err(rd.err(format!("tensor {name:?} has {n_dims} dimensions")));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(rd.u64()?);
        }
        dims.reverse(); // GGUF lists the innermost dimension first
        let ty = rd.u32()?;
        let offset = rd.u64()?;
        infos.push(Info {
            name,
            dims,
            ty,
            offset,
        });
    }

    let alignment = match metadata.get("general.alignment") {
        None => DEFAULT_ALIGNMENT,
        Some(v) => match v.as_u64() {
            Some(a) if a.is_power_of_two() && a <= 1 << 20 => a,
            _ => {
                return Err(rd.err(format!(
                    "general.alignment {v:?} is not a power of two up to 1 MiB"
                )));
            }
        },
    };
    let data_start = rd
        .pos
        .div_ceil(alignment)
        .checked_mul(alignment)
        .filter(|&d| d <= file_len)
        .ok_or_else(|| rd.err("data region starts past the end of the file"))?;

    let mut tensors = Vec::with_capacity(infos.len());
    for i in infos {
        let bad = |detail: String| ModelError::Tensor {
            path: path.to_owned(),
            name: i.name.clone(),
            detail,
        };
        let dtype = DType::from_ggml_id(i.ty).map_err(|e| bad(e.to_string()))?;
        let shape = Shape::new(&i.dims).map_err(|e| bad(e.to_string()))?;
        let bytes = dtype
            .storage_bytes(&shape)
            .map_err(|e| bad(e.to_string()))?;
        if i.offset % alignment != 0 {
            return Err(bad(format!(
                "offset {} is not {alignment}-byte aligned",
                i.offset
            )));
        }
        let offset = data_start
            .checked_add(i.offset)
            .filter(|&o| o.checked_add(bytes).is_some_and(|end| end <= file_len))
            .ok_or_else(|| {
                bad(format!(
                    "data [{}, +{bytes}) runs past the end of the file ({file_len} bytes)",
                    i.offset
                ))
            })?;
        tensors.push(RawTensor {
            name: i.name,
            dtype,
            shape,
            offset,
            bytes,
        });
    }
    tensors.sort_by(|a, b| a.offset.cmp(&b.offset).then_with(|| a.name.cmp(&b.name)));
    for w in tensors.windows(2) {
        if w[0].offset + w[0].bytes > w[1].offset {
            return Err(ModelError::Tensor {
                path: path.to_owned(),
                name: w[1].name.clone(),
                detail: format!("overlaps tensor {:?}", w[0].name),
            });
        }
    }
    let mut names: Vec<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
    names.sort_unstable();
    if let Some(w) = names.windows(2).find(|w| w[0] == w[1]) {
        return Err(rd.err(format!("duplicate tensor {:?}", w[0])));
    }

    Ok(GgufHeader {
        version,
        metadata,
        tensors,
        alignment,
        data_start,
    })
}
