//! SafeTensors header reader.
//!
//! Layout: an 8-byte little-endian header length `N`, `N` bytes of JSON, then the
//! data region. The JSON maps each tensor name to `{dtype, shape, data_offsets}`
//! (offsets relative to the data region); an optional `__metadata__` maps strings to
//! strings. As in the reference implementation, the tensors must tile the data region
//! exactly: no overlaps, no holes, nothing after the last tensor.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use qubivex_core::{DType, Shape};
use serde::Deserialize;

use crate::ModelError;
use crate::index::RawTensor;

/// Largest header accepted (the reference implementation's limit).
pub const MAX_HEADER_BYTES: u64 = 100 << 20;

/// A parsed SafeTensors header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafetensorsHeader {
    /// Tensors, sorted by offset. `offset` is absolute within the file.
    pub tensors: Vec<RawTensor>,
    /// The `__metadata__` map.
    pub metadata: BTreeMap<String, String>,
    /// Where the data region starts.
    pub data_start: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

fn dtype(s: &str) -> Option<DType> {
    match s {
        "F32" => Some(DType::F32),
        "F16" => Some(DType::F16),
        "BF16" => Some(DType::BF16),
        _ => None,
    }
}

/// Reads the header of a SafeTensors file of `file_len` bytes from `r`.
///
/// `path` is used only in error messages.
///
/// # Errors
/// [`ModelError::Header`] for a malformed header, [`ModelError::Tensor`] for a bad
/// entry (unsupported dtype, size not matching shape, overlap, out of bounds), and
/// [`ModelError::Io`] if reading fails.
pub fn parse(
    mut r: impl Read,
    file_len: u64,
    path: &Path,
) -> Result<SafetensorsHeader, ModelError> {
    let header = |detail: String| ModelError::Header {
        path: path.to_owned(),
        format: "safetensors",
        detail,
    };
    let io = |source| ModelError::Io {
        path: path.to_owned(),
        source,
    };
    let mut len = [0u8; 8];
    r.read_exact(&mut len).map_err(io)?;
    let n = u64::from_le_bytes(len);
    let data_start = n.checked_add(8).filter(|&s| s <= file_len).ok_or_else(|| {
        header(format!(
            "header length {n} exceeds the file ({file_len} bytes)"
        ))
    })?;
    if n > MAX_HEADER_BYTES {
        return Err(header(format!(
            "header length {n} exceeds the {MAX_HEADER_BYTES}-byte limit"
        )));
    }
    let n =
        usize::try_from(n).map_err(|_| header("header length does not fit in memory".into()))?;
    let mut json = vec![0u8; n];
    r.read_exact(&mut json).map_err(io)?;
    if !json.starts_with(b"{") {
        return Err(header("header is not a JSON object".into()));
    }
    let raw: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&json).map_err(|e| header(format!("bad JSON: {e}")))?;

    let mut metadata = BTreeMap::new();
    let mut tensors = Vec::with_capacity(raw.len());
    for (name, v) in raw {
        if name == "__metadata__" {
            metadata =
                serde_json::from_value(v).map_err(|e| header(format!("bad __metadata__: {e}")))?;
            continue;
        }
        let bad = |detail: String| ModelError::Tensor {
            path: path.to_owned(),
            name: name.clone(),
            detail,
        };
        let e: Entry = serde_json::from_value(v).map_err(|e| bad(format!("bad entry: {e}")))?;
        let dt = dtype(&e.dtype).ok_or_else(|| bad(format!("unsupported dtype {}", e.dtype)))?;
        let shape = Shape::new(&e.shape).map_err(|e| bad(e.to_string()))?;
        let bytes = dt.storage_bytes(&shape).map_err(|e| bad(e.to_string()))?;
        let [begin, end] = e.data_offsets;
        if begin > end || end - begin != bytes {
            return Err(bad(format!(
                "data_offsets [{begin}, {end}] hold {} bytes, but {dt} {:?} needs {bytes}",
                end.saturating_sub(begin),
                e.shape
            )));
        }
        let offset = data_start
            .checked_add(begin)
            .ok_or_else(|| bad("offset overflows".into()))?;
        tensors.push(RawTensor {
            name,
            dtype: dt,
            shape,
            offset,
            bytes,
        });
    }

    // The tensors must tile [data_start, file_len) exactly.
    tensors.sort_by(|a, b| a.offset.cmp(&b.offset).then_with(|| a.name.cmp(&b.name)));
    let mut cursor = data_start;
    for t in &tensors {
        if t.offset != cursor {
            let detail = if t.offset < cursor {
                "overlaps the previous tensor".to_owned()
            } else {
                format!("leaves a {}-byte hole before it", t.offset - cursor)
            };
            return Err(ModelError::Tensor {
                path: path.to_owned(),
                name: t.name.clone(),
                detail,
            });
        }
        cursor = t
            .offset
            .checked_add(t.bytes)
            .ok_or_else(|| ModelError::Tensor {
                path: path.to_owned(),
                name: t.name.clone(),
                detail: "end offset overflows".to_owned(),
            })?;
    }
    if cursor != file_len {
        return Err(header(format!(
            "tensors end at byte {cursor}, but the file has {file_len} bytes"
        )));
    }
    Ok(SafetensorsHeader {
        tensors,
        metadata,
        data_start,
    })
}
