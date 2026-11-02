//! Minimal SafeTensors and GGUF writers, for tests and tiny fixture models.
//!
//! These build whole files in memory, so they are meant for small models only.

use std::collections::BTreeMap;

use qubivex_core::DType;

use crate::GgufValue;

/// One tensor to write: name, dtype, row-major dims, and exactly its bytes.
#[derive(Clone, Debug)]
pub struct TensorData<'a> {
    /// Name.
    pub name: &'a str,
    /// Element type.
    pub dtype: DType,
    /// Row-major dims.
    pub dims: &'a [u64],
    /// The data; must be exactly the dtype's storage size for `dims`.
    pub data: &'a [u8],
}

fn st_dtype(d: DType) -> Option<&'static str> {
    match d {
        DType::F32 => Some("F32"),
        DType::F16 => Some("F16"),
        DType::BF16 => Some("BF16"),
        _ => None,
    }
}

/// A SafeTensors file. Tensors are laid out in the given order.
///
/// # Panics
/// If a dtype has no SafeTensors spelling (only f32, f16 and bf16 do).
#[must_use]
pub fn safetensors(tensors: &[TensorData<'_>], metadata: &BTreeMap<String, String>) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        header.insert("__metadata__".into(), serde_json::json!(metadata));
    }
    let mut off = 0u64;
    for t in tensors {
        let end = off + t.data.len() as u64;
        #[allow(clippy::expect_used)] // writer for tests and fixtures; a bad dtype is a caller bug
        let dtype = st_dtype(t.dtype).expect("dtype has a SafeTensors name");
        header.insert(
            t.name.into(),
            serde_json::json!({ "dtype": dtype, "shape": t.dims, "data_offsets": [off, end] }),
        );
        off = end;
    }
    let mut json = serde_json::Value::Object(header).to_string().into_bytes();
    // Pad with spaces to 8 bytes, as the reference writer does.
    while json.len() % 8 != 0 {
        json.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + json.len() + usize::try_from(off).unwrap_or(0));
    out.extend_from_slice(&(json.len() as u64).to_le_bytes());
    out.extend_from_slice(&json);
    for t in tensors {
        out.extend_from_slice(t.data);
    }
    out
}

fn gguf_type(v: &GgufValue) -> u32 {
    match v {
        GgufValue::U8(_) => 0,
        GgufValue::I8(_) => 1,
        GgufValue::U16(_) => 2,
        GgufValue::I16(_) => 3,
        GgufValue::U32(_) => 4,
        GgufValue::I32(_) => 5,
        GgufValue::F32(_) => 6,
        GgufValue::Bool(_) => 7,
        GgufValue::String(_) => 8,
        GgufValue::Array(_) => 9,
        GgufValue::U64(_) => 10,
        GgufValue::I64(_) => 11,
        GgufValue::F64(_) => 12,
    }
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_value(out: &mut Vec<u8>, v: &GgufValue) {
    match v {
        GgufValue::U8(x) => out.push(*x),
        GgufValue::I8(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::Bool(x) => out.push(u8::from(*x)),
        GgufValue::String(s) => put_str(out, s),
        GgufValue::Array(a) => {
            // An empty array is written as u8 elements.
            out.extend_from_slice(&a.first().map_or(0, gguf_type).to_le_bytes());
            out.extend_from_slice(&(a.len() as u64).to_le_bytes());
            for e in a {
                put_value(out, e);
            }
        }
        GgufValue::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
    }
}

/// The GGUF type id of a dtype.
///
/// # Panics
/// If the dtype has no GGUF id (`i8_row_scaled`).
fn ggml_id(d: DType) -> u32 {
    match d {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 30,
        DType::Ggml(t) => t.ggml_id(),
        #[allow(clippy::panic)] // writer for tests and fixtures; a bad dtype is a caller bug
        _ => panic!("{d} has no GGUF type id"),
    }
}

/// A GGUF v3 file with the default 32-byte alignment. Tensors are laid out in order.
#[must_use]
pub fn gguf(metadata: &[(&str, GgufValue)], tensors: &[TensorData<'_>]) -> Vec<u8> {
    const ALIGN: u64 = crate::gguf::DEFAULT_ALIGNMENT;
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (k, v) in metadata {
        put_str(&mut out, k);
        out.extend_from_slice(&gguf_type(v).to_le_bytes());
        put_value(&mut out, v);
    }
    let mut off = 0u64;
    let mut offsets = Vec::with_capacity(tensors.len());
    for t in tensors {
        put_str(&mut out, t.name);
        out.extend_from_slice(
            &u32::try_from(t.dims.len())
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
        for d in t.dims.iter().rev() {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&ggml_id(t.dtype).to_le_bytes());
        out.extend_from_slice(&off.to_le_bytes());
        offsets.push(off);
        off = (off + t.data.len() as u64).div_ceil(ALIGN) * ALIGN;
    }
    let pad = |out: &mut Vec<u8>| {
        while out.len() as u64 % ALIGN != 0 {
            out.push(0);
        }
    };
    pad(&mut out);
    let base = out.len();
    for (t, o) in tensors.iter().zip(offsets) {
        out.resize(base + usize::try_from(o).unwrap_or(0), 0);
        out.extend_from_slice(t.data);
    }
    pad(&mut out);
    out
}
