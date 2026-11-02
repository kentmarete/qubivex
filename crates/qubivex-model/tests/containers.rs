//! Container readers against well-formed, malformed and randomly corrupted files.

// Helpers outside `#[test]` fns are not covered by clippy.toml's test exemption.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation
)]

use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::Path;

use proptest::prelude::*;
use qubivex_core::{DType, GgmlType};
use qubivex_model::write::{self, TensorData};
use qubivex_model::{GgufValue, ModelError, TensorIndex, gguf, safetensors};

const P: &str = "test.bin";

fn st(bytes: &[u8]) -> Result<safetensors::SafetensorsHeader, ModelError> {
    safetensors::parse(Cursor::new(bytes), bytes.len() as u64, Path::new(P))
}

fn gg(bytes: &[u8]) -> Result<gguf::GgufHeader, ModelError> {
    gguf::parse(Cursor::new(bytes), bytes.len() as u64, Path::new(P))
}

fn err_text<T: std::fmt::Debug>(r: Result<T, ModelError>) -> String {
    r.expect_err("must be rejected").to_string()
}

fn sample_st() -> Vec<u8> {
    let a = vec![1u8; 2 * 3 * 2];
    let b = vec![2u8; 4 * 4];
    let mut meta = BTreeMap::new();
    meta.insert("format".to_owned(), "pt".to_owned());
    write::safetensors(
        &[
            TensorData {
                name: "a.weight",
                dtype: DType::BF16,
                dims: &[2, 3],
                data: &a,
            },
            TensorData {
                name: "b.bias",
                dtype: DType::F32,
                dims: &[4],
                data: &b,
            },
        ],
        &meta,
    )
}

/// A safetensors file with a hand-written header, padded data of `data_len` bytes.
fn raw_st(header: &str, data_len: usize) -> Vec<u8> {
    let mut v = (header.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(header.as_bytes());
    v.resize(v.len() + data_len, 0);
    v
}

#[test]
fn safetensors_round_trip() {
    let f = sample_st();
    let h = st(&f).unwrap();
    assert_eq!(h.metadata["format"], "pt");
    assert_eq!(h.tensors.len(), 2);
    let a = &h.tensors[0];
    assert_eq!(
        (a.name.as_str(), a.dtype, a.shape.dims(), a.bytes),
        ("a.weight", DType::BF16, &[2, 3][..], 12)
    );
    assert_eq!(a.offset, h.data_start);
    assert_eq!(h.tensors[1].offset, h.data_start + 12);
    assert_eq!(h.tensors[1].offset + 16, f.len() as u64);
}

#[test]
fn safetensors_rejections() {
    let cases: &[(&str, usize, &str)] = &[
        (
            r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
            4,
            "needs 8",
        ),
        (
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}"#,
            6,
            "overlaps",
        ),
        (
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
            8,
            "hole",
        ),
        (
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            12,
            "tensors end at byte",
        ),
        (
            r#"{"x":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}}"#,
            8,
            "unsupported dtype I64",
        ),
        (
            r#"{"x":{"dtype":"F32","shape":[1,1,1,1,1,1,1,1,1],"data_offsets":[0,4]}}"#,
            4,
            "rank 9",
        ),
        (
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#,
            4,
            "data_offsets",
        ),
        (
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":1}}"#,
            4,
            "bad entry",
        ),
        (
            r#"{"x":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,4]}}"#,
            4,
            "overflow",
        ),
        (r#"[1,2]"#, 0, "not a JSON object"),
        (r#"{"x":"#, 0, "bad JSON"),
    ];
    for (header, data, want) in cases {
        let t = err_text(st(&raw_st(header, *data)));
        assert!(t.contains(want), "{header}: {t}");
    }
    // Header length larger than the file.
    let mut f = sample_st();
    f[..8].copy_from_slice(&(u64::MAX - 3).to_le_bytes());
    assert!(err_text(st(&f)).contains("exceeds the file"));
    // Truncated file.
    let f = sample_st();
    assert!(st(&f[..f.len() - 1]).is_err());
}

fn sample_gguf() -> Vec<u8> {
    let q4k = vec![7u8; 4 * 144]; // 4 rows of one 256-element super-block
    let f32s = vec![3u8; 3 * 8 * 4];
    write::gguf(
        &[
            ("general.architecture", GgufValue::String("toyarch".into())),
            ("general.alignment", GgufValue::U32(32)),
            ("toyarch.block_count", GgufValue::U32(2)),
            ("toyarch.rope.freq_base", GgufValue::F32(10_000.0)),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array(vec![
                    GgufValue::String("<s>".into()),
                    GgufValue::String("a".into()),
                ]),
            ),
        ],
        &[
            TensorData {
                name: "blk.0.ffn_up.weight",
                dtype: DType::Ggml(GgmlType::Q4_K),
                dims: &[4, 256],
                data: &q4k,
            },
            TensorData {
                name: "output_norm.weight",
                dtype: DType::F32,
                dims: &[3, 8],
                data: &f32s,
            },
        ],
    )
}

#[test]
fn gguf_round_trip() {
    let f = sample_gguf();
    let h = gg(&f).unwrap();
    assert_eq!(h.version, 3);
    assert_eq!(h.alignment, 32);
    assert_eq!(h.metadata.architecture(), Some("toyarch"));
    assert_eq!(
        h.metadata
            .arch_value("block_count")
            .and_then(GgufValue::as_u64),
        Some(2)
    );
    assert_eq!(
        h.metadata
            .arch_value("rope.freq_base")
            .and_then(GgufValue::as_f64),
        Some(10_000.0)
    );
    let GgufValue::Array(toks) = h.metadata.get("tokenizer.ggml.tokens").unwrap() else {
        panic!()
    };
    assert_eq!(toks.len(), 2);
    let q = h
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_up.weight")
        .unwrap();
    assert_eq!(q.shape.dims(), &[4, 256], "dims are reversed to row-major");
    assert_eq!(q.bytes, 4 * 144);
    assert_eq!(q.offset % 32, 0);
    let n = h
        .tensors
        .iter()
        .find(|t| t.name == "output_norm.weight")
        .unwrap();
    assert_eq!((n.dtype, n.bytes), (DType::F32, 96));
    assert!(n.offset + n.bytes <= f.len() as u64);
}

/// A GGUF file built by hand, for malformed cases: kvs as (key, type, raw value bytes),
/// tensors as (name, dims innermost first, type, offset), then `data` bytes.
fn raw_gguf(
    version: u32,
    kvs: &[(&str, u32, Vec<u8>)],
    tensors: &[(&str, &[u64], u32, u64)],
    data: usize,
) -> Vec<u8> {
    let mut v = b"GGUF".to_vec();
    v.extend_from_slice(&version.to_le_bytes());
    v.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    v.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    let s = |v: &mut Vec<u8>, x: &str| {
        v.extend_from_slice(&(x.len() as u64).to_le_bytes());
        v.extend_from_slice(x.as_bytes());
    };
    for (k, t, b) in kvs {
        s(&mut v, k);
        v.extend_from_slice(&t.to_le_bytes());
        v.extend_from_slice(b);
    }
    for (name, dims, ty, off) in tensors {
        s(&mut v, name);
        v.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in *dims {
            v.extend_from_slice(&d.to_le_bytes());
        }
        v.extend_from_slice(&ty.to_le_bytes());
        v.extend_from_slice(&off.to_le_bytes());
    }
    while v.len() % 32 != 0 {
        v.push(0);
    }
    v.resize(v.len() + data, 0);
    v
}

#[test]
fn gguf_rejections() {
    let ok = raw_gguf(3, &[], &[("t", &[8], 0, 0)], 32);
    assert!(gg(&ok).is_ok());

    let mut bad_magic = ok.clone();
    bad_magic[0] = b'X';
    assert!(err_text(gg(&bad_magic)).contains("bad magic"));
    assert!(err_text(gg(&raw_gguf(1, &[], &[], 0))).contains("unsupported GGUF version 1"));
    assert!(err_text(gg(&raw_gguf(3u32.swap_bytes(), &[], &[], 0))).contains("big-endian"));
    assert!(err_text(gg(&ok[..6])).contains("end of file"));
    assert!(err_text(gg(&ok[..20])).contains("tensor count"));

    // A string length far beyond the file.
    let mut huge = (u64::MAX / 2).to_le_bytes().to_vec();
    huge.extend_from_slice(b"x");
    assert!(err_text(gg(&raw_gguf(3, &[("k", 8, huge)], &[], 0))).contains("larger than the file"));
    // A tensor count far beyond the file.
    let mut many = ok.clone();
    many[8..16].copy_from_slice(&(1u64 << 40).to_le_bytes());
    assert!(err_text(gg(&many)).contains("tensor count"));
    // An array claiming more elements than the file holds.
    let mut arr = 4u32.to_le_bytes().to_vec();
    arr.extend_from_slice(&(1u64 << 30).to_le_bytes());
    assert!(err_text(gg(&raw_gguf(3, &[("k", 9, arr)], &[], 0))).contains("array length"));
    // Arrays nested three deep.
    let mut nested = 9u32.to_le_bytes().to_vec();
    nested.extend_from_slice(&1u64.to_le_bytes());
    nested.extend_from_slice(&9u32.to_le_bytes());
    nested.extend_from_slice(&1u64.to_le_bytes());
    nested.extend_from_slice(&[0u8; 16]);
    assert!(err_text(gg(&raw_gguf(3, &[("k", 9, nested)], &[], 0))).contains("nested"));
    assert!(
        err_text(gg(&raw_gguf(3, &[("k", 77, vec![])], &[], 0))).contains("unknown value type 77")
    );
    assert!(err_text(gg(&raw_gguf(3, &[("k", 7, vec![2])], &[], 0))).contains("bool byte"));
    let dup = raw_gguf(3, &[("k", 0, vec![1]), ("k", 0, vec![2])], &[], 0);
    assert!(err_text(gg(&dup)).contains("duplicate metadata key"));
    let align = raw_gguf(
        3,
        &[("general.alignment", 4, 24u32.to_le_bytes().to_vec())],
        &[],
        0,
    );
    assert!(err_text(gg(&align)).contains("power of two"));

    assert!(
        err_text(gg(&raw_gguf(3, &[], &[("t", &[8], 99, 0)], 32)))
            .contains("unknown GGML type id 99")
    );
    assert!(err_text(gg(&raw_gguf(3, &[], &[("t", &[8], 0, 4)], 64))).contains("aligned"));
    assert!(err_text(gg(&raw_gguf(3, &[], &[("t", &[64], 0, 0)], 32))).contains("past the end"));
    let overlap = raw_gguf(3, &[], &[("a", &[16], 0, 0), ("b", &[8], 0, 32)], 64);
    assert!(err_text(gg(&overlap)).contains("overlaps"));
    let dup_t = raw_gguf(3, &[], &[("a", &[8], 0, 0), ("a", &[8], 0, 32)], 64);
    assert!(err_text(gg(&dup_t)).contains("duplicate tensor"));
    // Q4_K needs the innermost dim to be a multiple of 256.
    assert!(
        err_text(gg(&raw_gguf(3, &[], &[("q", &[100, 1], 12, 0)], 160))).contains("not a multiple")
    );
    let rank = raw_gguf(3, &[], &[("r", &[1; 9], 0, 0)], 32);
    assert!(err_text(gg(&rank)).contains("9 dimensions"));
}

#[test]
fn index_across_files() {
    let dir = std::env::temp_dir().join(format!("qubivex-model-index-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let one = dir.join("model-00001-of-00002.safetensors");
    let two = dir.join("model-00002-of-00002.safetensors");
    let g = dir.join("model.gguf");
    std::fs::write(&one, sample_st()).unwrap();
    let c = vec![0u8; 8];
    std::fs::write(
        &two,
        write::safetensors(
            &[TensorData {
                name: "c",
                dtype: DType::F16,
                dims: &[4],
                data: &c,
            }],
            &BTreeMap::new(),
        ),
    )
    .unwrap();
    std::fs::write(&g, sample_gguf()).unwrap();

    let mut idx = TensorIndex::new();
    idx.add_safetensors(&one).unwrap();
    idx.add_safetensors(&two).unwrap();
    assert_eq!(idx.len(), 3);
    assert_eq!(idx.total_bytes(), 12 + 16 + 8);
    assert_eq!(idx.get("c").unwrap().file, 1);
    assert_eq!(idx.bytes_by_dtype()[&DType::BF16], 12);

    // A second copy of a shard: every name collides; the index must not change.
    let before = idx.clone();
    let e = idx.add_safetensors(&one).unwrap_err();
    assert!(matches!(e, ModelError::DuplicateTensor { .. }), "{e}");
    assert_eq!(idx, before);

    let mut gi = TensorIndex::new();
    gi.add_gguf(&g).unwrap();
    assert_eq!(
        gi.gguf_metadata().and_then(|m| m.architecture()),
        Some("toyarch")
    );
    assert!(gi.add_gguf(&g).is_err(), "a second GGUF file is rejected");
    assert!(matches!(
        TensorIndex::new().add_gguf(&dir.join("missing.gguf")),
        Err(ModelError::Io { .. })
    ));

    std::fs::remove_dir_all(&dir).unwrap();
}

/// Accepted headers stay inside the file; nothing panics.
fn check_bounds(tensors: &[qubivex_model::RawTensor], len: usize) {
    for t in tensors {
        assert!(
            t.offset
                .checked_add(t.bytes)
                .is_some_and(|e| e <= len as u64),
            "{t:?}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn corrupted_safetensors_never_panic(flips in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..8), cut in any::<prop::sample::Index>()) {
        let mut f = sample_st();
        for (i, b) in &flips {
            let i = i.index(f.len());
            f[i] = *b;
        }
        let f = &f[..cut.index(f.len() + 1)];
        if let Ok(h) = st(f) {
            check_bounds(&h.tensors, f.len());
        }
    }

    #[test]
    fn corrupted_gguf_never_panic(flips in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..8), cut in any::<prop::sample::Index>()) {
        let mut f = sample_gguf();
        for (i, b) in &flips {
            let i = i.index(f.len());
            f[i] = *b;
        }
        let f = &f[..cut.index(f.len() + 1)];
        if let Ok(h) = gg(f) {
            check_bounds(&h.tensors, f.len());
        }
    }

    #[test]
    fn random_bytes_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        let _ = st(&bytes);
        let mut g = b"GGUF\x03\0\0\0".to_vec();
        g.extend_from_slice(&bytes);
        let _ = gg(&g);
    }
}
