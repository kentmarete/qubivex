//! Dequantization against gguf-py, llama.cpp's reference Python implementation
//! (`fixtures/oracle/ggml_dequant.json`, from `fixtures/oracle/ggml_dequant.py`).

#![allow(clippy::unwrap_used)]

use qubivex_backend_cpu::formats::dequant_row;
use qubivex_core::{DType, GgmlType};
use serde_json::Value;

#[test]
fn ggml_dequant_matches_gguf_py_bit_for_bit() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/oracle/ggml_dequant.json"
    );
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cases = v["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    for c in cases {
        let dtype = match c["dtype"].as_str().unwrap() {
            "q8_0" => DType::Ggml(GgmlType::Q8_0),
            "q4_k" => DType::Ggml(GgmlType::Q4_K),
            "q6_k" => DType::Ggml(GgmlType::Q6_K),
            other => panic!("unexpected {other}"),
        };
        let hex = c["bytes_hex"].as_str().unwrap();
        let bytes: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        let want: Vec<u32> = c["f32_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| u32::try_from(b.as_u64().unwrap()).unwrap())
            .collect();
        let cols = usize::try_from(c["cols"].as_u64().unwrap()).unwrap();
        let mut out = vec![0f32; cols];
        dequant_row(dtype, &bytes, &mut out);
        let diffs: Vec<usize> = (0..cols).filter(|&i| out[i].to_bits() != want[i]).collect();
        assert!(
            diffs.is_empty(),
            "{dtype}: {} of {cols} values differ; first at {}: {} vs {}",
            diffs.len(),
            diffs[0],
            out[diffs[0]],
            f32::from_bits(want[diffs[0]])
        );
    }
}
