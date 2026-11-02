//! Real model files written by other tools (transformers, llama.cpp), when available.
//!
//! Set `QUBIVEX_REAL_MODEL_FILES` to `:`-separated paths of `.safetensors` or `.gguf`
//! files. Without it this test does nothing, so CI stays offline.

#![allow(clippy::unwrap_used)]

use std::path::Path;

use qubivex_model::TensorIndex;

#[test]
fn real_files_index_cleanly() {
    let Ok(list) = std::env::var("QUBIVEX_REAL_MODEL_FILES") else {
        return;
    };
    for p in list.split(':').filter(|s| !s.is_empty()) {
        let path = Path::new(p);
        let mut idx = TensorIndex::new();
        let r = if p.ends_with(".gguf") {
            idx.add_gguf(path)
        } else {
            idx.add_safetensors(path).map(|_| ())
        };
        if let Err(e) = r {
            panic!("{p}: {e}");
        }
        let size = idx.files()[0].size;
        assert!(!idx.is_empty(), "{p}: no tensors");
        assert!(idx.iter().all(|t| t.tensor.offset + t.tensor.bytes <= size));
        eprintln!(
            "{p}: {} tensors, {} bytes of {} in the file, {:?}",
            idx.len(),
            idx.total_bytes(),
            size,
            idx.bytes_by_dtype()
        );
    }
}

/// Set `QUBIVEX_REAL_MODEL_DIRS` to `:`-separated model directories (each with a
/// `manifest.json`) to load and validate them end to end.
#[test]
fn real_model_dirs_validate() {
    let Ok(list) = std::env::var("QUBIVEX_REAL_MODEL_DIRS") else {
        return;
    };
    let reg = qubivex_model::ArchitectureRegistry::with_builtins();
    for d in list.split(':').filter(|s| !s.is_empty()) {
        let md = match qubivex_model::ModelMetadata::load(Path::new(d), &reg) {
            Ok(m) => m,
            Err(e) => panic!("{d}: {e}"),
        };
        let b = md.bytes();
        assert_eq!(
            b.total,
            md.tensors.total_bytes(),
            "{d}: every tensor has a role"
        );
        eprintln!(
            "{d}: {:?}, {} tensors, {b:?}",
            md.config.dims,
            md.roles.len()
        );
    }
}
