//! Expert extents of the fixtures (and the real GGUF, when available), read through
//! the store at several capacities.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::path::Path;
use std::sync::Arc;

use qubivex_core::{ExpertIdx, LayerIdx};
use qubivex_expert_stream::ExpertStore;
use qubivex_model::{ArchitectureRegistry, ExpertPart, ModelMetadata, TensorRole};
use qubivex_runtime::ModelExpertSource;

fn load(dir: &Path) -> ModelMetadata {
    ModelMetadata::load(dir, &ArchitectureRegistry::with_builtins()).unwrap()
}

/// Reads tensor bytes directly, for comparison.
fn tensor_bytes(meta: &ModelMetadata, name: &str) -> Vec<u8> {
    let t = meta.tensors.get(name).unwrap();
    let f = std::fs::read(&meta.tensors.files()[t.file].path).unwrap();
    f[t.tensor.offset as usize..(t.tensor.offset + t.tensor.bytes) as usize].to_vec()
}

fn check_model(meta: &ModelMetadata) {
    let src = Arc::new(ModelExpertSource::new(meta).unwrap());
    let d = meta.config.dims;
    assert_eq!(src.len(), d.layers as usize * d.experts as usize);
    assert_eq!(src.total_bytes(), meta.bytes().routed_experts);
    let largest = meta.bytes().largest_expert;
    // Expected bytes of each expert, assembled from the tensors in part order.
    let mut want = std::collections::BTreeMap::new();
    for (name, role) in &meta.roles {
        let bytes = tensor_bytes(meta, name);
        match *role {
            TensorRole::RoutedExpert {
                layer,
                expert,
                part,
            } => {
                want.entry((layer, expert))
                    .or_insert_with(Vec::new)
                    .push((part, bytes));
            }
            TensorRole::RoutedExpertStack {
                layer,
                part,
                experts,
            } => {
                let slice = bytes.len() / experts as usize;
                for e in 0..experts {
                    let s = bytes[e as usize * slice..(e as usize + 1) * slice].to_vec();
                    want.entry((layer, ExpertIdx(e)))
                        .or_insert_with(Vec::new)
                        .push((part, s));
                }
            }
            _ => {}
        }
    }
    for capacity in [largest, largest * 3, src.total_bytes()] {
        let store = ExpertStore::new(src.clone(), capacity);
        for (k, parts) in &mut want {
            parts.sort_by_key(|p| p.0);
            let expect: Vec<u8> = parts.iter().flat_map(|p| p.1.clone()).collect();
            let lease = store.acquire(*k).unwrap();
            assert_eq!(
                lease.bytes(),
                expect.as_slice(),
                "{k:?} capacity {capacity}"
            );
            let layout = src.layout(k.0, k.1).unwrap();
            assert_eq!(layout.size, expect.len());
            assert!(layout.part(ExpertPart::Down).is_some());
        }
    }
}

#[test]
fn fixture_experts_read_back_exactly() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/models");
    for id in ["olmoe-tiny", "qwen3-moe-tiny", "olmoe-tiny-gguf"] {
        check_model(&load(&root.join(id)));
    }
    // The stacked layout's parts: gate-up then down.
    let meta = load(&root.join("qwen3-moe-tiny"));
    let src = ModelExpertSource::new(&meta).unwrap();
    let l = src.layout(LayerIdx(1), ExpertIdx(7)).unwrap();
    let kinds: Vec<ExpertPart> = l.parts.iter().map(|p| p.part).collect();
    assert_eq!(kinds, [ExpertPart::GateUp, ExpertPart::Down]);
    assert_eq!((l.parts[0].rows, l.parts[0].cols), (64, 64));
}

/// Set QUBIVEX_OLMOE_GGUF_DIR to check the real model's expert extents.
#[test]
fn real_gguf_expert_extents() {
    let Ok(dir) = std::env::var("QUBIVEX_OLMOE_GGUF_DIR") else {
        return;
    };
    let meta = load(Path::new(&dir));
    let src = Arc::new(ModelExpertSource::new(&meta).unwrap());
    assert_eq!(src.len(), 16 * 64);
    assert_eq!(src.total_bytes(), meta.bytes().routed_experts);
    let store = ExpertStore::new(src.clone(), meta.bytes().largest_expert * 4);
    for l in [0u32, 7, 15] {
        for e in [0u32, 31, 63] {
            let lease = store.acquire((LayerIdx(l), ExpertIdx(e))).unwrap();
            let layout = src.layout(LayerIdx(l), ExpertIdx(e)).unwrap();
            assert_eq!(lease.bytes().len(), layout.size);
            let kinds: Vec<ExpertPart> = layout.parts.iter().map(|p| p.part).collect();
            assert_eq!(kinds, [ExpertPart::Gate, ExpertPart::Up, ExpertPart::Down]);
        }
    }
    assert!(store.stats().evictions > 0);
}
