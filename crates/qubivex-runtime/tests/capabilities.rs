//! The weight formats architectures declare are exactly the formats every backend runs.

use std::path::Path;

use qubivex_core::{DType, GgmlType};
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{BackendKind, refuse_weight_formats};

#[test]
fn declared_formats_match_every_backend() {
    for a in ArchitectureRegistry::with_builtins().list_supported() {
        for d in DType::all() {
            for b in BackendKind::ALL {
                assert_eq!(
                    a.capabilities.accepts(d),
                    b.supports(d),
                    "{} declares {} as {}, {} {}",
                    a.id,
                    d.name(),
                    a.capabilities.accepts(d),
                    b.name(),
                    b.supports(d)
                );
            }
        }
    }
}

/// A backend without a kernel for one of the model's formats is refused, naming the
/// format and what the backend runs; every real backend accepts the fixture.
#[test]
fn a_backend_lacking_a_weight_format_is_refused() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/models/olmoe-tiny-gguf");
    let meta = ModelMetadata::load(&dir, &ArchitectureRegistry::with_builtins()).unwrap();
    let q8 = DType::Ggml(GgmlType::Q8_0);
    assert!(
        meta.roles
            .keys()
            .any(|n| meta.tensors.get(n).is_some_and(|t| t.tensor.dtype == q8)),
        "the fixture stores some weights as q8_0"
    );
    let err = refuse_weight_formats(&meta, "fake", |d| d != q8)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("the fake backend has no kernel for q8_0 weights (it runs f32, f16, bf16"),
        "{err}"
    );
    for b in BackendKind::ALL {
        let name = b.name();
        refuse_weight_formats(&meta, name, |d| b.supports(d)).unwrap();
    }
}
