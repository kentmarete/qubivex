# Baseline audit

- Date: 2026-11-03
- Commit audited: `main` at `05ad5eb` (after PR #3)
- Machine: Apple M3 Pro (5 performance + 6 efficiency cores), 18 GB, macOS 27.2, Rust 1.99.1
- Purpose: Phase 0 of the Engineering Improvement Plan (EIP). It records what exists
  before any EIP work starts, maps every EIP recommendation onto the real code, and lists
  the gaps. No code was changed for this audit.

## 1. Summary

The repository is well past the EIP's assumed starting point. It already runs a real MoE
model end to end on CPU:

- `qubivex run` generates text from OLMoE-1B-7B-0125-Instruct Q4_K_M.
- The output agrees with `transformers` to a relative error of 7.2e-7, and with an
  independent f64 reference to 3.3e-7.
- It decodes at about 21 tok/s on the M3 Pro, losslessly.
- Its 160 generated tokens are identical to Colibri's lossless output.

The project has its own design (`docs/architecture/QUBIVEX_DESIGN.md`), its own roadmap
(`docs/architecture/MIGRATION_PLAN.md`, Phases 1–11, currently between Phase 4 and
Phase 5) and eight ADRs. **These number differently from the EIP**: the EIP's
"ADR-003 Tensor storage" collides with the existing "ADR-003 Model registry", and so on.
See [ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md).

How the EIP's eight phases map onto this repository:

| EIP phase | State in this repository |
|---|---|
| 0 Audit | This document |
| 1 Tensor foundations | **Partial.** `Shape` and `DType` are solid and overflow-checked. There are no strides, no general views and no owned tensor type |
| 2 Memory manager | **Designed, not built.** ADR-005 and ADR-008 exist. There is no `qubivex-memory` crate. The expert store is a minimal LRU |
| 3 CPU compute | **Largely done for decode.** Lossless, deterministic GEMV with fused NEON for Q4_K, Q6_K, Q8_0 and bf16. Prefill reuses the decode kernel. No AVX2 kernels |
| 4 Model execution | **Done for MoE** (`transformer_moe`, OLMoE). A dense model, SafeTensors execution and GQA on a real model are not yet tested |
| 5 Graph IR | Not started, and not in the existing design |
| 6 GPU | Not started (the existing roadmap has it at Phase 8) |
| 7 Advanced execution | Not started |
| 8 Server | Not started (the existing roadmap has it at Phase 9) |

## 2. Workspace inventory

One Cargo workspace: edition 2024, `resolver = "3"`, MSRV 1.85. It has 9 library
crates, the CLI and `xtask`. One standalone spike (`spikes/adr-002-tensor-engine`) is
excluded from the workspace.

Lines are counts of `src/**/*.rs`. Tests are `#[test]` functions, plus the proptest blocks
noted in the table.

| Crate | Lines | Tests | Responsibility | EIP equivalent |
|---|---:|---:|---|---|
| `qubivex-core` | 1,137 | 22 (+1 proptest) | IDs, `Bytes`, `DType` and `GgmlType` with exact storage sizes, `Shape` (rank ≤ 8, checked), `CoreError`, typed config | `qubivex-core` (no tensor descriptor, layout or strides yet) |
| `qubivex-hardware` | 2,409 | 25 | CPU, RAM, GPU and storage probing; `doctor --storage` read benchmark | Hardware discovery (the EIP folds it into backends and memory) |
| `qubivex-model` | 3,092 | 24 (+1 proptest) | GGUF and SafeTensors header readers, `TensorIndex`, manifest v1, `ArchitectureRegistry`, `transformer_moe` schema and validation, discovery and import | `qubivex-model` |
| `qubivex-compute` | 212 | 1 | `ComputeBackend` trait (host-only `matmul` and `dequant_row`), `WeightView` | `qubivex-backend` (the EIP's name) |
| `qubivex-backend-cpu` | 836 | 5 | Lossless, deterministic CPU kernels: f32, f16, bf16, Q8_0, Q4_K, Q6_K; NEON fused-dequant dots; fixed 64-lane reduction | `qubivex-cpu` |
| `qubivex-tokenizer` | 553 | 2 | HF `tokenizers` plus sandboxed `minijinja` chat templates | (not in the EIP) |
| `qubivex-expert-stream` | 398 | 7 (incl. 2 loom) | `ExpertStore`: leases, in-flight dedup on a condvar, byte-capacity LRU | MoE expert streaming (EIP §9.4) |
| `qubivex-arch` | 726 | 4 | `transformer_moe` forward pass; scalar RMSNorm, RoPE, softmax, SwiGLU and attention; `KvCache` | `qubivex-models` |
| `qubivex-runtime` | 829 | 13 | `Engine`: load, dense weights in RAM, store-backed experts, rayon pool, sampling, generation | `qubivex-runtime` |
| `apps/cli` | 1,019 | 15 | `qubivex system`, `doctor`, `models`, `run` | (CLI) |
| `xtask` | 1,013 | 9 | Dependency-direction check, model-name guard, fixture generation and check | CI governance |

Crates in the EIP that do not exist: `qubivex-tensor`, `qubivex-memory`, `qubivex-quant`,
`qubivex-ir`, `qubivex-compiler`, `qubivex-kv-cache`, `qubivex-metal`, `qubivex-cuda`,
`qubivex-vulkan`, `qubivex-server` and `qubivex-bench`. Under ADR-001 a crate is created
only in the phase that gives it real, tested code. That matches EIP Rule 3 (no placeholder
modules).

## 3. Dependency graph

Internal dependencies only. Arrows point from a crate to the crates it depends on.
`cargo xtask check-deps` enforces this direction in CI.

```text
apps/cli ──► runtime ──► arch ──────► compute ──► core
   │            │          └────────► model ────► core
   │            ├──► backend-cpu ───► compute
   │            ├──► expert-stream ─► core
   │            ├──► tokenizer ─────► core
   │            ├──► hardware ──────► core
   │            └──► model, compute, core
   └──► core, hardware, model, tokenizer
```

Third-party dependencies of the product crates:
- `serde` and `serde_json` (core, hardware, model, tokenizer, CLI)
- `thiserror`
- `rayon` (backend-cpu, runtime)
- `tokenizers` (pure Rust) and `minijinja` (tokenizer)
- `rustix` (hardware)
- `clap` (CLI)

Dev-only: `proptest` and `loom`. There is no `gemm` crate, no Accelerate binding and no
GPU SDK. Each dependency is declared once in `[workspace.dependencies]` (ADR-001), and
`cargo deny` gates them.

## 4. Functionality against the EIP

### 4.1 Implemented

- Checked shape arithmetic, exact storage sizes for 16 dtypes, and rejection of
  untrusted dimensions (proptest over `storage_bytes`) (EIP §4.1A, partly).
- GGUF and SafeTensors header parsing that validates every byte range against the file
  size (proptests over corrupted containers) (EIP §9.1).
- Manifest-driven architecture registry: OLMoE and a Qwen3-MoE fixture load through one
  `transformer_moe` factory with no code change (EIP §9.2).
- Lossless quantised GEMV with no full dequantisation of the matrix: one row is
  dequantised into a scratch row, and on NEON the dequantisation is fused into the dot
  product (EIP §6.2).
- Deterministic kernels, bitwise identical across thread counts, batch sizes and
  NEON/portable paths. This is stronger than EIP §1.2's principle 9.
- Expert leases that stop an expert in use from being evicted, plus in-flight dedup, with
  loom model checks (EIP §4.1C "leases", §9.4).
- Bitwise logit invariance across expert-cache capacities, tested on fixtures and the
  real model.
- Hardware discovery and storage benchmarking on recorded Linux and macOS fixtures.

### 4.2 Partially implemented

| Item | What exists | What is missing |
|---|---|---|
| Tensor descriptors | `Shape`, `DType`, `StorageLayout`; `WeightView` (2-D, row-major, borrowed) | Strides, offsets, views, transpose, slicing, broadcasting, owned tensors, device association |
| CPU kernels | GEMV/GEMM (`matmul`, any `s`), NEON fused decode | AVX2 and AVX-512 kernels (x86 runs the portable source compiled with `avx2,fma`); a fast prefill GEMM (ADR-002 chose dequantise + `gemm` crate, but it is not built); Q4_K is at about 20–24% of STREAM |
| Transformer ops | RMSNorm, RoPE, softmax, SwiGLU and attention in `qubivex-arch::ops` | They are scalar and single-path, and they sit outside `ComputeBackend`. ADR-007 puts them behind the backend so GPUs can run them |
| KV cache | `KvCache` as `Vec<Vec<Vec<f32>>>`, a single sequence | Contiguous slab, multiple sequences, memory accounting, paging |
| Expert store | Leases, dedup, LRU by bytes | Slab pools, LFRU and hysteresis (ADR-008), bounded I/O queue and I/O threads (ADR-006), prefetch |
| Thread pool | Dedicated rayon pool that the forward pass runs inside | The spinning pool ADR-002 chose (rayon measured 88% of Colibri on attention at 1k context, 11 threads) |
| Benchmarks | A standalone spike harness; a recorded end-to-end Colibri comparison | No in-workspace Criterion benches, no regression job in CI |

### 4.3 Not implemented

Memory planner and ledger, placement policies, NUMA, pinned memory, every GPU backend,
graph IR and compiler, a dense-model architecture, SafeTensors execution on a real model,
continuous batching, paged KV, the server, the metrics endpoint.

## 5. Duplicate and overlapping abstractions

1. **Three descriptions of a 2-D weight.**
   - `qubivex_model::RawTensor`: `{name, dtype, shape, offset, bytes}`, a file extent.
   - A private `qubivex_runtime::engine::Tensor`: `{dtype, rows, cols, bytes: Vec<u8>}`,
     owned dense weights.
   - `qubivex_compute::WeightView`: `{dtype, rows, cols, row_bytes, data: &[u8]}`, the
     kernel view.

   Each recomputes or rechecks the size from `dtype` and dimensions. One shared
   descriptor in `qubivex-core` removes this. This is the main motivation for the tensor
   work (see [TENSOR_ARCHITECTURE.md](../architecture/TENSOR_ARCHITECTURE.md)).
2. **Two "expert layout" notions.** `qubivex_runtime::experts::{ExpertLayout, PartLayout}`
   is implemented. `QUBIVEX_DESIGN.md` §4.5 places `ExpertLayout` behind `ExpertSource` in
   `qubivex-expert-stream`. The code is fine; the design document is out of date.
3. **Activations have no descriptor at all.** They are `&[f32]` slices whose shapes live
   in local variables of the forward pass. That works today, but nothing can describe a
   KV-cache view or a non-contiguous activation to a backend.

## 6. Unsafe code inventory

20 `unsafe` blocks or functions, all in `qubivex-backend-cpu`. That is one of the three
crate kinds ADR-001 allows to opt in.

| File | Count | Purpose | Guard |
|---|---:|---|---|
| `backend-cpu/src/lib.rs` | 2 | Calls the `avx2,fma`-compiled copy of the portable kernel after run-time feature detection | `is_x86_feature_detected!` immediately before the call; `SAFETY:` comment |
| `backend-cpu/src/neon.rs` | 18 | NEON intrinsics for the fused Q4_K, Q6_K, Q8_0 and bf16 dots | NEON is mandatory on aarch64; each block has a `SAFETY:` comment; tested bitwise against the portable twin |

The workspace denies `unsafe_code`, `unsafe_op_in_unsafe_fn` and
`undocumented_unsafe_blocks`. `qubivex-expert-stream` is allowed to opt in but has no
unsafe code today. Miri is not run in CI yet. The NEON code can't run under Miri, but the
portable twins can.

## 7. Test baseline (2026-11-03, `05ad5eb`)

| Check | Result |
|---|---|
| `cargo fmt --all --check` | clean |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo test --workspace` | **123 passed, 0 failed, 0 ignored** (38 test binaries) |
| Loom (`RUSTFLAGS="--cfg loom"`, expert store) | 2 model checks (in CI; not re-run for this audit) |
| Env-gated real-model tests (`QUBIVEX_OLMOE_GGUF_DIR`) | 5 tests that skip without the variable; passing as of PR #1 |
| CI jobs | fmt, clippy, test, doc and `xtask check` on Linux x86-64 and macOS; MSRV 1.85; loom; `cargo deny` |

Coverage gaps worth noting:
- `qubivex-compute` has 1 test.
- `qubivex-tokenizer` has 2 tests.
- `qubivex-arch` has 4 unit tests, but most of its checking comes from the runtime's
  oracle and reference tests.
- There are no stress tests (repeated generation memory growth, cancellation), because
  there is no cancellation path yet.

## 8. Benchmark baseline

Recorded results are carried forward unchanged and were not re-measured for this audit.
Sources: `docs/performance/ADR-002-SPIKE.md` and `COLIBRI-OLMOE-BENCHMARK.md`.

**End-to-end decode**, OLMoE-1B-7B Q4_K_M, 5 threads, warm, all experts resident, 160
greedy tokens:

| Engine | Numerics | tok/s | Peak RSS |
|---|---|---|---|
| QUBIVEX `8c5e20f` | lossless | 20.7–21.0 | 3.91–3.94 GB |
| Colibri `0a5504a1` default | lossy (integer dot) | 17.4–17.8 | 3.38–3.55 GB |
| Colibri `GQ_EXACT=1` | lossless | 3.0 | 3.22–3.44 GB |

**Kernel spike**, OLMoE expert shape at S=1, cold weights, 11 threads. STREAM reference:
116 GB/s.

| dtype | GB/s | % STREAM | EIP gate (§13.2) |
|---|---:|---:|---|
| f32 | 122.7 | 105% | ≥ 90%: **met** |
| bf16 | 117.5 | 101% | ≥ 90%: **met** |
| int8 row-scaled | 69.4 | 60% | ≥ 70% (quantised): not met |
| Q8_0 | 59.4 | 51% | ≥ 70%: not met |
| Q4_K | 23.2 | 20% | ≥ 70%: **far off** (compute-bound on integer widening) |

These spike numbers predate the fused NEON kernels in PR #2. The kernel figures need
re-measuring inside the workspace before they serve as a regression baseline. That is
the first benchmarking task below.

**Prefill:** the spike's dequantise + `gemm` path is not in the product. Prefill
currently runs the decode kernel with `s > 1`. Accelerate's f32 GEMM measured about 3×
faster than `gemm` on macOS. The EIP gate "within 15% of a selected optimised library" is
therefore unmet by a wide margin.

## 9. Documentation drift found during the audit

| Document | Drift |
|---|---|
| ADR-001, 003, 004, 006, 007, 008 | Still "Proposed (Phase 0)", although 001, 003, 004 and (minimally) 006 are implemented |
| `IMPLEMENTATION_STATUS.md` → "Architectural decisions" | Says ADR-002 is preliminary; ADR-002 is Accepted |
| `MIGRATION_PLAN.md` header and §2 | "Awaiting review before Phase 1"; describes the old Intel machine (a later section of `IMPLEMENTATION_STATUS.md` supersedes it) |
| `QUBIVEX_DESIGN.md` §4.4 | Describes the eventual `ComputeBackend`, not the implemented one, and doesn't say so |
| ADR-002 | Accepts a spinning pool and dequantise + `gemm` prefill; neither is built, and no status line says so |

## 10. Recommendations

In priority order. Each item is small enough to land as its own PR.

1. **Adopt one roadmap.** Fold the EIP into `MIGRATION_PLAN.md` instead of running two
   phase numberings side by side. Most EIP phases map onto existing phases (see
   ARCHITECTURE_DECISIONS §3).
2. **Number new ADRs from 009** in `docs/decisions/`, and refer to existing ADRs where the
   EIP proposes an equivalent (EIP ADR-004 memory → ADR-005; EIP ADR-005 backends →
   ADR-007).
3. **Tensor descriptors (EIP Phase 1, scoped):** add `TensorDesc` (byte strides, offset)
   to `qubivex-core` and typed views to `qubivex-compute`. Then retire the runtime's
   private `Tensor` and build `WeightView` from the shared descriptor. Kernel signatures
   stay unchanged, and decode tok/s must not regress.
4. **In-workspace benchmarks:** a Criterion bench for the existing kernels, and a
   `qubivex bench`-style harness for decode and prefill that records the EIP §13.3 fields.
   This has to come before any further kernel work (EIP Rule 4).
5. **Reconcile accepted-but-unbuilt decisions:** either build or explicitly defer the
   spinning pool and the `gemm` prefill (ADR-002), with a status line in the ADR.
6. **Then the existing Phase 5 (memory planner and ledger),** which the EIP also ranks
   P0.
7. **Update the drifting documents** in §9 as part of the same PRs that touch them.

## 11. Migration risks

| Risk | Impact | Mitigation |
|---|---|---|
| Replacing `WeightView` or kernel signatures regresses decode speed | High: 21 tok/s is the project's headline | Keep kernel signatures; build `WeightView` from the descriptor at load time; benchmark decode before and after |
| A general tensor library grows ahead of its consumers | Medium: dead code, churn | Build only the descriptor operations the tests and current consumers need; no elementwise broadcast kernels until an op needs them |
| Byte-to-`f32` reinterpretation needs `unsafe` or a new dependency | Medium | Typed storage (`Arc<[T]>`) keeps everything in safe Rust; see the tensor design §5 |
| Two roadmaps and two ADR numberings confuse future work | Medium | Recommendations 1 and 2 |
| Per-op CPU/GPU scheduling by live utilisation (EIP §11) breaks the bitwise-reproducibility invariant | High: invariant 1 is a core product claim | Device assignment is fixed per execution plan, not by live load; see ARCHITECTURE_DECISIONS §2.6 |
| A lossy "Performance" precision profile (EIP §6.3) contradicts the existing "no lossy profile" decision | Medium | Needs an owner decision; see ARCHITECTURE_DECISIONS §2.1 |
