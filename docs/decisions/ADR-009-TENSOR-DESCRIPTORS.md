# ADR-009: Tensor descriptors with byte strides; typed, Arc-shared host storage

- Status: **Accepted (2026-11-03)**, implemented on branch `phase-0-baseline-audit`
- Date: 2026-11-03
- Design: [`docs/architecture/TENSOR_ARCHITECTURE.md`](../architecture/TENSOR_ARCHITECTURE.md)

## Context
Three types described a 2-D weight, each re-deriving its size from dtype and
dimensions: `RawTensor` (model), a private `Tensor` (runtime) and `WeightView`
(compute). Nothing could describe a strided view, a KV-cache slice or a device buffer.
The Engineering Improvement Plan (EIP) Phase 1 asks for descriptors, strides, offsets,
views and ownership rules (BASELINE_AUDIT §5).

## Decision
- **`qubivex_core::TensorDesc`**: dtype, `Shape`, byte strides and a byte offset. It is
  `Copy` and pure. Byte strides address blocked and row-scaled dtypes as well as
  elementwise ones.
- **Packed dtypes** (blocked, row-scaled) keep their innermost dimension whole and
  innermost:
  - it can't be permuted or selected;
  - for blocked dtypes it can be sliced only on block boundaries;
  - for row-scaled dtypes it can't be sliced at all.
- Views (`permute`, `transpose`, `narrow`, `select`, `broadcast_to`, `reshape`) never
  copy. `reshape` uses PyTorch's stride rule and returns `ReshapeNeedsCopy` instead of
  copying silently.
- **`qubivex_compute::tensor`**:
  - `TensorView<'a, T>` and `TensorViewMut<'a, T>` borrow storage, so the borrow checker
    stops a view from outliving it.
  - `Tensor<T>` owns `Arc<[T]>`, shared by clones and views, and is mutable only while
    unique.
  - Storage is typed (`f32`, `u16` or `u8`), so no `unsafe` and no new dependency.
  - `to_contiguous` is the only copy.
- `WeightView` stays the kernel type, and a contiguous rank-2 byte view converts into
  it. Kernel signatures are unchanged.
- No `qubivex-tensor` crate (ADR-001). No device field on the descriptor: the storage
  carries the device.
- For asynchronous device execution (ADR-011, with the first GPU backend), an in-flight
  operation owns reference-counted handles to its buffers until its completion event.

## Consequences
- One validated description per tensor. Invalid descriptors are typed errors.
- The runtime's dense weights now use `Tensor<u8>`. Decode output is byte-identical, and
  decode speed did not regress: 21.06–21.33 tok/s against 19.98–20.81 tok/s before, on
  OLMoE Q4_K_M, 160 tokens, 3 runs each.
- External-scale quantisation formats (GPTQ, AWQ, FP8) will be separate tensors linked
  by role, not descriptor fields.
