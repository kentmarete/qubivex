# Tensor architecture

- Date: 2026-11-03
- Status: **Accepted (ADR-009) and implemented, 2026-11-03.** The code is
  `qubivex-core/src/tensor_desc.rs` and `qubivex-compute/src/tensor.rs`.
- Scope: tensor descriptors, strides, storage offsets, views, ownership, buffer lifetime
  and quantisation metadata.

## 1. Goals and non-goals

**Goals**
1. **One descriptor for every tensor.** Weights in a file, weights in RAM, expert slabs,
   activations, KV-cache slices and (later) device buffers all use it. It replaces the
   three ad-hoc 2-D descriptions found in the audit (BASELINE_AUDIT §5).
2. **Zero-copy views:** transpose, permute, slice, select, broadcast, and reshape where
   the strides allow it.
3. **Every size and offset computed with checked arithmetic.** An invalid descriptor is a
   typed error, never a panic or a wrap. This is the same rule as `Shape` and
   `DType::storage_bytes`, which already apply it to untrusted model headers.
4. **Safe Rust.** No byte-to-`f32` reinterpretation, no new `unsafe` and no new
   dependency.
5. **No change to the kernel hot path.** `WeightView` and the `ComputeBackend::matmul`
   signature stay. Decode speed must not regress.

**Non-goals for this step** (EIP Rule 10, ADR-001):
- No elementwise or broadcast arithmetic kernels. Broadcasting is a descriptor operation;
  kernels that consume it arrive with the first op that needs one.
- No `qubivex-tensor` crate. The descriptor lives in `qubivex-core`, and storage lives in
  `qubivex-compute`, next to `WeightView`.
- No device buffers. Their contract is fixed here (§6), and their code arrives with the
  first GPU backend (EIP Rule 3: no placeholder backends).

## 2. Placement in the workspace

```text
qubivex-core        TensorDesc (dtype, shape, byte strides, byte offset): pure,
                    Copy, no allocation, no I/O, no device types
     ▲
qubivex-model       RawTensor::desc(): a file extent described by a TensorDesc
     ▲
qubivex-compute     TensorView<'a, T>, TensorViewMut<'a, T>, Tensor<T> (Arc-shared),
                    WeightView: built from a 2-D TensorView<u8>
     ▲
backend-cpu, arch, runtime: unchanged kernel signatures; the runtime's private
                    Tensor is replaced by Tensor<u8>
```

`qubivex-model` depends only on `qubivex-core`, so the file layer can describe tensors
without knowing about compute storage. The dependency direction enforced by `xtask` stays
unchanged.

## 3. `TensorDesc` (`qubivex-core`)

```rust
pub struct TensorDesc {
    dtype: DType,
    shape: Shape,                 // existing type: row-major, rank ≤ MAX_RANK (8)
    strides: [u64; MAX_RANK],     // in BYTES, outermost first
    offset: u64,                  // in BYTES, from the start of the storage
}
```

The type is `Copy`, about 150 bytes, and never allocates.

### 3.1 Why byte strides

QUBIVEX stores three kinds of dtype (`StorageLayout`):

| Layout | Examples | Bytes per element |
|---|---|---|
| Elementwise | f32, f16, bf16 | fixed |
| Blocked | Q4_K, Q6_K, Q8_0, … | not defined: 256 elements share 144 bytes |
| RowScaled | `i8_row_scaled` | not linear: each row ends with a 4-byte scale |

Element-count strides can't address the second and third kinds: no element count
reaches row `r` of an `i8_row_scaled` matrix. Byte strides address all three uniformly,
which is also what GPU APIs and the memory planner work in. NumPy uses byte strides for
the same reason.

### 3.2 Packed dtypes

Blocked and RowScaled dtypes are called **packed**. For a packed dtype:

- The **innermost dimension is the packed row**. It has no per-element stride, and its
  slot in `strides()` reports 0. It can't be broadcast either, which keeps that 0
  unambiguous. Outer dimensions of a packed tensor may be broadcast.
- Operations that would move the innermost dimension (a permute or transpose involving
  it) return `CoreError::PackedInnermost`. Transposing a quantised weight needs a repack,
  which is a copy and is never done implicitly.
- Slicing the innermost dimension is allowed for Blocked dtypes **only on block
  boundaries** (start and length are multiples of `block_elems`). It is rejected for
  RowScaled, because a partial row has no scale.
- Outer dimensions behave exactly like elementwise ones. This is how one expert is
  selected from a stacked `[experts, rows, cols]` Q4_K tensor without a copy.

### 3.3 Construction

| Constructor | Result |
|---|---|
| `TensorDesc::contiguous(dtype, shape)` | Row-major, packed strides, offset 0. Errors on overflow or a misaligned block (reuses `DType::storage_bytes`) |
| `TensorDesc::from_parts(dtype, shape, strides, offset)` | Validates every rule in §3.2 and §3.5. Used by container readers and tests |

### 3.4 View operations

None of these copy data. All of them return a new `TensorDesc`, or an error.

| Operation | Rule |
|---|---|
| `permute(&[usize])` / `transpose(a, b)` | A valid permutation of `0..rank`; may not move a packed innermost dimension |
| `narrow(dim, start, len)` (slice) | `start + len ≤ dims[dim]` (checked); `offset += start × stride`; §3.2 rules for the packed innermost |
| `select(dim, index)` | `narrow(dim, index, 1)`, then drop the dimension; the rank falls by 1 |
| `broadcast_to(shape)` | NumPy rules, aligned on the right; size-1 or missing dimensions get stride 0; a packed dtype must keep its innermost size |
| `reshape(shape)` | Same element count. Zero-copy whenever the strides allow, using the PyTorch `computeStride` merge and split rule, not just for contiguous tensors. Otherwise `CoreError::ReshapeNeedsCopy`, and the caller decides whether to copy. A packed dtype must keep its innermost size |

### 3.5 Queries

| Query | Meaning |
|---|---|
| `is_contiguous()` | Strides equal the row-major packed strides (the stride of a size-1 dimension is ignored) |
| `is_non_overlapping()` | No two indices map to the same byte. Required for mutable views (§5) |
| `required_bytes()` | `offset + Σ (dims[i] − 1) × strides[i] + inner_extent`, checked; 0 if any dimension is 0. `inner_extent` is the element size for elementwise dtypes and the packed row's byte length for packed ones. **Storage must be at least this long.** It is checked once, when a view is built, so later indexing stays in bounds |
| `byte_offset(&[u64])` | The checked byte position of an index (outer index only for packed dtypes) |
| `numel()`, `storage_bytes()` | Delegated to `Shape` and `DType`, for contiguous descriptors |

### 3.6 Errors

These are new `CoreError` variants. The enum is already `#[non_exhaustive]`.

- `LengthMismatch`
- `DimOutOfRange`
- `IndexOutOfRange`
- `RangeOutOfBounds`
- `InvalidPermutation`
- `PackedInnermost`
- `BroadcastIncompatible`
- `ElementCountMismatch`
- `ReshapeNeedsCopy`
- `StrideMisaligned`

Variants that carry shapes box them, so that `Result<_, CoreError>` stays small on hot
paths. The existing `Overflow { what }` covers every arithmetic failure.

`qubivex-compute` adds two variants:
- `ComputeError::ElementType`: the storage element can't hold the dtype;
- `ComputeError::Shared`: mutation was attempted on shared storage.

## 4. Quantisation metadata

The existing `DType` and `StorageLayout` already describe quantisation independently of
storage:
- block size;
- block bytes;
- row scale;
- the GGML type id.

All supported formats keep their scales inside the block or row, so nothing more is
needed today.

Formats with **external** scales or zero-points (GPTQ, AWQ, FP8 with per-tensor scales)
will be described as **separate tensors linked by role** in the architecture's tensor
schema (ADR-003), not as extra fields on `TensorDesc`. That keeps the descriptor small
and `Copy`. The design is recorded now and built when the first such model is targeted.

## 5. Storage and views (`qubivex-compute`)

```rust
/// Element types that may back a tensor, and the dtypes each can hold.
pub trait Element: Copy + Send + Sync + 'static { fn holds(dtype: DType) -> bool; }
//   f32 → F32      u16 → F16, BF16      u8 → any dtype (packed dtypes require u8)

pub struct TensorView<'a, T: Element>    { desc: TensorDesc, data: &'a [T] }
pub struct TensorViewMut<'a, T: Element> { desc: TensorDesc, data: &'a mut [T] }
pub struct Tensor<T: Element>            { desc: TensorDesc, data: Arc<[T]> }
```

**Typed storage, not bytes.**
- Activations stay `f32` slices and weights stay `u8` slices, as today. Kernels never
  reinterpret bytes, so no `unsafe` code and no `bytemuck`.
- Descriptor byte strides and offsets must be multiples of `size_of::<T>()`. This is
  checked at construction, giving `StrideMisaligned`.

**Construction validates once.** `TensorView::new(desc, data)` checks:
- `T::holds(desc.dtype())`;
- the alignment rule above;
- `desc.required_bytes() ≤ data.len() × size_of::<T>()`.

After that, every in-range index is in bounds by construction.

**Views share storage.**
- `Tensor::view()` borrows: the borrow checker ties the view's lifetime to the tensor,
  so a view can never outlive its storage (EIP §4.1B).
- `Tensor::clone()` clones the `Arc`, not the data.
- Every §3.4 operation exists on views and tensors and returns the same type with a new
  descriptor.

**Mutable views.**
- `TensorViewMut::new` additionally requires `desc.is_non_overlapping()`. Without it,
  two indices could write the same byte through one `&mut`, which is legal Rust but a
  logic error.
- `Tensor::view_mut()` succeeds only while the `Arc` is unique (`Arc::get_mut`). Shared
  storage is never mutated behind another holder's back.

**Copies are explicit.** `to_contiguous()` is the only copying operation:
- a strided gather for elementwise dtypes;
- whole-row copies for packed dtypes.

It returns a new `Tensor<T>`. Reshape never copies silently: it returns
`ReshapeNeedsCopy`, and the caller calls `to_contiguous()` first.

**`WeightView` stays the kernel type.** `WeightView::try_from(TensorView<'a, u8>)`
requires:
- rank 2;
- a contiguous innermost dimension;
- a row stride equal to the row's byte length.

The kernels keep receiving exactly what they receive today, so the hot path is
unchanged. `WeightView::new` stays as a convenience.

## 6. Buffer lifetime and devices

| Storage | Lifetime mechanism | Exists |
|---|---|---|
| Borrowed host view | Rust borrow (`'a`) | This step |
| Owned or shared host tensor | `Arc<[T]>` reference count | This step |
| Expert in the cache | `Lease` (holds `Arc<[u8]>`; eviction skips leased slots) | Yes (Phase 4) |
| Device buffer | Opaque `DeviceBuffer` handle; reference-counted backend state | With the first GPU backend (ADR-011) |

The rule for asynchronous execution, fixed now so the GPU backend can't violate it:

> **An operation submitted to a device queue owns a reference-counted handle to every
> buffer it reads or writes, and releases those handles only when its completion event
> fires.**

Dropping the last user-side handle while the device still runs therefore can't free the
memory. Submission and completion stay distinct (EIP §5.1). The same rule applies to
expert leases: an in-flight GPU expert operation holds the lease until completion. This
is why leases hold an `Arc` rather than a borrow of the slab.

`TensorDesc` has no device field. **The storage carries the device**, so one descriptor
describes the same tensor in RAM and in VRAM, and a transfer copies storage without
touching the descriptor. CPU↔device transfers are explicit backend calls (`upload`,
`download` in ADR-007), never implicit.

## 7. Migration of existing code

| Today | After | Behaviour change |
|---|---|---|
| `qubivex_model::RawTensor { dtype, shape, offset, bytes }` | Unchanged fields, plus `desc()`, the contiguous descriptor of the extent | None |
| `qubivex_runtime` private `Tensor { dtype, rows, cols, bytes }` | `Tensor<u8>` | None; dense weights load exactly as before |
| `WeightView::new(dtype, rows, cols, data)` | Unchanged, plus `TryFrom<TensorView<'_, u8>>` | None |
| Expert slice of a stacked tensor (`rows_slice`) | `select(0, expert)` on the stacked descriptor (later, when the store hands out views) | None |
| `KvCache` (`Vec<Vec<Vec<f32>>>`) | Not changed in this step. A later candidate: one `Tensor<f32>` `[layer, kv_head, pos, head_dim]` with `narrow` views per step | Planned separately, with its own benchmark |

## 8. Acceptance tests

Unit tests in `qubivex-core` and `qubivex-compute`, plus one proptest:

| Area | Test |
|---|---|
| Shape arithmetic | Contiguous strides for ranks 0–8; `numel`, `required_bytes`; zero-size dimensions |
| Strides | Contiguous for f32, bf16, Q4_K and `i8_row_scaled`; `is_contiguous` after view operations |
| Transpose views | 2-D transpose shares storage (same pointer); element `(i, j)` of the transpose equals `(j, i)`; transposing twice gives the original descriptor; transposing a packed innermost dimension is rejected |
| Slicing | `narrow` offsets and strides; slice of a slice; `select` drops the rank; block-aligned narrow on Q4_K succeeds and a misaligned one fails; RowScaled innermost narrow fails |
| Broadcasting | `[3, 1]` → `[2, 3, 4]` gives stride 0 and correct values; incompatible shapes fail; packed dtypes fail |
| Storage offsets | A view at a non-zero offset reads the right elements; `required_bytes` with an offset |
| Contiguous conversion | `to_contiguous` of a transposed or sliced f32 view equals a reference gather; packed row copy is byte-exact |
| Reshape | Zero-copy for contiguous inputs and stride-compatible non-contiguous inputs; `ReshapeNeedsCopy` otherwise; element count mismatch fails |
| Invalid descriptors | Storage too short; misaligned strides for `T`; wrong `Element` for the dtype; overlapping mutable view; out-of-range dimension or index; invalid permutation |
| Ownership and lifetime | A view borrows a `Tensor`. A compile-fail doctest shows a view can't outlive its tensor. `view_mut` fails while the `Arc` is shared and succeeds once unique |
| Overflow | Descriptors with dimensions near `u64::MAX` return `Overflow` from `required_bytes`, `narrow` and `byte_offset` |
| Proptest | Random shapes, permutations and slices: `required_bytes` never panics. Every reachable `byte_offset` is `< required_bytes`. A view operation then `to_contiguous` equals a naive index-by-index gather |
| No regression | Workspace tests (123) still pass; real-model invariance tests pass; **decode tok/s on OLMoE Q4_K_M (160 tokens, 3 runs) within noise of the 20.7–21.0 baseline** |

## 9. Resolved and remaining questions

1. **Negative strides** (reversed views) are excluded: strides are `u64`. No current or
   planned operation needs them.
2. **The KV cache migration** was deliberately left out of this step, by owner
   decision. It is the next real consumer of `narrow` views and gets its own
   before-and-after benchmark.
