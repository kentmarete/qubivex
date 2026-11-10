# Tensor engine

QUBIVEX has its own tensor types and kernels. It does not use an external tensor
library. This page describes what is there and the rules it keeps. The design records
are [ADR-002](../decisions/ADR-002-HYBRID-COMPUTE.md) (own kernels),
[ADR-009](../decisions/ADR-009-TENSOR-DESCRIPTORS.md) (descriptors),
[ADR-010](../decisions/ADR-010-NUMERICAL-PRECISION.md) (precision) and
[ADR-011](../decisions/ADR-011-DEVICE-EXECUTION.md) (Metal). The full descriptor design
is [TENSOR_ARCHITECTURE.md](../architecture/TENSOR_ARCHITECTURE.md).

## Why its own

An inference engine for one family of architectures needs few operations: a matmul
over quantised weights, RMSNorm, softmax, rotary embeddings, attention and SwiGLU. What
it needs from them is unusual: exact reproducibility, weights used in their on-disk
quantised form with no conversion, and expert weights that arrive and leave during a
pass. The comparison in ADR-002 found that Candle has no lossless quantised CPU matmul
(its fast path quantises activations), and that own kernels ran attention decode 2 to 5
times faster while keeping the bit-identity rules below.

## The layers

| Crate | Holds |
|---|---|
| `qubivex-core` | `DType`, `Shape`, `TensorDesc`: pure descriptions, no storage |
| `qubivex-compute` | Storage and views (`Tensor`, `TensorView`, `WeightView`), and the `ComputeBackend` trait |
| `qubivex-backend-cpu` | The CPU kernels |
| `qubivex-backend-metal` | The Metal kernels and device buffers |
| `qubivex-arch` | The operations of a forward pass, written against the trait |

A backend never depends on the model, the runtime or the API. The forward pass never
names a backend.

## Descriptors and views

`TensorDesc` describes any tensor: dtype, shape (up to 8 dimensions), strides in bytes
and a byte offset. A weight in a file, a weight in RAM, an expert in the cache, an
activation and a slice of KV all use it.

- **Views are zero-copy.** Transpose, permute, slice, select and broadcast change the
  descriptor and share the storage. Reshape works where the strides allow it and
  otherwise returns an error; `to_contiguous()` is the only operation that copies.
- **Every size and offset uses checked arithmetic.** Shapes come from untrusted model
  files, so an impossible descriptor is a typed error, never an overflow or a panic.
- **Storage is typed.** Activations are `f32`, weights are bytes. Nothing reinterprets
  bytes as floats, so the descriptor and view code has no `unsafe`.
- **Lifetimes are enforced by the compiler.** A view borrows its tensor and cannot
  outlive it. Shared tensors are `Arc`s. An expert in use is held by a lease, which
  eviction respects.

## Dtypes

| Kind | Dtypes | Kernels |
|---|---|---|
| Floating point | `f32`, `f16`, `bf16` | Yes |
| GGML block quantisation | `Q8_0`, `Q4_K`, `Q6_K` | Yes |
| | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_1`, `Q2_K`, `Q3_K`, `Q5_K`, `Q8_K` | Parsed and sized; a model using one for a weight the forward pass needs is refused |
| | I-quants (`IQ*`) | Not supported |

Quantised weights are used as they are stored. A Q4_K matrix is never expanded to `f32`
in memory: each row is dequantised inside the dot product.

## Precision: lossless by default

Every kernel computes `f32` activations times the **exact** dequantised weights. The
quantisation is the model file's; QUBIVEX adds no approximation of its own. In
particular it does not quantise activations to use integer dot products. Fast quantised
paths elsewhere do, at a relative error of about 4e-3: on OLMoE, Colibri's lossy default
diverges from the lossless tokens after 86 of 160.

Anything that changes numerics would be an opt-in with its own name and a recorded
error. None exists yet. No memory profile changes numerics.

Against PyTorch `transformers`, logits agree to under 1e-6 relative on the tiny
reference models and within 1.2e-5 on Qwen3-0.6B, with identical greedy tokens.

## Determinism

The CPU backend is bitwise deterministic. An output element depends only on its weight
row and its activation row:

- Work is split across **output rows**, never across a reduction, so the thread count
  cannot change a sum's order.
- Every dot product uses one fixed reduction shape (64 lanes), whichever kernel
  computes it.
- `f32::mul_add` is a correctly rounded fused multiply-add on every target, so NEON,
  AVX2 with FMA, and the portable path agree.

So the same model and input give the same bits at any thread count, any batch size, on
arm64 and x86-64. The Metal kernels are written to reproduce the same bits, and tests
compare them exactly, not within a tolerance.

## CPU kernels

One matmul, `ys = W · xs` for `s` activation rows, has several paths chosen by `s` and
the dtype. All give the same bits:

| Activation rows | Path | Where |
|---|---|---|
| 1 (decode) | Fused: dequantise inside the row dot, no scratch buffer | NEON, for Q4_K, Q6_K, Q8_0, bf16 |
| 2 to 12 (decode for a few sequences) | Fused multi-row: each weight dequantised once into up to five rows at a time | NEON, same dtypes |
| 6 and more not covered above (prefill) | Register-blocked tiles | All targets |
| Anything else | Dequantise a row, one portable dot per output | All targets |

On x86-64 the portable source is compiled a second time with AVX2 and FMA enabled and
selected at run time when the CPU has them. There are no hand-written x86-64 kernels
yet, and x86-64 speed has not been measured.

Threads: a rayon pool sized to the CPU's performance cores (`--threads`). Efficiency
cores are left out by default, because a team of threads runs at the pace of its slowest
member.

## Metal

`--backend metal` (macOS) runs whole layers on the Apple GPU: attention, routing, the
experts and the head as compute kernels, one command-buffer submission per layer
(`--metal-ops layers`) or one per token with routing on the GPU (`--metal-ops token`,
the default). Weights are shared with the GPU without a copy on unified memory.

The rule for device buffers: a submitted operation holds a reference to every buffer it
reads or writes until it completes, so dropping a tensor while the GPU still runs cannot
free its memory. Expert leases follow the same rule.

Measured on an M3 Pro, OLMoE-1B-7B Q4_K_M, all experts resident: 60 tokens per second
against the CPU's 41 for one sequence. A dense model on Metal runs its matmuls one at a
time and is slower than the CPU.

## Not supported

- **A general tensor library.** There is no autograd, no training, no elementwise
  arithmetic API and no broadcasting kernels. Broadcasting exists as a view operation.
- **CUDA and Vulkan.** `ComputeBackend` is the boundary they would implement.
- **Hand-written AVX2 or AVX-512 kernels.**
- **Lossy fast paths:** integer dot products over quantised activations, `f16` KV,
  reduced-precision accumulation.
- **I-quant dtypes,** and kernels for the less common GGML block types listed above.
- **Whole-layer Metal for dense models.**
- **Multiple GPUs.**
