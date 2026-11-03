# ADR-011: Device execution, buffer lifetimes and GPU parity

- Status: Proposed (Phase 8 slice 1, 2026-11-03). Owner review pending.
- Date: 2026-11-03
- Related: ADR-007 (backends; amendment 2026-11-03), ADR-010 (lossless by default),
  `docs/engineering/ARCHITECTURE_DECISIONS.md` §2.4–2.5, TENSOR_ARCHITECTURE §6.

## Context
Phase 8 starts with Metal (`qubivex-backend-metal`) because the dev machine is an M3 Pro
(ADR-007 amendment). ADR-007 left these open until the first GPU backend: how close GPU
results must be to the CPU's, how device work is submitted and synchronised, and how buffers
stay alive while the GPU uses them. Slice 1 built the kernels and measured them, so those
questions can now be answered from evidence.

What slice 1 found on the M3 Pro (Apple GPU, 14 cores, unified memory):
- **Bit identity is achievable.** Each Metal kernel reproduces the CPU's dequantised weight
  values and its fixed 64-lane reduction shape: a 32-thread SIMD group, thread `t` holding
  lanes `t` and `t + 32`, with the halving tree done by `simd_shuffle_down`. Shaders compile
  from source at start-up with `MTLMathModeSafe`, and `#pragma METAL fp contract(off)`, so
  `a * b - c` stays two rounded operations. Every output is bit-identical to the CPU for all
  six dtypes, every tail shape and batch sizes 1, 3 and 8. On the real OLMoE Q4_K_M model,
  all 130 weight tensors (3.92 GiB, about 11M outputs) are bit-identical.
- **The one divergence: Apple GPUs flush f32 subnormals to zero**, in safe math mode too,
  and Metal has no option to keep them. A weight, activation, product or partial sum below
  1.18e-38 in magnitude can therefore differ. At model magnitudes this does not happen
  (the real-model test shows none), and f16 subnormals are unaffected because they are
  normal in f32.
- **Per-call synchronisation dominates small matrices.** A synchronous command buffer per
  matmul costs about 120 µs; an expert projection (1024 × 2048 Q4_K) takes 176 µs on Metal
  against 63 µs on 5 CPU threads. The 50304 × 2048 Q6_K output head takes 1.78 ms against
  2.77 ms.

## Decision
1. **Parity contract: bit-identical, except for f32 subnormals.** A GPU backend must give
   the CPU's bits for every output whose inputs, products and partial sums stay in the
   normal f32 range. This replaces ADR-007's "documented drift tolerance" for Metal. Any
   kernel that changes the reduction shape (for example `simdgroup_matrix` tiles) is a
   lossy opt-in under ADR-010, not a default. The parity tests (`tests/parity.rs` and the
   env-gated `qubivex-runtime/tests/metal_parity.rs`) are the gate for every Metal kernel
   change. CUDA and Vulkan get their own contract when they are built; the same target
   applies unless a measurement shows it cannot be met.
2. **Submission.** Device work is encoded into command buffers, and many operations go in
   one buffer: at least one layer, eventually one token. Each submission returns an `Event`
   that completes when the GPU finishes. Slice 1's synchronous per-call `matmul` is a
   correctness path only; the engine must not use it per expert.
3. **Lifetimes.** Every buffer an in-flight operation touches is held by a
   reference-counted handle that the operation owns until its event completes (the `Arc`
   rule of TENSOR_ARCHITECTURE §6). Expert leases move into the operation; their
   `Arc<[u8]>` already allows this. No Rust `async` on the compute path (ADR-006).
4. **Buffers.** An opaque `DeviceBuffer` handle (an `Arc` of backend state) keeps
   `ComputeBackend` object-safe. On unified memory, weights are wrapped in place
   (`newBufferWithBytesNoCopy` over the page-aligned mapping) instead of copied, and the
   ledger is charged once, for the host bytes. Copies are charged as device bytes.
5. **Placement.** Which device runs which operation is fixed in the execution plan from
   measured costs (ADR-007 amendment). Slice 1 shows the plan needs the per-submission cost,
   not only kernel bandwidth: with today's numbers, only the output head wins on Metal.
6. **Shaders compile at start-up from source.** No offline Metal toolchain is required
   (the dev machine has Command Line Tools only). Compilation takes well under a second; a
   cached `.metallib` can be added later if start-up time matters.

## Consequences
- Metal output keeps the reproducibility invariant (invariant 1) instead of a tolerance, so
  CPU and GPU runs can be compared token for token.
- The fastest Apple GPU matmul paths (simdgroup matrices, fast math) are excluded from the
  default; any of them comes as a named, measured, lossy opt-in.
- Slice 2 must build command-buffer batching and zero-copy weights before the engine can
  gain from Metal; the kernels alone do not give a speed-up for expert-sized matrices.
- A Q4_K kernel that dequantises per element reaches only about 21 GiB/s of kernel time on
  expert shapes; a block-reusing kernel (same bits) is the next kernel step, and prefill
  needs a kernel that reuses dequantised weights across activation rows.
