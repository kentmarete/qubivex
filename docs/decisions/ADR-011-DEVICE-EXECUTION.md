# ADR-011: Device execution, buffer lifetimes and GPU parity

- Status: Proposed (Phase 8 slice 1, 2026-11-03; amendment 1, slice 2). Owner review pending.
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

## Amendment 1 (2026-11-03): batching, zero-copy weights and placement in the engine, Phase 8 slice 2

### What was built
- **Batched submission (§2).** `ComputeBackend::matmul_many` takes independent matmuls as
  one submission (the default loops over `matmul`, so the CPU is unchanged). Metal encodes
  the whole batch in one command buffer, stages each distinct input once and reuses its
  staging buffers between submissions. The forward pass submits q, k and v together, and
  each expert's gate and up together.
- **Zero-copy weights (§4).** The engine reads every dense tensor into one page-aligned
  arena (`qubivex_compute::host::PageAligned`: an over-allocated `Vec`, aligned inside,
  so no unsafe code). `MetalBackend::map_host` wraps it with
  `newBufferWithBytesNoCopy`. Each placed matrix is an offset into that one buffer, and
  the buffer holds the arena's `Arc` (§3). The ledger charges the dense tensor bytes once,
  as before; the arena's alignment slack (under 40 KiB for OLMoE) is not charged.
- **Placement (§5).** `EngineOptions::{backend, metal_ops}` and the CLI's `--backend
  cpu|metal` and `--metal-ops attention,router,head`. Placement is fixed at load, by
  weight: the dense matmuls of the chosen kinds run on Metal, and everything else
  (embedding lookup, norms, attention, experts) runs on the CPU. `Engine::placement()`
  reports it, and `qubivex bench` records it as `config.device`.
- **Token parity.** The fixture invariant `metal_placement_never_changes_bits` checks
  every placement on all three fixtures. The env-gated `engine_on_metal_gives_the_cpu_tokens`
  places all 81 dense matmul weights of OLMoE on Metal (q, k, v, o and router for each of 16 layers, plus the head): the 10 × 50304 prompt logits are
  bit-identical, and so are the 48 generated tokens. Every `qubivex bench` run below
  matches the Phase 7 reference tokens.

### Measurements (M3 Pro, OLMoE Q4_K_M, all experts resident, 160 tokens, 3 runs)
Kernel level (Criterion, resident weights):

| | time |
|---|---|
| 64 × 64 f32, one submission (pure overhead) | 109 µs (slice 1: 120 µs; staging reuse) |
| q, k, v (3 × 2048 × 2048 Q4_K) as 3 submissions | 727 µs |
| the same in one command buffer | 424 µs |

Polling the command buffer's status instead of `waitUntilCompleted` did not lower the
109 µs (113 µs), so that cost is on the GPU side.

End to end (`qubivex bench`, records `docs/performance/baselines/2026-11-03-m3pro-phase8-*.json`):

| `--backend` / `--metal-ops` | decode tok/s (median step) | median step | prefill tok/s |
|---|---|---|---|
| cpu | 25.51 | 39.19 ms | 32.9 |
| metal / head | 25.07 | 39.90 ms | 32.6 |
| metal / attention,head | 22.16 | 45.13 ms | 30.1 |
| metal / attention,router,head | 21.01 | 47.60 ms | 29.6 |

An interleaved A/B of the CPU path against `main` (26.05 and 26.05 tok/s, then 26.05
and 26.13) shows the dense arena costs nothing.

**Why the head does not win in the engine.** In a tight loop the Q6_K head takes 1.78 ms
on Metal against 2.77 ms on the CPU. Timed inside decode, the Metal head takes a median
3.99 ms against the CPU's 3.18 ms. The GPU is idle for about 36 ms between heads, and a
lone submission after an idle period runs slower than one in a busy stream. Attention
loses for the reason slice 1 predicted: 5 submissions per layer at about 110 µs each
cost more than the kernels save.

### Decision
- **The CPU stays the default backend.** `--backend metal` is a correct, bit-identical
  opt-in. Its default placement is `head`, the only candidate whose kernel beats the CPU
  by more than one submission's cost.
- **A Metal speed-up needs the GPU kept busy, not more placed matmuls.** That means one
  submission per layer or per token, which needs norms, RoPE, attention and the expert
  FFN as Metal kernels with the same bit contract. Then the per-submission cost is paid
  once per token, and the GPU stays clocked up. That work, and the block-reusing Q4_K
  kernel, is slice 3.
- The scratch staging buffers (at most a few MiB) are not charged to the ledger yet.
  They will be once device tiers are charged per ADR-005.
