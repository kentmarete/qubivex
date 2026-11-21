# ADR-007: Compute backends behind one trait, runtime-selected, feature-gated

- Status: Proposed (Phase 0). **Amended 2026-11-03** (owner decision): backend order and device-assignment rule, see below. **Amendment 2 (2026-11-22, Proposed):** MXFP4 as an execution format.
- Date: 2026-11-01

## Decision
- `ComputeBackend` (see `QUBIVEX_DESIGN.md` §4.4) has **coarse, model-agnostic ops**:
  matmul (any supported weight dtype), norm, RoPE, attention, and grouped
  `expert_ffn_batch`. Fine-grained elementwise ops stay inside backends to avoid
  launch overhead.
- Order: CPU (Phase 4), CUDA (`cudarc` + ported Colibri kernels, Phase 8), Vulkan
  (`ash` + ported GLSL→SPIR-V, after CUDA), Metal (`objc2-metal`, later).
- Each GPU backend is a Cargo feature and a separate crate. The default build is
  CPU-only with no GPU SDK required. Selection happens at runtime from the hardware
  profile, measured capability and the plan. "Available" means probed and working,
  not merely compiled in.
- Every backend reports allocations to `MemoryLedger`.
- **Combination stays in rank order on the host or in a single device kernel**, so a
  hybrid CPU/GPU MoE layer gives the same summation order as the CPU-only path.
- Cross-backend parity: tokens against the CPU with a documented drift tolerance. The
  same backend must be bitwise reproducible across cache configurations.
- Backend choice per model is measured, not assumed: Colibri measured OLMoE *slower*
  on an integrated GPU.

## Amendment (2026-11-03)
From the Engineering Improvement Plan reconciliation (`docs/engineering/ARCHITECTURE_DECISIONS.md`
§2.4 and §2.6):
- **Order: Metal, then CUDA, then Vulkan.** The dev machine is an Apple M3 Pro, so Metal is
  the only GPU that can be tested locally. CUDA waits for NVIDIA hardware or a GPU CI runner
  (blocker B2). The Metal crate is `qubivex-backend-metal`, created with its first real
  kernels.
- **Device assignment is part of the execution plan.** It is fixed for a given model,
  configuration and hardware, and chosen from measured costs at plan time. The runtime may
  admit, delay or refuse work under memory pressure, but it never moves an operation to
  another device mid-request because of live load. This keeps output reproducible
  (invariant 1).
- `rms_norm`, `rope`, `attention` and `softmax` move behind `ComputeBackend` when the first
  GPU backend needs them, not before. Streams versus events, object safety and the
  `DeviceBuffer` handle are decided in ADR-011 alongside that backend.
- The existing crate names stay (`qubivex-compute`, `qubivex-backend-cpu`); the EIP's
  `qubivex-backend` and `qubivex-cpu` are the same crates.

## Amendment 2 (2026-11-22, Proposed): MXFP4 as an execution format
Owner brief: add MXFP4 for very large MoE models as a format of its own beside Q4_K and
Q6_K (which stay unchanged), consumed compressed inside the matmul, with FP32
dequantisation never the normal execution path.
- **Format.** GGML type 39 (`GgmlType::MXFP4`): blocks of 32 elements, 17 bytes. Byte 0 is
  an E8M0 scale (a power of two, `2^(e − 127)`); bytes 1..17 hold 32 E2M1 nibbles, element
  `i` in the low nibble of byte `i` for `i < 16`, else in the high nibble of byte `i − 16`.
  The E2M1 values are 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives.
- **Value of a weight.** As GGML and gguf-py: the doubled E2M1 value (an integer, |v| ≤ 12)
  times `2^(e − 128)`. Every weight is exact in f32. Only `e = 255` with |v| ≥ 8 overflows,
  to ±∞, and every backend computes the same infinity. As in GGML, `e = 255` is a finite
  scale, not the OCP specification's NaN.
- **Kernels read the packed blocks.**
  - CPU: fused NEON one-row and multi-row dots, used for 1 to 12 activation rows,
    convert 16 nibbles at a time by table lookup in registers.
  - Metal and Vulkan: the kernels read the scale byte and the nibble from device memory
    inside the 64-lane dot. These are the plain, token-tiled, routed and grouped kernels
    on Metal, and the plain and fused pipelines on Vulkan.
  - Two CPU paths still decode a scratch row, exactly as they do for the K-quants: the
    portable path (other ISAs) and the tiled prefill kernel (13 or more rows), which
    decodes a 4-row tile that stays in L1.
  - No f32 copy of a weight matrix is ever made.
- **Bits.** Each lane gets the same fused multiply-adds as the CPU's reduction, so every
  backend gives the CPU's bits. Apple GPUs flush f32 subnormals, and the scale `2^(e − 128)`
  is subnormal for `e < 2`. Metal and Vulkan therefore form that product as
  `(v · 2^(e − 2)) · 2^-126` from two normal factors, which is still exact. Only a weight
  that is itself subnormal is flushed: ADR-011's existing exception, which a test bounds.
- **Not yet:** the Hugging Face `*_blocks` / `*_scales` layout (gpt-oss SafeTensors). It
  packs two consecutive elements per byte (low nibble first) and stores the E8M0 bytes
  apart from the nibbles. It would be repacked into the GGML layout at import, which is a
  lossless byte shuffle. That needs the gpt-oss architecture options (attention sinks,
  biases, the clamped SwiGLU), so it is a later step.
- **Evidence:**
  - `qubivex-backend-cpu/tests/oracle.rs`: gguf-py's MXFP4 dequantizer on every E8M0
    byte, bit for bit.
  - `neon` unit tests: the fused kernels against the portable path for every scale byte
    and 1 to 5 rows.
  - Metal and Vulkan parity tests: typical and wild fills, routed and batched, and every
    scale byte, with a positive control showing the plain subnormal scale fails.
  - `qubivex-runtime/tests/mxfp4.rs`: a model with every quantized matrix in MXFP4 gives
    the logits and tokens of the same model stored as f32 with the decoded values, on the
    CPU, Metal (three modes), Sim and Vulkan.

## Consequences
GPU work cannot leak model knowledge into backends, and a CPU-only build stays
dependency-light.
