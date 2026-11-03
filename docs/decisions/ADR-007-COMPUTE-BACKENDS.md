# ADR-007: Compute backends behind one trait, runtime-selected, feature-gated

- Status: Proposed (Phase 0). **Amended 2026-11-03** (owner decision): backend order and device-assignment rule, see below.
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

## Consequences
GPU work cannot leak model knowledge into backends, and a CPU-only build stays
dependency-light.
