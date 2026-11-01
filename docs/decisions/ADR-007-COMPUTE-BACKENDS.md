# ADR-007: Compute backends behind one trait, runtime-selected, feature-gated

- Status: Proposed (Phase 0)
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

## Consequences
GPU work cannot leak model knowledge into backends, and a CPU-only build stays
dependency-light.
