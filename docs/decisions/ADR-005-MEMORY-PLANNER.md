# ADR-005: One measured, role-driven memory planner

- Status: Proposed (Phase 0)
- Date: 2026-11-01

## Context
Colibri has two planners: Python `resource_plan.py` for `coli plan`, and each engine's
own C budget at load time. They can disagree. The Python one carries family-specific
branches.

## Decision
- A single `MemoryPlanner` in `qubivex-memory` is used by `qubivex plan`, by load and
  by runtime rebalancing.
- Inputs: tensor roles and sizes (ADR-003), `HardwareProfile`, a live
  `ResourceSnapshot`, request requirements (context, concurrent slots) and the policy
  profile.
- RAM and VRAM budgets are separate and never summed. The defaults come from
  Colibri's measurements: `0.88 × MemAvailable` after the dense load, a 0.5 GB
  activation reserve, the KV and the expert cache **sharing** one room (the KV takes
  slots back as it grows), and a 1-slot-per-layer floor.
- Placement uses priority by expected latency cost (the specification's retention
  score). The baseline policy comes first. Smarter policies must beat it on a recorded
  benchmark.
- `MemoryLedger` records real allocations per tier and owner. The pressure monitor
  compares the ledger and the OS snapshot against the plan. Its responses, in order:
  stop prefetch, shrink caches, refuse new requests. It never kills an active request.
- Every number in a plan or recommendation is tagged `Measured` or `Estimated`.

## Consequences
The planner has no model knowledge beyond roles. Property tests can fuzz it with
synthetic models and machines.

## Amendment 1 (2026-11-03): first build, Phase 5 slice 1

Built in `qubivex-memory` and `qubivex plan`. The decision stands. These details were
settled while building it, and the ARCHITECTURE_DECISIONS §2.3 resolution is folded in:

- **A function, not a trait.** `plan(footprint, host, request)` is the one planner.
  A `MemoryPlanner` trait waits for a second implementation. `rebalance` and
  `recommend` come with the runtime integration and the catalog, respectively.
- **Footprint from roles only.** Dense bytes are every non-expert tensor; a cache slot
  is a layer's largest expert; the KV cost per token is the output width of each
  layer's K and V projections × 4 bytes (the KV cache is f32). No architecture
  knowledge: a new architecture gets planning by declaring roles.
- **Profiles are fractions of available RAM**: `performance` 0.88 (Colibri's measured
  default), `balanced` 0.75, `saver` 0.50, `custom` any fraction in (0, 1]. The last
  three are policy values, not measurements; revisit them with recorded runs. An
  explicit `--ram-budget` replaces the profile (Colibri's `RAM_GB`). The EIP's
  `CpuPreferred`/`LowMemory` map to `performance`/`saver`, and `GpuPreferred` waits
  for a GPU backend.
- **Before loading,** `available − dense` stands in for Colibri's after-load
  measurement, and the budget is tagged `Estimated`. Load time measures again with the
  dense weights resident.
- **Room sharing and admission.** The expert cache starts with the whole room (up to
  every expert) and gives it back as the KV grows (`MemoryPlan::expert_cache_for`).
  A context is admitted only if its full KV leaves one slot per layer. Without an
  explicit context, the model's context is shortened to what fits; an explicit one
  that does not fit is refused with the largest that does.
- **Non-uniform per-layer budgets:** slots go one at a time to the layer with the
  highest `weight / slots`. Equal weights spread evenly. Weights from measured
  per-layer demand arrive with Phase 6 telemetry. The expert store enforces only the
  byte total until Phase 6's per-layer slab pools; the plan says so.
- **Tags:** `Measured` (OS or file headers), `Estimated` (formulas and policy
  constants), and a third tag, `Requested`, for numbers the user gave.
- **`MemoryLedger`:** atomic counters per tier (RAM, pinned host, each GPU) and owner
  (dense, expert cache, KV, activations), charged through RAII `Charge`s. Disk is not
  a tier. Remote storage is out of scope until a user needs it.

Still to build in Phase 5: load-time planning and ledger charges in the engine, KV
growth shrinking the store, the ledger-against-RSS test on the real model, and the
pressure monitor with its RAM-hog test.
