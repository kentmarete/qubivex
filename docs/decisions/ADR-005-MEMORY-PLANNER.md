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
