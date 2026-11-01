# ADR-008: Expert cache: per-layer slab pools, LFRU with hysteresis, learned pins

- Status: Proposed (Phase 0)
- Date: 2026-11-01

## Decision
- Per-layer pools of fixed-size slabs, allocated once. Budgets may differ per layer
  (the planner may give layers with flatter routing more slots).
- Slot states: `Free | Loading | Ready { leases }`. A leased or loading slot is never
  evicted.
- Default policy: LRU, upgradeable to **LFRU** (score `heat << 8 | recency ≤ 255`,
  ported from Colibri `tier.h`). Heat decays by halving per window.
- Promotion and replacement between tiers (RAM↔VRAM, pinned↔unpinned) need
  `hot > cold + cold/4 + 4` (25% + 4 hysteresis, overflow-safe). This prevents
  ping-pong.
- Prefetch eviction guard: a speculative load may evict a resident expert only if its
  LFRU score beats the victim by the same hysteresis.
- Learned hot set: routing counts persist per model (a versioned file next to the
  model's state directory, not inside the model folder). They are used to warm and pin
  within a configurable share of the budget, always leaving LRU capacity (Colibri's
  auto-pin cap rule).
- A copy of an expert in RAM and one in VRAM are **two ledger entries**. When VRAM
  holds an expert, its RAM copy is the first eviction candidate (Colibri
  `vkt_ram_first`).
- Accounting and metrics: hits, misses, in-flight joins, evictions, bytes read, and
  prefetch useful/wasted.

## Consequences
The policies are pure functions over slot metadata, unit-testable without I/O. Any
replacement policy must beat this baseline on recorded routing traces (a trace-replay
simulator, like Colibri's `residency_sim.py`).
