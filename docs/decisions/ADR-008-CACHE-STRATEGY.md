# ADR-008: Expert cache: per-layer slab pools, LFRU with hysteresis, learned pins

- Status: Proposed (Phase 0); pools, slabs and LFRU built in Phase 6 (amendment 1), awaiting acceptance
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

## Amendment 1 (2026-11-03): per-layer pools and LFRU, Phase 6 slice 1

- **Pools.** `ExpertStore::new(source, &[PoolSpec], Policy)` keeps one pool per expert
  layer: a slot count and a slot size (the layer's largest expert). A miss evicts only
  an unleased expert of its own layer. The engine turns the plan's byte budget into
  slots with `qubivex_memory::distribute`, the function `qubivex plan` reports, so the
  plan and the store agree; as the KV grows, `Engine` calls `ExpertStore::set_pools`
  instead of the old byte `set_capacity`. A fixed `--expert-cache-mb` is spread the same
  way and never goes below one slot per layer.
- **Slot states.** `Loading | Ready { leases }`, as decided; `Free` is the absence of an
  entry. A leased or loading slot is never evicted. `StoreError::Full` now names the
  layer whose slots are all leased.
- **Slabs.** A miss in a full pool reads into the evicted expert's buffer, so once the
  pools fill, a streaming decode allocates no expert memory (`StoreStats::slab_allocs`
  counts the allocations: 0 during the measured 1 GiB runs). Buffers are allocated on
  first use, not up front, so an all-resident run's RSS is unchanged, and they are
  freed when a pool shrinks. Bytes live in `Arc<Vec<u8>>`; the store still has no
  `unsafe`.
- **LFRU is the default** (`Policy::LFRU`; `--cache-policy lru` keeps LRU). Heat is
  counted per layer for every access, resident or not, and halves every 256 accesses
  of the layer. The score and the 25% + 4 hysteresis (`policy::lfru_score`,
  `policy::beats`) are pure functions with unit tests. The hysteresis has no caller
  until prefetch (Phase 7) and tier promotion (Phase 8).
- **Trace replay.** `qubivex bench --routing-trace FILE` records every acquire;
  `cargo run --release -p qubivex-expert-stream --example replay -- FILE` replays it
  through the real store over a null source, against a global LRU of the same size.
  OLMoE, 3 runs of the benchmark prompt (50,625 acquires), hit rate by slots per layer:

  | Slots/layer | Global LRU | Pool LRU | Pool LFRU (window 256) |
  |---|---|---|---|
  | 2 | 0.0% | 0.0% | 7.3% |
  | 3 | 0.0% | 0.0% | 13.0% |
  | 6 | 0.0% | 2.6% | 25.0% |
  | 8 | 24.4% | 24.4% | 30.9% |
  | 16 | 44.1% | 44.5% | 51.3% |
  | 32 | 73.8% | 74.1% | 79.6% |

  Per-layer pools alone do not help LRU: a layer's experts are visited in ascending
  index order, so the misses at low indices evict the retained high-index experts
  before they are reached. Frequency is what keeps the hot experts. Windows of 64 to
  1024 and no decay are within 1.5 points of each other on this trace; 256 stays as the
  default because a trace of three identical runs flatters never decaying.
- **Measured end to end** (M3 Pro, 5 threads, `--ram-budget 1GiB`: 9 layers × 3 and
  7 layers × 2 slots, 149 MiB; 1 warm-up and 3 runs; tokens identical to the global-LRU
  record):

  | Store | Hit rate | Read per token | Decode tok/s (median step) |
  |---|---|---|---|
  | Global LRU (Phase 5, `c70dfa0`) | 0.0% | 479 MiB | 12.1 |
  | Pools, LRU | 0.0% | 479 MiB | 12.0 |
  | Pools, LFRU | 10.9% | 427 MiB | 14.9 |

  All resident, the decode step median is 38.04 ms against the 38.82 ms baseline,
  with matching tokens. Records: `docs/performance/baselines/2026-11-03-m3pro-1gib-lfru.json`
  and `…-1gib-pool-lru.json`; the global-LRU run's record was not kept (its numbers are
  from its text summary).
- **Not built yet (later Phase 6 slices):** I/O worker threads and the priority
  queue, `acquire_many`, extent coalescing, direct I/O and io_uring, multi-drive
  mirrors; the learned, persisted hot set and pins; the prefetch guard (Phase 7).

