# ADR-005: One measured, role-driven memory planner

- Status: Proposed (Phase 0); built in Phase 5 (amendments 1–3), awaiting acceptance
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

## Amendment 2 (2026-11-03): the engine plans at load, Phase 5 slice 2

- **Load-time planning.** `Engine::load` reads the dense weights, measures available
  RAM with them resident, and plans (Colibri's after-load measurement). The store
  starts at `expert_cache_for(0)`, and the context is the plan's. `--expert-cache-mb`
  still fixes the cache and bypasses the planner. Without it, `run` and `bench`
  accept `--profile`, `--ram-fraction` and `--ram-budget`. A model that does not fit
  fails to load with the planner's reason.
- **Room sharing at run time.** Before each forward pass the engine projects the KV
  after it, lowers the store's capacity to `expert_cache_for(projected)` with
  `ExpertStore::set_capacity` (LRU eviction of unleased experts), and refuses the pass
  if it would cross the floor. The engine runs one sequence at a time until the
  scheduler (Phase 9), so the projection follows the current cache.
- **The KV grows in 256-position chunks** (capped at the context) instead of doubling.
  Allocation stays within one chunk of use, and the planner prices whole chunks.
- **Ledger charges:** dense at load; the expert cache (`store.used()`) and the KV
  (allocated capacity) after every forward pass. Activation scratch is not charged:
  it is short-lived and covered by the 0.5 GiB reserve.
- **Measured on the real model (M3 Pro, 300 tokens).** With every expert resident,
  the ledger read 3.90 GiB and RSS grew 3.75 GiB. Under a 1 GiB budget the ledger read
  510 MiB and RSS grew 441 MiB. The ledger is the larger: reserved but untouched KV
  pages (about 52 MiB) are not resident, and macOS's `ps` excludes compressed pages
  (this second cause is likely but unverified). The test allows a 256 MiB difference
  either way.
- **Known limit, for Phase 6.** With 2–3 slots per layer (1 GiB budget) the global LRU
  thrashes: 0 hits in 20,971 loads, and decode drops from 20.4 to 12.0 tok/s on a warm
  page cache. Per-layer slot pools and LFRU (ADR-008) are the fix. The output is still
  byte-identical.

## Amendment 3 (2026-11-03): the pressure monitor, Phase 5 slice 3

- **`PressureMonitor`** (`qubivex-memory`) is pure logic. It maps available RAM to
  four levels, each adding to the response before it:
  - `Normal`;
  - `Elevated` (below 1 GiB): stop prefetch;
  - `High` (below 512 MiB): shrink the expert cache by the shortfall below the
    recovery point;
  - `Critical` (below 256 MiB): cache at the one-slot-per-layer floor, refuse new
    requests.

  It escalates at once and steps down only 256 MiB above a threshold (hysteresis).
  While pressure lasts the cap never grows, and it never goes below the floor, so an
  active request always finishes. The thresholds are policy values; revisit them with
  recorded runs.
- **Runtime.** A `qubivex-pressure` thread samples available RAM every 500 ms
  (`EngineOptions::pressure_interval_ms`; `Some(0)` turns it off, leaving
  `Engine::observe_pressure` for callers and tests) and stops when the engine drops.
  The cache cap applies at the next forward pass, between tokens. `generate` (the
  request boundary) calls `Engine::admit`, which refuses under `Critical`. Prefetch
  does not exist until Phase 7, so `prefetch: false` is recorded but has no effect
  yet. Fixed-cache engines (`--expert-cache-mb`) have no monitor.
- **Measured with an external RAM hog** (opt-in test `pressure_hog`, real model, M3
  Pro, three passes):
  - The monitor reached `Critical`, a new request was refused, and the active request
    finished with identical tokens with its cache at the floor.
  - After the hog exited, pressure returned to `Normal`, the plan's cache came back,
    and the tokens matched again.
- **What the hog test showed about macOS.** The OS answers a new allocation partly by
  compressing other memory, so available RAM falls by less than the allocation: 1.1–1.6
  GiB for a 2.1–2.4 GiB hog of incompressible data. A hog of uniform bytes lowered it by
  only 0.4 GiB, because the compressor squeezes such pages to almost nothing. Thresholds
  and tests must allow for this.
- **Feedback.** Dropping the cache to the floor frees more RAM (about 3.4 GiB here)
  than a moderate hog takes, so the monitor relieves its own trigger and the cache
  regrows after the hysteresis margin. That is the intended behaviour. A cache that
  oscillates under sustained pressure would need a longer hold, and the monitor
  records enough to add one later.

Phase 5 is complete.
