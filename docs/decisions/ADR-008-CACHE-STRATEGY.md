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
  mirrors; the learned, persisted hot set and pins; the prefetch guard (Phase 7, built:
  amendment 2).


## Amendment 2 (2026-11-03): router-lookahead prefetch and the eviction guard, Phase 7 slice 1

- **Prediction** (`qubivex_expert_stream::predict`, ported from Colibri's
  `pilot_prefetch`, PILOT=1). During decode only (one token per forward pass), after
  layer L's attention the forward pass applies layer L+1's MoE norm and router to the
  hidden state and hands the logits to `ExpertProvider::lookahead`. `Lookahead::predict`
  blends them with a per-layer moving average (`smooth` 0.3) and takes experts in
  probability order until the mass reaches `confidence` 0.92, at least top-K and at
  most top-K × `width` (1). The extra work is one router matmul per layer. Routing
  never reads these logits.
- **Issue order.** The provider holds the prediction and issues it right after layer
  L's `acquire_many`, so the prefetches queue behind that layer's demand misses.
  `ExpertStore::prefetch(keys, guard)` queues reads at `Priority::Prefetch`, best
  first. It never waits, never takes a leased or loading slot, and refuses every key
  without I/O threads.
- **Victims.** In a full pool a prefetch evicts the policy's victim, **never another
  expert of the same prediction**. Without that rule, a later candidate evicted an
  earlier one already resident; at 1 GiB this cost 15 extra reads per token and every
  gain.
- **The guard** (`policy::speculation_beats`, LFRU only). A prediction may evict a
  resident only if `heat << 8 | 255 > v + v/4 + (4 << 8)`, where `v` is the victim's
  LFRU score. This is Colibri's `pilot_realload` rule: heat against heat, with the 25%
  hysteresis and a margin of 4 heat. On by default (`--prefetch-guard off` disables it).
- **Stale speculation.** When layer L's demand arrives, its queued prefetches that are
  demanded are promoted to `Demand`, and the others are dropped, freeing their slots.
  A started read always finishes. A demanded expert whose layer is full of prefetches
  still loading waits for them instead of failing with `Full`. `Engine::end_turn` (and
  `generate`) drops every queued prefetch. Memory pressure at `Elevated` or above
  stops prediction and drops the queue (ADR-005).
- **Metrics.** `StoreStats::prefetch`: issued, present, refused, ready (the read was
  fully hidden), late (joined while loading), wasted (evicted unused, and its bytes),
  cancelled, and lead time. `Engine::prefetch_report` adds recall and precision against
  the router's real choice. `bench` records both under `experts.prefetch`.
- **Routing is identical with prediction on and off** (Phase 7 exit test,
  `prefetch_never_changes_routing_or_bits`). On the three fixtures, router logits,
  chosen experts, routing weights and output logits are bit-identical across prefetch
  off, guard on and off, width 1 and 3, and 0–8 I/O threads at 1–3 slots and all
  resident. The real model's 160 tokens match the Phase 6 exit record in every run
  below.
- **Measured** (M3 Pro, 5 threads, OLMoE Q4_K_M, 10 + 160 tokens, 1 warm-up + 3 runs,
  4 I/O threads; prediction recall and precision 79.7% at 8 predicted per layer):

  | Budget | Prefetch | Decode tok/s | Blocked per token | Read per token | Hit rate |
  |---|---|---|---|---|---|
  | 1 GiB | off | 19.87 | 8.29 ms | 426 MiB | 10.5% |
  | 1 GiB | guard on | 20.32 | 4.87 ms | 441 MiB | 18.8% |
  | 1 GiB | guard off | **20.50** | **3.95 ms** | 438 MiB | 22.6% |
  | 2 GiB | off | 23.28–23.62 | 0.49 ms | 169 MiB | 64.7% |
  | 2 GiB | guard on | 23.51 | 0.12 ms | 170 MiB | 65.4% |
  | 2 GiB | guard off | 23.07 | 0.02 ms | 199 MiB | 67.0% |
  | 3 GiB | off / guard on / guard off | 25.25 / 25.15 / 24.90 | 0.08 / 0.02 / 0.01 ms | 55 / 55 / 68 MiB | – |
  | all resident | off / on | 26.03 / 25.85 | 0 | 0 | 100% |

  Records: `docs/performance/baselines/2026-11-03-m3pro-phase7-*.json`.
- **What it says.** Prefetch meets the exit criterion: at 1 GiB, blocked time falls by
  41% with the guard and 52% without it. Throughput gains much less (+2–3%), because
  decode at 1 GiB is limited by read throughput, not latency. 8 I/O threads do no
  better than 4, and hiding a read does not make the copy cheaper. Prefetch only helps
  where it adds reads that are almost all used. At 2–3 GiB the pipelined demand path
  already hides nearly everything. There, without the guard, wasted prefetches add
  18–23% more reads and cost 1–2%; with the guard the result is within noise.
- **Defaults.** Prefetch stays **off** (`--prefetch` opts in): its gain is small and
  limited to tight budgets. When on, the guard is on: it never costs measurably and
  keeps most of the 1 GiB gain.
- **Not built yet:** lookahead beyond one layer (Colibri PILOT=2/3) and predicting
  layer 0 of the next token. The saturation throttle is amendment 3. A cancelled
  prefetch frees its slab buffer instead of keeping it: 197 allocations over 3 runs
  with 1 I/O thread, none at 4.

## Amendment 3 (2026-11-06): the saturation throttle, Phase 7

- **Measured first.** Warm reads come from the page cache, so a simulated disk
  (`fault::SlowDisk`, `bench --disk-mbps`, `--disk-latency-us`) gives every read a turn
  on one device with one bandwidth. It is a FIFO model: no real device's queueing, and
  no cold read has been measured.
- **The loss is real.** At `--ram-budget 1GiB` (2–3 slots per layer) on a saturated
  disk, prefetch with the guard costs 3–4%, because it reads 3.5% more bytes (441 vs
  426 MiB per token) and speed there follows the bytes read. A used speculative read
  gains nothing when no disk time is idle, and an unused one costs its transfer.
  Nothing was late or cancelled: the loss is waste, not stale queues.
  `--expert-cache-mb 1024` is not this regime: it gives 17–18 slots and hid the loss.
- **The signal.** `policy::SaturationGauge`: the share of wall time compute spent
  blocked on reads, from `StoreStats::wait_ns`, over windows of at least 100 ms. It
  holds at 60% and resumes below 40%. Disk utilisation was rejected: in early
  measurements 68% gained and 72% lost. The window closes at `Engine::end_turn`, and
  the verdict is kept, because idle time says nothing about the disk.
- **Where.** `Engine::forward`, beside the memory-pressure stop: while held, nothing is
  predicted (saving the router lookahead) and queued prefetches are dropped.
  `ExpertStore::prefetch` is not throttled, because the Metal pass uses it for experts
  it needs, not for speculation. On by default with `--prefetch`;
  `--prefetch-throttle off` disables it. `bench` records `held_rate`.
- **Measured** (M3 Pro, 5 threads, OLMoE Q4_K_M, 10 + 160 tokens, 1 warm-up, AC power;
  3 runs real disk, 2 at 6 and 2 GB/s, 1 at 500 MB/s; simulated disks add 80 µs per
  read; tokens identical in every run):

  | `--ram-budget 1GiB` | Prefetch off | Guard | Guard + throttle | Held | MiB per token |
  |---|---|---|---|---|---|
  | 6 GB/s | 9.93 | 9.52 | **9.99** | 100% | 426 / 441 / 426 |
  | 2 GB/s | 3.70 | 3.59 | **3.73** | 100% | 426 / 441 / 426 |
  | 500 MB/s | 0.99 | 0.96 | **1.00** | 100% | 426 / 441 / 426 |
  | real disk, warm | 21.4–22.3 | 20.7–21.3 | 20.4–22.5 | 0% | 426 / 441 / 441 |

  Records: `docs/performance/baselines/2026-11-06-m3pro-throttle-*.json`.
  On battery, so not baselines: 200 MB/s + 8 ms seek at 1 GiB, 48 tokens, one run:
  0.31 / 0.30 / 0.31, held 100%. `--expert-cache-mb 2048`, 86% hits: 6 GB/s
  32.2 / 32.3 / 32.4, held 0%; 2 GB/s 16.6 / 16.7 / 16.7, held 22%. The throttle costs
  nothing there, and prefetch gains nothing either.
- **What is still open.** On the real warm disk the gauge never holds (blocked share
  about 37%). The last two columns of that row are the same configuration, and they
  differ by more than the gap to prefetch off, so that loss is not resolved: pooled,
  prefetch is about 2.5% behind. Amendment 2's small gain (19.87 → 20.3) has not
  survived the faster Q4_K kernel. No regime measured here shows a throughput gain.
  Prefetch stays **off** by default. Owner decision (2026-11-06): it stays opt-in behind `--prefetch`, with no further work unless a real workload shows a gain.
