# ADR-006: Synchronous lease-based expert store with dedicated I/O threads

- Status: Proposed (Phase 0). **This deviates from the specification's async trait.**
  The queue, I/O threads and `acquire_many` were built in Phase 6 (amendment 1); awaiting acceptance.
- Date: 2026-11-01

## Context
A decode token performs layers × top-K expert lookups (OLMoE: 128). Most are cache
hits. Colibri's proven design is a lease (`busy` refcount) with in-flight dedup on a
condvar, plus a loader pool, with no async runtime.

## Decision
- `ExpertStore::acquire` / `acquire_many` are **synchronous** and return a `Lease`
  that releases on `Drop` and is `!Clone`. A hit is a hash probe plus an atomic
  increment.
- Misses go to a **bounded priority queue** (Demand > Prefetch > Warm) served by a
  fixed pool of dedicated I/O threads that do `pread` into preallocated slot slabs.
  io_uring (Linux) and O_DIRECT/`F_NOCACHE` are opt-in modes, defaulted from
  measurements.
- `acquire_many` submits all of a layer's misses before waiting on any.
- In-flight dedup: a second acquirer of a loading expert waits on the same load (a
  condvar, not a sleep poll).
- `ExpertSource` abstracts container layouts: SafeTensors (three tensors per expert,
  adjacent extents coalesced), GGUF (per-expert slices of the 3-D stacked expert
  tensors), and an optional **expert-packed** layout (one extent per expert, as in
  Colibri's `merged_weight`), produced by `qubivex models pack`. The original files
  always stay directly runnable.
- Every read size is validated against the config-derived size before the read.
- Tokio is used only at the gateway. `load_async` is a thin facade for non-hot-path
  callers.

## Consequences
- No executor overhead or blocking-in-async hazards on the hot path.
- Cancellation of prefetch is cooperative (queue entries are dropped). A demand read
  in progress always completes.

## Amendment 1 (2026-11-03): the I/O queue, I/O threads and `acquire_many`, Phase 6 slice 2

- **`acquire_many(keys) -> Batch`.** Every hit of the set is leased at once, so a
  miss of the same set can never evict it. Every miss that has a free slot of its
  layer is queued at `Demand` before the caller waits on any. The rest are deferred
  and issued as the caller releases earlier leases. `Batch` is an iterator that yields
  leases in completion order (hits first). A layer with fewer slots than routed
  experts therefore pipelines through the slots it has. `acquire` is a batch of one.
- **The MoE layer requests a whole layer at once.** `ExpertProvider::with_experts`
  has a default that loops `with_expert`. The order is unspecified, and each expert
  writes only its own rank slots before the rank-order combination, so arrival order
  cannot change a bit. A fixture test runs 0, 1, 4 and 8 I/O threads at several slot
  counts and checks bit-identical logits.
- **Queue and threads.** One priority queue (Demand > Prefetch > Warm, FIFO within a
  priority) sits under the store's lock. It is bounded by the slots, since every
  queued load holds one. `ExpertStore::with_io_threads(n)` starts `n` `qubivex-io-*`
  threads, which the store joins on drop.
  - Default: 4 threads (`--io-threads`, `EngineOptions::io_threads`).
  - With 0 threads, the waiting caller performs reads itself, one at a time and only
    when it has nothing else to take. Reading ahead on the caller's own thread was
    measured 15% slower than reading on demand (12.7 against 14.9 tok/s), with the same
    bytes and the same slab reuse: each `pread` cost more.
- **Errors and cancellation.**
  - A failed read becomes a `Failed` entry. Every waiter receives the error, and the
    last one removes the entry. Waiters no longer retry the load themselves.
  - Dropping a batch cancels its queued loads that no thread has started (counted as
    `cancelled`). A started read always completes and leaves an ordinary unleased
    expert.
- **Coalescing.** The model source merges an expert's parts that are adjacent both in
  the file and in the expert's bytes into one `pread` (SafeTensors experts stored
  gate, up, down back to back). GGUF stacks each part across experts, so its three
  parts stay three reads. Coalescing across experts belongs with the I/O-strategy
  matrix (slice 3).
- **Measured** on the M3 Pro (OLMoE Q4_K_M, 5 compute threads, `--ram-budget 1GiB`,
  LFRU, 1 warm-up + 3 runs). Tokens are identical to the slice-1 record in every row.

  | I/O threads | Decode tok/s | Step median (P95) | Blocked on loads per token | Prefill, 128 tokens |
  |---|---|---|---|---|
  | slice 1 (synchronous) | 14.9 | 67.0 ms | — | — |
  | 0 (reads on demand) | 15.3–15.6 | — | — | 42.7 tok/s |
  | 1 | 17.1 | 58.4 ms (61.9) | 16.8 ms | — |
  | 2 | 20.1 | 49.7 ms (55.1) | 8.0 ms | 48.6 tok/s |
  | **4 (default)** | **20.2** | **49.4 ms (52.6)** | **7.6 ms** | **49.5 tok/s** |
  | 6 | 20.2 | 49.5 ms (52.5) | 7.5 ms | 49.0 tok/s |

  Record: `docs/performance/baselines/2026-11-03-m3pro-1gib-lfru-io4.json`.
  - All resident: the decode step median is 38.06 ms (the baseline is 38.82 ms) and
    the tokens match. Prefill at 128 tokens is 45.0 against 45.2 tok/s.
  - At a 1 GiB budget, prefill (49.5 tok/s) beats all-resident prefill (45 tok/s).
    That is observed, not yet explained; slice 3's I/O matrix should look at it.
- **Stress tests** (`crates/qubivex-expert-stream/tests/io.rs`, 50 repeated runs with
  no failure):
  - every expert of a batch comes out exactly once at 1–8 slots and 0–4 threads;
  - with every expert resident, 8 threads asking for overlapping sets read each expert
    exactly once;
  - accounting survives contention with `Full`;
  - cancellation of queued loads;
  - error delivery to every waiter;
  - parallel reads;
  - shutdown.

  A third loom model covers a batch racing an acquire and being dropped mid-way.
- **Not built yet (slice 3):** `O_DIRECT`/`F_NOCACHE`, io_uring, multi-drive mirrors,
  per-device queues, the I/O strategy matrix, and the exit check at a 1 GB expert
  budget. Prefetch at `Priority::Prefetch` arrives in Phase 7.

