# ADR-006: Synchronous lease-based expert store with dedicated I/O threads

- Status: Proposed (Phase 0). **This deviates from the specification's async trait.**
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
