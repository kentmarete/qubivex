# ADR-006: Synchronous lease-based expert store with dedicated I/O threads

- Status: Proposed (Phase 0). **This deviates from the specification's async trait.**
  The queue, I/O threads and `acquire_many` were built in Phase 6 (amendment 1), and
  direct I/O in slice 3 (amendment 2); io_uring is deferred. Failure behaviour and the
  fault-injection suite: amendment 3; deadlines, retries, mirrors and recorded
  checksums: amendment 4; holding several leases of a batch: amendment 5. Awaiting
  acceptance.
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
  budget. Prefetch at `Priority::Prefetch` arrives in Phase 7 (built: ADR-008
  amendment 2, which also covers promotion and cancellation of queued prefetches).


## Amendment 2 (2026-11-03): direct I/O, the I/O matrix and the exit check, Phase 6 slice 3

- **Direct I/O, off by default** (`--direct-io`, `EngineOptions::expert_io`,
  `ExpertIo::Direct`). The model source opens the weight files so reads bypass the page
  cache:
  - macOS: `F_NOCACHE`, which takes any offset and length, so reads go straight into the
    expert's slab;
  - Linux: `O_DIRECT`, which needs 4 KiB-aligned offsets, lengths and buffers. GGUF
    extents are not aligned that far, so each extent is read as the aligned window around
    it into a per-thread bounce buffer (safe code: an over-allocated vector) and copied
    out. A window may run past the end of the file; a read that ends off a block
    boundary is taken as the end of the file;
  - other systems, and filesystems without `O_DIRECT` (`EINVAL`), refuse it with an
    error instead of silently reading buffered.

  The bench record holds `config.expert_io` (`buffered`, `f_nocache` or `o_direct`).
- **Tests.** A strict reader that fails on any misaligned offset, length or buffer, and
  returns short reads, checks the window logic on every platform: extents at unaligned
  offsets, extents ending at an unaligned end of file, and a read past the end (an
  error, not zeros). Direct reads of every fixture expert equal the buffered bytes, and
  so do sampled experts of the real model (env-gated). The engine's I/O-thread test has
  direct rows with bit-identical logits.
- **The I/O matrix, warm** (M3 Pro, OLMoE Q4_K_M, `--ram-budget 1GiB`, 149 MiB of slots,
  LFRU, 128 tokens, 1 warm-up + 3 runs). Tokens are identical to the slice-2 record in
  every row. Records: `docs/performance/baselines/2026-11-03-m3pro-1gib-lfru-io4.json`
  (buffered, slice 2) and `2026-11-03-m3pro-1gib-direct-io4.json`.

  | I/O threads | Buffered tok/s | `F_NOCACHE` tok/s | Blocked per token (buffered / direct) |
  |---|---|---|---|
  | 0 | 15.5 | 15.4 | 27.4 / 27.7 ms |
  | 1 | 18.1 | 17.8 | 13.4 / 14.3 ms |
  | 2 | 20.5 | 20.3 | 7.4 / 7.6 ms |
  | **4 (default)** | **20.5** | 20.2 | 7.7 / 9.0 ms |
  | 8 | 20.5 | 19.3 | 7.7 / 11.5 ms |

  Prefill at 128 tokens: 47.8 tok/s buffered and 48.1 tok/s direct. The hit rate (about
  10.6%) and the bytes read per token (about 449 MB) are the same in both modes.
- **These are warm numbers only.** The whole model was in the page cache from the first
  run on (`vm_stat` file-backed pages stopped growing), and macOS serves an `F_NOCACHE`
  read from pages already cached. The table therefore says that direct I/O costs nothing
  at 2–4 threads and a little at 8. It does not show what direct I/O is for: reading
  without filling the page cache, and cold-disk rates. Cold rows need the page cache
  emptied before each run (`sudo purge` on macOS, `drop_caches` on Linux) and are left
  for later. **The default stays buffered.**
- **Exit check (Phase 6): passed.** OLMoE at `--ram-budget 1GiB` (149 MiB of expert
  slots against 3.63 GiB of experts, about 4%), 10 prompt + 160 generated tokens:
  tokens identical to the all-resident record (`2026-11-03-m3pro-decode.json`), 19.9
  tok/s at the median step, 10.4% hit rate, 427 MiB read and 7.9 ms blocked per token,
  0 slab allocations. Record: `2026-11-03-m3pro-phase6-exit-1gib.json`. The duplicate-
  request and cancellation stress tests are those of amendment 1.
- **Not built, moved out of Phase 6:**
  - io_uring. It needs a new dependency with `unsafe` submission, cannot be built or
    measured on the development Mac, and the plan keeps it off until measured. It
    returns with a Linux measurement host, together with `O_DIRECT` cold rows;
  - multi-drive mirrors and per-device queues (no second drive yet);
  - cross-expert coalescing: in GGUF, an expert's parts are spread over three stacked
    tensors, so neighbouring experts are adjacent only within a part. A layer's misses
    are rarely neighbours (10.6% hits, top-8 of 64).

## Amendment 3 (2026-11-04): failure behaviour and the fault-injection suite

IMPROVEMENTS_EVALUATION §10 asks for an `ExpertSource` wrapper that injects faults, and
a suite showing that in every case nothing panics, no wrong token is emitted, the ledger
returns to baseline and the next request succeeds. Planned for Phase 3, built now.

- **`qubivex_expert_stream::fault`.** `FaultySource` wraps any source and applies
  `Rule`s (a target: any read, a layer or one expert; reads to let through first; reads
  to fail) with a `Fault`: `Eio`; `ShortRead { keep }` (the rest of the buffer is
  garbage, the read reports end of file); `Corrupt { offset, xor }` (reported as a
  success); `Panic` (after writing half the buffer); `Delay`. `VerifiedSource` checks
  every read against a checksum recorded from a known-good read (`checksum`: four
  multiply-rotate lanes, so any change within one 8-byte word always changes it), and
  refuses an expert it has no checksum for. `Engine::load_with_source` puts any wrapper
  between the model's source and the store.
- **Behaviour (unchanged except where noted):**
  - A failed read fails every waiter with `StoreError::Load`, is never published, and
    leaves no entry: the next acquire reads again. A failed prefetch has no waiter, and
    the demand that follows reads again.
  - **New:** a panicking source is caught in the load (`catch_unwind`) and becomes a
    load error. Before, it killed the I/O thread with the load unresolved, so its
    waiters waited forever; without I/O threads it unwound through the caller.
  - **New:** slab allocation is fallible (`PageAligned::try_zeroed`): a slot the
    allocator refuses is a load error, not an abort.
  - **New:** a forward pass that fails rolls the host KV cache back to the positions it
    held (`Transformer::forward`). Before, the layers before the failing one
    kept that step's keys and values, so the next step on the same cache attended to
    them. Generation drops its cache on error anyway; a server reusing a conversation's
    cache needs the rollback. The Metal pass advances the cache only on success.
  - **New:** the ledger is updated after a failed pass too.
  - Corrupt bytes that a read reports as good are invisible to a plain store (the size
    is right). Only `VerifiedSource` catches them. It is not on by default: recording
    reads every expert once, which only makes sense where the checksums are stored, at
    install time. That is a Phase 11 `models` item.
- **Tests.** Store (`qubivex-expert-stream/tests/fault.rs`, 0 and 4 I/O threads): EIO,
  short read and panic each fail one acquire, and the next succeeds; one failure in a
  layer spares the rest, whether the batch is consumed or dropped at the error; every
  waiter on a failed read gets the error; corruption gets through unverified and is
  caught verified; allocation failure (a 4 PiB slot) is an error and other layers keep
  working; dropping a batch mid-layer cancels queued loads and leaks nothing; a failed
  prefetch does not fail the demand. Each ends with no lease, load or failed entry, and
  `used` equal to the resident bytes (`ExpertStore::occupancy`). Engine
  (`qubivex-runtime/tests/faults.rs`, the three fixtures, CPU with every expert
  resident and with one slot per layer, 0 and 4 I/O threads, faults mid-prefill and
  mid-decode, EIO, short read, panic and checksummed corruption): the request fails
  naming the expert, the tokens emitted before it are the fault-free ones, nothing is
  held, the ledger matches the store, and the next request gives the fault-free tokens.
  With every expert resident, the ledger and occupancy after recovery equal a
  fault-free engine's. A decode step that fails can be followed on the same cache by a
  different token, and gives the fault-free logits bit for bit. This test fails
  without the rollback. The same suite runs on Metal whole layers. Removing
  `catch_unwind` fails the store panic test.
- **Speed: unchanged.** Real OLMoE Q4_K_M, M3 Pro, 128 tokens, 1 warm-up + 3 runs,
  warm: at `--expert-cache-mb 1024`, 21.1 tok/s (main) against 21.5 (this change); with
  the planned cache, 22.5 against 21.6. The run-to-run spread is larger than the
  difference (17.9–23.9). The tokens are identical in every run.
- **Not covered yet:** mirrors (retrying on another drive; none exist), a read that
  hangs forever (no read timeout), and per-request isolation in the server (Phase 9).

## Amendment 4 (2026-11-04): read deadlines, retries, mirrors and recorded checksums

Amendment 3 left three gaps: a read that never returns, no second copy to read from, and
checksums that existed but were never on. All three are closed here.

- **Deadlines (`ReadTimeout`, `StoreError::Timeout`).** A blocked `pread` cannot be
  interrupted, so the deadline belongs to the waiter. A caller waits for one expert at
  most `floor + size / min_rate`, counted from when it starts waiting (the engine's
  default is 5 s + size at 10 MB/s; `--read-timeout SECS|off`). Then:
  - the caller gets `StoreError::Timeout`, and the request fails naming the expert;
  - a queued read nobody waits for any more is cancelled;
  - a read already started is left to finish, and if it succeeds its bytes are
    published: a late good read is still good;
  - a later caller of the same expert joins the stuck read and times out in turn.

  Reads running past their deadline are counted (`Occupancy::stuck`). When every I/O
  thread is stuck, `Engine::admit` refuses new requests (`RuntimeError::Storage`)
  instead of queueing them behind a dead drive. Dropping the store detaches stuck
  threads instead of joining them, so shutdown does not hang. Without I/O threads the
  compute thread reads its own misses and cannot time out on those; the store's own
  default is no deadline, and the engine sets one.
- **Retries.** A failed read is attempted again inside the load before any waiter sees
  the error (`set_read_retries`). The engine retries once by default
  (`--read-retries N`). Every source error is retried, including a panic and a
  checksum mismatch; an allocation failure is not.
- **Mirrors (`mirror::MirroredSource`).** Reads go to the first healthy copy of the
  weight files and fail over to the next on an error or a panic. A copy that fails 3
  reads in a row is skipped for 30 s, then tried again; a success clears its count.
  When every copy is skipped, all are tried anyway. `ModelExpertSource::mirror_at`
  opens a copy under another directory (same relative paths; sizes must match the
  index). `Engine::load_mirrored` and `--mirror DIR` (repeatable) wire it up.
  Each copy is verified separately, so corrupt bytes on one copy fail over too. A hung
  read on one copy is not failed over: it blocks its thread, and the deadline above
  handles it. Hedged reads (asking a second copy after a delay) would fix that and are
  left until a second drive can be measured, together with per-device queues.
- **Recorded checksums (`qubivex_runtime::checksums`).** `qubivex models verify`
  records every expert's checksum in `experts.qvxsum` next to the model when there is
  none, and otherwise re-reads every expert and lists the ones that differ (an offline
  scrub). `qubivex models import --write` records them as well (`--no-checksums`
  skips it). The file also records each weight file's size and modification time.
  When they no longer match, the checksums are **stale**: they are reported (`run`
  warns) and not used, because a deliberately replaced model would fail every read.
  Silent corruption leaves size and time alone, so it is caught. The engine's default,
  `--verify-experts auto`, verifies every read when fresh checksums exist; `off`
  never does; `require` refuses to load without them. Bench records the mode.
- **The checksum** moved from four lanes to eight. One M3 Pro core, on one OLMoE
  expert (3.4 MiB): four lanes did 21 GiB/s, eight do 39 GiB/s (84 µs per expert);
  `memcpy` does 54 GiB/s. Bench: `qubivex-expert-stream/benches/checksum.rs`.
- **Measured** (OLMoE Q4_K_M, M3 Pro, 128 tokens, 1 warm-up + 3 runs, 3 alternating
  rounds, warm). Recording all 1024 experts (3.63 GiB) took 0.9 s and a full check
  0.4 s, both from the page cache. Decode, verified against unverified:

  | Cache | Verified tok/s | Unverified tok/s | Blocked per token |
  |---|---|---|---|
  | 1 GiB (streaming) | 22.3 / 23.2 / 23.0 | 23.2 / 23.5 / 23.4 | 2.0–6.6 / 2.2–3.0 ms |
  | Planned (resident) | 24.6 / 25.7 / 26.3 | 25.2 / 24.8 / 25.8 | 0.0–0.1 ms |

  Verification costs about 2–3% when every token streams experts, and nothing
  measurable when they stay resident. The tokens are identical in every run. **It is
  on by default**; `--verify-experts off` remains for streaming at the edge.
- **Tests.** Store: a hung read times out while the layer's other experts load; a
  second caller joins the stuck read; the late read is published; with every thread
  stuck, queued reads time out and are cancelled and the drop does not wait; retries
  recover a failed read, and the error counts the attempts. Mirrors: a failing copy is
  skipped, then tried again after the cooldown and preferred once it reads well;
  every copy failing gives one error naming each; corrupt bytes on one copy fail over
  through the store. Engine: a transient failure is retried with fault-free tokens; a
  hung read mid-decode fails the request with a timeout, a lone stuck thread makes the
  engine refuse new requests, and requests succeed again once the read returns; the
  model's files truncated under a running engine are read from a mirror with fault-free
  tokens, while the same engine without a mirror fails, and a mirror of the wrong size
  is refused. Checksums: record, save and load round-trip; the scrub of a clean model
  is empty; silent corruption of one expert (mtime kept) is named by the scrub,
  fails a verified request, and is read through a mirror with the right tokens;
  touching a weight file makes the checksums stale, and `require` then refuses to load.
- **Still open:** hedged reads and per-device queues (need a second drive), and a
  read timeout for a compute thread that reads its own misses (0 I/O threads).

## Amendment 5 (2026-11-04): holding several leases of one batch

The compute side now computes the experts of a layer concurrently (ADR-002 amendment 1),
so a caller can hold several leases of one `Batch` at once. The blocking `Batch::next`
still fails with `StoreError::Full` when the remaining experts need slots that leases
hold and no load is in flight; with the caller's own leases out, that would be a false
failure (or, on one compute thread, a deadlock).

`Batch::try_next` is `next` without waiting. It returns `Next::Lease` for an expert that
is ready (or an error for one), `Next::Later` where `next` would wait for a load or fail
with `Full`, and `Next::Done` at the end. Without I/O threads it never reads on the
caller's thread. The protocol is: take what `try_next` gives, compute and release, and
call the blocking `next` only while holding nothing. `StoreProvider::with_experts` does
this: each finished expert releases its lease and immediately starts whatever has become
ready, so loads into freed slots begin as soon as a slot frees, as they did when experts
ran one at a time. Tests: `try_next_lets_a_caller_hold_several_leases` and
`try_next_does_not_wait_for_loads` (store), and `compute_threads_never_change_bits`
(engine, one-slot pools, 1 and 3 threads).
