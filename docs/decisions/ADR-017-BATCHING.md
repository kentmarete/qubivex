# ADR-017: Continuous batching: sequences share a forward pass

- Status: **Accepted (2026-11-08)** by owner decision, including the order for job 8: the 2-to-5-row CPU kernel first, then sections 2 to 7
- Date: 2026-11-07

## Context
Spec §7.4 and Phase 8 (S8) ask for sequences that share real model execution batches,
in stages: dynamic batching with compatible grouping and per-sequence stopping (stage 2),
then continuous admission and removal, KV-aware scheduling and variable lengths (stage 3).
Batching must not change any request's semantics, and nothing may claim continuous
batching until sequences actually share a pass. Job 7 of `PLATFORM_AUDIT` asks for this
ADR with a measured prototype; job 8 builds it.

Today every running request has its own thread and its own `KvCache`, and calls
`Engine::forward` for each prefill chunk and each generated token. The engine's turn lock
runs those passes one at a time (audit decision 3, ADR-015): each pass already uses every
compute thread, and passes at once would compete for expert slots. Four conversations
decoding together therefore cost four passes per token, and each pass reads every dense
weight and every routed expert again.

Audit finding 2 named the obstacle: `Transformer::forward` took one token list and one
cache. Reading the pass shows how little of it is per sequence:

- Embedding, norms, the q/k/v/o projections, the router, routing, the experts and the
  head are per token. Matmuls are batch-invariant (`ComputeBackend::matmul`: each output
  element is the same function of its weight row and activation row whatever the row
  count), which is how a prompt already gives the same bits in one pass as token by token
  (invariant 1).
- The MoE layer is already expert-major: each routed expert runs once over every token
  routed to it. Tokens from several sequences join those groups unchanged.
- Only RoPE (the position), the KV write and attention depend on the sequence: each
  token needs its own position and must attend over its own sequence's keys and values.

## Decision
**A forward pass carries segments, one per sequence. Request threads keep their loop; the
engine's turn becomes a combining point, where the thread that takes it runs every
waiting step in one pass.**

### 1. The pass: `Transformer::forward_batch`
`forward_batch(&mut [Segment])`, where a `Segment` is a sequence's new tokens, its own
`&mut KvCache` and which logits it wants. The tokens of every segment are concatenated;
each token carries its segment and its position. Everything per token runs once over the
whole batch. RoPE uses the token's position, keys and values go to the token's own
cache, and attention reads only that cache, up to that position. The head runs once over
the rows any segment wants, and each segment gets back what `forward` would have returned
for it alone, with the same bits.

`forward` is now `forward_batch` with one segment, so there is one code path, and every
existing invariant test runs through it.

A segment can be a decode token or a prefill chunk, so a pass mixes them (variable
lengths, stage 3). A cache is counted as advanced only once the whole pass has succeeded,
so a failure anywhere leaves every cache as it was, and the same steps can run again.

### 2. The scheduler: combining at the turn
`Engine::forward` keeps its signature. Inside, a request thread posts its step (tokens,
its cache moved into a slot, logits wanted) to the engine's pending list and waits. The
thread that takes the turn drains the list, runs one `forward_batch` over every posted
step, hands each its logits and cache back, and releases the turn. A thread whose step
was taken by another just wakes with its result.

Consequences of this shape:

- **No added latency.** Nothing waits for a batch to fill. At one request a pass carries
  one segment, as now. Under load, steps accumulate while the current pass runs, so the
  next pass carries all of them: the batch width follows the load.
- **Continuous admission and removal fall out.** A sequence joins at its next step and
  leaves by not posting one. Stopping, sampling, stop strings, deadlines, cancellation
  and prefix reuse stay in each request's own loop (`generate_with`), unchanged, so they
  stay per sequence by construction.
- **The embedding API does not change.** An application calling `forward` or
  `run_request` from several threads gets batching with no server. `forward_batch` stays
  public for a caller that wants to drive the batch itself.
- **Fair by construction.** Every running sequence is in every pass it posted a step for,
  so one conversation can no longer wait out a full round of the others' turns. ADR-015's
  rule stands: priorities choose who starts, and the generation slots bound the width.
- **Caches move, not shared references.** A waiting thread moves its `KvCache` into its
  slot and receives it back, so the leader owns every cache it computes on and no cache
  is reachable from two threads at once. This is safe Rust and needs no new `unsafe`.

**Amendment 1 (2026-11-09, owner decision): a bounded wait keeps callers together.**
Measured in job 8, the rule above splits passes: the callers a pass answers are still
sampling their next token when the next leader takes the queue, so at N = 2 passes
mostly alternated one step each, and concurrent callers on OLMoE got 0.99x to 1.22x
where caller-driven batches got 1.17x to 1.46x. A leader now waits, up to
`EngineOptions::batch_wait_us` (default 1 ms; 0 turns it off), for as many posts as the
last pass answered with logits. Any post counts, so a lone caller never waits, and a
caller that has stopped costs one bounded wait, after which it is no longer expected
(`BatchStats::wait_timeouts` counts these). Sampling takes microseconds and a pass tens
of milliseconds, so the bound is reached only when a caller left. With the wait,
concurrent callers matched caller-driven batches: 1.19x to 1.53x at N = 2 to 8, every
pass at full width.

### 3. Pass composition: a token budget
A pass takes every waiting decode step first (one token each), then prefill chunks while
the total stays within a token budget (default: the prefill chunk, 256). A long prompt
therefore delays the others' next token by at most one chunk, as today, and decode is not
starved by prefill. The budget is an engine option (`EngineOptions`), and later a serve
flag.

### 4. Failure isolation
One request's bad step must not fail the others. Before the pass, the leader checks each
step on its own (context room, token ids in range) and answers a failing step with its
error alone. An error inside the pass, such as an expert read failure, is the model's,
not a sequence's: every step in that pass gets it, and since no cache advanced, each
request fails the same way it would have alone. A traced `forward` (tests) never
combines.

### 5. Memory
A pass reserves room for the sum of its segments' new positions, not each alone
(`fit_room` over the batch). Batching adds no KV: each sequence's cache was already
admitted and reserved by the scheduler (ADR-015), so KV-aware scheduling is the existing
admission, and the batch width is bounded by the generation slots. Activations grow with
the tokens in a pass, which the token budget bounds.

### 6. Metal
The Metal pass (ADR-011) already runs `s` tokens per layer submission, and its kernels
reproduce the CPU bits. Batching it means one KV write and one attention dispatch per
segment, against that segment's device buffers and positions, inside the same command
buffer; every other kernel runs over all tokens unchanged. Job 8 adds this with the same
bit-identity tests, on both Metal placements.

Built in job 8: attention is three dependent dispatches (scores, softmax, values), so the
segments' attentions run stage by stage (`Batch::attention_many`), each stage for every
segment at once, not as one group of whole attentions, which races. The combining point
and `Engine::forward_batch` run on Metal as on the CPU.

### 7. Prefetch
The lookahead router runs only when a pass has one token: a batch routes to the union of
its tokens' experts, which one token's prediction does not cover. Job 8 measures whether
predicting the union is worth adding when experts stream; it changes no bits either way.

Measured in job 8 (2026-11-10, on power, OLMoE Q4_K_M, CPU, `examples/batch.rs`, which
reports the store's blocked time as a share of decode time): **not worth adding.**
Predicting the union could save at most the time a batched pass waits on reads, and
batching already hides most of it. Experts load in the background while the pass computes
the experts it already holds, and a wider pass gives each load more compute to overlap.
With prefetch off, sequences taking turns wait 5% to 19% of the time; concurrent callers in
shared passes, at N = 2 to 8, wait 0% to 2% with a 1024 MiB expert cache and 2% to 5% with
512 MiB. No union predictor is built, and the lookahead stays for one-token passes. This
fits the owner's rule from 2026-11-06 that prefetch gets no further work unless a real
workload shows a gain.

## The prototype (job 7)
Built in this job: `Transformer::forward_batch`, `Segment`, `Engine::forward_batch` (CPU
only, not yet called by the generation loop), `tests/batch.rs`, the measurement
`examples/batch.rs` and the `batch_s<N>` kernel benchmark. Section 2 to 5 are job 8.

### Tests
`tests/batch.rs`, on the four fixtures (three MoE, one dense):

- A pass mixing two decode steps at different positions, a prefill chunk with every
  logit and a fresh one-token prompt gives each segment the bits of its own pass.
- Four sequences decoded greedily in lock-step, one pass per step, with sequences
  leaving at different steps, give each the tokens it gets alone.
- A failing segment (a full context; an out-of-range token found inside the pass) fails
  the pass and leaves every cache's length unchanged; the next pass gives the bits of a
  run where nothing failed.
- A one-expert cache, where a batch needs the union of four prompts' experts, waits for
  the slot instead of failing, with the same bits as an all-resident cache.
- Empty batches and empty segments.

Changing the rules in turn: attention reading the first segment's cache, positions not
offset by the segment, and RoPE at the batch index each fail tests. Dropping the
rollback survived, which showed it guarded only a failure in the final head matmul;
counting positions after the head instead removed the need for it, and
`KvCache::rollback` was deleted.

### Measurements
On power, M3 Pro, OLMoE-1B-7B-0125-Instruct Q4_K_M, 5 compute threads, 2026-11-08.
`examples/batch.rs`: greedy decode of N different prompts, 32 tokens each, one pass per
sequence per step against one `forward_batch` per step, median of 3 interleaved runs.
Prefill is per sequence and not timed.

All experts resident:

| N | sequential tok/s | batched tok/s | speed-up | tokens identical |
|---:|---:|---:|---:|---|
| 1 | 33.34 | 37.48 | 1.12x | yes |
| 2 | 41.01 | 28.22 | 0.69x | yes |
| 4 | 36.69 | 32.78 | 0.89x | yes |
| 8 | 35.68 | 35.28 | 0.99x | yes |

1024 MiB expert cache (experts stream):

| N | sequential tok/s | batched tok/s | speed-up | tokens identical |
|---:|---:|---:|---:|---|
| 1 | 32.69 | 34.93 | 1.07x | yes |
| 2 | 31.05 | 25.89 | 0.83x | yes |
| 4 | 27.78 | 16.33 | 0.59x | yes |
| 8 | 28.15 | 28.12 | 1.00x | yes |

At N = 1 both columns run the same code, so their 7 to 12% gap is this harness's noise.
**The tokens are identical at every size, but sharing a pass gives no throughput on
the CPU today, and costs up to 41% at N = 2 to 4.**

The cause is the CPU kernel at 2 to 5 activation rows, not the pass. On Apple silicon,
one row uses a fused dequantise-and-dot kernel; 2 to 5 rows fall back to the portable
path (dequantise each weight row into a buffer, then one portable dot per activation
row); the register-blocked kernel starts at 6 (`tile::MIN_ROWS`). The new `batch_s<N>`
group in `benches/kernels.rs` measures it, on the compute team (5 threads):

| rows | Q4_K expert 1024×2048 | per row | Q6_K head 50304×2048 | per row |
|---:|---:|---:|---:|---:|
| 1 | 42.0 µs | 42.0 µs | 2.91 ms | 2.91 ms |
| 2 | 129.9 µs | 65.0 µs | 12.06 ms | 6.03 ms |
| 3 | 155.2 µs | 51.7 µs | 13.45 ms | 4.48 ms |
| 4 | 180.2 µs | 45.1 µs | 14.20 ms | 3.55 ms |
| 6 | 226.2 µs | 37.7 µs | 15.64 ms | 2.61 ms |
| 8 | 259.9 µs | 32.5 µs | 17.33 ms | 2.17 ms |

So a two-row matmul costs 1.5× (expert) and 2.1× (head) two one-row matmuls, and only
from 6 rows does sharing the weight read pay, by at most 1.3× at 8 rows. In a decode
batch every dense matmul and the head see N rows, and an expert sees the few tokens
routed to it, mostly 1 or 2.

Expert sharing is small at these widths anyway. With 8 of 64 experts per token,
uniform routing gives N tokens about 64·(1 − (56/64)^N) distinct experts per layer:
15 for 16 expert runs at N = 2, 26.5 for 32 at N = 4, 42 for 64 at N = 8. The saving
that matters is the dense weights (attention projections and head, about a third of a
token's weight bytes), read once per pass instead of once per sequence, and that saving
depends on a matmul whose cost grows slowly with its row count.

**Consequence for job 8: a fused kernel for 2 to 5 rows comes first.** It dequantises
each weight block once, in registers, and accumulates it into up to 5 activation rows,
with each row's lane order unchanged so the bits stay those of the one-row kernel
(the same argument as `tile`, ADR-002 amendment 1). The combining point (section 2) is
built only after that kernel shows a speed-up on this table; without it, combining
would make concurrent decode slower than taking turns.

Single-sequence regression check (`qubivex bench`, OLMoE, all resident, 3 interleaved
rounds of main 7fa510b against this branch, defaults of 1 warm-up and 5 runs):

| | main | branch |
|---|---:|---:|
| decode tok/s at the median step, median of rounds | 42.41 | 42.09 |
| prefill tok/s, median of rounds | 43.10 | 43.32 |
| tokens match main | | yes, every round |

The 0.8% decode difference is inside the rounds' spread (41.87 to 43.18 on main), so
routing `forward` through `forward_batch` costs nothing measurable. The `batch` example
reports less than `qubivex bench` at N = 1 because it times whole steps including
argmax and the per-step vectors, where `bench` reports the median step.

## Not built
- Sections 2 to 7, which are job 8, after the 2-to-5-row CPU kernel the measurements
  call for.
- **Paged KV.** Each sequence keeps its contiguous cache. Attention runs per (token,
  head) over that sequence's rows, so a batch needs no shared layout.
- **Speculative decoding, beam search, `n > 1`.** They need several sequences per
  request; the pass supports it, the request API does not.

## Consequences
- The turn lock's exception to spec §7.3 (audit decision 3) ends with job 8: requests
  still take turns, but a turn serves all of them.
- Invariant 1 widens: a sequence's tokens do not depend on what else shares its pass.
  The batch tests guard it on the CPU, and job 8 adds Metal.
- `forward` and `forward_batch` are one code path, so the single-sequence path pays a
  few small vectors per pass for the layout. No kernel changed.
