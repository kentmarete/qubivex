# ADR-017: Continuous batching: sequences share a forward pass

- Status: **Proposed (2026-11-07)**, for the owner to accept
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
bit-identity tests, on both Metal placements. Until then `Engine::forward_batch` refuses
an engine with layers on Metal, and the combining point runs one segment per pass there.

### 7. Prefetch
The lookahead router runs only when a pass has one token: a batch routes to the union of
its tokens' experts, which one token's prediction does not cover. Job 8 measures whether
predicting the union is worth adding when experts stream; it changes no bits either way.

## The prototype (job 7)
Built in this job: `Transformer::forward_batch`, `Segment`, `Engine::forward_batch` (CPU
only, not yet called by the generation loop), `tests/batch.rs`, and the measurement
`examples/batch.rs`. Section 2 to 5 are job 8.

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
<!-- MEASUREMENTS -->

## Not built
- Sections 2 to 7, which are job 8.
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
