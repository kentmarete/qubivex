# ADR-013: The serving gateway (Phase 9)

- Status: Accepted (2026-11-04), built with Phase 9.
- Date: 2026-11-04
- (ADR-012 stays reserved for the graph IR, ARCHITECTURE_DECISIONS §1.)

## Context
QUBIVEX_DESIGN §4.7 and MIGRATION_PLAN Phase 9 ask for an Axum gateway with OpenAI's
`/v1/chat/completions` and `/v1/completions` (SSE), `/v1/models`, `/health`, `/metrics`,
`/v1/system*`, admin load and unload, admission control from the planner's KV
accounting, cancellation, timeouts, request ids and structured errors with no paths.
Exit: an OpenAI SDK client works, concurrent conversations get independent KV, and a
cancellation test passes. RUST_GUIDELINES §4: Tokio at the edge only, bounded channels,
no lock across `.await`, CPU work never on Tokio workers.

## Decision
**A `qubivex-api` crate and a `qubivex serve` subcommand** (no separate `apps/server`;
the design allows either).

- **Threads.** Every generation runs on its own dedicated thread (`worker.rs`): chat
  template, tokenization, KV reservation, the decode loop and detokenization. It talks
  to the request over a bounded channel (64 events). Tokio runs only HTTP and the
  forwarding of events to SSE. Model loading uses `spawn_blocking`.
- **One forward pass at a time per engine, first come first served** (a ticket lock in
  `Engine::forward`). Each pass already uses every compute thread, so running two at
  once gains no throughput, and two passes would compete for expert-cache slots: with
  a one-slot pool (a tight plan), the second fails with "pool is full" instead of
  waiting (the gateway's admission test hit exactly that). Conversations interleave a
  pass at a time; `Engine::generate_with` splits a prompt into chunks (default 256
  tokens), so a long prompt holds the turn for one chunk. Chunking never changes bits
  (invariant 1). Batching passes across conversations is Phase 10.
- **Independent KV.** Each conversation owns its `KvCache` for the duration of its
  request. Concurrent results equal sequential ones (tested over HTTP and with the
  OpenAI SDK on two real models).
- **Admission from the planner's KV accounting.** `Engine::reserve_kv(prompt +
  max_tokens)` before the first pass: all live reservations together must leave the
  plan one expert slot per layer (`MemoryPlan::expert_cache_for`). A refusal is a 503
  `overloaded`; the reservation returns its room when the conversation ends. The
  engine's room sharing and its KV ledger charge use the reservations' total when
  several are live. A per-model semaphore (`--max-concurrent`, default 4) bounds
  generation threads; a request that waits longer than `--queue-timeout` gets a 429.
- **Cancellation.** A shared flag the engine checks before every prefill chunk and every
  token. The request side holds a guard that sets it when dropped: the handler future
  for a non-streaming request (Axum drops it when the client disconnects), the SSE
  forwarding task for a stream (its send fails once the client's body is gone).
- **Timeouts.** `--request-timeout` ends a generation as if it reached `max_tokens`
  (`finish_reason: "length"`; counted in `/metrics`).
- **Errors.** OpenAI-shaped `{"error": {message, type, param, code}}`. Admission and
  prompt errors arrive before the first byte, so they are HTTP statuses even for a
  stream; a failure mid-stream is an error event. Runtime errors that can contain paths
  or internals are logged with the request id; the client gets "see the server log under
  request id …". Memory and storage refusals (sizes and counts only) are passed on.
- **Request ids.** `x-request-id` from the client (≤ 64 of `[A-Za-z0-9_-]`) or
  generated, on every response and in completion ids.
- **Auth.** Optional `--api-key` on `/v1/*`; admin endpoints only with `--admin-token`
  (otherwise 403). Constant-time comparison.
- **What is refused rather than ignored:** `n > 1`, `best_of > 1`, logprobs, tools,
  non-text response formats or content parts, non-zero penalties, `logit_bias`, `echo`,
  `suffix`. Defaults follow OpenAI except `max_tokens`, which defaults to
  `--default-max-tokens` (512) so a reservation is not the whole context.
- **Metrics.** Hand-written Prometheus text from atomics (requests by endpoint and
  status, active, tokens, cancelled, timed out, failed; per model: reserved KV, expert
  cache, free slots). No `metrics` crate: the counters are few and fixed.
- **Dependencies:** `axum` 0.8 (`http1`, `json`, `tokio`, `query`), `tokio`,
  `tokio-stream`; licences checked by `cargo deny`.

## Consequences
- The exit criteria are met: `crates/qubivex-api/tests/gateway.rs` (over HTTP, tiny
  fixtures: shapes, streaming equals non-streaming, concurrent equals sequential,
  cancellation of streamed and non-streamed requests frees the slot, KV admission
  refuses the fifth of four-that-fit, auth, admin, errors without paths), and
  `crates/qubivex-api/sdk/openai_client.py` with the official SDK against
  OLMoE-1B-7B (GGUF) and Qwen3-0.6B (SafeTensors) served together.
- Cancellation lands within one prefill chunk or one token, not between layers. A
  between-layer check needs the forward pass to take a flag; it can come with Phase 10's
  scheduler if chunk-level latency proves too coarse.
- The stream decoder re-decodes the whole completion per token (quadratic in its
  length; fine at the lengths served now).
