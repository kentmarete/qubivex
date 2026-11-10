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
  cache, free slots). No `metrics` crate: the counters are few and fixed. (Amendment 5
  adds histograms and the engine's counters, still hand-written.)
- **Dependencies:** `axum` 0.8 (`http1`, `json`, `tokio`, `query`; Kerniq Server since
  Amendment 4), `tokio`,
  `tokio-stream`; licences checked by `cargo deny`.

## Amendment 1 (2026-11-06): the generation contract moves to the runtime
The worker's chat template, tokenization, context checks, KV reservation, stop sequences and
detokenization are now `Engine::run_request` (`qubivex-runtime/src/request.rs`), with
`GenerationRequest`, `GenerationEvent`, `GenerationOutcome` and `RequestError`. `qubivex run`
uses it too, so a host embedding the runtime gets the same behaviour with no server, CLI or
Tokio. The worker keeps only channels and metrics, and threads, admission and cancellation
are unchanged. `cargo xtask check` now fails if the runtime gains `tokio`, `axum`, `hyper`,
`tower`, `clap`, `qubivex-api` or `qubivex-cli`, directly or through another crate.

- `Finish` gains `StopSequence` and `TimedOut`. `Sampling` gains `top_k` (0: off), applied
  before `top_p` as in `transformers`, and exposed as the `top_k` request field and `--top-k`.
- `qubivex run` now reserves KV like the gateway, and refuses a prompt that fills the context
  (it used to generate one token).
- A KV reservation now ends just before the final event, not just after it.

## Amendment 2 (2026-11-07): the scheduler replaces the semaphore
ADR-015 replaces the per-model semaphore with `qubivex-scheduler`: a queue bounded by count,
priorities, per-request deadlines and memory admission that can wait. Loads are deduplicated,
a loaded model reloads as a 200 rather than a 409, shutdown drains, and a stream nobody reads
ends. The threads, the turn lock and the cancellation flag described above are unchanged.

## Amendment 3 (2026-11-10): hardening (platform plan job 9)
- **Listening beyond loopback.** `Gateway::serve` and `qubivex serve` refuse a listener
  that is not loopback when no API key is set, unless `allow_unauthenticated_remote`
  (`--allow-unauthenticated`) says otherwise. Before, the CLI only warned.
- **Rate limit.** Optional token bucket per peer address on `/v1/*` (`--rate-limit`,
  `--rate-limit-burst`): 429 `rate_limit_exceeded` with `retry-after`. IPv6 counts by
  /64. The table holds at most 4096 clients. A single API key cannot tell clients apart,
  so the address is the client; behind a proxy, limit at the proxy.
- **Request limits.** Body bytes (`--max-body-bytes`, 4 MiB, 413), prompt tokens after
  the template (`--max-prompt-tokens`, 400 `prompt_too_long`) and `max_tokens`
  (`--max-generation-tokens`, 400). The prompt check runs after tokenization, before
  admission.
- **CORS.** Hand-written middleware rather than `tower-http`: configured origins or
  `*`, preflights answered with a 204 or refused with a 403, `x-request-id` and
  `retry-after` exposed, never credentials.
- **Config file.** `qubivex serve --config` (or `$QUBIVEX_CONFIG`) reads a TOML `[serve]`
  table; flag > environment > file > default. Unknown keys are errors, and a file
  holding a key or token must not be readable by others. `toml` joins the CLI's
  dependencies.

## Amendment 4 (2026-11-10): Kerniq Server over Axum
The owner chose Kerniq Server (`kerniq-server`, a thin layer over Axum, Tower and Hyper)
in place of bare Axum. It is a thin swap: the handlers, the OpenAI-shaped errors, the
body limit, the rate limit and CORS (`limits.rs`) are unchanged and run as Tower layers on
a Kerniq `App`, in the same order. What changes:

- `Gateway::serve` runs `App::serve` with our shutdown signal, so the accept loop is
  Kerniq's (hyper-util, HTTP/1 and HTTP/2 auto-detected). It adds a 30 s header read
  timeout and closes connections still open after `shutdown_grace` plus the stop margin;
  the grace and stop logic around it are as before.
- A wrong method on a known path answers 405 with Kerniq's JSON error body instead of an
  empty one.
- `Gateway::router` still returns an `axum::Router` (`App::into_axum_router`).
- Kerniq's request timeout, concurrency limit and body limit stay off: streams run for
  minutes, the schedulers queue, and our body limit answers in the OpenAI shape.
- The crate is pinned `=0.1.3`: its first release was 2026-11-07, from an organisation
  with no named maintainer. Its `axum` re-export is the gateway's Axum. It needs Rust
  1.88, so the MSRV moves from 1.85 to 1.88. `cargo xtask check` also keeps
  `kerniq-server` out of the runtime.

## Amendment 5 (2026-11-10): observability (platform plan job 10)
Spec §8 asks for inference, memory and serving metrics in a Prometheus-compatible form,
and structured logs with request, model and sequence ids and no private text.

- **Still hand-written.** The owner chose (2026-11-10) to extend the gateway's own
  metrics over Kerniq Observe (OpenTelemetry): no new dependency, the `qubivex_*` names
  stay, and the "no `metrics` crate" decision above stands although the counters are no
  longer few. The cost: no OTLP export and no traces.
- **Histograms** are fixed buckets of atomics (`metrics.rs`), per model: scheduler wait,
  time to first token and total latency (all from the request's arrival), the gap
  between tokens, prompt and generation tokens per second, and model load time; and per
  endpoint, the time to answer. A token costs one clock read and a few atomic adds.
- **Engine numbers are read at scrape**, not copied: expert store counters (hits, loads,
  evictions, bytes read, time blocked on reads, retries, timeouts), batch passes and
  steps, the memory ledger by owner and the plan's budget. Expert load latency is a sum
  to divide by loads, not a histogram: the store keeps only the total.
- **Process memory** comes from `qubivex-hardware`. On macOS that runs `ps` (off the
  Tokio workers), and the peak is the highest value a scrape has read.
- **Not measured:** peak VRAM (the ledger has no GPU tier while Metal shares host
  memory and CUDA is blocked) and fragmentation.
- **Logs** (`log.rs`) are one line per event on standard error, text by default or JSON
  (`--log-format text|json|off`, `ServerConfig::log_format`), with a level and a UTC
  timestamp. `request` is written for every API request, `/health` and `/metrics`
  excepted. A **sequence id** is new: the generation's number among those the server has
  admitted, given at the grant, so lines about sequences that share forward passes can
  be told apart. A host embedding the gateway can take the lines
  (`Gateway::with_log_writer`).
- **No prompt text, by construction and by test.** A log line is an event name plus
  fields of ids, counts, durations and codes; no logging call receives a prompt, a
  message or generated text. `tests/observability.rs` sends marked prompts through both
  endpoints and a failing request and searches the log for them. The one free-text field
  is `detail` on `internal_error` and `model_not_found`: the runtime's description of an
  internal failure, which may name a file and never reaches the client.
- `--no-metrics` (`ServerConfig::metrics_enabled`) turns `/metrics` into a 404.
  `/metrics` still needs no key, so it belongs behind the same network boundary as the
  server.

## Amendment 6 (2026-11-10): deployment (platform plan job 11)
- **SIGTERM drains, as SIGINT does.** `qubivex serve` stopped only on Ctrl-C. `docker stop`,
  systemd and Kubernetes send SIGTERM, which then ended the process mid-request, or, as
  process 1 in a container, was ignored until Docker killed it after 10 s. Both signals
  now start the same shutdown (ADR-015). `apps/cli/tests/serve_config.rs` sends SIGTERM to
  the real binary and expects status 0. In the image, `docker stop` during a stream gave
  the error event, `[DONE]` and status 0 at the grace.
- **A `Dockerfile`:** a Debian image with the release binary and the fixture models,
  running as a non-root user, with a health check on `/health`. Its default command
  listens on `0.0.0.0`, so it starts only with an API key or `--allow-unauthenticated`:
  amendment 3's refusal is what makes the image safe by default. CI builds it and runs
  the smoke test.
- **No TLS in the server.** The runbooks put a proxy in front. Kerniq Server is still
  built with the `json` feature only.
- **Not built:** published images or packages, a system-wide launchd daemon, and
  Kubernetes manifests. `docs/deployment/` says so.

## Amendment 7 (2026-11-10): per-client API keys

Accepted by the owner on 2026-11-11.
Amendment 3 left two gaps, both because one key serves every client: `user`, which
scopes the prefix cache and fairness, comes from the client unchecked, and the rate limit
counts addresses, so behind a proxy every client shares one allowance.
- **Named keys.** `ServerConfig::client_keys` lists `ClientKey { client, key_sha256,
  rate_limit }`. `client` is a name (`[A-Za-z0-9._-]`, 1 to 64 bytes, unique) that
  appears in the log; the key is stored only as its SHA-256, so a leaked config file
  gives no key. A bearer is hashed once and compared with every digest in constant time.
  Keys are random 256-bit values, so a plain hash needs no slow KDF. The single
  `api_key` stays as before, as a shared key with no client name; both may be set.
- **Who a request is.** A named key makes the request that client's:
  - Fairness counts the client, not `user`: one client sending many `user` values gets
    no larger share.
  - The prefix-cache scope is the client plus `user`, so clients never share retained KV
    and a client can still keep its own users apart. Unnamed requests (shared key or no
    key) keep `user` as their scope, now in a separate namespace, so no `user` value can
    name a client's scope.
  - The `generation_started` log line gains `client` (the name, never the key or its
    hash).
- **Rate limit by key.** The limiter counts a named key's requests under that key, with
  the key's own `rate_limit` if it has one, else the server's. Other requests still count
  by address. An invalid bearer counts by address, so made-up keys cannot buy
  allowances. The limiter runs before the handlers' 401, as before.
- **Configuration.** Keys live only in the config file (`[[serve.client_keys]]`, the file
  must be `chmod 600`), not on the command line or in the environment, where they would
  show in `ps` or the process environment. `qubivex keys new <client>` prints a fresh key
  once and the table to paste; `qubivex keys hash` hashes a key read from standard input.
  A named key counts as a key for the loopback check.
- **Not built:** reloading keys without a restart, per-client model access, per-client
  quotas on tokens rather than requests, and per-client metrics labels (unbounded label
  sets are what amendment 5 avoided).

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
