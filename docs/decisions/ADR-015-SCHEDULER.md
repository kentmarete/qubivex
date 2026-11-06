# ADR-015: The request scheduler and the concurrency model

- Status: Proposed
- Date: 2026-11-07

## Context
Spec §7 and job 5 of `PLATFORM_AUDIT` ask for concurrency that is correct before any
batching starts: a bounded queue, priorities, deadlines, load dedup, safe unloading and a
clean shutdown, proven by the §12.3 tests. ADR-013 gave each model a semaphore with a
timeout. Waiters were limited by time and not by count, a burst parked without bound, and
concurrent loads of one model both read the weights.

The owner's rule is that the runtime stays usable with no server, so the scheduler cannot
live in `qubivex-api`, and cannot depend on Tokio.

## Decision
**A new crate, `qubivex-scheduler`, with no dependencies** and no knowledge of models,
HTTP or Tokio. A request takes a place with `enter`, says what it costs with
`Ticket::ready`, and hears the outcome through a callback. Dropping the `Ticket` leaves the
queue, and dropping the `Grant` frees the slot and the memory. It has no timers and no
threads: a host enforces its own limits by dropping the ticket.

- **Bounded.** Running plus waiting never exceeds `max_running + max_queue`. The place is
  taken before tokenizing, so tokenizing is bounded too, and a burst past the limit costs
  nothing. Past it, the answer is an immediate 429.
- **Priorities.** `low`, `normal`, `high`, as the `priority` request field. A waiter gains
  a level per `priority_aging` (10 s), so a low priority is delayed, not starved.
- **Fairness.** Within a level, the client with the fewest running requests goes first,
  then the oldest. The client is the OpenAI `user` field, and requests with none count as
  one client.
- **Memory admission.** The `Admission` trait, with `KvAdmission` in the runtime:
  reservations are taken in the scheduler against the plan. `--memory-policy queue` (the
  default) waits; `refuse` answers 503, ADR-013's behaviour. A waiter short of memory stays
  at the head and nothing overtakes it, because letting small requests pass would starve a
  large one for as long as they kept arriving. This gives up some throughput to do that.
- **Deadlines.** The `timeout` request field, in seconds, covers the queue and the
  generation. Passing it in the queue is a 504. Passing it while running ends the
  generation as `--request-timeout` does. It is now checked before every prefill chunk as
  well as every token, so a long prompt cannot run past it.
- **The runtime splits a request in two.** `Engine::prepare` tokenizes and checks the
  context, and `Engine::run_prepared` runs under a reservation. `run_request` is both.
  `KvReservation` no longer borrows the engine, so a scheduler can hold it.
- **Model lifecycle.** The gateway keeps each model as loading, loaded or failed, shown in
  `/v1/system`. A load belongs to the server, not the request that started it, so a client
  that goes away does not fail the others. Concurrent loads join one load, and loading a
  loaded model answers 200. A generation that names a model still loading waits for it.
  One directory has one instance: a second id for it is a 409. `POST /v1/models/load` and
  `/unload` are added, and the admin routes stay as aliases (audit §5, decision 4).
- **Unloading** closes the model's queue, so waiters get a 503 at once. Running requests
  hold an `Arc` of the engine and finish. The weights go when the last ends, and
  `/v1/system` shows `models_draining` until then.
- **Shutdown.** The listener closes, and waiters get a 503 at once. Running requests get
  `--shutdown-grace` (30 s) to finish. After that a stream gets an error event and
  `[DONE]`, and a whole reply gets a 503. `serve` returns once the generation threads have
  ended, and a connection that will not close cannot hold it up past 5 s.
- **Stalled clients.** A client that holds its connection open and stops reading used to
  block the worker in its channel send, where neither the cancel flag nor the deadline is
  looked at, so it held a slot and its KV for as long as it liked. A stream now ends after
  `--stall-timeout` (30 s) with nothing read. The audit had not found this.

## The concurrency model
| What | Owner | Lifetime and protection |
|---|---|---|
| Weights, tokenizer, memory plan | The `Engine`, shared by `Arc` | Read-only after load |
| Expert slots | The store | Leases keep an expert resident while a pass reads it (ADR-006); checked with loom |
| KV cache | One request's thread | Never shared; dropped when the request ends |
| KV reservation | The request's `Grant` | RAII: a counter against the plan |
| Generation slot | The request's `Grant` | RAII: back before the client hears the end, so its next request finds it free |
| Queue place | The `Ticket` | RAII: leaves on timeout, disconnect or shutdown |
| Forward passes | The engine's ticket lock | One at a time, first come first served (audit decision 3) |
| A loaded model | `Arc<Loaded>` in the gateway's table | The table's lock is never held across an `.await` |

Priorities decide who starts, and do not reorder forward passes: a running high-priority
request takes turns equally with everything else running. The turn lock stays until real
batches replace it.

A request's path: validate, take a place, tokenize on a blocking thread, wait for a grant,
run on its own thread, stream over bounded channels, free. Each step has its failure: 429
at a full queue or queue timeout, 400 for what can never run, 503 for memory or closing,
504 for its own deadline.

## Not built
- **A `high` priority is open to any client with the API key.** Per-key limits belong to
  job 9, as does per-client rate limiting: `user` is what the client says it is, so
  fairness holds among clients that are not lying.
- **Each model has its own scheduler**, its own limits and its own compute threads, and
  each engine plans RAM alone when it loads. Two models running at once oversubscribe the
  cores and can together pass what either planned for; the pressure monitor is what
  responds to that. Nothing here measures or arbitrates it, so §12.3 test 13 proves
  correctness across models, not shared capacity.
- **Cancellation** still lands between prefill chunks and tokens, not inside a pass, so a
  disk read or a forward pass finishes first. A queued conversation can wait out one full
  round of turns before it notices.
- **Metrics** gain queue depth and rejections only. The rest of §8 is job 10.

## Verification
- `crates/qubivex-api/tests/concurrency.rs` covers §12.3 tests 1, 2, 6, 9, 11, 12 and 13
  over HTTP, and each checks that slots, the queue and KV reservations come back. Tests 3,
  4, 5, 7, 8 and 10 were covered and still pass; test 10 gains the queue policy.
- `qubivex-scheduler` has unit tests, including a panic and a stress test, and loom models
  of a finish racing a queue, memory given back racing a waiter, and a ticket dropped while
  granted. `job verify` and CI run them.
- ThreadSanitizer, with its own run on Rust nightly:

  ```zsh
  CARGO_TARGET_DIR=target/tsan RUSTFLAGS="-Zsanitizer=thread" \
    cargo +nightly test -Zbuild-std --target aarch64-apple-darwin \
    -p qubivex-scheduler -p qubivex-api -p qubivex-expert-stream -p qubivex-runtime \
    --test scheduler --test concurrency --test gateway --test store --test prefetch --test requests
  ```

  64 tests, no reports. A deliberately racy program built the same way was reported, so the
  sanitizer was active. It sees Rust memory only, so it does not cover Metal.

  CI runs the same tests on Linux, also with no reports, under one suppression in
  `.github/tsan-suppressions.txt`. Without it, the gateway's tests gave 14 reports, all
  against the allocation of tokio's `ScheduledIo`, none touching QUBIVEX memory. Tokio's
  driver reaches that object through a pointer kept by epoll, an ordering TSan does not
  see. That explanation comes from reading the reports. It is not proven, and the
  suppression would also hide a real race inside that tokio module.
- Miri is clean on the scheduler's tests. It cannot run the engine's SIMD or file I/O.

## Consequences
- `POST` of a loaded model is now 200, not 409.
- `--memory-policy queue` is the default, so a request that does not fit waits rather than
  getting a 503 at once. This changes ADR-013's behaviour, and `refuse` restores it.
- `--max-queue` defaults to 32, so a burst that used to park is now refused with a 429.
- `KvReservation` loses its lifetime parameter, a small change to the runtime's API.
- A deadline that has already passed now ends a request before its prefill, with no tokens,
  where it used to produce one.
