# Scheduler

The scheduler decides which request starts next and whether it may start at all. Each
loaded model has its own. It is the crate `qubivex-scheduler`, which has no dependencies
and knows nothing of HTTP, so an application embedding the runtime can use it too. The
design record is [ADR-015](../decisions/ADR-015-SCHEDULER.md).

What happens to requests once they are running is in
[Concurrent inference](concurrent-inference.md).

## A request's path

```text
arrive ── validate ── take a place ── tokenize ── wait for a grant ── run ── free
           │              │                          │                 │
          400            429                    429 / 503 / 504     stream, or
                    (queue full)             (timeout, memory,      one response
                                              deadline)
```

1. **Validate.** A malformed request or an unsupported field is a 400.
2. **Take a place.** The model has `--max-concurrent` running slots (4) and
   `--max-queue` waiting places (32). If both are full the answer is an immediate 429.
   The place is taken before tokenizing, so a burst past the limit costs the server
   nothing.
3. **Tokenize.** The chat template is rendered and the prompt tokenized on a blocking
   thread. A prompt that can never run (longer than the context or the server's limit)
   is a 400.
4. **Wait for a grant.** The request waits until a slot is free and the memory plan can
   hold its KV. See [Admission](#admission).
5. **Run** on its own thread, streaming over a bounded channel.
6. **Free.** The slot and the KV reservation return before the client hears the end, so
   its next request finds them free.

## Limits

| Setting | Default | What it bounds |
|---|---|---|
| `--max-concurrent` | 4 | Generations running at once, per model |
| `--max-queue` | 32 | Requests waiting, per model. Running plus waiting never exceeds the two together |
| `--queue-timeout` | 30 s | How long a request waits for a grant. Then a 429 |
| `--request-timeout` | 600 s | How long a generation may run. Then it ends with `finish_reason: "length"` |
| The request's `timeout` | none | The queue and the generation together, for that request |

The scheduler itself has no timers and no threads. The gateway enforces the time limits
by giving up its place.

## Order

Among waiting requests, the next to start is chosen by, in order:

1. **Priority.** The request field `priority`: `high`, `normal` (the default) or `low`.
2. **Fairness.** Within a priority, the client with the fewest running requests goes
   first. The client is the request's `user` field; requests with no `user` count as
   one client.
3. **Age.** Then the oldest.

A waiting request gains one priority level for every 10 s it waits, so a low priority
is delayed and never starved.

Priority decides who **starts**. It does not make a running request faster: every
running conversation takes part in every forward pass it has a step for.

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{"model":"qwen3-0.6b","user":"batch-job-7","priority":"low","timeout":120,
       "messages":[{"role":"user","content":"Summarise this."}]}'
```

## Admission

A request is granted only when the model's memory plan can hold the KV of every running
conversation, including this one, and still keep one expert slot per layer. What a
request costs is known before it starts: the most KV positions it can use, which is its
prompt plus its `max_tokens`, capped by the context.

What happens when the memory is not free is `--memory-policy`:

| Policy | Behaviour |
|---|---|
| `queue` (default) | The request waits for memory, up to `--queue-timeout` |
| `refuse` | The request gets a 503 `overloaded` at once |

A request that waits for memory stays at the head of the queue and nothing overtakes
it. Letting small requests pass would starve a large one for as long as small ones kept
arriving. This gives up some throughput for that guarantee.

A request always fits on an idle server: the context is sized when the model loads so
that one full conversation fits the plan, and a prompt too long for the context is a 400
before it reaches the queue.

When memory pressure reaches its critical level (under 256 MiB available on the
machine), new requests are refused with a 503 while running ones finish.

## Deadlines

The `timeout` field is in seconds and covers the queue and the generation.

- It passes **while waiting**: 504 `deadline_exceeded`.
- It passes **while running**: the generation ends, and the response is a normal one
  with `finish_reason: "length"` and whatever was generated.

It is checked before every prefill chunk and every token, so a long prompt cannot run
past it by more than one chunk.

## Model lifecycle

The gateway keeps each model as `loading`, `loaded` or `failed`, shown in `/v1/system`.

- A load belongs to the server, not to the request that started it. A client that
  disconnects does not fail the load for others.
- Concurrent loads of one model join one load.
- A generation that names a model still loading waits for it.
- Unloading closes the model's queue: waiters get a 503 at once. Running requests
  finish, and the weights go when the last one ends.

## Shutdown

The listener closes and waiters get a 503 at once. Running requests get
`--shutdown-grace` (30 s). After that a stream gets an error event and `[DONE]`, and a
non-streaming request gets a 503. The process exits once the generation threads have
ended.

## Observing it

| Metric | Meaning |
|---|---|
| `qubivex_queue_depth` | Requests waiting now |
| `qubivex_generation_slots_free` | Slots free now |
| `qubivex_queue_granted_total` | Requests that started |
| `qubivex_queue_rejected_total` | Requests refused because the queue was full |
| `qubivex_queue_wait_seconds` | From arrival to grant |
| `qubivex_generations_timed_out_total` | Generations cut short by a time limit |

`/v1/system` shows the same per model, with the KV positions reserved.

## Guarantees and how they are checked

- Every request reaches a terminal state: a response, an error, or a cancelled stream.
- A slot, a queue place and a KV reservation are each held by a value that releases
  them when dropped, so a timeout, a disconnect, a panic or a shutdown cannot leak one.
- `crates/qubivex-api/tests/concurrency.rs` drives the spec's concurrency cases over
  HTTP and checks after each that slots, the queue and reservations came back.
- The scheduler has loom models of its races (a finish against a queue, memory returned
  against a waiter, a ticket dropped while granted), and CI runs the scheduler, gateway
  and store tests under ThreadSanitizer.

## Not supported

- **Per-client limits and quotas.** One API key serves everyone, `user` is whatever the
  client sends, and `priority: "high"` is open to any client with the key.
- **Arbitration between models.** Each model has its own scheduler, limits and compute
  threads. Two busy models oversubscribe the cores and can together exceed what either
  planned for.
- **Preemption.** A running generation is never paused for a higher priority.
- **Cancellation inside a forward pass.** It lands between tokens or prefill chunks.
