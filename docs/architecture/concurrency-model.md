# Concurrency model

Which threads exist, what each owns, what is shared, and how the shared things are
protected. The serving view of the same subject (queues, priorities, batching) is in
[Scheduler](../serving/scheduler.md) and
[Concurrent inference](../serving/concurrent-inference.md). The design records are
[ADR-006](../decisions/ADR-006-EXPERT-STREAMING.md),
[ADR-015](../decisions/ADR-015-SCHEDULER.md) and
[ADR-017](../decisions/ADR-017-BATCHING.md).

## The principle

**Async at the edge, threads underneath.** Tokio runs the HTTP gateway and nothing
else. Inference, expert I/O and memory monitoring run on ordinary threads with ordinary
locks. A forward pass takes tens of milliseconds of CPU; running it on an async executor
would stall every connection that executor serves.

The runtime therefore has no async code and no Tokio dependency, and an application can
embed it with plain threads.

## Threads

In a `qubivex serve` process with one model:

| Threads | Count | Do | Owned by |
|---|---|---|---|
| `qubivex-http` | 2 | Accept connections, parse requests, write responses and SSE events | The gateway's Tokio runtime |
| Tokio blocking pool | As needed | Tokenize prompts, load models, read `/v1/system` hardware | Tokio |
| Generation threads | One per running request, at most `--max-concurrent` | Run one request's loop: prefill, sample, emit | The gateway, per request |
| Compute pool (rayon) | `--threads`: the performance cores | Split each kernel's work across output rows | The engine |
| `qubivex-io-*` | `--io-threads`: 4 | Read experts from disk into cache slots | The expert store |
| `qubivex-pressure` | 1 | Sample available RAM every 500 ms | The engine |

Each loaded model has its own generation threads, compute pool, I/O threads and
pressure thread. Two models busy at once therefore ask for more compute threads than
there are cores; nothing arbitrates that.

`qubivex run` and an embedding application have everything from the generation threads
down, and no Tokio.

## Crossing from async to threads

A request crosses the boundary twice.

**In:** the gateway takes a queue place (a synchronous call that never blocks), sends
tokenizing to the blocking pool, waits for the scheduler's grant, then starts a
generation thread.

**Out:** the generation thread sends events through a **bounded channel**. A task on
the Tokio side forwards them to the connection. If the client reads slowly the channel
fills and the generation thread waits, which slows that request only. If the client
reads nothing for `--stall-timeout`, the task ends and the request is cancelled, so a
stalled client cannot hold a slot.

No lock is held across an `.await`. The gateway's model table is a short-held lock
around a map lookup.

## What is shared, and how

| Thing | Shared between | Protection |
|---|---|---|
| Weights, tokenizer, memory plan | Every thread of a model | Immutable after load, behind `Arc`. No lock |
| The expert cache's table | Generation threads, I/O threads | One mutex, held only to look up, lease, queue or evict. Never held while an expert is read or used |
| An expert's bytes | A pass that leased it, and the cache | `Arc`. A lease holds a reference; eviction skips leased slots |
| A loading expert | Every request waiting for it | One in-flight load, with waiters on a condition variable |
| A KV cache | **Nobody.** One request owns it | Ownership. It moves into a pass and back; it is never reachable from two threads |
| The pending steps of a pass | Generation threads of one model | One mutex and two condition variables (the combining point) |
| The prefix index | Requests starting and finishing | One mutex around the index. Retained KV is immutable, held by `Arc`, and copied out |
| Scheduler state | The gateway's tasks | One mutex inside the scheduler |
| The ledger, counters, metrics | Everything | Atomics |
| The cancel flag of a request | The gateway task and the generation thread | An atomic boolean |

Rules that follow:

- **No lock is held across I/O or a forward pass.**
- **Shared mutable state is small.** Nearly everything is immutable and shared, or
  mutable and owned by one thread.

## The combining point

Running conversations meet at the forward pass. Each generation thread posts its next
step and its KV cache, and waits. One thread, the leader, takes every posted step and
runs them as one pass, then hands each thread its logits and its cache back.

- The leader **owns** every cache it computes on, by move. No cache is reachable from
  two threads at once, so this needs no `unsafe` and no per-cache lock.
- A cache counts as advanced only when the whole pass succeeded. A failure leaves every
  cache as it was.
- A leader waits at most 1 ms for the callers of the previous pass to post again, so
  conversations stay in step. A lone caller never waits.

## Ownership of resources

Every resource a request holds is a value that releases it when dropped. There is no
manual cleanup to forget, and an early return, an error, a timeout, a disconnect and a
panic all take the same path.

| Resource | Held by | Released when |
|---|---|---|
| A place in the queue | `Ticket` | It is dropped: on grant, timeout, disconnect or shutdown |
| A generation slot and a KV reservation | `Grant` | It is dropped, before the client hears the end |
| An expert in the cache | `Lease` | It is dropped, at the end of the layer |
| Ledger bytes | `Charge` | It is dropped |
| A loaded model | `Arc<Loaded>` | The last running request drops it, after an unload |
| A registry lock | A file lock (`flock`) | The process releases it, or dies |

## Cancellation

A request is cancelled by setting its cancel flag: when the client disconnects, when its
deadline passes, when its stream stalls, or at the end of the shutdown grace.

The generation thread checks the flag **between** units of work: before each prefill
chunk (256 tokens) and before each token. It is not checked inside a forward pass or a
disk read, so a cancel takes effect within one chunk or one token.

A cancelled request releases everything through the drops above, and with a prefix
cache its completed KV is kept.

## Panics

- A panic in a generation thread is caught at the request boundary. The request fails
  with a 500, the log gets `generation_panicked`, and the server continues. The release
  profile keeps `panic = "unwind"` for this reason.
- A panic in an expert reader is caught in the load and becomes a failed read for its
  waiters.
- Locks are taken in a way that survives poisoning, so one panic does not disable a
  shared structure.

## Shutdown

1. SIGTERM or SIGINT arrives. The listener closes.
2. Every model's queue closes: waiters get a 503.
3. Running requests continue for `--shutdown-grace`.
4. The stop flag is set. Remaining requests are cancelled at their next token.
5. Generation threads end, then the stores join their I/O threads, and the process
   exits. A connection that will not close cannot delay this past 5 s.

## How it is verified

| Tool | Covers |
|---|---|
| Rust's ownership and `Send`/`Sync` | Data races in safe code are compile errors. `unsafe` exists only in the two backend crates |
| loom | The expert store's lease, load and eviction interleavings; the scheduler's races between finishing, queueing, returning memory and dropping a ticket |
| ThreadSanitizer (CI, Linux; also run on macOS) | The scheduler, gateway, store, request and prefix tests, with one documented suppression inside Tokio |
| Miri | The scheduler's tests |
| Concurrency tests over HTTP | `crates/qubivex-api/tests/concurrency.rs`: bursts, cancellation, unloading while active, shutdown with active requests, different context lengths, several models. Each checks that slots, queue places and reservations came back |
| Fault injection | Failed, short, corrupt, slow and panicking reads, with waiters, through the whole engine |
| Bit-identity tests | A conversation's tokens are the same alone, in a batch, and under concurrent callers, on the CPU and on Metal |

## Not supported

- **Work stealing between models,** or a compute pool shared by models.
- **Cancellation inside a forward pass or a read.**
- **Preempting a running generation.**
- **An async runtime API.** An async application calls the engine from its blocking
  pool, as the gateway does.
- **Multi-process serving of one model with shared weights.** Each process loads its
  own copy; the OS page cache is the only sharing.
- **NUMA-aware thread placement** and thread pinning.
- **Sanitizer coverage of Metal.** ThreadSanitizer sees Rust memory only.
