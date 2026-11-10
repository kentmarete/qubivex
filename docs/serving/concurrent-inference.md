# Concurrent inference

How one loaded model serves several conversations at once: what they share, what each
owns, and what that costs and gains. The scheduler that decides who starts is in
[Scheduler](scheduler.md). The design records are
[ADR-015](../decisions/ADR-015-SCHEDULER.md) (concurrency) and
[ADR-017](../decisions/ADR-017-BATCHING.md) (batching).

## What is shared and what is not

| Thing | Shared? | How it is protected |
|---|---|---|
| Weights, tokenizer, memory plan | Shared by every conversation | Read-only after load |
| Expert cache | Shared | A lease keeps an expert in memory while a pass reads it. Two conversations that need the same expert share one load |
| KV cache | **One per conversation** | Owned by that request's thread and never reachable from two threads. Dropped when the request ends |
| Sampling state (seed, temperature, stop strings) | One per conversation | Lives in the request's own loop |
| Retained KV (prefix cache) | Read-only entries, copied on use | Never shared with a running conversation. Separated by `user` |

One conversation cannot read or change another's attention state, and the model's
weights are loaded once however many conversations run.

## Shared forward passes

Each request runs its own loop on its own thread: prefill the prompt, then sample a
token, feed it back, and repeat. The loop's forward passes are where conversations meet.

When a request needs a pass, it posts its step (its new tokens and its KV cache) and
waits. One thread takes the turn, collects every step that is waiting, and runs them as
**one forward pass**: every matmul runs once over all the tokens, each token attends
only to its own conversation's KV, and each request gets back exactly the logits it
would have got alone.

```text
conversation A  ──step──┐
conversation B  ──step──┼──►  one pass over 3 tokens  ──►  A's logits, B's logits, C's logits
conversation C  ──step──┘
```

Consequences:

- **The batch follows the load.** Nothing waits for a batch to fill. One conversation
  alone runs passes of one. Under load, steps collect while the current pass runs, and
  the next pass carries all of them.
- **Joining and leaving are free.** A conversation joins at its next step and leaves by
  not posting another. Stop strings, deadlines and cancellation stay per conversation.
- **Decode is not starved by prefill.** A pass takes every waiting decode step first,
  then prefill chunks within a budget of 256 tokens. A long prompt delays the others'
  next token by at most one chunk.
- **A short wait keeps conversations together.** After a pass, its callers are sampling
  their next tokens. The next pass waits up to 1 ms for as many steps as the last one
  answered, so the conversations stay in step. A lone conversation never waits.

This happens on the CPU backend and on Metal, in the server and in an application that
calls the runtime from several threads.

## The output does not change

A conversation's tokens are bit-identical whether it runs alone or in a pass with
others, at any batch width, on the CPU and on Metal. The kernels compute each output
from its own weight row and activation row only, so neither the batch size nor the
thread count can change a bit. Tests check this on the fixture models and on
OLMoE-1B-7B (`crates/qubivex-runtime/tests/batch.rs`).

So the same request with the same `seed` gives the same text on a busy server as on an
idle one.

## What it gains

Measured on an Apple M3 Pro on power, OLMoE-1B-7B Q4_K_M, 32 tokens per conversation,
tokens per second summed over the conversations. "In turn" is the same conversations
with passes taken one at a time.

| Backend | Conversations | In turn | Shared passes |
|---|---:|---:|---:|
| CPU, all experts in RAM | 1 | 41 | 42 |
| | 2 | 41 | 49 |
| | 4 | 41 | 54 |
| | 8 | 41 | 62 (1.50×) |
| CPU, 1024 MiB expert cache | 8 | 29 | 49 (1.73×) |
| Metal (`layers`) | 1 | 54 | 54 |
| | 2 | 54 | 66 (1.22×) |
| | 4 | 54 | 81 (1.51×) |
| | 8 | 54 | 102 (1.89×) |

Total throughput rises with concurrency, and each conversation's own speed falls: at
eight on the CPU each gets about 7.7 tokens per second where one alone gets 42. Choose
`--max-concurrent` for the balance you want. The full tables are in
[IMPLEMENTATION_STATUS.md](../../IMPLEMENTATION_STATUS.md).

With experts streaming from disk, shared passes help more than they do in RAM: reads
for one conversation overlap with compute for the others, and conversations taking
turns waited on reads 5% to 19% of the time against 0% to 5% in shared passes.

## Failure isolation

- A step that is wrong on its own (no room in its context, a bad token id) fails that
  request only. It is checked before the pass.
- An error inside a pass, such as an expert that cannot be read, belongs to the model,
  so every request in that pass gets it. No KV cache advances in a failed pass, so each
  conversation is left as it was and the same steps can run again.
- A panic in one generation is caught at the request boundary. That request fails with
  a 500 and the server keeps running.
- A client that stops reading is dropped after `--stall-timeout`, so it cannot hold a
  slot and its KV.

## Memory

Batching adds no KV: each conversation's cache was reserved by the scheduler before it
started. Activations grow with the tokens in a pass, which the 256-token budget bounds.
The expert cache shrinks as the conversations' KV grows, and regrows as they finish
(see [Memory planner](../runtime/memory-planner.md)).

## Prefix reuse

With `--prefix-cache-mb`, a finished conversation's KV is retained, and a new prompt
that starts with the same tokens copies it instead of running them again. Chat clients
resend the whole conversation each turn, so this removes most of the prefill from every
turn after the first. See [KV cache](../runtime/kv-cache.md).

## Several models

Each loaded model is independent: its own weights, plan, scheduler, queue and compute
threads. Requests to different models never share a pass.

## Embedding

None of this needs the server. An application that shares one `Arc<Engine>` and calls
`run_request` from several threads gets the same shared passes. `Engine::forward_batch`
is public for a caller that wants to build the batches itself.
`crates/qubivex-runtime/examples/batch.rs` does both and is what produced the table
above.

## Not supported

- **Capacity shared between models.** Two models busy at once oversubscribe the cores,
  and nothing arbitrates.
- **Batching across models,** or across processes.
- **More than one GPU, or splitting a model across devices.**
- **Speculative or beam decoding.** One sampled continuation per request.
- **A serve flag for the pass budget and the wait.** They are engine options
  (`EngineOptions::batch_tokens`, `batch_wait_us`) with fixed defaults in the server.
