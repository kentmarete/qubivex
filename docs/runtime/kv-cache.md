# KV cache

Attention needs, for every earlier token of a conversation, that token's keys and
values in every layer. The KV cache holds them so each token is computed once. This
page covers the cache itself and **prefix reuse**, which lets a new request start from
the KV an earlier one left behind. The design record is
[ADR-016](../decisions/ADR-016-KV-SUBSYSTEM.md).

## One cache per conversation

Every generation has its own KV cache, owned by its thread.

- It is never shared with another conversation, so one request cannot read or change
  another's attention state.
- It is `f32`. Its cost per token is fixed by the model: the output width of each
  layer's key and value projections × 4 bytes. `qubivex plan` shows it as
  "KV per token": 1.00 MiB for OLMoE-1B-7B.
- It grows in chunks of 256 positions, allocated as the conversation grows and capped
  at the context. Nothing is allocated for context that is not used.
- It is dropped when the request ends, however it ends.
- On Metal the KV lives in device buffers, which on Apple Silicon are host memory and
  count in the same budget.

Before a request starts, the scheduler reserves its KV against the memory plan
([Memory planner](memory-planner.md)). A request that would not fit waits or is
refused; it never starts and then runs out.

A forward pass that fails rolls the cache back to the positions it held before the
pass, so a failed step never leaves half-written keys and values behind.

## Prefix reuse

A chat client resends the whole conversation on every turn. Without reuse, each turn
runs the entire history again: on OLMoE-1B-7B a 700-token conversation cost about 6.6 s
of prefill per turn on the CPU, even when only the last 20 tokens were new.

With `--prefix-cache-mb N` on `qubivex serve` (or `EngineOptions::prefix_cache_bytes`
when embedding), finished KV is **retained**, and a new prompt that starts with the same
tokens copies the matching part into its own cache and runs only the rest.

Measured on OLMoE-1B-7B: a chat's second turn of 714 tokens took 0.57 s instead of
6.59 s, with the same text.

From the Qwen3-0.6B server, the same conversation one turn later:

```json
"usage": {"completion_tokens": 6,  "prompt_tokens": 53, "prompt_tokens_details": {"cached_tokens": 0},  "total_tokens": 59}
"usage": {"completion_tokens": 11, "prompt_tokens": 78, "prompt_tokens_details": {"cached_tokens": 53}, "total_tokens": 89}
```

The second request's first 53 prompt tokens were not run again.

### Why it is exact

Two facts make reuse safe. A position's keys and values depend only on the tokens up to
it. And a prompt's logits are the same however it is split into passes. So a prompt
that starts with tokens an earlier generation ran can start from a copy of that KV and
give the same bits as running everything. Tests compare the two bit for bit, on the CPU
and on Metal, across chunk boundaries and with sampling.

### The rules

- **Copy, not share.** Retained KV is immutable. A hit copies the prefix into the
  request's own cache. No running conversation shares KV memory with another or with
  the retained entries.
- **The longest match wins.** The index finds the retained entry sharing the longest
  prefix with the prompt.
- **The last prompt token always runs,** because the next token needs its logits.
- **Matches under 32 tokens are not used or kept.** An entry costs a whole 256-position
  chunk, and short prompts would fill the budget for little gain.
- **What is kept is what ran:** the prompt plus the output, however the generation
  ended: finished, stopped, cancelled or timed out. A cancelled prefill keeps the
  chunks that completed, so a retry continues from them. A failed pass keeps nothing.
- **One entry per conversation.** When a longer run of the same tokens arrives, it
  replaces the shorter entry, so a ten-turn chat holds one entry and not ten.

### Budget and eviction

`--prefix-cache-mb` is a budget per model, converted to KV positions.

- When full, the least recently used entry goes.
- A conversation larger than the whole budget is not kept, and evicts nothing.
- **Live requests come first.** Retained KV is priced in the memory plan, so the expert
  cache shrinks to hold it. A new request is never delayed or refused because of it:
  retained KV is evicted until the plan holds the request.
- **Memory pressure drops it first.** At the `Elevated` level all retained KV is
  cleared and nothing new is kept until pressure falls.
- It lives in process memory only. It is never written to disk, and it goes when the
  model unloads or the server stops.

Size it in whole conversations. An entry costs its positions, rounded up to
256-position chunks, times the model's KV per token, which `qubivex plan` shows: 1.00 MiB
for OLMoE-1B-7B and 224 KiB for Qwen3-0.6B. A 700-token OLMoE conversation is three
chunks, about 768 MiB; the same conversation on Qwen3-0.6B is about 168 MiB.

### Scopes: who can reuse what

A reused prefix makes the first token arrive sooner, and that timing tells the caller
that someone already sent that prefix. A cache shared by everyone would let one client
confirm a guess about another client's prompt.

So the cache is partitioned by **scope**, and an entry is only matched within its own.
The server uses the request's `user` field as the scope. Requests with no `user` share
one scope.

- Chat clients should send a stable `user` per end user, to get reuse and stay apart
  from others.
- `user` is whatever the client sends. A client that sends another's `user` reaches
  that scope. Among callers holding the same API key that is the current trust model.
  **If you cannot trust clients to send their own `user`, leave the prefix cache off.**

### Observing it

| Where | What |
|---|---|
| The response | `usage.prompt_tokens_details.cached_tokens`, also in a stream's final usage chunk |
| `/metrics` | `qubivex_prefix_cache_lookups_total`, `_hits_total`, `_tokens_reused_total`, `_evictions_total`, `qubivex_prefix_cache_positions` |
| `/v1/system` | A `prefix_cache` object per model: entries, positions, lookups, hits, tokens reused, evicted |
| The log | `cached_tokens` on `generation_finished` |
| Embedding | `GenerationOutcome::cached_tokens`, `Engine::prefix_stats`, `Engine::clear_prefix_cache` |

It is off by default (budget 0). With it off, generation is exactly the code path that
existed before it, so `qubivex run`, `qubivex bench` and every recorded benchmark are
unaffected.

## Not supported

- **Paged KV.** Each conversation's KV is contiguous per layer, and a prefix hit is a
  copy, so a reused prefix exists twice while the request runs. The index is designed
  so paging could replace the copy later; it is not built.
- **Sharing KV between running conversations** (copy-on-write).
- **Reuse of anything but a prefix:** no reuse of a matching middle or suffix.
- **KV on disk,** and KV that survives a restart or moves between servers.
- **Reduced-precision KV** (`f16` or quantised).
- **Per-API-key scopes.** There is one key, so the scope is the `user` field.
- **Sliding the context.** When the context is full the generation ends with
  `finish_reason: "length"`; old tokens are not dropped to make room.
