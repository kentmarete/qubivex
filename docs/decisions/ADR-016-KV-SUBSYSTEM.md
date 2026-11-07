# ADR-016: The KV subsystem: retained KV and prefix reuse

- Status: Proposed
- Date: 2026-11-07

## Context
Spec §4.7 asks for a KV subsystem with context reuse, prefix caching "where safe", bounded
memory, eviction, isolation between sequences and cleanup on cancellation, with interfaces
that allow paged KV later. Job 6 of `PLATFORM_AUDIT` makes it concrete: prefix reuse with a
bit-identity test, bounded and evictable.

Before this, each request built its KV cache from nothing and dropped it at the end. A chat
client resends the whole conversation every turn, so on OLMoE-1B-7B a 700-token
conversation cost about 6.6 s of prefill per turn on the CPU, even when only the last 20
tokens were new.

Two facts make reuse safe. First, a position's keys and values depend only on the tokens up
to it. Second, a prompt's logits are the same however it is split into passes (invariant 1).
A prompt that starts with tokens a finished generation already ran can therefore start from
a copy of that generation's KV and run only the rest, with the same bits.

## Decision
**Finished KV is retained, and a new prompt copies the longest matching prefix into its own
cache.**

- **A new crate, `qubivex-kv-cache`, with no dependencies.** `PrefixCache<V>` is the index.
  It finds the entry sharing the longest prefix with a prompt, keeps a budget by evicting
  the least recently used entry, and replaces an entry when a longer run of the same tokens
  arrives, so a conversation holds one entry and not one per turn. It knows nothing of
  tensors, so its rules are tested without a model. The runtime holds it as
  `Mutex<PrefixCache<KvCache>>`.
- **Copy, not share.** A retained `KvCache` is immutable. A hit holds it by `Arc`, and
  `KvCache::restore_from` (host) or `MetalForward::restore` (device buffers) copies the
  prefix into the request's own empty cache. No running sequence shares KV memory with
  another or with the index, so one request cannot change another's attention state. On the
  CPU a repeated 651-token prompt gave its first token 78 ms after the request, copy and
  last pass included, against 6.1 s for the first run in a fresh engine.
- **The last prompt token always runs**, because the next token needs its logits. Matches
  shorter than `MIN_REUSE` (32 tokens) are neither kept nor used: an entry costs a whole KV
  chunk (256 positions), and short prompts would fill the budget for a few passes.
- **What is kept is what ran.** At the end of a generation the cache holds the first `len`
  tokens of prompt plus output, since the last token generated has not been fed back in. That
  holds however it ended: finished, stopped, cancelled or past its deadline. A cancelled
  prefill keeps the chunks that ran, so a retry continues from them. A failed pass keeps
  nothing.
- **Bounded.** `EngineOptions::prefix_cache_bytes` and `qubivex serve --prefix-cache-mb` set
  the budget, converted to KV positions. A conversation larger than the whole budget is not
  kept, and evicts nothing.
- **Live requests come first.** Retained KV is priced in the memory plan with the
  reservations, so the expert cache shrinks to hold it. A reservation never waits or fails
  because of it: `reserve_kv` admits against reservations alone and then evicts retained KV
  until the plan holds both. A forward pass's `fit_room` does the same.
- **Memory pressure drops it first.** At `Elevated` (the level where prefetch stops),
  retained KV is cleared and nothing new is kept until the level falls. The sampling thread
  does this, so an idle server under pressure gives the memory back too.
- **Off by default.** With no budget, generation is the code path before this ADR, so `run`,
  `bench` and every recorded baseline are unchanged.

### Where it is safe: scopes
A reused prefix makes the first token arrive sooner, and that tells the caller that someone
already sent that prefix. A shared cache would let one client confirm a guess about another
client's prompt by timing it. `GenerateOptions::cache_scope` partitions the cache: entries
are only matched within their own scope. The gateway uses the OpenAI `user` field as the
scope, and requests with no `user` share one scope.

`user` is what the client says it is (ADR-015), so a client that claims another client's
`user` reaches that client's entries. Among callers holding the same API key that is the
current trust model. Per-key scopes belong to job 9. An operator who cannot trust clients to
send their own `user` should leave the cache off.

Retained KV stays in process memory and is never written to disk. It goes when the model is
unloaded.

### Reporting
`usage.prompt_tokens_details.cached_tokens`, as OpenAI reports it, in replies and in the
final stream chunk. `GenerationOutcome::cached_tokens` and `Generation::cached_tokens` carry
it for an embedding application. `/metrics` gains `qubivex_prefix_cache_lookups_total`,
`_hits_total`, `_tokens_reused_total`, `_evictions_total` and `_positions`, and `/v1/system`
gains a `prefix_cache` object per model. `Engine::prefix_stats` and
`Engine::clear_prefix_cache` are public.

### Paged KV later
The forward pass reads a layer's keys and values as one contiguous run of positions per KV
head (host) or per buffer (Metal). Paging would change `KvCache`'s storage and those reads,
plus the restore copy, which would become a page-table copy. The index would not change,
because it never sees the storage: `V` would be a set of pages, and a hit would share pages
copy-on-write instead of copying them. That is the step that removes the copy and the second
copy of a prefix while a request runs. It is not built, because nothing measured needs it
yet.

### Hardware
On the CPU the copy is a `memcpy` per KV head. On Metal it is one device submission of buffer
copies, and the KV stays in device buffers. On Apple's unified memory both count in the same
RAM ledger. A discrete GPU would need a tier choice for retained KV (device or host); there
is none to measure on yet.

## Verification
- `crates/qubivex-kv-cache`: 10 unit tests for the index (longest prefix, the last token,
  minimum length, scopes, budget and LRU order, over-budget refusal, one entry per
  conversation, replacement before eviction, a hit outliving eviction, counters).
- `crates/qubivex-runtime/tests/prefix.rs`, 19 tests on the fixtures:
  - A restored cache gives bit-identical logits at prefixes on both sides of a chunk
    boundary, and writing to a copy leaves its source unchanged (all four fixtures).
  - The same session of prompts with reuse on and off (repeat, next turn, shared start,
    shorter prompt, nothing shared) gives identical tokens, with prefill chunks of 7 and
    256, sampled at temperature 0.8, on the CPU and on Metal in both placements.
  - Scopes; the budget; the ledger; reservations evicting retained KV, and a refused one
    evicting nothing; a finishing request counted once; the plan pricing retained KV and
    the expert cache coming back; pressure clearing it with no request running.
  - A prefill cancelled by progress keeps whole chunks, and its retry is bit-identical. A
    deadline already past and a failed pass keep nothing. A stopped generation keeps all
    but its last token.
  - Four threads running overlapping prompts against a two-entry budget, so hits, inserts,
    replacements and evictions race: every result equals the run with no reuse.
- `crates/qubivex-api/tests/gateway.rs`: over HTTP, a repeat reports `cached_tokens`,
  matches a model with reuse off, does not reach another `user`, and shows in `/metrics`
  and `/v1/system`, streamed or not.
- **Removing each rule in turn**, 33 rules: 29 caught, two of them only after adding the test
  each exposed (the ledger after an eviction, a finishing request counted twice). Three
  mutations showed a rule was redundant, so it was removed: a guard against evicting when a
  pass's own KV cannot fit (reservations rule that out first), pricing the copy before it
  runs (the reservation and the next pass already do), and preferring the latest entry on a
  tie. One survives on purpose: the restore copy takes a forward-pass turn. No test can show
  it is needed, since the device lock orders submissions and ThreadSanitizer does not see
  Metal. It is kept because device KV has only ever been allocated under the turn, and it
  costs at most one pass of waiting.
- ThreadSanitizer on nightly, as in ADR-015: no reports over the 19 prefix tests and the 6
  gateway tests on macOS. The positive control was ADR-015's, with the same build. `--test
  prefix` is added to the sanitizer run in CI and to ADR-015's command.
- **Real model**, OLMoE-1B-7B Q4_K_M, M3 Pro, on power:
  - `tests/real_model.rs` (env-gated): a 651-token prompt, its repeat and its next turn give
    the tokens of a full run on the CPU and on Metal.
  - Over HTTP, `serve --prefix-cache-mb 512 --context 2048` against `--prefix-cache-mb 0`,
    both with warm experts, a chat of three turns with 48 greedy tokens each:

    | Turn | Prompt tokens | Reused | With reuse | Without |
    |---|---|---|---|---|
    | 1 | 673 | 0 | 6.57 s | 6.62 s |
    | 2 | 714 | 695 | 0.57 s | 6.59 s |
    | 3 | 738 | 720 | 0.45 s | 6.78 s |
    | 1, another `user` | 673 | 0 | 6.50 s | 6.57 s |

    The text of every turn is identical with and without reuse. The template's
    re-rendering of the assistant's reply tokenizes back to what the model generated, so the
    second turn misses only the new user message. These are single runs of whole requests,
    not benchmarks by `BENCHMARKING.md`.

## Not built
- **Sharing without a copy**, which is paged KV (above). While a request runs, its prefix
  exists twice: once retained and once in its own cache.
- **KV-aware scheduling** (spec §7.4 stage 3): the scheduler does not prefer requests that
  would hit, and a hit does not lower a request's reservation. That belongs to batching.
- **Persisting retained KV** across restarts, or to disk under pressure.
- **Per-key scopes**, with job 9.
- **Fixed expert caches** (`--expert-cache-mb`) have no plan and no pressure monitor, so
  retained KV there is bounded by its budget alone.

## Consequences
- `GenerateOptions` is no longer `Copy`, because it carries the scope.
- `Generation` and `GenerationOutcome` gain `cached_tokens`, and `GenerationRequest` gains
  `cache_scope`: struct literals in an embedding application need the new fields.
- Replies gain `usage.prompt_tokens_details`.
- With reuse on, the ledger's KV charge includes retained KV, and the expert cache is smaller
  by what is retained until a request needs the room.
