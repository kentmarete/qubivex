# ADR-021: Context management for requests longer than the context

- Status: **Proposed (2026-11-22)**; implemented. It acts only on requests the runtime
  refused before ("the prompt has N tokens; the context holds C"). Every request that
  ran before runs the same way, through the same calls.
- Date: 2026-11-22
- Crate: `qubivex-context` (new), wired through `qubivex-runtime` and `qubivex-api`

## 1. Assessment: what exists (derived from the source, 2026-11-22, `main` de652f3)

### 1.1 Request lifecycle
`POST /v1/chat/completions` and `POST /v1/completions` (`qubivex-api/src/lib.rs`):

1. `check_key` → `json_body` → `ChatRequest::messages()` turns the body into
   `(role, text)` pairs. Roles: `system`, `developer` (sent to the template as
   `system`), `user`, `assistant`. Content is text or text parts. **`tools` and
   `tool_choice` are refused (400); there are no tool messages.**
2. `Common::params` validates sampling, `max_tokens`, `priority`, `timeout`, `user`.
3. `generate()`: model lookup → `Scheduler::enter` (a bounded queue place) →
   `worker::request` builds a `GenerationRequest` → **`Engine::prepare` on a blocking
   thread** (chat template + exact tokenization + the context check) →
   `max_prompt_tokens` cap → `Ticket::ready(positions)` (memory admission) → grant →
   `worker::spawn` runs `Engine::run_prepared` on a dedicated thread → events over a
   bounded channel → JSON or SSE.

### 1.2 The context check today
`Engine::prepare` (`qubivex-runtime/src/request.rs`):

```
prompt = tokenize(apply_chat_template(messages, add_generation_prompt = true))
if prompt >= max_context → 400 "the prompt has N tokens; the context holds C, leaving no room to generate"
max_new = min(max_tokens or default_max_tokens, max_context − prompt)
```

`max_tokens` is an upper bound: a prompt one token short of the context still runs and
ends with `length` after one token. This is the behaviour a client sees today and the
behaviour this ADR must keep for every request that fits.

### 1.3 Context limits
`Engine::load` derives one number, `max_context`:

- model limit: `manifest.context_length`, else `config.json` `max_position_embeddings`
  (`dims.max_context`), else 4096;
- configured limit: `EngineOptions::max_context` (`--context`, `context` on
  `POST /v1/models/load`), capped by the model limit;
- memory limit: the planner (ADR-005, `sequences: 1`) shortens the context to what one
  conversation's KV fits beside one expert slot per layer.

So `max_context` is already the most restrictive of model, configuration and memory for
**one** conversation. Concurrent conversations share the plan through
`Engine::reserve_kv`; a request that does not fit *now* waits in the scheduler
(`MemoryPolicy::Queue`) or is refused (`Refuse`). The device KV pool (ADR-019 am. 5)
declines layers to the host when full, so it never shortens the context.

### 1.4 Tokenizer
`qubivex-tokenizer`: Hugging Face `tokenizer.json` or a GGUF vocabulary, chat templates
in sandboxed MiniJinja. Counting is exact: the same `apply_chat_template` + `encode`
the forward pass uses. Templates are not additive per message (each adds its own
framing), so only a whole rendering gives an exact count.

### 1.5 KV cache
Each request owns its `KvCache`. `qubivex-kv-cache::PrefixCache` (ADR-016) keeps
finished KV by token prefix, scoped by `cache_scope` (named key or `user`), off by
default (`--prefix-cache-mb 0`). A prompt that starts with tokens a finished request
ran starts from a copy of that KV. Nothing else keeps KV between requests.

### 1.6 Conversations and storage
The API is stateless: the client sends the whole history every time. The server has no
conversation id, no message store and no database; persistent files are the model
registry and caches under `~/.qubivex`. No crate embeds an embedding model or a vector
index.

### 1.7 Concurrency, cancellation, observability
Weights and the expert store are shared; each generation has its own thread, KV and
reservation; `CancelOnDrop` sets a flag the engine checks between prefill chunks and
tokens. `/metrics` is plain atomics rendered as Prometheus text; the log is one line per
event and never holds prompt or generated text.

### 1.8 Integration point
There is exactly one place where an over-long request fails: `Engine::prepare`. A
context engine that runs **only after `prepare` refused for length** costs a request
that fits nothing at all: not a token count, not a branch in the hot path.

## 2. Decision

### 2.1 Activation rule
```
prepared = engine.prepare(request)              // unchanged
match prepared {
  Ok(p)                         => run p        // the existing path, untouched
  Err(ContextOverflow) if on    => manage, then engine.prepare(managed) again
  Err(e)                        => the same error as before
}
```

- On by default, because it changes only requests that were refused before.
  `--no-context-management` (or `context_management = false`) restores the refusal,
  byte for byte.
- The second `prepare` is the final check: the managed prompt goes through the same
  template, tokenizer and context check as any request, so a managed request can never
  exceed the context.
- `RequestError` gains `ContextOverflow { prompt_tokens, context }`; its message is the
  old one, and the gateway maps it to the same 400, so clients see no difference when
  management is off or fails.

### 2.2 Budget (`qubivex_context::Budget`)
```
context  = engine.max_context()                      (model ∧ configured ∧ memory)
reserve  = requested max_tokens, else default_max_tokens, capped at context / 2
input    = min(context − reserve, max_prompt_tokens)
```
`Engine::context_limits()` reports the three limits separately (model, configured,
effective) for diagnostics and `/v1/system`. The reserve cap keeps at least half the
context for input; it is a fraction of the model's own context, not a fixed number. A
managed request still has `max_tokens` as an upper bound, as every request does.

KV under concurrency is **not** an input to the budget. A request that fits the
context but not the memory left by other conversations waits or is refused by the
scheduler, as today. Compacting it instead would make a conversation's answer depend on
server load, and the space would be free a moment later.

### 2.3 Overflow detection
`Overflow` classifies a refused request from exact counts:

- `Mandatory`: system and developer messages alone exceed the input budget. Never
  compacted (Scenario 5): a 400 naming their token count and the context, suggesting a
  larger `--context` or a model with a longer context.
- `CurrentMessage`: the last user message (with the system messages) exceeds the
  budget: document reduction (2.5).
- `History`: the conversation exceeds the budget: history compaction (2.4).

Tokenization failure keeps the original error (the template or tokenizer message) and
never falls back to an estimate. Per-message counts (each message's content encoded
alone) guide selection; only whole renderings decide whether something fits.

### 2.4 History compaction (Scenario 2)
Messages are classified:

- **pinned**: every system/developer message (kept verbatim, in place), the last user
  message and anything after it;
- **turns**: older messages grouped as a user message with the assistant messages that
  follow it, so the user/assistant alternation templates require survives.

Recent turns are kept verbatim, newest first, while they fit, leaving a share of the
budget (`note_share`, default 25 %) for older history. Older turns become a **context
note**: verbatim excerpts, never paraphrase, so it cannot introduce a fact:

```
[Earlier conversation, condensed to fit the model's context: messages 2–9 of 14 are
shown as excerpts. Nothing was added.]
#2 user: …
#3 assistant: …
[End of earlier conversation]
```

- Excerpts are sentences (prose) or whole fenced code blocks (code is never cut or
  paraphrased; a block that does not fit is named with its line count and omitted).
- Sentences are ranked by BM25 relevance to the current user message, salience (numbers,
  identifiers, code, decision and obligation words), and recency; the chosen ones are
  printed in the original order with their message number and role (provenance).
- The note goes in a **user-role** message (merged in front of the first kept user
  message, so alternation holds). Putting user text in a system message would give it
  system authority; a user message gives quoted assistant text none it lacked.
- Every kept message keeps its role; `#n` numbers refer to the client's message list.

### 2.5 Document reduction (Scenarios 3 and 4)
When one message is larger than its share (the current user message, a pasted file, a
`/v1/completions` prompt), it is cut into blocks at blank lines; fenced code blocks are
blocks; a block over the budget is cut at line boundaries. The first and last blocks
(where instructions and the question usually are) are always kept; the others are
ranked by BM25 against them and kept in their original order, with markers naming what
was omitted:

```
[… lines 41–212 of this message omitted to fit the model's context (1,830 tokens) …]
```

Code is kept or omitted verbatim, never summarised. Identifier-aware tokenization for
ranking splits `snake_case` and `camelCase`, so a question about `reserve_kv` finds the
block that defines it.

### 2.6 Retrieval and isolation
Retrieval is in-request: the index is built from the request's own messages and
dropped with it, so it cannot reach another user's data. BM25 is local, exact and needs
no model. **Semantic (embedding) retrieval is not implemented**: the runtime cannot run
an embedding model today. A ranker trait would be added with the first embedding model,
not before.

### 2.7 Checkpoints and the store (incremental compaction)
A compacted conversation is resent next turn with one more exchange. Recomputing the
note each turn would change it each turn and defeat the prefix cache. A **checkpoint**
records, for a conversation prefix, the note's excerpts and the messages it covers. The
next request whose first *k* messages hash the same reuses that note unchanged and adds
only what is new, so the rendered prompt keeps the same prefix (system + note) turn after
turn, and the prefix cache (when enabled) reuses its KV: the gateway reports it as
`cached_tokens` in `usage`, as for any request.

- Key: SHA-256 over the cache scope, the model id, the budget and the messages
  (length-prefixed). Different scopes never share a checkpoint, as with ADR-016.
- `MemoryStore`: bounded LRU (default 64 MiB), the default.
- `FileStore` (`--context-store <dir>`): one JSON file per checkpoint, written to a
  temporary file and renamed, so a crash leaves the old checkpoint or the new one, never
  half of one; age and size retention (default 7 days, 256 MiB) enforced on open and on
  write. Off by default: it keeps excerpts of conversations on disk.
- A store failure never fails a request: the conversation is compacted from scratch.
- The store holds no messages: the client owns its history.

### 2.8 Concurrency and cancellation
Preparation runs on the blocking pool, as `prepare` does today, and holds no lock while
it works. The store's lock covers one map lookup or insert. Each request computes its
own plan; two requests of one conversation compute the same note, and the last write
wins with an equal value. Preparation checks a cancellation flag and the request's
deadline between phases, so an abandoned request stops preparing.

### 2.9 Observability and API
- Response headers on managed requests only: `x-qubivex-context: compacted` (or
  `reduced`, or both, comma-separated) and
  `x-qubivex-context-tokens: <original>;<prepared>`. Bodies, chunks and streams are
  unchanged; `usage.prompt_tokens` is what the model ran.
- Log event `context_managed` (counts and timings, never text) and
  `context_refused` (why).
- Metrics: `qubivex_context_overflows_total`, `qubivex_context_managed_total{strategy}`,
  `qubivex_context_failures_total{reason}`, `qubivex_context_tokens_saved_total`,
  `qubivex_context_checkpoint_hits_total`, `qubivex_context_prepare_seconds`.
- `/v1/system`: `context_management` settings and, per model, `context_limits`.
- `POST /v1/context/inspect`: a chat or completion body; answers the budget, the
  decision and the provenance (which messages and lines were kept, condensed or
  omitted) without generating. The prepared text is included only with
  `"include_text": true`, and it is only ever the caller's own request.

### 2.10 Configuration
| Setting | CLI (`serve`) | Config file | Default |
|---|---|---|---|
| On or off | `--no-context-management` | `context_management` | on |
| Share for older history | `--context-note-share` | `context_note_share` | 0.25 |
| Checkpoint store | `--context-store <dir>` | `context_store` | in memory |
| Store size | `--context-store-mb` | `context_store_mb` | 64 (memory), 256 (files) |
| Store age | `--context-store-days` | `context_store_days` | 7 |

The output reserve follows the request's `max_tokens` and `--default-max-tokens`; the
input budget follows `--context` and `--max-prompt-tokens`. No new limit duplicates
them.

## 3. What this does not do (and why)
- **Abstractive summaries by the loaded model.** A summary from the model can state
  what the conversation did not say, which the requirement forbids, and running it inside
  a request's preparation would take a second KV reservation outside the scheduler. The
  note is extractive instead. A model summariser needs its own scheduled request kind;
  it is future work, not a stub.
- **Semantic retrieval**: no embedding model runs in the runtime (2.6).
- **Tool schemas and tool results**: the API refuses tools, so there are none to keep.
  When tools arrive (roadmap P5) they are pinned like system messages.
- **KV kept between requests other than the prefix cache**: none exists; checkpoints
  make compacted prompts prefix-stable so the existing cache can work, nothing more.
- **`qubivex run`**: one prompt, no history; unchanged.

## 4. Consequences
- A request that fits: identical, same calls (`prepare` once). Verified by tests that
  compare responses with management on and off, and by the benchmark.
- A request that did not fit: answered from a prepared context instead of a 400, with
  headers that say so. With management off: the old 400.
- New crate `qubivex-context` (no internal dependencies; `serde`, `serde_json`, `sha2`,
  `thiserror`); `qubivex-runtime` depends on it and re-exports it as
  `qubivex_runtime::context`.

## 5. Implementation plan (each step tested on its own)
1. `RequestError::ContextOverflow`; `Engine::context_limits()`; exact counter.
2. `qubivex-context`: budget, overflow classification, segmentation and BM25.
3. History compaction and document reduction, with the shrink-and-recount loop.
4. Checkpoints: memory and file stores, retention, incremental reuse.
5. `Engine::prepare_managed` (runtime) and the gateway: activation, headers, cancel.
6. Metrics, log events, `/v1/system`, `POST /v1/context/inspect`; CLI and config file.
7. Tests (unit, gateway with fixture models, a real-model test behind the env vars) and
   a benchmark (`cargo bench -p qubivex-context`, plus the gateway timing in
   `docs/performance`).
