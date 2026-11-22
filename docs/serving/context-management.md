# Context management

A request longer than the model's context used to get a 400 ("the prompt has N tokens;
the context holds C, leaving no room to generate"). With context management (on by
default) the server prepares it to fit instead: it keeps the instructions and the
current message, keeps the most recent turns verbatim, quotes the most relevant parts of
older turns, and cuts a long pasted document down to its relevant blocks. Requests that
fit are never touched. Decision record: [ADR-021](../decisions/ADR-021-CONTEXT-MANAGEMENT.md).

## Where it sits

```text
client ──► gateway (qubivex-api)
             │  auth, JSON, params, scheduler place
             ▼
           Engine::prepare_managed (qubivex-runtime)
             │
             ├── Engine::prepare ── fits ─────────────────────────────┐  the path every
             │     (chat template + tokenizer + context check)        │  request took before
             │                                                        │
             └── too long, management on                              │
                   ▼                                                  │
                 ContextEngine::prepare (qubivex-context)             │
                   │  budget: context − answer reserve (∧ server cap) │
                   │  instructions too long? → 400 context_length_exceeded
                   │  current message too long? → keep relevant blocks│
                   │  older turns → note of quoted excerpts           │
                   │  exact recount after every change                │
                   │  checkpoint store (memory or files)              │
                   ▼                                                  │
                 Engine::prepare again (the same check as any request)│
                   │                                                  │
                   ▼                                                  ▼
           scheduler admission (KV) ──► worker thread ──► forward passes, MoE routing,
                                                          experts, sampling: unchanged
```

The context engine knows nothing of models: the runtime gives it an exact counter (the
model's own chat template and tokenizer), and the prepared request goes through the same
`prepare` as every other one, so it cannot exceed the context. It does not touch the
scheduler, the KV cache, expert routing or any tensor code.

## What a client sees

- The answer, in the usual shape. Streams stream as usual. Nothing about the preparation
  is added to the assistant's message.
- Two response headers, only on prepared requests:
  - `x-qubivex-context: compacted`, `reduced`, or `compacted,reduced`
  - `x-qubivex-context-tokens: <tokens sent>;<tokens prepared>`
- `usage.prompt_tokens` is what the model ran: the prepared prompt.
- When it cannot fit without dropping what must stay, a 400 with code
  `context_length_exceeded` and `param` `messages` (or `prompt`) saying why.

## What is kept

| Part of the request | What happens |
|---|---|
| System and developer messages | Always verbatim. If they alone exceed the budget: 400, never shortened |
| The last user message (and anything after it) | Verbatim, unless it alone is too long: then its first and last blocks and the blocks most relevant to them, with a marker for each run left out |
| Recent turns | Verbatim, newest first, while they fit |
| Older turns | Condensed into a note of quoted sentences and whole code blocks, each with its message number and role |

The note is in a user message in front of the first kept turn, so quoted text never
gains system authority:

```text
[Earlier conversation, condensed to fit the model's context. Quoted excerpts only; nothing was added.]
#8 user: Record 3: the code word for locker 3 is dune-3.
#9 assistant: Recorded: locker 3 has code word dune-3.
[Also from the earlier conversation, relevant to the current message:]
#16 user: Record 7: the code word for locker 7 is basalt-7.
[End of earlier conversation: messages #2–#99.]
```

A reduced message keeps its text verbatim around markers such as
`[… lines 41–212 of this message omitted to fit the model's context …]`. Code is kept or
left out whole, never paraphrased.

Excerpts are ranked by relevance to the current message (BM25 over the request's own
text, with identifiers split so `reserve_kv` matches "reserve kv"), by salience (numbers,
identifiers, code, words such as *decided*, *must*, *never*) and by recency. Nothing is
summarised by a model, so nothing can be invented.

## The budget

```text
context  = the model's context limit ∧ --context ∧ what the memory plan holds
reserve  = max_tokens (or --default-max-tokens), at most half the context
input    = context − reserve, and at most --max-prompt-tokens
```

`GET /v1/system` shows each model's `context_limits` (`model`, `configured`, `memory`,
`effective`). A request that fits the context but waits for memory other conversations
hold is queued by the scheduler as before; it is not compacted.

## Later turns

The note is stored as a checkpoint, found by the conversation's earlier messages, the
model and the client's scope (`user` or named key, as for the prefix cache). The next
turn reuses it unchanged and extends it, so the prompt keeps the same beginning and, with
`--prefix-cache-mb`, its KV is reused (`usage.prompt_tokens_details.cached_tokens`).
After the stored part come excerpts chosen for the current message alone, which are never
stored. Checkpoints of one client are never used for another.

Checkpoints live in memory (64 MiB by default) or, with `--context-store DIR`, in files
that survive a restart (256 MiB, kept 7 days after last use; they contain excerpts of
conversations). A failed write never fails a request; the next turn rebuilds the note.

## Diagnostics

`POST /v1/context/inspect` takes a chat or completion body and answers what would be
done, without generating (abridged):

```json
{
  "object": "context.inspection", "model": "qwen2.5-0.5b-instruct",
  "managed": true, "prompt_tokens": 929,
  "context_limits": {"model": 32768, "configured": 1024, "memory": 1024, "effective": 1024},
  "budget": {"context": 1024, "reserve": 12, "input": 1012},
  "report": {
    "original_tokens": 3426, "prepared_tokens": 929, "compacted": true, "reduced": false,
    "messages_condensed": 98, "excerpts": 9, "checkpoint": "miss",
    "messages": [{"index": 0, "role": "system", "fate": "kept"},
                 {"index": 1, "role": "user", "fate": "condensed", "excerpts": 0, "units": 2}],
    "total_time": 7913
  }
}
```

Add `"include_text": true` to see the prepared messages (the caller's own text only).
Times are in microseconds. The log has one `context_managed` or `context_refused` line
per prepared or refused request, with counts and timings and never text. Metrics:
`qubivex_context_overflows_total`, `qubivex_context_managed_total{strategy}`,
`qubivex_context_failures_total{reason}`, `qubivex_context_tokens_saved_total`,
`qubivex_context_checkpoint_hits_total`, `qubivex_context_prepare_seconds`.

## Settings

| Flag | Config key | Default | Meaning |
|---|---|---|---|
| `--no-context-management` | `context_management = false` | on | Refuse long requests with the old 400 |
| `--context-note-share F` | `context_note_share` | 0.25 | Share of the room after the instructions and current message kept for the note |
| `--context-store DIR` | `context_store` | memory | Keep checkpoints in files there |
| `--context-store-mb N` | `context_store_mb` | 64 / 256 | Checkpoint store size |
| `--context-store-days N` | `context_store_days` | 7 | How long a checkpoint file is kept after its last use |

An application embedding the runtime calls `Engine::prepare_managed` with a
`qubivex_runtime::context::ContextEngine`; `Engine::prepare` is unchanged.

## Limits

- No model-written summaries: the note quotes. A model summariser would need its own
  scheduled request, and could state what the conversation did not say.
- No semantic (embedding) retrieval: the runtime runs no embedding model yet. Ranking is
  lexical (BM25), salience and recency.
- Tools are refused by the API, so there are no tool schemas or results to keep.
- `qubivex run` takes one prompt and is unchanged.

## Measured cost

Qwen2.5-0.5B-Instruct tokenizer, M3 Pro, release build (`context_bench` example):

| Case | Time |
|---|---|
| A request that fits (268 tokens): `prepare` / `prepare_managed`, on / off | 199 µs / 193 µs / 187 µs (no difference beyond noise) |
| The context check that refuses a 19 988-token conversation (as before) | 11.2 ms |
| Preparing it to 3 614 tokens (4096 context), note rebuilt | 64.5 ms |
| The same with a stored note | 46.0 ms |
| A 91 KiB source file in one message, to 3 182 tokens | 87.5 ms |

The engine alone (byte counter, criterion): 100 turns 2.2 ms, 1 000 turns 40.5 ms (20.5 ms
with a stored note), a 250 KB document 7.7 ms. Most of the real-model time is the
tokenizer counting the candidates.

On Qwen2.5-0.5B-Instruct with a 1024 context, a 3 426-token conversation of 60 code-word
records was prepared to 929 tokens; the model answered the question about a condensed
record correctly ("basalt-7"), and the next turn, about another condensed record, reused
the stored note, answered correctly ("basalt-13") and skipped 170 prompt tokens through
the prefix cache (`crates/qubivex-runtime/tests/real_context.rs`).
