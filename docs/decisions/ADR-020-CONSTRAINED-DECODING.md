# ADR-020: Constrained decoding: JSON mode and JSON Schema

- Status: **Proposed (2026-11-15)**; implemented behind the request field only (no request
  without `response_format` changes)
- Date: 2026-11-15
- Roadmap: `docs/ROADMAP_GENERAL_PURPOSE_RUNTIME.md` step 6 (P2)

## Context
Clients ask for JSON through the OpenAI `response_format` field: `{"type":"json_object"}`
(any JSON object) or `{"type":"json_schema","json_schema":{"schema":...}}` (a value that
validates against a schema). The gateway refuses both today. Prompting alone is not
enough: the roadmap's acceptance is that *every* output parses and validates, at
temperature > 0 too, which only token-level constraint can promise.

The same machinery is needed again for tool calling (P5, constrained arguments) and for
any later grammar a client supplies, so it must not be a JSON special case inside the
sampler.

Facts the design rests on:

- Sampling happens in one place, `Engine::generate_with`, on the request's own thread;
  batched passes (ADR-017) only combine forward passes, never sampling. A per-request
  constraint therefore touches no other request.
- Every tokenizer the runtime loads today is byte-level BPE (GGUF is `gpt2`-only, and the
  supported `tokenizer.json` models use the `ByteLevel` decoder), so every token is a
  byte string and the 256 single bytes are all tokens.
- The roadmap's risk: a mask over a 150k vocabulary every token. A naive check (run the
  grammar over every token's bytes) costs ~1M grammar steps per token.

## Decision

### 1. A new crate, `qubivex-grammar`
Embeddable, no I/O, depending only on `serde_json` (already a workspace dependency).
It has three layers, each reusable on its own:

1. **Grammar IR**: rules of alternatives of sequences of *byte classes* (256-bit sets) and
   rule references. Repetition is right recursion; bounded counts are unrolled up to a
   cap. Compilation checks that no rule is left-recursive (directly or through nullable
   prefixes) and every rule is productive, so every live state can still finish.
2. **JSON Schema compiler** (`schema → Grammar`), and the fixed JSON-mode grammar (any
   object). The JSON text is RFC 8259: strings are valid UTF-8 with JSON escapes (lone
   surrogate escapes excluded, since parsers reject them), and numbers are bounded
   (at most 15 fraction digits and two exponent digits) so every number is finite in an
   f64 parser.
3. **Matcher**: the grammar's pushdown state, *determinised lazily*. A parse stack is a
   hash-consed linked list of grammar positions (equal stacks are one id), a state is
   the sorted set of live stacks (interned to an id), and the transition
   `(state, byte) → state` is computed once and cached. After warm-up, advancing by a
   byte is a table lookup.

### 2. Token masks over a vocabulary trie
`TokenTrie` holds every token's bytes in a byte trie, built once per tokenizer. The mask
for a state is a depth-first walk of the trie that follows cached transitions and prunes
a subtree as soon as its prefix is dead; a token is allowed when its last byte leaves a
live state. Masks are cached per state, so a state that recurs (inside a string, between
array items) costs one bitset lookup.

End-of-sequence ids are never matched as bytes: they are allowed exactly when the state
accepts (the top-level value is complete). Other special (control) tokens are never
allowed. Generation therefore ends with an EOS once the value is complete, unless the
value can still grow (a number), where the model chooses.

### 3. Sampling semantics
With a constraint, disallowed logits become −∞ **before** temperature, top-k and top-p,
so those settings act on the allowed tokens only (the order `transformers` uses for
logits processors). Greedy decoding first checks the unconstrained argmax against the
matcher and needs no mask when it is allowed; this is exact, including the lowest-id
tie rule. The RNG is consumed exactly as without a constraint, so a seed still fixes
the output.

If a state allows no token and does not accept (possible only with a vocabulary that
lacks some byte), the request fails with an error rather than emitting invalid output.

### 4. The JSON Schema subset
Supported: `type` (one or a list), `properties`, `required`, `additionalProperties`
(`false`, `true` or absent: only declared properties are generated, which validates in
every case; a schema value for extra properties is refused), `items`, `prefixItems`,
`minItems`, `maxItems`, `enum`, `const`, `anyOf`, `allOf` with one member, local `$ref`
into `$defs`/`definitions` (recursion included), `minLength`, `maxLength`, and the
annotation keywords (`title`, `description`, `$schema`, `$id`, `default`, `examples`,
`$comment`, `strict`-style vendor fields are ignored).

Properties are generated in schema order: required ones always, optional ones each
present or absent. Whitespace is allowed where JSON allows it, at most 16 bytes per gap,
so a model cannot stall in whitespace forever.

**Refused, never ignored** (400 `invalid_request_error`, naming the keyword): `oneOf`
(exactly-one is not a context-free property of the text), `not`, `pattern`,
`patternProperties`, `format`, numeric bounds (`minimum`, `maximum`, `exclusive*`,
`multipleOf`), `uniqueItems`, `contains`, `dependent*`, `if/then/else`,
`propertyNames`, `min/maxProperties`, `unevaluated*`, remote `$ref`, and any keyword the
compiler does not know. A schema that compiles is one whose every output validates.

Limits, all refusals: grammar size (rules and byte elements), bounded counts above 1024,
schema nesting depth 64.

### 5. Where it plugs in
- `qubivex-tokenizer`: `Tokenizer::token_bytes()` gives each id's bytes (`None` for
  special tokens), from the GPT-2 byte table for `ByteLevel` decoders. Other decoders
  return an error, so constrained requests on such a model are refused (no model loaded
  today is affected).
- `qubivex-runtime`: `GenerateOptions::constraint: Option<Arc<Grammar>>`, mirrored on
  `GenerationRequest`. The engine builds the `TokenTrie` lazily on first use and keeps
  it. Each request gets its own `Matcher`. Requests without a constraint run the code
  they run today.
- `qubivex-api`: `response_format` on `/v1/chat/completions` (`text`, `json_object`,
  `json_schema`). The schema is compiled at request validation, so errors are 400s
  before admission. `/v1/completions` is unchanged (OpenAI has no such field there).

`max_tokens` and `stop` still apply and can cut a value short; such responses finish
with `length` or `stop`, as OpenAI does. Only `stop`-by-EOS outputs are promised to
parse.

## Consequences
- One generic constraint path; JSON Schema is one front end. Tool-call arguments (P5) and
  a client grammar later are further front ends over the same IR and matcher.
- A cold mask walks the reachable part of the trie once per new state; the cost is
  measured on the real Qwen2.5 vocabulary (151 643 tokens) and recorded in the milestone
  report. Matcher memory is per request and bounded by a state cap; exceeding it fails
  the request.
- A new workspace crate: `xtask` dependency rules list it (no internal dependencies;
  the runtime may depend on it).

## Verification
- Grammar property tests: random walks through the matcher (no model) over a schema
  suite, thousands of outputs per schema, each parsed with `serde_json` and validated by
  an independent validator; the matcher's mask agrees with a brute-force per-token check.
- Fixture models (random weights, temperature 1, many seeds): every EOS-finished output
  parses and validates. Random weights are the hardest case: the model has no idea of
  JSON.
- Real Qwen2.5-0.5B-Instruct over the gateway at temperature 0 and 0.8.
- Requests without `response_format` keep their tokens bit for bit (the existing
  invariant and real-model tests).
