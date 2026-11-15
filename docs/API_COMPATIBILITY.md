# API compatibility

What `qubivex serve` answers today and what it will answer, by API. The field-by-field
reference for what is implemented stays in [serving/openai-compatibility.md](serving/openai-compatibility.md)
and [serving/http-api.md](serving/http-api.md); this page is the compatibility plan.

**Rule (unchanged):** a field the server understands but cannot honour is refused with a
400 naming it, never silently dropped. That holds for images, audio, `response_format`
and tools until each is supported end to end.

State as of 2026-11-15 (`main` at `8a70709`). ✅ implemented and tested over HTTP,
✗ missing.

## OpenAI-compatible API

| Endpoint / feature | Now | Target | Roadmap step |
|---|---|---|---|
| `GET /v1/models`, `GET /v1/models/{id}` | ✅ | ✅ + capabilities (modalities, structured output, tools) | 3 |
| `POST /v1/chat/completions` | ✅ text | + media parts, `response_format`, `tools` | 6, 10, 11 |
| `POST /v1/completions` | ✅ | ✅ (+ `response_format`) | 6 |
| Streaming (SSE), `stream_options.include_usage` | ✅ | ✅ | — |
| System / developer messages, history | ✅ | ✅ | — |
| Usage (prompt, completion, cached tokens) | ✅ | + media tokens | 10 |
| Cancellation on client disconnect | ✅ | ✅ | — |
| `response_format: json_object` | ✗ (400) | constrained decoding | 6 |
| `response_format: json_schema` (strict) | ✗ (400) | constrained decoding; output validated | 6 |
| `tools`, `tool_choice`, tool messages | ✗ (400) | template-driven, arguments constrained by the tool's schema | 11 |
| `image_url` parts (`data:` URIs) | ✗ (400) | supported for models with a vision component | 10 |
| `input_audio` parts | ✗ (400) | for models with an audio component | 13 |
| Remote `http(s)` image URLs | ✗ (400) | stay refused by default (SSRF); a flag with an allowlist only if needed | — |
| `logprobs`, `top_logprobs` | ✗ (400) | top-k log-probabilities from the sampler | later |
| `presence_penalty`, `frequency_penalty`, `logit_bias` | only neutral values | sampler penalties | later |
| `n` > 1 | ✗ (400) | later | later |
| `POST /v1/embeddings` | ✗ (404) | for embedding models | later |
| `/v1/responses`, assistants, files, images, audio endpoints | ✗ (404) | not planned | — |

## Ollama-compatible API

Compatibility means the documented request and response shapes and behaviour of
Ollama's HTTP API, not just its paths. Nothing is implemented yet (`/api/*` answer 404).

| Endpoint | Fields to support | Notes |
|---|---|---|
| `GET /api/tags` | `models[]` with `name`, `model`, `modified_at`, `size`, `digest`, `details` | Names map to Qubivex model ids; no aliasing beyond a documented rule |
| `POST /api/show` | `model` → `details`, `model_info`, `template`, `parameters`, `capabilities` | Values from the manifest |
| `POST /api/chat` | `model`, `messages[]` (`role`, `content`, `images` as base64), `format` (`"json"` or a JSON Schema), `stream` (default **true**, NDJSON), `options` (`temperature`, `top_p`, `top_k`, `seed`, `num_predict`, `stop`, `num_ctx`), `keep_alive`, `tools` | Streamed as newline-delimited JSON objects ending with `done: true` and timing fields (`total_duration`, `eval_count`, …) |
| `POST /api/generate` | `model`, `prompt`, `system`, `images`, `format`, `stream`, `options`, `raw`, `context` (if KV can be resumed by id) | `context` support depends on retained KV (ADR-016) |
| `POST /api/embeddings` / `/api/embed` | `model`, `prompt` / `input` | Only with embeddings support |
| `/api/pull`, `/api/create`, `/api/push`, `/api/delete`, `/api/copy`, `/api/ps` | — | Pull may map onto the Ollama registry importer (roadmap step 12); the rest are out of scope at first |

Unsupported Ollama options are refused with an Ollama-shaped error (`{"error": "..."}`)
and the right status, following the same never-drop rule.

## Errors (both APIs)

| Condition | Status | OpenAI `code` | Ollama |
|---|---|---|---|
| Unknown model | 404 | `model_not_found` | `{"error": "model '…' not found"}` |
| Model loading / not ready | 503 | `model_loading` | same text |
| Context too long | 400 | `context_length_exceeded` | same text |
| Unsupported modality for this model | 400 | `unsupported_modality` (new) | same text |
| Unsupported field or value | 400 | `unsupported_parameter` | same text |
| Bad or oversized media | 400 / 413 | `invalid_image` / `payload_too_large` (new) | same text |
| Queue full / memory pressure | 429 / 503 | `overloaded` | same text |
| Rate limited | 429 | `rate_limit_exceeded` | same text |
| Unauthenticated | 401 | `invalid_api_key` | same text |

Existing codes are listed in [serving/http-api.md](serving/http-api.md#errors); new ones
are added there when implemented.

## Verification

Each API is verified with its own official client where one exists (OpenAI Python SDK,
already used; the `ollama` Python client for the Ollama API), and with recorded payloads
in `docs/serving/`, as the current OpenAI pages are.
