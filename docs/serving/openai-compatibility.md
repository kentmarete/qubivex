# OpenAI compatibility

`qubivex serve` implements the part of the OpenAI API that text generation needs, so an
OpenAI client works by changing its base URL. This page says exactly which part, and
what happens to the rest.

The rule: **a field the server cannot honour is refused with a 400 that names it, never
silently ignored.** A client finds out at the first request, not from wrong output.
Fields the server has never heard of are ignored.

## Using an OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-local")

reply = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "Hello"}],
)
print(reply.choices[0].message.content)

stream = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

The QUBIVEX fields that OpenAI does not have go through `extra_body`:

```python
client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"top_k": 40, "priority": "high", "timeout": 20},
)
```

With no API key configured on the server, pass any non-empty string as `api_key`: the
client requires one and the server ignores it.

```js
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://127.0.0.1:8080/v1", apiKey: "sk-local" });
```

The official Python SDK is the one tested against real models here (ADR-013). Other
clients work to the extent that they send only what the tables below accept.

## Endpoints

| OpenAI endpoint | Status |
|---|---|
| `POST /v1/chat/completions` | Supported |
| `POST /v1/completions` | Supported |
| `GET /v1/models`, `GET /v1/models/{id}` | Supported. Lists the loaded models |
| `POST /v1/embeddings` | Not supported (404) |
| `POST /v1/responses` | Not supported (404) |
| Assistants, threads, files, batches, fine-tuning, images, audio, moderations | Not supported (404) |

## Chat completions: request fields

| Field | Status | Notes |
|---|---|---|
| `model` | Supported | Must be a loaded model's id. There are no aliases: `gpt-4o` is a 404 unless a model has that id |
| `messages` | Supported | Roles `system`, `developer`, `user`, `assistant`. Text content only |
| `messages[].content` as parts | Text parts only | An `image_url`, `input_audio` or `file` part is a 400 |
| `messages[].role` `tool` or `function` | Refused | 400 |
| `messages[].name`, `tool_calls` on a message | Ignored | |
| `max_tokens`, `max_completion_tokens` | Supported | Either name |
| `temperature` | Supported | 0 to 2, default 1 |
| `top_p` | Supported | (0, 1], default 1 |
| `seed` | Supported | Deterministic: the same request gives the same text |
| `stop` | Supported | Up to 4 strings |
| `stream`, `stream_options.include_usage` | Supported | See [Streaming](streaming.md) |
| `user` | Supported | Used for fairness and to separate prefix caches; under a named key, within that key's client |
| `n` | Only 1 | Any other value is a 400 |
| `tools`, `tool_choice` (other than `"none"`) | Refused | 400: "tool calls are not supported" |
| `response_format` | Only `{"type":"text"}` | `json_object` and `json_schema` are a 400 |
| `logprobs`, `top_logprobs` | Refused | `logprobs: true` is a 400. `false` or absent is fine |
| `presence_penalty`, `frequency_penalty` | Only 0 | A non-zero value is a 400 |
| `logit_bias` | Only empty | A non-empty map is a 400 |
| `parallel_tool_calls`, `reasoning_effort`, `service_tier`, `store`, `metadata`, `prediction`, `modalities`, `audio` | Ignored | The server does not know these fields |

Fields QUBIVEX adds: `top_k`, `priority` and `timeout`
([HTTP API](http-api.md#request-fields)).

## Completions: request fields

As chat completions, with these differences:

| Field | Status | Notes |
|---|---|---|
| `prompt` | One string | A list of one string works. Several prompts, or token arrays, are a 400 |
| `echo` | Refused | `true` is a 400 |
| `suffix` | Refused | A non-empty value is a 400 |
| `best_of` | Only 1 | |

## Responses

The response objects have OpenAI's shape and names: `chat.completion`,
`chat.completion.chunk`, `text_completion`, `list`, `model`.

| Field | Notes |
|---|---|
| `id` | `chatcmpl-` or `cmpl-` followed by the request id |
| `choices[0].message.content` | The model's raw text. A reasoning block (`<think>...</think>`) is not split into a separate `reasoning_content` field |
| `choices[0].finish_reason` | `stop` or `length`. Never `tool_calls` or `content_filter` |
| `choices[0].logprobs` | Always `null` |
| `usage` | `prompt_tokens`, `completion_tokens`, `total_tokens`, and `prompt_tokens_details.cached_tokens` |
| `system_fingerprint`, `service_tier` | Absent |
| `message.refusal`, `message.tool_calls`, `message.annotations` | Absent |

## Errors

Errors use OpenAI's body, `{"error":{"message","type","code","param"}}`, and OpenAI's
status codes where one exists, so SDK exception types map as expected:
`AuthenticationError` for 401, `NotFoundError` for an unknown model, `RateLimitError`
for 429, `BadRequestError` for 400. The full table is in
[HTTP API](http-api.md#errors).

Two differences from OpenAI matter to retry logic:

- A 429 `too_many_requests` means this server's queue is full, not that an account
  quota ran out. Retrying after a short wait is right. The official SDKs already retry
  429 and 5xx with backoff.
- A 429 `rate_limit_exceeded` carries `retry-after` in seconds.

## Behaviour differences

- **Sampling is deterministic by default.** `seed` defaults to 0, so two identical
  requests give identical text even at `temperature` 1. Send a different `seed` per
  request for variety.
- **Chat templates come from the model.** The server renders `messages` with the
  template in the model's files. A model without a chat template cannot serve chat
  completions; use `/v1/completions`.
- **The context is the model's,** or less when the memory plan shortened it to fit. A
  prompt that leaves no room to generate is a 400, not a truncation.
- **`max_tokens` defaults to 512** (`--default-max-tokens`), where OpenAI's default is
  the rest of the context.

## Not supported

Stated once, in full:

- Tool and function calling.
- Structured output: JSON mode and JSON schema.
- Logprobs, logit bias, presence and frequency penalties.
- More than one choice (`n`, `best_of`).
- Images, audio and files as input or output.
- Embeddings.
- The Responses, Assistants and Batch APIs.
- Separate reasoning content.
- Token-array prompts, `echo` and `suffix` on completions.
