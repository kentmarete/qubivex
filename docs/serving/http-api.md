# HTTP API

The reference for everything `qubivex serve` answers. The payloads below are real: they
were captured from the server running `Qwen/Qwen3-0.6B` in the Linux image, and from the
tiny fixture models where noted.

Related: [OpenAI compatibility](openai-compatibility.md) (what an OpenAI client can and
cannot use), [Streaming](streaming.md), [Scheduler](scheduler.md). The design record is
[ADR-013](../decisions/ADR-013-SERVING-GATEWAY.md).

## Endpoints

| Method and path | Auth | Purpose |
|---|---|---|
| `GET /health` | none | Liveness |
| `GET /metrics` | none | Prometheus metrics |
| `GET /v1/models` | API key | The loaded models |
| `GET /v1/models/{id}` | API key | One loaded model |
| `POST /v1/chat/completions` | API key | Chat completion, streaming or not |
| `POST /v1/completions` | API key | Text completion, streaming or not |
| `GET /v1/system` | API key | Hardware, memory and queue state |
| `POST /v1/context/inspect` | API key | What context management would do to a request, without generating ([Context management](context-management.md)) |
| `POST /v1/models/load` | admin token | Load a model |
| `POST /v1/models/unload` | admin token | Unload a model |
| `POST /v1/admin/models`, `DELETE /v1/admin/models/{id}` | admin token | Aliases of load and unload |
| `GET /v1/admin/models/installed` | admin token | Models on disk under `--dir`, and which are loaded |
| `GET /v1/admin/models/search?q=&limit=` | admin token | Search a model hub (with downloads on) |
| `POST /v1/admin/models/pull` | admin token | Download and install a model, as a job |
| `GET /v1/admin/jobs`, `GET /v1/admin/jobs/{id}` | admin token | Download jobs |
| `DELETE /v1/admin/jobs/{id}`, `POST /v1/admin/jobs/{id}/retry` | admin token | Cancel or retry one |

Anything else answers 404 with code `not_found`.

## Authentication

With an API key set (`--api-key` or `QUBIVEX_API_KEY`), every `/v1` request needs
`Authorization: Bearer <key>`. Without one the request gets:

```http
HTTP/1.1 401 Unauthorized
content-type: application/json
x-request-id: req-6af2c353-6

{"error":{"code":"invalid_api_key","message":"missing or invalid bearer token","param":null,"type":"authentication_error"}}
```

**Named keys** (`[[serve.client_keys]]` in the config file, made with `qubivex keys new`)
are accepted the same way, alongside the shared key or instead of it. A named key makes the
request its client's: the rate limit counts that key, fairness counts that client whatever
`user` says, and the prefix-cache scope is the client plus `user`. A bearer that is no key
counts against its address's allowance.

With no API key set, `/v1` is open, which the server allows only on loopback unless
started with `--allow-unauthenticated`.

The load and unload endpoints take the admin token (`--admin-token` or
`QUBIVEX_ADMIN_TOKEN`) as the bearer token instead. With no admin token set they answer
403 with code `admin_disabled`.

`/health` and `/metrics` never need a key and are never rate limited.

## Common behaviour

- Every response carries an `x-request-id` header. The same id is in the server's log
  lines for the request and in completion ids (`chatcmpl-<request id>`).
- Request bodies are JSON, at most `--max-body-bytes` (4 MiB). A larger one gets 413.
- Errors use OpenAI's shape and never contain file paths. See [Errors](#errors).
- Fields the server does not know are ignored. Fields it knows and cannot honour are
  refused with a 400 naming the field, never silently dropped.

## `GET /health`

```sh
curl -s http://127.0.0.1:8080/health
```

```json
{"models":1,"status":"ok"}
```

`models` is the number of loaded models. The port opens only after the models named at
start have loaded.

## `GET /v1/models`

```sh
curl -s http://127.0.0.1:8080/v1/models -H "Authorization: Bearer $KEY"
```

```json
{"data":[{"created":1794294609,"id":"olmoe-tiny","object":"model","owned_by":"qubivex"}],"object":"list"}
```

`GET /v1/models/{id}` returns one of those objects, or 404 `model_not_found`.

## `POST /v1/chat/completions`

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "system", "content": "Answer in one short sentence."},
      {"role": "user", "content": "What is the capital of France? /no_think"}
    ],
    "max_tokens": 40,
    "temperature": 0
  }'
```

```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "<think>\n\n</think>\n\nThe capital of France is Paris.",
        "role": "assistant"
      }
    }
  ],
  "created": 1794294981,
  "id": "chatcmpl-req-6af2c4c5-1",
  "model": "qwen3-0.6b",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 11,
    "prompt_tokens": 30,
    "prompt_tokens_details": {"cached_tokens": 0},
    "total_tokens": 41
  }
}
```

The messages are rendered with the model's own chat template. The reply is the model's
raw text: the server does not separate a reasoning block such as `<think>` from the
answer.

### Request fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `model` | string | required | The id of a loaded model |
| `messages` | array | required | Roles `system`, `developer` (treated as `system`), `user`, `assistant`. `content` is a string or a list of `{"type":"text","text":...}` parts |
| `max_tokens` or `max_completion_tokens` | integer | `--default-max-tokens` (512) | Most tokens to generate. Above `--max-generation-tokens` is a 400; past the room the prompt leaves in the context, generation ends when the context is full (`finish_reason` `length`) |
| `temperature` | number | 1.0 | 0 to 2. 0 is greedy |
| `top_p` | number | 1.0 | In (0, 1] |
| `top_k` | integer | 0 (off) | Keep the K likeliest tokens before `top_p`. Not an OpenAI field |
| `seed` | integer | 0 | The same seed, prompt and settings give the same text |
| `stop` | string or array | none | Up to 4 non-empty strings. Generation ends before the first one that appears, and it is not returned |
| `stream` | boolean | false | Server-sent events. See [Streaming](streaming.md) |
| `stream_options.include_usage` | boolean | false | Add a final chunk with `usage` to a stream |
| `user` | string | none | The client this request is for. Used for fairness and as the prefix-cache scope; with a named key, fairness counts the key's client and `user` only separates scopes within it |
| `priority` | string | `normal` | `low`, `normal` or `high`. Not an OpenAI field. See [Scheduler](scheduler.md) |
| `timeout` | number | none | Seconds this request may take, queue included. Not an OpenAI field |

### `finish_reason`

| Value | Meaning |
|---|---|
| `stop` | The model ended its turn, or a `stop` string appeared |
| `length` | `max_tokens`, the context, the request's `timeout` or `--request-timeout` was reached |

### `usage`

`prompt_tokens` counts the prompt after the chat template.
`prompt_tokens_details.cached_tokens` is how many of them were not run again because a
retained prefix covered them (with `--prefix-cache-mb`). The second turn of a
conversation, sent with the same `user`:

```json
"usage": {"completion_tokens": 11, "prompt_tokens": 78, "prompt_tokens_details": {"cached_tokens": 53}, "total_tokens": 89}
```

## `POST /v1/completions`

```sh
curl -s http://127.0.0.1:8080/v1/completions \
  -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{"model":"qwen3-0.6b","prompt":"The capital of France is","max_tokens":8,"temperature":0}'
```

```json
{
  "choices": [
    {"finish_reason": "length", "index": 0, "logprobs": null, "text": " Paris. The capital of Italy is Rome"}
  ],
  "created": 1794294983,
  "id": "cmpl-req-6af2c4c7-2",
  "model": "qwen3-0.6b",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 8,
    "prompt_tokens": 5,
    "prompt_tokens_details": {"cached_tokens": 0},
    "total_tokens": 13
  }
}
```

`prompt` is one string (or a list holding one string). No chat template is applied. The
other fields are those of chat completions, without `messages`.

## `GET /v1/system`

The hardware profile the server sees and each model's live state. The `hardware` object
is what `qubivex system --json` prints. The rest, from a server with one fixture model:

```json
{
  "capabilities": {"admin": true, "model_management": true, "model_downloads": true},
  "generations_active": 0,
  "limits": {"default_max_tokens": 512, "load_context": null, "max_generation_tokens": null, "max_prompt_tokens": null},
  "model_states": [{"id": "olmoe-tiny", "state": "loaded"}],
  "models": [
    {
      "architecture": "transformer_moe",
      "expert_cache_bytes": 196608,
      "generation_slots": 4,
      "generation_slots_free": 4,
      "id": "olmoe-tiny",
      "kv_reserved_positions": 0,
      "ledger_bytes": {"dense": 134272, "expert_cache": 196608, "kv": 524288},
      "max_context": 4096,
      "placement": "cpu",
      "plan_budget_bytes": 2827358622,
      "prefix_cache": {"entries": 1, "evicted": 0, "hits": 0, "lookups": 3, "positions": 256, "tokens_reused": 0},
      "queue_depth": 0,
      "queue_limit": 32
    }
  ],
  "models_draining": 0,
  "object": "system"
}
```

| Field | Meaning |
|---|---|
| `capabilities` | `admin`: the admin endpoints are on. `model_management`: `GET /v1/admin/models/installed` is on (with the admin token). `model_downloads`: the download routes below are on |
| `limits` | `--default-max-tokens`, `--max-generation-tokens`, `--max-prompt-tokens` and `--context` (`load_context`); `null` where unset, so the context limits it |
| `models[].max_context` | The context the model was loaded with, in tokens |
| `models[].context_limits` | Where `max_context` comes from: the model's own limit, the configured one, what the memory plan holds, and the effective (least) one |
| `context_management` | Whether long requests are prepared to fit, the note share, where checkpoints are kept (`memory` or `files`) and the store's counters. See [Context management](context-management.md) |
| `model_states[].state` | `loading`, `loaded` or `failed` |
| `plan_budget_bytes` | The RAM the memory plan may use for this model |
| `ledger_bytes` | What is allocated now, by owner |
| `kv_reserved_positions` | KV positions reserved by running conversations |
| `generation_slots`, `generation_slots_free` | `--max-concurrent` and how many are free |
| `queue_depth`, `queue_limit` | Requests waiting, and `--max-queue` |
| `models_draining` | Unloaded models whose last generations are still finishing |

In a container, `hardware.memory.cgroup_limit` is `{"known":{"limited":2147483648}}`,
and the plan is made inside it.

## Model load and unload

```sh
curl -s http://127.0.0.1:8080/v1/models/load \
  -H "Authorization: Bearer $ADMIN" -H 'content-type: application/json' \
  -d '{"model":"qwen3-moe-tiny"}'
```

```json
{"created":1794294611,"id":"qwen3-moe-tiny","object":"model","owned_by":"qubivex"}
```

The first load answers 201, and loading a loaded model answers 200. The id is looked up
in the server's `--dir` directories. The request returns when the model has loaded.
Concurrent loads of one model share one load, and a generation that names a model still
loading waits for it. A second id for a directory that is already loaded is a 409
`conflict`.

`context` (optional, a positive number of tokens) sets this load's context limit in place
of `--context`; the model's own limit still caps it, and `/v1/system` reports the result
as `max_context`. Anything else is a 400 with `param` `context`. A model already loaded,
or loading, with a different context is a 409 `conflict`: unload it first.

```sh
curl -s http://127.0.0.1:8080/v1/models/load \
  -H "Authorization: Bearer $ADMIN" -H 'content-type: application/json' \
  -d '{"model":"qwen3-moe-tiny","context":8192}'
```

```sh
curl -s http://127.0.0.1:8080/v1/models/unload \
  -H "Authorization: Bearer $ADMIN" -H 'content-type: application/json' \
  -d '{"model":"qwen3-moe-tiny"}'
```

```json
{"deleted":true,"draining":0,"id":"qwen3-moe-tiny","object":"model"}
```

Unloading closes the model's queue, so waiting requests get a 503, and lets running
generations finish. `draining` is how many were running.

## Installed models

`GET /v1/admin/models/installed` lists every model under the `--dir` directories, the ids
`POST /v1/models/load` accepts, loaded or not. `/v1/models` lists only loaded ones.

```sh
curl -s http://127.0.0.1:8080/v1/admin/models/installed -H "Authorization: Bearer $ADMIN"
```

```jsonc
{"object":"list","data":[{
  "id":"qwen3-0.6b","name":"Qwen3 0.6B","dir":"qwen3-0.6b",
  "status":"ok","error":null,"install_state":"installed",
  "loaded":false,"serve_ids":[],
  "architecture":"transformer_dense 1.0.0","format":"safetensors","quantization":"bf16",
  "source":{"provider":"huggingface","repository":"Qwen/Qwen3-0.6B","revision":"…"},
  "layers":28,"hidden":1024,"vocab":151936,"experts":0,"experts_per_token":0,
  "max_context":40960,"bytes":{"total":1503238553,"dense":1503238553,"routed_experts":0}
}]}
```

| Field | |
|---|---|
| `status` | `ok`, or `error` when the model does not validate; `error` then says why, and the model fields are absent |
| `dir` | The directory's name inside its models directory; no path is returned |
| `loaded` | `true` if any id serves it, else `"loading"`, `"failed"` or `false` |
| `serve_ids` | The ids it is loaded under |
| `install_state` | `installed`: only finished installs are on disk; failed downloads are jobs |

## Model downloads

With an admin token, `qubivex serve` can search Hugging Face and download models into the
first `--dir` directory, as `qubivex models pull` does. `/v1/system` then says
`"model_downloads": true`. The gateway library has no network code of its own: an
application embedding it supplies a `ModelSource`, and without one these routes answer 404
`downloads_disabled`. Searches and downloads use `HF_ENDPOINT` and `HF_TOKEN` (or the token
`hf auth login` saved) from the **server's** environment; clients never send a hub token.

### `GET /v1/admin/models/search`

```sh
curl -s "http://127.0.0.1:8080/v1/admin/models/search?q=qwen3-4b&limit=2" \
  -H "Authorization: Bearer $ADMIN"
```

```json
{"object":"list","query":"qwen3-4b","data":[
  {"repository":"Qwen/Qwen3-4B-Instruct-2507","downloads":2736684,"likes":1060,
   "formats":["safetensors"],"gated":false,"installed_as":["qwen3-4b-instruct-2507"]}]}
```

`q` is required; `limit` is 1 to 100 (default 20). `installed_as` lists the ids already
installed from that repository. A hub that fails or cannot be reached is a 502
`upstream_error`.

### `POST /v1/admin/models/pull`

```sh
curl -s http://127.0.0.1:8080/v1/admin/models/pull \
  -H "Authorization: Bearer $ADMIN" -H 'content-type: application/json' \
  -d '{"repository":"Qwen/Qwen3-4B-Instruct-2507"}'
```

The body mirrors `qubivex models pull`: `repository` (`owner/name`, required), and
optionally `revision`, `files` (paths in the repository; needed when it holds several
weight files), `id`, `name` and `set` (config values the checkpoint lacks, such as
`{"norm_topk_prob": false}` for an OLMoE GGUF). It answers **202** with the job and a
`location` header. The download runs on the server: closing the client does not stop it,
and the model is not loaded when it finishes.

| Refusal | Status, code |
|---|---|
| A pull of the same repository (or the same `id`) is running | 409 `conflict` |
| Two pulls are already running | 429 `too_many_requests` |
| A malformed body | 400 `invalid_request` |

### Jobs

`GET /v1/admin/jobs/{id}` returns the job; `GET /v1/admin/jobs` lists them, newest first.
Finished jobs are kept for an hour.

```json
{"object":"job","id":"job-6920b3a1-1","kind":"pull","status":"running",
 "model":null,"repository":"Qwen/Qwen3-4B-Instruct-2507","revision":"cdbee75…",
 "created_at":1795086241,"started_at":1795086241,"finished_at":null,
 "progress":{"bytes_done":734003200,"bytes_total":8053063680,"bytes_resumed":0,
   "rate_bps":27262976,"eta_seconds":268,
   "files":[{"name":"model-00001-of-00003.safetensors","done":734003200,"total":3957900840}],
   "retrying":null},
 "result":null,"error":null,"cancellable":true}
```

| Field | Meaning |
|---|---|
| `status` | `running`, `validating` (every file arrived; the model is being checked), `succeeded`, `failed` or `cancelled` |
| `model` | The id it installs as, once known; `result.id` when it succeeded. Load it with `POST /v1/models/load` |
| `progress.bytes_resumed` | Bytes already present from an earlier attempt, left out of `rate_bps` |
| `progress.retrying` | `{file, attempt, delay_s, reason}` while a transfer waits to retry |
| `error` | `{code, message}` when it failed or was cancelled. Messages hold no path, URL or token |

`DELETE /v1/admin/jobs/{id}` asks a running job to stop and answers 202; it reads
`cancelled` once it has. What arrived is kept, so `POST /v1/admin/jobs/{id}/retry` (on a
failed or cancelled job) starts a new job that continues it. Cancelling a finished job, or
retrying one that is not failed or cancelled, is a 409. At shutdown, running jobs are
cancelled the same way.

## `GET /metrics`

Prometheus text format, with `# HELP` and `# TYPE` lines. Per model (`model` label)
unless noted. `--no-metrics` turns the endpoint into a 404.

| What | Series |
|---|---|
| Time to first token, total latency, scheduler wait | `qubivex_time_to_first_token_seconds`, `qubivex_generation_duration_seconds`, `qubivex_queue_wait_seconds` (histograms, from the request's arrival) |
| Inter-token latency | `qubivex_inter_token_seconds` (histogram) |
| Prompt and generation speed | `qubivex_prompt_tokens_per_second`, `qubivex_generation_tokens_per_second` (histograms, one observation per generation) |
| Model load time | `qubivex_model_load_seconds` (histogram; loads through the admin endpoint), `qubivex_model_loads_total` (server-wide) |
| Memory | `qubivex_memory_{dense,expert_cache,kv,activations}_bytes`, `qubivex_memory_budget_bytes`, `qubivex_kv_reserved_positions`; for the process, `qubivex_process_resident_bytes` and `qubivex_process_peak_resident_bytes` |
| Expert cache | `qubivex_expert_cache_{hits,loads,evictions}_total`, `qubivex_expert_cache_bytes` |
| Discrete GPU (ADR-018; only with one, `--backend vulkan` or `sim`) | `qubivex_vram_{dense,expert_cache,used,peak,capacity,budget}_bytes`, `qubivex_device_expert_{hits,copies,evictions}_total`, `qubivex_device_{upload,activation}_bytes_total`, `qubivex_device_transfer_seconds_total` |
| Disk | `qubivex_expert_read_bytes_total`, `qubivex_expert_load_wait_seconds_total`, `qubivex_expert_read_{retries,timeouts}_total` |
| Serving | `qubivex_generations_active` (server-wide), `qubivex_queue_depth`, `qubivex_generation_slots_free`, `qubivex_queue_{granted,rejected}_total`, `qubivex_batch_{passes,steps}_total` |
| Outcomes, server-wide | `qubivex_generations_{cancelled,timed_out,failed}_total` |
| Requests and tokens, server-wide | `qubivex_requests_total{endpoint,status}`, `qubivex_http_request_duration_seconds{endpoint}`, `qubivex_prompt_tokens_total`, `qubivex_generated_tokens_total` |
| Retained KV | `qubivex_prefix_cache_{lookups,hits,tokens_reused,evictions}_total`, `qubivex_prefix_cache_positions` |

An excerpt:

```text
# HELP qubivex_requests_total Requests answered, by endpoint and status.
# TYPE qubivex_requests_total counter
qubivex_requests_total{endpoint="chat_completions",status="200"} 2
qubivex_requests_total{endpoint="chat_completions",status="400"} 2
qubivex_requests_total{endpoint="models",status="401"} 1
# HELP qubivex_generated_tokens_total Tokens generated; its rate is the aggregate tokens per second.
# TYPE qubivex_generated_tokens_total counter
qubivex_generated_tokens_total 13
# HELP qubivex_time_to_first_token_seconds From a request's arrival to its first generated token.
# TYPE qubivex_time_to_first_token_seconds histogram
qubivex_time_to_first_token_seconds_sum{model="olmoe-tiny"} 0.060438375
qubivex_time_to_first_token_seconds_count{model="olmoe-tiny"} 3
# HELP qubivex_expert_cache_hits_total Expert acquires served from memory.
# TYPE qubivex_expert_cache_hits_total counter
qubivex_expert_cache_hits_total{model="olmoe-tiny"} 64
```

Dashboards and alerts built on these are in
[Production operations](../deployment/production-operations.md#monitoring).

## Errors

Every error is JSON in OpenAI's shape:

```json
{"error":{"code":"model_not_found","message":"the model \"nope\" is not loaded","param":"model","type":"invalid_request_error"}}
```

| Status | `code` | `type` | When |
|---|---|---|---|
| 400 | `invalid_request` | `invalid_request_error` | Malformed JSON, a bad value, a field the server cannot honour (`param` names it), or a prompt that leaves no room in the context |
| 400 | `prompt_too_long` | `invalid_request_error` | The prompt exceeds `--max-prompt-tokens` |
| 400 | `context_length_exceeded` | `invalid_request_error` | A request longer than the context that cannot be prepared to fit: its instructions alone are too long, or the current message cannot be cut far enough. `param` is `messages` or `prompt`. See [Context management](context-management.md) |
| 401 | `invalid_api_key` | `authentication_error` | Missing or wrong bearer token |
| 403 | `admin_disabled` | `permission_error` | An admin endpoint with no admin token configured |
| 403 | | | A CORS preflight from an origin that is not allowed |
| 404 | `model_not_found` | `invalid_request_error` | The model is not loaded |
| 404 | `not_found` | `invalid_request_error` | No such endpoint |
| 409 | `conflict` | `invalid_request_error` | Loading a second id for a directory already loaded |
| 413 | `request_too_large` | `invalid_request_error` | The body exceeds `--max-body-bytes` |
| 429 | `rate_limit_exceeded` | `rate_limit_error` | The client address used its `--rate-limit` allowance. A `retry-after` header says when to try again |
| 429 | `too_many_requests` | `rate_limit_error` | The model's queue is full, or the request waited `--queue-timeout` for a slot |
| 500 | `internal_error` | `server_error` | The request failed. The message names only the request id; the log has the detail |
| 503 | `overloaded` | `server_error` | Memory cannot take the request now, the model is being unloaded, or the server is stopping |
| 504 | `deadline_exceeded` | `server_error` | The request's own `timeout` passed before it could start |

A failure after a stream has started arrives as an event in the stream, because the 200
status has already been sent. See [Streaming](streaming.md#errors-in-a-stream).

## CORS

Off by default: the server sends no CORS headers, so browsers refuse cross-origin calls.
`--cors-origin https://app.example.com` (repeatable) or `--cors-origin '*'` allows
origins. A preflight from an allowed origin is answered:

```http
HTTP/1.1 204 No Content
access-control-allow-origin: https://app.example.com
access-control-allow-methods: GET, POST, DELETE
access-control-allow-headers: authorization, content-type, x-request-id
access-control-expose-headers: x-request-id, retry-after
access-control-max-age: 600
vary: origin
```

A preflight from any other origin gets a 403 with no body. Credentials (cookies) are
never allowed; the API key travels in the `Authorization` header.

## Rate limiting

`--rate-limit N` allows each client address `N` requests per minute on `/v1`, as a token
bucket: `--rate-limit-burst` (default `N`) requests at once after being idle, then one
every `60/N` seconds. An IPv6 client is counted by its /64. The server remembers at most
4096 clients. It counts the address that connects, so behind a reverse proxy every
client shares one allowance: limit at the proxy instead.

```http
HTTP/1.1 429 Too Many Requests
content-type: application/json
x-request-id: req-6af2c532-f
retry-after: 10

{"error":{"code":"rate_limit_exceeded","message":"too many requests from this client; retry after the time in retry-after","param":null,"type":"rate_limit_error"}}
```

## Server settings

Every setting can come from a flag, an environment variable (for three of them) or the
config file. The highest wins: flag, then environment, then file, then default.

| Flag | Default | Meaning |
|---|---|---|
| `--config FILE` | `$QUBIVEX_CONFIG` | A TOML file with a `[serve]` table |
| `--model ID\|DIR` | none | A model to load at start (repeatable) |
| `--dir DIR` | `$QUBIVEX_MODELS_DIRS`, then `~/.qubivex/models` | Where ids are looked up (repeatable) |
| `--listen ADDR` | `127.0.0.1:8080` | Beyond loopback needs an API key or `--allow-unauthenticated` |
| `--api-key KEY` | `$QUBIVEX_API_KEY` | The bearer token for `/v1` |
| `--admin-token T` | `$QUBIVEX_ADMIN_TOKEN` | Enables load and unload |
| `--allow-unauthenticated` | off | Serve beyond loopback with no key |
| `--rate-limit N`, `--rate-limit-burst N` | unlimited | Requests per minute per client address |
| `--cors-origin O` | none | Allowed browser origins (repeatable) |
| `--max-body-bytes N` | 4194304 | Largest request body |
| `--max-prompt-tokens N` | the context | Longest prompt, after the chat template |
| `--max-generation-tokens N` | the context | Largest `max_tokens` |
| `--default-max-tokens N` | 512 | When a request sets none |
| `--max-concurrent N` | 4 | Generations per model at once |
| `--max-queue N` | 32 | Requests per model that may wait |
| `--queue-timeout S` | 30 | Seconds a request waits to start before a 429 |
| `--memory-policy queue\|refuse` | `queue` | Wait for memory, or answer 503 |
| `--request-timeout S` | 600 | A generation then ends with `finish_reason: "length"` |
| `--stall-timeout S` | 30 | End a stream the client has stopped reading |
| `--shutdown-grace S` | 30 | How long running requests get at shutdown |
| `--prefix-cache-mb N` | 0 | Retained KV per model for prefix reuse |
| `--context N` | the model's | Context limit per conversation |
| `--threads N` | the performance cores | Compute threads per model |
| `--backend cpu\|metal`, `--metal-ops` | `cpu` | Compute backend (Metal on macOS only) |
| `--profile`, `--ram-fraction`, `--ram-budget`, `--expert-cache-mb` | `performance` | Memory planning, per model. See [Memory planner](../runtime/memory-planner.md) |
| `--io-threads N` | 4 | Threads reading experts from disk |
| `--log-format text\|json\|off` | `text` | The log on standard error |
| `--no-metrics` | off | `GET /metrics` answers 404 |
| `--no-context-management` | on | Refuse requests longer than the context (400) instead of preparing them to fit |
| `--context-note-share F` | 0.25 | Room kept for the note on older turns. See [Context management](context-management.md) |
| `--context-store DIR`, `--context-store-mb N`, `--context-store-days N` | memory, 64 MiB | Keep checkpoints of condensed turns in files that survive a restart |

In the config file each key is the flag's name with `_` for `-`. The repeatable flags
are lists named `models`, `dirs` and `cors_origins`, and a list given on the command
line replaces the file's. Unknown keys are errors. Everything is checked before a model
loads. A file holding `api_key` or `admin_token` must not be readable by other users
(`chmod 600`).

```toml
[serve]
listen = "0.0.0.0:8080"
models = ["qwen3-0.6b"]
api_key = "sk-local"
rate_limit = 120
cors_origins = ["https://app.example.com"]
max_prompt_tokens = 8192
queue_timeout = 30
```

## Not supported

- TLS. The server speaks plain HTTP; terminate TLS at a proxy.
- More than one API key, per-key limits, and per-user authentication.
- Embeddings, the Responses API, the Assistants API, files, images, audio, fine-tuning,
  moderation and batch endpoints.
- Tool calls, structured output, logprobs, `n > 1`: see
  [OpenAI compatibility](openai-compatibility.md).
- Changing settings while the server runs.
