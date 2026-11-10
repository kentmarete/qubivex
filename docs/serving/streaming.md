# Streaming

With `"stream": true`, `/v1/chat/completions` and `/v1/completions` answer with
server-sent events (SSE), in OpenAI's format: one `data:` line per chunk, a blank line
between events, and `data: [DONE]` last.

## A chat stream

```sh
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "Say hi. /no_think"}],
    "max_tokens": 12,
    "temperature": 0,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

The response, captured from the server (some middle chunks left out):

```text
data: {"choices":[{"delta":{"content":"","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1794294984,"id":"chatcmpl-req-6af2c4c8-3","model":"qwen3-0.6b","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"<think>"},"finish_reason":null,"index":0,"logprobs":null}],"created":1794294984,"id":"chatcmpl-req-6af2c4c8-3","model":"qwen3-0.6b","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null,"index":0,"logprobs":null}],"created":1794294984,"id":"chatcmpl-req-6af2c4c8-3","model":"qwen3-0.6b","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"!"},"finish_reason":null,"index":0,"logprobs":null}],"created":1794294984,"id":"chatcmpl-req-6af2c4c8-3","model":"qwen3-0.6b","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{},"finish_reason":"length","index":0,"logprobs":null}],"created":1794294984,"id":"chatcmpl-req-6af2c4c8-3","model":"qwen3-0.6b","object":"chat.completion.chunk"}

data: {"choices":[],"created":1794294984,"id":"chatcmpl-req-6af2c4c8-3","model":"qwen3-0.6b","object":"chat.completion.chunk","usage":{"completion_tokens":12,"prompt_tokens":15,"prompt_tokens_details":{"cached_tokens":0},"total_tokens":27}}

data: [DONE]
```

The order is always:

1. For chat, a first chunk with `delta.role` `"assistant"` and empty content.
2. One chunk per piece of text, in `delta.content`.
3. A chunk with an empty `delta` and the `finish_reason`.
4. With `stream_options.include_usage`, a chunk with empty `choices` and the `usage`.
5. `data: [DONE]`.

A completions stream is the same without step 1, and with the text in
`choices[0].text`:

```text
data: {"choices":[{"finish_reason":null,"index":0,"logprobs":null,"text":" Paris"}],"created":1794295182,"id":"cmpl-req-6af2c58e-1","model":"qwen3-0.6b","object":"text_completion"}
```

## What a chunk holds

A chunk is usually one token's text, and not always:

- A token that ends in the middle of a multi-byte character is held until the character
  is complete, so every chunk is valid UTF-8.
- With `stop` strings, text that could still turn out to be the start of a stop string
  is held back until it is known not to be. The stop string itself is never sent.

Clients should concatenate `delta.content` and not assume one chunk per token. The
token count is in `usage`.

## When the response starts

The HTTP status is sent once the request has been admitted: validated, through the
queue, and given its memory. So everything that can refuse a request arrives as an
ordinary error response with its own status, before any event:

| Before the stream | Status |
|---|---|
| A bad request or an unsupported field | 400 |
| The queue is full, or the wait for a slot timed out | 429 |
| Memory cannot take it, or the model is unloading | 503 |
| The request's `timeout` passed in the queue | 504 |

The time from the request to the first chunk with text is the queue wait plus the
prefill of the prompt. On a busy server a streaming client should allow for
`--queue-timeout` (30 s) before the first byte.

## Errors in a stream

After the 200 has been sent, a failure arrives as an event holding the usual error
object, followed by `[DONE]`:

```text
data: {"error":{"code":"overloaded","message":"the server is shutting down","param":null,"type":"server_error"}}

data: [DONE]
```

That one was captured by stopping the container during a generation. It happens when the
server stops while the stream is running and the shutdown grace runs out, or when the generation fails (for example an expert that cannot be read:
`internal_error`). A client must therefore check each event for an `error` key. The
official OpenAI Python SDK raises an exception for such an event.

A stream that ends without `[DONE]` means the connection broke.

## Ending a stream early

**Close the connection** to cancel. The server notices at the next token or, during
prefill, at the next chunk of the prompt, stops the generation, and frees its slot and
memory. Nothing else is needed, and there is no cancel endpoint.

A cancelled generation is counted in `qubivex_generations_cancelled_total`. With a
prefix cache, the KV for the part that ran is kept, so resending the same conversation
continues from it.

## Time limits

| Limit | Default | Effect on a stream |
|---|---|---|
| The request's `timeout` field | none | Covers the queue and the generation. Passing it while generating ends the stream normally with `finish_reason: "length"` |
| `--request-timeout` | 600 s | The same, for every request |
| `--stall-timeout` | 30 s | A client that stops reading for this long is dropped and its generation cancelled. A slow reader that keeps reading is not affected |
| `--shutdown-grace` | 30 s | At shutdown, a running stream gets this long to finish, then the error event above |

The server sends no keep-alive comments. During a long prefill nothing is sent between
the response headers and the first text chunk, so a proxy's idle timeout must be longer
than the longest prefill you expect. An nginx configuration that streams correctly is in
[Linux deployment](../deployment/linux-deployment.md#tls-and-the-public-network): the
important line is `proxy_buffering off`.

## Backpressure

Each stream has a bounded buffer between the generation and the connection. A client
that reads slowly slows its own generation and no one else's, and holds its slot while
it does. That is why the stall timeout exists.

## Not supported

- **Resuming a stream.** There is no `Last-Event-ID`; a broken stream is a new request
  (which the prefix cache makes cheap).
- **WebSockets.**
- **Tool-call deltas, logprobs and `refusal` deltas.**
- **Keep-alive comments** during prefill.
- **A separate reasoning stream.** A model's `<think>` text arrives as ordinary content.
