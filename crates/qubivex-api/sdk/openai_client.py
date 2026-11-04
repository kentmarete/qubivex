"""The official OpenAI Python SDK against a running `qubivex serve` (Phase 9 exit check).

    qubivex serve --model <id> [--model <id> ...] --api-key sk-local &
    ~/.qubivex/oracle/bin/python -I crates/qubivex-api/sdk/openai_client.py http://127.0.0.1:8080/v1 sk-local

Lists the models, then for each: a chat completion, the same chat streamed (the deltas
must join to the same text), a text completion, two concurrent chats with different
prompts (each must equal its sequential result), and an error the SDK raises as
`BadRequestError`. Greedy sampling (temperature 0), so results are reproducible.
"""
import concurrent.futures
import sys

import openai
from openai import OpenAI

base, key = sys.argv[1], sys.argv[2]
client = OpenAI(base_url=base, api_key=key)
models = [m.id for m in client.models.list().data]
print("models:", models)
assert models, "no models served"


def chat(model, text, **kw):
    return client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": text}],
        max_tokens=24, temperature=0, **kw)


for model in models:
    r = chat(model, "Name three primary colors.")
    text = r.choices[0].message.content
    print(f"[{model}] chat ({r.choices[0].finish_reason}, {r.usage.completion_tokens} tokens): {text!r}")
    assert r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens

    stream = chat(model, "Name three primary colors.", stream=True,
                  stream_options={"include_usage": True})
    parts, usage = [], None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)
    assert "".join(parts) == text, ("streamed differs", "".join(parts), text)
    assert usage is not None and usage.completion_tokens == r.usage.completion_tokens
    print(f"[{model}] streamed: {len(parts)} deltas, same text, usage {usage.completion_tokens}")

    c = client.completions.create(model=model, prompt="The capital of France is",
                                  max_tokens=8, temperature=0)
    print(f"[{model}] completion: {c.choices[0].text!r}")

    prompts = ["Count from one to five.", "What is the opposite of hot?"]
    alone = [chat(model, p).choices[0].message.content for p in prompts]
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        together = list(pool.map(lambda p: chat(model, p).choices[0].message.content, prompts))
    assert together == alone, (together, alone)
    print(f"[{model}] two concurrent chats equal their sequential results")

    try:
        client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}], n=2)
        raise AssertionError("n=2 was accepted")
    except openai.BadRequestError as e:
        print(f"[{model}] n=2 refused: {e.body['message']!r}")

try:
    client.chat.completions.create(model="no-such-model", messages=[{"role": "user", "content": "hi"}])
    raise AssertionError("an unknown model was accepted")
except openai.NotFoundError as e:
    print("unknown model:", e.status_code, e.body["code"])
print("OK")
