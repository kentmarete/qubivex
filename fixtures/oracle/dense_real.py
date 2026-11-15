"""Reference greedy decode and logits of a real dense SafeTensors checkpoint, from
transformers in f32 (bf16 weights widened exactly).

    ~/.qubivex/oracle/bin/python -I fixtures/oracle/dense_real.py <model dir> \
        > fixtures/oracle/dense_real.json

    ~/.qubivex/oracle/bin/python -I fixtures/oracle/dense_real.py <model dir> --chat \
        > fixtures/oracle/qwen2_real.json

<model dir> is an official checkpoint (config.json, model.safetensors, tokenizer.json):
huggingface.co/Qwen/Qwen3-0.6B for dense_real.json, huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
(revision 7ae557604adf67be50417f59c2c2f167def9a775) for qwen2_real.json. For a fixed prompt: the prompt ids
from the tokenizer, 32 greedy tokens (full forward each step, no end-of-sequence stop),
and, for every prompt position, the TOP largest f32 logits (ids and bit patterns). With
--chat, also a conversation rendered by the checkpoint's chat template (its text and ids)
and CHAT_GREEDY greedy tokens after it, stopping at an end-of-sequence id. Checked by the
env-gated test crates/qubivex-runtime/tests/real_dense.rs.
"""
import json
import sys

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "The capital of France is"
GREEDY = 32
TOP = 32
CHAT = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Name three primary colours."},
]
CHAT_GREEDY = 24

torch.set_num_threads(4)
d = sys.argv[1]
tok = AutoTokenizer.from_pretrained(d)
ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
model, info = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32, output_loading_info=True)
assert not info["missing_keys"] and not info["mismatched_keys"], info
model.eval()
with torch.no_grad():
    logits = model(torch.tensor([ids])).logits[0]
    out = list(ids)
    for _ in range(GREEDY):
        out.append(int(model(torch.tensor([out])).logits[0, -1].argmax()))
chat = None
if "--chat" in sys.argv[2:]:
    rendered = tok.apply_chat_template(CHAT, tokenize=False, add_generation_prompt=True)
    chat_ids = tok(rendered, add_special_tokens=False)["input_ids"]
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    reply = []
    with torch.no_grad():
        for _ in range(CHAT_GREEDY):
            nxt = int(model(torch.tensor([chat_ids + reply])).logits[0, -1].argmax())
            reply.append(nxt)
            if nxt in eos:
                break
    chat = {"messages": CHAT, "rendered": rendered, "ids": chat_ids, "greedy": reply,
            "eos": sorted(eos), "text": tok.decode(reply, skip_special_tokens=True)}
top = torch.topk(logits, TOP, dim=-1)
top_ids = top.indices.tolist()
top_bits = [v.contiguous().numpy().view(np.uint32).tolist() for v in top.values]
json.dump({
    "generator": "fixtures/oracle/dense_real.py",
    "transformers": transformers.__version__,
    "torch": torch.__version__,
    "prompt": PROMPT,
    "prompt_ids": ids,
    "greedy": out[len(ids):],
    "vocab": logits.shape[1],
    "top_ids": top_ids,
    "top_logits": top_bits,
} | ({"chat": chat} if chat else {}), sys.stdout, separators=(",", ":"))
