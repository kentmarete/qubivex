"""Reference greedy decode and logits of a real dense SafeTensors checkpoint, from
transformers in f32 (bf16 weights widened exactly).

    ~/.qubivex/oracle/bin/python -I fixtures/oracle/dense_real.py <model dir> \
        > fixtures/oracle/dense_real.json

<model dir> is the official checkpoint (config.json, model.safetensors,
tokenizer.json) of huggingface.co/Qwen/Qwen3-0.6B. For a fixed prompt: the prompt ids
from the tokenizer, 32 greedy tokens (full forward each step, no end-of-sequence stop),
and, for every prompt position, the TOP largest f32 logits (ids and bit patterns). Checked by the env-gated
test crates/qubivex-runtime/tests/real_dense.rs.
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
}, sys.stdout, separators=(",", ":"))
