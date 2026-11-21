"""Reference tokenizer, chat template and first two layers of a real gpt-oss checkpoint
(huggingface.co/openai/gpt-oss-20b, revision 6cee5e81ee83917806bbde320786a8fb61efebee),
from transformers in f32 (MXFP4 experts dequantized exactly, then widened).

    ~/.qubivex/oracle/bin/python -I fixtures/oracle/gpt_oss_real.py <model dir> \
        > fixtures/oracle/gpt_oss_real.json

The whole model does not fit in memory in f32, so the reference is the model cut to its
first two layers (layer 0 has a sliding window, layer 1 full attention); a layer's output
does not depend on the layers after it, so the full model's layers 0 and 1 must match.
Recorded for a prompt longer than the 128-position window: the tokenizer's ids for a few
texts, the chat template rendered for one conversation (text and ids; the date line
comes from strftime_now and is compared without it), and per layer the router logits
and selected experts of every position, and the attention output (after o_proj), MoE
output and layer output at a few positions on both sides of the window's edge, as f32
bit patterns. Checked by the env-gated test crates/qubivex-runtime/tests/real_gpt_oss.rs.
"""
import json
import sys

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

TEXTS = [
    "The capital of France is",
    "Hello, world! It's 2025 — naïve café ünïcödé 日本語 🙂\n\n  indented\tTabs",
    "def f(x):\n    return x ** 2  # square\n",
]
CHAT = [{"role": "user", "content": "What is the capital of France?"}]
PROMPT = (
    "In the beginning the lighthouse keeper counted the ships that passed the rocks each night, "
    "writing their names in a ledger by the light of a small lamp. Over many years the ledger grew "
    "thick with names: fishing boats, ferries, tall ships with patched sails, and once a strange "
    "grey vessel that carried no lights at all. He never learned where it came from, but he wrote "
    "its shape in the margin and wondered about it until the end of his days. The village below "
    "the cliff told stories about that ship, and the children dared each other to watch for it. "
    "When the old keeper retired, his granddaughter took over the lamp and the ledger, and on her "
    "first winter night she saw, far out beyond the rocks, a grey hull moving without a single light."
)
POSITIONS_FROM_END = [1, 2]
POSITIONS = [0, 1, 63, 126, 127, 128, 129]
LAYERS = 2


def bits(t):
    return t.detach().to(torch.float32).contiguous().numpy().reshape(-1).view(np.uint32).tolist()


d = sys.argv[1]
tok = AutoTokenizer.from_pretrained(d)
out = {
    "generator": "fixtures/oracle/gpt_oss_real.py",
    "transformers": transformers.__version__,
    "torch": torch.__version__,
    "texts": [{"text": t, "ids": tok.encode(t, add_special_tokens=False)} for t in TEXTS],
}
chat = tok.apply_chat_template(CHAT, add_generation_prompt=True, tokenize=False)
out["chat"] = {"messages": CHAT, "text": chat, "ids": tok.encode(chat, add_special_tokens=False)}

ids = tok.encode(PROMPT, add_special_tokens=False)
assert len(ids) > 140, len(ids)
positions = POSITIONS + [len(ids) - k for k in POSITIONS_FROM_END]
out["prompt"] = ids
out["positions"] = positions

cfg = AutoConfig.from_pretrained(d)
cfg.num_hidden_layers = LAYERS
cfg.layer_types = cfg.layer_types[:LAYERS]
model = AutoModelForCausalLM.from_pretrained(d, config=cfg, dtype=torch.float32)
model = model.float()  # MXFP4 experts come back as bf16 (exact); run them in f32
model.eval()
cap = {}
for i, layer in enumerate(model.model.layers):
    layer.self_attn.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(("attn", i), o[0]))
    layer.mlp.router.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(("router", i), o))
    layer.mlp.register_forward_hook(
        lambda m, a, o, i=i: cap.__setitem__(("moe", i), o[0] if isinstance(o, tuple) else o))
    layer.register_forward_hook(
        lambda m, a, o, i=i: cap.__setitem__(("hidden", i), o[0] if isinstance(o, tuple) else o))
with torch.no_grad():
    model.model(torch.tensor([ids]))
layers = []
for i in range(LAYERS):
    logits, weights, idx = cap[("router", i)]
    at = lambda t: bits(t[0][positions])
    layers.append({
        "router_logits": bits(logits),
        "experts": idx.tolist(),
        "routing_weights": bits(weights),
        "attention": at(cap[("attn", i)]),
        "moe": at(cap[("moe", i)]),
        "hidden": at(cap[("hidden", i)]),
    })
out["layers"] = layers
json.dump(out, sys.stdout, separators=(",", ":"))
sys.stdout.write("\n")
