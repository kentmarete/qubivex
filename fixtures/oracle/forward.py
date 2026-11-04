"""Reference forward passes of the tiny fixtures, from transformers (f32).

    ~/.qubivex/oracle/bin/python -I fixtures/oracle/forward.py > fixtures/oracle/forward.json

For each SafeTensors fixture: per-layer attention output (after o_proj), router logits,
selected experts and their routing weights (rank order), MoE output and layer output
(a dense fixture: attention, FFN and layer output), then the final logits, for every
position of a 14-token prompt; and 12 greedy tokens
(full forward each step, no end-of-sequence stop). Values are f32 bit patterns.
Checked by crates/qubivex-runtime/tests/oracle.rs.
"""
import json
import sys

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

PROMPT = [72, 101, 108, 108, 111, 44, 32, 101, 120, 112, 101, 114, 116, 115]
GREEDY = 12


def bits(t):
    return t.detach().to(torch.float32).contiguous().numpy().reshape(-1).view(np.uint32).tolist()


out = {"generator": "fixtures/oracle/forward.py", "transformers": transformers.__version__,
       "torch": torch.__version__, "prompt": PROMPT, "models": {}}
for name in ["olmoe-tiny", "qwen3-moe-tiny", "dense-tiny"]:
    model, info = AutoModelForCausalLM.from_pretrained(
        f"fixtures/models/{name}", dtype=torch.float32, output_loading_info=True)
    assert not any(info.values()), f"{name}: weights not loaded cleanly: {info}"
    model.eval()
    cap = {}
    dense = not hasattr(model.model.layers[0].mlp, "gate")
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(("attn", i), o[0]))
        if not dense:
            layer.mlp.gate.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(("gate", i), o))
        layer.mlp.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(("moe", i), o))
        layer.register_forward_hook(
            lambda m, a, o, i=i: cap.__setitem__(("hidden", i), o[0] if isinstance(o, tuple) else o))
    with torch.no_grad():
        logits = model(torch.tensor([PROMPT])).logits[0]
    layers = []
    for i in range(len(model.model.layers)):
        if dense:
            layers.append({
                "attention": bits(cap[("attn", i)][0]),
                "ffn": bits(cap[("moe", i)][0]),
                "hidden": bits(cap[("hidden", i)][0]),
            })
            continue
        router_logits, weights, idx = cap[("gate", i)]
        layers.append({
            "attention": bits(cap[("attn", i)][0]),
            "router_logits": bits(router_logits),
            "experts": idx.tolist(),
            "routing_weights": bits(weights),
            "moe": bits(cap[("moe", i)][0]),
            "hidden": bits(cap[("hidden", i)][0]),
        })
    ids = list(PROMPT)
    with torch.no_grad():
        for _ in range(GREEDY):
            ids.append(int(model(torch.tensor([ids])).logits[0, -1].argmax()))
    out["models"][name] = {"layers": layers, "logits": bits(logits), "greedy": ids[len(PROMPT):]}

json.dump(out, sys.stdout, separators=(",", ":"))
sys.stdout.write("\n")
