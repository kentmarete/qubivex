# Numerical oracle fixtures

Reference values that QUBIVEX's kernels and forward pass are checked against. The
files here are small and committed; generating them needs Python, the tests do not.

## Environment

An isolated virtual environment, outside the repository:

```sh
python3 -m venv ~/.qubivex/oracle
~/.qubivex/oracle/bin/python -m pip install torch transformers safetensors gguf
```

Versions used for the committed files: torch 2.15.0, transformers 5.22.1,
safetensors 0.8.0, gguf 0.19.0, numpy 2.5.4.

## Files

| File | Generator | Checked by |
|---|---|---|
| `ggml_dequant.json` | `ggml_dequant.py` (gguf-py, llama.cpp's Python package) | `crates/qubivex-backend-cpu/tests/oracle.rs`: Q8_0, Q4_K, Q6_K dequantization, bit for bit |
| `olmoe_tokenizer.json` | `olmoe_tokenizer.py <dir>` (transformers, official OLMoE-1B-7B-0125-Instruct `tokenizer.json`) | `crates/qubivex-runtime/tests/tokenizer.rs` (env-gated): ids, decoded text and chat template |
| `forward.json` | `forward.py` (transformers, f32, the SafeTensors fixtures) | `crates/qubivex-runtime/tests/oracle.rs`: per-layer attention, router logits, selected experts and weights, MoE and layer outputs (dense: attention, FFN and layer outputs), logits (1e-5 relative), 12 greedy tokens |
| `dense_real.json` | `dense_real.py <dir>` (transformers, f32, the official `Qwen/Qwen3-0.6B` bf16 SafeTensors checkpoint) | `crates/qubivex-runtime/tests/real_dense.rs` (env-gated, `QUBIVEX_DENSE_SAFETENSORS_DIR`): prompt ids, 32 greedy tokens, the 32 largest logits of each prompt position (1e-4 relative) |

Regenerate with `~/.qubivex/oracle/bin/python -I fixtures/oracle/<script>.py > fixtures/oracle/<file>.json`.
