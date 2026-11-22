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
| `ggml_dequant.json` | `ggml_dequant.py` (gguf-py, llama.cpp's Python package) | `crates/qubivex-backend-cpu/tests/oracle.rs`: Q8_0, Q4_K, Q6_K and MXFP4 dequantization, bit for bit (MXFP4: every E8M0 scale byte) |
| `olmoe_tokenizer.json` | `olmoe_tokenizer.py <dir>` (transformers, official OLMoE-1B-7B-0125-Instruct `tokenizer.json`) | `crates/qubivex-runtime/tests/tokenizer.rs` (env-gated): ids, decoded text and chat template |
| `forward.json` | `forward.py` (transformers, f32, the SafeTensors fixtures) | `crates/qubivex-runtime/tests/oracle.rs`: per-layer attention, router logits, selected experts and weights, MoE and layer outputs (dense: attention, FFN and layer outputs), logits (1e-5 relative), 12 greedy tokens; `gpt-oss-tiny` has SafeTensors MXFP4 experts (transformers dequantizes them exactly; the script runs them in f32) |
| `dense_real.json` | `dense_real.py <dir>` (transformers, f32, the official `Qwen/Qwen3-0.6B` bf16 SafeTensors checkpoint) | `crates/qubivex-runtime/tests/real_dense.rs` (env-gated, `QUBIVEX_DENSE_SAFETENSORS_DIR`): prompt ids, 32 greedy tokens, the 32 largest logits of each prompt position (1e-4 relative) |
| `qwen2_real.json` | `dense_real.py <dir> --chat` (transformers, f32, the official `Qwen/Qwen2.5-0.5B-Instruct` bf16 SafeTensors checkpoint, revision `7ae557604adf67be50417f59c2c2f167def9a775`) | `crates/qubivex-runtime/tests/real_dense.rs` (env-gated, `QUBIVEX_QWEN2_SAFETENSORS_DIR`): as `dense_real.json`, plus the chat template's text and ids and a greedy chat reply; Metal, Vulkan and Sim against the CPU |
| `llama3_real.json` | `dense_real.py <dir>` (transformers, f32, the official `meta-llama/Llama-3.2-3B` bf16 SafeTensors checkpoint, revision `13afe5124825b4f3751f836b40dafda64c1ed062`) | `crates/qubivex-runtime/tests/real_dense.rs` (env-gated, `QUBIVEX_LLAMA3_SAFETENSORS_DIR`): as `dense_real.json`, through llama3 RoPE scaling |
| `gpt_oss_real.json` | `gpt_oss_real.py <dir>` (transformers, f32, the official `openai/gpt-oss-20b` checkpoint, revision `6cee5e81ee83917806bbde320786a8fb61efebee`, cut to two layers; MXFP4 experts dequantized exactly) | `crates/qubivex-runtime/tests/real_gpt_oss.rs` (env-gated, `QUBIVEX_GPT_OSS_DIR`): tokenizer ids, chat template (date line aside), layers 0 (sliding window) and 1 (full): router logits, experts and weights at every position, attention, MoE and layer outputs either side of the window (1e-5 relative); a greedy reply equal on every backend |

Regenerate with `~/.qubivex/oracle/bin/python -I fixtures/oracle/<script>.py > fixtures/oracle/<file>.json`.
