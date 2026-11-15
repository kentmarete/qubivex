# Benchmarks

What Qubivex has measured, how, and what is still unmeasured. The method (tools, power,
rounds, baselines) is in [performance/BENCHMARKING.md](performance/BENCHMARKING.md); raw
records are in [performance/baselines/](performance/baselines/). This page is the summary
the roadmap points to and is updated at each milestone.

## Rules

- Same machine, same model file, same settings, on mains power, alternated rounds; the
  tokens must be identical (or the difference explained) for a speed number to count.
- A comparison with another runtime names both versions, the model file, the
  quantisation, the threads and the generation settings, and says whether both are
  lossless. Lossy and lossless runs are not compared as equals.
- Every number below names its source. "Estimate" means not measured.

## Machine

Apple M3 Pro (6 performance + 6 efficiency cores), 18 GB, macOS; MoltenVK 1.4.2 for
Vulkan. No x86-64 or NVIDIA machine has been measured.

## Recorded results: OLMoE-1B-7B-0125-Instruct Q4_K_M

All lossless, tokens identical to the CPU reference.

| Backend | Setting | Decode | Prefill | Source |
|---|---|---|---|---|
| CPU | all experts resident | 42.4 tok/s | 115 tok/s (128 tokens) | IMPLEMENTATION_STATUS (job 8), Phase 8 slice 6 |
| Metal | all experts resident, token mode | 59.6 tok/s | 239 tok/s (128 tokens) | ADR-011 am. 5 |
| Metal | 1024 MiB expert cache | 29.5 tok/s (CPU 34.9) | 194 tok/s (CPU 117) | ADR-011 am. 6 |
| Vulkan (MoltenVK) | `main`, Step 3c, opt-ins | 34.3–34.5 tok/s | — | ADR-019 am. 4 |
| Vulkan (MoltenVK) | PR #67 (Step 5), attention on device | 43 tok/s (23.17–23.30 ms) | 166 ms | audit §24 (branch `vulkan-kernel-port`, not merged) |
| CPU vs Colibri | `qubivex run` vs Colibri lossless / default lossy | 20.7–21.0 vs 3.0 / 17.4–17.8 tok/s | — | [COLIBRI-OLMOE-BENCHMARK.md](performance/COLIBRI-OLMOE-BENCHMARK.md) (earlier build) |

Concurrent decode (CPU, 1024 MiB cache, prefetch on): 37.9, 43.7, 50.9 tok/s total at
2, 4, 8 sequences (IMPLEMENTATION_STATUS, ADR-017 §7).

## Not yet measured (required by the brief)

| Metric | State |
|---|---|
| Model load time | Not recorded systematically |
| Time to first token over HTTP on a real model | Only on the tiny fixture (30 ms, wiring check) |
| Peak RAM per run | Recorded in the Colibri comparison only |
| Peak VRAM | Not measured (Metal shares host memory; Vulkan pool sizes only) |
| KV cache consumption | Planned values only |
| Dense models (Qwen3-0.6B) throughput | Not recorded |
| Image preprocessing, vision encoding | Not applicable yet |
| Ollama and llama.cpp on the same model | **Not run**: neither is installed on the development machine. Installing them as reference tools is the owner's decision; they are never execution dependencies |

## Planned benchmark additions

1. `qubivex bench` records load time, time to first token, peak RSS and KV bytes in its
   JSON (step with P1).
2. Dense model baselines (Qwen3-0.6B, then Qwen2.5) on CPU, Metal and Vulkan.
3. Equal-settings comparisons with llama.cpp (`llama-bench`) and Ollama on the same GGUF,
   once installed, with lossy/lossless noted.
4. Multimodal: preprocessing and encoder time per image size, share of time to first
   token.
