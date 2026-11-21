# Roadmap: a general-purpose, hardware-aware inference runtime

- Started: 2026-11-15, from `main` at `8a70709` (PR #66).
- Source of the goals: the owner's brief "Build a general-purpose, hardware-aware AI
  inference runtime" (2026-11-15). Priorities P0–P8 below are the brief's.
- Target architecture: [ARCHITECTURE.md](ARCHITECTURE.md). Multimodal design:
  [MULTIMODAL_RUNTIME.md](MULTIMODAL_RUNTIME.md). APIs:
  [API_COMPATIBILITY.md](API_COMPATIBILITY.md). Measurements: [BENCHMARKS.md](BENCHMARKS.md).
- Progress of individual jobs stays in [`IMPLEMENTATION_STATUS.md`](../IMPLEMENTATION_STATUS.md);
  decisions in [`docs/decisions/`](decisions/). This file holds the plan and the matrix,
  and is updated at the end of every milestone.

**Rule for this file:** a feature is "Implemented" only when the source has it and a test
exercises it end to end. Every row cites where. "Partial" says what is missing.

---

## 1. Audit: what exists (P0)

Audited 2026-11-15 by reading the source and tests on `main` (`8a70709`), and by running
the workspace tests (§1.9). The project's own records (IMPLEMENTATION_STATUS.md, ADRs, the
2026-11-14 engineering health report) were used as pointers and checked against the code
before being repeated here.

### 1.1 Workspace

17 library crates, the CLI (`apps/cli`) and `xtask`. Source lines (excluding tests):

| Crate | Lines | Role |
|---|---:|---|
| `qubivex-core` | 2.3k | Ids, dtypes (f32, f16, bf16, Q4_0…Q8_K, I8RowScaled), errors |
| `qubivex-compute` | 2.8k | Tensor descriptors and views, `ComputeBackend` trait, fixed-order math (RMSNorm, RoPE, softmax, SwiGLU, attention) |
| `qubivex-backend-cpu` | 2.7k | CPU kernels: portable, NEON (fp16/fhm detected at run time), AVX2+FMA (detected at run time); register-blocked prefill; `SimDevice` |
| `qubivex-backend-metal` | 2.5k | Metal kernels and device, bit-identical to the CPU except f32 subnormals (ADR-011) |
| `qubivex-backend-vulkan` | 4.6k | Vulkan via `ash` (MoltenVK verified), 10 compute shaders, bit-identical to the CPU (ADR-019) |
| `qubivex-arch` | 1.8k | `Transformer`: one generic decoder (dense or MoE FFN), `forward`/`forward_batch` |
| `qubivex-model` | 4.0k | GGUF and SafeTensors readers, manifest v1.1, architecture registry, import, discovery |
| `qubivex-model-registry` | 0.6k | Installed-model registry (`registry.json`), atomic installs |
| `qubivex-model-download` | 2.1k | Hugging Face search and resumable, verified downloads |
| `qubivex-tokenizer` | 0.6k | HF `tokenizer.json`, GGUF vocab rebuild, MiniJinja chat templates, streaming decode |
| `qubivex-memory` | 1.6k | Memory planner and ledger |
| `qubivex-expert-stream` | 3.2k | Expert store: leases, LFRU cache, I/O queue, prefetch, mirrors, checksums |
| `qubivex-kv-cache` | 0.4k | Prefix cache index (retained KV, LRU, scopes) |
| `qubivex-scheduler` | 0.4k | Request queue: priorities, deadlines, fairness |
| `qubivex-runtime` | 6.8k | `Engine`: load, placement, generation contract, batching, prefix reuse |
| `qubivex-hardware` | 2.4k | CPU, memory, storage and GPU detection |
| `qubivex-api` | 4.2k | HTTP gateway (Kerniq Server/Axum): OpenAI-shaped endpoints, auth, limits, metrics |
| `apps/cli` | 5.0k | `run`, `serve`, `bench`, `system`, `doctor`, `plan`, `models …`, `keys` |

Boundaries are enforced: `cargo xtask check` (embeddable runtime, no server deps in it,
lints including a ban on model-family names in engine code, marked `model-name-ok` where
a file-format identifier needs one) and `cargo deny`.

### 1.2 Implemented (verified in source and tests)

| Area | What | Evidence |
|---|---|---|
| Model formats | GGUF v3 reader; SafeTensors (sharded) reader; manifest v1.1 with per-file sha256; import of HF directories and GGUF files into manifests | `qubivex-model/src/{gguf,safetensors,import,manifest}.rs`, `tests/` (38 tests) |
| Architecture registry | Factories by id, versioned, config parsing with cross-validation, tensor schema; two architectures: `transformer_moe`, `transformer_dense` | `qubivex-model/src/registry.rs`, `arch/` |
| Real models | OLMoE-1B-7B Q4_K_M GGUF (tokens identical to a committed reference, also to Colibri's lossless path); Qwen3-0.6B bf16 SafeTensors (32 greedy tokens and top-32 logits vs transformers) | `qubivex-runtime/tests/oracle.rs`, `tests/real_dense.rs` (env-gated) |
| Fixture models | Llama-config dense, OLMoE, Qwen3-MoE, OLMoE GGUF, all vs transformers in f32 per layer | `fixtures/models/`, `fixtures/oracle/forward.json` |
| Lossless kernels | f32, f16, bf16, Q8_0, Q4_K, Q6_K, MXFP4: matmul and dequant bit for bit against gguf-py | `qubivex-backend-cpu/tests/oracle.rs`, `fixtures/oracle/ggml_dequant.json` |
| Attention | MHA/GQA/MQA, optional per-head q/k RMSNorm, `clip_qkv`, NeoX RoPE with correctly rounded tables | `qubivex-arch/src/{lib,ops}.rs` |
| KV cache | Per-sequence caches, reservation against the memory plan, restore for prefix reuse | `qubivex-arch` `KvCache`, `qubivex-runtime/tests/prefix.rs` |
| MoE | Top-k routing (optional renormalisation), host or device routing, fused expert dispatch | `qubivex-arch`, `qubivex-runtime/tests/batch.rs` |
| Expert streaming | Experts read on demand (pread), LFRU slab cache of any size, prefetch, direct I/O option, checksums, mirror failover | `qubivex-expert-stream` (64 tests) |
| Memory planner | Budgets RAM (and VRAM for discrete devices), dense + experts + KV + activations, refuses plans that do not fit | `qubivex-memory`, ADR-005, ADR-018 |
| CPU backend | Portable, NEON, AVX2+FMA paths chosen at run time; rayon compute pool; batch-invariant kernels | `qubivex-backend-cpu/src/{lib,neon,tile}.rs` |
| Metal backend | Per-matmul placement for any model; whole-layer and whole-token passes for MoE; batching | `qubivex-backend-metal`, `qubivex-runtime/src/metal_forward.rs` |
| Vulkan backend | Discrete-device memory model, dense and expert matmuls, fused expert block, device routing (`--vulkan-route`, off by default), decode attention (`--vulkan-attn`, off by default) | `qubivex-backend-vulkan` (44 tests, `QUBIVEX_REQUIRE_VULKAN`-gated on hardware) |
| Batching | Sequences share forward passes (CPU and Metal), bit-identical to running alone | ADR-017, `tests/batch.rs` |
| Prefix reuse | Retained KV by longest prefix, per scope, priced in the plan | ADR-016, `tests/prefix.rs` |
| Scheduler | Bounded queue, priorities, deadlines, fairness per client, cancellation | `qubivex-scheduler`, `qubivex-api/tests/concurrency.rs` |
| Sampling | Greedy, temperature, top-p, top-k, seed (deterministic), stop strings | `qubivex-runtime/src/request.rs`, `tests/requests.rs` |
| Tokenizer | HF `tokenizer.json`; GGUF byte-level BPE for pre-tokenizer types `olmo`, `gpt-2`, `default`; chat templates in sandboxed MiniJinja | `qubivex-tokenizer/src/lib.rs` |
| OpenAI API | `/v1/chat/completions`, `/v1/completions` (SSE streaming or not), `/v1/models`, usage incl. cached tokens, OpenAI error shapes; unknown-but-understood fields refused, never dropped | `qubivex-api/src/openai.rs`, `tests/` (51 tests), `docs/serving/openai-compatibility.md` |
| Server operations | API keys (shared, per-client hashed), rate limits, CORS, admin load/unload, Prometheus metrics, structured logs, drain on SIGTERM, Docker image | `qubivex-api`, ADR-013, CI `docker` job |
| Model downloads | HF search, resumable verified downloads, atomic installs, registry with crash recovery | `qubivex-model-download` (44 tests), `qubivex-model-registry` (16) |
| Benchmarks | Criterion kernel benches; `qubivex bench` end to end; Vulkan baseline driver | `crates/qubivex-backend-cpu/benches`, `apps/cli/src/bench.rs`, `scripts/bench-vulkan-baseline` |

### 1.3 Partially implemented

| Area | What exists | What is missing |
|---|---|---|
| Dense on GPU | Dense matmuls placed one by one on Metal or Vulkan | Whole-layer/whole-token device passes exist for MoE only (`engine.rs:1154`); device attention is MoE-only (`qubivex-arch/src/lib.rs`, `dev_attn`) |
| Llama family | Llama-shaped config through `transformer_dense` (fixture checked vs transformers); Llama 3.x `llama3` RoPE scaling (Llama-3.2-3B checked vs transformers, 2026-11-19) | GGUF tokenizer `llama-bpe` refused |
| GGUF tokenizers | Byte-level BPE rebuilt from GGUF for 3 pre-tokenizer types | `qwen2`, `llama-bpe` and others refused; those models need `tokenizer.json` |
| Quantised formats | Kernels for f32/f16/bf16/Q8_0/Q4_K/Q6_K/MXFP4 (MXFP4 read packed in every kernel, ADR-007 amendment 2) | Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q5_K are known dtypes without kernels |
| Architecture registry | id, version, config schema, tensor schema | Declared quantisation support, backend support, positional and normalisation strategy, preprocessing requirements (brief §4) |
| Vulkan speed | Bit-identical; Step 5 kernel (−10 to −12.5% decode) on branch `vulkan-kernel-port` (PR #67) | PR #67 undecided; prefill attention on host; decode attention and routing off by default |
| CPU SIMD | NEON and AVX2+FMA | No AVX-512 path; x86-64 Linux not verified on real models (RW-05 in the 2026-11-14 action plan) |
| Streaming detokenization | Correct | Decodes the whole id list per token: O(n²) in output length (`qubivex-tokenizer/src/lib.rs`, `StreamDecoder::push`) |
| Weight loading | Dense weights read once into an arena; experts streamed | No memory-mapping (no `mmap` anywhere in `crates/`); a dense model must fit in RAM |
| Metrics | Spec §8 metrics | Peak VRAM not measured; real-model metrics not recorded |

### 1.4 Missing

| Area | State |
|---|---|
| ~~Q/K/V projection biases~~ | **Done in milestone 1 (§5)**: `transformer_dense` option `qkv_bias` |
| RoPE scaling variants | `default` and `llama3` accepted: no `linear`, `dynamic`, `yarn`, `mrope` |
| Sliding-window attention | Refused (`use_sliding_window`) |
| Non-SwiGLU FFNs, LayerNorm, attention/MLP biases other than q/k/v, learned positions | Refused or absent |
| ~~Structured output~~ | **Done in milestone 2 (§5)**: constrained decoding, ADR-020 |
| Tool calling | `tools` refused (400); no tool-call parsing |
| Embeddings endpoint | None |
| Ollama-compatible API | None (`/api/*` answer 404) |
| Ollama registry import | None |
| Multimodal | No image, audio or video decoding, no encoders, no projectors, no non-token inputs to the forward pass, `image_url` refused (400) |
| Paged KV | None (contiguous per-sequence caches) |
| Speculative decoding | None |
| Repetition/presence/frequency penalties | Only 0 accepted |
| Flash-style attention | None (host attention is exact two-pass softmax) |
| CUDA, multi-GPU, distributed | None (CUDA blocked: no NVIDIA hardware) |

### 1.5 Incorrect or incomplete

| Finding | Evidence | Effect |
|---|---|---|
| `cargo xtask check` fails in CI on every run since 2026-11-12 | `.github/workflows/ci.yml` runs it; `check_embeddable` uses `cargo tree --target all --offline` and hosted runners lack other targets' crates | The merge gate is red for every PR; merges happen on local verification |
| O(n²) streaming detokenization | `StreamDecoder::push` calls `decode(&self.ids)` on every token | Long completions spend growing time on the generating thread |
| Documentation drift | Listed in the 2026-11-14 health report §9 (for example `serve --help` names only `cpu or metal`; IMPLEMENTATION_STATUS test section dated 2026-11-02) | Readers get wrong limits |
| `qubivex serve` cannot reach Vulkan options | `apps/cli/src/serve.rs` builds `DeviceArgs` with Vulkan options fixed off | Vulkan opt-ins only through `run`/`bench` |

No data-integrity or security defect was found (consistent with the health report §3–4).

### 1.6 Architectural limitations

1. **One decoder shape.** `qubivex-arch::Transformer` is pre-norm RMSNorm, RoPE, SwiGLU,
   no biases. New families need *configurable components* (bias, norm, activation,
   position scheme), not new copies of the decoder.
2. **Token-only input.** `forward_segments` takes token ids and looks embeddings up
   itself; positions are `cache.len()..`. Multimodal input needs embedding spans and
   explicit (possibly multi-axis) positions.
3. **Prefix cache keyed by token ids.** Correct for text; unsafe for any input whose
   placeholder tokens do not identify the content (images, audio).
4. ~~**Registry declares structure, not capabilities.**~~ **Done in milestone 3 (§5)**:
   architectures declare capabilities; undeclared formats are refused at validation.
5. **Dense models must be resident.** Streaming exists for experts only.
6. **Device execution is MoE-first.** Dense models get per-matmul placement only.
7. **Serving is OpenAI-only and text-only**, without constrained decoding.

### 1.7 Performance bottlenecks (measured by the project)

| Bottleneck | Measurement | Source |
|---|---|---|
| Vulkan quantised matmuls are instruction-bound | Q4_K 37.4 GB/s, Q6_K 47.7 GB/s; decoding 65–75% of kernel time | audit §22–23, `docs/engineering/VULKAN_EXECUTION_AUDIT.md` |
| Vulkan submit latency | 3.1–4.2 ms per token over 17 submissions | audit §22 |
| Metal below half an expert pool | Loses to the CPU (29.5 vs 34.9 tok/s at 1024 MiB) and falls back | ADR-011 am. 6 |
| Detokenization | O(n²), not measured | §1.5 |
| Dense GPU | Not measured (per-matmul placement only) | §1.3 |

### 1.8 Technical dependencies (current)

Pure Rust except system APIs: `tokenizers` (fancy-regex), `minijinja`, `serde`, `rayon`,
`ash` (Vulkan loader at run time, no SDK at build), `objc2`/Metal, Tokio + Kerniq Server
(gateway only), `ring`/`rustls` (downloads). New ones need an ADR-001 amendment and
`cargo deny`. Reference tooling (outside the product): the oracle venv
(`~/.qubivex/oracle`: torch 2.15, transformers 5.22.1, gguf-py, Pillow), Colibri.
Ollama and llama.cpp are **not installed** on the development machine.

### 1.9 Test baseline

`cargo test --workspace --no-fail-fast` on `8a70709`, M3 Pro, 2026-11-15:
**573 passed, 0 failed, 6 ignored** (debug build). The Vulkan tests ran on MoltenVK. The
env-gated real-model tests (OLMoE, Qwen3-0.6B) were not enabled for this run.

---

## 2. Feature matrix

Complexity: S (days), M (1–2 weeks), L (several weeks), XL (months).
State: ✅ implemented and tested, ◐ partial, ✗ missing.

### A. Model management

| Feature | Now | Target | Depends on | Cx | Acceptance |
|---|---|---|---|---|---|
| GGUF read | ✅ | ✅ + all K-quant/legacy kernels | — | M | Each dtype bit-exact vs gguf-py dequant |
| SafeTensors read | ✅ | ✅ | — | — | — |
| Manifest | ✅ v1.1 | v1.2: model **components** (text, vision, audio, projector), each with format, config and weights | — | M | A manifest with GGUF text + SafeTensors vision imports, validates and round-trips |
| Architecture registry | ✅ structure and capabilities (milestone 3) | + preprocessing (multimodal) | — | S | **Met**: `models architectures` lists capabilities; validation and load refuse unsupported formats before reading weights |
| Hugging Face downloads | ✅ | + multi-component repos | manifest v1.2 | S | Pull of a VL repo installs all components |
| Ollama registry import | ✗ | Resolve Ollama manifests to blobs, import those Qubivex can execute | GGUF coverage | M | `models import ollama://…` of a supported model runs; unsupported blobs refused with reason |
| Weight validation | ✅ sha256, shapes, cross-checks | ✅ for every component | manifest v1.2 | S | — |

### B. Tensor engine

| Feature | Now | Target | Depends on | Cx | Acceptance |
|---|---|---|---|---|---|
| CPU portable/NEON/AVX2 | ✅ | + AVX-512 where available | x86 hardware | M | Bit-identical to portable; bench vs AVX2 |
| Quantised matmul | ◐ 6 formats | + Q4_0/Q4_1/Q5_0/Q5_1/Q5_K/Q3_K/Q2_K | — | M | Bit-exact vs gguf-py; Criterion numbers |
| Vulkan matmul/attention/norm | ◐ | + vision ops (conv-as-matmul, LayerNorm, GELU), prefill attention | multimodal ops | L | Bit-identical to CPU (ADR-019) + benchmark |
| Metal | ✅ MoE, ◐ dense | whole-layer dense | — | M | Dense tokens identical to CPU; tok/s recorded |
| Memory-mapped weights | ✗ | mmap for dense weights where the OS page cache suffices | planner | M | Dense model larger than budget runs with reported page-in cost |
| Memory planner | ✅ | + vision/audio weights, encoder activations, embedding spans | multimodal | S | Plan lists each component; refusal when it cannot fit |

### C. Architecture engine

| Feature | Now | Target | Depends on | Cx | Acceptance |
|---|---|---|---|---|---|
| Configurable decoder components | ◐ (q/k norm, clip, q/k/v bias) | bias on o and FFN, norm kind, activation, position scheme | — | M | Each option checked vs transformers on a fixture |
| Llama 1/2-style | ◐ fixture | real checkpoint verified | — | S | Real-model tokens vs transformers |
| Llama 3.x | ◐ SafeTensors | ~~`llama3` RoPE scaling~~ (done 2026-11-19), `llama-bpe` GGUF tokenizer | RoPE variants | M | **Met for SafeTensors**: Llama-3.2-3B tokens and logits vs transformers; Metal, Vulkan, Sim give the CPU's tokens |
| Qwen2 / Qwen2.5 | ✅ SafeTensors (milestone 1); ◐ GGUF needs `tokenizer.json` | + GGUF `qwen2` pre-tokenizer | q/k/v bias | S | **Met for SafeTensors**: Qwen2.5-0.5B-Instruct greedy tokens, logits, chat template and reply vs transformers; Metal, Vulkan, Sim give the CPU's tokens |
| Qwen3 dense | ✅ | ✅ | — | — | — |
| Qwen3-MoE, OLMoE | ✅ | ✅ | — | — | — |
| Qwen2.5-VL | ✗ | full | multimodal foundation, M-RoPE | L | **P4** criteria (brief §6) |
| Sliding window, other RoPE scalings | ✗ | as needed per family | — | M | per family |

### D. Multimodal engine

| Feature | Now | Target | Depends on | Cx | Acceptance |
|---|---|---|---|---|---|
| Unified input (token + embedding spans, N-D positions) | ✗ | in `qubivex-arch` / runtime | — | M | Existing tests unchanged; span input equals token input when spans hold table rows |
| Image decode/validate/preprocess pipeline | ✗ | declarative, from `preprocessor_config.json` | image dependency (ADR) | M | Pixel values vs the reference processor within stated tolerance |
| Vision encoder (generic ViT from config) | ✗ | windowed/full attention, 2-D RoPE or learned positions, LN/RMSNorm, GELU/SwiGLU | ops | L | Encoder output vs reference, given reference pixels |
| Projectors | ✗ | linear, MLP, patch-merge+MLP | encoder | S | vs reference |
| Multimodal RoPE | ✗ | sectioned multi-axis RoPE | unified input | S | Positions and logits vs reference |
| Media-aware prefix cache | ✗ | content digest per span in the key | unified input | S | Same text, different image: no reuse |
| Audio (decode, resample, mel, encoder) | ✗ | design first, then one model | vision foundation | L | per model |
| Video (frames, sampling, temporal positions) | ✗ | via vision path | vision, M-RoPE | L | per model |

### E. Execution engine

| Feature | Now | Target | Depends on | Cx | Acceptance |
|---|---|---|---|---|---|
| Continuous batching | ✅ CPU, Metal (Vulkan not measured) | + Vulkan | — | M | bits identical, throughput recorded |
| Prefix caching | ✅ | media-aware | D | S | above |
| Paged KV | ✗ | when measured to matter | — | L | memory per sequence and throughput at N |
| Speculative decoding | ✗ | draft model, exact acceptance | — | L | identical outputs to target model at temperature 0; speed-up recorded |
| Flash-style attention | ✗ | blocked attention (needed for vision encoders) | — | M | identical within tolerance; memory bound |
| Linear streaming detokenization | ✗ | O(n) | — | S | same text; time per token flat |

### F. Serving engine

| Feature | Now | Target | Depends on | Cx | Acceptance |
|---|---|---|---|---|---|
| OpenAI chat/completions/models, streaming | ✅ | + multimodal content, tools, `response_format` | C, D | — | `docs/serving/openai-compatibility.md` tables all green for supported fields |
| JSON mode / JSON Schema | ✅ (milestone 2) | token-level constrained decoding | tokenizer vocab | M–L | **P2**: every output parses and validates against the schema, over a schema test suite, temperature > 0 included |
| Tool calling | ✗ | template-driven tool definitions and parsing, constrained arguments | P2 | M | OpenAI SDK tool-call round trip |
| Ollama API | ✗ | `/api/tags`, `/api/chat`, `/api/generate`, `/api/show`, `/api/embeddings` | embeddings for the last | M | Documented request/response shapes, streaming NDJSON, `images`, `format`, `options` |
| Image/audio/video input | ✗ | per supported modality | D | — | refused with a clear error until supported (already true for images) |
| Embeddings | ✗ | pooled hidden states for embedding models | — | M | vs reference |

---

## 3. Implementation sequence

Each step is one job and one PR (`docs/development/JOB_WORKFLOW.md`). Order follows the
brief's priorities, with dependencies resolved inside them.

| # | Priority | Job | Notes |
|---|---|---|---|
| 0 | P0 | This audit and the four documents | this PR |
| 1 | P0 | Fix the CI merge gate (`xtask check --offline`) | small; everything after it gets real CI |
| 2 | P1 | **Qwen2/Qwen2.5 text**: configurable q/k/v biases in `transformer_dense`, GGUF `qwen2` pre-tokenizer, reference tests on a real Qwen2.5 checkpoint | first implementation milestone |
| 3 | P1 | Registry capabilities (quant formats, backends, strategies) | refuses unsupported combinations early |
| 4 | P1 | Linear streaming detokenization | small, independent |
| 5 | P1 | Llama 3.x (RoPE `llama3`, `llama-bpe`) | widens text coverage |
| 6 | P2 | JSON mode and JSON Schema constrained decoding | ADR first |
| 7 | P3 | Unified input: embedding spans and N-D positions; media-aware prefix key | ADR (multimodal) first |
| 8 | P3 | Image pipeline (decode, validate, preprocess) | new dependency, ADR-001 amendment |
| 9 | P3 | Generic vision encoder + projectors on the CPU | blocked attention |
| 10 | P4 | Qwen2.5-VL as configuration; reference parity; image chat over the API | the first multimodal acceptance target |
| 11 | P5 | Tool calling; Ollama API | |
| 12 | P6 | Manifest components on download; Ollama registry import | |
| 13 | P7 | Audio, then video | |
| 14 | P8 | Paged KV, speculative decoding, dense device passes, Vulkan for vision | measured first |

---

## 4. Lists

### Completed
- P0 audit (this document), 2026-11-15 (PR #68).
- Milestone 1, P1: Qwen2/Qwen2.5 text models (§5), 2026-11-15.
- Milestone 2, P2: JSON mode and JSON Schema constrained decoding (§5), 2026-11-15.
- Milestone 3, P1: registry capabilities (§5), 2026-11-15.

### Partially completed
- See §1.3.

### Outstanding
- Steps 1–14 of §3.

### Known limitations
- §1.6. In particular: text-only, OpenAI-only, dense models must fit in RAM.

### Technical risks
- **Reference parity for preprocessing** (resampling filters differ between libraries).
- **Vision encoder memory**: full-attention blocks are quadratic in patches.
- **Constrained decoding cost** per token over a 150k vocabulary.
- **Scope creep across families**: every family must be configuration of shared
  components, never a branch on its name (the lint in `xtask` enforces names).
- **Development machine**: 18 GB RAM limits which real checkpoints can be verified in f32
  and bf16; parity runs use quantised weights where needed.

### Blocked
- CUDA and multi-GPU: no NVIDIA hardware (blocker B2).
- AVX-512 and x86-64 Linux validation: no x86 machine available here.
- Ollama and llama.cpp comparisons: neither is installed; installing them is the owner's
  call (they are reference tools only, never execution dependencies).

---

## 5. Milestone reports

### Milestone 1 (2026-11-15): Qwen2 and Qwen2.5 text models (P1, roadmap step 2)

**Implemented.** Q/K/V projection biases as a configurable attention option of the shared
decoder, detected from the tensors at import, with no code path that names a model family.
Plain RoPE is now enforced in `rope_parameters` as well as `rope_scaling` (a defect found
on the way: a transformers-5 config could carry a scaled RoPE there unchecked).

**Files.** `qubivex-model`: `registry.rs` (roles `QBias`, `KBias`, `VBias`),
`arch/common.rs` (bias specs, RoPE check), `arch/transformer_dense.rs` (option, names,
version 1.1), `import.rs` (detection). `qubivex-arch/src/lib.rs` (bias vectors loaded
once, added after the projections; device-fused attention declined with biases).
`xtask/src/fixtures.rs` (fixture `qwen2-tiny`). Tests: `qubivex-model/tests/{fixtures,metadata}.rs`,
`qubivex-runtime/tests/{real_dense,batch,discrete,invariants,prefix,vulkan}.rs`. Oracles:
`fixtures/oracle/forward.py` (+ `qwen2-tiny`), `dense_real.py` (`--chat`),
`qwen2_real.json`.

**Decisions.** ADR-003 amendment 3.

**Tests.**
- `qwen2-tiny` vs transformers per layer: pass. Positive control: the oracle test fails
  with the bias add disabled.
- `qwen2-tiny` on CPU, Sim, Metal and Vulkan, batching, prefix reuse and the invariants:
  `cargo test -p qubivex-runtime` all pass.
- Real `Qwen/Qwen2.5-0.5B-Instruct` (bf16, revision `7ae5576`, sha256 checked), release
  build: prompt ids, 32 greedy tokens, top-32 logits at every prompt position (worst
  1.05e-5 relative), prefill = token by token bit for bit, 1 thread = 4 threads, chat
  template text and ids, greedy chat reply ("Three primary colours are red, yellow, and
  blue.") stopping at `<|im_end|>`; Metal, Vulkan and Sim give the CPU's 16 tokens.
- Regressions: Qwen3-0.6B real test (1.15e-5), OLMoE real-model and tokenizer tests,
  `cargo test -p qubivex-model`, `cargo xtask check` (deps, embeddable, model names,
  fixtures): pass. Full workspace run: see the PR.

**Performance.** No change for existing models (one branch per layer when there are no
biases). Indicative only, on battery (does not meet BENCHMARKING.md): Qwen2.5-0.5B decode
85.5 tok/s on the CPU and 40.3 tok/s on Metal (per-matmul placement), prefill ~310 tok/s
on both. This confirms the audit's "dense on GPU is partial" (§1.3).

**Remaining for Qwen2/Qwen2.5.** GGUF `qwen2` pre-tokenizer (GGUF files need
`tokenizer.json` meanwhile); dense whole-layer device passes; a benchmark on power.

**Next milestone.** Roadmap step 6 (P2): JSON mode and JSON Schema constrained decoding,
ADR first. Steps 3–5 (registry capabilities, linear detokenization, Llama 3.x) are
smaller and can be interleaved.

### Milestone 2 (2026-11-15): JSON mode and JSON Schema (P2, roadmap step 6)

**Implemented.** Token-level constrained decoding as a generic component: a new crate
`qubivex-grammar` (byte grammars, a JSON Schema front end, a lazily determinised matcher
and token masks over a vocabulary trie), one hook at the engine's single sampling site,
and `response_format` (`json_object`, `json_schema`) on `/v1/chat/completions`. Nothing
in it knows a model family; tool-call arguments (P5) and Ollama `format` are further
front ends over the same grammar and matcher. Requests without `response_format` run the
old path.

**Files.** `crates/qubivex-grammar/` (new: `grammar.rs`, `matcher.rs`, `trie.rs`,
`schema.rs`, `json.rs`; `tests/schema.rs`). `qubivex-tokenizer`: `Tokenizer::token_bytes`.
`qubivex-runtime`: `engine.rs` (`GenerateOptions::constraint`, `Engine::token_trie`,
`sample_constrained`), `request.rs`, `lib.rs` (`RuntimeError::Constraint`); tests
`constrained.rs`, `real_constrained.rs`. `qubivex-api`: `openai.rs`, `lib.rs`,
`worker.rs`; `tests/gateway.rs`. `xtask/src/deps.rs`. Oracle:
`fixtures/oracle/validate_json.py` (Python `jsonschema`, added to the oracle venv).

**Decisions.** ADR-020 (Proposed). Masks apply before temperature, top-k and top-p;
greedy takes the unconstrained argmax when allowed; the RNG is drawn as without a
constraint. Schemas are read with their key order kept (the workspace's `serde_json`
sorts keys), so properties are written in schema order. Unsupported keywords are 400s.

**Tests.**
- Grammar: 31 schemas, 400 random walks each (no model), every text parsed by `serde_json`
  and validated by an independent test validator; the same 12 400 texts validate under
  Python `jsonschema` (Draft 2020-12): 0 failures. Masks equal a per-token brute-force
  check at every prefix of walks over a vocabulary with multi-byte tokens across JSON
  structure. Known texts accepted and rejected (order, required, extra properties, lone
  surrogates, invalid UTF-8, `1e999`, leading zeros, whitespace bound). 13 refusals and
  9 invalid schemas. Repetition keeps states and frames constant.
- Fixture models (random weights, temperature 1, 12 seeds × 5 schemas, two models, plus
  JSON mode and greedy): every output stays in the grammar and every end-of-sequence
  output parses. Four constrained requests at once (batched passes) get the tokens each
  got alone.
- Real `Qwen2.5-0.5B-Instruct` (151 665 tokens): 150 203 token byte strings equal the
  tokenizer's own decoding; every byte is a token. Six chat prompts × (t = 0, t = 0.8 with
  3 seeds): all 20 schema replies end at end-of-sequence, parse and validate (Python
  `jsonschema`, with the 3 JSON-mode replies that ended: 23 of 23). JSON mode at t = 0
  loops on repeated keys until `max_tokens` (a 0.5B model's repetition; the text is still
  a valid prefix).
- Gateway: `response_format` replies over HTTP, streamed equal to non-streamed, schema
  order on the wire; 400 for an unsupported keyword, a missing schema and an unknown type.
- Regressions: full `job verify` (see the PR).

**Performance.** Qwen2.5 vocabulary trie: 42 ms, once per engine. Masks along a recipe
reply, each state's first visit: mean 0.11 ms, worst 2.0 ms per token, against a decode
step of about 12–15 ms on the CPU, so under 1%. States recur (inside strings, between
items) and their masks are cached. Indicative only, on battery.

**Remaining for structured output.** `pattern`, `format` and numeric bounds; a schema
for `additionalProperties`; duplicate-key prevention in JSON mode; a compiled-grammar
cache across requests with the same schema; `/v1/models` capability flags (step 3).

**Next milestone.** Steps 3–5 (registry capabilities, linear detokenization, Llama 3.x)
are small; step 7 (P3, unified multimodal input) starts the multimodal path, ADR first.

### Milestone 3 (2026-11-15): registry capabilities (P1, roadmap step 3)

**Implemented.** Architectures declare what they execute (`ArchCapabilities`: weight
formats, inputs, positions, RoPE scalings, norms, attention and FFN options). Validation
refuses tensors in undeclared formats before any weight is read; backends declare their
formats without a device, and engine load refuses a format the chosen backend or the CPU
lacks, before reading weights. `qubivex models architectures` lists all of it.

**Files.** `qubivex-core/src/dtype.rs` (`DType::all`). `qubivex-model`: `registry.rs`
(`ArchCapabilities`, `ArchitectureFactory::capabilities`, `ArchDescriptor`),
`arch/common.rs` (shared declaration; RoPE check reads it), `arch/transformer_{dense,moe}.rs`,
`metadata.rs` (format refusal). `qubivex-backend-vulkan/src/lib.rs` (`supported`).
`qubivex-runtime`: `placement.rs` (`BackendKind::ALL`, `supports`, `weight_formats`),
`engine.rs` (load refusal). `apps/cli/src/models.rs` (listing). Tests:
`qubivex-model/tests/metadata.rs`, `qubivex-runtime/tests/capabilities.rs`, a unit test
in `models.rs`.

**Decisions.** ADR-003 amendment 4.

**Tests.**
- A GGUF MoE with four Q5_0 attention tensors and one Q4_1 expert is refused at
  validation with one line per format (count, first tensor, formats run); the same
  model in declared formats loads (positive control).
- Every built-in declares f32 and not Q5_K, text input, `default` RoPE only.
- Declared formats equal every backend's kernels for every dtype (CPU, Sim, Metal,
  Vulkan).
- Engine load with a backend lacking Q8_0 refuses the `olmoe-tiny-gguf` fixture, naming
  Q8_0 and what the backend runs; every real backend accepts it.
- The CLI listing shows each architecture's capabilities and each backend's formats.

**Performance.** None: one pass over tensor descriptors at validation and at load.

**Remaining.** Capability flags in `/v1/models` (format support, `response_format`);
preprocessing capabilities when multimodal input arrives (P4); new quantised kernels
(Q4_0, Q5_K, ...) are a tensor-engine item and now show up as one clear refusal.

**Next milestone.** Roadmap step 4: linear detokenization, then step 5, Llama 3.x.

## 6. Baseline

| Date | Commit | Command | Result |
|---|---|---|---|
| 2026-11-15 | `8a70709` | `cargo test --workspace --no-fail-fast` | 573 passed, 0 failed, 6 ignored |
