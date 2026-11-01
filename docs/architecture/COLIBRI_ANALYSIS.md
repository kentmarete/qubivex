# Colibri: Repository Analysis

Phase 0 deliverable. Source inspected: `github.com/JustVugg/colibri`, `main` at
`0a5504a1` (release line 2.1.0), cloned 2026-11-01. Licence: Apache-2.0 (porting
algorithms, kernels and shaders is permitted with attribution in `NOTICE`).

File and line references are relative to the Colibri checkout (`c/olmoe.c:1367`).

---

## 1. What Colibri is, in numbers

| Measure | Value |
|---|---|
| Files (excluding `.git`) | 1,697 |
| C / CUDA / ObjC / Python lines under `c/` | ~435,000 |
| Engines (one `.c` per model family) | 13 (10 LLM, 1 image, 2 decision) |
| Largest engine | `deepseek_v4.c`, 22,741 lines |
| Smallest LLM engine | `olmoe.c`, 4,105 lines |
| Tests under `c/tests/` | 652 files (C unit tests, Python harnesses, CUDA benches) |
| Python control plane | `coli` (4.3k), `resource_plan.py` (3.9k), `family_registry.py` (2.3k), `setup_hw.py` (1k), `openai_server.py` |

Colibri is mature, heavily measured and heavily optimised. It is **not** a small codebase
to port line by line. QUBIVEX should take its *mechanisms and invariants*, not its code
structure.

## 2. Current architecture

```
 ┌──────────────── Python control plane ─────────────────┐
 │ coli (launcher/CLI)  setup_hw.py  family_registry.py  │
 │ resource_plan.py (coli plan)  doctor.py  autotune.py  │
 │ openai_server.py (OpenAI + Anthropic + System One)    │
 │ mcp_server.py                                         │
 └──────────────┬────────────────────────────────────────┘
                │  stdin/stdout line protocol (READY, CAPS, STAT,
                │  HWINFO, TIERS, EMAP, SUBMIT, CANCEL …) — docs/serve_protocol.md
 ┌──────────────▼────────────────────────────────────────┐
 │ Engine process: one binary per family                 │
 │  colibri.c (GLM)  qwen36.c  qwen38.c  deepseek_v4.c   │
 │  kimi_k3.c  mimo.c  inkling.c  olmoe.c  …             │
 │      │ #include (header-only shared code)             │
 │  st.h gguf.h quant.h idot.h expert_ffn.h tok.h        │
 │  tier.h route_trace.h kv_prefix.h pin_pool.h park.h   │
 │  uring.h hybrid_split.h serve_budget.h …              │
 │      │ optional, compile-time (CUDA=1 VK=1 METAL=1)   │
 │  backend_cuda.cu  backend_vulkan.c  backend_metal.mm  │
 │  vk_tier.c cu_tier.c xtier.h (shared device tier)     │
 └───────────────────────────────────────────────────────┘
```

Key structural facts:

1. **The engine is a subprocess.** Python owns HTTP, model resolution, planning and
   setup. Each engine is a separate executable that speaks a text-plus-framed-payload
   protocol over pipes. Forward compatibility rule: ignore unknown line kinds.
2. **Shared code is header-only and `#include`d into each engine.** There is no shared
   library or link-time interface for the inference path. Each engine is a single
   translation unit (DeepSeek V4 uses a multi-unit "amalgamation").
3. **Backends are compile-time switches.** `COLI_VULKAN`, `COLI_CUDA` and `COLI_METAL`
   are preprocessor flags, with `#ifdef` blocks inside every engine. Runtime enablement
   uses environment variables (`COLI_VULKAN=1`, `COLI_VK_CHAIN=0`).
4. **Configuration is overwhelmingly environment variables.** `docs/ENVIRONMENT.md`
   and `docs/SETTINGS.md` list hundreds (`PILOT`, `HOT`, `WIDE`, `SMOOTH`, `CONF_LIMIT`,
   `PIPE`, `URING`, `DIRECT`, `PIN_GB`, `RAM_GB`, `CAP_RAISE` and many more).

## 3. Model implementations

| Engine | Families served | Notes |
|---|---|---|
| `colibri.c` | GLM-5.2/5.3, Mistral-Large-4 | MLA attention, sigmoid + noaux_tc routing, shared expert, MTP head, the reference engine for most optimisations |
| `glm53.c` | GLM-5.3-Flash | vision |
| `qwen36.c` | Qwen3.6, Qwen3-Coder, Qwen3.8-27B dense, Flint, Muse Spark, LFM3-VL, Clef | DeltaNet hybrid layers, MTP, many checkpoints on one engine |
| `qwen38.c` | Qwen3.8-Flash-Next | n-gram tables, MTP, int4 sidecar |
| `deepseek_v4.c`, `deepseek_v41.c` | DeepSeek V4 / V4.1 Flash | sparse attention, mHC, the only user of `expert_store.h` |
| `kimi_k3.c`, `mimo.c`, `inkling.c` | Kimi K3, MiMo-V2.6, Inkling | |
| `olmoe.c` | OLMoE | smallest, most self-contained: MHA + QK-norm, softmax top-8 of 64, no shared expert |
| `qwenimage.c`, `laya.c`, `gliner_decide.c` | image / decision models | out of QUBIVEX scope |

Model resolution: `family_registry.py` keys families on `config.json`'s `model_type`.
Each family declares capabilities, limits, display variants and a **per-family Python
geometry function** (`_olmoe_geometry`, …) that computes byte sizes for planning.
`coli` then maps the family to an engine binary (`engine_for`, `c/coli:433`).

So Colibri already avoids `if name.contains("qwen")`, but per-family *code* is still
needed in three places for each new family: the engine `.c`, the registry entry with
its geometry function, and family-specific branches in the planner.

## 4. Shared inference components

| Header | Responsibility | Port to QUBIVEX? |
|---|---|---|
| `st.h` (1.6k) | SafeTensors index + typed reads, multi-shard | Re-implement (Rust `safetensors` crate for headers; own positional reader) |
| `gguf.h`, `st_gguf.h`, `ggml_quants.h` | GGUF reader, ggml block types IQ1_S…Q8_0, MXFP4 | Re-implement reader; port block kernels selectively |
| `quant.h`, `idot.h`, `qgemv.h`, `fused_simd.h` | Container decoders, integer dot kernels (AVX2/SSE4.1/NEON) | Port the hot kernels to Rust `std::arch` |
| `expert_ffn.h` (925) | Routed expert SwiGLU kernel | Port |
| `expert_store.h` | Vtable store with **lease contract** (`lookup` → `release`, advisory `prefetch`, `stats`) | **Adopt the contract** as the Rust trait |
| `expert_store_registry.c` | Name → open fn, registered by linker constructors, chosen by env | Replace with typed Rust registry |
| `tier.h` | Promotion hysteresis and LFRU scoring | **Port verbatim** (pure arithmetic, tested) |
| `hybrid_split.h` | Measured-rate split of experts between GPU upload and CPU compute | **Port verbatim** |
| `route_trace.h` | Routing counts, `.coli_usage` history, `ROUTE_TRACE` stream | Port the concept; new format with versioned header |
| `kv_prefix.h`, `pin_pool.h`, `park.h` | KV prefix reuse, nested prefix snapshots, parking a conversation | Later phase (Phase 10) |
| `uring.h` | Minimal io_uring reader | Use a Rust io_uring crate behind a feature flag |
| `serve_budget.h` | max_tokens vs context budget | Port (trivial, but contains hard-won semantics) |
| `tok.h` | BPE tokenizer in C | Replace with Hugging Face `tokenizers` |

## 5. Hardware detection

Detection lives in **Python** (`setup_hw.py`), separate from the engines:

- RAM from `/proc/meminfo` (MemAvailable), `sysctl` + `vm_stat` on macOS,
  `GlobalMemoryStatusEx` on Windows. Unknown is reported as unknown, not as zero.
- CPU from `/proc/cpuinfo` and Windows `IsProcessorFeaturePresent`, with SIMD feature
  flags checked against what the prebuilt binary needs.
- NVIDIA via `nvidia-smi` plus an in-process CUDA driver probe **run in a child process
  with a timeout** (a broken driver cannot hang setup).
- Vulkan via an in-process `libvulkan` probe, also in a child process, with
  `VK_EXT_memory_budget`, integrated/discrete type, compute-unit count and
  cooperative-matrix support.
- Disk free space, WSL `/mnt/c` and 9p warnings.
- cgroup v1/v2 memory limits (`resource_plan.py`), so containers are budgeted correctly.
- Hybrid-CPU performance-core detection (P-cores only for the OpenMP team).

Engines also detect at runtime: `mem_available_gb()` and `rss_gb()` in each engine,
`cpu_check.h` to fail fast on missing ISA, and `HWINFO` emitted to the gateway.

**Preserve:** child-process GPU probes with timeouts, cgroup awareness, P-core-only
thread teams, and "unknown ≠ zero".

## 6. Memory planning

There are **two planners**, and they do not share code:

1. **`resource_plan.py` (`coli plan`)** analyses tensors (`analyze_model`, cached in
   `.coli_analysis.json`), separates dense from routed-expert bytes (regex
   `layers.N.(mlp|ffn).experts.M.`), prices KV and device copies, and emits
   hot (VRAM), warm (RAM) and cold (disk) tiers with reasons and `next_actions`.
   Policies: `quality` and `balanced`. It contains many family-specific branches
   (`_q38_cpu_dense_tensors`, `_glm53_dense_tensors`, `_olmoe_chain_layout`, …).
2. **Each engine's own budget at load time.** In OLMoE (`c/olmoe.c:837`):
   - `budget = RAM_GB` if set, otherwise `resident + 0.88 × MemAvailable`, measured
     *after* the dense weights are resident (a measurement, not a projection);
   - `room = budget − resident − 0.5 GB` (activations);
   - `slots_per_layer = room / slot_bytes / n_layers`, clamped to `[1, n_experts]`;
   - **the KV cache and the expert cache share `room`**: `kv_room_fit()`
     (`c/olmoe.c:1462`) shrinks every layer's cap as the KV reaches new pages,
     evicting LRU, unpinned and non-busy slots. Measured motivation: a fixed cap of 8
     gave 22.8% hit rate and 0.045 tok/s; the derived cap 64 gave 99.4% and 0.215 tok/s.

**Preserve:** measuring after the dense load, KV/expert-cache room sharing, never
reserving the full-context KV up front, and the 1-slot-per-layer floor (always correct,
just slow).

**Redesign:** one planner in the runtime used by both `plan` and load; no
family-specific branches (derive everything from the architecture's tensor roles).

## 7. Expert storage, caching and prefetch

### 7.1 Storage layout
- OLMoE's native container (`tools/convert_olmoe_merged.py`) packs each expert's
  gate, up and down as **one contiguous int8 tensor** (`…experts.E.merged_weight`) plus
  one f32 scale array (`…experts.E.qs`), with per-row symmetric quantisation. One expert
  is one `pread` plus a small scale read.
- GLM reads "an expert's three matrices in one `pread`" from its int4-g64 container.
- GGUF experts are read as stored (`load_expert_gguf`). They are optionally repacked
  with rows interleaved by 16 for the CPU kernel.
- **Untrusted-container validation:** every expert read checks that the tensor's
  byte size exactly matches the config-derived size, and refuses to run if not
  (`c/olmoe.c:1150`). Without this check a crafted header causes a heap overflow.

### 7.2 Cache (OLMoE, `c/olmoe.c:120–1460`)
- Per-layer `LCache` of fixed-size **slots** (one slab per slot, allocated once, reused).
- `slot_by_expert[]` gives an O(1) index; slot `eid = -1` means a load is in flight.
- `busy` refcount (a lease): eviction never takes a busy slot. Before this existed,
  the prefetcher could overwrite the slot being multiplied.
- **In-flight deduplication:** `expert_get` waits on a condvar for an in-flight load
  of the same expert instead of reading it twice.
- Victim selection: LRU over unpinned, non-busy, non-in-flight slots. When a device
  tier holds a copy, that slot is evicted first, so RAM and VRAM avoid a second copy.
- Learned hot set: `HOT=N` pins the top-N experts per layer after `WARMUP` tokens.
  Pins persist to `hot_pinned.bin`, and history is kept in `.coli_usage`.
- Tier policy (`tier.h`): promote only if `hot > cold + cold/4 + 4` (25% plus 4
  hysteresis, overflow-safe). LFRU score is `heat<<8 | recency(≤255)`, so recency only
  breaks ties. Heat decays by halving.

### 7.3 Asynchronous loading
- GLM: `PIPE` loader pool on by default (overlaps `pread` with matmul, −18% disk
  service time), `URING=1` batches on Linux, `DIRECT=1` O_DIRECT (+65% on one Strix Halo
  host, neutral or worse on others), `COLI_MODEL_MIRROR` reads from a second drive with
  weighted striping (+37.5% decode on two independent NVMe drives), `COLI_NUMA=1`
  interleave, `mlock` for pinned sets.
- OLMoE: synchronous demand loads plus one PILOT prefetch worker.

### 7.4 Router-lookahead prefetch (`PILOT`)
`pilot_prefetch` (`c/olmoe.c:2368`):
1. Apply layer L+1's `post_attention_layernorm` and router to layer L's hidden state.
2. Blend with a per-layer EMA of router logits ("momentum", `SMOOTH=0.3`).
3. Take candidates in probability order until the cumulative mass reaches
   `CONF_LIMIT=0.92` (at least top-K, at most `K × WIDE`).
4. Sort candidates by expert id (SSD locality) and enqueue them in a fixed 4096-entry
   ring with an `is_queued` bitmap for deduplication.
5. Worker `pilot_realload` checks an **LFRU eviction guard**: a speculative load must
   not evict a resident expert unless it scores 25% + 4 higher.

It never changes routing: the EMA is only used for prediction. Measured: GLM-5.2's
next-layer routing is 71.6% predictable (against 41.3% for "same as last token"). On
disk-saturated hosts hint-only PILOT can be net negative.

### 7.5 Weaknesses in this area
- The cache is reimplemented per engine (OLMoE `LCache`, GLM `ecache`, DSV4 behind
  `expert_store.h`). The clean lease interface is used by **one** engine.
- Global mutable state: `g_pilot_mx`, the static `pilot_q[4096]` ring and `pilot_m`
  are process-wide. Only one model is possible per process.
- Busy-wait loops: the worker polls with `sleep_ms(1)`, and eviction waits with
  `sleep 1ms` and retry.
- `exit(1)` on OOM and on malformed containers. A server cannot survive a bad request
  path.
- The cache is sized in *slots per layer*, uniform across layers, rather than by
  per-layer measured demand.

## 8. MoE execution and numerical discipline

- Decode (S=1): for each routed expert, `expert_get` → SwiGLU → `expert_put`, then
  accumulate `w × h` in rank order.
- Prefill (S>1): `moe_by_expert` (`c/olmoe.c:1790`) groups (row, rank) pairs by expert
  with a counting sort in blocks of 128 rows. Each expert is fetched once per block and
  multiplied against all its rows. **The final sum is done in rank order, so the bits
  equal the one-row loop's.**
- Device tier: the GPU computes resident experts and the CPU computes the rest in the
  same frame. The output is still summed in rank order, so **the order of summation never
  depends on residency**.
- Lossy options are opt-in and named, with warnings: `IDOT` (Q8_0 activations),
  `FUSED3`, `--topp`, `CACHE_ROUTE`, `DEGRADE_ZERO`. A past bug where `FUSED3` silently
  enabled the lossy path was found by greedy-run divergence and fixed.
- NaN-safe top-K (`rt_router_pick`) prevents a −1 index from reaching the cache.

This is the core invariant QUBIVEX must keep: **placement changes speed, never bits**
(on a given backend and kernel set).

## 9. GPU execution

- **CUDA:** `backend_cuda.cu` is one source compiled for CUDA or HIP through
  `backend_gpu_compat.h` (a 14-symbol shim). It provides a routed-expert tier
  (`cu_tier.h`), dense matrices on the device (`cu_dense.h`), whole-layer chains
  (`cuda_chain.cu`), and GGUF block kernels (`cuda_bd.cuh`) using tensor cores for
  prefill. On Windows it is loaded as a DLL.
- **Vulkan:** `backend_vulkan.c` (7.9k) plus GLSL shaders. It has an expert tier
  (`vk_tier.c`), a "dense chain" that records one submission per layer, a staging path
  for cards without Resizable BAR, and a device host-pool for experts the card cannot
  hold.
- **Metal:** `backend_metal.mm` (2.6k), expert math on unified memory.
- `hybrid_split.h` decides how many missing experts to upload versus compute on the
  host: `q* = m × B_P / B_H`, with EMA-measured rates. It falls back to the historical
  behaviour while rates are unmeasured.
- Measured trade-off: on an integrated GPU, OLMoE *loses* with Vulkan (23.1 → 12.8
  tok/s). It wins on a discrete card (CUDA 354.6 tok/s against 47.4 on CPU, i5-13500 +
  RTX PRO 4000). Backend selection must be measured per model and device, not assumed.

## 10. Tokenisation, KV cache and serving

- Tokenizer: a C BPE (`tok.h`) with per-family Unicode tables. Chat templates are
  handled per family in Python and C.
- KV: f32 per layer (OLMoE), compressed MLA latent (GLM, 57× smaller), fp8 and
  PolarQuant options, f16 KV, KV persistence across restarts, prefix reuse, multi-slot
  (`--kv-slots N` ≤ 16) continuous batching via the mux protocol.
- API: OpenAI (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, streaming,
  logprobs, stop sequences), Anthropic (`/v1/messages`), System One (`/v1/systemone`),
  image generation, MCP server.

## 11. Numerical validation and tests

- **Tiny-fixture oracles in CI:** pinned-dependency Python scripts
  (`tools/make_*_tiny.py`, `make_glm_oracle.py`) build tiny random checkpoints with the
  vendor `transformers` implementation and record greedy token ids. The C engine must
  reproduce **the same token ids** (`ref.json`, `ref_olmoe_real.json` for the real
  OLMoE). `TF=1` teacher-forcing validates the prefill over the whole sequence.
- GPU parity: CI runs each engine's Vulkan path against the CPU's tokens on a
  software Vulkan driver, with drift allowed and documented (`docs/gpu-parity.md`).
- Sanitizer builds (ASan, UBSan) for the expert tier and chain.
- Unit tests for cache indexing (`test_olmoe_cache_index.c` with an injectable load
  hook), `moe_by_expert` bit equality, `matmul_q` IDOT versus FP32, reference bounds,
  container validation, serve framing and the store-registry contract.
- Benchmarks: `tools/benchmark_*.py`, `docs/benchmarking.md` protocol (record
  hardware, commit, command, cache state, hit rate, bytes read, quality).

**Preserve:** tiny vendor-generated fixtures in CI, token-id equality for greedy
decoding, teacher-forcing logits, explicit tolerance for cross-backend drift, and the
benchmark protocol.

## 12. Optimisation techniques worth carrying over (ranked by measured value)

1. Experts cached in RAM sized from measured free memory (5× on OLMoE prefill).
2. One contiguous read per expert, with in-flight deduplication and busy leases.
3. Expert-major prefill batching (`moe_by_expert`): one fetch per expert per block.
4. Async load pool overlapping I/O with compute (−18% disk service).
5. Router-lookahead prefetch with a confidence cutoff and an eviction guard.
6. Learned hot set with hysteresis (repeat workloads).
7. Multiple-drive striping (+37.5% on independent controllers).
8. O_DIRECT and io_uring, **only where measured**.
9. Hybrid GPU/CPU split from measured rates.
10. P-core-only thread teams on hybrid CPUs.

## 13. Areas to preserve and areas to redesign

| Preserve | Redesign |
|---|---|
| "Placement decides speed, never bits" invariant | One binary per family, header-only sharing → crates with typed interfaces |
| Lease-based store contract (`expert_store.h`) | Cache reimplemented per engine → one `qubivex-expert-stream` |
| Rank-order summation independent of residency | Global mutable state → per-model runtime objects; multiple models per process |
| `tier.h` / `hybrid_split.h` arithmetic | Busy-wait polling → condvars and bounded channels |
| Measured, not assumed, budgets; KV/cache room sharing | `exit(1)` on errors → typed errors; reject the request, not the process |
| Exact-size validation of untrusted containers | Hundreds of env vars → typed config file + CLI flags + env override layer |
| Tiny-fixture oracle CI with token-id equality | Python/C split planner → one Rust planner used by `plan` and load |
| Child-process GPU probes with timeouts | Family-specific planner branches → tensor-role-driven planning |
| Lossy paths opt-in, named and warned | Engine as subprocess with a text protocol → in-process runtime behind a Rust API |
| Benchmark protocol and negative results | Model support via a new C engine → manifest + registered architecture components |

## 14. What Colibri has that QUBIVEX's specification does not mention

These are real features. They are deliberately **out of scope** until the core runtime
works, and listed so the omission is a choice: MTP and speculative decoding, KV
persistence, conversation parking, System One decision scoring, an Anthropic-compatible
API, vision towers, image generation, local cluster mode, LoRA adapters, grammar-forced
drafts, and XDNA NPU support.
