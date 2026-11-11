# ADR-011: Device execution, buffer lifetimes and GPU parity

- Status: **Accepted (2026-11-10)** by owner decision, for the parity contract and
  amendments 1 to 6. Amendment 7 (2026-11-11, NaN payloads and signs) by owner decision
  with Vulkan Step 3a. The CPU stays the default backend; `--backend metal` is an opt-in.
- Date: 2026-11-03
- Related: ADR-007 (backends; amendment 2026-11-03), ADR-010 (lossless by default),
  `docs/engineering/ARCHITECTURE_DECISIONS.md` §2.4–2.5, TENSOR_ARCHITECTURE §6.

## Context
Phase 8 starts with Metal (`qubivex-backend-metal`) because the dev machine is an M3 Pro
(ADR-007 amendment). ADR-007 left these open until the first GPU backend: how close GPU
results must be to the CPU's, how device work is submitted and synchronised, and how buffers
stay alive while the GPU uses them. Slice 1 built the kernels and measured them, so those
questions can now be answered from evidence.

What slice 1 found on the M3 Pro (Apple GPU, 14 cores, unified memory):
- **Bit identity is achievable.** Each Metal kernel reproduces the CPU's dequantised weight
  values and its fixed 64-lane reduction shape: a 32-thread SIMD group, thread `t` holding
  lanes `t` and `t + 32`, with the halving tree done by `simd_shuffle_down`. Shaders compile
  from source at start-up with `MTLMathModeSafe`, and `#pragma METAL fp contract(off)`, so
  `a * b - c` stays two rounded operations. Every output is bit-identical to the CPU for all
  six dtypes, every tail shape and batch sizes 1, 3 and 8. On the real OLMoE Q4_K_M model,
  all 130 weight tensors (3.92 GiB, about 11M outputs) are bit-identical.
- **The one divergence: Apple GPUs flush f32 subnormals to zero**, in safe math mode too,
  and Metal has no option to keep them. A weight, activation, product or partial sum below
  1.18e-38 in magnitude can therefore differ. At model magnitudes this does not happen
  (the real-model test shows none), and f16 subnormals are unaffected because they are
  normal in f32.
- **Per-call synchronisation dominates small matrices.** A synchronous command buffer per
  matmul costs about 120 µs; an expert projection (1024 × 2048 Q4_K) takes 176 µs on Metal
  against 63 µs on 5 CPU threads. The 50304 × 2048 Q6_K output head takes 1.78 ms against
  2.77 ms.

## Decision
1. **Parity contract: bit-identical, except for f32 subnormals.** A GPU backend must give
   the CPU's bits for every output whose inputs, products and partial sums stay in the
   normal f32 range. This replaces ADR-007's "documented drift tolerance" for Metal. Any
   kernel that changes the reduction shape (for example `simdgroup_matrix` tiles) is a
   lossy opt-in under ADR-010, not a default. The parity tests (`tests/parity.rs` and the
   env-gated `qubivex-runtime/tests/metal_parity.rs`) are the gate for every Metal kernel
   change. CUDA and Vulkan get their own contract when they are built; the same target
   applies unless a measurement shows it cannot be met.
2. **Submission.** Device work is encoded into command buffers, and many operations go in
   one buffer: at least one layer, eventually one token. Each submission returns an `Event`
   that completes when the GPU finishes. Slice 1's synchronous per-call `matmul` is a
   correctness path only; the engine must not use it per expert.
3. **Lifetimes.** Every buffer an in-flight operation touches is held by a
   reference-counted handle that the operation owns until its event completes (the `Arc`
   rule of TENSOR_ARCHITECTURE §6). Expert leases move into the operation; their
   `Arc<[u8]>` already allows this. No Rust `async` on the compute path (ADR-006).
4. **Buffers.** An opaque `DeviceBuffer` handle (an `Arc` of backend state) keeps
   `ComputeBackend` object-safe. On unified memory, weights are wrapped in place
   (`newBufferWithBytesNoCopy` over the page-aligned mapping) instead of copied, and the
   ledger is charged once, for the host bytes. Copies are charged as device bytes.
5. **Placement.** Which device runs which operation is fixed in the execution plan from
   measured costs (ADR-007 amendment). Slice 1 shows the plan needs the per-submission cost,
   not only kernel bandwidth: with today's numbers, only the output head wins on Metal.
6. **Shaders compile at start-up from source.** No offline Metal toolchain is required
   (the dev machine has Command Line Tools only). Compilation takes well under a second; a
   cached `.metallib` can be added later if start-up time matters.

## Consequences
- Metal output keeps the reproducibility invariant (invariant 1) instead of a tolerance, so
  CPU and GPU runs can be compared token for token.
- The fastest Apple GPU matmul paths (simdgroup matrices, fast math) are excluded from the
  default; any of them comes as a named, measured, lossy opt-in.
- Slice 2 must build command-buffer batching and zero-copy weights before the engine can
  gain from Metal; the kernels alone do not give a speed-up for expert-sized matrices.
- A Q4_K kernel that dequantises per element reaches only about 21 GiB/s of kernel time on
  expert shapes; a block-reusing kernel (same bits) is the next kernel step, and prefill
  needs a kernel that reuses dequantised weights across activation rows.

## Amendment 1 (2026-11-03): batching, zero-copy weights and placement in the engine, Phase 8 slice 2

### What was built
- **Batched submission (§2).** `ComputeBackend::matmul_many` takes independent matmuls as
  one submission (the default loops over `matmul`, so the CPU is unchanged). Metal encodes
  the whole batch in one command buffer, stages each distinct input once and reuses its
  staging buffers between submissions. The forward pass submits q, k and v together, and
  each expert's gate and up together.
- **Zero-copy weights (§4).** The engine reads every dense tensor into one page-aligned
  arena (`qubivex_compute::host::PageAligned`: an over-allocated `Vec`, aligned inside,
  so no unsafe code). `MetalBackend::map_host` wraps it with
  `newBufferWithBytesNoCopy`. Each placed matrix is an offset into that one buffer, and
  the buffer holds the arena's `Arc` (§3). The ledger charges the dense tensor bytes once,
  as before; the arena's alignment slack (under 40 KiB for OLMoE) is not charged.
- **Placement (§5).** `EngineOptions::{backend, metal_ops}` and the CLI's `--backend
  cpu|metal` and `--metal-ops attention,router,head`. Placement is fixed at load, by
  weight: the dense matmuls of the chosen kinds run on Metal, and everything else
  (embedding lookup, norms, attention, experts) runs on the CPU. `Engine::placement()`
  reports it, and `qubivex bench` records it as `config.device`.
- **Token parity.** The fixture invariant `metal_placement_never_changes_bits` checks
  every placement on all three fixtures. The env-gated `engine_on_metal_gives_the_cpu_tokens`
  places all 81 dense matmul weights of OLMoE on Metal (q, k, v, o and router for each of 16 layers, plus the head): the 10 × 50304 prompt logits are
  bit-identical, and so are the 48 generated tokens. Every `qubivex bench` run below
  matches the Phase 7 reference tokens.

### Measurements (M3 Pro, OLMoE Q4_K_M, all experts resident, 160 tokens, 3 runs)
Kernel level (Criterion, resident weights):

| | time |
|---|---|
| 64 × 64 f32, one submission (pure overhead) | 109 µs (slice 1: 120 µs; staging reuse) |
| q, k, v (3 × 2048 × 2048 Q4_K) as 3 submissions | 727 µs |
| the same in one command buffer | 424 µs |

Polling the command buffer's status instead of `waitUntilCompleted` did not lower the
109 µs (113 µs), so that cost is on the GPU side.

End to end (`qubivex bench`, records `docs/performance/baselines/2026-11-03-m3pro-phase8-*.json`):

| `--backend` / `--metal-ops` | decode tok/s (median step) | median step | prefill tok/s |
|---|---|---|---|
| cpu | 25.51 | 39.19 ms | 32.9 |
| metal / head | 25.07 | 39.90 ms | 32.6 |
| metal / attention,head | 22.16 | 45.13 ms | 30.1 |
| metal / attention,router,head | 21.01 | 47.60 ms | 29.6 |

An interleaved A/B of the CPU path against `main` (26.05 and 26.05 tok/s, then 26.05
and 26.13) shows the dense arena costs nothing.

**Why the head does not win in the engine.** In a tight loop the Q6_K head takes 1.78 ms
on Metal against 2.77 ms on the CPU. Timed inside decode, the Metal head takes a median
3.99 ms against the CPU's 3.18 ms. The GPU is idle for about 36 ms between heads, and a
lone submission after an idle period runs slower than one in a busy stream. Attention
loses for the reason slice 1 predicted: 5 submissions per layer at about 110 µs each
cost more than the kernels save.

### Decision
- **The CPU stays the default backend.** `--backend metal` is a correct, bit-identical
  opt-in. Its default placement is `head`, the only candidate whose kernel beats the CPU
  by more than one submission's cost.
- **A Metal speed-up needs the GPU kept busy, not more placed matmuls.** That means one
  submission per layer or per token, which needs norms, RoPE, attention and the expert
  FFN as Metal kernels with the same bit contract. Then the per-submission cost is paid
  once per token, and the GPU stays clocked up. That work, and the block-reusing Q4_K
  kernel, is slice 3.
- The scratch staging buffers (at most a few MiB) are not charged to the ledger yet.
  They will be once device tiers are charged per ADR-005.

## Amendment 2 (2026-11-03): whole layers on the GPU, Phase 8 slice 3

### Portable numerics (a change to the CPU's bits)
Three CPU operations could not be reproduced on Metal: `rms_norm` summed squares in f64
(Metal has no f64), and softmax, SwiGLU and RoPE used the platform's `expf`, `sinf` and
`cosf`, which also differ between macOS and Linux. The owner chose to make the CPU
portable rather than emulate f64 and libm on the GPU, or accept a tolerance:
- `qubivex_compute::math::exp`: Cody–Waite reduction and the Cephes polynomial, built
  from IEEE basic operations and fused multiply-add only, within 2 ulp; results that
  would be subnormal are `+0`, so flush-to-zero cannot make them differ.
- `rms_norm` sums squares with the shared 64-lane reduction (moved to
  `qubivex_compute::reduce`), then `1 / sqrt(ss / n + eps)` in f32.
- RoPE takes a per-position cos/sin table, evaluated in f64 and rounded once (the
  correctly rounded value except in rare near-ties), built on the host and uploaded.

The CPU's output bits changed once. On real OLMoE Q4_K_M the 128 bench tokens equal the
previous reference and Colibri's 12 reference tokens still match. On the tiny fixtures
the worst error against transformers went from 7.2e-7 to 6.5e-7, and against the f64
reference from 3.3e-7 to 2.8e-7. A side effect: the bits no longer depend on the
platform's libm.

### What was built
- **Forward-pass kernels** (`kernels.metal`, `program.rs`): `rms_norm`, `rope`, clamp,
  add, copy, `swiglu`, `exp`, `kv_store`, attention (scores, softmax, values), the MoE
  combine, and row gather/scatter. Each mirrors `qubivex-arch`'s ops operation for
  operation. **Every division and square root is `precise::`**: with plain `/` and
  `sqrt`, the compiler turned `1 / sqrt(x)` into a reciprocal square root that is not
  correctly rounded (1 ulp off). `tests/ops_parity.rs` checks every op bit for bit.
- **Recorded submissions (§2).** `MetalBackend::submit` records any number of
  operations over device vectors (`DeviceVec`, `Span`, `Batch`) into one command buffer.
  Each operation checks its bounds before it is recorded.
- **Block-wise K-quant kernels.** Per 64-element chunk, the scales are computed once
  and each quant byte is loaded once for both of a thread's lanes; two rows per SIMD
  group share activation loads; Q4_K reads each block's 16-byte header as one `uint4`
  (so Q4_K weights must be 16-byte aligned, and a misaligned matrix is not placed).
  Lanes and order are unchanged, so the bits are unchanged (all 130 OLMoE tensors).
  Throughput, 96 distinct expert matrices in one submission: Q4_K 23 → 38 GiB/s (48 in
  a single dispatch), Q6_K 38 → 64 GiB/s (90); the CPU does about 17 GiB/s.
- **Whole layers (`--metal-ops layers`, now the default placement of `--backend
  metal`).** `qubivex-runtime`'s `MetalForward` makes each layer one submission: the
  previous layer's routed experts (gather, gate, up, SwiGLU, down, scatter), the
  combine and residual, then this layer's attention block, MoE norm and router. Only the
  router logits return to the host, where routing runs with the CPU pass's own
  `ops::route` and the store leases the chosen experts. The last submission adds the
  final norm and the head. The embedding lookup stays on the host.
- **Expert slabs in place (§4).** Expert slabs are now page-aligned (`Slab`: an
  `Arc<PageAligned>` and an attachment slot). The first GPU use of a loaded expert wraps
  its slab with `newBufferWithBytesNoCopy` and keeps the handle in the attachment; the
  store drops the attachment before it reuses the slab, so a handle never outlives its
  expert. When a layer is routed to more experts than its pool holds, they run in waves
  of at most the pool's slots (the store hands out more leases only as earlier ones are
  released).
- **Device KV.** Keys and values are position-major (`[pos][kv_head][d]`) device
  buffers in the `KvCache`'s device slot. They grow by the host cache's 256-position
  chunks (a prefix copy), and their bytes are charged to the ledger's KV owner.
- GPU tests now skip only when no Metal device exists; a shader that fails to compile
  fails them (it used to skip silently).

### Parity
The fixture invariant `metal_placement_never_changes_bits` covers the layers placement.
`metal_layers_match_the_cpu_in_waves_and_across_kv_growth` runs every fixture with
one-slot pools (a prompt needs many waves) through 300 decode steps past the 256-position
chunk: every step's logits are bit-identical. On real OLMoE (env-gated), the 10 × 50304
prompt logits are bit-identical and so are the 48 generated tokens.

### Measurements (M3 Pro, OLMoE Q4_K_M, 10 prompt + 128 generated tokens, 5 runs)
Records: `docs/performance/baselines/2026-11-03-m3pro-phase8s3-*.json`. Every run's
tokens match the CPU reference.

| | decode tok/s (median step) | median step | P95 step | prefill (10 tokens) |
|---|---|---|---|---|
| cpu, all experts resident | 26.41 | 37.86 ms | 47.80 ms | 300 ms (33 tok/s) |
| metal layers, all resident | **44.47** | 22.49 ms | 23.36 ms | 131 ms (77 tok/s) |
| cpu, `--ram-budget 1GiB` | 20.31 | 49.24 ms | 59.13 ms | 317 ms |
| metal layers, `--ram-budget 1GiB` | 14.66 | 68.22 ms | 70.36 ms | 364 ms |

With every expert resident, whole layers are 1.68× faster at decode and 2.3× at
prefill, and the P95 step is within 4% of the median. Under a 1 GiB budget the pools
hold 2–3 experts per layer, fewer than the 8 routed per token, so each layer runs about
three waves. Each wave waits for all of its loads and pays a submission, while the CPU
pass computes each expert the moment its load lands. Blocked time per token is 29 ms
against 7.5 ms.

### Decision
- **`--backend metal` defaults to whole layers.** Single dense matmuls on the GPU
  (amendment 1) never beat the CPU, and they stay available for comparison.
- **The CPU stays the default backend** until the owner reviews this amendment. The
  measured case for Metal is the resident one, which is what the planner chooses
  whenever RAM allows.
- Not yet done, for the next slice:
  - asynchronous submission (commit without waiting, with events per §2), so expert
    loads overlap GPU work when the pools are smaller than top-k;
  - a concurrent encoder for independent matmuls (serial dispatch costs about 25% of
    kernel time on the expert batch);
  - routing on the GPU, for one submission per token.
- Device scratch (activations and scores; a few MiB at decode) is not charged to the
  ledger yet; the device KV is.

## Amendment 3 (2026-11-03): small expert caches, Phase 8 slice 4

### The question
Amendment 2 left one regime where whole layers lose: an expert cache whose per-layer
pools hold fewer experts than a token routes (2–3 slots against top-k 8 at
`--ram-budget 1GiB`), where decode fell from 20.3 tok/s (CPU) to 14.7 (Metal). Slice 4
tried to make the GPU pass overlap expert loads.

### What was built
- **Asynchronous submission (§2).** `MetalBackend::submit_async` commits a recorded
  batch without waiting and returns a `Pending`. The `Pending` holds every wrapped host
  region its operations bind (the command buffer retains the Metal objects, but not the
  Rust allocations they wrap), and dropping it waits for the GPU, so host memory the GPU
  may read is never released early (§3). Host reads and writes of device vectors first
  drain outstanding submissions (an empty command buffer completes only after every
  earlier one). Tests: two asynchronous submissions run in commit order, and a matmul
  over a wrapped region gives the CPU's bits after the caller drops every handle to the
  region before waiting.

### What was measured, and not kept (M3 Pro, OLMoE Q4_K_M, `--ram-budget 1GiB`)
Every variant gave the reference tokens.

| expert scheme in the whole-layer pass | decode tok/s | blocked per token |
|---|---|---|
| waves of at most `slots` experts (amendment 2) | 14.66 | 29.1 ms |
| one async submission per arriving expert, released when the GPU is done | 9.02 | 62.8 ms |
| the same, releasing GPU groups before waiting on the store | 8.29 | 20.8 ms |
| hits on the GPU, misses on the CPU as they land | 14.66 | 26.6 ms |
| **the CPU pass** | **20.31** | 7.5 ms |

A lone submission after the GPU idles costs about 0.5 ms end to end. With about 117
misses per token, submitting experts as they stream in costs more than the GPU saves.
Holding leases until the GPU finishes also pins pool slots the store needs for its next
reads. Running the misses on the CPU recovers the blocking, but the per-layer GPU round
trips (one per layer, waiting for the router logits) remain. In this regime the CPU
pass, which overlaps each expert's compute with the next loads and has no round trips,
is fastest.

### Decision
- **The plan chooses the CPU pass when pools are smaller than top-k.** If `--backend
  metal` (whole layers) is asked for and the plan gives any layer fewer slots than the
  routed experts per token, the engine runs the CPU pass. `PlacementReport::fallback`
  says why; `qubivex run` prints it, and `qubivex bench` records it in `config.device`.
  At 1 GiB this gives 19.96 tok/s (record
  `2026-11-03-m3pro-phase8s4-metal-1gib-fallback.json`) instead of 14.66. The choice is
  made at load from the plan, as §5 requires; it does not follow live load.
- When memory pressure shrinks the pools after load, the whole-layer pass keeps running
  in waves, with the same bits (`metal_layers_match_the_cpu_in_waves_and_across_kv_growth`
  now shrinks the pools from 8 slots to 1 under critical pressure and checks the prompt
  and 300 decode steps). It is slower than the CPU pass there, but the placement is
  never moved mid-run.
- `submit_async` stays: it is the event mechanism §2 calls for. Overlapping loads with
  GPU work needs routing on the GPU first (no host round trip per layer). With that, a
  token could be one submission that waits on expert availability, which is the next
  step for this regime.

## Amendment 4 (2026-11-04): a token in one submission, Phase 8 slice 5

### The question
With every expert resident, whole layers (amendment 2) still made one submission per
layer, because routing ran on the host: each layer's router logits came back, the host
chose the experts, and the next submission computed them. Amendment 3 named routing on
the GPU as the step that removes those round trips, and amendment 2 named a concurrent
encoder for independent dispatches.

### What was built
- **`moe_route`** (`Batch::moe_route`): one thread per token runs `ops::route` with the
  same operations in the same order: the softmax (max, `qexp`, a sequential sum,
  `precise::divide`), then the top k by `f32::total_cmp` order with ties to the lower id
  (selection instead of a sort; the order it defines is the same), then the optional
  renormalisation (a sequential sum from −0.0, as `Iterator::sum` starts). Ids are
  written as u32 bits in an f32 buffer. Up to 256 experts (`MAX_ROUTED_EXPERTS`).
- **Routed matmuls** (`Batch::routed_matmul`, kernels `routed_<dtype>`): one dispatch over
  every (token, rank) entry. Entry `j` reads its expert id from the route buffer, takes
  that expert's weights from an address table passed inline (each matrix's
  `gpuAddress` plus its offset, as a struct of device pointers), and computes the same
  dot product as the plain matmul kernel, which now shares its row body with it. The
  input row is the token's (gate and up) or the entry's (down); the output row is the
  entry's, which is exactly the `contrib` layout the combine reads, so the gather and
  scatter disappear. Buffers reached only through the table are declared with
  `useResource` (read), and wrapped host memory is kept alive as for bound buffers (§3).
  An id past the table is clamped to the last expert: wrong numbers, never an
  out-of-bounds read.
- **Holding experts without using them.** The GPU chooses experts after the host has
  committed, so the host must guarantee that every expert it could choose stays
  resident: `ExpertStore::hold_layer` leases all of a layer's experts when they are all
  resident (`Held::All`), and otherwise says which are missing or that the pool is too
  small. Holding is not a use: heat, recency, hit counts and the trace are untouched, and
  releasing a hold does not refresh recency. After the submission the host reads the ids
  back and calls `ExpertStore::record_uses`, which counts each layer's distinct experts as
  hits exactly as the per-layer pass's acquires would (same hit totals on the benchmark:
  84 375 in both).
- **`MetalOps::token`** (`--metal-ops token`, now the default; `layers` keeps host
  routing). Per forward pass: if every layer's pool holds all its experts and they are
  resident, the whole pass (all layers, routing, routed experts, final norm and head) is
  one submission. If some are missing, they are queued as prefetches (with I/O threads;
  without, they are read at once) and this pass takes the per-layer path, so the first
  token does not wait for the whole model. Pools smaller than the expert count (the
  amendment 3 regime, or pressure shrinking pools after load) keep the per-layer path.
  `Engine::gpu_routed_passes` counts the passes that ran as one submission.
- **A concurrent encoder.** Command buffers now use `MTLDispatchType::Concurrent` with a
  buffer barrier before each dispatch, which keeps the serial order (measured: no cost).
  `Batch::concurrent` records a group of independent operations without barriers between
  them: q/k/v projections, their clamps and norms, the two RoPEs, the two KV stores,
  routed gate and up, and in the per-layer path each stage (gather, gate and up, SwiGLU,
  down, scatter) across all of a wave's experts.

### Parity
New tests: `moe_route_matches_ops_route` (64/8, 8/2, 60/4, 256/8 and 5/5 experts, with
equal logits, pairs of equal logits and a −inf logit, with and without renormalisation)
and `routed_matmuls_give_each_experts_cpu_bits` (every dtype and shape, typical and wild
values, wrapped and uploaded experts, token and entry inputs, a clamped id). The fixture
invariant runs `token` with and without I/O threads (with none, every one of the 12+
passes is routed on the GPU) and checks `layers` routes none. On real OLMoE (env-gated),
the 10 × 50 304 prompt logits and the 48 generated tokens are bit-identical; 40 of the
passes ran as one submission (the first ones took the per-layer path while experts
loaded in the background). The fault suite runs both placements: with `token` and I/O
threads, a failed prefetch is read again later, so the request may succeed, and its
tokens are the fault-free ones either way.

### Measurements (M3 Pro, OLMoE Q4_K_M, all experts resident, 5 runs)
Records: `docs/performance/baselines/2026-11-04-m3pro-phase8s5-*.json`. Every run's
tokens match the CPU's.

| 10 prompt + 128 generated | decode tok/s (median step) | P95 step | prefill (10 tokens) |
|---|---|---|---|
| cpu | 42.3 | 29.0 ms | 38.7 tok/s |
| metal layers, serial encoder (slice 3 code) | 44.9 | 23.4 ms | 75.8 tok/s |
| metal token, serial encoder | 57.1 | 18.2 ms | 89.9 tok/s |
| metal layers, concurrent groups | 53.1 | 19.7 ms | 86.5 tok/s |
| **metal token, concurrent groups** | **59.2** | 17.5 ms | 91.3 tok/s |

| 128 prompt + 32 generated | prefill tok/s |
|---|---|
| cpu (tiled, ADR-002 amendment 1) | **115.3** |
| metal token | 97.1 |
| metal layers | 95.9 |

One submission per token is 1.32× the per-layer pass and 1.40× the CPU at decode. The
concurrent groups add 4% to it and 18% to the per-layer pass, whose expert chains now
overlap. For long prompts the CPU is faster: the GPU matmul kernels read each weight row
once per token row (one threadgroup row per token), while the CPU's tiled kernel reuses
each weight load across six tokens.

### Decision
- **`--backend metal` defaults to whole tokens (`token`).** Routing on the GPU needs every
  expert resident, which is the regime where Metal wins; otherwise the pass falls back
  per pass to whole layers, and at load to the CPU pass as amendment 3 decided.
- **The CPU stays the default backend** until the owner reviews amendments 2–4.
- Consequence: with `token`, every expert of the model is read into the cache in the
  background after the first pass (on OLMoE, 1 024 experts and 3.63 GiB, against about
  760 and 3.47 GiB that a 128-token run touches), so resident memory reaches the plan's
  full expert budget sooner. The plan already budgets for it: `token` only routes on the
  GPU when every pool holds every expert.
- Next for Metal: tile the GPU matmuls over several tokens per weight load (keeping each
  output's reduction order), so prefill beats the CPU's; then the small-cache regime
  again, now that a token can be one submission that waits on expert availability.

## Amendment 5 (2026-11-05): token tiles for prefill, Phase 8 slice 6

### The question
Amendment 4 left long prompts to the CPU: at 128 tokens the CPU's tiled kernel (ADR-002
amendment 1) gave 115 tok/s and Metal 97, because every GPU matmul ran one token per
threadgroup row and so loaded and dequantised each weight once per token.

### What was built
- **Token-tiled matmuls** (`matmul_<dtype>_t`): one SIMD group computes its rows for up
  to 8 activation rows (`TILE`), loading and dequantising each weight once and keeping a
  separate pair of lane accumulators per (row, token). Every output gets the same fused
  multiply-adds in the same order and the same tree as the one-token kernel, so the bits
  do not change. The row bodies of every format (generic, Q4_K, Q6_K) are now templates
  over the tile width, shared by the one-token, tiled and routed kernels. `Batch::matmul`
  and the batched path switch to the tiled kernel from 4 rows (`TILE_MIN`).
- **Grouped routing** (`moe_group`, `Routing::Grouped`, kernels `routed_<dtype>_g`):
  routed entries are sorted by expert on the GPU (a counting sort in one thread), and
  each threadgroup row computes one tile of up to 8 entries of one expert. The grid
  covers `experts × ⌈tokens / 8⌉` tiles, because `moe_route` never picks an expert twice
  for one token; unused tiles return at once. `Batch::routed_matmul` now takes a
  `Routing` (`Ids` as in amendment 4, or `Grouped`). The one-submission pass groups when
  there are at least 2 entries per expert on average (`GROUPED_ENTRIES_PER_EXPERT`; on
  OLMoE from 16 tokens). The per-layer pass needs nothing new: it already gathers each
  expert's tokens, so its matmuls tile.

### Parity
`metal_matches_cpu_bit_for_bit` now covers 1, 3, 4, 8, 9, 13 and 17 activation rows
(full and partial tiles) for every format. `routed_matmuls_give_each_experts_cpu_bits`
runs 20 tokens with every token routed to one expert (tiles of 8, 8 and 4) through both
routings and both input kinds. The fixtures' 14-token prompt takes the grouped path. On
real OLMoE the env-gated test re-runs the prompt once every expert is resident, as one
routed submission with grouping, and gets the CPU's logits bit for bit. Every benchmark
run's tokens match the CPU's.

### Measurements (M3 Pro, OLMoE Q4_K_M, all experts resident)
Records: `docs/performance/baselines/2026-11-05-m3pro-tiles-*.json`.

| prefill | cpu | metal token, amendment 4 | **metal token** | metal layers |
|---|---|---|---|---|
| 10 tokens | 38.7 tok/s | 91.3 | **102.3** | 100.9 |
| 128 tokens | 115.3 | 97.1 | **239.0** | 248.0 |
| 1024 tokens | 114.6 | | **234.9** | |

Decode is unchanged (token 59.6 tok/s, layers 52.9). Thresholds, measured on prefill:
- plain tiles from 4 rows: at 6 tokens 68.5 → 62.3 ms; at 2–3 tokens tiles cost 3–7%;
- grouped routing from 2 entries per expert: at 10 tokens (1.25 per expert) grouping
  cost 25% (99 → 125 ms); at 16 tokens it gains 3%, at 32 tokens 287 → 194 ms, and at 64
  tokens 569 → 301 ms.

At 128 tokens the per-layer pass is 4% faster than the one-submission pass: its host
grouping is exact, while the grouped grid also launches the tiles of experts with fewer
entries. Both are about twice the CPU.

### Decision
- Metal prefill now beats the CPU at every prompt length measured, as decode already
  did. `token` stays the Metal default. The CPU stays the default backend until the owner
  reviews amendments 2–5.
- Next for Metal: the small-cache regime (pools smaller than top-k), now that a token
  can be one submission; then CUDA.

## Amendment 6 (2026-11-06): partial expert caches

### The question
Amendment 3 settled pools smaller than top-k. Between that and every expert resident,
pools hold a token's experts but not all of a layer's. Most budgets from 2 to 4 GiB land
here, and `--metal-ops token` needs every expert resident, so this regime gets the
per-layer path. Nothing had measured it.

### What was measured (M3 Pro, OLMoE Q4_K_M, on power)
A 10-token prompt, 160 generated, 5 runs; every record gave the reference tokens. A CPU
repeat at the end of the sweep gave 43.0 tok/s against 43.4 at the start. Records:
`docs/performance/baselines/2026-11-06-m3pro-smallcache-*.json`.

| expert cache | slots of 64 | CPU decode tok/s | Metal decode tok/s | Metal wait per token |
|---|---|---|---|---|
| all resident | 64 | 43.4 | **58.2** | 0 ms |
| 3072 MiB | 52–53 | 42.9 | **49.2** | 1.5 ms |
| 2048 MiB | 35–36 | 40.8 | 40.1 | 5.6 ms |
| 1900 MiB | 32–33 | **39.5** | 37.5 | not extracted |
| 1536 MiB | 26–27 | **38.6** | 34.5 | not extracted |
| `--ram-budget 2GiB` | 20–21 | **35.5** | 30.8 | not extracted |
| 1024 MiB | 17–18 | **34.9** | 29.5 | 12.0 ms |
| 768 MiB | 13–14 | **32.5** | 26.3 | 14.2 ms |

Metal waits on loads that the CPU pass overlaps with compute (the CPU waits 1.3 ms or
less). `token` and `layers` give the same numbers below 64 slots, which fits the
one-submission pass never running there; `Engine::gpu_routed_passes` was not recorded in
these runs.

Prefill goes the other way. Metal stays ahead at 128 tokens (1024 MiB: 194 against 117
tok/s; 2048 MiB: 221 against 117) and at 10 tokens (1024 MiB: 46.9 against 41.4).

### Decision
- **The plan also chooses the CPU pass when any layer's pool holds fewer than half its
  experts**, as amendment 3 does below top-k: at load, from the plan, with the reason in
  `PlacementReport::fallback`. `--backend metal` now decodes at 37.7 tok/s at 1536 MiB
  (was 34.5), 35.4 at `--ram-budget 2GiB` (was 30.8) and 34.1 at 1024 MiB (was 29.5). The
  cost is prefill in that range, which runs at the CPU's speed: at 1024 MiB, 128 tokens
  go from 194 to about 117 tok/s.
- **The line is a judgement.** Half is not the decode crossover, which is near 35 slots.
  Just above the line Metal gives up about 5% of decode (1900 MiB: 37.5 against 39.5) and
  keeps its prefill (58.1 against 40.6 at 10 tokens), so the line favours keeping Metal
  there.
- **Not built: Metal for prefill and the CPU for decode.** It is the fastest combination
  on paper (a 128-token prompt plus 160 tokens at 1024 MiB: about 5.2 s, against 5.7 s
  with this decision and 6.1 s before, not counting the handoff). It would move the KV
  between device buffers and host tensors at each switch and break §5's placement fixed
  at load, for at most about 8% in a regime only `--backend metal` reaches.
- Pools that pressure shrinks after load keep the pass chosen at load, as in amendment 3.
- The CPU stays the default backend until the owner reviews amendments 2–6.

## Amendment 7 (2026-11-11): NaN payloads and signs

Owner decision with Vulkan Step 3a (ADR-019 amendment 3). §1's contract, for every GPU
backend:
- **NaN payloads and NaN sign bits are not part of parity.** Where the CPU's output is
  NaN, a device's output must be NaN; its payload and sign are not compared. Apple GPUs
  (Metal, and Vulkan through MoltenVK) return the canonical NaN from arithmetic, while the
  CPU propagates an input's payload.
- **NaN classification is part of parity.** A NaN where the CPU has a number, or the
  reverse, is a difference, unless the subnormal exception explains it (for example a
  flushed subnormal times an infinity: `0 · ∞` is NaN where the CPU's `tiny · ∞` is
  infinite).
- **Subnormals are unchanged** (§1): the flush is the one other exception.

Parity tests that see NaNs classify each difference (subnormal, NaN payload, other),
print the counts and fail on any other; everywhere else they compare exact bits.
