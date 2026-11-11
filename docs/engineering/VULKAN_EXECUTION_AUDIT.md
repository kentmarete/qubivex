# Vulkan decode: execution audit and plan

- Date: 2026-11-11
- Base: `main` at 9721f58 (PR #55, ADR-019 amendment 1 Proposed)
- Scope: OLMoE-1B-7B-0125-Instruct Q4_K_M, `--backend vulkan`, MoltenVK 1.4.2 on the M3 Pro
- Status: audit only. No code changed. No benchmark was run for this report: the
  machine was on battery. Every number marked *recorded* comes from a committed record.
  Every number marked *estimate* is arithmetic on recorded numbers. Estimates are not
  results.

## 1. Summary

1. **Decode time is the dependent round trips, multiplied by the wait per round
   trip.** A decode token makes 305 submissions, but only 113 of them are on the
   critical path: 7 per layer × 16 layers, plus the head. The recorded caller wait per
   submission is about 320 µs, and 113 × 320 µs = 36.2 ms. The recorded median step is
   36.5 ms.
2. **Seven per layer, not five, because of threads.** The 8 routed experts run on the 5
   rayon compute threads. Each thread blocks in `vkWaitForFences` twice per expert
   (gate+up, then down). So the experts run in two waves of four sequential round
   trips. With 8 or 10 threads the recorded step was only 35.0 to 35.6 ms, not the
   ~26 ms that removing a wave would predict. So concurrent submissions do not overlap
   fully on the GPU. The cause is not established.
3. **There is a second ceiling: kernel bandwidth.** A decode token reads about 762 MB of
   weights (§6.2). The recorded kernel test runs a layer's 24 expert matrices in one
   submission at about 22.5 GB/s. If that is the end-to-end kernel rate, the GPU-busy
   floor is about 34 ms per token. That would cap this path near 30 tok/s even with zero
   round trips. The ADR's "GPU idle more than half the time" contradicts that floor.
   The records cannot settle which ceiling binds. **Step 0 is GPU timestamps, before any
   optimisation.**
4. **This repository has already solved the same problem on Metal.** ADR-011 amendments
   2 and 4 took Metal from per-matmul submissions to one submission per layer, then one
   per token, with the CPU's bits: 59.2 tok/s on this machine (*recorded*, 2026-11-04).
   ADR-019's Consequences already name "whole layers on the device" as the next step for
   Vulkan. The plan below follows that route. It adds the measurement and the cheap
   in-interface steps first, so each step is judged on its own.
5. **The main numerical risk is division and square root, not the matmuls.** Vulkan
   guarantees only 2.5 ULP for `x / y` and about 3 ULP for `sqrt`. The CPU's
   `rms_norm`, softmax and routing need correctly rounded results. On Metal this needed
   `precise::`. On Vulkan it needs a probe at open, as for `fma`, and probably an
   `fma`-based correction. Moving norms, attention or routing onto the GPU depends on
   this.

## 2. Execution map (Phase 1)

### 2.1 The call chain

```
Transformer::forward_segments            qubivex-arch/src/lib.rs:878
  └─ be.matmul / be.matmul_many          ComputeBackend = runtime::Placed
       └─ Placed::matmul_many            qubivex-runtime/src/placement.rs:330
            └─ OnDevice::matmul_many     qubivex-runtime/src/discrete.rs:303
                 (host address → DeviceWeights: arena range check, or
                  BTreeMap lookup under the expert-cache mutex)
                 └─ VulkanDevice::matmul_many   qubivex-backend-vulkan/src/lib.rs
                      └─ submitter() → Submitter::matmul_many   device.rs:994
                           └─ Submitter::run                    device.rs:891
```

Routed experts go through `StoreProvider::with_experts`
(`qubivex-runtime/src/engine.rs:478`). It spawns one rayon task per ready expert on
the compute pool, which has 5 threads by default. Each task calls the arch closure at
`lib.rs:1147`, which makes two blocking device calls.

### 2.2 One submission, step by step (`Submitter::matmul_many`)

| # | Step | Kind | Notes |
|---|---|---|---|
| 1 | Validate every op, build `plan: Vec<Planned>` | host, alloc | One `Vec` per call |
| 2 | Copy every op's `xs` into the submitter's host-visible `xs` buffer | **host → device** | One element at a time (`to_ne_bytes`). Identical inputs are copied once per op: QKV three times, gate/up twice |
| 3 | Grow `xs`/`ys` if needed | **device alloc** (rare) | Doubling. Bounded by the largest submission, per submitter |
| 4 | Grow the descriptor pool if `ops > sets` | **descriptor alloc** (rare) | Doubling, starts at 64 |
| 5 | `vkResetDescriptorPool` + `vkAllocateDescriptorSets` (one per op) | **descriptor alloc** | Every submission. `layouts` and `sets` are `Vec`s |
| 6 | `vkUpdateDescriptorSets`, 3 writes per op | **descriptor update** | Every submission. The weight binding is offset to `minStorageBufferOffsetAlignment` |
| 7 | `shared.pipelines.clone()` | host alloc | A `Vec` clone per submission |
| 8 | `vkResetCommandBuffer`, `vkBeginCommandBuffer(ONE_TIME_SUBMIT)` | **command buffer re-record** | One command buffer per submitter, recorded again every time |
| 9 | Per op: bind pipeline, bind set, push 24 B, `vkCmdDispatch(ceil(n/8), s, 1)` | recording | No barrier between dispatches: outputs are disjoint |
| 10 | One `vkCmdPipelineBarrier` COMPUTE_SHADER → HOST, SHADER_WRITE → HOST_READ | **barrier** | One per submission. It is needed |
| 11 | `vkQueueSubmit` under `queue_lock` | **submission** | About 34 µs host time (*recorded*) |
| 12 | `vkWaitForFences(u64::MAX)` then `vkResetFences` | **sync** | The caller blocks here. About 320 µs per submission (*recorded*) |
| 13 | Copy `ys` back into each op's output, one element at a time | **device → host** | |

Uploads (`Submitter::upload`) use a 64 MiB staging buffer, one fenced submission per
chunk, and no barrier after the copy. They run at load and on device expert-cache
misses. In the recorded benchmark every expert is resident, so there are no uploads
during decode (`experts.loads = 0`).

Before any copy and dispatch share a command buffer (Steps 3 and 4 below), add a
TRANSFER_WRITE → SHADER_READ barrier. Today the fence between submissions orders them.

### 2.3 Per decode token (s = 1)

| Item | Per layer | Per token (16 layers + head) |
|---|---|---|
| Submissions (= fence waits) | 3 dense (QKV, O, router) + 16 expert (8 × 2) = 19 | **305** |
| On the critical path | QKV, O, router, then 2 waves × (gate+up, down) = 7 | **113** |
| Pipeline barriers | 19 | 305 |
| Command buffer re-records | 19 | 305 |
| Descriptor sets allocated and written | 3 + 3 + 1 + 8 × 3 = 31 sets, 93 writes | 497 sets, 1 491 writes |
| Floats copied in | 51 200 (200 KiB) | ~0.82 M (3.1 MiB) |
| Floats copied out | 41 024 (160 KiB) | ~0.71 M (2.7 MiB) |
| Host heap allocations in the device path | ~6 per submission (plan, layouts, sets, pipelines clone, `ws`, `batch`) | ~1 800 |
| Host heap allocations in the arch pass | per expert `xe`, `g`, `u`, `y`; per layer `picks`, `contrib`, `by_expert` (64 `Vec`s) | ~1 600 |
| Device memory allocations | none | none |

The recorded "about 297 submissions per token" is a run average. The structural count
for a decode step is 305. The lookahead router is off in the benchmark: the provider
has no predictor.

**Lifetime and leaks.** Nothing grows without bound. The 8 submitters keep high-water
host buffers and descriptor pools. Weight and expert buffers are freed by
`DeviceBuffer`'s last clone. Every submission completes before it returns, so no
buffer can be freed while in flight (ADR-011 §3).

**Locks per submission.** These are the submitter mutex (a `try_lock` scan), the queue
lock (around `vkQueueSubmit` only), and for expert weights the device-cache mutex (one
`BTreeMap` lookup per op). None of them is contended enough to matter next to a
320 µs wait.

### 2.4 What stays on the host today

- Embedding lookup.
- All RMS norms (attention, QK, FFN, final).
- QKV clip, RoPE, KV-cache store.
- Attention: scores, softmax and values, parallel over heads with rayon.
- Residual adds.
- Routing: softmax and top-k.
- SwiGLU.
- Expert gather and combine (rank order, `fma`).

Only matmuls run on the device. That is why each layer has 7 synchronisation points.

## 3. Execution dependency diagram (one decode layer, s = 1)

```
 hs ──► [H] x = rms_norm(hs, attn)
          │
          ▼
 RT1   ┌─ S: matmul_many { Wq·x, Wk·x, Wv·x } ─┐  wait ~320 µs
       └──────────────────────────────────────┘
          │ q, k, v back on the host
          ▼
       [H] clip → qk_norm → RoPE → KV store → attention (16 heads, rayon)
          │ att
          ▼
 RT2   ┌─ S: matmul { Wo·att } ─┐
       └────────────────────────┘
          │ o
          ▼
       [H] hs += o ; x = rms_norm(hs, ffn)
          │
          ▼
 RT3   ┌─ S: matmul { Wrouter·x } ─┐   64 logits
       └───────────────────────────┘
          │
          ▼
       [H] route: softmax, top-8 → leases for 8 experts
          │
          ├──────────── 5 compute threads, each blocking on its own fence ──────────┐
          ▼                                                                         ▼
 RT4   S: { Wg_e·x, Wu_e·x }  e ∈ wave 1 (5 experts)           RT6   same, wave 2 (3 experts)
       [H] swiglu                                                    [H] swiglu
 RT5   S: { Wd_e·g }                                           RT7   S: { Wd_e·g }
          └──────────────────────────────┬──────────────────────────────────────┘
                                         ▼
       [H] combine in rank order (fma) ; hs += o          ──► next layer

 After layer 16: [H] final rms_norm → RT: matmul { Whead·x } → 50 304 logits
```

Which dependencies are real and which come from where the work runs:

| Edge | Real data dependency? | Removable by |
|---|---|---|
| RT1 → RT2 (attention needs q, k, v) | yes | running attention on the device: no host hop |
| RT2 → RT3 (router needs the normed residual) | yes | the same |
| RT3 → experts (which experts) | yes, on the host: the store leases experts | GPU routing over held experts (Metal amendment 4) |
| gate+up → down within an expert | yes | SwiGLU on the device, same submission |
| wave 1 → wave 2 | **no**: an artefact of 5 blocking threads | grouping a layer's experts into one submission |
| experts → combine → next layer | yes | combine on the device |

## 4. Phase 2 to 4 findings

### 4.1 Submission reduction, inside the current interface

| Opportunity | Bits | What it saves | Estimate | Verdict |
|---|---|---|---|---|
| Bulk copies in and out (`copy_nonoverlapping` / byte cast), one copy of a shared input | unchanged | ~1.5 M scalar copies per token on the critical path | 0.5–2 ms/token | **Do** (Step 1). Cheap, local |
| Push descriptors (`VK_KHR_push_descriptor`, listed by MoltenVK here) or sets cached per (buffer, offset) | unchanged | reset, allocate and 3 writes per op | small; part of the 34 µs submit cost | **Do** (Step 1) if Step 0 shows host time matters |
| Remove per-call `Vec`s (`pipelines.clone()`, `plan`, `layouts`, `ws`, `batch`) | unchanged | ~1 800 small allocations | < 0.5 ms/token | Do with the above |
| Reusable (pre-recorded) command buffers | unchanged | recording time | small at 1–3 dispatches per submission | **Defer** until a submission holds a whole layer or token (Step 4) |
| Fewer barriers | — | none: there is one per submission, and it is needed | 0 | Nothing to remove |
| Polling the fence instead of waiting | — | — | 0 (*recorded*: no change) | Already rejected |

These steps shorten each round trip. They do not remove any. Even if they saved 50 µs
per round trip, the result would be about 31 ms (32 tok/s). That is an optimistic bound,
not a plan.

### 4.2 Removing round trips (Phase 3)

These are in order of dependency, not ease. Each needs the one before it.

1. **One expert submission per stage per layer**: all gate+up together, then all down.
   This removes the wave artefact, taking the critical path from 7 to 5 per layer.
   It works at the arch level: when the backend asks for it and every routed expert is
   resident, the MoE block issues one `matmul_many` over every expert's gate and up,
   runs SwiGLU on the host, then one `matmul_many` over every down. With streaming (a
   small cache) the current per-expert path stays: ADR-011 amendment 3 measured that
   overlapping loads with compute wins there. The change is generic to any MoE.
2. **The rest of the layer on the device** (ADR-011 amendment 2's shape): norms, clip,
   RoPE, KV store, attention, residuals, SwiGLU and combine as Vulkan kernels. Only
   router logits come back. This takes the critical path to 1 per layer, 17 per token.
   It needs correctly rounded `/` and `sqrt` (§5).
3. **Routing on the device** (amendment 4's shape): the route kernel plus routed
   matmuls over a held layer. This makes 1 submission per token when every expert is
   resident on the device (the recorded benchmark's regime).

KV cache: the device KV belongs in `KvCache`'s existing device slot (`DeviceKv`), as on
Metal. It grows by the host cache's 256-position chunks and is charged to the ledger's
KV owner.

### 4.3 MoE expert execution (Phase 4)

The records already warn that grouping is not automatically faster:

- The kernel test's 24 expert matrices in **one** submission take 1.26 ms (*recorded*),
  about 22.5 GB/s.
- Today a layer's experts cost about 4 round trips, about 1.28 ms (*estimate*).
- So putting the same dispatches into one command buffer roughly **ties**. MoltenVK
  appears to run them back to back.

The gain has to come from the dispatch shape, not only the submission count. The
options to measure:

| Variant | Description | Bits |
|---|---|---|
| A. Independent (today) | one submission per expert stage, 5 threads | reference |
| B. One command buffer, N dispatches | 1 submission per stage per layer | same kernel, same bits |
| C. **Grouped dispatch** | one `vkCmdDispatch` over every matrix of a stage: workgroup `z` picks the matrix from a fixed array of storage-buffer bindings (dynamically uniform index; `shaderStorageBufferArrayDynamicIndexing` is a Vulkan 1.0 feature, checked at open). Same per-row body | same body, so same bits; parity test per dtype |
| D. Shared input | gate and up as one 2048-row matrix pass over one `x` load | needs adjacent weights in one buffer; GGUF stores them separately, so a copy at upload |
| E. Token tiles | several activation rows per weight load | prefill and batch only. Amendment 1 measured 4 rows per group **slower** end to end; revisit only with timestamps |

Measure each at experts {1, 2, 4, 8} × tokens {1, 4, 10, 32}, with timestamps and the
end-to-end bench. Keep a variant only if the bench improves. The kernel test alone does
not decide: amendment 1 recorded a kernel-test win that lost end to end.

The "expert-weight cache locality" item: on this machine every expert is resident
(`hit_rate 1.0`, 0 loads), so locality is not the cost now. Under `--ram-budget 1GiB`,
ADR-011 amendment 3's decision applies: the plan runs the CPU pass when pools are
smaller than top-k.

## 5. Parity constraints for anything beyond matmuls

| Operation | CPU definition | Vulkan guarantee | Plan |
|---|---|---|---|
| `fma` | single rounding | the driver may split it | probed at open (exists) |
| `a + b`, `a - b`, `a * b` | IEEE | correctly rounded; `NoContraction` forbids fusing | as amendment 1 |
| `x / y` | correctly rounded | **2.5 ULP** | a probe at open, plus an `fma`-based correction step in the shader, proven over the ranges used, or refuse the device |
| `sqrt` | correctly rounded | **inherited from `1/inversesqrt`**, about 3 ULP | the same. On Metal, `1/sqrt` became `rsqrt` until `precise::` (ADR-011 amendment 2) |
| `exp` | `qubivex_compute::math::exp`: basic ops and `fma` only | portable once `fma` is single | port it unchanged |
| 64-lane reductions | `qubivex_compute::reduce` | shared-memory tree (exists in `matmul.comp`) | reuse |
| RoPE | host table, f64 rounded once | — | upload the table, as on Metal |
| subnormals | Apple GPUs flush (ADR-011 §1) | — | the existing exception |

MoltenVK with fast math off probably gives correctly rounded `/` and `sqrt`, because
Metal's non-fast mode does. A probe and the op parity tests must prove that before
anything relies on it. Other Vulkan drivers make no such promise.

## 6. Benchmarks (Phase 6)

### 6.1 Recorded

| Record | Backend | Decode tok/s | ms/token | Prefill tok/s | Source |
|---|---|---|---|---|---|
| 2026-11-11 vulkan-before | Vulkan | 6.52 | 153.3 | 8.2 | `baselines/2026-11-11-m3pro-vulkan-before.json` |
| 2026-11-11 vulkan-after | Vulkan | **27.36** | 36.55 | 48.9 | `…-vulkan-after.json` |
| 2026-11-11 cpu (same session) | CPU | **43.50** | 22.99 | 77.0 | `…-vulkan-cpu.json` |
| 2026-11-04 metal token | Metal | 59.2 | ~16.9 | 91.3 | ADR-011 amendment 4 (different session, 128 generated) |

Vulkan-after also records:
- peak RSS 1.39 GB
- ledger: dense 311 MB, expert cache 3.61 GB, KV 67 MB
- every expert resident (hit rate 1.0)

### 6.2 Effective weight bandwidth

Per decode token:
- 128 routed experts × 3.52 MB (expert cache 3 606 822 912 B / 1 024 experts) = 451 MB
- 81 dense tensors on the device: 311 MB

That is about **762 MB per token**.

| | tok/s | Effective GB/s |
|---|---|---|
| Vulkan now | 27.36 | 20.8 |
| Target 1 | 30 | 22.9 |
| Target 2 | 35 | 26.7 |
| Stretch (CPU) | 43.5 | 33.1 |
| Metal token (recorded) | 59.2 | 45.1 |
| Vulkan kernel test, 24 expert matrices in one submission | — | ~22.5 |

If ~22.5 GB/s is what the kernels deliver end to end, **only the first target is within
reach by removing round trips**. The second and the stretch target need faster kernels
too, about 27 and 33 GB/s sustained. Metal reaches 45 GB/s on the same GPU, so the
hardware allows it.

### 6.3 What the bench does not capture yet

`qubivex bench` records ms/token, tok/s, prefill, peak RSS, the ledger and expert
counters. It does **not** record the device's `TransferStats`:
- submissions
- activation bytes in and out
- transfer time

It also lacks fence waits, host time in record and submit, time blocked in waits, and
GPU execution time. Phase 6 asks for all of these. GPU time needs timestamp queries
(`vkCmdWriteTimestamp`). `vulkaninfo` on this machine lists:
- `timestampComputeAndGraphics = true`, `timestampValidBits = 64`,
  `timestampPeriod = 1`
- `VK_KHR_push_descriptor`
- `shaderStorageBufferArrayDynamicIndexing = true`
- `bufferDeviceAddress = true`, which could hold Metal amendment 4's expert address
  table

How fine-grained MoltenVK's timestamps are within a command buffer is still to be
checked. GPU idle time follows from GPU time and wall time.

## 7. A reusable execution scheduler (Phase 5): evaluation

What the forward pass needs is already shown by `qubivex-backend-metal`'s `Batch` and
`qubivex-runtime`'s `MetalForward`:
- device vectors and spans
- ops recorded into one command buffer
- explicit **concurrent groups**, with a barrier between groups
- `submit`, `submit_async` returning a `Pending`
- `read` and `write`

The pass is a fixed sequence with known parallel groups. Its shape depends on the
model config and `s`, not on run-time data, except routing, which amendment 4 moved
into buffers.

**Recommendation: a recorded device program, not a dynamic dependency graph.**

- A backend-neutral `DeviceProgram` trait in `qubivex-compute`, shaped like Metal's
  `Batch`:
  - ops: `matmul`, `rms_norm`, `rope`, `clamp`, `add`, `copy`, `swiglu`, `kv_store`,
    `attention`, `moe_route`, `routed_matmul`, `moe_combine`, gather and scatter
  - `concurrent(|b| …)` groups
  - submit, async submit, read, write
- One generic `DeviceForward` in `qubivex-runtime`, from `MetalForward`. It is driven by
  the arch config, not by OLMoE, and adds a dense-FFN path, which `MetalForward` lacks
  ("whole Metal layers need routed experts"). The CPU pass in `qubivex-arch` stays the
  reference and is not rewritten (owner rule).
- What each Phase 5 item maps to:

  | Phase 5 item | In this design |
  |---|---|
  | Dependency tracking | program order, plus groups |
  | Synchronisation planning | a barrier between groups, a fence per submission, `Pending` for async |
  | Buffer lifetimes | a per-pass arena (Metal's `Bufs`), KV in `KvCache`'s device slot, pins as today |
  | Command buffer reuse | replay of a recorded decode program (Step 4) |
  | Backend strategies | the trait's implementations |

- A general DAG scheduler would add a run-time planner, and a place for ordering bugs,
  without a measured benefit. Revisit it if a model needs data-dependent control flow
  beyond routing.

Two ways to get there, to be chosen:

| | Extract first | Vulkan first |
|---|---|---|
| Order | make `MetalForward` generic over the trait (Metal's tests hold its bits), then implement Vulkan | write `VulkanForward` beside `MetalForward`, unify later |
| Risk | touches the default Metal path | none to Metal |
| Cost | one implementation | ~1 100 lines duplicated, two places to keep bit-identical |
| Recommendation | **this one**, behind Metal's existing parity suite | |

## 8. Proposed implementation sequence

Each step is its own commit series, behind its own switch where it changes behaviour,
and benchmarked on mains power with identical configuration (10-token prompt, 64
generated, 5 threads, 1 warm-up + 3 runs). Every step has the same gates:
- the `qubivex-backend-vulkan` parity tests
- `tests/vulkan.rs`
- the fixtures
- the env-gated OLMoE test: the CPU's prompt logits and Colibri's tokens
- every dtype
- bench tokens identical to the reference
- no prefill regression

| Step | What | Changes behaviour? | Expected (*estimate*) | Decides |
|---|---|---|---|---|
| **0** | Measurement: device counters and GPU timestamps in `TransferStats` and the bench record (submissions, waits, host submit µs, blocked µs, GPU µs, bytes in/out). Run the 8-thread and grouped-submission experiments with timestamps | no (timestamps off by default, or their cost measured) | 0 | which ceiling binds (§1.3); whether Steps 1–2 are worth doing |
| **1** | Host-side submission cost: bulk copies, one copy per shared input, push descriptors or cached sets, no per-call `Vec`s | no | 0.5–3 ms → 28–29.5 tok/s | |
| **2** | Grouped experts: arch-level one-submission-per-stage when every expert is resident (backend capability flag), plus the grouped-dispatch kernel (variant C) if Step 0 shows serial dispatch costs | no bits change; path switch | 7 → 5 round trips per layer. **30–34 tok/s** if the grouped dispatch is faster than serial dispatches; ~0 if not | target 1 |
| **3** | `DeviceProgram` trait; `MetalForward` → generic `DeviceForward` (Metal bits unchanged); Vulkan kernels for norms, RoPE, attention, KV store, SwiGLU, combine; correctly rounded `/` and `sqrt` with a probe. **Needs ADR-019 amendment 2** (the ADR calls this "its own decision") | new placement | 17 round trips per token. **35–45 tok/s** if kernels reach 27–33 GB/s; capped near 30–33 if they stay at 22.5 | target 2 |
| **4** | Routing on the device and routed matmuls; one submission per token; replay of the recorded decode program | new placement | the GPU-busy floor | stretch |
| **5** | Kernel throughput (only if Step 0 shows kernels bind): occupancy, grouped dispatch for QKV, revisit tiles for prefill | no bits change | toward Metal's 45 GB/s | stretch |

Steps 1 and 2 stay within the `DiscreteDevice` interface and are cheap to revert.
Step 3 is the substantial architectural change. Nothing in it should start before the
owner decides §9.

## 9. Decisions for the owner

1. **The order.** Measure first (Step 0), then the in-interface steps (1–2), then the
   device-resident pass (3–4)? Or go straight to Step 3 after Step 0?
2. **Extract first or Vulkan first** for the device program (§7).
3. **Division and square root on drivers that do not round them correctly:** emulate
   with an `fma` correction, refuse the device as for a split `fma`, or allow a lossy
   opt-in under ADR-010.
4. **ADR-019 amendment 1 is still Proposed.** Accept it before amendment 2 builds on it?
