# Vulkan decode: execution audit and plan

- Date: 2026-11-11
- Base: `main` at 9721f58 (PR #55; ADR-019 amendment 1 accepted 2026-11-11)
- Scope: OLMoE-1B-7B-0125-Instruct Q4_K_M, `--backend vulkan`, MoltenVK 1.4.2 on the M3 Pro
- Status: audit done. The owner's decisions (2026-11-11) are in §9, and the revised
  sequence is in §8. No code changed by the audit. No benchmark was run for this report:
  the machine was on battery. Every number marked *recorded* comes from a committed record.
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

The pass is a fixed sequence with known parallel groups. Its shape depends on the model
config and `s`, not on run-time data, except routing, which amendment 4 moved into
buffers. So the scheduler is a **recorded program, not a dynamic dependency graph**
(owner decision §9.2). It has three layers, and only the first two are shared.

1. **The logical program (shared, backend-neutral).** It is built from the arch config
   (dense or MoE), never from a model name. It is an ordered list of logical ops over
   named device vectors:
   - `matmul`, `rms_norm`, `rope`, `clamp`, `add`, `copy`, `swiglu`, `kv_store`,
     `attention`, `moe_route`, `routed_matmul`, `moe_combine`, gather and scatter
   - grouped into **parallel groups**: ops in a group are independent; consecutive
     groups depend on each other
   - with **host points**: the places where the host must read a value, such as router
     logits when routing runs on the host, or the final logits.

   The CPU pass in `qubivex-arch` stays the reference and is not rewritten (owner rule).
2. **The submission planner (shared).** It maps the logical program to physical
   submissions. It cuts only at host points: a genuine data dependency on a
   host-visible value. It also cuts where the backend's policy asks for it, for example
   a maximum number of dispatches per command buffer, or an op the device cannot run
   with the CPU's bits (§5), which goes to the host. Everything else inside a
   submission is groups separated by barriers. One logical pass can therefore be one
   submission (a whole token), one per layer, or finer, without changing the program.
   The plan is data, so it can be logged, tested and counted by the bench.
3. **The backend executor (per backend, independent).** It implements a small shared
   trait: allocate device vectors; record a group of ops; insert the barrier between
   groups; submit, submit async, wait; read and write host points. It also reports
   per-op capability: whether the op is supported with strict bits, and how. Metal
   keeps its implementation (`Batch`, `program.rs`, `kernels.metal`) and its parity
   tests, with an adapter onto the trait. Vulkan gets its own command encoding,
   synchronisation, memory management and GLSL kernels behind the same trait.

| Phase 5 item | In this design |
|---|---|
| Dependency tracking | program order and parallel groups; host points for host dependencies |
| Operation grouping | parallel groups; the planner's choice of submission boundaries |
| Synchronisation planning | barriers between groups; a fence or `Pending` per submission, decided by the planner |
| Buffer lifetimes | a per-pass arena of named vectors (Metal's `Bufs`), KV in `KvCache`'s device slot, pins as today |
| Command buffer reuse | replay of a recorded decode submission (Step 4) |
| Backend strategies | the executor trait and the planner's backend policy |

A general DAG scheduler would add a run-time planner, and a place for ordering bugs,
without a measured benefit. Revisit it only if a model needs data-dependent control
flow beyond routing.

## 8. Implementation sequence (revised 2026-11-11 after the owner's decisions)

The rules for every step:
- Each step is its own commit series, behind its own switch where it changes
  behaviour, and can be reverted on its own.
- Each step is benchmarked with the full `qubivex bench` on mains power, with identical
  configuration: 10-token prompt, 64 generated, 5 threads, 1 warm-up + 3 runs.
- `qubivex bench` is the authority. A change that improves kernel or shader timings but
  regresses the bench is not kept, and no improvement is claimed from kernel timings.

Every step has the same gates:
- the `qubivex-backend-vulkan` parity tests
- `tests/vulkan.rs`
- the fixtures
- the env-gated OLMoE test: the CPU's prompt logits and Colibri's tokens
- every dtype
- bench tokens identical to the reference
- no prefill regression
- no unbounded allocation

Targets: 30 tok/s, then 35, then the CPU's 43.5.

| Step | What | Changes behaviour? | Expected (*estimate*) | Gate to the next step |
|---|---|---|---|---|
| **0** | **Measurement (mandatory).** `TransferStats` and the bench record gain: submissions; fence waits; host time recording and submitting; time blocked in fence waits; GPU time from timestamp queries, as a per-submission sum and as the union of busy intervals, so GPU idle = wall − busy; activation bytes in and out; per-phase (prefill, decode) deltas per token. Timestamps are opt-in, and their cost is measured on against off. Then rerun the 5- and 8-thread decode with timestamps | no | 0 | which ceiling binds (§1.3) |
| **1** | Host-side submission cost: bulk copies, one copy per shared input, push descriptors or cached sets, no per-call `Vec`s. Each is a separate commit and measured alone | no | 0.5–3 ms → 28–29.5 tok/s | |
| **2** | Grouped experts: one submission per stage per layer when every expert is resident (a backend capability, generic to any MoE), plus the grouped-dispatch kernel (variant C) if Step 0 shows serial dispatch costs. Measured at experts {1, 2, 4, 8} × tokens {1, 4, 10, 32} against the current path | no bits change; path switch | 7 → 5 round trips per layer; **30–34 tok/s** if the grouped dispatch beats serial dispatches, ~0 if not | target 1 |
| **3a** | **ADR-019 amendment 2** (Proposed, for the owner): the shared GPU-pass contract (§7: logical program, submission planner, executor trait), the submission strategy, and the numerical capability requirements (3b). Written before the code it governs | — | — | owner acceptance |
| **3b** | **Numerical capabilities and conformance.** Device-open probes for `fma`, `/`, `sqrt`, the `1/sqrt` pair, `exp`'s building blocks, and subnormal handling. Shader correction (an `fma` refinement) where it is proven correct over the inputs used. Otherwise, per-op CPU fallback for the sensitive op, or refusing strict Vulkan on that device. Approximate arithmetic only as an explicitly selected mode under ADR-010, never silently. **One shared conformance suite**, CPU, Metal and Vulkan, op by op (norms, softmax, routing, RoPE, attention, SwiGLU, combine, matmul per dtype), bit for bit against the CPU | strict stays the default | 0 | suite green on Metal and MoltenVK |
| **3c** | **The shared layers and the Metal adapter.** The logical program and submission planner in a shared crate; Metal's adapter onto the executor trait. Metal's implementation, placement and parity tests unchanged; Metal bench within noise | no (Metal path re-plumbed only) | 0 | Metal bits and bench unchanged |
| **3d** | **The Vulkan executor and kernels.** Command encoding, barriers, the per-pass arena and device KV, and GLSL kernels for norms, RoPE, attention, KV store, SwiGLU and combine, each in the conformance suite. Then the whole-layer plan: one submission per layer, only router logits returned | new placement, opt-in until measured | 17 round trips per token; **35–45 tok/s** if kernels reach 27–33 GB/s, near 30–33 if they stay at 22.5 | target 2 |
| **4** | Routing on the device and routed matmuls (descriptor arrays or buffer device addresses); the planner makes a token one submission when every expert is resident; replay of the recorded decode submission | new plan | toward the GPU-busy floor | stretch |
| **5** | Kernel throughput, only if Step 0 or 3d shows kernels bind: occupancy, grouped QKV, token tiles for prefill revisited with timestamps | no bits change | toward Metal's 45 GB/s | 43.5 tok/s |

Steps 0 to 2 stay within the `DiscreteDevice` interface. Step 3 is the substantial
architectural change, and nothing in it starts before amendment 2 is accepted.

## 9. Owner decisions (2026-11-11)

1. **Order.** Step 0, then 1 and 2, then 3 and 4. Step 0 is mandatory: the records
   conflict on GPU time against submission overhead. Do not jump to Step 3.
2. **Shared GPU-pass architecture first.** Share the execution contract and the
   scheduler abstraction, not backend implementations. Metal is the reference: its
   implementation and parity tests are preserved. Vulkan's command encoding,
   synchronisation, memory management and kernels sit behind the shared interface.
   Start from a recorded sequence of operations with explicit parallel groups, not a
   general dependency-graph framework.
3. **Numerics.** Strict parity stays the default. Device-open capability tests cover
   division, square root, `fma` and other sensitive arithmetic. Use shader correction
   where it is reliable. Otherwise run the sensitive op on the CPU, or reject strict
   Vulkan on that device. Approximate arithmetic is only an explicitly selected mode;
   parity is never compromised silently. Shared CPU, Metal and Vulkan numerical
   conformance tests.
4. **ADR-019.** Amendment 1 accepted, subject to the existing parity and regression
   checks; they were rerun and passed (ADR-019, amendment 1 "Acceptance"). Amendment 2
   is separate: the shared GPU-pass architecture, submission strategy and numerical
   capability requirements.
5. **Logical passes are separate from physical submissions.** The scheduler groups
   logical operations into fewer submissions where genuine data dependencies allow
   (§7). Backends stay independent, and there are no model-specific shortcuts.
6. **Discipline.** Each optimisation is independent, benchmarked and revertible. The
   full bench is the authority. Targets: 30, then 35, then 43.5 tok/s. No claims from
   isolated shader timings.

## 10. Step 0 status (2026-11-11)

Built on branch `vulkan-submission-audit`:
- `TransferStats` gains:
  - `fence_waits`, `submit_ns`, `wait_ns`
  - `timed`, `gpu_ns`, `gpu_busy_ns`
  - `since()`
- `VulkanConfig::timestamps` / `--vulkan-timestamps` adds a two-query timestamp pool
  per submitter, read after the fence.
- The busy union tolerates out-of-order arrival up to half a 64-interval window.
- The bench record gains a per-phase `device` block (docs/performance/BENCHMARKING.md).

Tests:
- The Vulkan parity tests, now with a timed run checked bit for bit.
- The union's unit test.
- `tests/vulkan.rs`, including OLMoE.
- The CLI tests.

Behaviour is unchanged without the flag.

### Step 0 records (mains power, 2026-11-11)

Records: `docs/performance/baselines/2026-11-11-m3pro-vulkan-step0-*.json`.

Setup:
- OLMoE Q4_K_M, the default 10-token prompt, 64 generated, 1 warm-up + 3 runs.
- Run back to back on mains power.
- Every record's tokens are identical across runs and equal to the PR #55 record
  (`--reference …-vulkan-after.json`).

Build: the binary was built from the working tree that was committed as 4c8e167. Its
code is identical to that commit (the records say `835e761-dirty`; only docs changed
after the build).

The first untimed 5-thread run was disturbed and is not committed: copies 7.7 ms per
step against 0.9 ms in every other run, and P95 43 ms. Both 5-thread runs were then
repeated.

Decode, per token:

| Record | Threads | Timestamps | ms/token (median) | tok/s | Host submit Σ | Host wait Σ | GPU Σ | GPU busy | GPU idle |
|---|---|---|---|---|---|---|---|---|---|
| untimed-5t | 5 | off | 35.86 | 27.89 | 5.61 ms | 86.7 ms | — | — | — |
| untimed-5t-repeat | 5 | off | 35.51 | 28.16 | 5.67 ms | 85.3 ms | — | — | — |
| timed-5t | 5 | on | 39.00 | 25.64 | 10.76 ms | 90.8 ms | 19.52 ms | 19.52 ms | 19.39 ms |
| timed-5t-repeat | 5 | on | 39.78 | 25.14 | 10.77 ms | 92.8 ms | 19.74 ms | 19.74 ms | 19.90 ms |
| untimed-8t | 8 | off | 34.76 | 28.77 | 9.00 ms | 128.6 ms | — | — | — |
| timed-8t | 8 | on | 36.83 | 27.15 | 16.19 ms | 128.3 ms | 17.15 ms | 17.15 ms | 19.69 ms |
| cpu-5t | 5 | — | 23.11 | 43.28 | | | | | |

Every decode token:
- makes 305 submissions, each with a fence wait;
- copies 3.13 MiB in and 2.70 MiB out;
- spends 0.9 ms of host time in copies.

The host Σ columns sum over the threads that submit, so they exceed wall time.

### What Step 0 establishes

1. **GPU submissions never overlap.** In every timed run, summed GPU time equals busy
   time to within 0.01 ms. The 8 submitters overlap host work only: MoltenVK runs the
   command buffers on its queue one at a time. Parallel submission is not a source of
   GPU concurrency on this driver. Concurrency has to come from inside a command
   buffer, or from fewer, larger submissions.
2. **The binding ceiling is the round trips, not the kernels.**
   - GPU busy is 19.5–19.7 ms of a ~39 ms timed token (17.2 of 36.8 at 8 threads).
   - The GPU is idle about half the time.
   - About 762 MB in 19.6 ms is about 39 GB/s while busy.
   - The 34 ms kernel floor feared in §1.3 is refuted. With every round trip removed,
     the GPU work alone would allow about 50 tok/s.
3. **The cost of a round trip.** The first version of this item reconciled 113 × ~316 µs
   with 35.7 ms. That fit was a coincidence: it mixed two different intervals and
   counted GPU time twice. See "Reconciling the per-token time" below for the corrected
   accounting: about 16 ms of idle time per token, spread over about 113 gaps, ≈ 140 µs
   per gap.
4. **Host submission work is small.** Preparing and submitting takes 5.6 ms per token
   summed over threads, about 18 µs per submission, and copies take 0.9 ms. On the
   critical path that is at most about 113 × 18 µs ≈ 2 ms plus part of the copies.
   **Step 1 can save at most about 2–3 ms (to about 30 tok/s).** It is worth doing
   because it is cheap, but it cannot carry target 1 by itself.
5. **Timestamps cost about 3.5–4 ms per token** (35.5–35.9 against 39.0–39.8), mostly
   host submit time, which roughly doubles. Tok/s comes from untimed records only.
6. **8 threads:** 34.76 ms against 35.5–35.9 (about 3% faster). It is not a substitute
   for Step 2, and the default stays at 5.
7. **The session is comparable to PR #55's:** CPU 43.28 tok/s (43.50 then), and Vulkan
   untimed 27.9–28.2 tok/s (27.36 then).

Consequences for the sequence:
- Step 2's value depends on the grouped dispatch, not on submission concurrency.
  Two expert submissions per layer instead of four critical round trips remove about
  2 gaps per layer, 32 per token. At ≈ 140 µs per gap that is about 4.5 ms: about
  31 ms per token, **≈ 32 tok/s**, if the grouped submissions' GPU time is no worse than
  the serial dispatches they replace. The earlier "≈ 37 tok/s" came from the
  double-counted 316 µs per round trip and is withdrawn as an estimate; 37 tok/s stays
  a target, not an expected result. The kernel test's 24 matrices in one submission
  (1.26 ms) against ~16 × 64 µs ≈ 1.0 ms today says to measure GPU time before
  building on it.
- Steps 3–4 target the ~16–19 ms of idle time directly.

### Reconciling the per-token time (added 2026-11-11, at the owner's request)

The four figures measure different intervals, and some of them overlap:

| Figure | Interval measured | Summed over | Overlaps with |
|---|---|---|---|
| 35.7 ms/token | wall clock, one decode step | one step | — (the quantity to explain) |
| 305 submissions | each `vkQueueSubmit` | all submitting threads | 49 are serial (3 dense per layer + head); 256 are expert submissions issued by 5 threads at once |
| 280 µs "wait per submission" | fence wait: return of `vkQueueSubmit` to return of `vkWaitForFences` | **every thread**, 86 ms Σ per token | each other: up to 5 expert waits run at once, and each includes the GPU time of the command buffers queued ahead of it from other threads. Σ wait (86 ms) exceeds wall (35.7 ms) for this reason |
| 18 µs "host submit" | descriptor sets, record, `vkQueueSubmit` | every thread, 5.6 ms Σ | other threads' waits; only the part on the critical thread adds to wall |
| 19.6 ms GPU busy | union of TOP_OF_PIPE → BOTTOM_OF_PIPE timestamp intervals | the GPU queue | nothing: Σ GPU = busy, so no two command buffers ran at once |
| "220 µs non-GPU" | 280 µs − 64 µs average GPU | — | **invalid**: 280 µs is an average over overlapping waits that include other submissions' GPU time, so subtracting one submission's GPU time does not isolate fixed latency |

What does reconcile:
- The GPU runs one command buffer at a time and every token waits for all of its GPU
  work. So all 19.6 ms of GPU busy time lies on the critical path.
- Wall minus busy is the time the GPU sits idle while the token is in progress:
  35.7 − 19.6 ≈ **16.1 ms**.
  - The timed runs show 19.4–19.9 ms idle. That includes the ~3.5–4 ms the timestamp
    queries add on the host.
  - The GPU figure is taken from the timed runs on the assumption that timestamps do
    not change GPU time. Their busy time matches across the two runs (19.52, 19.74).
- Idle time is what each serial hand-off costs:
  - host work between submissions: CPU attention, norms, routing, SwiGLU and combine,
    copies, descriptor writes and recording;
  - MoltenVK's submit-to-start latency;
  - completion-to-wake latency.
- About 113 hand-offs per token:
  - 3 dense per layer;
  - about 4 expert stages per layer, in two waves;
  - plus the head.
- That is ≈ 16.1 / 113 ≈ **140 µs idle per hand-off**.
  - It is an average. The expert waves are pipelined across 5 threads rather than strict
    waves, so the expert stages' share of the gaps is not measured separately.
  - Per-stage timestamps are not recorded.
- The earlier "113 × 316 µs ≈ 35.7 ms" added the 280 µs average wait (which already
  contains GPU time, partly other submissions') to the critical-path count. GPU time was
  then also counted in the wall. Its agreement with the median was a coincidence.

Consequences:
- Step 1's ceiling still holds: host prep (≈ 18 µs per submission) plus copies is a
  small part of the ≈ 140 µs gap. Step 1a's neutral result agrees.
- Each removed hand-off saves about one gap, ≈ 140 µs, not 316 µs. Step 2's estimate
  above is corrected to ≈ 32 tok/s on that basis.
- Step 2's measurements report the following, each separately:
  - host submit time (Σ `submit_ms`, and per submission);
  - fence wait (Σ);
  - GPU execution (Σ and busy);
  - GPU idle.
  Throughput comes only from untimed, alternated runs.

## 11. Step 1a: bulk copies, one copy per shared input (2026-11-11)

What changed:
- `Submitter::matmul_many` copies each op's activations and outputs as whole slices,
  not float by float.
- Ops that read the same activation slice (same address and length: QKV, and
  gate/up) share one copy on the device.
- No shader change, so no bit change.
- New test `a_shared_input_is_copied_once`: four ops, three sharing one slice and one
  reading an equal but separate slice. It checks the CPU's bits and that two copies
  were made.

Bench: interleaved, base (a7bc0bf) then step 1a, twice, on mains power. Records:
`baselines/2026-11-11-m3pro-vulkan-step1a-{base,step1a}-{1,2}.json`. Tokens equal the
reference in all four.

| | Run 1 ms/token | Run 2 ms/token | Copies per token | Bytes in per token |
|---|---|---|---|---|
| base | 36.16 | 35.44 | 0.91 ms | 3.13 MiB |
| step 1a | 35.42 | 35.71 | 0.23 ms | 1.88 MiB |

**Verdict: no end-to-end change that can be measured.** Copy time per token fell by
0.7 ms and input bytes by 40%, but the medians differ by less than the run-to-run spread
(~0.7 ms). The copies were spread across the expert threads, not on the critical path.
No improvement is claimed. The change is kept because it does no harm, does less
work and is simpler. Revert it with one commit.

This confirms Step 0's bound for the rest of Step 1. The remaining host submission work
is about 18 µs per submission. Push descriptors would remove only part of that, at most
about 0.5 ms per token on the critical path, which is below the noise floor of a 3-run
bench, and they would add a second descriptor path to maintain.

## 12. Step 2: grouped experts (plan, 2026-11-11)

The owner approved Step 2 and set these conditions:
- skip 1b and keep 1a;
- parity with PR #55 on every run;
- expert dispatches per layer go from 7 to 5;
- host submit time measured apart from GPU execution;
- register pressure, occupancy, memory traffic and grouped shader time monitored;
- no throughput claim from timestamped runs;
- alternated baseline and candidate runs on mains power;
- isolated and reversible;
- the existing tests run;
- 37 tok/s is a target only;
- report and stop for review before anything further.

Design (variant B of §4.3; same shader, same pipeline):
- `ComputeBackend::groups_matmuls()` (default `false`): the backend asks callers to put
  independent matmuls into one `matmul_many`, because each call is a submission.
  `Placed` answers `true` for a discrete device.
- `ExpertProvider::with_experts_grouped(layer, experts, group, each)`:
  - The provider calls `group` once with every routed expert held, when all of them are
    ready now (resident, and pinned on the device).
  - Otherwise it calls `each` per expert, exactly as `with_experts` does.
  - The default always takes the per-expert path.
  - `StoreProvider` takes the leases from `acquire_many` with `try_next`, so hits,
    heat and recency are counted exactly as today. If any expert is not ready, it hands
    the leases it already holds to the per-expert path.
- The MoE block, with a grouping backend, does three things:
  - one `matmul_many` over every expert's gate and up;
  - host SwiGLU per expert;
  - one `matmul_many` over every down.
- Each op has the same weight, the same inputs and the same kernel as before. So each
  output has the same bits, and the combine is unchanged, in rank order.

What can and cannot be monitored on this device:
- MoltenVK reports `pipelineStatisticsQuery = false` and lists neither
  `VK_KHR_pipeline_executable_properties` nor `VK_KHR_performance_query`. Register
  counts and occupancy cannot be read through Vulkan here.
- Variant B does not change the shader, the specialization constants or the workgroup
  size. Register pressure and occupancy are therefore unchanged by construction.
- A new shader (variant C) would need Xcode's Metal capture to inspect them. It is
  built only if B shows a GPU-time penalty.
- GPU execution time comes from timestamps:
  - per submission;
  - Σ and busy per token;
  - per stage in the kernel test.
- Memory traffic is computed: weight bytes per stage divided by GPU time.

### Step 2 results so far (2026-11-11): grouping adds GPU time; not accepted

Built: `c2a114a` (grouped experts) and `6c8b691` (the `QUBIVEX_VK_PARTS`
experiment, default 1, which is today's behaviour). Each reverts on its own.

Correctness:
- With every expert resident, a decode step is 81 submissions (5 per layer plus the
  head) against 305.
- The tokens match the PR #55 reference.
- Every test passes:
  - the workspace (544);
  - the Vulkan parity tests;
  - `tests/vulkan.rs` with the real OLMoE (CPU prompt logits, Colibri tokens);
  - a new test that resident experts take two submissions, a small RAM cache takes the
    per-expert path, and the bits match either way.

**Kernel test** (`parity.rs`, `grouped_experts`, mains not required: a guide only).
Columns: wall ms per layer, host submit ms, then GPU busy ms from timestamps.
- A is today's shape: one thread per expert, at most 5 at once.
- B is grouped.
- The bits are identical in every shape.

| experts × tokens | A wall | B wall | A submit | B submit | A GPU | B GPU |
|---|---|---|---|---|---|---|
| 1 × 1 | 0.51–0.64 | 0.46–0.49 | 0.04 | 0.04 | 0.18–0.21 | 0.15–0.16 |
| 2 × 1 | 0.52–0.55 | 0.57–0.59 | 0.10 | 0.03 | 0.24 | 0.25 |
| 4 × 1 | 0.68–0.70 | 0.82–0.85 | 0.27 | 0.03 | 0.39–0.41 | 0.50 |
| 8 × 1 | 1.41–1.44 | 1.36–1.39 | 0.56 | 0.04 | 0.78–0.83 | 0.98–1.00 |
| 8 × 4 | 3.63–3.65 | 3.69–3.70 | 0.54 | 0.04–0.05 | 2.86–2.88 | 3.21–3.22 |
| 8 × 10 | 8.28 | 8.29 | 0.55 | 0.04 | 7.47 | 7.72 |
| 8 × 32 | 25.08 | 25.06 | 0.59 | 0.05 | 24.09 | 24.42 |

What the kernel test shows:
- Grouping cuts host submit time by 10–14× at 8 experts.
- It also costs about 20–25% more GPU time at 4–8 experts with 1 token.
- Several things changed nothing:
  - `MVK_CONFIG_USE_MTLHEAP` (untracked resources);
  - spreading the ops over up to 16 command buffers in one `VkSubmitInfo`;
  - a single 24 576-row dispatch of the same bytes, which was slower still
    (1.67 ms, about 17 GB/s).
- One `VkSubmitInfo` per command buffer (still one fence) recovered some wall time at
  8 × 1 (1.26–1.31 ms). GPU time did not move (0.95–0.99).
- MoltenVK 1.4.2 exposes one queue per family and no concurrent-dispatch setting.

**End-to-end diagnostic (battery: not records).** The Mac was on battery, so these
are single 32-token runs, run back to back, for attribution only. No throughput is
claimed from them.

| | ms/token (untimed) | GPU busy (timed) | GPU idle (timed) | submissions |
|---|---|---|---|---|
| base (Step 1a) | 35.89 | 19.59 | 19.25 | 305 |
| B, 1 command buffer | 38.27 | 23.06 | 17.87 | 81 |
| B, 8 `VkSubmitInfo`s | 37.34 | 22.75 | 16.29 | 81 |

Reading:
1. **Grouping adds 3.2–3.5 ms of GPU execution per token.** The kernel test predicted
   this: 16 layers × about 0.2 ms.
   - Dispatches in one command buffer run one after another.
   - Command buffers from separate threads evidently overlap on the GPU.
   - The busy union cannot see that overlap: MoltenVK's timestamps bracket each command
     buffer, and the queue reports them as serial.
   - The finding "submissions never overlap on the GPU" (§10, item 1) therefore holds
     for the timestamp intervals only. It is not evidence that the GPU executed them one
     at a time.
2. **Idle fell by only 1–3 ms, against the 4.5 ms estimated.** The 5 expert threads
   were already hiding most of the expert-stage hand-offs: while one thread runs
   SwiGLU on the host, another's command buffer runs.
   - Per-gap arithmetic is valid for the serial dense hand-offs (3 per layer plus the
     head).
   - It is not valid for the expert stages.
   - That is the overlap the reconciliation above could not separate.
3. **The net result is slower end to end.** The owner's condition, "no GPU execution
   penalty", is not met, so Step 2 as built is not accepted.

Options (owner to choose; nothing further is built until then):
- **(a) Revert** `c2a114a` and `6c8b691`. Back to Step 1a exactly.
- **(b) Keep the mechanism, off for Vulkan.** `groups_matmuls()` returns `false` for
  Vulkan (or for every device), and the parts experiment goes.
  - The grouped provider path and its tests remain for Step 3, where a whole layer is
    one submission anyway, and the same GPU-concurrency question has to be answered
    there.
- **(c) Investigate the GPU penalty first.**
  - It decides Step 3 as well: a layer in one command buffer serialises its dispatches
    the same way.
  - Candidates: a grouped-dispatch shader that keeps several independent matrices in
    flight in one dispatch (variant C, with work interleaved across matrices rather
    than one giant matrix); or measuring under Xcode's Metal capture, to see the real
    overlap and the per-dispatch tail.
- The formal mains-power, alternated records are still owed for whichever option
  ships. They were not run because the machine was on battery.

## 13. Step 2 investigation, option (c): what the GPU actually does (2026-11-11)

The owner chose (c): explain the grouped penalty before anything else. No optimisation was
built. `c2a114a`, `6c8b691` and Step 1a are unchanged.

### 13.1 Method, and why not Xcode

- This Mac has only the Command Line Tools: no Xcode, so no Metal GPU capture, no
  Instruments, no `xctrace`. The same capture would need Xcode installed; the owner can do
  that, and the questions below can then be rechecked in a capture.
- Instead:
  - `spikes/vulkan-step2-metal-timing` runs **MoltenVK's own MSL** for `matmul.comp`
    (Q4_K, taken from MoltenVK's debug log) in the layer's expert shapes.
  - It times them with Metal's clocks, which do not go through Vulkan:
    - `GPUStartTime`/`GPUEndTime` per command buffer;
    - stage-boundary counter samples per encoder.
- The MoltenVK 1.4.2 source was read for how it encodes dispatches, barriers and
  timestamps.
- End to end: untimed bench, alternated, mains power. Step 1a (5e0ca13) against grouped
  (7636c43), 3 rounds. Records: `baselines/2026-11-11-m3pro-vulkan-step2-{base,grouped}-{1,2,3}.json`.
  The tokens match the PR #55 reference in all six.

### 13.2 How MoltenVK maps our submissions onto Metal (source, v1.4.2)

- **Every compute dispatch goes into a serial encoder.** `getDispatchType()` returns
  `MTLDispatchTypeSerial` for everything except occlusion-query accumulation, and there is no
  setting to change it.
  - All dispatches recorded in one `VkCommandBuffer` share one serial
    `MTLComputeCommandEncoder`.
  - Metal runs them one after another, although our recording has no barrier between them.
- **Command buffers are chained by device-wide fences** when Metal residency sets exist
  (they do on this OS):
  - `syncFences` makes each command buffer wait on and update device-level barrier fences.
- **Timestamps.** On Apple GPUs (stage-boundary sampling only), `vkCmdWriteTimestamp` is
  deferred to the end of the current Metal encoder:
  - Each query becomes its own one-byte blit pass (`fillBuffer` on a shared dummy buffer)
    that samples at its start.
  - The BOTTOM sample's blit waits on a fence that the compute encoder updates. The TOP
    sample's blit has **no dependency on the compute encoder**: the compute work may start
    before it.

### 13.3 Findings

**1. Command buffers never overlap on this GPU: the GPU-overlap hypothesis is rejected.**
- By Metal's own times, with no timestamp queries, overlapping command-buffer pairs = 0 in
  every shape and size:
  - 5 submitting threads on one queue;
  - 8–16 command buffers committed back to back;
  - one command buffer per queue on 8 queues.
- Inside one command buffer, consecutive dispatches never overlap either, in a serial
  encoder or in separate encoders. Dispatch *n+1* starts 0.3–3 µs after dispatch *n* ends.
- §12's "command buffers from separate threads evidently overlap on the GPU" is withdrawn.
  §10 item 1's conclusion (the GPU runs one submission at a time) stands, now on a measurement
  that does not depend on timestamps.

**2. The grouped "+20–25% GPU time" is a timestamp artefact, not GPU work.** 8 experts × 1
token, one layer, ms:

| Shape | Metal GPU time (5 runs) | MoltenVK-style timestamps (`TS=1`) |
|---|---|---|
| A per-expert, 5 threads (16 command buffers) | 0.920–0.922 | 0.718–0.760 |
| A per-expert, 1 thread | 1.20 | 1.03 |
| B grouped, serial encoder (2 command buffers) | 0.907–0.920 | 0.918–0.921 |
| B grouped, concurrent encoder (not reachable through MoltenVK) | 0.762–0.763 | — |

- The MoltenVK-style A figure, 0.72–0.76 ms, is below what is physically possible: the 24
  dispatches alone take 16 × 39 + 8 × 37 µs ≈ 0.92 ms by stage counters.
- The timestamps under-count each command buffer by:
  - about 19 µs when other command buffers are queued behind it (5 threads);
  - about 10 µs with one thread;
  - about 3 µs for the grouped shape.
- Why: the TOP sample's blit lands after the compute encoder has started (13.2).
- The harness reproduces the Vulkan kernel test closely:

  | | A timestamps | B timestamps |
  |---|---|---|
  | Vulkan kernel test (§12) | 0.78–0.83 | 0.98–1.00 |
  | harness, `TS=1` | 0.72–0.76 | 0.92 |

- The same holds at 8 × 4 (ms):

  | | A | B |
  |---|---|---|
  | timestamps (Vulkan test) | 2.87 | 3.21 |
  | Metal | 3.15–3.32 | 3.14–3.30 |

- Grouping does not change any dispatch's duration. Inside the grouped command buffer:
  - gate/up dispatches take 38.6–39.3 µs;
  - down dispatches take 36.7–38.1 µs;
  - 1 expert alone: 38.5–38.8 µs and 37.2 µs.

**3. End to end, grouping is still slower, and the cost is host-side serialisation, not GPU
execution.** Untimed, mains, ms/token (median):

| Round | Step 1a | Grouped |
|---|---|---|
| 1 | 39.21 (disturbed: copies 10 ms, P95 51, the same first-run pattern as Step 0) | 38.71 |
| 2 | 35.28 | 36.97 |
| 3 | 34.28 | 38.45 |

- Grouped is slower by about 2–4 ms per token: 26.0–27.1 against 28.4–29.2 tok/s.
- It is not GPU time:
  - (2) shows that per layer the GPU work is the same;
  - Σ host submit time fell from 5.5–6.6 to 1.1 ms.
- The difference is overlap:
  - Step 1a keeps up to 5 expert submissions in flight. While one thread is in its round
    trip (wake, SwiGLU, copies, record, submit), another thread's command buffer runs, so the
    GPU queue rarely empties during the expert stage.
  - Grouped has exactly one submission in flight. So both of its round trips per layer are
    fully exposed: GPU idle from completion through wake, host work and submit, to the start
    of the next command buffer.
  - About 0.2 ms per layer × 16 ≈ 3 ms, which matches.
- §12 reading 2 named this. It is the cause, not a side effect.

**4. The kernel, not memory bandwidth, bounds GPU time, and serial dispatch adds about
7.5 µs per dispatch.** One dispatch, k = 2048, alone in a command buffer:

| Rows | Threadgroups | GPU µs | GB/s (weights) |
|---|---|---|---|
| 1024 | 128 | 57–67 | 18–21 |
| 2048 | 256 | 69–80 | 29–34 |
| 4096 | 512 | 130 | 36.3 |
| 16384 | 2048 | 502 | 37.6 |

- Steady state: 31.4 µs per 1024 rows, **37.6 GB/s**. That is about a quarter of this chip's
  memory bandwidth.
- Time scales with tokens × bytes: 8 × 4 costs 3.4 × 8 × 1, with the same weights. So the
  kernel is bound by its dequantisation and reduction instructions (byte extraction, the
  `optnone` `spvFAdd` helpers from NoContraction), not by memory traffic.
- A 1024-row expert dispatch in a serial encoder costs 39 µs, against 31.4 µs at steady
  state: about 7.5 µs of ramp and drain per dispatch. 24 dispatches per layer ≈ 0.18 ms,
  which is exactly what the concurrent encoder recovers (0.92 → 0.76 ms).
- §12's "single 24 576-row dispatch was slower still (1.67 ms)" does not reproduce under Metal
  timing, which predicts about 0.77 ms. It was measured with the biased timestamps and is not
  in the tree; treat it as unreconciled, not as evidence against large dispatches.
- What can and cannot be read:
  - The device exposes only the `timestamp` counter set through the public API. There are
    no bandwidth, cache, occupancy or utilisation counters; those need Xcode or Instruments.
  - The pipeline reports `maxTotalThreadsPerThreadgroup` 1024 for a 256-wide kernel. So
    register use does not limit the threadgroup size; occupancy beyond that cannot be read.

### 13.4 Wall, GPU busy and GPU idle: what each measures

| Figure | Measures | Bias on MoltenVK/M3 |
|---|---|---|
| wall (ms/token) | host clock, one decode step | none: the only throughput figure |
| GPU Σ / busy (`--vulkan-timestamps`) | per submission: TOP blit sample → BOTTOM blit sample; busy = union | **low**, by about 3–19 µs per submission. The bias grows with the number of submissions and with how many are in flight. Use it only between runs with the same submission shape |
| GPU idle | wall − busy | **high** by the same amount. It is not an independent measurement, so busy + idle = wall by construction |
| host submit Σ, wait Σ | summed over threads | they overlap each other and the GPU, so they are not additive with wall |
| Metal `GPUStart/EndTime` (harness) | command buffer on the GPU | when timed, it includes MoltenVK's timestamp blits |

Corrections to §10:
- Step 0's 19.6 ms of GPU busy per token is an under-count. 256 expert submissions ×
  10–19 µs plus 49 × 3–10 µs ≈ 2.7–5.4 ms is missing, so **true busy is about 22–25 ms**.
  Grouped: 23.1 ms + 81 × 3–10 µs ≈ 23.3–23.9 ms.
- Idle in Step 1a is therefore about 10–13 ms, not 16–19 ms, and the gap per hand-off is
  about 90–115 µs, not about 140 µs.
- §10's "≈ 50 tok/s with every round trip removed" becomes **about 40–45 tok/s** at today's
  kernel speed and serial dispatch.

### 13.5 Implications for Step 3 (a layer in one command buffer)

- **No inherent GPU penalty.** Putting dispatches into one command buffer costs the same GPU
  time as separate command buffers, because the GPU runs command buffers one at a time
  anyway (findings 1 and 2). The "same scheduling penalty" does not exist.
- **The real penalty, lost overlap, would hit Step 3 harder** if any host work stays between
  stages:
  - Step 3 removes every in-flight submission but one.
  - Each remaining host hand-off is then fully exposed, with nothing to hide it.
  - Step 3 pays off only if the layer has **no intermediate host round trip**: attention,
    norms, routing, SwiGLU and the combine must run on the GPU, under the §5 parity rules.
    Fewer submissions with the same number of host round trips on the critical path is what
    Step 2 measured: slower.
- **Serial dispatch remains a cost of about 0.18 ms per layer for the experts** (≈ 2.9 ms
  per token). One command buffer does not remove it; MoltenVK cannot use a concurrent encoder.
- **Measurement:** Vulkan timestamps cannot compare Step 3 with Step 1a, because the
  submission counts differ. Decide on untimed wall, alternated, and attribute GPU time with
  the Metal harness.

### 13.6 Recommendation for the next experiment

1. **Step 2 as built: option (b), keep it off for Vulkan.**
   - Keep the grouped provider path and its tests; Step 3 needs them.
   - `groups_matmuls()` → `false`.
   - Remove the `QUBIVEX_VK_PARTS` experiment (`6c8b691`): it measured nothing that the
     harness does not show more directly.
   - This leaves Step 1a's behaviour on Vulkan.
2. **Next experiment: a multi-matrix dispatch, prototyped in the harness first.** This
   needs no Vulkan shader change until it is reviewed.
   - One dispatch whose grid covers (matrix, row block) for all of a layer's gate and up
     matrices, from a small table of per-matrix offsets, then one for the downs.
   - It aims at the 31.4 µs/1024-row steady state from inside one dispatch, which is what
     the concurrent encoder shows is available. That is about 0.92 → 0.76 ms per layer, with
     each output's arithmetic unchanged (the same per-row reduction), so the same bits.
   - Accept it only if the harness shows Metal GPU time ≤ 0.78 ms at 8 × 1 and no
     regression at 8 × 4.
3. **Step 3, re-scoped:** one command buffer per layer *with the host work between matmuls
   moved to the GPU*, or not at all.
   - The budget is about 10–13 ms of real idle per token.
   - Together with (2), about 3 ms of expert GPU time.
   - Gate it on the same untimed, alternated mains records.
4. **Instrumentation:**
   - Note the timestamp bias in `BENCHMARKING.md`.
   - Optionally, record the TOP timestamp *after* a first, empty dispatch, so that MoltenVK
     closes an encoder before it. Check that against the harness before trusting it.
5. **If the owner installs Xcode**, one Metal capture of a decode step each way should
   confirm findings 1–4. It would also give the cache, occupancy and bandwidth counters that
   the public API does not expose.

## 14. Option (b) and the fused-dispatch prototype (2026-11-11)

The owner chose (b), then a harness prototype of a fused expert dispatch.
- No production shader changed.
- Nothing was pushed.
- c2a114a and Step 1a are unchanged.

### 14.1 Option (b), as built

- `825b84b`: `DiscreteDevice::groups_matmuls()`, default `false`, decides whether to group.
  - The simulator answers `true`, so the grouped expert path and its test
    (`resident_experts_are_grouped_into_two_submissions`) stay exercised.
  - Vulkan inherits `false`. Reverting this one commit turns grouping back on for Vulkan.
- `f4dd9c3`: `git revert` of the `QUBIVEX_VK_PARTS` experiment (6c8b691). The history is
  kept; the experiment is no longer in the code.
- Records: `baselines/2026-11-11-m3pro-vulkan-optb-{base,optb}-{1,2,3}.json`, untimed,
  alternated, mains. The tokens match the PR #55 reference in all six, at 305 submissions
  per token.

  | Round | Step 1a (5e0ca13), ms/token | (b) (f4dd9c3), ms/token |
  |---|---|---|
  | 1 | 35.10 | 35.95 |
  | 2 | 35.70 | 35.76 |
  | 3 | 35.68 | 35.63 |

  Equal within the run-to-run spread: (b) gives back Step 1a's behaviour.

### 14.2 The fused dispatch: design

`spikes/vulkan-step2-metal-timing`: `fuse.py`, the `FUSED` shape and `check` mode in
`main.m`, and `cpucheck/`.

- **One dispatch per stage covers every matrix of the stage.**
  - Gate/up: 16 matrices × 128 row blocks = 2048 threadgroups.
  - Down: 8 × 256.
  - The grid's x axis concatenates the matrices' row blocks, and y is the token, as before.
- **A table replaces the push constants.** One entry per matrix: the six push-constant
  fields (`n`, `k`, `row_bytes`, `w_base`, `x_base`, `y_base`) and the matrix's first row
  block.
- Each threadgroup:
  - finds its matrix `m`;
  - copies that entry into the local that the body reads as its push constants;
  - sets its row block to `wg.x − first[m]`;
  - then runs **MoltenVK's generated body unchanged**. `fuse.py` changes only the entry
    signature and adds 7 lines.
- Two lookups, both measured:
  - a search of the table;
  - `m = wg.x / blocks_per_matrix`, which is valid because every matrix of a stage has the
    same shape: `--equal`, the default.
- `m` is uniform within a threadgroup, so the body's barriers stay uniform.

### 14.3 Mathematical and execution approach

- Each output is still one row's dot product, computed by one 32-lane group, with the same:
  - weights (`w_base` is the matrix's byte offset in the bound range);
  - input row (`x_base`);
  - dequantisation;
  - 64-lane FMA accumulation;
  - reduction tree.
- So every output is computed by the same instruction sequence on the same operands as in
  its own dispatch, and has the same bits. Fusing only changes the mapping of threadgroups
  to work, not the arithmetic.
- **Why it is faster.** In a serial encoder each dispatch pays about 7.5 µs to ramp up and
  drain (§13, finding 4). One dispatch per stage pays it twice per layer instead of 24 times,
  and keeps the GPU at the kernel's large-dispatch rate (about 37 GB/s) across matrix
  boundaries. It recovers what a concurrent encoder would, without needing one. MoltenVK
  cannot use a concurrent encoder.

### 14.4 Correctness

`run.sh check`, for 1, 3 and 8 experts × 1, 2, 4, 10 and 32 tokens: every gate, up and down
output, 2 286 592 in all.
- Weights use the parity tests' rule for block formats; activations use their `acts`.
- Each down has its own input.

| Check | Result |
|---|---|
| fused vs unfused, on the GPU, bit for bit (both lookups) | **0 of 15 configurations differ** |
| fused and unfused vs `CpuBackend::matmul`, bit for bit (both lookups) | **0 of 30 differ** |
| positive control: one output bit flipped | detected (1 of 30) |
| non-finite outputs | none |

The unfused harness kernel also matches the CPU bit for bit. That confirms the harness runs
the same arithmetic as the Vulkan backend.

### 14.5 Benchmarks

Metal GPU time per layer (ms). Two command buffers, as on the grouped path, with no timestamp
queries. Mains; 5 interleaved repeats, 200 samples each, medians. The range is over repeats.

| Experts × tokens | A: Step 1a shape (16 cbs, 5 threads) | B: grouped, serial | B: concurrent encoder (reference only) | **F: fused, search** | **F: fused, O(1)** |
|---|---|---|---|---|---|
| 8 × 1 | 0.924–0.927 | 0.932–0.937 | 0.765–0.767 | 0.789–0.793 | **0.783** |
| — of which gate+up / down | — | 0.633 / 0.302 | 0.504 / 0.261 | 0.518 / 0.272 | 0.513 / 0.270 |
| 8 × 4 | 3.169–3.175 | 3.161–3.163 | 3.014–3.015 | 3.098–3.099 | 3.085 |
| 8 × 10 | 7.689–7.691 | 7.666–7.667 | 7.520 | 7.721 | — |
| 4 × 1 | 0.461–0.462 | 0.469–0.470 | 0.390–0.391 | 0.397–0.401 | — |
| 1 × 1 | 0.116–0.118 | 0.117–0.118 | 0.107–0.111 | 0.108–0.113 | — |

The O(1) column comes from 3 alternations against the search. Every repeat agrees to within
0.004 ms.

- **The target (≤ 0.76 ms at 8 × 1) is not met:**
  - fused: 0.783 ms, **−15.4% against Step 1a's shape** (0.926);
  - the concurrent encoder: 0.766 ms.
  - About 9 µs per stage remains between them, inside the dispatch. The lookup accounted for
    7 µs of the gap, which the O(1) lookup removed. Neither a table search nor the stage
    structure explains the rest.
- **At 10 tokens fused is 0.4% slower** (7.72 against 7.69). There the dispatches are long
  and the ramp cost is negligible. A fused path should apply at small token counts (decode)
  only, or be checked again for prefill.

### 14.6 Against Step 1a, end to end: what the prototype does and does not buy

Per token (16 layers), fused saves 16 × (0.926 − 0.783) ≈ **2.3 ms of GPU time**. But
fusing requires grouping (one submission per stage), and §13 measured grouping at 2–4 ms per
token slower end to end, because only one submission is in flight. So:

- **Fused + grouped as today (two submissions per layer, host SwiGLU between): about even
  with Step 1a.** Estimate: grouped's 37–38.7 ms − 2.3 ≈ 35–36 ms, against 34.3–35.7. Not
  worth building on its own.
- **Fused + SwiGLU on the device, one submission per layer for the experts**:
  - This removes one exposed round trip per layer, about 0.1–0.2 ms × 16 = 1.6–3.2 ms.
  - Estimate: about 32–34 ms per token (29–31 tok/s), a gain of about 1–3 ms over Step 1a.
  - This is an estimate built from §13's gap measurements, not a measurement.

### 14.7 Step 3, rescoped: removing host round trips inside a layer

Today a decode layer has these round trips on its critical path:
- QKV;
- O;
- router;
- the expert stages: 4, partly hidden by the 5 in-flight submissions.

What each host operation needs in order to move to the device:

| Host operation | Round trip it forces | What moving it needs | Parity risk (§5) |
|---|---|---|---|
| SwiGLU, `g·silu(g)` with `silu(x) = x/(1+exp(−x))` | gate/up → down (per layer) | a small elementwise kernel between the fused stages in one command buffer, with a barrier | `exp`: port `qubivex_compute::math::exp` (basic ops and `fma` only), then **division**: Vulkan allows 2.5 ULP. Needs the probe and/or an fma correction. Medium |
| Combine (rank order, `fma`) | none by itself: it follows the last wait | needed only when a whole layer is one submission | `fma` is already probed at open. Low |
| Router matmul + softmax + top-8 | router → experts | top-k on the device; the experts' device addresses from a table | `exp`, division, **tie order** in top-k. Leases and residency are host decisions: valid only when every expert of the layer is resident (the grouped path's condition), with the host reading the choice back afterwards for statistics. Medium–high |
| RMS norms (attention, FFN, final), QK norm | part of QKV → attention → O | reduction via `reduce`'s shape, then `1/sqrt` | **`sqrt` and division**: about 3 ULP allowed. Metal's non-fast mode is probably correct, but must be proven by probe. Medium |
| Clip, residual adds | — | elementwise | exact. Low |
| RoPE | QKV → attention | upload the host's f64-rounded table, as on Metal | low |
| KV store + attention (scores, softmax, values) | QKV → attention → O | **a device-resident KV cache**: today `qubivex-kv-cache` owns it on the host. This is an architecture change (ADR-level) | dot via `reduce::dot`'s shape, `exp`, division. High |

Proposed order, each step gated on untimed, alternated mains wall-clock:

1. **3a, experts in one submission per layer:** fused gate/up → SwiGLU → fused down. The
   first step that should beat Step 1a: about 1–3 ms per token, estimated.
2. **3b, router and routing join that submission:** for fully resident layers only.
   2 round trips per layer remain: QKV and O+router+experts.
3. **3c, norms, RoPE and attention on the device, with a device KV cache:** one submission
   per layer, then per token. The wall clock then approaches GPU busy: about 21–23 ms with
   fused experts, about 43–47 tok/s. This is the only route to the 43.5 tok/s target found so
   far.

### 14.8 Recommendation for the next Vulkan step (not started; for review)

Two harness checks come before any production shader change:

1. **SwiGLU parity in the harness.** A Metal kernel built from MoltenVK's translation of a
   GLSL SwiGLU (`exp` port, NoContraction), checked bit for bit against
   `CpuBackend`'s `swiglu` by `cpucheck`, over the full float range seen in decode plus edge
   cases (overflow of `exp`, subnormals). If division is not exact on MoltenVK, add the fma
   correction step and prove it there.
2. **How the fused dispatch reaches eight experts' weights.** Each expert's device copy is
   its own `VkBuffer` today (tied to its RAM slab), while the harness used one buffer with
   offsets. Measure in the harness, against the 0.783 ms result:
   - (a) the expert copies sub-allocated from one device arena (what the harness shows today);
   - (b) a descriptor array indexed by matrix (Metal argument buffer);
   - (c) buffer device addresses (`VK_KHR_buffer_device_address`, a GPU address in the table).

Then the Vulkan Step 3a would be:
- `matmul.comp` gains the fused entry: same body, table in a storage buffer, chosen by
  specialisation constant, so the unfused pipeline is unchanged;
- a SwiGLU shader;
- the grouped provider path, re-enabled for Vulkan only in this combined form: a device
  capability, not the old `groups_matmuls`;
- isolated commits, each revertible: fused entry, SwiGLU shader, binding change, enabling
  switch;
- parity tests per shader against the CPU, `tests/vulkan.rs` with OLMoE, and alternated
  mains records against Step 1a.

It is accepted only if the wall clock beats Step 1a beyond the run-to-run spread (about
0.7 ms).

## 15. Harness checks before Step 3a: SwiGLU parity and expert weight binding (2026-11-11)

Harness only, as approved:
- No production shader changed.
- Step 1a is unchanged.
- Nothing was pushed.
- Code: `spikes/vulkan-step2-metal-timing/{swiglu/, binding.m, fuse.py --binding, cpucheck/}`;
  `run.sh swiglu-check | swiglu-time | binding-check | binding` reproduce everything.
- Timings were taken on mains, 3 interleaved rounds. The exhaustive parity run was on
  battery, which does not affect bits.

### 15.1 Check 1: SwiGLU numerical parity

**What was tested.** `swiglu/swiglu.comp` is a GLSL port of `ops::swiglu`
(`g·u/(1+exp(−g))`, written `s = g/(1+exp(−g)); g = s·u`), with the `exp` port of
`math::exp`.
- It is compiled exactly as `matmul.comp`: `glslc -O`, then NoContraction on every add,
  subtract and fma.
- It is run as **MoltenVK's own MSL**, produced by creating the pipeline on MoltenVK
  (`mvkmsl.c`), compiled as MoltenVK compiles it: MathMode Safe, floating-point functions
  Precise.
- It is bound through an argument buffer, as MoltenVK binds it.

Variants:
- `-O0`, the same source;
- `precise`: `precise` on the `exp` intermediates, `-O`;
- **control**: the native Metal backend's parity-tested `swiglu`/`qexp`, extracted from
  `kernels.metal`.

**Inputs: every f32 bit pattern of g (2³²), twice.**
- Pass 1: u = 1.
- Pass 2: u is a pseudo-random bit pattern per element, covering every class: zeros,
  subnormals, normals, infinities, NaNs.
- Reference: a C port in the harness. It is **proven equal to `ops::swiglu` itself over all
  2 × 2³² inputs**: an order-independent hash of every output matches the Rust
  implementation's (`cpucheck` `rusthash`): `68f49b9c9e53d30f` and `8cfbe379131474ec`.

**Results.** Differences from the CPU, of 4 294 967 296 per pass:

| Class | `-O` (as `build.sh` builds) | `-O0` | `precise` | control (Metal backend) |
|---|---|---|---|---|
| input g or u subnormal (pass 1 / 2) | 16 777 212 / 15 684 871 | same | same | same |
| intermediate silu(g) subnormal | 16 777 214 / 10 003 426 | same | same | same |
| reference output subnormal (pass 2) | 181 775 278 | same | same | same |
| exact silu(g) subnormal, rounds up to `FLT_MIN` | (in "other") | 2 / 0 | 2 / 0 | 2 / 0 |
| NaN vs NaN, other payload | 16 777 213 / 33 295 226 | same | same | same |
| +0 vs −0 | 0 | 0 | 0 | 0 |
| **other, unexplained** | **472 469 120 / 443 609 392** | **0** | **0** | **0** |

1. **The `-O` build breaks parity: a shader-pipeline bug, not a GPU one.**
   - `glslc -O` folds `exp`'s rounding step `(x·log2e + SHIFT) − SHIFT` into `x·log2e`,
     and `int(kf)` then truncates instead of rounding.
   - This happens because the optimiser runs *before* `build.sh` adds NoContraction, so it
     treats float arithmetic as algebra.
   - Result: 1-ulp errors on ordinary values. For example, g = 8.94e-8 gives `33400001`
     where the CPU gives `33400002`.
   - `matmul.comp` contains no such cancellation and is proven bit-exact. But **any new
     shader using a rounding idiom must be built `-O0` or use `precise`**, and must be gated
     by an exhaustive or targeted test.
2. **`-O0`, `precise` and the control agree exactly with each other.** They differ from the
   CPU only in:
   - **Subnormals.** The GPU flushes them: ADR-011's exception. It includes the 2 values
     g = ±`0x00ffffff`: the exact g/2 is 2⁻¹²⁶ − 2⁻¹⁵⁰, a subnormal midpoint that IEEE
     rounding takes up to `FLT_MIN` but the GPU flushes before rounding.
   - **NaN payloads.** The GPU returns `7fc00000`; the CPU propagates the payload, and NaN-ness
     is always preserved. The native Metal backend already behaves this way. **The parity
     contract (ADR-011 §1) does not say whether NaN payloads count; the owner should decide
     on the wording.**
3. **No other differences:**
   - **Division** `g/(1+e)` is correctly rounded in MoltenVK's Safe mode on this device, so
     §5's 2.5-ULP concern does not arise here. Other Vulkan drivers still need the probe.
   - **`exp` overflow/underflow edges** are clean.
   - **Signed zero:** 0 cases, including `precise`'s `spvFMul = fma(l, r, +0)` (a −0
     product feeds `+ SHIFT`, so it never shows).

**Performance** (mains):

| n (= experts × tokens × 1024) | GPU, alone in a command buffer | host `ops::swiglu`, one thread |
|---|---|---|
| 1 024 | 3.5 µs | 4.5 µs |
| 8 192 (8 × 1) | 3.0 µs | 11.7 µs |
| 32 768 (8 × 4) | 4.0 µs | 27.3 µs |
| 81 920 (8 × 10) | 5.9 µs | 67.0 µs |

Inside the one-submission expert block, SwiGLU and its two encoder boundaries add at most
about 3 µs of GPU time (SE against SF below).

### 15.2 Check 2: expert weight binding

**Setup** (`binding.m`):
- Each layer has a pool of **64 resident experts**, OLMoE's count. The arena variant copies
  them into one buffer; the other variants give each expert its own private buffer.
- **Every call routes to a new random set** of e experts. Every shape in a round uses the
  same set.
- Shapes:
  - **SA**: Step 1a's shape. Unfused, a thread per expert, at most 5 in flight, the expert's
    own buffer bound at the matrix offset.
  - **SF**: fused gate/up, then fused down, in 2 command buffers.
  - **SE**: the expert block in one command buffer: fused gate/up → SwiGLU (the `-O0`
    kernel) → fused down. 3 encoders, as MoltenVK splits at pipeline barriers.
- Bindings (`fuse.py --binding`, same body):
  - **arena**: one buffer, slot offsets.
  - **array**: a pointer array indexed per matrix. This is what MoltenVK makes of a
    descriptor array on Metal 3.
  - **bda**: the buffer address in the matrix's table entry.
  - **bda-all**: bda, encoded as MoltenVK encodes a pipeline that uses buffer device
    addresses. `MVKDevice::encodeGPUAddressableBuffers` calls `useResource` on **every**
    addressable buffer per encoder, here the 64-expert pool plus 960 stand-ins (16 layers × 64).

**Correctness:**
- All 45 comparisons are bit-identical:
  - SF in every binding against SA (unfused, CPU-exact);
  - SE in every binding against SE-arena.
- `cpucheck` `blockcheck` checks the SE expert block against `CpuBackend::matmul` →
  `ops::swiglu` → `CpuBackend::matmul`, for e = 1, 3, 8 × s = 1, 4, 10 with routed experts.
  **0 of 9 configurations differ.** The positive control (one flipped bit) is detected.

**Timing** (mains, ms; ranges over 3 interleaved rounds of 200 calls; "wall" is one layer's
expert block with a host wait per command buffer):

| e × s | | SA (Step 1a) | SF-arena | SF-array | SF-bda | SF-bda-all | SE-arena | SE-array | SE-bda | SE-bda-all |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 × 1 | GPU | 0.116 | 0.111 | 0.112–0.113 | 0.110–0.111 | 0.110 | 0.111–0.115 | 0.116 | 0.114–0.115 | 0.114 |
| | wall | 0.41–0.43 | 0.41–0.44 | 0.41–0.45 | 0.41–0.44 | 0.57 | **0.26–0.28** | 0.27–0.28 | 0.26–0.28 | 0.36–0.37 |
| 1 × 4 | GPU | 0.397 | 0.398 | 0.401–0.405 | 0.399 | 0.398–0.399 | 0.401–0.402 | 0.405 | 0.403 | 0.402–0.403 |
| | wall | 0.69–0.73 | 0.70–0.73 | 0.71–0.73 | 0.69–0.73 | 0.85–0.87 | **0.55–0.57** | 0.56–0.58 | 0.56–0.57 | 0.65–0.66 |
| 1 × 10 | GPU | 0.961 | 0.975–0.976 | 0.983–0.986 | 0.977–0.978 | 0.977–0.978 | 0.976–0.977 | 0.985–0.986 | 0.980–0.981 | 0.979 |
| | wall | 1.33–1.36 | 1.33–1.35 | 1.33–1.37 | 1.33–1.36 | 1.50–1.55 | **1.18** | 1.19–1.21 | 1.19–1.21 | 1.26–1.28 |
| 3 × 1 | GPU | 0.346–0.347 | 0.301–0.305 | 0.305–0.309 | 0.302–0.306 | 0.302 | 0.301–0.304 | 0.304–0.308 | 0.302–0.305 | 0.302–0.305 |
| | wall | 0.55–0.59 | 0.58–0.63 | 0.59–0.64 | 0.59–0.63 | 0.75–0.76 | **0.45–0.47** | 0.46–0.48 | 0.45–0.47 | 0.54–0.56 |
| 3 × 4 | GPU | 1.190–1.192 | 1.165 | 1.177–1.180 | 1.169 | 1.169 | 1.168–1.169 | 1.180 | 1.172 | 1.172 |
| | wall | 1.39–1.40 | 1.55–1.56 | 1.56–1.57 | 1.55–1.56 | 1.75 | **1.36–1.37** | 1.40 | 1.39 | 1.47–1.48 |
| 3 × 10 | GPU | 2.885–2.888 | 2.894 | 2.922 | 2.904–2.905 | 2.904 | 2.897 | 2.925 | 2.907 | 2.907 |
| | wall | **3.09** | 3.32 | 3.36–3.37 | 3.35 | 3.53 | 3.12 | 3.16 | 3.14 | 3.21 |
| **8 × 1** | GPU | **0.927–0.929** | 0.784–0.785 | 0.792–0.793 | 0.786–0.787 | 0.786–0.787 | **0.786–0.787** | 0.795 | 0.789 | 0.789 |
| | wall | 1.36–1.38 | 1.15–1.16 | 1.17 | 1.16 | 1.35–1.37 | **0.98** | 1.01 | 1.00 | 1.08–1.09 |
| 8 × 4 | GPU | 3.175–3.176 | 3.085 | 3.115 | 3.095 | 3.095 | 3.088–3.089 | 3.118–3.119 | 3.099 | 3.099 |
| | wall | 3.58 | 3.51–3.52 | 3.56–3.57 | 3.55 | 3.72–3.74 | **3.31–3.32** | 3.35–3.36 | 3.33–3.34 | 3.41–3.42 |
| 8 × 10 | GPU | 7.693–7.698 | 7.695 | 7.769 | 7.722 | 7.722 | 7.698 | 7.774–7.775 | 7.726 | 7.726 |
| | wall | 8.10–8.11 | 8.14 | 8.24–8.25 | 8.19–8.20 | 8.37–8.38 | **7.94** | 8.02–8.03 | 7.97–7.98 | 8.05 |

Host encoding per call, µs:

| Variant | 8 × 1 | 8 × 10 |
|---|---|---|
| SA | 71–79 (16 command buffers) | 134–143 |
| SF / SE, arena | 4–20 | 4–23 |
| SF / SE, array | 4–29 | 4–29 |
| SF / SE, bda | 4–29 | 4–29 |
| bda-all | **27–96** | **27–96** |

**Binding overhead:**
- GPU time, against arena:
  - **bda: +0.3%**;
  - **array: +1.0%**, one more dependent load per threadgroup.
  - These are consistent across every size, and small next to fusion's −15% at 8 × 1.
- Host and wall time: arena, array and bda are equal within noise (≤ 0.03 ms).
- **bda-all costs +0.1 ms per SE call and +0.2 ms per SF call** (wall), with no GPU change:
  this is the cost of MoltenVK making every addressable buffer resident per encoder. It
  grows with the number of resident experts, not with routing. Through MoltenVK 1.4.2,
  buffer device addresses would give back about 1.6 ms per token of what the expert block
  gains.

**GPU execution:**
- Fusion pays at small token counts: 8 × 1 −15%; 3 × 1 −13%; 8 × 4 −3%.
- It is neutral at 8 × 10 and **slightly worse at 1 × 10 and 3 × 10** (+0.3–1.5%). Below
  4 tokens per expert the fused path should be used; above that, measure again for prefill.
- The SE block's single submission removes 0.18 ms of wall per layer at 8 × 1 compared with SF
  (0.98 against 1.16): one host round trip.

### 15.3 Memory and the expert streaming architecture

Facts:
- **Expert sizes are not uniform across layers.** Layers 0, 1, 4, 7, 10, 13, 14 and 15 have
  **Q6_K** down projections: 4 079 616 bytes per expert. The other 8 have Q4_K: 3 538 944
  bytes.
- Within a layer, every expert has the same size and dtypes. So a fused stage always has one
  dtype: one pipeline per (stage, dtype).
- The RAM cache already has **per-layer pools of fixed `slot_bytes`**. The device cache is
  separate: a global LRU under its own budget, **one `VkBuffer` per expert**, uploaded on
  first use, inclusive of the RAM cache (dropped when the slab is reused), and pinned by
  cloning the buffer while in use.
- Device limits: 4 GiB per storage binding (`maxStorageBufferRange`), 10.7 GB per buffer.
  The bench's expert cache is 3.72 GB, all 1024 experts.

| Binding | What the device cache becomes | Selective loading / changing residency | Memory | Risk |
|---|---|---|---|---|
| **arena** | Two slot-class arenas (3.54 MB and 4.08 MB slots), each one buffer: 1.81 GB and 2.09 GB for the whole model, so each fits one binding. LRU across layers within a class; an expert's address is (class, slot). | A load becomes a copy into a free slot (no allocation; today each load creates a `VkBuffer` on MoltenVK). Eviction frees a slot. Pins become per-slot counts. The arena's size is the budget, fixed at load, or grown in chunks (then up to a few bindings, as with `array`). | **No fragmentation** (uniform slots per class) and no padding. One allocation per class instead of up to 1024. | A real change to `qubivex-runtime::discrete`: a slot allocator and pin counts replace `Arc` buffer clones. The shader is unchanged from the harness. |
| **array** | Unchanged: one buffer per expert. Each fused dispatch writes a descriptor array of the routed experts' buffers (descriptor indexing; `runtimeDescriptorArray` and non-uniform indexing are supported; the non-update-after-bind limit is 31 storage buffers per stage, so it needs update-after-bind). | As today. | As today: one allocation per expert. | +1% GPU. MoltenVK's residency handling for descriptor arrays is **not measured** here; a Vulkan-level check is needed. |
| **bda** | Unchanged buffers; addresses go in the table. | As today. | As today, plus `VK_KHR_buffer_device_address`. | **Through MoltenVK, every resident expert buffer is made resident per encoder (+0.1–0.2 ms per call)**, growing with the cache. Not recommended on MoltenVK 1.4.2. |

### 15.4 End-to-end projection (estimate, not a measurement)

The harness wall does not translate directly to the bench. The harness put grouped (B, 2
submissions) at 0.05–0.1 ms per layer *faster* than SA, while end to end it was 2–4 ms per
token *slower* (§13). So I anchor on the measured grouped figure, the same in-flight
structure, rather than on Step 1a:

- end to end grouped: 37.0–38.7 ms per token;
- harness wall, SE-arena against B at 8 × 1: 0.98 against 1.28–1.32 ms, which is
  −0.30 to −0.34 ms per layer, × 16;
- **SE-arena ≈ 31.6–33.9 ms per token (29.5–31.6 tok/s), against Step 1a's 35.1–35.7: a gain
  of about 1.2–4.1 ms per token.**
- With bda through MoltenVK (bda-all), take about 1.6 ms off that gain.

Assumptions and gaps:
- Q4_K down only: **the 8 Q6_K-down layers were not timed fused** (same body, so the bit
  argument holds, but the speed is unmeasured).
- The anchor assumes the harness-to-bench offset is the same for SE as for B.
- Host combine and copies are unchanged.
- The 43.5 tok/s target still needs Step 3b/3c (§14.7).

### 15.5 Recommendation for Vulkan Step 3a (for review; not started)

1. **Binding: arena, two slot-class arenas.**
   - It is the fastest on every measured axis.
   - It turns expert loads into slot copies.
   - It uses no Vulkan feature beyond what Step 1a uses.
   - It avoids MoltenVK's per-encoder residency cost.
   - The work is in the device cache (slot allocator, pins), not in the shader.

   If that cache change is judged too large for one step, `array` is the fallback (+1% GPU),
   after a Vulkan-level check of MoltenVK's residency handling for descriptor arrays.
   Not bda on MoltenVK 1.4.2.
2. **SwiGLU shader: `-O0` (or `precise`), never `glslc -O`.**
   - Keep `swiglu-check` (exhaustive, about 2.5 min) as its gate.
   - Add a targeted parity test to the crate: the rounding boundaries of `exp`, the overflow
     and underflow edges, ±0, subnormal boundaries, NaN.
   - Give `build.sh` a per-shader optimisation flag.
3. **Before the Vulkan change:**
   - time the fused down on **Q6_K** in the harness;
   - decide the NaN-payload wording of the parity contract.
4. **Step 3a, as before:**
   - fused entry in `matmul.comp` by specialisation constant;
   - SwiGLU shader;
   - arena device cache;
   - enable the one-submission expert block for Vulkan, for ≤ 4 tokens per expert.

   Separate commits, each revertible. Accept only on alternated, untimed mains wall-clock
   beating Step 1a beyond the run-to-run spread (about 0.7 ms).
