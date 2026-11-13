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

## 16. Step 3a: the expert block in one submission (2026-11-11)

Built as the owner directed after §15 ("two-arena design and fused expert block").
Nothing was pushed. Every number below is *recorded*: mains power, untimed, alternated
with the permanent Step 1a baseline (tag `bench/vulkan-step1a`, driver
`scripts/bench-vulkan-baseline`; BENCHMARKING.md "Vulkan baseline").

### 16.1 What was built (separate commits, each revertible and each compiling alone)

| Commit | What |
|---|---|
| Compute | `GatedFfn` and `ComputeBackend::{fuses_gated_ffn, gated_ffn_many}`; `DiscreteDevice::{fuses_gated_ffn, gated_ffn_many, alloc, write, max_arena}`. Defaults keep today's steps, so no backend changes until it opts in. `math::swiglu` moves into `qubivex-compute`, unchanged. The MoE stage asks the backend by shape (blocks, largest rows per expert) and, when every routed expert is resident, runs the layer's experts as one `gated_ffn_many`. Generic to any SwiGLU MoE; no model-specific path. |
| Runtime | The device expert cache in fixed-slot arenas, one per expert size class (OLMoE: Q4_K-down 3 538 944 B and Q6_K-down 4 079 616 B). A copy is a write into a free slot; pins are per slot; slots follow the budget's distribution over layers, capped by the RAM cache (inclusive) and the device's largest addressable arena. No arenas when the device has none or the budget is below a slot per layer (the old per-expert buffers). `experts.used` is now arenas plus single copies (the ledger's charge); `experts.held` is the copies. The sim gets arenas and an opt-in fused block (`SimConfig::fuse_rows`). |
| Shaders | `matmul.comp`'s body moved to `matmul_body.glsl`: **`matmul.spv` is byte-identical**, so the unfused pipeline did not change. `matmul_fused.comp`: the same body, each workgroup's matrix from a table (O(1) lookup, `wg.x / blocks`), rows beyond a matrix's tokens skipped uniformly. `swiglu.comp`: the harness-proven port, **built -O0** (`build.sh` has a per-shader level). |
| Vulkan | `gated_ffn_many`: fused gate/up dispatch(es) → barrier → SwiGLU → barrier → fused down dispatch(es), one command buffer, intermediates in device-local memory. Matrices are grouped by (buffer, dtype, rows), so experts outside the arena or of mixed dtypes still run, as extra dispatches in the same submission. `alloc`/`write` give the cache its arenas. `--vulkan-fuse-rows N` (default 8, 0 off). |
| Threshold | Default 8 rows per expert, measured (§16.4). |

### 16.2 Correctness

- **Fused block** (`tests/fused.rs`), against the CPU's separate steps, bit for bit: Q4_K/Q4_K, Q4_K/**Q6_K**, Q6_K/Q6_K, Q8_0, f32/f16, bf16/f32 × 1, 3, 8 experts × 1, 2, 4, 10 rows (mixed rows per expert), in an arena and in separate buffers, one submission each; OLMoE's real shapes (2048 × 1024, 8 experts, Q4_K and Q6_K down) at 1 and 4 rows; empty blocks and shared inputs. All equal.
- **SwiGLU on the device, exhaustive** (`--ignored`, 27 s): all 2 × 2³² inputs (u = 1; u random). 8 298 858 322 equal, 240 810 096 subnormal (ADR-011), 50 266 174 NaN payload, **0 other**. The targeted test (every `exp` rounding boundary, overflow/underflow edges, ±0, subnormal and `FLT_MIN` boundaries, infinities, NaNs, 4 M random) runs in the normal suite.
- **Positive control:** with SwiGLU built `-O`, the targeted test (914 162 other), the fused-block test and the shared-input test all fail.
- **End to end:** `tests/vulkan.rs` with the real OLMoE (CPU prompt logits, Colibri's 12 tokens, a 64-expert device cache that evicts) passes with fusion on; every bench record's tokens equal the Step 1a reference (single-stream, 9 + 4 + 6 records) and every concurrent run's tokens are identical across modes.
- Sim: `fused_expert_blocks_are_one_submission` (4 per layer + head; shape selection; concurrent sequences).

### 16.3 Single stream (`qubivex bench`, 10-token prompt, 128 generated, 5 threads, 1 + 5 runs)

Session 1, threshold 4 (records `2026-11-11-m3pro-vulkan-step3a-{base,fused,unfused}-{1,2,3}.json`):

| Round | Step 1a, ms/token | Fused, ms/token | Arenas only (`--vulkan-fuse-rows 0`) |
|---|---|---|---|
| 1 | 35.88 | **33.02** | 35.40 |
| 2 | 35.78 | **33.00** | 35.40 |
| 3 | 35.59 | **32.99** | 35.57 |
| tok/s | 27.9–28.1 | **30.28–30.31** | 28.1–28.3 |
| submissions per decode token | 305 | **65** | 305 |
| prefill, ms | 200.4–201.0 | 199.1–201.5 | 201.2–203.2 |

Session 2, threshold 8 (`…-step3a-r8-*`; the machine was slower, swap above 9 GB): Step 1a
38.36 / 38.54, fused **34.91 / 34.93** ms/token (−9%); prefill 214.4 / 214.9 → 207.4 /
207.9 ms (the 10-token prompt now fuses some layers).

**−2.8 ms per token (−7.8%) against Step 1a, four times the run-to-run spread; 30.3
tok/s, past target 1.** The arenas alone change nothing: the gain is the single
submission, not the binding.

### 16.4 Concurrent inference (the batch example, 32 tokens per sequence, median of 3)

Concurrent tok/s (N threads each calling `forward`, combined at the engine's turn).
Rounds 2–3 of session 1 (round 1 ran before the machine settled: the baseline itself
fell from 27.9 to 26.0 between rounds 1 and 2; log `…-step3a-concurrent.txt`):

| N | Step 1a | Fused (≤ 4 rows) | Gain | Arenas only |
|---|---|---|---|---|
| 1 | 25.9–26.2 | 28.9–29.0 | +11% | 26.3–26.4 |
| 2 | 31.9 | 35.5–35.7 | +11.5% | 31.9–32.0 |
| 4 | 37.1–37.2 | 40.7–40.8 | +9.5% | 37.1–37.2 |
| 8 | 41.4 | 43.3–43.4 | +4.8% | 41.3–41.4 |

Threshold, 2 alternated rounds (`…-step3a-threshold.txt`): at N = 8, ≤ 4 gives 43.3–43.4,
**≤ 8 gives 43.7–43.8**, ≤ 64 43.8 (with 8 sequences no expert exceeds 8 rows); below N = 8
all equal. So the default is 8. A 128-token prompt (`…-step3a-p128-*`) never fuses: prefill
2317–2322 ms in all three variants, 1615 submissions each; its decode is 35.2–35.3 ms fused
against 38.7–39.3 for Step 1a.

The gain narrows with N because a batched pass already amortises each round trip over
N tokens; fusion removes round trips, not GPU work.

### 16.5 What this shows

Optimising the execution path, not the kernel: the fused block runs the same shader body
with the same GPU work per output. It removes 240 submissions per token (305 → 65) and
the host SwiGLU between the expert stages, and that alone is 2.8–3.5 ms per token.

### 16.6 Owner decisions (2026-11-11)

Step 3a is accepted. The open points are settled:
1. **NaN payloads and signs** are outside the cross-backend parity contract; NaN
   classification stays inside it, and subnormals stay ADR-011 §1's exception (ADR-011
   amendment 7). The tests already classified this way; their wording now cites it.
2. **ADR-019 amendment 3** records the gated-FFN block op and its shape selection, the two
   fixed-slot arenas and their accounting (`used` vs `held`, working buffers), the
   per-shader `-O0` rule, and the NaN wording.
3. **Next: Step 3b**, routing in the same submission, investigated separately.

## 17. Step 3b: routing in the expert submission (design, 2026-11-11)

Owner brief (2026-11-11): move MoE routing into the same GPU execution path as the fused
expert block, behind its own switch, with Step 3a kept as the fallback and the benchmark
reference. Acceptance: identical tokens, numerical correctness, no prefill or concurrent
regression, a measured single-stream gain over Step 3a; 35 tok/s is a target, not a
promise.

### 17.1 The routing implementation today (audit)

All in `Transformer::forward_segments` (`qubivex-arch`), per MoE layer:
1. `x = rms_norm(hs, ffn_norm)` on the host.
2. `be.matmul(router, x)`: **one submission**, s × ne logits read back.
3. `ops::route` per token on the host: `softmax` (max by `f32::max`, `exp(v − m)`, a
   sequential f32 sum from +0, a division per value), then a stable sort by
   `f32::total_cmp` descending with ties to the lower id, truncated to k, then (OLMoE:
   `norm_topk_prob = false`) an optional renormalisation by `Iterator::sum`.
4. Trace points `RouterLogits` and `RoutingWeights` (tests and diagnostics only).
5. `by_expert`: each routed expert's (token, rank) users; `used` in ascending expert id.
6. `ExpertProvider::with_experts_grouped(layer, used, …)`: `ExpertStore::acquire_many`
   (counts hits, heat, recency and the trace, and may start loads), `after_demand`
   (lookahead prediction bookkeeping), then if every lease is ready, device pins
   (`OnDevice::ensure`, copying any expert without a device copy) and **one fused
   submission** (Step 3a); otherwise the per-expert waves.
7. Host combine: per token, `out = 0; out = fma(w_r, y_r, out)` in rank order; then
   `hs += out`.

### 17.2 Every CPU/GPU synchronisation point (decode, s = 1, Step 3a)

| # | Per layer | Submission | Host work after the wait |
|---|---|---|---|
| 1 | QKV | `matmul_many` (3 matmuls) | clip, QK norm, RoPE, KV store, attention |
| 2 | O | `matmul` | residual, FFN norm |
| (2′) | lookahead router | `matmul`, only with a predictor (off in the bench) | prediction |
| 3 | router | `matmul` | softmax, top-k, `acquire_many`, pins |
| 4 | expert block | `gated_ffn_many` | combine, residual |
| — | head (per token) | `matmul` | sampling |

4 per layer + 1 = **65 per token**, the recorded count. Every one is a host wait on a
fence: the GPU idles while the host works. Step 0 measured about 140 µs of GPU idle per
serial hand-off.

### 17.3 What can move to the GPU, and what Step 3b moves

| Edge | Real dependency | Step 3b |
|---|---|---|
| router → softmax/top-k → expert block | data only: the GPU can choose | **removed**: router, routing, experts and combine in one submission |
| top-k → which expert weights | on the host today because the store leases experts | the host **holds every expert of the layer** before submitting (Metal's `hold_layer`, ADR-011 amendment 4); the GPU picks among them |
| expert block → combine | data only | **combine on the device** (rank-order `fma`, already probed single-rounded) |
| O → residual → FFN norm → router | data only, but the norm needs `sqrt` and division on the device (audit §5) | not in Step 3b (17.9) |
| QKV → attention → O | host KV cache | not in Step 3b (Step 3c) |

After Step 3b: **3 per layer + 1 = 49 submissions per token.**

### 17.4 Selected expert index → arena slot

Step 3a put every expert of a layer in one arena (its size class), at
`slot × stride`. Step 3b gives the GPU a per-layer **expert table**: for each of the
layer's ne experts, three table templates (gate, up, down) in the fused shader's 8-word
format, with `w_base = slot × stride + (matrix offset inside the expert)`. The host
writes them at submit time from `OnDevice::weights`, which already resolves a host
weight view to (buffer, offset). The route kernel copies the chosen experts' templates
into the dispatch table, filling in the input and output offsets. So:
- no descriptor indexing, no buffer device addresses: one binding for the whole arena,
  as in Step 3a (MoltenVK's per-encoder residency cost, §15.2, is avoided);
- the routed path is used only when **every expert of the layer resolves into one
  buffer with one dtype per matrix kind**; otherwise that layer runs Step 3a.

Residency rule (the condition for the GPU to choose):
- **RAM:** `ExpertStore::hold_layer(layer, ne)` returns `Held::All` (every expert ready;
  leases taken quietly: no heat, recency, hits or trace). `Held::Missing`: those experts
  are queued as prefetches with I/O threads (loaded now without), and this layer runs
  Step 3a. `Held::TooSmall`: Step 3a.
- **Device:** only when the device cache is **fully provisioned**: each arena has a slot
  for every expert of every layer in its class, so holding a layer can never evict
  anything. Missing device copies are made at hold time (counted misses); held
  copies are pinned for the submission and not counted as hits or refreshed.
- **After the submission** the routed ids come back with the outputs (no extra wait).
  The host then does what `acquire_many` would have done: `record_uses` for the layer's
  distinct experts in ascending id (same heat, recency, hits and trace as Step 3a),
  `after_demand` with the same list, and a device-cache hit and recency refresh for
  each.

### 17.5 Descriptor and dispatch dependencies (one command buffer)

All dispatches use Step 3a's fused pipeline layout (4 storage bindings, 16-byte push),
so no new layout; two new pipelines (`route`, `combine`).

| Step | Pipeline | Dispatch | Reads | Writes |
|---|---|---|---|---|
| 1 | fused matmul (router dtype) | `blocks(ne) × 1, s` | dense buffer, x, table[router] | logits |
| barrier | | | | |
| 2 | **route** (new) | `ceil(s/64)` | logits, table[templates] | logits → p in place, routes (ids, weights), table[routed entries] |
| barrier | | | | |
| 3 | fused matmul (gate/up dtype) | `2·s·k · blocks(ffn), 1` | arena, x, table | mid (gate, up) |
| barrier | | | | |
| 4 | SwiGLU (Step 3a's) | `s·k·ffn / 256` | mid | mid |
| barrier | | | | |
| 5 | fused matmul (down dtype) | `s·k · blocks(h), 1` | arena, mid, table | contrib |
| barrier | | | | |
| 6 | **combine** (new) | `s·h / 256` | contrib, routes | y |
| barrier → host | | | | |

- Dispatch sizes depend only on s, k, ne and the shapes, all known on the host: **no
  indirect dispatch**. The GPU decides only *which* expert each entry reads.
- Entries are **entry-major**: entry j = (token t, rank r) is one 1-row matrix of
  expert ids[j]. Step 3a's fused dispatch already reads the weights once per token row
  (grid y is the token), so this is the same GPU work as grouping by expert, and its
  output layout is the combine's `contrib` layout directly (no gather or scatter).
- 6 dispatches and 5 barriers, against Step 3a's router submission (1 dispatch) plus
  its block (3 dispatches, 2 barriers).

### 17.6 Bits

- Router matmul: the fused entry runs `matmul.comp`'s body unchanged (Step 3a proof).
- Route: `ops::route`'s operations in its order, one invocation per token, as Metal's
  `moe_route`: `f32::max` fold from −∞ (NaN-ignoring, as Rust's `max`), `math::exp`
  ported unchanged (shared with `swiglu.comp` through an include; `swiglu.spv` must stay
  byte-identical), a sequential sum from +0, a correctly rounded division; top-k by
  selection over `total_cmp` keys (sign-magnitude to two's complement) descending, ties
  to the lower id, which is the order the stable sort defines; renormalisation as a
  sequential sum from −0 (`Iterator::sum` for f32). Built `-O0` (it contains `exp`).
- Gate/up, SwiGLU, down: Step 3a's kernels.
- Combine: `out = 0; out = fma(w_r, y_r, out)` per rank, as the host loop.
- **Tests (new, each against the CPU, bit for bit):** route alone over adversarial
  logits (ties, pairs of ties, −∞, +∞, NaN, all equal, subnormal differences) and
  millions of random tokens, with and without renormalisation, at ne ∈ {8, 60, 64, 128}
  and k ∈ {1, 2, 8}; the whole routed layer (router + experts + combine) against the
  host steps for every dtype pairing and s ∈ {1, 2, 4, 8}, in arenas; a positive control
  (ties to the higher id) that must fail. Existing gates: fixtures, `tests/vulkan.rs`
  with real OLMoE (CPU logits, Colibri's tokens), bench tokens equal to the Step 1a
  reference.
- Division: correctly rounded on MoltenVK in Safe mode (§15.1, 2³³ divisions); the
  route test adds millions of softmax divisions. A device-open division probe for other
  drivers is not part of 3b (ADR-019 "Not decided").

### 17.7 Interfaces (generic, no model-specific path)

- `qubivex-compute`: `RoutedMoe` (router, the layer's gate/up/down for every expert, k,
  renormalise, xs, s; outputs ys, ids, weights), `ComputeBackend::{routes_moe(experts,
  k, s), routed_moe}` with defaults "no" / "not run". `softmax` and `route` move from
  `qubivex-arch::ops` into `qubivex-compute::math` unchanged (re-exported), as SwiGLU did,
  so a backend can be checked against them. `DiscreteDevice` gains the same pair.
- `qubivex-arch`: `ExpertProvider::with_layer_held(layer, experts, f)`, default "not
  held". The MoE stage asks it first when the backend routes this shape and no trace is
  being taken; if the backend or the provider declines, the layer runs Step 3a
  unchanged.
- `qubivex-runtime`: `StoreProvider::with_layer_held` (hold, device pins, record uses);
  `OnDevice::{routes_moe, routed_moe}` (fully provisioned check, templates).
- `qubivex-backend-vulkan`: `route.comp`, `combine.comp`, `exp.glsl` include; one
  submission; **`--vulkan-route on|off` (default off)** and `VulkanConfig::route`.
  Shape rule: s ≤ the fuse-rows threshold (8), ne ≤ 256, k ≤ ne.
- Sim device: an opt-in routed path for the runtime tests (`SimConfig::route`), running
  the host steps, so the provider and accounting are tested without a GPU.

### 17.8 Memory and execution overhead (estimates)

- **Device memory:** the fully provisioned condition means the whole model's experts on
  the device: OLMoE 1024 experts, 1.81 + 2.09 GB of arenas (the bench's
  `--expert-cache-mb 3720` already holds all of them). Nothing new in the ledger.
  Per-submitter working buffers grow by: logits s·ne·4 B, routes s·k·8 B, table
  (1 + 3ne + 3sk)·32 B ≈ 7 KiB, contrib s·k·h·4 B (64 KiB at s = 1, 512 KiB at s = 8):
  under 1 MiB per submitter, in the 256 MiB device reserve.
- **Host:** per layer, `hold_layer` (64 quiet leases) and 64 device pins under the cache
  lock, plus about 6 KiB of templates: an estimated 10–20 µs, against one host round
  trip removed.
- **GPU:** route ~ne·(1 + k) scalar steps per token on one invocation, and an
  elementwise combine: a few µs; two more barriers (MoltenVK splits encoders at each).
- **Gain:** 16 fewer serial hand-offs per token. At Step 0's ~140 µs of GPU idle each,
  plus the router submission's own host cost, **about 2–3.5 ms per token: 33.0 → about
  29.5–31 ms (32–34 tok/s)**. An estimate, not a measurement. 35 tok/s (28.6 ms) probably
  needs 17.9 as well.
- **First passes:** holding a layer needs all 64 of its experts in RAM, so missing ones
  are prefetched; until they land, that layer runs Step 3a. The bench's warm-up absorbs
  this.

### 17.9 Next, not in 3b: the O projection and FFN norm in the same submission

O → residual → FFN `rms_norm` → router would make a layer two submissions (QKV;
everything else), another 16 hand-offs. It needs `sqrt` and division on the device with
the CPU's bits (`1/sqrt(ss/n + eps)`), proven exhaustively over f32 like SwiGLU, and the
64-lane reduction for the sum of squares (the matmul body's tree). It is a separate,
separately switched step, after 3b is measured.

### 17.10 Benchmark plan

- Mains power, untimed, alternated, 3 rounds: `scripts/bench-vulkan-baseline --tag
  step3b --variant step3a= --variant routed='--vulkan-route on'`. That gives Step 1a
  (the permanent baseline), Step 3a (same binary, switch off) and Step 3b, per round, all
  tokens checked against the Step 1a reference.
- Concurrent: the batch example at 1, 2, 4 and 8 sequences, the three variants.
- Prefill: the bench's 10-token prompt (> 8 rows per expert, so not routed) and a
  128-token prompt record: must equal Step 3a.
- Report: decode ms/token, tok/s, submissions per token (expect 65 → 49), routed layers
  per token, prefill, concurrent tok/s; accept only beyond the run-to-run spread
  (~0.7 ms).

### 17.11 What was built (separate commits, each compiling alone)

| Commit | What |
|---|---|
| Compute | `RoutedMoe`, `ComputeBackend::{routes_moe, routed_moe}` and `DiscreteDevice`'s pair, declined by default; reference steps (`routed_moe_steps`, `route_tokens`, `combine_ranks`). `softmax` and `route` move into `qubivex-compute::math` unchanged. |
| Arch | `ExpertProvider::with_layer_held` (default: not held). The MoE stage offers the layer to the backend when it routes the shape and no trace is taken; otherwise Step 3a, unchanged. |
| Sim | `SimConfig::route_rows`: an opt-in routed layer in one submission, entry-major, for the runtime tests. |
| Runtime | `StoreProvider::with_layer_held`: quiet `hold_layer` leases and device pins (`OnDevice::hold`, no hit, no recency); missing experts prefetched (loaded now without I/O threads); after the pass `record_uses`, `after_demand` and the device-cache hit for the used experts only, so the counts equal Step 3a's (tested). Routing on the device only with a **fully provisioned** device cache (an arena slot for every expert of every layer). `DeviceReport::routed_layers`. |
| Shaders | `route.comp`, `combine.comp` (both `-O0`); `exp` moved into `exp.glsl`. **`swiglu.spv`, `matmul.spv` and `matmul_fused.spv` byte-identical.** |
| Vulkan | `routed_moe`: 6 dispatches, 5 barriers, one command buffer (§17.5), on Step 3a's fused layout; declined (Step 3a runs) when a matrix kind spans buffers, dtypes or shapes, or past the limits. `VulkanConfig::route`, `--vulkan-route on\|off` (**default off**), up to `--vulkan-fuse-rows` tokens, at most 256 experts. `VulkanDevice::route` (the shader alone, for conformance tests). |
| Route kernel | One workgroup per token, logits in shared memory, `exp` and ranks in parallel; the max and both sums stay sequential in one invocation. The first version (one invocation per token) added 1.4 ms of GPU time per token; this one adds 0.3. |
| CLI, bench | `--vulkan-route`; the batch example and `scripts/bench-vulkan-baseline` pass it on. |

### 17.12 Correctness

- **Route alone** (`tests/routed.rs`, against `math::route`): ids and weights bit for bit
  at 8, 60, 64 and 128 experts × k = 1, 2, 8 × renormalise on/off, over adversarial
  logits (ties, pairs of ties, all equal, ±0, ±∞, a spread past `exp`'s underflow,
  ×10³⁰ and ×10⁻³⁰) and 8192 random tokens each, plus 4 × 2¹⁸ tokens at OLMoE's shape.
  Tokens with a NaN logit: every weight NaN on both; ids not compared (`total_cmp`
  orders NaNs by payload, which is outside the contract, ADR-011 amendment 7).
- **Whole layer** (against `routed_moe_steps` on the CPU): outputs, ids and weights bit
  for bit for Q4_K/Q4_K, Q4_K/Q6_K (Q8_0 router), Q6_K/Q6_K (f16 router), Q8_0, and
  f32/f16 (bf16 router), 1, 2, 4 and 8 tokens, with and without a router tie; OLMoE's
  shape (64 × 2048 × 1024, top 8, Q4_K and Q6_K down) at 1 and 4 tokens; one submission
  each. Declined for separate buffers, past the token limit, and when off.
- **Positive controls** (each restored after): ties to the higher id fails 3 tests;
  `x · (1/sum)` instead of `x / sum` fails 4; the combine in reverse rank order fails 2;
  in the runtime, a wrong id-to-expert mapping fails the sim tests (the concurrent test
  was strengthened to compare logits after it first passed the mutation). Starting the
  softmax sum at −0 instead of +0 cannot differ (the maximum contributes `exp(0) = 1`).
- **Runtime:** sim (fixtures): the CPU's bits, 3 submissions per layer + 1, the same
  store and device-cache hits as Step 3a, nothing routed with a device cache short of the
  model, concurrent requests. Vulkan: the fixtures' logits and tokens with routing on,
  alone and 4 concurrent; **real OLMoE: the CPU's prompt logits and Colibri's 12 tokens,
  every layer routed** (1024 experts, 4.21 GB on the device). Step 3a's tests unchanged
  and passing.

### 17.13 Measured: battery diagnostic only (not an acceptance record)

The machine was on battery for the whole session, so `scripts/bench-vulkan-baseline`
refused to run and **nothing here is a mains record**. Same binary, Step 3a
(`--vulkan-route off`) alternated with Step 3b (`on`), OLMoE Q4_K_M, 5 threads, 10-token
prompt, 128 generated, 1 + 3 runs, every run's tokens equal to the Step 1a reference:

| Round | Step 3a ms/token | Step 3b ms/token | Step 3a tok/s | Step 3b tok/s |
|---|---|---|---|---|
| 1 | 33.08 | **31.14** | 30.23 | **32.11** |
| 2 | 32.85 | **31.25** | 30.44 | **32.00** |
| 3 | 32.95 | **31.23** | 30.35 | **32.02** |

- −1.6 to −1.9 ms per token (−5.5%), more than twice the run-to-run spread; submissions
  per decode token 65 → 49; prefill 197.4–198.7 against 198.0–199.8 ms (the 10-token
  prompt is not routed: unchanged).
- Timed (`--vulkan-timestamps`): GPU idle 14.39 → 11.86 ms per token (−2.5 ms, the
  estimate in §17.8), GPU busy 20.52 → 20.82 (+0.3 ms: the route and combine dispatches
  and two more encoder boundaries).
- Concurrent (batch example, 32 tokens per sequence, median of 3, 2 rounds, tokens
  identical in every mode):

  | N | Step 3a concurrent tok/s | Step 3b concurrent tok/s | Gain |
  |---|---|---|---|
  | 1 | 30.45–30.56 | 32.22–32.34 | +5.6% |
  | 2 | 37.57 | 39.23–39.29 | +4.5% |
  | 4 | 42.78–42.96 | 44.59–44.64 | +4.0% |
  | 8 | 45.99–46.04 | 47.90–47.91 | +4.1% |

- **Against the 35 tok/s target:** 32.0 tok/s on battery. Not reached. The remaining
  per-layer round trips are QKV and O; §17.9 (the O projection, residual and FFN norm
  joining the routed submission) removes another 16 per token, and needs `sqrt` and
  division proven on the device first.

### 17.14 Measured on mains (the acceptance records)

`scripts/bench-vulkan-baseline --tag step3b --variant step3a= --variant routed='--vulkan-route on'`,
AC power, untimed, 3 alternated rounds (records `2026-11-11-m3pro-vulkan-step3b-{base,step3a,routed}-{1,2,3}.json`,
log `…-step3b-mains.txt`). Every record's tokens equal the Step 1a reference.

| Round | Step 1a ms/token | Step 3a | **Step 3b** | Step 1a prefill ms | Step 3a | Step 3b |
|---|---|---|---|---|---|---|
| 1 | 35.60 | 33.06 | **30.82** | 199.8 | 195.1 | 195.0 |
| 2 | 35.40 | 33.15 | **31.28** | 199.7 | 195.5 | 195.4 |
| 3 | 36.21 | 33.14 | **31.16** | 200.5 | 195.6 | 195.3 |
| tok/s | 27.6–28.3 | 30.2–30.3 | **32.0–32.5** | | | |
| submissions/token | 305 | 65 | **49** | | | |

- Against Step 3a: **−1.8 to −2.3 ms per token (−6.4%)**, about three times the run-to-run
  spread. Against the permanent Step 1a baseline: −4.6 ms (−13%).
- Prefill: unchanged against Step 3a (195.0–195.4 against 195.1–195.6 ms).

Concurrent tok/s (batch example, 32 tokens per sequence, median of 3, per round; tokens
identical in every mode):

| N | Step 1a | Step 3a | **Step 3b** | Gain over 3a |
|---|---|---|---|---|
| 1 | 27.93–28.05 | 30.33–30.47 | **32.26–32.49** | +6.3% |
| 2 | 34.17–34.41 | 37.65–37.83 | **39.47–39.61** | +4.7% |
| 4 | 40.15–40.19 | 43.04–43.21 | **44.92–44.98** | +4.2% |
| 8 | 44.48–44.52 | 46.34–46.44 | **48.32–48.34** | +4.2% |

**Gates:** identical tokens (yes, every record and mode); numerical correctness (§17.12);
no prefill regression (yes); no concurrent regression (a gain at every N); a measured
single-stream gain over Step 3a (yes, −6.4%). **35 tok/s: not reached (32.0–32.5).**

### 17.15 Not done; for the owner

1. **Acceptance**, and whether `--vulkan-route` becomes the default (it is off).
2. **Default.** `--vulkan-route` stays off until accepted.
3. **ADR-019 amendment 4** (routing on the device: residency rule, the route kernel's
   order argument, the 256-expert limit) is not written; it follows acceptance, as
   amendment 3 did for Step 3a.
4. **A device-open division probe** for drivers other than MoltenVK (§17.6) is not
   built.
5. **Next:** §17.9.

## 18. Step 3c: the O projection and FFN norm in the routed submission (design, 2026-11-11)

Owner brief (2026-11-11): start §17.9. Acceptance as Step 3b: identical tokens, numerical
correctness, no prefill or concurrent regression, a measured single-stream gain over
Step 3b, alternated against the permanent Step 1a baseline. Behind its own switch, Step 3b
the fallback.

### 18.1 What moves

After Step 3b a MoE layer is three submissions: QKV; O; router through combine. Between
the second and third the host does the attention residual (`hs += o`) and the FFN
`rms_norm`. Both are data-only edges (§17.3). Step 3c runs them on the device, at the
head of the routed submission:

| # | Per layer | Submission | Host work after the wait |
|---|---|---|---|
| 1 | QKV | `matmul_many` (3 matmuls) | clip, QK norm, RoPE, KV store, attention |
| 2 | **O, residual, FFN norm, router, routing, experts, combine** | `routed_moe` with a prelude | `hs += y` |
| — | head (per token) | `matmul` | sampling |

**2 per layer + 1 = 33 submissions per token** (Step 3b: 49). The QKV → attention → O
edge stays on the host: the KV cache is in host memory (Step 4 in §8).

### 18.2 Bits

The device does the CPU's operations in its order:
- **O:** the fused matmul entry, `matmul.comp`'s body unchanged (the Step 3a proof).
- **Residual:** one f32 add per element, `hs[i] + o[i]`, as `*a += b` on the host.
- **Sum of squares:** the shared 64-lane shape (`reduce::dot(x, x)`): lane l takes
  elements l, l + 64, … in order, one `fma(v, v, acc)` each (a short last chunk lands in
  lanes 0..rem, as the loop does), then the fixed halving tree, w = 32, 16, 8, 4, then
  `(a0 + a1) + (a2 + a3)`. One workgroup of 64 invocations per token, the tree in shared
  memory, as the matmul's.
- **Scale:** `1 / sqrt(ss / n + eps)` with `n` and `eps` as f32
  (`math::rms_scale_of`, split out of `rms_scale` with the same operations). Needs a
  correctly rounded division and **square root**. Vulkan promises neither (2.5 and about
  3 ULP); MoltenVK with fast math off gives correctly rounded division (§15.1). The square
  root is checked here: `shaders/rms.glsl`, built -O0 (glslc -O may turn `1 / sqrt` into
  `inversesqrt`), run alone by a probe shader over **every f32 input** for six (n, eps)
  shapes, including (1, 0), which is `1 / sqrt(x)` itself.
- **Output:** `(v · r) · w`, two multiplies, as `rms_norm`.

`rms_norm`, `rms_scale` and the new `rms_scale_of` move from `qubivex-arch::ops` into
`qubivex-compute::math` unchanged (re-exported), as SwiGLU and routing did.

### 18.3 One command buffer

Step 3b's six dispatches, preceded by two:

| Step | Pipeline | Dispatch | Reads | Writes |
|---|---|---|---|---|
| 0a | fused matmul (O dtype) | `blocks(h) × 1, s` | dense buffer, attention outputs, table[O] | o |
| barrier | | | | |
| 0b | **resnorm** (new) | `s` workgroups of 64 | hs, o, table[norm weight] | hs (in place), x |
| barrier | | | | |
| 1–6 | Step 3b (§17.5), router reading x | | | |

8 dispatches, 7 barriers; all on the fused layout (4 storage bindings, 16-byte push). The
norm weight (h f32 values) goes into the table buffer, so no new binding. `hs` is a
host-visible buffer written in and read back; `x` is the activation buffer Step 3b
already binds for the router and the gate/up entries. Extra per-submitter memory:
`s × (q_out + 2h) × 4` bytes and h words of table: 48 KiB at s = 1, 384 KiB at s = 8.

### 18.4 Interfaces

- `qubivex-compute`: `MoePrelude { o, att, hs, norm, eps }` and `RoutedMoe::prelude:
  Option<MoePrelude>` (then `xs` is empty: the activations are computed);
  `ComputeBackend::routes_moe_prelude(experts, k, s)`, default `false`; the reference
  `routed_moe_steps` runs the prelude as separate steps (O matmul, the add, `rms_norm`)
  first. `DeviceRoutedMoe` and `DiscreteDevice` gain the same.
- `qubivex-arch`: before the O matmul, a MoE layer is offered whole (prelude included)
  when the backend routes it with a prelude, no trace is taken and no lookahead is wanted
  (the lookahead router reads `hs` after O on the host). If the provider cannot hold the
  layer or the backend declines, nothing has been written and the layer runs as before
  (Step 3b, then 3a).
- `qubivex-runtime`: `OnDevice` resolves the O view like the router's.
- Sim: `SimConfig::route_prelude`, the separate steps in one submission, for the runtime
  tests.
- `qubivex-backend-vulkan`: `resnorm.comp`, `rms_scale.comp` (the probe), `rms.glsl`;
  `VulkanConfig::route_o`, **`--vulkan-route-o on|off` (default off)**, effective only
  with `--vulkan-route on`. Declined (Step 3b runs) when the O matrix's dtype is not a
  kernel's, its rows differ from the residual width, or past the limits.

### 18.5 Estimates

- 16 fewer serial hand-offs per token. At Step 3b's measured gain per removed hand-off
  (2.0 ms for 16, §17.14), **about 2 ms per token: 31.0 → about 29 ms (about 34.5 tok/s)**.
  An estimate, not a measurement.
- GPU work added: the resnorm dispatch (one workgroup per token, 2048 elements, a scalar
  square root and division) and one more encoder boundary; the O matmul moves, it is not
  added.
- Prefill: unchanged (10 tokens > 8 rows: not routed).

### 18.6 Tests and benchmark

- Scale probe: edge values (±0, ±∞, NaN, perfect squares and neighbours, 2^20 random) in
  the normal suite; every f32 input for six shapes, ignored (`tests/norm.rs`).
- Whole layer with a prelude against the host steps, bit for bit, every dtype pairing,
  s ∈ {1, 2, 4, 8}, an n with a short last chunk, OLMoE's shape. Positive controls: the
  residual added after the norm, `inversesqrt`, a sequential (not tree) sum.
- Runtime: sim and Vulkan fixtures, concurrent, real OLMoE with every layer routed with
  its prelude: the CPU's logits and Colibri's tokens.
- Benchmark (mains, 3 alternated rounds): `scripts/bench-vulkan-baseline --tag step3c
  --variant routed='--vulkan-route on' --variant routedo='--vulkan-route on
  --vulkan-route-o on'`; decode, prefill, submissions per token (49 → 33), concurrent at
  1, 2, 4 and 8.

### 18.7 What was built (separate commits, each compiling alone)

| Commit | What |
|---|---|
| Compute | `rms_norm` and `rms_scale` move into `qubivex-compute::math` unchanged (re-exported by `qubivex-arch::ops`); `rms_scale_of(ss, n, eps)` is the scalar step, which `rms_scale` now calls. |
| Shaders | `rms.glsl` (`1 / sqrt(ss / n + eps)`), `resnorm.comp` (residual, 64-lane sum of squares, scale, two multiplies; one workgroup of 64 per token) and `rms_scale.comp` (the scale alone, for the conformance tests), all `-O0`. The five existing `.spv` files are byte-identical. `VulkanDevice::rms_scale`. |
| Interface | `MoePrelude` and `RoutedMoe::prelude` (then `xs` is empty); `residual_norm` and `prelude_steps` (the reference); `ComputeBackend::routes_moe_prelude` and `DiscreteDevice::routes_moe_prelude`, both `false` by default; `DeviceMoePrelude`. Arch: before the O matmul, a MoE layer is offered with its prelude when the backend says so, nothing is traced and no lookahead is wanted; declined, it runs as before. Runtime: `OnDevice` resolves the O view. Sim: `SimConfig::route_prelude`. |
| Vulkan | `routed_moe` with a prelude: the O projection as a fused entry over the dense buffer, then `resnorm` writing the activations the router and experts read, then Step 3b's six dispatches: 8 dispatches, 7 barriers, one command buffer. The norm weights travel in the table buffer. Declined (Step 3b runs) when the O dtype has no kernel or past the limits. `VulkanConfig::route_o`, off by default, effective only with `route`. |
| CLI, bench | `--vulkan-route-o on\|off` (needs `--vulkan-route on`); the batch example and `scripts/bench-vulkan-baseline` pass it on. |

Also fixed: Step 3b had left `Ffn`'s doc comment and derive above `RoutedLayer`.

### 18.8 Correctness

- **Scale, every input** (`tests/norm.rs`, ignored, 41 s): `1 / sqrt(ss / n + eps)` on
  the device against `rms_scale_of` on the CPU for every f32 bit pattern of `ss`, at
  (n, eps) = (1, 0), (2048, 1e-5), (4096, 1e-6), (128, 1e-6), (896, 1e-6) and (7, 1e-5):
  **0 differences** outside the excluded classes. At (1, 0), 16,777,214 subnormal
  inputs differ (the GPU flushes them: ADR-011 §1); every shape has 16,777,213 NaN
  inputs with different payloads (ADR-011 amendment 7). So MoltenVK's `sqrt` and
  division are correctly rounded over the whole range here. In the normal suite: edge
  values (±0, ±∞, NaN, perfect squares and their neighbours up to 4095², 2²⁰ random) at
  the same six shapes.
- **Whole layer with its prelude** against `routed_moe_steps` on the CPU (`tests/routed.rs`):
  the residual stream, ids, weights and outputs bit for bit, one submission each, for
  five layer shapes with O in Q4_K, Q6_K, Q8_0, f16 and f32, attention widths 70 to 768,
  a width of 65 (a one-element last chunk in the norm's sum), 1, 2, 4 and 8 tokens, eps
  1e-5 and 1e-6, with and without renormalisation; OLMoE's shape (O 2048 × 2048 in Q4_K
  and Q6_K, 64 experts, top 8) at 1 and 4 tokens. Declined with nothing written when
  `route_o` is off (Step 3b still runs), when the experts are in separate buffers, and
  without `route`.
- **Positive controls** (each restored after): `inversesqrt(q)` for `1 / sqrt(q)` fails
  the scale test (1-ulp differences); in `resnorm.comp`, a different final tree
  (`((a0 + a1) + a2) + a3`), `v · (r · w)` for `(v · r) · w`, and the norm taken
  before the residual each fail both prelude layer tests; in the reference, the norm
  before the residual fails both sim prelude tests.
- **Runtime:** sim (fixtures): the CPU's prompt logits and tokens, **2 submissions per
  layer + 1** per pass, the same store and device-cache hits as Step 3b, nothing routed
  with a device cache short of the model, concurrent requests. Vulkan: the fixtures'
  logits and tokens with the prelude, 2 per layer + 1 submissions, concurrent; **real
  OLMoE: the CPU's prompt logits and Colibri's 12 tokens, every layer routed with its
  prelude**. Step 3a and 3b tests unchanged and passing.

### 18.9 Measured on mains (the acceptance records)

`scripts/bench-vulkan-baseline --tag step3c --variant routed='--vulkan-route on' --variant routedo='--vulkan-route on --vulkan-route-o on'`,
AC power, untimed, 3 alternated rounds (records `2026-11-12-m3pro-vulkan-step3c-{base,routed,routedo}-{1,2,3}.json`,
log `…-step3c-mains.txt`). Every record's tokens equal the Step 1a reference. Swap was
about 7 GB in use for the whole session (left from earlier work); the rounds alternate,
so it affects every variant alike.

| Round | Step 1a ms/token | Step 3b | **Step 3c** | Step 1a prefill ms | Step 3b | Step 3c |
|---|---|---|---|---|---|---|
| 1 | 35.47 | 30.93 | **29.05** | 200.0 | 195.9 | 195.8 |
| 2 | 35.22 | 31.04 | **29.18** | 200.0 | 195.6 | 195.8 |
| 3 | 35.62 | 31.13 | **28.95** | 200.1 | 196.0 | 195.5 |
| tok/s | 28.1–28.4 | 32.1–32.3 | **34.3–34.5** | | | |
| submissions/token | 305 | 49 | **33** | | | |

- Against Step 3b: **−1.9 to −2.2 ms per token (−6.2%)**, as estimated (§18.5), about
  three times the run-to-run spread. Against the permanent Step 1a baseline: −6.4 ms
  (−18%).
- Prefill: unchanged (195.5–195.8 against 195.6–196.0 ms; the 10-token prompt is not
  routed).

Concurrent tok/s (batch example, 32 tokens per sequence, median of 3, per round; tokens
identical in every mode):

| N | Step 1a | Step 3b | **Step 3c** | Gain over 3b |
|---|---|---|---|---|
| 1 | 28.36–28.61 | 32.36–32.69 | **34.49–34.88** | +6.6% |
| 2 | 34.27–34.60 | 39.44–39.69 | **41.32–41.50** | +4.7% |
| 4 | 40.27–40.35 | 44.88–45.00 | **46.00–46.06** | +2.4% |
| 8 | 44.48–44.73 | 48.25–48.34 | **49.07–49.18** | +1.8% |

**Gates:** identical tokens (yes, every record and mode); numerical correctness (§18.8);
no prefill regression (yes); no concurrent regression (a gain at every N, smaller as N
grows, since more sequences already hide the round trips); a measured single-stream gain
over Step 3b (yes, −6.2%). **35 tok/s: not reached (34.3–34.5).** What remains per layer
is QKV → attention on the host (the KV cache in host memory, §8 Step 4).

### 18.10 Not done; for the owner

1. **Acceptance**, and the defaults: `--vulkan-route-o` is off, and needs
   `--vulkan-route` (also off, pending Step 3b's acceptance, §17.15).
2. **ADR-019 amendment** for Steps 3b and 3c (routing and the prelude on the device; the
   square root proven exhaustively on MoltenVK) is not written; it follows acceptance.
3. **A device-open probe** for division and square root on drivers other than MoltenVK is
   not built (the exhaustive test is the proof on this one).
4. **Next:** attention on the device (the KV cache on the device), which would make a
   layer one submission.

**Decided 2026-11-12 (owner):** Steps 3b and 3c accepted; `--vulkan-route` and
`--vulkan-route-o` stay off by default; the fallbacks stay. Recorded as ADR-019
amendment 4. Next: the GPU attention investigation (§19).

## 19. GPU attention: investigation and initial harness (2026-11-12)

Owner brief (2026-11-12): investigate a GPU-resident KV cache and attention on the device
(Q projection, RoPE, KV updates, QKᵀ, causal masking, softmax, AV), tiled attention and
online softmax, KV against expert residency, concurrent sequences, exact parity with the
CPU. Target: no CPU attention round trip, eventually one submission per layer. Accepted
production shaders untouched; experiments reversible and committed separately; nothing
pushed. Base: `main` at c1fcd1b (Step 3c merged).

### 19.1 Where attention runs today (Step 3c, decode)

Per layer: **QKV** submission → host: clip, Q/K norm (OLMoE: full width, 2048), RoPE (host
table, f64-evaluated, rounded once), KV store into the host cache, `ops::attend` per (token,
head) on the rayon pool → **O + residual + norm + routed MoE** submission → host `hs += y`.
So 2 submissions per layer + the head = 33 per token. Removing the host attention merges
both into **one per layer (17 per token)**.

Prior art in this repository: Metal's whole-layer and token modes (ADR-011 amendments 2
and 4) already run this block on the GPU with the CPU's bits: `rms_norm`, `rope`,
`kv_store`, `attn_scores`, `attn_softmax`, `attn_values`; device KV position-major per
sequence in the `KvCache`'s device slot (`DeviceKv`), grown in 256-position chunks by a
prefix copy, charged to the ledger's KV owner; batched passes record one attention set per
sequence in the same submission. The Vulkan design below follows it and changes only what
the measurements below require.

### 19.2 The parity constraint, and what it rules out

`ops::attend` fixes the order of three reductions:
- each score: `dot_seq(q, k)`, a sequential fma chain over d from 0, then `· (1 / sqrt(d))`;
- the softmax: max by `f32::max` (order-free: exact, NaN-ignoring; a ±0 maximum gives the
  same `v − m` for every v), `exp(v − m)`, a **sequential sum over positions** from +0,
  one division per position;
- each output dimension: a **sequential fma chain over positions** from 0, `o = fma(p_j,
  v_j[i], o)`.

Everything else is parallel: positions for the scores, exp and division; heads; dimensions;
tokens. Causal masking is the loop bound (`j ≤ p0 + t`), not −∞ scores, as on the CPU.

**Tiled attention with an online softmax (FlashAttention's order) cannot give these
bits**: it rescales partial sums by `exp(m_old − m_new)` and divides once at the end.
Simulated on the CPU with the same `math::exp` and fma (harness
`online_softmax_is_not_bit_exact`), at OLMoE's shape and one token:

| positions | tile | outputs that differ | max difference |
|---|---|---|---|
| 64 | 64 (one tile) | 1567 of 2048 (77%) | 1332 ulp |
| 138 | 64 | 1686 (82%) | 611 ulp |
| 1001 | 64 | 1873 (91%) | 1757 ulp |
| 4096 | 64 | 1978 (97%) | 3249 ulp |
| 4096 | 256 | 1986 (97%) | 3249 ulp |

(Large ulp counts are on outputs near zero; this is ordinary rounding drift, not an error
in the algorithm.) Even a single tile differs, because the division moves. So split-K
attention, flash decoding and online softmax are **out under the current contract**;
using them would be a lossy opt-in under ADR-010, an owner decision. The investigation
keeps exact attention and makes it fast within the fixed order.

### 19.3 The harness (feature `spike`, not production)

- `qubivex-backend-vulkan` gains a `spike` cargo feature (off by default): `VulkanDevice::
  {spike_kernel, spike_run}` build experimental SPIR-V on the fused layout (4 storage
  bindings, 16-byte push) and run steps in one submission over host buffers, timed with
  timestamps (`repeat` runs per submission, to amortise MoltenVK's late TOP timestamp,
  §13). Nothing in the default build changes; production shaders and `build.sh` are
  untouched.
- `spike/`: `attn_exact.comp` (one workgroup of 128 per (token, head), scores in a device
  buffer), `attn_exact_sh.comp` (scores in shared memory, ≤ 4096 positions),
  `attn_scores_t.comp` + `attn_softmax.comp` + `attn_values.comp` (three dispatches, keys
  stored dimension-major `[kv_head][d][pos]` so the scores read coalesced), their
  unrolled softmax and values (`*_u.comp`: eight loads issued, then the eight adds or fmas
  in the same order), and `rope.comp`. All built -O0 with NoContraction (`spike/build.sh`,
  the production recipe).
- `tests/attention_spike.rs` compares with `qubivex_arch::ops::{attend, Rope::rotate}`
  (the CPU reference itself, a new dev-dependency).

### 19.4 Numerical parity (measured)

- **Exact attention, every variant: 379,376 of 379,376 outputs bit-identical** to
  `ops::attend`: OLMoE (16 × 16 heads, d 128), grouped-query (8 heads on 2, d 64), d 96 and
  d 200 (more than a workgroup), contexts 1, 2, 64, 65, 138, 1001, 4096 positions, 1, 4, 8
  and 10 new tokens (prefill shapes, the causal bound per token), score spreads past
  `exp`'s underflow (inputs ±30), a NaN key and an infinite key.
- **RoPE: 25,165,824 of 25,165,824 bit-identical** with the host table at every position
  0–4095, magnitudes 10⁻³ to 10³.
- **Positive controls** (each restored after): the softmax sum pairwise instead of
  sequential fails attention (1000+ differences); RoPE's `x1·c − x2·s` written as
  `fma(x1, c, −x2·s)` fails RoPE.
- Already proven and reusable: the norm's `1 / sqrt` (every f32 input, §18.8), `exp`
  (SwiGLU, §15.1), the 64-lane sum of squares (`resnorm.comp`). Q/K norm is `resnorm`
  without the residual.

### 19.5 Kernel time (measured; GPU µs per layer call, OLMoE shape, mains)

| positions | tokens | one workgroup | 3 dispatches | **3, unrolled** | CPU attend, 5 threads | CPU, 1 thread |
|---|---|---|---|---|---|---|
| 138 | 1 | 140 | 120 | **52** | 100 | 196 |
| 512 | 1 | 218 | 244 | **106** | 243 | 658 |
| 1024 | 1 | 484 | 482 | **189** | 413 | 1296 |
| 2048 | 1 | 977 | 919 | **350** | 870 | 2588 |
| 4096 | 1 | 1918 | 1827 | **725** | 1656 | 5269 |
| 138 | 8 | 88 | 77 | **57** | | |
| 1024 | 8 | 605 | 526 | **288** | | |
| 4096 | 8 | 2475 | 1997 | **1333** | | |

- By phase (3 dispatches, not unrolled, 4096 positions): scores 281 µs (32 MB of keys,
  about 117 GB/s: near bandwidth), softmax 406 µs (one invocation's sequential sum from
  device memory), values 1188 µs (2048 sequential chains, one dependent load per step,
  about 27 GB/s). **The fixed-order chains are latency-bound, not bandwidth-bound**;
  issuing loads ahead (unrolling) took the whole kernel from 1827 to 725 µs with the same
  bits. More of the same (deeper unrolling, scores and probabilities staged in shared
  memory for the chains) is the next lever.
- The CPU's 5-thread figures include `std::thread::scope` start-up (about 40–60 µs per
  call); rayon's pool in the real pass costs less, so the true CPU figure lies between the
  1-thread time ÷ 5 and the 5-thread time. Host Q/K norms, RoPE and the KV store copy:
  about 1 µs.
- The 8-token rows share one sequence's keys (prefill-like). 8 independent sequences read
  8 caches: the chains stay parallel, but the key and value bytes are 8×; at 4096
  positions that is 512 MB per layer, which is bandwidth-bound on any device (and on the
  CPU).

### 19.6 KV cache design (proposed)

- **Device KV as a mirror; the host cache stays authoritative.** The host cache
  (ADR-009) keeps being written: in the routed layer submission the new K and V rows come
  back with the outputs (16 KiB per token per layer), the host copies them in. Any pass the
  device declines (prefill above the row limit, a traced pass, lookahead, a sequence whose
  device KV could not be allocated) runs today's host attention unchanged, then uploads its
  new rows to the device mirror at the start of the next submission (a copy, no extra
  submission). So prefill and every fallback keep their current path and bits, and a
  sequence can drop to host attention at any step without losing state.
- **Layout:** paged blocks of 256 positions per layer, in one device pool: keys
  dimension-major within a block (`[kv_head][d][256]`, coalesced scores), values
  position-major (`[256][kv_head][d]`, coalesced chains). A sequence's block table (one
  word per block) travels in the table buffer, so several sequences' attention runs in one
  dispatch with no buffer per sequence, no prefix copy on growth, and no descriptor
  indexing (the arena pattern of Step 3a). A block is allocated when a sequence crosses
  its boundary; freed when the sequence ends.
- **Kernels in the layer submission** (decode, s ≤ 8): QKV fused entries → Q/K norm and
  RoPE and KV store fused into one kernel → scores → softmax → values → then Step 3c's O,
  residual, norm, routing, experts and combine: about 12 dispatches, **one submission per
  layer**.
- **Concurrent sequences:** each token carries its sequence's position and block table;
  every (token, head) chain is independent, so a batched pass of 8 sequences is one set of
  dispatches; the 8 submitters keep independent passes in flight as today.

### 19.7 Memory (OLMoE; f32 KV, as the CPU; ADR-010 rules out f16 KV)

Per token: 16 layers × (K + V) × 16 heads × 128 × 4 bytes = **256 KiB** (16 KiB per layer).

| context | per sequence | 8 sequences |
|---|---|---|
| 138 (the bench) | 35 MiB | 276 MiB |
| 512 | 128 MiB | 1 GiB |
| 1024 | 256 MiB | 2 GiB |
| 4096 (the model's maximum) | 1 GiB | 8 GiB |

Device memory already used with routing on (fully provisioned, §17.4): **4.21 GB** (expert
arenas 1.81 + 2.09 GB, dense weights the rest) plus the 256 MiB reserve. KV pool room on a
device of: 8 GB → about 3.5 GB (about 14,000 positions: 3 full contexts, or 27 sequences
of 512); 12 GB → about 7.5 GB (30,000); 24 GB → about 19.5 GB (78,000). On this Mac the
device is the same 18 GB as RAM, which already holds the host expert cache (3.7 GB) and
the host KV, so the practical pool is a few GB.

**Competition with experts:** routing on the device needs every expert resident (§17.4), so
KV must never evict experts; a full pool must instead (a) refuse admission at the
scheduler (KV blocks as a resource, as the ledger already does for host KV), or (b) leave
that sequence on host attention (the mirror makes this free). Proposed: the plan reserves
the pool after the experts (`--vulkan-kv-pool SIZE`, default: what the budget leaves after
the experts and the reserve), the pool is charged to the ledger's KV owner on the `Vram`
tier, and a sequence that cannot get a block runs host attention.

### 19.8 Performance estimate (decode; an estimate, not a measurement)

Per layer, moving the attention block to the device:
- saves one hand-off: **about 125 µs** (Step 3c removed 16 for 2.0 ms, §18.9);
- saves the host attention block: the CPU attend at that context (1-thread ÷ 5 to the
  5-thread figure) plus about 1 µs;
- adds the GPU attention (unrolled) and about 4 more encoder boundaries after fusing
  (about 10 µs each on MoltenVK, §17.13: 40–60 µs).

| context | gain per layer | per token (16 layers) | decode estimate |
|---|---|---|---|
| ~138 (the bench) | +53 to +133 µs | 0.9–2.1 ms | 29.0 → 26.9–28.1 ms (**35.6–37.2 tok/s**) |
| 1024 | +145 to +299 µs | 2.3–4.8 ms | |
| 4096 | +404 to +1006 µs | 6.5–16 ms | about 46–55 → about 40 ms |

The gain grows with context because CPU attention grows faster than the device's.
Concurrent: at 8 sequences the CPU's attention is 8× the work on 5 threads while the
device runs the chains in parallel; Step 3c gained only 1.8% at N = 8, so attention may be
a larger share there, but **nothing here measures it**. Prefill: unchanged by design (the
host path runs it; uploading a 128-token prompt's rows is 32 MiB, a few ms). As the brief
says, one submission per layer does not by itself guarantee a gain: at short context the
estimate rests on the 52 µs kernel staying below the hand-off it replaces, and the
encoder boundaries could eat half of it. The end-to-end bench decides.

### 19.9 Risks

1. **Fixed-order latency.** Exact parity forbids splitting the softmax sum and the value
   chains across workgroups, so a token's attention latency is O(context) per layer
   (725 µs at 4096 even unrolled). It beats the CPU here, but a faster CPU or a slower
   GPU memory latency can flip it; the long-context case must be measured on each target.
2. **Dispatch and encoder overhead on MoltenVK** (about 10 µs per boundary) can cancel
   the short-context gain; fusing norm, RoPE and the store into one kernel, and scores
   into the same workgroup as softmax where context allows, is required, not optional.
3. **Online softmax / flash attention** would be several times faster at long context,
   but changes the bits (77–97% of outputs). Using it is a contract change (ADR-010 /
   ADR-011), the owner's call.
4. **Memory:** the fully provisioned rule plus f32 KV leaves little room on 8 GB devices
   and on this Mac (swap was already 7 GB in use during §18.9); a pool policy and the
   ledger charge are needed before anything ships.
5. **Driver correctness:** division, `sqrt` and `exp` are proven on MoltenVK only; the
   attention adds divisions (softmax) and `1 / sqrt(d)`. A device-open probe is still not
   built (§18.10).
6. **Two copies of KV** (host authoritative, device mirror) cost host RAM equal to today's
   and keep fallbacks trivial; dropping the host copy later saves 256 KiB per token but
   makes every fallback need a download.
7. **Compiler sensitivity:** the unrolled chains rely on the Metal compiler keeping the
   order; -O0 and NoContraction plus the parity tests are the guard, as for every shader.
8. **Measurement artefacts:** MoltenVK's timestamps under-count short submissions (§13);
   the harness amortises them, and acceptance must be wall-clock end to end.

### 19.10 Proposed next steps (for review; nothing started)

1. Shared-memory staging and deeper unrolling in the harness (target: under 40 µs at 138
   positions, under 600 at 4096), and a fused norm-RoPE-store kernel, with parity.
2. A harness measurement of a whole layer's GPU time with attention in the routed
   submission (all dispatches, one command buffer) against Step 3c's two submissions, at
   1 and 8 sequences, before any production code.
3. Then, behind its own switch and with the pool, ledger charge and fallbacks of §19.6–7:
   the production path, measured alternated against Step 1a and Step 3c, single-stream,
   concurrent and prefill.

## 20. GPU attention: faster exact kernels and a whole-layer measurement (2026-11-13)

Steps 1 and 2 of §19.10, harness only (feature `spike`, off by default). Production
shaders, `shaders/build.sh` and every default-build path are unchanged; the new SPIR-V is
under `spike/` and rebuilds byte for byte from `spike/build.sh`. Base: `main` at 7e536ad.
Measured on mains, GPU time from the queue's timestamps.

### 20.1 What was built

- `spike/attn_sv.comp`: softmax and values fused into one dispatch per (token, head).
  Scores are read once into shared memory (at most 4096 positions), then the tree maximum,
  `exp`, the sequential sum by one invocation and the divisions all stay in shared
  memory, followed by each output dimension's fma chain over positions. That chain issues
  `U` value loads before its `U` fmas, in the same order. Build options: `U` is 8, 16 or
  32; `VEC4` runs four dimensions' chains per invocation from one vec4 load; `ONE`
  computes the scores in the same workgroup, giving attention in **one dispatch**;
  `BATCH` makes each token its own sequence with its own cache (batched decode).
- `spike/attn_scores_u.comp`: the scores with q in shared memory and `U` key loads
  issued ahead of their fmas.
- `spike/qk_prelude.comp`: everything the CPU does between the QKV projections and
  attention, in one dispatch:
  - the clip (Rust's `clamp`);
  - the full-width Q and K norms (resnorm's 64-lane sum of squares and `rms.glsl`);
  - RoPE on every head, from the host's table;
  - the store of k into the dimension-major K^T and of v into the position-major V.

  q, k and v are read region-major, as the QKV submission writes them, so q is
  attention's input as it stands. Through a header stride in the push constants, one
  dispatch serves one sequence's tokens (prefill) or one token for each of several
  sequences (batched decode).
- A `spike`-only hook in `device.rs`: `VulkanDevice::spike_prefix` / `spike_arm` record
  experimental steps at the head of the next submission's command buffer, whatever
  records it, with a barrier after each step. `spike_last` returns the last submission's
  wait and GPU time. Without the feature nothing is compiled in.

### 20.2 Parity (measured)

- **Attention:** every variant matches `ops::attend` bit for bit: 16 kernels (6
  softmax-values after the plain scores, 6 pairings of the unrolled scores, 4 one-dispatch).
  Each gave **379,376 of 379,376 outputs** on §19.4's cases: OLMoE, grouped-query, d 96
  and 200, contexts 1–4096, 1–10 new tokens, spreads past `exp`'s underflow, a NaN key and
  an infinite key.
- **Prelude:** 1,225,728 written outputs (q and the stored k and v) bit-identical to the
  CPU's `clamp`, `rms_norm`, `Rope::rotate` and store. Every other word of the KV buffer
  was left unchanged; 347,296,768 words were compared in all. The cases covered:
  - OLMoE and two grouped-query shapes;
  - 1 and 8 tokens, positions up to 4095;
  - with and without the clip, magnitudes 10⁻³ to 10³;
  - a NaN, an infinity and a -0 in the inputs.
- **In the routed submission:** the prelude and one-dispatch attention ran as a prefix of
  Step 3c's routed command buffer for 1 and 8 sequences (138 to 1024 positions). They
  gave **36,864 of 36,864** outputs equal to the CPU's prelude followed by `attend`.
- **Positive controls** (each restored after, and the SPIR-V rebuilt identically):
  - the softmax sum pairwise instead of sequential: fails (125,672 differences in the
    first kernel checked);
  - each values block's fmas in reverse order: fails;
  - RoPE's `x1·c − x2·s` as `fma(x1, c, −x2·s)` in the prelude: fails.

### 20.3 Kernel time (measured; GPU µs per layer call, OLMoE shape)

Each cell is the median of 5 samples after a one-second warm-up; each sample is 50 runs in
one submission. Two runs are shown as ranges; a single figure means they agreed. "Tokens"
here share one sequence's cache (prefill-like), as in §19.5.

| positions | tokens | §19 best (3 dispatches, unrolled) | scores u16 + softmax-values u16 | **one dispatch, u16** | one dispatch, vec4 u16 |
|---:|---:|---:|---:|---:|---:|
| 138 | 1 | 45–50 | 32–35 | **17** | 18 |
| 138 | 8 | 60–61 | 60 | **50** | 51 |
| 512 | 1 | 110 | 79 | **62** | 64 |
| 512 | 8 | 150–153 | 160 | **142–144** | 150–152 |
| 1024 | 1 | 186–187 | 172–174 | **152–158** | 156 |
| 1024 | 8 | 286 | 310–315 | 301 | 327–340 |
| 2048 | 1 | 349–350 | 321–324 | **304–307** | 306–310 |
| 2048 | 8 | 599–602 | 592–598 | 589–591 | 630–632 |
| 4096 | 1 | 674–676 | 620–627 | 625–652 | 641–677 |
| 4096 | 8 | 1305–1396 | 1155–1197 | 1189–1195 | 1275–1296 |

- §19.10's targets: **under 40 µs at 138 positions: met (17 µs)**. Under 600 at 4096:
  **not met (620–652 µs)**. At 4096 positions one layer reads 67 MB of f32 keys and
  values, so 625 µs is about 107 GB/s, near this machine's bandwidth. What remains there
  is the bytes (f32 KV; ADR-010 rules out f16 KV), not the order of operations.
- The one-dispatch kernel gains most at short context because it removes a dispatch
  boundary and the scores' round trip through device memory. Its scores use 128
  invocations per head instead of one per position, which is enough at decode.
- One run of the per-phase timing (`attention_two_phase_timings`; one run only, not
  repeated): at 138 positions, unrolling the scores took them from 37 to 12–15 µs.
- More unrolling of the value chains (32 instead of 16) and four chains per invocation
  (vec4) gained nothing or lost time; 16 scalar is the best of those measured.
- **Prelude:** 16.2–16.8 µs for 1 or 8 tokens. Most of that is the fixed cost of a
  dispatch; the host does the same work in about 1 µs (§19.5).

### 20.4 A whole layer (measured, composed where stated)

`tests/layer_spike.rs`: OLMoE's layer shape on synthetic weights, with Q and K Q4_K, V
Q6_K, O Q4_K, an f32 router, and 64 experts (Q4_K gate and up, Q6_K down) in one arena.
Three calls alternate 23 times per cell; the medians of the last 21 are shown. The
prefix is the prelude plus one-dispatch attention (`attn_oneb_u16`: each sequence reads
its own cache, so 8 sequences read 8 caches, unlike the 8-token rows above). Host
attention is the CPU's prelude and `attend` per (sequence, head) on `std::thread::scope`
threads. Spawning those threads costs tens of µs, so the true figure lies between the
1-thread time ÷ 5 and the 5-thread time, as in §19.5. Two runs, as ranges.

| sequences | positions | QKV submission: wall / GPU µs | routed (Step 3c): wall / GPU | host attention: 5 threads / 1 thread | routed with attention at its head: wall / GPU | GPU added by attention |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 138 | 504–532 / 225 | 1298–1302 / 969–970 | 163–213 / 244–277 | 1349–1360 / 1026–1029 | 57–59 |
| 1 | 1024 | 455–482 / 225 | 1247–1281 / 969 | 668–727 / 1509–1523 | 1435–1499 / 1165–1171 | 196–202 |
| 1 | 4096 | 537–643 / 225–227 | 1286–1486 / 970–978 | 2880–2992 / 6581–7854 | 2081–2306 / 1681–1752 | 711–775 |
| 8 | 138 | 1911–2032 / 1591 | 7431–7535 / 7059–7065 | 664–803 / 1676–1879 | 7673–7790 / 7264–7284 | 205–219 |
| 8 | 1024 | 1931–1985 / 1590–1591 | 7452–7467 / 7060–7061 | 4136–4306 / 11732–12739 | 8650–8724 / 8236–8276 | 1176–1215 |
| 8 | 4096 | 1922–1952 / 1591 | 7415–7473 / 7059–7063 | 14622–15187 / 47974–48442 | 11932–12130 / 11516–11672 | 4457–4609 |

- **The layer's GPU time.** Step 3c's two submissions take 225 + 969 = 1194 µs of GPU
  time at 1 × 138. With attention in the routed submission it is 1026–1029 µs, plus the
  QKV dispatches (225 µs) and about 10 µs for one more encoder boundary (§17.13). That
  composes to **about 1260 µs**, but the QKV dispatches were not recorded into the same
  command buffer: their GPU time is added, not measured there.
- **Per layer, wall.** "Today" is the QKV call, the host attention and the routed call.
  The one-submission layer is composed as the measured routed-with-attention call plus
  QKV's GPU time:

  | sequences | positions | today µs | one submission µs (composed) | saved per layer µs |
  |---:|---:|---:|---:|---:|
  | 1 | 138 | 1855–2043 | 1574–1585 | 270–469 |
  | 1 | 1024 | 2004–2431 | 1660–1724 | 280–771 |
  | 1 | 4096 | 3139–5009 | 2306–2533 | 606–2703 |
  | 8 | 138 | 9677–10370 | 9264–9381 | 296–1106 |
  | 8 | 1024 | 11798–13758 | 10241–10314 | 1484–3517 |
  | 8 | 4096 | 19020–24524 | 13523–13721 | 5299–11001 |

  In this harness a QKV call's wall minus its GPU time is about 230–420 µs, more than the
  125 µs per hand-off derived from the real pass (§18.9). With §18.9's figure instead, at
  1 × 138 the saving is **115–281 µs per layer, 1.8–4.5 ms per token**. Applied to Step
  3c's 28.95–29.18 ms that gives **24.5–27.3 ms, 36.6–40.9 tok/s**. That is an
  estimate, not a measurement. The low end assumes rayon's attention costs about the
  1-thread time ÷ 5, which is the likelier case.
- **Concurrent (8 sequences).** This is the first measurement of it (§19.8 had none).
  Attention on the device adds 205–219 µs of GPU time at 138 positions, against
  664–803 µs (5 threads) of host attention. At 4096 positions it adds 4.5–4.6 ms against
  14.6–15.2 ms on the host (48 ms on one thread). The device reads 537 MB per layer there,
  which is bandwidth-bound. The concurrent gain therefore grows with context, as §19.8
  expected.
- **What this does not show.**
  - The armed steps use their own buffers: the O projection reads the host's attention
    outputs. So the GPU does a chained layer's work, with a barrier in between, but
    without the data dependency. A chained layer would not overlap them either: the
    barrier already serialises them.
  - Synthetic weights; the expert choice is fixed by the synthetic router.
  - MoltenVK records the first timestamp late (§13), so the GPU times may be slightly
    under-counted. The walls are not affected.
  - Prefill is not measured.
  - No end-to-end tokens per second: that needs the production path (§19.10 step 3).

### 20.5 What this changes in §19's plan

- §19.8's estimate for 1 × 138 positions (35.6–37.2 tok/s) rested on a 52 µs kernel. The
  kernel is now 17 µs, and attention adds 57–59 µs of GPU time in the routed submission
  (prelude included). The estimate is now 36.6–40.9 tok/s. Wall-clock acceptance is
  still owed (§19.9 risk 8).
- The kernels to carry forward are `qk_prelude` and `attn_one` (u16): one dispatch per
  layer for decode, `BATCH` for concurrent sequences. The two-dispatch path (unrolled
  scores, then fused softmax-values) is the fallback for more positions than shared
  memory holds: `MAXP` = 4096. Contexts beyond that need the paged layout of §19.6 and a
  chunked score buffer, which are not built.
- Open for the owner before step 3 (production, behind its own switch):
  - the KV pool policy and its ledger charge (§19.7);
  - whether the mirror stays host-authoritative (§19.6, §19.9 risk 6);
  - the 4096-position cap per dispatch, or chunking above it;
  - whether to measure prefill before step 3 or within it.
- **Decided by the owner (2026-11-13)**, each the proposal of §19.6–§19.7:
  - **KV pool:** `--vulkan-kv-pool SIZE`, default what the device budget leaves after the
    expert cache (and the reserve), never more than one buffer the device addresses at
    once; allocated on first use and charged whole to the ledger's KV owner on the VRAM
    tier. KV never evicts experts. A pass that cannot get a block runs attention on the
    host.
  - **Mirror:** the host KV cache stays authoritative; the device holds a mirror.
  - **Above 4096 positions:** the device declines and attention runs on the host through
    the mirror. No chunking.
  - **Prefill:** stays on the host path (its rows copied into the pool), and is measured
    within step 3.

## 21. Step 3 of §19.10: decode attention in the routed submission (2026-11-13)

Behind `--vulkan-attn on|off` (default **off**; needs `--vulkan-route-o on`). Base: `main`
at 4eb2c93. Step 3c is the fallback, and with the switch off nothing changes: the
default build's submissions, shaders and bits are as before.

### 21.1 What was built

- **One submission per decode layer.** With the switch on, a decode step (every
  sequence in the pass has exactly one new token, no trace, no lookahead) offers each
  MoE layer to the device whole. The routed command buffer records, in order:
  1. `kv_sync`: the host rows the device does not hold yet are copied into the pool
     (copies only);
  2. the Q, K and V projections, as fused-matmul dispatches into a device-local buffer
     (one per run of matrices with one dtype and one row-block count: Q and K together,
     then V, for OLMoE);
  3. `attn_prelude`: the clip, the full-width Q and K norms, RoPE, and the store of the
     new key and value into the token's block. A copy of both goes back for the host;
  4. `attn_decode`: attention in one dispatch per (head, token) over the token's own
     sequence;
  5. Step 3c's steps unchanged: O, residual and FFN norm, router, routing, experts,
     combine.

  So a decode layer makes one submission instead of Step 3c's two, and attention no longer
  runs on the host between them.
- **Kernels.** `attn_prelude` is the harness's `qk_prelude` (§20.1) and `attn_decode` is
  `attn_oneb_u16` (§20.1), with the same operations in the same order. Only the
  addressing changed:
  - the KV store and reads go through a block table;
  - each token has its own position;
  - the norms are optional (a model without Q and K norms skips them).

  `kv_sync` has no arithmetic, so `shaders/build.sh` validates it without the
  NoContraction pass. The three are checked by `spirv_is_current`, and every existing
  SPIR-V file rebuilt byte for byte.
- **The KV pool** (runtime, `discrete.rs`): one arena of 256-position blocks. Each block
  holds one layer's keys, dimension-major (`[kv_head][d][256]`, coalesced scores), and
  its values, position-major (`[256][kv_head][d]`, coalesced value chains). That is
  4 MiB per block for OLMoE. Blocks are taken as a sequence grows and given back when its
  cache is dropped.
- **The mirror** lives in the sequence's KV cache's device slot (`KvCache::device_mut`).
  For each layer it holds the sequence's blocks and how many positions they hold.
  - Before a device pass, every position the host computed and the device does not hold
    is listed to copy in: prefill, and any layer or pass that ran on the host.
  - A mirror counts no further than the host's position count, so a pass that failed
    after the device stored its row is written again.
  - `KvCache` gains an epoch, bumped by `clear` and `restore_from`. A mirror from another
    epoch gives its blocks back and starts again.
  - The device returns the new keys and values, and the host stores them as before, so
    the host cache is always complete.
- **Declines (host attention, nothing written):**
  - prefill or a mixed pass;
  - a traced pass, or lookahead;
  - per-head Q and K norms (Qwen3), which the prelude does not implement;
  - a position at or past 4096;
  - a head dimension over 512;
  - QKV weights the fused kernels cannot read;
  - no free block. The other layers of that pass still run on the device, and a block
    the sequence already took stays with it.

  A declined layer runs today's host path and then Step 3c, unchanged.
- **Interface:** `MoeAttention` / `DeviceMoeAttention` at the head of a
  `RoutedMoe` / `DeviceRoutedMoe` with a prelude; `routes_moe_attention`; and
  `DiscreteDevice::kv_pool`. `attend` and the RoPE rotation moved into
  `qubivex-compute::math` unchanged, so the simulated device runs the same operations.
- **The simulated device** implements the stage over its own pool (`SimConfig::route_attention`,
  `kv_pool`). So the runtime's pool, mirror, sync and epoch logic is tested without a GPU.
- **Reports:** `DeviceReport::kv_pool` gives the pool's bytes, whether it is allocated,
  blocks in all and free, layers run on the device, rows copied in, and layers declined
  for lack of a block.

### 21.2 Parity (measured)

- **Simulated device** (`tests/discrete.rs`):
  - every decode step's logits equal the CPU's on the three MoE fixtures (12 steps);
  - every decode layer ran on the device, the prompt's rows were copied in exactly once
    per layer, and every block was back in the pool after the cache was dropped;
  - a pool one block short of the model runs one layer per step on the host, with the
    CPU's bits;
  - a cache cleared and refilled with another prompt gives the CPU's bits;
  - 4 concurrent requests give the CPU's tokens.
  - **Positive controls:** without the row copy-in, all three tests fail; without the
    epoch reset, the clear-and-refill test fails.
- **Vulkan, MoltenVK** (`tests/vulkan.rs`):
  - every decode step's logits equal the CPU's on `olmoe-tiny` and `olmoe-tiny-gguf`.
    `qwen3-moe-tiny` declines to the host, as designed, and also matches;
  - one decode step makes one submission per layer plus the head;
  - 4 concurrent requests give the CPU's tokens;
  - a 250-token prompt decodes 20 steps past position 256, into a second block, with the
    CPU's logits at every step (250 × layers rows copied in).
  - **Positive controls** (shader restored and rebuilt after each): reading every key
    from the first block fails the block-crossing test only; the softmax sum reordered
    (`sum += a1 + a2; sum −= a2`) fails both.
- **The real OLMoE** (`olmoe_routed_on_vulkan_matches_the_cpu`, `QUBIVEX_OLMOE_GGUF_DIR`,
  256 MiB pool): with decode attention on the device it gives the CPU's prompt logits and
  Colibri's 12 greedy tokens. At least 11 × 16 decode layers ran attention on the device.

### 21.3 Measured on mains

`scripts/bench-vulkan-baseline --tag step3d --variant routedo='--vulkan-route on --vulkan-route-o on' --variant attn='--vulkan-route on --vulkan-route-o on --vulkan-attn on --vulkan-kv-pool 512MiB'`:
AC power, untimed, 3 alternated rounds (records
`2026-11-13-m3pro-vulkan-step3d-{base,routedo,attn}-{1,2,3}.json`, log
`…-step3d-mains.txt`). Every record's tokens equal the Step 1a reference.

**Memory pressure:** swap was 3.4 GB at the start and 10.5 GB at the end, with the IDE
open. Round 2's Step 1a and Step 3c medians are far off their other rounds (43.4 and
36.8 ms), and every record has slow outlier steps (maxima of 34–150 ms). So the medians
below carry that noise. The per-step minimum is shown too: interference cannot make a
step faster. A quieter rerun would tighten the single-stream range.

| Round | Step 1a ms/token | Step 3c | **Step 3c + attention** | Step 1a prefill ms | Step 3c | + attention |
|---|---|---|---|---|---|---|
| 1 | 37.24 | 30.19 | **27.59** | 199.7 | 205.8 | 207.6 |
| 2 | 43.42 (disturbed) | 36.82 (disturbed) | **27.58** | 210.5 | 231.7 | 205.5 |
| 3 | 35.21 | 29.17 | **25.82** | 201.8 | 200.6 | 198.5 |
| fastest step | 30.29–32.21 | 26.58–27.60 | **24.75–24.88** | | | |
| tok/s (median) | 23.0–28.4 | 27.2–34.3 | **36.2–38.7** | | | |
| submissions/token | 305 | 33 | **17** | | | |

- Against Step 3c in the same round (rounds 1 and 3): **−2.6 and −3.3 ms per token
  (−8.6%, −11.5%)**. The fastest steps are 1.8–2.8 ms apart. That is within §20.4's
  estimate of 1.8–4.5 ms. The fence waits per token fell from 26.4–27.3 ms (33 waits) to
  24.2–24.7 ms (17 waits).
- **35 tok/s: reached** (36.2–38.7 tok/s by median; the batch example's single stream
  measures 37.3–38.0, below).
- **Prefill:** no regression visible (198.5–207.6 ms against 200.6–231.7 for Step 3c and
  199.7–210.5 for Step 1a). The 10-token prompt is neither routed nor on the device. Its
  rows are copied into the pool by the first decode step, whose time is in the decode
  median.

Concurrent tok/s (batch example, 32 tokens per sequence, median of 3 per round; tokens
identical in every mode; ranges over the three rounds):

| N | Step 1a | Step 3c | **+ attention** | Gain over Step 3c (rounds 2–3) |
|---|---|---|---|---|
| 1 | 27.93–30.59 | 31.90–34.85 | **37.34–37.96** | +8.8% to +9.0% |
| 2 | 33.09–34.91 | 40.34–41.19 | **43.63–44.04** | +6.9% to +7.0% |
| 4 | 38.78–40.32 | 43.99–45.72 | **47.72–47.92** | +4.8% to +5.2% |
| 8 | 42.57–44.27 | 48.27–49.18 | **50.00–50.30** | +2.3% to +2.7% |

Round 1's Step 3c single-sequence figure (31.90) is low against its other rounds
(34.71–34.85), so the gains use rounds 2 and 3. The attention rows agree across all
three rounds within 1.7%.

**Gates:**
- identical tokens: yes, in every record and mode;
- numerical correctness: yes (§21.2);
- no prefill regression: none visible;
- no concurrent regression: a gain at every N, smaller as N grows;
- a single-stream gain over Step 3c: yes.

The pool was 512 MiB here: 128 blocks, enough for 8 sequences of up to 256 positions. The
default (what the budget leaves after the experts) was not measured. On this machine it
would be the largest buffer MoltenVK addresses, about 4 GB of unified memory, which the
pressure above argues against as a default here.

### 21.4 Open for the owner

- Accept Step 3 with the switch default off, as Steps 3b and 3c were (ADR-019 amendment
  5), or ask for a quieter single-stream rerun first.
- The default pool size. As decided it is what the budget leaves, up to one addressable
  buffer. On unified memory that can be about 4 GB taken from RAM at the first decode
  step; a cap (for example 1 GiB) or a size derived from the context and the
  concurrency limit may suit this machine better.
- Per-head Q and K norms (Qwen3) are declined to the host. Supporting them is a change to
  the prelude's reduction (one RMSNorm per head), with its own parity tests.
- Not built: chunked attention past 4096 positions; device attention for prefill
  (measured unchanged, as decided).
