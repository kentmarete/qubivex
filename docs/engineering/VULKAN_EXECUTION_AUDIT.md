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
3. **The cost of a round trip.**
   - GPU time per submission: 19.6 ms / 305 ≈ 64 µs.
   - Host wait per submission: 86 ms / 305 ≈ 280 µs. So about 220 µs per submission is
     fixed latency (submit to fence signalled), not GPU work.
   - The critical path is about 113 round trips × ~316 µs ≈ 35.7 ms, which matches the
     untimed median (35.5–35.9 ms).
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
  Two submissions per layer for the experts take the critical path from 7 round trips
  to 5 per layer, about 81 per token. At ~316 µs each that is about 26 ms, roughly
  37 tok/s, provided the grouped dispatch's GPU time is no worse than the serial
  dispatches it replaces. The kernel test's 24 matrices in one submission (1.26 ms)
  against ~16 × 64 µs ≈ 1.0 ms today says to measure that before building on it.
- Steps 3–4 target the ~16–19 ms of idle time directly.

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
