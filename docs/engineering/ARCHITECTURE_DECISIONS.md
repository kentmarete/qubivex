# Architecture decisions: reconciliation

- Date: 2026-11-03
- Status: **Owner decisions taken 2026-11-03** on §2.1 (named opt-ins only, ADR-010),
  §2.6 (Metal first, ADR-007 amendment), §3 (keep `MIGRATION_PLAN.md`, fold the EIP in)
  and the tensor scope (ADR-009, implemented). The ADR status changes proposed in §1
  for ADR-001, 003, 004 and 006 still await review.
- Inputs: ADR-001 to ADR-008 (`docs/decisions/`), `QUBIVEX_DESIGN.md`,
  `MIGRATION_PLAN.md`, `IMPROVEMENTS_EVALUATION.md`, and the Engineering Improvement Plan
  (EIP).

This document compares the EIP with the decisions already recorded, and proposes how
to resolve each conflict. It changes no ADR by itself. Each accepted resolution is
recorded in the ADR it affects, or in a new ADR numbered from 009.

## 1. ADR status register

| ADR | Title | Recorded status | Actual state | Proposed status |
|---|---|---|---|---|
| 001 | Rust workspace, phase-gated crates | Proposed | Implemented as written; enforced by lints, `xtask`, CI | **Accepted** (owner review pending since Phase 1) |
| 002 | Hybrid compute, own kernels, no Candle | Accepted | Own kernels built. Spinning pool and dequantise + `gemm` prefill **not built** | Accepted, plus an "Implementation status" note listing the two unbuilt parts |
| 003 | Architecture registry by manifest id | Proposed | Implemented; OLMoE and a Qwen3-MoE fixture load through one factory | **Accepted** |
| 004 | Versioned manifest, cross-validated | Proposed | Implemented (schema v1, import, cross-checks) | **Accepted** |
| 005 | One measured, role-driven memory planner | Proposed | Not built (Phase 5) | Proposed; extended by §2.3 |
| 006 | Synchronous lease-based expert store | Proposed | Leases, dedup, the priority I/O queue, I/O threads and `acquire_many` built (Phase 6 slice 2, amendment 1); direct I/O and io_uring not built | **Accepted** for the lease and sync model; direct I/O and io_uring stay Phase 6 |
| 007 | Backends behind one trait | Proposed | Trait exists in host-only form | Proposed; amended by §2.4 and §2.6 (backend order) |
| 008 | Expert cache: slabs, LFRU, hysteresis | Proposed | Per-layer pools, slab reuse and LFRU built (Phase 6 slice 1, amendment 1); pins and the persisted hot set not built | Proposed |

The EIP proposes `docs/adr/ADR-003-TENSOR-STORAGE.md` through `ADR-008-ASYNC-EXECUTION.md`.
Those numbers are taken. **Proposal:** keep `docs/decisions/`, and map the EIP's ADRs as
follows:

| EIP ADR | Maps to |
|---|---|
| ADR-003 Tensor storage | **ADR-009** Tensor descriptors, storage and views (Accepted) |
| ADR-004 Memory management | Existing **ADR-005** (planner) and **ADR-008** (cache), amended |
| ADR-005 Compute backends | Existing **ADR-007**, amended |
| ADR-006 Graph IR | **New ADR-012**, written only when Graph IR is scheduled |
| ADR-007 Numerical precision | **ADR-010** Numerical precision policy (Accepted) |
| ADR-008 Async execution | **New ADR-011** Asynchronous execution and buffer lifetimes |

## 2. Conflicts and proposed resolutions

### 2.1 Lossless versus lossy computation: **DECIDED**, named opt-ins only (ADR-010)

**Existing position.**
- `QUBIVEX_DESIGN.md` invariant 4: lossy is opt-in and named.
- `IMPROVEMENTS_EVALUATION.md`: "lossy options are never part of any profile".
- Pending review decision 9: no lossy option in any execution profile.
- Every kernel today is lossless: f32 activations times the exact dequantised weights.

**EIP position (§6.3).** A `PrecisionPolicy { Strict, Standard, Performance,
Explicit(DataType) }`, with lossless activations by default and approximate activation
quantisation "only when explicitly enabled".

**Conflict.** A `Performance` policy is a lossy profile. The two documents agree on the
default and on "never silent". They disagree on whether a profile may bundle lossy
behaviour.

**Proposed resolution (ADR-010).**
- **Default is lossless and is not a mode you can forget to set.** Weight quantisation
  is a property of the model file, not a precision choice. Running a Q4_K model
  losslessly means the dequantised weights are used exactly.
- **Lossy behaviour is a set of named, per-feature opt-ins**, for example
  `activation_q8_dot`, `bf16_accumulate` or `fp16_kv_cache`. Each is listed in
  `qubivex run --help`, logged at start-up, and recorded in every benchmark report.
- **No profile turns on a lossy opt-in.** "Performance" may mean more threads, a larger
  cache or more prefetching, but never different numerics.
- Every lossy opt-in documents its measured error against the lossless path, and the
  greedy-token divergence point on the reference prompt. Colibri's lossy default
  diverges after 86 tokens, and that is the kind of number to publish.
- The EIP's `Strict` corresponds to the default here. `Explicit(DataType)` corresponds
  to an individual opt-in.

### 2.2 Tensor ownership

**Existing position.** None is recorded. In practice:
- dense weights are owned `Vec<u8>`;
- experts are `Arc<[u8]>` held by a `Lease`;
- kernels borrow `WeightView<'_>`;
- activations are `&[f32]`.

**EIP position (§4.1B).** `Tensor { desc, storage }` with owned, borrowed and shared
storage. Views never outlive storage. Asynchronous operations retain buffers until they
complete.

**Conflict.** None in principle. The gap is that there is no shared descriptor (see
BASELINE_AUDIT §5).

**Proposed resolution (ADR-009).** Adopt the design in
[TENSOR_ARCHITECTURE.md](../architecture/TENSOR_ARCHITECTURE.md):
- `TensorDesc` in `qubivex-core`, with byte strides and a byte offset.
- Typed, safe views and `Arc`-shared owned tensors in `qubivex-compute`.
- No new crate (ADR-001).
- No change to kernel signatures.

### 2.3 Memory placement

**Existing position.**
- ADR-005: one planner, RAM and VRAM budgets never summed, placement by tensor role,
  `MemoryLedger` as ground truth, pressure responses in a fixed order (stop prefetch,
  shrink caches, refuse admission, never kill an active request).
- ADR-008: per-layer slab pools, LFRU with hysteresis.

**EIP position (§4.1C).**
- Five tiers: VRAM, RAM, pinned host memory, NVMe, remote.
- A generic `MemoryAllocator` trait.
- `MemoryPlacementPolicy { GpuPreferred, CpuPreferred, Balanced, LowMemory, Explicit }`.
- LRU, LFRU and priority eviction.
- NUMA.

**Conflicts.**
1. The EIP orders the memory manager (its Phase 2) before CPU compute (its Phase 3). Here
   CPU compute already exists, so the order doesn't matter in practice.
2. A general-purpose allocator trait against ADR-008's fixed slabs. Slabs are
   deliberately not a general allocator: a fixed size and fixed count give no
   fragmentation by construction.
3. The policy names differ. ADR-005 uses profiles `performance | balanced | saver |
   custom`.

**Proposed resolution (amend ADR-005).**
- Keep the planner and ledger design.
- Add **pinned host** as a tier, needed once a GPU backend exists. Record **remote
  storage** as out of scope until a user needs it.
- The allocator interface is the ledger-charging `allocate` of each backend (ADR-007),
  not a separate global allocator.
- Map the EIP's `GpuPreferred`, `CpuPreferred` and `LowMemory` onto the existing
  profiles, plus an explicit device override. One name per concept.
- NUMA stays in Phases 10–11 as planned; the M3 Pro has no NUMA nodes to test on.

### 2.4 Backend interfaces

**Existing position.**
- `qubivex-compute::ComputeBackend` is implemented with `name`, `supports`, `matmul`
  and `dequant_row`, host memory only.
- `QUBIVEX_DESIGN.md` §4.4 describes the eventual trait: `allocate`, `upload`,
  `download`, `matmul`, `rms_norm`, `rope`, `attention`, `expert_ffn_batch` and
  `synchronize`, returning `Event`.
- ADR-007: coarse, model-agnostic operations.

**EIP position (§5.1).** A similar trait, plus an explicit `ExecutionStream` argument
and a separate `qubivex-backend` crate.

**Conflicts.**
1. Crate naming: `qubivex-backend` and `qubivex-cpu` against the existing
   `qubivex-compute` and `qubivex-backend-cpu`.
2. Streams as parameters against events only.
3. Norm, RoPE and attention currently live in `qubivex-arch::ops`, outside the trait. This
   is a gap against both documents.

**Proposed resolution (amend ADR-007).**
- **Keep the existing crate names**, so nothing is renamed without a technical reason
  (EIP Rule 2).
- Move `rms_norm`, `rope`, `attention` and `softmax` behind `ComputeBackend` **when the
  first GPU backend needs them**, not before. Moving them now would add a dispatch layer
  with one implementation.
- Decide streams against events in ADR-011, when the first GPU backend is designed. The
  CPU backend needs neither.
- Resolve object safety and associated buffer types at the same time. A `dyn
  ComputeBackend` with an associated `Buffer` type is not object-safe. The likely shape
  is an opaque `DeviceBuffer` handle that holds an `Arc` of backend state.

### 2.5 Asynchronous execution

**Existing position.**
- ADR-006: a **synchronous** hot path with dedicated I/O threads. Tokio is used only at
  the future gateway.
- The CPU backend is synchronous on its own rayon pool.

**EIP position (§5.1, principle 7).** Asynchronous GPU execution. Event completion is
distinct from submission. Explicit dependencies, and no global synchronisation after
each kernel.

**Conflict.** There is none once terms are separated. "Async" in the EIP means device
queue semantics (Metal command buffers, CUDA streams), not Rust `async`. ADR-006 rules
out Rust `async` on the hot path only.

**Proposed resolution (ADR-011, written with the first GPU backend).**
- Device work is submitted to a queue and returns an `Event`.
- **Every buffer an in-flight operation touches is kept alive by a reference-counted
  handle owned by that operation until its event completes.** This is the tensor design's
  `Arc` rule (TENSOR_ARCHITECTURE §6).
- Expert leases become movable into an in-flight operation: the lease's `Arc<[u8]>`
  already allows this.
- No Rust `async` on the compute path.

### 2.6 CPU/GPU responsibilities and scheduling: **DECIDED**, Metal first (ADR-007 amendment)

**Existing position.**
- ADR-007 sets the order CUDA, Vulkan, then Metal.
- Results combine in rank order so hybrid execution keeps CPU summation order.
- Invariant 1: placement may change speed, never bits.
- Cross-backend parity uses a documented drift tolerance.

**EIP position.**
- Metal first, because the dev machine is an M3 Pro (§5.3).
- Then CUDA; Vulkan last.
- An "intelligent scheduler" choosing CPU or GPU per operation from live device
  utilisation and memory pressure (§11).

**Conflicts.**
1. **Backend order.** The environment changed after ADR-007: the M3 Pro has a GPU and
   there is no CUDA hardware (blocker B2). Metal is the only GPU that can be tested
   locally.
2. **Live per-operation scheduling against reproducibility.** CPU and GPU kernels don't
   produce identical bits. If device choice depends on live load, the same prompt can
   give different tokens from run to run, which breaks invariant 1.

**Proposed resolution (amend ADR-007).**
1. Order: **Metal, then CUDA, then Vulkan.** This needs your decision.
2. **Device assignment is part of the execution plan, fixed for a given model,
   configuration and hardware.** It is chosen from measured costs at plan time. The
   scheduler can admit, delay or refuse work by memory pressure, but it never moves an
   operation to another device mid-request. With that rule the EIP's goal (don't assume
   the GPU is faster) and invariant 1 both hold.

### 2.7 Graph IR and compiler

**Existing position.** Not in the design. The forward pass is Rust code in
`qubivex-arch`, built from coarse backend operations (ADR-007).

**EIP position.** Its Phase 5: a graph IR, shape inference, fusion, memory planning
and an execution planner (§8).

**Proposed resolution.**
- **Defer.** EIP Rule 6 ("do not implement graph compilation before the operator
  contracts are defined") and Rule 10 both point the same way. The operator contracts
  (the amended `ComputeBackend`) don't exist yet.
- Revisit when a second architecture family, or the first GPU backend, shows that
  hand-written forward passes are the bottleneck to adding models. Record the decision
  then as ADR-012.

### 2.8 Crate count

**Existing position.** ADR-001: 11 crates, each created only with real code.

**EIP position.** About 16 crates, adapted to the repository "rather than blindly
recreating".

**Proposed resolution.** Consistent with ADR-001. New crates appear in the phase that
fills them:
- `qubivex-memory` in Phase 5;
- `qubivex-backend-metal` with the first Metal kernels;
- `qubivex-server` in Phase 9.

The EIP's `qubivex-tensor` is **not** created: descriptors go to `qubivex-core` and
storage to `qubivex-compute`. The EIP's `qubivex-quant` is not created either:
`backend-cpu::formats` already holds dequantisation.

### 2.9 Determinism

**EIP principle 9.** Deterministic "under a fixed precision and scheduling
configuration".

**Existing position.** Stronger: bitwise identical across thread counts, batch sizes,
SIMD paths and cache capacities.

**Resolution.** Keep the stronger guarantee for the CPU backend, and state the GPU
tolerance per backend (ADR-007).

## 3. Roadmap reconciliation: **DECIDED**, keep MIGRATION_PLAN.md

**Proposal:** keep `MIGRATION_PLAN.md` as the single roadmap and fold the EIP's content
into it.

| EIP phase | Existing phase | Action |
|---|---|---|
| 0 Audit | (new) | This document and BASELINE_AUDIT |
| 1 Tensor foundations | (new, before Phase 5) | Scoped tensor-descriptor work (TENSOR_ARCHITECTURE) |
| 2 Memory manager | Phase 5 (planner, ledger), Phase 6 (slabs, I/O) | Unchanged, with the §2.3 amendments |
| 3 CPU compute | Phase 4 follow-ups | Add: in-workspace benchmarks first, then AVX2, `gemm` prefill and Q4_K speed-up, each measured |
| 4 Model execution | Phase 4 (done for MoE) | Add: a dense architecture and a real SafeTensors run, as an extensibility test |
| 5 Graph IR | (none) | Deferred (§2.7) |
| 6 GPU | Phase 8 | Metal first (§2.6) |
| 7 Advanced execution | Phases 10–11 | Unchanged |
| 8 Production inference | Phase 9 | Unchanged |

Prefetch (Phase 7) stays before GPU (Phase 8), as in pending review decision 7.
