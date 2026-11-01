# QUBIVEX: Architecture Design

Status: **Proposed. Awaiting architectural review before Phase 1.**

Inputs: the QUBIVEX specification and [COLIBRI_ANALYSIS.md](COLIBRI_ANALYSIS.md).
Where this design departs from the specification, the departure is marked
**[DEVIATION]** with its reason.

---

## 1. Design invariants

These rules are checked by tests and are not open to trade-off for speed.

1. **Placement decides speed, never bits.** For a fixed backend and kernel set, the
   tokens and logits are identical whether an expert came from VRAM, RAM or disk, and
   whatever the cache capacity (1 slot to fully resident). Expert outputs are combined
   in **router rank order**, independent of which experts were resident or which
   device computed them.
2. **Prediction never routes.** Prefetch predictors may read router state. They never
   write it.
3. **No model names in shared code.** Models are described by manifests. Architectures
   are registered components selected by an `architecture` id.
4. **Lossy is opt-in and named.** Any option that changes numerics (activation
   quantisation, top-p expert pruning) needs an explicit flag, logs a warning, and is
   reported in the response metadata.
5. **Untrusted inputs are validated, never repaired.** Byte sizes, shapes and dtypes
   come from config and are checked against file headers before any read into a buffer.
6. **Errors fail the request, not the process.** No `panic!` or `exit` on ordinary
   errors (OOM on allocation, a bad shard, a cancelled request).

## 2. Layered architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ APPLICATIONS      apps/cli (qubivex)      apps/server                │
├──────────────────────────────────────────────────────────────────────┤
│ GATEWAY           qubivex-api: Axum, OpenAI schema, SSE, auth,       │
│                   admission, cancellation, timeouts                  │
├──────────────────────────────────────────────────────────────────────┤
│ RUNTIME           qubivex-runtime: Engine, ModelHandle lifecycle,    │
│                   scheduler, sessions, sampling, KV cache            │
├───────────────┬──────────────┬──────────────┬────────────────────────┤
│ MODEL         │ MEMORY       │ EXPERT STREAM│ HARDWARE               │
│ qubivex-model │ qubivex-     │ qubivex-     │ qubivex-hardware       │
│ manifest,     │ memory:      │ expert-      │ profile, snapshots,    │
│ registry,     │ budgets,     │ stream:      │ storage probes         │
│ formats,      │ planner,     │ store, cache,│                        │
│ tensor index  │ policies     │ loaders,     │                        │
│               │              │ prefetch     │                        │
├───────────────┴──────────────┴──────────────┴────────────────────────┤
│ ARCHITECTURES     qubivex-arch: components (attention, norm, RoPE,   │
│                   router, SwiGLU expert) + registered architectures  │
├──────────────────────────────────────────────────────────────────────┤
│ COMPUTE           qubivex-compute: Backend trait, buffers, ops       │
│                   qubivex-backend-cpu (Phase 4)                      │
│                   qubivex-backend-cuda / -vulkan / -metal (Phase 8+) │
├──────────────────────────────────────────────────────────────────────┤
│ CORE              qubivex-core: ids, dtypes, shapes, errors, config  │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.1 Token data flow (one decode step, one MoE layer)

```
 hidden x ──► attention (arch) ──► post-norm ──► router logits ──► top-K (arch)
                                       │                              │
                                       │ (copy, read-only)            │ RoutingResult
                                       ▼                              ▼
                              Predictor (L+1)            ExpertStore.acquire_many(L, ids)
                                       │                    │ hit ─► lease
                         prefetch(L+1, ids) (advisory)      │ in-flight ─► wait on same load
                                       │                    │ miss ─► IoScheduler (bounded)
                                       ▼                    ▼
                                 IoScheduler ◄──────── ExpertLoad (pread / io_uring)
                                                            │ validated bytes ─► slot
                                                            ▼
                                  Backend.expert_ffn_batch(x, leases, ranks)
                                                            │ per-rank outputs
                                                            ▼
                                  combine in RANK ORDER: out += w_k · h_k
                                                            │
                                      leases dropped ──► slots evictable again
```

## 3. Cargo workspace

**[DEVIATION]** The specification lists about 25 crates. Splitting this finely before
the interfaces have stabilised creates churn and empty crates, which the specification
itself forbids. I propose **11 crates at Phase 1–7**, each created only in the phase
that first gives it real code. Crates split when a boundary is proven.

```
qubivex/
├── Cargo.toml                     # workspace, shared lints, profile.release (lto, codegen-units=1)
├── crates/
│   ├── qubivex-core/              # P1  ids, DType, Shape, error types, config types
│   ├── qubivex-hardware/          # P2  HardwareProfile, ResourceSnapshot, probes
│   ├── qubivex-model/             # P3  manifest, ArchitectureRegistry, TensorIndex,
│   │   └── src/formats/{safetensors,gguf}.rs        validation, discovery
│   ├── qubivex-compute/           # P4  ComputeBackend trait, buffers, op descriptors
│   ├── qubivex-backend-cpu/       # P4  SIMD kernels (AVX2, AVX-512, NEON, scalar ref)
│   ├── qubivex-arch/              # P4  components + architectures (transformer_moe)
│   ├── qubivex-tokenizer/         # P4  wraps HF `tokenizers`, chat templates
│   ├── qubivex-expert-stream/     # P4 (minimal) → P6/P7  store, cache, I/O, prefetch
│   ├── qubivex-memory/            # P5  budgets, planner, placement policies
│   ├── qubivex-runtime/           # P4 → P9  engine, lifecycle, KV cache, sampling, scheduler
│   ├── qubivex-api/               # P9  Axum gateway
│   ├── qubivex-test-utils/        # P3  tiny-fixture loaders, tolerance asserts
│   ├── qubivex-backend-cuda/      # P8
│   ├── qubivex-backend-vulkan/    # P8+
│   └── qubivex-backend-metal/     # later
├── apps/
│   ├── cli/                       # P2 `qubivex` binary (system, models, plan, run, serve)
│   └── server/                    # P9 (may be a CLI subcommand only)
├── models/manifests/              # P3 manifest examples (no weights)
├── fixtures/                      # P3 tiny fixtures + generator scripts (pinned deps)
├── benchmarks/                    # Criterion + end-to-end harness
├── tests/                         # workspace integration tests
└── docs/{architecture,decisions,development,performance,model-support}/
```

Things merged into other crates for now, to be split out when they grow:
`kv-cache` and `sampling` go into `runtime`; `quantization` goes into
`backend-cpu` and `model::formats`; `observability` uses `tracing` plus a `metrics`
module in `runtime`; `downloader` waits until Phase 11.

### 3.1 Dependency direction (enforced)

```
apps ─► api ─► runtime ─┬─► arch ─────┬─► compute ─► core
                        ├─► memory ───┤
                        ├─► expert-stream ─► compute, core
                        ├─► model ────┘ (no compute dependency)
                        ├─► tokenizer
                        └─► hardware ─► core
backend-* ─► compute ─► core      (backends never depend on runtime/api/model)
```

These rules are enforced by a workspace test that parses `cargo metadata` and fails on
a forbidden edge, for example `model → backend-*`, `memory → arch`,
`expert-stream → arch`, `backend-* → api`, or anything depending on `apps/*`. A
`clippy.toml` `disallowed-names` and a CI grep guard against model-name literals
(`olmoe`, `qwen`, `deepseek`, `glm`, …) outside `models/`, `fixtures/`, `docs/` and
tests.

## 4. Interface definitions

These are signatures, not implementations. Lifetimes and ownership are settled when
each phase implements its interface. Only the interfaces needed by a phase are
written in that phase.

### 4.1 Core identifiers and errors (`qubivex-core`)

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LayerIdx(pub u32);
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ExpertIdx(pub u32);

/// Interned per loaded model; cheap to copy, unlike a String model id.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ModelKey(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ExpertId { pub model: ModelKey, pub layer: LayerIdx, pub expert: ExpertIdx }

pub enum DType { F32, F16, BF16, I8RowScaled, Ggml(GgmlType), /* … */ }

// One error enum per domain (thiserror), each convertible into a top-level
// `QubivexError` at the runtime boundary. Public API maps to `ApiError` with
// paths and internal detail stripped.
pub enum ModelError { /* … */ }  pub enum StorageError { /* … */ }
pub enum PlanningError { /* … */ } pub enum BackendError { /* … */ }
pub enum ExecutionError { /* … */ } pub enum HardwareError { /* … */ }
```

**[DEVIATION]** `ExpertId` uses an interned `ModelKey`, not `model_id: String`. It is
hashed on every expert lookup (layers × top-K per token), and a heap string there is
measurable overhead.

### 4.2 Hardware (`qubivex-hardware`)

```rust
pub struct HardwareProfile {            // static; probed once, cached
    pub cpu: CpuInfo,                   // vendor, model, arch, cores (P/E), SIMD set, NUMA nodes
    pub memory: MemoryInfo,             // total RAM, cgroup limit
    pub gpus: Vec<GpuInfo>,             // per backend probe result, incl. "unavailable: reason"
    pub storage: Vec<StorageInfo>,      // mount, fs, device kind, capacity
    pub os: OperatingSystem,
}
pub struct ResourceSnapshot {           // dynamic; cheap; sampled by the pressure monitor
    pub taken_at: Instant,
    pub ram_available: Bytes,
    pub vram_available: HashMap<DeviceId, Bytes>,
    pub process_rss: Bytes,
}
pub struct StorageMeasurement {         // measured on demand (`qubivex doctor --storage`), cached per device
    pub seq_read_bps: f64, pub rand_read_4m_latency: Duration, pub direct_io_gain: Option<f64>,
}
pub trait ResourceProbe: Send + Sync { fn snapshot(&self) -> Result<ResourceSnapshot, HardwareError>; }
```

Each field is `Known(T)` or `Unknown(reason)`, never a silent zero. GPU probes run in
a child process with a timeout, following Colibri, so a broken driver cannot hang
QUBIVEX.

### 4.3 Model system (`qubivex-model`)

```rust
pub struct Manifest { pub schema_version: SemVer, pub model: ModelSection,
                      pub architecture: ArchRef /* id + version */, pub weights: WeightsSection,
                      pub tokenizer: TokenizerSection, pub overrides: serde_json::Value }

/// What the runtime trusts: manifest ∩ config.json ∩ tensor headers, cross-checked.
pub struct ModelMetadata { pub key: ModelKey, pub manifest: Manifest,
                           pub config: ArchConfig /* architecture-parsed */,
                           pub tensors: TensorIndex, pub roles: TensorRoles }

/// Every tensor tagged with a role; the planner works on roles, never on names.
pub enum TensorRole { Embedding, OutputHead, Norm, Attention, Router,
                      RoutedExpert { layer: LayerIdx, expert: ExpertIdx, part: ExpertPart },
                      SharedExpert { layer: LayerIdx }, DenseFfn, Other }

pub trait ArchitectureFactory: Send + Sync {
    fn id(&self) -> &'static str;                         // "transformer_moe"
    fn supported_versions(&self) -> VersionReq;
    /// Parse the architecture-specific config (from manifest + config.json).
    fn parse_config(&self, m: &Manifest, raw: &serde_json::Value) -> Result<ArchConfig, ModelError>;
    /// Map tensor names to roles and expected shapes/dtypes. Drives validation AND planning.
    fn tensor_schema(&self, cfg: &ArchConfig) -> Result<TensorSchema, ModelError>;
    fn build(&self, meta: &ModelMetadata, ctx: &BuildContext) -> Result<Box<dyn Model>, ModelError>;
}

pub struct ArchitectureRegistry { factories: HashMap<&'static str, Arc<dyn ArchitectureFactory>> }
impl ArchitectureRegistry {
    pub fn register(&mut self, f: Arc<dyn ArchitectureFactory>) -> Result<(), ModelError>; // duplicate → error
    pub fn unregister(&mut self, id: &str) -> Option<Arc<dyn ArchitectureFactory>>;
    pub fn resolve(&self, r: &ArchRef) -> Result<Arc<dyn ArchitectureFactory>, ModelError>; // id + version check
    pub fn list_supported(&self) -> Vec<ArchDescriptor>;
}
```

**[DEVIATION]** The specification has `ArchitectureFactory::create → ModelArchitecture
→ build`, which is two levels of indirection. I merged them into one factory with
`parse_config`, `tensor_schema` and `build`. `tensor_schema` is the important
addition. It is what makes the planner and the validator model-agnostic. Colibri needs
per-family Python geometry functions because it lacks this.

**[DEVIATION]** No `unregister` on a registry that is serving. Registration happens at
startup, and the registry is then frozen (`Arc<ArchitectureRegistry>`). `unregister`
exists only for tests.

Architecture variance is configuration, not code. `transformer_moe` takes, from the
manifest and config: attention kind (MHA / GQA / MQA), QK-norm on/off, norm kind,
RoPE parameters, router scoring (softmax / sigmoid), top-K normalisation, routed
scaling factor, shared-expert count, expert activation (SiLU-GLU) and tensor-name
patterns. OLMoE and Qwen3-MoE should both load through it with **zero code change**.
That is the extensibility acceptance test (see the migration plan).

### 4.4 Compute (`qubivex-compute`)

```rust
pub trait ComputeBackend: Send + Sync {
    fn device(&self) -> &DeviceInfo;
    fn capabilities(&self) -> &BackendCapabilities;  // dtypes × ops supported, alignment, max alloc
    fn allocate(&self, req: AllocationRequest) -> Result<DeviceBuffer, BackendError>;
    fn upload(&self, src: &[u8], dst: &DeviceBuffer, offset: u64) -> Result<Event, BackendError>;
    fn download(&self, src: &DeviceBuffer, dst: &mut [u8]) -> Result<Event, BackendError>;
    // Ops are coarse and model-agnostic. Fine-grained ops on a GPU cost launch overhead.
    fn matmul(&self, op: &MatmulOp) -> Result<Event, BackendError>;           // W·x, W any supported dtype
    fn rms_norm(&self, op: &NormOp) -> Result<Event, BackendError>;
    fn rope(&self, op: &RopeOp) -> Result<Event, BackendError>;
    fn attention(&self, op: &AttentionOp) -> Result<Event, BackendError>;
    /// Grouped SwiGLU over many (expert, rows) pairs, one call per MoE layer per block.
    fn expert_ffn_batch(&self, op: &ExpertBatchOp) -> Result<Event, BackendError>;
    fn synchronize(&self) -> Result<(), BackendError>;
}
```

On the CPU backend `Event` is a ready marker and the ops are synchronous, running on
the backend's own thread pool. Accounting is part of the contract: `allocate` charges
a `MemoryLedger` owned by `qubivex-memory` (see 4.6), so VRAM use is never a guess.

### 4.5 Expert streaming (`qubivex-expert-stream`)

**[DEVIATION]** The specification gives an `async fn load` per expert on Tokio. I
propose a **synchronous, lease-based hot path** with Tokio only at the gateway. The
reasons:

- Each decode token does `layers × top-K` lookups (OLMoE: 16 × 8 = 128). Nearly all
  are cache hits that should cost one hash probe and an atomic increment, not a future
  plus a waker.
- The compute loop is CPU-bound and runs on dedicated threads. Blocking it on Tokio
  executor threads is exactly what the specification forbids.
- Disk I/O runs on **dedicated I/O worker threads** (`pread`, optionally io_uring or
  O_DIRECT). This is what Colibri measured. The specification itself allows dedicated
  I/O workers.

An async facade (`load_async`) can wrap the same store for non-hot-path callers such
as cache warming.

```rust
/// Contract (from Colibri's expert_store.h, made type-safe):
/// - a Lease pins the slot; eviction never takes a leased slot;
/// - Lease is !Clone and releases on Drop (no "release exactly once" footgun);
/// - concurrent acquires of the same missing expert share ONE load (in-flight dedup);
/// - prefetch is advisory: never blocks, never evicts a leased slot, may be dropped.
pub trait ExpertStore: Send + Sync {
    fn acquire(&self, id: ExpertId, prio: Priority) -> Result<Lease<'_>, StorageError>;
    /// Batch form used by MoE layers: issues all misses first, then waits, so a layer's
    /// misses load in parallel instead of one after another.
    fn acquire_many(&self, layer: LayerIdx, ids: &[ExpertIdx], out: &mut Vec<Lease<'_>>)
        -> Result<(), StorageError>;
    fn prefetch(&self, ids: &[ExpertId]) -> PrefetchReceipt;      // counts accepted / dropped
    fn cancel_prefetch(&self, layer: LayerIdx);
    fn stats(&self) -> ExpertStoreStats;                          // hits, misses, inflight-joins, bytes, …
    fn set_budget(&self, b: CacheBudget) -> Result<(), StorageError>;   // shrink evicts unleased LRU
}

pub struct Lease<'s> { slot: SlotRef<'s> /* view: gate/up/down TensorViews + dtype */ }

/// Where bytes come from: one implementation per container layout.
pub trait ExpertSource: Send + Sync {
    fn layout(&self, id: ExpertId) -> Result<ExpertLayout, StorageError>; // extents on which file(s), validated sizes
    fn read_into(&self, id: ExpertId, dst: &mut SlotBuf) -> Result<u64, StorageError>; // bytes read
}

pub trait EvictionPolicy: Send { fn on_access(&mut self, e: ExpertIdx, now: u64);
                                 fn victim(&self, candidates: &[SlotMeta]) -> Option<SlotIdx>; }
pub trait Predictor: Send { fn observe(&mut self, layer: LayerIdx, r: &RoutingResult);
                            fn predict(&mut self, next: LayerIdx, router_logits: &[f32]) -> Prediction; }
```

Internals: per-layer slot pools of fixed-size slabs, allocated once and reused (no
allocator churn, and NUMA placement possible later); an O(1) `expert → slot` index;
slot states `Free | Loading(LoadId) | Ready{leases}`; a bounded MPMC I/O queue with
priorities `Demand > Prefetch > Warm`; a `Condvar` per layer for in-flight joins (no
`sleep(1ms)` polling); `tier.h` LFRU and hysteresis ported as the default
`EvictionPolicy`.

### 4.6 Memory (`qubivex-memory`)

```rust
pub trait MemoryPlanner: Send + Sync {
    fn create_plan(&self, m: &ModelMetadata, hw: &HardwareProfile, now: &ResourceSnapshot,
                   req: &InferenceRequirements, policy: &PlacementPolicy) -> Result<MemoryPlan, PlanningError>;
    fn rebalance(&self, plan: &MemoryPlan, now: &ResourceSnapshot) -> Result<PlanDelta, PlanningError>;
    fn recommend(&self, catalog: &[ModelMetadata], hw: &HardwareProfile, now: &ResourceSnapshot)
        -> Vec<ModelRecommendation>;   // every number tagged Measured | Estimated
}
pub struct MemoryPlan {
    pub model: ModelKey,
    pub ram: RamBudget { total, dense, expert_cache, kv_room_shared, activations, reserve },
    pub vram: HashMap<DeviceId, VramBudget>,
    pub placements: Vec<TensorPlacement>,       // per role group; experts as ranges, not 1-per-expert rows
    pub expert_cache: CacheBudget,              // per-layer slot counts (non-uniform allowed)
    pub storage: StorageRequirement,
    pub reasons: Vec<PlanReason>,               // human-readable "why", as Colibri's plan has
}
/// Single source of truth for bytes actually allocated, per tier. Planner budgets
/// are targets; the ledger is reality; the pressure monitor compares the two.
pub struct MemoryLedger { /* atomics per tier & owner */ }
```

Budget formulas follow the specification (RAM and VRAM are separate and never
summed). Colibri's measured defaults are the starting values: `0.88 × MemAvailable`
after the dense load, `0.5 GB` activation reserve, KV sharing the expert room, and a
1-slot-per-layer floor.

### 4.7 Runtime and gateway

```rust
pub struct Engine { /* registry, planner, models: RwLock<HashMap<ModelKey, ModelHandle>> */ }
impl Engine {
    pub fn load(&self, path: &Path, opts: LoadOptions) -> Result<ModelKey, QubivexError>;
    pub fn unload(&self, key: ModelKey, drain: DrainPolicy) -> Result<(), QubivexError>;
    pub fn submit(&self, req: GenerateRequest) -> Result<GenerationStream, AdmissionError>;
}
pub enum ModelState { Discovered, Validating, Planning, Loading, Ready, Draining, Unloading, Failed(Reason) }
```

`GenerationStream` is a bounded channel of token events. The gateway (Tokio and
Axum) bridges it to SSE. Cancellation is a `CancellationToken` checked between layers,
as Colibri's `prompt_break.h` does.

## 5. Dependency decisions

These are proposed. Versions are pinned in Phase 1. The tensor-engine decision waits
for the benchmark spike (ADR-002).

| Area | Choice | Rationale |
|---|---|---|
| Errors | `thiserror` (libs), `anyhow` only in `apps/` | Typed domain errors |
| Serialization | `serde`, `serde_json` | Manifests, config.json |
| CLI | `clap` (derive) | |
| Logging and tracing | `tracing`, `tracing-subscriber` | Spans per request and per layer, off on the hot path by default |
| Metrics | `metrics` + `metrics-exporter-prometheus` | Cheap atomic counters |
| HTTP | `axum` + `tokio` (gateway only) | |
| CPU parallelism | Dedicated pinned thread pool. `rayon` is evaluated against it in the spike | Colibri's OpenMP team uses P-cores only, with a barrier per matmul; rayon's work-stealing may cost on short GEMVs |
| SafeTensors | `safetensors` crate (header parse only) + own `pread` reader | We need positional reads into our own slabs, not whole-file mmap |
| GGUF | Own reader (small, spec is stable). `candle`'s reader is evaluated | |
| Tokenizer | `tokenizers` (HF) + `minijinja` for chat templates | |
| Memory info | `sysinfo` + direct `/proc`/`sysctl`/Win32 for MemAvailable and cgroups | `sysinfo` alone lacks cgroup v2 and MemAvailable semantics |
| CUDA (P8) | `cudarc` + ported Colibri `.cu` kernels | Driver API binding, no heavy framework |
| Vulkan (P8+) | `ash` + ported Colibri GLSL shaders (precompiled SPIR-V) | Needs subgroup ops and cooperative matrix. `wgpu` hides them |
| Metal (later) | `objc2-metal` | |
| Benchmarks | `criterion` (micro) + own end-to-end harness | |
| Tensor engine | **Decided by spike** (ADR-002): Candle vs own kernels | |

## 6. Configuration

One typed `QubivexConfig` (TOML) with layered sources: defaults, then
`~/.config/qubivex/config.toml`, then `--config`, then `QUBIVEX_*` environment
variables, then CLI flags. Every Colibri tunable that survives (prefetch width,
confidence limit, eviction guard, direct I/O, I/O workers, …) becomes a named, typed
and documented field. `qubivex config show` prints the effective values and their
source.

## 7. Open design questions for review

1. **Synchronous lease hot path vs async `ExpertStore`** (section 4.5). This is the
   most consequential deviation.
2. **Crate count** (11 initially vs about 25).
3. **Merged factory interface** with `tensor_schema` (section 4.3).
4. **Repacked expert container.** Do we allow an optional one-time "expert-packed"
   layout (one extent per expert, as Colibri's `merged_weight`)? Original SafeTensors
   and GGUF would always remain readable directly (ADR-006).
