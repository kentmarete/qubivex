# Workspace structure

QUBIVEX is one Cargo workspace: 16 library crates, one binary and one tooling crate.
This page says what each holds and where to look for what.

```text
qubivex/
├── Cargo.toml            workspace: members, shared dependency versions, lints, release profile
├── Cargo.lock            committed; CI builds with --locked
├── Dockerfile            the server image
├── apps/cli/             the `qubivex` binary
├── crates/               the 16 library crates below
├── xtask/                repository checks and the fixture generator
├── fixtures/models/      four tiny generated models, used by tests and the quick start
├── fixtures/oracle/      reference outputs from PyTorch transformers
├── spikes/               standalone experiments, outside the workspace
├── scripts/job           the job workflow helper
├── docs/                 this documentation
├── deny.toml             licence, advisory and source policy (cargo-deny)
├── clippy.toml           lint configuration
└── .github/workflows/    CI
```

## The crates

In dependency order, lowest first.

### Foundations

| Crate | Holds | Notable rules |
|---|---|---|
| `qubivex-core` | Ids (`LayerIdx`, `ExpertIdx`), `Bytes`, `DType` and GGML block sizes, `Shape`, `TensorDesc`, shared config and error types | Depends on no other workspace crate. Pure data: no I/O |
| `qubivex-hardware` | The hardware profile (CPU, core tiers, SIMD, NUMA, RAM, cgroup limits, GPUs, filesystems), live resource snapshots, the storage benchmark | No `unsafe`: system calls go through `rustix`. GPU probes run in a child process with a timeout |

### Models

| Crate | Holds | Notable rules |
|---|---|---|
| `qubivex-model` | SafeTensors and GGUF readers, the tensor index, the manifest, the architecture registry, the `transformer_moe` and `transformer_dense` configs and schemas, validation, discovery, import, checksums | Never depends on compute. Treats every file as hostile |
| `qubivex-model-registry` | `registry.json`: which models are installed, from where, and how far each install got. Atomic install and safe removal | Installs and removes models; never executes one |
| `qubivex-model-download` | The provider trait, the Hugging Face provider, file selection, resumable verified transfers | The only crate with network and TLS code. Blocking, on plain threads |
| `qubivex-tokenizer` | Tokenizers (Hugging Face `tokenizer.json`, or rebuilt from GGUF), chat templates, streaming detokenization | Templates run in a sandboxed engine with a step limit |

### Compute

| Crate | Holds | Notable rules |
|---|---|---|
| `qubivex-compute` | `Tensor`, `TensorView`, `WeightView`, the `ComputeBackend` trait, reference math | Typed storage; no byte reinterpretation |
| `qubivex-backend-cpu` | The CPU kernels: fused NEON row dots, multi-row kernels, tiled prefill kernels, the portable path and its AVX2 build | May use `unsafe` (SIMD). Every path gives the same bits |
| `qubivex-backend-metal` | The Metal device, buffers, kernels (`kernels.metal`) and the layer programs | macOS only; compiles to stubs elsewhere. May use `unsafe` |

### The engine

| Crate | Holds | Notable rules |
|---|---|---|
| `qubivex-arch` | The forward pass: embedding, RMSNorm, RoPE attention, routing, SwiGLU, the head, `KvCache`, `forward_batch` | Knows neither files nor caches: weights arrive through two traits |
| `qubivex-expert-stream` | The expert store: per-layer pools, leases, LFRU eviction, the I/O queue and threads, prefetch prediction, mirrors, checksum verification, fault injection | Synchronous. No `unsafe`. Model-checked with loom |
| `qubivex-memory` | Footprint from tensor roles, the planner, the assessment behind `analyse`, the ledger, the pressure monitor | No architecture knowledge beyond roles |
| `qubivex-kv-cache` | The prefix index: longest shared prefix, scopes, budget, LRU | No dependencies at all. Knows nothing of tensors |
| `qubivex-scheduler` | The bounded queue: priorities, fairness, aging, admission through a trait | No dependencies at all. No timers, no threads, no Tokio |
| `qubivex-runtime` | `Engine`: loading, planning at load, generation, sampling, stop strings, the combining point for shared passes, prefix reuse, the Metal forward pass | **Embeddable:** must not depend on a server, an async runtime or the CLI |

### Serving and the binary

| Crate | Holds | Notable rules |
|---|---|---|
| `qubivex-api` | The HTTP gateway: routes, authentication, rate and request limits, CORS, OpenAI request and response types, SSE, the model table, metrics, logs | The only crate that uses Tokio and Kerniq Server |
| `apps/cli` (`qubivex-cli`) | The `qubivex` binary: argument parsing and output for `run`, `serve`, `bench`, `plan`, `system`, `doctor`, `models` | Logic lives in `lib.rs`; `main.rs` is thin. Nothing depends on it |
| `xtask` | `cargo xtask check` (dependency direction, embeddable runtime, model names, fixtures) and `gen-fixtures` | Tooling only. Nothing depends on it |

## Where to find things

| Looking for | Look in |
|---|---|
| How a GGUF or SafeTensors header is parsed | `crates/qubivex-model/src/gguf.rs`, `safetensors.rs` |
| What a manifest may contain | `crates/qubivex-model/src/manifest.rs`, `docs/model-support/manifest.schema.json` |
| An architecture's config and tensor schema | `crates/qubivex-model/src/arch/` |
| The forward pass | `crates/qubivex-arch/src/lib.rs`, `ops.rs` |
| A kernel | `crates/qubivex-backend-cpu/src/neon.rs`, `tile.rs`, `formats.rs` |
| The Metal shaders | `crates/qubivex-backend-metal/src/kernels.metal` |
| Expert cache and eviction | `crates/qubivex-expert-stream/src/lib.rs`, `policy.rs`, `slab.rs` |
| The memory plan | `crates/qubivex-memory/src/plan.rs` |
| The generation loop | `crates/qubivex-runtime/src/engine.rs`, `request.rs` |
| Shared forward passes | `crates/qubivex-runtime/src/combine.rs` |
| An HTTP route | `crates/qubivex-api/src/lib.rs` |
| Request validation and OpenAI types | `crates/qubivex-api/src/openai.rs` |
| A metric | `crates/qubivex-api/src/metrics.rs` |
| A CLI flag | `apps/cli/src/lib.rs` and the command's own file |
| A download rule | `crates/qubivex-model-download/src/` |

## Tests

| Kind | Where | Run with |
|---|---|---|
| Unit tests | Beside the code | `cargo test --workspace` |
| Integration tests on the fixtures | `crates/*/tests/` | `cargo test --workspace` |
| Reference comparison with transformers | `crates/qubivex-runtime/tests/oracle.rs`, `fixtures/oracle/` | `cargo test --workspace` |
| The HTTP API, concurrency, hardening, observability | `crates/qubivex-api/tests/` | `cargo test -p qubivex-api` |
| The binary | `apps/cli/tests/` | `cargo test -p qubivex-cli` |
| Loom models | `crates/qubivex-expert-stream/tests/loom.rs`, `crates/qubivex-scheduler/tests/loom.rs` | `RUSTFLAGS="--cfg loom" cargo test --release -p <crate> --test loom` |
| Real models | Env-gated tests in `crates/qubivex-runtime/tests/` and `crates/qubivex-model/tests/` | See [Local development](../deployment/local-development.md#tests-that-need-real-models) |
| Kernel benchmarks | `crates/qubivex-backend-cpu/benches/` | `cargo bench -p qubivex-backend-cpu` |

## Workspace-wide settings

All in the root `Cargo.toml`:

- **One version of each third-party crate,** declared once under
  `[workspace.dependencies]`. Members write `{ workspace = true }`.
- **Lints.** `unsafe_code` is denied workspace-wide and allowed only in the two
  backend crates. `unwrap` is denied outside tests. Every `unsafe` block
  needs a `// SAFETY:` comment.
- **Release profile.** Fat LTO, one codegen unit, `panic = "unwind"` so the server can
  catch a panic at the request boundary.
- **Minimum Rust version:** 1.88, checked in CI.

## Why this many crates

The platform specification sketches about 25. The workspace has fewer because a crate is
created only when it has tested code and a boundary worth enforcing
([ADR-001](../decisions/ADR-001-RUST-WORKSPACE.md)). The boundaries that exist are the
ones a check enforces: models without compute, a runtime without a server, a scheduler
and a prefix index with no dependencies. Sampling lives in the runtime, quantisation in
the CPU backend and the model crate, and metrics in the gateway, until one of them
needs a boundary of its own.

Not present, and named in the specification or the original design: `qubivex-backend-cuda`,
`qubivex-backend-vulkan`, `qubivex-test-utils`, a separate `apps/server`, a
`benchmarks/` directory and `models/manifests/`.
