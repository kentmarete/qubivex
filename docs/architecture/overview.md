# Architecture overview

QUBIVEX is an inference platform for large language models, written in Rust, built
around one observation: a Mixture-of-Experts model uses only a few of its experts for
each token, so most of its weights can stay on disk until they are needed.

This page is the map. Each part links to its own page.

## Three products in one workspace

| Product | What it is | Crates |
|---|---|---|
| **Runtime** | An embeddable inference engine: load a model, generate text. No server, no CLI, no async runtime | `qubivex-runtime` and everything below it |
| **Model manager** | Find, download, verify, install, inspect and remove models | `qubivex-model`, `qubivex-model-registry`, `qubivex-model-download` |
| **Server** | An OpenAI-compatible HTTP API over the runtime | `qubivex-api` |

The `qubivex` binary (`apps/cli`) is a thin front end to all three.

The separation is enforced, not just intended: the runtime may not depend on an HTTP
server, an async runtime or the CLI, and the model manager may not depend on anything
that executes a model. `cargo xtask check` fails the build otherwise
([Dependency architecture](dependency-architecture.md)).

## The layers

```text
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ apps/cli            qubivex run · serve · bench · plan · system · models     │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ qubivex-api         HTTP gateway: auth, limits, OpenAI shapes, SSE, metrics  │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ qubivex-runtime     the engine: load, plan, generate, admit, combine passes  │
 ├───────────────┬─────────────────┬──────────────────┬─────────────────────────┤
 │ qubivex-arch  │ qubivex-memory  │ qubivex-expert-  │ qubivex-scheduler       │
 │ forward pass, │ footprint,      │ stream           │ queue, priorities       │
 │ KV, routing   │ planner, ledger,│ expert cache,    │ qubivex-kv-cache        │
 │               │ pressure        │ leases, I/O      │ prefix index            │
 │               │                 │                  │ qubivex-tokenizer       │
 ├───────────────┴─────────────────┴──────────────────┴─────────────────────────┤
 │ qubivex-compute     tensors, views, the ComputeBackend trait                 │
 │ qubivex-backend-cpu · qubivex-backend-metal        the kernels               │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ qubivex-model       containers, manifest, architecture registry, validation  │
 │ qubivex-hardware    CPU, RAM, cgroups, GPUs, storage                         │
 │ qubivex-core        ids, dtypes, shapes, descriptors, errors                 │
 └──────────────────────────────────────────────────────────────────────────────┘
        qubivex-model-registry · qubivex-model-download     (beside, not below:
        they install models and never run them)
```

Dependencies point downwards only. [Workspace structure](workspace-structure.md) lists
every crate.

## One request, end to end

A chat completion sent to `qubivex serve`:

1. **Gateway.** Authenticate, apply the rate limit and body limit, parse and validate
   the OpenAI request. Anything unsupported is refused by name.
   ([HTTP API](../serving/http-api.md))
2. **Scheduler.** Take a place in the model's bounded queue, or answer 429.
   ([Scheduler](../serving/scheduler.md))
3. **Prepare.** Render the chat template, tokenize, check the context.
4. **Admit.** Wait until a slot is free and the memory plan can hold this
   conversation's KV. ([Memory planner](../runtime/memory-planner.md))
5. **Prefill.** Run the prompt through the model in chunks, or copy a retained prefix
   and run only what is new. ([KV cache](../runtime/kv-cache.md))
6. **Each forward pass** combines the waiting steps of every running conversation.
   ([Concurrent inference](../serving/concurrent-inference.md)) In each layer:
   attention over the conversation's own KV; then the router picks experts, the expert
   store leases those in memory and reads the rest from disk, and the kernels run
   them. ([MoE routing](../runtime/moe-routing.md),
   [Expert streaming](../runtime/expert-streaming.md),
   [Tensor engine](../runtime/tensor-engine.md))
7. **Decode.** Sample a token, emit its text as an SSE event, feed it back, repeat.
   ([Streaming](../serving/streaming.md))
8. **Finish.** Free the slot and the KV reservation, retain the KV for reuse if
   configured, record metrics and one log line.

`qubivex run` is steps 3 to 7 with no gateway. An embedding application calls the same
engine function.

## Design invariants

These hold everywhere and are enforced by tests and CI.

1. **Placement decides speed, never bits.** The output is identical whatever the expert
   cache size, the device an expert ran on, the batch it ran in, the thread count, or
   whether a prefix was reused.
2. **Prediction never routes.** Prefetching may read router state and never changes it.
3. **No model names in shared code.** A model is described by a manifest. An
   architecture is a registered component. `cargo xtask check` rejects model-family
   names in `crates/` and `apps/`.
4. **Lossy is opt-in and named.** Every kernel is lossless by default, and no profile
   changes numerics.
5. **Untrusted inputs are validated, never repaired.** Model files, manifests,
   downloads and HTTP requests are checked in full and refused with a reason.
6. **Errors fail the request, not the process.**

## Where state lives

| State | Where | Lifetime |
|---|---|---|
| Model files, manifests, `registry.json` | The models directory | Until removed |
| Weights (dense), tokenizer, memory plan | The `Engine`, read-only | While the model is loaded |
| Routed experts | The expert store's cache | Until evicted |
| A conversation's KV | Its request's thread | The request |
| Retained KV | The engine's prefix cache | Until evicted, unloaded or pressure |
| Queue places, slots, reservations | Values that release on drop | The request |
| Loaded/loading/failed | The gateway's model table | The process |
| Metrics | In memory | The process |

A server keeps nothing on disk. Stopping one loses only caches.

## Platforms

| Platform | Backend | Status |
|---|---|---|
| macOS, Apple Silicon | CPU (NEON), Metal | Built, tested in CI, and the platform every benchmark comes from |
| Linux x86-64 | CPU (AVX2 and FMA when present) | Built and tested in CI. Not benchmarked |
| Linux arm64 | CPU (NEON) | Built and run in the Docker image. Not in CI |
| macOS x86-64 | CPU | Builds. Not tested in CI |
| CUDA | | Not built: needs NVIDIA hardware |
| Vulkan | Every matmul (`--backend vulkan`) | Bit-identical on MoltenVK (local); lavapipe is refused (it splits fma); slower than the CPU; not in CI (ADR-019, Proposed) |
| Windows | | Not supported |

## Further reading

| Topic | Document |
|---|---|
| The crates | [Workspace structure](workspace-structure.md) |
| Dependency rules | [Dependency architecture](dependency-architecture.md) |
| Threads, locks and ownership | [Concurrency model](concurrency-model.md) |
| The original design and its interfaces | [QUBIVEX_DESIGN.md](QUBIVEX_DESIGN.md) |
| The platform specification | [PLATFORM_SPEC.md](PLATFORM_SPEC.md) |
| Decisions | [`docs/decisions/`](../decisions) |
| Current status and measurements | [IMPLEMENTATION_STATUS.md](../../IMPLEMENTATION_STATUS.md) |
