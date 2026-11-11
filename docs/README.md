# QUBIVEX documentation

Start with the page for what you want to do.

| I want to | Read |
|---|---|
| Try it on my machine | [Local development](deployment/local-development.md) |
| Run it as a service | [Linux deployment](deployment/linux-deployment.md), [macOS deployment](deployment/macos-deployment.md) |
| Operate it | [Production operations](deployment/production-operations.md) |
| Call it from a program | [HTTP API](serving/http-api.md), [OpenAI compatibility](serving/openai-compatibility.md) |
| Get and manage models | [Model downloads](model-management/model-downloads.md), [Model lifecycle](model-management/model-lifecycle.md) |
| Use the engine inside a Rust program | [Inference engine](runtime/inference-engine.md) |
| Understand how it works | [Architecture overview](architecture/overview.md) |
| Make it faster | [Optimisation guide](performance/optimisation-guide.md) |
| Know what it cannot do | [Not supported](#not-supported) below |

## All documents

### Architecture

| Document | Contents |
|---|---|
| [overview.md](architecture/overview.md) | The products, the layers, one request end to end, the invariants |
| [workspace-structure.md](architecture/workspace-structure.md) | Every crate and where to find things |
| [dependency-architecture.md](architecture/dependency-architecture.md) | Which crate may depend on which, third-party code, the enforced rules |
| [concurrency-model.md](architecture/concurrency-model.md) | Threads, shared state, ownership, cancellation, shutdown |
| [PLATFORM_SPEC.md](architecture/PLATFORM_SPEC.md) | The platform specification this work follows |
| [QUBIVEX_DESIGN.md](architecture/QUBIVEX_DESIGN.md) | The original design: invariants and interfaces |
| [TENSOR_ARCHITECTURE.md](architecture/TENSOR_ARCHITECTURE.md) | Tensor descriptors, views and buffer lifetime, in full |
| [MIGRATION_PLAN.md](architecture/MIGRATION_PLAN.md), [COLIBRI_ANALYSIS.md](architecture/COLIBRI_ANALYSIS.md), [IMPROVEMENTS_EVALUATION.md](architecture/IMPROVEMENTS_EVALUATION.md) | History: the phase plan and what was taken from Colibri |

### Runtime

| Document | Contents |
|---|---|
| [inference-engine.md](runtime/inference-engine.md) | Loading, the forward pass, generation, sampling, embedding the engine |
| [tensor-engine.md](runtime/tensor-engine.md) | Descriptors, dtypes, lossless kernels, determinism, CPU and Metal |
| [moe-routing.md](runtime/moe-routing.md) | How experts are chosen and combined, and what routing guarantees |
| [expert-streaming.md](runtime/expert-streaming.md) | The expert cache, leases, reads, and failure behaviour |
| [memory-planner.md](runtime/memory-planner.md) | Budgets, the shared room, admission, the ledger, pressure |
| [kv-cache.md](runtime/kv-cache.md) | Per-conversation KV and prefix reuse |

### Model management

| Document | Contents |
|---|---|
| [model-downloads.md](model-management/model-downloads.md) | `models pull`: what it fetches and guarantees |
| [model-manifest.md](model-management/model-manifest.md) | `manifest.json`: fields, trust order, versions |
| [model-registry.md](model-management/model-registry.md) | The models directory, `registry.json`, the architecture registry |
| [model-lifecycle.md](model-management/model-lifecycle.md) | Installing, checking, loading, updating and removing |
| [manifest.schema.json](model-support/manifest.schema.json) | The manifest's JSON Schema |

### Serving

| Document | Contents |
|---|---|
| [http-api.md](serving/http-api.md) | Every endpoint, field, error and setting, with real payloads |
| [openai-compatibility.md](serving/openai-compatibility.md) | What an OpenAI client can use, and what is refused |
| [streaming.md](serving/streaming.md) | Server-sent events, errors in a stream, cancelling, time limits |
| [scheduler.md](serving/scheduler.md) | The queue, priorities, fairness, admission, deadlines |
| [concurrent-inference.md](serving/concurrent-inference.md) | Shared forward passes, isolation, measured throughput |

### Performance

| Document | Contents |
|---|---|
| [BENCHMARKING.md](performance/BENCHMARKING.md) | How to measure, the record format, the baselines |
| [memory-profiling.md](performance/memory-profiling.md) | The plan, the ledger and resident size, and why they differ |
| [optimisation-guide.md](performance/optimisation-guide.md) | Tuning a deployment; rules and history for making the code faster |
| [COLIBRI-OLMOE-BENCHMARK.md](performance/COLIBRI-OLMOE-BENCHMARK.md), [ADR-002-SPIKE.md](performance/ADR-002-SPIKE.md) | Recorded comparisons |

The specification names this group's first file `benchmarking.md`. It is
`BENCHMARKING.md` here: the file predates the specification, the CLI's help text and
several ADRs cite it by that name, and the two names are the same file on a
case-insensitive filesystem.

### Deployment

| Document | Contents |
|---|---|
| [local-development.md](deployment/local-development.md) | Build, try, test |
| [linux-deployment.md](deployment/linux-deployment.md) | systemd, Docker, Compose, the facts for Kubernetes |
| [macos-deployment.md](deployment/macos-deployment.md) | CPU or Metal, launchd |
| [production-operations.md](deployment/production-operations.md) | Sizing, monitoring, runbooks, upgrades |

### Decisions and status

| Document | Contents |
|---|---|
| [`decisions/`](decisions) | The ADRs: why each part is the way it is, with measurements |
| [ARCHITECTURE_DECISIONS.md](engineering/ARCHITECTURE_DECISIONS.md), [PLATFORM_AUDIT.md](engineering/PLATFORM_AUDIT.md), [BASELINE_AUDIT.md](engineering/BASELINE_AUDIT.md) | Reviews and the plan |
| [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) | What is done, with test results and numbers |
| [RUST_GUIDELINES.md](development/RUST_GUIDELINES.md), [JOB_WORKFLOW.md](development/JOB_WORKFLOW.md) | For contributors |

## Not supported

Every page ends with its own list. This is all of them in one place, so that nobody has
to discover a gap by hitting it.

**Hardware and platforms**

- CUDA. Vulkan runs (ADR-019) with the CPU's bits but slower than the CPU;
  Metal is the fast GPU backend, on macOS.
- Windows.
- Multiple GPUs, and splitting a model across devices.
- Hand-written x86-64 kernels. x86-64 runs the portable kernels built for AVX2 and FMA,
  and has not been benchmarked.

**Models**

- Architectures other than `transformer_moe` and `transformer_dense`.
- Shared-expert and partly dense MoE models, sigmoid routing, attention or FFN biases,
  non-SiLU activations, RoPE scaling, sliding-window attention.
- I-quant GGUF dtypes, split GGUF files, big-endian GGUF.
- Multimodal models, embedding models, LoRA adapters.
- Model sources other than Hugging Face.

**The API**

- Tool calls, JSON mode and JSON schema, logprobs, logit bias, penalties, `n > 1`.
- Embeddings, the Responses, Assistants and Batch APIs, images, audio, files.
- A separate reasoning field.
- TLS, token quotas per key, and changing keys without a restart.
- Loading a model on first request, and unloading idle ones.

**Serving**

- Capacity shared or arbitrated between loaded models.
- Preemption, and cancellation inside a forward pass.
- Settings changed without a restart.
- Tracing (OpenTelemetry). There are metrics and logs.

**Runtime**

- Lossy fast paths of any kind.
- Paged or on-disk KV, and a sliding context.
- `io_uring`, a packed expert file, remote storage as an expert source.
- A VRAM budget separate from RAM.

**Packaging**

- A crates.io release, a stable Rust API, a C API or Python bindings.
- Published binaries, packages or container images. Build from source.
