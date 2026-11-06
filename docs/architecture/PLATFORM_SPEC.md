# Qubivex: Complete Implementation Specification
## Standalone Hardware-Aware MoE Tensor and LLM Inference Platform

**Repository:** https://github.com/kentmarete/qubivex  
**Primary language:** Rust  
**Execution strategy:** Incremental, phased implementation  
**Initial model scope:** Mixture of Experts (MoE), with architecture extensible to dense transformer models  
**Deployment targets:** macOS Apple Silicon, Linux x86_64, Linux ARM64  
**Primary objective:** Build an independent model management, tensor execution and inference serving platform without depending on Ollama, llama.cpp or another external inference runtime.

---

# 1. Project Mission

You are working on Qubivex, an existing Rust-based hardware-aware Mixture of Experts tensor and inference engine.

Your responsibility is to evolve the existing project into a complete, standalone LLM inference platform.

Qubivex must independently support:

1. Downloading and managing language models.
2. Loading GGUF and SafeTensors models.
3. Performing hardware-aware model placement and memory planning.
4. Executing transformer inference.
5. Dynamically routing tokens to MoE experts.
6. Streaming experts between memory and storage.
7. Supporting concurrent inference requests.
8. Exposing an HTTP API for external applications.
9. Streaming generated tokens to clients.
10. Operating as a standalone local or remote inference server.

Qubivex must not require Ollama to download, manage, load or execute models.

Ollama may be used as an external benchmarking reference, but never as a runtime dependency.

The finished platform must have three independently usable products:

### Product A: Qubivex Runtime

The core Rust inference engine.

Responsibilities:

- Tensor operations.
- Model loading.
- Tokenization integration.
- Transformer execution.
- MoE routing.
- Expert streaming.
- KV cache management.
- Dynamic hardware-aware memory planning.
- CPU and GPU execution backends.
- Sampling and token generation.
- Concurrent sequence execution.

### Product B: Qubivex Model Manager

The model lifecycle and distribution component.

Responsibilities:

- Model downloads.
- Hugging Face integration.
- GGUF and SafeTensors support.
- Model metadata.
- Model manifests.
- Tokenizer and chat template management.
- Model validation.
- Local model registry.
- Model installation and removal.
- Model compatibility analysis.
- Download integrity and resumability.

### Product C: Qubivex Server

The HTTP inference and serving component.

Responsibilities:

- OpenAI-compatible HTTP APIs.
- Chat completions.
- Text completions.
- SSE token streaming.
- Model management endpoints.
- Request scheduling.
- Concurrent inference.
- Batching.
- Request cancellation.
- Memory admission control.
- Authentication and rate limiting.
- Observability and metrics.

These products must be independently usable but share common libraries and interfaces.

---

# 2. First Requirement: Audit the Existing Repository

Before writing code, inspect the entire existing repository.

Repository:

https://github.com/kentmarete/qubivex

Do not start by creating a new project or replacing existing modules.

You must understand what already exists.

Inspect:

- Cargo workspace structure.
- Existing crates and module boundaries.
- Tensor implementations.
- GGUF parser.
- SafeTensors parser.
- Model manifest implementation.
- Tokenizer integration.
- Transformer execution.
- MoE implementation.
- Expert routing.
- Expert cache.
- Expert streaming.
- CPU kernels.
- Metal implementation.
- Quantisation support.
- Memory management.
- Existing benchmarks.
- Tests and fixtures.
- Existing documentation.
- Current roadmap.

Produce an implementation audit containing:

1. Existing functionality.
2. Functionality partially implemented.
3. Functionality missing.
4. Technical debt.
5. Performance bottlenecks.
6. Architectural changes required.
7. Dependencies that need upgrading or introduction.

Clearly distinguish production-ready functionality from prototypes and experimental implementations.

Do not claim a feature is implemented simply because its module or interface exists.

### Preservation requirements

- Preserve working implementations.
- Avoid unnecessary rewrites.
- Reuse existing tensor and model loading code.
- Maintain backward compatibility where practical.
- Introduce abstractions around existing implementations rather than replacing them unnecessarily.
- Keep existing tests passing.
- Avoid duplicating model parsers, tensor operations or memory management logic.

After the audit, establish a phased implementation plan with dependencies between phases.

Begin implementation immediately after the audit. Do not stop after producing documentation.

---

# 3. Target Architecture

Use a Rust workspace architecture.

The proposed structure is a target, not a mandate to rename or relocate existing crates without justification.

```text
qubivex/
│
├── apps/
│   ├── cli/
│   │
│   ├── server/
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── config.rs
│   │   │   ├── state.rs
│   │   │   ├── api/
│   │   │   ├── middleware/
│   │   │   ├── services/
│   │   │   ├── scheduler/
│   │   │   ├── streaming/
│   │   │   └── observability/
│   │   └── Cargo.toml
│   │
│   └── model-manager/
│
├── crates/
│   ├── qubivex-core/
│   ├── qubivex-tensor/
│   ├── qubivex-runtime/
│   ├── qubivex-model/
│   ├── qubivex-model-registry/
│   ├── qubivex-model-download/
│   ├── qubivex-tokenizer/
│   ├── qubivex-memory/
│   ├── qubivex-hardware/
│   ├── qubivex-compute/
│   ├── qubivex-moe/
│   ├── qubivex-kv-cache/
│   ├── qubivex-scheduler/
│   └── qubivex-observability/
│
├── examples/
│   ├── basic-inference/
│   ├── streaming-inference/
│   ├── concurrent-inference/
│   └── embedded-runtime/
│
├── benchmarks/
│
├── tests/
│
└── docs/
    ├── architecture/
    ├── runtime/
    ├── model-management/
    ├── serving/
    ├── performance/
    └── deployment/
```

Adjust this structure to fit the existing workspace.

## Architectural dependency rules

The architecture must maintain strict separation.

```text
             Qubivex Applications
               /      |       \
              /       |        \
           CLI      Manager    Server
                      |          |
                      |          |
              Model Management   |
                      |          |
                      +----+-----+
                           |
                    Qubivex Runtime
                           |
                Tensor and Compute Engine
                           |
              Hardware and Memory Management
```

The runtime must never depend on the HTTP server.

The model manager must not implement inference.

The server must not implement tensor execution.

The runtime must be usable as an embedded Rust library without starting an HTTP server.

The model manager must be usable from the CLI or another application without running an inference server.

The server must be able to communicate with the runtime through stable, documented interfaces.

Avoid circular dependencies.

---

# 4. Product A: Qubivex Runtime

This is the central product.

Its purpose is to execute language models efficiently across available hardware.

## 4.1 Runtime interfaces

Introduce stable abstractions for model loading and inference.

Illustrative interfaces:

```rust
pub trait InferenceEngine: Send + Sync {
    async fn load_model(
        &self,
        config: ModelLoadConfig,
    ) -> Result<ModelHandle>;

    async fn unload_model(
        &self,
        model_id: &ModelId,
    ) -> Result<()>;

    async fn generate(
        &self,
        request: GenerationRequest,
    ) -> Result<GenerationResult>;

    async fn generate_stream(
        &self,
        request: GenerationRequest,
    ) -> Result<GenerationStream>;
}
```

Adapt the design to the actual execution architecture.

Do not introduce asynchronous traits or boxed futures unnecessarily if they conflict with the compute execution model.

Define appropriate interfaces for:

- Model executor.
- Tensor backend.
- Model loader.
- Tokenizer.
- MoE router.
- Expert manager.
- KV cache.
- Memory planner.
- Hardware detector.
- Generation scheduler.

All interfaces must have clear ownership and lifecycle rules.

## 4.2 Tensor engine

Continue developing the native tensor engine.

Support:

- Tensor creation and allocation.
- Tensor views and slicing.
- Matrix multiplication.
- Vector operations.
- Reductions.
- Broadcasting.
- Normalisation.
- Softmax.
- Attention operations.
- RoPE.
- Activation functions.
- Quantised tensor operations.
- Efficient memory reuse.

Support the required data types according to model compatibility:

- FP32.
- FP16.
- BF16.
- INT8.
- INT4.
- Existing GGUF quantisation formats, including Q4_K and other formats already supported.

Do not introduce lossy conversions into operations advertised as lossless.

For every kernel, provide numerical correctness tests and benchmarks.

## 4.3 Transformer execution

Support the full inference pipeline.

```text
Input Text
    |
Tokenizer
    |
Token IDs
    |
Embedding
    |
Transformer Layers
    |
Attention
    |
MoE or Dense Feed Forward
    |
Output Projection
    |
Logits
    |
Sampling
    |
Generated Token
    |
Detokenization
```

Ensure model-specific architectural configurations are respected.

Do not assume all MoE architectures have identical routing, expert layouts or attention implementations.

Initially prioritise the existing MoE model scope.

Keep dense transformer support architecturally possible without forcing an incomplete generalisation.

## 4.4 MoE routing

Implement and improve:

- Top-K expert selection.
- Router logits.
- Expert probability calculations.
- Expert weighting.
- Token-to-expert dispatch.
- Expert output aggregation.
- Routing statistics.
- Expert cache awareness.
- Expert prefetch hints.

Respect model-specific routing rules, including expert selection, normalization, scaling and shared experts where applicable.

Do not silently substitute a different routing algorithm.

## 4.5 Expert streaming

Expert streaming is one of Qubivex's main differentiators.

Implement an efficient expert lifecycle:

```text
Expert Requested
      |
Check Expert Cache
      |
  +---+---+
  |       |
Found    Missing
  |       |
Use     Check Memory
Cache      |
       Load Expert
           |
       Validate Data
           |
       Move to Compute
           |
       Execute
           |
       Release or Cache
```

Support:

- Disk-backed expert storage.
- Memory-mapped files where appropriate.
- Asynchronous expert loading.
- Bounded prefetch queues.
- Expert cache eviction.
- LRU or improved cost-aware eviction.
- Expert reference counting or leases.
- Concurrent expert access.
- In-flight load deduplication.
- Cancellation-safe loading.
- Cache metrics.
- Storage read metrics.

Avoid loading duplicate copies of the same expert when multiple requests need it simultaneously.

An expert currently executing must not be evicted.

Ensure memory reservations occur before loading an expert.

Prevent uncontrolled prefetching from exhausting available RAM.

## 4.6 Hardware-aware memory planner

Build a dynamic memory planner.

The planner must inspect:

- Available system RAM.
- Available GPU memory.
- CPU architecture.
- GPU architecture.
- Hardware acceleration capabilities.
- Model weight size.
- Tensor allocation requirements.
- Expert sizes.
- KV cache requirements.
- Context length.
- Concurrent inference demand.
- Backend memory constraints.

The planner must estimate:

```text
Total Memory Requirement =
    Model Weights
  + Resident Experts
  + Expert Loading Buffers
  + KV Cache
  + Temporary Tensor Buffers
  + Runtime Overhead
  + Safety Reserve
```

Support execution strategies:

1. Fully memory-resident model.
2. Partial expert residency.
3. Disk-backed expert streaming.
4. CPU-only inference.
5. Hybrid CPU/GPU execution.
6. GPU-resident weights where feasible.

The planner must make placement decisions dynamically.

Do not assume that available physical memory equals memory available to Qubivex.

Account for memory already allocated by the operating system and other processes where the platform exposes reliable measurements.

Provide explainable memory planning output.

Example:

```text
Model: OLMoE-1B-7B
Backend: Metal

System RAM:          36 GB
Available RAM:       24 GB
Estimated Weights:   4.8 GB
KV Cache:            1.2 GB
Expert Cache:        2.0 GB
Runtime Buffers:     0.8 GB

Execution Strategy: Hybrid
Expert Streaming:    Enabled
GPU Acceleration:    Enabled
Estimated Budget:    8.8 GB
```

The figures are illustrative.

## 4.7 KV cache management

Implement a dedicated KV cache subsystem.

Requirements:

- Per-sequence KV cache.
- Efficient allocation and deallocation.
- Context reuse where valid.
- Prefix caching where safe.
- Bounded memory consumption.
- Cache eviction.
- Concurrent sequence isolation.
- Cancellation cleanup.
- Context length enforcement.

Design interfaces that allow future paged KV caching.

Do not allow one request to corrupt or overwrite another request's attention state.

Support different cache strategies based on hardware availability.

## 4.8 Hardware backends

Keep compute backends modular.

Prioritise:

1. CPU.
2. Apple Metal.
3. NVIDIA CUDA.
4. Vulkan where practical.

The CPU implementation must remain a reliable fallback.

Use hardware capability detection rather than hardcoded assumptions.

Provide backend selection and fallback diagnostics.

Do not silently move to a slower backend when the user explicitly requests an unsupported backend.

---

# 5. Product B: Qubivex Model Manager

This product manages the entire model lifecycle.

It must work independently of the inference server.

## 5.1 Model downloads

Initially support Hugging Face model repositories.

Implement:

- Repository identification.
- Model metadata retrieval.
- File listing.
- Model revision selection.
- File selection.
- Resumable downloads.
- Concurrent file downloads with bounded concurrency.
- Download progress.
- Cancellation.
- Retry with exponential backoff.
- HTTP range requests where supported.
- Authentication for private repositories.
- Checksum verification when available.
- Atomic installation.

Avoid downloading unnecessary files.

Support GGUF and SafeTensors initially.

Allow future model sources to be introduced through provider interfaces.

## 5.2 CLI commands

Implement commands similar to:

```bash
qubivex models search <query>

qubivex models pull <repository>

qubivex models pull <repository> \
    --file model-Q4_K_M.gguf

qubivex models list

qubivex models inspect <model-id>

qubivex models analyse <model-id>

qubivex models validate <model-id>

qubivex models remove <model-id>
```

Also support:

```bash
qubivex system
```

This should display detected hardware, available memory and supported compute backends.

And:

```bash
qubivex models analyse <model-id>
```

This should estimate memory requirements and identify whether the model can run using the available hardware.

## 5.3 Model manifest

Develop a versioned model manifest.

It must describe:

- Model ID.
- Display name.
- Architecture.
- Model family.
- Model revision.
- Source repository.
- Weight format.
- Quantisation.
- Tensor dimensions.
- Number of transformer layers.
- Attention configuration.
- Hidden dimensions.
- Vocabulary size.
- Context length.
- Number of experts.
- Number of active experts.
- Expert dimensions.
- Shared expert configuration.
- Tokenizer metadata.
- Chat template.
- Weight file locations.
- Tensor-to-file mappings.
- Checksums.
- Supported execution backends.

The manifest must support future architecture extensions.

Use explicit schema versioning and migration strategies.

Do not make the runtime depend on Hugging Face-specific metadata.

Normalise external repository metadata into Qubivex's own internal manifest.

## 5.4 Model registry

Implement a local model registry.

Responsibilities:

- Register installed models.
- Discover existing model installations.
- Track model versions.
- Track file locations.
- Validate model integrity.
- Prevent duplicate installations.
- Support multiple quantisations.
- Track loaded and unloaded state.
- Provide model metadata to the server.
- Support safe model removal.

Use atomic registry updates.

The registry must remain consistent if a download or installation fails midway.

## 5.5 Model lifecycle

Support:

```text
Not Installed
     |
Downloading
     |
Validating
     |
Installed
     |
Loading
     |
Loaded
     |
Unloading
     |
Installed
```

Represent failed downloads, failed validation and failed model loading explicitly.

Avoid reporting a model as loaded until the runtime confirms successful initialisation.

---

# 6. Product C: Qubivex Server

Build an independent HTTP inference server.

Use Axum and Tokio unless the existing project provides a compelling reason to choose another stack.

The server must communicate with the runtime through stable interfaces.

## 6.1 OpenAI-compatible API

Implement:

```text
GET  /health
GET  /v1/models
GET  /v1/models/{id}

POST /v1/chat/completions
POST /v1/completions

GET  /metrics
```

Also introduce model lifecycle endpoints:

```text
POST /v1/models/load
POST /v1/models/unload
```

These management endpoints should use a clearly documented Qubivex-specific contract.

The chat completion API should support:

```json
{
  "model": "olmoe-1b-7b",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Explain mixture of experts."
    }
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 500,
  "stream": false
}
```

Support:

- System messages.
- User messages.
- Assistant messages.
- Tool message compatibility where implemented.
- Model-specific chat templates.
- Stop sequences.
- Temperature.
- Top-p.
- Top-k where supported.
- Maximum generation tokens.
- Streaming.
- Usage statistics.
- Structured error responses.

Maintain compatibility with common OpenAI client libraries.

Document any unsupported parameters rather than silently ignoring them.

## 6.2 Streaming

Implement SSE streaming.

For example:

```text
data: {"choices":[{"delta":{"role":"assistant"}}]}

data: {"choices":[{"delta":{"content":"Mixture"}}]}

data: {"choices":[{"delta":{"content":" of"}}]}

data: {"choices":[{"delta":{"content":" Experts"}}]}

data: [DONE]
```

Requirements:

- Send tokens as soon as they become available.
- Avoid buffering the entire response.
- Handle client disconnections.
- Cancel generation when appropriate.
- Propagate runtime errors.
- Release sequence resources after completion.
- Support usage reporting.
- Avoid blocking asynchronous HTTP workers.

Use bounded channels between the generation task and the HTTP streaming response.

Implement backpressure.

A slow client must not cause unlimited token buffering.

---

# 7. Concurrent Inference: First-Class Requirement

Concurrency is one of the most important requirements of this implementation.

Qubivex must support multiple clients using the same loaded model without loading a separate copy of the model for every request.

Design the serving system around a shared model runtime and independently managed inference sequences.

## 7.1 Request lifecycle

Each request should have:

- Unique request ID.
- Model ID.
- Input tokens.
- Generation parameters.
- Sequence ID.
- KV cache allocation.
- Cancellation token.
- Deadline or timeout.
- Resource reservation.
- Completion state.

Lifecycle:

```text
HTTP Request
     |
Validation
     |
Admission Control
     |
Request Queue
     |
Scheduler
     |
Memory Reservation
     |
KV Cache Allocation
     |
Inference Execution
     |
Token Streaming
     |
Completion or Cancellation
     |
Resource Cleanup
```

Every transition must have explicit failure handling.

## 7.2 Shared model instances

Implement a model lifecycle manager.

Requirements:

- One shared loaded model instance per compatible model configuration.
- Multiple requests sharing model weights.
- Per-request generation state.
- Per-request KV cache.
- Shared read-only model tensors.
- Safe concurrent expert access.
- Model loading deduplication.
- Safe model unloading.
- Reference tracking for active requests.

Prevent model unloading while active requests are using it.

Model loading must be idempotent or deduplicated.

Concurrent requests for the same model should wait for or reuse the same loading operation.

## 7.3 Request scheduler

Build a scheduler independent of the HTTP handlers.

It should support:

- Bounded request queues.
- Configurable concurrency.
- Request priorities.
- Fairness.
- Queue timeouts.
- Request cancellation.
- Model affinity.
- Memory admission control.
- Execution deadlines.

Avoid a single global lock around inference.

Do not serialize all requests unnecessarily.

Use fine-grained synchronisation and immutable shared model data wherever possible.

## 7.4 Continuous batching

Design the scheduler to support continuous batching.

Implement this in stages.

Stage 1:

- Multiple concurrent requests.
- Independent generation sequences.
- Shared model weights.
- Controlled compute execution.

Stage 2:

- Dynamic batching.
- Compatible sequence grouping.
- Shared model execution where supported.
- Per-sequence stopping conditions.

Stage 3:

- Continuous batching.
- Dynamic sequence admission.
- Removal of completed sequences.
- KV cache-aware scheduling.
- Variable sequence lengths.

Batching must not change the semantics of individual requests.

Do not claim continuous batching is implemented until multiple sequences actually share an execution batch.

## 7.5 Memory admission control

Before accepting a generation request, estimate its resource requirements.

Consider:

- Prompt token count.
- Maximum generation tokens.
- Context length.
- KV cache requirements.
- Current active sequences.
- Resident experts.
- Temporary buffers.
- Available memory.
- Expected expert streaming pressure.

Support configurable policies:

- Queue when memory is temporarily unavailable.
- Reject when the request exceeds hard resource limits.
- Apply per-request context limits.
- Apply per-model concurrency limits.
- Release resources promptly after completion.

Never permit uncontrolled allocations that can cause the process to exhaust system memory.

## 7.6 Concurrency safety

Pay special attention to:

- Shared expert cache.
- Concurrent expert loading.
- Expert eviction during execution.
- KV cache ownership.
- Cancellation during disk reads.
- Cancellation during tensor execution.
- Model unloading.
- Request timeout.
- HTTP client disconnection.
- Server shutdown.

Implement expert leases or equivalent lifetime protection.

Use RAII and explicit ownership where possible.

Avoid holding mutexes across await points.

Do not use unbounded Tokio channels.

Avoid unnecessary Arc<Mutex<...>> around the entire runtime.

Prefer per-resource synchronisation.

Document the concurrency model and ownership guarantees.

---

# 8. Performance and Hardware-Aware Optimisation

The main differentiator of Qubivex is hardware-aware execution.

Do not optimise for API throughput at the expense of the actual tensor execution engine.

Track the following:

### Inference performance

- Time to first token.
- Prompt processing tokens/second.
- Generation tokens/second.
- Total request latency.
- Inter-token latency.
- Model loading time.

### Memory performance

- Resident model memory.
- Peak RAM.
- Peak VRAM.
- KV cache utilisation.
- Expert cache hit rate.
- Expert cache eviction count.
- Disk read throughput.
- Expert load latency.
- Memory fragmentation where measurable.

### Concurrent serving

- Active requests.
- Queued requests.
- Rejected requests.
- Requests per second.
- Aggregate tokens per second.
- Per-request latency.
- Scheduler wait time.
- Cancellation count.

Use Prometheus-compatible metrics.

Provide structured logs with request IDs, model IDs and sequence IDs.

Do not log private prompts or generated content by default.

---

# 9. CLI Experience

Provide a coherent command-line interface.

Examples:

```bash
# Inspect the machine
qubivex system

# Discover models
qubivex models search "OLMoE"

# Download a model
qubivex models pull <repository>

# Inspect model
qubivex models inspect <model-id>

# Analyse hardware compatibility
qubivex models analyse <model-id>

# Run an interactive terminal chat
qubivex chat --model <model-id>

# Run one inference request
qubivex chat \
  --model <model-id> \
  --prompt "Explain MoE inference"

# Start the server
qubivex serve \
  --model <model-id> \
  --host 127.0.0.1 \
  --port 8080

# Show running server information
qubivex server status

# List available models
qubivex models list
```

The CLI must be a consumer of the runtime and model manager libraries.

Do not implement a second inference pipeline inside the CLI.

---

# 10. Configuration

Provide configuration through:

1. CLI arguments.
2. Environment variables.
3. Configuration files.

Define clear precedence.

Example:

```toml
[server]
host = "127.0.0.1"
port = 8080
max_concurrent_requests = 4
max_queue_size = 32
request_timeout_seconds = 300

[models]
directory = "~/.qubivex/models"
auto_load = true

[runtime]
backend = "auto"
memory_reserve_percent = 15
expert_streaming = true
expert_cache_size_mb = 2048

[scheduler]
continuous_batching = false
max_batch_size = 8
max_context_length = 8192

[observability]
metrics_enabled = true
structured_logging = true
```

All limits must be configurable.

Do not hardcode hardware-specific assumptions.

Validate configuration during startup and provide useful error messages.

---

# 11. Security and Production Readiness

Implement:

- Optional API key authentication.
- Request size limits.
- Maximum prompt length.
- Maximum generation tokens.
- Per-client rate limiting.
- Model download authentication.
- Secure credential storage.
- Path traversal protection.
- Safe model archive handling.
- Download checksum verification.
- CORS configuration.
- Graceful shutdown.

Do not expose an unauthenticated inference server on all network interfaces by default.

Localhost should be the default binding.

Do not expose filesystem paths or internal errors unnecessarily through HTTP responses.

---

# 12. Testing Strategy

Testing is mandatory throughout implementation.

## 12.1 Unit tests

Test:

- Tensor operations.
- Model manifests.
- GGUF parsing.
- SafeTensors parsing.
- Tokenizer behaviour.
- Chat templates.
- MoE routing.
- Expert cache.
- Memory planner.
- KV cache.
- Sampling.
- Request scheduler.

## 12.2 Integration tests

Test the complete flow:

```text
Download Model
     |
Validate Model
     |
Register Model
     |
Load Model
     |
Submit Chat Request
     |
Execute Inference
     |
Generate Tokens
     |
Return Response
```

Test both standard and streaming inference.

## 12.3 Concurrency tests

These are essential.

Test:

1. Ten simultaneous requests using one model.
2. Fifty simultaneous requests with bounded concurrency.
3. Multiple clients requesting the same expert.
4. Concurrent expert loading.
5. Expert eviction while other experts are executing.
6. Model loading deduplication.
7. Cancellation during inference.
8. Client disconnect during SSE streaming.
9. Model unloading while requests are active.
10. Memory exhaustion and admission control.
11. Server shutdown with active requests.
12. Concurrent requests with different context lengths.
13. Multiple models with shared hardware resources.

Verify:

- No data races.
- No KV cache contamination.
- No duplicate model instances.
- No expert use-after-free.
- No resource leaks.
- No unbounded queue growth.
- No deadlocks.
- No incorrect response interleaving.

Use stress tests and sanitising tools where supported.

## 12.4 Performance benchmarks

Create reproducible benchmarks for:

- CPU tensor operations.
- GPU tensor operations.
- Model loading.
- Expert loading.
- Expert cache hit and miss.
- Memory planner overhead.
- Single-request generation.
- Concurrent generation.
- Continuous batching.

Compare equivalent configurations and workloads.

Record hardware, model, quantisation, context length and generation settings.

Never fabricate benchmark results.

---

# 13. Phased Implementation Sequence

Follow this sequence. Do not jump ahead and build an incomplete HTTP server before establishing the runtime interfaces.

## Phase 0: Repository audit and architecture

Deliverables:

- Existing implementation audit.
- Updated architecture documentation.
- Dependency graph.
- Gap analysis.
- Detailed implementation plan.
- Baseline test and benchmark results.

Acceptance:

Existing working code remains intact and the project builds.

## Phase 1: Runtime foundations

Deliverables:

- Stable inference interfaces.
- Model lifecycle abstraction.
- Runtime configuration.
- Tensor execution interfaces.
- Generation request and response contracts.
- Model handle management.

Acceptance:

Existing models can execute through the new runtime abstraction.

## Phase 2: Model manager

Deliverables:

- Hugging Face integration.
- Resumable downloads.
- GGUF and SafeTensors installation.
- Manifest validation.
- Model registry.
- Model discovery.
- CLI model commands.

Acceptance:

Download, validate, register, inspect and remove a model without Ollama.

## Phase 3: Hardware-aware memory planner

Deliverables:

- Hardware discovery.
- Memory estimation.
- Memory reservations.
- Model placement strategies.
- Expert residency planning.
- Execution compatibility analysis.

Acceptance:

Qubivex can produce a hardware-aware execution plan before loading a model.

## Phase 4: Runtime and expert streaming

Deliverables:

- Improved expert lifecycle.
- Asynchronous expert loading.
- Expert leases.
- Cache coordination.
- In-flight loading deduplication.
- KV cache management.
- Cancellation-safe execution.

Acceptance:

Concurrent sequences can safely share model weights and expert resources.

## Phase 5: HTTP server foundations

Deliverables:

- Axum server.
- Shared application state.
- Model discovery endpoints.
- Chat completion endpoint.
- Standard JSON responses.
- Structured errors.
- Health endpoint.

Acceptance:

An external OpenAI-compatible client can execute a complete inference request.

## Phase 6: Streaming inference

Deliverables:

- Token streaming.
- SSE.
- Backpressure.
- Cancellation.
- Client disconnect handling.
- Correct stream termination.

Acceptance:

Tokens stream incrementally without buffering the full response.

## Phase 7: Concurrent inference scheduler

Deliverables:

- Bounded queues.
- Admission control.
- Request prioritisation.
- Concurrency limits.
- Shared model execution.
- Per-request KV cache.
- Resource lifecycle management.

Acceptance:

Multiple simultaneous requests execute correctly without duplicating model weights or corrupting sequence state.

## Phase 8: Continuous batching

Deliverables:

- Dynamic batching.
- Sequence grouping.
- KV cache-aware scheduling.
- Sequence admission and removal.
- Per-sequence completion handling.

Acceptance:

Multiple sequences share actual model execution batches while preserving independent generation state.

## Phase 9: Hardware acceleration

Deliverables:

- CPU optimisations.
- Metal backend.
- CUDA backend.
- Vulkan backend where practical.
- Hardware-aware backend selection.
- Hybrid execution.

Acceptance:

All supported backends pass correctness tests and provide reproducible benchmarks.

## Phase 10: Production readiness

Deliverables:

- Authentication.
- Rate limiting.
- Resource limits.
- Metrics.
- Structured logging.
- Graceful shutdown.
- Deployment documentation.
- Docker support where appropriate.
- Operational runbooks.

Acceptance:

Qubivex can run as a managed inference service and handle multiple clients reliably.

---

# 14. Required Documentation

Update the README and create detailed documentation.

Required documents:

```text
docs/
├── architecture/
│   ├── overview.md
│   ├── workspace-structure.md
│   ├── dependency-architecture.md
│   └── concurrency-model.md
│
├── runtime/
│   ├── inference-engine.md
│   ├── tensor-engine.md
│   ├── moe-routing.md
│   ├── expert-streaming.md
│   ├── memory-planner.md
│   └── kv-cache.md
│
├── model-management/
│   ├── model-downloads.md
│   ├── model-manifest.md
│   ├── model-registry.md
│   └── model-lifecycle.md
│
├── serving/
│   ├── http-api.md
│   ├── openai-compatibility.md
│   ├── streaming.md
│   ├── scheduler.md
│   └── concurrent-inference.md
│
├── performance/
│   ├── benchmarking.md
│   ├── memory-profiling.md
│   └── optimisation-guide.md
│
└── deployment/
    ├── local-development.md
    ├── linux-deployment.md
    ├── macos-deployment.md
    └── production-operations.md
```

Include working examples and actual API payloads.

Document unsupported features explicitly.

---

# 15. Engineering Rules

These rules are mandatory.

### Code quality

- Use idiomatic Rust.
- Minimise unnecessary dependencies.
- Prefer well-defined ownership.
- Avoid excessive Arc<Mutex<T>>.
- Avoid global mutable state.
- Use typed errors.
- Document public interfaces.
- Keep modules cohesive.
- Avoid oversized source files.
- Use traits where they provide real architectural value.
- Do not introduce unnecessary abstraction layers.

### Performance

- Avoid unnecessary tensor copies.
- Avoid blocking asynchronous executor threads.
- Use bounded queues.
- Reuse allocations where safe.
- Avoid loading duplicate experts.
- Keep inference hot paths efficient.
- Profile before optimising.
- Never trade correctness for benchmark improvements.

### Reliability

- Every allocation must have an ownership and release strategy.
- Every request must have a terminal state.
- Every model must have a well-defined lifecycle.
- Every cancellation path must release resources.
- Every background task must have shutdown behaviour.
- Every cache must have bounded capacity.
- Every download must support failure recovery.

### Architectural integrity

- Runtime must not depend on server.
- Model manager must not execute inference.
- Server must not implement tensor operations.
- All three products must be independently usable.
- Shared contracts must be versioned and documented.
- Existing implementations must be preserved unless a measurable technical reason justifies replacing them.

---

# 16. Definition of Done

The project is not complete simply because it compiles.

The final acceptance criteria are:

1. Qubivex downloads a supported GGUF or SafeTensors model independently.
2. Qubivex validates and registers the model.
3. Qubivex detects available hardware.
4. Qubivex produces a memory-aware execution plan.
5. Qubivex loads and executes the model through its own runtime.
6. Qubivex performs MoE routing and expert streaming.
7. Qubivex supports multiple concurrent inference sequences.
8. Multiple clients share the same loaded model.
9. Every request maintains isolated generation state and KV cache.
10. Qubivex exposes an OpenAI-compatible HTTP API.
11. Qubivex streams tokens using SSE.
12. Qubivex handles client cancellation and resource cleanup.
13. Qubivex supports bounded request queues and memory admission control.
14. Qubivex provides observability and performance metrics.
15. Qubivex runs without Ollama or another external inference runtime.
16. All implemented functionality has automated tests.
17. Documentation reflects the actual implementation.
18. Existing working functionality and benchmarks remain accounted for.

---

# 17. Final Instructions to Claude Code

Approach this as a serious systems engineering project.

Do not treat this as a simple HTTP wrapper around an existing inference library.

The primary objective is to build Qubivex into an independent inference engine with its own model management and serving infrastructure.

The tensor engine, MoE routing, expert streaming and hardware-aware memory planner are the core intellectual and technical components of the project.

The HTTP server and model manager should expose those capabilities without introducing dependencies into the core runtime.

### Execution instructions

1. Start by examining the existing repository.
2. Establish the current implementation baseline.
3. Identify what can be reused.
4. Produce the target architecture and implementation plan.
5. Implement phases in the specified order.
6. Complete each phase with tests and documentation.
7. Run the full existing test suite after architectural changes.
8. Do not leave core functionality as TODOs or placeholder implementations.
9. Do not mark a phase complete without satisfying its acceptance criteria.
10. Keep the repository buildable after every meaningful implementation milestone.

Use incremental commits or clearly separated implementation milestones where appropriate.

When a phase depends on hardware unavailable in the development environment, implement the abstraction, provide a testable fallback and clearly document what remains unverified.

Do not invent benchmark results or claim unsupported hardware compatibility.

At the end of each phase, report:

- What was implemented.
- Which existing modules were reused or modified.
- New modules and dependencies.
- Tests executed and their results.
- Performance impact, where measured.
- Known limitations.
- Remaining work.
- Whether the phase acceptance criteria have been met.

**Start with Phase 0 and continue implementation through the phases. Do not stop after creating a plan or documentation.**

The ultimate goal is a standalone, hardware-aware Rust inference platform that can download, manage, load, execute and serve MoE language models locally or remotely, with efficient expert streaming and reliable multi-client concurrent inference.

Qubivex must own the complete inference lifecycle.
