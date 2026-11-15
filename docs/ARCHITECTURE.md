# Qubivex target architecture

The logical architecture Qubivex is evolving towards (owner's brief, 2026-11-15), and
where each part lives in the workspace today. Detailed current-state documents:
[architecture/overview.md](architecture/overview.md),
[architecture/workspace-structure.md](architecture/workspace-structure.md),
[architecture/dependency-architecture.md](architecture/dependency-architecture.md),
[architecture/concurrency-model.md](architecture/concurrency-model.md). The plan and the
feature matrix: [ROADMAP_GENERAL_PURPOSE_RUNTIME.md](ROADMAP_GENERAL_PURPOSE_RUNTIME.md).

## The tree

```text
Qubivex Runtime
│
├── Model Registry
│   ├── Architecture Registry
│   ├── Weight Loader
│   └── Model Manager
│
├── Tensor Engine
│   ├── CPU Backend
│   ├── Vulkan Backend
│   └── Memory Planner
│
├── Architecture Engine
│   ├── Llama
│   ├── Qwen2
│   ├── Qwen2.5-VL
│   ├── Qwen3
│   └── Other Architectures
│
├── Multimodal Engine
│   ├── Vision
│   ├── Audio
│   ├── Video
│   └── Projectors
│
├── Execution Engine
│   ├── Attention
│   ├── MoE Router
│   ├── Expert Streaming
│   └── Scheduler
│
└── Serving Engine
    ├── OpenAI API
    ├── Ollama API
    ├── Structured Output
    └── Tool Calling
```

The brief's expanded form (A–F: model management, tensor engine, architecture engine,
multimodal engine, execution engine, serving engine) is the same tree with more leaves;
the matrix in the roadmap follows it.

## Principles

1. **Families are configuration, not code paths.** "Llama", "Qwen2", "Qwen2.5-VL" and
   "Qwen3" in the tree are *model families supported by the Architecture Engine*, not
   separate implementations. Each is a configuration of shared, registered components
   (decoder blocks, attention variants, position schemes, FFN kinds, encoders,
   projectors), validated by the Architecture Registry (ADR-003). Engine code never
   branches on a family name; `cargo xtask check` fails the build if a family name
   appears in engine code without a format-identifier exemption.
2. **One execution path per concept.** A new family that needs a new behaviour (a bias,
   a norm kind, a RoPE variant, a windowed attention pattern) adds that behaviour as an
   option of the shared component, with a fixture test against the reference.
3. **The runtime stays embeddable.** Everything above the Serving Engine works inside an
   application with no HTTP server, no CLI and no Tokio (`xtask` enforces it).
4. **Lossless by default, identical across backends.** Device backends give the CPU's
   bits (ADR-010, ADR-011, ADR-019) unless a named opt-in says otherwise.
5. **Hardware-aware is the differentiator.** The Memory Planner decides placement and
   streaming from measured hardware; no part of the tree assumes the model fits.
6. **No other runtime executes anything.** Ollama, llama.cpp and transformers are
   reference implementations for tests and comparisons only.
7. **New crates only for a real boundary** (a dependency that others must not carry, or
   code an embedder may leave out), not to mirror the tree.

## Mapping to the workspace

| Tree node | Crates and modules today | State | Planned change |
|---|---|---|---|
| **Model Registry** | | | |
| Architecture Registry | `qubivex-model/src/registry.rs`, `arch/` (`transformer_moe`, `transformer_dense`) | Implemented: config and tensor schemas | Declare capabilities (quant formats, backends, position/norm/FFN strategies, preprocessing); register encoder and projector components |
| Weight Loader | `qubivex-model` (`gguf.rs`, `safetensors.rs`, `import.rs`); `qubivex-runtime/src/engine.rs` (dense arena), `experts.rs` | Implemented: GGUF, SafeTensors, read into an arena; experts streamed | Components with separate files (manifest v1.2); optional memory-mapping |
| Model Manager | `qubivex-model-registry`, `qubivex-model-download`, `apps/cli/src/models.rs`, gateway load/unload | Implemented: HF downloads, installs, lifecycle | Multi-component downloads; Ollama registry import |
| **Tensor Engine** | | | |
| CPU Backend | `qubivex-backend-cpu`, `qubivex-compute` (trait, tensors, math) | Implemented: portable, NEON, AVX2 | More quantised formats; AVX-512; vision ops |
| Vulkan Backend | `qubivex-backend-vulkan` | Implemented, bit-identical; speed work ongoing (PR #67) | Vision ops, prefill attention, dense passes |
| *(Metal Backend)* | `qubivex-backend-metal` | Implemented (not in the tree above, kept: it is the fastest backend on Apple silicon today) | Dense whole-layer passes |
| Memory Planner | `qubivex-memory` | Implemented: RAM and VRAM budgets | Encoder weights and activations, embedding spans |
| **Architecture Engine** | `qubivex-arch` (`Transformer`, `ops`), `qubivex-model/src/arch/` (config) | One generic decoder, dense or MoE FFN | Configurable biases, norms, activations, position schemes; families as configs (Qwen2/2.5 first) |
| **Multimodal Engine** | none | Missing | New crate `qubivex-multimodal` (image/audio/video decoding and preprocessing, encoders, projectors): its decoding dependencies must stay out of text-only embedders. See [MULTIMODAL_RUNTIME.md](MULTIMODAL_RUNTIME.md) |
| **Execution Engine** | | | |
| Attention | `qubivex-arch` (host), `qubivex-compute` device ops, Metal/Vulkan kernels | Implemented for decoder self-attention | Blocked attention for long and bidirectional (encoder) attention |
| MoE Router | `qubivex-arch` (host routing), device routing in Metal/Vulkan | Implemented | — |
| Expert Streaming | `qubivex-expert-stream`, `qubivex-runtime/src/experts.rs` | Implemented | — |
| Scheduler | `qubivex-scheduler`, `qubivex-runtime` (batching, KV admission), `qubivex-kv-cache` (prefix reuse) | Implemented | Media-aware prefix keys; paged KV and speculative decoding when measured |
| **Serving Engine** | `qubivex-api` | OpenAI API implemented (text) | Structured output, tool calling, Ollama API, multimodal content. See [API_COMPATIBILITY.md](API_COMPATIBILITY.md) |
| OpenAI API | `qubivex-api/src/openai.rs` | Implemented | `response_format`, `tools`, image parts |
| Ollama API | — | Missing | `/api/*` in `qubivex-api` |
| Structured Output | — | Missing | Constraint engine in the runtime (so embedders get it), JSON Schema compiler; API maps `response_format` and Ollama `format` onto it |
| Tool Calling | — | Missing | Built on structured output plus the chat template's tool format |

## One request through the target architecture

1. **Serving Engine** parses an OpenAI or Ollama request into a `GenerationRequest`:
   messages with text and media parts, sampling, constraints (JSON Schema, tools).
2. **Model Manager** resolves the model; the **Memory Planner** admits the request
   (KV, encoder activations, media embeddings).
3. **Multimodal Engine** decodes and validates each media item, preprocesses it
   according to the model's declared pipeline, runs the encoder and projector, and
   returns an **embedding span** with its positions.
4. The chat template renders the conversation; placeholders become token runs; spans
   are attached at their positions.
5. **Execution Engine** schedules the sequence into a batch, reuses any retained prefix
   (keyed by tokens *and* media digests), and runs the decoder through the **Tensor
   Engine** on the placed backends, streaming experts as needed.
6. The sampler applies the constraint mask, then samples; tokens stream back through the
   Serving Engine in the client's protocol.
