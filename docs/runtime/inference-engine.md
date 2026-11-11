# Inference engine

The engine is `qubivex-runtime`: it loads a model, runs forward passes, and generates
text. `qubivex run`, `qubivex bench` and `qubivex serve` are all thin hosts around it,
and so is any Rust program that links it. It has no HTTP server, no CLI and no async
runtime in its dependencies, and `cargo xtask check` fails if it gains one.

## The pieces

```text
                         ┌──────────────── Engine (qubivex-runtime) ────────────────┐
 GenerationRequest ────► │ template + tokenizer ─► prefill ─► sample ─► detokenize │ ─► events
                         │                              ▲          │               │
                         │                              └── feed ──┘               │
                         │   forward pass (qubivex-arch)                           │
                         │     embedding ─► [ attention ─► FFN ] × layers ─► head  │
                         │        │              │           │                     │
                         │   dense weights    KV cache    routed experts           │
                         │   (resident)     (per request)  (expert store)          │
                         │                                                         │
                         │   memory plan · ledger · pressure monitor               │
                         └─────────────────────────────────────────────────────────┘
                               │ matmuls                         │ reads
                         compute backend                    model files
                         (CPU or Metal)
```

| Part | Crate | Page |
|---|---|---|
| Forward pass | `qubivex-arch` | this page |
| Kernels and tensors | `qubivex-compute`, `qubivex-backend-cpu`, `qubivex-backend-metal` | [Tensor engine](tensor-engine.md) |
| Routing | `qubivex-arch` | [MoE routing](moe-routing.md) |
| Experts on demand | `qubivex-expert-stream` | [Expert streaming](expert-streaming.md) |
| Budgets | `qubivex-memory` | [Memory planner](memory-planner.md) |
| KV and prefix reuse | `qubivex-arch`, `qubivex-kv-cache` | [KV cache](kv-cache.md) |

## Loading

`Engine::load(metadata, options)`:

1. takes a validated model (`ModelMetadata::load` has checked the manifest, the config
   and every tensor header);
2. reads the dense weights into memory. Routed experts are not read;
3. measures available RAM and makes the memory plan, which fixes the context and the
   expert cache;
4. builds the tokenizer, from `tokenizer.json` or from the GGUF vocabulary;
5. starts the expert store's I/O threads and the pressure monitor.

An `Engine` is `Send + Sync`. Share one by `Arc` between threads.

## The forward pass

One pass takes new tokens for one or more sequences and returns logits.

For each layer: RMSNorm, attention with rotary position embeddings over the sequence's
own KV cache, RMSNorm, then the feed-forward block. In `transformer_moe` the block is
routed experts (see [MoE routing](moe-routing.md)); in `transformer_dense` it is one
SwiGLU network. After the last layer, a final norm and the output head give logits.

The two architectures share everything except the feed-forward block.

**One evaluation order.** Each token's computation has one fixed order of floating-point
operations, whatever else is in the pass. From this the engine's invariants follow, and
tests check each one bit for bit:

| Changing this | changes the output? |
|---|---|
| The expert cache size, or where an expert was read from | No |
| Prefill in one pass, in chunks, or token by token | No |
| The number of compute threads | No |
| Running alone or in a pass with other sequences | No |
| CPU or Metal | No |
| Reusing a retained prefix | No |
| Prefetch on or off | No |

Only the model, the prompt, the sampling settings and the seed decide the tokens.

## Generation

A request goes through `Engine::run_request`:

1. **Prepare.** Render the chat template (for `Prompt::Chat`), tokenize, and check that
   the prompt leaves room in the context.
2. **Admit.** Reserve KV for the conversation against the memory plan. Under critical
   memory pressure new requests are refused.
3. **Prefill.** Run the prompt in chunks of 256 tokens. Between chunks the deadline and
   the cancel flag are checked. With a prefix cache, a matching prefix is copied in
   place of running it.
4. **Decode.** Sample a token from the logits, emit its text, feed it back. Repeat
   until end-of-sequence, a stop string, `max_tokens`, the context, the deadline or a
   cancel.

### Sampling

| Setting | Default in the engine | Effect |
|---|---|---|
| `temperature` | 0 | 0 is greedy: always the likeliest token. Above 0, logits are divided by it before sampling |
| `top_k` | 0 (off) | Keep only the K likeliest tokens |
| `top_p` | 1 | Keep the smallest set of tokens whose probability reaches P |
| `seed` | 0 | The random stream. The same seed, prompt and settings give the same tokens |

`qubivex run` uses the engine's defaults. The HTTP API follows OpenAI and defaults
`temperature` to 1.

### How a generation ends

| `Finish` | Cause | In the API |
|---|---|---|
| End of sequence | The model emitted an end token | `finish_reason: "stop"` |
| Stop string | A `stop` string appeared. The text ends before it | `"stop"` |
| Length | `max_tokens` or the context was reached | `"length"` |
| Deadline | The request's time limit passed | `"length"` |
| Cancelled | The callback returned `false`, or the cancel flag was set | The stream ends |

## Embedding it

```rust
use std::sync::atomic::AtomicBool;
use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{Engine, EngineOptions, GenerationEvent, GenerationRequest, Prompt};

let meta = ModelMetadata::load(&dir, &ArchitectureRegistry::with_builtins())?;
let engine = Engine::load(meta, EngineOptions::default())?;
let request = GenerationRequest::new(Prompt::Text("Hello".into()));
engine.run_request(&request, &AtomicBool::new(false), |event| {
    if let GenerationEvent::Text(piece) = event {
        print!("{piece}");
    }
    true // false stops the generation
})?;
```

A complete program that runs two requests on two threads is
[`crates/qubivex-runtime/examples/embed.rs`](../../crates/qubivex-runtime/examples/embed.rs):

```sh
cargo run --release -p qubivex-runtime --example embed -- fixtures/models/olmoe-tiny "Hello"
```

Several threads calling `run_request` on one `Arc<Engine>` share forward passes
automatically ([Concurrent inference](../serving/concurrent-inference.md)). For a
bounded queue with priorities in your own application, use `qubivex-scheduler` with
`Engine::prepare` and `Engine::run_prepared`, as the gateway does.

The lower-level entry points are public too: `Engine::forward` (one sequence's step),
`Engine::forward_batch` (several), `Engine::new_cache`, and `Engine::generate_with`.

## Errors

Errors fail the request, not the process.

- A failed expert read fails the requests waiting on it. No wrong token is emitted, the
  KV cache is rolled back to before the pass, memory is returned, and the next request
  reads the expert again.
- A panic in an expert source is caught and becomes a load error.
- A model that does not fit fails to load with the planner's reason.
- The fault-injection suite (`crates/qubivex-runtime/tests/faults.rs`) checks this for
  I/O errors, short reads, panics and corrupt bytes, mid-prefill and mid-decode, on the
  CPU and on Metal.

## Not supported

- **Architectures beyond `transformer_moe` and `transformer_dense`.** No multi-head
  latent attention, state-space or encoder models.
- **Sigmoid router scoring,** shared experts, attention or FFN biases, non-SiLU
  activations, RoPE scaling and sliding-window attention. Each is refused at load with
  a message naming it.
- **Repetition, presence and frequency penalties, logit bias, logprobs, beam search and
  grammar-constrained sampling.**
- **Multimodal input, embeddings output and LoRA adapters.**
- **A CUDA backend, and a fast Vulkan one.** `--backend vulkan` has the CPU's bits but
  decodes slower than the CPU (ADR-019).
- **Cancellation inside a forward pass.** It lands between tokens or prefill chunks.
- **Saving a conversation's KV to disk.**
