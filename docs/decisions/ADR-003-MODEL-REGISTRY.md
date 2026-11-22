# ADR-003: Architecture registry keyed by manifest architecture id

- Status: **Accepted (2026-11-08)** by owner decision, with amendments 1 and 2. Amendment 1
  (2026-11-04): a second architecture, `transformer_dense`. Amendment 2 (2026-11-06): a
  registry of installed models. Amendment 3 (2026-11-15, proposed in PR with the owner's
  general-purpose runtime brief): q/k/v biases as a `transformer_dense` option, and plain
  RoPE enforced in `rope_parameters`. Amendment 4 (2026-11-15, proposed in PR):
  architectures declare capabilities, and undeclared weight formats are refused at
  validation.
- Date: 2026-11-01

## Context
Colibri resolves families by `config.json` `model_type` in Python
(`family_registry.py`). Each family still needs its own C engine, its own Python
geometry function and planner branches. QUBIVEX must load a compatible model with no
runtime code change.

## Decision
- `ArchitectureRegistry` maps `architecture` id to `Arc<dyn ArchitectureFactory>`.
  The id and a semver requirement come from the manifest.
- One factory trait with `parse_config`, `tensor_schema` and `build` (this replaces
  the specification's two-level factory → architecture indirection).
- `tensor_schema` maps every tensor to a **role** (embedding, attention, router,
  routed expert (layer, expert, part), shared expert, norm, head) with its expected
  shape and dtype. Validation and the memory planner work **only on roles**, never on
  names or families.
- Registration happens at startup. The registry is then frozen and shared read-only.
  Duplicate ids are an error.
- Variation inside an architecture (MHA/GQA, QK-norm, softmax/sigmoid routing, top-K
  renormalisation, shared experts, routed scaling, tensor-name patterns) is
  **configuration**, parsed into a typed `ArchConfig`.
- Static registration (explicit `register_builtin(&mut registry)`) at first. Dynamic
  plugins (`libloading`) are out of scope until there is a need.

## Consequences
- The first architecture, `transformer_moe`, must express both OLMoE and Qwen3-MoE
  by configuration. This is the extensibility acceptance test.
- A genuinely new mechanism (MLA, DeltaNet) is a new component plus a new or extended
  architecture, and does not touch runtime code.

## Amendment 1 (2026-11-04): `transformer_dense`, the second architecture

The extensibility test of the consequences above (OLMoE and Qwen3-MoE through one
factory) covered variation inside one architecture. This adds a second architecture,
registered beside the first, and runs a real checkpoint through it.

- **`transformer_dense`** (`qubivex-model/src/arch/transformer_dense.rs`): decoder-only,
  RMSNorm, RoPE attention (MHA/GQA/MQA, optional q/k norm) and one dense SwiGLU FFN per
  layer. Its own `DenseConfig`, tensor-name presets (`hf`, `gguf`) and schema. It refuses
  a checkpoint with routed experts (that is `transformer_moe`).
- **Shared configuration** (`arch/common.rs`): reading values from `config.json` or GGUF
  metadata with manifest overrides, the decoder dimensions, the attention tensor
  schema, and the checks every architecture applies (attention and FFN biases, non-SiLU
  activations, RoPE scaling, sliding-window attention are refused). `transformer_moe`'s
  behaviour is unchanged.
- **A new role, `TensorRole::Ffn { layer, part }`**: dense FFN tensors are resident. The
  planner groups them as `dense FFN`; a model without routed experts has no expert
  layers, pools or checksums.
- **The forward pass** (`qubivex-arch`, `Transformer`, formerly `TransformerMoeModel`)
  shares the embedding, attention, KV cache and head; the FFN is configuration (routed
  experts or one dense SwiGLU). A new trace point, `Ffn`, is the dense block's output.
- **Runtime:** the engine is unchanged except that whole-layer Metal and expert
  prefetch apply to routed experts only. A dense model asked to run on Metal places its
  matmuls one by one, with the CPU's bits (33 vs 69 tok/s decode on the reference
  checkpoint; a dense whole-layer pass is future work).
- **Importer:** tries `transformer_moe`, then `transformer_dense`. A tied output head
  stored anyway (as some checkpoints do) is listed in `overrides.ignore_tensors`.
- **Evidence:** a tiny Llama-style fixture (`fixtures/models/dense-tiny`: no q/k norm,
  GQA, tied head) matches transformers layer by layer (5.2e-7 relative,
  `tests/oracle.rs`) and passes the invariants (prefill equals token by token, thread
  count, every Metal placement). The full-size `Qwen/Qwen3-0.6B` bf16 SafeTensors
  checkpoint (per-head q/k norm, GQA, tied head) imports with no manifest edits, gives
  transformers' 32 greedy tokens, logits within 1.2e-5 relative, and the invariants hold
  on it (`tests/real_dense.rs`, env-gated, `fixtures/oracle/dense_real.json`).

## Amendment 2 (2026-11-06): installed models get a registry
"Registry" above means the architecture registry. Plan job 2 (`docs/engineering/PLATFORM_AUDIT.md`)
adds a separate one for installed models, in a new crate, `qubivex-model-registry`:
`registry.json` in the models directory, with no effect on models that do not use it.

- A model directory stays the unit of installation, so discovery and the runtime are
  unchanged and a directory copied in by hand is adopted.
- Installs build in `.staging/<id>`, validate, then rename. Removal renames to `.trash`
  first. `registry.json` changes by temporary file, fsync and rename.
- Locks are `flock`s, because the kernel releases them when a process dies.
- A record's `dir` must be one path component, and a file that breaks this fails to load.
- Loaded state is not recorded: a file cannot say whether a process is alive.
- `models remove` deletes nothing without `--yes`.

## Amendment 3 (2026-11-15): q/k/v biases, and RoPE settings in either key
Roadmap step 2 (`docs/ROADMAP_GENERAL_PURPOSE_RUNTIME.md`, P1): Qwen2 and Qwen2.5 put
biases on the q, k and v projections (not on o or the FFN) and their `config.json` does
not say so. They are supported as configuration of the shared components, with no code
path that knows the family:

- **An attention option, not an architecture.** `transformer_dense` 1.1 takes
  `architecture.config.qkv_bias` (default `false`). The tensor schema then requires the
  three bias vectors with new roles `QBias`, `KBias`, `VBias` (so the planner's KV
  estimate, which reads the K and V roles, is unchanged). Tensor-name presets gain
  `q_bias`, `k_bias`, `v_bias` (`hf`: `…self_attn.q_proj.bias`; `gguf`:
  `blk.{L}.attn_q.bias`). A manifest using the option records version 1.1; one without
  it stays 1.0, so older builds still read it.
- **Detected from the tensors.** Import tries `qkv_bias` false and true like the q/k norm
  modes. Validation refuses unused tensors, so only the checkpoint's own layout
  validates; a checkpoint's biases can never be silently dropped.
- **Forward pass.** The biases are dequantized once at load and added to each token's
  projections right after the matmul, before clipping, q/k norm and RoPE, as
  `nn.Linear` does. Device-fused attention (MoE decode on Vulkan) is never used when a
  model has biases. `transformer_moe` does not take the option yet.
- **`attention_bias = true`** (biases on o as well) stays refused.
- **RoPE settings.** transformers 5 writes RoPE settings as `rope_parameters`, which
  was read for `rope_theta` but not checked, so a scaled RoPE stored there (for example
  `llama3`, or multimodal RoPE) would have loaded with wrong positions. Both
  `rope_scaling` and `rope_parameters` now accept only `rope_theta` and a `default` type;
  any other key is refused by name.

Verified: a new tiny fixture `qwen2-tiny` (random biases, untied head, from
`cargo xtask gen-fixtures`) matches transformers layer by layer
(`fixtures/oracle/forward.json`), runs on every backend and batching test, and the
oracle fails when the bias add is disabled. The full-size `Qwen/Qwen2.5-0.5B-Instruct`
bf16 checkpoint (revision `7ae5576`) imports with no edits and matches transformers in
prompt ids, 32 greedy tokens, top-32 logits (1.05e-5 relative), chat template text and
ids, and a greedy chat reply ending at its end-of-sequence id; Metal, Vulkan and the
simulated device give the CPU's tokens (`tests/real_dense.rs`, env-gated
`QUBIVEX_QWEN2_SAFETENSORS_DIR`, `fixtures/oracle/qwen2_real.json`).

## Amendment 4 (2026-11-15): architectures declare capabilities
Roadmap step 3 (`docs/ROADMAP_GENERAL_PURPOSE_RUNTIME.md`, P1). The registry described a
model's structure but not what the build can execute, so a checkpoint in a format with no
kernel (Q5_K, Q4_0, ...) passed validation and failed only when the forward pass met it.

- **`ArchitectureFactory::capabilities()`** returns `ArchCapabilities`: weight formats,
  inputs, positions, RoPE scaling types, norms, attention options and FFN strategies.
  Both built-ins compose the shared decoder components, so they share one declaration
  (`arch/common.rs`) and add their own options (`qkv_bias`; routed SwiGLU and the router
  options). The strategy lists are descriptive: config parsing still enforces options.
  The RoPE scaling check reads the same list, so the two cannot drift.
- **Validation refuses undeclared formats** with the other problems, one line per
  format: count, first tensor, and the formats the architecture runs. No weight is read.
- **Backends declare formats** without opening a device (`BackendKind::supports`, from
  each backend crate's `supported`). `Engine::load` refuses, before reading weights, a
  model with a format the chosen backend or the CPU lacks, naming the formats.
- **One set.** The architectures declare exactly the formats every backend runs
  (f32, f16, bf16, Q8_0, Q4_K, Q6_K). A test compares the two for every dtype, so
  adding a kernel to one backend, or a format to the declaration, fails until the other
  side follows. A per-backend subset (formats only some backends run) would need the
  architecture's set to become a union and the load check to carry the refusal; that
  check exists and is tested with a backend lacking Q8_0.
- **`qubivex models architectures`** prints each architecture's capabilities and each
  backend's formats. `ArchDescriptor` carries the capabilities, so an API can list them.

### Amendment: Llama 3 RoPE scaling (2026-11-19)

- **`rope_type: "llama3"`** is accepted in `rope_scaling` or `rope_parameters` with its
  four values (`factor` ≥ 1, `0 < low_freq_factor < high_freq_factor`,
  `original_max_position_embeddings` > 0); a missing or extra key is refused by name, and
  the two places must agree. The parsed `RopeScaling` is part of `DenseConfig` and
  `MoeConfig`, and `Rope::with_scaling` applies it to the inverse frequencies, so every
  backend's tables carry it. `ROPE_SCALINGS` is `["default", "llama3"]`, and the type
  check reads it.
- **Versions:** `transformer_dense` 1.2.0, `transformer_moe` 1.1.0, so an older build
  refuses a manifest that may need the scaling.
- **Bits:** the llama3 adjustment mirrors `transformers`' f32 operations (a Python scalar
  rounds to f32; `scalar / tensor` is `reciprocal(tensor) · scalar`) and matches it bit for
  bit. The base power `θ^(2i/d)` is now the correctly rounded f32 (an f64 power rounded
  once) instead of an f32 `powf`, which was one ulp off on macOS for some bases; torch's
  own f32 `pow` is within one ulp of it.

Verified: `qubivex-arch` unit tests against transformers' inverse frequencies;
`real_dense` on Llama-3.2-3B (`QUBIVEX_LLAMA3_SAFETENSORS_DIR`,
`fixtures/oracle/llama3_real.json`).

## Amendment 5 (2026-11-22, Proposed): gpt-oss as configuration of `transformer_moe`

gpt-oss SafeTensors checkpoints run on the shared decoder; nothing in the code
paths names the family. `transformer_moe` 1.2.0 (and `transformer_dense` 1.3.0 for
the shared parts) adds options, each read from the checkpoint:

| Option | Where it comes from | What it does |
|---|---|---|
| `attention_bias` | `config.json` | biases on q, k, v and the output projection |
| `attention_sinks` | detected: the `sinks` tensor exists | one learned logit per query head joins every attention softmax (in the maximum and, last, in the sum) and is dropped |
| per-layer windows | `layer_types` + `sliding_window` | a `sliding_attention` layer attends to its last `sliding_window` positions, the current one included |
| YaRN | `rope_scaling` / `rope_parameters`, `rope_type: "yarn"` | transformers' `_compute_yarn_parameters`, defaults included; `attention_factor` multiplies every cos and sin in the RoPE table |
| `router_bias` | detected | a bias on the router logits |
| `router_scoring: topk_softmax` | transformers' config-class default for the `model_type` (below) | the top-K logits, then a softmax over those K |
| `expert_bias` | detected (stacked gate-up layouts only) | per-expert biases after gate-up and after down |
| clamped SwiGLU | `swiglu_limit` (+ `swiglu_alpha`) | `(clamp(up, ±limit) + 1) · g · σ(alpha · g)`, `g = min(gate, limit)` |
| `hf_interleaved` preset | the importer's candidates | router at `mlp.router`, fused gate-up whose rows alternate gate, up |

- **Values a `config.json` leaves to its transformers config class** come from a small
  table in the importer keyed by `model_type` (here: `topk_softmax` routing and
  `swiglu_alpha` 1.702, as `GptOssTopKRouter` and `GptOssConfig` define them). Each
  applies only when the checkpoint omits it, and is written into the manifest, so the
  model reads the same without the table.
- **SafeTensors MXFP4.** The index pairs `P_blocks` (`[…, rows, cols/32, 16]` u8) with
  `P_scales` (`[…, rows, cols/32]` u8) into one MXFP4 tensor `P` (`SplitScales` records
  where the scales are). The expert source repacks each expert as it loads: the scale
  byte joins its 16 nibble bytes, the nibbles move from consecutive pairs (element `2j`
  low, `2j + 1` high) to the GGML order (element `i` low, `i + 16` high), and interleaved
  gate/up rows become gate rows then up rows. All of it moves bytes; values are
  unchanged, and the kernels read MXFP4 packed as before (ADR-007 amendment 2). The
  checkpoint on disk is not rewritten. Dense tensors stored split are refused.
- **Device paths.** The fused device paths (attention, routing and the expert block on
  the device; Metal's whole-layer pass) implement none of these options, so a model
  using any of them runs those steps on the host and places its matmuls on the device
  (`Transformer::fusable`). Every backend gives the CPU's bits.
- **Tokenizer.** The chat template is read from `chat_template.jinja` when
  `tokenizer_config.json` has none; minijinja's `macros` are on; `strftime_now(format)`
  is provided, in UTC (transformers uses the server's local time, so the date can differ
  by a day around midnight).

Verified: the `gpt-oss-tiny` fixture (MXFP4 experts with interleaved gate/up, biases,
sinks, a 4-position window, YaRN) matches transformers per layer and in logits and
greedy tokens (`fixtures/oracle/forward.json`; with the window, the sinks, the clamp or
YaRN's factor removed it fails), and gives the CPU's bits on Metal, Vulkan and Sim, in
batches and with prefix reuse. YaRN's frequencies match transformers bit for bit except
where torch's f32 `pow` is not correctly rounded (one ulp). The real checkpoint:
`real_gpt_oss.rs` (`QUBIVEX_GPT_OSS_DIR`, `fixtures/oracle/gpt_oss_real.json`).
