# ADR-003: Architecture registry keyed by manifest architecture id

- Status: **Accepted (2026-11-08)** by owner decision, with amendments 1 and 2. Amendment 1
  (2026-11-04): a second architecture, `transformer_dense`. Amendment 2 (2026-11-06): a
  registry of installed models. Amendment 3 (2026-11-15, proposed in PR with the owner's
  general-purpose runtime brief): q/k/v biases as a `transformer_dense` option, and plain
  RoPE enforced in `rope_parameters`.
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
