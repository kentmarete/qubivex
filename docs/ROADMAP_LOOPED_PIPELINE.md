# Roadmap: a separate looped-hybrid architecture and pipeline

Status: **Draft** (2026-11-22). Not started; no code exists for it.

First target: `thinkingmachines/Inkling-2-Mini` (config `model_type = inkling2_loop`,
released 2026-11-20, Apache 2.0). 26.4B parameters, 3.4B active per token, 5.4B effective
with its three loop passes.

## 0. Decision and ground rules

The owner decided on 2026-11-22 that this model gets **its own architecture and its own
execution pipeline**, beside the existing one, rather than more options on
`transformer_dense` / `transformer_moe`. The existing runtime must not be compromised.

These rules hold at every milestone. A milestone that breaks one is not merged.

| # | Rule | How it is checked |
|---|---|---|
| G1 | The existing forward pass (`qubivex_arch::Transformer`), its configs and its tensor schemas are not edited to make room for the new pipeline | Diff review: changes under `crates/qubivex-arch/src/` and `crates/qubivex-model/src/arch/{common,transformer_dense,transformer_moe}.rs` are limited to the single seam in G3 |
| G2 | Existing outputs stay bit-identical | All current fixture, oracle and parity tests pass unchanged; logits of the existing fixtures hashed before and after each milestone and compared |
| G3 | Exactly one seam in shared code: the engine holds a model enum (existing transformer or looped pipeline) instead of the concrete `Transformer` (`crates/qubivex-runtime/src/engine.rs:950`) | The seam lands alone, first, as its own PR, with no behaviour change (milestone L0) |
| G4 | Shared kernels in `qubivex-compute` and the backends are only **added to**, never changed | New ops are new functions; existing kernel tests and `fixtures/oracle/*` unchanged |
| G5 | Performance of the existing path does not regress | `bench/vulkan-step1a` baseline and the CPU decode baselines rerun at L0 and at the end; within noise |
| G6 | The new pipeline is opt-in by architecture id; models that resolve today resolve to exactly the same factory | Registry test: every existing fixture manifest resolves to the same id and version |
| G7 | Still embeddable, still no family names in engine code | `cargo xtask check` (component names describe mechanisms, not the model: e.g. `looped_hybrid_moe`, never `inkling`) |
| G8 | No support is claimed without an end-to-end test against the reference implementation | Each milestone's acceptance criteria below |

The new pipeline is separate, but its parts are named for what they do (loop block,
relative-position attention, short convolution), so a later looped or hybrid model can
reuse them without touching the existing transformer either.

## 1. Gap analysis

What the checkpoint needs, against what exists today.

| Need (from `config.json`) | Today | Where it goes |
|---|---|---|
| Weights: NVFP4 + MXFP8 (`modelopt`), ~20 GB | f32/f16/bf16, Q8_0, Q4_K, Q6_K, MXFP4 | New dequant kernels in `qubivex-compute` (additive, G4) |
| Loop: layers 6–17 run 3 passes, shared weights, rank-64 gate (`loop_config`) | Each layer runs once | New pipeline: loop block |
| 24 layers; 20 local (`local_layer_ids`, window 1024), 4 global | Per-layer windows from `layer_types` only | New pipeline: own layer plan |
| Relative positions (`d_rel = 16`, `rel_extent = 1024`), log-scaled attention (`log_scaling_*`), QK and rel L2 clipping, value norm without gamma | RoPE only | New pipeline: attention variant |
| Short convolution (`sconv_kernel_size = 4`) | None | New pipeline op + per-sequence conv state |
| Attention residual blocks (`n_layers_per_attn_res_block = 6`) | None | New pipeline |
| MoE: 128 experts, top 8, no shared expert, gate bias, `route_scale = 8`, gate/up softcaps | Routing exists in `transformer_moe`; no softcaps or this scoring | New pipeline's MoE layer; reuses `ExpertProvider` and expert streaming as they are |
| Final logit softcap 30.0; padded vocab (201024, 200058 real) | None | New pipeline head |
| Vision: 23-layer ViT, patch 20, merge 2, projector | No vision anywhere | Phase 3, after the general multimodal foundation (P3) |
| MTP head (3 layers, speculative) | None | Optional; Phase 4 |
| Context up to 1M | Contiguous KV | Local layers need only a 1024 ring; global layers need paged KV (roadmap P8) |
| Tokenizer and chat template (`<|message_user|>` etc., reasoning effort) | HF tokenizer path exists | Check coverage; template through the existing template engine |

Unknowns to settle in L1 before any kernel work: the exact semantics of the loop gate,
`d_rel`/`rel_extent`, log scaling, the attention residual blocks and `sconv`, read from the
transformers reference implementation, not inferred from field names.

## 2. Target architecture

```
qubivex-model     registry ── transformer_dense, transformer_moe   (unchanged)
                           └─ looped_hybrid_moe                    (new factory: config, tensor schema, capabilities)

qubivex-arch      Transformer                                      (unchanged)
qubivex-arch-looped  (new crate)
                  LoopedModel: embed → pre-loop layers → loop block ×N passes → post-loop layers → head
                  own layer plan, own KV state (local ring, global, conv state, loop state)

qubivex-compute   existing kernels (unchanged) + new: NVFP4/MXFP8 dequant, relative-position attention,
                  short conv, softcap                              (additive)

qubivex-runtime   Engine.model: Model::Transformer | Model::Looped (the one seam, G3)
                  scheduler, sampling, constrained decoding, API   (shared, unchanged)
```

Why a separate crate: the existing `Transformer` encodes invariant 1 (batch-invariant
bits) and its prefix reuse around one KV layout. The looped model's state (ring buffers,
conv state, per-pass activations) is different enough that sharing the struct would mean
editing it. A separate crate keeps the existing pipeline's code and tests untouched and lets
the new one hold the same invariants on its own.

What the new pipeline reuses as-is: `DenseWeights` / `ExpertProvider` traits, expert
streaming and the expert cache, the memory planner, the scheduler, sampling, JSON
constrained decoding, the tokenizer, the OpenAI API layer.

## 3. Milestones

Each milestone is one job and one PR (see `docs/development/JOB_WORKFLOW.md`).

### Phase 0: protect the ground

| Id | Work | Done when |
|---|---|---|
| L0 | Engine seam: `Model` enum around the existing `Transformer`, no other change. Record logit hashes for every existing fixture and the benchmark baselines | All tests pass unchanged; hashes identical; benchmarks within noise (G2, G3, G5) |
| L1 | Reference study and ADR-022 "Looped hybrid pipeline": read the transformers implementation, write down each op exactly, decide the KV/state layout. Build a tiny random-weight fixture with the reference (`fixtures/models/looped-hybrid-tiny/`) and its golden logits | ADR Proposed; fixture and goldens committed; owner review |

### Phase 1: text, bf16, CPU

| Id | Work | Done when |
|---|---|---|
| L2 | `looped_hybrid_moe` factory: config parse, tensor schema, capabilities. Refuses NVFP4/MXFP8 and vision until implemented | `models architectures` lists it; tiny fixture validates; every existing manifest resolves as before (G6) |
| L3 | New ops in `qubivex-compute`: relative-position attention with log scaling and clipping, short conv, softcap, value norm without gamma. Each checked against the reference | Op tests within tolerance of the reference; existing kernel tests unchanged (G4) |
| L4 | `qubivex-arch-looped`: layer plan, loop block with gate, attention residual blocks, MoE with softcaps and gate bias, head with softcap and vocab trim | Tiny fixture logits match goldens; prefill-in-one-batch equals token-by-token |
| L5 | Engine wiring: generation, sampling, chat template, API | `qubivex run` and `/v1/chat/completions` work on the tiny fixture end to end |

### Phase 2: the real checkpoint

| Id | Work | Done when |
|---|---|---|
| L6 | NVFP4 and MXFP8 dequant kernels (CPU), bit-exact vs a reference dequant; import of the `modelopt` checkpoint | Dequant matches reference bit for bit; import validates the real checkpoint by sha256 |
| L7 | Real-model parity: Inkling-2-Mini text, greedy | Top-1 tokens match transformers over a fixed prompt set; logits within the ADR-010 tolerance |
| L8 | Memory and placement: local-layer ring KV, expert streaming for 128 experts, plan fits on a 32 GB Mac | `qubivex plan` shows the fit; long-context run without swap growth |
| L9 | Metal / Vulkan for the new ops, each bit-identical to CPU (ADR-019 rule) | Parity tests; decode tok/s measured and recorded |

### Phase 3: vision (after roadmap P3)

| Id | Work | Done when |
|---|---|---|
| L10 | Plug the ViT + projector into the generic vision encoder once P3 exists; image preprocessing (patch 20, merge 2, temporal patch 2) | Image chat over the API matches the reference on a fixed image set |

### Phase 4: optional

| Id | Work | Done when |
|---|---|---|
| L11 | MTP speculative decoding | Same outputs as without it; measured speed-up |
| L12 | Global-layer paged KV for very long context (with roadmap P8) | 128k+ context run within memory plan |

## 4. Risks

| Risk | Mitigation |
|---|---|
| The seam (L0) changes behaviour or speed | It lands alone with hash and benchmark checks before any new code |
| Op semantics guessed wrong from field names | L1 reads the reference code; every op tested against it (L3) |
| NVFP4 numerics differ between implementations | Bit-exact dequant against the reference library; decide in ADR-022 which library is the oracle |
| 26.4B total parameters do not fit comfortably | Expert streaming already handles routed experts; L8 measures it; bf16 parity runs may need a larger machine or a quantised oracle |
| Two pipelines drift apart (fixes made in one, not the other) | Shared things stay shared (scheduler, sampling, API, weights traits); only the forward pass and its state are separate |
| Scope creep into the existing pipeline | G1–G7 are merge gates, checked in every PR description |

## 5. Open decisions for the owner

1. Approve the one seam in the engine (G3), or require the new pipeline behind a separate
   `Engine` type with no change to the existing one at all.
2. Text-only first (Phases 1–2) before vision, or wait for P3 and ship both.
3. Priority against the current roadmap (Qwen2.5-VL, P3 multimodal foundation).
4. Which NVFP4 reference is the oracle (ModelOpt, or the transformers dequant path).
