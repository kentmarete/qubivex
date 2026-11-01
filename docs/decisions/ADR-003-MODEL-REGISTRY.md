# ADR-003: Architecture registry keyed by manifest architecture id

- Status: Proposed (Phase 0)
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
