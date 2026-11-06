# ADR-001: Rust Cargo workspace with phase-gated crates

- Status: **Accepted (2026-11-06)** by owner decision, as implemented since Phase 1.
  One point is not yet in force: no crate defines Cargo features yet (the Metal backend
  is selected by `target_os`), so the feature-powerset CI job has nothing to build. It
  is added with the first feature.
- Date: 2026-11-01

## Context
The specification proposes about 25 crates. Colibri shows that the tightly coupled
code (the cache, the loader and the MoE layer) changes together often. Creating many
crates before their interfaces stabilise produces churn and empty crates, which the
specification forbids.

## Decision
- One Cargo workspace, edition 2024, MSRV 1.85, `resolver = "3"`.
- Start with **11 crates** (see `QUBIVEX_DESIGN.md` §3). A crate is created only in
  the phase that first gives it real, tested code.
- Workspace-wide lints, inherited by every crate via `[lints] workspace = true`
  (full rationale in `docs/development/RUST_GUIDELINES.md`):

  ```toml
  [workspace.lints.rust]
  unsafe_code = "deny"            # opted out per crate with #![allow(unsafe_code)] in
                                  # backend-cpu, expert-stream, backend-{cuda,vulkan,metal} only
  unsafe_op_in_unsafe_fn = "deny"
  missing_docs = "warn"
  unexpected_cfgs = { level = "warn", check-cfg = ['cfg(loom)'] }

  [workspace.lints.clippy]
  correctness = { level = "deny", priority = -1 }
  suspicious  = { level = "deny", priority = -1 }
  style       = { level = "warn", priority = -1 }
  complexity  = { level = "warn", priority = -1 }
  perf        = { level = "warn", priority = -1 }
  unwrap_used = "deny"            # tests allowed via clippy.toml allow-unwrap-in-tests
  undocumented_unsafe_blocks = "deny"
  missing_safety_doc = "deny"
  cast_possible_truncation = "warn"
  float_cmp = "deny"
  ```
- `[workspace.dependencies]` pins every third-party version once. Member crates use
  `dep = { workspace = true }`.
- `rust-version` is declared and checked by an MSRV CI job. GPU and io_uring features
  are strictly additive, and CI builds the feature powerset of the CPU-buildable
  features.
- `profile.release`: `lto = "fat"`, `codegen-units = 1`, `panic = "unwind"` (the
  server catches per-request panics at the gateway boundary).
- Dependency direction is enforced by a test over `cargo metadata`. A CI guard rejects
  model-name literals in shared crates.

## Consequences
- Fewer crates to begin with. Splits (KV cache, sampling, observability) happen when a
  second consumer or a compile-time benefit appears.
- Boundary violations fail CI instead of relying on review.
