# QUBIVEX Rust Guidelines

Status: Proposed with Phase 0. These are enforced by CI from Phase 1.

This is a curated, project-specific subset adapted from
[leonardomso/rust-skills](https://github.com/leonardomso/rust-skills) (MIT,
© 2025 Leonardo Maldonado, reviewed at commit `fd2a861`), plus rules learned from
Colibri's bug history. Rule ids in brackets (`[err-no-unwrap-prod]`) refer to that
repository's `rules/*.md` for the full rationale and examples.

Rules are tagged **MUST** (CI-enforced or review-blocking) or **SHOULD**.

---

## 1. Errors and panics

- **MUST** Library crates use `thiserror` domain errors. `anyhow` is allowed only in
  `apps/*` and tests. [err-thiserror-lib, err-anyhow-app]
- **MUST** No `unwrap()` outside tests. `expect("…")` is allowed only for real
  invariants (a bug if it fires), with a message saying which invariant.
  [err-no-unwrap-prod, err-expect-bugs-only]
- **MUST** Preserve error chains with `#[source]` or `#[from]`. Log an error once, at
  the boundary that handles it, with its full chain. [err-source-chain, obs-error-chain]
- **MUST** Allocation failure on model-sized buffers uses `try_reserve` /
  `Vec::try_with_capacity`-style fallible allocation and becomes an error. Colibri's
  `exit(1)` on OOM is the anti-pattern this replaces. [anti-panic-expected]
- **SHOULD** Error messages are lowercase with no trailing punctuation. Public fns
  document an `# Errors` section. [err-lowercase-msg, err-doc-errors]

## 2. Untrusted input and numeric safety

Model files are untrusted input. Colibri needed several SEC fixes here (oversized
headers overflowing slabs, an all-NaN router producing index −1).

- **MUST** Every size derived from file headers or config uses checked arithmetic
  (`checked_mul`, `checked_add`) and `TryFrom` for narrowing. Do not use `as` for any
  size, offset or index derived from input. [num-overflow-explicit, num-cast-try-from]
- **MUST** Router top-K and every float ordering use `f32::total_cmp`, never
  `partial_cmp().unwrap()`. NaN logits must produce a typed error, never an
  out-of-range index. [num-float-compare]
- **MUST** Numerical tests compare with explicit tolerances, except for the invariant
  tests that are bitwise by design (same kernel set, different cache capacity).
  [num-float-compare]
- **SHOULD** Use `NonZeroU32` / `NonZeroUsize` for counts that cannot be zero
  (`experts_per_token`, `num_layers`, slot capacity). [num-nonzero]

## 3. Unsafe code

- **MUST** `unsafe_code = "deny"` workspace-wide. Only these crates opt in, with a
  crate-root `#![allow(unsafe_code)]` and a module-level justification:
  `qubivex-backend-cpu` (SIMD), `qubivex-expert-stream` (slab I/O), and
  `qubivex-backend-{cuda,vulkan,metal}` (FFI). [lint-workspace-lints]
- **MUST** Every `unsafe` block has a `// SAFETY:` comment, and every `unsafe fn` has
  a `# Safety` doc section (`clippy::undocumented_unsafe_blocks`,
  `clippy::missing_safety_doc`). [unsafe-safety-comment, lint-unsafe-doc]
- **MUST** Keep `unsafe` minimal in scope: wrap it in a safe abstraction in the
  smallest module. [unsafe-minimize-scope]
- **MUST** Use `MaybeUninit` for uninitialised slabs, never `mem::zeroed` on types with
  validity invariants. [unsafe-maybeuninit]
- **MUST** Use Rust 2024 forms: `unsafe extern "C" { … }` with per-item
  `safe`/`unsafe`, and `#[unsafe(no_mangle)]`. [unsafe-extern-block, unsafe-no-mangle-unsafe]
- **MUST** Manual `Send` and `Sync` impls (device buffers, slabs) document the
  invariant. [unsafe-send-sync-manual]
- **MUST** CI runs `cargo miri test` (pinned nightly,
  `-Zmiri-strict-provenance`) on the unsafe-bearing crates' unit tests. SIMD
  intrinsics are not Miri-supported, so each SIMD kernel has a scalar twin, and Miri
  runs the scalar path. [unsafe-miri-ci]

## 4. Concurrency (expert store, I/O pool, prefetcher)

- **MUST** Lease, slot-state, in-flight-dedup and eviction code is tested with `loom`
  under `cfg(loom)`, with `loom::sync` substituted for `std::sync`. Colibri's slot
  races (a prefetcher overwriting a slot mid-matmul; two writers on one slab) are
  exactly the bugs loom exists to find. [test-loom-concurrency]
- **MUST** Use the weakest correct atomic ordering, with a comment on every
  non-`Relaxed` one naming its paired operation. Statistics counters use `Relaxed`.
  [conc-atomic-ordering]
- **MUST** Use only bounded channels and queues. Unbounded ones are disallowed via
  `clippy.toml` `disallowed-methods`. [async-bounded-channel]
- **MUST** (gateway) Never hold a lock across `.await`. CPU work never runs on Tokio
  workers: inference runs on dedicated threads and is bridged with bounded channels.
  Use `CancellationToken` for request cancellation and shutdown.
  [anti-lock-across-await, async-spawn-blocking, async-cancellation-token]
- **SHOULD** Use `std::thread::scope` for borrowing per-layer buffers across worker
  threads. [conc-scoped-threads]

## 5. Performance

- **MUST** Profile before optimising, and benchmark with Criterion using `black_box`.
  Any kernel change records before and after numbers in the PR.
  [perf-profile-first, perf-black-box-bench, test-criterion-bench]
- **MUST** No `format!`, `String` building or allocation on the per-token hot path.
  Reuse scratch buffers per thread and layer. [anti-format-hot-path, mem-reuse-collections]
- **MUST** Hot metadata structs have size assertions
  (`const _: () = assert!(size_of::<SlotMeta>() == N);`) so growth is deliberate.
  [mem-assert-type-size]
- **SHOULD** Use `FxHashMap` / `ahash` for internal maps keyed by our own ids (no DoS
  exposure). Use `std` `HashMap` for anything keyed by request data. [perf-ahash]
- **SHOULD** Mark error and cold paths `#[cold]` / `#[inline(never)]`. Use
  `#[inline(always)]` only with profiler evidence. [opt-cold-unlikely, opt-inline-always-rare]
- **SHOULD** Write hot loops in iterator and `chunks_exact` form so bounds checks are
  elided. Verify with `cargo asm` when it matters. [opt-bounds-check]

## 6. Types and API design

- **MUST** Use newtype ids (`LayerIdx`, `ExpertIdx`, `ModelKey`, `DeviceId`), never
  bare `u32` or `String` across crate boundaries. [type-newtype-ids, anti-stringly-typed]
- **MUST** States are enums (`ModelState`, slot state), not boolean flags.
  [type-enum-states]
- **MUST** Take `&[T]` / `&str` parameters, not `&Vec<T>` / `&String`. Borrow instead
  of cloning. [own-slice-over-vec, own-borrow-over-clone]
- **SHOULD** Builders for `LoadOptions` and `GenerateRequest`, with `#[must_use]`.
  [api-builder-pattern, api-builder-must-use]
- **SHOULD** Prefer generics in hot paths and `dyn` at plugin boundaries
  (`ArchitectureFactory`, `ComputeBackend` selection), never per element.
  [trait-dyn-vs-generic]

## 7. Observability

- **MUST** Use `tracing` with structured fields. Library crates never install a
  subscriber; only `apps/*` do. [obs-tracing-over-log, obs-library-facade, obs-structured-fields]
- **MUST** No prompt text, tokens or auth tokens in logs or spans by default. Wrap the
  admin token in `secrecy::SecretString`. Model paths are logged locally and never
  returned in API errors. [obs-no-sensitive-data]
- **SHOULD** Per-token spans are `TRACE` level, compiled in but disabled by default.

## 8. Project hygiene

- **MUST** Use workspace dependency inheritance (`[workspace.dependencies]`) and
  workspace lints (`[workspace.lints]`, with `lints.workspace = true` in every crate).
  [proj-workspace-deps, lint-workspace-lints]
- **MUST** Declare MSRV (`rust-version`) and test it in CI. [proj-msrv-declare]
- **MUST** Cargo features are strictly additive. `cuda`, `vulkan`, `metal` and
  `io-uring` only add backends or modes, and every combination compiles.
  [proj-feature-additive]
- **MUST** Use `unexpected_cfgs` with `check-cfg` declaring `cfg(loom)`, so a typo in a
  cfg cannot silently disable a code path. [lint-cfg-check]
- **MUST** Keep `main.rs` thin and put logic in libraries. [proj-lib-main-split]
- **SHOULD** Use `pub(crate)` by default and re-export the public API from crate roots.
  [proj-pub-crate-internal, proj-pub-use-reexport]

## 9. Rules adopted with changes, or rejected

| Upstream rule | Decision | Reason |
|---|---|---|
| `opt-target-cpu` (`target-cpu=native`) | **Rejected for distributed builds** | Binaries must run on any x86-64 / aarch64 baseline. Use runtime dispatch (`is_x86_feature_detected!` + `#[target_feature]`), as the rule's own "Feature Detection at Runtime" section shows. Colibri needed `cpu_check.h` after prebuilt binaries crashed on older CPUs. `native` is allowed for local benchmarking only |
| `opt-simd-portable` (`std::simd`) | **Rejected** | Nightly only. We target stable. Use `std::arch` intrinsics with scalar reference twins |
| `async-fn-in-trait` | Adopted with a caveat | Native `async fn` in traits is not dyn-compatible. Our hot-path traits are synchronous anyway (ADR-006) |
| `opt-pgo-profile` | Deferred | Revisit in Phase 10 once end-to-end benchmarks are stable |
| `lint-missing-docs` | `warn`, raised to `deny` from Phase 9 | Interfaces churn in early phases |
| `clippy::pedantic` | Selective only | [lint-pedantic-selective] |
