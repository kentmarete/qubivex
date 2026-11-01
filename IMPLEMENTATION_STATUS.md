# QUBIVEX Implementation Status

Last updated: 2026-11-02

## Current phase
**Phase 1: Rust foundation, in progress.** The workspace, `qubivex-core`, the CLI
skeleton, the repository checks and CI are done. The ADR-002 tensor-engine
benchmark spike has not started.

## Completed
| Phase | Deliverables |
|---|---|
| 0 | `docs/architecture/COLIBRI_ANALYSIS.md`, `QUBIVEX_DESIGN.md`, `MIGRATION_PLAN.md`, `IMPROVEMENTS_EVALUATION.md`, `docs/decisions/ADR-001` … `ADR-008`, `docs/development/RUST_GUIDELINES.md` |
| 1 (partial) | Cargo workspace (edition 2024, resolver 3, MSRV 1.85, ADR-001 workspace lints, `profile.release`), `clippy.toml`, `deny.toml`; `qubivex-core`; `apps/cli` (`qubivex --version`); `xtask` (`check-deps`, `check-model-names`); `.github/workflows/ci.yml` |

Colibri reference point: `JustVugg/colibri` @ `0a5504a1` (2.1.0), Apache-2.0.

### `qubivex-core` contents
- Newtype ids: `LayerIdx`, `ExpertIdx`, `ModelKey`, `ExpertId` (12 bytes, size-asserted), `DeviceId` / `DeviceKind`.
- `Bytes` with checked arithmetic and binary-unit display.
- `Shape` (row-major, last dim contiguous, rank ≤ 8) with checked `numel` / `rows`.
- `DType` (`f32`, `f16`, `bf16`, `i8_row_scaled`, GGML `q4_0` … `q8_k`). `storage_bytes(&Shape)` gives exact byte sizes with overflow and block-alignment errors, and maps GGUF type ids.
- `CoreError` (thiserror).
- `config::QubivexConfig` with a `log` section (`deny_unknown_fields`, defaults).

### Repository checks (`cargo xtask check`)
- **check-deps** parses `cargo metadata` and enforces QUBIVEX_DESIGN §3.1. Every workspace crate must be listed in `xtask/src/deps.rs`. Forbidden edges, such as `model → compute/backend-*`, `memory → arch`, `expert-stream → arch`, `backend-* → runtime/api` and anything depending on an app, fail.
- **check-model-names** scans `crates/` and `apps/` (excluding `tests/`, `fixtures/`, `benches/`) for tokens that start with a model family name.

### CI jobs
- fmt, clippy (`-D warnings`), test, doc and xtask checks on Linux x86-64 and macOS arm64.
- MSRV 1.85 check and test.
- `cargo deny` (advisories, bans, licences, sources).

The `miri` and `loom` jobs arrive with the first unsafe or concurrent code (Phase 4), as planned. There is no feature-powerset job yet because no crate has features.

## Remaining in Phase 1
- ADR-002 spike: tensor-engine benchmark on expert shapes (2048×1024, 1024×2048) at S=1 and S=128 (Candle against own kernels; f32, bf16, int8 row-scaled, Q4_K, Q8_0). It needs Colibri's kernels for the ≥ 90% throughput rule, so building Colibri (third-party code) needs confirmation.
- Move ADR-001 to Accepted after review.

## Environment change (supersedes MIGRATION_PLAN §2)
Development moved to an **Apple M3 Pro (arm64, 18 GB RAM, macOS)** with about 374 GB free.
- **NEON** is now the first local SIMD target. AVX2 and AVX-512 are exercised on Linux x86-64 CI.
- Rust stable 1.99.1 and 1.85 (MSRV) are installed through rustup in `~/.cargo`.
- Apple GPU: Metal becomes the locally testable GPU backend. CUDA still needs other hardware.

## Pending review decisions
1. Synchronous lease-based expert store instead of the specification's async trait (ADR-006).
2. 11 initial crates instead of about 25 (ADR-001).
3. Merged architecture factory with `tensor_schema` (ADR-003).
4. Minimal expert store moved into Phase 4 (MIGRATION_PLAN §3).
5. Reference model: OLMoE-1B-7B-0125-Instruct, plus a tiny Qwen3-MoE fixture.
6. Optional expert-packed container layout (ADR-006).
7. Keep prefetch (Phase 7) before GPU (Phase 8).
8. Treat failure handling as P0.
9. No lossy option in any execution profile.

## Blockers
- ~~**B1** Disk~~: resolved by the machine change (about 374 GB free). The bf16 OLMoE checkpoint fits.
- **B2** No CUDA hardware for Phase 8.
- ~~**B3** Git repository~~: resolved (`origin` = `github.com/kentmarete/qubivex`).

## Test results (2026-11-02, M3 Pro, Rust 1.99.1 and 1.85)
- `cargo test --workspace`: 35 passed, 0 failed (core 22, including a proptest over untrusted dims; CLI 4; xtask 9).
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo fmt --check`, `cargo doc` (`-D warnings`), `cargo deny check` and `cargo xtask check`: clean.
- `cargo +1.85 check/test --workspace --all-targets`: clean.

## Benchmark results
None yet (ADR-002 spike pending).

## Architectural decisions
ADR-001 … ADR-008, all Proposed. ADR-001 is implemented as written. ADR-002 is pending the spike.
