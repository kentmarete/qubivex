# QUBIVEX Implementation Status

Last updated: 2026-11-02

## Current phase
**Phase 2: hardware detection, in progress.** `qubivex-hardware` and `qubivex system
[--json]` are done and tested on recorded Linux and macOS fixtures. Remaining: the
optional `qubivex doctor --storage` read benchmark, and seeing `qubivex system` on a
real Linux host (CI runs only the live smoke test there).

Phase 1 exit criteria are met (clean workspace; ADR-002 Accepted: hybrid, without
Candle). ADR-001 still awaits your review.

## Completed
| Phase | Deliverables |
|---|---|
| 0 | `docs/architecture/COLIBRI_ANALYSIS.md`, `QUBIVEX_DESIGN.md`, `MIGRATION_PLAN.md`, `IMPROVEMENTS_EVALUATION.md`, `docs/decisions/ADR-001` … `ADR-008`, `docs/development/RUST_GUIDELINES.md` |
| 1 | Cargo workspace (edition 2024, resolver 3, MSRV 1.85, ADR-001 workspace lints, `profile.release`), `clippy.toml`, `deny.toml`; `qubivex-core`; `apps/cli` (`qubivex --version`); `xtask` (`check-deps`, `check-model-names`); `.github/workflows/ci.yml`; ADR-002 spike and Colibri baseline |
| 2 (partial) | `qubivex-hardware` (profile, snapshots, fixtures), `qubivex system [--json]` |

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

### ADR-002 spike (`spikes/adr-002-tensor-engine`, standalone, not a workspace member)
- Own kernels: a fixed 64-lane reduction shape, NEON twins tested bitwise against portable twins, and lossless f32 activations.
- Candle's quantized path is lossy (Q8 activations, about 4e-3 relative error) and fast. Own lossless f32 and bf16 GEMV reach memory bandwidth. int8, Q8_0 and Q4_K reach 60%, 51% and 20% of it.
- Prefill via dequantize plus the `gemm` crate matches Candle. Accelerate is about 3× on macOS.
- Everything is bitwise deterministic across 1–11 threads.
- Colibri baseline (`colibri-baseline/colibri_bench.c`, identical inputs via `dump`): every lossless own kernel is ≥ 90% of Colibri's lossless equivalent at 1 and 11 threads. f32 and bf16 are at per-core parity. Q8_0 and Q4_K are about 10× Colibri's exact path. Attention passes only on the spinning pool (rayon gives 88%). Colibri's fast GGUF path is lossy by default.
- Decision: hybrid without Candle (ADR-002 Accepted).

### `qubivex-hardware` (Phase 2)
- `HardwareProfile` (probed once): CPU (vendor, model, logical and physical cores, **core tiers** fastest first, run-time SIMD set, NUMA nodes), memory (total and **cgroup limit**), GPUs per backend (CUDA, Vulkan, Metal: devices or "unavailable: reason"), storage (mount, fs, SSD/HDD/network, capacity, free), OS.
- `ResourceSnapshot` via `ResourceProbe` / `SystemProbe`: available RAM capped by the cgroup's room, and process RSS.
- Every value is `Probed::Known` or `Probed::Unknown(reason)`. A cgroup outside the namespace is unknown, not unlimited.
- Ported from Colibri: the cgroup v2/v1 walk (`compat.h`: room = limit − (usage − file cache), tightest ancestor), macOS available = free + inactive + speculative + purgeable pages (`setup_hw.py`), P-core teams (`omp_tune.h`: every tier but the slowest; Linux `cpu_core`/`cpu_atom`), child-process GPU probes with timeouts.
- No `unsafe`. Every helper program (`sysctl`, `vm_stat`, `ps`, `mount`, `df`, `diskutil`, `system_profiler`, `nvidia-smi`, `vulkaninfo`) runs in a child process with a timeout, with output drained on threads. Probes read through `Host`, so `Host::fixture` replays recorded `/proc`, `/sys` and command outputs.
- Fixtures: a hybrid i5-13500 in a cgroup-v2 container with NVIDIA, an EPYC with 2 NUMA nodes, cgroup v1 and an HDD root, an arm big.LITTLE with a cgroup outside its namespace, and a recorded M3 Pro.
- **Deviations from QUBIVEX_DESIGN §4.2.** `vram_available` is a `Vec<DeviceMemory>`, not a `HashMap`, so it serializes. CUDA is probed with `nvidia-smi` (NVML's CLI), not an in-process driver call. A macOS snapshot spawns `vm_stat` and `ps` (a few ms) because the crate has no FFI.

## Remaining in Phase 2
- Optional `qubivex doctor --storage` (sequential and random 4 MB reads, O_DIRECT / F_NOCACHE gain) producing `StorageMeasurement`.
- Check `qubivex system` output on a real Linux host.

## Remaining in Phase 1
- ADR-002 follow-ups (not blocking): an AVX2 measurement on Linux x86-64 against Colibri, and optional lossless Q4_K speed-up (24% of STREAM today).
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
- `cargo test --workspace`: 61 passed, 0 failed (core 22, including a proptest over untrusted dims; hardware 16 unit + 6 fixture, including a live smoke test; CLI 8; xtask 9).
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo fmt --check`, `cargo doc` (`-D warnings`), `cargo deny check` and `cargo xtask check`: clean.
- `cargo +1.85 check/test --workspace --all-targets`: clean.

## Benchmark results
ADR-002 spike, M3 Pro: [`docs/performance/ADR-002-SPIKE.md`](docs/performance/ADR-002-SPIKE.md).

## Architectural decisions
ADR-001 … ADR-008, all Proposed. ADR-001 is implemented as written. ADR-002 has spike data and a preliminary choice (hybrid without Candle), pending the Colibri baseline.
