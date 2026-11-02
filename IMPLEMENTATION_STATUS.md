# QUBIVEX Implementation Status

Last updated: 2026-11-03

## Current phase
**Phase 4: CPU inference prototype, in progress (step 1 of 5 done: kernels).**
The plan has five steps: (1) compute interface and CPU kernels; (2) tokenizer and chat
templates; (3) a minimal expert store; (4) the `transformer_moe` forward pass, KV
cache, sampling and `qubivex run`; (5) numerical exit tests against `transformers`
and an independent f64 reference.

**Phase 3: model metadata and registry, done.** Containers, manifest, registry,
`transformer_moe`, validation, discovery, import and `qubivex models
list|inspect|import|architectures` are done, with three committed tiny fixtures. The
exit check passed on the real OLMoE-1B-7B-0125-Instruct Q4_K_M GGUF. **Next: Phase 4,
the CPU inference prototype.**

Phase 2 (hardware detection) is done: `qubivex-hardware`, `qubivex system [--json]`
and `qubivex doctor --storage`. They are tested on recorded Linux and macOS fixtures and
live on macOS. Remaining: seeing both commands' output on a real Linux host (CI runs
only the live tests there).

Phase 1 exit criteria are met (clean workspace; ADR-002 Accepted: hybrid, without
Candle). ADR-001 still awaits your review.

## Completed
| Phase | Deliverables |
|---|---|
| 0 | `docs/architecture/COLIBRI_ANALYSIS.md`, `QUBIVEX_DESIGN.md`, `MIGRATION_PLAN.md`, `IMPROVEMENTS_EVALUATION.md`, `docs/decisions/ADR-001` … `ADR-008`, `docs/development/RUST_GUIDELINES.md` |
| 1 | Cargo workspace (edition 2024, resolver 3, MSRV 1.85, ADR-001 workspace lints, `profile.release`), `clippy.toml`, `deny.toml`; `qubivex-core`; `apps/cli` (`qubivex --version`); `xtask` (`check-deps`, `check-model-names`); `.github/workflows/ci.yml`; ADR-002 spike and Colibri baseline |
| 2 (partial) | `qubivex-hardware` (profile, snapshots, fixtures, storage measurement), `qubivex system [--json]`, `qubivex doctor --storage` |

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

- `measure_storage` / `qubivex doctor --storage [--path DIR] [--size-mib N] [--json]`: writes an incompressible scratch file (refuses without size + 1 GiB free), then measures an uncached sequential read (`O_DIRECT` / `F_NOCACHE`, 8 MiB requests), random 4 MiB uncached reads (median, p90), and a buffered cold read (Linux `fadvise(DONTNEED)`, macOS `F_NOCACHE` writes). The scratch file is removed on every path. It uses `rustix` (safe syscall wrappers) instead of `unsafe` FFI. On this M3 Pro: 5.7 GB/s uncached, 6.1–6.6 GB/s buffered cold, 0.83 ms per random 4 MiB read, and direct I/O at 0.87–0.93× buffered.

### `qubivex-model` (Phase 3, step 1 of 3: containers)
- `safetensors::parse` and `gguf::parse` (v2/v3, little-endian) read headers as untrusted input. The SafeTensors header is capped at 100 MiB. GGUF has caps on strings, keys, arrays, metadata and tensor counts, each also bounded by the bytes left in the file, and arrays nest at most 2 deep. Each tensor's byte length must equal `DType::storage_bytes(shape)`. Offsets are checked against the real file size. SafeTensors tensors must tile the data region exactly. GGUF offsets must be aligned and must not overlap. Duplicate keys and tensors are rejected. GGUF dims are reversed to row-major.
- `TensorIndex` holds tensors by name across files, with file sizes, GGUF metadata, totals and per-dtype bytes. A duplicate tensor across files is an error and leaves the index unchanged.
- `write::{safetensors, gguf}` produce small files for tests and fixtures.
- Tests cover round trips, one rejection per rule, and proptests of corrupted and truncated files and random bytes (no panics, accepted tensors stay in the file). An env-gated `real_files` test (`QUBIVEX_REAL_MODEL_FILES`) indexed the `hf-internal-testing/tiny-random-{Olmoe,Qwen3Moe}ForCausalLM` checkpoints cleanly.
- **Finding for the tensor schema.** Those transformers-5 checkpoints store each layer's experts fused (`mlp.experts.gate_up_proj [E, 2I, H]`, `down_proj [E, H, I]`), while released checkpoints store them per expert (`mlp.experts.{e}.gate_proj.weight`). The router name differs (`mlp.gate` / `mlp.router`), and q/k-norm is full-width in one model and per-head in the other. `transformer_moe` must take both layouts and the names from configuration. A fused tensor's expert slice is still one contiguous byte range.

### `qubivex-model` (Phase 3, step 2 of 3: manifest, registry, `transformer_moe`, validation)
- `Manifest` v1 (`deny_unknown_fields`). It checks the schema major, a `[a-z0-9._-]` id, and relative paths with no `..`/absolute/drive/backslash forms. It checks sha256 format and the GGUF single-shard rule. The JSON Schema is published at `docs/model-support/manifest.schema.json`, and a test keeps it in step with the type.
- `ArchitectureRegistry`: register (duplicate = error), resolve (same major, minor not newer), list, and unregister (tests). The `ArchitectureFactory` trait has `parse_config` and `tensor_schema`; `build` comes in Phase 4.
- `TensorRole`: embedding, head, final norm, layer norms, attention parts, router, `RoutedExpert{layer, expert, part}`, and `RoutedExpertStack{layer, part, experts}` (expert `e` is the `e`-th equal slice).
- `transformer_moe` reads dims from `config.json` names, or from GGUF `<general.architecture>.*` keys (the vocab may come from the token list). `overrides.config` fills gaps, and a value that contradicts the checkpoint is an error (relative 1e-6). `architecture.config` has `tensor_names` (preset `hf` / `hf_fused` / `gguf`, or `{base, name: pattern}`), `qk_norm` (`none`/`full`/`per_head`) and `router_scoring`. Unsupported features are refused with a reason: dense or shared-expert layers, attention bias, non-SiLU activations, and RoPE scaling. Products are checked.
- `ModelMetadata::load` reads the manifest, checks shard sizes, indexes the files, reads `config.json` (4 MiB cap), resolves the factory, builds the schema, and validates. Every missing, misshapen and unexpected tensor is reported at once; `overrides.ignore_tensors` allows extras. `bytes()` gives total, routed-expert, dense and largest-expert bytes.
- **Extensibility check.** The two `transformers` tiny checkpoints load through the same code with only manifest differences: OLMoE-style (`hf_fused`, full q/k-norm) and Qwen3-MoE-style (`hf_fused` with a router-name override, per-head q/k-norm). Every tensor has a role.
- **Deviations.** Dimensions are never in the manifest, only in config or GGUF; ADR-004 permits them as cross-checked duplicates, but v1 has no such fields. Facts the checkpoint omits (q/k-norm, GGUF top-K renormalisation) are in `architecture.config` / `overrides.config`, for `models import` (step 3) to infer.

### `qubivex-model` and CLI (Phase 3, step 3 of 3: discovery, import, fixtures)
- `discover(roots)`: each root and its subdirectories that hold `manifest.json`, loaded and validated. `find(id|dir)` reports ambiguous ids.
- `import(dir|file.gguf)` finds shards (`model.safetensors.index.json`, `*.safetensors`, or one `.gguf`). It tries each layout preset (plus the alternative router name) and each q/k-norm mode against the real tensors, and keeps the first that validates. Config errors are reported directly, and if nothing validates it reports the closest candidate's problems. `--set KEY=VALUE` supplies values the checkpoint lacks. It records shard sizes and the dominant dtype.
- CLI: `qubivex models list [--dir]... [--json]`, `inspect <id|dir> [--json] [--tensors]`, `import <path> [--id] [--name] [--set]... [--write [--force]]`, and `architectures`. Model dirs come from `--dir`, else `$QUBIVEX_MODELS_DIRS`, else `~/.qubivex/models`.
- Fixtures in `fixtures/models/` (~900 KB), from `cargo xtask gen-fixtures`. They are deterministic, and `cargo xtask check` regenerates them and fails on drift. Manifests come from the importer.
  - `olmoe-tiny`: per-expert `hf` layout, full q/k-norm.
  - `olmoe-tiny-gguf`: the same weights as GGUF Q8_0/F32 with `olmoe.*` keys and `--set norm_topk_prob=false`.
  - `qwen3-moe-tiny`: `hf_fused`, per-head q/k-norm, `head_dim` ≠ hidden/heads.
- The importer also recognised the `transformers` tiny checkpoints (downloaded, not committed) with no hand-written manifest.
- **Deviation from MIGRATION_PLAN §3.** Fixtures are written by our own writer, with configs that mirror the released checkpoints' keys, not by pinned `transformers` scripts. There is no Python ML stack here, and the `transformers` tiny models are 13–39 MB. Numerical oracles for these fixtures (Phase 4) will still need a reference implementation.
- `xtask` may now depend on `qubivex-core` and `qubivex-model`, for the generator. Nothing depends on `xtask`.

### Phase 3 exit check (2026-11-03): real OLMoE GGUF
- `allenai/OLMoE-1B-7B-0125-Instruct-GGUF`, file `OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf`: 4,213,512,192 bytes, sha256 `4ddc0e53…c684f`, matching the published LFS object. Stored at `~/.qubivex/models/olmoe-1b-7b-0125-instruct/`.
- `qubivex models import … --set norm_topk_prob=false --write` found the layout (`gguf`, `qk_norm: full`) from headers alone, in 0.02 s. Without `--set`, it names the missing value; the CLI now adds a `--set KEY=VALUE` hint.
- `models list` and `inspect` show 16 layers, hidden 2048, vocab 50304, 64 experts (top 8) and context 4096, matching the published config.
  - 195 tensors, every one with a role: 3.63 GiB routed experts, 296.62 MiB dense.
  - Dtypes: q4_k 2.99 GiB, q6_k 946.85 MiB, f32 8.51 MiB.
  - Largest expert 4,079,616 bytes: two Q4_K 1024×2048 matrices plus a Q6_K 2048×1024 one.
- The env-gated `real_files` tests pass on the file and the model directory.

### Phase 4, step 1: `qubivex-compute` and `qubivex-backend-cpu`
- `qubivex-compute`:
  - `WeightView` is a validated `rows × cols` view of stored weights; its byte length equals `DType::storage_bytes` and row slicing is checked.
  - `ComputeBackend` has `matmul` (`s` activation rows) and `dequant_row`, in host-memory form. Device buffers join it with the first GPU backend.
- `qubivex-backend-cpu` covers f32, f16, bf16, Q8_0, Q4_K and Q6_K (every dtype in the OLMoE Q4_K_M GGUF and the fixtures).
  - Each row is dequantized exactly, as `ggml-quants.c` evaluates it, then dotted with f32 activations using the one fixed reduction shape: 64 lanes, chunks in `k` order, a defined tail in lanes `0..rem`, and a fixed halving tree.
  - Work is split across output rows only, on rayon.
  - On x86-64 the same source is also compiled with `avx2,fma` and selected at run time. That one feature-gated call is the crate's only `unsafe`.
  - Hand-written NEON and AVX2 kernels, the spinning thread pool and `gemm`-crate prefill (ADR-002) are a later, measured optimisation that must reproduce these bits.
- Tests:
  - f16 to f32 conversion is checked exhaustively over all 65,536 values.
  - Q8_0, Q4_K and Q6_K dequantization matches **gguf-py bit for bit** (`fixtures/oracle/ggml_dequant.json`).
  - For all six dtypes, including a 32-column tail, batched equals one-at-a-time, 1, 3 and 7 threads give identical bits, portable equals dispatched, and results match an f64 reference to within 1e-6 relative.
- Oracle environment: `~/.qubivex/oracle` with torch 2.15.0, transformers 5.22.1, safetensors 0.8.0 and gguf 0.19.0, outside the repository; see `fixtures/oracle/README.md`. Generated reference files are committed, and tests never need Python.
- `rayon` is a new workspace dependency.

## Remaining in Phase 2
- Check `qubivex system` and `qubivex doctor --storage` output on a real Linux host.
- Caching `StorageMeasurement` per device waits for a state directory (config work, Phase 3 or later).

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
- `cargo test --workspace`: 98 passed, 0 failed on Rust 1.99.1 and 1.85 (2026-11-03, Phase 4 step 1). Before that: 92. Earlier: 91 passed on Rust 1.99.1 and 1.85. Breakdown: core 22 (including a proptest over untrusted dims); hardware 24 (including live smoke and storage tests); model 23 (including proptests over corrupted containers and the committed fixtures); CLI 14; xtask 9. The env-gated `real_files` tests are skipped without `QUBIVEX_REAL_MODEL_*`.
- Cross-checks: `cargo clippy -D warnings` and `cargo +1.85 check` for `x86_64-unknown-linux-gnu` and `x86_64-apple-darwin`.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo fmt --check`, `cargo doc` (`-D warnings`), `cargo deny check` and `cargo xtask check`: clean.
- `cargo +1.85 check/test --workspace --all-targets`: clean.

## Benchmark results
ADR-002 spike, M3 Pro: [`docs/performance/ADR-002-SPIKE.md`](docs/performance/ADR-002-SPIKE.md).

## Architectural decisions
ADR-001 … ADR-008, all Proposed. ADR-001 is implemented as written. ADR-002 has spike data and a preliminary choice (hybrid without Candle), pending the Colibri baseline.
