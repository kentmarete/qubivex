# QUBIVEX Implementation Status

Last updated: 2026-11-03

## Current phase
**Phase 8 (GPU acceleration, Metal first): slices 1–4 done (2026-11-03).**

**Phase 8 slice 4 done (2026-11-03): small expert caches.**
- Code: `MetalBackend::submit_async` and `Pending` (keeps wrapped host memory alive until the GPU is done; dropping waits; host reads and writes drain outstanding work). `PlacementReport::fallback`: with `--backend metal`, a plan whose pools are smaller than top-k runs the CPU pass and says why (`run` prints it; `bench` records it).
- Measured at `--ram-budget 1GiB` (ADR-011 amendment 3): every way of overlapping streamed experts with GPU work was slower than the CPU pass (async per expert 8.3–9.0 tok/s, hits on GPU and misses on CPU 14.7, waves 14.7, CPU 20.3), because each lone GPU submission costs about 0.5 ms and every layer needs a host round trip for routing. `--backend metal` at 1 GiB now gives 19.96 tok/s (the CPU pass) instead of 14.66.
- Tests: 241 pass. New: async ordering and keep-alive; the wave test now shrinks pools from 8 slots to 1 under critical pressure after load (the plan keeps whole layers, the waves run, and the bits match through 300 decode steps), and checks that a plan with one-slot pools falls back.
- **Next (slice 5):** routing on the GPU, so a token is one submission, and a concurrent encoder. Then CUDA. Phase 7 slice 1 is done and its exit criterion is met; its remaining items (a disk-saturation throttle, deeper lookahead) are parked.

**Phase 8 slice 3 done (2026-11-03): whole layers on Metal, bit-identical, 1.68× decode with experts resident.**
- Portable numerics (owner's choice; the CPU's bits changed once): `qubivex_compute::math::exp` (IEEE basic ops and fma, ≤ 2 ulp, subnormal results are 0), `rms_norm` on the shared 64-lane reduction (`qubivex_compute::reduce`, moved from backend-cpu), and RoPE from a host cos/sin table evaluated in f64. Real OLMoE: the 128 bench tokens match the previous reference and Colibri's 12 tokens still match. Fixtures: transformers error 7.2e-7 → 6.5e-7, f64 reference 3.3e-7 → 2.8e-7.
- Code:
  - `qubivex-backend-metal`: forward-pass kernels (norm, RoPE, clamp, add, copy, SwiGLU, exp, KV store, attention, MoE combine, row gather/scatter), all with `precise::` division and sqrt; `MetalBackend::submit` records many ops over `DeviceVec`s into one command buffer (`program.rs`); block-wise Q4_K/Q6_K kernels (Q4_K needs 16-byte alignment).
  - `qubivex-expert-stream`: page-aligned slabs (`Slab`, with an attachment slot dropped before reuse); `qubivex-compute`: `PageAligned::resized`.
  - `qubivex-arch`: `KvCache` device slot (`device_mut`, `advance`, `set_device_bytes`), `ops::route` shared by both passes, `layer_name` public.
  - `qubivex-runtime`: `MetalForward` (one submission per layer; routing on the host; experts wrapped in place once per load; waves when a pool is smaller than the routed set; device KV in 256-position chunks, charged). `MetalOps::layers`, now the default of `--backend metal`.
  - CLI: `--metal-ops layers`.
- Tests: 240 pass (Rust 1.99.1; 1.85 checks). New: `ops_parity.rs` (every op bit-identical, including exp over −110..110), gather/scatter, misaligned Q4_K is declined, `metal_layers_match_the_cpu_in_waves_and_across_kv_growth` (one-slot pools, 300 decode steps past 256 positions, three fixtures). Env-gated: whole layers on real OLMoE give bit-identical prompt logits and 48 identical tokens; all 130 tensors stay bit-identical with the new kernels. GPU tests now fail (not skip) on a shader compile error.
- Measured (M3 Pro, OLMoE Q4_K_M, ADR-011 amendment 2; records `2026-11-03-m3pro-phase8s3-*.json`): all resident, decode 26.41 → 44.47 tok/s (median step 37.86 → 22.49 ms, P95 47.80 → 23.36 ms), prefill 33 → 77 tok/s. At `--ram-budget 1GiB`, Metal is slower (14.66 vs 20.31 tok/s): the pools hold 2–3 experts, so each layer runs about three waves that each wait for all their loads. Kernel throughput (`metal_stream`): Q4_K 23 → 38 GiB/s, Q6_K 38 → 64 GiB/s.
- Decision (pending owner review): the CPU stays the default backend; `--backend metal` means whole layers.
- Slice 4 (above) handled small expert caches.

**Phase 8 slice 2 done (2026-11-03): Metal in the engine (batching, zero-copy weights, placement, token parity).**
- Code:
  - `qubivex-compute`: `ComputeBackend::matmul_many` (independent matmuls as one submission; the default loops) and `host::PageAligned` (a page-aligned region from an over-allocated `Vec`, no unsafe).
  - `qubivex-backend-metal`: one command buffer per batch, each distinct input staged once, staging buffers reused; `MetalBackend::map_host` wraps a host region in place (`newBufferWithBytesNoCopy`), `MetalHost::weights` gives a matrix at an offset in it, and `matmul_many_resident` runs a batch. Buffers hold the host region's `Arc`.
  - `qubivex-arch`: q, k and v are one `matmul_many`, and so are each expert's gate and up; `TransformerMoeModel::dense_matmuls` lists the placeable weights by kind.
  - `qubivex-runtime`: dense tensors are read into one page-aligned arena (the ledger still charges tensor bytes); `placement::Placed` routes each matmul by weight (Metal for the placed ones, the CPU for the rest); `EngineOptions::{backend, metal_ops}`, `Engine::placement()`.
  - CLI: `run` and `bench` gain `--backend cpu|metal` and `--metal-ops attention,router,head` (default `head`); `bench` records `config.device`.
- Tests: 229 pass on Rust 1.99.1 and 1.85. New: batched and wrapped parity (every dtype, unaligned offsets, a shared input, shrinking batches), batch shape errors, the page-aligned region, and the fixture invariant `metal_placement_never_changes_bits` (every placement, three fixtures, prompt logits and 12 tokens). Env-gated: `engine_on_metal_gives_the_cpu_tokens` places all 81 dense matmul weights of OLMoE on Metal, and the prompt logits are bit-identical and the 48 generated tokens identical. The slice-1 per-tensor parity, `ledger_rss`, `real_model` and `reference` still pass.
- Measured (M3 Pro, all resident, 160 tokens; ADR-011 amendment 1): one submission costs 109 µs (120 before staging reuse); q, k and v in one command buffer take 424 µs against 727 µs as three. End to end, decode is cpu 25.51, metal head 25.07, attention + head 22.16, all dense 21.01 tok/s, with every run's tokens matching the Phase 7 reference. The head's kernel wins in a loop (1.78 vs 2.77 ms), but in decode it takes 3.99 ms against 3.18 ms, because the GPU idles between tokens. An A/B against `main` shows the dense arena costs nothing on the CPU (26.05 vs 26.05/26.13 tok/s).
- Decision: the CPU stays the default; `--backend metal` is a bit-identical opt-in.
- Slice 3 (above) kept the GPU busy with whole layers.

**Phase 8 slice 1 done (2026-11-03): the Metal kernels and their parity.**
- Code:
  - New crate `qubivex-backend-metal` (objc2-metal; macOS only, an empty stub elsewhere). Kernels in `src/kernels.metal`, compiled from source at start-up with `MTLMathModeSafe` and contraction off; no offline Metal toolchain needed.
  - matmul for f32, f16, bf16, Q8_0, Q4_K and Q6_K: one 32-thread SIMD group per output, reproducing the CPU's dequantised values and 64-lane reduction shape.
  - `MetalBackend` implements `ComputeBackend` (it copies weights per call: a correctness path); `MetalBackend::upload` and `matmul_resident` keep weights on the device. `dequant_row` stays on the CPU.
  - `ComputeError::Device`. Dependency rule: a GPU backend may dev-depend on `qubivex-backend-cpu` as its parity reference (`xtask` test added).
- Tests: synthetic parity (all six dtypes, tails of 32–37 lanes, s = 1, 3, 8, typical and wide-range values) is bit-identical; resident and copying paths agree; errors are reported. The env-gated `qubivex-runtime/tests/metal_parity.rs` is bit-identical on all 130 real OLMoE tensors (3.92 GiB, about 11M outputs).
- Finding: Apple GPUs flush f32 subnormals to zero (safe mode too). That is the only divergence; `subnormals_are_the_only_divergence` documents it. Parity contract: ADR-011 (Proposed).
- Kernel benches (`cargo bench -p qubivex-backend-metal --bench kernels`), resident weights, against the CPU baseline at 5 threads: expert Q4_K 1024 × 2048 decode 176 µs vs 63 µs; Q6_K head 1.78 ms vs 2.77 ms; Q4_K prefill s = 128 4.84 ms vs 3.44 ms; a 64 × 64 matmul costs 120 µs, which is per-call synchronisation.
- Slice 2 (above) built the batching, zero-copy weights, placement, `--backend` and token parity.

**Phase 7 slice 1 done (2026-11-03): router-lookahead prefetch.**
- Code:
  - `qubivex-expert-stream`: `predict::Lookahead` (EMA 0.3, confidence 0.92, width 1; pure, unit-tested); `ExpertStore::prefetch(keys, guard)` at `Priority::Prefetch`; `cancel_prefetch`; `policy::speculation_beats` (Colibri's guard); `StoreStats::prefetch`.
  - Store rules: a prediction never evicts its own experts; demand promotes its queued prefetches and drops the stale ones of the layer; a demand waits for in-flight prefetches instead of failing `Full`.
  - `qubivex-arch`: `ExpertProvider::{wants_lookahead, lookahead}`. During decode, the next layer's norm and router run on the post-attention state; routing never reads the result.
  - `qubivex-runtime`: `EngineOptions::prefetch`, the predictor state in the engine, issue after the layer's demand, recall and precision (`Engine::prefetch_report`), `Engine::end_turn`, and a stop under memory pressure.
  - CLI: `run` and `bench` gain `--prefetch`, `--prefetch-guard on|off`, `--prefetch-width`, `--prefetch-confidence` and `--prefetch-smooth`; `bench` records `config.prefetch` and `experts.prefetch`.
- Tests: 217 pass, plus 3 loom models. New: 11 store prefetch tests (50 repeated runs, no failure), the predictor unit tests, and the Phase 7 exit test `prefetch_never_changes_routing_or_bits` (router logits, experts, weights and logits bit-identical on the three fixtures across every prefetch setting).
- Real model, M3 Pro, 160 tokens, tokens identical to the Phase 6 exit record. Recall is 79.7%. At 1 GiB, decode goes 19.87 → 20.32 tok/s (guard on) or 20.50 (guard off), and blocked time per token goes 8.29 → 4.87 or 3.95 ms. At 2–3 GiB and all resident there is no gain. Decode at 1 GiB is limited by read throughput, so hiding latency gains little. Prefetch stays opt-in; the guard is on by default. Tables: ADR-008 amendment 2.
- Not built: a disk-saturation throttle, lookahead beyond one layer, and layer-0 prediction for the next token.

**Engineering Improvement Plan: Phase 0 audit and the tensor-descriptor step (Phase 4.5), done (2026-11-03).**
- Audit: `docs/engineering/BASELINE_AUDIT.md`. Reconciliation: `docs/engineering/ARCHITECTURE_DECISIONS.md`. The EIP is folded into `MIGRATION_PLAN.md` §3a.
- Owner decisions: one roadmap (MIGRATION_PLAN); lossless by default with named lossy opt-ins only (ADR-010); Metal first (ADR-007 amendment); tensor scope = descriptors, views and migration (ADR-009).
- Code: `TensorDesc` in `qubivex-core`; `TensorView`, `TensorViewMut` and `Tensor<T>` in `qubivex-compute`; `RawTensor::desc()`; the runtime's dense weights on `Tensor<u8>`; `WeightView: TryFrom<TensorView<u8>>`. Kernel signatures are unchanged.
- Tests: 145 pass (123 before), plus the env-gated real-model generation and expert tests.
- Decode, OLMoE Q4_K_M, 160 tokens, 3 runs: 19.98 / 20.03 / 20.81 tok/s before and 21.33 / 21.24 / 21.06 after. The output is byte-identical (sha1 `dc51dd49`).
- **In-workspace benchmarks, done (2026-11-03).** Criterion kernel benches (`cargo bench -p qubivex-backend-cpu --bench kernels`) and `qubivex bench` (prefill and decode with build, hardware, median/P95, peak RSS and a token check, as a JSON record). Baseline on the M3 Pro, warm, 5 threads: decode 25.8 tok/s at the median step (P95 step 51.6 ms vs median 38.8 ms); prefill 44 tok/s at 128 tokens. Q4_K decode is limited by compute (17 GiB/s whether the matrix is in cache or in DRAM, against 80 GiB/s for bf16). Protocol and tables: `docs/performance/BENCHMARKING.md`; records: `docs/performance/baselines/`.
- **Next:** Phase 5 (memory planner), with measured kernel follow-ups (MIGRATION_PLAN §3a).

**Phase 5: memory planner, slice 1 of 3 done (2026-11-03).**
- Code: the `qubivex-memory` crate.
  - `ModelFootprint` is derived from tensor roles only. The KV cost per token is the K/V projection width × 4 bytes.
  - `plan()` gives one RAM budget per profile (performance 0.88, balanced 0.75, saver 0.50, custom), with the room the KV and expert caches share and admission that keeps one expert slot per layer.
  - Non-uniform per-layer slots; `MemoryLedger` with RAII charges; `qubivex plan <model> [--ram-budget] [--ctx] [--profile] [--sequences] [--json]`.
- Tests: the fixture footprints agree with the model's byte summary and the architecture's head counts. The admission test keeps the floor. Property tests (2,000 cases each) show plans never exceed their budget at any KV length, refusals are real, and the slot distribution is tight.
- Real model, M3 Pro: every expert (3.63 GiB) fits beside a 1 GiB full-context KV in a 5.53 GiB budget. At `--ram-budget 2GiB` the cache shrinks from 1.21 GiB to 215 MiB at full context; at 1 GiB the context is shortened to 629 tokens.

**Phase 5 slice 2 done (2026-11-03): the engine uses the plan.**
- `Engine::load` plans with the dense weights resident. Before each forward pass, the store's capacity is lowered to make room for the projected KV (`ExpertStore::set_capacity`). The KV grows in 256-position chunks. The ledger is charged for dense, expert-cache and KV bytes.
- `run` and `bench` gain `--profile`, `--ram-fraction` and `--ram-budget`; `--expert-cache-mb` still fixes the cache. The `run` stats line and the `bench` record show the plan's budget and the ledger.
- Tests (all pass):
  - room sharing is bit-identical to all-resident on the three fixtures, with a monotonic capacity that never drops below the floor;
  - the store shrink test;
  - the ledger-against-RSS test on the real model: the ledger is 3.90 GiB against 3.75 GiB of RSS growth, and 510 MiB against 441 MiB under a 1 GiB budget.
- Real model: output is byte-identical (sha1 `dc51dd49`) both all-resident and under `--ram-budget 1GiB`. Decode is unchanged (median step 38.63 ms against the 38.82 ms baseline, and the tokens match the baseline record).
- Known limit, for Phase 6: under a 1 GiB budget the global LRU gets 0 hits (12.0 tok/s). Per-layer pools and LFRU (ADR-008) fix this.

**Phase 5 slice 3 done (2026-11-03): the pressure monitor. Phase 5 is complete.**
- `PressureMonitor` levels, each adding to the one before:
  - Normal;
  - Elevated (below 1 GiB available): stop prefetch;
  - High (below 512 MiB): shrink the cache by the shortfall;
  - Critical (below 256 MiB): cache at the floor, refuse new requests.

  It escalates at once and recovers only past a 256 MiB hysteresis margin. It never kills an active request.
- Runtime: a sampling thread (500 ms, stopped on drop). The cap applies between tokens, and `generate` checks `Engine::admit`. The pressure level is in `run`'s stats line.
- Tests (all pass):
  - unit tests for order, hysteresis and the floor;
  - a property test (2,000 random sequences);
  - a fixture test: critical pressure drops the cache to the floor with bit-identical logits, refuses a new request, and recovers;
  - the sampling thread stops with the engine;
  - the opt-in RAM-hog test on the real model (`QUBIVEX_PRESSURE_TEST=1`), which passed 3 of 3 runs.
- Finding: macOS compresses memory, so available RAM falls by only 50–65% of an incompressible allocation, and by far less for uniform bytes. See ADR-005 amendment 3.
- **Next:** Phase 6 (full expert streaming). The first target is the per-layer slab pools and LFRU of ADR-008: under a 1 GiB budget the global LRU gets 0 hits. Measured kernel follow-ups remain in MIGRATION_PLAN §3a.

**Phase 6 slice 3 done (2026-11-03): direct I/O, the I/O matrix and the exit check. Phase 6 is complete.**
- Code (`qubivex-runtime`, ADR-006 amendment 2): `ExpertIo::{Buffered, Direct}` and `IoPath`. Direct I/O uses `F_NOCACHE` on macOS (straight into the slab) and `O_DIRECT` on Linux (the aligned window around each extent goes into a per-thread bounce buffer, then is copied out). It is refused, with an error, where unsupported. `--direct-io` on `run` and `bench`; `config.expert_io` in the bench record.
- Tests: the window reader under a strict aligned reader that also returns short reads (unaligned extents, an unaligned end of file, a read past the end); direct bytes equal buffered bytes on every fixture expert and on the real model (env-gated); direct rows in the engine's bit-identity test.
- Warm I/O matrix at `--ram-budget 1GiB` (identical tokens in every row): buffered 15.5 / 18.1 / 20.5 / 20.5 / 20.5 tok/s at 0 / 1 / 2 / 4 / 8 I/O threads; `F_NOCACHE` 15.4 / 17.8 / 20.3 / 20.2 / 19.3. The model stayed in the page cache, so this is no cold measurement; the default stays buffered.
- Exit check: 160 tokens at 1 GiB match the all-resident record; 19.9 tok/s, 10.4% hits, 427 MiB read per token, 0 slab allocations (`docs/performance/baselines/2026-11-03-m3pro-phase6-exit-1gib.json`).
- Deferred: io_uring and cold `O_DIRECT` rows (a Linux host), multi-drive mirrors, per-device queues.

**Phase 6 slice 2 done (2026-11-03): the I/O queue, I/O threads and `acquire_many`.**
- Code (`qubivex-expert-stream`, ADR-006 amendment 1):
  - `acquire_many` returns a `Batch` that leases hits, queues every miss with a free slot before waiting, and yields leases as loads finish;
  - a priority queue (Demand > Prefetch > Warm) with `with_io_threads(n)`;
  - read errors reach every waiter, and dropping a batch cancels its unstarted loads;
  - without I/O threads, the caller reads on demand.
- Arch and runtime: `ExpertProvider::with_experts` (order-free; the MoE layer requests a whole layer); the provider pipelines through `acquire_many`; within-expert extent coalescing in `ModelExpertSource`; `--io-threads` (default 4). The bench records the blocked time per token and the I/O threads.
- Tests: 197 pass, plus 20 real-model runtime tests and 3 loom models. New: the I/O stress suite (exactly-once batches, duplicate requests, contention, cancellation, error delivery, parallel reads, shutdown; 50 repeated runs, no failure), I/O-thread bit-identity on the fixtures, and coalescing.
- Real model, M3 Pro, `--ram-budget 1GiB`: decode goes from 14.9 to 20.2 tok/s (blocked time 7.6 ms per token), and prefill at 128 tokens from 42.7 to 49.5 tok/s, with identical tokens. All resident: unchanged (38.06 ms step).
- **Next:** slice 3: direct I/O (`F_NOCACHE`/`O_DIRECT`) and io_uring behind options, the I/O strategy matrix, and the exit check (OLMoE correct and benchmarked well below its expert size).

**Phase 6: full expert streaming, slice 1 done (2026-11-03): per-layer pools and LFRU.**
- Code (`qubivex-expert-stream`, ADR-008 amendment 1):
  - one pool per expert layer (`PoolSpec`: slots and slot size); a miss evicts only within its layer;
  - slab reuse: a miss reads into the evicted expert's buffer, so a streaming decode allocates no expert memory once the pools fill;
  - `Policy::LFRU` (default; heat per layer, halved every 256 accesses) or `Policy::Lru`; `policy::lfru_score` and the 25% + 4 hysteresis `policy::beats` are pure and unit-tested;
  - trace recording (`ExpertStore::record`) and `replay`, which runs a trace through the real store over a null source.
- Runtime and CLI: the engine sizes the pools with the plan's `distribute` and resizes them as the KV grows (`set_pools`). `run` and `bench` gain `--cache-policy`; `bench` gains `--routing-trace` and records the hit rate, bytes per token, slab allocations and per-layer slots. Replay: `cargo run --release -p qubivex-expert-stream --example replay -- trace.txt`.
- Tests: 186 pass, plus 18 real-model runtime tests. New: per-layer isolation, LFRU scan resistance, slab reuse at every slot count and policy, recording, replay against a global LRU, and per-layer pools in the engine's room-sharing test. The loom checks pass.
- Real model, M3 Pro, `--ram-budget 1GiB` (149 MiB of slots: 9 layers × 3, 7 × 2): the hit rate goes from 0% to 10.9%, reads per token from 479 to 427 MiB, decode from 12.1 to 14.9 tok/s. Tokens are identical. Pool LRU alone stays at 0% (experts are visited in ascending index order). All resident: no change (38.04 ms median step against 38.82 ms).
- **Next:** slice 2, the I/O queue (Demand > Prefetch > Warm), the worker pool and `acquire_many` (issue a layer's misses before waiting), then direct I/O, io_uring and the I/O strategy matrix.

**Phase 4: CPU inference prototype, done.** `qubivex run` generates correct text on CPU from the real OLMoE GGUF, and every numerical test passes:
- against `transformers` the worst relative error is ≤ 7.2e-7, with identical experts and greedy tokens;
- against an independent f64 reference it is ≤ 3.3e-7;
- the expert-cache sweep gives bit-identical logits on the fixtures and the real model.

**Next: Phase 5, the memory planner.** A measured performance step (SIMD kernels, spinning thread pool, `gemm` prefill, a Colibri benchmark) can come before or alongside it.
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

### Phase 4, step 2: `qubivex-tokenizer` and a minimal `qubivex-runtime`
- `qubivex-tokenizer` wraps Hugging Face `tokenizers` 0.23.2. It is pure Rust: `fancy-regex` instead of Oniguruma (C), and no esaxx C++ code.
  - `from_hf_dir` reads `tokenizer.json`, special tokens and the chat template from `tokenizer_config.json`, and EOS ids from `generation_config.json` / `config.json`. File sizes are capped.
  - `from_gguf` rebuilds a byte-level BPE from `tokenizer.ggml.*`. Token types map to the official flags: control → special and not normalized; user-defined → added and normalized; unused → vocabulary only.
  - Only GGUF pre-tokenizer types verified against an official `tokenizer.json` are accepted (`olmo` with NFC; `gpt-2` and `default` without). Anything else is refused with "supply tokenizer.json".
  - `apply_chat_template` runs MiniJinja 3 in a sandbox: no loader (no `include`), a fuel limit, `trim_blocks`/`lstrip_blocks` as `transformers` sets them, Python string methods (`minijinja-contrib` pycompat), and `raise_exception`.
  - `StreamDecoder` holds back incomplete UTF-8 when decoding token by token.
- `qubivex-runtime` (new): `load_tokenizer(&ModelMetadata)` and `gguf_vocab`. The tokenizer crate may depend only on `core`, so the runtime converts GGUF metadata to plain data.
- **Verified against `transformers` 5.22.1** (`fixtures/oracle/olmoe_tokenizer.json`, from the official OLMoE-1B-7B-0125-Instruct `tokenizer.json`). The corpus covers 17 strings: NFC/NFD accents, runs of spaces (the user-defined tokens), special-token text, emoji, CJK, code, numbers, CRLF and contractions. Both QUBIVEX's tokenizer rebuilt from the real GGUF and the official `tokenizer.json` loaded by QUBIVEX produce identical ids and decoded text. The chat template renders the identical string with identical ids, and EOS is 50279. These are env-gated tests (`QUBIVEX_OLMOE_GGUF_DIR`, `QUBIVEX_OLMOE_TOKENIZER_JSON`).
- Fixtures now share one byte-level BPE tokenizer (256 GPT-2 byte symbols, no merges, EOS = byte 1, a chat template) in both `tokenizer.json` and GGUF form, so the SafeTensors and GGUF fixtures tokenize identically.
- **Policy changes for review:**
  - `deny.toml` ignores RUSTSEC-2024-0436: `paste` is unmaintained, a compile-time macro used by `tokenizers`, with no vulnerability.
  - The model-name guard accepts a single line marked `model-name-ok: <reason>`, used once, for the GGUF format identifier `"olmo"`.

### Phase 4, step 3: `qubivex-expert-stream` (minimal) and expert extents
- `ExpertStore` (ADR-006, Phase 4 form):
  - Synchronous `acquire` returns a `Lease` that releases on drop and is not `Clone`.
  - The capacity is in bytes, and the least-recently-used unleased experts are evicted first; expert sizes differ by layer in Q4_K_M.
  - Loads run outside the lock. In-flight dedup: a second caller waits on a condvar for the first caller's load.
  - When everything resident is leased it returns `Full` instead of waiting, so one generating thread cannot deadlock itself. `TooLarge` covers an expert bigger than the whole capacity.
  - Stats: hits, loads, dedup waits, evictions and bytes read.
  - Expert bytes are `Arc<[u8]>`, so the store has no `unsafe`. Slabs, I/O threads, `O_DIRECT` and prefetch belong to Phase 6.
- **Loom.** Mutex, condvar and atomics come from `loom::sync` under `cfg(loom)`. The model checks (dedup loads once; eviction never takes a leased expert) explore 35 and 18 interleavings. A planted dedup bug (a loading slot treated as a miss) makes them fail. New CI job: `loom (expert store)`.
- `qubivex-runtime::ModelExpertSource` builds each expert's byte extents from the tensor roles: per-expert tensors, or the `e`-th slice of a stacked tensor, with parts in a fixed order. It records each part's dtype and shape and reads with `pread`.
- Tests:
  - Capacity sweep on synthetic experts: every capacity returns identical bytes, and all-resident loads each expert once.
  - LRU order, `Full`, `TooLarge` and load-error recovery.
  - 8 threads × 300 acquires stay within the capacity.
  - For all three fixtures, at three capacities, every expert read through the store is byte-identical to its tensors.
  - On the real OLMoE GGUF (env-gated), all 1,024 experts are located and read correctly through a 4-expert store that evicts.

### Phase 4, step 4: `qubivex-arch`, the engine, `qubivex run`
- `qubivex-arch` runs the `transformer_moe` forward pass:
  - embedding;
  - per layer: RMSNorm (f64 mean), Q/K/V, optional `clip_qkv`, q/k-norm (`full` or `per_head`), NeoX RoPE (inv_freq as `transformers` computes it), causal GQA attention against the KV cache, the output projection and the residual;
  - then RMSNorm, router softmax and top-K (ties to the lower id), optional renormalisation, SwiGLU experts, a combination **in router rank order**, and the residual;
  - final norm and head (tied or untied).
- Weights arrive only through `DenseWeights` (resident) and `ExpertProvider` (one expert at a time). The arch crate knows neither files nor the cache.
- Every computation has one fixed order per token, so **a batched prefill is bit-identical to feeding tokens one at a time**.
- A trace hook exposes embedding, attention, router logits, routing weights, MoE output, hidden states and logits for the numerical tests.
- `qubivex-runtime::Engine`:
  - resident tensors are read into memory in their stored dtype;
  - the expert store is the `ExpertProvider`, with a stacked gate-up split into gate and up rows;
  - the KV cache is f32;
  - `generate` does greedy decoding, or temperature + top-p with a seeded xorshift RNG;
  - EOS ids come from the tokenizer.
- `qubivex run --model <id|dir> --prompt … [--chat] [--max-tokens] [--temperature] [--top-p] [--seed] [--expert-cache-mb] [--context] [--quiet]` streams text and prints prefill/decode tok/s and expert-cache counters.
- **Real model (OLMoE-1B-7B-0125-Instruct Q4_K_M, M3 Pro):**
  - "The capital of France is" → " Paris.\n\nThe official language of France is French." These are exactly the 12 tokens of Colibri's `ref_olmoe_real.json`, which Colibri produced from int8 weights.
  - The output is identical with every expert resident, with a 512 MiB cache and with a 64 MiB cache (about 16 of 1,024 experts; 1,837 loads, 1,821 evictions).
  - A chat prompt gives a coherent answer at about 11 tok/s decode with the portable kernels.
- Tests on all three fixtures:
  - batched prefill is bit-identical to token-by-token;
  - an expert-cache sweep (1, 2 and 5 experts, all resident) gives bit-identical logits and greedy tokens, with the 1-expert cache evicting;
  - seeded sampling is reproducible;
  - errors (token out of range, context full, empty prompt) are reported;
  - `qubivex run` works on a fixture, with and without `--chat`.

### Phase 4, step 5: numerical exit tests
- **`transformers` 5.22.1 / torch 2.15.0, f32** (`fixtures/oracle/forward.json`; the loader asserts that every weight was loaded and none newly initialized):
  - checked at every layer and position: attention output, router logits, routing weights, MoE output and layer output, plus the final logits;
  - worst relative error: **4.6e-7** (OLMoE-style fixture) and **7.2e-7** (Qwen3-MoE-style fixture), against a 1e-5 bound;
  - selected experts identical at every layer and token, and 12 greedy tokens identical.
- **Independent f64 reference** (`tests/reference.rs`: a deliberately naive second implementation sharing only config parsing and the gguf-py-verified dequantizers):
  - worst logits error **2.1e-7 / 2.2e-7 / 3.3e-7** on all three fixtures, including the GGUF Q8_0 one that `transformers` cannot load;
  - expert selections identical.
- **Real model** (env-gated `tests/real_model.rs`):
  - greedy tokens equal Colibri's 12 reference tokens with every expert resident, at 256 MiB and at 32 MiB;
  - logits bit-identical across the three capacities.
- Not possible here: a `transformers` run of the full 7B model, which needs 14–28 GB to dequantize on an 18 GB machine. The real-model check is the Colibri token match plus the cache invariance.

## Phase 4 follow-ups (performance, measured)
- **Done: fused NEON kernels** (`qubivex-backend-cpu/src/neon.rs`) for Q4_K, Q6_K, Q8_0 and bf16, used for single-token matmuls (decode) on aarch64.
  - They reproduce the portable bits exactly: the same dequantized values (separate multiply and subtract), the same lane per element, the same chunk order, the same halving tree. The existing batched-versus-single tests compare them directly, across all dtypes and tails.
  - No `#[target_feature]` (NEON is baseline on aarch64). Each kernel is an `unsafe fn` with a single `unsafe` body, and lengths are asserted; this builds on Rust 1.85 (MSRV) and current Rust.
  - Measured on the M3 Pro, OLMoE shapes, warm, 11 threads. Q4_K expert gate/up: 146 → about 80–115 µs. Q6_K expert down: 246 → about 72–86 µs. Q6_K head (50304×2048): 8.1 → 2.0 ms. **Decode: about 11 → 14–18 tok/s.** Real-model output and the cache-invariance test are unchanged.
- **Done: the forward pass runs inside the engine's own rayon pool** (`Engine` owns a `ThreadPool`; `forward` runs under `install`).
  - The `sample` profiler showed the main thread calling each matmul from outside the pool. That is rayon's cold path: inject the job and block on a condvar, about 465 times per token, with the caller idle.
  - Default threads: the compute team from `qubivex-hardware::probe_cpu` (5 performance cores on the M3 Pro). Decode scaling stops at 5 threads; the efficiency cores add nothing. `qubivex run --threads N` overrides it.
  - `DenseWeights` and `ExpertProvider` are now `Sync`, and `Tracer` is `Send`.
- **Measured steady-state decode** (OLMoE-1B-7B Q4_K_M, M3 Pro, 160 tokens, every expert resident, same prompt):

  | Build | Decode |
  |---|---|
  | portable kernels (Phase 4) | 12.0–12.4 tok/s |
  | + NEON kernels | 18.6–19.3 tok/s |
  | + in-pool forward | 18.6–21.3 tok/s, about 1.7× overall |

  Short runs read lower, because the first tokens load experts from disk on demand.
- **Done: benchmark against Colibri `olmoe`** (`docs/performance/COLIBRI-OLMOE-BENCHMARK.md`): same GGUF, prompt, 160 tokens, 5 threads, warm, every expert resident.
  - QUBIVEX (lossless): **20.7–21.0 tok/s**.
  - Colibri default (lossy integer dot): 17.4–17.8 tok/s.
  - Colibri lossless (`GQ_EXACT=1`): 3.0 tok/s.
  - QUBIVEX's 160 tokens are **identical** to Colibri's lossless tokens. Colibri's default drifts after 86 tokens.
- Next candidates: one parallel region per layer for all of a token's experts; a faster lossless Q4_K kernel (about 20 GB/s, compute-bound); AVX2 kernels for Linux; streaming benchmarks with small, cold expert caches.
- Then: fused AVX2 kernels for Linux x86-64. Rosetta on this Mac reports no AVX2, so they can only be verified by CI's bitwise tests on Linux runners.
- Hand-written NEON and AVX2 kernels reproducing the portable bits, a spinning thread pool for decode, and `gemm`-crate prefill (ADR-002). Then a benchmark against Colibri `olmoe` on the same machine (MIGRATION_PLAN §4).
- Today: about 11 tok/s decode on OLMoE-1B-7B Q4_K_M (M3 Pro, every expert resident, portable kernels). A cold first prompt is slower because it reads experts from disk.

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
- `cargo test --workspace`: **145 passed**, 0 failed on Rust 1.99.1 and 1.85 (2026-11-03, after the tensor-descriptor step: core 35, compute 9 + 1 compile-fail doctest). Clippy `-D warnings` is clean natively and for `x86_64-unknown-linux-gnu`. Before that: `cargo test --workspace`: 123 passed, 0 failed on Rust 1.99.1 and 1.85 (2026-11-03, Phase 4 done). After step 4: 120; step 3: 110. Loom: 2 model checks. Env-gated real-model tests: tokenizer (3), expert extents (1) and generation with cache invariance (1), all passing. After step 2: 103; step 1: 98; Phase 3: 92. Loom: 2 model checks (`RUSTFLAGS="--cfg loom"`). Earlier: 91 passed on Rust 1.99.1 and 1.85. Breakdown: core 22 (including a proptest over untrusted dims); hardware 24 (including live smoke and storage tests); model 23 (including proptests over corrupted containers and the committed fixtures); CLI 14; xtask 9. The env-gated `real_files` tests are skipped without `QUBIVEX_REAL_MODEL_*`.
- Cross-checks: `cargo clippy -D warnings` and `cargo +1.85 check` for `x86_64-unknown-linux-gnu` and `x86_64-apple-darwin`.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo fmt --check`, `cargo doc` (`-D warnings`), `cargo deny check` and `cargo xtask check`: clean.
- `cargo +1.85 check/test --workspace --all-targets`: clean.

## Benchmark results
ADR-002 spike, M3 Pro: [`docs/performance/ADR-002-SPIKE.md`](docs/performance/ADR-002-SPIKE.md).

## Architectural decisions
Register and proposed status changes: `docs/engineering/ARCHITECTURE_DECISIONS.md` §1.
- Accepted: ADR-002 (hybrid, no Candle; the spinning pool and `gemm` prefill are not built yet), ADR-009 (tensor descriptors), ADR-010 (numerical precision).
- Amended: ADR-007 (Metal first; device assignment fixed per execution plan).
- Proposed: ADR-001, 003, 004 and 006 are implemented and proposed for Accepted, awaiting owner review. ADR-005 and ADR-008 are not built yet.
