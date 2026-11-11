# QUBIVEX Implementation Status

Last updated: 2026-11-11

## Current phase
**Vulkan submission plan, Step 0: measurement (2026-11-11, done, branch `vulkan-submission-audit`).** See docs/engineering/VULKAN_EXECUTION_AUDIT.md (§8 sequence, §9 decisions, §10 results).
- Built: device counters and opt-in GPU timestamps (`--vulkan-timestamps`), and a per-phase `device` block in `qubivex bench` records. Behaviour is unchanged without the flag; the tests pass, timed runs included.
- Measured on mains power (`baselines/2026-11-11-m3pro-vulkan-step0-*.json`, tokens identical to the reference):
  - Vulkan: 27.9–28.2 tok/s untimed at 5 threads, 28.8 at 8. CPU: 43.3.
  - GPU busy is 19.5–19.7 ms per token, idle about the same.
  - Submissions never overlap on the GPU.
  - Wall minus GPU busy leaves about 16 ms of GPU idle per token over about 113 serial hand-offs, ≈ 140 µs each. (A first reading of "220 µs fixed latency per submission" double-counted overlapping waits; corrected in the audit §10.)
  - Host submit work is about 18 µs per submission.
  - So the round trips bind, not the kernels.
- Step 1a (bulk copies) committed: neutral end to end. 1b and the other per-call clean-ups skipped (owner, 2026-11-11).
- Next: Step 2 (grouped experts), estimate ≈ 32 tok/s; 37 is a target, not an expectation.

**Vulkan kernel speed (2026-11-11; ADR-019 amendment 1, accepted 2026-11-11).** Decode on MoltenVK 4.2x faster with the same bits.
- Shader: Q4_K, Q6_K and Q8_0 decode a block at a time (scales once per block, each byte once for all the weights it holds); a Q4_K header is read as four words. No `precise`: `shaders/build.sh` marks every add, subtract and `fma` `NoContraction`, because SPIRV-Cross turned every `precise` multiply into an unoptimised helper call. Q4_K's `t - m` is `fma(t, 1, -m)` with `t` kept rounded by an integer OR.
- Device: 8 submitters (own command buffer, fence and host buffers), the queue locked only around `vkQueueSubmit`, so a layer's experts overlap their round trips.
- Measured, OLMoE Q4_K_M, mains power: decode 153.3 -> 36.55 ms per token (6.52 -> 27.36 tok/s), prefill 8.2 -> 48.9 tok/s, tokens identical. The CPU in the same session: 43.50 tok/s, prefill 77.0. Records `docs/performance/baselines/2026-11-11-m3pro-vulkan-*.json`; steps in ADR-019 amendment 1 and BENCHMARKING.md.
- Positive controls: fast math forced on, the parity tests pass; fast math on with only `Fma` decorated, 3 fail. Four rows per group, fence polling and more threads were measured and not kept.
- Now bounded by round trips, not kernels: about 297 submissions per token, the GPU busy about 74 µs of a 320 µs wait each, about 7 dependent round trips per layer.
- Tests: the 8 parity tests and the 4 runtime Vulkan tests (OLMoE included) pass; a new ignored timing test (`parity.rs`, `speed`).
- **Next:** whole layers on the Vulkan device (one submission per token, as Metal's token mode), or token tiles for prefill and batched decode; CUDA once hardware exists.

**Vulkan backend (2026-11-11; ADR-019, accepted 2026-11-11).** A real GPU behind ADR-018's `DiscreteDevice`, bit-identical through MoltenVK on the M3 Pro; lavapipe in a Linux container is detected and refused.
- `qubivex-backend-vulkan`: `VulkanDevice` on `ash` 0.38 (the loader is opened at run time, so the crate builds everywhere). It needs Vulkan 1.1 and no optional features. Weights go through a staging buffer into device-local memory with a counted capacity; each `matmul_many` is one command buffer waited on a fence. One shader (`shaders/matmul.comp`, SPIR-V checked in, dtype as a specialization constant) covers f32, f16, bf16, Q8_0, Q4_K and Q6_K.
- Bits: the CPU's 64-lane shape with the tree in shared memory (not subgroups), integer f16 conversion, `precise` everywhere. On MoltenVK, fast math is also turned off through `VK_EXT_layer_settings`.
- Runtime and CLI: `--backend vulkan [--vulkan-device N --vulkan-vram SIZE --vram-budget SIZE]`. The device is opened before planning, so the plan gets its capacity. Placement, the ledger, the device line and `/metrics` are as for `sim`. The copy time is measured, not modelled.
- Measured on MoltenVK: all six dtypes are bit-identical across tails, batches, offsets and 100-op submissions. Fixtures give the CPU's logits and tokens. OLMoE Q4_K_M gives the CPU's prompt logits and Colibri's tokens. `qubivex run` gives the same text at 5.7 tok/s decode against 31.6 on the CPU (per-element dequantisation, a synchronous wait on each of about 15 submissions per token, and a churning device cache).
- Positive control: an unfused multiply-add in the shader fails the parity tests. With fast math forced on (MathMode Fast), they still pass, because of `precise`.
- Lavapipe (Mesa 22.3 and 25.0, Linux container) splits every `fma` into a multiply and an add, even with `NoContraction`, which `shaders/build.sh` now adds because glslang omits it. A diagnostic matched its outputs exactly to an unfused-fma CPU reference (99 of 99 f32, 16 of 16 Q4_K). Everything else passed there. `VulkanDevice::new` now probes one fma and refuses such a device with the reason, so lavapipe is refused rather than trusted, and the tests skip on it.
- Tests: 1 unit test (SPIR-V freshness, needs glslc and SPIRV-Tools) and 8 parity tests in the crate; 4 in `qubivex-runtime/tests/vulkan.rs` (fixtures with ledger = device memory, a one-expert device cache, 4 concurrent requests, and the env-gated OLMoE test). All skip without a device unless `QUBIVEX_REQUIRE_VULKAN=1`, so CI (no Vulkan) skips them.
- Not built: fast kernels (block-wise K-quants, token tiles), layers or activations on the device, asynchronous submission, an integrated-GPU default (MoltenVK reports all RAM as device memory; cap it with `--vulkan-vram`), and lavapipe as a parity target (an emulated exact fma, or a lossy tolerance; ADR-019 lists both).
- **Next was** Vulkan kernel speed (done, above).

**Discrete device memory on a simulated GPU done (2026-11-11; ADR-018, accepted 2026-11-11).** Groundwork for CUDA and Vulkan, built and tested without either.
- `qubivex-compute::discrete`: a `DiscreteDevice` trait (`upload`, `matmul_many` over `DeviceWeights`, `usage`, `transfers`), an `Arc`-backed `DeviceBuffer` whose optional guard (the ledger charge) lives as long as the allocation, and `ComputeError::OutOfDeviceMemory`.
- `qubivex-backend-cpu::sim::SimDevice`: its own allocations with a hard capacity, real copies, refusal of another device's buffers, copy time modelled from bandwidth and latency (PCIe 4.0 by default) and optionally slept. It runs the CPU kernels on its own bytes.
- Planner: `PlanRequest::vram` (`VramRequest`) gives `MemoryPlan::vram` its first entries: budget, dense copy, a 256 MiB device reserve, and an expert cache capped at the RAM cache at full context; `Location::Vram(d)` and `VramCache(d)`. `qubivex plan --vram SIZE [--vram-budget SIZE]`.
- Runtime: `--backend sim [--sim-vram SIZE --sim-bandwidth GB/s --sim-delay --vram-budget SIZE]`. The dense arena is copied once. Experts are copied on first use into a device cache that evicts least recently used and stays inclusive with the RAM cache through slab attachments. A forward pass pins the experts it is computing. Every matmul is one device submission, found by host address; a weight with no device copy is an error. The ledger's `Vram(0)` tier equals the device's memory.
- Reporting: a device line in `qubivex run`, and the `qubivex_vram_*` and `qubivex_device_*` series in `/metrics`.
- Measured, OLMoE Q4_K_M, 3 GiB RAM budget, 1.5 GiB device: the same text as the CPU. The 767 MiB device cache made 4572 copies against 2065 hits (16.5 GiB, 0.93 s modelled at PCIe 4.0), so a device cache below the working set churns.
- Tests: 6 sim unit tests; a planner test; 5 fixture tests in `qubivex-runtime/tests/discrete.rs` (bits on all fixtures, a one-expert device cache, RAM-slab reuse, 4 concurrent requests, refusal of a too-small device) plus one env-gated OLMoE test (Colibri's tokens and the CPU's logits with a 64-expert device cache); one `/metrics` test. Removing the expert upload fails them with "no device copy".
- Not built: whole layers or activations on the device, asynchronous copies, pinned host memory, prefetch into the device cache, several devices, and CUDA or Vulkan themselves.
- **Next:** a Vulkan backend implementing `DiscreteDevice`, tested locally through MoltenVK (installed 2026-11-11) and lavapipe in a Linux container. CUDA is still blocked on NVIDIA hardware.

**Per-client API keys done (2026-11-10; ADR-013 amendment 7, accepted 2026-11-11).** Closes job 9's two gaps: `user` was unauthenticated, and behind a proxy every client shared one rate limit.
- Named keys: `ServerConfig::client_keys` (`ClientKey { client, key_sha256, rate_limit }`), from `[[serve.client_keys]]` tables in the config file only (`chmod 600`), never flags or environment. Only the SHA-256 is stored; a bearer is hashed once and compared with every digest in constant time. The shared `api_key` still works alongside them; either counts as a key for the loopback check.
- A named key makes the request its client's: the rate limit counts that key (its own `rate_limit`, else the server's); fairness counts the client whatever `user` says; the prefix-cache scope is the client plus `user`, and unnamed scopes are in a separate namespace, so no `user` value reaches a client's KV. A bearer that is no key counts against its address. `generation_started` logs `client`.
- CLI: `qubivex keys new CLIENT [--rate-limit N] [--rate-limit-burst N]` prints a `qvx_` key (32 bytes from `/dev/urandom`) once and the table to paste; `qubivex keys hash` hashes a key from standard input. `sha2` joins `qubivex-api` (already in the workspace for downloads).
- Tests: 4 over HTTP in `tests/client_keys.rs` (auth, separate allowances from one address, no KV reuse across clients whatever `user`, log names the client and never the key), 4 unit tests in `keys.rs`, 2 in the CLI, 1 config test, and one through the binary (`keys new`, config, `serve`, 200/401/401). Reverting the scope to `user` alone fails the KV test.
- Not built: reloading keys without a restart, per-client model access, token quotas, a cap on `priority: "high"` per client, per-client metrics labels.
- **Next:** open items: per-client quotas and priority caps (not started), CUDA (blocked on NVIDIA hardware), Vulkan, real-model metrics measurement (needs mains power).

**Deployment done (2026-11-10; job 11, ADR-013 amendment 6).** The last job of the platform plan (`PLATFORM_AUDIT` §4): Docker, runbooks and the spec §14 documents.
- `qubivex serve` now drains on SIGTERM as it does on Ctrl-C. Before, `docker stop` and systemd ended it mid-request (as process 1 in a container the signal was ignored until Docker killed it at 10 s). A test sends SIGTERM to the real binary and expects status 0.
- `Dockerfile` and `.dockerignore`: `rust:1.88-bookworm` builds, `debian:bookworm-slim` runs, as uid 10001, with the fixture models under `/usr/share/qubivex/models`, `/models` as the volume, and a `HEALTHCHECK` on `/health` that needs no curl. The image is 156 MB on arm64. Its default command listens on `0.0.0.0`, so it refuses to start without `QUBIVEX_API_KEY` or `--allow-unauthenticated`.
- Checked in the image (Docker Desktop, linux/arm64): the fixture model and the real `Qwen/Qwen3-0.6B` serve chat, completions and streams; `--read-only` with all capabilities dropped works (the server writes nothing); under `--memory 2g` the plan's budget was 1.76 GiB (the cgroup limit is read); `docker ps` reports `healthy`; `models search` reaches Hugging Face with no CA package (the roots are compiled in); a volume is writable by the image's user. `docker stop` during a 4000-token stream with `--shutdown-grace 1` ended it with the `overloaded` error event and `[DONE]`, and the container exited 0 after 1.25 s.
- Under `--memory 512m` even the fixture models are refused: the plan's 512 MiB activation reserve does not fit. That is the planner working, and the Linux runbook says so.
- Documentation: 25 new documents under `docs/` in the spec §14 layout (architecture 4, runtime 6, model-management 4, serving 5, performance 2, deployment 4), plus `docs/README.md` as the index with one list of everything not supported. Each page ends with its own "Not supported" section. `docs/performance/BENCHMARKING.md` stands for the spec's `benchmarking.md`: the two names are one file on a case-insensitive filesystem, and the CLI help cites the existing name.
- The payloads in `docs/serving/` were captured from the server (the Qwen3-0.6B run in the image and the fixtures locally), and the documented OpenAI Python SDK calls were run against it, including the refusals. The systemd unit passes `systemd-analyze verify`, the launchd plist `plutil -lint`, the Compose file `docker compose config`, and the config example the binary's own parser. Every relative link and anchor in `docs/` and the README resolves.
- CI gains a `docker` job: it builds the image, serves the fixture model, calls it, and checks that `docker stop` exits 0.
- Not done: nothing was run on Linux x86-64 hardware beyond CI, on Kubernetes, or as a launchd agent across a reboot; the runbooks say which steps are untested. No image or package is published. The alert thresholds in the operations runbook are starting points, not measurements.
- Fixed after the job (2026-11-10): `qubivex plan` printed "No VRAM budget: no GPU backend is built yet (Phase 8)", which predated Metal. It now says Metal shares host memory and is counted in the RAM budget, and that discrete GPUs (CUDA, Vulkan) are not supported yet.
- Tests: 502 passed, 0 failed, 1 ignored (one new: SIGTERM through the binary).
- **Next:** the platform plan is complete. Open: CUDA (blocked on NVIDIA hardware), Vulkan, and the owner decisions under "Pending review decisions". ADR-005, 008, 011 and 014 were accepted on 2026-11-10 (the CPU stays the default backend), so no ADR is Proposed.

**Observability done (2026-11-10; job 10, ADR-013 amendment 5).** The spec §8 metrics and structured logs on the gateway, still hand-written (the owner chose that over Kerniq Observe: no new dependency).
- `/metrics` gains per-model histograms: `qubivex_time_to_first_token_seconds`, `qubivex_generation_duration_seconds` and `qubivex_queue_wait_seconds` (all from the request's arrival), `qubivex_inter_token_seconds`, `qubivex_prompt_tokens_per_second`, `qubivex_generation_tokens_per_second`, `qubivex_model_load_seconds`; and `qubivex_http_request_duration_seconds` per endpoint.
- Read from the engine at scrape, per model: expert cache hits, loads, evictions, bytes read, time blocked on reads, retries and timeouts; batch passes and steps; ledger bytes by owner and the plan's budget; granted requests. For the process: resident size and its peak.
- Logs: one line per event on standard error, text by default or JSON (`--log-format text|json|off`), with a level, a UTC timestamp, and request, model and sequence ids on every line about a generation. The sequence id is new: the generation's number among those admitted. `--no-metrics` turns `/metrics` into a 404. Both are config-file keys (`log_format`, `no_metrics`).
- No prompt or generated text is logged: no logging call receives any, and `tests/observability.rs` sends marked prompts through both endpoints and a failing request and searches the log for them.
- Not measured: peak VRAM (no GPU tier in the ledger while Metal shares host memory; CUDA is blocked) and fragmentation. On macOS the peak resident size is the highest value a scrape has read. Expert load latency is a sum over loads, not a histogram.
- Not measured on a real model in this job: the machine was on battery. On the tiny fixture the server reported a 30 ms time to first token and 6 tokens for one request, which only shows the numbers are wired.
- Tests: 501 passed, 0 failed, 1 ignored. New: 3 in `log.rs`, 3 in `metrics.rs`, 2 over HTTP in `tests/observability.rs`, 1 precedence test in `serve.rs`.
- The gateway has run on Kerniq Server since PR #46 (ADR-013 amendment 4, MSRV 1.88); this file had not recorded it.
- **Next:** job 11, deployment (Docker, runbooks, the §14 docs).

**Hardening done (2026-11-10; job 9, ADR-013 amendment 3).** Spec §11 limits on the gateway.
- Listening beyond loopback without an API key is now refused by `Gateway::serve` and `qubivex serve` (it only warned before), unless `--allow-unauthenticated`.
- Per-client rate limit on `/v1/*` (`--rate-limit`, `--rate-limit-burst`): a token bucket per peer address (IPv6 by /64), 429 `rate_limit_exceeded` with `retry-after`, health and metrics exempt, at most 4096 clients remembered.
- Request limits: body bytes (`--max-body-bytes`, 4 MiB default, 413), prompt tokens after the template (`--max-prompt-tokens`, 400 `prompt_too_long`) and `max_tokens` (`--max-generation-tokens`, 400).
- CORS for configured origins or `*` (`--cors-origin`): preflights answered or refused with a 403, no credentials.
- `qubivex serve --config` (or `$QUBIVEX_CONFIG`): a TOML `[serve]` table; flag > environment > file > default; unknown keys refused; a file holding a key or token must be `chmod 600`. New dependency `toml` (CLI only).
- Tests: 492 passed, 0 failed, 1 ignored. New: 7 unit tests in `limits.rs`, 5 over HTTP in `tests/hardening.rs`, 5 precedence and validation tests in `serve.rs`, 3 through the binary in `tests/serve_config.rs`.
- Limits: one API key serves every client, so `user` (prefix-cache scopes, fairness) is still unauthenticated, and the rate limit counts addresses, so behind a proxy every client shares one allowance. Per-client keys would fix both.

**Batching done (2026-11-10; job 8, ADR-017 stages 2 and 3).** Sequences share forward passes on the CPU and on whole Metal layers.
- Fused CPU matmul for 2 to 5 activation rows (Q4_K, Q6_K, Q8_0, bf16), with the one-row kernel's bits. `batch_s<N>` on power, 5 threads, median (job 7's portable path in brackets): Q4_K expert 1024×2048 at 1, 2, 3, 4, 5 rows 39.2, 59.5 (129.9), 70.2 (155.2), 85.6 (180.2), 103.6 µs; Q6_K head 50304×2048 2.75, 3.07 (12.06), 4.01 (13.45), 4.12 (14.20), 4.80 ms. 6 to 12 rows now run through the same kernels in balanced groups of up to 5 (6 as 3 + 3, 8 as 4 + 4), each weight row once per group, same bits; `tile` starts at 13 rows for these dtypes. Grouped against `tile` (in brackets): expert at 6, 8, 12, 16 rows 136.7 (216.0), 169.9 (253.1), 252.2 (295.5), 375.5 (351.1) µs; head 7.23 (15.60), 8.20 (17.54), 12.34 (18.88), 16.81 (21.23) ms. 16 rows is even on the expert, so the cut-off is 12.
- OLMoE-1B-7B Q4_K_M on power (`examples/batch.rs`, 32 tokens per sequence, median of 3 rotated runs), decode tok/s, tokens identical in every cell. "Concurrent" is N threads each calling `Engine::forward` through the combining point, with the 1 ms bounded wait:

  | cache | N | sequential | batched (caller-driven) | concurrent | widest pass |
  |---|---:|---:|---:|---:|---:|
  | all resident | 1 | 40.93 | 42.60 (1.04x) | 43.14 (1.05x) | 1 |
  | | 2 | 41.40 | 49.86 (1.20x) | 49.07 (1.19x) | 2 |
  | | 4 | 40.54 | 56.21 (1.39x) | 56.80 (1.40x) | 4 |
  | | 8 | 41.13 | 53.42 (1.30x) | 51.89 (1.26x) | 8 |
  | 1024 MiB | 1 | 35.15 | 35.29 (1.00x) | 35.26 (1.00x) | 1 |
  | | 2 | 31.27 | 38.78 (1.24x) | 38.14 (1.22x) | 2 |
  | | 4 | 29.24 | 42.25 (1.44x) | 41.89 (1.43x) | 4 |
  | | 8 | 28.69 | 44.18 (1.54x) | 43.98 (1.53x) | 8 |

  Sharing a pass now pays (job 7: 0.59x to 1.00x). Without the wait, concurrent callers reached only 0.99x to 1.22x: a caller answered by a pass is still sampling when the next leader takes the queue, so at N = 2 passes mostly alternated one step each. The owner chose a bounded wait (2026-11-09; ADR-017 §2 amended): a leader waits up to `EngineOptions::batch_wait_us` (1 ms) for as many posts as the last pass answered. All resident, N = 8 fell below N = 4 because 8 rows used the `tile` path. With the grouped kernel (same run, N = 1, 2, 4, 6, 8, tokens identical): all resident, concurrent 42.27, 48.66, 54.37, 57.08, 61.63 tok/s (N = 8: 1.50x sequential; 51.89 before); 1024 MiB, concurrent 35.18, 38.72, 43.54, 44.45, 49.26 (N = 8: 1.73x; 43.98 before). Batched at N = 8: 63.18 and 49.57.
- Metal batching (ADR-017 §6), on both whole-layer placements (`layers`, routed on the host, and `token`, routed on the GPU): one submission per layer as before, every kernel over all the pass's tokens, and per segment one KV write and one attention against that segment's own device cache and positions. `Batch::attention_many` runs each attention stage (scores, softmax, values) for every segment at once, with a barrier between stages; a first version grouped whole attentions and raced. The head gathers the rows each segment wants. `Engine::forward_batch` and the combining point now run on Metal too; only a traced pass stays per caller. New tests in `tests/batch.rs` on the three MoE fixtures: a mixed batch gives the CPU's bits, concurrent callers get the CPU's tokens in shared passes, and a bad step fails only its caller, on both placements; attention on the first segment's cache, one shared scores range, and RoPE at the batch index each fail them. OLMoE `engine_on_metal_gives_the_cpu_tokens` passes on every placement. `examples/batch.rs --metal layers|token` gives identical tokens at N = 2, 4, 8 with every pass at full width (on battery, so no timings recorded yet).
- Metal on power (OLMoE Q4_K_M, all experts resident, `examples/batch.rs --metal`, 32 tokens per sequence, median of 3 rotated runs), decode tok/s, tokens identical in every cell, every pass at full width:

  | placement | N | sequential | batched (caller-driven) | concurrent |
  |---|---:|---:|---:|---:|
  | layers | 1 | 54.26 | 54.25 | 54.26 |
  | layers | 2 | 54.04 | 66.84 (1.24x) | 66.07 (1.22x) |
  | layers | 4 | 53.96 | 82.18 (1.52x) | 81.49 (1.51x) |
  | layers | 6 | 54.06 | 94.14 (1.74x) | 93.36 (1.73x) |
  | layers | 8 | 54.13 | 102.99 (1.90x) | 102.20 (1.89x) |
  | token | 1 | 60.35 | 60.33 | 60.25 |
  | token | 2 | 60.46 | 72.53 (1.20x) | 72.13 (1.19x) |
  | token | 4 | 60.48 | 86.92 (1.44x) | 86.48 (1.43x) |
  | token | 6 | 59.43 | 96.03 (1.62x) | 93.31 (1.57x) |
  | token | 8 | 58.59 | 101.31 (1.73x) | 102.20 (1.74x) |

  `token` (routing on the GPU) is faster for one sequence. From N = 2 the two placements are close, and at N = 8 `layers` slightly leads.
- Prefetch for batches (ADR-017 §7): not worth adding. `examples/batch.rs` now reports the store's blocked time as a share of decode time, and loads per token, and takes `--prefetch on|off`. On the CPU with experts streaming and prefetch off, concurrent callers at N = 2, 4, 8 wait on reads 1%, 2%, 1% of the time with a 1024 MiB cache and 5%, 2%, 2% with 512 MiB, against 5% to 19% for the same sequences taking turns. Predicting the union of a batch's experts could save at most that wait, so it is not built. With prefetch on (1024 MiB), concurrent decode measured 37.92, 43.73, 50.85 tok/s at N = 2, 4, 8 against 35.22, 38.91, 48.48 off. The lookahead never runs in those passes, so the gap is the cache state that earlier one-token passes leave behind, plus run-to-run spread: in the same prefetch-off run, N = 1 measured 30.79 tok/s sequential and 34.43 batched for identical work. Tokens identical in every cell.
- Job 8's done-when is met: the fused kernel, the combining point (§2–7, Metal included), bit-identical tokens on the CPU and both Metal placements, and concurrent decode measured on power.

**Batching design done (2026-11-08); ADR-017, accepted 2026-11-08.** Job 7 of the platform plan.
- `Transformer::forward_batch` over `Segment`s (a sequence's new tokens, its own `KvCache`, the logits it wants): everything per token runs once over the batch; RoPE, the KV write and attention use each token's own sequence. `forward` is `forward_batch` with one segment. `Engine::forward_batch` is public, CPU only, and not yet used by generation. Caches advance only after the whole pass succeeds, so `KvCache::rollback` was deleted.
- Tests: 464 passed, 0 failed, 1 ignored. `tests/batch.rs`, 5 tests on the four fixtures (mixed decode and prefill segments, lock-step decode with sequences leaving, failure isolation, a one-expert cache, empty batches), each bit-identical to the sequences run alone. Three wrong rules (attention on the first cache, unshifted positions, RoPE at the batch index) each fail them.
- Real model, OLMoE-1B-7B Q4_K_M on power (`examples/batch.rs`, N = 1, 2, 4, 8): tokens identical at every size, all resident and with a 1024 MiB expert cache, but no speed-up: 0.59x to 1.00x at N ≥ 2. The cause is the CPU matmul at 2 to 5 activation rows, which drops from the fused one-row kernel to the portable path: two rows cost 1.5x (Q4_K expert) and 2.1x (Q6_K head) two one-row calls (`batch_s<N>` in `benches/kernels.rs`). Single-sequence decode is unchanged: 42.41 tok/s on main against 42.09 on the branch over three interleaved rounds, tokens matching.
- **Next:** job 8, starting with a fused 2-to-5-row CPU kernel with the one-row bits, then the combining point at the engine turn (ADR-017 §2–7). ADR-017 was accepted on 2026-11-08, with that order.

**KV subsystem done (2026-11-07); ADR-016, accepted 2026-11-07.** Job 6 of the platform plan.
- New crate `qubivex-kv-cache`, no dependencies: `PrefixCache`, which finds the retained KV sharing the longest prefix with a prompt, keeps a budget with LRU eviction, and holds one entry per conversation. The runtime keeps finished caches in it and copies a hit's prefix into the request's own cache (`KvCache::restore_from` on the host, `MetalForward::restore` on the device), so no running sequence shares KV memory. What is kept is what ran, also for a cancelled or late request; a failed pass keeps nothing.
- Retained KV is priced in the memory plan, and reservations and passes evict it rather than wait or fail; memory pressure at `Elevated` clears it, from the sampling thread. Scoped by the request's `user`, because a hit is faster and so would show one client another's prompt. Off by default: `EngineOptions::prefix_cache_bytes`, `serve --prefix-cache-mb`.
- Reported as `usage.prompt_tokens_details.cached_tokens`, five `qubivex_prefix_cache_*` metrics and `prefix_cache` in `/v1/system`.
- Tests: 459 passed, 0 failed, 1 ignored. 10 index tests, 19 in `tests/prefix.rs` (bit-identical logits from a restored cache; a session of prompts with reuse on and off giving identical sampled tokens on the CPU and both Metal placements; isolation; scopes; budget; ledger; eviction for reservations and the plan; pressure; cancellation; four racing threads), one over HTTP, one env-gated on OLMoE. Removing each of 33 rules: 29 caught, two only after adding the test each exposed; three rules were redundant and removed; one, the forward-pass turn around the restore copy, survives by design (ADR-016). ThreadSanitizer: no reports over the prefix and gateway tests on macOS; CI adds `--test prefix` on Linux.
- Real model, OLMoE-1B-7B Q4_K_M on power: tokens identical with and without reuse on the CPU and Metal. Over HTTP with warm experts, a chat's second turn (714 tokens, 695 reused) took 0.57 s instead of 6.59 s and the third (738, 720) 0.45 s instead of 6.78 s, with identical text; another `user` got no hit. Single runs, not benchmarks; no kernel changed.
- Limits (ADR-016): a running request's prefix exists twice (retained and its own copy) until paged KV; the scheduler does not prefer hits; `user` is unauthenticated, so scopes hold among honest clients (job 9); a fixed `--expert-cache-mb` has no plan, so only the budget bounds it.
- ADR-016 was accepted on 2026-11-07 and ADR-015 on 2026-11-08.

**Concurrency correctness and the scheduler done (2026-11-07); ADR-015, accepted 2026-11-08.** Job 5 of the platform plan, the gate on batching.
- New crate `qubivex-scheduler`, no dependencies: a queue bounded by count, priorities with aging, fewest-running-first per client, and memory admission that waits or refuses. The runtime gains `Engine::prepare` and `run_prepared`, so an embedding application can schedule the same way. Deadlines are now checked before each prefill chunk too.
- Gateway: `--max-queue`, `--memory-policy`, `--stall-timeout`, `--shutdown-grace`; `priority`, `timeout` and `user` request fields; `POST /v1/models/load` and `/unload`; concurrent loads share one load and one directory has one instance; loading, loaded and failed states in `/v1/system`; unloading ends waiters and lets running requests finish; shutdown drains.
- Fixed, not in the audit: a client that kept its connection open and stopped reading held its slot and KV indefinitely, because the worker blocked in a channel send where neither the cancel flag nor the deadline was checked. A stream now ends after `--stall-timeout`.
- Tests: 428 passed, 0 failed, 1 ignored. Every §12.3 test passes; `tests/concurrency.rs` adds tests 1, 2, 6, 9, 11, 12 and 13 over HTTP. Loom models three scheduler races. ThreadSanitizer on nightly: no reports over 64 tests on macOS, with a deliberately racy program reported as a positive control, and none on Linux in CI with one suppression, for tokio's epoll hand-off (`.github/tsan-suppressions.txt`). It does not see Metal. Miri is clean on the scheduler's 17 tests.
- Removing each of 30 rules in turn: 29 caught, two of them only after adding the test each exposed. The one left is the grant being dropped just before the final event rather than just after it, a gap too small for a client to hit over HTTP. One more mutation showed a check was redundant, so it was removed.
- Real model, OLMoE-1B-7B Q4_K_M on `--max-concurrent 4 --max-queue 4`: ten at once gave eight 200s and two immediate 429s; all eight, and eight more in a second round, match sequential output exactly. SIGINT during a long stream ended it with an error event and `[DONE]` at the 3 s grace, and the process exited 0 at 3.1 s. No benchmarks: the machine was on battery, and no kernel changed.
- Limits (ADR-015): `high` is open to any client with the key and `user` is not authenticated (job 9); each model plans RAM and compute alone, so two busy models oversubscribe, and test 13 proves correctness, not shared capacity; priorities choose who starts, not the order of forward passes; cancellation lands between chunks and tokens, not inside a pass.
- Next was job 6 (above).

**`models analyse` and manifest 1.1 done (2026-11-07); ADR-004 amendment 1.** Job 4 of the platform plan.
- `qubivex models analyse <id>` gives a verdict (fits, streams, or does not fit), the RAM to start, to reach the context and to cache every expert, and what each backend would do. It takes `plan`'s flags and `--json`. "Does not fit" is an answer, so it exits 0. `qubivex_memory::assess` holds the logic, so an embedding application can call it.
- Manifest 1.1 adds `source` and `files` (config, tokenizer and the rest, with checksums). `models pull` and `models import --write` fill them; `models migrate` upgrades a 1.0 manifest. A newer minor is refused by name. Writes are atomic.
- Fixed: `Shard.sha256` was documented as checked by `models verify`, and nothing checked it. Verify now checks every file first, then experts. `--record` rewrites both sets of checksums, and only `--record` accepts a changed file.
- Tests: 392 passed, 0 failed, 1 ignored. Fixtures stay at 1.0 as migration input. Removing each of 22 rules in turn: all caught, one only after adding the case it exposed (a registry record of another directory). One more mutation showed a flag was redundant, so it was removed.
- Real models: `analyse` on OLMoE-1B-7B Q4_K_M gives all three outcomes (default budget fits; 2 GiB streams; 700 MiB does not fit, needing 866.99 MiB). A copy of Qwen3-0.6B migrated with checksums equal to `shasum -a 256`, then verified in 3.1 s; `experts.qvxsum` needs no change, because it covers weight files only. The installed manifests were left at 1.0: run `qubivex models migrate <id> --write` to move them.
- Limits: verify reads every file one at a time, with no mtime cache; a local import records no source; `import --force` drops a recorded source; pull's recording was tested against the fake hub, with no live pull. No benchmarks: no tensor or kernel code changed.
- Next was job 5 (above).

**Model downloads done (2026-11-06); ADR-014.** Job 3 of the platform plan. New crate `qubivex-model-download`, the workspace's first network and TLS code (`ureq` + `rustls`, blocking, on plain threads): `qubivex models pull` and `models search`, Hugging Face first, behind a `ModelProvider` trait. A pull pins the commit, fetches only what a model needs, verifies each file, and installs through the registry, so a model is complete in place or absent.
- Failure: stalls, drops and slow links share one path (a request budget, then resume at the offset). Retries back off exponentially, honour `Retry-After`, and reset on progress. Ctrl-C keeps the files and the same command continues them. Files that do not import stay, so a missing `--set` costs no second download.
- Threat review in ADR-014: the token goes only to the configured origin, on every hop, so it never reaches the CDN; no downgrade to plain HTTP; file names cannot leave staging; bodies stop at the listed size; errors carry no signed URL or token.
- Tests: 30 fault tests against a raw-socket hub, 14 unit tests, 4 through the binary. Each safety rule was removed in turn to check a test fails: 13 of 13 caught, one only after adding the case it exposed (a scheme with an explicit port).
- Real pull: `Qwen/Qwen3-0.6B` from the live hub, cancelled by SIGINT at 210 MB and resumed. `model.safetensors` and `tokenizer.json` match the published SHA-256 and the existing install; 32 generated tokens are byte-identical to it. Live search and refusals (a missing repository, a repository of many GGUF files) work. No benchmarks: the machine was on battery.
- Limits (ADR-014): split GGUF files are refused; a model's architecture is checked only after it arrives; published checksums protect against corruption, not a compromised hub; TLS uses bundled roots, so an inspecting proxy needs work. Source and revision were added to the manifest in job 4.
- `deny.toml` gained a licence exception scoped to `webpki-roots` (the CA list, as data).
- Next was job 4 (above).

**Model registry done (2026-11-06); ADR-003 amendment 2.** Job 2 of the platform plan. New crate `qubivex-model-registry`: `registry.json` records each installed model's source, revision and install state, with atomic updates. Installs build in staging and rename into place; `reconcile` recovers interrupted installs and adopts directories found on disk.
- CLI: `qubivex models remove <id> [--yes]` (without `--yes` it deletes nothing), and `models validate` as an alias of `verify`.
- Tests: 15 registry tests (a crash at each install and removal step, path traversal, tampered and newer registry files, symbolic links, 16 concurrent writers) and one test through the binary.
- Limits: removal does not check for a process serving the model.
- Next was job 3 (downloads, above).

**Generation contract done (2026-11-06); ADR-013 amendment 1.** Job 1 of the platform plan. `Engine::run_request` now holds the chat template, tokenization, context checks, KV reservation, stop sequences and detokenization. `qubivex run` and the gateway both use it, and so can an application embedding the runtime with no server, CLI or Tokio. No tensor or kernel code changed.
- Added: `top_k` (runtime, `--top-k`, request field); `examples/embed.rs`; `tests/requests.rs` (6 tests, including concurrent requests on a shared engine equalling sequential ones); `cargo xtask check` fails if the runtime gains a server dependency.
- Tests: 300 passed, 0 failed, 1 ignored. `qubivex run` on real OLMoE-1B-7B Q4_K_M (32 tokens, raw and chat) is byte-identical to the build before the change. No benchmarks: the machine was on battery.
- **Next:** job 2 of the plan (registry). The audit's §5 decisions are made.

**Platform spec audit done (2026-11-06).** The owner's implementation spec is audited in [`docs/engineering/PLATFORM_AUDIT.md`](docs/engineering/PLATFORM_AUDIT.md): status per requirement, a mapping of its phases (S0–S10) to the phases here, and an 11-job plan. Largest gaps: no model downloads or install registry (S2), no batching (S8), and no structured logs or config files (S10). Baseline: 293 tests passed, 0 failed, 1 ignored.
- Spec: [`docs/architecture/PLATFORM_SPEC.md`](docs/architecture/PLATFORM_SPEC.md). This replaces the "Phase 10" pointer below.

**Phase 9 (serving) done (2026-11-04).** Phase 8 (GPU acceleration, Metal first): slices 1–6 done (2026-11-05). The improvement-plan backlog (MIGRATION_PLAN §3a) is cleared.

**Metal at partial expert caches done (2026-11-06); ADR-011 amendment 6.** Measured on power with reference tokens at every size: Metal decode leads with most experts resident (3072 MiB: 49.2 against 42.9 tok/s), ties near 2048 MiB (40.1 against 40.8) and loses below it (1024 MiB: 29.5 against 34.9), because it waits on loads the CPU pass overlaps. Metal prefill leads at every size (128 tokens at 1024 MiB: 194 against 117).
- Code: `cpu_pass_reason` in `engine.rs`: `--backend metal` runs the CPU pass when a layer's pool holds fewer than top-k or fewer than half its experts, with the reason in `PlacementReport::fallback`. Decode goes 34.5 → 37.7 tok/s at 1536 MiB, 30.8 → 35.4 at `--ram-budget 2GiB`, 29.5 → 34.1 at 1024 MiB; prefill in that range drops to the CPU's speed.
- Tests: a unit test of both limits. Existing Metal tests pass unchanged.
- Not built: Metal for prefill with the CPU for decode (at most about 8% before the KV handoff, and it breaks placement fixed at load). Just above the line Metal gives up about 5% of decode to keep its prefill.
- Records: `docs/performance/baselines/2026-11-06-m3pro-smallcache-*.json`.
- **Next:** Phase 10.

**Phase 7 saturation throttle done (2026-11-06); ADR-008 amendment 3.** On a saturated simulated disk at `--ram-budget 1GiB`, prefetch with the guard lost 3–4% (9.93 → 9.52 tok/s at 6 GB/s, 3.70 → 3.59 at 2 GB/s, 0.99 → 0.96 at 500 MB/s); the throttle removes that (9.99, 3.73, 1.00), and tokens stay identical.
- Code: `fault::SlowDisk` and `bench --disk-mbps` / `--disk-latency-us`; `policy::SaturationGauge` (share of wall time blocked on reads, 100 ms windows, holds at 60%, resumes below 40%); checked in `Engine::forward`; `--prefetch-throttle on|off`, default on; `held_rate` in `bench` records.
- Tests: gauge thresholds, windows and idle turns; shared bandwidth of `SlowDisk`; an engine test that a saturated disk holds speculation with identical tokens.
- Owner decision (2026-11-06): prefetch stays opt-in behind `--prefetch`, with no further work unless a real workload shows a gain. No regime measured shows it gaining throughput: on the real warm disk it is about 2.5% behind, within the spread of its own repeat runs, and the throttle never holds there (about 37% blocked).
- Not on power, so not recorded as baselines: the 200 MB/s + 8 ms row and the `--expert-cache-mb 2048` rows.
- Next was Metal at partial expert caches (done, above).

**Phase 8 slice 6 done (2026-11-05): token tiles for Metal prefill; ADR-011 amendment 5.** Prefill with every expert resident: 128 tokens 97 → 239 tok/s, 1024 tokens 235 tok/s (CPU 115 for both), 10 tokens 91 → 102; decode unchanged (59.6); tokens bit-identical.
- Code: token-tiled matmul kernels (`matmul_<dtype>_t`, up to 8 activation rows per weight load, per-(row, token) accumulators in the one-token order; row bodies are templates shared by every kernel), used from 4 rows; `moe_group` (GPU counting sort of routed entries by expert) and `Routing::Grouped` / `routed_<dtype>_g`, used by the one-submission pass from 2 entries per expert.
- Tests: tile sizes 1–17 for every format, grouped routing with multi-tile experts, the fixtures' prompt through the grouped path, and the real-model prompt re-run as one grouped submission (bit-identical).

**Phase 8 slice 5 done (2026-11-04): a token in one Metal submission; ADR-011 amendment 4.** Decode with every expert resident 44.9 → 59.2 tok/s (CPU 42.3); 10-token prefill 76 → 91 tok/s; tokens bit-identical.
- Code: `moe_route` (GPU top-k routing with `ops::route`'s bits) and routed matmuls over an address table of every expert of a layer (`Batch::moe_route`, `Batch::routed_matmul`; one dispatch per gate/up/down, no gather or scatter); `ExpertStore::hold_layer`/`Held` (lease a whole resident layer without counting a use) and `record_uses` (count the GPU's choices afterwards as hits); `MetalOps::token` (`--metal-ops token`, the new default for `--backend metal`; `layers` keeps host routing), falling back per pass to whole layers while experts are missing (queued as prefetches) or pools are smaller than the expert count; `Engine::gpu_routed_passes`. A concurrent encoder: barriers between dispatches, and `Batch::concurrent` groups (q/k/v, RoPE, KV stores, gate/up, and each per-expert stage of the per-layer pass, which goes 44.9 → 53.1 tok/s).
- Tests: new GPU route and routed-matmul parity tests, the store's hold/record test, the fixture invariant with `token` (every pass routed without I/O threads), the Metal fault test for both placements, and real OLMoE (prompt logits and 48 tokens bit-identical; 40 passes routed on the GPU).
- Not better yet: 128-token prefill (CPU 115 tok/s, Metal 97): the GPU matmuls read each weight row once per token. Next was tiling them (slice 6, above).

**Phase 9 done (2026-11-04): the serving gateway; ADR-013.**
- `qubivex-api` (Axum + Tokio at the edge only) and `qubivex serve`: `/v1/chat/completions` and `/v1/completions` (SSE or not), `/v1/models`, `/health`, `/metrics` (Prometheus text), `/v1/system`, admin load/unload behind `--admin-token`, optional `--api-key`, request ids, OpenAI-shaped errors without paths. Every generation on its own thread with bounded channels.
- Runtime: `Engine::generate_with` (chunked prefill, cancellation flag checked per chunk and per token, `Finish` reasons), `Engine::reserve_kv` (KV of concurrent conversations admitted together against the plan; room sharing and the KV ledger charge use the total), forward passes take FIFO turns per engine (a ticket lock; concurrent passes on one-slot pools failed with "pool is full"), `StreamDecoder::finish` (a trailing incomplete character is no longer dropped).
- Exit criteria: the official OpenAI Python SDK (`crates/qubivex-api/sdk/openai_client.py`) works against OLMoE-1B-7B (GGUF) and Qwen3-0.6B (SafeTensors) served together: chat, streamed chat equal to non-streamed, completions, two concurrent chats equal to their sequential results, SDK error types. Over HTTP on the fixtures (`tests/gateway.rs`): concurrent = sequential, cancellation of streamed and non-streamed requests frees the slot, the fifth of four-that-fit conversations gets 503, auth/admin/errors. A curl disconnect on the real model shows up as one cancelled generation.
- Tests: 284 on 1.85 and stable; `cargo deny` passes with the new dependencies. OLMoE single-request decode unchanged (43.2 tok/s, tokens match).
- Next was Metal slice 5 (done, above).

**Dense architecture and a real SafeTensors run done (2026-11-04), §3a item; ADR-003 amendment 1.**
- `transformer_dense` registered beside `transformer_moe` (own config, presets `hf`/`gguf`, schema; refuses routed experts); shared config parsing in `qubivex-model/src/arch/common.rs`; new `TensorRole::Ffn` (planner group `dense FFN`). The forward pass (`qubivex_arch::Transformer`, renamed from `TransformerMoeModel`) shares embedding, attention, KV and head; the FFN is configuration. Whole-layer Metal and prefetch stay MoE-only; a dense model on Metal places its matmuls one by one (same bits).
- Importer tries both architectures; a tied head stored anyway goes to `overrides.ignore_tensors`; no expert checksums for a dense model. Also refused now by every architecture: `mlp_bias`, `use_sliding_window`.
- Real run: `Qwen/Qwen3-0.6B` bf16 SafeTensors (1.4 GiB) imports with no edits and generates coherent text (plain and chat template). Against transformers f32: 32 greedy tokens identical, the top-32 logits of every prompt position within 1.2e-5 relative; prefill equals token by token and 1 vs 4 threads bit for bit (`tests/real_dense.rs`, env-gated `QUBIVEX_DENSE_SAFETENSORS_DIR`).
- Measured (M3 Pro, 5 threads): decode 68.6 tok/s (about 77 GiB/s of bf16 weights per token, near the machine's bandwidth), prefill 157 tok/s (10 tokens) and 281 tok/s (128 tokens); Metal per-op 33 tok/s with identical tokens. Records `docs/performance/baselines/2026-11-04-m3pro-dense-cpu-*.json`. OLMoE unchanged (42.65 tok/s, tokens match).
- Tests: 276 on 1.85 and stable. New: `dense-tiny` fixture (Llama-style, no q/k norm, GQA, tied head) against transformers per layer (5.2e-7), in the prefill/thread/Metal invariants, fixture import round trip, refusals (`num_experts`, `mlp_bias`, sliding window, attention bias).

**KV cache on `Tensor<f32>` done (2026-11-04), §3a item; ADR-009's next consumer.**
- `qubivex-arch` `KvCache`: one `Tensor<f32>` per layer for keys and one for values, `[kv_head, pos, head_dim]`, with room for the reserved positions (still whole 256-position chunks, so the planner's KV charge is unchanged). A step writes each head's new rows through one `narrow` view; attention reads each head's positions `0..=p` as one contiguous `narrow` view; a failed step only resets the position count (rows past it are never read and are overwritten next). Per-layer tensors, so growing copies one layer at a time.
- `qubivex-compute`: `TensorViewMut::{reborrow, narrow, select, contiguous_data_mut}`.
- Measured against `main` (interleaved): 1024-token prompt + 300 generated, prefill 114.8 / 116.7 / 111.7 vs 111.3 / 110.6 / 117.5 tok/s, decode 32.5 / 33.4 / 33.1 vs 32.6 / 33.4 / 33.7; 10-token prompt decode 42.6 / 43.0 vs 42.5 / 42.2. Tokens identical in every run.
- Tests: 274 on 1.85 and stable. New: KV growth across chunk boundaries (one 300-token step, then single tokens past 512) equals token by token, a refused step adds no positions, a cleared cache keeps its room and gives a fresh cache's bits; mutable tensor views.

**Faster lossless Q4_K done (2026-11-04), §3a item; ADR-002 amendment 2.** Decode 30.5 → 41.6–42.5 tok/s all resident and 19.0 → 22.4–22.7 at 1 GiB; prefill unchanged; tokens identical to every reference.
- Kernel: `qubivex-backend-cpu/src/neon.rs`. `d1 · q` is exact in f32 (17-bit `d1`, 4-bit `q`), so `fma(d1, q, −m1)` rounds exactly where the portable multiply-then-subtract does. On FEAT_FHM CPUs (run-time detected; every Apple M-series) `fmlal` forms `d · (sc · q)` from f16 operands (`sc · q < 1024`, exact in f16) in one instruction. 23 → 13 vector instructions per 16 weights; `decode_t5` gate/up 62.1 → 38.8 µs (17.5 → 28 GiB/s).
- Tests: 272 on 1.85 and stable. New: every f16 `d` and `dmin` (with random scales, nibbles, activations) bit-identical across the portable, baseline-NEON and FHM paths (NaN results only have to be NaN); multi-block rows; the block-scale unpacking against GGML's `get_scale_min_k4`. Records: `docs/performance/baselines/2026-11-04-m3pro-q4k-fhm-*.json`.

**Prefill done (2026-11-04), §3a item; ADR-002 amendment 1.** 128-token prefill 44.7 → 113–117 tok/s (faster than Metal whole layers' 77); decode 25 → 30 tok/s; tokens identical to every reference.
- Kernel: `qubivex-backend-cpu/src/tile.rs`, a register-blocked matmul (4 weight rows × 6 activation rows, one 4-lane quad of the 64-lane reduction at a time, lane-major packed operands) that reproduces the decode dot product's bits exactly. It is used from 6 activation rows (the measured crossover); below that the row path stays. The `gemm` crate is not used: its accumulation order breaks invariant 1 (prefill equals token-by-token). Kernel 2.1× (`prefill_s128_t5` Q4_K gate/up 3.44 → 1.66 ms).
- Concurrency: `ExpertProvider::with_experts` takes an `Fn + Sync` callback; `StoreProvider` computes ready experts concurrently in a rayon scope and starts newly ready ones as each finishes (new `Batch::try_next` / `Next` in the store; the blocking `next` runs only with no lease held). Attention runs on the pool per (token, head).
- Spinning pool: deferred. After these changes workers are asleep 3–5% and spinning 1–5% of the time, so a pool could recover at most that, while replacing rayon's nested work stealing.
- Tests: 269 on 1.85 (debug), all env-gated real-model tests (Metal parity, transformers, Colibri tokens, ledger) pass. New: `compute_threads_never_change_bits` (1 and 3 threads × one-slot pools × I/O threads), `try_next` store tests, matmul bit identity at s = 2, 5–11, 13. Records: `docs/performance/baselines/2026-11-04-m3pro-tiled-*.json`; numbers in BENCHMARKING.md.

**Read hardening done (2026-11-04): the three gaps fault injection left; ADR-006 amendment 4.**
- Hung reads: a per-waiter deadline (`ReadTimeout`; engine default 5 s + size at 10 MB/s; `--read-timeout`), `StoreError::Timeout`, `Occupancy::stuck`, `Engine::admit` refuses when every I/O thread is stuck (`RuntimeError::Storage`), and the store's drop detaches stuck threads. Retries: once by default (`--read-retries`).
- Mirrors: `mirror::MirroredSource` (failover, skip after 3 failures for 30 s), `ModelExpertSource::mirror_at`, `Engine::load_mirrored`, `--mirror DIR`.
- Checksums on by default: `experts.qvxsum` recorded by `qubivex models verify` (which also scrubs) and `models import --write`; the engine verifies every read when the file matches the weights (`--verify-experts auto|off|require`); stale files are reported, not used. The checksum went from 4 to 8 lanes (21 → 39 GiB/s). The cost is about 2–3% of decode at 1 GiB streaming, and nothing measurable when resident.
- Tests: 265 pass on 1.99.1 and 1.85. Still open: hedged reads and per-device queues (need a second drive).

**Fault injection done (2026-11-04), Addendum §4 Phase 3 item; ADR-006 amendment 3.**
- Code: `qubivex_expert_stream::fault` (`FaultySource` with `Rule`/`Target`/`Fault`: EIO, short read, corrupt bytes, panic, delay; `VerifiedSource` and `checksum`); `ExpertStore::occupancy`; `Engine::load_with_source` and `Engine::store_occupancy`; `ModelExpertSource::keys`; `PageAligned::try_zeroed`/`try_resized`.
- Fixes found by the suite: a panicking source killed an I/O thread and left its waiters hanging (now a load error); slab allocation aborted on failure (now a load error); a failed forward pass left partial keys and values in the KV cache (now rolled back); the ledger was not updated after a failed pass.
- Tests: 253 pass on 1.99.1 and 1.85 (7 store and 4 engine fault tests are new, including Metal whole layers). Speed is unchanged within noise (21.1 → 21.5 tok/s at 1 GiB; tokens identical).
- Next was prefill (done, above).

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
- Decision (accepted with ADR-011 on 2026-11-10): the CPU stays the default backend; `--backend metal` meant whole layers here, and whole tokens since slice 5.
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
- Finding: Apple GPUs flush f32 subnormals to zero (safe mode too). That is the only divergence; `subnormals_are_the_only_divergence` documents it. Parity contract: ADR-011 (accepted 2026-11-10).
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
Candle). ADR-001 was accepted on 2026-11-06.

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
- ~~Move ADR-001 to Accepted after review~~: accepted 2026-11-06.

## Environment change (supersedes MIGRATION_PLAN §2)
Development moved to an **Apple M3 Pro (arm64, 18 GB RAM, macOS)** with about 374 GB free.
- **NEON** is now the first local SIMD target. AVX2 and AVX-512 are exercised on Linux x86-64 CI.
- Rust stable 1.99.1 and 1.85 (MSRV) are installed through rustup in `~/.cargo`.
- Apple GPU: Metal becomes the locally testable GPU backend. CUDA still needs other hardware.

## Pending review decisions
1. ~~Synchronous lease-based expert store instead of the specification's async trait (ADR-006)~~: accepted with ADR-006 (2026-11-08).
2. ~~11 initial crates instead of about 25 (ADR-001)~~: accepted with ADR-001 (2026-11-06).
3. ~~Merged architecture factory with `tensor_schema` (ADR-003)~~: accepted with ADR-003 (2026-11-08).
4. Minimal expert store moved into Phase 4 (MIGRATION_PLAN §3).
5. Reference model: OLMoE-1B-7B-0125-Instruct, plus a tiny Qwen3-MoE fixture.
6. ~~Optional expert-packed container layout (ADR-006)~~: accepted with ADR-006 (2026-11-08).
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
- Accepted: ADR-001 (workspace, phase-gated crates; 2026-11-06), ADR-003 (architecture registry; 2026-11-08), ADR-004 (versioned manifest; 2026-11-08), ADR-006 (synchronous lease-based expert store, io_uring deferred; 2026-11-08), ADR-015 (scheduler and concurrency model; 2026-11-08), ADR-016 (KV subsystem: retained KV and prefix reuse; 2026-11-07), ADR-002 (hybrid, no Candle; amendment 1: own register-blocked prefill instead of `gemm`, spinning pool deferred), ADR-009 (tensor descriptors), ADR-010 (numerical precision).
- Amended: ADR-007 (Metal first; device assignment fixed per execution plan).
- Accepted 2026-11-10: ADR-005 (memory planner; no VRAM budget yet), ADR-008 (expert cache; hot set, pins and tier promotion not built), ADR-011 (device execution and GPU parity; the CPU stays the default backend), ADR-014 (model downloads and threat review).
- Accepted 2026-11-11: ADR-018 (discrete device memory, simulated device), ADR-019 (Vulkan backend; the CPU's bits, a device that splits fma is refused).
- Proposed: none.
