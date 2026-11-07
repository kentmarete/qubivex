# Platform audit (spec Phase 0)

- Date: 2026-11-06
- Commit audited: `main` at `4db44c0`
- Machine: Apple M3 Pro, 18 GB. Tests only: the machine was on battery, so no benchmarks ran.
- Baseline: `cargo test --workspace`: 293 passed, 0 failed, 1 ignored. Benchmarks are the
  committed records in `docs/performance/baselines/`.

This audits the repo against the owner's "Complete Implementation Specification" (the
platform spec, [`docs/architecture/PLATFORM_SPEC.md`](../architecture/PLATFORM_SPEC.md)),
received 2026-11-06. It is the spec's Phase 0 deliverable.

Ratings describe what the code does, not what a module is called:

- **Production:** built, tested through its public surface, and exercised on real models.
- **Partial:** works, but a named part of the requirement is absent.
- **Missing:** no code.

## 1. Two phase numberings

The spec's phases are not the repo's. They overlap in topic and differ in number: the
spec's Phase 10 is production readiness, and the repo's Phase 10 is batching, prefix reuse
and paged KV. Below, **S0–S10** are spec phases. Phase numbers without the S stay with
`MIGRATION_PLAN.md`, which every ADR and baseline already cites.

| Spec phase | Scope | Repo phases | State | Acceptance met? |
|---|---|---|---|---|
| S0 | Audit and plan | | This document | Yes |
| S1 | Runtime foundations | 4 | Partial | Models run, but through the concrete `Engine`; there is no stable contract |
| S2 | Model manager | 3 | Partial | No: nothing downloads or removes a model |
| S3 | Memory planner | 5 | Mostly production | Yes, through `qubivex plan` and `models analyse` |
| S4 | Runtime, expert streaming | 6, 7 | Mostly production | Yes |
| S5 | HTTP server | 9 | Production | Yes |
| S6 | Streaming | 9 | Production | Yes |
| S7 | Concurrent scheduler | 9 | Production | Yes: `qubivex-scheduler` (ADR-015) |
| S8 | Continuous batching | 10 | Missing | No |
| S9 | Hardware acceleration | 8 | Partial | CPU and Metal yes; CUDA blocked (B2), Vulkan not started |
| S10 | Production readiness | 10–11 | Partial | No |

The spec orders S1 to S10, but the repo reached S9's Metal work first, so the order below
applies only to what remains.

## 2. Status by area

### Runtime (Product A)

| Requirement | State | Evidence |
|---|---|---|
| Tensor engine, kernels | Production | `qubivex-core`, `qubivex-compute`, `qubivex-backend-cpu`; Criterion benches |
| Data types | Production for f32, f16, bf16, Q8_0, Q4_K, Q6_K | The rest of the GGUF `dtype.rs` set parses; check support per format before advertising it |
| Transformer execution | Production | `qubivex-arch`; matches `transformers` to under 1e-6 relative |
| MoE routing | Production | Softmax and top-K renormalised routing; GPU routing has `ops::route`'s bits |
| Expert lifecycle | Production | `qubivex-expert-stream`: leases, in-flight dedup, LRU, bounded prefetch, cancellation of unstarted loads, loom checks |
| Memory planner | Production | `qubivex-memory::plan`, cgroup-aware, explainable through `qubivex plan` |
| KV cache | Partial (job 6: prefix reuse done) | `KvCache` in `qubivex-arch` is per-conversation, bounded and reserved through the planner. Job 6 added retained KV with prefix reuse, a budget and eviction in `qubivex-kv-cache` (ADR-016). No paging |
| Backends | Partial | CPU and Metal bit-identical. CUDA and Vulkan exist only as detection |
| Stable interfaces | Partial | Traits exist for `ComputeBackend`, `ExpertSource`, `DenseWeights`, `ExpertProvider`, `ArchitectureFactory`, `ResourceProbe`. `Engine::run_request` is the generation contract (ADR-013 amendment 1). There is no model handle yet |
| Sampling | Production | Greedy, temperature, top-k, top-p, seed |

### Model manager (Product B)

| Requirement | State | Evidence |
|---|---|---|
| Containers, manifest 1.1, validation | Production | `qubivex-model`: hardened GGUF and SafeTensors readers, proptests, JSON schema |
| Import, discovery, verify | Production | `models import / list / inspect / verify` |
| Downloads | Production | `qubivex-model-download` (ADR-014): resumable, verified, atomic; 30 fault tests against a local server; a real interrupted pull resumed and matched the reference byte for byte |
| `models pull / search` | Production | Hugging Face first, behind `ModelProvider`. Split GGUF files are refused (ADR-014) |
| `models remove` | Production | Needs `--yes`; deletes only inside the models directory |
| `models analyse` | Production | Says whether a model runs here, the RAM each way of running it needs, and what each backend would do. Built on the planner, through `qubivex_memory::assess` |
| Registry | Production | `qubivex-model-registry`: atomic records of source, revision and install state, crash recovery, safe removal. `models pull` installs through it |
| Lifecycle states | Production | Install states are recorded (ADR-003 amendment 2). Loading, loaded and failed states are held by the serving process and shown in `/v1/system` (ADR-015) |
| Manifest fields | Production | Schema 1.1: source, revision and every file's sha256, with `models migrate`. Dimensions and expert fields stay in the checkpoint on purpose (ADR-004 amendment 1) |

### Server (Product C)

| Requirement | State | Evidence |
|---|---|---|
| OpenAI chat and completions, SSE | Production | `qubivex-api`; official SDK against OLMoE and Qwen3 |
| Backpressure, disconnects | Production | 64-event bounded channel; drop guards; tests over HTTP |
| Unsupported parameters | Production | Refused, not ignored (ADR-013) |
| Top-k | Production | `top_k` request field |
| Model endpoints | Production | `POST /v1/models/load` and `/unload`; the admin routes stay as aliases |
| Load dedup | Production | Concurrent loads join one load; one directory, one instance (ADR-015) |
| Safe unload | Production | `Arc` holds the model for running requests; the queue closes at once; tested with a live stream |
| Queue | Production | `qubivex-scheduler`: bounded by count, refused at once past the limit |
| Priorities, deadlines, per-client fairness | Production | `priority` and `timeout` request fields, aging, fewest-running-first by `user`. `user` is not authenticated (job 9) |
| Memory admission | Production | Against the plan through the scheduler; queues or refuses by `--memory-policy` |
| Batching | **Missing** | See §3 |
| Auth | Partial | Optional API key and admin token, constant-time. No key refuses a non-localhost bind |
| Rate limiting | Missing | The 429 means queue timeout only |
| Request limits | Partial | Axum's default body limit only; no configured prompt length, no explicit limit |
| CORS | Missing | |
| Graceful shutdown | Production | Waiters end at once, running requests get `--shutdown-grace`, then stop with an error. Tested over HTTP |
| Metrics | Partial | Requests, tokens, active, cancelled, failed, KV, cache bytes, slots, queue depth, rejections. No TTFT, tokens/s, latency histograms, hit rate, evictions or disk throughput |
| Structured logs | **Missing** | `eprintln!` with a request id. No model or sequence ids, levels or format |
| Config files | **Missing** | Flags and three environment variables. No TOML, no precedence rule |

### CLI

| Spec command | State |
|---|---|
| `system` | Done |
| `models list / inspect` | Done |
| `models validate` | Done, alias of `models verify` |
| `serve` | Done, with `--listen` instead of `--host` and `--port` |
| `chat --model --prompt` | Done as `run`. Interactive chat is missing |
| `server status` | Missing |
| `models remove` | Done |
| `models search / pull` | Done |
| `models analyse` | Production |

`run` and the gateway both call `Engine::run_request`, so there is one inference pipeline,
as the spec asks.

### Tests the spec requires

| Test (§12.3) | State |
|---|---|
| 1. Ten simultaneous requests | Covered: `concurrency.rs` |
| 2. Fifty, bounded | Covered: `concurrency.rs` |
| 3, 4. Same expert, concurrent loads | Covered in the store's tests and loom |
| 5. Eviction while executing | Covered by the store's leases |
| 6. Model load dedup | Covered: `concurrency.rs` |
| 7, 8. Cancellation, SSE disconnect | Covered |
| 9. Unload while active | Covered: a live stream and a waiter |
| 10. Memory exhaustion | Covered for KV admission, both policies |
| 11. Shutdown with active requests | Covered: within the grace and past it |
| 12. Different context lengths | Covered: `concurrency.rs` |
| 13. Multiple models | Covered for correctness and separate queues in CI. Shared capacity is not arbitrated (ADR-015) |

### Docs and packaging

No `docs/deployment/`, `docs/runtime/`, `docs/serving/` or `docs/model-management/`, no
Dockerfile, and no `examples/`. CI covers Linux and macOS, MSRV 1.85, loom and `cargo deny`.

## 3. Findings that need more than filling a gap

1. **The turn lock conflicts with the spec's wording.** §7.3 says to avoid a global lock
   around inference. `Engine::forward` runs one pass at a time per engine through a ticket
   lock, on purpose: each pass uses every compute thread, and two passes in a tight pool
   fail with "pool is full" (ADR-013). Removing the lock would gain nothing and would bring
   that failure back. It should stay until S8 replaces it with real batches, where one pass
   carries many sequences. S1–S7 can treat it as an internal detail of "controlled compute
   execution", which is §7.4 stage 1.
2. **S8 is the largest item left.** `Transformer::forward` takes one token list and one
   `KvCache`. A batch needs per-sequence routing, positions and KV inside one pass, on CPU
   and in the Metal kernels, with tokens still bit-identical to a sequential run
   (invariant 1). Expert grouping across sequences is where the speed would come from, and
   it is unmeasured. It needs its own ADR, and ADR-012 stays reserved for the graph IR.
3. **Generated-token decoding is quadratic** in completion length (ADR-013). It should be
   fixed before longer outputs are served.
4. **Downloads add the workspace's first outbound network and TLS code** (§5 decision 2).
   The threat review is in ADR-014: redirects, path traversal from repository file names,
   and token handling.

Rules the code already follows: the runtime does not depend on the server, the server has
no tensor code, and the model crate executes nothing. `cargo tree -p qubivex-runtime` on this
commit shows no `tokio`, `axum` or `clap`. Job 1 added the proof: `examples/embed.rs`,
`tests/requests.rs` and the `check-embeddable` gate.

**Not adopted from the spec's target layout:** `qubivex-tensor`, `qubivex-moe`, `apps/server`
and `apps/model-manager`. Splitting them now would move working code with no behaviour
gained, against ADR-001 and the spec's own preservation rules. New crates are justified for
new code only: `qubivex-model-download` and `qubivex-model-registry`.

## 4. Plan

Each row is one job and one PR. Dependencies run downwards.

Owner rules for this plan (2026-11-06):

- **No tensor engine rewrite.** `qubivex-core`, `qubivex-compute` and the backends stay. This
  audit establishes what works first, and any architectural change has to build on that.
- **Concurrency correctness comes before batching.** Shared weights, independent KV caches,
  expert leases and cancellation-safe resource management must be proven first. Job 5 is a
  gate: jobs 7 and 8 do not start until it is merged.
- **The runtime stays embeddable.** Qubivex must work inside another Rust application with
  no HTTP server and no CLI. Job 1 proves it, and a check keeps it true.

| # | Job | Spec | Depends on | Done when |
|---|---|---|---|---|
| 1 | Generation contract (**done**) | S1 | | A `GenerationRequest`, stop handling and templates live in the runtime; `run` and the gateway both use them; top-k added; tokens unchanged; a compiled embedding example uses only the runtime; CI fails if the runtime gains `tokio`, `axum` or `qubivex-api` |
| 2 | Registry (**done**) | S2 | | Atomic store of installed models, revisions and lifecycle states; survives a failed install; `models remove`, `validate` alias |
| 3 | Downloader (**done**) | S2 | 2 | Resumable, verified, atomic `models pull` and `search` from Hugging Face; fault tests against a local server; one real pull |
| 4 | Analyse and manifest v1.1 (**done**) | S2, S3 | 2 | `models analyse`; source, revision and checksums in the manifest, with a migration test |
| 5 | Concurrency correctness and scheduler (gate) (**done**) | S4, S7 | 1 | Every §12.3 test passes, including 1, 2, 6, 9, 11, 12, 13, with a sanitizer run where supported; queue bounded by count, priorities, deadlines, load dedup |
| 6 | KV subsystem (**done**) | S4 | 1 | Prefix reuse with a bit-identity test; bounded and evictable |
| 7 | Batching design | S8 | 5 (gate), 6 | ADR with a measured prototype |
| 8 | Batching, stages 2 and 3 | S8 | 7 | Sequences share a pass; tokens bit-identical; measured on power |
| 9 | Hardening | S10 | 5 | Rate limits, request limits, CORS, non-localhost refusal, config files |
| 10 | Observability | S10 | 5 | The §8 metrics; structured logs with no prompt text |
| 11 | Deployment | S10 | 9, 10 | Docker, runbooks, the §14 docs |
| Blocked | CUDA | S9 | B2 | Needs NVIDIA hardware; abstraction only until then |
| Later | Vulkan | S9 | | Spec says "where practical" |

## 5. Owner decisions (2026-11-06)

All six were taken as recommended.

1. **Numbering.** `MIGRATION_PLAN.md` keeps its phases, and S-numbers mean spec phases.
2. **HTTP client.** `ureq` with `rustls`, blocking, on dedicated threads. `qubivex-model`
   stays free of Tokio.
3. **Turn lock.** `Engine::forward` keeps it until S8 replaces it with real batches
   (finding 1). This is a deliberate exception to spec §7.3 until then.
4. **Endpoints.** `POST /v1/models/load` and `/unload` are added, and the admin routes stay
   as aliases.
5. **Crate layout.** The "not adopted" list in §3 stands. New crates only for new code.
6. **Spec.** Committed as `docs/architecture/PLATFORM_SPEC.md`.
