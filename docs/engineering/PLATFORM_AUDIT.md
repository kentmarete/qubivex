# Platform audit (spec Phase 0)

- Date: 2026-11-06
- Commit audited: `main` at `4db44c0`
- Machine: Apple M3 Pro, 18 GB. Tests only: the machine was on battery, so no benchmarks ran.
- Baseline: `cargo test --workspace`: 293 passed, 0 failed, 1 ignored. Benchmarks are the
  committed records in `docs/performance/baselines/`.

This audits the repo against the owner's "Complete Implementation Specification" (the
platform spec), received 2026-11-06. It is the spec's Phase 0 deliverable. The spec itself
is not committed; the plan below carries its phases, and §5 asks whether to add it.

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
| S3 | Memory planner | 5 | Mostly production | Yes through `qubivex plan`; `models analyse` is absent |
| S4 | Runtime, expert streaming | 6, 7 | Mostly production | Yes |
| S5 | HTTP server | 9 | Production | Yes |
| S6 | Streaming | 9 | Production | Yes |
| S7 | Concurrent scheduler | 9 | Partial | Requests are correct and share weights; the queue is not bounded by size and there are no priorities |
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
| KV cache | Partial | `KvCache` in `qubivex-arch` is per-conversation, bounded and reserved through the planner. No prefix reuse, eviction or paging, and no crate boundary |
| Backends | Partial | CPU and Metal bit-identical. CUDA and Vulkan exist only as detection |
| Stable interfaces | Partial | Traits exist for `ComputeBackend`, `ExpertSource`, `DenseWeights`, `ExpertProvider`, `ArchitectureFactory`, `ResourceProbe`. There is no engine contract, no generation request type shared across products, and no model handle |
| Sampling | Partial | Greedy, temperature, top-p, seed. **No top-k** |

### Model manager (Product B)

| Requirement | State | Evidence |
|---|---|---|
| Containers, manifest v1, validation | Production | `qubivex-model`: hardened GGUF and SafeTensors readers, proptests, JSON schema |
| Import, discovery, verify | Production | `models import / list / inspect / verify` |
| Downloads | **Missing** | No HTTP client in the workspace. `QUBIVEX_DESIGN.md:126` deferred it to Phase 11. The spec makes it S2 and acceptance criterion 1 |
| `models pull / search / remove` | **Missing** | |
| `models analyse` | **Missing** as named | `qubivex plan` does the work; needs wiring |
| Registry | Partial | "Registry" today means the architecture registry. Models are found by scanning directories. No record of revisions, install state, duplicates or failures, and no atomic updates |
| Lifecycle states | Missing | Only the gateway's loaded map |
| Manifest fields | Partial | No source repository, revision pin, per-file checksums or expert fields; checksums are `experts.qvxsum`. The schema is versioned, with no migration path yet |

### Server (Product C)

| Requirement | State | Evidence |
|---|---|---|
| OpenAI chat and completions, SSE | Production | `qubivex-api`; official SDK against OLMoE and Qwen3 |
| Backpressure, disconnects | Production | 64-event bounded channel; drop guards; tests over HTTP |
| Unsupported parameters | Production | Refused, not ignored (ADR-013) |
| Top-k | Missing | Follows the runtime gap |
| Model endpoints | Partial | `POST /v1/admin/models`, `DELETE /v1/admin/models/{id}`. The spec names `POST /v1/models/load` and `/unload` |
| Load dedup | **Missing** | `admin_load` checks, then loads outside any lock: two concurrent loads of one id both read the weights, and the second gets a 409 only afterwards |
| Safe unload | Production | `Arc` holds the model for running requests; slots close at once |
| Queue | Partial | A semaphore with a timeout. Waiters are limited by time, not count, so a burst parks without bound for up to `--queue-timeout` |
| Priorities, deadlines, per-client fairness | Missing | FIFO only. `--request-timeout` is a global setting |
| Memory admission | Production | `reserve_kv` against the plan |
| Batching | **Missing** | See §3 |
| Auth | Partial | Optional API key and admin token, constant-time. No key refuses a non-localhost bind |
| Rate limiting | Missing | The 429 means queue timeout only |
| Request limits | Partial | Axum's default body limit only; no configured prompt length, no explicit limit |
| CORS | Missing | |
| Graceful shutdown | Partial | `with_graceful_shutdown` exists; shutdown with active generations is untested |
| Metrics | Partial | Requests, tokens, active, cancelled, failed, KV, cache bytes, slots. No TTFT, tokens/s, latency histograms, queue depth, hit rate, evictions or disk throughput |
| Structured logs | **Missing** | `eprintln!` with a request id. No model or sequence ids, levels or format |
| Config files | **Missing** | Flags and three environment variables. No TOML, no precedence rule |

### CLI

| Spec command | State |
|---|---|
| `system` | Done |
| `models list / inspect` | Done |
| `models validate` | Done as `models verify`; needs an alias |
| `serve` | Done, with `--listen` instead of `--host` and `--port` |
| `chat --model --prompt` | Done as `run`. Interactive chat is missing |
| `server status` | Missing |
| `models search / pull / analyse / remove` | Missing |

`run` and the gateway both call `Engine::generate_with`, so there is one inference
pipeline, as the spec asks. Template rendering and stop-sequence handling sit in
`qubivex-api/src/worker.rs`, so embedders of the runtime do not get them.

### Tests the spec requires

| Test (§12.3) | State |
|---|---|
| 1. Ten simultaneous requests | Missing at that count; 4-that-fit tested |
| 2. Fifty, bounded | Missing |
| 3, 4. Same expert, concurrent loads | Covered in the store's tests and loom |
| 5. Eviction while executing | Covered by the store's leases |
| 6. Model load dedup | Missing, and the code fails it |
| 7, 8. Cancellation, SSE disconnect | Covered |
| 9. Unload while active | Not tested with a live request |
| 10. Memory exhaustion | Covered for KV admission |
| 11. Shutdown with active requests | Missing |
| 12. Different context lengths | Missing |
| 13. Multiple models | Covered only by the SDK script, which is not in CI |

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
4. **Downloads add the workspace's first outbound network and TLS code.** That needs a
   dependency decision (§5) and a threat review: redirects, path traversal from repository
   file names, and token handling.

Rules the code already follows: the runtime does not depend on the server, the server has
no tensor code, and the model crate executes nothing.

**Not adopted from the spec's target layout:** `qubivex-tensor`, `qubivex-moe`, `apps/server`
and `apps/model-manager`. Splitting them now would move working code with no behaviour
gained, against ADR-001 and the spec's own preservation rules. New crates are justified for
new code only: `qubivex-model-download` and `qubivex-model-registry`.

## 4. Plan

Each row is one job and one PR. Dependencies run downwards.

| # | Job | Spec | Depends on | Done when |
|---|---|---|---|---|
| 1 | Generation contract | S1 | | A `GenerationRequest`, stop handling and templates live in the runtime; `run` and the gateway both use them; top-k added; tokens unchanged |
| 2 | Registry | S2 | | Atomic store of installed models, revisions and lifecycle states; survives a failed install; `models remove`, `validate` alias |
| 3 | Downloader | S2 | 2 | Resumable, verified, atomic `models pull` and `search` from Hugging Face; fault tests against a local server; one real pull |
| 4 | Analyse and manifest v1.1 | S2, S3 | 2 | `models analyse`; source, revision and checksums in the manifest, with a migration test |
| 5 | Scheduler | S7 | 1 | Queue bounded by count, priorities, deadlines, load dedup; §12.3 tests 1, 2, 6, 9, 11, 12, 13 |
| 6 | KV subsystem | S4 | 1 | Prefix reuse with a bit-identity test; bounded and evictable |
| 7 | Batching design | S8 | 5, 6 | ADR with a measured prototype |
| 8 | Batching, stages 2 and 3 | S8 | 7 | Sequences share a pass; tokens bit-identical; measured on power |
| 9 | Hardening | S10 | 5 | Rate limits, request limits, CORS, non-localhost refusal, config files |
| 10 | Observability | S10 | 5 | The §8 metrics; structured logs with no prompt text |
| 11 | Deployment | S10 | 9, 10 | Docker, runbooks, the §14 docs |
| Blocked | CUDA | S9 | B2 | Needs NVIDIA hardware; abstraction only until then |
| Later | Vulkan | S9 | | Spec says "where practical" |

## 5. Owner decisions

1. **Numbering.** Keep `MIGRATION_PLAN.md`'s phases and use S-numbers for the spec, as here?
   Recommended: yes, because renumbering would break references in every ADR.
2. **HTTP client.** Recommended: `ureq` with `rustls`, blocking, on dedicated threads. It
   matches the runtime's thread model, and `qubivex-model` stays free of Tokio. `reqwest` is
   the alternative, and it pulls in Tokio.
3. **Turn lock.** Keep it until S8 (finding 1)? Recommended: yes.
4. **Endpoint names.** Add `POST /v1/models/load` and `/unload` beside the admin routes, or
   replace them? Recommended: add, and keep the admin routes as aliases.
5. **Crate layout.** Accept the "not adopted" list in §3? It restates ADR-001.
6. **Spec in the repo.** Commit the spec as `docs/architecture/PLATFORM_SPEC.md`? The earlier
   specification the docs cite is not in the repo either. Recommended: yes.
