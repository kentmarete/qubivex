# Evaluation of the Proposed Improvements Over Colibri (Addendum §4)

Status: Phase 0 deliverable. Proposed, awaiting review.
Source of truth: Colibri `0a5504a1`. References are relative to its checkout. See
[COLIBRI_ANALYSIS.md](COLIBRI_ANALYSIS.md) for the full analysis.

Each item below follows the addendum's procedure: **does it exist in Colibri**
(evidence), **what is limited**, **the QUBIVEX change**, **which goal it serves**, **how
the benefit is measured**, and a **verdict**.

Goals, abbreviated: **LAT** lower latency, **HW** hardware utilisation, **MEM** lower
memory, **COMPAT** model compatibility, **PRED** predictable execution, **MAINT**
maintenance, **EXT** extensibility.

Verdicts:
- **Adopt**: build it as described.
- **Adopt-modified**: build it, changed as stated.
- **Port**: Colibri already does it well; keep the behaviour and fix the structure.
- **Gate**: build it as an option; it becomes the default only if a benchmark shows a gain.
- **Defer**: not before the stated phase.

### Headline finding

Colibri already implements a substantial part of this list: router-lookahead
prefetch, in-flight dedup, LFRU with hysteresis, a Prometheus endpoint, resource
policies, a model recommender, cross-request expert sharing, and memory-derived cache
sizing. QUBIVEX's demonstrable gains are therefore in five areas:

1. **Extensibility.** Colibri needs a new C engine per family.
2. **One measured planner** instead of two disconnected ones (Python and C).
3. **Failure isolation.** Colibri has hundreds of `exit(1)` calls.
4. **Measured storage and runtime bottleneck diagnosis.**
5. **Memory-aware admission and pressure response.**

Raw speed parity with Colibri is a *target to match*, not an assumed win.

---

## §2 Eliminate hardcoded model implementations (P0)

**In Colibri:** partly.
- No `if name.contains(…)`. Families resolve by `config.json` `model_type` in
  `family_registry.py`.
- But every family has its own engine (`olmoe.c`, `qwen36.c`, …, 13 in total), its
  own Python geometry function (`_olmoe_geometry`, `family_registry.py:560`) and
  family-specific planner branches (`_q38_*`, `_glm53_*` in `resource_plan.py`).
- Checkpoints with an *identical* architecture already run unchanged (README: KAT-Coder
  on the Qwen3.6 engine).

**Limitation:** a new family with a known structure (for example another softmax-top-K
GQA MoE) still needs a new or modified C engine plus Python registry and planner code.

**QUBIVEX:** the manifest, `ArchitectureRegistry`, and `tensor_schema` that assigns a
role to every tensor (ADR-003/004). The planner and validator operate on roles only.

**Goals:** COMPAT, EXT, MAINT.

**Measure:** OLMoE and a tiny Qwen3-MoE (GQA, top-K renormalisation, different tensor
names) both load through `transformer_moe` with **zero runtime diff**. A CI guard
enforces this.

**Verdict: Adopt.**

## §3 Hardware-aware memory planning (P0)

**In Colibri:** yes, but split in two.
- `resource_plan.py` (`coli plan`) gives hot, warm and cold tiers, policies, cgroups,
  `next_actions`.
- Each engine computes its own load-time budget (`c/olmoe.c:837`:
  `resident + 0.88 × MemAvailable`, a 0.5 GB reserve, KV sharing the expert room via
  `kv_room_fit`).
- **Storage speed is measured only** by the GLM engine on Metal and macOS startup
  (`.coli_ssd`, `resource_plan.py:2614`). Everywhere else the plan says "storage
  measurement required".
- No runtime re-planning beyond KV-driven cache shrinking.

**Limitations:**
- Two sources of truth.
- No storage measurement on Linux or Windows.
- No reaction to *other* processes taking RAM mid-session.
- Uniform slots per layer.

**QUBIVEX:**
- One planner used by `plan`, by load and at runtime (ADR-005).
- `qubivex doctor --storage` measures sequential and expert-sized random reads on every
  OS, cached per device id.
- A `MemoryLedger` plus a pressure monitor (§6, §8).
- Per-layer non-uniform cache budgets.

**Goals:** HW, MEM, PRED.

**Measure:**
- Plan-vs-ledger-vs-RSS error stays under 5% on OLMoE.
- Property tests: plans never exceed their budget.
- An external RAM hog test: QUBIVEX shrinks its cache and keeps serving, where Colibri
  keeps its budget.
- Tokens/s at an equal RAM budget is no worse than Colibri.

**Verdict:** **Adopt-modified.** Colibri's measured constants are kept as defaults.

## §4 Predictive expert loading (P2)

**In Colibri:** yes. `PILOT` 1–3 layer lookahead (`c/olmoe.c:2368`):
- the next layer's norm and router applied to the current hidden state;
- an EMA momentum, a confidence cutoff (0.92) and a width cap;
- an `is_queued` dedup and an LFRU eviction guard;
- persistent usage history (`.coli_usage`), and `PILOT_REAL` in the GLM engine.

It measured 71.6% next-layer recall on GLM-5.2. It also **measured net-negative on
disk-saturated hosts**, and it is off by default.

**Limitations:**
- One worker thread polls with `sleep(1ms)`.
- A fixed 4096-entry ring, with no priorities.
- **No cancellation of queued speculative reads that went stale.** Entries are only
  dropped when an `is_queued` flag is cleared.
- Recall is not reported per turn.
- Demand misses and speculative reads compete on equal terms.

**QUBIVEX:**
- The same predictor (port, not reinvent).
- Prefetch sits in the I/O scheduler at `Prefetch` priority, below `Demand`.
- When layer L+1 starts, any unfinished speculation for L+1 is cancelled.
- Prefetch is throttled when the measured disk queue is saturated (this addresses
  Colibri's known negative case).
- Metrics: recall, useful prefetch rate, wasted bytes, lead time, and blocking time
  saved.
- Deeper lookahead or learned predictors only when a benchmark shows they help.

**Goals:** LAT.

**Measure:** blocking expert-wait time per token with prefetch on vs off, at an OLMoE
cache budget of 1 GB and 2 GB, cold and warm. A test also proves routing is identical
with prefetch on and off.

**Verdict:** **Port plus Adopt-modified** (priority, cancellation, saturation throttle).
**Gate:** default-on only if blocking time falls with no throughput loss.

## §5 Disk I/O and streaming (P0/P1)

**In Colibri:** extensive, in the GLM engine.
- A `PIPE` loader pool (on by default, −18% disk service).
- `URING=1`, `DIRECT=1` (+65% on one host, worse on others), `COLI_MMAP`.
- `COLI_MODEL_MIRROR` with weighted striping (+37.5% on two NVMe drives).
- One read per expert, sorted by expert id for locality.

OLMoE, by contrast, has synchronous demand loads plus one prefetch thread.

**Limitations:**
- Each engine has its own I/O path. The best one (GLM) isn't shared.
- Modes are chosen by environment variables with no per-machine measurement (except
  `coli tune`).
- No backpressure across concurrent requests.
- No cancellation.

**QUBIVEX:**
- One `qubivex-expert-stream` used by every architecture (ADR-006).
- A bounded priority queue, dedicated I/O workers, slab pool, extent coalescing.
- Per-device queues so mirrors and multiple drives are scheduled independently.
- Cancellation of queued work.
- **A benchmark matrix:** mmap, `pread` worker pool, io_uring (Linux), and O_DIRECT /
  `F_NOCACHE`, run per OS and device. The default is picked from `doctor --storage`
  results, not hardcoded.

**Goals:** LAT, HW, MAINT.

**Measure:** expert-load latency p50/p99, GB/s against the device's measured ceiling,
and decode tok/s at a fixed cache budget, compared with Colibri's GLM-style pool on the
same files (both read the same GGUF).

**Verdict: Adopt.** The strategy is chosen per device by measurement.

## §6 Cache architecture (P1/P2)

**In Colibri:**
- RAM LRU per layer with leases (`busy`) and pins.
- `tier.h` LFRU with 25% + 4 hysteresis and decay.
- VRAM tiers (`vk_tier.c`, `cu_tier.c`) with "evict the RAM copy first if the GPU holds
  it" (`vkt_ram_first`).
- The KV cache shares room with the expert cache.
- A persistent usage history.
- Coverage analysis tooling (`tools/usage_coverage.py`) and a trace-replay simulator
  (`tools/residency_sim.py`).

**Limitations:**
- Policy is hardwired per engine.
- **No cost-awareness:** every expert is treated as equally costly to reload.
- Cache sizing is uniform per layer.

**QUBIVEX:**
- `EvictionPolicy` trait with LRU, LFU, decayed LFU (LFRU) and cost-aware
  implementations. Cost = measured reload latency from the expert's source device ×
  bytes.
- Independent RAM and VRAM tiers, each copy accounted separately.
- Pinning within a capped share of the budget.
- A trace-replay simulator in Rust for offline policy comparison.

**Expectation:** cost-aware eviction likely shows **no gain on OLMoE from one disk**,
because all experts have the same size and source. It should matter with mixed
per-layer quantisation (GGUF layers differ in type), multiple drives, or RAM↔VRAM
transfer costs. This is stated in advance so a null result is not mistaken for a bug.

**Goals:** LAT, MEM.

**Measure:** the simulator replays recorded routing traces and reports hit rate and
modelled stall time per policy. The live benchmark only runs for policies that win in
the simulator.

**Verdict:** **Port** LRU and LFRU (the LFRU default). **Gate** cost-aware and
predictive retention.

## §7 Compute backend independence (P1)

**In Colibri:**
- `backend_cuda.cu` is one source for CUDA and HIP via a 14-symbol shim.
- `xtier.h` shares the expert tier across Vulkan and CUDA.
- But backends are compile-time `#ifdef` blocks inside every engine, and there's no
  operation-level capability query. Each engine knows what its backend supports.

**Limitations:** adding a backend touches every engine. Unsupported operations show up
as engine-specific `exit(1)` ("no kernel for a GGUF expert of type …",
`c/olmoe.c:1729`).

**QUBIVEX:**
- `ComputeBackend` with `capabilities()`, a matrix of dtype × op × layout (ADR-007).
- The planner places a tensor on a device only if every op that touches it is
  supported there. Otherwise there's a **controlled fallback** to the CPU for that
  tensor group, reported in the plan.
- If no backend supports an op, the result is an error at *plan* time, never at token
  time.

**Goals:** EXT, PRED, MAINT.

**Measure:** a test backend that declares a partial capability set produces a correct
mixed plan, and its output matches CPU tokens.

**Verdict: Adopt.**

## §8 Resource scheduling and concurrency (P1)

**In Colibri:**
- One engine process per model, a single shared expert cache.
- `--kv-slots N` (≤16) gives independent KV per conversation.
- Mux continuous batching decodes all slots' rows in **one forward**. Through
  expert-major batching (`moe_by_expert`), each expert is fetched once per step across
  requests ("route, union, place").
- The gateway `GenerationScheduler` (`openai_server.py:179`) has a bounded queue
  (429 when full), a queue timeout, cancellation and metrics.

**So "concurrent requests share resident experts and don't duplicate loads" is already
true in Colibri.**

**Limitations:**
- Admission counts **slots**, not memory. KV growth is absorbed by shrinking the expert
  cache, which can quietly collapse performance for everyone.
- No priorities or fairness between long and short requests.
- One model per process. Loading a model means a new process.
- A fatal error in one request can take down the engine process, and every
  conversation with it.

**QUBIVEX:**
- An in-process scheduler with **memory-aware admission**: a request is admitted only
  if its projected KV fits without pushing the expert cache below the planner's floor.
  Otherwise it waits or is rejected with a reason.
- Per-step fair batching (prefill chunking so a long prompt can't starve decodes).
- Cancellation propagated to queued I/O.
- Several models per process with shared immutable weights (`Arc`).
- Per-request error isolation.

**Goals:** PRED, LAT, HW.

**Measure:**
- At N = 1, 4 and 8 concurrent requests: p99 time-to-first-token and per-request
  tok/s spread, plus expert bytes read per step (it must not scale with N for shared
  experts).
- Cache-floor violations: zero.
- One request with a forced failure: the others complete.

**Verdict:** **Port** the shared-expert batching. **Adopt** memory-aware admission,
fairness and isolation.

## §9 Observability (P2)

**In Colibri:**
- `/metrics` Prometheus with scheduler gauges, counters and histograms (queue wait,
  first output, engine call; `openai_server.py:313`).
- Per-turn `PROF` lines: wall time, disk, expert matmul, attention, head.
- `TIERS` and `EMAP` (expert map), a dashboard Profiling page, `coli doctor`.
- `resource_plan.py` classifies a *projected* bottleneck (disk or compute) with
  `next_actions`.

**Limitations:**
- No runtime expert-cache hit, latency or prefetch metrics in Prometheus.
- Bottleneck diagnosis is static (projected at plan time), not measured from the
  running workload.

**QUBIVEX:**
- `metrics` counters and histograms from every subsystem: cache hit/miss/evict per
  tier, load latency, bytes per token, prefetch recall and waste, queue depth, ledger
  bytes per tier.
- Exposed through `/metrics`, `GET /v1/system/resources` and `qubivex status`.
- A **runtime diagnoser** classifies each turn from measured stage times into: waiting
  on storage, storage bandwidth saturated, CPU-compute bound, GPU waiting on weights,
  eviction thrash (evictions of experts reused within N tokens), KV pressure (the cache
  floor reached), or insufficient VRAM (the planner capped the GPU tier). Each verdict
  comes with its evidence numbers.

**Goals:** PRED, MAINT.

**Measure:**
- Metrics overhead under 1% of decode time (A/B with metrics compiled out).
- The diagnoser labels synthetic scenarios correctly: a throttled disk is labelled
  storage-bound, cache 1 is labelled thrash, a large context is labelled KV pressure.

**Verdict:** **Adopt.** Colibri's PROF stage breakdown is kept as the basis.

## §10 Failure handling and recovery (P0 for correctness)

**In Colibri:**
- Validation is strong: exact tensor-size checks reject crafted headers, config range
  checks, NaN-safe router picks. Load failures are reported to the gateway (the "load
  fail" protocol path).
- But error handling is mostly `exit(1)`: 41 call sites in `olmoe.c`, 97 in
  `colibri.c`, 128 in `qwen36.c`. That includes OOM during inference, missing kernels
  and container errors. The gateway survives; the model and every conversation on it
  do not.

**QUBIVEX:**
- Typed errors per domain.
- Load failures move the model to `Failed(reason)` and release everything (RAII,
  verified by the ledger reaching zero).
- Request-time failures fail the request with a structured error and leave the model
  `Ready`.
- A read error on one mirror retries on another.
- A corrupt expert read (size or optional checksum mismatch) is surfaced and never
  substituted with another expert.

**Goals:** PRED, MAINT.

**Measure:** a fault-injection suite where an `ExpertSource` wrapper injects EIO,
short reads, corrupt bytes, allocation failure and cancellation mid-layer. In every
case: no panic, no wrong tokens emitted, the ledger returns to baseline, and the next
request succeeds.

**Verdict: Adopt.** This is the clearest structural improvement over Colibri.

## §11 Benchmarking and numerical validation (P0)

**In Colibri:**
- Strong: tiny vendor-generated fixtures with token-id equality in CI, teacher forcing,
  GPU-vs-CPU parity on a software driver, sanitizers.
- A benchmark protocol (`docs/benchmarking.md`) and benchmark tools.
- Gap: intermediate tensors (router logits, expert outputs) are not compared
  systematically for every engine. Comparison is mostly at the token level.

**QUBIVEX:**
- Per-layer intermediate comparison (router logits, selected ids, routing weights,
  attention output, expert output, logits) against `transformers` on the fixtures.
- Cache-capacity sweep with **bitwise** logits.
- A benchmark harness running the matrix: RAM budget × cache size × quantisation ×
  context × concurrency, cold and warm, for QUBIVEX and Colibri on the same files.
  Results are written as JSON with hardware and commit metadata.

**Goals:** PRED, MAINT.

**Verdict: Adopt.** It extends Colibri's practice rather than replacing it.

## §12 Configuration and execution profiles (P2)

**In Colibri:**
- Policies `quality`, `balanced` and `experimental-fast` (`resource_plan.py:3309`).
- `coli tune` measures and saves per-machine settings.
- But the knobs are hundreds of environment variables, varying per engine
  (`docs/ENVIRONMENT.md`).

**QUBIVEX:**
- Profiles `auto` (default), `performance`, `balanced`, `memory-saver` and `custom`.
- A typed TOML config with layered sources (QUBIVEX_DESIGN §6).
- Overrides for RAM and VRAM budget, cache capacity, prefetch depth, concurrency,
  storage dirs, backend and quantisation preference.
- `qubivex config show` prints effective values and where each came from.
- **Lossy options are never part of any profile.** They're always explicit (invariant 4;
  Colibri's `experimental-fast` changes router decisions).

**Goals:** MAINT, PRED.

**Verdict: Adopt-modified.** No lossy profile.

## §13 Model discovery and recommendations (P2)

**In Colibri:**
- `setup_catalog.recommend()` uses a **hand-maintained catalog** (`ram_min_gb`,
  `ram_good_gb`, `dense_gb` per entry) against total RAM and free disk.
- `--list` explains why each model does or doesn't fit.
- Measured speeds are in the README tables, not used by the recommender.

**Limitations:**
- Catalog numbers are maintained by hand and use total RAM rather than available RAM.
- No VRAM consideration in recommendations.
- No context estimate.

**QUBIVEX:** `qubivex models recommend`:
- Runs the **real planner** over manifests, so sizes come from tensor schemas, not
  hand-entered numbers.
- Considers RAM, VRAM and disk, and reports the maximum context at the chosen profile.
- Expected speed is shown only when a benchmark record exists for a matching hardware
  class, labelled `measured`. Otherwise a roofline estimate (bytes per token ÷ measured
  bandwidth) is labelled `estimated`.

**Goals:** COMPAT, PRED.

**Verdict: Adopt.** Defer to Phase 11, after the planner and benchmark records exist.

---

## Priority mapping onto phases

| Addendum priority | Item | Phase |
|---|---|---|
| P0 | Manifest and registry | 3 |
| P0 | Memory planner | 5 (budget basics already in 4) |
| P0 | Disk-backed expert streaming | 4 (minimal store) → 6 |
| P0 | Numerical correctness | 3–4, then every phase |
| P0* | Failure handling (§10) | 3 onward. *I'm treating this as P0 because wrong output is a correctness failure* |
| P1 | Multi-tier caching | 6 (RAM), 8 (VRAM) |
| P1 | Async loading and dedup | 4 (dedup) → 6 |
| P1 | CPU/GPU hybrid | 8 |
| P1 | Memory-aware scheduling | 9 (admission logic is unit-tested in 5) |
| P2 | Predictive preloading | 7 |
| P2 | Advanced cache optimisation | 7–10 (gated by the simulator) |
| P2 | Observability | Metrics from 4. Diagnoser in 9 |
| P3 | Multi-GPU, distributed | Out of scope for MVP |

**Ordering note:** the addendum ranks CPU/GPU hybrid (P1) above predictive preloading
(P2), while the phase plan puts prefetch (7) before GPU (8). I propose keeping 7 before
8. GPU work is blocked on hardware (blocker B2), and prefetch builds directly on Phase 6
using the same machine. If CUDA hardware becomes available earlier, the two phases can
swap without interface changes.

## Acceptance criteria mapping (addendum §15)

| # | Criterion | Verified by | Milestone |
|---|---|---|---|
| 1 | A compatible model is added with no runtime change | Qwen3-MoE-tiny via manifest; CI grep and dependency guards | M3 |
| 2 | Hardware identified automatically | `qubivex system` on Linux and macOS; recorded-fixture unit tests | M2 |
| 3 | Valid allocation plan | Planner property tests; plan vs ledger vs RSS | M5 |
| 4 | Runs when experts don't fit in VRAM | Hybrid run with the VRAM tier smaller than the expert set (CPU run with a RAM budget below the expert set at M6) | M6 / M8 |
| 5 | Loading overlaps compute | Stage timings: blocking wait is less than total load time at a constrained cache | M6 |
| 6 | Concurrent requests share resident experts | Bytes read per step don't scale with N; one ledger entry per expert | M9 |
| 7 | Pressure triggers controlled rebalancing | External RAM-hog test: prefetch stops, cache shrinks, active requests finish | M5 / M9 |
| 8 | Prefetch reduces blocking load latency | On/off A/B at 1 and 2 GB budgets, cold and warm | M7 |
| 9 | Numerics within tolerance | Per-layer fixture comparisons; bitwise capacity sweep; greedy tokens | M4 onward |
| 10 | Equivalent comparisons | Benchmark harness records hardware, files, flags and cache state for both engines | M6 onward |

"Measurable improvement over Colibri" will be reported per criterion as **better /
equal / worse with numbers**. On raw single-request decode speed, matching Colibri is
the realistic goal. The expected wins are in criteria 1, 3, 6, 7 and §10, plus storage
measurement on platforms where Colibri has none.
