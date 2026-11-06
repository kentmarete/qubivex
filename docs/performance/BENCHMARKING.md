# Benchmarking

Every kernel or runtime change that can affect speed is measured before and after, on
the same machine, with the two tools below (MIGRATION_PLAN §3a). A change that claims a
speed-up includes both measurements in its PR.

| Tool | Measures | Where |
|---|---|---|
| Criterion kernel benches | One matmul or block operation, isolated | `crates/qubivex-backend-cpu/benches/kernels.rs` |
| `qubivex bench` | Prefill and decode of a real model, end to end | `apps/cli/src/bench.rs` |

## Kernel benches (Criterion)

```sh
cargo bench -p qubivex-backend-cpu --bench kernels -- --save-baseline before
# ... change the kernel ...
cargo bench -p qubivex-backend-cpu --bench kernels -- --baseline before
# One group only (a regex over benchmark ids):
cargo bench -p qubivex-backend-cpu --bench kernels -- decode_stream --baseline before
```

Shapes are the expert projections of the OLMoE reference model: gate/up `1024 × 2048`,
down `2048 × 1024`, plus the `50304 × 2048` output head. Throughput is weight bytes per
call, so Criterion prints GiB/s. Weights are random but valid (sane block scales); the
numbers do not depend on their values.

| Group | What it isolates |
|---|---|
| `decode_1t` | One activation row, one thread: the kernel alone, no pool |
| `decode_t<N>` | One row on the compute team; the matrix stays in cache |
| `decode_stream_t<N>` | One row on the team, cycling through 512 MiB of copies so every call reads from DRAM, as a decode step reads each expert once per token. **Bandwidth targets are measured here.** |
| `decode_portable_1t` | The portable path for the dtypes that have a fused kernel on this CPU, so the fused gain stays visible |
| `prefill_s128_t<N>` | 128 activation rows on the team (the register-blocked kernel) |
| `prefill_s16_t<N>` | 16 rows, Q4_K: what one expert sees in a 128-token prompt |
| `dequant_row`, `dot` | The building blocks, one 2048-wide row |

The team size `N` is `QUBIVEX_BENCH_THREADS`, else the performance cores on macOS, else
every logical CPU. It is part of the group name, so results at different team sizes are
never compared by accident. Use the same `N` as the engine (`qubivex system` shows the
performance cores).

Criterion baselines live in `target/criterion/` and are not committed. Copy the
relevant medians into the PR.

## End-to-end harness (`qubivex bench`)

```sh
cargo build --release
./target/release/qubivex bench --model <id> --label before --out before.json
# ... change ...
./target/release/qubivex bench --model <id> --label after --out after.json --reference before.json
```

Each run starts from an empty KV cache. Prefill is one forward pass over the prompt.
Each decode step is one forward pass over one token, timed on its own. Decoding is
greedy, and an end-of-sequence token does not stop a run, so every run does the same
work. Warm-up runs (default 1) are not measured. They also fill the expert cache, so
measured runs see no expert loads at the default all-resident capacity. Pass
`--warmup 0 --runs 1` to measure a cold process instead.

| Option | Default | |
|---|---|---|
| `--prompt` | the Colibri-comparison prompt | encoded as `qubivex run` encodes it |
| `--prompt-tokens N` | the encoded prompt | the prompt repeated or cut to `N` tokens |
| `--decode-tokens` | 128 | tokens generated per run; the first comes from prefill |
| `--warmup` / `--runs` | 1 / 5 | |
| `--threads`, `--expert-cache-mb`, `--context` | as `qubivex run` | |
| `--reference FILE` | | a previous record, or a JSON array of ids, whose tokens must match |
| `--out FILE`, `--json`, `--label` | | |
| `--cache-policy` | `lfru` | the expert pools' replacement policy, `lfru` or `lru` (ADR-008) |
| `--io-threads N` | 4 | expert I/O threads; 0 makes the compute thread read its own misses (ADR-006) |
| `--direct-io` | off | read experts bypassing the page cache (`F_NOCACHE` on macOS, `O_DIRECT` on Linux); recorded as `config.expert_io` (ADR-006 amendment 2) |
| `--routing-trace FILE` | | writes every expert acquire of the measured runs, for trace replay (below) |

The JSON record (schema 1) holds:

- **build:** version, git commit (`-dirty` with uncommitted changes), rustc, profile,
  opt-level, debug assertions, target, compile-time target features, RUSTFLAGS;
- **hardware:** the CPU (model, core tiers, run-time SIMD), RAM and OS, as `qubivex
  system` reports them;
- **config:** threads, expert-cache size, context, prompt and generated tokens, warm-up
  and measured runs;
- **prefill:** time per run (min, median, P95, max, mean), and prompt tokens per second at
  the median;
- **decode:** time of every measured step (min, median, P95, max, mean), tokens per
  second at the median step, and each run's tokens per second;
- **memory:** RSS after load and the peak RSS of the process. On Linux the peak is the
  kernel's high-water mark (`VmHWM`). macOS reports a peak only through `getrusage`,
  which this code base does not call (it would need FFI), so the peak there is
  `sampled`: the largest RSS seen after load and after each run, outside timed regions.
  A sampled peak can miss transient scratch inside a forward pass;
- **experts:** expert-store hits, loads, bytes read and evictions during measured runs;
  the hit rate, bytes read per generated token, slab allocations and the time the
  forward pass was blocked on loads per token (Phase 6). The config holds the policy,
  the I/O threads and the slots of each layer's pool. Records from before
  Phase 6 lack these fields and still load;
- **check (the error field):** whether every run produced the same tokens (invariant
  1), and whether they match `--reference`. Lossless kernel changes must keep the tokens
  identical. The command fails if runs disagree with each other.

The text summary prints the same fields on one screen.

### Trace replay (cache policies)

A routing trace replays through the real expert store over a source that reads
nothing, so policies and slot counts are compared in seconds, without the model:

```sh
./target/release/qubivex bench --model <id> --ram-budget 1GiB --routing-trace trace.txt
cargo run --release -p qubivex-expert-stream --example replay -- trace.txt [slots...]
```

It prints the hit rate per slots-per-layer for a global LRU of the same size, pool LRU
and pool LFRU at several decay windows. A new policy must beat the default here before
it ships (ADR-008). Results: ADR-008 amendment 1.

### Prefetch (Phase 7)

`--prefetch` (with `--prefetch-guard on|off` and the predictor's settings) adds an
`experts.prefetch` section to the record: recall and precision of the prediction,
issued, present and refused predictions, ready and late uses, wasted reads, cancelled
prefetches and the mean lead time. Compare it against a run without `--prefetch` at the
same budget, interleaved. Blocked time per token is the latency prefetch hides, and the
decode rate says whether that mattered. Results: ADR-008 amendment 2.

### Device placement (Phase 8)

`--backend metal` runs whole tokens on the Apple GPU (`--metal-ops token`, the default:
whole layers with routing on the GPU, one submission per pass while every expert is
resident), whole layers with routing on the host (`--metal-ops layers`), or with
`--metal-ops attention,router,head` only those dense matmuls, and adds
`config.device` to the record: backend, device, placed kinds, tensor count and bytes.
Placement never changes bits, so always pass `--reference` with a CPU record: the
tokens must match. Kernel-level numbers come from the Criterion groups of
`cargo bench -p qubivex-backend-metal --bench kernels`: `metal_batch` (batching) and
`metal_stream` (kernel throughput over 96 distinct expert matrices in one submission,
with no per-call overhead). Measure Metal both with every expert resident and under a
small `--ram-budget`: the two regimes rank the backends differently. When the plan's
pools are smaller than top-k, or any holds fewer than half its layer's experts,
`--backend metal` runs the CPU pass, and the record's
`config.device.fallback` says so. Measure prefill at several prompt lengths (`--prompt-tokens`): the token-tiled and grouped kernels switch on at 4 tokens and at 2 routed entries per expert. Results: ADR-011 amendments 1–6.

### How to read the numbers

- **Report the median, and the P95 next to it.** A P95 far above the median means jitter
  (other processes, thermal state, the efficiency cores), not kernel speed. If P95 moves
  and the median doesn't, check the machine before blaming the change.
- **`qubivex run` reports less than `qubivex bench`** for the same model, prompt and
  length (21 vs about 25 tok/s on the baseline machine). This is expected, not a
  discrepancy. `run` is one cold process: its first tokens load about 1,000 experts
  (3.5 GiB) from the file during decode. `bench` measures warm runs. Both are real; say
  which one you quote.
- **Same machine, same build settings, same thread count, back to back.** Close other
  heavy programs, keep the laptop on power, and interleave before and after runs when
  the difference is small.

## Baseline: 2026-11-03, Apple M3 Pro

The first measurements, against which later changes are compared. Build: `05ad5eb` plus
the uncommitted tensor-descriptor work (`-dirty`), `cargo build --release` (`lto =
"fat"`, no `target-cpu=native`), Rust 1.99.1. Machine: Apple M3 Pro (5 performance + 6
efficiency cores), 18 GiB, macOS 27.2. Five threads.

### End to end (OLMoE-1B-7B-0125-Instruct Q4_K_M)

Records: `baselines/2026-11-03-m3pro-decode.json` and
`baselines/2026-11-03-m3pro-prefill128.json`. Both: 1 warm-up and 5 measured runs, all
experts resident, tokens identical across runs.

| Operating point | Prefill median (P95) | Prefill tok/s | Decode step median (P95) | Decode tok/s (median step) | Per-run decode tok/s | Peak RSS (sampled) |
|---|---|---|---|---|---|---|
| 10-token prompt, 160 generated (the Colibri comparison) | 323.8 ms (410.7) | 30.9 | 38.82 ms (51.59) | 25.76 | 22.81–25.42 | 3.74 GiB |
| 128-token prompt, 32 generated | 2908.6 ms (2961.8) | 44.0 | 39.46 ms (47.44) | 25.34 | 21.39–25.39 | 3.22 GiB |

### Kernels (Criterion medians)

Team = 5 threads. GiB/s is weight bytes per call.

| dtype | `decode_1t` gate/up | `decode_t5` gate/up | `decode_stream_t5` gate/up | `decode_stream_t5` down | `decode_portable_1t` gate/up | `prefill_s128_t5` gate/up |
|---|---|---|---|---|---|---|
| f32 | 230 µs, 33.9 GiB/s | 63.3 µs, 123 GiB/s | – | – | – | 3.49 ms |
| f16 | 1.27 ms, 3.07 GiB/s | 281 µs, 13.9 GiB/s | – | – | – | – |
| bf16 | 170 µs, 23.0 GiB/s | 45.5 µs, 85.9 GiB/s | 47.5 µs, 82.2 GiB/s | 48.9 µs, 79.9 GiB/s | 301 µs, 13.0 GiB/s | 3.51 ms |
| q8_0 | 204 µs, 10.2 GiB/s | 64.7 µs, 32.1 GiB/s | 54.5 µs, 38.1 GiB/s | 56.3 µs, 36.9 GiB/s | 724 µs, 2.87 GiB/s | 3.56 ms |
| q4_k | 240 µs, 4.58 GiB/s | 62.8 µs, 17.5 GiB/s | 63.1 µs, 17.4 GiB/s | 64.0 µs, 17.2 GiB/s | 373 µs, 2.95 GiB/s | 3.44 ms |
| q6_k | 243 µs, 6.59 GiB/s | 63.9 µs, 25.1 GiB/s | 64.8 µs, 24.7 GiB/s | 66.0 µs, 24.3 GiB/s | 1.01 ms, 1.59 GiB/s | 3.59 ms |

Output head, Q6_K `50304 × 2048`, `decode_t5`: 2.77 ms (28.4 GiB/s).
Blocks (2048 values): `dot` 150 ns; `dequant_row` f32 95 ns, bf16 191 ns, q4_k 258 ns,
q8_0 559 ns, q6_k 877 ns, f16 995 ns.

### What the baseline says

- **The quantised decode kernels are limited by compute, not memory.** For every dtype,
  `decode_stream` (DRAM) matches `decode_t5` (in cache), and bf16 pulls 80 GiB/s from
  DRAM. Q4_K reaches 17 GiB/s and Q6_K 24 GiB/s: roughly 22% and 30% of what this
  machine demonstrably streams. The EIP's quantised-GEMV target is at least 70% of STREAM.
  A faster lossless Q4_K kernel is the largest available decode gain. (Done 2026-11-04:
  17 → 25–28 GiB/s, see below and ADR-002 amendment 2.)
- **Prefill does not use batching yet.** At 128 rows every dtype takes about 3.5 ms per
  expert projection, so the per-row scalar dot dominates, and end-to-end prefill is
  44 tok/s. (Done 2026-11-04 with an own kernel instead of `gemm`: see below and
  ADR-002 amendment 1.)
- **f16 has no fused kernel** and is 5–7× slower than bf16. No current reference-model
  tensor is f16, so this is low priority.
- **Decode jitter:** the step P95 is 20–30% above the median in both end-to-end runs,
  with occasional steps over 100 ms. Before attributing a few-percent decode change to a
  kernel, compare medians over interleaved runs.
- No STREAM number was recorded on this machine. bf16's 80 GiB/s is a measured lower
  bound for achievable bandwidth.

## 2026-11-04: prefill (ADR-002 amendment 1)

Build `f2ff9d5` plus the change (`-dirty`), same machine and flags as the baseline,
5 threads, 1 warm-up and 5 measured runs. Every record's tokens match its reference
record (`--reference`).

| Operating point | Record | Prefill median | Prefill tok/s | Decode tok/s (median step) |
|---|---|---|---|---|
| 10-token prompt, 160 generated | `2026-11-04-m3pro-tiled-decode.json` (ref. `…-decode.json`) | 244.7 ms (was 323.8) | 40.9 (was 30.9) | 30.50 (was 25.76) |
| 128-token prompt, 32 generated | `2026-11-04-m3pro-tiled-prefill128.json` (ref. `…-prefill128.json`) | 1132.6 ms (was 2908.6) | 113.0 (was 44.0) | 30.20 (was 25.34) |
| 10-token prompt, 160 generated, `--ram-budget 1GiB` | `2026-11-04-m3pro-tiled-1gib.json` (ref. `…-phase6-exit-1gib.json`) | 307.6 ms | 32.5 | 19.12 |

Kernel, `prefill_s128_t5` (Criterion medians; baseline in brackets):

| dtype | gate/up | down |
|---|---|---|
| f32 | 1.65 ms (3.49) | 1.81 ms |
| bf16 | 1.64 ms (3.51) | 1.82 ms |
| q8_0 | 1.72 ms (3.56) | 1.89 ms |
| q4_k | 1.66 ms (3.44) | 1.84 ms |
| q6_k | 1.81 ms (3.59) | 2.03 ms |

`prefill_s16_t5` Q4_K: gate/up 356 µs, down 377 µs.

Notes:
- The end-to-end gain is more than the kernel's 2.1×: experts now run concurrently and
  attention runs on the pool (it was single-threaded), so the compute team went from about
  50% to 95% busy in a profile.
- Decode gains 15–20% from computing the 8 routed experts concurrently.
- At `--ram-budget 1GiB` a 10-token prompt is I/O-bound and within noise of the previous
  build (main measured 320–326 ms in interleaved runs); a 128-token prompt at 1 GiB went
  48–49 → 98–100 tok/s.

## 2026-11-04: lossless Q4_K decode kernel (ADR-002 amendment 2)

Build `e06fa21` plus the change, same machine and flags, 5 threads, 1 warm-up and 5
measured runs. Every record's tokens match its reference record.

| Operating point | Record | Prefill tok/s | Decode tok/s (median step) |
|---|---|---|---|
| 10-token prompt, 160 generated | `2026-11-04-m3pro-q4k-fhm-decode.json` (ref. `…-tiled-decode.json`) | 43.1 | 41.65 (24.01 ms; was 30.50) |
| 128-token prompt, 32 generated | `2026-11-04-m3pro-q4k-fhm-prefill128.json` (ref. `…-tiled-prefill128.json`) | 114.4 | 42.49 (23.54 ms; was 30.20) |
| 10-token prompt, 160 generated, `--ram-budget 1GiB` | `2026-11-04-m3pro-q4k-fhm-1gib.json` (ref. `…-tiled-1gib.json`) | 33.0 | 22.39 (44.66 ms; was 19.12) |

Interleaved against a `main` build (two rounds each): 128-token prompt, prefill 113.8 /
113.8 vs 113.9 / 115.0 tok/s, decode 30.2 / 29.9 vs 42.2 / 42.5; at 1 GiB, decode
18.98 / 18.92 vs 22.70 / 22.58. One unrecorded run of the branch gave 101.5 tok/s
prefill after a long benchmark session; the interleaved runs show prefill unchanged, as
expected (prefill does not use the single-row kernel).

Kernels, Q4_K (Criterion medians; baseline `q4before` in brackets):

| group | gate/up | down |
|---|---|---|
| `decode_1t` | 141–144 µs (244) | 145–153 µs (235) |
| `decode_t5` | 38.8–42.2 µs (62.1) | 40.0–41.9 µs (66.1) |
| `decode_stream_t5` | 43.4 µs (63.0) | 45.4 µs (64.2) |
| `prefill_s128_t5` | 1.66–1.76 ms (1.66) | 1.83–1.89 ms (1.83) |

The ranges are two sessions; the second ran warm after the end-to-end benchmarks.
`decode_t5` gate/up moves 1.125 MiB per call, so 38.8 µs is 28.3 GiB/s.

## 2026-11-04: a dense model (`transformer_dense`, ADR-003 amendment 1)

`Qwen/Qwen3-0.6B`, the official bf16 SafeTensors checkpoint (1.11 GiB resident with the
tied head; no experts). Same machine and flags, 5 threads, 1 warm-up and 5 runs.

| Operating point | Record | Prefill tok/s | Decode tok/s (median step) |
|---|---|---|---|
| 10-token prompt, 160 generated, CPU | `2026-11-04-m3pro-dense-cpu-decode.json` | 156.8 | 68.57 (14.58 ms) |
| 128-token prompt, 32 generated, CPU | `2026-11-04-m3pro-dense-cpu-prefill128.json` | 281.4 | 65.99 (15.15 ms) |
| 10-token prompt, 160 generated, `--backend metal` | not recorded (tokens match the CPU record) | 194.9 | 33.44 (29.90 ms) |

Each decode step reads every weight once (about 1.13 GiB including the head), so 68.6
tok/s is about 77 GiB/s, close to the 80 GiB/s bf16 streams in the kernel benchmarks.
On Metal a dense model's matmuls are placed one by one (about 196 synchronisations per
token), which is why it is slower than the CPU; a dense whole-layer pass is future work.

## Saturated disks

Warm reads never saturate here, so `bench --disk-mbps N` (with `--disk-latency-us`) puts
reads behind a simulated disk with one shared bandwidth. Use `--ram-budget 1GiB` for the
tight regime (2–3 slots per layer). `--expert-cache-mb 1024` is a different regime: it
gives the experts the whole GiB (17–18 slots). Results: ADR-008 amendment 3.
