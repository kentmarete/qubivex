# ADR-002: Hybrid compute; the tensor engine is chosen by a benchmark spike

- Status: **Accepted (2026-11-02): option 3, hybrid, without Candle.** The decision rule was met against the Colibri `0a5504a1` baseline. Amendment 1 (2026-11-04): prefill uses an own register-blocked kernel instead of the `gemm` crate; the spinning pool is deferred.
- Date: 2026-11-01

## Context
Colibri's speed comes from hand-written quantized GEMV kernels (int8 row-scaled, ggml
blocks, AVX2/SSE4.1/NEON) with no BLAS, and from controlling exactly when weights are
touched. A general tensor framework may help with breadth (many ops and dtypes) but
can hide memory placement and add per-op overhead at S=1 decode.

## Options
1. **Candle as the primary CPU path.** Mature quantized types (q4_k, q8_0), CUDA and
   Metal backends. Risk: its own allocator and layout choices, and less control over
   the slab-based expert residency.
2. **Own kernels** in `qubivex-backend-cpu` (Rust `std::arch`) behind
   `ComputeBackend`. Full control. Higher effort.
3. **Hybrid:** own kernels for the hot path (expert SwiGLU, quantized GEMV, attention
   decode), and Candle or another library for cold or rare ops and reference checks.
4. **FFI to Colibri's C kernels** as a stop-gap for specific ops.

## Spike (3 days, Phase 1)
Shapes: OLMoE expert (2048→1024 gate/up, 1024→2048 down) at S=1 and S=128, attention
decode with 16 heads × 128 at 1k and 4k context.
Dtypes: f32, bf16, int8 row-scaled, Q4_K, Q8_0.
Metrics: ns/op, effective GB/s against STREAM bandwidth, max relative error against
an f64 reference, and bitwise determinism across runs and thread counts.

## Decision rule
Pick the option that reaches **≥ 90% of Colibri's kernel throughput** on the same
shapes with deterministic results. If several meet that, prefer the one with fewer
dependencies. Whatever is chosen stays behind `ComputeBackend`, so it can be replaced.

## Consequences
The decision is recorded with the spike data in `docs/performance/`.

## Spike results (2026-11-02)
Full data and caveats: [`docs/performance/ADR-002-SPIKE.md`](../performance/ADR-002-SPIKE.md).

- Candle's fast quantized CPU path is lossy. It quantizes activations to Q8 (relative
  error about 4e-3, against about 1e-7 lossless), the equivalent of Colibri's opt-in
  `IDOT`. Candle has no lossless quantized CPU matmul. Its dense f32 S=1 path runs at
  43% of memory bandwidth, and its bf16 path is slow and lossy.
- Own lossless kernels: f32 and bf16 GEMV reach memory bandwidth. int8, Q8_0 and Q4_K
  reach 60%, 51% and 20% (compute-bound on the integer widening, with NEON).
  Attention decode is 2–5× faster than Candle's.
- Prefill: dequantize to f32, then the `gemm` crate, matches Candle's f32 GEMM.
  Accelerate is about 3× faster on macOS.
- Every own and Candle case is bitwise deterministic across 1–11 threads.

- **Colibri baseline** (same bytes, same machine, same method): every lossless own kernel
  reaches ≥ 90% of Colibri's lossless equivalent at 1 and 11 threads. Per core, f32 and
  bf16 GEMV are at parity (99–102%) and int8 is 1.6–1.8×. Q8_0 and Q4_K are 8.5–13×
  Colibri's exact `gq_matmul_f32`, and prefill is 1.5–2.4×. Attention decode is 105–116%,
  but only on the spinning pool: on rayon it is 88% at 1k context with 11 threads.
  Colibri's faster quantized paths are lossy by default (integer dot, about 4e-3), like
  Candle's. Colibri is also bitwise deterministic across thread counts.

## Implementation status (2026-11-04)
Built: own lossless GEMV kernels with fused NEON decode (PR #2), and the prefill path of
amendment 1 (register-blocked kernel, experts computed concurrently, parallel attention).
**Not built:** the persistent spinning pool, deferred by amendment 1 with the
measurements that justify it. The engine keeps its dedicated rayon pool.

## Decision
**Option 3 (hybrid) without Candle.** Own kernels on the decode hot path (GEMV and
attention, on a persistent spinning pool), and own dequantize plus the `gemm` crate for
prefill, with Accelerate as an additive macOS feature. Candle is not a dependency.
Follow-ups that don't block the decision: an AVX2 measurement on Linux x86-64 against
Colibri, and optional lossless Q4_K speed-up work.

## Amendment 1 (2026-11-04): an own register-blocked prefill kernel; the spinning pool deferred

### Prefill does not use the `gemm` crate
Invariant 1 (`qubivex-runtime/tests/invariants.rs`, `prefill_equals_token_by_token`) and
ADR-010 require the CPU's bits not to depend on the batch size: a prompt's logits must
equal the logits of feeding it one token at a time. The `gemm` crate accumulates in its
own blocked order, so its outputs differ from the decode dot product in the last bits
(the spike measured 1e-6 relative, "bitwise vs S=1: NO" in every row). Adopting it would
break the invariant, so it is not used.

Instead, `qubivex-backend-cpu/src/tile.rs` keeps the decode reduction exactly. The shared
dot product (`qubivex_compute::reduce`) gives each output 64 independent lanes, each
accumulated by fused multiply-add in `k` order, then one fixed halving tree. Lanes never
interact before the tree, so the kernel computes one 4-lane quad at a time for a tile of
4 weight rows × 6 activation rows: 24 accumulators held in registers, 10 vector loads per
24 FMAs (the aarch64 inner loop compiles to `ldp`/`fmla.4s` with no spills). Each lane
sees exactly the operations `dot` performs, so the bits equal the row path's for every
batch size, thread count and ISA (tests: `batching_threads_and_isa_never_change_bits` at
s = 2, 5–11, 13 against the portable row path, on every dtype including ragged `k`).

Operands are packed lane-major so the kernel reads them contiguously. Measured
alternatives:
- reading the row-major rows in place: 2× slower, because rows 8 KiB apart map to the
  same L1 sets;
- a quad copied with a run-time length: compiled to `memmove` calls (5.8k of 22k profile
  samples). Fixed-size copies took prefill from 53 to 86 tok/s.

Below 6 activation rows, dequantising and packing each weight row costs more than the
tile saves (Q4_K expert shape: 0.98× at 5 rows, 1.11× at 6, 1.6× at 16), so `matmul`
uses the tiled kernel from 6 rows and the row path below. Decode (1 row) is unchanged.

Kernel (Criterion, 5 threads, `prefill_s128_t5` gate/up): f32 3.49 → 1.65 ms, bf16
3.51 → 1.64, Q8_0 3.56 → 1.72, Q4_K 3.44 → 1.66, Q6_K 3.59 → 1.81. That is about
85 GFLOP/s on one P-core, against a peak near 130.

### Concurrency outside the kernel
Profiling the 128-token prefill after the kernel change showed the compute team idle
about half the time. The causes were 64 small expert products per layer, each with its
own fork-join, and attention running on one thread. Two changes fixed this, neither
changing any bits:
- **Experts run concurrently.** `ExpertProvider::with_experts` now takes an `Fn + Sync`
  callback that may be called from several threads. `StoreProvider` runs a rayon scope:
  each expert that is ready is spawned. When one finishes, its lease is released and
  whatever has become ready is spawned at once, so loads into freed slots start
  immediately. The store gained `Batch::try_next`, a non-blocking `next` that answers
  `Next::Later` where `next` would wait or fail with `Full`. The blocking `next` is
  called only while no lease is held, so a layer with fewer slots than routed experts
  still pipelines, and one compute thread cannot deadlock (test:
  `compute_threads_never_change_bits`, 1 and 3 threads, one-slot pools, with and without
  I/O threads). Each expert writes only its own rank slots, combined in rank order
  afterwards.
- **Attention runs on the pool,** one task per (token, head) pair. The KV cache already
  holds the whole batch, so the pairs are independent; each is the same sequential
  `attend`.

### Results (M3 Pro, OLMoE-1B-7B Q4_K_M, 5 threads; tokens identical to the references)
| | before | after |
|---|---|---|
| prefill, 128-token prompt, all resident | 44.7 tok/s | 113–117 tok/s |
| prefill, 128-token prompt, `--ram-budget 1GiB` | 48–49 tok/s | 98–100 tok/s |
| prefill, 10-token prompt, all resident | 34.7 tok/s | 39–41 tok/s |
| prefill, 10-token prompt, 1 GiB | 31.0 tok/s | 31–32.5 tok/s |
| decode, 160 tokens, all resident (median step) | 24.9–25.8 tok/s | 29.6–30.8 tok/s |
| decode, 1 GiB | 18.9 tok/s | 19.1 tok/s |

The CPU prefill of a 128-token prompt is now faster than whole layers on Metal (77 tok/s,
ADR-011 amendment 2). Records: `docs/performance/baselines/2026-11-04-m3pro-tiled-*.json`.

### The spinning pool is deferred
The spike preferred a spinning pool because rayon's wake-up cost about 25–30 µs per
isolated GEMV and attention reached only 88% of Colibri at 1k context. In the engine, a
sample of the compute threads after the changes above shows the workers asleep 3–5% of
the time and spinning 1–5%, in prefill and decode alike (it was 18% asleep before; that
idle time was single-threaded code, not wake-up latency). A spinning pool can recover at
most that. It would also have to provide the nested work stealing that concurrent
experts rely on, which means replacing rayon in the CPU backend, the forward pass and the
provider. That is too large a change for at most 5–8%. The pool is deferred until a
profile shows wake-up latency costing more than that, for example at higher thread
counts or for a dense model with fewer, larger products.
