# ADR-002: Hybrid compute; the tensor engine is chosen by a benchmark spike

- Status: Proposed. **Spike run (2026-11-02); the decision is pending the Colibri baseline.**
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

## Spike results (2026-11-02, partial)
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

**Preliminary choice: option 3 (hybrid) without Candle.** Own kernels on the decode hot
path, and own dequantize plus the `gemm` crate for prefill, with Accelerate as an
additive macOS feature. The ≥ 90%-of-Colibri rule is still unchecked.
