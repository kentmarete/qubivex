# ADR-002 spike: tensor engine on expert-FFN and attention-decode shapes

- Date: 2026-11-02
- Code: `spikes/adr-002-tensor-engine/` (standalone package, excluded from the workspace)
- Raw output: [`adr-002/raw-default.md`](adr-002/raw-default.md), [`adr-002/raw-accelerate.md`](adr-002/raw-accelerate.md);
  Colibri baseline: [`adr-002/raw-colibri-1t.md`](adr-002/raw-colibri-1t.md), [`adr-002/raw-colibri-11t.md`](adr-002/raw-colibri-11t.md),
  [`adr-002/raw-own-same-session.md`](adr-002/raw-own-same-session.md)
- Status: **complete.** The Colibri baseline ran on 2026-11-02 (section "Colibri baseline" below).
  Every lossless own kernel reaches ≥ 90% of Colibri's lossless equivalent.

## Setup

| Item | Value |
|---|---|
| Machine | Apple M3 Pro (5 P + 6 E cores), 18 GB, macOS |
| Toolchain | Rust 1.99.1, `-C target-cpu=native` (local benchmarking only, RUST_GUIDELINES §9), `lto = "fat"` |
| Candle | `candle-core` / `candle-nn` 0.11.1, default CPU build and a second build with `accelerate` |
| Threads | 11 (Candle's barrier pool and rayon both read `RAYON_NUM_THREADS`) |
| Memory bandwidth reference | 116 GB/s, all-thread vectorized read of 1 GiB ("STREAM" below) |
| Weights | Synthetic, roughly Gaussian (σ ≈ 0.02). Q8_0 and Q4_K blocks come from Candle's GGML quantizer, so both engines multiply identical bytes |
| Hot / cold | Hot: one matrix, cache-resident. Cold: rotating over ≥ 512 MiB of copies, so weights come from DRAM, as for streamed experts |
| Error | `max|y − ref| / max|ref|`, with `ref` in f64 on the same dequantized weights and f32 activations. This isolates kernel error from weight-quantization error |
| Determinism | Output bit hashes from fresh processes with 1, 2, 4 and 11 threads, each run twice |

### Engines
- **own:** the spike's kernels. The reduction has a fixed shape: 64 f32 lanes (16 FMA chains), chunks in `k` order, then a fixed halving tree. Each SIMD (NEON) kernel has a portable twin that a test requires to match it **bit for bit**, so Miri can run the scalar path later. Activations stay f32, so nothing is lossy. GEMV runs on Candle's persistent spinning `BarrierPool` (public API). The rayon variant appears in the raw data.
- **own deq + gemm crate (S=128):** dequantize the expert to f32, then one `gemm` 0.19 call. This is the same GEMM library Candle's CPU backend uses.
- **candle:** `x.matmul(w.t())` for f32/bf16 (bf16 also converts activations to bf16) and `QMatMul::forward` for Q8_0/Q4_K, the way `candle-transformers` layers call them.

## Results: expert matrices, S = 1 (decode)

Cold µs is the DRAM-bound number that matters for streamed experts. The gate/up shape (1024 × 2048) is shown here; the down shape (2048 × 1024) is within ±5%.

| dtype | engine | hot µs | cold µs | cold GB/s | % STREAM | rel err |
|---|---|---|---|---|---|---|
| f32 | own | 25.5 | 68.4 | 122.7 | 105% | 1.4e-7 |
| f32 | candle | 167.6 | 168.7 | 49.7 | 43% | 3.7e-7 |
| f32 | candle + Accelerate | 29.1 | 81.8 | 102.6 | 88% | 1.5e-7 |
| bf16 | own | 18.3 | 35.7 | 117.5 | 101% | 1.1e-7 |
| bf16 | candle | 317.6 | 319.0 | 13.1 | 11% | **3.2e-3** (bf16 activations) |
| int8 row-scaled | own | 27.3 | 30.3 | 69.4 | 60% | 1.6e-7 |
| int8 row-scaled | candle | – | – | – | – | no such type |
| q8_0 | own | 35.8 | 37.5 | 59.4 | 51% | 1.6e-7 |
| q8_0 | candle | 6.7 | 19.0 | 117.0 | 101% | **4.2e-3** (Q8_0 activations) |
| q4_k | own | 49.8 | 50.8 | 23.2 | 20% | 1.4e-7 |
| q4_k | candle | 8.2 | 11.2 | 105.6 | 91% | **5.3e-3** (Q8_K activations) |

## Results: expert matrices, S = 128 (prefill)

These runs are compute-bound, so hot and cold times are within 5% of each other. Down shape shown (gate/up is 5–15% slower).

| dtype | engine | µs | GFLOP/s | rel err | bits equal S=1 path |
|---|---|---|---|---|---|
| f32 | own (decode-order GEMM) | 2059 | 261 | 1.3e-7 | yes |
| f32 | own deq + gemm crate | 1270 | 423 | 1.3e-6 | no |
| f32 | candle | 1279 | 420 | 1.3e-6 | – |
| f32 | candle + Accelerate | 441 | 1217 | 1.3e-6 | – |
| bf16 | own deq + gemm crate | 1390 | 386 | 2.1e-6 | no |
| bf16 | candle / + Accelerate | 2636 / 698 | 204 / 770 | 3.8e-3 | – |
| q8_0 | own deq + gemm crate | 1367 | 393 | 1.0e-6 | no |
| q8_0 | candle | 573 | 937 | 4.4e-3 | – |
| q4_k | own deq + gemm crate | 1365 | 393 | 1.1e-6 | no |
| q4_k | candle | 845 | 635 | 4.2e-3 | – |

## Results: attention decode (16 heads × 128, f32 KV)

| context | own µs | candle µs | candle + Accelerate µs | own KV GB/s | rel err (own / candle) |
|---|---|---|---|---|---|
| 1024 | 77 | 365 | 221 | 217 (cache-resident) | 1.3e-6 / 9.3e-7 |
| 4096 | 579 | 1265 | 918 | 116 | 2.5e-6 / 2.6e-6 |

## Determinism
All 50 cases, own and Candle (default and Accelerate builds), produce **identical bits run to run and across 1, 2, 4 and 11 threads**. The own kernels split work across rows only, never across `k`.

## Findings

1. **Candle's fast quantized path is lossy.** `QMatMul` quantizes activations to Q8_0/Q8_K and uses integer dot products. The relative error is about 4e-3, against about 1e-7 for f32-activation kernels. That is Colibri's opt-in `IDOT` mode, which pending decision 9 ("no lossy option in any execution profile") rules out as a default. Candle has no lossless quantized CPU matmul. Its f32 dense path is 2.5× slower than own at S=1 (43% of STREAM), and its bf16 path is slow and lossy.
2. **Own lossless dense kernels are at the memory bound.** At S=1, f32 and bf16 GEMV stream at 100–105% of the STREAM proxy. The proxy slightly understates peak.
3. **Own lossless quantized kernels are compute-bound.** int8 reaches 60%, Q8_0 51% and Q4_K 20% of STREAM, even with explicit NEON. The int-to-f32 widening (u8 → u16 → u32 → f32) dominates. A lossless Q4_K matrix (51 µs) is **slower than the same matrix in bf16** (36 µs), even though it reads 3.6× fewer bytes. Its win today is storage, disk bytes and cache capacity, not decode compute. The Colibri baseline below shows these kernels are still well ahead of Colibri's lossless equivalents. *(Correction: an earlier draft said Colibri's speed claims assume a lossless default. That holds for `olmoe.c` and `inkling.c` only. Its GGUF path `gq_matmul` and `colibri.c` (GLM) default to the lossy integer dot; `GQ_EXACT=1` or `IDOT=0` opts out.)*
4. **Dispatch overhead matters at S=1.** The rayon global pool adds about 25–30 µs per GEMV, doubling hot f32 time. A persistent spinning pool on dedicated threads (Candle's `BarrierPool`, or our own) is needed for decode. That is consistent with QUBIVEX_DESIGN §4.5 (sync hot path, dedicated threads).
5. **Prefill wants a real GEMM library.** Our decode-order GEMM keeps S=128 bits equal to S=1 bits, but at 0.6× the `gemm` crate and 0.2× Accelerate. "Dequantize the expert to f32, then GEMM" costs nothing measurable over dense f32, so one f32 GEMM serves every dtype. Results stay deterministic across thread counts, but prefill bits differ from decode bits. Invariant 1 (cache capacity changes speed, never bits) still holds, because the kernel is chosen by S, never by residency. Colibri's prefill-equals-decode property (`moe_by_expert`) would cost about 1.6× prefill time on CPU if we kept it.
6. **Own attention decode is 2–5× faster than Candle's** matmul–softmax–matmul composition, and deterministic.

## Colibri baseline (2026-11-02)

Harness: [`spikes/adr-002-tensor-engine/colibri-baseline/colibri_bench.c`](../../spikes/adr-002-tensor-engine/colibri-baseline/colibri_bench.c).
It builds against a `JustVugg/colibri` checkout at `0a5504a1`. `adr-002-tensor-engine dump` writes the spike's exact
weight bytes, activations and f64 references, so both engines multiply identical data. The method matches the spike
(median, hot and cold over ≥ 512 MiB, same error metric). Colibri is compiled with `clang -O3 -mcpu=native` and
Homebrew libomp, as its macOS Makefile does when libomp is present. Both engines ran back to back in one session.
STREAM was 114 GB/s (11 threads) and 81 GB/s (1 thread) that session.

Colibri arms, each one its shipped function. Engine-static ones are copied verbatim:
- f32: `matmul_f32.h` `matmul` (lanes). bf16: `matmul_f16.h` `matmul_bf16w`.
- int8 row-scaled: `olmoe.c` `matmul_q` / `matmul_q_rows` (IDOT off, OLMoE's default) and `quant.h` `matmul_q` (NEON, f32 activations).
- Q8_0 / Q4_K: `gq_matmul_f32` (exact, the `GQ_EXACT=1` path) and `gq_matmul` (integer dot with Q8 activations, the default).
- Attention: `bench_attn_f32.c`'s decode loop, with the shipped scalar dot and with the `matmul_f32.h` lanes dot.

### Decode S = 1, cold µs (gate/up · down)

The ratio is Colibri time over own time. It is the share of Colibri's throughput that own reaches, so ≥ 90% passes.

| dtype | own | best Colibri **lossless** | own vs lossless | Colibri **lossy** default (rel err ≈ 4e-3) |
|---|---|---|---|---|
| **11 threads** | | | | |
| f32 | 76.0 · 75.3 | 93.6 · 95.7 | **123% · 127%** | – |
| bf16 | 40.1 · 39.9 | 63.2 · 63.2 | **158% · 158%** | – |
| int8 row | 25.8 · 26.2 | 80.1 · 74.6 (`quant.h`) | **310% · 285%** | – |
| q8_0 | 31.9 · 32.4 | 422.5 · 425.5 (`gq_matmul_f32`) | **13×** | 59.8 · 57.8 (own is 1.8× faster) |
| q4_k | 42.8 · 43.0 | 432.0 · 435.1 (`gq_matmul_f32`) | **10×** | 92.9 · 85.2 (own is 2.0–2.2× faster) |
| **1 thread** (per-core kernel quality) | | | | |
| f32 | 117.6 · 119.3 | 119.7 · 120.8 | **102% · 101%** | – |
| bf16 | 80.4 · 82.1 | 79.7 · 81.2 | **99% · 99%** | – |
| int8 row | 124.3 · 127.0 | 223.8 (`olmoe`) · 197.8 (`quant.h`) | **180% · 156%** | – |
| q8_0 | 164.2 · 167.3 | 1957 · 2099 | **12×** | 90.3 · 88.5 (Colibri is 1.8× faster) |
| q4_k | 228.8 · 232.1 | 1939 · 2102 | **8.5–9×** | 230.6 · 204.9 (own at 100% · 88%) |

### Prefill S = 128, µs, 11 threads (1 thread in brackets)

| | own deq + `gemm` crate | best Colibri lossless | own vs lossless | Colibri lossy q8_0 · q4_k |
|---|---|---|---|---|
| gate/up | 1484–1589 (5282–5496) | 2365 bf16 (12048) | **150–160% (2.2–2.3×)** | 2595 · 3590 |
| down | 1299–1387 (5198–5395) | 2391 bf16 (12369) | **170–185% (2.3–2.4×)** | 2361 · 3136 |

### Attention decode, µs (interleaved rounds, medians of 3)

| context | own, rayon | own, spinning pool | Colibri lanes | Colibri scalar (shipped) | own pool vs lanes |
|---|---|---|---|---|---|
| 11 threads, 1024 | 120.5 | 101.5 | 106.3 | 224.1 | **105%** (rayon: 88%) |
| 11 threads, 4096 | 576.5 | 520.7 | 556.3 | 677.5 | **107%** (rayon: 97%) |
| 1 thread, 1024 | 289.0 | 277.8 | 323.2 | – | **116%** |
| 1 thread, 4096 | 1138.9 | 1123.5 | 1286.3 | – | **114%** |

With rayon, own attention misses the bar at 1k context and 11 threads (88%). The ~20 µs is rayon dispatch,
the same cost as finding 4. On the spinning pool (`attn::own_pool`, bitwise equal to the rayon version by test) it passes.
In the earlier single-thread build without an OpenMP runtime (`-fopenmp-simd`), Colibri's lanes attention measured
265 / 1098 µs, so own was at 91% / 96%. The libomp build is the one Colibri ships, and that is the one used above.

### Determinism and error
Colibri's outputs are also bitwise identical run to run and across 1, 2, 4 and 11 threads for every arm above.
Its lossless arms land at 1.4e-7 to 2e-6 relative error. Its integer-dot defaults land at 3.9e-3 to 4.4e-3, the same
mechanism and size as Candle's `QMatMul`.

### Reading
- **The ≥ 90% rule is met for every lossless arm, at both thread counts.** The tightest case is per-core f32 and bf16
  GEMV, at parity (99–102%). Both are simple lane loops that the compiler vectorizes equally well.
- Colibri's lossless quantized path (`gq_matmul_f32`) dequantizes a row and then runs a scalar `fp-contract=off` dot.
  It is a reference path, roughly 10× slower. Its fast quantized paths are the lossy integer dots.
- Colibri's 11-thread scaling on the M3 Pro is weak (int8: 2.6–2.9× from 1 to 11 threads, against 4.8× for own). OpenMP
  `schedule(static)` gives the six E-cores the same row share as the P-cores. Own chunks dynamically. On homogeneous
  x86 cores this part of the gap should shrink, so the per-core rows are the fairer kernel comparison.
- The only place Colibri wins is its **lossy** Q8_0 integer dot, per core (1.8×). Decision 9 rules out a lossy
  default, and at 11 threads own lossless Q8_0 is faster anyway.

## Recommendation

**Option 3, hybrid, without a Candle dependency:**
- Hot decode path: own kernels in `qubivex-backend-cpu`, using the fixed reduction shape, NEON and AVX2 behind runtime dispatch, and portable twins. Attention decode is included.
- Prefill: own dequantize, then the **`gemm` crate**. That is one small dependency instead of Candle, and it is what Candle uses internally. On macOS, an optional Accelerate `sgemm` is about 3× faster, as an additive feature.
- Candle is not needed. Its unique advantage is the lossy integer-dot path. If a named, opt-in lossy profile is ever accepted, we can port that path ourselves (as Colibri's `IDOT` did) behind `ComputeBackend`.
- Keep the capacity-sweep test comparing prefill against prefill and decode against decode.

- Attention decode, like GEMV, runs on the persistent spinning pool, not on rayon.

Follow-ups (not blocking the decision):
1. An AVX2 measurement on Linux x86-64, the first production target, again against Colibri. The portable twins exist, but on x86 they need `#[target_feature(enable = "avx2,fma")]` dispatch, or `mul_add` becomes a libm call.
2. Optional lossless Q4_K speed-up. It is already 10× Colibri's lossless path, but it is only at 24% of STREAM. Candidates are an algebraic min/scale factorization (`d·Σqx − m·Σx`, with `Σx` per sub-block shared across rows) and a table-lookup nibble decode. Each needs its own bit-exactness check.

## Caveats
- One machine, synthetic weights, `target-cpu=native`, and no thermal control. Medians over ≥ 15 iterations or 0.4–0.8 s, whichever is longer.
- P and E cores are mixed. Work is chunked dynamically, so E cores take smaller shares.
- Cold mode rotates copies of one matrix. Real streamed experts also pay for the disk read and the page cache, which belong to Phase 6.
