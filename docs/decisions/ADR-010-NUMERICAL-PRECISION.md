# ADR-010: Lossless by default; lossy only as named opt-ins, never in a profile

- Status: **Accepted (2026-11-03)**, by owner decision on the Engineering Improvement
  Plan reconciliation (`docs/engineering/ARCHITECTURE_DECISIONS.md` §2.1)
- Date: 2026-11-03

## Context
Every QUBIVEX kernel is lossless: f32 activations times the exact dequantised weights.
Colibri's and Candle's fast quantised paths quantise activations (relative error about
4e-3). On OLMoE, Colibri's lossy default diverges from the lossless tokens after 86 of
160 tokens. The EIP proposed a `PrecisionPolicy` with a `Performance` level. The design
(invariant 4) and pending review decision 9 say no profile may be lossy.

## Decision
- **The default is lossless.** It is not a mode a user can forget to set. Weight
  quantisation is a property of the model file: running a Q4_K model means using its
  dequantised weights exactly.
- **Lossy behaviour exists only as named, per-feature opt-ins**, for example
  `activation_q8_dot`, `bf16_accumulate` or `fp16_kv_cache`. Each one:
  - is listed in the CLI help, logged at start-up, and recorded in benchmark reports;
  - documents its measured error against the lossless path and its greedy-token
    divergence point on the reference prompt;
  - has a reference test.
- **No profile enables a lossy opt-in.** Profiles (performance, balanced, saver) change
  threads, caches and prefetching, never numerics.
- The CPU backend stays bitwise deterministic across thread counts, batch sizes, SIMD
  paths and cache capacities. A GPU backend states its tolerance against the CPU
  (ADR-007).

## Consequences
- The EIP's `Strict` is the default here. Its `Explicit(DataType)` is an individual
  opt-in. Its `Performance` level is not adopted.
- No lossy opt-in exists yet. The first one must ship with its error and divergence
  measurements.
