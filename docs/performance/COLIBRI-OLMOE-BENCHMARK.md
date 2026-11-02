# QUBIVEX against Colibri `olmoe`: OLMoE-1B-7B Q4_K_M decode

- Date: 2026-11-03
- Protocol: MIGRATION_PLAN §4; reported in the shape of Colibri's `docs/benchmarking.md`
  ("Minimum viable report")
- **Result:** lossless QUBIVEX decodes at **20.7–21.0 tok/s**, faster than Colibri's
  default lossy kernels (17.4–17.8 tok/s) and about 7× faster than Colibri's lossless
  path (3.0 tok/s). Its 160 generated tokens are **identical** to Colibri's lossless
  output.

## Results

Three interleaved rounds (Colibri default, Colibri lossless, QUBIVEX), same prompt,
160 generated tokens, greedy decoding.

| Engine | Numerics | Decode (3 runs) | Peak RSS | Expert hits |
|---|---|---|---|---|
| Colibri `olmoe` (default) | lossy: activations quantized for an integer dot (`gq_matmul`) | 17.36 / 17.52 / 17.80 tok/s | 3.38–3.55 GB | 95.5% |
| Colibri `olmoe`, `GQ_EXACT=1` | lossless | 3.04 / 3.06 / 3.04 tok/s | 3.22–3.44 GB | 95.5% |
| **QUBIVEX** `qubivex run` | lossless | **20.97 / 20.73 / 20.67 tok/s** | 3.91–3.94 GB | (see below) |

**Independent control: the generated text.**
- QUBIVEX's 160 tokens are identical to Colibri's lossless (`GQ_EXACT=1`) tokens. These are two independent implementations computing the same exact arithmetic.
- Colibri's default path agrees for the first 86 tokens, then drifts; that is the expected effect of its activation quantization.
- All three QUBIVEX runs produced byte-identical output. The Colibri speeds were stable across rounds, so no run was disturbed by thermal drift or another process.

## Operating point

| Item | Value |
|---|---|
| QUBIVEX | `main` at `8c5e20f` (fused NEON kernels, in-pool forward pass), `cargo build --release` (`lto = "fat"`, no `target-cpu=native`), Rust 1.99.1 |
| Colibri | `JustVugg/colibri` at `0a5504a1` (v2.1.0), `make olmoe` with Homebrew libomp, Apple clang 16.0.0 |
| Model | `allenai/OLMoE-1B-7B-0125-Instruct-GGUF`, `OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf`, 4,213,512,192 bytes, sha256 `4ddc0e53159ed512b8dd67914a66e27bc618f694672ba43a9a0454eabd9c684f`. Experts Q4_K gate/up and Q6_K down, head Q6_K, norms f32 |
| Model loading | Both engines read the same GGUF in place. Colibri: `coli import <gguf> --out <dir> --mode link` writes a directory that points at the file. QUBIVEX: `qubivex models import … --set norm_topk_prob=false --write` |
| Prompt | `Write a long story about a lighthouse keeper.`, 10 tokens: `[10639, 247, 1048, 2926, 670, 247, 1708, 5967, 43775, 15]`, no chat template, no BOS |
| Generation | 160 tokens, greedy (temperature 0), no end-of-sequence stop reached |
| Threads | 5 for both: the M3 Pro's performance cores. Colibri: `[OMP] olmoe: 5 physical-core threads`. QUBIVEX: the default compute team |
| Expert cache | Every expert may stay resident: Colibri `cache=64/layer` (all 64), QUBIVEX's default all-resident capacity. Both start each process empty and load on first use, so the first tokens read experts from the file (Colibri: 975 misses, 20,657 hits) |
| I/O and cache state | Buffered reads. The page cache was warm: the model file had been read many times before these runs, and the rounds ran back to back |
| Machine | Apple M3 Pro (5 performance + 6 efficiency cores), 18 GB, macOS 27.2, internal SSD |

### Commands

```sh
# Colibri (SNAP is the imported directory; bench_ref.json has the 10 prompt ids and 160 placeholders)
SNAP=<dir> ./olmoe 64 8 bench_ref.json               # default
GQ_EXACT=1 SNAP=<dir> ./olmoe 64 8 bench_ref.json    # lossless

# QUBIVEX
qubivex run --model olmoe-1b-7b-0125-instruct --prompt "Write a long story about a lighthouse keeper." --max-tokens 160
```

## Caveats

- **The speed metrics are not defined identically.** Colibri's `Speed` is 160 tokens over its decode time (`TUNE decode: 160 tokens in …`). QUBIVEX's `decode` excludes the first token, which is produced by the prefill. For 160 tokens the difference is under 1%.
- This is one machine, warm, with every expert resident. Streaming with a smaller expert cache, cold caches and x86-64 still need measuring. QUBIVEX's AVX2 kernels don't exist yet, so x86-64 runs its portable path.
- QUBIVEX's peak RSS is about 0.4 GB higher, mostly because experts are held in per-expert allocations. Preallocated slabs come in Phase 6.
- Colibri can trade exactness for more speed elsewhere (for example its int8 checkpoint path with `IDOT=1`). This compares the GGUF file both engines read, at each engine's default and lossless settings.
