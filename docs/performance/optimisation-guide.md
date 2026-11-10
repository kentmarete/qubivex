# Optimisation guide

Two audiences: **operators** who want a deployment to go faster, and **contributors**
who want to make the code faster. Both parts rest on the same rule: measure first, and
never trade correctness for a number.

How to measure is in [BENCHMARKING.md](BENCHMARKING.md), and the recorded results are
there and in [IMPLEMENTATION_STATUS.md](../../IMPLEMENTATION_STATUS.md). The figures
below are from an Apple M3 Pro with OLMoE-1B-7B Q4_K_M unless stated. No other machine
has been measured, so treat them as shapes, not promises.

## Part 1: tuning a deployment

### Find the bottleneck first

| Symptom | Likely bound by | Check |
|---|---|---|
| Low tokens per second, expert cache hit rate near 100% | Compute | `qubivex_generation_tokens_per_second`; CPU use |
| Low tokens per second, many expert loads per token | Disk or memory | `qubivex_expert_cache_loads_total` against tokens; `qubivex_expert_load_wait_seconds_total` |
| Slow first token, fast after | Prefill, or the queue | `qubivex_queue_wait_seconds` against `qubivex_time_to_first_token_seconds` |
| Good single speed, poor under load | Concurrency settings | `qubivex_generation_slots_free`, mean batch width |

### Levers, in the order to try them

**1. Give the expert cache memory.** For a MoE model nothing else matters as much. With
every expert resident, decode is about 41 tokens per second; with a 1024 MiB cache out
of 3.6 GiB of experts it is about 35 on a warm file cache, and much less on a cold or
slow disk. Use `qubivex plan` to see the cache you will get, and raise the budget
(`--ram-budget`, a larger container limit) or lower what competes with it (`--context`,
`--max-concurrent`, `--prefix-cache-mb`).

**2. Use Metal on Apple Silicon for MoE models** whose experts mostly fit:
`--backend metal` gives 60 tokens per second against 41 for one conversation, and 102
against 62 for eight. Keep the CPU for dense models. With a small expert cache nothing
needs choosing: when any layer's pool holds fewer than half its experts, the engine runs
the CPU pass even under `--backend metal` (ADR-011 amendment 6).

**3. Turn on the prefix cache for chat.** `--prefix-cache-mb` removes most of the
prefill from every turn after the first: 0.57 s instead of 6.59 s for a 714-token
second turn. Clients must send a stable `user`. It costs KV memory, which comes out of
the expert cache, so size it deliberately ([KV cache](../runtime/kv-cache.md)).

**4. Set concurrency for your goal.** Total throughput rises with `--max-concurrent`
(1.5× at eight conversations on the CPU, 1.9× on Metal), and each conversation slows.
For interactive use keep it low; for batch work raise it as far as memory allows.

**5. Put the model on fast local storage.** With experts streaming, speed follows the
disk. `qubivex doctor --storage --path <models dir>` measures it. NVMe over SATA over
network storage.

**6. Leave `--threads` alone** unless other work shares the machine. The default is the
performance cores. Adding efficiency cores usually slows a kernel down, because the
team waits for its slowest thread.

**7. Choose the quantisation for the hardware.** A Q4_K_M file is about a quarter of
the bf16 size, so four times as many experts fit in the same cache. The fused kernels
cover Q4_K, Q6_K, Q8_0 and bf16.

### Things that do not help

- **Prefetch** (`--prefetch`, `run` and `bench` only). It helps only on a disk slow
  enough for reads to hold compute back, and it is off by default. With several
  conversations in a pass, reads already overlap with compute.
- **More I/O threads** beyond the default 4, unless `qubivex_expert_load_wait_seconds_total`
  shows requests waiting on reads and the disk has headroom.
- **Lowering precision.** There is no such switch. Every kernel is lossless.
- **A debug build.** Always measure `--release`.

### Limits to plan around

- Requests to different models never share a pass, and two busy models compete for the
  same cores.
- On x86-64 the kernels are the portable ones compiled for AVX2 and FMA. Expect less
  than the NEON figures per core until hand-written x86-64 kernels exist. This has not
  been measured.
- Inside a virtual machine or container on a CPU with performance and efficiency cores,
  the core tiers may not be visible, and the default thread count then includes the slow
  cores. Set `--threads` to the number of performance cores given to the VM.

## Part 2: making the code faster

### The rules

1. **Profile before optimising.** Find where the time goes with a profiler, not by
   reading code.
2. **Bits do not change.** A kernel or scheduling change must produce bit-identical
   output: across thread counts, batch sizes, cache sizes and backends. The invariant
   tests fail otherwise, and `qubivex bench --reference` checks the tokens.
3. **A change that alters numerics is not an optimisation.** It is a named, opt-in
   feature with its measured error, and needs an ADR
   ([ADR-010](../decisions/ADR-010-NUMERICAL-PRECISION.md)).
4. **Measured on power, quiet machine, before and after interleaved.** Report the
   median with the P95 beside it.
5. **Record what was tried and not kept.** A negative result saves the next person the
   same experiment. The ADRs hold several.

### The workflow

```sh
cargo build --release
./target/release/qubivex bench --model <ID> --label before --out before.json
# ... change ...
cargo build --release
./target/release/qubivex bench --model <ID> --label after --out after.json --reference before.json
cargo test --workspace
```

`--reference` makes the run fail if the tokens differ from the earlier record. For a
kernel, start with the Criterion benches:

```sh
cargo bench -p qubivex-backend-cpu -- batch_s
```

For concurrency, `crates/qubivex-runtime/examples/batch.rs` measures sequential,
caller-driven and concurrent decode at several widths and checks the tokens match in
every cell.

### Where the time goes

For one decode token on the CPU with every expert resident, the time is almost entirely
matmuls: per layer the four attention projections, the router, and three matrices for
each chosen expert, then the output head, which is the largest single matmul (vocabulary
× hidden). Attention over the KV, norms, routing and sampling are small beside them.

With a small expert cache, time moves to reads: each miss is one expert-sized read, a
few MiB.

For prefill, the same matmuls run over many rows at once, which is what the tiled
kernel is for.

### What has been done, and what it gained

| Change | Result | Record |
|---|---|---|
| Fused dequantise-and-dot NEON kernels for one row | The decode path | ADR-002 amendment 2 |
| Register-blocked tiles for prefill | Replaced an external GEMM | ADR-002 amendment 1 |
| Fused multi-row kernels for 2 to 12 rows | Q4_K expert at 2 rows: 59.5 µs against 129.9 µs through the portable path | ADR-017, job 8 |
| Per-layer expert pools with LFRU | Ended thrashing at small caches | ADR-008 amendment 1 |
| Reading a layer's misses together on I/O threads | Reads overlap with compute | ADR-006 amendment 1 |
| Whole layers on Metal; a token in one submission | 1.4× the CPU at decode | ADR-011 amendments 2 and 4 |
| Shared passes with a 1 ms bounded wait | 1.5× (CPU) and 1.9× (Metal) at eight conversations | ADR-017 |
| Prefix reuse | A 714-token turn in 0.57 s instead of 6.59 s | ADR-016 |

### What was measured and not kept

- **A lossy integer dot product.** Faster, and it changes tokens (ADR-010).
- **Predicting the union of a batch's experts for prefetch.** Could save at most the 0%
  to 5% of time shared passes wait on reads (ADR-017 §7).
- **The Metal pass with small expert caches.** Below half a layer's experts the engine
  uses the CPU pass instead (ADR-011 amendment 6).
- **The portable path and tiles for 2 to 12 rows.** Slower than the fused multi-row
  kernels; the cut-over to tiles is at 13 rows for the fused dtypes.

### Open opportunities

Not built, and each would need its own measurement:

- Hand-written AVX2 and AVX-512 kernels, and any measurement at all on x86-64.
- A whole-layer Metal pass for dense models.
- Paged KV, which would remove the copy on a prefix hit.
- A packed expert file with one contiguous extent per expert.
- `io_uring` for expert reads on Linux.
- Per-layer cache slots weighted by measured routing demand.
- A compute pool shared across loaded models.
- CUDA.

### Checks every performance change must pass

```sh
cargo test --workspace                 # includes the bit-identity invariants
cargo xtask check
RUSTFLAGS="--cfg loom" cargo test --release -p qubivex-expert-stream --test loom   # if the store changed
RUSTFLAGS="--cfg loom" cargo test --release -p qubivex-scheduler --test loom       # if the scheduler changed
```

and, with the real model available, the env-gated tests in
`crates/qubivex-runtime/tests/` ([Local development](../deployment/local-development.md#tests-that-need-real-models)).
