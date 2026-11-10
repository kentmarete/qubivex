# Memory profiling

How to find out where a QUBIVEX process's memory is, and whether it matches what was
planned. Three sources answer this, from the inside out: the **plan** (what was
allowed), the **ledger** (what QUBIVEX allocated) and the **resident size** (what the
operating system counts).

How the budget is decided is in [Memory planner](../runtime/memory-planner.md).

## Before running: the plan

```sh
qubivex models analyse <ID>                    # does it fit, and what does it need?
qubivex plan <ID> --ctx 8192 --sequences 4     # the placement for a given load
qubivex plan <ID> --ram-budget 8GiB --json     # the same, as data
```

`plan` prints every component with its size and a tag: measured, estimated or
requested. The lines to read first:

| Line | Tells you |
|---|---|
| `dense weights` | What is resident whatever happens |
| `shared room` | What the KV and the expert cache divide between them |
| `KV at full context` | The cost of every planned conversation at full length |
| `expert cache, empty KV` and `full KV` | The cache's range. If `full KV` is at the floor, long conversations will stream nearly every expert |
| `KV per token` | The slope: multiply by tokens × conversations |
| `Expert slots per layer` | How many of each layer's experts stay in memory |

## While running: the ledger

The ledger counts what the engine has allocated, by owner.

```sh
curl -s http://127.0.0.1:8080/v1/system -H "Authorization: Bearer $KEY"
```

```json
"ledger_bytes": {"dense": 134272, "expert_cache": 196608, "kv": 524288},
"plan_budget_bytes": 2827358622,
"kv_reserved_positions": 0,
"expert_cache_bytes": 196608
```

The same, over time, from `/metrics`:

| Series | Meaning |
|---|---|
| `qubivex_memory_dense_bytes` | Resident weights |
| `qubivex_memory_expert_cache_bytes` | Experts cached now |
| `qubivex_memory_kv_bytes` | KV allocated now (live and retained) |
| `qubivex_memory_activations_bytes` | Activations and scratch |
| `qubivex_memory_budget_bytes` | The plan's budget |
| `qubivex_expert_cache_bytes` | The expert cache's capacity now. It falls as KV grows |
| `qubivex_kv_reserved_positions` | KV positions reserved by running conversations |
| `qubivex_prefix_cache_positions` | KV positions retained for reuse |

`qubivex run` prints the budget, the ledger by owner and the pressure level at the end
of a run, on standard error:

```text
memory: performance budget 3.10 GiB; ledger 483.12 KiB (dense 131.12 KiB, experts 96.00 KiB, KV 256.00 KiB); pressure normal
```

**What to check:** the sum of the ledger's owners stays under the budget; the expert
cache capacity moves opposite to KV; after the conversations end, KV returns to the
retained amount and the cache regrows.

## From outside: resident size

| Source | Gives |
|---|---|
| `qubivex_process_resident_bytes` | The process's resident set, read at each scrape |
| `qubivex_process_peak_resident_bytes` | Its peak. On Linux, the kernel's high-water mark. On macOS, the highest value a scrape has read |
| `qubivex bench --json` | `memory`: resident size after load, and the peak of the run |
| `qubivex system` | "RSS (this process)", total and available RAM, and the cgroup limit |
| `ps -o rss= -p <PID>`, `/proc/<PID>/status`, `docker stats` | The OS's view |

### Why the ledger and the resident size differ

They measure different things, and a difference of a few hundred MiB is normal. On
OLMoE-1B-7B with every expert resident, the ledger read 3.90 GiB while resident size
grew 3.75 GiB.

| Cause | Direction |
|---|---|
| KV is reserved in 256-position chunks; pages never written are not resident | Ledger above resident |
| On macOS, compressed pages are not counted as resident | Ledger above resident |
| The tokenizer, the binary, thread stacks and allocator overhead are not in the ledger | Resident above ledger |
| Short-lived activation scratch inside a pass is covered by a fixed 512 MiB reserve, not charged per allocation | Resident above ledger, briefly |
| The OS page cache holding model file pages | In neither. It is the kernel's, though a cgroup limit counts it |

The test `crates/qubivex-runtime/tests/ledger_rss.rs` compares the two on the real model
and allows 256 MiB either way.

## The page cache

Experts are read with ordinary file reads, so the OS keeps the pages it has read in its
page cache while memory allows. This is separate from QUBIVEX's expert cache:

- It makes a second run fast even with a small expert cache.
- It is not in the budget, the ledger or the resident size.
- The OS drops it under pressure, at no cost but later reads.
- `--direct-io` (on `run` and `bench`) bypasses it, so measurements show the disk and
  the file cache does not fill.

To measure true cold reads, start from a dropped page cache (`sudo purge` on macOS,
`echo 3 | sudo tee /proc/sys/vm/drop_caches` on Linux) or use `--direct-io`. Results
after either are not comparable with warm results.

## Memory pressure

The pressure monitor's level is visible through its effects: `qubivex_expert_cache_bytes`
falling without KV growing means the monitor is shrinking the cache, and 503
`overloaded` responses with a "memory:" message mean it has reached its critical level.
The levels and thresholds are in
[Memory planner](../runtime/memory-planner.md#memory-pressure).

The opt-in test `pressure_hog` (real model) runs a generation while another process
takes memory, and checks that the tokens do not change and that the cache returns when
the pressure does.

## Recipes

**How much memory will `--max-concurrent 8` at 4,096 tokens take?**
`qubivex plan <ID> --ctx 4096 --sequences 8`.

**Is the server using what it planned?** Add `qubivex_memory_dense_bytes`,
`_expert_cache_bytes`, `_kv_bytes` and `_activations_bytes`, and compare the sum with
`qubivex_memory_budget_bytes`. Then compare `qubivex_process_resident_bytes` with both.

**Is memory leaking?** Run a fixed load, let every conversation end, and compare
`qubivex_memory_kv_bytes` and `qubivex_process_resident_bytes` with their values before.
KV should return to the retained amount (0 with the prefix cache off). The test suites
check this after every failure case: the ledger returns to its baseline.

**What does one more conversation cost?** `KV per token` × its context, rounded up to
256-position chunks, taken from the expert cache.

**Why did the expert cache shrink?** KV grew (check `qubivex_kv_reserved_positions`),
retained KV grew (`qubivex_prefix_cache_positions`), or pressure rose (available RAM on
the host).

## Allocation profiling

For where allocations come from, use the platform's tools on a release build with
debug symbols (`CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release`):

- macOS: Instruments (Allocations), or `heap <PID>` and `vmmap --summary <PID>`.
- Linux: `heaptrack`, or `perf` with `--call-graph dwarf`.

The engine is built to make this uninteresting in steady state: once the expert pools
are full, a streaming decode reuses slot buffers and allocates no expert memory, and
KV grows only in whole chunks.

## Not supported

- **Peak GPU memory.** On Apple Silicon the GPU shares host RAM, which the ledger and
  the resident size already count. There is no discrete-GPU backend.
- **A fragmentation measure.**
- **A true kernel peak on macOS.** It needs an FFI call the code does not make; the
  peak there is sampled.
- **Per-request memory accounting** in the API. KV is reserved per request internally,
  and reported per model.
- **A built-in heap profiler or a memory dump endpoint.**
