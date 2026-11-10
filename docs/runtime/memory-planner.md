# Memory planner

The memory planner decides, for one model on one machine, how much memory each part may
use: the dense weights, the expert cache, the KV cache and working space. One planner
serves `qubivex plan`, `qubivex models analyse`, model loading and the running engine,
so what the tools report is what the engine does. The design record is
[ADR-005](../decisions/ADR-005-MEMORY-PLANNER.md).

## The model of memory

```text
 available RAM (or the cgroup's room)
 └─ × profile fraction ──► budget
      ├─ dense weights         always resident                          measured
      ├─ activation reserve    512 MiB of working space                 fixed
      └─ shared room           everything left
           ├─ KV cache         grows as conversations grow
           └─ expert cache     takes what the KV is not using
                               floor: one expert slot per layer
```

The KV cache and the expert cache **share one room**. The expert cache starts with all
of it and gives bytes back as KV grows. So an idle server has the largest expert cache,
and a server full of long conversations has the smallest.

Two rules hold at all times:

- The expert cache never goes below one slot per layer, so a forward pass can always
  run.
- A conversation is admitted only if its full KV leaves that floor. This is what turns
  "out of memory" into a queue or a 503 instead of a crash.

## Seeing a plan

```sh
qubivex plan olmoe-1b-7b-0125-instruct --ram-budget 2GiB --sequences 4
```

```text
Memory plan for olmoe-1b-7b-0125-instruct: performance profile, context 1181 × 4 sequences

RAM
  budget                       2.00 GiB  (requested)
  dense weights              296.62 MiB  (measured)
  activations                512.00 MiB  (estimated)
  shared room                  1.21 GiB  (estimated)
    KV at full context         1.15 GiB  (measured)
    expert cache, empty KV     1.21 GiB  (estimated)
    expert cache, full KV     58.38 MiB  (estimated)
    cache floor               58.12 MiB  (measured)
  all routed experts           3.63 GiB  (measured)
  KV per token                 1.00 MiB  (measured)
VRAM
  none

Placement
  embedding         55.27 MiB  RAM        dense: used by every token
  output head       80.60 MiB  RAM        dense: used by every token
  norms            264.00 KiB  RAM        dense: used by every token
  attention        152.50 MiB  RAM        dense: used by every token
  routers            8.00 MiB  RAM        dense: used by every token
  routed experts     1.21 GiB  RAM cache  loaded on first use; the KV takes bytes back as it grows
  routed experts     3.58 GiB  disk       beyond the cache at full context: read from the model file on a miss

Expert slots per layer
  empty KV  5 layers × 22, 11 layers × 21 (of 64 experts per layer)
  full KV   16 layers × 1 (of 64 experts per layer)

Why
  - RAM budget 2.00 GiB, as requested.
  - Room = budget − dense 296.62 MiB − activation reserve 512.00 MiB = 1.21 GiB, shared by the KV cache and the expert cache.
  - Context shortened from the model's 4096 to 1181: the full-context KV must leave one expert slot per layer.
  - KV: 1.00 MiB per token × 1181 context × 4 sequences (included per token) = 1.15 GiB at full context. It is allocated as it grows, never up front.
  - Expert cache 1.21 GiB with an empty KV, shrinking to 58.38 MiB at full context (at least 1 of 64 experts per layer); the rest stream from disk. Each layer's pool holds its slots and evicts only its own experts.
```

Every number carries a tag: **measured** (from the OS or the file headers),
**estimated** (a formula or a policy constant) or **requested** (you gave it). The
"Why" section states each decision in words. `--json` gives the same as data.

Reading this plan: with 2 GiB and four conversations, each conversation can reach 1,181
tokens, and at that point the cache is down to one expert per layer and nearly every
expert is read from disk. Fewer conversations, a shorter `--ctx` or more RAM each buy
cache back.

## Choosing the budget

| Setting | Budget |
|---|---|
| `--profile performance` (default) | 0.88 of the RAM available at load |
| `--profile balanced` | 0.75 |
| `--profile saver` | 0.50 |
| `--ram-fraction F` | `F` of the RAM available, in (0, 1] |
| `--ram-budget 8GiB` | Exactly that. Replaces the profile |
| `--expert-cache-mb N` | Bypasses the planner: a fixed expert cache, no pressure monitor |

"Available" is the memory free for use when the model loads. In a container or a
systemd unit with a memory limit it is the **cgroup's** room, not the machine's:
`qubivex system` shows both. Under `docker run --memory 2g`, the plan's budget was
1.76 GiB.

0.88 is a measured default inherited from Colibri. 0.75 and 0.50 are policy values, not
measurements. Profiles change budgets only; no profile changes numerics.

A model whose dense weights, activation reserve and one expert per layer exceed the
budget does not load, and says by how much:

```text
qubivex: memory: does not fit: dense 131.12 KiB + activations 512.00 MiB + one expert per layer 24.00 KiB + one KV token 1.00 KiB exceed the 448.22 MiB budget (needs 512.15 MiB, budget 448.22 MiB)
```

## The context

Without `--context`, the model's context is shortened to what the budget can hold for
the planned number of conversations, as above. An explicit `--context` that does not
fit is refused, with the largest that does.

`qubivex models analyse` reports this in its verdict:

```text
Verdict  runs here, everything in RAM, with the context shortened from 40960 to 6438
```

## At load

The plan made before loading uses an estimate (`available − dense`). At load the engine
reads the dense weights, measures available RAM again with them resident, and plans from
that measurement. The context and the per-layer expert slots come from this plan.

## While running

- **KV grows in 256-position chunks,** allocated as used, never up front. The planner
  prices whole chunks.
- **Before each forward pass** the engine projects the KV after it, lowers the expert
  cache to match (evicting unleased experts), and refuses the pass if it would cross
  the floor.
- **Per-layer slots.** The cache's bytes are divided into slots per layer, evenly, one
  at a time to the layer with the fewest. Each layer's pool evicts only its own
  experts.
- **Retained KV** (the prefix cache) is priced with live KV, so the expert cache shrinks
  to hold it, and it is the first thing given up when a live request needs room.

## The ledger

The `MemoryLedger` records what is actually allocated, by owner: dense, expert cache,
KV, activations. It is charged through values that return their bytes when dropped, so
an error path cannot leak a charge.

- `/v1/system` shows it per model as `ledger_bytes`, beside `plan_budget_bytes`.
- `/metrics` has `qubivex_memory_{dense,expert_cache,kv,activations}_bytes` and
  `qubivex_memory_budget_bytes`.
- A test on the real model checks the ledger against the process's resident size: with
  every expert resident the ledger read 3.90 GiB while resident size grew 3.75 GiB.
  The ledger is the larger, because reserved KV pages that were never touched are not
  resident.

## Memory pressure

A monitor thread samples the machine's available RAM every 500 ms and compares it with
thresholds. Each level adds to the one before:

| Level | Available RAM | Response |
|---|---|---|
| Normal | 1 GiB or more | None |
| Elevated | Under 1 GiB | Stop prefetch; drop retained KV |
| High | Under 512 MiB | Shrink the expert cache |
| Critical | Under 256 MiB | Expert cache at the floor; refuse new requests (503) |

It escalates at once and steps down only when memory is 256 MiB above a threshold, so
it does not flap. It never cancels a running request: at the floor a request can still
finish, with the same tokens, more slowly.

The thresholds are policy values. An engine with a fixed `--expert-cache-mb` has no
monitor.

On macOS the system answers a new allocation partly by compressing other memory, so
available RAM falls by less than another process allocates. The monitor therefore sees
less pressure than the same load would show on Linux.

## Several conversations and several models

`--max-concurrent N` on `serve` plans for `N` conversations' KV. Use
`qubivex plan <ID> --sequences N` to see the result before starting.

Each loaded model plans **alone**, against the memory available when it loads. Two
models loaded one after the other each take their fraction of what was free at their
own load, and nothing arbitrates between them afterwards. Give each an explicit
`--ram-budget`, or run one model per server.

## A discrete GPU (ADR-018)

Metal on Apple Silicon shares host RAM and is counted there. A GPU with its own memory
gets a separate VRAM budget, never summed with the RAM one. `qubivex plan --vram 16GiB`
plans one without loading anything. `--vram-budget` replaces the profile's fraction of
the device's memory. From the budget, the planner takes:

- the dense weights' device copy (the host keeps its copy for the work done there);
- a 256 MiB device reserve;
- a device expert cache from the rest, never larger than the RAM expert cache at full
  context, because the device only holds copies of experts the RAM cache holds.

No CUDA or Vulkan backend exists yet. `qubivex run --backend sim --sim-vram SIZE`
runs on a simulated device that copies weights into its own memory with a hard
capacity, so the plan, the ledger's VRAM tier and the device expert cache can be
checked on any machine. The output bits are the CPU's.

## Not supported

- **A real discrete GPU.** The VRAM budget is planned and simulated (`--backend sim`);
  CUDA and Vulkan are not built.
- **A shared budget across models** in one server, or across servers on one machine.
- **Per-layer slot counts from measured demand.** Slots are spread evenly.
- **Swap-aware planning.** The planner budgets RAM and treats swapping as a failure of
  the budget, not a resource.
- **NUMA placement** and huge pages.
- **Fragmentation metrics.** Peak VRAM is reported for a discrete device
  (`qubivex_vram_peak_bytes`); there is none to measure on Metal.
- **Changing the budget of a loaded model.** Reload it.
