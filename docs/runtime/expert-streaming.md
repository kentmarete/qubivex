# Expert streaming

Expert streaming is how QUBIVEX runs a MoE model whose experts do not all fit in memory:
experts live in a cache, and the ones a token needs that are not there are read from the
model file while the token is computed. This page covers the cache, the reads, and what
happens when a read fails. The design records are
[ADR-006](../decisions/ADR-006-EXPERT-STREAMING.md) (the store) and
[ADR-008](../decisions/ADR-008-CACHE-STRATEGY.md) (the cache).

Why it works is in [MoE routing](moe-routing.md): a token uses a few experts per layer,
so most of the model's bytes are idle at any moment.

## The trade

| Expert cache | Memory | Speed |
|---|---|---|
| Every expert | The whole model | Fastest: no reads after warm-up |
| Part | Dense weights + cache + KV | Slower by the reads that miss |
| One slot per layer (the floor) | Dense weights + KV + a few MiB | Slowest: nearly every expert is read |

**The output is identical in every row.** The cache decides speed, never tokens.

On OLMoE-1B-7B Q4_K_M (M3 Pro, CPU, one conversation, warm file cache): about 41 tokens
per second with every expert resident, about 35 with a 1024 MiB cache out of 3.6 GiB of
experts.

## Where experts come from

The original model files, always. There is no conversion step and no second copy on
disk.

| Container | An expert is |
|---|---|
| SafeTensors, one tensor per expert | Three tensors (gate, up, down); adjacent ones are read together |
| SafeTensors, stacked | Three slices of the layer's stacked tensors |
| GGUF | Three slices of the layer's 3-D expert tensors |

Every read's size is checked against the size the model's config implies before it is
read.

## The cache

- **One pool per layer.** Each layer has its own slots, sized to that layer's largest
  expert. A miss evicts only an expert of its own layer. The memory planner decides how
  many slots each layer gets ([Memory planner](memory-planner.md)).
- **Slots are reused.** A miss in a full pool reads into the evicted expert's buffer, so
  once the pools are full, streaming allocates no expert memory.
- **Eviction is LFRU by default:** the coldest expert goes, by a heat count that decays
  over time, with recency as the tie-break (`--cache-policy lfru|lru` on `run` and
  `bench`).
- **The cache resizes while running.** As conversations' KV grows, the planner lowers
  the cache; as they end, it grows back. Under memory pressure it shrinks to the floor.
  It never goes below one slot per layer, so a running request can always finish.

## Leases

A forward pass holds a **lease** on each expert it is using. A leased expert is never
evicted, and the lease releases when dropped. This is what makes it safe for several
conversations, the I/O threads and the eviction policy to share one cache:

- A hit is a table lookup and a counter increment. No lock is held while the expert is
  used.
- Two requests for an expert that is being read share that one read.
- A layer asks for all its experts at once. Hits are leased immediately, so a miss in
  the same set cannot evict them, and every miss with a free slot is queued before the
  layer waits for any.
- A layer with fewer slots than chosen experts still works: it runs through the slots
  it has, releasing each expert as it finishes with it.

The lease and eviction logic is model-checked with loom, and exercised under
ThreadSanitizer in CI.

## The reads

Misses go to a queue served by dedicated I/O threads (4 by default, `--io-threads`),
which `pread` into the slot. The store is synchronous: there is no async runtime under
the forward pass.

Queue priority: a read a token is waiting for comes before a speculative prefetch.

| Option (on `run` and `bench`) | Effect |
|---|---|
| `--io-threads N` | Reader threads. 0 makes the compute thread read its own misses |
| `--direct-io` | Bypass the OS page cache (`O_DIRECT` on Linux, `F_NOCACHE` on macOS), so streaming does not fill the file cache with experts the cache will evict anyway |
| `--prefetch` | Read predicted experts ahead ([MoE routing](moe-routing.md#prefetch)) |
| `--read-timeout S` | Fail a request whose expert read takes longer than this plus the expert's size at 10 MB/s. Default 5 |
| `--read-retries N` | Extra attempts after a failed read. Default 1 |
| `--mirror DIR` | A directory holding a copy of the weight files, read when a read from the model's own files fails (repeatable) |
| `--verify-experts auto\|off\|require` | Check each read against the per-expert checksums. Default `auto`: when `experts.qvxsum` exists and matches the weights |

`qubivex serve` takes `--io-threads` and uses the defaults for the rest.

### How fast must the disk be?

One OLMoE Q4_K_M expert is about 3.9 MiB. `qubivex doctor --storage` reports what
matters for it:

```sh
qubivex doctor --storage --path ~/.qubivex/models
```

It measures sequential throughput with and without the page cache, and the median and
90th-percentile time of a random 4 MiB uncached read, which is about one expert. Compare
those with how many misses per token your cache size gives
(`qubivex_expert_cache_loads_total` against `qubivex_generated_tokens_total`).

The first run after a boot reads experts from the disk itself; later runs find them in
the OS page cache, which is why a second run is faster even with a small expert cache.

## When a read fails

A failed read fails the requests waiting on that expert, and nothing else:

- The expert is never published half-read. The slot is freed and the next request reads
  it again.
- No wrong token is emitted. A pass that fails rolls its KV back, and the tokens
  already sent are the ones a fault-free run would have sent.
- Memory is returned: the ledger after a failure equals the store's contents.
- A reader thread that panics is caught, and its waiters get an error instead of
  waiting forever.
- An allocation the system refuses is an error, not an abort.

Before failing, a read is retried (`--read-retries`), then tried on each mirror. A read
that hangs is abandoned at the deadline.

**Silent corruption** (a read that succeeds with wrong bytes) cannot be seen from sizes.
The per-expert checksums catch it:

```sh
qubivex models verify <ID>     # records experts.qvxsum if there is none
```

With that file present, every expert read is checked, and a mismatch is a failed read
(then a retry, then a mirror). `--verify-experts require` refuses to run a model that
has no checksums.

The fault-injection suite injects I/O errors, short reads, corrupt bytes, panics and
delays at the store and through the whole engine, and checks after each that nothing is
leaked and the next request gives the fault-free tokens.

## Observing it

| Metric | Meaning |
|---|---|
| `qubivex_expert_cache_hits_total`, `_loads_total` | Hit rate is hits over hits plus loads |
| `qubivex_expert_cache_evictions_total` | Churn |
| `qubivex_expert_cache_bytes` | The cache's capacity now |
| `qubivex_expert_read_bytes_total` | Its rate is the disk throughput in use |
| `qubivex_expert_load_wait_seconds_total` | Time generations spent blocked on reads |
| `qubivex_expert_read_retries_total`, `_timeouts_total` | Trouble |

`qubivex run` prints the same counters for one run, and `qubivex bench` records them.

## Not supported

- **`io_uring`.** Reads are `pread` on threads. It stays deferred until measured on a
  Linux host.
- **A packed expert file** (one contiguous extent per expert). Experts are read from
  the original layout.
- **A VRAM tier separate from RAM.** On Apple Silicon the GPU shares host memory; there
  is no discrete-GPU backend to tier for.
- **Remote and object storage** as a source. Experts are read from local files (a
  network filesystem works and is slow).
- **Routing statistics kept between runs** to pre-warm the cache.
- **Streaming dense weights.** Only routed experts stream; a dense model must fit.
- **`--direct-io`, `--prefetch`, mirrors and read tuning on `serve`.** They are on
  `run` and `bench`.
