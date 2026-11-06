<p align="center">
  <img src="docs/assets/qubivex-logo.png" alt="QUBIVEX logo" width="480">
</p>

# QUBIVEX

**Run large Mixture-of-Experts (MoE) language models on ordinary hardware by streaming
experts from disk and RAM instead of holding the whole model in memory.**

QUBIVEX is a Rust runtime for MoE models such as OLMoE and Qwen3-MoE. In an MoE model,
each token uses only a few experts out of many (for example 8 of 64 per layer). Most of
the weights are therefore idle at any moment. QUBIVEX keeps the small, always-used part
of the model (embeddings, attention, routers, norms) resident. It treats the experts as
a cache: experts are loaded from fast storage when the router picks them, kept while
they are hot, and evicted under memory pressure. A model several times larger than your
RAM or VRAM can still run, and more memory makes it faster, never different.

QUBIVEX is a Rust redesign of the mechanisms proven in
[Colibri](https://github.com/JustVugg/colibri) (C, Apache-2.0). Colibri showed the
approach works. QUBIVEX's aim is to make it safe, model-agnostic and maintainable:
models are described by manifests rather than per-family engines, untrusted model files
are validated before any byte is used, and results are bitwise reproducible.

> **Status: early development.** QUBIVEX now generates text on CPU (`qubivex run`), with
> experts streamed through a cache of any size; the output is identical whatever the
> cache holds. It is a prototype: about 21 tok/s decode for OLMoE-1B-7B Q4_K_M on
> an Apple M3 Pro, lossless. That is faster than Colibri's default lossy kernels on the
> same machine, with output identical to Colibri's lossless path
> ([benchmark](docs/performance/COLIBRI-OLMOE-BENCHMARK.md)). No GPU, no server yet. See
> [Roadmap](#roadmap) and [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md).

---

## Contents

- [Why QUBIVEX](#why-qubivex)
- [What works today](#what-works-today)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Command reference](#command-reference)
- [Model manifests](#model-manifests)
- [Embedding the runtime](#embedding-the-runtime)
- [Supported models and formats](#supported-models-and-formats)
- [Design principles](#design-principles)
- [Roadmap](#roadmap)
- [Repository layout](#repository-layout)
- [Development](#development)
- [Documentation](#documentation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Why QUBIVEX

Large MoE models have a structure that most runtimes ignore:

| | Dense part | Routed experts |
|---|---|---|
| Used by every token | yes | no: only the top-K chosen by the router |
| Share of the weights | small (OLMoE-1B-7B: well under 1 GB) | large (OLMoE-1B-7B: 1,024 experts ≈ 6.4 GB in int8) |
| Where QUBIVEX keeps it | always resident | an expert cache across VRAM, RAM and disk |

Because only a few experts are active per token, the experts can live on fast storage
and be fetched on demand. The cost moves from "how much memory do I have" to "how fast
can I read an expert", and QUBIVEX measures both (`qubivex system`,
`qubivex doctor --storage`) so it can plan memory for your machine.

What QUBIVEX adds over a straight port:

- **One runtime, many models.** An architecture is a registered component, and a model
  is a `manifest.json`. A new checkpoint of a supported architecture loads with no code
  change. Tiny checkpoints of two MoE families already load through the same code
  path.
- **Untrusted files are validated, never repaired.** Every tensor's byte length must
  match its dtype and shape, offsets must fit inside the file, and the whole tensor set
  must match what the architecture's config requires. Problems are reported all at once
  and by name.
- **Placement never changes the output.** Whether an expert came from VRAM, RAM or disk,
  and whatever the cache size, the tokens and logits are bitwise identical.
- **Lossless by default.** Anything that changes numerics (for example quantising
  activations for integer dot products) is opt-in and named.
- **Memory safety where it matters.** The model, hardware and CLI crates contain no
  `unsafe` code. When kernels and expert streaming arrive (Phase 4 onward), `unsafe`
  is allowed only in those crates, and CI will run Miri and Loom on them.

## What works today

| Area | Capability |
|---|---|
| Hardware | CPU model, core tiers (performance/efficiency) and compute-team size, SIMD set, NUMA nodes, RAM, **cgroup memory limits** in containers, GPUs per backend (CUDA, Vulkan, Metal), filesystems with SSD/HDD/network and free space, OS |
| Storage | Sequential read throughput with and without the page cache (`O_DIRECT` / `F_NOCACHE`), random 4 MiB read latency (about one expert) |
| Model files | SafeTensors (single or sharded) and GGUF v2/v3 readers, hardened against malformed and malicious headers |
| Models | Manifest v1, architecture registry, the `transformer_moe` and `transformer_dense` architectures, full cross-validation of manifest, config and tensors, discovery, automatic manifest import |
| Inference | CPU forward pass for `transformer_moe` (OLMoE, Qwen3-MoE) and `transformer_dense` (dense Qwen3, Llama-style): lossless kernels for f32, f16, bf16, Q8_0, Q4_K and Q6_K; an expert cache of any size with leases and LRU eviction; KV cache; greedy, temperature and top-p sampling; chat templates |
| Tokenizers | Hugging Face `tokenizer.json`, or rebuilt from a GGUF file's vocabulary (verified identical to the official tokenizer) |
| Serving | `qubivex serve`: OpenAI-compatible `/v1/chat/completions` and `/v1/completions` (streaming or not), `/v1/models`, `/health`, `/metrics`, `/v1/system`, admin load and unload; independent KV per conversation, admission against the memory plan, cancellation when a client disconnects |
| CLI | `qubivex run`, `qubivex serve`, `qubivex bench`, `qubivex system`, `qubivex doctor --storage`, `qubivex models search / pull / list / inspect / analyse / import / migrate / verify / remove / architectures` |

See the [Roadmap](#roadmap) for what is next.

## Installation

QUBIVEX is not published to crates.io yet. Build it from source.

**Requirements**

- Rust 1.85 or newer (`rustup` recommended). The workspace uses edition 2024.
- Linux (x86-64 or arm64) or macOS (arm64 or x86-64). CI tests Linux x86-64 and macOS arm64.
- No C toolchain, Python or GPU SDK is needed for the current features.

**Build and install**

```sh
git clone https://github.com/kentmarete/qubivex.git
cd qubivex
cargo build --release
./target/release/qubivex --version

# or put `qubivex` on your PATH:
cargo install --path apps/cli
```

## Quick start

The repository ships four tiny models in [`fixtures/models/`](fixtures/models), so
you can try everything without downloading anything.

```sh
# What hardware does QUBIVEX see?
qubivex system

# List the bundled tiny models
qubivex models list --dir fixtures/models
```

```text
ID               FORMAT            LAYERS  EXPERTS    SIZE        STATUS
olmoe-tiny       safetensors bf16  2       8 (top 2)  323.12 KiB  ok
olmoe-tiny-gguf  gguf q8_0         2       8 (top 2)  173.31 KiB  ok
qwen3-moe-tiny   safetensors bf16  2       8 (top 2)  354.88 KiB  ok
```

```sh
# Look inside one
qubivex models inspect olmoe-tiny --dir fixtures/models
```

```text
Model    olmoe-tiny (Tiny OLMoE-style fixture)
Dir      fixtures/models/olmoe-tiny
Arch     transformer_moe 1.0.0
Shape    2 layers, hidden 64, vocab 256, 8 experts (top 2), context 4096
Bytes    323.12 KiB total: 192.00 KiB routed experts, 131.12 KiB dense; largest expert 12.00 KiB
Dtypes   bf16 323.12 KiB
Files
  fixtures/models/olmoe-tiny/model.safetensors  330.41 KiB
Tensors  69
  attention           12  64.50 KiB
  embedding            1  32.00 KiB
  norms                5  640 B
  output head          1  32.00 KiB
  routed experts      48  192.00 KiB
  router               2  2.00 KiB
Config   {...}
```

### Using a real model

1. Download a supported checkpoint, either a Hugging Face directory (`config.json` and
   `*.safetensors`) or a GGUF file. For example, the official OLMoE GGUF:

   ```sh
   mkdir -p ~/.qubivex/models/olmoe-1b-7b-0125-instruct
   cd ~/.qubivex/models/olmoe-1b-7b-0125-instruct
   curl -LO https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct-GGUF/resolve/main/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf
   ```

   The file is about 4.2 GB, so have about 5 GB free. If the download is interrupted,
   resume it with `curl -C - -LO <same URL>`. The full-precision SafeTensors checkpoint
   is about 14 GB.

   The model can live anywhere. `~/.qubivex/models` is only the default search
   directory. Elsewhere, add `--dir /path/to/models` to `models` and `run` commands,
   or pass the model's own folder as `--model /path/to/folder`.

2. Let QUBIVEX write a manifest. It detects the tensor layout by validating each
   candidate against the real tensors:

   ```sh
   qubivex models import ~/.qubivex/models/olmoe-1b-7b-0125-instruct \
       --id olmoe-1b-7b-0125-instruct --set norm_topk_prob=false --write
   ```

   Some facts are not recorded in every checkpoint. GGUF files, for instance, do not
   always say whether the top-K expert weights are renormalised. QUBIVEX then stops and
   names the missing value, and you supply it with `--set KEY=VALUE`. Take the value
   from the model's original `config.json` (OLMoE: `norm_topk_prob=false`; Qwen3-MoE:
   `norm_topk_prob=true`).

3. List and inspect it. `~/.qubivex/models` is the default model directory, so no
   `--dir` is needed:

   ```sh
   qubivex models list
   qubivex models inspect olmoe-1b-7b-0125-instruct
   ```

4. Generate:

   ```sh
   qubivex run --model olmoe-1b-7b-0125-instruct --prompt "The capital of France is" --max-tokens 12
   ```

   ```text
   The capital of France is Paris.

   The official language of France is French.
   ```

   Add `--chat` to wrap the prompt in the model's chat template, and
   `--expert-cache-mb 64` to stream experts through a small cache. The text stays the
   same; only the speed changes.

   The first run is slower, because it reads the experts from disk; later runs find
   them in the operating system's page cache.

## Command reference

Every command accepts `--help`.

### `qubivex run --model <ID|DIR> --prompt TEXT [options]`

Generates text on the CPU, streaming it to standard output, then prints prompt and
generation speed and expert-cache counters to standard error.

| Option | Meaning |
|---|---|
| `--chat` | Wrap the prompt as a user message with the model's chat template |
| `--max-tokens N` | Most tokens to generate (default 128); generation also stops at end-of-sequence |
| `--temperature T`, `--top-p P`, `--seed S` | Sampling. Temperature 0 (the default) is greedy; the same seed gives the same text |
| `--expert-cache-mb N` | Expert cache size (default: every expert resident). Smaller caches stream experts from disk and never change the output |
| `--context N` | Context limit in tokens (default: the model's) |
| `--threads N` | Compute threads (default: the CPU's performance cores) |
| `--quiet` | Do not print the statistics line |
| `--dir DIR` | Model directories to search for the id (as for `models list`) |
 Commands exit with 0 on success, 1 on a runtime error,
and 2 on a usage error.

### `qubivex serve [--model <ID|DIR>]... [options]`

Serves models over an OpenAI-compatible HTTP API (ADR-013). Any OpenAI client works:

```sh
qubivex serve --model olmoe-1b-7b-0125-instruct --model qwen3-0.6b --api-key sk-local
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-local")
r = client.chat.completions.create(model="qwen3-0.6b",
        messages=[{"role": "user", "content": "Hello"}], stream=True)
```

| Endpoint | |
|---|---|
| `POST /v1/chat/completions`, `POST /v1/completions` | Streaming (SSE, `stream_options.include_usage`) or not; `max_tokens`, `temperature`, `top_p`, `seed`, `stop`. Refused, not ignored: `n > 1`, tools, logprobs, penalties |
| `GET /v1/models`, `GET /v1/models/{id}` | The loaded models |
| `GET /health`, `GET /metrics` | Liveness; Prometheus counters (requests, tokens, cancellations, reserved KV) |
| `GET /v1/system` | Hardware and each model's memory (plan, ledger, reserved KV, free slots) |
| `POST /v1/admin/models` `{"model": ID}`, `DELETE /v1/admin/models/{id}` | Load and unload, with `--admin-token` only |

| Option | Default | |
|---|---|---|
| `--listen ADDR` | `127.0.0.1:8080` | |
| `--api-key KEY` | `$QUBIVEX_API_KEY` | Required as a bearer token on `/v1` when set |
| `--admin-token T` | `$QUBIVEX_ADMIN_TOKEN` | Enables the admin endpoints |
| `--max-concurrent N` | 4 | Generations per model at once; more wait up to `--queue-timeout` (30 s), then 429 |
| `--default-max-tokens N` | 512 | When a request does not set `max_tokens` |
| `--request-timeout S` | 600 | A generation then ends with `finish_reason: "length"` |
| memory, device, `--context`, `--threads` | as `qubivex run` | Per model |

Each conversation has its own KV cache and is admitted only if the memory plan can hold
every live conversation's KV with one expert slot per layer (503 otherwise). Forward
passes of concurrent conversations take turns, first come first served, and long
prompts run in chunks, so one request cannot hold the others up for long. A client
that disconnects cancels its generation within one chunk or one token. Errors follow
OpenAI's shape, carry the `x-request-id`, and never contain file paths.

### `qubivex system [--json]`

Prints the hardware profile and the resources available right now:

- the OS and kernel;
- the CPU model, logical and physical cores, and core tiers. The **compute team** is
  every physical core outside the slowest tier: on a CPU with efficiency cores, a
  barrier-synchronised team runs at the pace of its slowest thread;
- run-time SIMD features (for example `avx2`, `neon`, `i8mm`) and NUMA nodes;
- total RAM, the **cgroup limit** when running in a container, and RAM free now. Inside
  a container, the cgroup's room is what counts, not the machine's free memory;
- one line per GPU backend: devices found, or why it is unavailable;
- mounted filesystems with type, SSD/HDD/network and capacity.

A value that cannot be determined is shown as `unknown (reason)`, never as zero. With
`--json`, every such value is `{"known": ...}` or `{"unknown": "reason"}`. GPU probes
run in a child process with a timeout, so a broken driver cannot hang the command.

### `qubivex doctor --storage [--path DIR] [--size-mib N] [--json]`

Measures the filesystem that holds `DIR` (default: the current directory). This is
where expert loading time comes from. It writes a scratch file of `N` MiB (default
1024), then measures:

- a sequential read that bypasses the page cache (`O_DIRECT` on Linux, `F_NOCACHE` on
  macOS), in 8 MiB requests;
- random 4 MiB uncached reads (median and 90th percentile), about the size of one expert;
- a buffered sequential read with the file evicted from the cache;
- the direct-I/O gain, the uncached rate divided by the buffered rate.

It refuses to run unless the scratch file plus 1 GiB fits in the free space, and it
removes the scratch file afterwards, even on error.

### `qubivex models list [--dir DIR]... [--json]`

Lists every model directory (a directory holding `manifest.json`) found in each model
directory or one level below it. Every model is fully validated, and a broken model is
listed with its error instead of being skipped.

Model directories, in order of precedence:

1. `--dir DIR` (repeatable);
2. `QUBIVEX_MODELS_DIRS`, `:`-separated;
3. `~/.qubivex/models`.

### `qubivex models inspect <ID|DIR> [--dir DIR]... [--json] [--tensors]`

Shows a model's dimensions, byte breakdown, files, dtypes, tensor counts by role and its
parsed architecture config. The byte breakdown covers routed experts (streamable) versus
dense (resident), and the largest single expert, which is the unit the expert cache
moves. `--tensors` lists every tensor with its dtype, shape and role. `<ID>` is
looked up in the model directories; a path to a model directory also works.

### `qubivex models analyse <ID|DIR> [options]`

Says whether a model can run on this machine, and what it would take:

```bash
qubivex models analyse olmoe-1b-7b-0125-instruct
qubivex models analyse olmoe-1b-7b-0125-instruct --ram-budget 2GiB --ctx 8192 --json
```

The verdict is one of: everything in RAM, runs while reading some experts from disk, or
does not fit. It lists the RAM to start, to reach the context, and to keep every expert
cached, then this machine's RAM and what each backend would do. It reads only headers and
changes nothing. "Does not fit" is an answer, so the command still exits 0; scripts should
read `assessment.verdict` from `--json`. It takes the options of `qubivex plan`, which shows
the placement behind the verdict. `analyze` also works.

### `qubivex models verify <ID|DIR> [--record] [--threads N]`

Checks every file against the checksums in its manifest, then every expert against
`experts.qvxsum`, which it records if there are none. `--record` records both again from
the files as they are, after replacing them on purpose; nothing else accepts a changed
file. `qubivex models validate` is an alias.

### `qubivex models migrate <ID|DIR> [--write] [--no-checksums]`

Brings a manifest up to the current schema (1.1), adding the source from the registry and
the size and checksum of every file. Prints the result unless given `--write`, which
replaces `manifest.json` and reads every file once. A checksum already recorded stays, so
migrating cannot hide a file that changed. Models keep working without this: 1.0 manifests
load as they are.

### `qubivex models import <DIR|FILE.gguf> [options]`

Writes a `manifest.json` for a checkpoint:

| Option | Meaning |
|---|---|
| `--id ID` | Model id used on the command line (default: the directory or file name, sanitised) |
| `--name NAME` | Display name |
| `--set KEY=VALUE` | A config value the checkpoint does not record, by its `config.json` name (repeatable). `VALUE` is parsed as JSON when possible (`true`, `1e-6`), otherwise taken as a string |
| `--write` | Write `manifest.json` next to the weights instead of printing it |
| `--force` | Replace an existing `manifest.json` |

The importer finds the weight files: a `model.safetensors.index.json`, all
`*.safetensors`, or a single `.gguf`. It then tries each architecture
(`transformer_moe`, then `transformer_dense`), tensor-name layout and q/k-norm mode, and
keeps the first combination under which every tensor validates. If none does, it
reports the problems of the closest candidate. A tied output head reads the embedding;
a copy of it stored under the head's name is listed in `overrides.ignore_tensors`. It records shard sizes,
which are checked at every load, and the dominant dtype. With `--write` it also records
each file's checksum and the per-expert checksums, unless given `--no-checksums`.

### `qubivex models search <QUERY> [--limit N] [--json]`

Finds models on Hugging Face, most downloaded first.

### `qubivex models pull <OWNER/NAME> [options]`

Downloads a model and installs it, with no other tools needed:

```bash
qubivex models pull Qwen/Qwen3-0.6B
qubivex models pull owner/Model-GGUF --file Model-Q4_K_M.gguf --set norm_topk_prob=false
```

Only the files a model needs are fetched, each is checked against its published checksum,
and the model appears in the models directory only when complete. If a pull is interrupted,
the same command continues it. A repository with several weight files needs `--file`.

| Option | Meaning |
|---|---|
| `--file PATH` | A file to fetch, by its path in the repository (repeatable) |
| `--revision REV` | A branch, tag or commit. The exact commit is recorded either way |
| `--id ID`, `--name NAME` | The model id and display name |
| `--set KEY=VALUE` | A config value the checkpoint lacks, as for `models import` |
| `--dir DIR` | The models directory to install into |
| `--concurrency N` | Files fetched at once (default 4) |

Private and gated models read `HF_TOKEN`, or the token `hf auth login` saved; there is no
token flag, because arguments show up in process listings. `HF_ENDPOINT` selects a mirror.

### `qubivex models remove <ID> [--dir DIR]... [--yes]`

Deletes a model from its models directory. Without `--yes` it only says what would be
deleted. It refuses symbolic links and models outside the directory's top level, and
does not check whether a running server has the model loaded.

### `qubivex models architectures`

Lists the architectures this build supports.

## Model manifests

A model directory holds the weights, `config.json` (for SafeTensors) and a
`manifest.json`. The manifest **does not repeat dimensions**: those are read from
`config.json` or the GGUF metadata. It names the files and the architecture, and adds
only what the checkpoint does not say. A manifest for a SafeTensors checkpoint:

```json
{
  "schema_version": "1.0.0",
  "model": { "id": "qwen3-moe-tiny", "name": "Tiny Qwen3-MoE-style fixture" },
  "architecture": {
    "id": "transformer_moe",
    "version": "1.0.0",
    "config": { "tensor_names": "hf_fused", "qk_norm": "per_head" }
  },
  "weights": {
    "format": "safetensors",
    "shards": [{ "path": "model.safetensors", "size": 365960 }],
    "config": "config.json"
  },
  "tokenizer": { "kind": "huggingface", "path": "tokenizer.json" },
  "quantization": "bf16"
}
```

| Field | Purpose |
|---|---|
| `schema_version` | Manifest schema version. This build reads `1.x.x` |
| `model.id` | Stable id: `a-z`, `0-9`, `.`, `_`, `-`; up to 128 characters |
| `architecture.id` / `version` | A registered architecture. Accepted when the major version matches and the minor is not newer than the build's |
| `architecture.config` | Architecture options the checkpoint does not declare (below) |
| `weights.format` | `safetensors` or `gguf` (a GGUF model has exactly one shard and no `config`) |
| `weights.shards[]` | Paths relative to the model directory. Optional `size` (checked at load) and `sha256` |
| `tokenizer` | `{"kind": "huggingface", "path": "tokenizer.json"}` or `{"kind": "gguf"}` |
| `context_length`, `capabilities`, `quantization` | Optional metadata |
| `overrides.config` | Values missing from `config.json` / GGUF metadata. A value the checkpoint also states must agree with it |
| `overrides.ignore_tensors` | Tensors present in the files but deliberately unused |

**`transformer_moe` options** (`architecture.config`):

| Key | Values | Default |
|---|---|---|
| `tensor_names` | `"hf"` (one tensor per expert, as released Hugging Face checkpoints), `"hf_fused"` (stacked experts, as newer `transformers` saves), `"gguf"` (standard GGUF names), or `{"base": <preset>, "<slot>": "<pattern>"}` to rename individual tensors, with `{L}` for the layer and `{E}` for the expert | `"hf"` for SafeTensors, `"gguf"` for GGUF |
| `qk_norm` | `"none"`, `"full"` (one norm across all heads), `"per_head"` | `"none"` |
| `router_scoring` | `"softmax"`, `"sigmoid"` | `"softmax"` |

Manifests are strict. Unknown fields, absolute paths, `..` and a schema or architecture
version this build does not implement are all errors. The full JSON Schema is in
[`docs/model-support/manifest.schema.json`](docs/model-support/manifest.schema.json).

**Trust order.** Tensor headers, checked against the real file sizes, take precedence
over the config, which takes precedence over the manifest. At load, QUBIVEX lists every
missing tensor, wrong shape and unexpected tensor in one error. A wrong manifest fails
loudly instead of producing wrong numbers.

## Embedding the runtime

`qubivex-runtime` works inside another Rust program, with no server, CLI or async runtime.
Share one `Arc<Engine>` and give each request a thread:

```rust
let meta = ModelMetadata::load(&dir, &ArchitectureRegistry::with_builtins())?;
let engine = Engine::load(meta, EngineOptions::default())?;
let request = GenerationRequest::new(Prompt::Text("Hello".into()));
engine.run_request(&request, &AtomicBool::new(false), |event| {
    if let GenerationEvent::Text(piece) = event {
        print!("{piece}");
    }
    true // false stops the generation
})?;
```

A complete program is in
[`crates/qubivex-runtime/examples/embed.rs`](crates/qubivex-runtime/examples/embed.rs).

## Supported models and formats

**Architectures:**
- `transformer_moe`, a decoder-only transformer with RMSNorm, RoPE attention (MHA, GQA
  or MQA, optional q/k norm) and routed SwiGLU experts in every layer, with softmax or
  sigmoid routing and optional top-K renormalisation.
- `transformer_dense`, the same attention with one dense SwiGLU FFN in every layer. The
  forward pass shares the embedding, attention, KV cache and head with
  `transformer_moe`; only the FFN differs. It runs on the CPU; on Metal its matmuls are
  placed one by one (same bits, slower than the CPU for now).

| Model family | Status |
|---|---|
| OLMoE (for example `allenai/OLMoE-1B-7B-0125-Instruct`) | Supported by configuration in per-expert SafeTensors, stacked SafeTensors and GGUF layouts. Validated on tiny checkpoints and on the full-size Q4_K_M GGUF |
| Qwen3-MoE (for example `Qwen/Qwen3-30B-A3B`) | Supported by configuration in per-expert and stacked SafeTensors layouts. Tested on tiny checkpoints; not yet on a full-size one |
| Dense Qwen3 (for example `Qwen/Qwen3-0.6B`) | `transformer_dense` by configuration (per-head q/k norm, GQA, tied head). Validated on the full-size bf16 SafeTensors checkpoint: 32 greedy tokens equal transformers', logits within 1.2e-5 |
| Dense Llama-style (no q/k norm, GQA, tied or separate head) | `transformer_dense` by configuration. Tested on a tiny checkpoint against transformers |
| Shared-expert or partly dense MoE (Qwen2-MoE, DeepSeek, GLM) | Refused with a clear message for now |
| Attention or FFN biases, non-SiLU activations, RoPE scaling (YaRN and similar), sliding-window attention | Refused with a clear message for now |

**Containers:** SafeTensors (single file or sharded with `model.safetensors.index.json`)
and GGUF v2/v3, little-endian, single file.

**Dtypes:** `f32`, `f16`, `bf16`, and the GGML block types `Q4_0`, `Q4_1`, `Q5_0`,
`Q5_1`, `Q8_0`, `Q8_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` and `Q8_K`. I-quants (`IQ*`)
are not supported yet.

Adding a model family that fits `transformer_moe` or `transformer_dense` takes a
manifest, not code. A new
mechanism (for example multi-head latent attention) is a new architecture registered
with `ArchitectureRegistry`, and runtime code does not change.

## Design principles

These invariants are enforced by tests and CI
([`docs/architecture/QUBIVEX_DESIGN.md`](docs/architecture/QUBIVEX_DESIGN.md)):

1. **Placement decides speed, never bits.** Outputs are identical whatever the cache
   capacity or the device an expert ran on.
2. **Prediction never routes.** Prefetching may read router state, never change it.
3. **No model names in shared code.** `cargo xtask check` rejects model-family names in
   `crates/` and `apps/`. Models are described only by manifests.
4. **Lossy is opt-in and named.**
5. **Untrusted inputs are validated, never repaired.**
6. **Errors fail the request, not the process.**

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Rust workspace, core types, CI. ADR-002 tensor-engine decision: own kernels, no Candle | Done |
| 2 | Hardware detection, `qubivex system`, `qubivex doctor --storage` | Done |
| 3 | Model containers, manifest, architecture registry, validation, `qubivex models` | Done (checked on the real OLMoE-1B-7B Q4_K_M GGUF) |
| 4 | CPU inference prototype: kernels, tokenizer, a minimal expert store, generation | Done: matches `transformers` to under 1e-6 relative; identical output at every expert-cache size |
| 5 | Memory planner: fits the model to RAM, VRAM and cgroup limits | Done |
| 6 | Full expert streaming: tiered cache, I/O pool, packed expert files | Done |
| 7 | Predictive expert prefetching | Router lookahead and the disk-saturation throttle done; off by default |
| 8 | GPU backends: CUDA, Vulkan, Metal | Metal slices 1–6 done (bit-identical; 1.4× the CPU at decode, 2× at prefill); CUDA next |
| 9 | OpenAI-compatible HTTP server and serving | Done: `qubivex serve` (ADR-013); the OpenAI SDK works against real models |
| 10–11 | Hardening, packaging, documentation | Planned |

The phase plan, exit criteria and risks are in
[`docs/architecture/MIGRATION_PLAN.md`](docs/architecture/MIGRATION_PLAN.md).
Current progress and test results are in
[`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md).

## Repository layout

```text
apps/cli/                 the `qubivex` binary (logic in lib.rs; main.rs is thin)
crates/qubivex-core/      ids, Bytes, DType and GGML block sizes, Shape, config types
crates/qubivex-hardware/  hardware profile, resource snapshots, storage measurement
crates/qubivex-model/     SafeTensors/GGUF readers, tensor index, manifest,
                          architecture registry, transformer_moe/dense, discovery, import
crates/qubivex-tokenizer/ tokenizers and chat templates
crates/qubivex-compute/   tensors, views, the backend trait
crates/qubivex-backend-*/ CPU (lossless SIMD kernels) and Metal backends
crates/qubivex-arch/      the forward pass (shared decoder, MoE or dense FFN), KV cache
crates/qubivex-expert-stream/  expert store: pools, leases, I/O threads, prefetch
crates/qubivex-memory/    footprint, memory planner, ledger, pressure monitor
crates/qubivex-runtime/   the engine: loading, generation, admission
crates/qubivex-api/       the OpenAI-compatible HTTP gateway (`qubivex serve`)
xtask/                    repository checks and fixture generator (`cargo xtask ...`)
fixtures/models/          tiny generated models used by tests and the quick start
spikes/                   standalone experiments (ADR-002 tensor-engine benchmark)
docs/                     architecture, decisions (ADRs), guidelines, performance data
```

Crates are added in the phase that first gives them tested code (ADR-001). A dependency
rule (`cargo xtask check-deps`) keeps the layering one-way: for example, the model crate
never depends on compute backends.

## Development

```sh
cargo test --workspace                                   # every test
cargo clippy --workspace --all-targets -- -D warnings   # lints (CI treats warnings as errors)
cargo fmt --all --check
cargo xtask check                                        # dependency rules, model-name guard, fixtures
cargo deny check                                         # advisories, licences, sources (needs cargo-deny)
```

CI runs all of these on Linux x86-64 and macOS arm64, plus a build and test on the
minimum supported Rust version, 1.85.

**Fixtures.** `fixtures/models/` is generated deterministically by
`cargo xtask gen-fixtures`, and `cargo xtask check` fails if the committed files drift
from the generator. After changing the generator, regenerate and commit.

**Tests against real checkpoints** are opt-in, so CI stays offline:

```sh
QUBIVEX_REAL_MODEL_FILES=/path/a.safetensors:/path/b.gguf \
QUBIVEX_REAL_MODEL_DIRS=/path/to/model-dir \
cargo test -p qubivex-model --test real_files -- --nocapture
```

**Guidelines.** Read [`docs/development/RUST_GUIDELINES.md`](docs/development/RUST_GUIDELINES.md)
before contributing. In short:

- domain errors with `thiserror`, and no `unwrap` outside tests;
- checked arithmetic for every size derived from a file;
- `unsafe` only in the designated crates, each block with a `// SAFETY:` comment;
- no model names in shared code.

Architectural changes go through an ADR in [`docs/decisions/`](docs/decisions).

## Documentation

| Document | Contents |
|---|---|
| [`QUBIVEX_DESIGN.md`](docs/architecture/QUBIVEX_DESIGN.md) | Architecture, invariants, crate layering, core interfaces |
| [`MIGRATION_PLAN.md`](docs/architecture/MIGRATION_PLAN.md) | Phases, exit criteria, reference model, risks |
| [`COLIBRI_ANALYSIS.md`](docs/architecture/COLIBRI_ANALYSIS.md) | What QUBIVEX keeps, and changes, from Colibri |
| [`docs/decisions/`](docs/decisions) | ADR-001 to ADR-008: workspace, compute, registry, manifest, planner, streaming, backends, cache |
| [`ADR-002-SPIKE.md`](docs/performance/ADR-002-SPIKE.md) | Kernel benchmarks against Candle and Colibri |
| [`COLIBRI-OLMOE-BENCHMARK.md`](docs/performance/COLIBRI-OLMOE-BENCHMARK.md) | End-to-end decode against Colibri `olmoe` on OLMoE-1B-7B Q4_K_M |
| [`manifest.schema.json`](docs/model-support/manifest.schema.json) | Manifest JSON Schema |
| [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md) | What is done, deviations, test results |

## Acknowledgements

QUBIVEX builds on the design and measurements of
[Colibri](https://github.com/JustVugg/colibri) by its authors (Apache-2.0). The expert
streaming approach, the cgroup and memory measurements, and many lessons recorded in
Colibri's history shaped this design. Reference models come from the
[OLMoE](https://huggingface.co/allenai) and [Qwen](https://huggingface.co/Qwen) teams.

## License

The workspace is declared as Apache-2.0 in `Cargo.toml`. A `LICENSE` file with the full
text has not been added to the repository yet.
