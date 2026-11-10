# Model manifest

A model is a directory holding weight files and a `manifest.json`. The manifest names
the files and the architecture, and adds only what the checkpoint does not say about
itself. The design record is [ADR-004](../decisions/ADR-004-MODEL-MANIFEST.md), and the
JSON Schema is [`manifest.schema.json`](../model-support/manifest.schema.json).

You rarely write one by hand: `qubivex models pull` and `qubivex models import` write
it.

## Examples

A SafeTensors model, as `models import` wrote it for `Qwen/Qwen3-0.6B`:

```json
{
  "schema_version": "1.0.0",
  "model": { "id": "qwen3-0.6b", "name": "Qwen3-0.6B" },
  "architecture": {
    "id": "transformer_dense",
    "version": "1.0.0",
    "config": { "qk_norm": "per_head", "tensor_names": "hf" }
  },
  "weights": {
    "format": "safetensors",
    "shards": [{ "path": "model.safetensors", "size": 1503300328 }],
    "config": "config.json"
  },
  "tokenizer": { "kind": "huggingface", "path": "tokenizer.json" },
  "quantization": "bf16",
  "overrides": { "ignore_tensors": ["lm_head.weight"] }
}
```

A GGUF model, at schema 1.1:

```json
{
  "schema_version": "1.1.0",
  "model": { "id": "olmoe-tiny-gguf", "name": "Tiny OLMoE-style fixture, GGUF Q8_0" },
  "architecture": {
    "id": "transformer_moe",
    "version": "1.0.0",
    "config": { "qk_norm": "full", "tensor_names": "gguf" }
  },
  "weights": {
    "format": "gguf",
    "shards": [{ "path": "olmoe-tiny-q8_0.gguf", "size": 183584 }]
  },
  "tokenizer": { "kind": "gguf" },
  "quantization": "q8_0",
  "overrides": { "config": { "norm_topk_prob": false } }
}
```

## Fields

| Field | Required | Purpose |
|---|---|---|
| `schema_version` | yes | The manifest schema, as semver. This build reads `1.0.x` and `1.1.x` |
| `model.id` | yes | The id used on the command line and in the API: `a-z`, `0-9`, `.`, `_`, `-`, up to 128 characters |
| `model.name` | no | A display name |
| `architecture.id`, `architecture.version` | yes | A registered architecture: `transformer_moe` or `transformer_dense`. Accepted when the major version matches and the minor is not newer than the build's |
| `architecture.config` | no | Options the checkpoint does not declare (below) |
| `weights.format` | yes | `safetensors` or `gguf` |
| `weights.shards[]` | yes | `path` relative to the model directory, optional `size` (checked at every load) and `sha256` (checked by `models verify`). A GGUF model has exactly one shard |
| `weights.config` | SafeTensors | The checkpoint's `config.json`. A GGUF model has none: its metadata is in the file |
| `tokenizer` | yes | `{"kind":"huggingface","path":"tokenizer.json"}`, or `{"kind":"gguf"}` to rebuild it from the GGUF vocabulary |
| `context_length`, `capabilities`, `quantization` | no | Metadata |
| `overrides.config` | no | Values missing from `config.json` or the GGUF metadata. A value the checkpoint also states must agree with it |
| `overrides.ignore_tensors` | no | Tensors present in the files and deliberately unused |
| `source` (1.1) | no | `provider`, `repository` and the exact `revision` the model came from |
| `files` (1.1) | no | `path`, `size` and `sha256` of the config, tokenizer and other files beside the weights |

### `architecture.config`

| Key | Values | Default |
|---|---|---|
| `tensor_names` | `"hf"` (one tensor per expert, as released Hugging Face checkpoints), `"hf_fused"` (stacked experts, as newer `transformers` saves), `"gguf"` (standard GGUF names), or `{"base": <preset>, "<slot>": "<pattern>"}` to rename individual tensors, with `{L}` for the layer and `{E}` for the expert | `"hf"` for SafeTensors, `"gguf"` for GGUF |
| `qk_norm` | `"none"`, `"full"` (one norm across all heads), `"per_head"` | `"none"` |
| `router_scoring` (`transformer_moe`) | `"softmax"`, `"sigmoid"` | `"softmax"` |

## What the manifest does not hold

**Dimensions.** Layer count, hidden size, expert count, top-K and vocabulary size are
read from `config.json` or the GGUF metadata, never repeated in the manifest. A second
copy could only disagree. `qubivex models inspect <ID>` shows them.

**Tensor-to-file mappings.** They are in the file headers.

**Supported backends.** They depend on the machine, so `qubivex models analyse` reports
them.

**Anything executable.** Chat templates come from the tokenizer files and run in a
sandboxed template engine with a step limit.

## Trust order

The files are untrusted input. When sources disagree, the one closer to the bytes wins:

1. tensor headers, checked against the real file sizes;
2. the checkpoint's config;
3. the manifest.

At load, every tensor's byte length must match its dtype and shape, every offset must
lie inside its file, and the set of tensors must be exactly what the architecture's
config requires. All problems are reported in one error, by name. A wrong manifest
fails loudly at load and never produces wrong numbers. Nothing is repaired.

Manifests are strict: unknown fields, absolute paths, `..`, and a schema or architecture
version this build does not implement are all errors.

## Versions and migration

| Schema | Adds |
|---|---|
| 1.0 | The base format |
| 1.1 | `source` and `files`, both optional |

A 1.0 manifest loads unchanged, and nothing requires migrating. `source` or `files` in
a manifest that says 1.0 is an error. A manifest from a newer schema is refused by
name.

```sh
qubivex models migrate <ID>            # print the upgraded manifest
qubivex models migrate <ID> --write    # replace manifest.json
```

`migrate --write` adds the source from the registry and the size and SHA-256 of every
file, reading each file once (`--no-checksums` skips the hashing). A checksum already
recorded is never replaced by migrating, so migration cannot hide a file that changed.

## Checksums

```sh
qubivex models verify <ID>
```

checks every file against the manifest's `sha256` values, then every expert against
`experts.qvxsum` (a per-expert checksum file beside the weights), recording that file if
there is none. The engine checks expert reads against `experts.qvxsum` when it matches
the weights.

After replacing a model's files on purpose, `qubivex models verify <ID> --record`
records both again from the files as they are. Nothing else accepts a changed file.

## Writing a manifest for a checkpoint

```sh
qubivex models import <DIR|FILE.gguf> --id my-model --write
```

The importer finds the weight files, then tries each architecture, tensor-name layout
and q/k-norm mode, and keeps the first combination under which every tensor validates.
If none does, it reports the problems of the closest candidate. It stops and names any
value the checkpoint does not record; supply it with `--set KEY=VALUE`.

## Not supported

- **Architectures other than `transformer_moe` and `transformer_dense`.** Shared-expert
  and partly dense MoE models, attention or FFN biases, non-SiLU activations, RoPE
  scaling and sliding-window attention are refused with a message that names the
  feature.
- **`router_scoring: "sigmoid"` at run time.** The manifest accepts it and the forward
  pass refuses it: only softmax routing is implemented.
- **I-quant GGUF dtypes (`IQ*`),** big-endian GGUF and GGUF v1.
- **A packed expert file** (`qubivex models pack`). Experts are read from the original
  files.
- **Loading sha256 verification.** Sizes are checked at every load; full hashes only by
  `models verify`.
