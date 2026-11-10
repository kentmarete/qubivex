# Model registry

QUBIVEX has two things called a registry. This page covers both, because they answer
different questions.

| Registry | Question | Where |
|---|---|---|
| Installed-model registry | Which models are installed here, where did they come from, and did their install finish? | `registry.json` in each models directory; crate `qubivex-model-registry` |
| Architecture registry | Which model architectures can this build run? | In the binary; `ArchitectureRegistry` in `qubivex-model` |

The design record for both is
[ADR-003](../decisions/ADR-003-MODEL-REGISTRY.md).

## The models directory

```text
~/.qubivex/models/
├── registry.json              installed models and their sources
├── .registry.lock             advisory lock for changes to registry.json
├── .staging/<id>/             a model being downloaded or validated
├── .trash/                    a model being removed
├── qwen3-0.6b/
│   ├── manifest.json
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
└── olmoe-1b-7b-0125-instruct/
    ├── manifest.json
    ├── OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf
    └── experts.qvxsum         per-expert checksums, from `models verify`
```

Model directories are searched in this order: `--dir` (repeatable), then
`QUBIVEX_MODELS_DIRS` (`:`-separated), then `~/.qubivex/models`. A model is a directory
holding `manifest.json`: the models directory itself, or a directory directly inside it.

**A model directory is the unit of installation.** The runtime and `models list` need
no registry: they read the directories. So a model copied in by hand works, and a models
directory with no `registry.json` is valid. The registry adds what a directory cannot
say: the source, and whether an install is still in progress.

## `registry.json`

One record per model:

```json
{
  "schema_version": 1,
  "models": {
    "qwen3-0.6b": {
      "id": "qwen3-0.6b",
      "dir": "qwen3-0.6b",
      "state": { "kind": "installed" },
      "source": {
        "provider": "huggingface",
        "repository": "Qwen/Qwen3-0.6B",
        "revision": "<the 40-digit commit>"
      },
      "updated_at": 1794294609
    }
  }
}
```

| Field | Meaning |
|---|---|
| `id` | The model id |
| `dir` | Its directory name under the models directory. It must be one path component; a file that says otherwise fails to load |
| `state.kind` | `downloading`, `validating`, `installed`, `download_failed` or `validation_failed` (the last two carry a `reason`) |
| `source` | Provider, repository and exact revision. `null` for a model adopted from disk whose manifest names no source |
| `updated_at` | When the state last changed, in seconds since the Unix epoch |

The file is not meant to be edited. `qubivex models pull` and `qubivex models remove`
write it, and `qubivex models migrate` reads the source from it.

Whether a model is **loaded** is not recorded. A file cannot say whether a process is
still alive, so that state belongs to the server: see `/v1/system`.

## Crash safety

- **Installs are atomic.** A model is built in `.staging/<id>`, validated there, then
  renamed into place. A model directory is complete or absent.
- **`registry.json` changes atomically:** a temporary file, an fsync and a rename, under
  a lock.
- **Locks are `flock`s,** which the kernel releases when a process dies. A crash never
  leaves a stale lock.
- **Reconciliation.** The next `pull` or `remove` brings the file in line with the
  directory: it finishes installs a crash interrupted after the rename, adopts valid
  model directories that have no record, drops records whose directories have gone, and
  empties `.trash`. An install another process is running is left alone.
- **One install per model.** A second `pull` of a model that is being pulled is refused.

## Removal safety

```sh
qubivex models remove qwen3-0.6b          # says what would be deleted
qubivex models remove qwen3-0.6b --yes    # deletes it
```

```text
qubivex: this would delete fixtures/models/olmoe-tiny (336.52 KiB); pass --yes to delete it
```

Removal deletes only what a record names, and only one directory directly under the
models directory. It refuses symbolic links and a model whose install is running. The
directory is renamed into `.trash` before the record goes, so a crash cannot leave a
half-deleted model that still looks installed.

It does not check whether a running server has the model loaded. Unload it first.

## The architecture registry

An architecture is a registered component, and a model names one in its manifest:

```sh
qubivex models architectures
```

lists what this build supports: `transformer_moe` and `transformer_dense`. Loading a
model looks up `architecture.id` and checks its version.

An architecture provides three things: how to parse its config, the **tensor schema**
(every tensor it expects, with its shape, dtype and role), and how to build the forward
pass. Validation and the memory planner work only on roles (embedding, attention,
router, routed expert, dense FFN, norm, head), never on tensor names or model families.
That is why a new checkpoint of a supported architecture loads with a manifest and no
code change, and why the planner needs no knowledge of any model.

Variation inside an architecture is configuration: attention with one or grouped KV
heads, q/k normalisation, top-K renormalisation, tied output head, tensor-name layouts.
A genuinely new mechanism is a new architecture registered beside the others; the
runtime does not change.

`cargo xtask check` enforces the other half of this: no model-family name may appear in
shared code.

## Not supported

- **A registry shared between machines, or a remote registry.** It is a file in a
  directory.
- **Several revisions of one model under one id.** Install a second revision with a
  different `--id`.
- **Garbage collection of unused models** and disk quotas.
- **Dynamically loaded architectures (plugins).** Architectures are compiled in.
- **`models update`.** To move to a newer revision, remove the model and pull it again,
  or pull under a new id.
