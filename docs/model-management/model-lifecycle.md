# Model lifecycle

A model passes through two lifecycles: **installation** on disk, managed by the CLI, and
**loading** in a server process. They are separate on purpose. A file cannot say whether
a process is alive, so the disk records what is installed and each server records what
it has loaded.

```text
   on disk (qubivex models ...)                 in a server (qubivex serve)

   not installed
        │ pull / import
        ▼
   downloading ──► download_failed ─┐
        │                 ▲ retry ──┘
        ▼
   validating ───► validation_failed
        │
        ▼                               load
   installed ───────────────────────────────────►  loading ──► failed
        │                                             │
        │ remove                                      ▼
        ▼                                          loaded
   not installed                                      │ unload, or shutdown
                                                      ▼
                                                   draining ──► unloaded
```

## Installation states

Recorded in `registry.json` ([Model registry](model-registry.md)).

| State | Meaning | What you can do |
|---|---|---|
| Not installed | No directory, no record | `models pull`, or copy files in and `models import --write` |
| `downloading` | Files are arriving in `.staging/<id>` | Wait. If the pull was interrupted, run the same `pull` again: it continues |
| `download_failed` | The download ended without finishing. The files are kept | Run the same `pull` again. From a different source or revision, the old files are discarded first |
| `validating` | Every file arrived and the model is being checked | Wait |
| `validation_failed` | The files arrived and are not a valid, supported model. The reason is recorded | Fix the cause (often a missing `--set`) and pull again, or `models remove <ID> --yes` |
| `installed` | In place and valid | Use it |

A model is never visible to `models list`, `run` or `serve` until it is `installed`.

## Installing

### By pulling

```sh
qubivex models pull Qwen/Qwen3-0.6B
```

Download, verify, validate and install in one step. See
[Model downloads](model-downloads.md).

### By importing files you already have

```sh
mkdir -p ~/.qubivex/models/my-model
cp /somewhere/{config.json,tokenizer.json,model.safetensors} ~/.qubivex/models/my-model/
qubivex models import ~/.qubivex/models/my-model --id my-model --write
```

or, for a GGUF file:

```sh
qubivex models import ~/.qubivex/models/my-model/model.gguf --id my-model --write
```

`import --write` writes `manifest.json` beside the weights and records each file's
checksum and the per-expert checksums (`--no-checksums` skips that). The model is then
adopted into the registry at the next `pull` or `remove`, and works without that.

## Checking an installed model

| Command | Answers |
|---|---|
| `qubivex models list` | What is installed, and whether each is valid. A broken model is listed with its error, not skipped |
| `qubivex models inspect <ID>` | Dimensions, bytes by role, files, dtypes |
| `qubivex models analyse <ID>` | Whether it can run on this machine, and what it needs |
| `qubivex plan <ID>` | How it would be placed in memory |
| `qubivex models verify <ID>` | Whether every file and every expert matches its recorded checksum |

`analyse` on a real model:

```text
Model    qwen3-0.6b (Qwen3-0.6B)
Source   local
Verdict  runs here, everything in RAM, with the context shortened from 40960 to 6438

Needs
  on disk                          1.40 GiB
  RAM to start                     1.61 GiB
  RAM at context 40960            10.36 GiB

This machine
  RAM total                       18.00 GiB
  RAM available now                3.24 GiB
  budget (performance profile)     2.99 GiB

Backends
  cpu: available (the default)
  cuda: unavailable: CUDA is not supported on macOS
  vulkan: unavailable: vulkaninfo not found
  metal (Apple M3 Pro): matmuls, with --backend metal
```

The verdict is one of: everything in RAM, runs while reading some experts from disk, or
does not fit. "Does not fit" is an answer, so the command still exits 0; scripts should
read `assessment.verdict` from `--json`.

## Loading states

Kept by each server, per model, and shown in `GET /v1/system`:

```json
"model_states": [{"id": "qwen3-0.6b", "state": "loaded"}]
```

| State | Meaning |
|---|---|
| `loading` | The dense weights are being read and the memory plan made. Requests naming the model wait for it |
| `loaded` | Serving |
| `failed` | The load failed. The reason was returned to the caller and logged as `model_load_failed` |
| draining | Unloaded, with generations still finishing. Counted in `models_draining` |

### What loading does

1. Reads and validates the manifest, the config and every tensor header.
2. Reads the dense weights (embedding, attention, norms, routers, head) into memory.
   Routed experts are **not** read: they are loaded on first use.
3. Measures the RAM available with the dense weights resident, and makes the memory
   plan: the context, the expert cache and its per-layer slots.
4. Builds the tokenizer and the chat template.
5. Starts the model's I/O threads, pressure monitor, scheduler and queue.

A model that does not fit fails here, with the planner's reason. Load time is mostly
step 2, so it grows with the dense bytes and not with the expert bytes: OLMoE-1B-7B has
about 300 MiB dense out of 4 GiB.

### Loading and unloading

| How | When |
|---|---|
| `qubivex serve --model <ID>` (repeatable), or `models = [...]` in the config | At start. The port opens after these have loaded |
| `POST /v1/models/load {"model": "<ID>"}` | While running, with the admin token |
| `POST /v1/models/unload {"model": "<ID>"}` | While running, with the admin token |
| Shutdown | Unloads everything |

Rules:

- Concurrent loads of one model join one load.
- A load belongs to the server. If the client that asked disconnects, the load
  continues.
- One directory is one loaded model. A second id for the same directory is a 409.
- Unloading closes the model's queue (waiters get a 503) and lets running generations
  finish. The memory is freed when the last one ends.
- Loaded models are not remembered across a restart, except those in the config.

## Updating a model

There is no in-place update. To move to a new revision without downtime for the old
one:

```sh
qubivex models pull Qwen/Qwen3-0.6B --revision <new> --id qwen3-0.6b-v2
curl -s localhost:8080/v1/models/load   -H "Authorization: Bearer $ADMIN" -H 'content-type: application/json' -d '{"model":"qwen3-0.6b-v2"}'
# move clients to the new id, then:
curl -s localhost:8080/v1/models/unload -H "Authorization: Bearer $ADMIN" -H 'content-type: application/json' -d '{"model":"qwen3-0.6b"}'
qubivex models remove qwen3-0.6b --yes
```

Both models are in memory between the load and the unload, so check that both fit.

Never change the files of a loaded model. The server reads experts from them while it
runs, and a file that changes underneath it gives failed reads (with checksums) or
wrong numbers (without).

## Removing

```sh
qubivex models remove <ID>          # dry run
qubivex models remove <ID> --yes
```

Unload the model from every server first: `remove` does not check.

## Not supported

- **Automatic loading on first request.** A request for a model that is not loaded is a
  404. Load it at start or through the admin endpoint.
- **Automatic unloading of idle models,** and a limit on how many are loaded.
- **In-place update and rollback of a model.** Use a second id, as above.
- **Reference counting between servers.** Two servers can load the same directory;
  each has its own copy in memory, and neither knows about the other.
- **A loaded state that survives a restart.**
