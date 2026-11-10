# Model downloads

`qubivex models pull` downloads a model from Hugging Face and installs it, with no other
tool. This page covers how to use it and what it guarantees. The design record and the
threat review are [ADR-014](../decisions/ADR-014-MODEL-DOWNLOADS.md).

## Find and pull

```sh
qubivex models search qwen3 --limit 3
```

```text
REPOSITORY                  DOWNLOADS   LIKES  FORMATS
Qwen/Qwen3-0.6B              20728069    1888  safetensors
Qwen/Qwen3-14B               12904408     536  safetensors
Qwen/Qwen3-Embedding-0.6B     9129136    1339  safetensors
```

Search lists repositories. It does not say whether QUBIVEX supports a model's
architecture: that is checked when the model is installed.

```sh
qubivex models pull Qwen/Qwen3-0.6B
qubivex models list
qubivex run --model qwen3-0.6b --chat --prompt "Hello"
```

The model id defaults to the repository's name in lower case. It installs into the
first directory of `QUBIVEX_MODELS_DIRS`, or `~/.qubivex/models`, or `--dir`.

### GGUF repositories

A GGUF repository usually holds one file per quantisation. Pulling it without choosing
is an error that lists the files, because a guess could be a 40 GiB download of the
wrong one. Name the file:

```sh
qubivex models pull allenai/OLMoE-1B-7B-0125-Instruct-GGUF \
  --file OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf \
  --id olmoe-1b-7b-0125-instruct \
  --set norm_topk_prob=false
```

`--set KEY=VALUE` supplies a config value the checkpoint does not record. GGUF files do
not always say whether the top-K expert weights are renormalised, so QUBIVEX stops and
names the missing value. Take it from the model's original `config.json` (OLMoE:
`norm_topk_prob=false`; Qwen3-MoE: `true`). If a pull fails only for a missing `--set`,
the files are kept: run the command again with the value and nothing is downloaded
twice.

### Options

| Option | Meaning |
|---|---|
| `--file PATH` | A file to fetch, by its path in the repository (repeatable) |
| `--revision REV` | A branch, tag or commit. The exact commit is recorded either way |
| `--id ID`, `--name NAME` | The model id and display name |
| `--set KEY=VALUE` | A config value the checkpoint lacks (repeatable) |
| `--dir DIR` | The models directory to install into |
| `--concurrency N` | Files fetched at once (default 4) |

### Private and gated models

The token comes from `HF_TOKEN`, or from the file `hf auth login` saves (`HF_TOKEN_PATH`
or under `HF_HOME`). There is no token flag, because command-line arguments show up in
process listings. `HF_ENDPOINT` selects a mirror; the token is sent to that endpoint and
nowhere else.

```sh
HF_TOKEN=hf_... qubivex models pull owner/private-model
```

## What gets downloaded

Only what a model needs, chosen by an allow-list:

- the SafeTensors shards the index names, or the one weight file;
- `config.json`, the tokenizer files and the licence.

Pickle and PyTorch files (`.bin`, `.pt`, `.pth`) are never selected. QUBIVEX cannot
execute code from a model file: SafeTensors and GGUF are parsed by its own readers.

## What it guarantees

- **Pinned revision.** A branch or tag is resolved to a commit first, and every file
  comes from that commit. The commit is recorded in the registry and the manifest.
- **Verified.** Each file is checked against the checksum the hub publishes: SHA-256
  for large (LFS) files, the git blob id for the rest. A resumed file is re-hashed from
  the start, so the check covers the whole file.
- **Atomic.** A file arrives as `<name>.part` and takes its name only after its check.
  The model is built in a staging directory and appears in the models directory only
  when complete and valid. `qubivex models list` never shows a half-installed model.
- **Resumable.** If a pull is interrupted, by Ctrl-C, a crash or the network, the same
  command continues from the bytes already there.
- **Bounded.** Free space is checked before downloading, and a body never grows past
  the size the listing gave.

## Failures

| Situation | What happens |
|---|---|
| The connection stalls or drops | The request ends after 120 s without finishing, and the next continues from its offset. Retries back off from 500 ms to 60 s and honour `Retry-After` |
| No progress in eight attempts | The pull ends. The files stay: run it again later |
| A 4xx other than 408 or 429 | Final: a wrong name, no access, or a gated model without a token |
| A checksum does not match | The file is downloaded again, once |
| The server ignores a range request | That file restarts from the beginning |
| The model is not a supported architecture | The install fails validation after the download, and says why |
| Ctrl-C | Stops cleanly and keeps the files. A second Ctrl-C exits at once, which is also safe |
| Not enough disk space | Refused before downloading |

To abandon a partial pull and delete its files:

```sh
qubivex models remove <ID> --yes
```

## Without `pull`

A model fetched some other way works the same once it has a manifest:

```sh
mkdir -p ~/.qubivex/models/my-model
# copy config.json, *.safetensors and tokenizer.json, or one .gguf, into it
qubivex models import ~/.qubivex/models/my-model --write
```

See [Model lifecycle](model-lifecycle.md).

## Not supported

- **Providers other than Hugging Face.** The provider is a trait
  (`ModelProvider`), and Hugging Face is the only implementation.
- **Split GGUF files** (`-00001-of-00003.gguf`). The importer reads one file.
- **TLS-inspecting proxies.** The trusted roots are compiled in, so a proxy that
  re-signs TLS fails. Ordinary HTTP proxy variables are honoured.
- **Checking the architecture before downloading.** An unsupported model costs a full
  download before it is refused.
- **Reserving disk space.** Two pulls at once can both pass the free-space check.
- **Pulling through the server.** `pull` is a CLI command; the HTTP API loads only
  models already installed.
- **Protection against a compromised hub.** The checksums come from the same hub as
  the files. They catch corruption, not a hub serving a different model.
