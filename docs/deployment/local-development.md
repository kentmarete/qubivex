# Local development

How to build QUBIVEX, run it against the bundled tiny models, and run the checks CI runs.

## Requirements

- Rust 1.88 or newer, installed with `rustup`. The workspace uses edition 2024.
- Linux (x86-64 or arm64) or macOS (arm64 or x86-64).
- A C compiler. `ring`, the TLS crypto library behind `qubivex models pull`, compiles C
  and assembly. On macOS the Command Line Tools provide one; on Debian or Ubuntu install
  `build-essential`.
- No Python and no GPU SDK. The Metal kernels are compiled by the system at run time.

## Build

```sh
git clone https://github.com/kentmarete/qubivex.git
cd qubivex
cargo build                 # debug, for development
cargo build --release       # optimised: fat LTO, so it takes several minutes
./target/release/qubivex --version
```

Use the release build for anything you time. A debug build of the kernels is many times
slower.

## Try it with no download

`fixtures/models/` holds four tiny generated models (under 400 KiB each). Their weights
are random, so the text they produce is noise, but every code path runs.

```sh
qubivex models list --dir fixtures/models
qubivex run --dir fixtures/models --model olmoe-tiny --prompt "Hello" --max-tokens 8
qubivex serve --dir fixtures/models --model olmoe-tiny
```

The server listens on `127.0.0.1:8080` and needs no key on loopback:

```sh
curl -s http://127.0.0.1:8080/health
```

```json
{"models":1,"status":"ok"}
```

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"olmoe-tiny","messages":[{"role":"user","content":"Hello"}],"max_tokens":6}'
```

Stop it with Ctrl-C. It lets running requests finish first.

## A real model

```sh
qubivex models pull Qwen/Qwen3-0.6B          # about 1.4 GiB, into ~/.qubivex/models
qubivex models analyse qwen3-0.6b            # will it fit on this machine?
qubivex run --model qwen3-0.6b --chat --prompt "Hello"
```

[Model downloads](../model-management/model-downloads.md) covers GGUF repositories,
private models and interrupted pulls.

## Checks

These are the gates CI runs. Run them before pushing.

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo xtask check          # dependency direction, embeddable runtime, model names, fixtures
cargo deny check           # advisories, licences, sources (needs cargo-deny)
```

`scripts/job verify` runs all of them, plus the Rust 1.88 build and the loom models. The
loom models alone:

```sh
RUSTFLAGS="--cfg loom" cargo test --release -p qubivex-expert-stream --test loom
RUSTFLAGS="--cfg loom" cargo test --release -p qubivex-scheduler --test loom
```

## Tests that need real models

They are skipped unless their variables are set, so CI stays offline. Set the variables
on the one command, not with `export`: under `cargo test --workspace` the real-model tests
would then run in a debug build and take many minutes.

| Variable | Used by |
|---|---|
| `QUBIVEX_OLMOE_GGUF_DIR` | The OLMoE Q4_K_M model directory: `real_model`, `metal_parity`, `ledger_rss` and others in `crates/qubivex-runtime/tests` |
| `QUBIVEX_OLMOE_TOKENIZER_JSON` | The official `tokenizer.json`, to compare with the tokenizer rebuilt from the GGUF |
| `QUBIVEX_DENSE_SAFETENSORS_DIR` | A dense SafeTensors model directory: `real_dense` |
| `QUBIVEX_REAL_MODEL_FILES`, `QUBIVEX_REAL_MODEL_DIRS` | `:`-separated files and directories for the container readers: `qubivex-model --test real_files` |

```sh
QUBIVEX_OLMOE_GGUF_DIR=~/.qubivex/models/olmoe-1b-7b-0125-instruct \
  cargo test --release -p qubivex-runtime --test real_model -- --nocapture
```

## Fixtures

`cargo xtask gen-fixtures` writes `fixtures/models/` deterministically, and
`cargo xtask check` fails if the committed files differ from the generator. After
changing the generator, regenerate and commit.

## Running the image locally

```sh
docker build -t qubivex .
docker run --rm -p 8080:8080 -e QUBIVEX_API_KEY=sk-local qubivex \
  serve --listen 0.0.0.0:8080 --dir /usr/share/qubivex/models --model olmoe-tiny
```

On a Mac, Docker runs a Linux virtual machine: the container uses the CPU backend and
never Metal. See [Linux deployment](linux-deployment.md#option-2-docker) for the full options.

## Where to go next

- [Workspace structure](../architecture/workspace-structure.md): what each crate does.
- [RUST_GUIDELINES.md](../development/RUST_GUIDELINES.md): the rules for contributions.
- [JOB_WORKFLOW.md](../development/JOB_WORKFLOW.md): how one piece of work runs from
  start to merge in this repository.
- [Benchmarking](../performance/BENCHMARKING.md): how to measure a change.
