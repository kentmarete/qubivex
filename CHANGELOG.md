# Changelog

Notable changes to QUBIVEX, newest first. Versions follow the workspace `version` in
`Cargo.toml`, which `qubivex --version` prints. There are no tagged releases yet:
`qubivex update` installs the newest commit of the branch you follow.

## Unreleased

### Added

- **Context management** (ADR-021, Proposed). A request longer than the model's context
  is prepared to fit instead of refused with a 400: system and developer messages are
  kept verbatim, recent turns are kept, older turns become a note of quoted excerpts
  (never model-written), and a long pasted document or source file is cut to its most
  relevant blocks with markers naming what was left out. Every change is recounted with
  the model's own template and tokenizer. Requests that fit are untouched. Prepared
  responses carry `x-qubivex-context` headers; `POST /v1/context/inspect` shows what
  would be done; new `qubivex_context_*` metrics. Notes are reused across turns (in
  memory, or in files with `--context-store`). `--no-context-management` restores the
  old refusal. New crate `qubivex-context`.

## 0.1.2 (2026-11-22)

### Added

- **gpt-oss SafeTensors** (#84). `openai/gpt-oss-20b` checkpoints run on the shared MoE
  decoder as configuration of `transformer_moe` 1.2.0: q/k/v/o biases, attention sinks,
  per-layer sliding windows, YaRN RoPE, router and per-expert biases, a softmax over the
  top-K logits, interleaved gate/up rows and the clamped SwiGLU. Hugging Face MXFP4
  `*_blocks`/`*_scales` are repacked into the GGML layout as each expert loads, with
  values unchanged. Chat templates are read from `chat_template.jinja`. The fused device
  paths decline these models; the host runs the new steps and the device the matmuls,
  with the CPU's bits on every backend. Real gpt-oss-20b layers match transformers f32
  (worst 2.83e-6), and the greedy chat reply is the same on CPU, Metal, Vulkan and Sim.
  ADR-003 amendment 5 (Proposed).
- **MXFP4 execution format** (#83). GGUF type 39 (32 E2M1 values with an E8M0 scale per
  17-byte block) beside the K-quants, read packed on the CPU (fused NEON kernels), Metal
  and Vulkan. Never expanded to an f32 copy. Subnormal scales are formed from two
  normal factors so GPU flushing only affects weights that are themselves subnormal.
  Bit for bit with gguf-py over all 256 scale bytes. ADR-007 amendment 2 (Proposed).
- **Update notice** (#86). After a command in a terminal, a notice on stderr when the
  branch the installation follows has a newer workspace version, suggesting
  `qubivex update`. Looked up at most daily in the background (3 s timeout, cached in
  `~/.qubivex/update-check.json`). Never shown with piped output, `--json`,
  `--version`, after `run`, `serve`, `bench` or `update`, after an error, in CI, or with
  `QUBIVEX_NO_UPDATE_CHECK=1`.
- **CLI command reference** (#85) at `docs/cli/COMMAND_REFERENCE.md`, linked from the
  README and the docs index.

### Changed

- The architectures listing and the metadata refusal name `mxfp4` among the weight
  formats `transformer_moe` runs (#83).

## 0.1.1 (2026-11-19)

### Added

- **`qubivex update`** (#81). Rebuilds from the branch recorded in
  `~/.qubivex/install-receipt` when it has new commits, through that commit's
  `install.sh --update`. `--check`, `--branch`, `--jobs`, `--no-test`; `--detach` runs
  it in the background (log `~/.qubivex/update.log`) and `--status` reports on it.
  Binaries older than 0.1.1 need one `install.sh --update` first.
- **`qubivex serve --detach | --status | --stop`** (#81). Starts the server in the
  background and returns once it accepts connections; one server per port, log in
  `~/.qubivex/serve-<port>.log`. `--stop` drains as Ctrl-C does.
- **Llama 3.x** (#80). `rope_type: "llama3"` (banded) RoPE scaling is parsed, checked and
  applied on every backend, operation for operation as transformers in f32.
  Llama-3.2-3B bf16 matches transformers in tokenizer ids, 32 greedy tokens and logits
  (6.79e-6). `transformer_dense` 1.2.0, `transformer_moe` 1.1.0; the higher version is
  recorded only for models that use RoPE scaling, so older builds refuse them.

### Changed

- **Plain RoPE frequencies** (#80) are now the correctly rounded f32 power (f64 `pow`
  rounded once) instead of f32 `powf`, which was one ulp off on macOS at θ = 500 000.
  This affects every model by at most one ulp in a frequency table entry.
- `rope_scaling` and `rope_parameters` must agree, and so must `rope_type` and `type`
  (#80). A config pairing `rope_type: "default"` with `type: "mrope"` is now refused.

### Fixed

- `cargo xtask check` passes again: a test in `qubivex-api` no longer uses a model family
  name as a sample directory (#82).
