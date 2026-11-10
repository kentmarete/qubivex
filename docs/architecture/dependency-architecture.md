# Dependency architecture

Which crate may depend on which, what third-party code QUBIVEX ships, and how both are
kept in check. The rules here are executable: `cargo xtask check` and `cargo deny check`
run in CI and fail the build when one is broken.

## Internal dependencies

The actual edges, from `cargo metadata` (normal dependencies only):

```text
qubivex-cli ──► api, runtime, model-download, model-registry, model, memory,
                expert-stream, tokenizer, hardware, core

qubivex-api ──► runtime, memory, model, tokenizer, hardware

qubivex-runtime ──► arch, memory, expert-stream, kv-cache, scheduler, tokenizer,
                    model, hardware, compute, core, backend-cpu, backend-metal

qubivex-arch ──────────► model, compute, core
qubivex-memory ────────► model, hardware, core
qubivex-expert-stream ─► compute, core
qubivex-backend-cpu ───► compute, core
qubivex-backend-metal ─► compute, core
qubivex-compute ───────► core

qubivex-model-download ► model-registry, model
qubivex-model-registry ► model
qubivex-model ─────────► core
qubivex-tokenizer ─────► core
qubivex-hardware ──────► core

qubivex-scheduler      (no dependencies)
qubivex-kv-cache       (no dependencies)
qubivex-core           (no workspace dependencies)
```

Every edge points from a higher layer to a lower one. There are no cycles, and Cargo
would refuse one.

## The rules

Each workspace crate has a list of crates it may depend on, in `xtask/src/deps.rs`. A
crate that is not in the list fails the check, so adding a crate forces a decision about
its edges.

| Rule | Why |
|---|---|
| **`qubivex-model` never depends on compute or a backend** | Describing and validating a model must not need the code that runs it. The model manager can then ship without the engine |
| **The model registry and downloader never depend on anything that executes a model** | Code that touches the network and untrusted files stays apart from the engine |
| **Backends depend only on `compute` and `core`** | A backend cannot know about models, the runtime or the API, so a new one is a drop-in |
| **`qubivex-expert-stream` and `qubivex-memory` never depend on `qubivex-arch`** | The cache and the planner work on tensor roles and bytes, not on an architecture |
| **`qubivex-scheduler` and `qubivex-kv-cache` depend on nothing** | Their rules can be tested and model-checked without a model, and reused by an embedding application |
| **Nothing depends on an application** (`apps/*`) or on `xtask` | Logic cannot hide in the binary |
| **The runtime is embeddable** | See below |

Dev-dependencies have three narrow exceptions, each written in the rules with its
reason: the gateway's tests use the expert store's fault injection, and a GPU backend's
parity tests use the CPU backend and the forward pass as their reference.

### The embeddable runtime

`qubivex-runtime` must work inside another Rust program with no HTTP server and no CLI.
The check runs `cargo tree` on the runtime and fails if any of these appears anywhere
in its dependency tree, direct or transitive:

`axum`, `clap`, `hyper`, `kerniq-server`, `tokio`, `tower`, `qubivex-api`, `qubivex-cli`

A compiled example (`crates/qubivex-runtime/examples/embed.rs`) uses only the runtime
and the model crate, so the claim is also built on every CI run.

### No model names in shared code

`cargo xtask check-model-names` scans `crates/` and `apps/` and rejects model-family
names outside tests. A model is described by its manifest and an architecture by its
registration, so supporting a new checkpoint of a known architecture needs no code.

### Running the checks

```sh
cargo xtask check            # all of the below
cargo xtask check-deps       # dependency direction and the embeddable runtime
cargo xtask check-model-names
cargo xtask check-fixtures   # committed fixtures equal the generator's output
```

## Third-party dependencies

Versions are declared once, in the root `Cargo.toml`, with a comment where the choice
needs one. Exact versions are in the committed `Cargo.lock`, and CI builds with
`--locked`.

| Crate | Used by | For |
|---|---|---|
| `serde`, `serde_json` | Most crates | Manifests, configs, API bodies, JSON output |
| `thiserror` | Most crates | Typed errors |
| `rayon` | arch, backend-cpu, runtime | The compute thread pool |
| `rustix` | hardware, registry, download, runtime | Safe system calls: `statvfs`, `flock`, `O_DIRECT`, `F_NOCACHE`, `fadvise` |
| `tokenizers` | tokenizer | Hugging Face tokenizers, pure-Rust build (no Oniguruma, no C++) |
| `minijinja`, `minijinja-contrib` | tokenizer | Chat templates, sandboxed with a step limit |
| `sha2`, `sha1` | model, download | File checksums and git blob ids |
| `ureq` (with `rustls`) | download | Blocking HTTPS. Brings `ring` and `webpki-roots` |
| `objc2`, `objc2-metal`, `objc2-foundation` | backend-metal | Metal, on macOS only |
| `kerniq-server` | api | The HTTP server: a thin layer over Axum, Tower and Hyper. Pinned to an exact version while the crate is new |
| `tokio`, `tokio-stream` | api, cli | The async runtime, at the gateway only |
| `clap` | cli | Argument parsing |
| `toml` | cli | The serve config file |
| `loom` | expert-stream, scheduler | Model checking, under `cfg(loom)` only |
| `criterion`, `proptest` | dev only | Kernel benchmarks and property tests |

What is deliberately absent:

- **A tensor library** (Candle, Burn, ndarray, tch). QUBIVEX has its own kernels
  ([ADR-002](../decisions/ADR-002-HYBRID-COMPUTE.md)).
- **An async runtime below the gateway.** Inference and expert I/O run on dedicated
  threads.
- **Python, at build time or run time.**
- **C or C++ code,** with one exception: `ring`, the crypto library behind TLS for
  downloads, compiles C and assembly. It is why a build needs a C compiler.
- **Logging and metrics frameworks.** The gateway writes its own log lines and
  Prometheus text.

### Policy checks (`cargo deny`)

`deny.toml` sets, and CI enforces:

| Check | Rule |
|---|---|
| Advisories | No known vulnerabilities; yanked crates denied. One documented exception: an unmaintained compile-time macro pulled in by `tokenizers` |
| Licences | An allow-list: Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, ISC, Zlib, Unicode-3.0. One scoped exception for the CA list in `webpki-roots` |
| Sources | crates.io only. No git dependencies, no other registries |
| Bans | No wildcard version requirements |

### Adding a dependency

1. Check that it is needed: the engineering rules ask for the fewest dependencies.
2. Declare it once in the root `Cargo.toml` at the oldest version that works, with a
   comment saying which crate uses it and why, and turn off default features that are
   not needed.
3. Run `cargo xtask check` and `cargo deny check`. A dependency that would put Tokio or
   an HTTP stack under the runtime fails the first; a licence outside the allow-list
   fails the second.
4. A dependency that changes the architecture (a new backend, a new I/O model) needs an
   ADR.

## Build features

There are no optional Cargo features for backends: the CPU backend is always built, and
the Metal backend compiles on every platform and does nothing outside macOS. A binary's
capabilities are decided by the target it was built for and the hardware it runs on,
and `qubivex system` reports them.

## Not supported

- **A crates.io release.** Every crate is `publish = false`. Embedding the runtime
  today means a git or path dependency.
- **A stable API.** The public Rust interfaces change with the ADRs. Versions are all
  0.1.0.
- **A C API, Python bindings or WebAssembly.**
- **Dynamically loaded backends or architectures.**
