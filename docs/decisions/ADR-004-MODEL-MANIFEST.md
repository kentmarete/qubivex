# ADR-004: Versioned JSON manifest, cross-validated, never trusted alone

- Status: Proposed (Phase 0)
- Date: 2026-11-01

## Decision
- `manifest.json` with `schema_version` (semver). A published JSON Schema lives in
  `docs/model-support/manifest.schema.json`.
- Sections: `model` (id, name, version), `architecture` (id, version), `weights`
  (format, shards with optional size and sha256), `tokenizer`, `context`,
  `capabilities`, `quantization`, and an optional `overrides` block for tensor-name
  patterns or config fields missing from `config.json`.
- **Dimensions are not duplicated by hand.** By default they are read from
  `config.json` or GGUF metadata. When a manifest states them, they are cross-checked
  and any mismatch is an error.
- Trust order: tensor headers (checked against actual file sizes) take precedence
  over config, which takes precedence over the manifest. Every byte size the runtime
  reads into a buffer is derived from config and checked for equality with the header
  (Colibri's untrusted-container rule).
- sha256 verification is optional at load (it is slow for hundreds of GB), mandatory
  in `qubivex models verify`, and cached with the file's mtime and size.
- `qubivex models import <dir|gguf>` generates a manifest from a checkpoint, so users
  rarely write one by hand.
- Manifests contain no executable content: no templates evaluated with side effects.
  Chat templates run in a sandboxed `minijinja` environment.

## Consequences
A wrong manifest fails loudly at load time instead of producing wrong numerics.
