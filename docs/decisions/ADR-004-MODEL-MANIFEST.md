# ADR-004: Versioned JSON manifest, cross-validated, never trusted alone

- Status: Proposed (Phase 0). Amendment 1 (2026-11-07): schema 1.1, with source, checksums
  and a migration path.
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

## Amendment 1 (2026-11-07): schema 1.1
Plan job 4 (`docs/engineering/PLATFORM_AUDIT.md`).

- **Added, both optional:** `source` (provider, repository, exact revision) and `files`
  (path, size and sha256 of the config, tokenizer and other files beside the weights). The
  runtime reads neither, so it stays free of provider metadata. `source` uses the registry's
  type, which moved here, and adoption now keeps a source the manifest states.
- **Version rules.** The version is read first, so a newer manifest is refused by name, not
  as an unknown field. A build reads its own minor and older ones, as for `architecture`. A
  minor adds optional fields; 1.0 manifests load unchanged, and `source` or `files` in one
  is an error.
- **Migration.** `Manifest::upgraded` moves a manifest forward. `qubivex models migrate`
  adds the source from the registry and hashes the files, and prints the result unless
  given `--write`. Manifests are written by temporary file and rename. The fixtures stay at
  1.0 as the migration test's input.
- **A recorded checksum is never replaced by filling in.** Migrating a model whose file
  changed leaves the check failing. Only `models verify --record` accepts a change, and it
  rewrites both the manifest's checksums and `experts.qvxsum`.
- **`models verify` now checks `sha256`.** This ADR required it, but nothing did. A pull
  reuses the provider's SHA-256, which the transfer already checked, and hashes only files
  with a git blob id, so large files are not read twice. `models import --write` reads every
  file unless given `--no-checksums`.

Not adopted:

- **Dimensions and expert fields from spec §5.3.** The trust order above stands: they come
  from the checkpoint, and `models inspect` and `analyse` show them. A second copy could only
  disagree. The same goes for tensor-to-file mappings, which are in the headers. Supported
  backends depend on the machine and the budget, so `analyse` reports them.
- **Caching sha256 by mtime and size.** Not built. `models verify` reads every file, at
  about 500 MB/s here (a 1.4 GiB model in 3.1 s), one file at a time.
