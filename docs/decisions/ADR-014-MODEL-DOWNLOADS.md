# ADR-014: Model downloads and their threat review

- Status: Proposed
- Date: 2026-11-06

## Context
The platform spec (S2, acceptance criterion 1) requires Qubivex to download models itself.
`PLATFORM_AUDIT` section 5, decision 2 chose `ureq` with `rustls`, blocking, on dedicated
threads. This is the workspace's first outbound network and TLS code, and audit finding 4
asks for a threat review of redirects, path traversal and tokens.

## Decision
- **A new crate, `qubivex-model-download`.** It depends on the model and registry crates and
  nothing that executes a model. `qubivex-runtime` and `qubivex-api` gain no network or TLS
  dependency. It runs on plain threads, so an application can embed it.
- **`ModelProvider` is the source boundary.** Hugging Face is the first provider. What
  crosses the trait is normalised, so the registry and runtime see no provider formats.
- **The revision is pinned first.** A branch resolves to a commit, every file comes from that
  commit, and the commit goes in the registry. A branch moving between attempts discards the
  old files, through the registry's same-source rule, instead of mixing commits.
- **Selection is an allow-list.** SafeTensors shards named by the index, or one weight file,
  plus config, tokenizer and licence files. Several candidates are an error that lists them,
  because a guess can be a 40 GiB download of the wrong quantisation.
- **Each file arrives as `<name>.part`** and takes its name after its size and checksum pass.
  SHA-256 for LFS files, the git blob id for the others. Resuming re-hashes what is there, so
  the check covers the whole file, and one clean download is allowed if it fails.
- **One path for stalls, drops and slow links.** ureq's body timeout covers a whole body and
  cannot see a stall, so a request has a 120 s budget and the next continues from its offset.
  A request that moved the file forward is not a failure: it continues at once with fresh
  attempts, so a long download over a bad link ends only when the link stops moving.
- **Retries** double from 500 ms to 60 s with jitter and honour `Retry-After` up to that cap.
  Eight attempts without progress end the pull. 4xx statuses other than 408 and 429 are final.
- **Failures keep their files**, recorded as `download_failed`, including files that do not
  import: a pull that only lacked a `--set` value should not download 40 GiB again.

## Threat review
Everything a server returns is treated as untrusted. "Test" names the check in
`crates/qubivex-model-download`. Each rule was removed in turn to confirm a test then fails.

| Threat | Rule | Test |
|---|---|---|
| A redirect carries the token to another host, or the CDN | Redirects are followed by this crate. The token goes only to the configured origin (scheme, host and port), checked on every hop | `the_token_reaches_the_hub_and_never_another_host` |
| A pagination link sends the token elsewhere | `Link` targets must be on the configured origin | `a_listing_cannot_page_the_token_to_another_host` |
| A redirect downgrades to plain HTTP, or reaches a local service over it | From an https endpoint only https is followed. Plain http needs a loopback endpoint, and stays on loopback | `https_never_downgrades`, `plain_http_stays_on_this_machine` |
| A redirect to `file:`, `ftp:` or similar | Only http and https, with a limit of 8 hops | `redirects_to_other_schemes_are_refused_and_stay_out_of_errors`, `a_redirect_loop_ends` |
| Repository file names escape the model directory | Selected paths must be relative with normal parts, and are written under a flat, checked name. Hidden names, `.part` and `manifest.json` are refused. An index can only name files in the listing | `hostile_paths_are_refused`, `file_names_cannot_leave_the_model_directory`, `an_index_cannot_name_a_file_outside_the_listing` |
| A repository id or revision rewrites the URL | Ids are validated and revisions percent-encoded; a resolved revision must be 40 hex digits | `repository_ids`, `a_bad_repository_or_endpoint_sends_nothing` |
| A server fills the disk or memory | Bodies stop at the listed size; JSON is capped at 64 MiB; listings at 64 pages; free space is checked first | `a_body_longer_than_its_listing_stops_at_the_listed_size` |
| Corruption in transit or at rest | The published checksum, after every resume | `corrupt_bytes_are_downloaded_again`, `a_bad_partial_file_is_caught_by_the_final_check` |
| A reply for a range that was not asked for | The range must match, or the pull stops. A plain 200 restarts the file | `a_range_that_was_not_asked_for_is_refused`, `a_server_that_ignores_ranges_restarts_the_file` |
| A token leaks through logs or arguments | No command-line flag; `Token` prints as redacted; a token that cannot be a header value is refused, which stops header injection | `tokens_stay_out_of_debug_output` |
| A signed URL leaks through errors | Errors are built from status codes and error kinds, not client messages | `redirects_to_other_schemes_are_refused_and_stay_out_of_errors` |
| Malicious weights execute code | Pickle and PyTorch files are never selected. GGUF and SafeTensors are parsed by the hardened readers, and the model crate executes nothing | `safetensors_take_only_what_a_model_needs` |

## Limits
- Published checksums come from the same hub as the bytes. They catch corruption, not a
  compromised hub. TLS is the only protection against that.
- An https redirect to a private address is followed. The response only reaches a file that
  must match a checksum, and no token goes with it.
- Files with no published checksum rely on TLS alone. Every file of the real pull had one;
  that is one repository, not a guarantee.
- Roots are the bundled `webpki-roots`, so a TLS-inspecting proxy fails. The proxy variables
  ureq reads are honoured.
- `HF_ENDPOINT` decides where the token goes, as it does for the Hugging Face tools.
- Credentials are read, not stored: `HF_TOKEN`, or the file `hf auth login` writes.
- A cancel can wait up to one budget (120 s) on a stalled connection. A second Ctrl-C exits at
  once, which is safe, as a killed pull leaves what a cancelled one does.
- A model's architecture is checked only after it arrives, so an unsupported model costs a
  full download.
- Split GGUF files are refused: the importer reads one file.
- The free-space check does not reserve anything, so two pulls can both pass it.

## Consequences
- The binary gains `ureq`, `rustls` and `ring`. `ring` is the first dependency in a shipped
  path to compile C and assembly; it is the crypto provider ureq's `rustls` feature uses.
- `deny.toml` gains an exception scoped to `webpki-roots`, whose CA list is data under
  CDLA-Permissive-2.0.
- The source and revision are in the registry and, since plan job 4, the manifest too
  (ADR-004 amendment 1).
