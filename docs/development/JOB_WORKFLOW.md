# Job Workflow (macOS)

How to run one piece of QUBIVEX work from start to finish, then leave the Mac and Claude
clean for the next one. All commands go through `scripts/job`.

To type `job` instead of `./scripts/job`, add this to `~/.zshrc`:

```zsh
alias job="$HOME/fin/qubivex/scripts/job"
```

A job is one reviewable outcome on one branch, such as "add the disk throttle". If it
needs two PRs, it is two jobs.

## The cycle

| Step | You run | What it does |
|---|---|---|
| 1. Check | `job status` | Shows the open job, git, running processes, RAM, disk and notes, then a verdict. Changes nothing. |
| 2. Start | `job start <slug> "<goal>" "<done when>"` | Needs a clean tree. Creates or switches to the branch and opens the job card. |
| 3. Brief Claude | Paste the start prompt below | Starts the session from the card, not from memory of the last chat. |
| 4. Work | `job save "<where things stand>"` | Adds a checkpoint. Run it after each milestone and before `/compact`. |
| 5. Commit | `job commit "<message>" [paths...]` | Commits on the job branch and adds a checkpoint. Repeat as often as needed. |
| 6. Verify | `job verify` | Runs the CI gates locally and stops at the first failure. On a clean tree, records that this commit passed. |
| 7. Raise a PR | `job pr` | Pushes the branch and opens its PR, or updates the open one. Needs a verified commit. |
| 8. Merge, when wanted | `job merge` | Merges once CI passes and you confirm. Skip it to leave the PR for review. |
| 9. Close | Paste the close prompt, then `job finish "<outcome>"` | Needs a clean tree and a merged PR, unless you pass `--unmerged`. Logs the job and PR, then removes the card. |
| 10. Clear | `job clean`, then `/clear` in Claude Code | Frees disk and empties the conversation. |
| 11. Confirm | `job status` | Should say `clean, ready for the next job`. |

`job start` branches from the current commit. Use `--from main` when the job doesn't build
on the current branch. To adopt work already in progress, give the current branch's name as
the slug; uncommitted changes stay as they are.

## Commits, PRs and merges

**`job commit`** commits all changes, or only the paths you list. With no paths it refuses
to run while a build or benchmark is running, because another session's edits could be in
the same tree. It refuses to commit on `main`, and refuses secrets, model files and
attribution lines.

**`job pr`** options:

| Option | Use |
|---|---|
| `"<title>"` | Sets the title. The default is the latest commit's subject. |
| `--base <branch>` | Stacks the PR on a branch already on GitHub. The default is `main`. |
| `--draft` | Opens a draft PR. |
| `--body-file <path>` | Uses your description. The default is built from the job card and commits. |
| `--skip-verify` | Skips the `job verify` requirement, for docs-only changes. |

Run `job pr` again after new commits to update the open PR. It never force-pushes.

**`job merge`** uses a merge commit, as the repo's history does, and keeps the branch. It
stops unless the PR is open, not a draft, mergeable, matches your local commit, and has
passing checks. It then asks you to type `merge`, and updates local `main` afterwards.

`job merge` and `job finish` check that the PR on the card belongs to the job's branch.
`job finish` asks GitHub whether the PR merged, so a merge made outside `job merge` counts.
To close a job and leave its PR for review, use `job finish "<outcome>" --unmerged`.

Claude may pass `job merge --yes` only when you ask for that merge in the same request. In
a stack, merge from the bottom up, then retarget the next PR to `main`.

After merging, start the next job with `--from main`.

## Prompts

**Start of a job, after `/clear`:**

```text
Read the current-job memory and the "Current phase" section of IMPLEMENTATION_STATUS.md.
Tell me the goal, the done-when and your first step, then begin.
```

**Resuming a job, after `/clear` or a new session:**

```text
Read the current-job memory. Run `git status` and check the last checkpoint against the
code. Tell me what is done and what is next, then continue.
```

**Closing a job:**

```text
Close this job. Update IMPLEMENTATION_STATUS.md with the result, measured numbers and
next step. Add an ADR amendment if a decision changed. Update memory with only what the
repo doesn't record: gotchas, my decisions, and environment facts. Commit with `job commit`,
run `job verify`, and raise the PR with `job pr`. Don't merge unless I ask. Then give me a
one-line outcome for `job finish`.
```

## Where things live

Each fact has one home. Don't copy progress into memory.

| What | Home | Lifetime |
|---|---|---|
| Open job: goal, done-when, checkpoints | `current-job.md` in Claude memory | Until `job finish` |
| Finished jobs, one line each | `job-log.md` in Claude memory | Permanent |
| Project progress and next steps | `IMPLEMENTATION_STATUS.md` | Permanent, committed |
| Decisions | `docs/decisions/` | Permanent, committed |
| Performance numbers | `docs/performance/` | Permanent, committed |
| Gotchas, your decisions, machine facts | One small file per fact in Claude memory | Until out of date |
| Throwaway output | Session scratchpad | Can vanish on reboot |

Claude memory is the project's `memory/` folder under `~/.claude/projects/`. `job start`
adds the card to `MEMORY.md`, so a new session sees the open job straight away.

## Managing the three kinds of memory

### Claude's context

- Use `/clear` between jobs. It is the only full reset, and only you can run it.
- Use `/compact` when a long job gets heavy but isn't finished. Run `job save` first.
- Use `/context` to see what is filling the conversation.
- Keep `MEMORY.md` to one short line per note, under 30 lines.
- Keep each note under 4 KB. `job status` flags larger ones.
- Delete or fix a note as soon as it goes stale. Don't add a correction beside it.

### RAM and swap

This Mac has 18 GB of RAM, and the real model and GPU runs need much of it.

- Quit Docker Desktop when you don't need it. Its VM can hold over 1 GB.
- Quit the JetBrains IDE before long test or benchmark runs.
- Swap only falls when the processes using it exit or the Mac reboots. Read
  [Before you reboot](#before-you-reboot) first.
- `sudo purge` drops the file cache but doesn't clear swap. Use it only for a cold-read
  measurement, because it invalidates warm results.
- Don't run builds, tests or a second session during a benchmark. Follow
  [BENCHMARKING.md](../performance/BENCHMARKING.md), and stay on power.

### Disk

| Command | Removes | Cost |
|---|---|---|
| `job clean` | Incremental debug data and side build directories | Next debug build is slower |
| `job clean --deep` | All of `target/` | Full rebuild |
| `job clean --scratch` | Session scratchpads with no file changed in 7 days | None |

Add `--dry-run` to see the list and sizes first. Use `--deep` when `target/` passes about
30 GB, or for a clean-build check.

`job clean` and `job verify` refuse to run while cargo, rustc, or a QUBIVEX or Colibri
binary is running, because that process may belong to another session. The script never
kills anything. `job clean --scratch` also skips scratchpads holding tokenizer or Colibri
files.

`job clean` never touches `~/.qubivex` (models and the oracle venv), git data, Claude
memory, or `qubivex-backup-2026-11-03.bundle`.

## Rules

- Run one job at a time in a working tree. For a parallel job, use a `git worktree` with
  its own branch, and don't benchmark while the other job builds.
- Before pushing, run `job verify`. It covers the checks CI runs, including Rust 1.88 and
  loom.
- Push with `job pr`. It uses HTTPS, because SSH to `origin` fails from this environment.
- Commits and PRs carry no Claude attribution. `job commit` and `job pr` enforce this.
- Claude commits and raises PRs when the job calls for it, but merges only when you ask.
- If a job stops halfway, commit or stash, run `job save`, and leave the card open. The
  next session resumes from it.

## Before you reboot

macOS clears `/private/tmp` on reboot, so every session scratchpad goes with it. Before
rebooting, run `job status`, then move anything still needed to `~/.qubivex/`. At the
moment that means:

- The tokenizer files that `QUBIVEX_OLMOE_TOKENIZER_JSON` needs.
- The Colibri source, build and import used by the comparison benchmark.
