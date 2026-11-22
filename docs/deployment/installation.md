# Installation

`install.sh` at the repository root builds the `qubivex` CLI from source and installs it
for the current user. It needs no sudo, never touches your models, and never replaces a
working binary with a failed build.

```sh
curl -fsSL https://raw.githubusercontent.com/kentmarete/qubivex/main/install.sh | bash
```

## Reading the script first

`curl ... | bash` runs code you have not read. To look first:

```sh
curl -fsSL https://raw.githubusercontent.com/kentmarete/qubivex/main/install.sh -o install.sh
less install.sh
bash install.sh --help
bash install.sh
```

The whole script sits in functions and runs from `main` on its last line, so a download
cut short does nothing. It downloads one other thing, and only when you allow it to
install Rust: `rustup-init` from `static.rust-lang.org`. It verifies that file against
its published SHA-256 before running it, with `--no-modify-path`.

## Options

| Option | Effect |
|---|---|
| `--version VERSION` | Install a tag. `0.2.0` and `v0.2.0` both find tag `v0.2.0` (or `0.2.0`) |
| `--branch BRANCH` | Install a branch (default `main`) |
| `--install-dir DIR` | Put the binary in `DIR` (created if missing; must be writable by you) |
| `--jobs N` | Parallel build jobs, passed to cargo. Lower it if the fat-LTO link runs out of memory |
| `--update` | Rebuild the installed branch if it has new commits; does nothing if it has none |
| `--uninstall` | Remove the binary and the install record; keep `~/.qubivex` |
| `--install-rust` | Allow the installer to install Rust with rustup if it is missing or too old |
| `--no-test` | Skip `qubivex system` and `qubivex doctor --storage` after installing |
| `--no-cleanup` | Keep the temporary clone and build directory (its path is printed) |
| `-h`, `--help` | Usage |

| Environment | Effect |
|---|---|
| `QUBIVEX_REPO_URL` | Build from another repository URL (a fork or mirror). Credentials in the URL are never printed or stored |
| `QUBIVEX_NO_PROMPT=1` | Never ask questions (CI). Without `--install-rust`, a missing Rust is then an error |
| `NO_COLOR` | Plain output |
| `CARGO_HOME` | The default install directory is `$CARGO_HOME/bin` |

Conflicting options are refused with exit status 2: `--version` with `--branch`,
`--update` with `--uninstall` or `--version`, and `--uninstall` with any build option.

## Steps

1. **Platform.** `uname` gives the OS and CPU. Linux and macOS on x86-64 or arm64 are
   accepted, and anything else stops with a message. An x86-64 shell on Apple Silicon
   (Rosetta) gets a warning to run natively.
2. **Tools.** git, then a C compiler: the Xcode Command Line Tools on macOS (checked with
   `xcode-select -p`), or `cc`, `gcc` or `clang` on Linux. `ring`, the TLS library
   behind `qubivex models pull`, compiles C and assembly. On macOS the linker comes from
   the Command Line Tools as well.
3. **Rust.** `rustc` must be at least the workspace's `rust-version` (1.88 today; read
   from the cloned `Cargo.toml`, so a later bump is followed automatically). If the default toolchain is
   older but rustup has a newer `stable`, that one is used through `cargo +stable`,
   without changing your default. If Rust is missing or too old, the installer asks, or
   uses `--install-rust`. It never changes your rustup default or your shell files.
4. **Source.** The branch or tag is checked with `git ls-remote`, then shallow-cloned
   into a fresh `mktemp -d` directory. The installer checks that `apps/cli/Cargo.toml` is
   the `qubivex-cli` package with a `qubivex` binary.
5. **Build.** `cargo install --path apps/cli --locked --root <tmp>/stage --no-track`,
   with `CARGO_TARGET_DIR` inside the temporary directory. `--locked` builds exactly
   `Cargo.lock`. The workspace release profile applies: fat LTO and one codegen unit,
   which is why the final link takes minutes. `--no-track` and the staging root
   leave `~/.cargo/.crates.toml` alone.
6. **Check.** The staged binary must print `qubivex <version>` for `--version` and
   succeed on `--help`. Otherwise nothing is installed.
7. **Install.** The binary is copied next to the target as a hidden temporary file,
   then renamed over `qubivex`. The rename is atomic, and on macOS it avoids rewriting a
   code-signed binary in place. If the installed binary does not run, the previous one is
   restored.
8. **Record.** `~/.qubivex/install-receipt` stores the directory, branch or tag, commit
   and version, as `key=value` lines. The installer reads it as data and never sources it.
9. **Report.** `qubivex system` (CPU, RAM, GPU backends) and
   `qubivex doctor --storage --path ~/.qubivex/models --size-mib 64` (disk read speed;
   it needs about 1.1 GiB free). A failure in either is a warning, because GPU
   backends are optional.
10. **PATH.** If the install directory is not on `PATH`, the installer prints the line to add for your
    shell (zsh, bash, fish). If another `qubivex` comes first on `PATH`, it says so.
11. **Cleanup.** An `EXIT` trap removes the temporary directory, also on failure or
    Ctrl-C, unless `--no-cleanup` was given.

GPU backends need nothing at build time. The Vulkan backend opens the loader at run
time (`ash` with `loaded`), and the Metal backend compiles its kernels when a model
loads. The binary therefore builds without the CUDA, Vulkan or Metal SDKs, and
`qubivex system` reports which backends the machine can use.

## Layout

| Path | Contents | Removed by `--uninstall` |
|---|---|---|
| `~/.cargo/bin/qubivex` (or `--install-dir`) | The binary | Yes |
| `~/.qubivex/install-receipt` | What was installed, from where | Yes |
| `~/.qubivex/update-check.json` | The latest version last looked up, for the update notice | No |
| `~/.qubivex/models/` | Models (the CLI's default models directory) | **Never** |
| `~/.qubivex/` (anything else) | Your configuration and data | **Never** |

## Platforms

| Platform | Status |
|---|---|
| macOS Apple Silicon | Installer tests and a full build-and-install run on CI (`macos-latest`) |
| Linux x86-64 (glibc) | Installer tests and a full build-and-install run on CI (`ubuntu-latest`) |
| Linux arm64 (glibc) | Accepted by the installer. The Dockerfile builds the workspace for arm64, but CI runs neither that build nor the installer on arm64 |
| macOS Intel | Accepted by the installer. The workspace builds there, but neither the build nor the installer is run on it in CI |
| Linux musl (Alpine) | Accepted with a warning; untested |
| Windows | Refused; use WSL 2 |

## Updating

```sh
./install.sh --update                 # same branch, same directory, as recorded
./install.sh --update --branch dev    # move the installation to another branch
./install.sh --version 0.3.0          # move to a tag (an installation pinned to a tag is not --update'd)
```

`--update` compares the branch's current commit with the recorded one and returns
straight away if they match. Otherwise it rebuilds and replaces the binary in the same
way as a fresh install. Running the plain installer again always rebuilds, which is a
safe way to repair an installation.

### From the CLI

An installed `qubivex` can update itself the same way, without a copy of the script:

```sh
qubivex update --check           # is there a newer commit? (installs nothing)
qubivex update                   # rebuild in this terminal
qubivex update --detach          # rebuild in the background and return at once
qubivex update --status          # is a background update running? end of its log
qubivex update --branch dev      # follow another branch (also leaves a pinned tag)
```

`qubivex update` reads `~/.qubivex/install-receipt`, asks the recorded repository for
the branch's newest commit with `git ls-remote`, and when that differs from the
installed one, shallow-clones that commit and runs its `install.sh --update`. It takes
`--jobs N` and `--no-test`, which it passes to the installer. Without a receipt (a
`cargo install` by hand) it refuses and says how to update instead.

`--detach` starts the update in its own process group, so Ctrl-C and closing the
terminal do not stop it. Its output goes to `~/.qubivex/update.log` (replaced by each
background update) and its pid to `~/.qubivex/update.pid`; a second `--detach` while
one runs is refused. It never prompts: a missing or old Rust fails the update, and the
log says what to do. Follow it with `tail -f ~/.qubivex/update.log`. The running
`qubivex` keeps working while the binary is replaced; the new version is used from the
next command.

### Update notices

When a newer version is out, a command run in a terminal ends with a notice on stderr,
after its own output:

```text
----------------------------------------
Qubivex Update Available

Current version: 0.1.2
Latest version:  0.2.0

Run `qubivex update` to upgrade.
----------------------------------------
```

- **What counts as newer.** The latest version is the `version` in the workspace
  `Cargo.toml` on the branch the installation follows (the one in
  `~/.qubivex/install-receipt`), which is what `qubivex update` would install. It is
  compared with the version compiled into the binary (the one `qubivex --version`
  prints) by semantic-version precedence, so `0.10.0` is newer than `0.9.0` and
  `0.2.0-rc.1` older than `0.2.0`. New commits without a version bump are not
  announced; `qubivex update --check` reports those. A build ahead of the branch gets
  no notice, and after `qubivex update` the notice stops from the next command.
- **How often.** At most one lookup a day. It never runs in the command's process: when
  the cache `~/.qubivex/update-check.json` is a day old, the command starts a lookup in
  the background (giving up after 3 seconds) and shows what the cache already holds.
  The first notice therefore comes on the command after the one that found it. A failed
  lookup (offline, GitHub down, a bad answer) is silent and leaves the last version
  found in place until the next day. A missing or damaged cache file counts as none.
- **`qubivex --version`.** Asking for the version looks the latest up there and then
  (waiting 2 seconds at most), whatever the cache holds, so a version released since
  the daily lookup is shown at once, after the `qubivex X.Y.Z` line. If the lookup
  fails, the cache answers. The version line on stdout is unchanged.
- **When it is never shown.** When stdout or stderr is not a terminal (piped,
  redirected, cron, scripts); after any `--json` output, `run`, `serve`,
  `bench` and `update`; after an error; when `CI` is set; and for installations that
  `qubivex update` cannot move: no receipt (a checkout or `cargo install`), pinned to a
  tag, or not hosted on GitHub. In none of these cases does it look anything up either.
- **Turning it off.** `export QUBIVEX_NO_UPDATE_CHECK=1` (any value but `0` or
  `false`). It then neither looks anything up nor shows anything.

## Uninstalling

```sh
./install.sh --uninstall
./install.sh --uninstall --install-dir ~/.local/bin   # a binary somewhere else
```

The installer removes the binary only if its `--version` says `qubivex`, so it cannot
delete an unrelated file of the same name. It never removes `~/.qubivex`. Delete that
yourself with `rm -rf ~/.qubivex` if you want your models gone too. A binary installed
with `sudo` elsewhere is left alone, and the installer tells you so.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `the Xcode Command Line Tools are not installed` | `xcode-select --install`, then rerun |
| `no C compiler found` | Debian/Ubuntu `sudo apt-get install build-essential`; Fedora `sudo dnf install gcc make`; Arch `sudo pacman -S base-devel` |
| `Rust 1.88 or newer is required` | `rustup update stable`, or rerun with `--install-rust` |
| `qubivex: command not found` after installing | Add the printed `export PATH=...` line to your shell's startup file and open a new shell |
| The build is killed or the link step fails with little RAM | `--jobs 1` or `--jobs 2`. Fat LTO links the whole program in one step |
| `no write permission for DIR` | Pick a directory you own: `--install-dir ~/.local/bin` |
| A different `qubivex` runs | The installer prints which one comes first on `PATH`. Remove it or reorder `PATH` |
| `doctor --storage did not complete` | Usually too little free space for its scratch file. The install itself succeeded |
| Anything else | Rerun with `--no-cleanup` and look in the printed build directory |

## Binary releases (planned)

The installer is split so that prebuilt binaries can be added later without changing
the rest of it. `acquire_binary` produces a staged executable. Everything after it is
independent of where the binary came from: the check, the atomic install, the record,
the hardware report and the PATH help. A `binary` method would:

1. use the target triple the installer already computes (`aarch64-apple-darwin`,
   `x86_64-unknown-linux-gnu`, ...);
2. look up the GitHub release for `--version` (or the latest);
3. download `qubivex-<version>-<triple>.tar.gz` and `SHA256SUMS`;
4. check the archive's SHA-256 (the `sha256_of` helper used for rustup-init already does this);
5. extract `qubivex` into the staging directory;
6. continue with the existing steps 6 to 11.

Building from source would remain the fallback for platforms without a published archive.

## Testing the installer

```sh
shellcheck install.sh tests/install/run.sh
tests/install/run.sh             # about a minute; needs git and Rust, nothing else
tests/install/run.sh update      # only tests whose name contains "update"
```

The tests run the installer against a small fixture repository with the same layout
(`apps/cli`, package `qubivex-cli`, binary `qubivex`), served from a `file://` URL. The
fixture has branches, tags, a branch that does not compile and a branch that needs a
future Rust. Each test has its own `HOME`, `CARGO_HOME`, `TMPDIR` and install
directory. Missing tools, other operating systems and other CPUs are simulated with a
reduced `PATH` and stub commands. CI also runs the real installer against the real
repository on Linux and macOS.
