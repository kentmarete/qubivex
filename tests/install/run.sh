#!/usr/bin/env bash
# Tests for install.sh. No dependencies beyond bash, git and a Rust toolchain.
#
#   tests/install/run.sh            all tests
#   tests/install/run.sh uninstall  only tests whose name contains "uninstall"
#
# The installer runs against a fixture repository: a tiny Cargo workspace laid out like
# QUBIVEX (apps/cli, package qubivex-cli, binary qubivex), with branches and tags, served
# from a local file:// URL. Each test gets its own HOME, CARGO_HOME, TMPDIR and install
# directory, so nothing outside a scratch directory is touched. Missing tools and other
# platforms are simulated with a reduced PATH and stub commands.
#
# A real build of QUBIVEX itself is not part of this suite (fat LTO takes minutes); CI
# runs that separately (the `installer` job).

# Stubs are single-quoted on purpose (they expand when run); "~" is passed literally on purpose.
# shellcheck disable=SC2016,SC2088

set -Eeuo pipefail

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALLER="$ROOT/install.sh"
FILTER="${1:-}"
REAL_RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
REAL_CARGO_BIN="$(dirname "$(command -v cargo)")"

SUITE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/qubivex-install-tests.XXXXXXXX")"
SUITE_DIR="$(cd -P "$SUITE_DIR" && pwd)"
trap 'rm -rf "$SUITE_DIR"' EXIT

PASS=0
FAIL=0
FAILED_NAMES=()

# ---------------------------------------------------------------------------------------
# Fixture repository.
# ---------------------------------------------------------------------------------------

FIXTURE="$SUITE_DIR/fixture-repo"
FIXTURE_URL="file://$FIXTURE"

# write_main VERSION: a qubivex stand-in. `doctor` fails on purpose, to check that a
# failed post-install check does not fail the install.
write_main() {
    cat >"$FIXTURE/apps/cli/src/main.rs" <<EOF
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.first().map(String::as_str) {
        Some("--version") => println!("qubivex $1"),
        Some("--help") => println!("usage: qubivex"),
        Some("system") => println!("fixture system: cpu ok"),
        Some("doctor") => { eprintln!("fixture doctor: refusing"); std::process::exit(3) }
        _ => std::process::exit(2),
    }
}
EOF
}

commit_all() { git -C "$FIXTURE" add -A && git -C "$FIXTURE" commit -q -m "$1"; }

make_fixture() {
    mkdir -p "$FIXTURE/apps/cli/src"
    git -C "$FIXTURE" init -q -b main
    git -C "$FIXTURE" config user.email test@example.invalid
    git -C "$FIXTURE" config user.name test
    git -C "$FIXTURE" config commit.gpgsign false
    git -C "$FIXTURE" config tag.gpgsign false
    cat >"$FIXTURE/Cargo.toml" <<'EOF'
[workspace]
resolver = "3"
members = ["apps/cli"]

[workspace.package]
version = "0.1.0"
edition = "2024"
rust-version = "1.88"

[profile.release]
lto = "fat"
codegen-units = 1
EOF
    cat >"$FIXTURE/apps/cli/Cargo.toml" <<'EOF'
[package]
name = "qubivex-cli"
version.workspace = true
edition.workspace = true
rust-version.workspace = true

[[bin]]
name = "qubivex"
path = "src/main.rs"
EOF
    write_main "0.1.0-main1"
    (cd "$FIXTURE" && CARGO_HOME="$SUITE_DIR/cargo-home" cargo generate-lockfile -q --offline)
    printf 'target/\n' >"$FIXTURE/.gitignore"
    commit_all "main 1"

    git -C "$FIXTURE" tag v0.2.0
    git -C "$FIXTURE" tag 0.3.0-rc1

    git -C "$FIXTURE" checkout -q -b dev
    write_main "0.1.0-dev"
    commit_all "dev"

    git -C "$FIXTURE" checkout -q -b broken main
    printf 'fn main() { this does not compile }\n' >"$FIXTURE/apps/cli/src/main.rs"
    commit_all "broken"

    git -C "$FIXTURE" checkout -q -b future-rust main
    sed -i.bak 's/rust-version = "1.88"/rust-version = "9.0"/' "$FIXTURE/Cargo.toml"
    rm -f "$FIXTURE/Cargo.toml.bak"
    commit_all "needs rust 9"

    git -C "$FIXTURE" checkout -q main
}

# A new commit on main, for --update.
advance_main() {
    git -C "$FIXTURE" checkout -q main
    write_main "$1"
    commit_all "main $1"
}

# ---------------------------------------------------------------------------------------
# Per-test sandbox and assertions.
# ---------------------------------------------------------------------------------------

T=""        # this test's scratch directory
OUT=""      # combined output of the last run
STATUS=0    # exit status of the last run

new_sandbox() {
    T="$(mktemp -d "$SUITE_DIR/t.XXXXXX")"
    mkdir -p "$T/home" "$T/tmp" "$T/home/.cargo"
}

# Runs the installer in the sandbox. Extra environment as VAR=value before "--".
run_installer() {
    local envs=()
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ "${1:-}" = "--" ] && shift
    set +e
    OUT="$(
        cd "$T" && env -i \
            HOME="$T/home" \
            CARGO_HOME="$T/home/.cargo" \
            RUSTUP_HOME="$REAL_RUSTUP_HOME" \
            TMPDIR="$T/tmp" \
            PATH="$REAL_CARGO_BIN:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin" \
            SHELL=/bin/zsh \
            TERM=dumb \
            QUBIVEX_NO_PROMPT=1 \
            QUBIVEX_REPO_URL="$FIXTURE_URL" \
            ${envs[@]+"${envs[@]}"} \
            bash "$INSTALLER" "$@" 2>&1 </dev/null
    )"
    STATUS=$?
    set -e
}

# Records the failure (a test function keeps going, so all its failures are shown).
fail() {
    : >"$T/.failed"
    printf '    FAIL: %s\n' "$*"
    printf '    --- installer output ---\n%s\n    ------------------------\n' "$OUT" | sed 's/^/    /'
    return 1
}
assert_status() { [ "$STATUS" -eq "$1" ] || fail "exit status $STATUS, expected $1"; }
assert_out() { case "$OUT" in *"$1"*) ;; *) fail "output lacks: $1" ;; esac; }
refute_out() { case "$OUT" in *"$1"*) fail "output has: $1" ;; *) ;; esac; }
assert_file() { [ -f "$1" ] || fail "missing file $1"; }
assert_no_file() { [ ! -e "$1" ] || fail "unexpected file $1"; }
assert_eq() { [ "$1" = "$2" ] || fail "expected '$2', got '$1'"; }

BIN() { printf '%s' "$T/home/.cargo/bin/qubivex"; }
installed_version() { "$(BIN)" --version 2>/dev/null || echo "none"; }
no_leftover_tmp() {
    local left
    left="$(find "$T/tmp" -maxdepth 1 -name 'qubivex-install.*' | head -n 1)"
    [ -z "$left" ] || fail "temporary directory left behind: $left"
}

# A directory of symlinks to the tools the installer may use, minus the excluded ones.
# Usage: make_path DIR excluded-tool...
make_path() {
    local dir="$1" tool src
    shift
    mkdir -p "$dir"
    for tool in bash sh env uname sed awk grep head mktemp chmod mkdir rm cp mv cat \
        dirname basename date ls tr find sysctl xcode-select cc clang gcc ldd ld \
        git cargo rustc rustup curl shasum sha256sum ar; do
        case " $* " in *" $tool "*) continue ;; esac
        src="$(PATH="$REAL_CARGO_BIN:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin" command -v "$tool" 2>/dev/null || true)"
        [ -n "$src" ] && ln -sf "$src" "$dir/$tool"
    done
    return 0
}

stub() { # stub DIR NAME BODY
    printf '#!/bin/sh\n%s\n' "$3" >"$1/$2"
    chmod +x "$1/$2"
}

# ---------------------------------------------------------------------------------------
# Unit tests of helper functions (the script without its last line, `main "$@"`).
# ---------------------------------------------------------------------------------------

with_functions() {
    bash -c 'source <(sed "\$d" "$1"); shift; "$@"' _ "$INSTALLER" "$@"
}

test_unit_version_compare() {
    local c
    for c in "1.88.0 1.88 0" "1.88 1.88.0 0" "1.99.1 1.88 0" "2.0.0 1.88 0" \
        "rustc_1.90.0-nightly_(abc) 1.88 0" "1.87.9 1.88 1" "1.8.0 1.88 1" "0.99 1.88 1" "garbage 1.88 1"; do
        # shellcheck disable=SC2086 # split "have need expected"
        set -- $c
        local st=0
        with_functions version_ge "${1//_/ }" "$2" || st=$?
        [ "$st" -eq "$3" ] || { OUT="version_ge '$1' '$2' -> $st"; fail "expected $3"; }
    done
}

test_unit_redact_url() {
    OUT="$(with_functions redact_url 'https://user:ghp_secret@github.com/kentmarete/qubivex.git')"
    assert_eq "$OUT" "https://github.com/kentmarete/qubivex.git"
    OUT="$(with_functions redact_url 'https://github.com/kentmarete/qubivex.git')"
    assert_eq "$OUT" "https://github.com/kentmarete/qubivex.git"
}

# ---------------------------------------------------------------------------------------
# Options.
# ---------------------------------------------------------------------------------------

test_help() {
    run_installer -- --help
    assert_status 0
    assert_out "--install-dir DIR"
    assert_out "--uninstall"
}

test_invalid_options() {
    local args
    while IFS= read -r args; do
        [ -n "$args" ] || continue
        # shellcheck disable=SC2086 # each line is a list of arguments
        run_installer -- $args
        [ "$STATUS" -eq 2 ] || fail "'$args' exited $STATUS, expected 2"
        assert_out "error:"
    done <<'EOF'
--bogus
positional
--version
--branch
--install-dir
--version 1.0 --branch main
--update --uninstall
--update --version 1.0
--uninstall --branch main
--uninstall --no-test
--uninstall --jobs 2
--jobs 0
--jobs abc
--branch ../etc
--branch -x
--version a..b
EOF
}

# ---------------------------------------------------------------------------------------
# Installing.
# ---------------------------------------------------------------------------------------

test_fresh_install() {
    run_installer -- --no-test
    assert_status 0
    assert_eq "$(installed_version)" "qubivex 0.1.0-main1"
    assert_file "$T/home/.qubivex/install-receipt"
    [ -d "$T/home/.qubivex/models" ] || fail "models directory not created"
    grep -q '^ref=main$' "$T/home/.qubivex/install-receipt" || fail "receipt lacks ref=main"
    grep -q '^ref_kind=branch$' "$T/home/.qubivex/install-receipt" || fail "receipt lacks ref_kind"
    assert_out "Installed qubivex 0.1.0-main1"
    assert_no_file "$T/home/.cargo/.crates.toml"
    assert_no_file "$T/home/.cargo/.crates2.json"
    no_leftover_tmp
}

test_path_hint_when_not_on_path() {
    run_installer -- --no-test
    assert_status 0
    assert_out "is not on your PATH"
    assert_out "export PATH=\"$T/home/.cargo/bin:\$PATH\""
    assert_out ".zshrc"
}

test_no_path_hint_when_on_path() {
    mkdir -p "$T/home/.cargo/bin"
    run_installer PATH="$T/home/.cargo/bin:$REAL_CARGO_BIN:/usr/bin:/bin:/usr/sbin:/sbin" -- --no-test
    assert_status 0
    refute_out "is not on your PATH"
}

test_post_install_checks_do_not_fail_install() {
    run_installer
    assert_status 0
    assert_out "fixture system: cpu ok"
    assert_out "qubivex doctor --storage"
    assert_out "warning: 'qubivex doctor --storage' did not complete"
    assert_out "Installed qubivex"
}

test_no_test_skips_checks() {
    run_installer -- --no-test
    assert_status 0
    refute_out "fixture system"
    assert_out "post-install checks skipped"
}

test_reinstall_keeps_models_and_config() {
    run_installer -- --no-test
    mkdir -p "$T/home/.qubivex/models/my-model"
    printf 'weights' >"$T/home/.qubivex/models/my-model/model.safetensors"
    printf 'listen = "x"\n' >"$T/home/.qubivex/serve.toml"
    run_installer -- --no-test
    assert_status 0
    assert_out "existing install"
    assert_eq "$(cat "$T/home/.qubivex/models/my-model/model.safetensors")" "weights"
    assert_file "$T/home/.qubivex/serve.toml"
    assert_eq "$(installed_version)" "qubivex 0.1.0-main1"
}

test_branch_install() {
    run_installer -- --no-test --branch dev
    assert_status 0
    assert_eq "$(installed_version)" "qubivex 0.1.0-dev"
    grep -q '^ref=dev$' "$T/home/.qubivex/install-receipt" || fail "receipt lacks ref=dev"
}

test_tag_install_with_and_without_v() {
    run_installer -- --no-test --version 0.2.0
    assert_status 0
    assert_out "tag v0.2.0"
    grep -q '^ref_kind=tag$' "$T/home/.qubivex/install-receipt" || fail "receipt lacks ref_kind=tag"
    run_installer -- --no-test --version v0.2.0
    assert_status 0
    run_installer -- --no-test --version=0.3.0-rc1
    assert_status 0
    assert_out "tag 0.3.0-rc1"
}

test_unknown_branch_and_tag() {
    run_installer -- --no-test --branch nope
    assert_status 1
    assert_out "no branch 'nope'"
    run_installer -- --no-test --version 9.9.9
    assert_status 1
    assert_out "no release tag '9.9.9'"
    assert_no_file "$(BIN)"
}

test_custom_install_dir_relative_and_tilde() {
    run_installer -- --no-test --install-dir rel/bin
    assert_status 0
    [ -x "$T/rel/bin/qubivex" ] || fail "relative --install-dir not honoured"
    run_installer -- --no-test --install-dir "~/.local/bin"
    assert_status 0
    [ -x "$T/home/.local/bin/qubivex" ] || fail "~ in --install-dir not expanded"
    grep -q "^install_dir=$T/home/.local/bin\$" "$T/home/.qubivex/install-receipt" ||
        fail "receipt install_dir wrong"
}

test_install_dir_not_writable() {
    mkdir -p "$T/locked"
    chmod 555 "$T/locked"
    run_installer -- --no-test --install-dir "$T/locked"
    chmod 755 "$T/locked"
    assert_status 1
    assert_out "no write permission for"
    assert_out "does not use sudo"
}

test_install_dir_is_a_file() {
    : >"$T/afile"
    run_installer -- --no-test --install-dir "$T/afile"
    assert_status 1
    assert_out "is not a directory"
}

test_jobs_passed_to_cargo() {
    run_installer -- --no-test --jobs 2
    assert_status 0
    assert_out "--jobs 2"
}

test_no_cleanup_keeps_build_dir() {
    run_installer -- --no-test --no-cleanup
    assert_status 0
    assert_out "Kept the source and build files in: $T/tmp/qubivex-install."
    [ -n "$(find "$T/tmp" -maxdepth 1 -name 'qubivex-install.*')" ] || fail "build directory removed"
}

# ---------------------------------------------------------------------------------------
# Failures leave the existing install alone.
# ---------------------------------------------------------------------------------------

test_compile_failure_keeps_existing_binary() {
    run_installer -- --no-test
    assert_status 0
    local before
    before="$(installed_version)"
    run_installer -- --no-test --branch broken
    assert_status 1
    assert_out "the build failed; nothing was installed"
    assert_eq "$(installed_version)" "$before"
    grep -q '^ref=main$' "$T/home/.qubivex/install-receipt" || fail "receipt changed by a failed build"
    [ -z "$(find "$T/home/.cargo/bin" -name '.qubivex.new.*')" ] || fail "partial binary left in install dir"
    no_leftover_tmp
}

test_missing_git() {
    make_path "$T/path" git
    run_installer PATH="$T/path" -- --no-test
    assert_status 1
    assert_out "git is not installed"
}

test_missing_rust() {
    make_path "$T/path" cargo rustc rustup
    run_installer PATH="$T/path" CARGO_HOME="$T/empty-cargo" -- --no-test
    assert_status 1
    assert_out "Rust (rustc and cargo) is not installed"
    assert_out "--install-rust"
}

test_old_rust() {
    make_path "$T/path" cargo rustc rustup
    stub "$T/path" rustc 'echo "rustc 1.80.0 (051478957 2024-07-21)"'
    stub "$T/path" cargo 'echo "cargo 1.80.0"'
    run_installer PATH="$T/path" CARGO_HOME="$T/empty-cargo" -- --no-test
    assert_status 1
    assert_out "Rust 1.88.0 or newer is required; found: rustc 1.80.0"
    run_installer PATH="$T/path" CARGO_HOME="$T/empty-cargo" -- --no-test --install-rust
    assert_status 1
    assert_out "was not installed with rustup"
}

test_workspace_rust_version_is_enforced() {
    run_installer -- --no-test --branch future-rust
    assert_status 1
    assert_out "Rust 9.0 or newer is required"
    assert_no_file "$(BIN)"
}

test_unsupported_os() {
    make_path "$T/path" uname
    stub "$T/path" uname 'case "$1" in -s) echo FreeBSD ;; -m) echo amd64 ;; *) echo FreeBSD ;; esac'
    run_installer PATH="$T/path" -- --no-test
    assert_status 1
    assert_out "unsupported operating system: FreeBSD"
    stub "$T/path" uname 'case "$1" in -s) echo MINGW64_NT-10.0 ;; *) echo x86_64 ;; esac'
    run_installer PATH="$T/path" -- --no-test
    assert_status 1
    assert_out "WSL 2"
}

test_unsupported_arch() {
    make_path "$T/path" uname
    stub "$T/path" uname "case \"\$1\" in -s) echo $(uname -s) ;; -m) echo riscv64 ;; esac"
    run_installer PATH="$T/path" -- --no-test
    assert_status 1
    assert_out "unsupported CPU architecture: riscv64"
}

# ---------------------------------------------------------------------------------------
# Update and uninstall.
# ---------------------------------------------------------------------------------------

test_update_without_install() {
    run_installer -- --update
    assert_status 1
    assert_out "no QUBIVEX installation found"
}

test_update_up_to_date_then_new_commit() {
    run_installer -- --no-test
    assert_status 0
    run_installer -- --update --no-test
    assert_status 0
    assert_out "Already up to date"
    advance_main "0.1.0-main2"
    run_installer -- --update --no-test
    local st="$STATUS"
    git -C "$FIXTURE" reset -q --hard HEAD~1
    STATUS="$st"
    assert_status 0
    assert_eq "$(installed_version)" "qubivex 0.1.0-main2"
}

test_update_follows_installed_branch_and_dir() {
    run_installer -- --no-test --branch dev --install-dir "$T/custom"
    assert_status 0
    run_installer -- --update --no-test
    assert_status 0
    assert_out "Already up to date"
    assert_out "branch dev"
}

test_update_refuses_pinned_tag() {
    run_installer -- --no-test --version 0.2.0
    run_installer -- --update
    assert_status 1
    assert_out "pinned to tag 'v0.2.0'"
}

test_uninstall_keeps_data() {
    run_installer -- --no-test
    mkdir -p "$T/home/.qubivex/models/m"
    printf 'w' >"$T/home/.qubivex/models/m/w.gguf"
    run_installer -- --uninstall
    assert_status 0
    assert_no_file "$(BIN)"
    assert_no_file "$T/home/.qubivex/install-receipt"
    assert_file "$T/home/.qubivex/models/m/w.gguf"
    assert_out "kept your data"
    run_installer -- --uninstall
    assert_status 0
    assert_out "Nothing to uninstall"
}

test_uninstall_custom_dir_from_receipt() {
    run_installer -- --no-test --install-dir "$T/custom"
    run_installer -- --uninstall
    assert_status 0
    assert_no_file "$T/custom/qubivex"
}

test_uninstall_refuses_foreign_binary() {
    mkdir -p "$T/home/.cargo/bin"
    stub "$T/home/.cargo/bin" qubivex 'echo "something-else 1.0"'
    run_installer -- --uninstall
    assert_status 1
    assert_out "does not look like the qubivex CLI"
    assert_file "$(BIN)"
}

# ---------------------------------------------------------------------------------------

main() {
    printf 'Building the fixture repository...\n'
    make_fixture
    local name
    for name in $(declare -F | awk '{print $3}' | grep '^test_'); do
        case "$name" in *"$FILTER"*) ;; *) continue ;; esac
        new_sandbox
        "$name" || : >"$T/.failed"
        if [ ! -e "$T/.failed" ]; then
            PASS=$((PASS + 1))
            printf 'ok    %s\n' "${name#test_}"
        else
            FAIL=$((FAIL + 1))
            FAILED_NAMES+=("$name")
            printf 'FAIL  %s\n' "${name#test_}"
        fi
        chmod -R u+w "$T" 2>/dev/null || true
    done
    printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
    [ "$FAIL" -eq 0 ] || { printf 'failed: %s\n' "${FAILED_NAMES[*]}"; exit 1; }
}

main
