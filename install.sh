#!/usr/bin/env bash
# QUBIVEX installer: builds the `qubivex` CLI from source and installs it for the
# current user, without sudo.
#
#   curl -fsSL https://raw.githubusercontent.com/kentmarete/qubivex/main/install.sh | bash
#   ./install.sh --help
#
# To read the script before running it:
#
#   curl -fsSL https://raw.githubusercontent.com/kentmarete/qubivex/main/install.sh -o install.sh
#   less install.sh
#   bash install.sh
#
# What it does: checks the platform and the build tools (git, Rust >= the workspace's
# rust-version, a C compiler), shallow-clones the repository into a temporary directory,
# runs `cargo install --path apps/cli --locked` into a staging directory there, checks
# the staged binary, then moves it into place with a rename, so a failed build never
# touches a working installation. It never deletes ~/.qubivex (models, configuration)
# and never edits shell startup files.
#
# Written for bash 3.2 (macOS /bin/bash) and later, with POSIX tools only.
#
# Everything runs from `main` at the bottom, so a truncated download runs nothing.

set -Eeuo pipefail

# ---------------------------------------------------------------------------------------
# Constants. QUBIVEX_REPO_URL overrides the repository (a fork or a local mirror).
# ---------------------------------------------------------------------------------------

readonly INSTALLER_NAME="qubivex install.sh"
readonly DEFAULT_REPO_URL="https://github.com/kentmarete/qubivex.git"
readonly DEFAULT_BRANCH="main"
readonly BIN_NAME="qubivex"
readonly CLI_PACKAGE="qubivex-cli"
readonly CLI_PATH="apps/cli"
# Used before the source is cloned; afterwards the workspace's own rust-version wins.
readonly FALLBACK_MIN_RUST="1.88.0"
# `qubivex doctor --storage` refuses less than 64 MiB.
readonly DOCTOR_SIZE_MIB=64

# ---------------------------------------------------------------------------------------
# Options and state (set by parse_args and the steps below).
# ---------------------------------------------------------------------------------------

ACTION="install"          # install | update | uninstall
REF_KIND=""               # branch | tag (empty: default branch)
REF_NAME=""
OPT_VERSION=""
OPT_BRANCH=""
OPT_INSTALL_DIR=""
OPT_JOBS=""
RUN_TESTS=1
CLEANUP=1
ALLOW_RUST_INSTALL=0

REPO_URL="${QUBIVEX_REPO_URL:-$DEFAULT_REPO_URL}"
DATA_DIR="${HOME:-}/.qubivex"
RECEIPT_FILE=""
INSTALL_DIR=""
WORK_DIR=""
OS=""                     # linux | macos
ARCH=""                   # x86_64 | aarch64
TARGET_TRIPLE=""
CARGO_CMD=()              # cargo, or cargo +stable when the installer added a toolchain
RUSTC_CMD=()
RESOLVED_COMMIT=""

# ---------------------------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------------------------

if [ -t 2 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ]; then
    C_BOLD=$'\033[1m'; C_RED=$'\033[31m'; C_YELLOW=$'\033[33m'; C_GREEN=$'\033[32m'
    C_DIM=$'\033[2m'; C_RESET=$'\033[0m'
else
    C_BOLD=""; C_RED=""; C_YELLOW=""; C_GREEN=""; C_DIM=""; C_RESET=""
fi

# All messages go to stderr; stdout carries only what a command is asked to print.
say() { printf '%s\n' "$*" >&2; }
step() { printf '%s==>%s %s%s%s\n' "$C_GREEN" "$C_RESET" "$C_BOLD" "$*" "$C_RESET" >&2; }
info() { printf '    %s\n' "$*" >&2; }
warn() { printf '%swarning:%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
die() {
    printf '%serror:%s %s\n' "$C_RED" "$C_RESET" "$1" >&2
    shift
    local line
    for line in "$@"; do printf '       %s\n' "$line" >&2; done
    exit 1
}
# Exit code 2 for usage errors, as most CLIs do.
usage_error() {
    printf '%serror:%s %s\n' "$C_RED" "$C_RESET" "$1" >&2
    printf '       Run with --help for the options.\n' >&2
    exit 2
}

# A repository URL without credentials (https://user:token@host/... -> https://host/...).
redact_url() {
    printf '%s' "$1" | sed -E 's#^([a-zA-Z][a-zA-Z0-9+.-]*://)[^/@]*@#\1#'
}

usage() {
    cat <<EOF
QUBIVEX installer: builds the qubivex CLI from source and installs it for this user.

Usage:
  install.sh [OPTIONS]
  curl -fsSL https://raw.githubusercontent.com/kentmarete/qubivex/main/install.sh | bash -s -- [OPTIONS]

Options:
  --version VERSION   Install a release tag (VERSION or vVERSION, e.g. 0.2.0 or v0.2.0)
  --branch BRANCH     Install from a git branch (default: ${DEFAULT_BRANCH})
  --install-dir DIR   Where to put the qubivex binary (default: \${CARGO_HOME:-~/.cargo}/bin)
  --jobs N            Parallel build jobs (default: cargo's, the number of CPUs)
  --update            Rebuild the installed branch if it has new commits
  --uninstall         Remove the qubivex binary; models and ~/.qubivex are kept
  --install-rust      Allow installing Rust with rustup if it is missing or too old
  --no-test           Skip the post-install checks (qubivex system, doctor --storage)
  --no-cleanup        Keep the temporary source and build directory
  -h, --help          Show this help

Paths:
  binary        \${CARGO_HOME:-~/.cargo}/bin/qubivex, or --install-dir
  data          ~/.qubivex           (never deleted by this script)
  models        ~/.qubivex/models    (never deleted by this script)
  receipt       ~/.qubivex/install-receipt (what was installed, from where)

Environment:
  QUBIVEX_REPO_URL    Repository to build from (default: ${DEFAULT_REPO_URL})
  QUBIVEX_NO_PROMPT   Set to 1 to never ask questions (CI); Rust then needs --install-rust
  NO_COLOR            Disable coloured output

Requirements: Linux (x86-64, arm64) or macOS (Apple Silicon, Intel); git; a C
compiler (Xcode Command Line Tools on macOS, gcc/build-essential on Linux); Rust and
Cargo at least the version in the workspace Cargo.toml (1.88 today). No GPU SDK.
No sudo: the default install directory belongs to you.
EOF
}

# ---------------------------------------------------------------------------------------
# Arguments.
# ---------------------------------------------------------------------------------------

# Value of an option that takes one; rejects a missing value or one that is an option.
need_value() {
    if [ "$#" -lt 2 ] || [ -z "$2" ] || [ "${2#-}" != "$2" ]; then
        usage_error "$1 needs a value"
    fi
}

parse_args() {
    local saw_update=0 saw_uninstall=0 saw_no_test=0 saw_no_cleanup=0
    while [ "$#" -gt 0 ]; do
        case "$1" in
            -h | --help) usage; exit 0 ;;
            --version) need_value "$@"; OPT_VERSION="$2"; shift 2 ;;
            --version=*) OPT_VERSION="${1#*=}"; [ -n "$OPT_VERSION" ] || usage_error "--version needs a value"; shift ;;
            --branch) need_value "$@"; OPT_BRANCH="$2"; shift 2 ;;
            --branch=*) OPT_BRANCH="${1#*=}"; [ -n "$OPT_BRANCH" ] || usage_error "--branch needs a value"; shift ;;
            --install-dir) need_value "$@"; OPT_INSTALL_DIR="$2"; shift 2 ;;
            --install-dir=*) OPT_INSTALL_DIR="${1#*=}"; [ -n "$OPT_INSTALL_DIR" ] || usage_error "--install-dir needs a value"; shift ;;
            --jobs | -j) need_value "$@"; OPT_JOBS="$2"; shift 2 ;;
            --jobs=*) OPT_JOBS="${1#*=}"; shift ;;
            --update) saw_update=1; shift ;;
            --uninstall) saw_uninstall=1; shift ;;
            --install-rust) ALLOW_RUST_INSTALL=1; shift ;;
            --no-test) saw_no_test=1; RUN_TESTS=0; shift ;;
            --no-cleanup) saw_no_cleanup=1; CLEANUP=0; shift ;;
            --) shift; [ "$#" -eq 0 ] || usage_error "unexpected argument: $1" ;;
            -*) usage_error "unknown option: $1" ;;
            *) usage_error "unexpected argument: $1" ;;
        esac
    done

    if [ -n "$OPT_VERSION" ] && [ -n "$OPT_BRANCH" ]; then
        usage_error "--version and --branch cannot be combined: pick a tag or a branch"
    fi
    if [ "$saw_update" -eq 1 ] && [ "$saw_uninstall" -eq 1 ]; then
        usage_error "--update and --uninstall cannot be combined"
    fi
    if [ "$saw_update" -eq 1 ] && [ -n "$OPT_VERSION" ]; then
        usage_error "--update follows a branch; to move to a tag, run without --update: --version $OPT_VERSION"
    fi
    if [ "$saw_uninstall" -eq 1 ]; then
        local bad=""
        [ -n "$OPT_VERSION" ] && bad="--version"
        [ -n "$OPT_BRANCH" ] && bad="--branch"
        [ -n "$OPT_JOBS" ] && bad="--jobs"
        [ "$ALLOW_RUST_INSTALL" -eq 1 ] && bad="--install-rust"
        [ "$saw_no_test" -eq 1 ] && bad="--no-test"
        [ "$saw_no_cleanup" -eq 1 ] && bad="--no-cleanup"
        [ -z "$bad" ] || usage_error "--uninstall cannot be combined with $bad"
        ACTION="uninstall"
    elif [ "$saw_update" -eq 1 ]; then
        ACTION="update"
    fi

    if [ -n "$OPT_JOBS" ]; then
        case "$OPT_JOBS" in
            '' | *[!0-9]* | 0*) usage_error "--jobs must be a positive integer, got '$OPT_JOBS'" ;;
        esac
    fi
    if [ -n "$OPT_VERSION" ]; then validate_ref "--version" "$OPT_VERSION"; fi
    if [ -n "$OPT_BRANCH" ]; then validate_ref "--branch" "$OPT_BRANCH"; fi
    return 0
}

# A conservative subset of git's ref-name rules: enough to reject anything that could be
# read as an option, a path escape or a revision expression.
validate_ref() {
    local opt="$1" ref="$2"
    case "$ref" in
        -* | */ | /* | *..* | *//* | *.lock | *@\{* | *[!A-Za-z0-9._/+-]*)
            usage_error "$opt: '$ref' is not a valid git ref name" ;;
    esac
}

# ---------------------------------------------------------------------------------------
# Platform.
# ---------------------------------------------------------------------------------------

detect_platform() {
    local kernel machine
    kernel="$(uname -s 2>/dev/null || echo unknown)"
    machine="$(uname -m 2>/dev/null || echo unknown)"

    case "$kernel" in
        Linux) OS="linux" ;;
        Darwin) OS="macos" ;;
        MINGW* | MSYS* | CYGWIN* | Windows_NT)
            die "Windows is not supported by this installer." \
                "QUBIVEX targets Linux and macOS. On Windows, use WSL 2 (Ubuntu) and run the installer there." ;;
        *)
            die "unsupported operating system: $kernel" \
                "Supported: Linux (x86-64, arm64) and macOS (Apple Silicon, Intel)." ;;
    esac

    case "$machine" in
        x86_64 | amd64) ARCH="x86_64" ;;
        aarch64 | arm64) ARCH="aarch64" ;;
        *)
            die "unsupported CPU architecture: $machine ($kernel)" \
                "Supported: x86-64 and arm64 (aarch64). QUBIVEX's CPU kernels are written for these two." ;;
    esac

    if [ "$OS" = "macos" ] && [ "$ARCH" = "x86_64" ]; then
        # An x86-64 shell on Apple Silicon runs under Rosetta; build natively instead.
        if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ]; then
            warn "this shell runs under Rosetta 2 on Apple Silicon; the build will be x86-64."
            info "For a native arm64 build (and the Metal backend at full speed), run the installer"
            info "from a native terminal, or: arch -arm64 bash install.sh"
        fi
    fi

    case "$OS" in
        linux) TARGET_TRIPLE="${ARCH}-unknown-linux-gnu" ;;
        macos) TARGET_TRIPLE="${ARCH}-apple-darwin" ;;
    esac

    if [ "$OS" = "linux" ] && command -v ldd >/dev/null 2>&1 &&
        ldd --version 2>&1 | grep -qi musl; then
        TARGET_TRIPLE="${ARCH}-unknown-linux-musl"
        warn "musl libc detected (e.g. Alpine): QUBIVEX is only tested on glibc distributions."
    fi
}

# ---------------------------------------------------------------------------------------
# Versions: "1.88" or "rustc 1.90.0-nightly (...)" -> compared numerically, no sort -V.
# ---------------------------------------------------------------------------------------

# Prints "MAJOR MINOR PATCH" for the first x.y[.z] in the input, or nothing.
parse_version() {
    printf '%s\n' "$1" | sed -nE 's/^[^0-9]*([0-9]+)\.([0-9]+)(\.([0-9]+))?.*$/\1 \2 \4/p' | head -n 1
}

# version_ge HAVE NEED: true when HAVE >= NEED.
version_ge() {
    local have need h1 h2 h3 n1 n2 n3
    have="$(parse_version "$1")"
    need="$(parse_version "$2")"
    [ -n "$have" ] && [ -n "$need" ] || return 1
    # shellcheck disable=SC2086 # word splitting is the point
    set -- $have
    h1="$1"; h2="$2"; h3="${3:-0}"
    # shellcheck disable=SC2086
    set -- $need
    n1="$1"; n2="$2"; n3="${3:-0}"
    [ "$h1" -gt "$n1" ] && return 0
    [ "$h1" -lt "$n1" ] && return 1
    [ "$h2" -gt "$n2" ] && return 0
    [ "$h2" -lt "$n2" ] && return 1
    [ "$h3" -ge "$n3" ]
}

# ---------------------------------------------------------------------------------------
# Dependencies.
# ---------------------------------------------------------------------------------------

have() { command -v "$1" >/dev/null 2>&1; }

check_git() {
    have git && return 0
    case "$OS" in
        macos) die "git is not installed." "Install the Xcode Command Line Tools (they include git): xcode-select --install" ;;
        linux) die "git is not installed." \
            "Debian/Ubuntu: sudo apt-get install git" \
            "Fedora/RHEL:   sudo dnf install git" \
            "Arch:          sudo pacman -S git" ;;
    esac
}

# `ring` (TLS for `qubivex models pull`) compiles C and assembly, and on macOS the
# linker comes from the Command Line Tools too.
check_c_compiler() {
    if [ "$OS" = "macos" ]; then
        if ! xcode-select -p >/dev/null 2>&1; then
            die "the Xcode Command Line Tools are not installed (needed for clang and the linker)." \
                "Run: xcode-select --install" \
                "then run this installer again. Full Xcode is not needed."
        fi
        if ! have cc && ! have clang; then
            die "no C compiler found although the Command Line Tools are set up." \
                "Try: sudo xcode-select --reset   or reinstall them with xcode-select --install"
        fi
        return 0
    fi
    if [ -n "${CC:-}" ] && have "${CC%% *}"; then return 0; fi
    if have cc || have gcc || have clang; then return 0; fi
    die "no C compiler found (cc, gcc or clang). The TLS library 'ring' compiles C code." \
        "Debian/Ubuntu: sudo apt-get install build-essential" \
        "Fedora/RHEL:   sudo dnf install gcc make" \
        "Arch:          sudo pacman -S base-devel" \
        "Alpine:        sudo apk add build-base"
}

# Finds cargo and rustc. Looks in CARGO_HOME/bin too, for a rustup install whose PATH
# line has not been loaded into this shell yet.
find_rust() {
    local cargo_bin="${CARGO_HOME:-$HOME/.cargo}/bin"
    if ! have cargo && [ -x "$cargo_bin/cargo" ]; then
        PATH="$cargo_bin:$PATH"
        export PATH
    fi
    if have cargo && have rustc; then
        CARGO_CMD=(cargo)
        RUSTC_CMD=(rustc)
        return 0
    fi
    return 1
}

rustc_version() { "${RUSTC_CMD[@]}" -V 2>/dev/null || true; }

# Reads yes/no from the terminal, also when the script itself arrives on stdin (curl | bash).
confirm() {
    local answer=""
    [ -z "${QUBIVEX_NO_PROMPT:-}" ] || return 1
    if [ -r /dev/tty ] && { : </dev/tty; } 2>/dev/null; then
        printf '%s [y/N] ' "$1" >/dev/tty
        IFS= read -r answer </dev/tty || answer=""
    else
        return 1
    fi
    case "$answer" in y | Y | yes | YES | Yes) return 0 ;; *) return 1 ;; esac
}

# Rust is installed only with consent: --install-rust, or "y" at an interactive prompt.
# Nothing in the user's existing setup is changed: rustup gets --no-modify-path, and an
# existing rustup only gains a stable toolchain, used here through `cargo +stable`.
ensure_rust() {
    local need="$1" current=""
    if find_rust; then
        current="$(rustc_version)"
        if version_ge "$current" "$need"; then
            info "rust: $current"
            return 0
        fi
        if have rustup && "${CARGO_CMD[@]}" +stable -V >/dev/null 2>&1 &&
            version_ge "$(rustc +stable -V 2>/dev/null || true)" "$need"; then
            CARGO_CMD=(cargo +stable)
            RUSTC_CMD=(rustc +stable)
            info "rust: $(rustc_version) (the default toolchain, $current, is older than $need)"
            return 0
        fi
        rust_permission "Rust $need or newer is required; found: ${current:-unknown}." || die \
            "Rust $need or newer is required; found: ${current:-unknown}." \
            "With rustup: rustup update stable" \
            "Or rerun with --install-rust to let this installer add a stable toolchain."
        if have rustup; then
            step "Installing the stable Rust toolchain with rustup"
            rustup toolchain install stable --profile minimal ||
                die "rustup could not install the stable toolchain." "Run: rustup toolchain install stable"
            CARGO_CMD=(cargo +stable)
            RUSTC_CMD=(rustc +stable)
        else
            die "Rust $current was not installed with rustup, so the installer cannot upgrade it." \
                "Upgrade Rust with the tool that installed it, or install rustup: https://rustup.rs"
        fi
    else
        rust_permission "Rust (rustc and cargo) is not installed." || die \
            "Rust (rustc and cargo) is not installed." \
            "Install it from https://rustup.rs, then run this installer again," \
            "or rerun with --install-rust to let this installer run rustup for you."
        install_rustup
        find_rust || die "Rust was installed but cargo is not in ${CARGO_HOME:-$HOME/.cargo}/bin."
    fi
    current="$(rustc_version)"
    version_ge "$current" "$need" || die "Rust $need or newer is required; found: $current."
    info "rust: $current"
}

rust_permission() {
    [ "$ALLOW_RUST_INSTALL" -eq 1 ] && return 0
    warn "$1"
    confirm "Install Rust with rustup (https://rustup.rs) now?"
}

# Downloads rustup-init for this platform from static.rust-lang.org and checks its
# published SHA-256 before running it (no `curl | sh`).
install_rustup() {
    local url base sum_file bin expected actual
    have curl || die "curl is required to download rustup." "Install curl, or install Rust yourself from https://rustup.rs"
    base="https://static.rust-lang.org/rustup/dist/${TARGET_TRIPLE}"
    url="${base}/rustup-init"
    bin="$WORK_DIR/rustup-init"
    sum_file="$WORK_DIR/rustup-init.sha256"

    step "Downloading rustup-init for $TARGET_TRIPLE"
    curl --proto '=https' --tlsv1.2 -fsSL "$url" -o "$bin" || die "could not download $url"
    curl --proto '=https' --tlsv1.2 -fsSL "${url}.sha256" -o "$sum_file" || die "could not download ${url}.sha256"
    expected="$(awk '{print $1; exit}' "$sum_file")"
    actual="$(sha256_of "$bin")"
    if [ -z "$expected" ] || [ "$expected" != "$actual" ]; then
        die "rustup-init checksum mismatch (expected '$expected', got '$actual'); not running it."
    fi
    info "checksum ok: $actual"
    chmod 700 "$bin"
    step "Installing Rust (stable, minimal profile; shell startup files are not modified)"
    "$bin" -y --profile minimal --default-toolchain stable --no-modify-path ||
        die "rustup-init failed. See its output above, or install Rust from https://rustup.rs"
    PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
    export PATH
}

sha256_of() {
    if have sha256sum; then
        sha256sum "$1" | awk '{print $1}'
    elif have shasum; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        die "neither sha256sum nor shasum is available to verify downloads."
    fi
}

# ---------------------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------------------

default_install_dir() { printf '%s/bin' "${CARGO_HOME:-$HOME/.cargo}"; }

# Turns the --install-dir value into an absolute, existing, writable directory.
resolve_install_dir() {
    local dir="$1"
    # A literal "~" or "~/..." reaches here when the value was quoted.
    case "$dir" in
        *$'\n'* | *$'\r'*) die "--install-dir must not contain line breaks." ;;
        \~) dir="$HOME" ;;
        \~/*) dir="$HOME/${dir#\~/}" ;;
    esac
    case "$dir" in /*) ;; *) dir="$PWD/$dir" ;; esac
    if [ -e "$dir" ] && [ ! -d "$dir" ]; then
        die "--install-dir $dir exists and is not a directory."
    fi
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir" 2>/dev/null || die "cannot create $dir." \
            "Choose a directory you own, for example: --install-dir \"$HOME/.local/bin\""
    fi
    # Logical path (symlinks kept), so it matches how the directory appears in PATH.
    dir="$(cd "$dir" && pwd)"
    if [ ! -w "$dir" ]; then
        die "no write permission for $dir." \
            "This installer does not use sudo. Choose a directory you own, for example:" \
            "  --install-dir \"$HOME/.local/bin\""
    fi
    INSTALL_DIR="$dir"
}

# Safe temporary directory: mktemp gives a fresh 0700 directory; removed on exit.
make_work_dir() {
    WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/qubivex-install.XXXXXXXX")" ||
        die "could not create a temporary directory in ${TMPDIR:-/tmp}."
    chmod 700 "$WORK_DIR"
}

cleanup() {
    local status=$?
    if [ -n "$WORK_DIR" ] && [ -d "$WORK_DIR" ]; then
        if [ "$CLEANUP" -eq 1 ]; then
            # The build directory is ours alone: created by mktemp under a fixed prefix.
            case "$WORK_DIR" in
                */qubivex-install.*) rm -rf "$WORK_DIR" ;;
            esac
        else
            say "${C_DIM}Kept the source and build files in: $WORK_DIR${C_RESET}"
        fi
    fi
    return "$status"
}

on_error() {
    local status=$? line="$1"
    # `die` already explained itself; this catches anything unexpected.
    if [ "$status" -ne 0 ] && [ "$status" -ne 1 ] && [ "$status" -ne 2 ]; then
        printf '%serror:%s the installer stopped unexpectedly (exit %s, line %s).\n' \
            "$C_RED" "$C_RESET" "$status" "$line" >&2
        printf '       No installed binary was changed. Rerun with --no-cleanup to keep the build files.\n' >&2
    fi
}

# ---------------------------------------------------------------------------------------
# Receipt: plain key=value lines, read line by line, never sourced.
# ---------------------------------------------------------------------------------------

receipt_get() {
    local key="$1" line
    [ -f "$RECEIPT_FILE" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in "$key="*) printf '%s' "${line#*=}"; return 0 ;; esac
    done <"$RECEIPT_FILE"
}

write_receipt() {
    local version="$1" tmp
    mkdir -p "$DATA_DIR"
    tmp="$DATA_DIR/.install-receipt.$$"
    {
        printf '# Written by %s. Safe to delete; used by --update and --uninstall.\n' "$INSTALLER_NAME"
        printf 'binary=%s\n' "$INSTALL_DIR/$BIN_NAME"
        printf 'install_dir=%s\n' "$INSTALL_DIR"
        printf 'ref_kind=%s\n' "$REF_KIND"
        printf 'ref=%s\n' "$REF_NAME"
        printf 'commit=%s\n' "$RESOLVED_COMMIT"
        printf 'repo=%s\n' "$(redact_url "$REPO_URL")"
        printf 'version=%s\n' "$version"
        printf 'target=%s\n' "$TARGET_TRIPLE"
        printf 'installed_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    } >"$tmp"
    mv -f "$tmp" "$RECEIPT_FILE"
}

# ---------------------------------------------------------------------------------------
# Source: resolve the ref, clone it.
# ---------------------------------------------------------------------------------------

# git without prompts or the user's credential helpers leaking into output.
git_quiet() { GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false git "$@"; }

# Sets REF_KIND/REF_NAME/RESOLVED_COMMIT from the options, checking the ref exists.
resolve_ref() {
    local refs="" candidate sha
    if [ -n "$OPT_VERSION" ]; then
        refs="$(git_quiet ls-remote --tags "$REPO_URL" 2>/dev/null)" ||
            die "could not reach $(redact_url "$REPO_URL")." "Check your network connection and the repository URL."
        for candidate in "v${OPT_VERSION#v}" "${OPT_VERSION#v}" "$OPT_VERSION"; do
            sha="$(printf '%s\n' "$refs" | awk -v r="refs/tags/$candidate^{}" '$2 == r {print $1; exit}')"
            [ -n "$sha" ] || sha="$(printf '%s\n' "$refs" | awk -v r="refs/tags/$candidate" '$2 == r {print $1; exit}')"
            if [ -n "$sha" ]; then
                REF_KIND="tag"; REF_NAME="$candidate"; RESOLVED_COMMIT="$sha"
                return 0
            fi
        done
        die "no release tag '$OPT_VERSION' (or 'v${OPT_VERSION#v}') in $(redact_url "$REPO_URL")." \
            "List the tags with: git ls-remote --tags $(redact_url "$REPO_URL")" \
            "or install a branch with --branch $DEFAULT_BRANCH"
    fi
    REF_KIND="branch"
    REF_NAME="${OPT_BRANCH:-$DEFAULT_BRANCH}"
    refs="$(git_quiet ls-remote --heads "$REPO_URL" "refs/heads/$REF_NAME" 2>/dev/null)" ||
        die "could not reach $(redact_url "$REPO_URL")." "Check your network connection and the repository URL."
    sha="$(printf '%s\n' "$refs" | awk -v r="refs/heads/$REF_NAME" '$2 == r {print $1; exit}')"
    [ -n "$sha" ] || die "no branch '$REF_NAME' in $(redact_url "$REPO_URL")."
    RESOLVED_COMMIT="$sha"
}

fetch_source() {
    local src="$WORK_DIR/src"
    step "Cloning $(redact_url "$REPO_URL") ($REF_KIND $REF_NAME, ${RESOLVED_COMMIT:0:12})"
    git_quiet -c advice.detachedHead=false clone --quiet --depth 1 --branch "$REF_NAME" \
        "$REPO_URL" "$src" || die "git clone failed for $REF_KIND '$REF_NAME'."
    local head
    head="$(git -C "$src" rev-parse HEAD)"
    if [ "$head" != "$RESOLVED_COMMIT" ]; then
        # The branch moved between ls-remote and clone; build what was cloned.
        RESOLVED_COMMIT="$head"
    fi
    [ -f "$src/Cargo.toml" ] && [ -f "$src/Cargo.lock" ] && [ -f "$src/$CLI_PATH/Cargo.toml" ] ||
        die "the checkout has no $CLI_PATH/Cargo.toml or Cargo.lock; is this the QUBIVEX repository?"
    grep -q "^name = \"$CLI_PACKAGE\"" "$src/$CLI_PATH/Cargo.toml" ||
        die "$CLI_PATH/Cargo.toml is not the $CLI_PACKAGE package."
    grep -q "^name = \"$BIN_NAME\"" "$src/$CLI_PATH/Cargo.toml" ||
        die "$CLI_PATH/Cargo.toml does not declare a '$BIN_NAME' binary."
}

# The workspace's rust-version, e.g. "1.88"; empty when not declared.
workspace_rust_version() {
    sed -nE 's/^rust-version[[:space:]]*=[[:space:]]*"([0-9.]+)".*$/\1/p' "$WORK_DIR/src/Cargo.toml" | head -n 1
}

# ---------------------------------------------------------------------------------------
# Build: `cargo install --path apps/cli --locked` into a staging root inside WORK_DIR.
# --root keeps the user's ~/.cargo/bin and install metadata untouched until the final
# rename, and --no-track writes no .crates.toml there. The workspace's release profile
# (fat LTO, one codegen unit) applies: cargo builds the package inside its workspace.
# ---------------------------------------------------------------------------------------

build_from_source() {
    local stage="$WORK_DIR/stage" jobs_args=()
    [ -z "$OPT_JOBS" ] || jobs_args=(--jobs "$OPT_JOBS")
    step "Building $BIN_NAME (release, fat LTO: this takes several minutes)"
    info "cargo install --path $CLI_PATH --locked${OPT_JOBS:+ --jobs $OPT_JOBS}"
    if ! (
        # The staging bin on PATH silences cargo's "add it to your PATH" note about it.
        cd "$WORK_DIR/src" &&
            PATH="$stage/bin:$PATH" CARGO_TARGET_DIR="$WORK_DIR/target" "${CARGO_CMD[@]}" install \
                --path "$CLI_PATH" --locked --bin "$BIN_NAME" \
                --root "$stage" --no-track \
                ${jobs_args[@]+"${jobs_args[@]}"}
    ); then
        die "the build failed; nothing was installed and any existing $BIN_NAME is unchanged." \
            "Common causes: no C compiler, an outdated Rust, or too little memory for fat LTO" \
            "(try --jobs 2). Rerun with --no-cleanup to keep the build directory."
    fi
    STAGED_BIN="$stage/bin/$BIN_NAME"
    [ -x "$STAGED_BIN" ] || die "the build finished but produced no $BIN_NAME binary."
}

# Prints the binary's `--version` line if it is a working qubivex; fails otherwise.
binary_version() {
    local out
    out="$("$1" --version 2>/dev/null)" || return 1
    case "$out" in "$BIN_NAME "*) printf '%s' "$out" ;; *) return 1 ;; esac
}

# Checks the staged binary before it replaces anything.
check_staged() {
    local version
    step "Checking the new binary"
    version="$(binary_version "$STAGED_BIN")" ||
        die "the built binary does not run or is not $BIN_NAME; nothing was installed."
    "$STAGED_BIN" --help >/dev/null 2>&1 || die "the built binary fails '--help'; nothing was installed."
    info "$version"
}

# Copy next to the target, then rename: atomic on one filesystem, and a running or
# code-signed (macOS) old binary is replaced as a whole file, never rewritten in place.
install_binary() {
    local target="$INSTALL_DIR/$BIN_NAME" tmp="$INSTALL_DIR/.$BIN_NAME.new.$$" backup=""
    step "Installing to $target"
    if [ -e "$target" ]; then
        backup="$WORK_DIR/$BIN_NAME.previous"
        cp -p "$target" "$backup" || backup=""
    fi
    cp "$STAGED_BIN" "$tmp" || { rm -f "$tmp"; die "could not write to $INSTALL_DIR."; }
    chmod 755 "$tmp"
    mv -f "$tmp" "$target" || { rm -f "$tmp"; die "could not replace $target."; }

    if ! INSTALLED_VERSION="$(binary_version "$target")"; then
        if [ -n "$backup" ]; then
            cp -p "$backup" "$tmp" && mv -f "$tmp" "$target"
            die "the installed binary did not run; the previous one was restored."
        fi
        rm -f "$target"
        die "the installed binary did not run and was removed."
    fi
    info "$INSTALLED_VERSION"
}

# ---------------------------------------------------------------------------------------
# After install.
# ---------------------------------------------------------------------------------------

post_install_checks() {
    local bin="$INSTALL_DIR/$BIN_NAME"
    [ "$RUN_TESTS" -eq 1 ] || { info "post-install checks skipped (--no-test)"; return 0; }

    step "Hardware: qubivex system"
    if ! "$bin" system >&2; then
        warn "'qubivex system' reported an error. The install is fine; GPU backends are optional."
    fi
    step "Storage: qubivex doctor --storage --path $DATA_DIR/models --size-mib $DOCTOR_SIZE_MIB"
    if ! "$bin" doctor --storage --path "$DATA_DIR/models" --size-mib "$DOCTOR_SIZE_MIB" >&2; then
        warn "'qubivex doctor --storage' did not complete (it needs about 1.1 GiB free). The install is fine."
    fi
}

path_contains() {
    case ":${PATH}:" in *":$1:"*) return 0 ;; *) return 1 ;; esac
}

print_path_help() {
    local dir="$1" shell_name rc
    shell_name="$(basename "${SHELL:-sh}")"
    if path_contains "$dir"; then
        local found
        found="$(command -v "$BIN_NAME" 2>/dev/null || true)"
        if [ -n "$found" ] && [ "$found" != "$dir/$BIN_NAME" ]; then
            warn "another $BIN_NAME comes first on PATH: $found"
            info "Remove it or put $dir before its directory in PATH."
        fi
        return 0
    fi
    say ""
    say "${C_BOLD}$dir is not on your PATH.${C_RESET} Add it with:"
    case "$shell_name" in
        fish)
            say "  fish_add_path \"$dir\"" ;;
        zsh)
            say "  echo 'export PATH=\"$dir:\$PATH\"' >> ~/.zshrc && source ~/.zshrc" ;;
        bash)
            # Login shells on macOS read ~/.bash_profile; on Linux interactive shells read ~/.bashrc.
            # shellcheck disable=SC2088 # printed for the user to type, not expanded here
            rc='~/.bashrc'
            # shellcheck disable=SC2088
            if [ "$OS" = "macos" ]; then rc='~/.bash_profile'; fi
            say "  echo 'export PATH=\"$dir:\$PATH\"' >> $rc && source $rc" ;;
        *)
            say "  export PATH=\"$dir:\$PATH\"    # add this line to your shell's startup file" ;;
    esac
    if [ "$dir" = "$(default_install_dir)" ] && [ -f "${CARGO_HOME:-$HOME/.cargo}/env" ]; then
        say "  (or, for a rustup setup: . \"${CARGO_HOME:-$HOME/.cargo}/env\")"
    fi
}

# ---------------------------------------------------------------------------------------
# Actions.
# ---------------------------------------------------------------------------------------

# Future: a `binary` method will query GitHub Releases for $TARGET_TRIPLE, download the
# archive and its SHA256SUMS, verify, extract to $WORK_DIR/stage/bin and set STAGED_BIN.
# Everything after acquire_binary (check, atomic install, receipt) is shared.
acquire_binary() {
    local need
    check_git
    check_c_compiler
    step "Checking Rust"
    ensure_rust "$FALLBACK_MIN_RUST"
    resolve_ref
    fetch_source
    need="$(workspace_rust_version)"
    if [ -n "$need" ] && ! version_ge "$(rustc_version)" "$need"; then
        ensure_rust "$need"
    fi
    build_from_source
}

do_install() {
    local previous=""
    step "Installing QUBIVEX for $OS/$ARCH"
    resolve_install_dir "${OPT_INSTALL_DIR:-$(default_install_dir)}"
    if [ -x "$INSTALL_DIR/$BIN_NAME" ]; then
        previous="$(binary_version "$INSTALL_DIR/$BIN_NAME" || echo "an unknown build")"
        info "existing install: $INSTALL_DIR/$BIN_NAME ($previous); it is kept until the new build succeeds"
    fi
    make_work_dir
    acquire_binary
    check_staged
    install_binary
    mkdir -p "$DATA_DIR/models"
    write_receipt "$INSTALLED_VERSION"
    post_install_checks
    say ""
    step "Installed $INSTALLED_VERSION ($REF_KIND $REF_NAME, ${RESOLVED_COMMIT:0:12})"
    info "binary: $INSTALL_DIR/$BIN_NAME"
    info "models: $DATA_DIR/models"
    print_path_help "$INSTALL_DIR"
    say ""
    say "Next: qubivex system    |   qubivex models pull Qwen/Qwen3-0.6B   |   qubivex --help"
}

do_update() {
    local installed_bin installed_commit ref_kind ref_name
    installed_bin="$(receipt_get binary)"
    if [ -z "$installed_bin" ]; then
        installed_bin="${OPT_INSTALL_DIR:+$OPT_INSTALL_DIR/$BIN_NAME}"
        [ -n "$installed_bin" ] || installed_bin="$(default_install_dir)/$BIN_NAME"
    fi
    if [ ! -x "$installed_bin" ]; then
        die "no QUBIVEX installation found (looked for $installed_bin)." \
            "Install it first: run this script without --update."
    fi
    [ -n "$OPT_INSTALL_DIR" ] || OPT_INSTALL_DIR="$(dirname "$installed_bin")"
    ref_kind="$(receipt_get ref_kind)"
    ref_name="$(receipt_get ref)"
    installed_commit="$(receipt_get commit)"
    if [ -z "$OPT_BRANCH" ]; then
        if [ "$ref_kind" = "tag" ]; then
            die "the installation is pinned to tag '$ref_name'; --update follows branches." \
                "Install another tag with --version VERSION, or follow a branch with --update --branch $DEFAULT_BRANCH"
        fi
        OPT_BRANCH="${ref_name:-$DEFAULT_BRANCH}"
    fi
    check_git
    resolve_ref
    if [ -n "$installed_commit" ] && [ "$installed_commit" = "$RESOLVED_COMMIT" ] &&
        [ "$(dirname "$installed_bin")" = "$OPT_INSTALL_DIR" ] &&
        binary_version "$installed_bin" >/dev/null; then
        step "Already up to date: $(binary_version "$installed_bin") (branch $REF_NAME, ${RESOLVED_COMMIT:0:12})"
        return 0
    fi
    info "updating ${installed_commit:+${installed_commit:0:12} -> }${RESOLVED_COMMIT:0:12} on branch $REF_NAME"
    do_install
}

do_uninstall() {
    local bin removed=0 dir
    bin="$(receipt_get binary)"
    if [ -n "$OPT_INSTALL_DIR" ]; then
        resolve_install_dir "$OPT_INSTALL_DIR"
        bin="$INSTALL_DIR/$BIN_NAME"
    fi
    [ -n "$bin" ] || bin="$(default_install_dir)/$BIN_NAME"

    step "Uninstalling QUBIVEX"
    if [ -e "$bin" ]; then
        if [ -x "$bin" ] && ! binary_version "$bin" >/dev/null; then
            die "$bin does not look like the qubivex CLI (its --version is not 'qubivex ...'); not removing it."
        fi
        dir="$(dirname "$bin")"
        [ -w "$dir" ] || die "no write permission for $dir; cannot remove $bin." \
            "If it was installed with sudo, remove it the same way: sudo rm \"$bin\""
        rm -f "$bin"
        removed=1
        info "removed $bin"
    else
        info "no binary at $bin"
    fi
    if [ -f "$RECEIPT_FILE" ]; then
        rm -f "$RECEIPT_FILE"
        info "removed $RECEIPT_FILE"
    fi
    local other
    other="$(command -v "$BIN_NAME" 2>/dev/null || true)"
    if [ -n "$other" ]; then
        warn "another $BIN_NAME is still on PATH: $other (not installed by this script's settings; left alone)"
    fi
    if [ -d "$DATA_DIR" ]; then
        info "kept your data: $DATA_DIR (models and configuration). Delete it yourself if you no longer need it."
    fi
    if [ "$removed" -eq 1 ]; then step "QUBIVEX uninstalled"; else step "Nothing to uninstall"; fi
}

main() {
    parse_args "$@"
    [ -n "${HOME:-}" ] || die "HOME is not set."
    unset CDPATH
    RECEIPT_FILE="$DATA_DIR/install-receipt"
    trap cleanup EXIT
    trap 'on_error "$LINENO"' ERR
    trap 'exit 130' INT
    trap 'exit 143' TERM
    umask 022
    detect_platform
    case "$ACTION" in
        install) do_install ;;
        update) do_update ;;
        uninstall) do_uninstall ;;
    esac
}

STAGED_BIN=""
INSTALLED_VERSION=""
main "$@"
