//! Records how the binary was built, for `qubivex bench` records: git commit, rustc,
//! profile, opt-level, target, compile-time target features and RUSTFLAGS.

use std::process::Command;

fn output(cmd: &str, args: &[&str]) -> Option<String> {
    let o = Command::new(cmd).args(args).output().ok()?;
    o.status
        .success()
        .then(|| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .filter(|s| !s.is_empty())
}

fn main() {
    let env = |k: &str| std::env::var(k).unwrap_or_default();
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let git = output("git", &["rev-parse", "--short=12", "HEAD"]).map_or_else(
        || "unknown".to_owned(),
        |rev| {
            // Uncommitted changes make the commit alone an incomplete description.
            let dirty = output("git", &["status", "--porcelain", "--untracked-files=no"]).is_some();
            if dirty { format!("{rev}-dirty") } else { rev }
        },
    );
    let vars = [
        ("QUBIVEX_BUILD_GIT", git),
        (
            "QUBIVEX_BUILD_RUSTC",
            output(&rustc, &["-V"]).unwrap_or_else(|| "unknown".to_owned()),
        ),
        ("QUBIVEX_BUILD_PROFILE", env("PROFILE")),
        ("QUBIVEX_BUILD_OPT_LEVEL", env("OPT_LEVEL")),
        ("QUBIVEX_BUILD_TARGET", env("TARGET")),
        (
            "QUBIVEX_BUILD_TARGET_FEATURES",
            env("CARGO_CFG_TARGET_FEATURE"),
        ),
        // Flags are separated by 0x1f in the encoded form.
        (
            "QUBIVEX_BUILD_RUSTFLAGS",
            env("CARGO_ENCODED_RUSTFLAGS").replace('\u{1f}', " "),
        ),
    ];
    for (k, v) in vars {
        println!("cargo:rustc-env={k}={v}");
    }
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    // A commit, a staged change or an edit to any source (directories are scanned
    // recursively) changes the git description.
    for p in [
        "../../.git/HEAD",
        "../../.git/index",
        "../../Cargo.toml",
        "../../Cargo.lock",
        "../../crates",
        "src",
    ] {
        println!("cargo:rerun-if-changed={p}");
    }
}
