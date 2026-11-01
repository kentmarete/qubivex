//! Repository checks, run as `cargo xtask <command>`.
//!
//! - `check-deps`: enforces the crate dependency direction (QUBIVEX_DESIGN §3.1).
//! - `check-model-names`: rejects model-name literals in shared code (design invariant 3).
//! - `check`: runs both.

mod deps;
mod model_names;

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    let root = workspace_root();
    let result = match cmd.as_str() {
        "check-deps" => check_deps(&root),
        "check-model-names" => check_model_names(&root),
        "check" => check_deps(&root).and_then(|()| check_model_names(&root)),
        _ => Err("usage: cargo xtask <check | check-deps | check-model-names>".to_owned()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::FAILURE
        }
    }
}

fn workspace_root() -> PathBuf {
    // xtask lives at <root>/xtask.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
}

fn check_deps(root: &Path) -> Result<(), String> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let out = Command::new(cargo)
        .args([
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
            "--offline",
        ])
        .current_dir(root)
        .output()
        .map_err(|e| format!("failed to run cargo metadata: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo metadata failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let meta: deps::Metadata = serde_json::from_slice(&out.stdout)
        .map_err(|e| format!("cannot parse cargo metadata: {e}"))?;
    let violations = deps::violations(&meta);
    if violations.is_empty() {
        println!("check-deps: ok ({} workspace crates)", meta.packages.len());
        Ok(())
    } else {
        Err(format!(
            "check-deps: {} violation(s) of QUBIVEX_DESIGN §3.1:\n  {}",
            violations.len(),
            violations.join("\n  ")
        ))
    }
}

fn check_model_names(root: &Path) -> Result<(), String> {
    let hits = model_names::scan(root).map_err(|e| format!("check-model-names: {e}"))?;
    if hits.is_empty() {
        println!("check-model-names: ok");
        Ok(())
    } else {
        Err(format!(
            "check-model-names: model names in shared code (design invariant 3). \
             Describe models with manifests instead:\n  {}",
            hits.join("\n  ")
        ))
    }
}
