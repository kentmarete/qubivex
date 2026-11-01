//! The `qubivex` binary. Logic lives in the library (RUST_GUIDELINES §8).

use std::process::ExitCode;

fn main() -> ExitCode {
    qubivex_cli::run(std::env::args_os())
}
