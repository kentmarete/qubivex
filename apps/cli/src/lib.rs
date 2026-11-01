//! The `qubivex` command-line interface.
//!
//! Phase 1 provides only `--version` and `--help`. Subcommands (`system`, `models`,
//! `plan`, `run`, `serve`) are added in the phases that implement them
//! (MIGRATION_PLAN §3).

use std::ffi::OsString;
use std::process::ExitCode;

use clap::Parser;

/// Run Mixture-of-Experts models with expert streaming.
#[derive(Debug, Parser)]
#[command(name = "qubivex", version = qubivex_core::VERSION, about, long_about = None)]
#[command(arg_required_else_help = true)]
pub struct Cli {}

/// Parses `args` (including the program name) and runs the command.
///
/// Returns the process exit code: 0 on success, 2 on a usage error.
pub fn run<I, T>(args: I) -> ExitCode
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    match Cli::try_parse_from(args) {
        Ok(Cli {}) => ExitCode::SUCCESS,
        Err(e) => {
            // `--help` and `--version` arrive here as "errors" with exit code 0.
            let code = e.exit_code();
            // Printing can fail only on a closed stdout/stderr, where nothing useful remains to do.
            let _ = e.print();
            ExitCode::from(u8::try_from(code).unwrap_or(2))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }
}
