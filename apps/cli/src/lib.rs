//! The `qubivex` command-line interface.
//!
//! Subcommands arrive with the phases that implement them (MIGRATION_PLAN §3):
//! `system` (Phase 2); `models`, `plan`, `run` and `serve` later.

mod system;

use std::ffi::OsString;
use std::io::Write;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use qubivex_hardware::{Host, ResourceProbe, SystemProbe};

/// Run Mixture-of-Experts models with expert streaming.
#[derive(Debug, Parser)]
#[command(name = "qubivex", version = qubivex_core::VERSION, about, long_about = None)]
#[command(arg_required_else_help = true)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Show the hardware profile and the resources available now.
    System {
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
}

/// Parses `args` (including the program name) and runs the command.
///
/// Returns the process exit code: 0 on success, 1 on a runtime error, 2 on a usage
/// error.
pub fn run<I, T>(args: I) -> ExitCode
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    match Cli::try_parse_from(args) {
        Ok(cli) => match execute(&cli.command) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("qubivex: {e}");
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            // `--help` and `--version` arrive here as "errors" with exit code 0.
            let code = e.exit_code();
            // Printing can fail only on a closed stdout/stderr, where nothing useful remains to do.
            let _ = e.print();
            ExitCode::from(u8::try_from(code).unwrap_or(2))
        }
    }
}

fn execute(cmd: &Command) -> Result<(), String> {
    match cmd {
        Command::System { json } => {
            let host = Host::live();
            let profile = qubivex_hardware::probe(&host);
            let snapshot = SystemProbe::new(host)
                .snapshot()
                .map_err(|e| e.to_string())?;
            let text = if *json {
                let report = system::Report {
                    profile: &profile,
                    snapshot: &snapshot,
                };
                let mut t = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
                t.push('\n');
                t
            } else {
                system::render(&profile, &snapshot)
            };
            std::io::stdout()
                .lock()
                .write_all(text.as_bytes())
                .map_err(|e| format!("cannot write output: {e}"))
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
