//! The `qubivex` command-line interface.
//!
//! Subcommands arrive with the phases that implement them (MIGRATION_PLAN §3):
//! `system` and `doctor` (Phase 2); `models`, `plan`, `run` and `serve` later.

mod doctor;
mod system;

use std::ffi::OsString;
use std::io::Write;
use std::path::PathBuf;
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
    /// Measure things `system` cannot see by probing.
    Doctor {
        /// Measure read throughput and 4 MiB latency of the filesystem holding --path.
        /// Writes and removes a scratch file there.
        #[arg(long)]
        storage: bool,
        /// Directory on the filesystem to measure.
        #[arg(long, default_value = ".")]
        path: PathBuf,
        /// Scratch file size in MiB (at least 64).
        #[arg(long, default_value_t = 1024)]
        size_mib: u64,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
}

/// An error and its chain of causes, `: `-separated.
fn chain(e: &dyn std::error::Error) -> String {
    let mut s = e.to_string();
    let mut cur = e.source();
    while let Some(c) = cur {
        s.push_str(": ");
        s.push_str(&c.to_string());
        cur = c.source();
    }
    s
}

fn emit(text: &str) -> Result<(), String> {
    std::io::stdout()
        .lock()
        .write_all(text.as_bytes())
        .map_err(|e| format!("cannot write output: {e}"))
}

fn to_json(v: &impl serde::Serialize) -> Result<String, String> {
    let mut t = serde_json::to_string_pretty(v).map_err(|e| e.to_string())?;
    t.push('\n');
    Ok(t)
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
            let snapshot = SystemProbe::new(host).snapshot().map_err(|e| chain(&e))?;
            let text = if *json {
                to_json(&system::Report {
                    profile: &profile,
                    snapshot: &snapshot,
                })?
            } else {
                system::render(&profile, &snapshot)
            };
            emit(&text)
        }
        Command::Doctor {
            storage,
            path,
            size_mib,
            json,
        } => {
            if !storage {
                return Err("nothing to check; pass --storage".to_owned());
            }
            emit(&doctor::storage(path, *size_mib, *json)?)
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
