//! The `qubivex` command-line interface.
//!
//! Subcommands arrive with the phases that implement them (MIGRATION_PLAN §3):
//! `system` and `doctor` (Phase 2), `models` (Phase 3), `run` and `bench` (Phase 4);
//! `plan` and `serve` later.

mod bench;
mod doctor;
mod models;
mod run;
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
    /// Generate text with a model (CPU).
    Run {
        /// A model id, or a model directory.
        #[arg(long)]
        model: String,
        /// A model directory to search for the id (repeatable).
        #[arg(long = "dir")]
        dirs: Vec<PathBuf>,
        /// The prompt.
        #[arg(long)]
        prompt: String,
        /// Wrap the prompt as a user message with the model's chat template.
        #[arg(long)]
        chat: bool,
        /// Most tokens to generate.
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// Sampling temperature; 0 is greedy.
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Nucleus sampling mass.
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        /// Random seed for sampling.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Expert cache size in MiB (default: every expert resident).
        #[arg(long)]
        expert_cache_mb: Option<u64>,
        /// Context limit in tokens (default: the model's).
        #[arg(long)]
        context: Option<usize>,
        /// Compute threads (default: the CPU's performance cores).
        #[arg(long)]
        threads: Option<usize>,
        /// Do not print timing and cache statistics.
        #[arg(long)]
        quiet: bool,
    },
    /// Time prefill and decode, and record them with hardware, build flags, peak
    /// memory and a token check (docs/performance/BENCHMARKING.md).
    Bench {
        /// A model id, or a model directory.
        #[arg(long)]
        model: String,
        /// A model directory to search for the id (repeatable).
        #[arg(long = "dir")]
        dirs: Vec<PathBuf>,
        /// The prompt (encoded as `run` encodes it, without a chat template).
        #[arg(long, default_value = bench::DEFAULT_PROMPT)]
        prompt: String,
        /// Prompt length in tokens: the encoded prompt repeated or cut to this length.
        /// Default: the encoded prompt as is.
        #[arg(long)]
        prompt_tokens: Option<usize>,
        /// Tokens generated per run (the first comes from prefill).
        #[arg(long, default_value_t = 128)]
        decode_tokens: usize,
        /// Unmeasured runs first (they also fill the expert cache).
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        /// Measured runs.
        #[arg(long, default_value_t = 5)]
        runs: usize,
        /// Compute threads (default: the CPU's performance cores).
        #[arg(long)]
        threads: Option<usize>,
        /// Expert cache size in MiB (default: every expert resident).
        #[arg(long)]
        expert_cache_mb: Option<u64>,
        /// Context limit in tokens (default: the model's).
        #[arg(long)]
        context: Option<usize>,
        /// A label stored in the record, for example `before` or `after`.
        #[arg(long)]
        label: Option<String>,
        /// A previous record (or a JSON array of token ids) whose tokens must match.
        #[arg(long)]
        reference: Option<PathBuf>,
        /// Also write the JSON record to this file.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Print the JSON record instead of text.
        #[arg(long)]
        json: bool,
    },
    /// List, inspect and import models.
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ModelsCommand {
    /// List the models in the model directories.
    List {
        /// A model directory to scan (repeatable). Default: $QUBIVEX_MODELS_DIRS, else ~/.qubivex/models.
        #[arg(long = "dir")]
        dirs: Vec<PathBuf>,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Show a model's dimensions, bytes, files and tensors.
    Inspect {
        /// A model id, or a model directory.
        model: String,
        /// A model directory to search for the id (repeatable).
        #[arg(long = "dir")]
        dirs: Vec<PathBuf>,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
        /// List every tensor with its dtype, shape and role.
        #[arg(long)]
        tensors: bool,
    },
    /// Write a manifest for a checkpoint directory or a .gguf file.
    Import {
        /// The checkpoint directory, or a .gguf file.
        path: PathBuf,
        /// Model id (default: from the directory or file name).
        #[arg(long)]
        id: Option<String>,
        /// Display name.
        #[arg(long)]
        name: Option<String>,
        /// A config value the checkpoint does not record, as KEY=VALUE (repeatable).
        #[arg(long = "set")]
        set: Vec<String>,
        /// Write manifest.json next to the weights instead of printing it.
        #[arg(long)]
        write: bool,
        /// Replace an existing manifest.json.
        #[arg(long)]
        force: bool,
    },
    /// List the architectures this build supports.
    Architectures,
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
        Command::Run {
            model,
            dirs,
            prompt,
            chat,
            max_tokens,
            temperature,
            top_p,
            seed,
            expert_cache_mb,
            context,
            threads,
            quiet,
        } => run::run(&run::RunArgs {
            model,
            dirs,
            prompt,
            chat: *chat,
            max_tokens: *max_tokens,
            sampling: qubivex_runtime::Sampling {
                temperature: *temperature,
                top_p: *top_p,
                seed: *seed,
            },
            expert_cache_mb: *expert_cache_mb,
            context: *context,
            threads: *threads,
            quiet: *quiet,
        }),
        Command::Bench {
            model,
            dirs,
            prompt,
            prompt_tokens,
            decode_tokens,
            warmup,
            runs,
            threads,
            expert_cache_mb,
            context,
            label,
            reference,
            out,
            json,
        } => emit(&bench::bench(&bench::BenchArgs {
            model,
            dirs,
            prompt,
            prompt_tokens: *prompt_tokens,
            decode_tokens: *decode_tokens,
            warmup: *warmup,
            runs: *runs,
            threads: *threads,
            expert_cache_mb: *expert_cache_mb,
            context: *context,
            label: label.as_deref(),
            reference: reference.as_deref(),
            out: out.as_deref(),
            json: *json,
        })?),
        Command::Models { command } => emit(&match command {
            ModelsCommand::List { dirs, json } => models::list(dirs, *json)?,
            ModelsCommand::Inspect {
                model,
                dirs,
                json,
                tensors,
            } => models::inspect(model, dirs, *json, *tensors)?,
            ModelsCommand::Import {
                path,
                id,
                name,
                set,
                write,
                force,
            } => models::import_cmd(&models::ImportArgs {
                path,
                id: id.clone(),
                name: name.clone(),
                set,
                write: *write,
                force: *force,
            })?,
            ModelsCommand::Architectures => models::architectures(),
        }),
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
