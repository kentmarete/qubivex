use std::path::PathBuf;
use std::time::Duration;

use thiserror::Error;

/// Errors raised while probing the host.
///
/// Probes turn these into [`crate::Probed::Unknown`] reasons. They surface as errors
/// only where a caller asked for one specific value.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum HardwareError {
    /// A file could not be read.
    #[error("cannot read {}", path.display())]
    Io {
        /// The file.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },

    /// A helper program is not installed or not on `PATH`.
    #[error("{program} not found")]
    NotFound {
        /// The program name.
        program: String,
    },

    /// A helper program could not be started or waited on.
    #[error("cannot run {program}")]
    Spawn {
        /// The program name.
        program: String,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },

    /// A helper program did not finish in time and was killed.
    #[error("{program} did not finish within {:?}", timeout)]
    Timeout {
        /// The program name.
        program: String,
        /// The time limit.
        timeout: Duration,
    },

    /// A helper program exited unsuccessfully.
    #[error("{program} failed ({status}): {stderr}")]
    CommandFailed {
        /// The program name.
        program: String,
        /// The exit status, as text.
        status: String,
        /// The first line of its standard error.
        stderr: String,
    },

    /// Probe output did not have the expected format.
    #[error("cannot parse {what}: {detail}")]
    Parse {
        /// What was being parsed, for example "/proc/meminfo".
        what: String,
        /// What was wrong.
        detail: String,
    },
}

impl HardwareError {
    pub(crate) fn parse(what: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::Parse {
            what: what.into(),
            detail: detail.into(),
        }
    }
}
