//! Typed configuration (QUBIVEX_DESIGN §6).
//!
//! Each section is added in the phase that first reads it. Loading and layering
//! (defaults, user file, `--config`, `QUBIVEX_*` environment, CLI flags) lives in the
//! applications; this module only defines the types and their defaults.

use std::fmt;

use serde::{Deserialize, Serialize};

/// The effective QUBIVEX configuration.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct QubivexConfig {
    /// Logging.
    pub log: LogConfig,
}

/// Logging configuration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct LogConfig {
    /// A `tracing` filter directive, for example `info` or `qubivex=debug,warn`.
    pub filter: String,
    /// Output format.
    pub format: LogFormat,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            filter: "warn".to_owned(),
            format: LogFormat::Text,
        }
    }
}

/// Log output format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    /// Human-readable text.
    #[default]
    Text,
    /// One JSON object per line.
    Json,
}

impl fmt::Display for LogFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Text => "text",
            Self::Json => "json",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_fields_take_defaults() {
        let c: QubivexConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(c, QubivexConfig::default());
        let c: QubivexConfig = serde_json::from_str(r#"{"log":{"format":"json"}}"#).unwrap();
        assert_eq!(c.log.format, LogFormat::Json);
        assert_eq!(c.log.filter, "warn");
    }

    #[test]
    fn unknown_fields_are_rejected() {
        // A typo in a config key must not be silently ignored.
        assert!(serde_json::from_str::<QubivexConfig>(r#"{"logg":{}}"#).is_err());
        assert!(serde_json::from_str::<QubivexConfig>(r#"{"log":{"fmt":"json"}}"#).is_err());
    }
}
