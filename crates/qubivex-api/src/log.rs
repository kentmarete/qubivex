//! The gateway's log (spec §8): one line per event, as text or as one JSON object.
//!
//! Every line names its event, and events about a request carry its request id, its
//! model id and, once it is admitted, its sequence id. A line is built from an event
//! name and fields the gateway computed (ids, counts, durations, status and error
//! codes): nothing here takes a prompt, a message or generated text, so none can be
//! logged. The one free-text field is `detail`, the runtime's own description of an
//! internal error, which the client never sees.

use std::fmt::Write as _;
use std::io::Write;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

/// How the gateway writes its log.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LogFormat {
    /// `qubivex: <time> <level> <event> key=value ...`
    #[default]
    Text,
    /// One JSON object per line: `ts`, `level`, `event`, then the event's fields.
    Json,
    /// Nothing is written.
    Off,
}

impl LogFormat {
    /// The format named `text`, `json` or `off`.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "text" => Some(Self::Text),
            "json" => Some(Self::Json),
            "off" => Some(Self::Off),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Level {
    Info,
    Warn,
    Error,
}

impl Level {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
        }
    }
}

/// One field of a log line. The key is fixed in the source.
pub(crate) type Field = (&'static str, Value);

/// Where log lines go: standard error, or a writer a host or a test supplied.
pub(crate) struct Log {
    format: LogFormat,
    sink: Option<Mutex<Box<dyn Write + Send>>>,
}

impl std::fmt::Debug for Log {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Log")
            .field("format", &self.format)
            .finish_non_exhaustive()
    }
}

/// `secs` after the Unix epoch as `YYYY-MM-DDTHH:MM:SS.mmmZ` (UTC).
fn rfc3339(secs: u64, millis: u32) -> String {
    let days = secs / 86_400;
    let rem = secs % 86_400;
    // Days to a civil date (Howard Hinnant's algorithm), for dates from 1970 on.
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z % 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        rem / 3_600,
        rem % 3_600 / 60,
        rem % 60
    )
}

fn now() -> String {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    rfc3339(d.as_secs(), d.subsec_millis())
}

/// A value in a text line: bare when that is unambiguous, quoted as JSON otherwise.
fn text_value(out: &mut String, v: &Value) {
    match v {
        Value::String(s)
            if !s.is_empty()
                && s.bytes()
                    .all(|b| b.is_ascii_graphic() && b != b'"' && b != b'=') =>
        {
            out.push_str(s);
        }
        other => {
            let _ = write!(out, "{other}");
        }
    }
}

impl Log {
    /// A log in `format` on standard error.
    pub(crate) const fn stderr(format: LogFormat) -> Self {
        Self { format, sink: None }
    }

    /// A log in `format` on `sink`.
    pub(crate) fn to(format: LogFormat, sink: Box<dyn Write + Send>) -> Self {
        Self {
            format,
            sink: Some(Mutex::new(sink)),
        }
    }

    fn line(&self, time: &str, level: Level, event: &str, fields: &[Field]) -> Option<String> {
        let mut out = String::with_capacity(160);
        match self.format {
            LogFormat::Off => return None,
            LogFormat::Json => {
                let _ = write!(
                    out,
                    "{{\"ts\":\"{time}\",\"level\":\"{}\",\"event\":\"{event}\"",
                    level.as_str()
                );
                for (k, v) in fields {
                    // `Value`'s `Display` is compact JSON, with strings escaped.
                    let _ = write!(out, ",\"{k}\":{v}");
                }
                out.push('}');
            }
            LogFormat::Text => {
                let _ = write!(out, "qubivex: {time} {} {event}", level.as_str());
                for (k, v) in fields {
                    let _ = write!(out, " {k}=");
                    text_value(&mut out, v);
                }
            }
        }
        Some(out)
    }

    /// Writes one event. A log that cannot be written is not an error to anyone.
    pub(crate) fn event(&self, level: Level, event: &'static str, fields: &[Field]) {
        let Some(line) = self.line(&now(), level, event, fields) else {
            return;
        };
        match &self.sink {
            // `eprintln!`, so a test harness captures it.
            None => eprintln!("{line}"),
            Some(sink) => {
                if let Ok(mut w) = sink.lock() {
                    let _ = writeln!(w, "{line}");
                    let _ = w.flush();
                }
            }
        }
    }

    pub(crate) fn info(&self, event: &'static str, fields: &[Field]) {
        self.event(Level::Info, event, fields);
    }

    pub(crate) fn warn(&self, event: &'static str, fields: &[Field]) {
        self.event(Level::Warn, event, fields);
    }

    pub(crate) fn error(&self, event: &'static str, fields: &[Field]) {
        self.event(Level::Error, event, fields);
    }
}

/// Milliseconds, to a microsecond, for a log field.
pub(crate) fn millis(d: std::time::Duration) -> Value {
    Value::from((d.as_secs_f64() * 1e6).round() / 1e3)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use serde_json::{Value, json};

    use super::{Level, Log, LogFormat, rfc3339};

    #[test]
    fn times_are_utc_rfc3339() {
        assert_eq!(rfc3339(0, 0), "1970-01-01T00:00:00.000Z");
        assert_eq!(rfc3339(951_782_400, 7), "2000-02-29T00:00:00.007Z");
        assert_eq!(rfc3339(1_700_000_000, 120), "2023-11-14T22:13:20.120Z");
        assert_eq!(rfc3339(1_735_689_599, 999), "2024-12-31T23:59:59.999Z");
    }

    #[test]
    fn a_json_line_is_one_object_with_its_fields_escaped() {
        let log = Log::stderr(LogFormat::Json);
        let fields = [
            ("request_id", json!("req-1")),
            ("status", json!(200)),
            ("detail", json!("a \"quoted\"\nline")),
        ];
        let line = log.line("T", Level::Warn, "request", &fields).unwrap();
        assert!(!line.contains('\n'));
        let v: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(
            v,
            json!({
                "ts": "T", "level": "warn", "event": "request", "request_id": "req-1",
                "status": 200, "detail": "a \"quoted\"\nline",
            })
        );
    }

    #[test]
    fn a_text_line_quotes_only_what_needs_it() {
        let log = Log::stderr(LogFormat::Text);
        let fields = [
            ("model", json!("tiny-model")),
            ("detail", json!("two words")),
            ("status", json!(429)),
        ];
        assert_eq!(
            log.line("T", Level::Info, "request", &fields).unwrap(),
            "qubivex: T info request model=tiny-model detail=\"two words\" status=429"
        );
        assert_eq!(
            Log::stderr(LogFormat::Off).line("T", Level::Info, "request", &fields),
            None
        );
    }
}
