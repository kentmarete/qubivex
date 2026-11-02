//! Values that a probe either measured or could not determine.

use serde::{Deserialize, Serialize};

/// A probed value: [`Probed::Known`], or [`Probed::Unknown`] with the reason.
///
/// A failed probe is never reported as zero (QUBIVEX_DESIGN §4.2). A planner that
/// needs the value must decide what an unknown means, explicitly.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Probed<T> {
    /// The value was measured.
    Known(T),
    /// The value could not be determined, and why.
    Unknown(String),
}

impl<T> Probed<T> {
    /// An unknown value with the given reason.
    pub fn unknown(reason: impl Into<String>) -> Self {
        Self::Unknown(reason.into())
    }

    /// The value, if known.
    #[must_use]
    pub const fn known(&self) -> Option<&T> {
        match self {
            Self::Known(v) => Some(v),
            Self::Unknown(_) => None,
        }
    }

    /// The value by value, if known.
    #[must_use]
    pub fn into_known(self) -> Option<T> {
        match self {
            Self::Known(v) => Some(v),
            Self::Unknown(_) => None,
        }
    }

    /// Maps a known value.
    #[must_use]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Probed<U> {
        match self {
            Self::Known(v) => Probed::Known(f(v)),
            Self::Unknown(r) => Probed::Unknown(r),
        }
    }

    /// Chains a probe that needs this value.
    #[must_use]
    pub fn and_then<U>(self, f: impl FnOnce(T) -> Probed<U>) -> Probed<U> {
        match self {
            Self::Known(v) => f(v),
            Self::Unknown(r) => Probed::Unknown(r),
        }
    }
}

impl<T, E: std::fmt::Display> From<Result<T, E>> for Probed<T> {
    fn from(r: Result<T, E>) -> Self {
        match r {
            Ok(v) => Self::Known(v),
            Err(e) => Self::Unknown(e.to_string()),
        }
    }
}

impl<T: std::fmt::Display> std::fmt::Display for Probed<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Known(v) => v.fmt(f),
            Self::Unknown(r) => write!(f, "unknown ({r})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_tagged() {
        let k: Probed<u32> = Probed::Known(3);
        let u: Probed<u32> = Probed::unknown("no sysfs");
        assert_eq!(serde_json::to_string(&k).unwrap(), r#"{"known":3}"#);
        assert_eq!(
            serde_json::to_string(&u).unwrap(),
            r#"{"unknown":"no sysfs"}"#
        );
    }

    #[test]
    fn display_and_combinators() {
        let u: Probed<u32> = Probed::unknown("x");
        assert_eq!(u.to_string(), "unknown (x)");
        assert_eq!(Probed::Known(2).map(|v: u32| v * 2), Probed::Known(4));
        assert_eq!(u.and_then(|v| Probed::Known(v + 1)), Probed::unknown("x"));
    }
}
