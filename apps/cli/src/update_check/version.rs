//! Semantic versions (semver.org 2.0.0) as far as precedence needs them: `MAJOR.MINOR.PATCH`,
//! an optional `-prerelease` and an ignored `+build`. A leading `v` is accepted, as tags
//! (`v0.2.0`) and `install.sh --version` spell it.

use std::cmp::Ordering;
use std::fmt;

/// One dot-separated prerelease identifier.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Ident {
    Numeric(u64),
    Alpha(String),
}

impl Ord for Ident {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Numeric(a), Self::Numeric(b)) => a.cmp(b),
            (Self::Alpha(a), Self::Alpha(b)) => a.cmp(b),
            // Numeric identifiers have lower precedence than alphanumeric ones (§11.4.3).
            (Self::Numeric(_), Self::Alpha(_)) => Ordering::Less,
            (Self::Alpha(_), Self::Numeric(_)) => Ordering::Greater,
        }
    }
}

impl PartialOrd for Ident {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A parsed version. Equality and order follow semver precedence, so build metadata is
/// not part of either.
#[derive(Clone, Debug)]
pub(crate) struct Version {
    major: u64,
    minor: u64,
    patch: u64,
    pre: Vec<Ident>,
    text: String,
}

fn number(s: &str) -> Option<u64> {
    let digits = !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    // No leading zeros (§2, §9).
    (digits && (s == "0" || !s.starts_with('0')))
        .then(|| s.parse().ok())
        .flatten()
}

impl Version {
    /// Parses `s`, or `None` if it is not a semantic version.
    pub(crate) fn parse(s: &str) -> Option<Self> {
        let text = s.trim();
        let core = text.strip_prefix('v').unwrap_or(text);
        let core = match core.split_once('+') {
            Some((c, build)) => {
                let ok = build
                    .split('.')
                    .all(|id| !id.is_empty() && id.bytes().all(is_ident_byte));
                ok.then_some(c)?
            }
            None => core,
        };
        let (nums, pre) = match core.split_once('-') {
            Some((n, p)) => (n, Some(p)),
            None => (core, None),
        };
        let mut parts = nums.split('.');
        let major = number(parts.next()?)?;
        let minor = number(parts.next()?)?;
        let patch = number(parts.next()?)?;
        if parts.next().is_some() {
            return None;
        }
        let pre = match pre {
            None => Vec::new(),
            Some(p) => p
                .split('.')
                .map(|id| {
                    if id.is_empty() || !id.bytes().all(is_ident_byte) {
                        None
                    } else if id.bytes().all(|b| b.is_ascii_digit()) {
                        number(id).map(Ident::Numeric)
                    } else {
                        Some(Ident::Alpha(id.to_owned()))
                    }
                })
                .collect::<Option<Vec<_>>>()?,
        };
        Some(Self {
            major,
            minor,
            patch,
            pre,
            text: core.to_owned(),
        })
    }
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'-'
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.major, self.minor, self.patch)
            .cmp(&(other.major, other.minor, other.patch))
            .then_with(|| match (self.pre.is_empty(), other.pre.is_empty()) {
                (true, true) => Ordering::Equal,
                // A prerelease comes before its release (§11.3).
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                // Identifier by identifier; a shorter list that is a prefix comes first.
                (false, false) => self.pre.cmp(&other.pre),
            })
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Version {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Version {}

impl fmt::Display for Version {
    /// The version as written, without a leading `v`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> Version {
        Version::parse(s).unwrap_or_else(|| panic!("{s} should parse"))
    }

    #[test]
    fn the_compiled_version_parses() {
        assert!(Version::parse(qubivex_core::VERSION).is_some());
    }

    #[test]
    fn table_from_the_brief() {
        assert!(v("0.1.2") > v("0.1.1"));
        assert!(v("0.2.0") > v("0.1.1"));
        assert!(v("1.0.0") > v("0.1.1"));
        assert!(v("0.1.2") <= v("0.1.2"));
        assert!(v("0.1.2") <= v("0.2.0"));
    }

    #[test]
    fn numbers_compare_as_numbers_not_strings() {
        assert!(v("0.10.0") > v("0.9.0"));
        assert!(v("0.1.10") > v("0.1.9"));
        assert!(v("10.0.0") > v("9.99.99"));
    }

    #[test]
    fn semver_spec_precedence_example() {
        // semver.org §11.4's ordered list.
        let order = [
            "1.0.0-alpha",
            "1.0.0-alpha.1",
            "1.0.0-alpha.beta",
            "1.0.0-beta",
            "1.0.0-beta.2",
            "1.0.0-beta.11",
            "1.0.0-rc.1",
            "1.0.0",
        ];
        for w in order.windows(2) {
            assert!(v(w[0]) < v(w[1]), "{} < {}", w[0], w[1]);
        }
    }

    #[test]
    fn build_metadata_and_a_leading_v_are_ignored_for_precedence() {
        assert_eq!(v("1.2.3+abc.5"), v("1.2.3"));
        assert_eq!(v("v1.2.3"), v("1.2.3"));
        assert_eq!(v("v1.2.3").to_string(), "1.2.3");
    }

    #[test]
    fn invalid_versions_are_refused() {
        for s in [
            "",
            "1",
            "1.2",
            "1.2.3.4",
            "01.2.3",
            "1.02.3",
            "1.2.x",
            "1.2.3-",
            "1.2.3-a..b",
            "1.2.3-01",
            "1.2.3+",
            "1.2.3-a_b",
            "latest",
            "<html>",
            "-1.2.3",
            " ",
        ] {
            assert!(Version::parse(s).is_none(), "{s:?} should not parse");
        }
    }
}
