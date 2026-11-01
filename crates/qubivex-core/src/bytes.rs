//! Byte quantities.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::CoreError;

/// A number of bytes.
///
/// Arithmetic is explicit: use [`Bytes::checked_add`] and friends for sizes derived
/// from untrusted input, and [`Bytes::saturating_sub`] for budget headroom.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Bytes(pub u64);

impl Bytes {
    /// Zero bytes.
    pub const ZERO: Self = Self(0);
    /// One kibibyte (1024 bytes).
    pub const KIB: Self = Self(1 << 10);
    /// One mebibyte.
    pub const MIB: Self = Self(1 << 20);
    /// One gibibyte.
    pub const GIB: Self = Self(1 << 30);

    /// The raw byte count.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Adds two byte counts.
    ///
    /// # Errors
    /// Returns [`CoreError::Overflow`] if the sum does not fit in `u64`.
    pub fn checked_add(self, rhs: Self) -> Result<Self, CoreError> {
        self.0
            .checked_add(rhs.0)
            .map(Self)
            .ok_or(CoreError::Overflow { what: "byte sum" })
    }

    /// Multiplies a byte count by an integer.
    ///
    /// # Errors
    /// Returns [`CoreError::Overflow`] if the product does not fit in `u64`.
    pub fn checked_mul(self, rhs: u64) -> Result<Self, CoreError> {
        self.0
            .checked_mul(rhs)
            .map(Self)
            .ok_or(CoreError::Overflow {
                what: "byte product",
            })
    }

    /// Subtracts, clamping at zero.
    #[must_use]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }
}

impl fmt::Display for Bytes {
    /// Formats with binary units, for example `6.30 MiB`. Exact counts below 1 KiB.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const UNITS: [&str; 6] = ["KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
        if self.0 < 1024 {
            return write!(f, "{} B", self.0);
        }
        // Lossy only for display; never used for accounting.
        #[allow(clippy::cast_precision_loss)]
        let mut value = self.0 as f64 / 1024.0;
        let mut unit = UNITS[0];
        for next in &UNITS[1..] {
            if value < 1024.0 {
                break;
            }
            value /= 1024.0;
            unit = next;
        }
        write!(f, "{value:.2} {unit}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_uses_binary_units() {
        assert_eq!(Bytes(0).to_string(), "0 B");
        assert_eq!(Bytes(1023).to_string(), "1023 B");
        assert_eq!(Bytes::KIB.to_string(), "1.00 KiB");
        assert_eq!(Bytes(6_606_028).to_string(), "6.30 MiB");
        assert_eq!(Bytes::GIB.checked_mul(14).unwrap().to_string(), "14.00 GiB");
        assert_eq!(Bytes(u64::MAX).to_string(), "16.00 EiB");
    }

    #[test]
    fn checked_arithmetic() {
        assert_eq!(Bytes(1).checked_add(Bytes(2)), Ok(Bytes(3)));
        assert!(Bytes(u64::MAX).checked_add(Bytes(1)).is_err());
        assert!(Bytes(u64::MAX).checked_mul(2).is_err());
        assert_eq!(Bytes(3).saturating_sub(Bytes(5)), Bytes::ZERO);
    }
}
