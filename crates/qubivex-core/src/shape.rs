//! Tensor shapes.

use std::fmt;

use crate::CoreError;

/// The maximum tensor rank QUBIVEX accepts.
///
/// GGUF allows 4 dimensions. SafeTensors allows any rank; tensors above this rank are
/// rejected at validation time rather than silently reshaped.
pub const MAX_RANK: usize = 8;

/// A tensor shape in row-major order: the **last** dimension is contiguous in memory.
///
/// This is the SafeTensors and PyTorch convention. GGUF lists dimensions innermost
/// first (`ne[0]` is contiguous), so GGUF readers reverse them before building a
/// `Shape`.
///
/// A rank-0 shape is a scalar with one element. Zero-length dimensions are allowed
/// and give zero elements.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: [u64; MAX_RANK],
    rank: u8,
}

const _: () = assert!(size_of::<Shape>() == 72);

impl Shape {
    /// A rank-0 (scalar) shape.
    pub const SCALAR: Self = Self {
        dims: [0; MAX_RANK],
        rank: 0,
    };

    /// Builds a shape from dimensions, outermost first.
    ///
    /// # Errors
    /// Returns [`CoreError::RankTooLarge`] if there are more than [`MAX_RANK`]
    /// dimensions.
    pub fn new(dims: &[u64]) -> Result<Self, CoreError> {
        if dims.len() > MAX_RANK {
            return Err(CoreError::RankTooLarge {
                rank: dims.len(),
                max: MAX_RANK,
            });
        }
        let mut out = [0; MAX_RANK];
        out[..dims.len()].copy_from_slice(dims);
        let rank = u8::try_from(dims.len()).map_err(|_| CoreError::RankTooLarge {
            rank: dims.len(),
            max: MAX_RANK,
        })?;
        Ok(Self { dims: out, rank })
    }

    /// The dimensions, outermost first.
    #[must_use]
    pub fn dims(&self) -> &[u64] {
        &self.dims[..usize::from(self.rank)]
    }

    /// The number of dimensions.
    #[must_use]
    pub fn rank(&self) -> usize {
        usize::from(self.rank)
    }

    /// The contiguous (last) dimension, or `None` for a scalar.
    #[must_use]
    pub fn innermost(&self) -> Option<u64> {
        self.dims().last().copied()
    }

    /// The total number of elements.
    ///
    /// # Errors
    /// Returns [`CoreError::Overflow`] if the product does not fit in `u64`.
    pub fn numel(&self) -> Result<u64, CoreError> {
        self.dims().iter().try_fold(1u64, |acc, &d| {
            acc.checked_mul(d).ok_or(CoreError::Overflow {
                what: "element count",
            })
        })
    }

    /// The number of rows: the product of every dimension except the last.
    /// A rank-1 shape has one row; a scalar has one row.
    ///
    /// # Errors
    /// Returns [`CoreError::Overflow`] if the product does not fit in `u64`.
    pub fn rows(&self) -> Result<u64, CoreError> {
        let outer = self.dims().split_last().map_or(&[][..], |(_, rest)| rest);
        outer.iter().try_fold(1u64, |acc, &d| {
            acc.checked_mul(d)
                .ok_or(CoreError::Overflow { what: "row count" })
        })
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Shape {
    /// Formats as `[2048, 1024]`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (i, d) in self.dims().iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{d}")?;
        }
        f.write_str("]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let s = Shape::new(&[16, 2048, 1024]).unwrap();
        assert_eq!(s.rank(), 3);
        assert_eq!(s.dims(), &[16, 2048, 1024]);
        assert_eq!(s.innermost(), Some(1024));
        assert_eq!(s.numel().unwrap(), 16 * 2048 * 1024);
        assert_eq!(s.rows().unwrap(), 16 * 2048);
        assert_eq!(s.to_string(), "[16, 2048, 1024]");
    }

    #[test]
    fn scalar_and_vector() {
        assert_eq!(Shape::SCALAR.numel().unwrap(), 1);
        assert_eq!(Shape::SCALAR.rows().unwrap(), 1);
        assert_eq!(Shape::SCALAR.innermost(), None);
        assert_eq!(Shape::SCALAR, Shape::new(&[]).unwrap());
        let v = Shape::new(&[7]).unwrap();
        assert_eq!(v.rows().unwrap(), 1);
    }

    #[test]
    fn zero_dimension_gives_zero_elements() {
        assert_eq!(Shape::new(&[4, 0, 3]).unwrap().numel().unwrap(), 0);
    }

    #[test]
    fn rejects_excess_rank() {
        let err = Shape::new(&[1; MAX_RANK + 1]).unwrap_err();
        assert_eq!(
            err,
            CoreError::RankTooLarge {
                rank: MAX_RANK + 1,
                max: MAX_RANK
            }
        );
    }

    #[test]
    fn overflow_is_an_error_not_a_wrap() {
        let s = Shape::new(&[u64::MAX, 2]).unwrap();
        assert!(matches!(s.numel(), Err(CoreError::Overflow { .. })));
        let s = Shape::new(&[u64::MAX, 2, 1]).unwrap();
        assert!(matches!(s.rows(), Err(CoreError::Overflow { .. })));
    }

    #[test]
    fn equality_ignores_unused_slots() {
        // Shapes are built only through `new`, so unused slots are always zero.
        assert_eq!(Shape::new(&[3, 4]).unwrap(), Shape::new(&[3, 4]).unwrap());
        assert_ne!(
            Shape::new(&[3, 4]).unwrap(),
            Shape::new(&[3, 4, 1]).unwrap()
        );
    }
}
