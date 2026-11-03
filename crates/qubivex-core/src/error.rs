use thiserror::Error;

use crate::Shape;

/// Errors raised by core type constructors and size computations.
///
/// Other domains (model, storage, planning, backend, execution, hardware) define
/// their own error enums in their own crates and wrap this one with `#[from]`.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum CoreError {
    /// A size, element count or offset computation overflowed `u64`.
    #[error("arithmetic overflow while computing {what}")]
    Overflow {
        /// What was being computed, for example "element count".
        what: &'static str,
    },

    /// A shape had more dimensions than [`crate::MAX_RANK`].
    #[error("shape rank {rank} exceeds the maximum of {max}")]
    RankTooLarge {
        /// The rank that was requested.
        rank: usize,
        /// The maximum supported rank.
        max: usize,
    },

    /// The innermost dimension is not a multiple of the dtype's block size.
    #[error("innermost dimension {len} is not a multiple of the {dtype} block size {block}")]
    BlockMisaligned {
        /// The dtype name.
        dtype: &'static str,
        /// The innermost dimension length.
        len: u64,
        /// The required block size, in elements.
        block: u64,
    },

    /// A dtype needs a shape of at least this rank (for example row-scaled int8 needs rows).
    #[error("{dtype} requires a shape of rank >= {min}, got rank {rank}")]
    RankTooSmall {
        /// The dtype name.
        dtype: &'static str,
        /// The minimum rank.
        min: usize,
        /// The rank that was given.
        rank: usize,
    },

    /// A numeric GGML type id that QUBIVEX does not know.
    #[error("unknown GGML type id {0}")]
    UnknownGgmlType(u32),

    /// A list (strides, an index, a permutation) has the wrong length for the rank.
    #[error("{what} has {got} entries, expected {expected}")]
    LengthMismatch {
        /// What was given, for example "strides".
        what: &'static str,
        /// The length given.
        got: usize,
        /// The length required.
        expected: usize,
    },

    /// A dimension number is not below the rank.
    #[error("dimension {dim} is out of range for rank {rank}")]
    DimOutOfRange {
        /// The dimension requested.
        dim: usize,
        /// The tensor rank.
        rank: usize,
    },

    /// An index is not below the dimension's size.
    #[error("index {index} is out of range for dimension {dim} of size {size}")]
    IndexOutOfRange {
        /// The dimension.
        dim: usize,
        /// The index requested.
        index: u64,
        /// The dimension's size.
        size: u64,
    },

    /// A slice `start .. start + len` does not fit in the dimension.
    #[error("slice {start}..+{len} does not fit dimension {dim} of size {size}")]
    RangeOutOfBounds {
        /// The dimension.
        dim: usize,
        /// First index of the slice.
        start: u64,
        /// Number of indices in the slice.
        len: u64,
        /// The dimension's size.
        size: u64,
    },

    /// A permutation does not contain every dimension exactly once.
    #[error("not a permutation of the {rank} dimensions")]
    InvalidPermutation {
        /// The tensor rank.
        rank: usize,
    },

    /// The operation would split or move the packed innermost dimension of a
    /// blocked or row-scaled dtype.
    #[error("{op} cannot change the packed innermost dimension of a {dtype} tensor")]
    PackedInnermost {
        /// The dtype name.
        dtype: &'static str,
        /// The operation.
        op: &'static str,
    },

    /// A shape cannot be broadcast to the target shape.
    #[error("shape {from} cannot be broadcast to {to}")]
    BroadcastIncompatible {
        /// The source shape (boxed: errors are cold, results stay small).
        from: Box<Shape>,
        /// The target shape.
        to: Box<Shape>,
    },

    /// Two shapes have different element counts.
    #[error("shape {from} and shape {to} have different element counts")]
    ElementCountMismatch {
        /// The source shape (boxed: errors are cold, results stay small).
        from: Box<Shape>,
        /// The target shape.
        to: Box<Shape>,
    },

    /// The strides do not allow this reshape without copying; make the tensor
    /// contiguous first.
    #[error("reshaping {from} to {to} needs a copy with these strides")]
    ReshapeNeedsCopy {
        /// The source shape (boxed: errors are cold, results stay small).
        from: Box<Shape>,
        /// The target shape.
        to: Box<Shape>,
    },

    /// A stride or offset is not a multiple of the element (or storage unit) size.
    #[error("byte stride or offset {stride} is not a multiple of the {dtype} unit of {unit} bytes")]
    StrideMisaligned {
        /// The dtype name.
        dtype: &'static str,
        /// The offending stride or offset, in bytes.
        stride: u64,
        /// The required unit, in bytes.
        unit: u64,
    },
}
