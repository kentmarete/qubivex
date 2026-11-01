use thiserror::Error;

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
}
