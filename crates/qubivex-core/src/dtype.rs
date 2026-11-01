//! Tensor element types and their storage sizes.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{CoreError, Shape};

/// A GGML block-quantized type.
///
/// Plain float types (`F32`, `F16`, `BF16`) are not here; they are [`DType`] variants
/// directly, so each type has exactly one representation. Block layouts follow
/// `ggml-common.h`; K-quants use super-blocks of 256 elements.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[non_exhaustive]
#[allow(non_camel_case_types)] // Match the GGML spelling used in every GGUF tool.
pub enum GgmlType {
    /// 4-bit, 32-element blocks, f16 scale.
    Q4_0,
    /// 4-bit, 32-element blocks, f16 scale and min.
    Q4_1,
    /// 5-bit, 32-element blocks, f16 scale.
    Q5_0,
    /// 5-bit, 32-element blocks, f16 scale and min.
    Q5_1,
    /// 8-bit, 32-element blocks, f16 scale.
    Q8_0,
    /// 8-bit, 32-element blocks, f16 scale and sum.
    Q8_1,
    /// 2-bit K-quant, 256-element super-blocks.
    Q2_K,
    /// 3-bit K-quant, 256-element super-blocks.
    Q3_K,
    /// 4-bit K-quant, 256-element super-blocks.
    Q4_K,
    /// 5-bit K-quant, 256-element super-blocks.
    Q5_K,
    /// 6-bit K-quant, 256-element super-blocks.
    Q6_K,
    /// 8-bit K-quant (activation type), 256-element super-blocks.
    Q8_K,
}

impl GgmlType {
    /// Every supported block type, in GGML id order.
    pub const ALL: [Self; 12] = [
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q8_1,
        Self::Q2_K,
        Self::Q3_K,
        Self::Q4_K,
        Self::Q5_K,
        Self::Q6_K,
        Self::Q8_K,
    ];

    /// The numeric type id used in GGUF tensor headers.
    #[must_use]
    pub const fn ggml_id(self) -> u32 {
        match self {
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::Q8_K => 15,
        }
    }

    /// Elements per block.
    #[must_use]
    pub const fn block_elems(self) -> u64 {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
        }
    }

    /// Bytes per block.
    #[must_use]
    pub const fn block_bytes(self) -> u64 {
        match self {
            Self::Q4_0 => 2 + 16,
            Self::Q4_1 => 2 + 2 + 16,
            Self::Q5_0 => 2 + 4 + 16,
            Self::Q5_1 => 2 + 2 + 4 + 16,
            Self::Q8_0 => 2 + 32,
            Self::Q8_1 => 2 + 2 + 32,
            Self::Q2_K => 16 + 64 + 2 + 2,
            Self::Q3_K => 32 + 64 + 12 + 2,
            Self::Q4_K => 2 + 2 + 12 + 128,
            Self::Q5_K => 2 + 2 + 12 + 32 + 128,
            Self::Q6_K => 128 + 64 + 16 + 2,
            Self::Q8_K => 4 + 256 + 32,
        }
    }

    /// The lowercase name, for example `q4_k`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q8_1 => "q8_1",
            Self::Q2_K => "q2_k",
            Self::Q3_K => "q3_k",
            Self::Q4_K => "q4_k",
            Self::Q5_K => "q5_k",
            Self::Q6_K => "q6_k",
            Self::Q8_K => "q8_k",
        }
    }
}

/// How a dtype lays out bytes, which determines its storage size.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StorageLayout {
    /// A fixed number of bytes per element.
    Elementwise {
        /// Bytes per element.
        bytes_per_elem: u64,
    },
    /// Fixed-size blocks along the innermost dimension.
    Blocked {
        /// Elements per block.
        block_elems: u64,
        /// Bytes per block.
        block_bytes: u64,
    },
    /// One byte per element plus one scale per row, rows stored contiguously.
    RowScaled {
        /// Bytes of the per-row scale.
        scale_bytes: u64,
    },
}

/// A tensor element type.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[non_exhaustive]
pub enum DType {
    /// IEEE-754 binary32.
    F32,
    /// IEEE-754 binary16.
    F16,
    /// bfloat16.
    BF16,
    /// Signed int8 weights with one f32 scale per row (Colibri's int8 expert format).
    /// Each row is stored as `cols` int8 values followed by its f32 scale.
    I8RowScaled,
    /// A GGML block-quantized type.
    Ggml(GgmlType),
}

impl DType {
    /// Maps a GGUF tensor type id to a dtype.
    ///
    /// # Errors
    /// Returns [`CoreError::UnknownGgmlType`] for ids QUBIVEX does not support.
    pub fn from_ggml_id(id: u32) -> Result<Self, CoreError> {
        match id {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            30 => Ok(Self::BF16),
            _ => GgmlType::ALL
                .iter()
                .find(|t| t.ggml_id() == id)
                .map(|&t| Self::Ggml(t))
                .ok_or(CoreError::UnknownGgmlType(id)),
        }
    }

    /// The lowercase name, for example `bf16` or `q4_k`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::I8RowScaled => "i8_row_scaled",
            Self::Ggml(t) => t.name(),
        }
    }

    /// The byte layout of this dtype.
    #[must_use]
    pub const fn layout(self) -> StorageLayout {
        match self {
            Self::F32 => StorageLayout::Elementwise { bytes_per_elem: 4 },
            Self::F16 | Self::BF16 => StorageLayout::Elementwise { bytes_per_elem: 2 },
            Self::I8RowScaled => StorageLayout::RowScaled { scale_bytes: 4 },
            Self::Ggml(t) => StorageLayout::Blocked {
                block_elems: t.block_elems(),
                block_bytes: t.block_bytes(),
            },
        }
    }

    /// The exact number of bytes a tensor of this dtype and shape occupies on disk
    /// and in memory.
    ///
    /// # Errors
    /// - [`CoreError::Overflow`] if any intermediate product overflows `u64`.
    /// - [`CoreError::RankTooSmall`] if a blocked or row-scaled dtype is given a scalar.
    /// - [`CoreError::BlockMisaligned`] if the innermost dimension is not a multiple of
    ///   the block size.
    pub fn storage_bytes(self, shape: &Shape) -> Result<u64, CoreError> {
        let overflow = CoreError::Overflow {
            what: "tensor byte size",
        };
        match self.layout() {
            StorageLayout::Elementwise { bytes_per_elem } => {
                shape.numel()?.checked_mul(bytes_per_elem).ok_or(overflow)
            }
            StorageLayout::Blocked {
                block_elems,
                block_bytes,
            } => {
                let inner = self.innermost_or_err(shape)?;
                if inner % block_elems != 0 {
                    return Err(CoreError::BlockMisaligned {
                        dtype: self.name(),
                        len: inner,
                        block: block_elems,
                    });
                }
                (inner / block_elems)
                    .checked_mul(block_bytes)
                    .and_then(|row| row.checked_mul(shape.rows().ok()?))
                    .ok_or(overflow)
            }
            StorageLayout::RowScaled { scale_bytes } => {
                let inner = self.innermost_or_err(shape)?;
                inner
                    .checked_add(scale_bytes)
                    .and_then(|row| row.checked_mul(shape.rows().ok()?))
                    .ok_or(overflow)
            }
        }
    }

    fn innermost_or_err(self, shape: &Shape) -> Result<u64, CoreError> {
        shape.innermost().ok_or(CoreError::RankTooSmall {
            dtype: self.name(),
            min: 1,
            rank: shape.rank(),
        })
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Error returned when parsing an unknown dtype name.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown dtype {0:?}")]
pub struct ParseDTypeError(String);

impl FromStr for DType {
    type Err = ParseDTypeError;

    /// Parses a lowercase dtype name as produced by [`DType::name`].
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::BF16),
            "i8_row_scaled" => Ok(Self::I8RowScaled),
            _ => GgmlType::ALL
                .iter()
                .find(|t| t.name() == s)
                .map(|&t| Self::Ggml(t))
                .ok_or_else(|| ParseDTypeError(s.to_owned())),
        }
    }
}

impl Serialize for DType {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for DType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = <std::borrow::Cow<'de, str>>::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn shape(d: &[u64]) -> Shape {
        Shape::new(d).unwrap()
    }

    #[test]
    fn ggml_block_sizes_match_ggml_common() {
        let expected = [
            (GgmlType::Q4_0, 32, 18),
            (GgmlType::Q4_1, 32, 20),
            (GgmlType::Q5_0, 32, 22),
            (GgmlType::Q5_1, 32, 24),
            (GgmlType::Q8_0, 32, 34),
            (GgmlType::Q8_1, 32, 36),
            (GgmlType::Q2_K, 256, 84),
            (GgmlType::Q3_K, 256, 110),
            (GgmlType::Q4_K, 256, 144),
            (GgmlType::Q5_K, 256, 176),
            (GgmlType::Q6_K, 256, 210),
            (GgmlType::Q8_K, 256, 292),
        ];
        for (t, elems, bytes) in expected {
            assert_eq!((t.block_elems(), t.block_bytes()), (elems, bytes), "{t}");
        }
    }

    #[test]
    fn ggml_ids_round_trip() {
        for t in GgmlType::ALL {
            assert_eq!(DType::from_ggml_id(t.ggml_id()), Ok(DType::Ggml(t)));
        }
        assert_eq!(DType::from_ggml_id(0), Ok(DType::F32));
        assert_eq!(DType::from_ggml_id(1), Ok(DType::F16));
        assert_eq!(DType::from_ggml_id(30), Ok(DType::BF16));
        // 4 and 5 were Q4_2 / Q4_3, removed from GGML.
        assert_eq!(DType::from_ggml_id(4), Err(CoreError::UnknownGgmlType(4)));
        assert_eq!(
            DType::from_ggml_id(u32::MAX),
            Err(CoreError::UnknownGgmlType(u32::MAX))
        );
    }

    #[test]
    fn names_round_trip() {
        let mut all = vec![DType::F32, DType::F16, DType::BF16, DType::I8RowScaled];
        all.extend(GgmlType::ALL.map(DType::Ggml));
        for d in all {
            assert_eq!(d.name().parse::<DType>(), Ok(d));
            let json = serde_json::to_string(&d).unwrap();
            assert_eq!(json, format!("\"{}\"", d.name()));
            assert_eq!(serde_json::from_str::<DType>(&json).unwrap(), d);
        }
        assert!("Q4_K".parse::<DType>().is_err());
        assert!(serde_json::from_str::<DType>("\"f64\"").is_err());
    }

    #[test]
    fn expert_ffn_sizes() {
        // A routed SwiGLU expert with hidden 2048 and FFN 1024: gate and up are
        // [1024, 2048], down is [2048, 1024].
        let gate_up = shape(&[1024, 2048]);
        let down = shape(&[2048, 1024]);
        let total =
            |d: DType| 2 * d.storage_bytes(&gate_up).unwrap() + d.storage_bytes(&down).unwrap();
        assert_eq!(total(DType::BF16), 3 * 2048 * 1024 * 2);
        // int8 + one f32 scale per row: 1024*(2048+4)*2 + 2048*(1024+4).
        assert_eq!(total(DType::I8RowScaled), 6_307_840);
        assert_eq!(
            total(DType::Ggml(GgmlType::Q4_K)),
            3 * 2048 * 1024 / 256 * 144
        );
        assert_eq!(
            total(DType::Ggml(GgmlType::Q8_0)),
            3 * 2048 * 1024 / 32 * 34
        );
    }

    #[test]
    fn blocked_dtype_rejects_misaligned_rows() {
        let err = DType::Ggml(GgmlType::Q4_K)
            .storage_bytes(&shape(&[4, 100]))
            .unwrap_err();
        assert_eq!(
            err,
            CoreError::BlockMisaligned {
                dtype: "q4_k",
                len: 100,
                block: 256
            }
        );
    }

    #[test]
    fn blocked_and_row_scaled_reject_scalars() {
        for d in [DType::I8RowScaled, DType::Ggml(GgmlType::Q8_0)] {
            assert!(matches!(
                d.storage_bytes(&Shape::SCALAR),
                Err(CoreError::RankTooSmall {
                    min: 1,
                    rank: 0,
                    ..
                })
            ));
        }
        assert_eq!(DType::F32.storage_bytes(&Shape::SCALAR), Ok(4));
    }

    #[test]
    fn overflow_is_an_error() {
        let huge = shape(&[u64::MAX / 2, 4]);
        assert!(matches!(
            DType::F32.storage_bytes(&huge),
            Err(CoreError::Overflow { .. })
        ));
        let huge = shape(&[u64::MAX / 2, 256]);
        assert!(matches!(
            DType::Ggml(GgmlType::Q4_K).storage_bytes(&huge),
            Err(CoreError::Overflow { .. })
        ));
        let huge = shape(&[u64::MAX / 2, 1]);
        assert!(matches!(
            DType::I8RowScaled.storage_bytes(&huge),
            Err(CoreError::Overflow { .. })
        ));
    }

    fn any_dtype() -> impl Strategy<Value = DType> {
        prop_oneof![
            Just(DType::F32),
            Just(DType::F16),
            Just(DType::BF16),
            Just(DType::I8RowScaled),
            proptest::sample::select(GgmlType::ALL.to_vec()).prop_map(DType::Ggml),
        ]
    }

    enum Reference {
        Misaligned,
        /// The exact size, or `None` if it does not even fit in u128.
        Size(Option<u128>),
    }

    /// Reference size computed in checked u128 arithmetic.
    fn reference_bytes(d: DType, dims: &[u64]) -> Reference {
        let inner = u128::from(*dims.last().unwrap());
        let rows = dims[..dims.len() - 1]
            .iter()
            .try_fold(1u128, |acc, &x| acc.checked_mul(u128::from(x)));
        let per_row = match d.layout() {
            StorageLayout::Elementwise { bytes_per_elem } => {
                inner.checked_mul(u128::from(bytes_per_elem))
            }
            StorageLayout::Blocked {
                block_elems,
                block_bytes,
            } => {
                if inner % u128::from(block_elems) != 0 {
                    return Reference::Misaligned;
                }
                (inner / u128::from(block_elems)).checked_mul(u128::from(block_bytes))
            }
            StorageLayout::RowScaled { scale_bytes } => inner.checked_add(u128::from(scale_bytes)),
        };
        Reference::Size(rows.zip(per_row).and_then(|(r, p)| r.checked_mul(p)))
    }

    proptest! {
        /// Untrusted header dimensions never panic, and a returned size is exact.
        #[test]
        fn storage_bytes_is_exact_or_errors(
            d in any_dtype(),
            dims in proptest::collection::vec(
                prop_oneof![0u64..=4096, Just(u64::MAX), 0u64..=u64::MAX], 1..=4),
        ) {
            let s = Shape::new(&dims).unwrap();
            let max = u128::from(u64::MAX);
            match (d.storage_bytes(&s), reference_bytes(d, &dims)) {
                (Ok(n), Reference::Size(Some(e))) => prop_assert_eq!(u128::from(n), e),
                (Err(CoreError::BlockMisaligned { .. }), Reference::Misaligned) => {}
                (Err(CoreError::Overflow { .. }), Reference::Size(e)) => {
                    // Overflow is reported only when the exact answer exceeds u64, or
                    // when an intermediate does (the row count or the element count).
                    prop_assert!(
                        e.is_none_or(|e| e > max) || s.rows().is_err() || s.numel().is_err()
                    );
                }
                (got, _) => prop_assert!(false, "unexpected result {got:?}"),
            }
        }
    }
}
