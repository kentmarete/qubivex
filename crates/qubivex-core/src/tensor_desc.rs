//! Tensor descriptors: a dtype, a shape, byte strides and a byte offset.
//!
//! A [`TensorDesc`] says where every element (or, for packed dtypes, every row) of a
//! tensor lives inside some storage, without owning or touching that storage. View
//! operations (transpose, slice, broadcast, reshape) return a new descriptor and never
//! copy. Every computation is checked: an invalid descriptor is a [`CoreError`], never a
//! panic or a wrapped integer. Design: `docs/architecture/TENSOR_ARCHITECTURE.md`.

use std::fmt;

use crate::{CoreError, DType, MAX_RANK, Shape, StorageLayout};

/// Where a tensor's bytes are inside its storage.
///
/// - `strides` and `offset` are in **bytes**, so blocked (`q4_k`) and row-scaled
///   (`i8_row_scaled`) dtypes, which have no fixed bytes per element, are addressed the
///   same way as `f32`.
/// - For a **packed** dtype ([`DType::is_packed`]) the innermost dimension is a packed
///   row: it has no per-element stride (its slot in [`TensorDesc::strides`] is 0), and
///   no view may move it or split it except on block boundaries.
/// - A stride of 0 on an outer dimension is a broadcast.
///
/// The descriptor has no device: the storage carries the device, so the same
/// descriptor describes a tensor in RAM or in VRAM.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorDesc {
    dtype: DType,
    shape: Shape,
    strides: [u64; MAX_RANK],
    offset: u64,
}

fn overflow(what: &'static str) -> CoreError {
    CoreError::Overflow { what }
}

/// Bytes of one whole innermost row of `inner` elements.
fn row_bytes_of(dtype: DType, inner: u64) -> Result<u64, CoreError> {
    dtype.storage_bytes(&Shape::new(&[inner])?)
}

/// Row-major strides for `dims`, where the innermost listed dimension has stride
/// `unit` (for packed outer dimensions: the row's byte length).
fn packed_strides(dims: &[u64], unit: u64) -> Result<[u64; MAX_RANK], CoreError> {
    let mut s = [0; MAX_RANK];
    let mut acc = unit;
    for (i, &d) in dims.iter().enumerate().rev() {
        s[i] = acc;
        if i > 0 {
            acc = acc.checked_mul(d).ok_or(overflow("tensor strides"))?;
        }
    }
    Ok(s)
}

/// Contiguous strides for a whole shape of `dtype`.
fn contiguous_strides(dtype: DType, dims: &[u64]) -> Result<[u64; MAX_RANK], CoreError> {
    let Some((&inner, outer)) = dims.split_last() else {
        return Ok([0; MAX_RANK]);
    };
    match dtype.elem_bytes() {
        Some(e) => packed_strides(dims, e),
        None => packed_strides(outer, row_bytes_of(dtype, inner)?),
    }
}

/// The zero-copy strides for viewing (`old`, `old_strides`) as `new`, or `None` when
/// the strides do not allow it. This is PyTorch's `computeStride` rule: dimensions can
/// be merged or split only inside runs that are contiguous with each other.
fn view_strides(
    old: &[u64],
    old_strides: &[u64],
    new: &[u64],
    unit: u64,
) -> Result<Option<[u64; MAX_RANK]>, CoreError> {
    // A scalar or empty tensor has no layout to preserve.
    if old.is_empty() || old.contains(&0) {
        return packed_strides(new, unit).map(Some);
    }
    let ovf = || overflow("reshape strides");
    let mut out = [0; MAX_RANK];
    let mut view_d = new.len(); // dimensions of `new` still to assign, from the back
    let mut chunk_base = old_strides[old.len() - 1];
    let mut tensor_numel = 1u64;
    let mut view_numel = 1u64;
    for td in (0..old.len()).rev() {
        tensor_numel = tensor_numel.checked_mul(old[td]).ok_or_else(ovf)?;
        let chunk_ends = td == 0
            || (old[td - 1] != 1
                && tensor_numel.checked_mul(chunk_base) != Some(old_strides[td - 1]));
        if chunk_ends {
            while view_d > 0 && (view_numel < tensor_numel || new[view_d - 1] == 1) {
                out[view_d - 1] = view_numel.checked_mul(chunk_base).ok_or_else(ovf)?;
                view_numel = view_numel.checked_mul(new[view_d - 1]).ok_or_else(ovf)?;
                view_d -= 1;
            }
            if view_numel != tensor_numel {
                return Ok(None);
            }
            if td > 0 {
                chunk_base = old_strides[td - 1];
                tensor_numel = 1;
                view_numel = 1;
            }
        }
    }
    Ok((view_d == 0).then_some(out))
}

impl TensorDesc {
    /// A contiguous row-major tensor at offset 0.
    ///
    /// # Errors
    /// As [`DType::storage_bytes`]: overflow, a misaligned block, or a packed dtype
    /// with a scalar shape.
    pub fn contiguous(dtype: DType, shape: Shape) -> Result<Self, CoreError> {
        dtype.storage_bytes(&shape)?;
        Ok(Self {
            dtype,
            shape,
            strides: contiguous_strides(dtype, shape.dims())?,
            offset: 0,
        })
    }

    /// A descriptor from explicit byte strides (one per dimension) and a byte offset.
    ///
    /// # Errors
    /// - [`CoreError::LengthMismatch`] if `strides` does not have one entry per dimension;
    /// - [`CoreError::RankTooSmall`] or [`CoreError::BlockMisaligned`] for an invalid
    ///   packed shape, and [`CoreError::PackedInnermost`] if a packed innermost stride
    ///   is not 0;
    /// - [`CoreError::StrideMisaligned`] if an elementwise stride or the offset is not a
    ///   multiple of the element size;
    /// - [`CoreError::Overflow`] if the furthest byte does not fit in `u64`.
    pub fn from_parts(
        dtype: DType,
        shape: Shape,
        strides: &[u64],
        offset: u64,
    ) -> Result<Self, CoreError> {
        let rank = shape.rank();
        if strides.len() != rank {
            return Err(CoreError::LengthMismatch {
                what: "strides",
                got: strides.len(),
                expected: rank,
            });
        }
        match dtype.elem_bytes() {
            Some(unit) => {
                if let Some(&bad) = std::iter::once(&offset)
                    .chain(strides)
                    .find(|&&s| s % unit != 0)
                {
                    return Err(CoreError::StrideMisaligned {
                        dtype: dtype.name(),
                        stride: bad,
                        unit,
                    });
                }
            }
            None => {
                let inner = shape.innermost().ok_or(CoreError::RankTooSmall {
                    dtype: dtype.name(),
                    min: 1,
                    rank,
                })?;
                row_bytes_of(dtype, inner)?;
                if strides[rank - 1] != 0 {
                    return Err(CoreError::PackedInnermost {
                        dtype: dtype.name(),
                        op: "from_parts",
                    });
                }
            }
        }
        let mut s = [0; MAX_RANK];
        s[..rank].copy_from_slice(strides);
        let d = Self {
            dtype,
            shape,
            strides: s,
            offset,
        };
        d.required_bytes()?;
        Ok(d)
    }

    /// Element type.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Shape.
    #[must_use]
    pub const fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Dimensions, outermost first.
    #[must_use]
    pub fn dims(&self) -> &[u64] {
        self.shape.dims()
    }

    /// Number of dimensions.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Byte strides, one per dimension. A packed innermost dimension reports 0.
    #[must_use]
    pub fn strides(&self) -> &[u64] {
        &self.strides[..self.rank()]
    }

    /// Byte offset of the first element from the start of the storage.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Whether the dtype is packed (see [`DType::is_packed`]).
    #[must_use]
    pub const fn is_packed(&self) -> bool {
        self.dtype.is_packed()
    }

    /// Number of elements.
    ///
    /// # Errors
    /// [`CoreError::Overflow`] if it does not fit in `u64`.
    pub fn numel(&self) -> Result<u64, CoreError> {
        self.shape.numel()
    }

    /// Bytes of one whole innermost row as stored contiguously (for a packed dtype, the
    /// packed row; for an elementwise dtype, `innermost × element size`).
    ///
    /// # Errors
    /// [`CoreError::RankTooSmall`] for a scalar, or as [`DType::storage_bytes`].
    pub fn row_bytes(&self) -> Result<u64, CoreError> {
        let inner = self.shape.innermost().ok_or(CoreError::RankTooSmall {
            dtype: self.dtype.name(),
            min: 1,
            rank: 0,
        })?;
        row_bytes_of(self.dtype, inner)
    }

    /// The number of storage bytes this view needs: one past its furthest byte.
    /// Storage at least this long makes every in-range index in bounds. 0 if the
    /// tensor has no elements.
    ///
    /// # Errors
    /// [`CoreError::Overflow`] if the furthest byte does not fit in `u64`.
    pub fn required_bytes(&self) -> Result<u64, CoreError> {
        let dims = self.dims();
        if dims.contains(&0) {
            return Ok(0);
        }
        let ovf = || overflow("tensor extent");
        let (spanned, last) = match self.dtype.elem_bytes() {
            Some(e) => (self.rank(), e),
            None => (self.rank() - 1, self.row_bytes()?),
        };
        let mut end = self.offset;
        for (&d, &s) in dims[..spanned].iter().zip(&self.strides) {
            end = (d - 1)
                .checked_mul(s)
                .and_then(|x| end.checked_add(x))
                .ok_or_else(ovf)?;
        }
        end.checked_add(last).ok_or_else(ovf)
    }

    /// Whether the elements are laid out row-major with no gaps from
    /// [`offset`](Self::offset) on. Strides of size-1 dimensions are ignored; a tensor
    /// with no elements is contiguous.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        let dims = self.dims();
        if dims.contains(&0) {
            return true;
        }
        contiguous_strides(self.dtype, dims).is_ok_and(|c| {
            dims.iter()
                .zip(self.strides.iter().zip(&c))
                .all(|(&d, (s, c))| d <= 1 || s == c)
        })
    }

    /// Whether no two indices reach the same byte, which a mutable view requires.
    /// Broadcast (stride 0) dimensions of size > 1 overlap.
    #[must_use]
    pub fn is_non_overlapping(&self) -> bool {
        let dims = self.dims();
        if dims.contains(&0) {
            return true;
        }
        let (spanned, unit) = match self.dtype.elem_bytes() {
            Some(e) => (self.rank(), e),
            None => match self.row_bytes() {
                Ok(r) => (self.rank() - 1, r),
                Err(_) => return false,
            },
        };
        let mut runs = [(0u64, 0u64); MAX_RANK];
        let mut n = 0;
        for (&d, &s) in dims[..spanned].iter().zip(&self.strides) {
            if d > 1 {
                runs[n] = (s, d);
                n += 1;
            }
        }
        let runs = &mut runs[..n];
        runs.sort_unstable();
        // Each dimension must step past everything the smaller-stride dimensions span.
        let mut span = unit;
        for &(s, d) in runs.iter() {
            if s < span {
                return false;
            }
            match s.checked_mul(d - 1).and_then(|x| x.checked_add(span)) {
                Some(next) => span = next,
                None => return false,
            }
        }
        true
    }

    /// The byte position of an index. Elementwise dtypes take one index per
    /// dimension; packed dtypes take the outer indices and return the row's start.
    ///
    /// # Errors
    /// [`CoreError::LengthMismatch`], [`CoreError::IndexOutOfRange`] or
    /// [`CoreError::Overflow`].
    pub fn byte_offset(&self, index: &[u64]) -> Result<u64, CoreError> {
        let expected = if self.is_packed() {
            self.rank() - 1
        } else {
            self.rank()
        };
        if index.len() != expected {
            return Err(CoreError::LengthMismatch {
                what: "index",
                got: index.len(),
                expected,
            });
        }
        let mut off = self.offset;
        for (dim, ((&i, &size), &s)) in index.iter().zip(self.dims()).zip(&self.strides).enumerate()
        {
            if i >= size {
                return Err(CoreError::IndexOutOfRange {
                    dim,
                    index: i,
                    size,
                });
            }
            off = i
                .checked_mul(s)
                .and_then(|x| off.checked_add(x))
                .ok_or(overflow("byte offset"))?;
        }
        Ok(off)
    }

    fn check_dim(&self, dim: usize) -> Result<(), CoreError> {
        if dim < self.rank() {
            Ok(())
        } else {
            Err(CoreError::DimOutOfRange {
                dim,
                rank: self.rank(),
            })
        }
    }

    fn is_packed_inner(&self, dim: usize) -> bool {
        self.is_packed() && dim + 1 == self.rank()
    }

    fn packed_err(&self, op: &'static str) -> CoreError {
        CoreError::PackedInnermost {
            dtype: self.dtype.name(),
            op,
        }
    }

    /// Reorders the dimensions: dimension `i` of the result is `perm[i]` of `self`.
    ///
    /// # Errors
    /// [`CoreError::LengthMismatch`] or [`CoreError::InvalidPermutation`] for a bad
    /// permutation, and [`CoreError::PackedInnermost`] if it moves a packed innermost
    /// dimension.
    pub fn permute(&self, perm: &[usize]) -> Result<Self, CoreError> {
        let rank = self.rank();
        if perm.len() != rank {
            return Err(CoreError::LengthMismatch {
                what: "permutation",
                got: perm.len(),
                expected: rank,
            });
        }
        let mut seen = [false; MAX_RANK];
        for &p in perm {
            if p >= rank || seen[p] {
                return Err(CoreError::InvalidPermutation { rank });
            }
            seen[p] = true;
        }
        if self.is_packed() && perm[rank - 1] != rank - 1 {
            return Err(self.packed_err("permute"));
        }
        let mut dims = [0; MAX_RANK];
        let mut strides = [0; MAX_RANK];
        for (i, &p) in perm.iter().enumerate() {
            dims[i] = self.dims()[p];
            strides[i] = self.strides[p];
        }
        Ok(Self {
            shape: Shape::new(&dims[..rank])?,
            strides,
            ..*self
        })
    }

    /// Swaps two dimensions.
    ///
    /// # Errors
    /// [`CoreError::DimOutOfRange`], or as [`permute`](Self::permute).
    pub fn transpose(&self, a: usize, b: usize) -> Result<Self, CoreError> {
        self.check_dim(a)?;
        self.check_dim(b)?;
        let mut perm = [0; MAX_RANK];
        for (i, p) in perm.iter_mut().enumerate() {
            *p = i;
        }
        perm.swap(a, b);
        self.permute(&perm[..self.rank()])
    }

    /// Keeps indices `start .. start + len` of dimension `dim` (a slice).
    ///
    /// On a packed innermost dimension the slice must be on block boundaries for
    /// blocked dtypes, and is refused for row-scaled ones (a partial row has no scale).
    ///
    /// # Errors
    /// [`CoreError::DimOutOfRange`], [`CoreError::RangeOutOfBounds`],
    /// [`CoreError::BlockMisaligned`], [`CoreError::PackedInnermost`] or
    /// [`CoreError::Overflow`].
    pub fn narrow(&self, dim: usize, start: u64, len: u64) -> Result<Self, CoreError> {
        self.check_dim(dim)?;
        let size = self.dims()[dim];
        if start.checked_add(len).is_none_or(|end| end > size) {
            return Err(CoreError::RangeOutOfBounds {
                dim,
                start,
                len,
                size,
            });
        }
        let ovf = || overflow("slice offset");
        let skipped = if self.is_packed_inner(dim) {
            match self.dtype.layout() {
                StorageLayout::Blocked {
                    block_elems,
                    block_bytes,
                } => {
                    if let Some(&bad) = [start, len].iter().find(|&&x| x % block_elems != 0) {
                        return Err(CoreError::BlockMisaligned {
                            dtype: self.dtype.name(),
                            len: bad,
                            block: block_elems,
                        });
                    }
                    (start / block_elems)
                        .checked_mul(block_bytes)
                        .ok_or_else(ovf)?
                }
                _ if start == 0 && len == size => 0,
                _ => return Err(self.packed_err("narrow")),
            }
        } else {
            start.checked_mul(self.strides[dim]).ok_or_else(ovf)?
        };
        let mut dims = [0; MAX_RANK];
        dims[..self.rank()].copy_from_slice(self.dims());
        dims[dim] = len;
        Ok(Self {
            shape: Shape::new(&dims[..self.rank()])?,
            offset: self.offset.checked_add(skipped).ok_or_else(ovf)?,
            ..*self
        })
    }

    /// Fixes dimension `dim` at `index` and removes it; the rank falls by one.
    ///
    /// # Errors
    /// [`CoreError::DimOutOfRange`], [`CoreError::IndexOutOfRange`], or
    /// [`CoreError::PackedInnermost`] for a packed innermost dimension.
    pub fn select(&self, dim: usize, index: u64) -> Result<Self, CoreError> {
        self.check_dim(dim)?;
        if self.is_packed_inner(dim) {
            return Err(self.packed_err("select"));
        }
        let size = self.dims()[dim];
        if index >= size {
            return Err(CoreError::IndexOutOfRange { dim, index, size });
        }
        let n = self.narrow(dim, index, 1)?;
        let rank = self.rank();
        let mut dims = [0; MAX_RANK];
        let mut strides = [0; MAX_RANK];
        let keep = (0..rank).filter(|&i| i != dim);
        for (o, i) in keep.enumerate() {
            dims[o] = n.dims()[i];
            strides[o] = n.strides[i];
        }
        Ok(Self {
            shape: Shape::new(&dims[..rank - 1])?,
            strides,
            ..n
        })
    }

    /// Views the tensor as `target` by NumPy broadcasting: dimensions align on the
    /// right, and missing or size-1 dimensions repeat with stride 0. No data is copied.
    /// A packed dtype must keep its innermost size.
    ///
    /// # Errors
    /// [`CoreError::BroadcastIncompatible`] or [`CoreError::PackedInnermost`].
    pub fn broadcast_to(&self, target: &Shape) -> Result<Self, CoreError> {
        let (rank, trank) = (self.rank(), target.rank());
        let incompatible = CoreError::BroadcastIncompatible {
            from: Box::new(self.shape),
            to: Box::new(*target),
        };
        if trank < rank {
            return Err(incompatible);
        }
        if self.is_packed() && target.innermost() != self.shape.innermost() {
            return Err(self.packed_err("broadcast_to"));
        }
        let lead = trank - rank;
        let mut strides = [0; MAX_RANK];
        for (i, &t) in target.dims().iter().enumerate().skip(lead) {
            let j = i - lead;
            let s = self.dims()[j];
            strides[i] = if s == t {
                self.strides[j]
            } else if s == 1 {
                0
            } else {
                return Err(incompatible);
            };
        }
        Ok(Self {
            shape: *target,
            strides,
            ..*self
        })
    }

    /// Views the same elements, in the same row-major order, as `target`, without
    /// copying. A packed dtype must keep its innermost size.
    ///
    /// # Errors
    /// [`CoreError::ElementCountMismatch`], [`CoreError::PackedInnermost`], or
    /// [`CoreError::ReshapeNeedsCopy`] when the strides do not allow a view (copy with
    /// a contiguous conversion first).
    pub fn reshape(&self, target: &Shape) -> Result<Self, CoreError> {
        if self.numel()? != target.numel()? {
            return Err(CoreError::ElementCountMismatch {
                from: Box::new(self.shape),
                to: Box::new(*target),
            });
        }
        let strides = if let Some(e) = self.dtype.elem_bytes() {
            view_strides(self.dims(), self.strides(), target.dims(), e)?
        } else {
            if target.rank() == 0 || target.innermost() != self.shape.innermost() {
                return Err(self.packed_err("reshape"));
            }
            let (old_outer, _) = self.dims().split_at(self.rank() - 1);
            let (new_outer, _) = target.dims().split_at(target.rank() - 1);
            view_strides(
                old_outer,
                &self.strides[..self.rank() - 1],
                new_outer,
                self.row_bytes()?,
            )?
        };
        let strides = strides.ok_or(CoreError::ReshapeNeedsCopy {
            from: Box::new(self.shape),
            to: Box::new(*target),
        })?;
        Ok(Self {
            shape: *target,
            strides,
            ..*self
        })
    }
}

impl fmt::Debug for TensorDesc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} strides {:?} offset {}",
            self.dtype,
            self.shape,
            self.strides(),
            self.offset
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GgmlType;
    use proptest::prelude::*;

    const Q4K: DType = DType::Ggml(GgmlType::Q4_K);

    fn shape(d: &[u64]) -> Shape {
        Shape::new(d).unwrap()
    }

    fn f32s(d: &[u64]) -> TensorDesc {
        TensorDesc::contiguous(DType::F32, shape(d)).unwrap()
    }

    #[test]
    fn contiguous_strides_per_layout() {
        assert_eq!(f32s(&[2, 3, 4]).strides(), &[48, 16, 4]);
        let b = TensorDesc::contiguous(DType::BF16, shape(&[5, 8])).unwrap();
        assert_eq!(b.strides(), &[16, 2]);
        // q4_k: 512 elements per row = 2 blocks of 144 bytes; innermost reports 0.
        let q = TensorDesc::contiguous(Q4K, shape(&[3, 4, 512])).unwrap();
        assert_eq!(q.strides(), &[4 * 288, 288, 0]);
        assert_eq!(q.row_bytes().unwrap(), 288);
        let i8r = TensorDesc::contiguous(DType::I8RowScaled, shape(&[2, 10])).unwrap();
        assert_eq!(i8r.strides(), &[14, 0]);
        for d in [f32s(&[2, 3, 4]), b, q, i8r] {
            assert!(d.is_contiguous() && d.is_non_overlapping(), "{d:?}");
            assert_eq!(
                d.required_bytes().unwrap(),
                d.dtype().storage_bytes(d.shape()).unwrap()
            );
        }
    }

    #[test]
    fn scalars_and_empty_tensors() {
        let s = TensorDesc::contiguous(DType::F32, Shape::SCALAR).unwrap();
        assert_eq!((s.rank(), s.required_bytes().unwrap()), (0, 4));
        assert_eq!(s.byte_offset(&[]).unwrap(), 0);
        assert!(TensorDesc::contiguous(Q4K, Shape::SCALAR).is_err());
        let e = f32s(&[3, 0, 5]);
        assert_eq!(e.required_bytes().unwrap(), 0);
        assert!(e.is_contiguous() && e.is_non_overlapping());
    }

    #[test]
    fn transpose_swaps_strides_and_round_trips() {
        let d = f32s(&[2, 3]);
        let t = d.transpose(0, 1).unwrap();
        assert_eq!((t.dims(), t.strides()), (&[3, 2][..], &[4, 12][..]));
        assert!(!t.is_contiguous() && t.is_non_overlapping());
        assert_eq!(t.required_bytes().unwrap(), 24);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(
                    d.byte_offset(&[i, j]).unwrap(),
                    t.byte_offset(&[j, i]).unwrap()
                );
            }
        }
        assert_eq!(t.transpose(0, 1).unwrap(), d);
        assert_eq!(
            d.transpose(0, 2),
            Err(CoreError::DimOutOfRange { dim: 2, rank: 2 })
        );
    }

    #[test]
    fn packed_innermost_cannot_move() {
        let q = TensorDesc::contiguous(Q4K, shape(&[8, 4, 256])).unwrap();
        assert!(matches!(
            q.transpose(1, 2),
            Err(CoreError::PackedInnermost { op: "permute", .. })
        ));
        // Outer dimensions move freely.
        let t = q.transpose(0, 1).unwrap();
        assert_eq!(t.strides(), &[144, 576, 0]);
        assert!(matches!(
            q.select(2, 0),
            Err(CoreError::PackedInnermost { .. })
        ));
    }

    #[test]
    fn permute_validation() {
        let d = f32s(&[2, 3, 4]);
        assert_eq!(d.permute(&[2, 0, 1]).unwrap().dims(), &[4, 2, 3]);
        assert_eq!(
            d.permute(&[0, 0, 1]),
            Err(CoreError::InvalidPermutation { rank: 3 })
        );
        assert_eq!(
            d.permute(&[0, 1, 3]),
            Err(CoreError::InvalidPermutation { rank: 3 })
        );
        assert!(matches!(
            d.permute(&[0, 1]),
            Err(CoreError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn narrow_and_select_offsets() {
        let d = f32s(&[4, 6]);
        let n = d.narrow(0, 1, 2).unwrap().narrow(1, 2, 3).unwrap();
        assert_eq!((n.dims(), n.offset()), (&[2, 3][..], 24 + 8));
        assert_eq!(n.strides(), d.strides());
        assert!(!n.is_contiguous());
        // Furthest element: row 2, column 4 → (2*6 + 4)*4 + 4 bytes.
        assert_eq!(n.required_bytes().unwrap(), (2 * 6 + 4) * 4 + 4);
        let row = d.select(0, 3).unwrap();
        assert_eq!((row.dims(), row.offset()), (&[6][..], 72));
        assert!(row.is_contiguous());
        let col = d.select(1, 5).unwrap();
        assert_eq!(
            (col.dims(), col.strides(), col.offset()),
            (&[4][..], &[24][..], 20)
        );
        assert_eq!(
            d.narrow(1, 4, 3),
            Err(CoreError::RangeOutOfBounds {
                dim: 1,
                start: 4,
                len: 3,
                size: 6
            })
        );
        assert!(matches!(
            d.narrow(1, u64::MAX, 2),
            Err(CoreError::RangeOutOfBounds { .. })
        ));
        assert!(matches!(
            d.select(0, 4),
            Err(CoreError::IndexOutOfRange { .. })
        ));
    }

    #[test]
    fn expert_of_a_stacked_tensor() {
        // [experts, rows, cols] q4_k: one expert is a select on dimension 0.
        let q = TensorDesc::contiguous(Q4K, shape(&[64, 1024, 2048])).unwrap();
        let e = q.select(0, 5).unwrap();
        let per_expert = Q4K.storage_bytes(&shape(&[1024, 2048])).unwrap();
        assert_eq!(e.offset(), 5 * per_expert);
        assert!(e.is_contiguous());
        assert_eq!(e.required_bytes().unwrap(), 6 * per_expert);
    }

    #[test]
    fn packed_innermost_narrow() {
        let q = TensorDesc::contiguous(Q4K, shape(&[2, 1024])).unwrap();
        let n = q.narrow(1, 256, 512).unwrap();
        assert_eq!((n.offset(), n.row_bytes().unwrap()), (144, 288));
        assert!(!n.is_contiguous() && n.is_non_overlapping());
        assert!(matches!(
            q.narrow(1, 100, 256),
            Err(CoreError::BlockMisaligned { len: 100, .. })
        ));
        assert!(matches!(
            q.narrow(1, 0, 100),
            Err(CoreError::BlockMisaligned { len: 100, .. })
        ));
        let r = TensorDesc::contiguous(DType::I8RowScaled, shape(&[2, 8])).unwrap();
        assert!(r.narrow(1, 0, 8).is_ok(), "a whole row is fine");
        assert!(matches!(
            r.narrow(1, 0, 4),
            Err(CoreError::PackedInnermost { op: "narrow", .. })
        ));
    }

    #[test]
    fn broadcasting() {
        let d = f32s(&[3, 1]);
        let b = d.broadcast_to(&shape(&[2, 3, 4])).unwrap();
        assert_eq!(b.strides(), &[0, 4, 0]);
        assert!(!b.is_non_overlapping());
        assert_eq!(b.required_bytes().unwrap(), 12);
        assert_eq!(b.byte_offset(&[1, 2, 3]).unwrap(), 8);
        assert_eq!(d.broadcast_to(&shape(&[3, 2])).unwrap().strides(), &[4, 0]);
        assert!(matches!(
            d.broadcast_to(&shape(&[2, 4])),
            Err(CoreError::BroadcastIncompatible { .. })
        ));
        assert!(matches!(
            d.broadcast_to(&shape(&[3])),
            Err(CoreError::BroadcastIncompatible { .. })
        ));
        let s = TensorDesc::contiguous(DType::F32, Shape::SCALAR).unwrap();
        assert_eq!(s.broadcast_to(&shape(&[5])).unwrap().strides(), &[0]);
        let q = TensorDesc::contiguous(Q4K, shape(&[1, 256])).unwrap();
        assert_eq!(
            q.broadcast_to(&shape(&[4, 256])).unwrap().strides(),
            &[0, 0]
        );
        assert!(matches!(
            q.broadcast_to(&shape(&[1, 512])),
            Err(CoreError::PackedInnermost { .. })
        ));
    }

    #[test]
    fn reshape_views_and_refusals() {
        let d = f32s(&[2, 3, 4]);
        let r = d.reshape(&shape(&[6, 4])).unwrap();
        assert_eq!(r.strides(), &[16, 4]);
        assert_eq!(d.reshape(&shape(&[24])).unwrap().strides(), &[4]);
        assert_eq!(
            d.reshape(&shape(&[2, 1, 12])).unwrap().strides(),
            &[48, 48, 4]
        );
        // A row slice is non-contiguous but its rows can still be split.
        let n = f32s(&[4, 8]).narrow(1, 0, 4).unwrap();
        let split = n.reshape(&shape(&[4, 2, 2])).unwrap();
        assert_eq!(split.strides(), &[32, 8, 4]);
        // ... but rows cannot be merged across the gap.
        assert_eq!(
            n.reshape(&shape(&[16])),
            Err(CoreError::ReshapeNeedsCopy {
                from: Box::new(shape(&[4, 4])),
                to: Box::new(shape(&[16]))
            })
        );
        // A transpose cannot be flattened without a copy.
        assert!(matches!(
            f32s(&[2, 3]).transpose(0, 1).unwrap().reshape(&shape(&[6])),
            Err(CoreError::ReshapeNeedsCopy { .. })
        ));
        assert!(matches!(
            d.reshape(&shape(&[5, 5])),
            Err(CoreError::ElementCountMismatch { .. })
        ));
        let q = TensorDesc::contiguous(Q4K, shape(&[4, 6, 256])).unwrap();
        assert_eq!(q.reshape(&shape(&[24, 256])).unwrap().strides(), &[144, 0]);
        assert!(matches!(
            q.reshape(&shape(&[12, 512])),
            Err(CoreError::PackedInnermost { .. })
        ));
    }

    #[test]
    fn invalid_descriptors() {
        let s = shape(&[2, 3]);
        assert!(matches!(
            TensorDesc::from_parts(DType::F32, s, &[12], 0),
            Err(CoreError::LengthMismatch { .. })
        ));
        assert_eq!(
            TensorDesc::from_parts(DType::F32, s, &[12, 4], 2),
            Err(CoreError::StrideMisaligned {
                dtype: "f32",
                stride: 2,
                unit: 4
            })
        );
        assert!(matches!(
            TensorDesc::from_parts(DType::F32, s, &[4, 4], 0),
            Ok(d) if !d.is_non_overlapping()
        ));
        let q = shape(&[2, 256]);
        assert!(matches!(
            TensorDesc::from_parts(Q4K, q, &[144, 1], 0),
            Err(CoreError::PackedInnermost { .. })
        ));
        assert!(matches!(
            TensorDesc::from_parts(Q4K, shape(&[2, 100]), &[144, 0], 0),
            Err(CoreError::BlockMisaligned { .. })
        ));
        assert!(TensorDesc::from_parts(Q4K, q, &[144, 0], 7).is_ok());
    }

    #[test]
    fn overflow_is_an_error_not_a_wrap() {
        let big = shape(&[u64::MAX / 2, 2]);
        assert!(matches!(
            TensorDesc::contiguous(DType::F32, big),
            Err(CoreError::Overflow { .. })
        ));
        assert!(matches!(
            TensorDesc::from_parts(DType::F32, shape(&[3]), &[u64::MAX - 3], 0),
            Err(CoreError::Overflow { .. })
        ));
        // The furthest byte would be 2^64: refused at construction, so every later
        // view of a valid descriptor stays representable.
        assert!(matches!(
            TensorDesc::from_parts(DType::F32, shape(&[2]), &[4], u64::MAX - 7),
            Err(CoreError::Overflow { .. })
        ));
        let wide = TensorDesc::from_parts(DType::F32, shape(&[2]), &[(u64::MAX / 2) & !3], 0);
        assert!(wide.is_ok());
        assert!(matches!(
            wide.unwrap().byte_offset(&[2]),
            Err(CoreError::IndexOutOfRange { .. })
        ));
    }

    /// A random view of a random contiguous f32 tensor: a list of operations.
    #[derive(Clone, Debug)]
    enum Op {
        Transpose(usize, usize),
        Narrow(usize, u64, u64),
        Select(usize, u64),
    }

    fn apply(d: TensorDesc, op: &Op) -> Option<TensorDesc> {
        let r = d.rank().max(1);
        match *op {
            Op::Transpose(a, b) => d.transpose(a % r, b % r).ok(),
            Op::Narrow(dim, s, l) => {
                let dim = dim % r;
                let size = *d.dims().get(dim)?;
                let s = s % (size + 1);
                d.narrow(dim, s, l % (size - s + 1)).ok()
            }
            Op::Select(dim, i) => {
                let dim = dim % r;
                let size = *d.dims().get(dim)?;
                (size > 0).then(|| d.select(dim, i % size).ok())?
            }
        }
    }

    /// Every index of `d` in row-major order.
    fn indices(dims: &[u64]) -> Vec<Vec<u64>> {
        let mut out = vec![vec![]];
        for &n in dims {
            out = out
                .into_iter()
                .flat_map(|p| {
                    (0..n).map(move |i| {
                        let mut q = p.clone();
                        q.push(i);
                        q
                    })
                })
                .collect();
        }
        out
    }

    proptest! {
        /// Views never reach past `required_bytes` or past the base storage, a
        /// non-overlapping verdict is true, and a zero-copy reshape keeps the
        /// row-major element order.
        #[test]
        fn views_stay_in_bounds(
            dims in proptest::collection::vec(0u64..5, 1..=4),
            ops in proptest::collection::vec(
                prop_oneof![
                    (0usize..4, 0usize..4).prop_map(|(a, b)| Op::Transpose(a, b)),
                    (0usize..4, 0u64..5, 0u64..5).prop_map(|(d, s, l)| Op::Narrow(d, s, l)),
                    (0usize..4, 0u64..5).prop_map(|(d, i)| Op::Select(d, i)),
                ],
                0..5,
            ),
        ) {
            let base = f32s(&dims);
            let storage = base.required_bytes().unwrap();
            let mut d = base;
            for op in &ops {
                if let Some(n) = apply(d, op) {
                    d = n;
                }
            }
            let req = d.required_bytes().unwrap();
            prop_assert!(req <= storage);
            let all = indices(d.dims());
            let offs: Vec<u64> = all.iter().map(|i| d.byte_offset(i).unwrap()).collect();
            for &o in &offs {
                prop_assert!(o + 4 <= req);
            }
            let mut sorted = offs.clone();
            sorted.sort_unstable();
            sorted.dedup();
            prop_assert_eq!(sorted.len(), offs.len(), "views without broadcast never overlap");
            prop_assert!(d.is_non_overlapping());
            if d.is_contiguous() && !offs.is_empty() {
                prop_assert!(offs.windows(2).all(|w| w[1] == w[0] + 4));
            }
            let flat = shape(&[d.numel().unwrap()]);
            if let Ok(r) = d.reshape(&flat) {
                for (k, &o) in offs.iter().enumerate() {
                    prop_assert_eq!(r.byte_offset(&[k as u64]).unwrap(), o);
                }
            }
        }
    }
}
