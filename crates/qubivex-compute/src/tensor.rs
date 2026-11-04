//! Host tensors: typed storage described by a [`TensorDesc`].
//!
//! - [`TensorView`] borrows storage, so it can never outlive it.
//! - [`TensorViewMut`] borrows mutably and requires a non-overlapping descriptor.
//! - [`Tensor`] owns storage behind an `Arc`: clones and views share it, and it is
//!   mutable only while unshared.
//!
//! Storage is typed (`f32`, `u16` or `u8`), so no byte is ever reinterpreted as another
//! type and this module needs no `unsafe`. View operations change only the descriptor;
//! [`TensorView::to_contiguous`] is the only copy. Design:
//! `docs/architecture/TENSOR_ARCHITECTURE.md`.

use std::sync::Arc;

use qubivex_core::{CoreError, DType, Shape, TensorDesc};

use crate::{ComputeError, WeightView};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u16 {}
    impl Sealed for u8 {}
}

/// A storage element type, and the dtypes it can hold.
///
/// | Element | Holds |
/// |---|---|
/// | `f32` | `f32` |
/// | `u16` | `f16`, `bf16` (raw bits) |
/// | `u8` | any dtype as raw bytes; packed dtypes are always `u8` |
pub trait Element: sealed::Sealed + Copy + Default + Send + Sync + 'static {
    /// Whether storage of this element type can hold `dtype`.
    fn holds(dtype: DType) -> bool;
}

impl Element for f32 {
    fn holds(dtype: DType) -> bool {
        dtype == DType::F32
    }
}

impl Element for u16 {
    fn holds(dtype: DType) -> bool {
        matches!(dtype, DType::F16 | DType::BF16)
    }
}

impl Element for u8 {
    fn holds(_: DType) -> bool {
        true
    }
}

const fn unit<T>() -> u64 {
    size_of::<T>() as u64
}

/// Checks that `len` elements of `T` can back `desc`: the element type holds the
/// dtype, every stride and the offset are whole elements, and the storage reaches the
/// view's furthest byte.
fn validate<T: Element>(desc: &TensorDesc, len: usize) -> Result<(), ComputeError> {
    let dtype = desc.dtype();
    if !T::holds(dtype) {
        return Err(ComputeError::ElementType {
            dtype,
            element: std::any::type_name::<T>(),
        });
    }
    let u = unit::<T>();
    if let Some(&bad) = std::iter::once(&desc.offset())
        .chain(desc.strides())
        .find(|&&s| s % u != 0)
    {
        return Err(CoreError::StrideMisaligned {
            dtype: dtype.name(),
            stride: bad,
            unit: u,
        }
        .into());
    }
    let need = desc.required_bytes()?;
    let have = (len as u64).checked_mul(u).ok_or(CoreError::Overflow {
        what: "storage bytes",
    })?;
    if need > have {
        return Err(ComputeError::Shape {
            op: "tensor view",
            detail: format!("{desc:?} needs {need} bytes of storage, got {have}"),
        });
    }
    Ok(())
}

/// The index of storage element `byte / size_of::<T>()`. The descriptor was validated
/// against the storage, so in-range byte offsets fit in `usize`.
fn elem_index<T>(byte: u64) -> Result<usize, ComputeError> {
    usize::try_from(byte / unit::<T>()).map_err(|_| {
        CoreError::Overflow {
            what: "storage index",
        }
        .into()
    })
}

/// The storage-element range of a contiguous view.
fn contiguous_range<T>(desc: &TensorDesc) -> Result<std::ops::Range<usize>, ComputeError> {
    let start = elem_index::<T>(desc.offset())?;
    let bytes = desc.dtype().storage_bytes(desc.shape())?;
    Ok(start..start + elem_index::<T>(bytes)?)
}

/// Copies every item (an element, or a packed row) of `desc` from `src`, in
/// row-major order. `src` was validated against `desc`.
fn gather<T: Element>(desc: &TensorDesc, src: &[T]) -> Result<Vec<T>, ComputeError> {
    let (item_dims, item_bytes) = match desc.dtype().elem_bytes() {
        Some(e) => (desc.dims(), e),
        None => (&desc.dims()[..desc.rank() - 1], desc.row_bytes()?),
    };
    let item = elem_index::<T>(item_bytes)?;
    let total = elem_index::<T>(desc.dtype().storage_bytes(desc.shape())?)?;
    let mut out = Vec::with_capacity(total);
    if total == 0 {
        return Ok(out);
    }
    if desc.is_contiguous() {
        out.extend_from_slice(&src[contiguous_range::<T>(desc)?]);
        return Ok(out);
    }
    // An odometer over the item index space.
    let mut index = vec![0u64; item_dims.len()];
    loop {
        let at = elem_index::<T>(desc.byte_offset(&index)?)?;
        out.extend_from_slice(&src[at..at + item]);
        let mut d = index.len();
        loop {
            if d == 0 {
                return Ok(out);
            }
            d -= 1;
            index[d] += 1;
            if index[d] < item_dims[d] {
                break;
            }
            index[d] = 0;
        }
    }
}

/// A borrowed, read-only view of typed storage.
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a, T: Element> {
    desc: TensorDesc,
    data: &'a [T],
}

impl<'a, T: Element> TensorView<'a, T> {
    /// A view of `data` described by `desc`.
    ///
    /// # Errors
    /// [`ComputeError::ElementType`] if `T` cannot hold the dtype,
    /// [`CoreError::StrideMisaligned`] if a stride or the offset is not a whole number
    /// of `T`, and [`ComputeError::Shape`] if `data` is shorter than the view needs.
    pub fn new(desc: TensorDesc, data: &'a [T]) -> Result<Self, ComputeError> {
        validate::<T>(&desc, data.len())?;
        Ok(Self { desc, data })
    }

    /// The descriptor.
    #[must_use]
    pub const fn desc(&self) -> &TensorDesc {
        &self.desc
    }

    /// The whole backing storage (not just the elements this view reaches).
    #[must_use]
    pub const fn storage(&self) -> &'a [T] {
        self.data
    }

    /// The elements in row-major order without copying, if the view is contiguous.
    ///
    /// # Errors
    /// As [`DType::storage_bytes`].
    pub fn contiguous_data(&self) -> Result<Option<&'a [T]>, ComputeError> {
        if !self.desc.is_contiguous() {
            return Ok(None);
        }
        Ok(Some(&self.data[contiguous_range::<T>(&self.desc)?]))
    }

    /// One element of an elementwise tensor whose storage element is the dtype's
    /// element (`f32` for `f32`, `u16` for `f16` and `bf16`).
    ///
    /// # Errors
    /// [`ComputeError::ElementType`] for packed dtypes or byte storage of wider
    /// elements, and as [`TensorDesc::byte_offset`].
    pub fn get(&self, index: &[u64]) -> Result<T, ComputeError> {
        if self.desc.dtype().elem_bytes() != Some(unit::<T>()) {
            return Err(ComputeError::ElementType {
                dtype: self.desc.dtype(),
                element: std::any::type_name::<T>(),
            });
        }
        Ok(self.data[elem_index::<T>(self.desc.byte_offset(index)?)?])
    }

    /// A copy of the viewed elements as a contiguous tensor. This is the only
    /// operation that copies data.
    ///
    /// # Errors
    /// As [`TensorDesc::byte_offset`] and [`DType::storage_bytes`].
    pub fn to_contiguous(&self) -> Result<Tensor<T>, ComputeError> {
        let data = gather(&self.desc, self.data)?;
        Tensor::from_vec(self.desc.dtype(), *self.desc.shape(), data)
    }

    /// The same storage under another descriptor (checked against the storage).
    fn with(&self, desc: Result<TensorDesc, CoreError>) -> Result<Self, ComputeError> {
        Self::new(desc?, self.data)
    }

    /// See [`TensorDesc::permute`].
    ///
    /// # Errors
    /// As [`TensorDesc::permute`].
    pub fn permute(&self, perm: &[usize]) -> Result<Self, ComputeError> {
        self.with(self.desc.permute(perm))
    }

    /// See [`TensorDesc::transpose`].
    ///
    /// # Errors
    /// As [`TensorDesc::transpose`].
    pub fn transpose(&self, a: usize, b: usize) -> Result<Self, ComputeError> {
        self.with(self.desc.transpose(a, b))
    }

    /// See [`TensorDesc::narrow`].
    ///
    /// # Errors
    /// As [`TensorDesc::narrow`].
    pub fn narrow(&self, dim: usize, start: u64, len: u64) -> Result<Self, ComputeError> {
        self.with(self.desc.narrow(dim, start, len))
    }

    /// See [`TensorDesc::select`].
    ///
    /// # Errors
    /// As [`TensorDesc::select`].
    pub fn select(&self, dim: usize, index: u64) -> Result<Self, ComputeError> {
        self.with(self.desc.select(dim, index))
    }

    /// See [`TensorDesc::broadcast_to`].
    ///
    /// # Errors
    /// As [`TensorDesc::broadcast_to`].
    pub fn broadcast_to(&self, shape: &Shape) -> Result<Self, ComputeError> {
        self.with(self.desc.broadcast_to(shape))
    }

    /// See [`TensorDesc::reshape`]. Never copies: on
    /// [`CoreError::ReshapeNeedsCopy`], call [`to_contiguous`](Self::to_contiguous)
    /// first.
    ///
    /// # Errors
    /// As [`TensorDesc::reshape`].
    pub fn reshape(&self, shape: &Shape) -> Result<Self, ComputeError> {
        self.with(self.desc.reshape(shape))
    }
}

impl<'a> TryFrom<TensorView<'a, u8>> for WeightView<'a> {
    type Error = ComputeError;

    /// A kernel weight view of a contiguous rank-2 byte view.
    fn try_from(v: TensorView<'a, u8>) -> Result<Self, ComputeError> {
        let dims = v.desc().dims();
        let (&[rows, cols], true) = (dims, v.desc().is_contiguous()) else {
            return Err(ComputeError::Shape {
                op: "weight view",
                detail: format!("needs a contiguous rank-2 tensor, got {:?}", v.desc()),
            });
        };
        let size = |n: u64| {
            usize::try_from(n).map_err(|_| CoreError::Overflow {
                what: "weight dimension",
            })
        };
        let bytes = v.contiguous_data()?.unwrap_or_default();
        Self::new(v.desc().dtype(), size(rows)?, size(cols)?, bytes)
    }
}

/// A borrowed, mutable view of typed storage. No two indices reach the same element.
#[derive(Debug)]
pub struct TensorViewMut<'a, T: Element> {
    desc: TensorDesc,
    data: &'a mut [T],
}

impl<'a, T: Element> TensorViewMut<'a, T> {
    /// A mutable view of `data` described by `desc`.
    ///
    /// # Errors
    /// As [`TensorView::new`], and [`ComputeError::Shape`] if `desc` overlaps itself
    /// (for example a broadcast).
    pub fn new(desc: TensorDesc, data: &'a mut [T]) -> Result<Self, ComputeError> {
        validate::<T>(&desc, data.len())?;
        if !desc.is_non_overlapping() {
            return Err(ComputeError::Shape {
                op: "mutable tensor view",
                detail: format!("{desc:?} reaches some elements more than once"),
            });
        }
        Ok(Self { desc, data })
    }

    /// The descriptor.
    #[must_use]
    pub const fn desc(&self) -> &TensorDesc {
        &self.desc
    }

    /// A read-only view of the same elements.
    #[must_use]
    pub fn as_view(&self) -> TensorView<'_, T> {
        TensorView {
            desc: self.desc,
            data: self.data,
        }
    }

    /// A shorter-lived mutable view of the same elements, so a view can be narrowed
    /// more than once.
    pub fn reborrow(&mut self) -> TensorViewMut<'_, T> {
        TensorViewMut {
            desc: self.desc,
            data: self.data,
        }
    }

    /// See [`TensorDesc::narrow`]; the view keeps the storage borrow.
    ///
    /// # Errors
    /// As [`TensorDesc::narrow`].
    pub fn narrow(self, dim: usize, start: u64, len: u64) -> Result<Self, ComputeError> {
        let desc = self.desc.narrow(dim, start, len)?;
        Self::new(desc, self.data)
    }

    /// See [`TensorDesc::select`]; the view keeps the storage borrow.
    ///
    /// # Errors
    /// As [`TensorDesc::select`].
    pub fn select(self, dim: usize, index: u64) -> Result<Self, ComputeError> {
        let desc = self.desc.select(dim, index)?;
        Self::new(desc, self.data)
    }

    /// The elements in row-major order, mutably and without copying, if the view is
    /// contiguous.
    ///
    /// # Errors
    /// As [`DType::storage_bytes`].
    pub fn contiguous_data_mut(self) -> Result<Option<&'a mut [T]>, ComputeError> {
        if !self.desc.is_contiguous() {
            return Ok(None);
        }
        let range = contiguous_range::<T>(&self.desc)?;
        Ok(Some(&mut self.data[range]))
    }

    /// Sets one element. Same rules as [`TensorView::get`].
    ///
    /// # Errors
    /// As [`TensorView::get`].
    pub fn set(&mut self, index: &[u64], value: T) -> Result<(), ComputeError> {
        if self.desc.dtype().elem_bytes() != Some(unit::<T>()) {
            return Err(ComputeError::ElementType {
                dtype: self.desc.dtype(),
                element: std::any::type_name::<T>(),
            });
        }
        let i = elem_index::<T>(self.desc.byte_offset(index)?)?;
        self.data[i] = value;
        Ok(())
    }
}

/// An owned tensor. Storage is shared by `Arc`: [`Clone`] and view operations never
/// copy data, and storage lives until its last holder (including an in-flight
/// operation) drops it.
#[derive(Clone, Debug)]
pub struct Tensor<T: Element> {
    desc: TensorDesc,
    data: Arc<[T]>,
}

impl<T: Element> Tensor<T> {
    /// A tensor over `data`, described by `desc`.
    ///
    /// # Errors
    /// As [`TensorView::new`].
    pub fn new(desc: TensorDesc, data: impl Into<Arc<[T]>>) -> Result<Self, ComputeError> {
        let data = data.into();
        validate::<T>(&desc, data.len())?;
        Ok(Self { desc, data })
    }

    /// A contiguous tensor that takes `data`, which must be exactly the storage size.
    ///
    /// # Errors
    /// As [`TensorDesc::contiguous`] and [`TensorView::new`], and
    /// [`ComputeError::Shape`] if `data` is longer or shorter than the tensor.
    pub fn from_vec(dtype: DType, shape: Shape, data: Vec<T>) -> Result<Self, ComputeError> {
        let desc = TensorDesc::contiguous(dtype, shape)?;
        let bytes = dtype.storage_bytes(&shape)?;
        if (data.len() as u64).checked_mul(unit::<T>()) != Some(bytes) {
            return Err(ComputeError::Shape {
                op: "tensor",
                detail: format!(
                    "{dtype} {shape} is {bytes} bytes, got {} elements of {} bytes",
                    data.len(),
                    unit::<T>()
                ),
            });
        }
        Self::new(desc, data)
    }

    /// A contiguous tensor of zero bits.
    ///
    /// # Errors
    /// As [`from_vec`](Self::from_vec).
    pub fn zeros(dtype: DType, shape: Shape) -> Result<Self, ComputeError> {
        let n = elem_index::<T>(dtype.storage_bytes(&shape)?)?;
        Self::from_vec(dtype, shape, vec![T::default(); n])
    }

    /// The descriptor.
    #[must_use]
    pub const fn desc(&self) -> &TensorDesc {
        &self.desc
    }

    /// A borrowed view; it cannot outlive this tensor.
    ///
    /// ```compile_fail
    /// use qubivex_compute::tensor::{Tensor, TensorView};
    /// use qubivex_core::{DType, Shape};
    /// let view: TensorView<'_, f32>;
    /// {
    ///     let t = Tensor::<f32>::zeros(DType::F32, Shape::new(&[2]).unwrap()).unwrap();
    ///     view = t.view();
    /// } // `t` dropped here while still borrowed
    /// let _ = view.get(&[0]);
    /// ```
    #[must_use]
    pub fn view(&self) -> TensorView<'_, T> {
        TensorView {
            desc: self.desc,
            data: &self.data,
        }
    }

    /// A mutable view, only while no clone shares the storage.
    ///
    /// # Errors
    /// [`ComputeError::Shared`] if the storage is shared, or as
    /// [`TensorViewMut::new`].
    pub fn view_mut(&mut self) -> Result<TensorViewMut<'_, T>, ComputeError> {
        let data = Arc::get_mut(&mut self.data).ok_or(ComputeError::Shared)?;
        TensorViewMut::new(self.desc, data)
    }

    /// The same storage under another descriptor, without copying; for example
    /// `t.with_desc(t.desc().transpose(0, 1)?)`.
    ///
    /// # Errors
    /// As [`TensorView::new`].
    pub fn with_desc(&self, desc: TensorDesc) -> Result<Self, ComputeError> {
        validate::<T>(&desc, self.data.len())?;
        Ok(Self {
            desc,
            data: Arc::clone(&self.data),
        })
    }

    /// Whether two tensors share storage.
    #[must_use]
    pub fn shares_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qubivex_core::GgmlType;

    fn shape(d: &[u64]) -> Shape {
        Shape::new(d).unwrap()
    }

    /// An f32 tensor holding 0, 1, 2, … in row-major order.
    fn iota(d: &[u64]) -> Tensor<f32> {
        let n = shape(d).numel().unwrap();
        let v = (0..n).map(|i| i as f32).collect();
        Tensor::from_vec(DType::F32, shape(d), v).unwrap()
    }

    fn values(v: &TensorView<'_, f32>) -> Vec<f32> {
        v.to_contiguous()
            .unwrap()
            .view()
            .contiguous_data()
            .unwrap()
            .unwrap()
            .to_vec()
    }

    #[test]
    fn transpose_shares_storage_and_reads_transposed() {
        let t = iota(&[2, 3]);
        let tt = t.with_desc(t.desc().transpose(0, 1).unwrap()).unwrap();
        assert!(tt.shares_storage(&t));
        assert_eq!(tt.view().storage().as_ptr(), t.view().storage().as_ptr());
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(
                    t.view().get(&[i, j]).unwrap().to_bits(),
                    tt.view().get(&[j, i]).unwrap().to_bits()
                );
            }
        }
        assert_eq!(values(&tt.view()), [0., 3., 1., 4., 2., 5.]);
        assert_eq!(tt.view().contiguous_data().unwrap(), None);
    }

    #[test]
    fn slices_offsets_and_contiguous_conversion() {
        let t = iota(&[4, 6]);
        let v = t.view().narrow(0, 1, 2).unwrap().narrow(1, 2, 3).unwrap();
        assert_eq!(values(&v), [8., 9., 10., 14., 15., 16.]);
        let row = t.view().select(0, 3).unwrap();
        assert_eq!(
            row.contiguous_data().unwrap().unwrap(),
            [18., 19., 20., 21., 22., 23.]
        );
        let col = t.view().select(1, 5).unwrap();
        assert_eq!(values(&col), [5., 11., 17., 23.]);
        // A copy is a new storage and contiguous.
        let c = v.to_contiguous().unwrap();
        assert!(c.desc().is_contiguous() && !c.shares_storage(&t));
    }

    #[test]
    fn broadcasting_repeats_without_copying() {
        let t = iota(&[3, 1]);
        let b = t.view().broadcast_to(&shape(&[2, 3, 2])).unwrap();
        assert_eq!(b.storage().len(), 3);
        assert_eq!(values(&b), [0., 0., 1., 1., 2., 2., 0., 0., 1., 1., 2., 2.]);
    }

    #[test]
    fn reshape_is_a_view_or_an_explicit_refusal() {
        let t = iota(&[2, 3]);
        let r = t.view().reshape(&shape(&[3, 2])).unwrap();
        assert_eq!(values(&r), [0., 1., 2., 3., 4., 5.]);
        let tr = t.view().transpose(0, 1).unwrap();
        let err = tr.reshape(&shape(&[6])).unwrap_err();
        assert!(matches!(
            err,
            ComputeError::Core(CoreError::ReshapeNeedsCopy { .. })
        ));
        let c = tr.to_contiguous().unwrap();
        assert_eq!(
            values(&c.view().reshape(&shape(&[6])).unwrap()),
            [0., 3., 1., 4., 2., 5.]
        );
    }

    #[test]
    fn packed_rows_copy_byte_exact() {
        // Two Q8_0 rows of 64 elements (2 blocks of 34 bytes each), viewed transposed
        // over a third outer dimension.
        let q8 = DType::Ggml(GgmlType::Q8_0);
        let bytes: Vec<u8> = (0..3 * 2 * 68)
            .map(|i| u8::try_from(i % 251).unwrap())
            .collect();
        let t = Tensor::from_vec(q8, shape(&[3, 2, 64]), bytes.clone()).unwrap();
        let v = t.view().transpose(0, 1).unwrap();
        let c = v.to_contiguous().unwrap();
        let got = c.view().contiguous_data().unwrap().unwrap();
        // Row (i, j) of the copy is row (j, i) of the original.
        for i in 0..2 {
            for j in 0..3 {
                let dst = (i * 3 + j) * 68;
                let src = (j * 2 + i) * 68;
                assert_eq!(&got[dst..dst + 68], &bytes[src..src + 68]);
            }
        }
        // Block-aligned slice of the packed rows: the second block of each row.
        let half = t.view().narrow(2, 32, 32).unwrap().to_contiguous().unwrap();
        let h = half.view().contiguous_data().unwrap().unwrap();
        assert_eq!(h.len(), 6 * 34);
        assert_eq!(&h[..34], &bytes[34..68]);
    }

    #[test]
    fn invalid_views_fail_safely() {
        let d = TensorDesc::contiguous(DType::F32, shape(&[2, 3])).unwrap();
        // Storage too short.
        assert!(matches!(
            TensorView::new(d, &[0f32; 5]),
            Err(ComputeError::Shape { .. })
        ));
        // Wrong element type for the dtype.
        assert!(matches!(
            TensorView::new(d, &[0u16; 12]),
            Err(ComputeError::ElementType { .. })
        ));
        let q = TensorDesc::contiguous(DType::Ggml(GgmlType::Q4_K), shape(&[1, 256])).unwrap();
        assert!(matches!(
            TensorView::<f32>::new(q, &[0.; 64]),
            Err(ComputeError::ElementType { .. })
        ));
        // Bytes may hold any dtype, but `get` refuses to read a 4-byte element as a byte.
        let b = TensorView::new(d, &[0u8; 24]).unwrap();
        assert!(matches!(
            b.get(&[0, 0]),
            Err(ComputeError::ElementType { .. })
        ));
        // A view at an offset needs storage up to its furthest element.
        let odd = TensorDesc::from_parts(DType::BF16, shape(&[2]), &[2], 2).unwrap();
        assert!(TensorView::new(odd, &[0u16; 3]).is_ok());
        assert!(matches!(
            TensorView::<u16>::new(odd, &[0u16; 2]),
            Err(ComputeError::Shape { .. })
        ));
        // Out-of-range access is an error, not a panic.
        let t = iota(&[2, 3]);
        assert!(t.view().get(&[2, 0]).is_err());
        assert!(t.view().get(&[0]).is_err());
        assert!(Tensor::from_vec(DType::F32, shape(&[2, 3]), vec![0f32; 7]).is_err());
    }

    #[test]
    fn mutation_needs_unique_storage_and_no_overlap() {
        let mut t = iota(&[2, 3]);
        t.view_mut().unwrap().set(&[1, 2], 50.).unwrap();
        assert_eq!(t.view().get(&[1, 2]).unwrap().to_bits(), 50f32.to_bits());
        let shared = t.clone();
        assert!(matches!(t.view_mut(), Err(ComputeError::Shared)));
        drop(shared);
        assert!(t.view_mut().is_ok());
        let mut data = [0f32; 3];
        let bcast = TensorDesc::contiguous(DType::F32, shape(&[3]))
            .unwrap()
            .broadcast_to(&shape(&[2, 3]))
            .unwrap();
        assert!(matches!(
            TensorViewMut::new(bcast, &mut data),
            Err(ComputeError::Shape { .. })
        ));
        let mut transposed = TensorViewMut::new(
            t.desc().transpose(0, 1).unwrap(),
            Arc::get_mut(&mut t.data).unwrap(),
        )
        .unwrap();
        transposed.set(&[0, 1], -1.).unwrap();
        assert_eq!(
            transposed.as_view().get(&[0, 1]).unwrap().to_bits(),
            (-1f32).to_bits()
        );
        assert_eq!(t.view().get(&[1, 0]).unwrap().to_bits(), (-1f32).to_bits());
    }

    #[test]
    fn mutable_views_narrow_and_expose_contiguous_runs() {
        let mut t = iota(&[2, 4, 3]);
        let mut m = t.view_mut().unwrap();
        // Row 1 of head 1: contiguous, written through a slice.
        let run = m.reborrow().select(0, 1).unwrap().narrow(0, 1, 1).unwrap();
        run.contiguous_data_mut()
            .unwrap()
            .unwrap()
            .copy_from_slice(&[-1., -2., -3.]);
        // A column is strided: no slice.
        let col = m.reborrow().narrow(2, 0, 1).unwrap();
        assert!(col.contiguous_data_mut().unwrap().is_none());
        assert!(m.reborrow().narrow(1, 3, 2).is_err());
        let v = t.view();
        for (j, want) in [-1f32, -2., -3.].into_iter().enumerate() {
            assert_eq!(v.get(&[1, 1, j as u64]).unwrap().to_bits(), want.to_bits());
        }
        assert_eq!(v.get(&[1, 0, 2]).unwrap().to_bits(), 14f32.to_bits());
        assert_eq!(v.get(&[1, 2, 0]).unwrap().to_bits(), 18f32.to_bits());
    }

    #[test]
    fn weight_view_from_tensor_view() {
        let q4 = DType::Ggml(GgmlType::Q4_K);
        let t = Tensor::<u8>::zeros(q4, shape(&[4, 3, 512])).unwrap();
        // One "expert" of a stacked tensor, without copying.
        let e = t.view().select(0, 2).unwrap();
        let w = WeightView::try_from(e).unwrap();
        assert_eq!((w.rows(), w.cols(), w.row_bytes()), (3, 512, 288));
        assert_eq!(
            w.bytes().as_ptr(),
            t.view().storage()[2 * 3 * 288..].as_ptr()
        );
        // A transposed or sliced weight is not a kernel view.
        assert!(
            WeightView::try_from(t.view().transpose(0, 1).unwrap().select(0, 0).unwrap()).is_err()
        );
        assert!(WeightView::try_from(t.view()).is_err(), "rank 3");
        let f = Tensor::<u8>::zeros(DType::F32, shape(&[2, 4])).unwrap();
        assert!(WeightView::try_from(f.view().narrow(1, 0, 2).unwrap()).is_err());
    }
}
