//! Page-aligned host memory that a unified-memory GPU can use in place (ADR-011 §4).
//!
//! Metal wraps host memory without a copy only if the pointer is page-aligned and the
//! length a whole number of pages. [`PageAligned`] gets that from an ordinary `Vec`
//! (over-allocated by one page, then aligned inside), so no unsafe code is needed.

/// The alignment and length granule: 16 KiB, the page size of Apple Silicon (a multiple
/// of the 4 KiB page elsewhere).
pub const DEVICE_PAGE: usize = 16 * 1024;

/// A zero-initialised, page-aligned byte region whose padded length is a whole number of
/// [`DEVICE_PAGE`]s. It never reallocates, so its address is stable for its lifetime.
pub struct PageAligned {
    buf: Vec<u8>,
    start: usize,
    len: usize,
    padded: usize,
}

impl std::fmt::Debug for PageAligned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PageAligned")
            .field("len", &self.len)
            .field("padded", &self.padded)
            .finish_non_exhaustive()
    }
}

impl PageAligned {
    /// `len` zero bytes (at least one page is reserved, so the region is never empty).
    #[must_use]
    pub fn zeroed(len: usize) -> Self {
        let padded = len.max(1).div_ceil(DEVICE_PAGE) * DEVICE_PAGE;
        let buf = vec![0u8; padded + DEVICE_PAGE];
        let start = buf.as_ptr().align_offset(DEVICE_PAGE);
        debug_assert!(start < DEVICE_PAGE);
        Self {
            buf,
            start,
            len,
            padded,
        }
    }

    /// The region holding `len` bytes, reusing this allocation when its padded length
    /// is large enough (the address stays the same) and allocating a zeroed one
    /// otherwise. Reused bytes keep their old contents; callers overwrite them.
    #[must_use]
    pub fn resized(mut self, len: usize) -> Self {
        if len <= self.padded {
            self.len = len;
            self
        } else {
            Self::zeroed(len)
        }
    }

    /// The `len` bytes asked for.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.buf[self.start..self.start + self.len]
    }

    /// The `len` bytes asked for, mutably.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buf[self.start..self.start + self.len]
    }

    /// The whole page-aligned region: [`PageAligned::as_slice`] plus zero padding to a
    /// page boundary.
    #[must_use]
    pub fn padded(&self) -> &[u8] {
        &self.buf[self.start..self.start + self.padded]
    }

    /// Bytes asked for.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether no bytes were asked for.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether `bytes` lies inside this region; if so, its offset from the start.
    #[must_use]
    pub fn offset_of(&self, bytes: &[u8]) -> Option<usize> {
        let base = self.as_slice().as_ptr() as usize;
        let p = bytes.as_ptr() as usize;
        let off = p.checked_sub(base)?;
        (off.checked_add(bytes.len())? <= self.len).then_some(off)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_padded_and_zeroed() {
        for len in [
            0,
            1,
            DEVICE_PAGE - 1,
            DEVICE_PAGE,
            DEVICE_PAGE + 1,
            5 * DEVICE_PAGE,
        ] {
            let mut r = PageAligned::zeroed(len);
            assert_eq!(r.as_slice().as_ptr() as usize % DEVICE_PAGE, 0);
            assert_eq!(r.len(), len);
            assert_eq!(r.padded().len() % DEVICE_PAGE, 0);
            assert!(r.padded().len() >= len.max(1));
            assert!(r.padded().iter().all(|&b| b == 0));
            r.as_mut_slice().fill(7);
            assert!(r.padded()[len..].iter().all(|&b| b == 0));
        }
    }

    #[test]
    fn resized_reuses_when_it_fits() {
        let r = PageAligned::zeroed(3 * DEVICE_PAGE);
        let p = r.as_slice().as_ptr();
        let r = r.resized(DEVICE_PAGE + 5);
        assert_eq!((r.as_slice().as_ptr(), r.len()), (p, DEVICE_PAGE + 5));
        let r = r.resized(3 * DEVICE_PAGE);
        assert_eq!(r.as_slice().as_ptr(), p);
        let r = r.resized(3 * DEVICE_PAGE + 1);
        assert_eq!(r.len(), 3 * DEVICE_PAGE + 1);
        assert_eq!(r.as_slice().as_ptr() as usize % DEVICE_PAGE, 0);
    }

    #[test]
    fn offsets_only_inside() {
        let r = PageAligned::zeroed(100);
        let s = r.as_slice();
        assert_eq!(r.offset_of(&s[10..20]), Some(10));
        assert_eq!(r.offset_of(&s[100..]), Some(100));
        let other = [0u8; 4];
        assert_eq!(r.offset_of(&other), None);
        assert_eq!(r.offset_of(&r.padded()[90..110]), None, "past len");
    }
}
