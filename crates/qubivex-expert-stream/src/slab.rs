//! An expert's resident bytes (ADR-008, ADR-011 §4).

use std::any::Any;
use std::sync::{Arc, OnceLock};

use qubivex_compute::host::PageAligned;

/// A device backend's handle on a slab (for example a GPU buffer that wraps it in place).
pub type Attachment = Arc<dyn Any + Send + Sync>;

/// One expert's bytes in a page-aligned region, so a unified-memory GPU can use them in
/// place, plus a slot where a device backend keeps its handle on the region.
///
/// The handle lives exactly as long as the slab holds this expert: when the slab is
/// reused for another expert the handle is dropped first, so a backend never sees stale
/// bytes through it. The region is only written while no lease and no handle exist.
pub struct Slab {
    // Declared before `region` so it is dropped first.
    attachment: OnceLock<Attachment>,
    region: Arc<PageAligned>,
}

impl std::fmt::Debug for Slab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Slab")
            .field("len", &self.region.len())
            .field("attached", &self.attachment.get().is_some())
            .finish()
    }
}

impl Slab {
    pub(crate) fn new(region: PageAligned) -> Self {
        Self {
            attachment: OnceLock::new(),
            region: Arc::new(region),
        }
    }

    /// The expert's bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        self.region.as_slice()
    }

    /// Bytes held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.region.len()
    }

    /// Whether no bytes are held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.region.is_empty()
    }

    /// The page-aligned region, for a backend that wraps it.
    #[must_use]
    pub const fn region(&self) -> &Arc<PageAligned> {
        &self.region
    }

    /// The backend's handle, set at most once per expert load.
    #[must_use]
    pub const fn attachment(&self) -> &OnceLock<Attachment> {
        &self.attachment
    }

    /// The region back for reuse, if nothing else holds it. The attachment (and with it
    /// any backend reference to the region) is dropped first.
    pub(crate) fn into_region(self) -> Option<PageAligned> {
        let Self { attachment, region } = self;
        drop(attachment);
        Arc::try_unwrap(region).ok()
    }
}
