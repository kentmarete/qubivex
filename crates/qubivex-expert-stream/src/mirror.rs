//! Reading experts from several copies of a model's files (ADR-006 amendment 4).
//!
//! [`MirroredSource`] reads each expert from the first healthy copy and, when that read
//! fails, from the next, so a failing drive costs a slower read instead of a failed
//! request. A copy that fails [`MirrorPolicy::trip`] reads in a row is skipped for
//! [`MirrorPolicy::cooldown`], then tried again; a success clears its count. When every
//! copy is skipped, all are tried anyway: skipping is a preference, never a refusal.
//!
//! Copies must hold the same bytes. Sizes come from the first copy. Put a
//! [`crate::fault::VerifiedSource`] around each copy to fail over on corrupt bytes too.
//!
//! Not handled here: a read that hangs. It blocks its I/O thread, so this source never
//! sees it end; the store's deadline ([`crate::ReadTimeout`]) fails its waiters instead.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::{ExpertKey, ExpertSource, StoreError};

/// When a failing copy is skipped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MirrorPolicy {
    /// Failed reads in a row after which a copy is skipped.
    pub trip: u32,
    /// How long a tripped copy is skipped before it is tried again.
    pub cooldown: Duration,
}

impl Default for MirrorPolicy {
    fn default() -> Self {
        Self {
            trip: 3,
            cooldown: Duration::from_secs(30),
        }
    }
}

/// One copy's record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyHealth {
    /// The copy's name (for example its directory).
    pub name: String,
    /// Reads served.
    pub reads: u64,
    /// Reads failed.
    pub failures: u64,
    /// Failed reads since its last success.
    pub failing: u32,
    /// Whether it is being skipped now.
    pub skipped: bool,
}

struct Copy {
    name: String,
    source: Arc<dyn ExpertSource>,
    reads: AtomicU64,
    failures: AtomicU64,
    /// Failures in a row, and when the last one happened.
    state: Mutex<(u32, Option<Instant>)>,
}

impl Copy {
    fn skipped(&self, p: MirrorPolicy) -> bool {
        let s = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        s.0 >= p.trip && s.1.is_some_and(|t| t.elapsed() < p.cooldown)
    }

    fn record(&self, ok: bool) {
        let mut s = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if ok {
            self.reads.fetch_add(1, Ordering::Relaxed);
            *s = (0, None);
        } else {
            self.failures.fetch_add(1, Ordering::Relaxed);
            *s = (s.0.saturating_add(1), Some(Instant::now()));
        }
    }
}

/// An [`ExpertSource`] over several copies of the same experts, failing over between
/// them.
pub struct MirroredSource {
    copies: Vec<Copy>,
    policy: MirrorPolicy,
    failovers: AtomicU64,
}

impl std::fmt::Debug for MirroredSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MirroredSource")
            .field("copies", &self.health())
            .field("failovers", &self.failovers())
            .finish_non_exhaustive()
    }
}

impl MirroredSource {
    /// Reads from `copies` (name, source), preferring them in order.
    ///
    /// # Panics
    /// If `copies` is empty.
    #[must_use]
    pub fn new(
        copies: impl IntoIterator<Item = (String, Arc<dyn ExpertSource>)>,
        policy: MirrorPolicy,
    ) -> Self {
        let copies: Vec<Copy> = copies
            .into_iter()
            .map(|(name, source)| Copy {
                name,
                source,
                reads: AtomicU64::new(0),
                failures: AtomicU64::new(0),
                state: Mutex::new((0, None)),
            })
            .collect();
        assert!(
            !copies.is_empty(),
            "a mirrored source needs at least one copy"
        );
        Self {
            copies,
            policy,
            failovers: AtomicU64::new(0),
        }
    }

    /// Each copy's record, in preference order.
    #[must_use]
    pub fn health(&self) -> Vec<CopyHealth> {
        self.copies
            .iter()
            .map(|c| {
                // Read the count and release the lock before `skipped` takes it again.
                let failing = c
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .0;
                CopyHealth {
                    name: c.name.clone(),
                    reads: c.reads.load(Ordering::Relaxed),
                    failures: c.failures.load(Ordering::Relaxed),
                    failing,
                    skipped: c.skipped(self.policy),
                }
            })
            .collect()
    }

    /// Reads served by a copy other than the first one tried.
    #[must_use]
    pub fn failovers(&self) -> u64 {
        self.failovers.load(Ordering::Relaxed)
    }
}

impl ExpertSource for MirroredSource {
    fn expert_size(&self, key: ExpertKey) -> Result<u64, StoreError> {
        self.copies[0].source.expert_size(key)
    }

    fn read_expert(&self, key: ExpertKey, dst: &mut [u8]) -> Result<(), StoreError> {
        // Healthy copies first, in preference order, then the skipped ones.
        let (healthy, skipped): (Vec<&Copy>, Vec<&Copy>) =
            self.copies.iter().partition(|c| !c.skipped(self.policy));
        let mut errors = Vec::new();
        for (i, c) in healthy.into_iter().chain(skipped).enumerate() {
            // A panic in one copy is that copy's failure, not the whole read's.
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                c.source.read_expert(key, dst)
            }));
            match r {
                Ok(Ok(())) => {
                    c.record(true);
                    if i > 0 {
                        self.failovers.fetch_add(1, Ordering::Relaxed);
                    }
                    return Ok(());
                }
                Ok(Err(e)) => {
                    c.record(false);
                    errors.push(format!("{}: {}", c.name, detail(e)));
                }
                Err(_) => {
                    c.record(false);
                    errors.push(format!("{}: the source panicked", c.name));
                }
            }
        }
        Err(StoreError::Load {
            key,
            detail: format!("every copy failed ({})", errors.join("; ")),
        })
    }
}

fn detail(e: StoreError) -> String {
    match e {
        StoreError::Load { detail, .. } => detail,
        other => other.to_string(),
    }
}
