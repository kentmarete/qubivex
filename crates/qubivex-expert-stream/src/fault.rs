//! Fault injection and verification for expert sources (IMPROVEMENTS_EVALUATION §10).
//!
//! [`FaultySource`] wraps any [`ExpertSource`] and makes chosen reads fail the way
//! storage fails: an I/O error, a short read that leaves the buffer partly written,
//! corrupt bytes reported as a success, a panic, or a slow read. The fault suites in
//! `tests/fault.rs` (store) and the runtime's `tests/faults.rs` (engine) use it to check
//! that every failure is reported, never panics, never yields wrong tokens and leaks
//! nothing.
//!
//! Corrupt bytes that a read reports as good are invisible to the store: the bytes have
//! the right size. [`VerifiedSource`] catches them by comparing each read with a
//! checksum recorded from a known-good read ([`checksum`]).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use qubivex_core::LayerIdx;

use crate::{ExpertKey, ExpertSource, StoreError};

/// How one read fails.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Fault {
    /// The read fails with an I/O error (`EIO`), writing nothing.
    Eio,
    /// Only the first `keep` bytes arrive; the rest of the buffer is overwritten with
    /// garbage, and the read reports end of file.
    ShortRead {
        /// Bytes read before the end of file.
        keep: usize,
    },
    /// The read succeeds, but the byte at `offset` (modulo the expert's size) is
    /// XORed with `xor` (non-zero).
    Corrupt {
        /// Which byte.
        offset: usize,
        /// The bits flipped.
        xor: u8,
    },
    /// The source panics mid-read, after writing part of the buffer.
    Panic,
    /// The read succeeds after a delay.
    Delay(Duration),
    /// The read blocks, as on a failing drive or a lost network mount, until
    /// [`FaultySource::release_hangs`]; then it succeeds.
    Hang,
}

/// Which reads a rule watches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// Every read.
    Any,
    /// Reads of one layer's experts.
    Layer(LayerIdx),
    /// Reads of one expert.
    Expert(ExpertKey),
}

impl Target {
    const fn matches(self, key: ExpertKey) -> bool {
        match self {
            Self::Any => true,
            Self::Layer(l) => key.0.0 == l.0,
            Self::Expert(k) => k.0.0 == key.0.0 && k.1.0 == key.1.0,
        }
    }
}

/// A fault on some reads: of the reads `target` matches, the first `skip` pass, the
/// next `times` fail with `fault`, and the rest pass.
#[derive(Clone, Debug)]
pub struct Rule {
    /// The reads watched.
    pub target: Target,
    /// Matching reads that pass before the fault starts.
    pub skip: u64,
    /// Matching reads that fail (`u64::MAX`: all of them).
    pub times: u64,
    /// How they fail.
    pub fault: Fault,
}

impl Rule {
    /// `fault` on every read of `target`, from the first.
    #[must_use]
    pub const fn always(target: Target, fault: Fault) -> Self {
        Self {
            target,
            skip: 0,
            times: u64::MAX,
            fault,
        }
    }

    /// `fault` on the read of `target` after `skip` good ones, once.
    #[must_use]
    pub const fn once_after(target: Target, skip: u64, fault: Fault) -> Self {
        Self {
            target,
            skip,
            times: 1,
            fault,
        }
    }
}

struct Armed {
    rule: Rule,
    seen: AtomicU64,
}

/// An [`ExpertSource`] that injects [`Fault`]s into chosen reads of another.
///
/// Rules are checked in order and the first that fires on a read decides its fault.
/// Sizes are never faked: [`ExpertSource::expert_size`] is the inner source's.
pub struct FaultySource {
    inner: Arc<dyn ExpertSource>,
    rules: Vec<Armed>,
    reads: AtomicU64,
    injected: AtomicU64,
    /// Set once hung reads may return.
    released: Mutex<bool>,
    release: Condvar,
    hanging: AtomicU64,
}

impl std::fmt::Debug for FaultySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FaultySource")
            .field(
                "rules",
                &self.rules.iter().map(|a| &a.rule).collect::<Vec<_>>(),
            )
            .field("reads", &self.reads())
            .field("injected", &self.injected())
            .finish_non_exhaustive()
    }
}

impl FaultySource {
    /// `inner` with the faults of `rules`.
    #[must_use]
    pub fn new(inner: Arc<dyn ExpertSource>, rules: impl IntoIterator<Item = Rule>) -> Self {
        Self {
            inner,
            rules: rules
                .into_iter()
                .map(|rule| Armed {
                    rule,
                    seen: AtomicU64::new(0),
                })
                .collect(),
            reads: AtomicU64::new(0),
            injected: AtomicU64::new(0),
            released: Mutex::new(false),
            release: Condvar::new(),
            hanging: AtomicU64::new(0),
        }
    }

    /// Lets every hung read (and every later [`Fault::Hang`]) return.
    pub fn release_hangs(&self) {
        *self
            .released
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        self.release.notify_all();
    }

    /// Reads blocked in [`Fault::Hang`] now.
    #[must_use]
    pub fn hanging(&self) -> u64 {
        self.hanging.load(Ordering::SeqCst)
    }

    /// Reads asked for so far.
    #[must_use]
    pub fn reads(&self) -> u64 {
        self.reads.load(Ordering::Relaxed)
    }

    /// Reads given a fault so far.
    #[must_use]
    pub fn injected(&self) -> u64 {
        self.injected.load(Ordering::Relaxed)
    }

    /// The fault for this read of `key`, if a rule fires. Every matching rule counts
    /// the read, whether or not an earlier rule fired.
    fn fault_for(&self, key: ExpertKey) -> Option<Fault> {
        let mut fired = None;
        for a in &self.rules {
            if !a.rule.target.matches(key) {
                continue;
            }
            let n = a.seen.fetch_add(1, Ordering::Relaxed);
            let on = n >= a.rule.skip && n - a.rule.skip < a.rule.times;
            if on && fired.is_none() {
                fired = Some(a.rule.fault.clone());
            }
        }
        fired
    }
}

fn load_err(key: ExpertKey, detail: impl Into<String>) -> StoreError {
    StoreError::Load {
        key,
        detail: detail.into(),
    }
}

impl ExpertSource for FaultySource {
    fn expert_size(&self, key: ExpertKey) -> Result<u64, StoreError> {
        self.inner.expert_size(key)
    }

    fn read_expert(&self, key: ExpertKey, dst: &mut [u8]) -> Result<(), StoreError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        let Some(fault) = self.fault_for(key) else {
            return self.inner.read_expert(key, dst);
        };
        self.injected.fetch_add(1, Ordering::Relaxed);
        match fault {
            Fault::Eio => Err(load_err(
                key,
                format!("injected: {}", std::io::Error::from_raw_os_error(5)),
            )),
            Fault::ShortRead { keep } => {
                self.inner.read_expert(key, dst)?;
                let keep = keep.min(dst.len());
                dst[keep..].fill(0xA5);
                Err(load_err(
                    key,
                    format!(
                        "injected: short read, {keep} of {} bytes before end of file",
                        dst.len()
                    ),
                ))
            }
            Fault::Corrupt { offset, xor } => {
                self.inner.read_expert(key, dst)?;
                if !dst.is_empty() {
                    dst[offset % dst.len()] ^= xor;
                }
                Ok(())
            }
            Fault::Panic => {
                let half = dst.len() / 2;
                dst[..half].fill(0x5A);
                panic!("injected: expert source panicked reading {key:?}");
            }
            Fault::Delay(d) => {
                std::thread::sleep(d);
                self.inner.read_expert(key, dst)
            }
            Fault::Hang => {
                self.hanging.fetch_add(1, Ordering::SeqCst);
                let mut released = self
                    .released
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                while !*released {
                    released = self
                        .release
                        .wait(released)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                drop(released);
                self.hanging.fetch_sub(1, Ordering::SeqCst);
                self.inner.read_expert(key, dst)
            }
        }
    }
}

/// A 64-bit checksum of `bytes`: four multiply-rotate lanes over little-endian words,
/// then the tail and the length. Not cryptographic; it detects storage corruption,
/// and any change confined to one 8-byte word always changes it (every step is a
/// bijection of the lane state). About as fast as memory bandwidth on one core.
#[must_use]
pub fn checksum(bytes: &[u8]) -> u64 {
    const P: [u64; 4] = [
        0x9E37_79B1_85EB_CA87,
        0xC2B2_AE3D_27D4_EB4F,
        0x1656_67B1_9E37_79F9,
        0x85EB_CA77_C2B2_AE63,
    ];
    let word = |c: &[u8]| {
        let mut w = [0u8; 8];
        w[..c.len()].copy_from_slice(c);
        u64::from_le_bytes(w)
    };
    let mut lanes = P;
    let blocks = bytes.chunks_exact(32);
    let tail = blocks.remainder();
    for b in blocks {
        for (i, lane) in lanes.iter_mut().enumerate() {
            let w = word(&b[i * 8..i * 8 + 8]);
            *lane = (*lane ^ w).wrapping_mul(P[i]).rotate_left(31);
        }
    }
    for (i, c) in tail.chunks(8).enumerate() {
        lanes[i] = (lanes[i] ^ word(c)).wrapping_mul(P[i]).rotate_left(31);
    }
    let mut h = bytes.len() as u64;
    for (i, lane) in lanes.iter().enumerate() {
        h = (h ^ *lane).wrapping_mul(P[(i + 1) % 4]).rotate_left(27);
    }
    h ^ (h >> 33)
}

/// An [`ExpertSource`] that checks every read against a checksum recorded from a
/// known-good read, so corrupt bytes fail the read ([`StoreError::Load`]) instead of
/// reaching the forward pass. An expert without a recorded checksum fails too: it
/// cannot be vouched for.
pub struct VerifiedSource {
    inner: Arc<dyn ExpertSource>,
    sums: HashMap<ExpertKey, u64>,
}

impl std::fmt::Debug for VerifiedSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifiedSource")
            .field("experts", &self.sums.len())
            .finish_non_exhaustive()
    }
}

impl VerifiedSource {
    /// `inner`, checked against `sums` (expert → [`checksum`] of its bytes).
    #[must_use]
    pub fn new(inner: Arc<dyn ExpertSource>, sums: HashMap<ExpertKey, u64>) -> Self {
        Self { inner, sums }
    }

    /// Records the checksums of `keys` by reading each once from `inner`.
    ///
    /// # Errors
    /// The first size or read error.
    pub fn record(
        inner: &dyn ExpertSource,
        keys: impl IntoIterator<Item = ExpertKey>,
    ) -> Result<HashMap<ExpertKey, u64>, StoreError> {
        let mut buf = Vec::new();
        let mut sums = HashMap::new();
        for key in keys {
            let size = inner.expert_size(key)?;
            let len = usize::try_from(size).map_err(|_| StoreError::TooLarge {
                key,
                size,
                capacity: size,
            })?;
            buf.clear();
            buf.resize(len, 0);
            inner.read_expert(key, &mut buf)?;
            sums.insert(key, checksum(&buf));
        }
        Ok(sums)
    }
}

impl ExpertSource for VerifiedSource {
    fn expert_size(&self, key: ExpertKey) -> Result<u64, StoreError> {
        self.inner.expert_size(key)
    }

    fn read_expert(&self, key: ExpertKey, dst: &mut [u8]) -> Result<(), StoreError> {
        let want = *self
            .sums
            .get(&key)
            .ok_or_else(|| load_err(key, "no recorded checksum for this expert"))?;
        self.inner.read_expert(key, dst)?;
        let got = checksum(dst);
        if got == want {
            Ok(())
        } else {
            Err(load_err(
                key,
                format!("checksum mismatch: read {got:016x}, recorded {want:016x}"),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_sees_every_single_byte_change() {
        let base: Vec<u8> = (0..1000u32).map(|i| (i * 31 % 251) as u8).collect();
        let c = checksum(&base);
        for len in [0, 1, 7, 8, 31, 32, 33, 999, 1000] {
            let s = &base[..len];
            let cs = checksum(s);
            for i in 0..len {
                for xor in [1u8, 0x80, 0xFF] {
                    let mut m = s.to_vec();
                    m[i] ^= xor;
                    assert_ne!(checksum(&m), cs, "len {len} byte {i} xor {xor:#x}");
                }
            }
        }
        // Length matters even when the extra bytes are zero.
        let mut longer = base.clone();
        longer.push(0);
        assert_ne!(checksum(&longer), c);
    }
}
