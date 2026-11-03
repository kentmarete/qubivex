//! Bytes actually allocated, per tier and owner (ADR-005).
//!
//! Every long-lived allocation (dense weights, expert slots, KV pages, activation
//! buffers) holds a [`Charge`] for its bytes; dropping the charge releases them, so the
//! ledger cannot drift from the allocations it describes. The pressure monitor
//! compares the ledger and the OS snapshot against the plan.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use qubivex_core::Bytes;
use serde::Serialize;

/// A memory tier. Disk is not a tier here: nothing is allocated on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Ordinary host RAM.
    Ram,
    /// Page-locked host RAM for device transfers (with the first GPU backend).
    PinnedHost,
    /// One GPU's memory.
    Vram(u32),
}

/// What an allocation is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Owner {
    /// Resident (non-expert) weights.
    Dense,
    /// The routed-expert cache.
    ExpertCache,
    /// KV caches.
    Kv,
    /// Activations and scratch buffers.
    Activations,
}

impl Owner {
    /// Every owner, in index order.
    pub const ALL: [Self; 4] = [Self::Dense, Self::ExpertCache, Self::Kv, Self::Activations];

    const fn index(self) -> usize {
        match self {
            Self::Dense => 0,
            Self::ExpertCache => 1,
            Self::Kv => 2,
            Self::Activations => 3,
        }
    }
}

const OWNERS: usize = Owner::ALL.len();

/// Counters per tier and owner. Share it through an `Arc`.
#[derive(Debug)]
pub struct MemoryLedger {
    /// `[ram, pinned host, vram 0, vram 1, ...][owner]`.
    counters: Vec<[AtomicU64; OWNERS]>,
}

/// Bytes per owner in one tier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct LedgerSnapshot {
    /// Resident weights.
    pub dense: Bytes,
    /// Expert cache.
    pub expert_cache: Bytes,
    /// KV caches.
    pub kv: Bytes,
    /// Activations and scratch.
    pub activations: Bytes,
}

impl LedgerSnapshot {
    /// The sum over owners.
    #[must_use]
    pub const fn total(&self) -> Bytes {
        Bytes(self.dense.0 + self.expert_cache.0 + self.kv.0 + self.activations.0)
    }
}

impl MemoryLedger {
    /// A ledger for host memory and `gpus` devices.
    #[must_use]
    pub fn new(gpus: u32) -> Arc<Self> {
        let tiers = 2 + gpus as usize;
        Arc::new(Self {
            counters: (0..tiers).map(|_| Default::default()).collect(),
        })
    }

    fn slot(&self, tier: Tier) -> &[AtomicU64; OWNERS] {
        let i = match tier {
            Tier::Ram => 0,
            Tier::PinnedHost => 1,
            Tier::Vram(d) => 2 + d as usize,
        };
        // A tier outside the ledger is a wiring bug: caught in debug builds; in
        // release it is counted nowhere rather than panicking in an allocation path.
        static NOWHERE: [AtomicU64; OWNERS] = [const { AtomicU64::new(0) }; OWNERS];
        debug_assert!(i < self.counters.len(), "{tier:?} is not in this ledger");
        self.counters.get(i).unwrap_or(&NOWHERE)
    }

    /// Records `bytes` for `owner` in `tier` until the returned charge is dropped.
    #[must_use = "the bytes are released when the charge is dropped"]
    pub fn charge(self: &Arc<Self>, tier: Tier, owner: Owner, bytes: Bytes) -> Charge {
        self.slot(tier)[owner.index()].fetch_add(bytes.0, Ordering::Relaxed);
        Charge {
            ledger: Arc::clone(self),
            tier,
            owner,
            bytes: bytes.0,
        }
    }

    /// Bytes per owner in `tier` now.
    #[must_use]
    pub fn snapshot(&self, tier: Tier) -> LedgerSnapshot {
        let c = self.slot(tier);
        let get = |o: Owner| Bytes(c[o.index()].load(Ordering::Relaxed));
        LedgerSnapshot {
            dense: get(Owner::Dense),
            expert_cache: get(Owner::ExpertCache),
            kv: get(Owner::Kv),
            activations: get(Owner::Activations),
        }
    }
}

/// Bytes recorded in a ledger; released on drop.
#[derive(Debug)]
pub struct Charge {
    ledger: Arc<MemoryLedger>,
    tier: Tier,
    owner: Owner,
    bytes: u64,
}

impl Charge {
    /// The bytes charged.
    #[must_use]
    pub const fn bytes(&self) -> Bytes {
        Bytes(self.bytes)
    }

    /// Changes the charge to `bytes` (a KV cache that grew, a cache that shrank).
    pub fn resize(&mut self, bytes: Bytes) {
        let c = &self.ledger.slot(self.tier)[self.owner.index()];
        if bytes.0 >= self.bytes {
            c.fetch_add(bytes.0 - self.bytes, Ordering::Relaxed);
        } else {
            c.fetch_sub(self.bytes - bytes.0, Ordering::Relaxed);
        }
        self.bytes = bytes.0;
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.ledger.slot(self.tier)[self.owner.index()].fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charges_are_released_on_drop() {
        let l = MemoryLedger::new(1);
        let a = l.charge(Tier::Ram, Owner::Dense, Bytes(100));
        let mut kv = l.charge(Tier::Ram, Owner::Kv, Bytes(10));
        let v = l.charge(Tier::Vram(0), Owner::ExpertCache, Bytes(7));
        assert_eq!(l.snapshot(Tier::Ram).total(), Bytes(110));
        kv.resize(Bytes(50));
        assert_eq!(l.snapshot(Tier::Ram).kv, Bytes(50));
        kv.resize(Bytes(5));
        assert_eq!(l.snapshot(Tier::Ram).kv, Bytes(5));
        assert_eq!(l.snapshot(Tier::Vram(0)).expert_cache, Bytes(7));
        assert_eq!(
            l.snapshot(Tier::PinnedHost).total(),
            Bytes(0),
            "tiers are separate"
        );
        drop((a, kv, v));
        assert_eq!(l.snapshot(Tier::Ram), LedgerSnapshot::default());
        assert_eq!(l.snapshot(Tier::Vram(0)), LedgerSnapshot::default());
    }

    #[test]
    fn concurrent_charges_balance() {
        let l = MemoryLedger::new(0);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for i in 0..1000 {
                        let _c = l.charge(Tier::Ram, Owner::Activations, Bytes(i));
                    }
                });
            }
        });
        assert_eq!(l.snapshot(Tier::Ram).total(), Bytes(0));
    }
}
