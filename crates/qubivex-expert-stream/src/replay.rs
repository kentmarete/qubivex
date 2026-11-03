//! Trace replay (ADR-008): runs a recorded routing trace through the real store over
//! a source that reads nothing, so cache policies and per-layer budgets are compared
//! on real routing without the model or any I/O. Any new policy must beat the default
//! on recorded traces before it ships.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Write};
use std::sync::Arc;

use qubivex_core::{ExpertIdx, LayerIdx};

use crate::{ExpertKey, ExpertSource, ExpertStore, Policy, PoolSpec, StoreError, StoreStats};

/// Every expert is one byte and reading it does nothing.
struct Null;

impl ExpertSource for Null {
    fn expert_size(&self, _: ExpertKey) -> Result<u64, StoreError> {
        Ok(1)
    }
    fn read_expert(&self, _: ExpertKey, _: &mut [u8]) -> Result<(), StoreError> {
        Ok(())
    }
}

/// Replays `trace` with `slots[layer]` slots per layer (one lease at a time, as a
/// decode step takes them) and returns the store's counters.
///
/// # Errors
/// [`StoreError::NoPool`] for a trace layer without slots, and [`StoreError::Full`] if
/// a layer has 0 slots.
pub fn replay(
    trace: &[ExpertKey],
    slots: &HashMap<LayerIdx, u32>,
    policy: Policy,
) -> Result<StoreStats, StoreError> {
    let pools: Vec<PoolSpec> = slots
        .iter()
        .map(|(&layer, &slots)| PoolSpec {
            layer,
            slots,
            slot_bytes: 1,
        })
        .collect();
    let store = ExpertStore::new(Arc::new(Null), &pools, policy);
    for &k in trace {
        drop(store.acquire(k)?);
    }
    Ok(store.stats())
}

/// What a global LRU of `capacity` experts would hit on `trace`: the Phase 4–5 store,
/// kept as the baseline that per-layer pools must beat.
#[must_use]
pub fn global_lru_hits(trace: &[ExpertKey], capacity: usize) -> u64 {
    let mut last: HashMap<ExpertKey, u64> = HashMap::new();
    let mut order: std::collections::BTreeMap<u64, ExpertKey> = std::collections::BTreeMap::new();
    let mut hits = 0;
    for (t, &k) in (0u64..).zip(trace) {
        if let Some(old) = last.insert(k, t) {
            order.remove(&old);
            hits += 1;
        } else if last.len() > capacity {
            if let Some((_, v)) = order.pop_first() {
                last.remove(&v);
            }
        }
        order.insert(t, k);
    }
    hits
}

/// The layers of a trace and their distinct experts.
#[must_use]
pub fn layers(trace: &[ExpertKey]) -> Vec<(LayerIdx, usize)> {
    let mut seen: HashMap<LayerIdx, HashSet<ExpertIdx>> = HashMap::new();
    for &(l, e) in trace {
        seen.entry(l).or_default().insert(e);
    }
    let mut v: Vec<(LayerIdx, usize)> = seen.into_iter().map(|(l, s)| (l, s.len())).collect();
    v.sort();
    v
}

/// Writes a trace as text, one `layer expert` pair per line.
///
/// # Errors
/// I/O errors.
pub fn write_trace(trace: &[ExpertKey], mut w: impl Write) -> std::io::Result<()> {
    for (l, e) in trace {
        writeln!(w, "{} {}", l.0, e.0)?;
    }
    Ok(())
}

/// Reads a trace written by [`write_trace`].
///
/// # Errors
/// I/O errors, or [`std::io::ErrorKind::InvalidData`] for a malformed line.
pub fn read_trace(r: impl BufRead) -> std::io::Result<Vec<ExpertKey>> {
    let bad = |n: usize| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("line {n}"));
    let mut v = Vec::new();
    for (n, line) in r.lines().enumerate() {
        let line = line?;
        let mut it = line.split_ascii_whitespace();
        let (Some(l), Some(e), None) = (it.next(), it.next(), it.next()) else {
            return Err(bad(n + 1));
        };
        let l = l.parse().map_err(|_| bad(n + 1))?;
        let e = e.parse().map_err(|_| bad(n + 1))?;
        v.push((LayerIdx(l), ExpertIdx(e)));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn k(l: u32, e: u32) -> ExpertKey {
        (LayerIdx(l), ExpertIdx(e))
    }

    /// A decode-like trace: `layers` layers in order per token, `top` experts each,
    /// from a skewed distribution (expert 0 is hottest).
    fn decode_trace(tokens: u32, layers: u32, top: u32, experts: u32) -> Vec<ExpertKey> {
        let mut x = 0x9E37_79B9_7F4A_7C15_u64;
        let mut v = Vec::new();
        for _ in 0..tokens {
            for l in 0..layers {
                let mut chosen = Vec::new();
                while chosen.len() < top as usize {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    // Squaring a uniform draw skews it toward 0.
                    let u = (x >> 11) as f64 / (1u64 << 53) as f64;
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let e = ((u * u) * f64::from(experts)) as u32;
                    if !chosen.contains(&e) {
                        chosen.push(e);
                    }
                }
                chosen.sort_unstable();
                v.extend(chosen.into_iter().map(|e| k(l, e)));
            }
        }
        v
    }

    #[test]
    fn per_layer_pools_beat_a_global_lru_of_the_same_size() {
        let (layers, top, experts, per) = (16, 8, 64, 3);
        let trace = decode_trace(200, layers, top, experts);
        let slots: HashMap<LayerIdx, u32> = (0..layers).map(|l| (LayerIdx(l), per)).collect();
        let global = global_lru_hits(&trace, (layers * per) as usize);
        assert_eq!(global, 0, "the decode sweep defeats a global LRU");
        let lru = replay(&trace, &slots, Policy::Lru).unwrap();
        let lfru = replay(&trace, &slots, Policy::LFRU).unwrap();
        assert_eq!(lru.hits + lru.loads, trace.len() as u64);
        assert!(lru.hits > 0);
        assert!(
            lfru.hits > lru.hits,
            "LFRU {} vs LRU {}",
            lfru.hits,
            lru.hits
        );
        assert!(
            lfru.slab_allocs <= u64::from(layers * per),
            "slabs are reused"
        );
    }

    #[test]
    fn traces_round_trip_as_text() {
        let t = vec![k(0, 3), k(15, 63)];
        let mut buf = Vec::new();
        write_trace(&t, &mut buf).unwrap();
        assert_eq!(buf, b"0 3\n15 63\n");
        assert_eq!(read_trace(&buf[..]).unwrap(), t);
        assert!(read_trace(&b"0 1 2\n"[..]).is_err());
        assert_eq!(layers(&t), [(LayerIdx(0), 1), (LayerIdx(15), 1)]);
    }

    #[test]
    fn global_lru_baseline_counts_hits() {
        let t = [k(0, 1), k(0, 2), k(0, 1), k(0, 3), k(0, 2)];
        assert_eq!(global_lru_hits(&t, 2), 1, "2 was evicted by 3");
        assert_eq!(global_lru_hits(&t, 3), 2);
    }
}
