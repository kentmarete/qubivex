//! Replacement-policy arithmetic (ADR-008), ported from Colibri's `tier.h`. Pure
//! functions over slot metadata, so they are tested without a store or any I/O.

/// The most recency can add to an LFRU score: recency only breaks heat ties.
pub const RECENCY_MAX: u64 = 255;

/// The LFRU score of a resident expert: `heat << 8 | recency`, where recency is
/// `255 − age` clamped to `0..=255`. Higher is more valuable; the lowest unleased score
/// is evicted. `age` counts the pool's accesses since the expert was last used.
#[must_use]
pub fn lfru_score(heat: u32, age: u64) -> u64 {
    (u64::from(heat) << 8) | (RECENCY_MAX - age.min(RECENCY_MAX))
}

/// Whether `hot` beats `cold` by the promotion hysteresis, `hot > cold + cold/4 + 4`
/// (25% plus 4). It keeps two close candidates from swapping back and forth. Saturates
/// instead of overflowing, so a `cold` near `u64::MAX` is never beaten.
#[must_use]
pub const fn beats(hot: u64, cold: u64) -> bool {
    hot > cold.saturating_add(cold / 4).saturating_add(4)
}

/// Whether a predicted expert of `heat` may evict a resident expert whose LFRU score is
/// `victim` (ADR-008's prefetch eviction guard, Colibri's `pilot_realload`). The
/// prediction scores as if just used, so this is heat against heat with the 25%
/// hysteresis and a margin of 4 heat: `heat << 8 | 255 > victim + victim/4 + 4 << 8`.
/// An expert the layer keeps using is not displaced by a guess.
#[must_use]
pub fn speculation_beats(heat: u32, victim: u64) -> bool {
    lfru_score(heat, 0) > victim.saturating_add(victim / 4).saturating_add(4 << 8)
}

/// Accesses per expert, halved every `window` accesses so that old popularity fades.
#[derive(Clone, Debug, Default)]
pub struct Heat {
    counts: std::collections::HashMap<u32, u32>,
    since_decay: u32,
}

impl Heat {
    /// Counts one access to `expert` and decays every count once `window` accesses
    /// have passed (`window` 0 never decays).
    pub fn touch(&mut self, expert: u32, window: u32) {
        let h = self.counts.entry(expert).or_insert(0);
        *h = h.saturating_add(1);
        self.since_decay += 1;
        if window > 0 && self.since_decay >= window {
            self.since_decay = 0;
            self.counts.retain(|_, h| {
                *h >>= 1;
                *h > 0
            });
        }
    }

    /// The heat of `expert` (0 if never seen or decayed away).
    #[must_use]
    pub fn get(&self, expert: u32) -> u32 {
        self.counts.get(&expert).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heat_dominates_and_recency_breaks_ties() {
        assert!(
            lfru_score(2, 1000) > lfru_score(1, 0),
            "one heat beats any recency"
        );
        assert!(lfru_score(1, 0) > lfru_score(1, 1));
        assert_eq!(lfru_score(0, 0), 255);
        assert_eq!(lfru_score(0, 255), 0);
        assert_eq!(lfru_score(0, u64::MAX), 0, "age clamps");
        assert_eq!(lfru_score(u32::MAX, 0), (u64::from(u32::MAX) << 8) | 255);
    }

    #[test]
    fn hysteresis_is_25_percent_plus_4() {
        assert!(!beats(4, 0));
        assert!(beats(5, 0));
        assert!(!beats(129, 100), "100 + 25 + 4 = 129");
        assert!(beats(130, 100));
        assert!(!beats(u64::MAX, u64::MAX - 1), "saturates, no overflow");
        assert!(!beats(u64::MAX, u64::MAX));
    }

    #[test]
    fn a_speculation_needs_clearly_more_heat() {
        // A resident of heat 4, just used: 4 << 8 | 255 = 1279.
        let victim = lfru_score(4, 0);
        // 1279 + 319 + 1024 = 2622: heat 10 (2815) wins, heat 9 (2559) does not.
        assert!(!speculation_beats(9, victim));
        assert!(speculation_beats(10, victim));
        // A cold, stale resident (score 0) falls to any prediction with heat 4.
        assert!(!speculation_beats(3, 0));
        assert!(speculation_beats(4, 0));
        assert!(!speculation_beats(u32::MAX, u64::MAX), "saturates");
    }

    #[test]
    fn heat_counts_and_halves_per_window() {
        let mut h = Heat::default();
        for _ in 0..5 {
            h.touch(7, 0);
        }
        h.touch(3, 0);
        assert_eq!((h.get(7), h.get(3), h.get(9)), (5, 1, 0));
        let mut h = Heat::default();
        for _ in 0..3 {
            h.touch(1, 4);
        }
        h.touch(2, 4); // fourth access: decay, 3 → 1 and 1 → 0 (dropped)
        assert_eq!((h.get(1), h.get(2)), (1, 0));
    }
}
