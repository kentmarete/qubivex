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

/// Whether reads hold compute back: the share of wall time callers spent blocked on
/// loads, over windows of at least [`SaturationGauge::WINDOW_NS`] (ADR-008 amendment 3).
///
/// On a saturated disk, speed follows the bytes read, so a speculative read that goes
/// unused costs throughput and one that is used gains nothing. Disk utilisation does not
/// tell that case from a useful one: the measured regimes overlap. This does, and
/// [`crate::StoreStats::wait_ns`] already counts it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SaturationGauge {
    /// When the open window started, and the blocked time then.
    start: Option<(u64, u64)>,
    saturated: bool,
}

impl SaturationGauge {
    /// The shortest window judged: about two tokens at the speed of a fast disk.
    pub const WINDOW_NS: u64 = 100_000_000;
    /// Saturated from this blocked share, in percent.
    pub const HOLD_PERCENT: u64 = 60;
    /// No longer saturated below this blocked share, in percent. The gap keeps the
    /// gauge from flipping when prefetch itself moves the share.
    pub const RESUME_PERCENT: u64 = 40;

    /// Takes a reading (`now_ns` on any monotonic clock, `wait_ns` the blocked time so
    /// far) and says whether reads are saturated.
    pub fn observe(&mut self, now_ns: u64, wait_ns: u64) -> bool {
        let Some((t0, w0)) = self.start else {
            self.start = Some((now_ns, wait_ns));
            return self.saturated;
        };
        let elapsed = now_ns.saturating_sub(t0);
        if elapsed >= Self::WINDOW_NS {
            let blocked = u128::from(wait_ns.saturating_sub(w0)) * 100;
            let share = blocked / u128::from(elapsed);
            if share >= u128::from(Self::HOLD_PERCENT) {
                self.saturated = true;
            } else if share < u128::from(Self::RESUME_PERCENT) {
                self.saturated = false;
            }
            self.start = Some((now_ns, wait_ns));
        }
        self.saturated
    }

    /// Closes the open window, keeping the verdict: idle time between requests says
    /// nothing about the disk.
    pub const fn pause(&mut self) {
        self.start = None;
    }
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
    use super::SaturationGauge;

    const W: u64 = SaturationGauge::WINDOW_NS;

    #[test]
    fn gauge_holds_at_60_percent_and_resumes_below_40() {
        let mut g = SaturationGauge::default();
        assert!(!g.observe(0, 0));
        // 59%: not yet.
        assert!(!g.observe(W, W * 59 / 100));
        // 60% of the next window.
        assert!(g.observe(2 * W, W * 119 / 100));
        // 50%: inside the gap, the verdict stands.
        assert!(g.observe(3 * W, W * 169 / 100));
        // 39%: resumed.
        assert!(!g.observe(4 * W, W * 208 / 100));
    }

    #[test]
    fn gauge_judges_only_full_windows() {
        let mut g = SaturationGauge::default();
        g.observe(0, 0);
        assert!(!g.observe(W - 1, W - 1), "a short window is not judged");
        assert!(g.observe(W, W), "the window kept running");
    }

    #[test]
    fn idle_between_requests_does_not_clear_the_verdict() {
        let mut g = SaturationGauge::default();
        g.observe(0, 0);
        assert!(g.observe(W, W));
        g.pause();
        // A minute idle, then the first reading of the next request.
        assert!(g.observe(60_000_000_000, W));
        assert!(g.observe(60_000_000_000 + W, 2 * W));
    }

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
