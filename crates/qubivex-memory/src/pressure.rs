//! The pressure monitor's decisions (ADR-005): what to do when the machine runs short
//! of RAM, in a fixed order. Stop prefetch, then shrink the expert cache, then refuse
//! new requests. An active request is never killed: the cache never goes below one
//! slot per layer, which always runs (only slower).
//!
//! This is pure logic over observations, so it is tested without a machine under
//! pressure. The runtime samples available RAM on a thread and applies each decision
//! between tokens.

use qubivex_core::Bytes;
use serde::Serialize;

/// Available-RAM thresholds. Policy values, not measurements: revisit them with
/// recorded pressure runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct PressureConfig {
    /// Below this: stop prefetch.
    pub elevated: Bytes,
    /// Below this: shrink the expert cache by the shortfall.
    pub high: Bytes,
    /// Below this: cache at the floor, refuse new requests.
    pub critical: Bytes,
    /// A level is left only once available RAM is this far above its threshold, so
    /// the monitor does not flap around a boundary.
    pub hysteresis: Bytes,
}

impl Default for PressureConfig {
    fn default() -> Self {
        Self {
            elevated: Bytes(1 << 30),
            high: Bytes(512 << 20),
            critical: Bytes(256 << 20),
            hysteresis: Bytes(256 << 20),
        }
    }
}

/// How short of RAM the machine is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PressureLevel {
    /// Enough RAM.
    #[default]
    Normal,
    /// Prefetch stops.
    Elevated,
    /// The expert cache shrinks.
    High,
    /// The cache is at its floor and new requests are refused.
    Critical,
}

/// What the runtime should do now.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct PressureDecision {
    /// The level.
    pub level: PressureLevel,
    /// Whether prefetch may run.
    pub prefetch: bool,
    /// A cap on the expert cache below the plan's (`None`: the plan's).
    pub cache_cap: Option<Bytes>,
    /// Whether new requests are admitted. Active ones always continue.
    pub admit: bool,
}

/// The monitor's state between observations.
#[derive(Clone, Debug)]
pub struct PressureMonitor {
    config: PressureConfig,
    floor: Bytes,
    decision: PressureDecision,
}

impl PressureMonitor {
    /// A monitor whose cache never goes below `floor` (one slot per layer).
    #[must_use]
    pub const fn new(config: PressureConfig, floor: Bytes) -> Self {
        Self {
            config,
            floor,
            decision: PressureDecision {
                level: PressureLevel::Normal,
                prefetch: true,
                cache_cap: None,
                admit: true,
            },
        }
    }

    /// The thresholds.
    #[must_use]
    pub const fn config(&self) -> PressureConfig {
        self.config
    }

    /// Replaces the thresholds (the level is re-evaluated on the next observation).
    pub const fn set_config(&mut self, config: PressureConfig) {
        self.config = config;
    }

    /// The latest decision.
    #[must_use]
    pub const fn decision(&self) -> PressureDecision {
        self.decision
    }

    fn level_of(&self, available: Bytes) -> PressureLevel {
        let c = &self.config;
        if available < c.critical {
            PressureLevel::Critical
        } else if available < c.high {
            PressureLevel::High
        } else if available < c.elevated {
            PressureLevel::Elevated
        } else {
            PressureLevel::Normal
        }
    }

    /// Decides from `available` RAM and the expert cache's resident bytes.
    pub fn observe(&mut self, available: Bytes, cache_used: Bytes) -> PressureDecision {
        let c = self.config;
        let current = self.decision.level;
        let target = self.level_of(available);
        // Up at once; down only as far as the hysteresis margin allows.
        let level = if target >= current {
            target
        } else {
            current.min(self.level_of(available.saturating_sub(c.hysteresis)))
        };
        let cache_cap = match level {
            PressureLevel::Normal => None,
            // Hold any cap from a higher level until pressure is gone.
            PressureLevel::Elevated => self.decision.cache_cap,
            PressureLevel::High => {
                // Free the shortfall below the recovery point, never below the floor,
                // and never grow while pressure lasts.
                let shortfall = c
                    .high
                    .0
                    .saturating_add(c.hysteresis.0)
                    .saturating_sub(available.0);
                let base = self
                    .decision
                    .cache_cap
                    .map_or(cache_used, |cap| cap.min(cache_used));
                Some(Bytes(base.0.saturating_sub(shortfall)).max(self.floor))
            }
            PressureLevel::Critical => Some(self.floor),
        };
        self.decision = PressureDecision {
            level,
            prefetch: level == PressureLevel::Normal,
            cache_cap,
            admit: level < PressureLevel::Critical,
        };
        self.decision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    fn m() -> PressureMonitor {
        PressureMonitor::new(PressureConfig::default(), Bytes(100 * MIB))
    }

    #[test]
    fn responses_come_in_order() {
        let mut p = m();
        let used = Bytes(3000 * MIB);
        let d = p.observe(Bytes(4096 * MIB), used);
        assert_eq!(
            (d.level, d.prefetch, d.cache_cap, d.admit),
            (PressureLevel::Normal, true, None, true)
        );
        let d = p.observe(Bytes(900 * MIB), used);
        assert_eq!(
            (d.level, d.prefetch, d.cache_cap, d.admit),
            (PressureLevel::Elevated, false, None, true)
        );
        // 400 MiB available: free 512 + 256 − 400 = 368 MiB.
        let d = p.observe(Bytes(400 * MIB), used);
        assert_eq!(d.level, PressureLevel::High);
        assert_eq!(d.cache_cap, Some(Bytes(3000 * MIB - 368 * MIB)));
        assert!(d.admit && !d.prefetch);
        let d = p.observe(Bytes(100 * MIB), used);
        assert_eq!(
            (d.level, d.cache_cap, d.admit),
            (PressureLevel::Critical, Some(Bytes(100 * MIB)), false)
        );
    }

    #[test]
    fn hysteresis_and_recovery() {
        let mut p = m();
        let used = Bytes(3000 * MIB);
        p.observe(Bytes(200 * MIB), used); // critical
        // Above the critical threshold but within the margin: stay critical.
        assert_eq!(
            p.observe(Bytes(400 * MIB), used).level,
            PressureLevel::Critical
        );
        // Past critical + margin, still under high + margin: high, the cap held low.
        let d = p.observe(Bytes(600 * MIB), Bytes(100 * MIB));
        assert_eq!(
            (d.level, d.cache_cap),
            (PressureLevel::High, Some(Bytes(100 * MIB)))
        );
        assert!(d.admit);
        let d = p.observe(Bytes(1000 * MIB), Bytes(100 * MIB));
        assert_eq!(
            (d.level, d.cache_cap),
            (PressureLevel::Elevated, Some(Bytes(100 * MIB)))
        );
        let d = p.observe(Bytes(1300 * MIB), Bytes(100 * MIB));
        assert_eq!(
            (d.level, d.prefetch, d.cache_cap),
            (PressureLevel::Normal, true, None)
        );
    }

    #[test]
    fn shrinking_never_grows_and_stops_at_the_floor() {
        let mut p = m();
        let mut used = Bytes(3000 * MIB);
        let mut last = u64::MAX;
        for _ in 0..20 {
            let d = p.observe(Bytes(300 * MIB), used);
            let cap = d.cache_cap.unwrap().0;
            assert!(cap <= last && cap >= 100 * MIB);
            last = cap;
            used = Bytes(cap); // the runtime applied it
        }
        assert_eq!(last, 100 * MIB, "repeated shortfall reaches the floor");
    }
}
