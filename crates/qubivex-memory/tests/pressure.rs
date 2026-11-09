//! Pressure-monitor properties over random observation sequences.

use proptest::prelude::*;
use qubivex_core::Bytes;
use qubivex_memory::{PressureConfig, PressureLevel, PressureMonitor};

const MIB: u64 = 1 << 20;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn decisions_keep_their_guarantees(
        floor in 1u64..(512 * MIB),
        used0 in 0u64..(8192 * MIB),
        obs in proptest::collection::vec(0u64..(4096 * MIB), 1..60),
    ) {
        let cfg = PressureConfig::default();
        let mut m = PressureMonitor::new(cfg, Bytes(floor));
        let mut used = Bytes(used0);
        let mut prev = m.decision();
        for a in obs {
            let d = m.observe(Bytes(a), used);
            // Never below the floor: an active request always runs.
            if let Some(cap) = d.cache_cap {
                prop_assert!(cap.0 >= floor);
            }
            // The fixed order of responses.
            prop_assert_eq!(d.prefetch, d.level == PressureLevel::Normal);
            prop_assert_eq!(d.admit, d.level < PressureLevel::Critical);
            // No cap without pressure; always one from High up. Elevated only stops
            // prefetch and holds whatever cap it had.
            if d.level == PressureLevel::Normal {
                prop_assert!(d.cache_cap.is_none());
            }
            if d.level >= PressureLevel::High {
                prop_assert!(d.cache_cap.is_some());
            }
            if d.level == PressureLevel::Critical {
                prop_assert_eq!(d.cache_cap, Some(Bytes(floor)));
            }
            // Escalation is immediate.
            if a < cfg.critical.0 {
                prop_assert_eq!(d.level, PressureLevel::Critical);
            }
            // The cap never grows while pressure lasts.
            if let (Some(p), Some(c)) = (prev.cache_cap, d.cache_cap)
                && (d.level >= PressureLevel::High || c.0 > floor) {
                    prop_assert!(c <= p || prev.level == PressureLevel::Normal);
                }
            // Leaving a level needs the hysteresis margin.
            if d.level < prev.level {
                prop_assert!(a >= cfg.critical.0.min(cfg.high.0) + cfg.hysteresis.0);
            }
            if let Some(c) = d.cache_cap {
                used = used.min(c); // the runtime applied it
            }
            prev = d;
        }
    }
}
