//! Router-lookahead prediction (Phase 7), ported from Colibri's `pilot_prefetch`.
//!
//! During decode, the forward pass applies layer L+1's MoE norm and router to the
//! hidden state just after layer L's attention, and gives the logits to
//! [`Lookahead::predict`]. The prediction blends them with a per-layer moving average
//! of earlier predictions ("momentum"), then takes experts in probability order until
//! their mass reaches a confidence cutoff: at least top-K, at most `top-K × width`.
//!
//! A prediction is advisory. It only chooses what to read ahead
//! ([`crate::ExpertStore::prefetch`]). Routing always uses the real router logits of
//! the real hidden state, so the output is the same with prediction on or off.

use qubivex_core::ExpertIdx;

/// Prediction settings (Colibri's `SMOOTH`, `CONF_LIMIT` and `WIDE`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LookaheadConfig {
    /// The weight of the moving average when blending, in `[0, 1)`; 0 uses the
    /// lookahead logits alone.
    pub smooth: f32,
    /// Stop adding candidates once their probability mass reaches this, in `(0, 1]`.
    pub confidence: f32,
    /// At most `top_k × width` candidates (at least 1).
    pub width: u32,
}

impl Default for LookaheadConfig {
    /// Colibri's defaults: `SMOOTH=0.3`, `CONF_LIMIT=0.92`, `WIDE=1`.
    fn default() -> Self {
        Self {
            smooth: 0.3,
            confidence: 0.92,
            width: 1,
        }
    }
}

/// The predictor: the configuration and one moving average of router logits per
/// layer.
#[derive(Clone, Debug)]
pub struct Lookahead {
    cfg: LookaheadConfig,
    top_k: usize,
    ema: Vec<Option<Vec<f32>>>,
}

impl Lookahead {
    /// A predictor for `layers` layers routing `top_k` experts per token.
    #[must_use]
    pub fn new(cfg: LookaheadConfig, layers: usize, top_k: usize) -> Self {
        Self {
            cfg,
            top_k: top_k.max(1),
            ema: vec![None; layers],
        }
    }

    /// The configuration.
    #[must_use]
    pub const fn config(&self) -> LookaheadConfig {
        self.cfg
    }

    /// Forgets the moving averages.
    pub fn reset(&mut self) {
        self.ema.iter_mut().for_each(|e| *e = None);
    }

    /// The experts `layer` is predicted to route to, most probable first, from its
    /// router's `logits` on the lookahead state. Updates the layer's moving average.
    /// A layer out of range predicts from the logits alone.
    pub fn predict(&mut self, layer: usize, logits: &[f32]) -> Vec<ExpertIdx> {
        let s = self.cfg.smooth.clamp(0.0, 0.99);
        let blended = match self.ema.get_mut(layer) {
            Some(slot) if s > 0.0 => {
                let b: Vec<f32> = match slot.as_ref() {
                    Some(ema) if ema.len() == logits.len() => logits
                        .iter()
                        .zip(ema)
                        .map(|(&x, &m)| (1.0 - s).mul_add(x, s * m))
                        .collect(),
                    _ => logits.to_vec(),
                };
                *slot = Some(b.clone());
                b
            }
            _ => logits.to_vec(),
        };
        let width = usize::try_from(self.cfg.width.max(1)).unwrap_or(usize::MAX);
        candidates(
            &blended,
            self.top_k,
            self.top_k.saturating_mul(width),
            self.cfg.confidence,
        )
    }
}

/// Experts in descending probability (softmax of `logits`; ties: lower id), taken until
/// their mass reaches `confidence` with at least `min` taken, and never more than
/// `max` (both clamped to the number of experts).
#[must_use]
pub fn candidates(logits: &[f32], min: usize, max: usize, confidence: f32) -> Vec<ExpertIdx> {
    let n = logits.len();
    let max = max.max(min).min(n);
    let min = min.min(max);
    let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let p: Vec<f32> = logits.iter().map(|&x| (x - top).exp()).collect();
    let total: f32 = p.iter().sum();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    let mut out = Vec::with_capacity(max);
    let mut mass = 0f32;
    for &e in order.iter().take(max) {
        out.push(ExpertIdx(u32::try_from(e).unwrap_or(u32::MAX)));
        mass += p[e];
        if out.len() >= min && mass >= confidence * total {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(v: &[ExpertIdx]) -> Vec<u32> {
        v.iter().map(|e| e.0).collect()
    }

    #[test]
    fn takes_top_k_then_stops_at_the_cutoff() {
        // Probabilities ∝ e^x: expert 2 holds almost all the mass.
        let l = [0.0f32, 1.0, 10.0, 2.0, -1.0];
        assert_eq!(ids(&candidates(&l, 1, 5, 0.9)), [2]);
        assert_eq!(ids(&candidates(&l, 2, 5, 0.9)), [2, 3], "at least min");
        assert_eq!(
            ids(&candidates(&l, 1, 3, 1.0)),
            [2, 3, 1],
            "at most max, even below the cutoff"
        );
        assert_eq!(ids(&candidates(&l, 9, 9, 0.5)).len(), 5, "clamped to n");
    }

    #[test]
    fn ties_take_the_lower_id() {
        assert_eq!(ids(&candidates(&[1.0, 1.0, 1.0], 2, 2, 1.0)), [0, 1]);
    }

    #[test]
    fn momentum_blends_with_earlier_predictions() {
        let cfg = LookaheadConfig {
            smooth: 0.5,
            confidence: 1.0,
            width: 1,
        };
        let mut p = Lookahead::new(cfg, 2, 1);
        // First call: the logits alone.
        assert_eq!(ids(&p.predict(0, &[3.0, 0.0])), [0]);
        // 0.5·[0, 2] + 0.5·[3, 0] = [1.5, 1]: momentum keeps expert 0.
        assert_eq!(ids(&p.predict(0, &[0.0, 2.0])), [0]);
        // Layer 1 has its own average.
        assert_eq!(ids(&p.predict(1, &[0.0, 2.0])), [1]);
        // Without smoothing the newest logits decide.
        let mut q = Lookahead::new(LookaheadConfig { smooth: 0.0, ..cfg }, 2, 1);
        q.predict(0, &[3.0, 0.0]);
        assert_eq!(ids(&q.predict(0, &[0.0, 2.0])), [1]);
        p.reset();
        assert_eq!(ids(&p.predict(0, &[0.0, 2.0])), [1], "reset forgets");
    }

    #[test]
    fn width_caps_the_candidates() {
        let cfg = LookaheadConfig {
            smooth: 0.0,
            confidence: 1.0,
            width: 2,
        };
        let mut p = Lookahead::new(cfg, 1, 2);
        assert_eq!(p.predict(0, &[0.0; 10]).len(), 4, "top_k × width");
    }
}
