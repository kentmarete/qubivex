//! Portable elementary functions with fixed bits on every target.
//!
//! The platform `expf` (macOS libm, glibc, the GPU's `exp`) differs between targets,
//! so a forward pass that used it could not give the same bits on the CPU, on Linux and
//! on a GPU (ADR-011 §1). [`exp`] is built only from IEEE-754 operations that every
//! target rounds the same way: `+`, `−`, `×`, fused multiply-add and exact power-of-two
//! scaling. A GPU kernel that writes the same operations in the same order (with
//! floating-point contraction off) gets the same bits.
//!
//! Results that would be f32 subnormals are returned as zero, so the GPU's
//! flush-to-zero cannot make them differ.

/// `log2(e)`.
const LOG2E: f32 = std::f32::consts::LOG2_E;
/// `ln 2` split for Cody–Waite reduction: `HI` has 8 trailing zero bits, so `k · HI`
/// is exact for every `|k| ≤ 2^8`.
const LN2_HI: f32 = f32::from_bits(0x3f31_7200);
/// The rest of `ln 2`.
const LN2_LO: f32 = f32::from_bits(0x35bf_be8e);
/// `1.5 · 2^23`: adding and subtracting it rounds to the nearest integer (ties to even).
const SHIFT: f32 = 12_582_912.0;
/// Above this, `e^x` overflows f32.
const OVERFLOW: f32 = 88.722_84;
/// Below this, `e^x` is under `f32::MIN_POSITIVE` (or close enough to round there).
const UNDERFLOW: f32 = -87.336_55;
/// Cephes `expf` coefficients for `(e^r − 1 − r) / r²` on `[−ln2/2, ln2/2]`, highest
/// first, as exact bit patterns (the last is 0.5). Device kernels use the same bits.
const P: [f32; 6] = [
    f32::from_bits(0x3950_6967),
    f32::from_bits(0x3ab7_43ce),
    f32::from_bits(0x3c08_8908),
    f32::from_bits(0x3d2a_a9c1),
    f32::from_bits(0x3e2a_aaaa),
    f32::from_bits(0x3f00_0000),
];

/// `2^k` for `−126 ≤ k ≤ 127`.
#[inline(always)]
fn pow2(k: i32) -> f32 {
    #[allow(clippy::cast_sign_loss)]
    f32::from_bits(((k + 127) as u32) << 23)
}

/// `e^x`, within 2 ulp of the exact value for normal results.
///
/// `exp(NaN)` is NaN, results above `f32::MAX` are `+∞`, and results below
/// `f32::MIN_POSITIVE` are `+0`.
#[must_use]
#[inline]
pub fn exp(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x > OVERFLOW {
        return f32::INFINITY;
    }
    if x < UNDERFLOW {
        return 0.0;
    }
    // x = k·ln2 + r, |r| ≤ ln2/2 (plus rounding).
    let kf = (x * LOG2E + SHIFT) - SHIFT;
    let r = (-kf).mul_add(LN2_HI, x);
    let r = (-kf).mul_add(LN2_LO, r);
    let mut p = P[0];
    for c in &P[1..] {
        p = p.mul_add(r, *c);
    }
    let y = p.mul_add(r * r, r) + 1.0;
    // y · 2^k in two exact steps, so neither factor leaves the normal range.
    #[allow(clippy::cast_possible_truncation)]
    let k = kf as i32;
    let k1 = k >> 1;
    let v = (y * pow2(k1)) * pow2(k - k1);
    if v < f32::MIN_POSITIVE { 0.0 } else { v }
}

/// `silu(g) · u`, in place into `g`: `silu(a) = a / (1 + exp(−a))`, then one multiply.
///
/// A device kernel gives the same bits if it evaluates the same operations in the same
/// order, with this module's [`exp`], a correctly rounded division and contraction off.
pub fn swiglu(g: &mut [f32], u: &[f32]) {
    for (a, b) in g.iter_mut().zip(u) {
        let s = *a / (1.0 + exp(-*a));
        *a = s * b;
    }
}

/// Softmax in place (max-subtracted, sequential sum).
pub fn softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = exp(*v - m);
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// Softmax routing of one token: the top `k` experts by probability (ties: the lower
/// id), in rank order, with their probabilities (renormalised to sum to 1 when `norm`).
#[must_use]
pub fn route(logits: &[f32], k: usize, norm: bool) -> (Vec<u32>, Vec<f32>) {
    let mut p = logits.to_vec();
    softmax(&mut p);
    let mut idx: Vec<u32> = (0..u32::try_from(p.len()).unwrap_or(u32::MAX)).collect();
    idx.sort_by(|a, b| p[*b as usize].total_cmp(&p[*a as usize]).then(a.cmp(b)));
    idx.truncate(k);
    let mut w: Vec<f32> = idx.iter().map(|&e| p[e as usize]).collect();
    if norm {
        let sum: f32 = w.iter().sum();
        w.iter_mut().for_each(|x| *x /= sum);
    }
    (idx, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distance in ulps between two finite positive f32 values.
    fn ulps(a: f32, b: f32) -> u32 {
        a.to_bits().abs_diff(b.to_bits())
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn exp_is_within_two_ulp_over_the_normal_range() {
        let mut worst = 0;
        let mut x = UNDERFLOW;
        while x < OVERFLOW {
            let want = f64::from(x).exp();
            if want >= f64::from(f32::MIN_POSITIVE) && want <= f64::from(f32::MAX) {
                let e = ulps(exp(x), want as f32);
                if e > worst {
                    worst = e;
                }
            }
            // About 4 million points, denser near zero where softmax and SiLU live.
            x += (x.abs() * 1e-5).max(1e-6);
        }
        assert!(worst <= 2, "worst error {worst} ulp");
    }

    #[test]
    fn exp_edges() {
        assert_eq!(exp(0.0).to_bits(), 1f32.to_bits());
        assert_eq!(exp(-0.0).to_bits(), 1f32.to_bits());
        assert!(exp(f32::NAN).is_nan());
        assert_eq!(exp(f32::INFINITY), f32::INFINITY);
        assert_eq!(exp(f32::NEG_INFINITY).to_bits(), 0);
        assert_eq!(exp(100.0), f32::INFINITY);
        assert_eq!(exp(-100.0).to_bits(), 0);
        assert_eq!(exp(-87.5).to_bits(), 0, "subnormal results are zero");
        assert!(exp(88.7) < f32::INFINITY);
        assert!(exp(-87.3365) >= f32::MIN_POSITIVE);
        assert!((exp(1.0) - std::f32::consts::E).abs() <= f32::EPSILON * 4.0);
    }
}
