//! The one reduction shape every dot product and norm uses.
//!
//! 64 f32 lanes, one fused multiply-add per lane per 64-element chunk, chunks in
//! increasing `k`, a final partial chunk in lanes `0..rem`, then a fixed halving tree.
//! Because the shape is fixed, a dot product's bits depend only on its two inputs:
//! not on the batch size, the thread count, or which code path (portable or
//! feature-dispatched) ran it. `f32::mul_add` is a correctly rounded fused operation
//! on every target, so compiling this code with or without hardware FMA gives the
//! same results.

/// Accumulator lanes.
pub const LANES: usize = 64;

/// The fixed halving tree: `a[i] += a[i + w]` for w = 32, 16, 8, 4, then
/// `(a0 + a1) + (a2 + a3)`.
#[inline(always)]
pub fn reduce(acc: &[f32; LANES]) -> f32 {
    let mut a = *acc;
    let mut w = LANES / 2;
    while w >= 4 {
        for i in 0..w {
            a[i] += a[i + w];
        }
        w /= 2;
    }
    (a[0] + a[1]) + (a[2] + a[3])
}

/// `Σ w[i] · x[i]` with the shared shape. `w` and `x` must have the same length.
#[inline(always)]
pub fn dot(w: &[f32], x: &[f32]) -> f32 {
    debug_assert_eq!(w.len(), x.len());
    let mut acc = [0f32; LANES];
    let full = w.len() / LANES * LANES;
    for (wc, xc) in w[..full]
        .chunks_exact(LANES)
        .zip(x[..full].chunks_exact(LANES))
    {
        for l in 0..LANES {
            acc[l] = wc[l].mul_add(xc[l], acc[l]);
        }
    }
    for (l, (a, b)) in w[full..].iter().zip(&x[full..]).enumerate() {
        acc[l] = a.mul_add(*b, acc[l]);
    }
    reduce(&acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_f64_and_handles_tails() {
        for n in [0, 1, 31, 32, 63, 64, 65, 100, 128, 1000, 2048] {
            let w: Vec<f32> = (0..n)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 64.0)
                .collect();
            let x: Vec<f32> = (0..n)
                .map(|i| ((i * 53 % 97) as f32 - 48.0) / 32.0)
                .collect();
            let want: f64 = w
                .iter()
                .zip(&x)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let got = f64::from(dot(&w, &x));
            assert!(
                (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                "n={n}: {got} vs {want}"
            );
        }
    }
}
