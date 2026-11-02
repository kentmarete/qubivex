//! Exact dequantization of one row, per storage format.
//!
//! Each function writes the f32 value of every weight as the format defines it. The
//! GGML formats follow `ggml-quants.c` `dequantize_row_*`, including the evaluation
//! order (separate multiply and subtract, never fused), so a weight's value is the
//! same here as in the reference dequantizer.

use qubivex_core::{DType, GgmlType};

/// Q8_0 block: f16 scale, 32 × i8.
pub const Q8_0_BYTES: usize = 34;
/// K-quant super-block elements.
pub const QK_K: usize = 256;
/// Q4_K block: f16 d, f16 dmin, 12 scale/min bytes, 128 nibble bytes.
pub const Q4_K_BYTES: usize = 144;
/// Q6_K block: 128 low-nibble bytes, 64 high-bit bytes, 16 i8 scales, f16 d.
pub const Q6_K_BYTES: usize = 210;

/// IEEE binary16 to f32, exactly (every f16 is representable in f32).
#[must_use]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (u32::from(h) & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: shift the mantissa up to an implicit leading 1.
            let mut m = m;
            let mut e: u32 = 127 - 15 + 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

#[inline(always)]
fn f16_at(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

/// Whether [`dequant_row`] supports `dtype`.
#[must_use]
pub const fn supported(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F32
            | DType::F16
            | DType::BF16
            | DType::Ggml(GgmlType::Q8_0 | GgmlType::Q4_K | GgmlType::Q6_K)
    )
}

/// Writes the f32 values of one row. `row` must be exactly one row of `dtype` for
/// `out.len()` columns (the caller's `WeightView` guarantees it). Unsupported dtypes
/// are rejected by the caller via [`supported`].
#[inline(always)]
pub fn dequant_row(dtype: DType, row: &[u8], out: &mut [f32]) {
    match dtype {
        DType::F32 => {
            for (o, b) in out.iter_mut().zip(row.chunks_exact(4)) {
                *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        DType::F16 => {
            for (o, b) in out.iter_mut().zip(row.chunks_exact(2)) {
                *o = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            }
        }
        DType::BF16 => {
            for (o, b) in out.iter_mut().zip(row.chunks_exact(2)) {
                *o = f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16);
            }
        }
        DType::Ggml(GgmlType::Q8_0) => q8_0(row, out),
        DType::Ggml(GgmlType::Q4_K) => q4_k(row, out),
        DType::Ggml(GgmlType::Q6_K) => q6_k(row, out),
        _ => out.fill(f32::NAN),
    }
}

/// `y = q · d`.
#[inline(always)]
fn q8_0(row: &[u8], out: &mut [f32]) {
    for (blk, o) in row.chunks_exact(Q8_0_BYTES).zip(out.chunks_exact_mut(32)) {
        let d = f16_at(blk, 0);
        for (y, &q) in o.iter_mut().zip(&blk[2..]) {
            #[allow(clippy::cast_possible_wrap)]
            let q = q as i8;
            *y = f32::from(q) * d;
        }
    }
}

/// GGML `get_scale_min_k4`.
#[inline(always)]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// `y = d1 · q − m1`: four 64-weight groups per block, low nibbles then high.
#[inline(always)]
fn q4_k(row: &[u8], out: &mut [f32]) {
    for (blk, o) in row.chunks_exact(Q4_K_BYTES).zip(out.chunks_exact_mut(QK_K)) {
        let d = f16_at(blk, 0);
        let dmin = f16_at(blk, 2);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        for j in 0..4 {
            let (sc, m) = scale_min_k4(2 * j, scales);
            let (d1, m1) = (d * f32::from(sc), dmin * f32::from(m));
            let (sc, m) = scale_min_k4(2 * j + 1, scales);
            let (d2, m2) = (d * f32::from(sc), dmin * f32::from(m));
            let q = &qs[32 * j..32 * j + 32];
            let (lo, hi) = o[64 * j..64 * j + 64].split_at_mut(32);
            for ((a, b), &v) in lo.iter_mut().zip(hi.iter_mut()).zip(q) {
                *a = d1 * f32::from(v & 0xF) - m1;
                *b = d2 * f32::from(v >> 4) - m2;
            }
        }
    }
}

/// `y = (d · scale) · q`, q a signed 6-bit value: two 128-weight halves per block.
#[inline(always)]
fn q6_k(row: &[u8], out: &mut [f32]) {
    for (blk, o) in row.chunks_exact(Q6_K_BYTES).zip(out.chunks_exact_mut(QK_K)) {
        let (ql_all, rest) = blk.split_at(128);
        let (qh_all, rest) = rest.split_at(64);
        let (sc_all, d) = rest.split_at(16);
        let d = f16_at(d, 0);
        for half in 0..2 {
            let ql = &ql_all[64 * half..64 * half + 64];
            let qh = &qh_all[32 * half..32 * half + 32];
            let sc = &sc_all[8 * half..8 * half + 8];
            let y = &mut o[128 * half..128 * half + 128];
            for l in 0..32 {
                let is = l / 16;
                #[allow(clippy::cast_possible_wrap)]
                let q = |low: u8, high: u8| f32::from(((low & 0xF) | ((high & 3) << 4)) as i8 - 32);
                let q1 = q(ql[l], qh[l]);
                let q2 = q(ql[l + 32], qh[l] >> 2);
                let q3 = q(ql[l] >> 4, qh[l] >> 4);
                let q4 = q(ql[l + 32] >> 4, qh[l] >> 6);
                #[allow(clippy::cast_possible_wrap)]
                let s = |i: usize| d * f32::from(sc[i] as i8);
                y[l] = s(is) * q1;
                y[l + 32] = s(is + 2) * q2;
                y[l + 64] = s(is + 4) * q3;
                y[l + 96] = s(is + 6) * q4;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_is_exact_for_every_value() {
        for h in 0..=u16::MAX {
            let got = f16_to_f32(h);
            let s = if h & 0x8000 != 0 { -1.0f64 } else { 1.0 };
            let e = i32::from((h >> 10) & 0x1f);
            let m = f64::from(h & 0x3ff);
            let want = match e {
                0 => s * m * 2f64.powi(-24),
                31 if m == 0.0 => s * f64::INFINITY,
                31 => f64::NAN,
                _ => s * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
            };
            if want.is_nan() {
                assert!(got.is_nan(), "{h:#06x}");
            } else {
                assert_eq!(f64::from(got).to_bits(), want.to_bits(), "{h:#06x}");
            }
        }
    }
}
