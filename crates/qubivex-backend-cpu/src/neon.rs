//! Fused NEON row dots for aarch64.
//!
//! Each kernel computes, bit for bit, what [`crate::formats::dequant_row`] followed by
//! [`crate::reduce::dot`] computes:
//! - the same dequantized value per weight. Mostly by the same operations (a separate
//!   multiply and subtract, never fused); Q4_K instead uses one fused multiply-add
//!   whose product is provably exact, so it rounds once, where the subtract does (see
//!   `q4_k_impl`). On CPUs with FEAT_FHM (every Apple M-series), Q4_K forms that
//!   product from two f16 operands with `fmlal` (`q4_k_fhm_impl`);
//! - the same fused multiply-add into the same lane (element `i` of each 64-element
//!   chunk goes to lane `i`; accumulator vector `j` holds lanes `4j..4j+4`);
//! - the same halving-tree reduction.
//!
//! So switching between these and the portable path never changes results. The
//! `matmul` tests compare them directly.

#![allow(clippy::cast_possible_wrap)]

use core::arch::aarch64::{
    float32x4_t, int8x16_t, uint8x16_t, uint16x8_t, vaddq_f32, vandq_u8, vcvtq_f32_s32,
    vcvtq_f32_u32, vdupq_n_f32, vdupq_n_s8, vdupq_n_u8, vdupq_n_u16, vfmaq_f32, vget_low_s8,
    vget_low_s16, vget_low_u8, vget_low_u16, vgetq_lane_f32, vld1q_f32, vld1q_u8, vld1q_u16,
    vmovl_high_s8, vmovl_high_s16, vmovl_high_u16, vmovl_s8, vmovl_s16, vmovl_u16, vmull_high_u8,
    vmull_u8, vmulq_f32, vorrq_u8, vqtbl1q_u8, vreinterpretq_f32_u16, vreinterpretq_f32_u32,
    vreinterpretq_s8_u8, vreinterpretq_u32_u8, vshlq_n_u8, vshlq_n_u32, vshrq_n_u8, vst1q_f32,
    vsubq_s8,
};

use crate::formats::{Q4_K_BYTES, Q6_K_BYTES, Q8_0_BYTES, QK_K, f16_to_f32};
use crate::reduce::LANES;

/// Accumulator vectors: 64 lanes / 4.
const V: usize = LANES / 4;

/// The portable halving tree on vectors: `a[j] += a[j + w]` for w = 8, 4, 2, 1
/// (lane `l` of vector `j` is lane `4j + l`, so this is `a[i] += a[i + 4w]`), then
/// `(l0 + l1) + (l2 + l3)`.
#[inline]
fn reduce(mut a: [float32x4_t; V]) -> f32 {
    // SAFETY: register-only NEON arithmetic; NEON is part of the aarch64 baseline.
    unsafe {
        let mut w = V / 2;
        while w >= 1 {
            for j in 0..w {
                a[j] = vaddq_f32(a[j], a[j + w]);
            }
            w /= 2;
        }
        let v = a[0];
        (vgetq_lane_f32::<0>(v) + vgetq_lane_f32::<1>(v))
            + (vgetq_lane_f32::<2>(v) + vgetq_lane_f32::<3>(v))
    }
}

/// 16 activations starting at `x[off]`.
///
/// # Safety
/// `off + 16 <= x.len()`.
#[inline]
unsafe fn load_x16(x: &[f32], off: usize) -> [float32x4_t; 4] {
    debug_assert!(off + 16 <= x.len());
    // SAFETY: the caller guarantees 16 floats are in bounds from `off`.
    unsafe {
        let p = x.as_ptr().add(off);
        [
            vld1q_f32(p),
            vld1q_f32(p.add(4)),
            vld1q_f32(p.add(8)),
            vld1q_f32(p.add(12)),
        ]
    }
}

/// `acc[base + i] = fma(w[i], x[i], acc[base + i])` for 16 lanes.
#[inline]
fn fma16(acc: &mut [float32x4_t; V], base: usize, w: [float32x4_t; 4], x: [float32x4_t; 4]) {
    // SAFETY: register-only NEON arithmetic; NEON is part of the aarch64 baseline.
    unsafe {
        for i in 0..4 {
            acc[base + i] = vfmaq_f32(acc[base + i], w[i], x[i]);
        }
    }
}

#[inline]
fn u8x16_to_f32(v: uint8x16_t) -> [float32x4_t; 4] {
    // SAFETY: register-only NEON conversions; NEON is part of the aarch64 baseline.
    unsafe {
        use core::arch::aarch64::{vcvtq_f32_u32, vget_low_u8, vmovl_high_u8, vmovl_u8};
        let lo = vmovl_u8(vget_low_u8(v));
        let hi = vmovl_high_u8(v);
        [
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))),
            vcvtq_f32_u32(vmovl_high_u16(lo)),
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))),
            vcvtq_f32_u32(vmovl_high_u16(hi)),
        ]
    }
}

#[inline]
fn i8x16_to_f32(v: int8x16_t) -> [float32x4_t; 4] {
    // SAFETY: register-only NEON conversions; NEON is part of the aarch64 baseline.
    unsafe {
        let lo = vmovl_s8(vget_low_s8(v));
        let hi = vmovl_high_s8(v);
        [
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),
            vcvtq_f32_s32(vmovl_high_s16(lo)),
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
            vcvtq_f32_s32(vmovl_high_s16(hi)),
        ]
    }
}

#[inline]
fn f16_at(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

/// GGML `get_scale_min_k4` (as in `formats`), the reference for [`q4_k_scales_mins`].
#[cfg(test)]
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

/// The 8 scales then the 8 mins of a Q4_K block, as `scale_min_k4(j)` gives them
/// for `j = 0..8` (the 12 packed bytes rearranged as four words with three masks).
#[inline]
fn q4_k_scales_mins(q: &[u8]) -> [u8; 16] {
    const M6: u32 = 0x3f3f_3f3f;
    const M4: u32 = 0x0f0f_0f0f;
    const M2: u32 = 0x0303_0303;
    let w = |i: usize| u32::from_le_bytes([q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]]);
    let (a, b, c) = (w(0), w(1), w(2));
    let sc_lo = a & M6;
    let sc_hi = (c & M4) | (((a >> 6) & M2) << 4);
    let m_lo = b & M6;
    let m_hi = ((c >> 4) & M4) | (((b >> 6) & M2) << 4);
    let mut out = [0u8; 16];
    for (o, v) in out.chunks_exact_mut(4).zip([sc_lo, sc_hi, m_lo, m_hi]) {
        o.copy_from_slice(&v.to_le_bytes());
    }
    out
}

/// `tbl` indices that zero-extend bytes `4i..4i + 4` of a vector into four u32 lanes.
const WIDEN: [[u8; 16]; 4] = {
    let mut t = [[0xFF; 16]; 4];
    let mut i = 0u8;
    while i < 16 {
        t[i as usize / 4][(i as usize % 4) * 4] = i;
        i += 1;
    }
    t
};

/// Q4_K row · x.
///
/// The portable dequantiser computes `d1 · q − m1` as a multiply then a subtract. Here
/// it is one `fma(d1, q, −m1)`, which gives the same bits: `d1 = d · sc` is an f16
/// (11 significant bits) times a 6-bit integer, so it has at most 17 bits, and `d1 · q`
/// with a 4-bit `q` at most 21; both products are exact in f32 (no underflow either:
/// every nonzero product is at least 2⁻²⁴). The only rounding is then the subtraction's
/// in both cases, and `a − b = a + (−b)` exactly, signed zeros included. The same holds
/// for `−m1 = (−dmin) · m`. With a NaN block scale both paths give NaN, but not
/// necessarily the same NaN (the fused form propagates the addend's).
///
/// # Safety
/// None beyond running on aarch64; lengths are asserted.
unsafe fn q4_k_impl(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is part of the aarch64 baseline. Every load is in bounds: the
    // asserts at the top fix the lengths, and each load notes its bound.
    unsafe {
        assert_eq!(row.len() / Q4_K_BYTES * QK_K, x.len());
        let mut acc = [vdupq_n_f32(0.0); V];
        let mask = vdupq_n_u8(0x0F);
        let widen = WIDEN.map(|t| vld1q_u8(t.as_ptr()));
        let mut dm = [0f32; 16];
        for (b, blk) in row.chunks_exact(Q4_K_BYTES).enumerate() {
            let d = vdupq_n_f32(f16_at(blk, 0));
            let nmin = vdupq_n_f32(-f16_at(blk, 2));
            let sm = q4_k_scales_mins(&blk[4..16]);
            // SAFETY: `sm` has 16 bytes.
            let f = u8x16_to_f32(vld1q_u8(sm.as_ptr()));
            // dm[0..8] = d · sc_j, dm[8..16] = −dmin · m_j: exact products.
            for (i, (v, s)) in f.into_iter().zip([d, d, nmin, nmin]).enumerate() {
                // SAFETY: `dm` has 16 floats; 4i + 4 <= 16.
                vst1q_f32(dm.as_mut_ptr().add(4 * i), vmulq_f32(s, v));
            }
            for j in 0..4 {
                let q = &blk[16 + 32 * j..16 + 32 * j + 32];
                // SAFETY: `q` has 32 bytes.
                let (qa, qb) = { (vld1q_u8(q.as_ptr()), vld1q_u8(q.as_ptr().add(16))) };
                let (d1, m1) = (vdupq_n_f32(dm[2 * j]), vdupq_n_f32(dm[8 + 2 * j]));
                let (d2, m2) = (vdupq_n_f32(dm[2 * j + 1]), vdupq_n_f32(dm[9 + 2 * j]));
                let off = b * QK_K + 64 * j;
                // SAFETY (each load_x16): off + 64 <= (b + 1) * 256 <= x.len().
                let lanes = |qv: uint8x16_t, dd: float32x4_t, mm: float32x4_t| {
                    [0, 1, 2, 3].map(|i| {
                        let q = vcvtq_f32_u32(vreinterpretq_u32_u8(vqtbl1q_u8(qv, widen[i])));
                        vfmaq_f32(mm, dd, q)
                    })
                };
                fma16(
                    &mut acc,
                    0,
                    lanes(vandq_u8(qa, mask), d1, m1),
                    load_x16(x, off),
                );
                fma16(
                    &mut acc,
                    4,
                    lanes(vandq_u8(qb, mask), d1, m1),
                    load_x16(x, off + 16),
                );
                fma16(
                    &mut acc,
                    8,
                    lanes(vshrq_n_u8::<4>(qa), d2, m2),
                    load_x16(x, off + 32),
                );
                fma16(
                    &mut acc,
                    12,
                    lanes(vshrq_n_u8::<4>(qb), d2, m2),
                    load_x16(x, off + 48),
                );
            }
        }
        reduce(acc)
    }
}

/// `fmlal{2} acc.4s, a.4h, d.h[0]` (`HI` picks elements 4..8 of `a`): f32 lanes of
/// `acc + a · d`, the f16 product exact and the sum rounded once.
///
/// # Safety
/// The CPU must support FEAT_FHM.
#[inline]
#[target_feature(enable = "fp16,fhm")]
unsafe fn fmlal<const HI: bool>(acc: float32x4_t, a: float32x4_t, d: float32x4_t) -> float32x4_t {
    let mut acc = acc;
    // SAFETY: register-only; the caller guarantees FEAT_FHM.
    unsafe {
        if HI {
            core::arch::asm!("fmlal2 {acc:v}.4s, {a:v}.4h, {d:v}.h[0]", acc = inout(vreg) acc, a = in(vreg) a, d = in(vreg_low16) d, options(pure, nomem, nostack, preserves_flags));
        } else {
            core::arch::asm!("fmlal {acc:v}.4s, {a:v}.4h, {d:v}.h[0]", acc = inout(vreg) acc, a = in(vreg) a, d = in(vreg_low16) d, options(pure, nomem, nostack, preserves_flags));
        }
    }
    acc
}

/// `ucvtf v.8h, v.8h`: eight u16 lanes to f16, exactly for values below 2048.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[inline]
#[target_feature(enable = "fp16")]
unsafe fn u16x8_to_f16(v: uint16x8_t) -> float32x4_t {
    let out: float32x4_t;
    // SAFETY: register-only; the caller guarantees FEAT_FP16.
    unsafe {
        core::arch::asm!("ucvtf {o:v}.8h, {i:v}.8h", i = in(vreg) v, o = lateout(vreg) out, options(pure, nomem, nostack, preserves_flags));
    }
    out
}

/// Q4_K row · x with FEAT_FHM: the same bits as [`q4_k_impl`].
///
/// `d1 · q = d · (sc · q)`: `sc · q` is an integer below 64 · 16, exact in f16, and `d`
/// is the block's f16 scale, so `fmlal` forms the same exact product from two f16
/// operands and adds `−m1` with the one rounding that `q4_k_impl`'s `fma(d1, q, −m1)`
/// has (see there). The integer product comes from one widening multiply.
///
/// # Safety
/// The CPU must support FEAT_FP16 and FEAT_FHM; lengths are asserted.
#[target_feature(enable = "fp16,fhm")]
unsafe fn q4_k_fhm_impl(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is part of the aarch64 baseline and the caller guarantees FP16 and
    // FHM. Every load is in bounds: the asserts at the top fix the lengths, and each
    // load notes its bound.
    unsafe {
        assert_eq!(row.len() / Q4_K_BYTES * QK_K, x.len());
        let mut acc = [vdupq_n_f32(0.0); V];
        let mask = vdupq_n_u8(0x0F);
        let mut nm = [0f32; 8];
        for (b, blk) in row.chunks_exact(Q4_K_BYTES).enumerate() {
            // The raw f16 `d` in element 0 (the other elements are unused).
            let d = vreinterpretq_f32_u16(vdupq_n_u16(u16::from_le_bytes([blk[0], blk[1]])));
            let nmin = vdupq_n_f32(-f16_at(blk, 2));
            let sm = q4_k_scales_mins(&blk[4..16]);
            // SAFETY: `sm` has 16 bytes.
            let f = u8x16_to_f32(vld1q_u8(sm.as_ptr()));
            // nm[j] = −dmin · m_j, exact.
            // SAFETY: `nm` has 8 floats.
            vst1q_f32(nm.as_mut_ptr(), vmulq_f32(nmin, f[2]));
            vst1q_f32(nm.as_mut_ptr().add(4), vmulq_f32(nmin, f[3]));
            for j in 0..4 {
                let q = &blk[16 + 32 * j..16 + 32 * j + 32];
                // SAFETY: `q` has 32 bytes.
                let (qa, qb) = (vld1q_u8(q.as_ptr()), vld1q_u8(q.as_ptr().add(16)));
                let (s1, m1) = (vdupq_n_u8(sm[2 * j]), vdupq_n_f32(nm[2 * j]));
                let (s2, m2) = (vdupq_n_u8(sm[2 * j + 1]), vdupq_n_f32(nm[2 * j + 1]));
                let off = b * QK_K + 64 * j;
                let lanes = |qv: uint8x16_t, sc: uint8x16_t, mm: float32x4_t| {
                    let lo = u16x8_to_f16(vmull_u8(vget_low_u8(qv), vget_low_u8(sc)));
                    let hi = u16x8_to_f16(vmull_high_u8(qv, sc));
                    [
                        fmlal::<false>(mm, lo, d),
                        fmlal::<true>(mm, lo, d),
                        fmlal::<false>(mm, hi, d),
                        fmlal::<true>(mm, hi, d),
                    ]
                };
                // SAFETY (each load_x16): off + 64 <= (b + 1) * 256 <= x.len().
                fma16(
                    &mut acc,
                    0,
                    lanes(vandq_u8(qa, mask), s1, m1),
                    load_x16(x, off),
                );
                fma16(
                    &mut acc,
                    4,
                    lanes(vandq_u8(qb, mask), s1, m1),
                    load_x16(x, off + 16),
                );
                fma16(
                    &mut acc,
                    8,
                    lanes(vshrq_n_u8::<4>(qa), s2, m2),
                    load_x16(x, off + 32),
                );
                fma16(
                    &mut acc,
                    12,
                    lanes(vshrq_n_u8::<4>(qb), s2, m2),
                    load_x16(x, off + 48),
                );
            }
        }
        reduce(acc)
    }
}

/// Q6_K row · x.
///
/// # Safety
/// None beyond running on aarch64; lengths are asserted.
unsafe fn q6_k_impl(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is part of the aarch64 baseline. Every load is in bounds: the
    // asserts at the top fix the lengths, and each load notes its bound.
    unsafe {
        assert_eq!(row.len() / Q6_K_BYTES * QK_K, x.len());
        let mut acc = [vdupq_n_f32(0.0); V];
        let m4 = vdupq_n_u8(0x0F);
        let m2 = vdupq_n_u8(0x03);
        let k32 = vdupq_n_s8(32);
        for (b, blk) in row.chunks_exact(Q6_K_BYTES).enumerate() {
            let d = f16_at(blk, 208);
            for half in 0..2 {
                let ql = &blk[64 * half..64 * half + 64];
                let qh = &blk[128 + 32 * half..128 + 32 * half + 32];
                let sc = &blk[192 + 8 * half..192 + 8 * half + 8];
                let s = |i: usize| vdupq_n_f32(d * f32::from(sc[i] as i8));
                // Two 16-element vectors per 32-weight group: l in 0..16 and 16..32.
                for v16 in 0..2 {
                    let lo = 16 * v16;
                    // SAFETY: ql has 64 bytes and qh 32; lo + 16 <= 32 and lo + 32 + 16 <= 64.
                    let (a, c, h) = {
                        (
                            vld1q_u8(ql.as_ptr().add(lo)),
                            vld1q_u8(ql.as_ptr().add(lo + 32)),
                            vld1q_u8(qh.as_ptr().add(lo)),
                        )
                    };
                    let to_q = |low: uint8x16_t, high: uint8x16_t| -> int8x16_t {
                        vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(low, vshlq_n_u8::<4>(vandq_u8(high, m2)))),
                            k32,
                        )
                    };
                    // q1..q4 → elements lo.., lo+32.., lo+64.., lo+96.. of the 128-weight half.
                    let qs = [
                        (to_q(vandq_u8(a, m4), h), 0, s(v16)),
                        (to_q(vandq_u8(c, m4), vshrq_n_u8::<2>(h)), 32, s(v16 + 2)),
                        (to_q(vshrq_n_u8::<4>(a), vshrq_n_u8::<4>(h)), 64, s(v16 + 4)),
                        (to_q(vshrq_n_u8::<4>(c), vshrq_n_u8::<6>(h)), 96, s(v16 + 6)),
                    ];
                    for (q, base, scale) in qs {
                        let mut w = i8x16_to_f32(q);
                        for v in &mut w {
                            *v = vmulq_f32(scale, *v);
                        }
                        let e = 128 * half + base + lo; // element within the block
                        // SAFETY: b * 256 + e + 16 <= (b + 1) * 256 <= x.len().
                        let xv = { load_x16(x, b * QK_K + e) };
                        fma16(&mut acc, (e % LANES) / 4, w, xv);
                    }
                }
            }
        }
        reduce(acc)
    }
}

/// Q8_0 row · x; an odd block count ends in a half chunk (lanes 0..32), as the
/// portable tail does.
///
/// # Safety
/// None beyond running on aarch64; lengths are asserted.
unsafe fn q8_0_impl(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is part of the aarch64 baseline. Every load is in bounds: the
    // asserts at the top fix the lengths, and each load notes its bound.
    unsafe {
        assert_eq!(row.len() / Q8_0_BYTES * 32, x.len());
        let mut acc = [vdupq_n_f32(0.0); V];
        for (bi, blk) in row.chunks_exact(Q8_0_BYTES).enumerate() {
            let d = vdupq_n_f32(f16_at(blk, 0));
            for h in 0..2 {
                // SAFETY: the block has 34 bytes; 2 + 16h + 16 <= 34.
                let q = { core::arch::aarch64::vld1q_s8(blk.as_ptr().add(2 + 16 * h).cast()) };
                let mut w = i8x16_to_f32(q);
                for v in &mut w {
                    *v = vmulq_f32(*v, d);
                }
                let e = 32 * bi + 16 * h;
                // SAFETY: e + 16 <= 32 * (bi + 1) <= x.len().
                let xv = { load_x16(x, e) };
                fma16(&mut acc, (e % LANES) / 4, w, xv);
            }
        }
        reduce(acc)
    }
}

/// bf16 row · x; a tail shorter than 64 fills lanes `0..rem`.
///
/// # Safety
/// None beyond running on aarch64; lengths are asserted.
unsafe fn bf16_impl(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is part of the aarch64 baseline. Every load is in bounds: the
    // asserts at the top fix the lengths, and each load notes its bound.
    unsafe {
        let n = x.len();
        assert_eq!(row.len(), 2 * n);
        let mut acc = [vdupq_n_f32(0.0); V];
        let full16 = n / 16 * 16;
        let mut e = 0;
        while e < full16 {
            // SAFETY: 32 bytes from 2e are in bounds (e + 16 <= n); x likewise.
            let (lo, hi, xv) = {
                let p = row.as_ptr().add(2 * e).cast::<u16>();
                (vld1q_u16(p), vld1q_u16(p.add(8)), load_x16(x, e))
            };
            let w = [
                vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(vget_low_u16(lo)))),
                vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_high_u16(lo))),
                vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(vget_low_u16(hi)))),
                vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_high_u16(hi))),
            ];
            fma16(&mut acc, (e % LANES) / 4, w, xv);
            e += 16;
        }
        let mut a = [0f32; LANES];
        for (j, v) in acc.iter().enumerate() {
            // SAFETY: `a` has 64 floats; 4j + 4 <= 64.
            core::arch::aarch64::vst1q_f32(a.as_mut_ptr().add(4 * j), *v);
        }
        for i in full16..n {
            let w =
                f32::from_bits(u32::from(u16::from_le_bytes([row[2 * i], row[2 * i + 1]])) << 16);
            a[i % LANES] = w.mul_add(x[i], a[i % LANES]);
        }
        crate::reduce::reduce(&a)
    }
}

// Safe entry points.

/// Q4_K row dot: the FEAT_FHM kernel where the CPU has it, else the baseline NEON one.
/// Both give the same bits.
pub(crate) fn q4_k(row: &[u8], x: &[f32]) -> f32 {
    if has_fhm() {
        // SAFETY: FP16 and FHM were detected just above.
        unsafe { q4_k_fhm_impl(row, x) }
    } else {
        q4_k_base(row, x)
    }
}

/// Q4_K row dot with baseline NEON only.
pub(crate) fn q4_k_base(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is always available on aarch64.
    unsafe { q4_k_impl(row, x) }
}

/// Whether this CPU has the FP16 and FHM extensions the fast Q4_K kernel needs (the
/// result is cached by `std`).
pub(crate) fn has_fhm() -> bool {
    std::arch::is_aarch64_feature_detected!("fp16")
        && std::arch::is_aarch64_feature_detected!("fhm")
}

/// Q6_K row dot.
pub(crate) fn q6_k(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is always available on aarch64.
    unsafe { q6_k_impl(row, x) }
}

/// Q8_0 row dot.
pub(crate) fn q8_0(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is always available on aarch64.
    unsafe { q8_0_impl(row, x) }
}

/// bf16 row dot.
pub(crate) fn bf16(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: NEON is always available on aarch64.
    unsafe { bf16_impl(row, x) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::dequant_row;
    use qubivex_core::{DType, GgmlType};

    /// xorshift64*.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
        #[allow(clippy::cast_possible_truncation)]
        fn byte(&mut self) -> u8 {
            (self.next() >> 56) as u8
        }
        #[allow(clippy::cast_precision_loss)]
        fn f(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    #[test]
    fn q4_k_scales_and_mins_unpack_like_ggml() {
        let mut rng = Rng(7);
        for _ in 0..20_000 {
            let q: Vec<u8> = (0..12).map(|_| rng.byte()).collect();
            let got = q4_k_scales_mins(&q);
            for j in 0..8 {
                assert_eq!((got[j], got[8 + j]), scale_min_k4(j, &q), "{q:?} j{j}");
            }
        }
    }

    /// Every Q4_K kernel against dequantise-then-dot, for every f16 value of `d` and of
    /// `dmin` (subnormal, negative, infinite and NaN included) with random scales,
    /// mins, nibbles and activations. Results must have the same bits; a NaN only has
    /// to be NaN (see `q4_k_impl`).
    #[test]
    fn q4_k_kernels_match_the_portable_path_for_every_block_scale() {
        let dt = DType::Ggml(GgmlType::Q4_K);
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let (mut w, mut blk) = (vec![0f32; QK_K], vec![0u8; Q4_K_BYTES]);
        let mut checked = 0u32;
        for sweep in 0..2 {
            for h in 0..=u16::MAX {
                blk.iter_mut().for_each(|b| *b = rng.byte());
                // The other scale: a typical one, so the swept one decides the result.
                let other = 0x1c00u16 | u16::from(rng.byte() & 0x3f);
                let (dh, mh) = if sweep == 0 { (h, other) } else { (other, h) };
                blk[0..2].copy_from_slice(&dh.to_le_bytes());
                blk[2..4].copy_from_slice(&mh.to_le_bytes());
                let x: Vec<f32> = (0..QK_K).map(|_| rng.f()).collect();
                dequant_row(dt, &blk, &mut w);
                let want = crate::reduce::dot(&w, &x);
                let mut got = vec![("base", q4_k_base(&blk, &x))];
                if has_fhm() {
                    // SAFETY: the features were detected.
                    got.push(("fhm", unsafe { q4_k_fhm_impl(&blk, &x) }));
                }
                for (name, g) in got {
                    if want.is_nan() {
                        assert!(g.is_nan(), "{name} d {dh:#06x} dmin {mh:#06x}: {g} vs NaN");
                    } else {
                        assert_eq!(
                            g.to_bits(),
                            want.to_bits(),
                            "{name} d {dh:#06x} dmin {mh:#06x}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        // Most blocks give a number, not NaN, so the bit comparison did the work.
        assert!(checked > 100_000, "{checked}");
    }

    /// Multi-block rows with random bytes everywhere except normal-range scales.
    #[test]
    fn q4_k_kernels_match_the_portable_path_on_long_rows() {
        let dt = DType::Ggml(GgmlType::Q4_K);
        let mut rng = Rng(3);
        for blocks in [1, 2, 8, 11] {
            for _ in 0..200 {
                let mut row: Vec<u8> = (0..blocks * Q4_K_BYTES).map(|_| rng.byte()).collect();
                for b in row.chunks_exact_mut(Q4_K_BYTES) {
                    for o in [0, 2] {
                        let h =
                            0x1000u16 | (u16::from(rng.byte()) << 4) | u16::from(rng.byte() >> 4);
                        b[o..o + 2].copy_from_slice(&(h & 0xbfff).to_le_bytes());
                    }
                }
                let x: Vec<f32> = (0..blocks * QK_K).map(|_| rng.f()).collect();
                let mut w = vec![0f32; x.len()];
                dequant_row(dt, &row, &mut w);
                let want = crate::reduce::dot(&w, &x).to_bits();
                assert_eq!(q4_k_base(&row, &x).to_bits(), want);
                assert_eq!(q4_k(&row, &x).to_bits(), want);
            }
        }
    }
}
