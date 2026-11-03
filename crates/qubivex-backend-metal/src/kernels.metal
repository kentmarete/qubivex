// QUBIVEX Metal kernels (ADR-007, ADR-011).
//
// Every kernel reproduces, bit for bit, what the CPU backend computes:
// - each weight's value is the value `formats::dequant_row` gives it (the same
//   operations in the same order, with a separate multiply and subtract, never fused);
// - the dot product uses the CPU's reduction shape (`reduce::dot`): 64 f32 lanes, one
//   fused multiply-add per lane per 64-element chunk in increasing k, a partial final
//   chunk in lanes 0..rem, then the halving tree a[i] += a[i + w] for w = 32, 16, 8, 4,
//   then (a0 + a1) + (a2 + a3).
//
// One SIMD group of 32 threads computes one output element. Thread t holds lanes t and
// t + 32, so the w = 32 step is a per-thread add and the later steps are shuffles.
// Fast math is off (compile options), and contraction is off here, so `a * b - c`
// stays two rounded operations and only the explicit `fma` calls are fused.

#include <metal_stdlib>
using namespace metal;

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

struct Params {
    uint n;         // output rows
    uint k;         // input columns
    uint s;         // activation rows
    uint row_bytes; // bytes per weight row
};

static inline float f16_at(device const uchar *b, uint i) {
    ushort h = ushort(b[i]) | (ushort(b[i + 1]) << 8);
    return float(as_type<half>(h));
}

// ---- Per-element dequantisation: the value of weight `j` of one row. ----

static inline float w_f32(device const uchar *row, uint j) {
    device const uchar *p = row + 4 * j;
    uint u = uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
    return as_type<float>(u);
}

static inline float w_f16(device const uchar *row, uint j) {
    return f16_at(row, 2 * j);
}

static inline float w_bf16(device const uchar *row, uint j) {
    uint u = (uint(row[2 * j]) | (uint(row[2 * j + 1]) << 8)) << 16;
    return as_type<float>(u);
}

// Q8_0: 34-byte blocks of f16 d then 32 x i8; y = q * d.
static inline float w_q8_0(device const uchar *row, uint j) {
    device const uchar *blk = row + 34 * (j / 32);
    float d = f16_at(blk, 0);
    float q = float(as_type<char>(blk[2 + j % 32]));
    return q * d;
}

// GGML get_scale_min_k4.
static inline uchar2 scale_min_k4(uint j, device const uchar *q) {
    if (j < 4) {
        return uchar2(q[j] & 63, q[j + 4] & 63);
    }
    return uchar2((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                  (q[j + 4] >> 4) | ((q[j] >> 6) << 4));
}

// Q4_K: 144-byte blocks of 256; y = (d * sc) * q - (dmin * m).
static inline float w_q4_k(device const uchar *row, uint j) {
    device const uchar *blk = row + 144 * (j / 256);
    uint i = j % 256;
    uint g = i / 64;
    uint w = i % 64;
    float d = f16_at(blk, 0);
    float dmin = f16_at(blk, 2);
    uint sub = 2 * g + (w >= 32 ? 1 : 0);
    uchar2 sm = scale_min_k4(sub, blk + 4);
    float d1 = d * float(sm.x);
    float m1 = dmin * float(sm.y);
    uchar v = blk[16 + 32 * g + (w % 32)];
    float q = float(w < 32 ? (v & 0xF) : (v >> 4));
    return d1 * q - m1;
}

// Q6_K: 210-byte blocks of 256 (ql 128, qh 64, scales 16 x i8, f16 d);
// y = (d * scale) * q, q a signed 6-bit value.
static inline float w_q6_k(device const uchar *row, uint j) {
    device const uchar *blk = row + 210 * (j / 256);
    uint i = j % 256;
    uint half_ = i / 128;
    uint r = i % 128;
    uint quarter = r / 32;
    uint l = r % 32;
    device const uchar *ql = blk + 64 * half_;
    device const uchar *qh = blk + 128 + 32 * half_;
    device const uchar *sc = blk + 192 + 8 * half_;
    float d = f16_at(blk, 208);
    uchar low = (quarter & 1) ? ql[l + 32] : ql[l];
    low = (quarter >= 2) ? (low >> 4) : (low & 0xF);
    uchar high = (qh[l] >> (2 * quarter)) & 3;
    float q = float(int(low | (high << 4)) - 32);
    uint is = l / 16 + 2 * quarter;
    float s = d * float(as_type<char>(sc[is]));
    return s * q;
}

// ---- The shared reduction. ----

// `a` is thread t's sum of lanes t and t + 32 (the w = 32 step). Returns the full sum
// in thread 0.
static inline float tree(float a) {
    a = a + simd_shuffle_down(a, 16);
    a = a + simd_shuffle_down(a, 8);
    a = a + simd_shuffle_down(a, 4);
    // Threads 0..3 hold a0..a3: (a0 + a1) + (a2 + a3).
    float b = a + simd_shuffle_down(a, 1);
    return b + simd_shuffle_down(b, 2);
}

// Rows per threadgroup (one SIMD group each).
constant uint ROWS = 8;

#define MATMUL_KERNEL(NAME, WFN)                                                       \
kernel void NAME(device const uchar *w [[buffer(0)]],                                  \
                 device const float *xs [[buffer(1)]],                                 \
                 device float *ys [[buffer(2)]],                                       \
                 constant Params &p [[buffer(3)]],                                     \
                 uint2 tg [[threadgroup_position_in_grid]],                            \
                 uint sg [[simdgroup_index_in_threadgroup]],                           \
                 uint lane [[thread_index_in_simdgroup]]) {                            \
    uint r = tg.x * ROWS + sg;                                                         \
    uint t = tg.y;                                                                     \
    if (r >= p.n) {                                                                    \
        return;                                                                        \
    }                                                                                  \
    device const uchar *row = w + ulong(r) * p.row_bytes;                              \
    device const float *x = xs + ulong(t) * p.k;                                       \
    float a0 = 0.0f;                                                                   \
    float a1 = 0.0f;                                                                   \
    uint full = p.k / 64 * 64;                                                         \
    for (uint c = 0; c < full; c += 64) {                                              \
        uint j0 = c + lane;                                                            \
        uint j1 = j0 + 32;                                                             \
        a0 = fma(WFN(row, j0), x[j0], a0);                                             \
        a1 = fma(WFN(row, j1), x[j1], a1);                                             \
    }                                                                                  \
    uint rem = p.k - full;                                                             \
    if (lane < rem) {                                                                  \
        a0 = fma(WFN(row, full + lane), x[full + lane], a0);                           \
    }                                                                                  \
    if (lane + 32 < rem) {                                                             \
        a1 = fma(WFN(row, full + lane + 32), x[full + lane + 32], a1);                 \
    }                                                                                  \
    float y = tree(a0 + a1);                                                           \
    if (lane == 0) {                                                                   \
        ys[ulong(t) * p.n + r] = y;                                                    \
    }                                                                                  \
}

MATMUL_KERNEL(matmul_f32, w_f32)
MATMUL_KERNEL(matmul_f16, w_f16)
MATMUL_KERNEL(matmul_bf16, w_bf16)
MATMUL_KERNEL(matmul_q8_0, w_q8_0)
MATMUL_KERNEL(matmul_q4_k, w_q4_k)
MATMUL_KERNEL(matmul_q6_k, w_q6_k)
