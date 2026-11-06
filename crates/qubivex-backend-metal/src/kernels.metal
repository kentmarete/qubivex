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

// Token tiles (ADR-011 amendment 5): a `_t` kernel computes up to TILE activation rows
// per SIMD group, so each weight is loaded and dequantised once for all of them. Every
// (row, token) output keeps its own two lane accumulators, the same fused multiply-adds
// in the same order and the same tree: the bits are those of the one-token kernel.
// Must equal `TILE` in `lib.rs`.
constant uint TILE = 8;

typedef device const float *xptr;
typedef device float *yptr;

// Routed matmuls (Phase 8 slice 5, ADR-011 amendment 4): entry j of `s * top_k` routes
// one token to one expert. Its expert is idx[j] (u32 bits in a float buffer, from
// `moe_route`), whose weights start at ptrs[expert]; its input row is xs[j] (per
// entry) or xs[j / top_k] (the token's row), and its output row ys[j]. Each output is
// the same dot product, in the same order, as the plain matmul's: the bits match.
//
// The grouped form (`_g`) takes the entries sorted by expert (`moe_group`): threadgroup
// row y covers tile y % tiles of expert y / tiles, up to TILE of its entries.
struct RoutedParams {
    uint n;         // output rows
    uint k;         // input columns
    uint row_bytes; // bytes per weight row
    uint top_k;     // entries per token
    uint ne;        // experts in the table
    uint per_entry; // 1: input row j; 0: input row j / top_k
    uint entries;   // s * top_k
    uint tiles;     // grouped: tiles per expert
};

// One expert's weights: a device address, written by the host from the buffer's
// gpuAddress plus the matrix's offset (the buffer is declared with useResource).
struct ExpertPtr {
    device const uchar *w;
};

// ---- Row bodies: rows r0.. (RPS of them, n in all) against NT input rows. ----

// Generic per-element formats: one row per SIMD group.
#define GENERIC_ROWS(NAME, WFN)                                                        \
template <uint NT>                                                                     \
static inline void rows_##NAME(device const uchar *w, thread xptr *xp, thread yptr *yp, \
                               uint nt, uint n, uint k, uint row_bytes, uint r,        \
                               uint lane) {                                            \
    device const uchar *row = w + ulong(r) * row_bytes;                                \
    float a0[NT];                                                                      \
    float a1[NT];                                                                      \
    for (uint t = 0; t < NT; t++) {                                                    \
        a0[t] = 0.0f;                                                                  \
        a1[t] = 0.0f;                                                                  \
    }                                                                                  \
    uint full = k / 64 * 64;                                                           \
    for (uint c = 0; c < full; c += 64) {                                              \
        uint j0 = c + lane;                                                            \
        uint j1 = j0 + 32;                                                             \
        float w0 = WFN(row, j0);                                                       \
        float w1 = WFN(row, j1);                                                       \
        for (uint t = 0; t < NT; t++) {                                                \
            a0[t] = fma(w0, xp[t][j0], a0[t]);                                         \
            a1[t] = fma(w1, xp[t][j1], a1[t]);                                         \
        }                                                                              \
    }                                                                                  \
    uint rem = k - full;                                                               \
    if (lane < rem) {                                                                  \
        float w0 = WFN(row, full + lane);                                              \
        for (uint t = 0; t < NT; t++) {                                                \
            a0[t] = fma(w0, xp[t][full + lane], a0[t]);                                \
        }                                                                              \
    }                                                                                  \
    if (lane + 32 < rem) {                                                             \
        float w1 = WFN(row, full + lane + 32);                                         \
        for (uint t = 0; t < NT; t++) {                                                \
            a1[t] = fma(w1, xp[t][full + lane + 32], a1[t]);                           \
        }                                                                              \
    }                                                                                  \
    for (uint t = 0; t < NT; t++) {                                                    \
        float v = tree(a0[t] + a1[t]);                                                 \
        if (lane == 0 && t < nt) {                                                     \
            yp[t][r] = v;                                                              \
        }                                                                              \
    }                                                                                  \
}

GENERIC_ROWS(f32, w_f32)
GENERIC_ROWS(f16, w_f16)
GENERIC_ROWS(bf16, w_bf16)
GENERIC_ROWS(q8_0, w_q8_0)

// ---- Block-wise K-quant rows. ----
//
// The same lanes and order as the generic rows, and each weight's value is computed
// with the CPU's operations (`formats::dequant_row`):
// - Q4_K (144-byte blocks of 256: f16 d, f16 dmin, 12 scale bytes, 128 nibble bytes):
//   y = (d * sc) * q - (dmin * m), with (sc, m) from GGML get_scale_min_k4;
// - Q6_K (210-byte blocks of 256: ql 128, qh 64, 16 x i8 scales, f16 d):
//   y = (d * scale) * q, q a signed 6-bit value.
// A 64-element chunk's scales are computed once, and each quant byte is loaded once for
// both of the thread's lanes. Every K-quant row is a whole number of 256-element blocks
// (k % 256 == 0), so there is no tail.

// Rows per SIMD group for the K-quant kernels: each activation chunk is loaded once and
// used for NR rows. Each row keeps its own lanes and order. Must equal `KQ_ROWS` in
// `device.rs`.
constant uint NR = 2;

static inline float f16_at_t(device const uchar *b) {
    return float(as_type<half>(ushort(ushort(b[0]) | (ushort(b[1]) << 8))));
}

// Byte `i` (0..15) of a 16-byte block header held as a uint4.
static inline uint hbyte(uint4 h, uint i) {
    return (h[i / 4] >> (8 * (i % 4))) & 0xFF;
}

// GGML get_scale_min_k4 on the 12 scale bytes at header bytes 4..15.
static inline uint2 scale_min_h(uint j, uint4 h) {
    if (j < 4) {
        return uint2(hbyte(h, 4 + j) & 63, hbyte(h, 8 + j) & 63);
    }
    return uint2((hbyte(h, 8 + j) & 0xF) | ((hbyte(h, j) >> 6) << 4),
                 (hbyte(h, 8 + j) >> 4) | ((hbyte(h, 4 + j) >> 6) << 4));
}

// Q4_K rows r0.. (up to NR). Each block's 16-byte header (d, dmin, 12 scale bytes) is
// loaded once as a uint4: the weights must start 16-byte aligned (`offset_ok`).
template <uint NT>
static inline void rows_q4_k(device const uchar *w, thread xptr *xp, thread yptr *yp,
                             uint nt, uint n, uint k, uint row_bytes, uint r0,
                             uint lane) {
    uint nr = min(NR, n - r0);
    device const uchar *rows[NR];
    for (uint i = 0; i < NR; i++) {
        rows[i] = w + ulong(r0 + min(i, nr - 1)) * row_bytes;
    }
    float a0[NR][NT];
    float a1[NR][NT];
    for (uint i = 0; i < NR; i++) {
        for (uint t = 0; t < NT; t++) {
            a0[i][t] = 0.0f;
            a1[i][t] = 0.0f;
        }
    }
    uint blocks = k / 256;
    for (uint b = 0; b < blocks; b++) {
        uint4 hd[NR];
        float d[NR];
        float dmin[NR];
        for (uint i = 0; i < NR; i++) {
            hd[i] = *((device const uint4 *)(rows[i] + 144 * b));
            d[i] = float(as_type<half>(ushort(hd[i].x & 0xFFFF)));
            dmin[i] = float(as_type<half>(ushort(hd[i].x >> 16)));
        }
        for (uint g = 0; g < 4; g++) {
            float x0[NT];
            float x1[NT];
            for (uint t = 0; t < NT; t++) {
                x0[t] = xp[t][256 * b + 64 * g + lane];
                x1[t] = xp[t][256 * b + 64 * g + 32 + lane];
            }
            for (uint i = 0; i < NR; i++) {
                // Lanes 0..31 are sub-block 2g (low nibbles), lanes 32..63 sub-block
                // 2g + 1 (high nibbles of the same bytes).
                uint2 sm0 = scale_min_h(2 * g, hd[i]);
                uint2 sm1 = scale_min_h(2 * g + 1, hd[i]);
                float d0 = d[i] * float(sm0.x);
                float m0 = dmin[i] * float(sm0.y);
                float d1 = d[i] * float(sm1.x);
                float m1 = dmin[i] * float(sm1.y);
                uchar v = rows[i][144 * b + 16 + 32 * g + lane];
                float w0 = d0 * float(v & 0xF) - m0;
                float w1 = d1 * float(v >> 4) - m1;
                for (uint t = 0; t < NT; t++) {
                    a0[i][t] = fma(w0, x0[t], a0[i][t]);
                    a1[i][t] = fma(w1, x1[t], a1[i][t]);
                }
            }
        }
    }
    for (uint i = 0; i < NR; i++) {
        for (uint t = 0; t < NT; t++) {
            float v = tree(a0[i][t] + a1[i][t]);
            if (lane == 0 && i < nr && t < nt) {
                yp[t][r0 + i] = v;
            }
        }
    }
}

// Q6_K rows r0.. (up to NR).
template <uint NT>
static inline void rows_q6_k(device const uchar *w, thread xptr *xp, thread yptr *yp,
                             uint nt, uint n, uint k, uint row_bytes, uint r0,
                             uint lane) {
    uint nr = min(NR, n - r0);
    device const uchar *rows[NR];
    for (uint i = 0; i < NR; i++) {
        rows[i] = w + ulong(r0 + min(i, nr - 1)) * row_bytes;
    }
    float a0[NR][NT];
    float a1[NR][NT];
    for (uint i = 0; i < NR; i++) {
        for (uint t = 0; t < NT; t++) {
            a0[i][t] = 0.0f;
            a1[i][t] = 0.0f;
        }
    }
    uint blocks = k / 256;
    uint is = lane / 16;
    for (uint b = 0; b < blocks; b++) {
        for (uint m = 0; m < 4; m++) {
            float x0[NT];
            float x1[NT];
            for (uint t = 0; t < NT; t++) {
                x0[t] = xp[t][256 * b + 64 * m + lane];
                x1[t] = xp[t][256 * b + 64 * m + 32 + lane];
            }
            // Chunk m: half m / 2; lanes 0..31 are quarter 2(m % 2), lanes 32..63
            // quarter 2(m % 2) + 1, both at l = lane.
            uint half_ = m / 2;
            uint q0 = 2 * (m % 2);
            for (uint i = 0; i < NR; i++) {
                device const uchar *blk = rows[i] + 210 * b;
                float d = f16_at_t(blk + 208);
                device const uchar *ql = blk + 64 * half_;
                uchar hb = blk[128 + 32 * half_ + lane];
                device const uchar *scl = blk + 192 + 8 * half_;
                float s0 = d * float(as_type<char>(scl[is + 2 * q0]));
                float s1 = d * float(as_type<char>(scl[is + 2 * (q0 + 1)]));
                // Quarter q0 (even) reads ql[l], quarter q0 + 1 (odd) ql[l + 32]; both
                // take the high nibble when the quarter is >= 2.
                uchar lo0 = ql[lane];
                uchar lo1 = ql[lane + 32];
                lo0 = (q0 >= 2) ? (lo0 >> 4) : (lo0 & 0xF);
                lo1 = (q0 + 1 >= 2) ? (lo1 >> 4) : (lo1 & 0xF);
                uchar h0 = (hb >> (2 * q0)) & 3;
                uchar h1 = (hb >> (2 * (q0 + 1))) & 3;
                float w0 = s0 * float(int(lo0 | (h0 << 4)) - 32);
                float w1 = s1 * float(int(lo1 | (h1 << 4)) - 32);
                for (uint t = 0; t < NT; t++) {
                    a0[i][t] = fma(w0, x0[t], a0[i][t]);
                    a1[i][t] = fma(w1, x1[t], a1[i][t]);
                }
            }
        }
    }
    for (uint i = 0; i < NR; i++) {
        for (uint t = 0; t < NT; t++) {
            float v = tree(a0[i][t] + a1[i][t]);
            if (lane == 0 && i < nr && t < nt) {
                yp[t][r0 + i] = v;
            }
        }
    }
}

// ---- Kernels: plain, token-tiled, routed and grouped routed, per format. ----
// RPS: output rows per SIMD group (1, or NR for the K-quants).

#define TG_ARGS                                                                        \
    uint2 tg [[threadgroup_position_in_grid]],                                         \
    uint sg [[simdgroup_index_in_threadgroup]],                                        \
    uint lane [[thread_index_in_simdgroup]]

#define MATMUL_KERNELS(NAME, RPS)                                                      \
kernel void matmul_##NAME(device const uchar *w [[buffer(0)]],                         \
                          device const float *xs [[buffer(1)]],                        \
                          device float *ys [[buffer(2)]],                              \
                          constant Params &p [[buffer(3)]],                            \
                          TG_ARGS) {                                                   \
    uint r0 = (tg.x * ROWS + sg) * RPS;                                                \
    if (r0 >= p.n) {                                                                   \
        return;                                                                        \
    }                                                                                  \
    xptr xp[1] = {xs + ulong(tg.y) * p.k};                                             \
    yptr yp[1] = {ys + ulong(tg.y) * p.n};                                             \
    rows_##NAME<1>(w, xp, yp, 1, p.n, p.k, p.row_bytes, r0, lane);                     \
}                                                                                      \
kernel void matmul_##NAME##_t(device const uchar *w [[buffer(0)]],                     \
                              device const float *xs [[buffer(1)]],                    \
                              device float *ys [[buffer(2)]],                          \
                              constant Params &p [[buffer(3)]],                        \
                              TG_ARGS) {                                               \
    uint r0 = (tg.x * ROWS + sg) * RPS;                                                \
    uint t0 = tg.y * TILE;                                                             \
    if (r0 >= p.n || t0 >= p.s) {                                                      \
        return;                                                                        \
    }                                                                                  \
    uint nt = min(TILE, p.s - t0);                                                     \
    xptr xp[TILE];                                                                     \
    yptr yp[TILE];                                                                     \
    for (uint t = 0; t < TILE; t++) {                                                  \
        uint tt = t0 + min(t, nt - 1);                                                 \
        xp[t] = xs + ulong(tt) * p.k;                                                  \
        yp[t] = ys + ulong(tt) * p.n;                                                  \
    }                                                                                  \
    rows_##NAME<TILE>(w, xp, yp, nt, p.n, p.k, p.row_bytes, r0, lane);                 \
}                                                                                      \
kernel void routed_##NAME(device const float *xs [[buffer(0)]],                        \
                          device float *ys [[buffer(1)]],                              \
                          device const float *idx [[buffer(2)]],                       \
                          constant ExpertPtr *ptrs [[buffer(3)]],                      \
                          constant RoutedParams &p [[buffer(4)]],                      \
                          TG_ARGS) {                                                   \
    uint r0 = (tg.x * ROWS + sg) * RPS;                                                \
    if (r0 >= p.n) {                                                                   \
        return;                                                                        \
    }                                                                                  \
    uint j = tg.y;                                                                     \
    uint e = min(as_type<uint>(idx[j]), p.ne - 1);                                     \
    xptr xp[1] = {xs + ulong(p.per_entry != 0 ? j : j / p.top_k) * p.k};               \
    yptr yp[1] = {ys + ulong(j) * p.n};                                                \
    rows_##NAME<1>(ptrs[e].w, xp, yp, 1, p.n, p.k, p.row_bytes, r0, lane);             \
}                                                                                      \
kernel void routed_##NAME##_g(device const float *xs [[buffer(0)]],                    \
                              device float *ys [[buffer(1)]],                          \
                              device const float *order [[buffer(2)]],                 \
                              constant ExpertPtr *ptrs [[buffer(3)]],                  \
                              constant RoutedParams &p [[buffer(4)]],                  \
                              device const float *off [[buffer(5)]],                   \
                              TG_ARGS) {                                               \
    uint r0 = (tg.x * ROWS + sg) * RPS;                                                \
    uint e = min(tg.y / p.tiles, p.ne - 1);                                            \
    uint end = min(as_type<uint>(off[e + 1]), p.entries);                              \
    uint start = as_type<uint>(off[e]) + (tg.y % p.tiles) * TILE;                      \
    if (r0 >= p.n || start >= end) {                                                   \
        return;                                                                        \
    }                                                                                  \
    uint nt = min(TILE, end - start);                                                  \
    xptr xp[TILE];                                                                     \
    yptr yp[TILE];                                                                     \
    for (uint t = 0; t < TILE; t++) {                                                  \
        uint j = min(as_type<uint>(order[start + min(t, nt - 1)]), p.entries - 1);     \
        xp[t] = xs + ulong(p.per_entry != 0 ? j : j / p.top_k) * p.k;                  \
        yp[t] = ys + ulong(j) * p.n;                                                   \
    }                                                                                  \
    rows_##NAME<TILE>(ptrs[e].w, xp, yp, nt, p.n, p.k, p.row_bytes, r0, lane);         \
}

MATMUL_KERNELS(f32, 1)
MATMUL_KERNELS(f16, 1)
MATMUL_KERNELS(bf16, 1)
MATMUL_KERNELS(q8_0, 1)
MATMUL_KERNELS(q4_k, NR)
MATMUL_KERNELS(q6_k, NR)

// ---- Forward-pass operations (Phase 8 slice 3). ----
//
// Each mirrors a function of `qubivex-arch`'s `ops` (or the forward pass itself) with
// the same IEEE operations in the same order, so the bits match the CPU's. Exponentials
// use `qexp`, the port of `qubivex_compute::math::exp`; nothing calls the GPU's `exp`.

constant float QEXP_LOG2E = as_type<float>(0x3fb8aa3bu);
constant float QEXP_LN2_HI = as_type<float>(0x3f317200u);
constant float QEXP_LN2_LO = as_type<float>(0x35bfbe8eu);
constant float QEXP_SHIFT = 12582912.0f;
constant float QEXP_OVERFLOW = as_type<float>(0x42b17218u);
constant float QEXP_UNDERFLOW = as_type<float>(0xc2aeac50u);

static inline float qpow2(int k) {
    return as_type<float>(uint(k + 127) << 23);
}

// `qubivex_compute::math::exp`, operation for operation.
static inline float qexp(float x) {
    if (isnan(x)) {
        return x;
    }
    if (x > QEXP_OVERFLOW) {
        return INFINITY;
    }
    if (x < QEXP_UNDERFLOW) {
        return 0.0f;
    }
    float kf = (x * QEXP_LOG2E + QEXP_SHIFT) - QEXP_SHIFT;
    float r = fma(-kf, QEXP_LN2_HI, x);
    r = fma(-kf, QEXP_LN2_LO, r);
    float p = as_type<float>(0x39506967u);
    p = fma(p, r, as_type<float>(0x3ab743ceu));
    p = fma(p, r, as_type<float>(0x3c088908u));
    p = fma(p, r, as_type<float>(0x3d2aa9c1u));
    p = fma(p, r, as_type<float>(0x3e2aaaaau));
    p = fma(p, r, as_type<float>(0x3f000000u));
    float y = fma(p, r * r, r) + 1.0f;
    int k = int(kf);
    int k1 = k >> 1;
    float v = (y * qpow2(k1)) * qpow2(k - k1);
    return v < FLT_MIN ? 0.0f : v;
}

struct NormParams {
    uint n;    // row length
    float eps; // `eps as f32`
};

// `ops::rms_norm` on each row of `x`, in place. One SIMD group per row; the sum of
// squares has the shared 64-lane shape.
kernel void rms_norm(device float *x [[buffer(0)]],
                     device const float *w [[buffer(1)]],
                     constant NormParams &p [[buffer(2)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]]) {
    device float *xr = x + ulong(row) * p.n;
    float a0 = 0.0f;
    float a1 = 0.0f;
    uint full = p.n / 64 * 64;
    for (uint c = 0; c < full; c += 64) {
        float v0 = xr[c + lane];
        float v1 = xr[c + lane + 32];
        a0 = fma(v0, v0, a0);
        a1 = fma(v1, v1, a1);
    }
    uint rem = p.n - full;
    if (lane < rem) {
        float v = xr[full + lane];
        a0 = fma(v, v, a0);
    }
    if (lane + 32 < rem) {
        float v = xr[full + lane + 32];
        a1 = fma(v, v, a1);
    }
    float ss = simd_broadcast(tree(a0 + a1), 0);
    // `precise::` keeps 1 / sqrt from becoming a (not correctly rounded) rsqrt.
    float r = precise::divide(1.0f, precise::sqrt(precise::divide(ss, float(p.n)) + p.eps));
    simdgroup_barrier(mem_flags::mem_device);
    for (uint j = lane; j < p.n; j += 32) {
        xr[j] = (xr[j] * r) * w[j];
    }
}

struct RopeParams {
    uint half_;          // pairs per head
    uint heads_per_row;  // heads per token
    uint rows;           // tokens
};

// `Rope::rotate` on every head; token t uses table row t (`[cos…, sin…]`).
kernel void rope(device float *x [[buffer(0)]],
                 device const float *cs [[buffer(1)]],
                 constant RopeParams &p [[buffer(2)]],
                 uint gid [[thread_position_in_grid]]) {
    uint per_token = p.heads_per_row * p.half_;
    if (gid >= p.rows * per_token) {
        return;
    }
    uint t = gid / per_token;
    uint hd = (gid % per_token) / p.half_;
    uint i = gid % p.half_;
    device float *h = x + (ulong(t) * p.heads_per_row + hd) * 2 * p.half_;
    device const float *c = cs + ulong(t) * 2 * p.half_;
    float x1 = h[i];
    float x2 = h[i + p.half_];
    float co = c[i];
    float si = c[p.half_ + i];
    h[i] = x1 * co - x2 * si;
    h[i + p.half_] = x2 * co + x1 * si;
}

struct EwParams {
    uint n;
    float f;
};

// `f32::clamp(-c, c)` (NaN stays NaN).
kernel void clamp_abs(device float *x [[buffer(0)]],
                      constant EwParams &p [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) {
        return;
    }
    float v = x[gid];
    x[gid] = v < -p.f ? -p.f : (v > p.f ? p.f : v);
}

// a += b.
kernel void add_in_place(device float *a [[buffer(0)]],
                         device const float *b [[buffer(1)]],
                         constant EwParams &p [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) {
        return;
    }
    a[gid] = a[gid] + b[gid];
}

// dst = src.
kernel void copy_f32(device float *dst [[buffer(0)]],
                     device const float *src [[buffer(1)]],
                     constant EwParams &p [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) {
        return;
    }
    dst[gid] = src[gid];
}

// `ops::swiglu`: g = silu(g) * u.
kernel void swiglu(device float *g [[buffer(0)]],
                   device const float *u [[buffer(1)]],
                   constant EwParams &p [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) {
        return;
    }
    float a = g[gid];
    float s = precise::divide(a, 1.0f + qexp(-a));
    g[gid] = s * u[gid];
}

struct KvParams {
    uint s;         // tokens
    uint kv_heads;
    uint d;         // head dim
    uint p0;        // first position
};

// Appends `s` tokens' keys or values (`[t][g][d]`) to a position-major cache
// (`[pos][g][d]`): one contiguous copy.
kernel void kv_store(device float *cache [[buffer(0)]],
                     device const float *src [[buffer(1)]],
                     constant KvParams &p [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    uint per_token = p.kv_heads * p.d;
    if (gid >= p.s * per_token) {
        return;
    }
    cache[ulong(p.p0) * per_token + gid] = src[gid];
}

struct AttnParams {
    uint s;         // tokens
    uint heads;
    uint group;     // query heads per KV head
    uint d;
    uint p0;        // first token's position
    uint span;      // p0 + s: score slots per (token, head)
    uint kv_heads;  // the cache is `[pos][kv_head][d]`
    float scale;    // 1 / sqrt(d)
};

// Scores of `ops::attend`: q·k (sequential fma) times the scale, for every visible
// position. Grid: (position, head, token).
kernel void attn_scores(device const float *q [[buffer(0)]],
                        device const float *kc [[buffer(1)]],
                        device float *scores [[buffer(2)]],
                        constant AttnParams &p [[buffer(3)]],
                        uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint h = gid.y;
    uint t = gid.z;
    if (j >= p.span || h >= p.heads || t >= p.s || j > p.p0 + t) {
        return;
    }
    device const float *qh = q + (ulong(t) * p.heads + h) * p.d;
    device const float *kh = kc + (ulong(j) * p.kv_heads + h / p.group) * p.d;
    float acc = 0.0f;
    for (uint i = 0; i < p.d; i++) {
        acc = fma(qh[i], kh[i], acc);
    }
    scores[(ulong(t) * p.heads + h) * p.span + j] = acc * p.scale;
}

// `ops::softmax` over each (token, head)'s visible scores: max, then exp and a
// sequential sum, then division.
kernel void attn_softmax(device float *scores [[buffer(0)]],
                         constant AttnParams &p [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= p.s * p.heads) {
        return;
    }
    uint t = gid / p.heads;
    uint n = p.p0 + t + 1;
    device float *sc = scores + ulong(gid) * p.span;
    float m = -INFINITY;
    for (uint j = 0; j < n; j++) {
        m = fmax(m, sc[j]);
    }
    float sum = 0.0f;
    for (uint j = 0; j < n; j++) {
        float v = qexp(sc[j] - m);
        sc[j] = v;
        sum += v;
    }
    for (uint j = 0; j < n; j++) {
        sc[j] = precise::divide(sc[j], sum);
    }
}

// Output of `ops::attend`: out[i] = Σ_j p_j · v_j[i] (sequential fma from zero).
// Grid: (head × d, token).
kernel void attn_values(device const float *scores [[buffer(0)]],
                        device const float *vc [[buffer(1)]],
                        device float *out [[buffer(2)]],
                        constant AttnParams &p [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint hi = gid.x;
    uint t = gid.y;
    if (hi >= p.heads * p.d || t >= p.s) {
        return;
    }
    uint h = hi / p.d;
    uint i = hi % p.d;
    device const float *sc = scores + (ulong(t) * p.heads + h) * p.span;
    device const float *vh = vc + ulong(h / p.group) * p.d + i;
    ulong pos_stride = ulong(p.kv_heads) * p.d;
    uint n = p.p0 + t + 1;
    float o = 0.0f;
    for (uint j = 0; j < n; j++) {
        o = fma(sc[j], vh[ulong(j) * pos_stride], o);
    }
    out[ulong(t) * p.heads * p.d + hi] = o;
}

struct CombineParams {
    uint s;     // tokens
    uint k;     // experts per token
    uint h;     // hidden
};

// The MoE combine: out[t] = Σ_r w[t][r] · contrib[t][r] (sequential fma from zero, rank
// order). Grid: (h, token).
kernel void moe_combine(device const float *contrib [[buffer(0)]],
                        device const float *w [[buffer(1)]],
                        device float *out [[buffer(2)]],
                        constant CombineParams &p [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint t = gid.y;
    if (i >= p.h || t >= p.s) {
        return;
    }
    float o = 0.0f;
    for (uint r = 0; r < p.k; r++) {
        o = fma(w[t * p.k + r], contrib[(ulong(t) * p.k + r) * p.h + i], o);
    }
    out[ulong(t) * p.h + i] = o;
}

// x = qexp(x).
kernel void exp_in_place(device float *x [[buffer(0)]],
                         constant EwParams &p [[buffer(1)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) {
        return;
    }
    x[gid] = qexp(x[gid]);
}

struct RowsParams {
    uint h;     // row length
    uint n;     // rows
};

// dst[j] = src[idx[j]] for n rows of h floats.
kernel void gather_rows(device float *dst [[buffer(0)]],
                        device const float *src [[buffer(1)]],
                        constant uint *idx [[buffer(2)]],
                        constant RowsParams &p [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.h || gid.y >= p.n) {
        return;
    }
    dst[ulong(gid.y) * p.h + gid.x] = src[ulong(idx[gid.y]) * p.h + gid.x];
}

// dst[idx[j]] = src[j] for n rows of h floats (the indices are distinct).
kernel void scatter_rows(device float *dst [[buffer(0)]],
                         device const float *src [[buffer(1)]],
                         constant uint *idx [[buffer(2)]],
                         constant RowsParams &p [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.h || gid.y >= p.n) {
        return;
    }
    dst[ulong(idx[gid.y]) * p.h + gid.x] = src[ulong(gid.y) * p.h + gid.x];
}

struct RouteParams {
    uint s;     // tokens
    uint ne;    // experts
    uint k;     // experts per token
    uint norm;  // 1: renormalise the top k to sum to 1
};

// The order of `f32::total_cmp`, as a signed integer key.
static inline int total_key(float f) {
    int i = as_type<int>(f);
    return i ^ int(uint(i >> 31) >> 1);
}

// `ops::route` for each token (one thread per token): the softmax of its logits (in
// place: the logits become probabilities), the top k by probability (ties: the lower
// id) in rank order, and their probabilities, renormalised when `norm`. Writes ids as
// u32 bits into `idx` and probabilities into `wts`, k per token. ne <= 256 (the host
// checks).
kernel void moe_route(device float *logits [[buffer(0)]],
                      device float *idx [[buffer(1)]],
                      device float *wts [[buffer(2)]],
                      constant RouteParams &p [[buffer(3)]],
                      uint t [[thread_position_in_grid]]) {
    if (t >= p.s) {
        return;
    }
    device float *x = logits + ulong(t) * p.ne;
    float m = -INFINITY;
    for (uint e = 0; e < p.ne; e++) {
        m = fmax(m, x[e]);
    }
    float sum = 0.0f;
    for (uint e = 0; e < p.ne; e++) {
        float v = qexp(x[e] - m);
        x[e] = v;
        sum += v;
    }
    for (uint e = 0; e < p.ne; e++) {
        x[e] = precise::divide(x[e], sum);
    }
    uint taken[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    device float *oi = idx + ulong(t) * p.k;
    device float *ow = wts + ulong(t) * p.k;
    for (uint r = 0; r < p.k; r++) {
        uint best = 0;
        int key = 0;
        bool found = false;
        for (uint e = 0; e < p.ne; e++) {
            if ((taken[e / 32] >> (e % 32)) & 1) {
                continue;
            }
            int ke = total_key(x[e]);
            if (!found || ke > key) {
                best = e;
                key = ke;
                found = true;
            }
        }
        taken[best / 32] |= 1u << (best % 32);
        oi[r] = as_type<float>(best);
        ow[r] = x[best];
    }
    if (p.norm != 0) {
        float s = -0.0f;
        for (uint r = 0; r < p.k; r++) {
            s += ow[r];
        }
        for (uint r = 0; r < p.k; r++) {
            ow[r] = precise::divide(ow[r], s);
        }
    }
}

struct GroupParams {
    uint entries;   // s * top_k
    uint ne;        // experts
};

// The routed entries sorted by expert (counting sort, entries ascending within each
// expert): order[off[e]..off[e + 1]] are expert e's entries. One thread; ne <= 256.
kernel void moe_group(device const float *idx [[buffer(0)]],
                      device float *order [[buffer(1)]],
                      device float *off [[buffer(2)]],
                      constant GroupParams &p [[buffer(3)]],
                      uint tid [[thread_position_in_grid]]) {
    if (tid != 0) {
        return;
    }
    uint cur[257];
    for (uint e = 0; e <= p.ne; e++) {
        cur[e] = 0;
    }
    for (uint j = 0; j < p.entries; j++) {
        cur[min(as_type<uint>(idx[j]), p.ne - 1) + 1] += 1;
    }
    for (uint e = 0; e < p.ne; e++) {
        cur[e + 1] += cur[e];
    }
    for (uint e = 0; e <= p.ne; e++) {
        off[e] = as_type<float>(cur[e]);
    }
    for (uint j = 0; j < p.entries; j++) {
        uint e = min(as_type<uint>(idx[j]), p.ne - 1);
        order[cur[e]] = as_type<float>(j);
        cur[e] += 1;
    }
}
