// The matmul body shared by `matmul.comp` (one matrix per dispatch) and
// `matmul_fused.comp` (several matrices per dispatch), included after the includer
// defines the parameter macros P_N, P_K, P_ROW_BYTES, P_W_BASE, P_X_BASE, P_Y_BASE,
// the row block ROW_BLOCK, the token TOKEN, MATMUL_PARAMS (the declarations they
// read) and MATMUL_PROLOGUE (statements run first in main). See matmul.comp for what
// the body computes and why it gives the CPU's bits.

// Must match `Dtype` in `device.rs`.
const uint DT_F32 = 0;
const uint DT_F16 = 1;
const uint DT_BF16 = 2;
const uint DT_Q8_0 = 3;
const uint DT_Q4_K = 4;
const uint DT_Q6_K = 5;
const uint DT_MXFP4 = 6;

// MXFP4's E2M1 values, doubled, by nibble (`formats::MXFP4_VALUES`).
const float MXFP4_VALUES[16] = float[16](0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
                                         0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0);

layout(constant_id = 0) const uint DTYPE = 0;

// The weights as little-endian 32-bit words; bytes are extracted by hand so the
// shader needs no 8- or 16-bit storage feature.
layout(std430, set = 0, binding = 0) readonly buffer Weights {
    uint w[];
};
layout(std430, set = 0, binding = 1) readonly buffer Xs {
    float xs[];
};
layout(std430, set = 0, binding = 2) writeonly buffer Ys {
    float ys[];
};

MATMUL_PARAMS

shared float red[256];

uint byte_at(uint i) {
    return (w[i >> 2] >> ((i & 3u) * 8u)) & 0xFFu;
}

// The 4 bytes at `i`, little-endian, in at most two loads.
uint word_at(uint i) {
    uint q = i >> 2;
    uint sh = (i & 3u) * 8u;
    return sh == 0u ? w[q] : (w[q] >> sh) | (w[q + 1u] << (32u - sh));
}

uint u16_at(uint i) {
    return byte_at(i) | (byte_at(i + 1u) << 8);
}

uint u32_at(uint i) {
    return u16_at(i) | (u16_at(i + 2u) << 16);
}

int i8_at(uint i) {
    return int(byte_at(i) << 24) >> 24;
}

// Exactly `formats::f16_to_f32`.
float f16_bits(uint h) {
    uint sign = (h & 0x8000u) << 16;
    uint e = (h >> 10) & 0x1Fu;
    uint m = h & 0x3FFu;
    uint bits;
    if (e == 0u) {
        if (m == 0u) {
            bits = sign;
        } else {
            uint ex = 127u - 15u + 1u;
            while ((m & 0x400u) == 0u) {
                m <<= 1;
                ex -= 1u;
            }
            bits = sign | (ex << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (e == 0x1Fu) {
        bits = sign | 0x7F800000u | (m << 13);
    } else {
        bits = sign | ((e + 127u - 15u) << 23) | (m << 13);
    }
    return uintBitsToFloat(bits);
}

float f16_at(uint i) {
    return f16_bits(u16_at(i));
}

// GGML get_scale_min_k4 on the 12 scale bytes at `q`.
uvec2 scale_min_k4(uint j, uint q) {
    if (j < 4u) {
        return uvec2(byte_at(q + j) & 63u, byte_at(q + j + 4u) & 63u);
    }
    return uvec2((byte_at(q + j + 4u) & 0xFu) | ((byte_at(q + j - 4u) >> 6) << 4),
                 (byte_at(q + j + 4u) >> 4) | ((byte_at(q + j) >> 6) << 4));
}

// `v` itself, through an integer OR with a zero no compiler can prove is zero
// (`n` < 2^31), so `v` is a rounded value: a multiply producing it cannot be fused into
// what consumes it.
float opaque(float v) {
    return uintBitsToFloat(floatBitsToUint(v) | (P_N >> 31));
}

// `scale_min_k4` on the 12 scale bytes held in three words.
uint sbyte(uvec3 sw, uint i) {
    return (sw[i >> 2] >> ((i & 3u) * 8u)) & 0xFFu;
}

uvec2 scale_min_k4w(uint j, uvec3 sw) {
    if (j < 4u) {
        return uvec2(sbyte(sw, j) & 63u, sbyte(sw, j + 4u) & 63u);
    }
    return uvec2((sbyte(sw, j + 4u) & 0xFu) | ((sbyte(sw, j - 4u) >> 6) << 4),
                 (sbyte(sw, j + 4u) >> 4) | ((sbyte(sw, j) >> 6) << 4));
}

// MXFP4 element `i` (0..32) of the block at byte `blk`: the doubled E2M1 value from
// the packed nibble (byte i's low nibble for i < 16, else byte i - 16's high one)
// times 2^(e - 128), exact as on the CPU. For e < 2 that scale is an f32 subnormal,
// which MoltenVK's Apple GPUs flush, so the product is formed from two normal
// factors, (v * 2^(e - 2)) * 2^-126: still exact, and the CPU's value whenever the
// weight itself is normal.
float mxfp4_at(uint blk, uint i) {
    uint e = byte_at(blk);
    uint b = byte_at(blk + 1u + (i & 15u));
    float v = MXFP4_VALUES[i < 16u ? (b & 0xFu) : (b >> 4)];
    if (e >= 2u) {
        float y = v * uintBitsToFloat((e - 1u) << 23);
        return y;
    }
    float t = opaque(v * uintBitsToFloat((e + 125u) << 23));
    float y = t * uintBitsToFloat(1u << 23);
    return y;
}

// The value of weight `j` of the row starting at byte `row`.
float weight(uint row, uint j) {
    if (DTYPE == DT_MXFP4) {
        // 17-byte blocks of 32: the E8M0 scale byte, then 16 nibble bytes.
        return mxfp4_at(row + 17u * (j / 32u), j % 32u);
    }
    if (DTYPE == DT_F32) {
        return uintBitsToFloat(u32_at(row + 4u * j));
    } else if (DTYPE == DT_F16) {
        return f16_at(row + 2u * j);
    } else if (DTYPE == DT_BF16) {
        return uintBitsToFloat(u16_at(row + 2u * j) << 16);
    } else if (DTYPE == DT_Q8_0) {
        // 34-byte blocks: f16 d, then 32 x i8. y = q * d.
        uint blk = row + 34u * (j / 32u);
        float d = f16_at(blk);
        float q = float(i8_at(blk + 2u + j % 32u));
        float y = q * d;
        return y;
    } else if (DTYPE == DT_Q4_K) {
        // 144-byte blocks of 256: f16 d, f16 dmin, 12 scale bytes, 128 nibble bytes.
        // Group g of 64: its first 32 weights are the low nibbles of qs[32g..32g+32]
        // (sub-block 2g), the next 32 the high nibbles (sub-block 2g + 1).
        // y = (d * sc) * q - (dmin * m).
        uint blk = row + 144u * (j / 256u);
        uint e = j % 256u;
        uint g = e / 64u;
        uint hi = (e % 64u) / 32u;
        float d = f16_at(blk);
        float dmin = f16_at(blk + 2u);
        uvec2 sm = scale_min_k4(2u * g + hi, blk + 4u);
        float d1 = d * float(sm.x);
        float m1 = dmin * float(sm.y);
        uint v = byte_at(blk + 16u + 32u * g + e % 32u);
        uint q = hi == 0u ? (v & 0xFu) : (v >> 4);
        float t = d1 * float(q);
        float y = t - m1;
        return y;
    } else {
        // Q6_K, 210-byte blocks of 256: ql 128, qh 64, 16 x i8 scales, f16 d.
        // Half h of 128, quarter qq of 32, position l: low bits from
        // ql[64h + l + 32(qq & 1)] (high nibble when qq >= 2), high bits
        // (qh[32h + l] >> 2qq) & 3, scale sc[8h + l / 16 + 2qq].
        // y = (d * scale) * (q - 32).
        uint blk = row + 210u * (j / 256u);
        uint e = j % 256u;
        uint h = e / 128u;
        uint qq = (e % 128u) / 32u;
        uint l = e % 32u;
        uint lo = byte_at(blk + 64u * h + l + 32u * (qq & 1u));
        lo = qq >= 2u ? (lo >> 4) : (lo & 0xFu);
        uint hb = (byte_at(blk + 128u + 32u * h + l) >> (2u * qq)) & 3u;
        int q = int(lo | (hb << 4)) - 32;
        float d = f16_at(blk + 208u);
        float s = d * float(i8_at(blk + 192u + 8u * h + l / 16u + 2u * qq));
        float y = s * float(q);
        return y;
    }
}

// Q4_K and Q6_K: the block's scale products, each computed once per block by one lane
// of the row's group and read by all 32 (audit §23). Group g's products start at 16g:
// Q4_K's d * sc_j at kd[16g + j] and dmin * m_j at km[8g + j], j < 8; Q6_K's
// d * sc_j at kd[16g + j], j < 16. Each is the same product of the same operands as
// `weight` computes per weight, so the bits are unchanged; shared memory rather than
// subgroup shuffles, so the result does not depend on the subgroup size.
shared float kd[ROWS * 16u];
shared float km[ROWS * 8u];

void main() {
    MATMUL_PROLOGUE
    // Uniform across the workgroup (as P_K is), so the barriers below stay uniform.
    if (P_N == 0u) {
        return;
    }
    uint t = gl_LocalInvocationID.x;
    uint lane = t % 32u;
    uint grp = t / 32u;
    uint r = ROW_BLOCK * ROWS + grp;
    uint tok = TOKEN;

    float a0 = 0.0;
    float a1 = 0.0;
    // Every invocation runs the whole loop, so every one reaches every barrier: a row
    // past `n` computes row n - 1 again and writes nothing.
    {
        uint row = P_W_BASE + min(r, P_N - 1u) * P_ROW_BYTES;
        uint x = P_X_BASE + tok * P_K;
        uint full = P_K / 64u * 64u;
        // The quantised formats decode a block at a time: the block's scales once, and
        // each byte once for every weight it holds. Each lane still gets the same
        // weights, from the same operations, in the same order as `weight` gives.
        uint c = 0u;
        if (DTYPE == DT_Q4_K) {
            // A chunk of 64 is one group g: lane l's two weights are the low and high
            // nibbles of the same byte, sub-blocks 2g and 2g + 1.
            // y = t - m as fma(t, 1, -m): the same single rounding, without the call
            // SPIRV-Cross makes for a NoContraction subtract. `opaque` keeps the product
            // rounded on its own.
#define K4G(G)                                                                     \
    {                                                                              \
        float d0 = kd[16u * grp + 2u * (G)];                                       \
        float m0 = km[8u * grp + 2u * (G)];                                        \
        float d1 = kd[16u * grp + 2u * (G) + 1u];                                  \
        float m1 = km[8u * grp + 2u * (G) + 1u];                                   \
        uint v = byte_at(blk + 16u + 32u * (G) + lane);                            \
        float y0 = fma(opaque(d0 * float(v & 0xFu)), 1.0, -m0);                    \
        float y1 = fma(opaque(d1 * float(v >> 4)), 1.0, -m1);                      \
        uint j = x + c + 64u * (G) + lane;                                         \
        a0 = fma(y0, xs[j], a0);                                                   \
        a1 = fma(y1, xs[j + 32u], a1);                                             \
    }
            for (; c + 256u <= full; c += 256u) {
                uint blk = row + 144u * (c / 256u);
                if (lane < 8u) {
                    // The header in four words: d and dmin, then the 12 scale bytes.
                    uint h0 = word_at(blk);
                    uvec3 sw = uvec3(word_at(blk + 4u), word_at(blk + 8u), word_at(blk + 12u));
                    uvec2 s = scale_min_k4w(lane, sw);
                    kd[16u * grp + lane] = f16_bits(h0 & 0xFFFFu) * float(s.x);
                    km[8u * grp + lane] = f16_bits(h0 >> 16) * float(s.y);
                }
                memoryBarrierShared();
                barrier();
                K4G(0u)
                K4G(1u)
                K4G(2u)
                K4G(3u)
                // No lane overwrites a product before every lane has read it.
                memoryBarrierShared();
                barrier();
            }
#undef K4G
        } else if (DTYPE == DT_Q6_K) {
            // A half of 128 is two chunks: lane l's four weights share qh[32h + l] and
            // take ql[64h + l] and ql[64h + l + 32], low nibbles in the first chunk and
            // high nibbles in the second; quarter qq has scale 8h + l / 16 + 2qq.
#define K6Q(H, QQ, LO, A)                                                          \
    {                                                                              \
        int q = int((LO) | (((hb >> (2u * (QQ))) & 3u) << 4)) - 32;                \
        float s = kd[16u * grp + 8u * (H) + lane / 16u + 2u * (QQ)];               \
        float y = s * float(q);                                                    \
        A = fma(y, xs[j + 32u * (QQ)], A);                                         \
    }
#define K6H(H)                                                                     \
    {                                                                              \
        uint la = byte_at(blk + 64u * (H) + lane);                                 \
        uint lb = byte_at(blk + 64u * (H) + lane + 32u);                           \
        uint hb = byte_at(blk + 128u + 32u * (H) + lane);                          \
        uint j = x + c + 128u * (H) + lane;                                        \
        K6Q(H, 0u, la & 0xFu, a0)                                                  \
        K6Q(H, 1u, lb & 0xFu, a1)                                                  \
        K6Q(H, 2u, la >> 4, a0)                                                    \
        K6Q(H, 3u, lb >> 4, a1)                                                    \
    }
            for (; c + 256u <= full; c += 256u) {
                uint blk = row + 210u * (c / 256u);
                if (lane < 16u) {
                    kd[16u * grp + lane] = f16_at(blk + 208u) * float(i8_at(blk + 192u + lane));
                }
                memoryBarrierShared();
                barrier();
                K6H(0u)
                K6H(1u)
                memoryBarrierShared();
                barrier();
            }
#undef K6H
#undef K6Q
        } else if (DTYPE == DT_MXFP4) {
            // A chunk of 64 is two blocks of 32: lane l takes element l of each, read
            // from the packed nibbles.
            for (; c < full; c += 64u) {
                uint b0 = row + 17u * (c / 32u);
                float y0 = mxfp4_at(b0, lane);
                float y1 = mxfp4_at(b0 + 17u, lane);
                a0 = fma(y0, xs[x + c + lane], a0);
                a1 = fma(y1, xs[x + c + lane + 32u], a1);
            }
        } else if (DTYPE == DT_Q8_0) {
            // A chunk of 64 is two blocks of 32: lane l takes weight l of each.
            for (; c < full; c += 64u) {
                uint b0 = row + 34u * (c / 32u);
                uint b1 = b0 + 34u;
                float d0 = f16_at(b0);
                float d1 = f16_at(b1);
                float q0 = float(i8_at(b0 + 2u + lane));
                float q1 = float(i8_at(b1 + 2u + lane));
                float y0 = q0 * d0;
                float y1 = q1 * d1;
                a0 = fma(y0, xs[x + c + lane], a0);
                a1 = fma(y1, xs[x + c + lane + 32u], a1);
            }
        }
        for (; c < full; c += 64u) {
            uint j0 = c + lane;
            uint j1 = j0 + 32u;
            a0 = fma(weight(row, j0), xs[x + j0], a0);
            a1 = fma(weight(row, j1), xs[x + j1], a1);
        }
        uint rem = P_K - full;
        if (lane < rem) {
            a0 = fma(weight(row, full + lane), xs[x + full + lane], a0);
        }
        if (lane + 32u < rem) {
            a1 = fma(weight(row, full + lane + 32u), xs[x + full + lane + 32u], a1);
        }
    }

    // w = 32.
    float s = a0 + a1;
    red[t] = s;
    memoryBarrierShared();
    barrier();
    // w = 16, 8, 4 within each group of 32.
    for (uint wd = 16u; wd >= 4u; wd /= 2u) {
        float v = 0.0;
        if (lane < wd) {
            v = red[t] + red[t + wd];
        }
        memoryBarrierShared();
        barrier();
        if (lane < wd) {
            red[t] = v;
        }
        memoryBarrierShared();
        barrier();
    }
    if (lane == 0u && r < P_N) {
        float b0 = red[t] + red[t + 1u];
        float b1 = red[t + 2u] + red[t + 3u];
        float y = b0 + b1;
        ys[P_Y_BASE + tok * P_N + r] = y;
    }
}
