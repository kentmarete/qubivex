// Exact attention for the harness (audit §19): `qubivex_arch::ops::attend` for one
// (token, query head) per workgroup, with the CPU's bits. Parallel only where the order
// cannot change a result:
// - scores: each position's q·k is its own sequential fma chain over d (dot_seq), times
//   the scale `1 / sqrt(d)`; positions in parallel;
// - the maximum: Rust's `f32::max` fold from -inf ignores NaNs and is exact, so a tree
//   gives the same value (a ±0 maximum gives the same `v - m` for every finite v);
// - exp(v - m) in parallel; the sum sequential from +0 in one invocation, position order;
//   one division per position in parallel;
// - values: each output dimension is its own sequential fma chain over positions, from
//   zero, in position order; dimensions in parallel.
// The includer defines SCORE(j) (read and write) and declares the buffers Q, KV, Out.
// Push constants: d | group << 16, kv_heads | heads << 16, p0 (the first token's
// position), v_base (V's word offset in KV). Grid: (heads, tokens).

const uint LOCAL = 128u;
shared float red[LOCAL];
shared float stot;

#include "exp.glsl"

float fmax_rust(float a, float b) {
    return isnan(a) ? b : (isnan(b) ? a : max(a, b));
}

void main() {
    uint d = p.dg & 0xFFFFu;
    uint group = p.dg >> 16;
    uint kvh = p.kh & 0xFFFFu;
    uint heads = p.kh >> 16;
    uint h = gl_WorkGroupID.x;
    uint t = gl_WorkGroupID.y;
    uint s = gl_NumWorkGroups.y;
    uint span = p.p0 + s;
    uint n = p.p0 + t + 1u;
    uint l = gl_LocalInvocationID.x;
    uint qb = (t * heads + h) * d;
    uint g = h / group;
    uint stride = kvh * d;
    SCORE_SETUP
    float scale = 1.0 / sqrt(float(d));

    for (uint j = l; j < n; j += LOCAL) {
        uint kb = j * stride + g * d;
        float acc = 0.0;
        for (uint i = 0u; i < d; i++) {
            acc = fma(q[qb + i], kv[kb + i], acc);
        }
        SCORE(j) = acc * scale;
    }
    SCORE_BARRIER
    float m = uintBitsToFloat(0xff800000u);
    for (uint j = l; j < n; j += LOCAL) {
        m = fmax_rust(m, SCORE(j));
    }
    red[l] = m;
    memoryBarrierShared();
    barrier();
    for (uint w = LOCAL / 2u; w > 0u; w /= 2u) {
        float v = 0.0;
        if (l < w) {
            v = fmax_rust(red[l], red[l + w]);
        }
        memoryBarrierShared();
        barrier();
        if (l < w) {
            red[l] = v;
        }
        memoryBarrierShared();
        barrier();
    }
    m = red[0];
    for (uint j = l; j < n; j += LOCAL) {
        SCORE(j) = qexp(SCORE(j) - m);
    }
    SCORE_BARRIER
    if (l == 0u) {
        float sum = 0.0;
        for (uint j = 0u; j < n; j++) {
            sum += SCORE(j);
        }
        stot = sum;
    }
    SCORE_BARRIER
    float sum = stot;
    for (uint j = l; j < n; j += LOCAL) {
        SCORE(j) = SCORE(j) / sum;
    }
    SCORE_BARRIER
    for (uint i = l; i < d; i += LOCAL) {
        uint vb = p.v_base + g * d + i;
        float o = 0.0;
        for (uint j = 0u; j < n; j++) {
            o = fma(SCORE(j), kv[vb + j * stride], o);
        }
        out_[qb + i] = o;
    }
}
