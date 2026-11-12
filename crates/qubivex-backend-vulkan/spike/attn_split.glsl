// Exact attention in three dispatches for the harness (audit §19), the same operations
// and order as attn_body.glsl, with keys stored dimension-major so the scores read
// coalesced:
//   bindings: 0 q [token][head][d]; 1 KV: K^T [kv_head][d][span] at 0, V [pos][kv_head][d]
//   at v_base; 2 out [token][head][d]; 3 scores [token][head][span].
//   push: d | group << 16, kv_heads | heads << 16, p0, v_base; span = p0 + tokens.
layout(std430, set = 0, binding = 0) readonly buffer Q { float q[]; };
layout(std430, set = 0, binding = 1) readonly buffer KV { float kv[]; };
layout(std430, set = 0, binding = 2) writeonly buffer Out { float out_[]; };
layout(std430, set = 0, binding = 3) buffer Scores { float sc[]; };
layout(push_constant) uniform P { uint dg; uint kh; uint p0; uint v_base; } p;
#define D (p.dg & 0xFFFFu)
#define GROUP (p.dg >> 16)
#define KVH (p.kh & 0xFFFFu)
#define HEADS (p.kh >> 16)
