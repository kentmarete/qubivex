/* ADR-002 Colibri baseline: Colibri 0a5504a1 kernels on the spike's exact inputs.
 *
 * Inputs come from `adr-002-tensor-engine dump <dir>` (same weight bytes, same
 * activations, f64 references). Timing mirrors the spike: median of >= 15 iterations
 * or the budget, hot = one matrix, cold = rotating over >= 512 MiB of copies.
 *
 *   colibri_bench <dumpdir>         markdown tables
 *   colibri_bench <dumpdir> det     output bit hashes per case
 *
 * Arms are Colibri's own functions, included from its headers, or copied verbatim
 * where they are static in an engine file (olmoe.c matmul_q / matmul_q_rows,
 * quant.h matmul_q NEON, bench_attn_f32.c ATTN_CORE).
 *
 * The copied functions are from JustVugg/colibri @ 0a5504a1, Copyright the Colibri
 * authors, Apache License 2.0. Spike tooling only: not product code.
 *
 * Build (macOS arm64, from a Colibri checkout at 0a5504a1; brew install libomp):
 *   O=$(brew --prefix libomp)
 *   clang -O3 -mcpu=native -Xpreprocessor -fopenmp -I$O/include -I <colibri>/c \
 *         colibri_bench.c -o colibri_bench -L$O/lib -lomp -lm
 * Run:
 *   adr-002-tensor-engine dump <dir>
 *   OMP_NUM_THREADS=11 ./colibri_bench <dir> [det|attn] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <arm_neon.h>
#include "matmul_f32.h"
#include "matmul_f16.h"
#include "ggml_quants.h"

/* ---- olmoe.c @0a5504a1 matmul_q, IDOT off (its default): lossless ---- */
static void olmoe_matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        #pragma omp simd reduction(+:acc)
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
}
/* olmoe.c matmul_q_rows (prefill, moe_by_expert), IDOT off */
static void olmoe_matmul_q_rows(float *const *y, const float *const *x, int n,
                                const int8_t *q, const float *scale, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        for (int t = 0; t < n; t++) {
            const float *xt = x[t];
            float acc = 0.f;
            #pragma omp simd reduction(+:acc)
            for (int i = 0; i < I; i++) acc += xt[i] * (float)w[i];
            y[t][o] = acc * scale[o];
        }
    }
}
/* quant.h @0a5504a1 matmul_q, NEON branch (f32 activations): lossless */
static void quant_matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const int8_t *w=q+(int64_t)o*I; float sc=scale[o];
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+8<=I;i+=8){ int16x8_t w16=vmovl_s8(vld1_s8(w+i));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),   vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)))); }
            a=vaddvq_f32(vaddq_f32(ac0,ac1));
            for(;i<I;i++) a+=xs[i]*(float)w[i]; y[(int64_t)s*O+o]=a*sc; } }
}

/* ---- attention, bench_attn_f32.c (qwen36/olmoe decode loop) ---- */
typedef struct { int H, KV, hd, T; const float *q, *K, *V; float *ctx, *sc; } Attn;
static void softmax_row(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}
#define ATTN_CORE(name, DOT)                                                          \
static void name(const Attn *a) {                                                     \
    int q_per_kv = a->H / a->KV, hd = a->hd, qpos = a->T - 1;                         \
    float scale = 1.f / sqrtf((float)hd);                                             \
    _Pragma("omp parallel for schedule(static)")                                      \
    for (int hh = 0; hh < a->H; hh++) {                                               \
        int kvh = hh / q_per_kv, tid = omp_get_thread_num();                          \
        const float *qv = a->q + (int64_t)hh * hd;                                    \
        float *sc = a->sc + (int64_t)tid * a->T;                                      \
        for (int t = 0; t <= qpos; t++) {                                             \
            const float *kv = a->K + ((int64_t)kvh * a->T + t) * hd;                  \
            float acc; DOT;                                                           \
            sc[t] = acc * scale;                                                      \
        }                                                                             \
        softmax_row(sc, qpos + 1);                                                    \
        float *cx = a->ctx + (int64_t)hh * hd;                                        \
        for (int dd = 0; dd < hd; dd++) cx[dd] = 0;                                   \
        for (int t = 0; t <= qpos; t++) {                                             \
            const float *vrow = a->V + ((int64_t)kvh * a->T + t) * hd;                \
            float w = sc[t]; for (int dd = 0; dd < hd; dd++) cx[dd] += w * vrow[dd];  \
        }                                                                             \
    }                                                                                 \
}
ATTN_CORE(attn_scalar, acc = 0; for (int dd = 0; dd < hd; dd++) acc += qv[dd] * kv[dd])
ATTN_CORE(attn_lanes, acc = dot_f32_lanes(qv, kv, hd))

/* ---- harness ---- */
static const char *DIR;
static void *load(const char *name, size_t *len) {
    char p[1024]; snprintf(p, sizeof p, "%s/%s", DIR, name);
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc((size_t)n); if (fread(b, 1, (size_t)n, f) != (size_t)n) { perror(p); exit(1); }
    fclose(f); if (len) *len = (size_t)n; return b;
}
static double now_ns(void){ return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static int cmpd(const void *a, const void *b){ double x=*(const double*)a, y=*(const double*)b; return (x>y)-(x<y); }

typedef void (*run_fn)(void *ctx, int i);
static double time_ns(run_fn f, void *ctx, int warm, double budget_ns) {
    for (int i = 0; i < warm; i++) f(ctx, i);
    static double v[2000]; int n = 0, i = warm; double st = now_ns();
    while ((now_ns() - st < budget_ns || n < 15) && n < 2000) { double t = now_ns(); f(ctx, i++); v[n++] = now_ns() - t; }
    qsort(v, n, sizeof *v, cmpd); return v[n/2];
}
static double rel_err(const float *y, const double *r, int n) {
    double sc = 0, e = 0;
    for (int i = 0; i < n; i++) { sc = fmax(sc, fabs(r[i])); e = fmax(e, fabs((double)y[i] - r[i])); }
    return sc > 0 ? e / sc : e;
}
static uint64_t bits_hash(const float *y, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { uint32_t u; memcpy(&u, &y[i], 4);
        for (int b = 0; b < 4; b++) { h ^= (u >> (8*b)) & 0xff; h *= 0x100000001b3ull; } }
    return h;
}
static uint16_t f32_to_bf16_rne(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffff) > 0x7f800000) return (uint16_t)((u >> 16) | 0x40);
    return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

enum { A_F32, A_BF16, A_I8_OLMOE, A_I8_QUANT, A_Q8_EXACT, A_Q8_INT, A_Q4_EXACT, A_Q4_INT, A_N };
static const char *ARM_NAME[A_N] = {
    "f32 | matmul_f32.h `matmul` (lanes)",
    "bf16 | matmul_f16.h `matmul_bf16w`",
    "int8-row | olmoe.c `matmul_q` / `_rows` (IDOT off, default)",
    "int8-row | quant.h `matmul_q` NEON (f32 act)",
    "q8_0 | `gq_matmul_f32` (exact, = GQ_EXACT=1)",
    "q8_0 | `gq_matmul` (integer dot, default)",
    "q4_k | `gq_matmul_f32` (exact, = GQ_EXACT=1)",
    "q4_k | `gq_matmul` (integer dot, default)",
};
static const char *ARM_REF[A_N] = { "f32", "bf16", "int8-row", "int8-row", "q8_0", "q8_0", "q4_k", "q4_k" };

typedef struct {
    int arm, S, I, O; const float *x; float *y;
    void **w; float **sc; int pool; int cold;
    float **yrows; const float **xrows;
} Ctx;

static void run_arm(void *vc, int it) {
    Ctx *c = vc; int k = c->cold ? it % c->pool : 0;
    void *w = c->w[k]; float *sc = c->sc ? c->sc[k] : NULL;
    switch (c->arm) {
    case A_F32:  matmul(c->y, c->x, w, c->S, c->I, c->O); break;
    case A_BF16: matmul_bf16w(c->y, c->x, w, c->S, c->I, c->O); break;
    case A_I8_OLMOE:
        if (c->S == 1) olmoe_matmul_q(c->y, c->x, w, sc, c->I, c->O);
        else olmoe_matmul_q_rows(c->yrows, c->xrows, c->S, w, sc, c->I, c->O);
        break;
    case A_I8_QUANT: quant_matmul_q(c->y, c->x, w, sc, c->S, c->I, c->O); break;
    case A_Q8_EXACT: gq_matmul_f32(c->y, c->x, w, GQ_Q8_0, c->S, c->I, c->O); break;
    case A_Q8_INT:   gq_matmul(c->y, c->x, w, GQ_Q8_0, c->S, c->I, c->O); break;
    case A_Q4_EXACT: gq_matmul_f32(c->y, c->x, w, GQ_Q4_K, c->S, c->I, c->O); break;
    case A_Q4_INT:   gq_matmul(c->y, c->x, w, GQ_Q4_K, c->S, c->I, c->O); break;
    }
}
static void run_attn_scalar(void *a, int i) { (void)i; attn_scalar(a); }
static void run_attn_lanes(void *a, int i) { (void)i; attn_lanes(a); }

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <dumpdir> [det]\n", argv[0]); return 2; }
    DIR = argv[1]; int det = argc > 2 && !strcmp(argv[2], "det"); int attn_only = argc > 2 && !strcmp(argv[2], "attn");
    const int SH[2][2] = { {1024, 2048}, {2048, 1024} };   /* (n=O, k=I) as in the spike */
    const char *SHL[2] = { "gate/up 2048→1024", "down 1024→2048" };
    const int SEQ[2] = { 1, 128 };
    const size_t COLD = (size_t)512 << 20;
    if (!det) {
        printf("# Colibri 0a5504a1 baseline on the ADR-002 spike inputs\n\n- OpenMP threads: %d\n\n", omp_get_max_threads());
        printf("| shape | S | dtype | Colibri kernel | hot µs | cold µs | cold GB/s | hot GFLOP/s | rel err |\n|---|---|---|---|---|---|---|---|---|\n");
    }
    for (int si = 0; si < (attn_only ? 0 : 2); si++) {
        int O = SH[si][0], I = SH[si][1]; char nm[256];
        #define LOADF(name) (snprintf(nm, sizeof nm, "n%d_k%d_%s", O, I, name), load(nm, &len))
        size_t len;
        float *wf = LOADF("w_f32");
        int8_t *wi8 = LOADF("w_i8"); float *si8 = LOADF("s_i8");
        uint8_t *wq8 = LOADF("w_q8_0"); size_t lq8 = len;
        uint8_t *wq4 = LOADF("w_q4_k"); size_t lq4 = len;
        if ((int64_t)lq8 != gq_row_bytes(GQ_Q8_0, I) * O || (int64_t)lq4 != gq_row_bytes(GQ_Q4_K, I) * O) { fprintf(stderr, "row bytes mismatch\n"); return 1; }
        uint16_t *wbf = malloc((size_t)O * I * 2); for (size_t i = 0; i < (size_t)O * I; i++) wbf[i] = f32_to_bf16_rne(wf[i]);
        for (int qi = 0; qi < 2; qi++) {
            int S = SEQ[qi];
            snprintf(nm, sizeof nm, "n%d_k%d_x_s%d", O, I, S); float *x = load(nm, NULL);
            int ntok = S == 1 ? 1 : 3, toks[3] = { 0, 37, S - 1 };
            float *y = calloc((size_t)S * O, 4);
            float **yrows = malloc(S * sizeof *yrows); const float **xrows = malloc(S * sizeof *xrows);
            for (int t = 0; t < S; t++) { yrows[t] = y + (size_t)t * O; xrows[t] = x + (size_t)t * I; }
            for (int a = 0; a < A_N; a++) {
                void *base; size_t wb; float *sc = NULL;
                switch (a) {
                case A_F32: base = wf; wb = (size_t)O*I*4; break;
                case A_BF16: base = wbf; wb = (size_t)O*I*2; break;
                case A_I8_OLMOE: case A_I8_QUANT: base = wi8; wb = (size_t)O*I; sc = si8; break;
                case A_Q8_EXACT: case A_Q8_INT: base = wq8; wb = lq8; break;
                default: base = wq4; wb = lq4; break;
                }
                size_t wtot = wb + (sc ? (size_t)O * 4 : 0);
                int pool = det ? 1 : (int)(COLD / wtot > 2 ? COLD / wtot : 2);
                void **ws = malloc(pool * sizeof *ws); float **ss = sc ? malloc(pool * sizeof *ss) : NULL;
                ws[0] = base; if (sc) ss[0] = sc;
                for (int p = 1; p < pool; p++) { ws[p] = malloc(wb); memcpy(ws[p], base, wb);
                    if (sc) { ss[p] = malloc((size_t)O*4); memcpy(ss[p], sc, (size_t)O*4); } }
                Ctx c = { a, S, I, O, x, y, ws, ss, pool, 0, yrows, xrows };
                if (det) {
                    uint64_t h[2]; for (int r = 0; r < 2; r++) { run_arm(&c, 0); h[r] = bits_hash(y, (size_t)S*O); }
                    printf("%s S=%d %s\t%016llx\n", SHL[si], S, ARM_NAME[a], (unsigned long long)(h[0] == h[1] ? h[0] : 0));
                } else {
                    double budget = S == 1 ? 400e6 : 800e6;
                    double hot = time_ns(run_arm, &c, 5, budget);
                    c.cold = 1; double cold = time_ns(run_arm, &c, pool, budget); c.cold = 0;
                    run_arm(&c, 0);
                    snprintf(nm, sizeof nm, "n%d_k%d_ref_%s_s%d", O, I, ARM_REF[a], S); double *ref = load(nm, NULL);
                    double err = 0; for (int t = 0; t < ntok; t++) err = fmax(err, rel_err(y + (size_t)toks[t]*O, ref + (size_t)t*O, O));
                    free(ref);
                    printf("| %s | %d | %s | %.1f | %.1f | %.1f | %.1f | %.1e |\n", SHL[si], S, ARM_NAME[a],
                           hot/1e3, cold/1e3, wtot/cold, 2.0*O*I*S/hot, err);
                    fflush(stdout);
                }
                for (int p = 1; p < pool; p++) { free(ws[p]); if (sc) free(ss[p]); }
                free(ws); free(ss);
            }
            free(x); free(y); free(yrows); free(xrows);
        }
        free(wf); free(wi8); free(si8); free(wq8); free(wq4); free(wbf);
    }
    /* attention */
    if (!det) printf("\n## Attention decode (16 heads × 128, f32 KV)\n\n| context | Colibri kernel | µs | KV GB/s | rel err |\n|---|---|---|---|---|\n");
    for (int ti = 0; ti < 2; ti++) {
        int T = ti ? 4096 : 1024; char nm[128];
        snprintf(nm, sizeof nm, "attn_t%d_q", T); float *q = load(nm, NULL);
        snprintf(nm, sizeof nm, "attn_t%d_k", T); float *K = load(nm, NULL);
        snprintf(nm, sizeof nm, "attn_t%d_v", T); float *V = load(nm, NULL);
        snprintf(nm, sizeof nm, "attn_t%d_ref", T); double *ref = load(nm, NULL);
        float *ctx = calloc(16 * 128, 4), *scr = calloc((size_t)omp_get_max_threads() * T, 4);
        Attn at = { 16, 16, 128, T, q, K, V, ctx, scr };
        run_fn fs[2] = { run_attn_scalar, run_attn_lanes };
        const char *ns[2] = { "bench_attn_f32 scalar (olmoe/qwen36 as shipped)", "bench_attn_f32 lanes (matmul_f32.h dot)" };
        for (int a = 0; a < 2; a++) {
            if (det) { uint64_t h[2]; for (int r = 0; r < 2; r++) { fs[a](&at, 0); h[r] = bits_hash(ctx, 16*128); }
                printf("attn ctx=%d %s\t%016llx\n", T, ns[a], (unsigned long long)(h[0] == h[1] ? h[0] : 0)); continue; }
            double t = time_ns(fs[a], &at, 10, 400e6); fs[a](&at, 0);
            printf("| %d | %s | %.1f | %.1f | %.1e |\n", T, ns[a], t/1e3, 2.0*16*T*128*4/t, rel_err(ctx, ref, 16*128));
        }
        free(q); free(K); free(V); free(ref); free(ctx); free(scr);
    }
    return 0;
}
