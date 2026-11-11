// Exhaustive SwiGLU parity (audit §15): every f32 bit pattern of g, against a C port of
// `qubivex_arch::ops::swiglu` + `qubivex_compute::math::exp`.
//   pass 1: u = 1.0 (isolates silu: s * 1 is exact for normal s)
//   pass 2: u = a pseudo-random bit pattern per element (every class: zero, subnormal,
//           normal, inf, NaN), the same generator as the Rust hasher.
// Each GPU variant is a kernel file: MoltenVK's MSL (bound through an argument buffer, as
// MoltenVK binds it, push constant at buffer 8) or the native Metal backend's control
// kernel (buffers 0, 1, params at 2). Differences are classified; the C reference's
// outputs are hashed per pass and compared with the Rust implementation (rusthash).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <float.h>
#include <stdatomic.h>

// ---- C port of the CPU reference (compiled with -ffp-contract=off). ----
static inline float bits_f(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
static inline uint32_t f_bits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
static inline float pow2(int k) { return bits_f((uint32_t)(k + 127) << 23); }
static float ref_exp(float x) {
    if (isnan(x)) return x;
    if (x > bits_f(0x42b17218u)) return INFINITY;
    if (x < bits_f(0xc2aeac50u)) return 0.0f;
    float kf = (x * bits_f(0x3fb8aa3bu) + 12582912.0f) - 12582912.0f;
    float r = fmaf(-kf, bits_f(0x3f317200u), x);
    r = fmaf(-kf, bits_f(0x35bfbe8eu), r);
    float p = bits_f(0x39506967u);
    p = fmaf(p, r, bits_f(0x3ab743ceu));
    p = fmaf(p, r, bits_f(0x3c088908u));
    p = fmaf(p, r, bits_f(0x3d2aa9c1u));
    p = fmaf(p, r, bits_f(0x3e2aaaaau));
    p = fmaf(p, r, bits_f(0x3f000000u));
    float y = fmaf(p, r * r, r) + 1.0f;
    int k = (int)kf;
    int k1 = k >> 1;
    float v = (y * pow2(k1)) * pow2(k - k1);
    return v < FLT_MIN ? 0.0f : v;
}
static inline float ref_silu(float a) { return a / (1.0f + ref_exp(-a)); }

static inline uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
static inline uint32_t u_bits(int pass, uint64_t i) { return pass == 1 ? 0x3f800000u : (uint32_t)mix64(i ^ 0x5851f42d4c957f2dull); }

// ---- Classification. ----
enum { C_SUB_IN, C_SUB_MID, C_SUB_OUT, C_TINY, C_NAN_PAYLOAD, C_ZERO_SIGN, C_OTHER, NCAT };
static const char *catName[] = {"input g or u subnormal", "intermediate silu(g) subnormal", "reference output subnormal",
                                "exact silu(g) subnormal, rounds up to FLT_MIN",
                                "NaN vs NaN, other payload", "+0 vs -0", "OTHER (unexplained)"};
static inline int sub(float x) { return x != 0 && fabsf(x) < FLT_MIN; }

typedef struct {
    NSString *name; id<MTLComputePipelineState> pso; BOOL mvk; id<MTLArgumentEncoder> ae; id<MTLBuffer> args;
    id<MTLBuffer> out; _Atomic uint64_t count[NCAT]; uint32_t ex[NCAT][3][3]; _Atomic int nex[NCAT]; double gpuSec;
} Variant;

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "usage: check KERNEL.metal... (env CHUNK_LOG2, PASSES)\n"); return 2; }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        int clog = getenv("CHUNK_LOG2") ? atoi(getenv("CHUNK_LOG2")) : 25;
        uint64_t chunk = 1ull << clog, total = 1ull << 32;
        const char *passes = getenv("PASSES") ? getenv("PASSES") : "12";
        id<MTLBuffer> gin = [dev newBufferWithLength:chunk * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> uin = [dev newBufferWithLength:chunk * 4 options:MTLResourceStorageModeShared];
        int nv = argc - 1;
        Variant *vs = calloc(nv, sizeof(Variant));
        for (int v = 0; v < nv; v++) {
            NSError *err = nil;
            NSString *src = [NSString stringWithContentsOfFile:@(argv[v + 1]) encoding:NSUTF8StringEncoding error:&err];
            MTLCompileOptions *o = [MTLCompileOptions new];
            o.mathMode = MTLMathModeSafe;  // as MoltenVK with fast math off
            o.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
            id<MTLLibrary> lib = [dev newLibraryWithSource:src options:o error:&err];
            if (!lib) { printf("%s: %s\n", argv[v + 1], err.description.UTF8String); return 1; }
            vs[v].mvk = [src containsString:@"spvDescriptorSet0"];
            id<MTLFunction> fn = [lib newFunctionWithName:vs[v].mvk ? @"main0" : @"swiglu"];
            vs[v].pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            vs[v].name = [@(argv[v + 1]) lastPathComponent];
            vs[v].out = [dev newBufferWithLength:chunk * 4 options:MTLResourceStorageModeShared];
            if (vs[v].mvk) {
                vs[v].ae = [fn newArgumentEncoderWithBufferIndex:0];
                vs[v].args = [dev newBufferWithLength:vs[v].ae.encodedLength options:MTLResourceStorageModeShared];
                [vs[v].ae setArgumentBuffer:vs[v].args offset:0];
                [vs[v].ae setBuffer:vs[v].out offset:0 atIndex:1];  // g (in/out)
                [vs[v].ae setBuffer:uin offset:0 atIndex:2];        // u
            }
        }
        for (const char *pc = passes; *pc; pc++) {
            int pass = *pc - '0';
            uint64_t hash = 0;
            for (int v = 0; v < nv; v++) { for (int c = 0; c < NCAT; c++) { vs[v].count[c] = 0; vs[v].nex[c] = 0; } vs[v].gpuSec = 0; }
            uint32_t *g = gin.contents, *u = uin.contents;
            for (uint64_t base = 0; base < total; base += chunk) {
                dispatch_apply(64, DISPATCH_APPLY_AUTO, ^(size_t b) {
                    for (uint64_t i = b * chunk / 64; i < (b + 1) * chunk / 64; i++) { g[i] = (uint32_t)(base + i); u[i] = u_bits(pass, base + i); }
                });
                for (int v = 0; v < nv; v++) {
                    memcpy(vs[v].out.contents, g, chunk * 4);
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                    [e setComputePipelineState:vs[v].pso];
                    uint32_t n = (uint32_t)chunk;
                    if (vs[v].mvk) {
                        [e setBuffer:vs[v].args offset:0 atIndex:0];
                        [e useResource:vs[v].out usage:MTLResourceUsageRead | MTLResourceUsageWrite];
                        [e useResource:uin usage:MTLResourceUsageRead];
                        [e setBytes:&n length:4 atIndex:8];
                    } else {
                        struct { uint32_t n; float f; } p = {n, 0};
                        [e setBuffer:vs[v].out offset:0 atIndex:0];
                        [e setBuffer:uin offset:0 atIndex:1];
                        [e setBytes:&p length:8 atIndex:2];
                    }
                    [e dispatchThreads:MTLSizeMake(chunk, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    vs[v].gpuSec += cb.GPUEndTime - cb.GPUStartTime;
                }
                __block _Atomic uint64_t h = 0;
                dispatch_apply(64, DISPATCH_APPLY_AUTO, ^(size_t b) {
                    uint64_t lh = 0;
                    for (uint64_t i = b * chunk / 64; i < (b + 1) * chunk / 64; i++) {
                        float a = bits_f(g[i]), m = bits_f(u[i]);
                        float s = ref_silu(a), want = s * m;
                        uint32_t wb = f_bits(want);
                        lh ^= mix64(((base + i) << 32) | wb);
                        for (int v = 0; v < nv; v++) {
                            uint32_t gb = ((uint32_t *)vs[v].out.contents)[i];
                            if (gb == wb) continue;
                            float got = bits_f(gb);
                            int c = (sub(a) || sub(m)) ? C_SUB_IN
                                  : sub(s) ? C_SUB_MID
                                  : sub(want) ? C_SUB_OUT
                                  : (got == 0 && fabsf(s) == FLT_MIN) ? C_TINY
                                  : (isnan(got) && isnan(want)) ? C_NAN_PAYLOAD
                                  : (got == 0 && want == 0) ? C_ZERO_SIGN : C_OTHER;
                            atomic_fetch_add(&vs[v].count[c], 1);
                            int k = atomic_fetch_add(&vs[v].nex[c], 1);
                            if (k < 3) { vs[v].ex[c][k][0] = g[i]; vs[v].ex[c][k][1] = u[i]; vs[v].ex[c][k][2] = gb; }
                        }
                    }
                    atomic_fetch_xor(&h, lh);
                });
                hash ^= h;
            }
            printf("pass %d (%s): C reference hash %016llx\n", pass, pass == 1 ? "u = 1" : "u = mix64 bits", (unsigned long long)hash);
            for (int v = 0; v < nv; v++) {
                uint64_t diff = 0; for (int c = 0; c < NCAT; c++) diff += vs[v].count[c];
                printf("  %-22s differs in %llu of 4294967296 (GPU %.2f ns/element)\n", vs[v].name.UTF8String, (unsigned long long)diff, vs[v].gpuSec * 1e9 / (double)total);
                for (int c = 0; c < NCAT; c++) {
                    if (!vs[v].count[c]) continue;
                    printf("    %-34s %12llu", catName[c], (unsigned long long)vs[v].count[c]);
                    int k = vs[v].nex[c] < 3 ? vs[v].nex[c] : 3;
                    for (int j = 0; j < k; j++) {
                        uint32_t gg = vs[v].ex[c][j][0], uu = vs[v].ex[c][j][1];
                        float a = bits_f(gg), m = bits_f(uu);
                        printf("  | g=%08x (%g) u=%08x got %08x want %08x", gg, a, uu, vs[v].ex[c][j][2], f_bits(ref_silu(a) * m));
                    }
                    printf("\n");
                }
            }
            fflush(stdout);
        }
    }
    return 0;
}
