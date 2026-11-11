// Expert weight binding (audit §15, check 2), harness only.
//
// A layer's pool of 64 resident experts (OLMoE), each call routed to `e` of them at
// random (a new set every call, as routing does), in these shapes:
//   SA        Step 1a: a thread per expert (at most 5 in flight), unfused dispatches,
//             gate+up then down, each its own command buffer, the expert's own buffer.
//   SF-<b>    fused gate/up, then fused down: two command buffers.
//   SE-<b>    the expert block in one command buffer: fused gate/up, SwiGLU (MoltenVK's
//             MSL of swiglu/swiglu.comp built -O0), fused down reading SwiGLU's output.
//             Three encoders, as MoltenVK splits at a pipeline barrier with residency sets.
// with <b> the weight binding: arena (one buffer, slot offsets), array (a pointer array
// indexed per matrix), bda (the pointer in the matrix's table entry).
//
// Reports per call: GPU time (Metal's clock), wall, and host encoding time. `check` mode
// compares every SF/SE output across bindings and with SA bit for bit, and dumps SE for
// cpucheck's `block` mode (CPU matmul + ops::swiglu + matmul).
//
// Usage: binding KERNEL_DIR [bench|check] [OUT_DIR]; KERNEL_DIR holds kernel.metal,
// fused-{arena,array,bda}.metal and swiglu/swiglu-O0.metal.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#include <stdatomic.h>

enum { H = 2048, F = 1024, POOL = 64, MAXE = 8, MAXT = 32 };
static int rb(int k) { return k / 256 * 144; }
static const int EB = 2 * F * (H / 256 * 144) + H * (F / 256 * 144);  // expert bytes
static int matOff(int m) { return m == 0 ? 0 : m == 1 ? F * rb(H) : 2 * F * rb(H); }
static int matN(int m) { return m == 2 ? H : F; }
static int matK(int m) { return m == 2 ? F : H; }
// Activations: x rows at 0; each routed slot's down input (SF) at XD + slot * MAXT * F.
static const int XD = MAXT * H;
// Outputs, blocked: every gate, then every up, then every down.
static const int UB = MAXE * MAXT * F, DB = 2 * MAXE * MAXT * F;

typedef struct { uint32_t n, k, rowBytes, wBase, xBase, yBase; } Params;
typedef struct { Params p; uint32_t first, pad, wi, pad2; uint64_t w; } Entry;  // 48 bytes, as QvxFused
// BDA_ALL: the bda kernel, encoded as MoltenVK encodes a pipeline that uses buffer device
// addresses: useResource on every GPU-addressable buffer of the device per encoder
// (MVKDevice::encodeGPUAddressableBuffers), here the pool plus stand-ins for 16 layers' experts.
typedef enum { ARENA, ARRAY, BDA, BDA_ALL } Binding;
static const char *bname[] = {"arena", "array", "bda", "bda-all"};
enum { NALL = 16 * POOL };
static id<MTLResource> allBufs[NALL];

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;
static id<MTLComputePipelineState> pso, psoF[3], psoSw;
static id<MTLArgumentEncoder> swAE;
static id<MTLBuffer> arena, per[POOL], xbuf, ybuf;
static uint64_t perAddr[POOL];
static int reps = 200;

// One matrix of the call: routed slot `i` (0..e-1), pool expert `x`, matrix `m`.
typedef struct { int i, x, m; } Mat;

static void encodeFused(id<MTLComputeCommandEncoder> enc, Binding b, const Mat *ms, int n, int s,
                        id<MTLBuffer> xb, const uint32_t *xBase, const uint32_t *yBase) {
    Entry tab[2 * MAXE]; uint32_t first = 0;
    for (int j = 0; j < n; j++) {
        const Mat *m = &ms[j];
        uint32_t wb = b == ARENA ? (uint32_t)(m->x * EB + matOff(m->m)) : (uint32_t)matOff(m->m);
        tab[j] = (Entry){{matN(m->m), matK(m->m), rb(matK(m->m)), wb, xBase[j], yBase[j]}, first, 0, (uint32_t)m->x, 0,
                         b >= BDA ? perAddr[m->x] : 0};
        first += (matN(m->m) + 7) / 8;
    }
    tab[0].pad = (matN(ms[0].m) + 7) / 8;  // O(1) lookup: every matrix of a stage has this many row blocks
    uint32_t count = n;
    [enc setComputePipelineState:psoF[b == BDA_ALL ? BDA : b]];
    if (b == ARENA) [enc setBuffer:arena offset:0 atIndex:0];
    else if (b == BDA_ALL) [enc useResources:allBufs count:NALL usage:MTLResourceUsageRead];
    else for (int j = 0; j < n; j++) if (j == 0 || ms[j].x != ms[j - 1].x) [enc useResource:per[ms[j].x] usage:MTLResourceUsageRead];
    if (b == ARRAY) [enc setBytes:perAddr length:sizeof perAddr atIndex:10];  // every resident expert of the layer
    [enc setBuffer:xb offset:0 atIndex:1];
    [enc setBuffer:ybuf offset:0 atIndex:2];
    [enc setBytes:tab length:sizeof(Entry) * n atIndex:8];
    [enc setBytes:&count length:4 atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake(first, s, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void encodeUnfused(id<MTLComputeCommandEncoder> enc, int x, int m, int s, uint32_t xBase, uint32_t yBase) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:per[x] offset:matOff(m) atIndex:0];  // the expert's buffer at the matrix (a descriptor offset)
    [enc setBuffer:xbuf offset:0 atIndex:1];
    [enc setBuffer:ybuf offset:0 atIndex:2];
    Params p = {matN(m), matK(m), rb(matK(m)), 0, xBase, yBase};
    [enc setBytes:&p length:sizeof p atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake((matN(m) + 7) / 8, s, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// SwiGLU over every routed slot's gate (in place) and up: one dispatch, n = e * s * F.
static void encodeSwiglu(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> args, int e, int s) {
    [swAE setArgumentBuffer:args offset:0];
    [swAE setBuffer:ybuf offset:0 atIndex:1];
    [swAE setBuffer:ybuf offset:(NSUInteger)UB * 4 atIndex:2];
    uint32_t n = (uint32_t)(e * s * F);
    [enc setComputePipelineState:psoSw];
    [enc setBuffer:args offset:0 atIndex:0];
    [enc useResource:ybuf usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [enc setBytes:&n length:4 atIndex:8];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

typedef enum { SA, SF, SE } Shape;
typedef struct { double gpu, encode; } Cost;

static uint64_t rng = 0x243F6A8885A308D3ull;
static uint64_t nextr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static void route(int e, int *sel) {
    for (int i = 0; i < e; i++) {
        int x, dup;
        do { x = (int)(nextr() % POOL); dup = 0; for (int j = 0; j < i; j++) dup |= sel[j] == x; } while (dup);
        sel[i] = x;
    }
}

static Cost run(Shape sh, Binding b, int e, int s, const int *sel, id<MTLBuffer> swArgs) {
    __block _Atomic uint64_t gpuNs = 0, encNs = 0;
    if (sh == SA) {
        for (int wave = 0; wave < e; wave += 5) {
            int n = e - wave < 5 ? e - wave : 5;
            dispatch_apply(n, DISPATCH_APPLY_AUTO, ^(size_t k) {
                int i = wave + (int)k, x = sel[i];
                double t0 = CACurrentMediaTime();
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                encodeUnfused(enc, x, 0, s, 0, i * s * F);
                encodeUnfused(enc, x, 1, s, 0, UB + i * s * F);
                [enc endEncoding];
                atomic_fetch_add(&encNs, (uint64_t)((CACurrentMediaTime() - t0) * 1e9));
                [cb commit]; [cb waitUntilCompleted];
                atomic_fetch_add(&gpuNs, (uint64_t)((cb.GPUEndTime - cb.GPUStartTime) * 1e9));
                t0 = CACurrentMediaTime();
                id<MTLCommandBuffer> cb2 = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc2 = [cb2 computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                encodeUnfused(enc2, x, 2, s, XD + i * MAXT * F, DB + i * s * H);
                [enc2 endEncoding];
                atomic_fetch_add(&encNs, (uint64_t)((CACurrentMediaTime() - t0) * 1e9));
                [cb2 commit]; [cb2 waitUntilCompleted];
                atomic_fetch_add(&gpuNs, (uint64_t)((cb2.GPUEndTime - cb2.GPUStartTime) * 1e9));
            });
        }
    } else {
        Mat gu[2 * MAXE], dn[MAXE];
        uint32_t guX[2 * MAXE], guY[2 * MAXE], dnX[MAXE], dnY[MAXE];
        for (int i = 0; i < e; i++) {
            gu[2 * i] = (Mat){i, sel[i], 0}; guX[2 * i] = 0; guY[2 * i] = i * s * F;
            gu[2 * i + 1] = (Mat){i, sel[i], 1}; guX[2 * i + 1] = 0; guY[2 * i + 1] = UB + i * s * F;
            dn[i] = (Mat){i, sel[i], 2}; dnY[i] = DB + i * s * H;
            dnX[i] = sh == SE ? (uint32_t)(i * s * F) : (uint32_t)(XD + i * MAXT * F);
        }
        if (sh == SF) {
            for (int st = 0; st < 2; st++) {
                double t0 = CACurrentMediaTime();
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                if (st == 0) encodeFused(enc, b, gu, 2 * e, s, xbuf, guX, guY);
                else encodeFused(enc, b, dn, e, s, xbuf, dnX, dnY);
                [enc endEncoding];
                encNs += (uint64_t)((CACurrentMediaTime() - t0) * 1e9);
                [cb commit]; [cb waitUntilCompleted];
                gpuNs += (uint64_t)((cb.GPUEndTime - cb.GPUStartTime) * 1e9);
            }
        } else {
            double t0 = CACurrentMediaTime();
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
            encodeFused(enc, b, gu, 2 * e, s, xbuf, guX, guY);
            [enc endEncoding];
            enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
            encodeSwiglu(enc, swArgs, e, s);
            [enc endEncoding];
            enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
            encodeFused(enc, b, dn, e, s, ybuf, dnX, dnY);
            [enc endEncoding];
            encNs += (uint64_t)((CACurrentMediaTime() - t0) * 1e9);
            [cb commit]; [cb waitUntilCompleted];
            gpuNs += (uint64_t)((cb.GPUEndTime - cb.GPUStartTime) * 1e9);
        }
    }
    return (Cost){gpuNs / 1e6, encNs / 1e6};
}

static int cmpD(const void *a, const void *b) { double d = *(const double *)a - *(const double *)b; return d < 0 ? -1 : d > 0; }
static double med(double *v, int n) { qsort(v, n, sizeof(double), cmpD); return v[n / 2]; }

static id<MTLComputePipelineState> pipeline(NSString *path, NSString *fn, id<MTLArgumentEncoder> __strong *ae) {
    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
    if (!src) { printf("%s: %s\n", path.UTF8String, err.description.UTF8String); exit(1); }
    MTLCompileOptions *o = [MTLCompileOptions new];
    o.mathMode = MTLMathModeSafe;  // as MoltenVK with fast math off
    o.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:o error:&err];
    if (!lib) { printf("%s: %s\n", path.UTF8String, err.description.UTF8String); exit(1); }
    MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
    uint32_t dt = 4; [fc setConstantValue:&dt type:MTLDataTypeUInt atIndex:0];  // Q4_K
    id<MTLFunction> f = [src containsString:@"function_constant"] ? [lib newFunctionWithName:fn constantValues:fc error:&err]
                                                                 : [lib newFunctionWithName:fn];
    if (!f) { printf("%s: no %s\n", path.UTF8String, fn.UTF8String); exit(1); }
    if (ae) *ae = [f newArgumentEncoderWithBufferIndex:0];
    return [dev newComputePipelineStateWithFunction:f error:&err];
}

int main(int argc, char **argv) {
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        q = [dev newCommandQueue];
        if (getenv("REPS")) reps = atoi(getenv("REPS"));
        NSString *dir = @(argv[1]);
        const char *mode = argc > 2 ? argv[2] : "bench";
        pso = pipeline([dir stringByAppendingPathComponent:@"kernel.metal"], @"main0", nil);
        for (int b = 0; b < 3; b++)
            psoF[b] = pipeline([dir stringByAppendingPathComponent:[NSString stringWithFormat:@"fused-%s.metal", bname[b]]],
                               [NSString stringWithFormat:@"fused_%s", bname[b]], nil);
        id<MTLArgumentEncoder> ae = nil;
        psoSw = pipeline([dir stringByAppendingPathComponent:@"swiglu/swiglu-O0.metal"], @"main0", &ae);
        swAE = ae;
        id<MTLBuffer> swArgs = [dev newBufferWithLength:swAE.encodedLength options:MTLResourceStorageModeShared];
        printf("argument buffers tier %lu; expert %d bytes; pool %d experts (%.1f MB)\n",
               (unsigned long)dev.argumentBuffersSupport + 1, EB, POOL, POOL * (double)EB / 1e6);

        // The pool's weights: the parity tests' rule for block formats (odd bytes & 0x3b).
        size_t total = (size_t)POOL * EB;
        uint8_t *host = malloc(total);
        uint64_t r = 0x9E3779B97F4A7C15ull;
        for (size_t i = 0; i < total; i++) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; host[i] = (uint8_t)r; }
        for (size_t i = 1; i < total; i += 2) host[i] &= 0x3b;
        id<MTLBuffer> staging = [dev newBufferWithBytes:host length:total options:MTLResourceStorageModeShared];
        arena = [dev newBufferWithLength:total options:MTLResourceStorageModePrivate];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        [bl copyFromBuffer:staging sourceOffset:0 toBuffer:arena destinationOffset:0 size:total];
        for (int x = 0; x < POOL; x++) {
            per[x] = [dev newBufferWithLength:EB options:MTLResourceStorageModePrivate];
            [bl copyFromBuffer:staging sourceOffset:(NSUInteger)x * EB toBuffer:per[x] destinationOffset:0 size:EB];
            perAddr[x] = per[x].gpuAddress;
            allBufs[x] = per[x];
        }
        for (int i = POOL; i < NALL; i++) allBufs[i] = [dev newBufferWithLength:4096 options:MTLResourceStorageModePrivate];
        [bl endEncoding]; [cb commit]; [cb waitUntilCompleted];
        size_t xn = (size_t)XD + MAXE * MAXT * F;
        xbuf = [dev newBufferWithLength:xn * 4 options:MTLResourceStorageModeShared];
        float *xp = xbuf.contents;
        for (size_t i = 0; i < xn; i++) xp[i] = (float)((int)(((i + 5) * 37) % 101) - 50) / 25.0f;  // the parity tests' acts
        size_t yn = (size_t)MAXE * MAXT * (2 * F + H);
        ybuf = [dev newBufferWithLength:yn * 4 options:MTLResourceStorageModeShared];

        int es[] = {1, 3, 8}, ss[] = {1, 4, 10};
        if (!strcmp(mode, "check")) {
            const char *out = argc > 3 ? argv[3] : ".";
            NSString *od = @(out);
            [[NSData dataWithBytesNoCopy:host length:total freeWhenDone:NO] writeToFile:[od stringByAppendingPathComponent:@"pool.bin"] atomically:NO];
            [[NSData dataWithBytesNoCopy:xbuf.contents length:xn * 4 freeWhenDone:NO] writeToFile:[od stringByAppendingPathComponent:@"x.bin"] atomically:NO];
            int bad = 0, n = 0;
            for (int a = 0; a < 3; a++) for (int c = 0; c < 3; c++) {
                int e = es[a], s = ss[c], sel[MAXE];
                route(e, sel);
                size_t l0 = (size_t)e * s * F * 4, l2 = (size_t)e * s * H * 4;
                NSMutableData *(^grab)(void) = ^{
                    NSMutableData *d = [NSMutableData data];
                    [d appendBytes:ybuf.contents length:l0];
                    [d appendBytes:(uint8_t *)ybuf.contents + (size_t)UB * 4 length:l0];
                    [d appendBytes:(uint8_t *)ybuf.contents + (size_t)DB * 4 length:l2];
                    return d;
                };
                memset(ybuf.contents, 0xFF, yn * 4); run(SA, ARENA, e, s, sel, swArgs);
                NSData *ref = grab();
                for (int b = 0; b < 3; b++) {
                    memset(ybuf.contents, 0xFF, yn * 4); run(SF, b, e, s, sel, swArgs);
                    int diff = ![grab() isEqualToData:ref];
                    printf("check e%d s%-2d SF-%-5s vs SA (unfused, per-expert buffers): %s\n", e, s, bname[b], diff ? "DIFFER" : "identical");
                    bad += diff; n++;
                }
                NSData *se0 = nil;
                for (int b = 0; b < 3; b++) {
                    memset(ybuf.contents, 0xFF, yn * 4); run(SE, b, e, s, sel, swArgs);
                    NSData *d = grab();
                    if (!se0) {
                        se0 = d;
                        NSString *f = [od stringByAppendingPathComponent:[NSString stringWithFormat:@"se-e%d-s%d.bin", e, s]];
                        NSMutableData *w = [NSMutableData dataWithBytes:sel length:sizeof(int) * MAXE];
                        [w appendData:d]; [w writeToFile:f atomically:NO];
                    } else {
                        int diff = ![d isEqualToData:se0];
                        printf("check e%d s%-2d SE-%-5s vs SE-arena: %s\n", e, s, bname[b], diff ? "DIFFER" : "identical");
                        bad += diff; n++;
                    }
                }
            }
            printf("check: %d of %d comparisons differ\n", bad, n);
            return bad != 0;
        }

        if (!strcmp(mode, "swiglu")) {
            // The SwiGLU dispatch alone in a command buffer, at a layer's sizes (e * s * F).
            int ns[] = {1 * 1 * F, 3 * 1 * F, 8 * 1 * F, 8 * 4 * F, 8 * 10 * F};
            for (int c = 0; c < 5; c++) {
                double *t = calloc(reps, 8);
                for (int i = -10; i < reps; i++) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                    encodeSwiglu(enc, swArgs, 1, ns[c] / F);
                    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    if (i >= 0) t[i] = (cb.GPUEndTime - cb.GPUStartTime) * 1e6;
                }
                printf("swiglu alone, n = %6d: GPU %.1f us per command buffer\n", ns[c], med(t, reps));
            }
            return 0;
        }

        // bench: interleaved shapes per configuration, a new routed set every call.
        printf("%-10s %-4s | GPU ms (median) | wall ms | host encode us\n", "shape", "e×s");
        for (int a = 0; a < 3; a++) for (int c = 0; c < 3; c++) {
            int e = es[a], s = ss[c];
            struct { Shape sh; Binding b; const char *name; } shapes[] = {
                {SA, ARENA, "SA"}, {SF, ARENA, "SF-arena"}, {SF, ARRAY, "SF-array"}, {SF, BDA, "SF-bda"},
                {SF, BDA_ALL, "SF-bda-all"},
                {SE, ARENA, "SE-arena"}, {SE, ARRAY, "SE-array"}, {SE, BDA, "SE-bda"}, {SE, BDA_ALL, "SE-bda-all"}};
            enum { NS = 9 };
            double *gpu[NS], *wall[NS], *enc[NS];
            for (int k = 0; k < NS; k++) { gpu[k] = calloc(reps, 8); wall[k] = calloc(reps, 8); enc[k] = calloc(reps, 8); }
            int sel[MAXE];
            for (int w = 0; w < 10; w++) for (int k = 0; k < NS; k++) { route(e, sel); run(shapes[k].sh, shapes[k].b, e, s, sel, swArgs); }
            for (int i = 0; i < reps; i++) {
                route(e, sel);  // the same routed set for every shape in this round
                for (int k = 0; k < NS; k++) {
                    double t0 = CACurrentMediaTime();
                    Cost cst = run(shapes[k].sh, shapes[k].b, e, s, sel, swArgs);
                    wall[k][i] = (CACurrentMediaTime() - t0) * 1e3; gpu[k][i] = cst.gpu; enc[k][i] = cst.encode * 1e3;
                }
            }
            for (int k = 0; k < NS; k++)
                printf("%-10s %dx%-2d | %7.3f | %7.3f | %6.1f\n", shapes[k].name, e, s, med(gpu[k], reps), med(wall[k], reps), med(enc[k], reps));
            printf("\n"); fflush(stdout);
        }
    }
    return 0;
}
