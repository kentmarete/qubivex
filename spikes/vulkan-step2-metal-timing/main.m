// Metal-level timing harness for the QUBIVEX expert matmuls (Step 2 investigation).
// Runs MoltenVK's own MSL translation of matmul.comp (Q4_K) in the shapes the Vulkan
// backend submits, and records Metal's GPUStartTime/GPUEndTime per command buffer,
// plus stage-boundary counter samples per encoder.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#include <mach/mach_time.h>
#include <stdlib.h>

enum { H = 2048, F = 1024, MAXE = 8, MAXT = 32 };
static int rowBytes(int k) { return k / 256 * 144; }
typedef struct { int n, k, off; } Mat;
typedef struct { Mat m; int xBase, yBase; } Op;
typedef struct { uint32_t n, k, rowBytes, wBase, xBase, yBase; } Params;
typedef struct { double a, b; } Iv;

static id<MTLDevice> dev;
static id<MTLComputePipelineState> pso;
static id<MTLBuffer> wbuf, xbuf, ybuf;
static id<MTLCommandQueue> q0;
static NSArray<id<MTLCommandQueue>> *queues;
static Mat mats[MAXE][3];
static int reps = 200;
static int threads = 5;

static void encodeOp(id<MTLComputeCommandEncoder> enc, Op op, int s) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:wbuf offset:op.m.off atIndex:0];
    [enc setBuffer:xbuf offset:0 atIndex:1];
    [enc setBuffer:ybuf offset:0 atIndex:2];
    Params p = {op.m.n, op.m.k, rowBytes(op.m.k), 0, op.xBase, op.yBase};
    [enc setBytes:&p length:sizeof p atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake((op.m.n + 7) / 8, s, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// gu[2e], gu[2e+1]: gate, up of expert e; down[e].
static void makeOps(int experts, int s, Op *gu, Op *down) {
    int y = 0;
    for (int e = 0; e < experts; e++) {
        gu[2 * e] = (Op){mats[e][0], 0, y}; y += s * F;
        gu[2 * e + 1] = (Op){mats[e][1], 0, y}; y += s * F;
        down[e] = (Op){mats[e][2], MAXT * H, y}; y += s * H;
    }
}

typedef enum { PER_EXPERT, GROUPED_SERIAL, GROUPED_SERIAL_ENC, GROUPED_CONC, MULTI_CB, MULTI_QUEUE } Shape;
static const char *shapeName[] = {"A per-expert (5 thr)", "B serial, 1 encoder", "B serial, enc/dispatch",
                                  "B concurrent, 1 enc", "B multi-cb, 1 queue", "B multi-queue"};

typedef struct { Iv iv[64]; int n; Iv ts[64]; int nts; } Rec;
// MoltenVK-style timestamps (TS=1): what vkCmdWriteTimestamp(TOP) ... (BOTTOM) becomes on an
// Apple GPU: a blit pass sampling at its start, before the compute encoder; the compute
// encoder updates a per-command-buffer fence; a second blit pass waits on it and samples.
// Each blit fills one byte of a shared private dummy buffer (MVKDevice::getDummyBlitMTLBuffer).
static int tsMode; static id<MTLBuffer> dummy; static id<MTLCounterSet> tsSet; static double nsPerTick = 1;
static id<MTLCounterSampleBuffer> tsPool[64]; static volatile int tsNext;
static void tsBlit(id<MTLCommandBuffer> cb, int slot, int idx, id<MTLFence> f) {
    MTLBlitPassDescriptor *bp = [MTLBlitPassDescriptor new];
    bp.sampleBufferAttachments[0].sampleBuffer = tsPool[slot];
    bp.sampleBufferAttachments[0].startOfEncoderSampleIndex = idx;
    bp.sampleBufferAttachments[0].endOfEncoderSampleIndex = idx;
    id<MTLBlitCommandEncoder> e = [cb blitCommandEncoderWithDescriptor:bp];
    [e waitForFence:f]; [e fillBuffer:dummy range:NSMakeRange(0, 1) value:0]; [e endEncoding];
}
static int tsBegin(id<MTLCommandBuffer> cb, id<MTLFence> __strong *f) {
    if (!tsMode) return -1;
    int slot = __sync_fetch_and_add(&tsNext, 1);
    *f = [dev newFence]; tsBlit(cb, slot, 0, *f); return slot;
}
static void tsEnd(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc, int slot, id<MTLFence> f) {
    if (slot >= 0) [enc updateFence:f];
    [enc endEncoding];
    if (slot >= 0) tsBlit(cb, slot, 1, f);
}
static void add(Rec *r, id<MTLCommandBuffer> cb, int slot) {
    Iv t = {0, 0};
    if (slot >= 0) {
        NSData *d = [tsPool[slot] resolveCounterRange:NSMakeRange(0, 2)];
        const MTLCounterResultTimestamp *ts = d.bytes;
        t = (Iv){ts[0].timestamp * nsPerTick * 1e-9, ts[1].timestamp * nsPerTick * 1e-9};
    }
    @synchronized(queues) {
        r->iv[r->n++] = (Iv){cb.GPUStartTime, cb.GPUEndTime};
        if (slot >= 0) r->ts[r->nts++] = t;
    }
}

static void run(Shape shape, int experts, int s, Rec *rec) {
    Op gu[2 * MAXE], down[MAXE];
    makeOps(experts, s, gu, down);
    switch (shape) {
    case PER_EXPERT:
        // Today: one thread per expert, at most 5 at once; gate+up, wait, down, wait.
        { Op *gp = gu, *dp = down;
        for (int wave = 0; wave < experts; wave += threads) {
            int n = experts - wave < threads ? experts - wave : threads;
            dispatch_apply(n, DISPATCH_APPLY_AUTO, ^(size_t i) {
                int e = wave + (int)i;
                id<MTLCommandBuffer> cb = [q0 commandBuffer]; id<MTLFence> f1 = nil;
                int s1 = tsBegin(cb, &f1);
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                encodeOp(enc, gp[2 * e], s); encodeOp(enc, gp[2 * e + 1], s);
                tsEnd(cb, enc, s1, f1); [cb commit]; [cb waitUntilCompleted]; add(rec, cb, s1);
                id<MTLCommandBuffer> cb2 = [q0 commandBuffer]; id<MTLFence> f2 = nil;
                int s2 = tsBegin(cb2, &f2);
                id<MTLComputeCommandEncoder> enc2 = [cb2 computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                encodeOp(enc2, dp[e], s);
                tsEnd(cb2, enc2, s2, f2); [cb2 commit]; [cb2 waitUntilCompleted]; add(rec, cb2, s2);
            });
        }
        }
        break;
    case GROUPED_SERIAL: case GROUPED_SERIAL_ENC: case GROUPED_CONC: {
        MTLDispatchType dt = shape == GROUPED_CONC ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial;
        for (int st = 0; st < 2; st++) {
            Op *ops = st ? down : gu; int n = st ? experts : 2 * experts;
            id<MTLCommandBuffer> cb = [q0 commandBuffer]; id<MTLFence> f = nil;
            int slot = shape == GROUPED_SERIAL ? tsBegin(cb, &f) : -1;
            if (shape == GROUPED_SERIAL_ENC) {
                for (int i = 0; i < n; i++) {
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:dt];
                    encodeOp(enc, ops[i], s); [enc endEncoding];
                }
            } else {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:dt];
                for (int i = 0; i < n; i++) encodeOp(enc, ops[i], s);
                tsEnd(cb, enc, slot, f);
            }
            [cb commit]; [cb waitUntilCompleted]; add(rec, cb, slot);
        }
        break;
    }
    case MULTI_CB: case MULTI_QUEUE:
        // One command buffer per expert, committed back to back; one wait per stage.
        for (int st = 0; st < 2; st++) {
            NSMutableArray<id<MTLCommandBuffer>> *cbs = [NSMutableArray array];
            for (int e = 0; e < experts; e++) {
                id<MTLCommandQueue> q = shape == MULTI_QUEUE ? queues[e % queues.count] : q0;
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                if (st == 0) { encodeOp(enc, gu[2 * e], s); encodeOp(enc, gu[2 * e + 1], s); }
                else encodeOp(enc, down[e], s);
                [enc endEncoding]; [cb commit]; [cbs addObject:cb];
            }
            for (id<MTLCommandBuffer> cb in cbs) { [cb waitUntilCompleted]; add(rec, cb, -1); }
        }
        break;
    }
}

static int cmpIv(const void *x, const void *y) { double d = ((const Iv *)x)->a - ((const Iv *)y)->a; return d < 0 ? -1 : d > 0; }
static double unionIv(const Iv *iv, int n) {
    if (!n) return 0;
    Iv s[64]; memcpy(s, iv, sizeof(Iv) * n); qsort(s, n, sizeof(Iv), cmpIv);
    double tot = 0, ca = s[0].a, cb = s[0].b;
    for (int i = 1; i < n; i++) {
        if (s[i].a <= cb) { if (s[i].b > cb) cb = s[i].b; }
        else { tot += cb - ca; ca = s[i].a; cb = s[i].b; }
    }
    return tot + cb - ca;
}
static int overlapping(Rec *r) {
    int n = 0;
    for (int i = 0; i < r->n; i++) for (int j = i + 1; j < r->n; j++)
        if (r->iv[i].a < r->iv[j].b && r->iv[j].a < r->iv[i].b) n++;
    return n;
}
static int cmpD(const void *x, const void *y) { double d = *(const double *)x - *(const double *)y; return d < 0 ? -1 : d > 0; }
static double med(double *v, int n) { qsort(v, n, sizeof(double), cmpD); return v[n / 2]; }

static void measure(Shape shape, int experts, int s) {
    for (int i = 0; i < 10; i++) { Rec r = {0}; tsNext = 0; run(shape, experts, s, &r); }
    double *wall = calloc(reps, 8), *sum = calloc(reps, 8), *uni = calloc(reps, 8), *span = calloc(reps, 8), *pairs = calloc(reps, 8);
    double *tsum = calloc(reps, 8), *tuni = calloc(reps, 8);
    int ncb = 0;
    for (int k = 0; k < reps; k++) {
        Rec r = {0}; tsNext = 0;
        double t0 = CACurrentMediaTime();
        run(shape, experts, s, &r);
        wall[k] = CACurrentMediaTime() - t0;
        double sm = 0, lo = 1e30, hi = 0;
        for (int i = 0; i < r.n; i++) { sm += r.iv[i].b - r.iv[i].a; if (r.iv[i].a < lo) lo = r.iv[i].a; if (r.iv[i].b > hi) hi = r.iv[i].b; }
        double tsm = 0; for (int i = 0; i < r.nts; i++) tsm += r.ts[i].b - r.ts[i].a;
        tsum[k] = tsm; tuni[k] = unionIv(r.ts, r.nts);
        sum[k] = sm; uni[k] = unionIv(r.iv, r.n); span[k] = hi - lo; pairs[k] = overlapping(&r); ncb = r.n;
    }
    printf("%-24s e%d s%-2d | cbs %2d | wall %.3f | GPU sum %.3f | GPU union %.3f | first->last %.3f | overlapping pairs %.0f",
           shapeName[shape], experts, s, ncb, med(wall, reps) * 1e3, med(sum, reps) * 1e3, med(uni, reps) * 1e3,
           med(span, reps) * 1e3, med(pairs, reps));
    if (tsMode && (shape == PER_EXPERT || shape == GROUPED_SERIAL))
        printf(" | MVK-style ts sum %.3f union %.3f", med(tsum, reps) * 1e3, med(tuni, reps) * 1e3);
    printf("\n");
    fflush(stdout);
    free(wall); free(sum); free(uni); free(span); free(pairs);
}

// Per-dispatch GPU intervals inside one command buffer: one encoder per dispatch,
// stage-boundary timestamps at the start and end of each encoder.
static void perDispatch(int experts, int s, MTLDispatchType dt) {
    id<MTLCounterSet> set = nil;
    for (id<MTLCounterSet> cs in dev.counterSets) if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) set = cs;
    if (!set || ![dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) { printf("no stage timestamps\n"); return; }
    mach_timebase_info_data_t tb; mach_timebase_info(&tb);
    Op gu[2 * MAXE], down[MAXE];
    makeOps(experts, s, gu, down);
    for (int st = 0; st < 2; st++) {
        Op *ops = st ? down : gu; int n = st ? experts : 2 * experts;
        MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
        d.counterSet = set; d.storageMode = MTLStorageModeShared; d.sampleCount = 2 * n;
        NSError *err = nil;
        id<MTLCounterSampleBuffer> sb = [dev newCounterSampleBufferWithDescriptor:d error:&err];
        if (!sb) { printf("sample buffer: %s\n", err.description.UTF8String); return; }
        double *starts = calloc(n * reps, 8), *durs = calloc(n * reps, 8), *cbt = calloc(reps, 8), *last = calloc(reps, 8);
        for (int rep = 0; rep < reps + 10; rep++) {
            MTLTimestamp c0, g0, c1, g1;
            [dev sampleTimestamps:&c0 gpuTimestamp:&g0];
            id<MTLCommandBuffer> cb = [q0 commandBuffer];
            for (int i = 0; i < n; i++) {
                MTLComputePassDescriptor *pd = [MTLComputePassDescriptor computePassDescriptor];
                pd.dispatchType = dt;
                pd.sampleBufferAttachments[0].sampleBuffer = sb;
                pd.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * i;
                pd.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * i + 1;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDescriptor:pd];
                encodeOp(enc, ops[i], s); [enc endEncoding];
            }
            [cb commit]; [cb waitUntilCompleted];
            [dev sampleTimestamps:&c1 gpuTimestamp:&g1];
            if (rep < 10) continue;
            int k = rep - 10;
            double nsPerTick = (double)(c1 - c0) / (double)(g1 - g0); (void)tb;
            NSData *data = [sb resolveCounterRange:NSMakeRange(0, 2 * n)];
            const MTLCounterResultTimestamp *ts = data.bytes;
            uint64_t t0 = ts[0].timestamp; double end = 0;
            for (int i = 0; i < n; i++) {
                starts[i * reps + k] = (double)(ts[2 * i].timestamp - t0) * nsPerTick / 1e3;
                durs[i * reps + k] = (double)(ts[2 * i + 1].timestamp - ts[2 * i].timestamp) * nsPerTick / 1e3;
                double e = (double)(ts[2 * i + 1].timestamp - t0) * nsPerTick / 1e3; if (e > end) end = e;
            }
            last[k] = end;
            cbt[k] = (cb.GPUEndTime - cb.GPUStartTime) * 1e6;
        }
        printf("  %s e%d s%d, %s, one encoder per dispatch: cb GPU %.1f us, first start -> last end %.1f us\n",
               st ? "down" : "gate+up", experts, s, dt == MTLDispatchTypeSerial ? "serial" : "concurrent", med(cbt, reps), med(last, reps));
        double sumd = 0;
        for (int i = 0; i < n; i++) {
            double ms = med(starts + i * reps, reps), md = med(durs + i * reps, reps); sumd += md;
            printf("    dispatch %2d (n=%4d k=%4d): start +%6.1f us  dur %5.1f us  end +%6.1f\n", i, ops[i].m.n, ops[i].m.k, ms, md, ms + md);
        }
        printf("    sum of dispatch durations %.1f us\n", sumd);
        free(starts); free(durs); free(cbt); free(last);
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        if (getenv("REPS")) reps = atoi(getenv("REPS"));
        int untracked = getenv("UNTRACKED") && atoi(getenv("UNTRACKED"));
        tsMode = getenv("TS") && atoi(getenv("TS"));
        if (getenv("THREADS")) threads = atoi(getenv("THREADS"));
        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&err];
        MTLCompileOptions *opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) { printf("compile: %s\n", err.description.UTF8String); return 1; }
        MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
        uint32_t dt = 4; [fc setConstantValue:&dt type:MTLDataTypeUInt atIndex:0];
        id<MTLFunction> fn = [lib newFunctionWithName:@"main0" constantValues:fc error:&err];
        pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        printf("device %s  cores? | pipeline maxThreadsPerTG %lu  simd width %lu  static TG mem %lu B | untracked %d\n",
               dev.name.UTF8String, (unsigned long)pso.maxTotalThreadsPerThreadgroup, (unsigned long)pso.threadExecutionWidth,
               (unsigned long)pso.staticThreadgroupMemoryLength, untracked);
        const char *pn[] = {"draw", "dispatch", "blit", "stage", "tile"};
        MTLCounterSamplingPoint pts[] = {MTLCounterSamplingPointAtDrawBoundary, MTLCounterSamplingPointAtDispatchBoundary,
                                         MTLCounterSamplingPointAtBlitBoundary, MTLCounterSamplingPointAtStageBoundary,
                                         MTLCounterSamplingPointAtTileDispatchBoundary};
        for (int i = 0; i < 5; i++) printf("  counter sampling at %s boundary: %d\n", pn[i], [dev supportsCounterSampling:pts[i]]);
        for (id<MTLCounterSet> cs in dev.counterSets) {
            printf("  counter set %s:", cs.name.UTF8String);
            for (id<MTLCounter> c in cs.counters) printf(" %s", c.name.UTF8String);
            printf("\n");
        }

        int off = 0;
        for (int e = 0; e < MAXE; e++) {
            int shp[3][2] = {{F, H}, {F, H}, {H, F}};
            for (int m = 0; m < 3; m++) { mats[e][m] = (Mat){shp[m][0], shp[m][1], off}; off += shp[m][0] * rowBytes(shp[m][1]); }
        }
        uint8_t *host = malloc(off);
        uint64_t r = 0x9E3779B97F4A7C15ull;
        #define NEXT() (r ^= r << 13, r ^= r >> 7, r ^= r << 17, r)
        for (int i = 0; i < off; i++) host[i] = (uint8_t)NEXT();
        for (int b = 0; b < off; b += 144)
            for (int h = 0; h < 2; h++) { uint16_t v = 0x1800 + (uint16_t)(NEXT() % 0x1000); host[b + 2 * h] = v & 0xFF; host[b + 2 * h + 1] = v >> 8; }
        MTLResourceOptions priv = MTLResourceStorageModePrivate | (untracked ? MTLResourceHazardTrackingModeUntracked : 0);
        MTLResourceOptions shared = MTLResourceStorageModeShared | (untracked ? MTLResourceHazardTrackingModeUntracked : 0);
        id<MTLBuffer> staging = [dev newBufferWithBytes:host length:off options:MTLResourceStorageModeShared];
        wbuf = [dev newBufferWithLength:off options:priv];
        q0 = [dev newCommandQueue];
        NSMutableArray *qs = [NSMutableArray array];
        for (int i = 0; i < 8; i++) [qs addObject:[dev newCommandQueue]];
        queues = qs;
        id<MTLCommandBuffer> cb = [q0 commandBuffer];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        [bl copyFromBuffer:staging sourceOffset:0 toBuffer:wbuf destinationOffset:0 size:off];
        [bl endEncoding]; [cb commit]; [cb waitUntilCompleted];
        dummy = [dev newBufferWithLength:1 options:MTLResourceStorageModePrivate];
        for (id<MTLCounterSet> cs in dev.counterSets) if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) tsSet = cs;
        for (int i = 0; i < 64; i++) {
            MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
            d.counterSet = tsSet; d.storageMode = MTLStorageModeShared; d.sampleCount = 2;
            tsPool[i] = [dev newCounterSampleBufferWithDescriptor:d error:&err];
        }
        { MTLTimestamp c0, g0, c1, g1; [dev sampleTimestamps:&c0 gpuTimestamp:&g0]; usleep(200000);
          [dev sampleTimestamps:&c1 gpuTimestamp:&g1]; nsPerTick = (double)(c1 - c0) / (double)(g1 - g0);
          printf("GPU timestamp: %.4f ns per tick\n", nsPerTick); }
        xbuf = [dev newBufferWithLength:4 * MAXT * (H + F) options:shared];
        float *xp = xbuf.contents;
        for (int i = 0; i < MAXT * (H + F); i++) xp[i] = (float)((int)(NEXT() % 2001) - 1000) / 1000.0f;
        ybuf = [dev newBufferWithLength:4 * MAXE * MAXT * (2 * F + H) options:shared];
        printf("expert weight bytes %d (gate/up %d each, down %d)\n\n", off / MAXE, mats[0][1].off - mats[0][0].off, off / MAXE - 2 * (mats[0][1].off - mats[0][0].off));

        const char *which = argc > 2 ? argv[2] : "all";
        if (!strcmp(which, "all") || !strcmp(which, "shapes")) {
            int cfg[][2] = {{1, 1}, {4, 1}, {8, 1}, {8, 4}};
            for (int c = 0; c < 4; c++) {
                for (int sh = 0; sh <= MULTI_QUEUE; sh++) measure((Shape)sh, cfg[c][0], cfg[c][1]);
                printf("\n");
            }
        }
        if (!strcmp(which, "scale")) {
            // One dispatch of n rows (k = 2048, contiguous Q4_K rows from offset 0), alone in a
            // command buffer: GPU time and effective weight bandwidth.
            int ns[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
            for (int c = 0; c < 8; c++) {
                Op op = {{ns[c], H, 0}, 0, 0};
                double *t = calloc(reps, 8);
                for (int k = -10; k < reps; k++) {
                    id<MTLCommandBuffer> cb = [q0 commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                    encodeOp(enc, op, 1); [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                    if (k >= 0) t[k] = cb.GPUEndTime - cb.GPUStartTime;
                }
                double m = med(t, reps);
                printf("one dispatch n=%5d k=%d (%4d threadgroups, %5.2f MB): GPU %7.1f us  %5.1f GB/s  %.2f us per 1024 rows\n",
                       ns[c], H, (ns[c] + 7) / 8, ns[c] * rowBytes(H) / 1e6, m * 1e6, ns[c] * rowBytes(H) / m / 1e9, m * 1e6 * 1024 / ns[c]);
                free(t);
            }
        }
        if (!strcmp(which, "focus")) {
            int cfg[][2] = {{8, 1}, {8, 4}};
            for (int c = 0; c < 2; c++) {
                measure(PER_EXPERT, cfg[c][0], cfg[c][1]); measure(GROUPED_SERIAL, cfg[c][0], cfg[c][1]);
                measure(GROUPED_CONC, cfg[c][0], cfg[c][1]);
            }
        }
        if (!strcmp(which, "all") || !strcmp(which, "dispatch")) {
            perDispatch(1, 1, MTLDispatchTypeSerial);
            perDispatch(8, 1, MTLDispatchTypeSerial);
            perDispatch(8, 1, MTLDispatchTypeConcurrent);
        }
    }
    return 0;
}
