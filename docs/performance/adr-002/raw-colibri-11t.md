# Raw: Colibri 0a5504a1 baseline, 11 threads (libomp)


- OpenMP threads: 11

| shape | S | dtype | Colibri kernel | hot µs | cold µs | cold GB/s | hot GFLOP/s | rel err |
|---|---|---|---|---|---|---|---|---|
| gate/up 2048→1024 | 1 | f32 | matmul_f32.h `matmul` (lanes) | 58.8 | 93.6 | 89.6 | 71.4 | 1.9e-07 |
| gate/up 2048→1024 | 1 | bf16 | matmul_f16.h `matmul_bf16w` | 54.8 | 63.2 | 66.4 | 76.6 | 1.9e-07 |
| gate/up 2048→1024 | 1 | int8-row | olmoe.c `matmul_q` / `_rows` (IDOT off, default) | 86.6 | 87.7 | 24.0 | 48.4 | 2.0e-07 |
| gate/up 2048→1024 | 1 | int8-row | quant.h `matmul_q` NEON (f32 act) | 78.7 | 80.1 | 26.2 | 53.3 | 3.8e-07 |
| gate/up 2048→1024 | 1 | q8_0 | `gq_matmul_f32` (exact, = GQ_EXACT=1) | 422.3 | 422.5 | 5.3 | 9.9 | 1.1e-06 |
| gate/up 2048→1024 | 1 | q8_0 | `gq_matmul` (integer dot, default) | 56.8 | 59.8 | 37.3 | 73.9 | 4.1e-03 |
| gate/up 2048→1024 | 1 | q4_k | `gq_matmul_f32` (exact, = GQ_EXACT=1) | 440.2 | 432.0 | 2.7 | 9.5 | 1.4e-06 |
| gate/up 2048→1024 | 1 | q4_k | `gq_matmul` (integer dot, default) | 92.6 | 92.9 | 12.7 | 45.3 | 4.1e-03 |
| gate/up 2048→1024 | 128 | f32 | matmul_f32.h `matmul` (lanes) | 2957.6 | 2976.3 | 2.8 | 181.5 | 1.8e-07 |
| gate/up 2048→1024 | 128 | bf16 | matmul_f16.h `matmul_bf16w` | 2358.7 | 2365.2 | 1.8 | 227.6 | 1.6e-07 |
| gate/up 2048→1024 | 128 | int8-row | olmoe.c `matmul_q` / `_rows` (IDOT off, default) | 5630.6 | 5664.7 | 0.4 | 95.3 | 2.1e-07 |
| gate/up 2048→1024 | 128 | int8-row | quant.h `matmul_q` NEON (f32 act) | 4420.3 | 4429.0 | 0.5 | 121.5 | 3.0e-07 |
| gate/up 2048→1024 | 128 | q8_0 | `gq_matmul_f32` (exact, = GQ_EXACT=1) | 26141.7 | 25731.0 | 0.1 | 20.5 | 1.2e-06 |
| gate/up 2048→1024 | 128 | q8_0 | `gq_matmul` (integer dot, default) | 2593.2 | 2595.2 | 0.9 | 207.0 | 4.3e-03 |
| gate/up 2048→1024 | 128 | q4_k | `gq_matmul_f32` (exact, = GQ_EXACT=1) | 25588.2 | 25401.0 | 0.0 | 21.0 | 1.5e-06 |
| gate/up 2048→1024 | 128 | q4_k | `gq_matmul` (integer dot, default) | 3592.2 | 3590.0 | 0.3 | 149.5 | 4.1e-03 |
| down 1024→2048 | 1 | f32 | matmul_f32.h `matmul` (lanes) | 63.2 | 95.7 | 87.7 | 66.4 | 1.5e-07 |
| down 1024→2048 | 1 | bf16 | matmul_f16.h `matmul_bf16w` | 51.1 | 63.2 | 66.4 | 82.0 | 1.4e-07 |
| down 1024→2048 | 1 | int8-row | olmoe.c `matmul_q` / `_rows` (IDOT off, default) | 84.4 | 85.1 | 24.7 | 49.7 | 2.1e-07 |
| down 1024→2048 | 1 | int8-row | quant.h `matmul_q` NEON (f32 act) | 74.8 | 74.6 | 28.2 | 56.1 | 2.5e-07 |
| down 1024→2048 | 1 | q8_0 | `gq_matmul_f32` (exact, = GQ_EXACT=1) | 429.1 | 425.5 | 5.2 | 9.8 | 1.2e-06 |
| down 1024→2048 | 1 | q8_0 | `gq_matmul` (integer dot, default) | 57.6 | 57.8 | 38.5 | 72.8 | 3.9e-03 |
| down 1024→2048 | 1 | q4_k | `gq_matmul_f32` (exact, = GQ_EXACT=1) | 433.8 | 435.1 | 2.7 | 9.7 | 1.2e-06 |
| down 1024→2048 | 1 | q4_k | `gq_matmul` (integer dot, default) | 85.4 | 85.2 | 13.8 | 49.1 | 4.0e-03 |
| down 1024→2048 | 128 | f32 | matmul_f32.h `matmul` (lanes) | 3016.1 | 3020.7 | 2.8 | 178.0 | 1.4e-07 |
| down 1024→2048 | 128 | bf16 | matmul_f16.h `matmul_bf16w` | 2388.7 | 2390.9 | 1.8 | 224.8 | 1.3e-07 |
| down 1024→2048 | 128 | int8-row | olmoe.c `matmul_q` / `_rows` (IDOT off, default) | 5748.5 | 5735.7 | 0.4 | 93.4 | 1.8e-07 |
| down 1024→2048 | 128 | int8-row | quant.h `matmul_q` NEON (f32 act) | 4155.3 | 4172.7 | 0.5 | 129.2 | 3.0e-07 |
| down 1024→2048 | 128 | q8_0 | `gq_matmul_f32` (exact, = GQ_EXACT=1) | 25208.2 | 25334.9 | 0.1 | 21.3 | 1.1e-06 |
| down 1024→2048 | 128 | q8_0 | `gq_matmul` (integer dot, default) | 2361.0 | 2361.2 | 0.9 | 227.4 | 4.4e-03 |
| down 1024→2048 | 128 | q4_k | `gq_matmul_f32` (exact, = GQ_EXACT=1) | 25026.2 | 25131.3 | 0.0 | 21.5 | 1.2e-06 |
| down 1024→2048 | 128 | q4_k | `gq_matmul` (integer dot, default) | 3141.9 | 3136.1 | 0.4 | 170.9 | 4.4e-03 |

## Attention decode (16 heads × 128, f32 KV)

| context | Colibri kernel | µs | KV GB/s | rel err |
|---|---|---|---|---|
| 1024 | bench_attn_f32 scalar (olmoe/qwen36 as shipped) | 224.1 | 74.9 | 9.9e-07 |
| 1024 | bench_attn_f32 lanes (matmul_f32.h dot) | 106.7 | 157.3 | 1.3e-06 |
| 4096 | bench_attn_f32 scalar (olmoe/qwen36 as shipped) | 677.5 | 99.1 | 2.2e-06 |
| 4096 | bench_attn_f32 lanes (matmul_f32.h dot) | 562.3 | 119.3 | 2.2e-06 |

## Colibri determinism

Output hashes from fresh processes with OMP_NUM_THREADS = 1, 2, 4, 11, each case run twice: all identical (no run-to-run or cross-thread differences).
