# Raw: Colibri 0a5504a1 baseline, 1 thread (`-fopenmp-simd`, no OpenMP runtime)


- OpenMP threads: 1

| shape | S | dtype | Colibri kernel | hot µs | cold µs | cold GB/s | hot GFLOP/s | rel err |
|---|---|---|---|---|---|---|---|---|
| gate/up 2048→1024 | 1 | f32 | matmul_f32.h `matmul` (lanes) | 115.5 | 119.7 | 70.1 | 36.3 | 1.9e-07 |
| gate/up 2048→1024 | 1 | bf16 | matmul_f16.h `matmul_bf16w` | 80.2 | 79.7 | 52.6 | 52.3 | 1.9e-07 |
| gate/up 2048→1024 | 1 | int8-row | olmoe.c `matmul_q` / `_rows` (IDOT off, default) | 223.5 | 223.8 | 9.4 | 18.8 | 2.0e-07 |
| gate/up 2048→1024 | 1 | int8-row | quant.h `matmul_q` NEON (f32 act) | 231.0 | 231.6 | 9.1 | 18.2 | 3.8e-07 |
| gate/up 2048→1024 | 1 | q8_0 | `gq_matmul_f32` (exact, = GQ_EXACT=1) | 1956.7 | 1957.2 | 1.1 | 2.1 | 1.1e-06 |
| gate/up 2048→1024 | 1 | q8_0 | `gq_matmul` (integer dot, default) | 89.6 | 90.3 | 24.7 | 46.8 | 4.1e-03 |
| gate/up 2048→1024 | 1 | q4_k | `gq_matmul_f32` (exact, = GQ_EXACT=1) | 2085.8 | 1938.5 | 0.6 | 2.0 | 1.4e-06 |
| gate/up 2048→1024 | 1 | q4_k | `gq_matmul` (integer dot, default) | 230.8 | 230.6 | 5.1 | 18.2 | 4.1e-03 |
| gate/up 2048→1024 | 128 | f32 | matmul_f32.h `matmul` (lanes) | 14765.8 | 14808.5 | 0.6 | 36.4 | 1.8e-07 |
| gate/up 2048→1024 | 128 | bf16 | matmul_f16.h `matmul_bf16w` | 12033.9 | 12047.5 | 0.3 | 44.6 | 1.6e-07 |
| gate/up 2048→1024 | 128 | int8-row | olmoe.c `matmul_q` / `_rows` (IDOT off, default) | 28541.8 | 28480.0 | 0.1 | 18.8 | 2.1e-07 |
| gate/up 2048→1024 | 128 | int8-row | quant.h `matmul_q` NEON (f32 act) | 29649.0 | 29651.9 | 0.1 | 18.1 | 3.0e-07 |
| gate/up 2048→1024 | 128 | q8_0 | `gq_matmul_f32` (exact, = GQ_EXACT=1) | 171703.2 | 170928.4 | 0.0 | 3.1 | 1.2e-06 |
| gate/up 2048→1024 | 128 | q8_0 | `gq_matmul` (integer dot, default) | 11428.8 | 11467.1 | 0.2 | 47.0 | 4.3e-03 |
| gate/up 2048→1024 | 128 | q4_k | `gq_matmul_f32` (exact, = GQ_EXACT=1) | 172926.4 | 171109.5 | 0.0 | 3.1 | 1.5e-06 |
| gate/up 2048→1024 | 128 | q4_k | `gq_matmul` (integer dot, default) | 20387.0 | 20226.6 | 0.1 | 26.3 | 4.1e-03 |
| down 1024→2048 | 1 | f32 | matmul_f32.h `matmul` (lanes) | 117.3 | 120.8 | 69.5 | 35.7 | 1.5e-07 |
| down 1024→2048 | 1 | bf16 | matmul_f16.h `matmul_bf16w` | 80.8 | 81.2 | 51.6 | 51.9 | 1.4e-07 |
| down 1024→2048 | 1 | int8-row | olmoe.c `matmul_q` / `_rows` (IDOT off, default) | 224.8 | 225.2 | 9.4 | 18.7 | 2.1e-07 |
| down 1024→2048 | 1 | int8-row | quant.h `matmul_q` NEON (f32 act) | 197.2 | 197.8 | 10.6 | 21.3 | 2.5e-07 |
| down 1024→2048 | 1 | q8_0 | `gq_matmul_f32` (exact, = GQ_EXACT=1) | 2098.3 | 2099.2 | 1.1 | 2.0 | 1.2e-06 |
| down 1024→2048 | 1 | q8_0 | `gq_matmul` (integer dot, default) | 87.8 | 88.5 | 25.2 | 47.8 | 3.9e-03 |
| down 1024→2048 | 1 | q4_k | `gq_matmul_f32` (exact, = GQ_EXACT=1) | 2103.2 | 2101.8 | 0.6 | 2.0 | 1.2e-06 |
| down 1024→2048 | 1 | q4_k | `gq_matmul` (integer dot, default) | 205.1 | 204.9 | 5.8 | 20.5 | 4.0e-03 |
| down 1024→2048 | 128 | f32 | matmul_f32.h `matmul` (lanes) | 15117.5 | 15064.0 | 0.6 | 35.5 | 1.4e-07 |
| down 1024→2048 | 128 | bf16 | matmul_f16.h `matmul_bf16w` | 12197.5 | 12369.0 | 0.3 | 44.0 | 1.3e-07 |
| down 1024→2048 | 128 | int8-row | olmoe.c `matmul_q` / `_rows` (IDOT off, default) | 28901.4 | 28858.2 | 0.1 | 18.6 | 1.8e-07 |
| down 1024→2048 | 128 | int8-row | quant.h `matmul_q` NEON (f32 act) | 25344.5 | 25322.9 | 0.1 | 21.2 | 3.0e-07 |
| down 1024→2048 | 128 | q8_0 | `gq_matmul_f32` (exact, = GQ_EXACT=1) | 161479.8 | 160412.4 | 0.0 | 3.3 | 1.1e-06 |
| down 1024→2048 | 128 | q8_0 | `gq_matmul` (integer dot, default) | 11161.6 | 11161.9 | 0.2 | 48.1 | 4.4e-03 |
| down 1024→2048 | 128 | q4_k | `gq_matmul_f32` (exact, = GQ_EXACT=1) | 161239.3 | 159033.5 | 0.0 | 3.3 | 1.2e-06 |
| down 1024→2048 | 128 | q4_k | `gq_matmul` (integer dot, default) | 18089.1 | 18110.2 | 0.1 | 29.7 | 4.4e-03 |

## Attention decode (16 heads × 128, f32 KV)

| context | Colibri kernel | µs | KV GB/s | rel err |
|---|---|---|---|---|
| 1024 | bench_attn_f32 scalar (olmoe/qwen36 as shipped) | 769.4 | 21.8 | 9.9e-07 |
| 1024 | bench_attn_f32 lanes (matmul_f32.h dot) | 265.2 | 63.3 | 1.3e-06 |
| 4096 | bench_attn_f32 scalar (olmoe/qwen36 as shipped) | 3127.5 | 21.5 | 2.2e-06 |
| 4096 | bench_attn_f32 lanes (matmul_f32.h dot) | 1098.3 | 61.1 | 2.2e-06 |
