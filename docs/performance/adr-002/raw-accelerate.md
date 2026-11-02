# ADR-002 spike results

- rayon threads: 11 (Candle reads RAYON_NUM_THREADS too)
- candle accelerate feature: true
- STREAM-style read bandwidth: 81.1 GB/s (1 thread), 116.3 GB/s (all threads)

## Expert-FFN matrices

Hot: one matrix, cache-resident. Cold: rotating over ≥ 512 MiB of copies (DRAM-bound).
GB/s is weight bytes over cold time. Error is max|y−ref|/max|ref| against f64 on the same
dequantized weights and f32 activations.

| shape | S | dtype | engine | hot µs | cold µs | cold GB/s | % STREAM | hot GFLOP/s | rel err | S=1 ≡ S=128 bits |
|---|---|---|---|---|---|---|---|---|---|---|
| gate/up 2048→1024 | 1 | f32 | own, rayon pool | 52.3 | 96.5 | 86.9 | 75% | 80.1 | 1.4e-7 | – |
| gate/up 2048→1024 | 1 | f32 | own | 25.3 | 68.2 | 122.9 | 106% | 165.6 | 1.4e-7 | yes |
| gate/up 2048→1024 | 1 | f32 | candle | 29.1 | 81.8 | 102.6 | 88% | 144.0 | 1.5e-7 | – |
| gate/up 2048→1024 | 1 | bf16 | own, rayon pool | 45.4 | 64.4 | 65.2 | 56% | 92.4 | 1.1e-7 | – |
| gate/up 2048→1024 | 1 | bf16 | own | 18.3 | 35.7 | 117.5 | 101% | 229.3 | 1.1e-7 | yes |
| gate/up 2048→1024 | 1 | bf16 | candle | 181.1 | 182.6 | 23.0 | 20% | 23.2 | 3.2e-3 | – |
| gate/up 2048→1024 | 1 | int8-row | own, rayon pool | 52.5 | 53.9 | 39.0 | 34% | 79.9 | 1.6e-7 | – |
| gate/up 2048→1024 | 1 | int8-row | own | 27.3 | 30.3 | 69.4 | 60% | 153.5 | 1.6e-7 | yes |
| gate/up 2048→1024 | 1 | q8_0 | own, rayon pool | 57.9 | 58.6 | 38.0 | 33% | 72.5 | 1.6e-7 | – |
| gate/up 2048→1024 | 1 | q8_0 | own | 35.8 | 37.5 | 59.5 | 51% | 117.2 | 1.6e-7 | yes |
| gate/up 2048→1024 | 1 | q8_0 | candle | 6.7 | 19.0 | 117.3 | 101% | 629.1 | 4.2e-3 | – |
| gate/up 2048→1024 | 1 | q4_k | own, rayon pool | 70.0 | 69.2 | 17.0 | 15% | 59.9 | 1.4e-7 | – |
| gate/up 2048→1024 | 1 | q4_k | own | 49.8 | 50.7 | 23.3 | 20% | 84.2 | 1.4e-7 | yes |
| gate/up 2048→1024 | 1 | q4_k | candle | 8.2 | 11.2 | 105.3 | 90% | 510.9 | 5.3e-3 | – |
| gate/up 2048→1024 | 128 | f32 | own | 2306.5 | 2323.7 | 3.6 | 3% | 232.8 | 1.5e-7 | yes |
| gate/up 2048→1024 | 128 | f32 | own deq + gemm crate | 1461.6 | 1500.9 | 5.6 | 5% | 367.3 | 1.4e-6 | NO |
| gate/up 2048→1024 | 128 | f32 | candle | 497.2 | 614.7 | 13.6 | 12% | 1079.8 | 1.4e-6 | – |
| gate/up 2048→1024 | 128 | bf16 | own | 2326.7 | 2325.2 | 1.8 | 2% | 230.7 | 1.3e-7 | yes |
| gate/up 2048→1024 | 128 | bf16 | own deq + gemm crate | 1592.2 | 1584.9 | 2.6 | 2% | 337.2 | 2.0e-6 | NO |
| gate/up 2048→1024 | 128 | bf16 | candle | 701.9 | 711.0 | 5.9 | 5% | 764.9 | 3.0e-3 | – |
| gate/up 2048→1024 | 128 | int8-row | own | 2320.5 | 2322.1 | 0.9 | 1% | 231.4 | 1.3e-7 | yes |
| gate/up 2048→1024 | 128 | int8-row | own deq + gemm crate | 1561.5 | 1559.8 | 1.3 | 1% | 343.8 | 2.1e-6 | NO |
| gate/up 2048→1024 | 128 | q8_0 | own | 2330.5 | 2329.7 | 1.0 | 1% | 230.4 | 1.4e-7 | yes |
| gate/up 2048→1024 | 128 | q8_0 | own deq + gemm crate | 1556.7 | 1556.0 | 1.4 | 1% | 344.9 | 1.3e-6 | NO |
| gate/up 2048→1024 | 128 | q8_0 | candle | 638.4 | 644.5 | 3.5 | 3% | 841.0 | 4.3e-3 | – |
| gate/up 2048→1024 | 128 | q4_k | own | 2344.8 | 2347.3 | 0.5 | 0% | 229.0 | 1.3e-7 | yes |
| gate/up 2048→1024 | 128 | q4_k | own deq + gemm crate | 1557.5 | 1557.3 | 0.8 | 1% | 344.7 | 1.5e-6 | NO |
| gate/up 2048→1024 | 128 | q4_k | candle | 1118.0 | 1118.7 | 1.1 | 1% | 480.2 | 4.4e-3 | – |
| down 1024→2048 | 1 | f32 | own, rayon pool | 54.5 | 97.7 | 85.9 | 74% | 77.0 | 1.2e-7 | – |
| down 1024→2048 | 1 | f32 | own | 25.5 | 68.3 | 122.8 | 106% | 164.2 | 1.2e-7 | yes |
| down 1024→2048 | 1 | f32 | candle | 31.5 | 90.0 | 93.2 | 80% | 133.3 | 1.4e-7 | – |
| down 1024→2048 | 1 | bf16 | own, rayon pool | 49.0 | 67.6 | 62.1 | 53% | 85.5 | 1.4e-7 | – |
| down 1024→2048 | 1 | bf16 | own | 18.7 | 36.0 | 116.6 | 100% | 224.7 | 1.4e-7 | yes |
| down 1024→2048 | 1 | bf16 | candle | 181.7 | 184.8 | 22.7 | 20% | 23.1 | 3.4e-3 | – |
| down 1024→2048 | 1 | int8-row | own, rayon pool | 55.9 | 55.2 | 38.1 | 33% | 75.0 | 1.3e-7 | – |
| down 1024→2048 | 1 | int8-row | own | 27.9 | 30.9 | 68.2 | 59% | 150.5 | 1.3e-7 | yes |
| down 1024→2048 | 1 | q8_0 | own, rayon pool | 61.5 | 64.0 | 34.8 | 30% | 68.2 | 1.1e-7 | – |
| down 1024→2048 | 1 | q8_0 | own | 36.5 | 38.2 | 58.3 | 50% | 115.0 | 1.1e-7 | yes |
| down 1024→2048 | 1 | q8_0 | candle | 6.5 | 18.9 | 117.8 | 101% | 645.3 | 3.9e-3 | – |
| down 1024→2048 | 1 | q4_k | own, rayon pool | 72.2 | 73.4 | 16.1 | 14% | 58.1 | 1.2e-7 | – |
| down 1024→2048 | 1 | q4_k | own | 50.5 | 51.5 | 22.9 | 20% | 83.0 | 1.2e-7 | yes |
| down 1024→2048 | 1 | q4_k | candle | 8.0 | 11.0 | 106.8 | 92% | 524.3 | 4.2e-3 | – |
| down 1024→2048 | 128 | f32 | own | 2061.7 | 2088.7 | 4.0 | 3% | 260.4 | 1.3e-7 | yes |
| down 1024→2048 | 128 | f32 | own deq + gemm crate | 1265.0 | 1289.5 | 6.5 | 6% | 424.4 | 1.3e-6 | NO |
| down 1024→2048 | 128 | f32 | candle | 441.0 | 538.2 | 15.6 | 13% | 1217.3 | 1.3e-6 | – |
| down 1024→2048 | 128 | bf16 | own | 2081.0 | 2098.5 | 2.0 | 2% | 258.0 | 1.2e-7 | yes |
| down 1024→2048 | 128 | bf16 | own deq + gemm crate | 1382.5 | 1399.0 | 3.0 | 3% | 388.3 | 2.1e-6 | NO |
| down 1024→2048 | 128 | bf16 | candle | 697.6 | 703.2 | 6.0 | 5% | 769.6 | 3.8e-3 | – |
| down 1024→2048 | 128 | int8-row | own | 2072.8 | 2110.8 | 1.0 | 1% | 259.0 | 1.1e-7 | yes |
| down 1024→2048 | 128 | int8-row | own deq + gemm crate | 1374.1 | 1377.7 | 1.5 | 1% | 390.7 | 1.3e-6 | NO |
| down 1024→2048 | 128 | q8_0 | own | 2113.9 | 2138.4 | 1.0 | 1% | 254.0 | 1.4e-7 | yes |
| down 1024→2048 | 128 | q8_0 | own deq + gemm crate | 1364.7 | 1372.9 | 1.6 | 1% | 393.4 | 1.0e-6 | NO |
| down 1024→2048 | 128 | q8_0 | candle | 571.7 | 576.7 | 3.9 | 3% | 939.1 | 4.4e-3 | – |
| down 1024→2048 | 128 | q4_k | own | 2143.2 | 2177.0 | 0.5 | 0% | 250.5 | 1.3e-7 | yes |
| down 1024→2048 | 128 | q4_k | own deq + gemm crate | 1367.0 | 1372.2 | 0.9 | 1% | 392.7 | 1.1e-6 | NO |
| down 1024→2048 | 128 | q4_k | candle | 841.4 | 847.7 | 1.4 | 1% | 638.1 | 4.2e-3 | – |

## Attention decode (16 heads × 128, f32 KV)

| context | engine | µs | KV GB/s | rel err |
|---|---|---|---|---|
| 1024 | own | 78.4 | 214.1 | 1.3e-6 |
| 1024 | candle | 221.2 | 75.9 | 9.3e-7 |
| 4096 | own | 582.4 | 115.2 | 2.5e-6 |
| 4096 | candle | 918.2 | 73.1 | 2.6e-6 |

## Bitwise determinism

Output hashes from fresh processes with RAYON_NUM_THREADS = [1, 2, 4, 11]; each process also
runs every case twice (a run-to-run mismatch shows as `0000000000000000`).

| case | run-to-run | across thread counts |
|---|---|---|
| attn ctx=1024 candle | same | same |
| attn ctx=1024 own | same | same |
| attn ctx=4096 candle | same | same |
| attn ctx=4096 own | same | same |
| down 1024→2048 S=1 bf16 candle | same | same |
| down 1024→2048 S=1 bf16 own | same | same |
| down 1024→2048 S=1 f32 candle | same | same |
| down 1024→2048 S=1 f32 own | same | same |
| down 1024→2048 S=1 int8-row own | same | same |
| down 1024→2048 S=1 q4_k candle | same | same |
| down 1024→2048 S=1 q4_k own | same | same |
| down 1024→2048 S=1 q8_0 candle | same | same |
| down 1024→2048 S=1 q8_0 own | same | same |
| down 1024→2048 S=128 bf16 candle | same | same |
| down 1024→2048 S=128 bf16 own | same | same |
| down 1024→2048 S=128 bf16 own deq + gemm crate | same | same |
| down 1024→2048 S=128 f32 candle | same | same |
| down 1024→2048 S=128 f32 own | same | same |
| down 1024→2048 S=128 f32 own deq + gemm crate | same | same |
| down 1024→2048 S=128 int8-row own | same | same |
| down 1024→2048 S=128 int8-row own deq + gemm crate | same | same |
| down 1024→2048 S=128 q4_k candle | same | same |
| down 1024→2048 S=128 q4_k own | same | same |
| down 1024→2048 S=128 q4_k own deq + gemm crate | same | same |
| down 1024→2048 S=128 q8_0 candle | same | same |
| down 1024→2048 S=128 q8_0 own | same | same |
| down 1024→2048 S=128 q8_0 own deq + gemm crate | same | same |
| gate/up 2048→1024 S=1 bf16 candle | same | same |
| gate/up 2048→1024 S=1 bf16 own | same | same |
| gate/up 2048→1024 S=1 f32 candle | same | same |
| gate/up 2048→1024 S=1 f32 own | same | same |
| gate/up 2048→1024 S=1 int8-row own | same | same |
| gate/up 2048→1024 S=1 q4_k candle | same | same |
| gate/up 2048→1024 S=1 q4_k own | same | same |
| gate/up 2048→1024 S=1 q8_0 candle | same | same |
| gate/up 2048→1024 S=1 q8_0 own | same | same |
| gate/up 2048→1024 S=128 bf16 candle | same | same |
| gate/up 2048→1024 S=128 bf16 own | same | same |
| gate/up 2048→1024 S=128 bf16 own deq + gemm crate | same | same |
| gate/up 2048→1024 S=128 f32 candle | same | same |
| gate/up 2048→1024 S=128 f32 own | same | same |
| gate/up 2048→1024 S=128 f32 own deq + gemm crate | same | same |
| gate/up 2048→1024 S=128 int8-row own | same | same |
| gate/up 2048→1024 S=128 int8-row own deq + gemm crate | same | same |
| gate/up 2048→1024 S=128 q4_k candle | same | same |
| gate/up 2048→1024 S=128 q4_k own | same | same |
| gate/up 2048→1024 S=128 q4_k own deq + gemm crate | same | same |
| gate/up 2048→1024 S=128 q8_0 candle | same | same |
| gate/up 2048→1024 S=128 q8_0 own | same | same |
| gate/up 2048→1024 S=128 q8_0 own deq + gemm crate | same | same |
