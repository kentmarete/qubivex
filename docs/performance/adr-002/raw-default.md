# ADR-002 spike results

- rayon threads: 11 (Candle reads RAYON_NUM_THREADS too)
- candle accelerate feature: false
- STREAM-style read bandwidth: 82.9 GB/s (1 thread), 116.4 GB/s (all threads)

## Expert-FFN matrices

Hot: one matrix, cache-resident. Cold: rotating over ≥ 512 MiB of copies (DRAM-bound).
GB/s is weight bytes over cold time. Error is max|y−ref|/max|ref| against f64 on the same
dequantized weights and f32 activations.

| shape | S | dtype | engine | hot µs | cold µs | cold GB/s | % STREAM | hot GFLOP/s | rel err | S=1 ≡ S=128 bits |
|---|---|---|---|---|---|---|---|---|---|---|
| gate/up 2048→1024 | 1 | f32 | own, rayon pool | 52.5 | 96.2 | 87.2 | 75% | 80.0 | 1.4e-7 | – |
| gate/up 2048→1024 | 1 | f32 | own | 25.5 | 68.4 | 122.7 | 105% | 164.2 | 1.4e-7 | yes |
| gate/up 2048→1024 | 1 | f32 | candle | 167.6 | 168.7 | 49.7 | 43% | 25.0 | 3.7e-7 | – |
| gate/up 2048→1024 | 1 | bf16 | own, rayon pool | 46.1 | 64.5 | 65.0 | 56% | 90.9 | 1.1e-7 | – |
| gate/up 2048→1024 | 1 | bf16 | own | 18.3 | 35.7 | 117.5 | 101% | 229.3 | 1.1e-7 | yes |
| gate/up 2048→1024 | 1 | bf16 | candle | 317.6 | 319.0 | 13.1 | 11% | 13.2 | 3.2e-3 | – |
| gate/up 2048→1024 | 1 | int8-row | own, rayon pool | 52.5 | 54.5 | 38.6 | 33% | 79.9 | 1.6e-7 | – |
| gate/up 2048→1024 | 1 | int8-row | own | 27.3 | 30.3 | 69.4 | 60% | 153.7 | 1.6e-7 | yes |
| gate/up 2048→1024 | 1 | q8_0 | own, rayon pool | 58.3 | 57.9 | 38.5 | 33% | 71.9 | 1.6e-7 | – |
| gate/up 2048→1024 | 1 | q8_0 | own | 35.8 | 37.5 | 59.4 | 51% | 117.0 | 1.6e-7 | yes |
| gate/up 2048→1024 | 1 | q8_0 | candle | 6.7 | 19.0 | 117.0 | 101% | 629.1 | 4.2e-3 | – |
| gate/up 2048→1024 | 1 | q4_k | own, rayon pool | 70.4 | 70.8 | 16.7 | 14% | 59.6 | 1.4e-7 | – |
| gate/up 2048→1024 | 1 | q4_k | own | 49.8 | 50.8 | 23.2 | 20% | 84.2 | 1.4e-7 | yes |
| gate/up 2048→1024 | 1 | q4_k | candle | 8.2 | 11.2 | 105.6 | 91% | 513.6 | 5.3e-3 | – |
| gate/up 2048→1024 | 128 | f32 | own | 2300.0 | 2310.4 | 3.6 | 3% | 233.4 | 1.5e-7 | yes |
| gate/up 2048→1024 | 128 | f32 | own deq + gemm crate | 1462.6 | 1494.2 | 5.6 | 5% | 367.1 | 1.4e-6 | NO |
| gate/up 2048→1024 | 128 | f32 | candle | 1386.5 | 1410.8 | 5.9 | 5% | 387.2 | 1.4e-6 | – |
| gate/up 2048→1024 | 128 | bf16 | own | 2319.2 | 2328.6 | 1.8 | 2% | 231.5 | 1.3e-7 | yes |
| gate/up 2048→1024 | 128 | bf16 | own deq + gemm crate | 1588.0 | 1585.4 | 2.6 | 2% | 338.1 | 2.0e-6 | NO |
| gate/up 2048→1024 | 128 | bf16 | candle | 2327.2 | 2386.0 | 1.8 | 2% | 230.7 | 3.0e-3 | – |
| gate/up 2048→1024 | 128 | int8-row | own | 2322.2 | 2323.0 | 0.9 | 1% | 231.2 | 1.3e-7 | yes |
| gate/up 2048→1024 | 128 | int8-row | own deq + gemm crate | 1561.7 | 1571.4 | 1.3 | 1% | 343.8 | 2.1e-6 | NO |
| gate/up 2048→1024 | 128 | q8_0 | own | 2331.0 | 2333.5 | 1.0 | 1% | 230.3 | 1.4e-7 | yes |
| gate/up 2048→1024 | 128 | q8_0 | own deq + gemm crate | 1559.2 | 1565.3 | 1.4 | 1% | 344.3 | 1.3e-6 | NO |
| gate/up 2048→1024 | 128 | q8_0 | candle | 639.0 | 644.6 | 3.5 | 3% | 840.2 | 4.3e-3 | – |
| gate/up 2048→1024 | 128 | q4_k | own | 2342.9 | 2343.4 | 0.5 | 0% | 229.2 | 1.3e-7 | yes |
| gate/up 2048→1024 | 128 | q4_k | own deq + gemm crate | 1555.0 | 1560.8 | 0.8 | 1% | 345.2 | 1.5e-6 | NO |
| gate/up 2048→1024 | 128 | q4_k | candle | 1107.1 | 1104.9 | 1.1 | 1% | 484.9 | 4.4e-3 | – |
| down 1024→2048 | 1 | f32 | own, rayon pool | 55.0 | 99.3 | 84.5 | 73% | 76.2 | 1.2e-7 | – |
| down 1024→2048 | 1 | f32 | own | 25.7 | 68.3 | 122.8 | 106% | 163.4 | 1.2e-7 | yes |
| down 1024→2048 | 1 | f32 | candle | 150.7 | 151.9 | 55.2 | 47% | 27.8 | 2.8e-7 | – |
| down 1024→2048 | 1 | bf16 | own, rayon pool | 49.2 | 67.8 | 61.9 | 53% | 85.2 | 1.4e-7 | – |
| down 1024→2048 | 1 | bf16 | own | 18.7 | 36.0 | 116.6 | 100% | 224.2 | 1.4e-7 | yes |
| down 1024→2048 | 1 | bf16 | candle | 299.6 | 301.2 | 13.9 | 12% | 14.0 | 3.4e-3 | – |
| down 1024→2048 | 1 | int8-row | own, rayon pool | 55.9 | 57.0 | 37.0 | 32% | 75.0 | 1.3e-7 | – |
| down 1024→2048 | 1 | int8-row | own | 27.9 | 30.8 | 68.5 | 59% | 150.5 | 1.3e-7 | yes |
| down 1024→2048 | 1 | q8_0 | own, rayon pool | 62.3 | 63.7 | 35.0 | 30% | 67.3 | 1.1e-7 | – |
| down 1024→2048 | 1 | q8_0 | own | 36.5 | 38.2 | 58.4 | 50% | 115.0 | 1.1e-7 | yes |
| down 1024→2048 | 1 | q8_0 | candle | 6.6 | 18.8 | 118.3 | 102% | 633.1 | 3.9e-3 | – |
| down 1024→2048 | 1 | q4_k | own, rayon pool | 75.5 | 80.3 | 14.7 | 13% | 55.6 | 1.2e-7 | – |
| down 1024→2048 | 1 | q4_k | own | 50.5 | 51.5 | 22.9 | 20% | 83.0 | 1.2e-7 | yes |
| down 1024→2048 | 1 | q4_k | candle | 8.0 | 11.0 | 107.2 | 92% | 524.3 | 4.2e-3 | – |
| down 1024→2048 | 128 | f32 | own | 2059.2 | 2071.3 | 4.0 | 3% | 260.7 | 1.3e-7 | yes |
| down 1024→2048 | 128 | f32 | own deq + gemm crate | 1269.5 | 1290.5 | 6.5 | 6% | 422.9 | 1.3e-6 | NO |
| down 1024→2048 | 128 | f32 | candle | 1279.0 | 1324.0 | 6.3 | 5% | 419.8 | 1.3e-6 | – |
| down 1024→2048 | 128 | bf16 | own | 2093.2 | 2092.5 | 2.0 | 2% | 256.5 | 1.2e-7 | yes |
| down 1024→2048 | 128 | bf16 | own deq + gemm crate | 1389.8 | 1392.0 | 3.0 | 3% | 386.3 | 2.1e-6 | NO |
| down 1024→2048 | 128 | bf16 | candle | 2635.6 | 2562.6 | 1.6 | 1% | 203.7 | 3.8e-3 | – |
| down 1024→2048 | 128 | int8-row | own | 2072.8 | 2074.5 | 1.0 | 1% | 259.0 | 1.1e-7 | yes |
| down 1024→2048 | 128 | int8-row | own deq + gemm crate | 1379.3 | 1379.9 | 1.5 | 1% | 389.2 | 1.3e-6 | NO |
| down 1024→2048 | 128 | q8_0 | own | 2079.5 | 2079.1 | 1.1 | 1% | 258.2 | 1.4e-7 | yes |
| down 1024→2048 | 128 | q8_0 | own deq + gemm crate | 1367.4 | 1372.2 | 1.6 | 1% | 392.6 | 1.0e-6 | NO |
| down 1024→2048 | 128 | q8_0 | candle | 572.9 | 577.6 | 3.9 | 3% | 937.2 | 4.4e-3 | – |
| down 1024→2048 | 128 | q4_k | own | 2093.3 | 2109.4 | 0.6 | 0% | 256.5 | 1.3e-7 | yes |
| down 1024→2048 | 128 | q4_k | own deq + gemm crate | 1365.2 | 1374.7 | 0.9 | 1% | 393.2 | 1.1e-6 | NO |
| down 1024→2048 | 128 | q4_k | candle | 844.9 | 837.2 | 1.4 | 1% | 635.4 | 4.2e-3 | – |

## Attention decode (16 heads × 128, f32 KV)

| context | engine | µs | KV GB/s | rel err |
|---|---|---|---|---|
| 1024 | own | 77.3 | 216.9 | 1.3e-6 |
| 1024 | candle | 365.2 | 45.9 | 9.3e-7 |
| 4096 | own | 578.7 | 116.0 | 2.5e-6 |
| 4096 | candle | 1264.5 | 53.1 | 2.6e-6 |

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
