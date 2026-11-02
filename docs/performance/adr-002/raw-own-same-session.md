# Raw: own kernels, same session as the Colibri baseline

## 1 thread


- rayon threads: 1 (Candle reads RAYON_NUM_THREADS too)
- candle accelerate feature: false
- STREAM-style read bandwidth: 81.3 GB/s (1 thread), 81.7 GB/s (all threads)

## Expert-FFN matrices

Hot: one matrix, cache-resident. Cold: rotating over ≥ 512 MiB of copies (DRAM-bound).
GB/s is weight bytes over cold time. Error is max|y−ref|/max|ref| against f64 on the same
dequantized weights and f32 activations.

| shape | S | dtype | engine | hot µs | cold µs | cold GB/s | % STREAM | hot GFLOP/s | rel err | S=1 ≡ S=128 bits |
|---|---|---|---|---|---|---|---|---|---|---|
| gate/up 2048→1024 | 1 | f32 | own, rayon pool | 122.5 | 125.9 | 66.6 | 82% | 34.3 | 1.4e-7 | – |
| gate/up 2048→1024 | 1 | f32 | own | 111.2 | 117.6 | 71.3 | 87% | 37.7 | 1.4e-7 | yes |
| gate/up 2048→1024 | 1 | f32 | candle | 166.1 | 167.4 | 50.1 | 61% | 25.3 | 3.7e-7 | – |
| gate/up 2048→1024 | 1 | bf16 | own, rayon pool | 91.7 | 91.9 | 45.7 | 56% | 45.7 | 1.1e-7 | – |
| gate/up 2048→1024 | 1 | bf16 | own | 80.3 | 80.4 | 52.2 | 64% | 52.2 | 1.1e-7 | yes |
| gate/up 2048→1024 | 1 | bf16 | candle | 316.8 | 318.4 | 13.2 | 16% | 13.2 | 3.2e-3 | – |
| gate/up 2048→1024 | 1 | int8-row | own, rayon pool | 135.5 | 135.9 | 15.5 | 19% | 31.0 | 1.6e-7 | – |
| gate/up 2048→1024 | 1 | int8-row | own | 124.1 | 124.3 | 16.9 | 21% | 33.8 | 1.6e-7 | yes |
| gate/up 2048→1024 | 1 | q8_0 | own, rayon pool | 175.0 | 175.5 | 12.7 | 16% | 24.0 | 1.6e-7 | – |
| gate/up 2048→1024 | 1 | q8_0 | own | 163.7 | 164.2 | 13.6 | 17% | 25.6 | 1.6e-7 | yes |
| gate/up 2048→1024 | 1 | q8_0 | candle | 26.7 | 31.2 | 71.3 | 87% | 157.0 | 4.2e-3 | – |
| gate/up 2048→1024 | 1 | q4_k | own, rayon pool | 240.2 | 240.6 | 4.9 | 6% | 17.5 | 1.4e-7 | – |
| gate/up 2048→1024 | 1 | q4_k | own | 228.8 | 228.8 | 5.2 | 6% | 18.3 | 1.4e-7 | yes |
| gate/up 2048→1024 | 1 | q4_k | candle | 33.6 | 34.0 | 34.7 | 43% | 124.7 | 5.3e-3 | – |
| gate/up 2048→1024 | 128 | f32 | own | 13395.4 | 13411.2 | 0.6 | 1% | 40.1 | 1.5e-7 | yes |
| gate/up 2048→1024 | 128 | f32 | own deq + gemm crate | 5220.0 | 5281.9 | 1.6 | 2% | 102.8 | 1.4e-6 | NO |
| gate/up 2048→1024 | 128 | f32 | candle | 5241.9 | 5261.0 | 1.6 | 2% | 102.4 | 1.4e-6 | – |
| gate/up 2048→1024 | 128 | bf16 | own | 13490.5 | 13492.8 | 0.3 | 0% | 39.8 | 1.3e-7 | yes |
| gate/up 2048→1024 | 128 | bf16 | own deq + gemm crate | 5362.5 | 5388.1 | 0.8 | 1% | 100.1 | 2.0e-6 | NO |
| gate/up 2048→1024 | 128 | bf16 | candle | 5465.7 | 5507.0 | 0.8 | 1% | 98.2 | 3.0e-3 | – |
| gate/up 2048→1024 | 128 | int8-row | own | 13477.7 | 13471.0 | 0.2 | 0% | 39.8 | 1.3e-7 | yes |
| gate/up 2048→1024 | 128 | int8-row | own deq + gemm crate | 5486.1 | 5492.4 | 0.4 | 0% | 97.9 | 2.1e-6 | NO |
| gate/up 2048→1024 | 128 | q8_0 | own | 13530.9 | 13525.3 | 0.2 | 0% | 39.7 | 1.4e-7 | yes |
| gate/up 2048→1024 | 128 | q8_0 | own deq + gemm crate | 5415.7 | 5407.2 | 0.4 | 1% | 99.1 | 1.3e-6 | NO |
| gate/up 2048→1024 | 128 | q8_0 | candle | 2412.7 | 2534.5 | 0.9 | 1% | 222.5 | 4.3e-3 | – |
| gate/up 2048→1024 | 128 | q4_k | own | 13662.4 | 13679.8 | 0.1 | 0% | 39.3 | 1.3e-7 | yes |
| gate/up 2048→1024 | 128 | q4_k | own deq + gemm crate | 5498.7 | 5496.2 | 0.2 | 0% | 97.6 | 1.5e-6 | NO |
| gate/up 2048→1024 | 128 | q4_k | candle | 3004.3 | 3038.5 | 0.4 | 0% | 178.7 | 4.4e-3 | – |
| down 1024→2048 | 1 | f32 | own, rayon pool | 125.4 | 128.8 | 65.1 | 80% | 33.4 | 1.2e-7 | – |
| down 1024→2048 | 1 | f32 | own | 114.0 | 119.3 | 70.3 | 86% | 36.8 | 1.2e-7 | yes |
| down 1024→2048 | 1 | f32 | candle | 150.3 | 152.0 | 55.2 | 68% | 27.9 | 2.8e-7 | – |
| down 1024→2048 | 1 | bf16 | own, rayon pool | 93.2 | 93.6 | 44.8 | 55% | 45.0 | 1.4e-7 | – |
| down 1024→2048 | 1 | bf16 | own | 83.7 | 82.1 | 51.1 | 63% | 50.1 | 1.4e-7 | yes |
| down 1024→2048 | 1 | bf16 | candle | 299.2 | 301.1 | 13.9 | 17% | 14.0 | 3.4e-3 | – |
| down 1024→2048 | 1 | int8-row | own, rayon pool | 138.7 | 138.7 | 15.2 | 19% | 30.2 | 1.3e-7 | – |
| down 1024→2048 | 1 | int8-row | own | 126.9 | 127.0 | 16.6 | 20% | 33.0 | 1.3e-7 | yes |
| down 1024→2048 | 1 | q8_0 | own, rayon pool | 178.7 | 179.4 | 12.4 | 15% | 23.5 | 1.1e-7 | – |
| down 1024→2048 | 1 | q8_0 | own | 166.8 | 167.3 | 13.3 | 16% | 25.1 | 1.1e-7 | yes |
| down 1024→2048 | 1 | q8_0 | candle | 26.9 | 31.3 | 71.2 | 87% | 155.8 | 3.9e-3 | – |
| down 1024→2048 | 1 | q4_k | own, rayon pool | 244.3 | 244.3 | 4.8 | 6% | 17.2 | 1.2e-7 | – |
| down 1024→2048 | 1 | q4_k | own | 232.0 | 232.1 | 5.1 | 6% | 18.1 | 1.2e-7 | yes |
| down 1024→2048 | 1 | q4_k | candle | 33.8 | 34.0 | 34.7 | 42% | 124.3 | 4.2e-3 | – |
| down 1024→2048 | 128 | f32 | own | 13380.2 | 13371.8 | 0.6 | 1% | 40.1 | 1.3e-7 | yes |
| down 1024→2048 | 128 | f32 | own deq + gemm crate | 5123.5 | 5197.8 | 1.6 | 2% | 104.8 | 1.3e-6 | NO |
| down 1024→2048 | 128 | f32 | candle | 5134.7 | 5213.3 | 1.6 | 2% | 104.6 | 1.3e-6 | – |
| down 1024→2048 | 128 | bf16 | own | 13507.9 | 13529.2 | 0.3 | 0% | 39.7 | 1.2e-7 | yes |
| down 1024→2048 | 128 | bf16 | own deq + gemm crate | 5252.2 | 5255.6 | 0.8 | 1% | 102.2 | 2.1e-6 | NO |
| down 1024→2048 | 128 | bf16 | candle | 5382.4 | 5382.2 | 0.8 | 1% | 99.7 | 3.8e-3 | – |
| down 1024→2048 | 128 | int8-row | own | 13482.7 | 13485.5 | 0.2 | 0% | 39.8 | 1.1e-7 | yes |
| down 1024→2048 | 128 | int8-row | own deq + gemm crate | 5353.7 | 5354.0 | 0.4 | 0% | 100.3 | 1.3e-6 | NO |
| down 1024→2048 | 128 | q8_0 | own | 13441.3 | 13429.5 | 0.2 | 0% | 39.9 | 1.4e-7 | yes |
| down 1024→2048 | 128 | q8_0 | own deq + gemm crate | 5287.8 | 5287.3 | 0.4 | 1% | 101.5 | 1.0e-6 | NO |
| down 1024→2048 | 128 | q8_0 | candle | 2356.0 | 2483.7 | 0.9 | 1% | 227.9 | 4.4e-3 | – |
| down 1024→2048 | 128 | q4_k | own | 13595.4 | 13594.3 | 0.1 | 0% | 39.5 | 1.3e-7 | yes |
| down 1024→2048 | 128 | q4_k | own deq + gemm crate | 5394.4 | 5395.5 | 0.2 | 0% | 99.5 | 1.1e-6 | NO |
| down 1024→2048 | 128 | q4_k | candle | 2745.5 | 2744.8 | 0.4 | 1% | 195.5 | 4.2e-3 | – |

## Attention decode (16 heads × 128, f32 KV)

| context | engine | µs | KV GB/s | rel err |
|---|---|---|---|---|
| 1024 | own | 292.6 | 57.3 | 1.3e-6 |
| 1024 | candle | 319.6 | 52.5 | 9.3e-7 |
| 4096 | own | 1145.8 | 58.6 | 2.5e-6 |
| 4096 | candle | 1218.7 | 55.1 | 2.6e-6 |


## 11 threads


- rayon threads: 11 (Candle reads RAYON_NUM_THREADS too)
- candle accelerate feature: false
- STREAM-style read bandwidth: 80.5 GB/s (1 thread), 114.3 GB/s (all threads)

## Expert-FFN matrices

Hot: one matrix, cache-resident. Cold: rotating over ≥ 512 MiB of copies (DRAM-bound).
GB/s is weight bytes over cold time. Error is max|y−ref|/max|ref| against f64 on the same
dequantized weights and f32 activations.

| shape | S | dtype | engine | hot µs | cold µs | cold GB/s | % STREAM | hot GFLOP/s | rel err | S=1 ≡ S=128 bits |
|---|---|---|---|---|---|---|---|---|---|---|
| gate/up 2048→1024 | 1 | f32 | own, rayon pool | 52.0 | 93.0 | 90.2 | 79% | 80.7 | 1.4e-7 | – |
| gate/up 2048→1024 | 1 | f32 | own | 27.5 | 76.0 | 110.4 | 97% | 152.3 | 1.4e-7 | yes |
| gate/up 2048→1024 | 1 | f32 | candle | 166.5 | 167.7 | 50.0 | 44% | 25.2 | 3.7e-7 | – |
| gate/up 2048→1024 | 1 | bf16 | own, rayon pool | 33.0 | 58.2 | 72.1 | 63% | 126.9 | 1.1e-7 | – |
| gate/up 2048→1024 | 1 | bf16 | own | 15.4 | 40.1 | 104.6 | 92% | 272.1 | 1.1e-7 | yes |
| gate/up 2048→1024 | 1 | bf16 | candle | 315.5 | 317.0 | 13.2 | 12% | 13.3 | 3.2e-3 | – |
| gate/up 2048→1024 | 1 | int8-row | own, rayon pool | 39.6 | 44.5 | 47.3 | 41% | 105.8 | 1.6e-7 | – |
| gate/up 2048→1024 | 1 | int8-row | own | 22.8 | 25.8 | 81.5 | 71% | 183.7 | 1.6e-7 | yes |
| gate/up 2048→1024 | 1 | q8_0 | own, rayon pool | 45.8 | 48.6 | 45.8 | 40% | 91.6 | 1.6e-7 | – |
| gate/up 2048→1024 | 1 | q8_0 | own | 29.8 | 31.9 | 69.9 | 61% | 140.6 | 1.6e-7 | yes |
| gate/up 2048→1024 | 1 | q8_0 | candle | 9.4 | 24.1 | 92.5 | 81% | 445.4 | 4.2e-3 | – |
| gate/up 2048→1024 | 1 | q4_k | own, rayon pool | 64.5 | 61.9 | 19.1 | 17% | 65.0 | 1.4e-7 | – |
| gate/up 2048→1024 | 1 | q4_k | own | 41.8 | 42.8 | 27.6 | 24% | 100.4 | 1.4e-7 | yes |
| gate/up 2048→1024 | 1 | q4_k | candle | 11.9 | 15.0 | 78.9 | 69% | 353.2 | 5.3e-3 | – |
| gate/up 2048→1024 | 128 | f32 | own | 2294.3 | 2301.0 | 3.6 | 3% | 234.0 | 1.5e-7 | yes |
| gate/up 2048→1024 | 128 | f32 | own deq + gemm crate | 1457.3 | 1484.0 | 5.7 | 5% | 368.4 | 1.4e-6 | NO |
| gate/up 2048→1024 | 128 | f32 | candle | 1481.0 | 1503.0 | 5.6 | 5% | 362.5 | 1.4e-6 | – |
| gate/up 2048→1024 | 128 | bf16 | own | 2316.3 | 2320.3 | 1.8 | 2% | 231.8 | 1.3e-7 | yes |
| gate/up 2048→1024 | 128 | bf16 | own deq + gemm crate | 1580.5 | 1588.9 | 2.6 | 2% | 339.7 | 2.0e-6 | NO |
| gate/up 2048→1024 | 128 | bf16 | candle | 1787.1 | 1787.4 | 2.3 | 2% | 300.4 | 3.0e-3 | – |
| gate/up 2048→1024 | 128 | int8-row | own | 2324.5 | 2322.7 | 0.9 | 1% | 231.0 | 1.3e-7 | yes |
| gate/up 2048→1024 | 128 | int8-row | own deq + gemm crate | 1565.1 | 1559.6 | 1.3 | 1% | 343.0 | 2.1e-6 | NO |
| gate/up 2048→1024 | 128 | q8_0 | own | 2321.0 | 2351.2 | 0.9 | 1% | 231.3 | 1.4e-7 | yes |
| gate/up 2048→1024 | 128 | q8_0 | own deq + gemm crate | 1550.8 | 1549.0 | 1.4 | 1% | 346.2 | 1.3e-6 | NO |
| gate/up 2048→1024 | 128 | q8_0 | candle | 543.5 | 558.9 | 4.0 | 3% | 987.9 | 4.3e-3 | – |
| gate/up 2048→1024 | 128 | q4_k | own | 2357.8 | 2405.7 | 0.5 | 0% | 227.7 | 1.3e-7 | yes |
| gate/up 2048→1024 | 128 | q4_k | own deq + gemm crate | 1545.9 | 1550.8 | 0.8 | 1% | 347.3 | 1.5e-6 | NO |
| gate/up 2048→1024 | 128 | q4_k | candle | 1117.2 | 1106.4 | 1.1 | 1% | 480.5 | 4.4e-3 | – |
| down 1024→2048 | 1 | f32 | own, rayon pool | 43.5 | 92.5 | 90.6 | 79% | 96.4 | 1.2e-7 | – |
| down 1024→2048 | 1 | f32 | own | 21.6 | 75.3 | 111.4 | 97% | 194.0 | 1.2e-7 | yes |
| down 1024→2048 | 1 | f32 | candle | 150.7 | 152.0 | 55.2 | 48% | 27.8 | 2.8e-7 | – |
| down 1024→2048 | 1 | bf16 | own, rayon pool | 34.8 | 56.0 | 74.9 | 66% | 120.7 | 1.4e-7 | – |
| down 1024→2048 | 1 | bf16 | own | 15.7 | 39.9 | 105.1 | 92% | 267.0 | 1.4e-7 | yes |
| down 1024→2048 | 1 | bf16 | candle | 299.5 | 300.9 | 13.9 | 12% | 14.0 | 3.4e-3 | – |
| down 1024→2048 | 1 | int8-row | own, rayon pool | 40.2 | 42.8 | 49.2 | 43% | 104.3 | 1.3e-7 | – |
| down 1024→2048 | 1 | int8-row | own | 23.3 | 26.2 | 80.5 | 70% | 180.1 | 1.3e-7 | yes |
| down 1024→2048 | 1 | q8_0 | own, rayon pool | 45.4 | 52.7 | 42.3 | 37% | 92.4 | 1.1e-7 | – |
| down 1024→2048 | 1 | q8_0 | own | 30.3 | 32.4 | 68.7 | 60% | 138.5 | 1.1e-7 | yes |
| down 1024→2048 | 1 | q8_0 | candle | 9.0 | 23.9 | 93.2 | 81% | 468.2 | 3.9e-3 | – |
| down 1024→2048 | 1 | q4_k | own, rayon pool | 57.6 | 57.2 | 20.6 | 18% | 72.8 | 1.2e-7 | – |
| down 1024→2048 | 1 | q4_k | own | 42.1 | 43.0 | 27.4 | 24% | 99.6 | 1.2e-7 | yes |
| down 1024→2048 | 1 | q4_k | candle | 12.0 | 14.3 | 82.3 | 72% | 349.5 | 4.2e-3 | – |
| down 1024→2048 | 128 | f32 | own | 2051.6 | 2080.0 | 4.0 | 4% | 261.7 | 1.3e-7 | yes |
| down 1024→2048 | 128 | f32 | own deq + gemm crate | 1270.4 | 1298.8 | 6.5 | 6% | 422.6 | 1.3e-6 | NO |
| down 1024→2048 | 128 | f32 | candle | 1287.4 | 1313.7 | 6.4 | 6% | 417.0 | 1.3e-6 | – |
| down 1024→2048 | 128 | bf16 | own | 2091.7 | 2116.8 | 2.0 | 2% | 256.7 | 1.2e-7 | yes |
| down 1024→2048 | 128 | bf16 | own deq + gemm crate | 1389.5 | 1386.5 | 3.0 | 3% | 386.4 | 2.1e-6 | NO |
| down 1024→2048 | 128 | bf16 | candle | 1597.9 | 1592.0 | 2.6 | 2% | 336.0 | 3.8e-3 | – |
| down 1024→2048 | 128 | int8-row | own | 2091.2 | 2120.7 | 1.0 | 1% | 256.7 | 1.1e-7 | yes |
| down 1024→2048 | 128 | int8-row | own deq + gemm crate | 1377.0 | 1377.8 | 1.5 | 1% | 389.9 | 1.3e-6 | NO |
| down 1024→2048 | 128 | q8_0 | own | 2113.8 | 2136.0 | 1.0 | 1% | 254.0 | 1.4e-7 | yes |
| down 1024→2048 | 128 | q8_0 | own deq + gemm crate | 1363.0 | 1361.8 | 1.6 | 1% | 393.9 | 1.0e-6 | NO |
| down 1024→2048 | 128 | q8_0 | candle | 511.9 | 507.2 | 4.4 | 4% | 1048.8 | 4.4e-3 | – |
| down 1024→2048 | 128 | q4_k | own | 2148.3 | 2155.2 | 0.5 | 0% | 249.9 | 1.3e-7 | yes |
| down 1024→2048 | 128 | q4_k | own deq + gemm crate | 1362.9 | 1364.8 | 0.9 | 1% | 393.9 | 1.1e-6 | NO |
| down 1024→2048 | 128 | q4_k | candle | 773.1 | 782.8 | 1.5 | 1% | 694.4 | 4.2e-3 | – |

## Attention decode (16 heads × 128, f32 KV)

| context | engine | µs | KV GB/s | rel err |
|---|---|---|---|---|
| 1024 | own | 122.6 | 136.8 | 1.3e-6 |
| 1024 | candle | 373.2 | 45.0 | 9.3e-7 |
| 4096 | own | 585.1 | 114.7 | 2.5e-6 |
| 4096 | candle | 1258.9 | 53.3 | 2.6e-6 |


## Attention A/B, interleaved rounds (own rayon / own spinning pool / Colibri lanes)

### 11 threads

| context | engine | µs | KV GB/s | rel err |
|---|---|---|---|---|
| 1024 | own (rayon) | 122.0 | 137.5 |
| 1024 | own (spinning pool) | 99.9 | 168.0 |
| 4096 | own (rayon) | 576.5 | 116.4 |
| 4096 | own (spinning pool) | 523.0 | 128.3 |
| 1024 | bench_attn_f32 lanes (matmul_f32.h dot) | 106.7 | 157.2 | 1.3e-06 |
| 4096 | bench_attn_f32 lanes (matmul_f32.h dot) | 555.2 | 120.9 | 2.2e-06 |
| 1024 | own (rayon) | 120.5 | 139.2 |
| 1024 | own (spinning pool) | 102.6 | 163.5 |
| 4096 | own (rayon) | 574.4 | 116.8 |
| 4096 | own (spinning pool) | 520.7 | 128.9 |
| 1024 | bench_attn_f32 lanes (matmul_f32.h dot) | 105.7 | 158.8 | 1.3e-06 |
| 4096 | bench_attn_f32 lanes (matmul_f32.h dot) | 556.3 | 120.6 | 2.2e-06 |
| 1024 | own (rayon) | 119.1 | 140.8 |
| 1024 | own (spinning pool) | 101.5 | 165.4 |
| 4096 | own (rayon) | 580.4 | 115.6 |
| 4096 | own (spinning pool) | 519.3 | 129.2 |
| 1024 | bench_attn_f32 lanes (matmul_f32.h dot) | 106.3 | 157.8 | 1.3e-06 |
| 4096 | bench_attn_f32 lanes (matmul_f32.h dot) | 560.5 | 119.7 | 2.2e-06 |

### 1 thread (Colibri built with libomp)

| context | engine | µs | KV GB/s | rel err |
|---|---|---|---|---|
| 1024 | own (rayon) | 289.0 | 58.0 |
| 1024 | own (spinning pool) | 277.8 | 60.4 |
| 4096 | own (rayon) | 1138.3 | 59.0 |
| 4096 | own (spinning pool) | 1136.4 | 59.1 |
| 1024 | bench_attn_f32 lanes (matmul_f32.h dot) | 323.2 | 51.9 | 1.3e-06 |
| 4096 | bench_attn_f32 lanes (matmul_f32.h dot) | 1286.3 | 52.2 | 2.2e-06 |
| 1024 | own (rayon) | 288.8 | 58.1 |
| 1024 | own (spinning pool) | 277.3 | 60.5 |
| 4096 | own (rayon) | 1138.9 | 58.9 |
| 4096 | own (spinning pool) | 1119.0 | 60.0 |
| 1024 | bench_attn_f32 lanes (matmul_f32.h dot) | 330.4 | 50.8 | 1.3e-06 |
| 4096 | bench_attn_f32 lanes (matmul_f32.h dot) | 1307.0 | 51.3 | 2.2e-06 |
| 1024 | own (rayon) | 290.9 | 57.7 |
| 1024 | own (spinning pool) | 279.1 | 60.1 |
| 4096 | own (rayon) | 1140.9 | 58.8 |
| 4096 | own (spinning pool) | 1123.5 | 59.7 |
| 1024 | bench_attn_f32 lanes (matmul_f32.h dot) | 321.4 | 52.2 | 1.3e-06 |
| 4096 | bench_attn_f32 lanes (matmul_f32.h dot) | 1281.8 | 52.4 | 2.2e-06 |
