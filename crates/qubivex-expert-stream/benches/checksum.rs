//! The cost of verifying expert reads (ADR-006 amendment 4): `fault::checksum` on one
//! OLMoE-1B-7B Q4_K_M expert (3.5 MiB), in cache and cycling through 512 MiB of
//! copies so every call reads from DRAM, as a streamed expert arrives. `memcpy` of the
//! same bytes is the yardstick. Criterion reports GB/s.
//!
//! ```sh
//! cargo bench -p qubivex-expert-stream --bench checksum
//! ```

#![allow(missing_docs, clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use qubivex_expert_stream::fault::checksum;

const EXPERT: usize = 3_538_944;
const STREAM: usize = 512 << 20;

fn bench(c: &mut Criterion) {
    let one: Vec<u8> = (0..EXPERT).map(|i| (i * 31 % 251) as u8).collect();
    let many: Vec<u8> = (0..STREAM).map(|i| (i * 31 % 251) as u8).collect();
    let mut dst = vec![0u8; EXPERT];
    let mut g = c.benchmark_group("expert");
    g.throughput(Throughput::Bytes(EXPERT as u64));
    g.bench_function("checksum_cached", |b| b.iter(|| checksum(black_box(&one))));
    let mut i = 0;
    g.bench_function("checksum_dram", |b| {
        b.iter(|| {
            let off = (i * EXPERT) % (STREAM - EXPERT);
            i += 1;
            checksum(black_box(&many[off..off + EXPERT]))
        });
    });
    let mut j = 0;
    g.bench_function("memcpy_dram", |b| {
        b.iter(|| {
            let off = (j * EXPERT) % (STREAM - EXPERT);
            j += 1;
            dst.copy_from_slice(black_box(&many[off..off + EXPERT]));
            black_box(&dst);
        });
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
