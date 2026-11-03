//! Replays a routing trace (from `qubivex bench --routing-trace`) through the expert
//! store at several per-layer slot counts and policies, against a global LRU of the
//! same size, and prints the hit rates.
//!
//! ```sh
//! cargo run --release -p qubivex-expert-stream --example replay -- trace.txt [slots...]
//! ```

use std::collections::HashMap;
use std::io::BufReader;

use qubivex_core::LayerIdx;
use qubivex_expert_stream::Policy;
use qubivex_expert_stream::replay::{global_lru_hits, layers, read_trace, replay};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: replay <trace> [slots per layer...]")?;
    let mut slot_counts: Vec<u32> = args.map(|a| a.parse()).collect::<Result<_, _>>()?;
    if slot_counts.is_empty() {
        slot_counts = vec![1, 2, 3, 4, 6, 8, 12, 16, 24, 32];
    }
    let trace = read_trace(BufReader::new(std::fs::File::open(&path)?))?;
    let ls = layers(&trace);
    println!(
        "{path}: {} acquires over {} layers ({} distinct experts)",
        trace.len(),
        ls.len(),
        ls.iter().map(|l| l.1).sum::<usize>()
    );
    let policies = [
        ("lru", Policy::Lru),
        ("lfru-64", Policy::Lfru { decay_window: 64 }),
        ("lfru-256", Policy::LFRU),
        ("lfru-1024", Policy::Lfru { decay_window: 1024 }),
        ("lfru-none", Policy::Lfru { decay_window: 0 }),
    ];
    print!("{:>6} {:>10}", "slots", "global-lru");
    for (name, _) in &policies {
        print!(" {name:>10}");
    }
    println!();
    #[allow(clippy::cast_precision_loss)]
    let pct = |h: u64| 100.0 * h as f64 / trace.len().max(1) as f64;
    for &n in &slot_counts {
        let slots: HashMap<LayerIdx, u32> = ls.iter().map(|&(l, _)| (l, n)).collect();
        print!(
            "{n:>6} {:>9.1}%",
            pct(global_lru_hits(&trace, ls.len() * n as usize))
        );
        for (_, p) in &policies {
            print!(" {:>9.1}%", pct(replay(&trace, &slots, *p)?.hits));
        }
        println!();
    }
    Ok(())
}
