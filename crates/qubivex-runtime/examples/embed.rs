//! QUBIVEX inside another Rust program: no server, no CLI, no async runtime.
//!
//! ```text
//! cargo run -p qubivex-runtime --example embed -- <MODEL_DIR> "Hello"
//! ```

use std::error::Error;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{Engine, EngineOptions, GenerationEvent, GenerationRequest, Prompt};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().ok_or("usage: embed <MODEL_DIR> <PROMPT>")?);
    let prompt = args.next().ok_or("usage: embed <MODEL_DIR> <PROMPT>")?;

    let meta = ModelMetadata::load(&dir, &ArchitectureRegistry::with_builtins())?;
    let engine = Arc::new(Engine::load(meta, EngineOptions::default())?);

    // Each request gets a thread and its own KV cache; the weights are shared.
    let workers: Vec<_> = (0..2)
        .map(|_| {
            let engine = Arc::clone(&engine);
            let mut request = GenerationRequest::new(Prompt::Text(prompt.clone()));
            request.default_max_tokens = 16;
            std::thread::spawn(move || {
                let cancel = AtomicBool::new(false);
                let mut text = String::new();
                let outcome = engine.run_request(&request, &cancel, |event| {
                    if let GenerationEvent::Text(piece) = event {
                        text.push_str(&piece);
                    }
                    true
                });
                outcome.map(|o| (o, text))
            })
        })
        .collect();

    let mut out = std::io::stdout().lock();
    for worker in workers {
        let (outcome, text) = worker.join().map_err(|_| "a generation panicked")??;
        writeln!(out, "{text:?} ({outcome:?})")?;
    }
    Ok(())
}
