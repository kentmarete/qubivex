//! ADR-017 prototype: sequences sharing one forward pass get the bits they get alone.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use qubivex_model::{ArchitectureRegistry, ModelMetadata};
use qubivex_runtime::{Engine, EngineOptions, KvCache, Logits, Segment};

fn dir(id: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/models")
        .join(id)
}

fn engine(id: &str, cache: Option<u64>, context: usize) -> Engine {
    let meta = ModelMetadata::load(&dir(id), &ArchitectureRegistry::with_builtins()).unwrap();
    Engine::load(
        meta,
        EngineOptions {
            expert_cache_bytes: cache,
            max_context: Some(context),
            ..EngineOptions::default()
        },
    )
    .unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    u32::try_from(best).unwrap()
}

const ALL: [&str; 4] = [
    "olmoe-tiny",
    "olmoe-tiny-gguf",
    "qwen3-moe-tiny",
    "dense-tiny",
];

/// Prompts of different lengths, so the sequences sit at different positions.
const PROMPTS: [&[u32]; 4] = [
    &[72, 101, 108, 108, 111, 44, 32, 101, 120, 112],
    &[5, 9],
    &[200, 17, 33, 91, 4, 4, 4, 18, 77, 61, 3, 12, 99, 1, 2, 3, 40],
    &[11],
];

/// Prefills of different lengths and logits kinds, beside decode steps of sequences
/// already in progress, in one pass: every segment's logits equal its own pass's.
#[test]
fn a_mixed_batch_equals_separate_passes() {
    for id in ALL {
        let e = engine(id, None, 256);
        // Two sequences already part-way through, two starting.
        let mut alone: Vec<KvCache> = (0..4).map(|_| e.new_cache()).collect();
        let mut shared: Vec<KvCache> = (0..4).map(|_| e.new_cache()).collect();
        for i in [0, 2] {
            e.forward(PROMPTS[i], &mut alone[i], Logits::Last, None)
                .unwrap();
            e.forward(PROMPTS[i], &mut shared[i], Logits::Last, None)
                .unwrap();
        }
        let step: [(&[u32], Logits); 4] = [
            (&[7], Logits::Last),
            (PROMPTS[1], Logits::All),
            (&[8, 9, 10], Logits::Last),
            (PROMPTS[3], Logits::Last),
        ];
        let want: Vec<Vec<f32>> = alone
            .iter_mut()
            .zip(&step)
            .map(|(c, (t, l))| e.forward(t, c, *l, None).unwrap())
            .collect();
        let mut segments: Vec<Segment<'_>> = shared
            .iter_mut()
            .zip(&step)
            .map(|(cache, &(tokens, logits))| Segment {
                tokens,
                cache,
                logits,
            })
            .collect();
        let got = e.forward_batch(&mut segments).unwrap();
        assert_eq!(got.len(), 4, "{id}");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(bits(g), bits(w), "{id} segment {i}");
        }
        for (a, s) in alone.iter().zip(&shared) {
            assert_eq!(a.len(), s.len(), "{id}");
        }
    }
}

/// Greedy decoding of several sequences in lock-step, one pass per step, gives each
/// sequence the tokens it gets decoded alone; a sequence that stops leaves the batch.
#[test]
fn lock_step_decoding_gives_each_sequence_its_own_tokens() {
    const STEPS: [usize; 4] = [12, 5, 9, 12];
    for id in ALL {
        let e = engine(id, None, 256);
        let want: Vec<Vec<u32>> = PROMPTS
            .iter()
            .zip(STEPS)
            .map(|(p, n)| {
                let mut c = e.new_cache();
                let mut next = argmax(&e.forward(p, &mut c, Logits::Last, None).unwrap());
                let mut out = vec![next];
                while out.len() < n {
                    next = argmax(&e.forward(&[next], &mut c, Logits::Last, None).unwrap());
                    out.push(next);
                }
                out
            })
            .collect();

        let mut caches: Vec<KvCache> = (0..4).map(|_| e.new_cache()).collect();
        let mut got: Vec<Vec<u32>> = vec![Vec::new(); 4];
        // The first pass prefills every prompt together.
        let mut feed: Vec<Vec<u32>> = PROMPTS.iter().map(|p| p.to_vec()).collect();
        loop {
            let live: Vec<usize> = (0..4).filter(|&i| got[i].len() < STEPS[i]).collect();
            if live.is_empty() {
                break;
            }
            let mut segments: Vec<Segment<'_>> = caches
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| live.contains(i))
                .map(|(i, cache)| Segment {
                    tokens: &feed[i],
                    cache,
                    logits: Logits::Last,
                })
                .collect();
            let logits = e.forward_batch(&mut segments).unwrap();
            drop(segments);
            for (&i, l) in live.iter().zip(&logits) {
                let next = argmax(l);
                got[i].push(next);
                feed[i] = vec![next];
            }
        }
        assert_eq!(got, want, "{id}");
    }
}

/// A segment that cannot run fails the pass and leaves every cache as it was, so the
/// same batch without it gives the same bits as if the failure never happened.
#[test]
fn a_failed_batch_rolls_back_every_cache() {
    for id in ALL {
        let e = engine(id, None, 16);
        let mut a = e.new_cache();
        let mut b = e.new_cache();
        e.forward(PROMPTS[0], &mut a, Logits::Last, None).unwrap();
        let mut full = e.new_cache();
        e.forward(&[1; 15], &mut full, Logits::Last, None).unwrap();
        let mut segments = [
            Segment {
                tokens: &[3],
                cache: &mut a,
                logits: Logits::Last,
            },
            Segment {
                tokens: PROMPTS[1],
                cache: &mut b,
                logits: Logits::Last,
            },
            Segment {
                tokens: &[1, 2],
                cache: &mut full,
                logits: Logits::Last,
            },
        ];
        assert!(e.forward_batch(&mut segments).is_err(), "{id}");
        assert_eq!((a.len(), b.len(), full.len()), (10, 0, 15), "{id}");

        // An out-of-range token fails inside the pass, after the checks: still nothing.
        let mut segments = [
            Segment {
                tokens: &[3],
                cache: &mut a,
                logits: Logits::Last,
            },
            Segment {
                tokens: &[u32::MAX],
                cache: &mut b,
                logits: Logits::Last,
            },
        ];
        assert!(e.forward_batch(&mut segments).is_err(), "{id}");
        assert_eq!((a.len(), b.len()), (10, 0), "{id}");

        let mut fresh = e.new_cache();
        e.forward(PROMPTS[0], &mut fresh, Logits::Last, None)
            .unwrap();
        let want = e.forward(&[3], &mut fresh, Logits::Last, None).unwrap();
        let mut segments = [Segment {
            tokens: &[3],
            cache: &mut a,
            logits: Logits::Last,
        }];
        let got = e.forward_batch(&mut segments).unwrap();
        assert_eq!(bits(&got[0]), bits(&want), "{id}");
    }
}

/// An expert cache of one expert: a batch routes to the union of its sequences'
/// experts, which must wait for the slot, not fail, and keep the bits.
#[test]
fn a_one_expert_cache_keeps_the_bits() {
    for id in ["olmoe-tiny", "olmoe-tiny-gguf", "qwen3-moe-tiny"] {
        let full = engine(id, None, 256);
        let largest = full.metadata().bytes().largest_expert;
        let small = engine(id, Some(largest), 256);
        let run = |e: &Engine| {
            let mut caches: Vec<KvCache> = (0..4).map(|_| e.new_cache()).collect();
            let mut segments: Vec<Segment<'_>> = caches
                .iter_mut()
                .zip(PROMPTS)
                .map(|(cache, tokens)| Segment {
                    tokens,
                    cache,
                    logits: Logits::Last,
                })
                .collect();
            e.forward_batch(&mut segments).unwrap()
        };
        let (a, b) = (run(&full), run(&small));
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(bits(x), bits(y), "{id}");
        }
    }
}

#[test]
fn an_empty_batch_and_empty_segments_return_nothing() {
    let e = engine("olmoe-tiny", None, 256);
    assert!(e.forward_batch(&mut []).unwrap().is_empty());
    let mut a = e.new_cache();
    let mut b = e.new_cache();
    let mut segments = [
        Segment {
            tokens: &[],
            cache: &mut a,
            logits: Logits::Last,
        },
        Segment {
            tokens: &[5],
            cache: &mut b,
            logits: Logits::Last,
        },
    ];
    let out = e.forward_batch(&mut segments).unwrap();
    assert!(out[0].is_empty());
    assert!(!out[1].is_empty());
    assert_eq!((a.len(), b.len()), (0, 1));
}

/// Greedy decoding of `n` tokens after `prompt`, through [`Engine::forward`].
fn decode(e: &Engine, prompt: &[u32], n: usize) -> Vec<u32> {
    let mut c = e.new_cache();
    let mut next = argmax(&e.forward(prompt, &mut c, Logits::Last, None).unwrap());
    let mut out = vec![next];
    while out.len() < n {
        next = argmax(&e.forward(&[next], &mut c, Logits::Last, None).unwrap());
        out.push(next);
    }
    out
}

/// Threads calling `forward` at once share passes at the engine's turn (ADR-017
/// section 2), and each still gets the tokens it gets alone, with a full expert cache
/// and with one expert slot (where a pass waits for the union of its experts).
#[test]
fn concurrent_callers_share_passes_and_keep_their_tokens() {
    const STEPS: [usize; 4] = [24, 7, 16, 24];
    let mut widest = 0;
    for id in ALL {
        let full = engine(id, None, 256);
        let largest = full.metadata().bytes().largest_expert;
        let mut engines = vec![full];
        if id != "dense-tiny" {
            engines.push(engine(id, Some(largest), 256));
        }
        for e in &engines {
            let want: Vec<Vec<u32>> = PROMPTS
                .iter()
                .zip(STEPS)
                .map(|(p, n)| decode(e, p, n))
                .collect();
            let before = e.batch_stats();
            let start = std::sync::Barrier::new(PROMPTS.len());
            let got: Vec<Vec<u32>> = std::thread::scope(|s| {
                let handles: Vec<_> = PROMPTS
                    .iter()
                    .zip(STEPS)
                    .map(|(p, n)| {
                        let start = &start;
                        s.spawn(move || {
                            start.wait();
                            decode(e, p, n)
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            assert_eq!(got, want, "{id}");
            let after = e.batch_stats();
            // Every step ran exactly once: 4 prompts and the decode steps after them.
            let steps: usize = STEPS.iter().sum();
            assert_eq!(after.steps - before.steps, steps as u64, "{id}");
            widest = widest.max(after.widest);
        }
    }
    // Four threads starting together post steps while each pass runs.
    assert!(widest > 1, "no pass combined two callers");
}

/// A caller whose own step is bad (a token out of range, a full context) gets its
/// error alone; the callers sharing its passes get their tokens unchanged.
#[test]
fn a_bad_step_fails_only_its_caller() {
    for id in ALL {
        let e = engine(id, None, 32);
        let want: Vec<Vec<u32>> = PROMPTS[..3].iter().map(|p| decode(&e, p, 8)).collect();
        let start = std::sync::Barrier::new(5);
        std::thread::scope(|s| {
            let good: Vec<_> = PROMPTS[..3]
                .iter()
                .map(|p| {
                    let (e, start) = (&e, &start);
                    s.spawn(move || {
                        start.wait();
                        decode(e, p, 8)
                    })
                })
                .collect();
            let (e1, s1) = (&e, &start);
            let bad_token = s.spawn(move || {
                s1.wait();
                let mut c = e1.new_cache();
                let mut errs = 0;
                for _ in 0..8 {
                    let r = e1.forward(&[3, u32::MAX], &mut c, Logits::Last, None);
                    errs += usize::from(r.is_err());
                }
                (errs, c.len())
            });
            let (e2, s2) = (&e, &start);
            let full = s.spawn(move || {
                let mut c = e2.new_cache();
                e2.forward(&[1; 30], &mut c, Logits::Last, None).unwrap();
                s2.wait();
                let mut errs = 0;
                for _ in 0..8 {
                    let r = e2.forward(&[1, 2, 3], &mut c, Logits::Last, None);
                    errs += usize::from(r.is_err());
                }
                (errs, c.len())
            });
            let got: Vec<Vec<u32>> = good.into_iter().map(|h| h.join().unwrap()).collect();
            assert_eq!(got, want, "{id}");
            assert_eq!(bad_token.join().unwrap(), (8, 0), "{id}");
            assert_eq!(full.join().unwrap(), (8, 30), "{id}");
        });
    }
}
