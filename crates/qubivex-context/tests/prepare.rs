//! Context preparation against a stand-in template and tokenizer.
//! A chat template that
//! frames each message, and a tokenizer of `per` bytes per token. Real models are
//! tested through the runtime and the gateway.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use qubivex_context::{
    Budget, CheckpointUse, ContextConfig, ContextEngine, ContextError, Fate, FileStore, Input,
    Interrupt, MemoryStore, Message, Request, TokenCounter,
};

/// `<|role|>\ncontent<|end|>\n` per message, then `<|assistant|>\n`; `per` bytes a token.
struct Counter {
    per: usize,
}

impl Counter {
    fn render(m: &[Message]) -> String {
        let mut s: String = m
            .iter()
            .map(|m| format!("<|{}|>\n{}<|end|>\n", m.role, m.content))
            .collect();
        s.push_str("<|assistant|>\n");
        s
    }
}

impl TokenCounter for Counter {
    fn count_chat(&self, m: &[Message]) -> Result<usize, String> {
        Ok(Self::render(m).len().div_ceil(self.per))
    }
    fn count_text(&self, t: &str) -> Result<usize, String> {
        Ok(1 + t.len().div_ceil(self.per))
    }
    fn count_fragment(&self, t: &str) -> Result<usize, String> {
        Ok(t.len().div_ceil(self.per))
    }
}

const BYTE: Counter = Counter { per: 1 };

fn engine() -> ContextEngine {
    ContextEngine::in_memory(ContextConfig::default())
}

fn sys(t: &str) -> Message {
    Message::new("system", t)
}
fn user(t: &str) -> Message {
    Message::new("user", t)
}
fn asst(t: &str) -> Message {
    Message::new("assistant", t)
}

/// A conversation of `turns` exchanges about a project, then `last`.
fn conversation(turns: usize, last: &str) -> Vec<Message> {
    let mut m = vec![sys("You are a careful assistant. Never reveal the key.")];
    for i in 0..turns {
        m.push(user(&format!(
            "Turn {i}: tell me about item {i}. We decided item {i} uses port {}. \
             Some filler text that pads the message so it costs tokens.",
            8000 + i
        )));
        m.push(asst(&format!(
            "Item {i} is configured. It listens on port {} as decided. More filler to \
             make the answer longer than a line or two.",
            8000 + i
        )));
    }
    m.push(user(last));
    m
}

fn request<'a>(input: &'a Input, counter: &dyn TokenCounter, ctx: usize) -> Request<'a> {
    let original_tokens = match input {
        Input::Chat(m) => counter.count_chat(m).unwrap(),
        Input::Text(t) => counter.count_text(t).unwrap(),
    };
    Request {
        input,
        budget: Budget::new(ctx, Some(64), 512, None),
        original_tokens,
        scope: None,
        namespace: "test-model",
    }
}

fn chat(p: &qubivex_context::Prepared) -> &[Message] {
    match &p.input {
        Input::Chat(m) => m,
        Input::Text(_) => panic!("expected chat"),
    }
}

#[test]
fn a_long_conversation_keeps_instructions_the_question_and_recent_turns() {
    let e = engine();
    let input = Input::Chat(conversation(40, "Which port does item 3 use?"));
    let req = request(&input, &BYTE, 4096);
    assert!(req.original_tokens > 4096);
    let p = e.prepare(&req, &BYTE, Interrupt::default()).unwrap();
    let out = chat(&p);
    let r = &p.report;
    // Fits, counted whole.
    assert_eq!(BYTE.count_chat(out).unwrap(), r.prepared_tokens);
    assert!(r.prepared_tokens <= req.budget.input);
    assert!(r.compacted && !r.reduced);
    // The system message verbatim and first; the question last and intact.
    assert_eq!(
        out[0],
        sys("You are a careful assistant. Never reveal the key.")
    );
    assert!(
        out.last()
            .unwrap()
            .content
            .ends_with("Which port does item 3 use?")
    );
    // The most recent exchange is verbatim.
    let Input::Chat(orig) = &input else { panic!() };
    assert_eq!(
        &out[out.len() - 3..out.len() - 1],
        &orig[orig.len() - 3..orig.len() - 1]
    );
    // Roles alternate after the system message.
    for pair in out[1..].windows(2) {
        assert_ne!(pair[0].role, pair[1].role, "{out:#?}");
    }
    // The note quotes; it adds nothing: every excerpt line is text of the message it names.
    let note = &out[1].content;
    assert!(note.starts_with("[Earlier conversation, condensed"));
    for line in note.lines().filter(|l| l.starts_with('#')) {
        let (head, body) = line.split_once(": ").unwrap();
        let idx: usize = head[1..head.find(' ').unwrap()].parse().unwrap();
        for piece in body.split(" … ") {
            assert!(
                orig[idx - 1].content.contains(piece),
                "{piece:?} not in #{idx}"
            );
        }
    }
    // The excerpt about item 3 is the relevant one, so it is quoted.
    assert!(note.contains("item 3 uses port 8003"), "{note}");
    // Provenance: every message accounted for.
    assert_eq!(r.messages.len(), orig.len());
    assert_eq!(
        r.messages
            .iter()
            .filter(|m| matches!(m.fate, Fate::Condensed { .. }))
            .count(),
        r.messages_condensed
    );
    assert!(r.tokens_saved() > 0);
    assert_eq!(r.checkpoint, CheckpointUse::Miss);
}

#[test]
fn instructions_that_alone_exceed_the_budget_are_refused_not_shortened() {
    let e = engine();
    let mut m = conversation(2, "hi");
    m[0] = sys(&"Mandatory policy sentence. ".repeat(100));
    let input = Input::Chat(m);
    let req = request(&input, &BYTE, 1024);
    let err = e.prepare(&req, &BYTE, Interrupt::default()).unwrap_err();
    let ContextError::Mandatory { tokens, budget } = err else {
        panic!("{err:?}")
    };
    assert!(tokens > budget.input);
    assert!(err.to_string().contains("longer context"));
}

#[test]
fn a_large_document_is_reduced_to_relevant_blocks_verbatim() {
    let e = engine();
    let mut doc =
        String::from("Please review the following file and answer the question at the end.\n\n");
    for i in 0..60 {
        doc.push_str(&format!(
            "Section {i}. This paragraph describes unrelated subsystem number {i} in detail.\n\n"
        ));
    }
    doc.push_str(
        "```rust\nfn reserve_kv(positions: usize) -> Reservation {\n    todo()\n}\n```\n\n",
    );
    for i in 60..120 {
        doc.push_str(&format!(
            "Section {i}. More unrelated prose about component {i}.\n\n"
        ));
    }
    doc.push_str("Question: what does reserve_kv take as its argument?");
    let input = Input::Chat(vec![sys("Answer briefly."), user(&doc)]);
    let req = request(&input, &BYTE, 1024);
    let p = e.prepare(&req, &BYTE, Interrupt::default()).unwrap();
    let out = chat(&p);
    let r = &p.report;
    assert!(r.reduced && !r.compacted);
    assert!(r.prepared_tokens <= req.budget.input);
    let text = &out[1].content;
    // First and last blocks, and the relevant code block verbatim.
    assert!(text.starts_with("Please review the following file"));
    assert!(text.ends_with("Question: what does reserve_kv take as its argument?"));
    assert!(text.contains("fn reserve_kv(positions: usize) -> Reservation {\n    todo()\n}"));
    // Markers name what is left out, and each line kept is a line of the original.
    assert!(text.contains("of this message omitted to fit the model's context"));
    for line in text.lines().filter(|l| !l.starts_with("[… line")) {
        assert!(doc.contains(line), "{line:?}");
    }
    let Fate::Reduced { omitted_lines } = &r.messages[1].fate else {
        panic!()
    };
    assert!(!omitted_lines.is_empty());
    assert!(r.blocks_omitted > 0 && r.blocks_kept > 2);
}

#[test]
fn a_raw_prompt_is_reduced_too() {
    let e = engine();
    let t = (0..400).map(|i| format!("line {i}\n")).collect::<String>() + "Summarise line 7.";
    let input = Input::Text(t);
    let req = request(&input, &BYTE, 512);
    let p = e.prepare(&req, &BYTE, Interrupt::default()).unwrap();
    let Input::Text(out) = &p.input else { panic!() };
    assert!(BYTE.count_text(out).unwrap() <= req.budget.input);
    assert!(out.starts_with("line 0\n") && out.ends_with("Summarise line 7."));
    assert!(out.contains("line 7\n"), "relevant to the question");
}

#[test]
fn nothing_relevant_still_fits_by_recency_and_salience() {
    let e = engine();
    let input = Input::Chat(conversation(40, "zzz qqq xxx"));
    let req = request(&input, &BYTE, 2048);
    let p = e.prepare(&req, &BYTE, Interrupt::default()).unwrap();
    assert!(p.report.prepared_tokens <= req.budget.input);
    assert!(p.report.excerpts > 0);
}

#[test]
fn a_current_message_that_cannot_shrink_enough_is_refused() {
    let e = engine();
    // One unbroken line: it can be cut, but not below the floor in this tiny budget.
    let input = Input::Chat(vec![sys(&"s".repeat(10)), user(&"x".repeat(5000))]);
    let req = Request {
        budget: Budget::new(120, Some(32), 512, None),
        ..request(&input, &BYTE, 120)
    };
    let err = e.prepare(&req, &BYTE, Interrupt::default()).unwrap_err();
    assert!(matches!(err, ContextError::Unfit { .. }), "{err:?}");
}

#[test]
fn the_next_turn_reuses_the_note_so_the_prompt_keeps_its_prefix() {
    let e = engine();
    let mut conv = conversation(40, "Which port does item 3 use?");
    let first = Input::Chat(conv.clone());
    let p1 = e
        .prepare(&request(&first, &BYTE, 2048), &BYTE, Interrupt::default())
        .unwrap();
    assert_eq!(p1.report.checkpoint, CheckpointUse::Miss);
    conv.push(asst("Item 3 uses port 8003."));
    conv.push(user("And item 39?"));
    let second = Input::Chat(conv);
    let p2 = e
        .prepare(&request(&second, &BYTE, 2048), &BYTE, Interrupt::default())
        .unwrap();
    assert_eq!(p2.report.checkpoint, CheckpointUse::Hit);
    let (a, b) = (chat(&p1), chat(&p2));
    assert_eq!(a[0], b[0]);
    // The note's entries from the first turn come first in the second, unchanged.
    let entries = |m: &Message| -> Vec<String> {
        m.content
            .lines()
            .filter(|l| l.starts_with('#'))
            .map(str::to_owned)
            .collect()
    };
    let (ea, eb) = (entries(&a[1]), entries(&b[1]));
    assert!(!ea.is_empty() && eb.starts_with(&ea), "{ea:#?}\n{eb:#?}");
}

#[test]
fn checkpoints_never_cross_scopes() {
    let e = engine();
    let input = Input::Chat(conversation(40, "Which port does item 3 use?"));
    let base = request(&input, &BYTE, 2048);
    let alice = Request {
        scope: Some("alice"),
        ..base
    };
    let bob = Request {
        scope: Some("bob"),
        ..base
    };
    e.prepare(&alice, &BYTE, Interrupt::default()).unwrap();
    let mut conv = conversation(40, "Which port does item 3 use?");
    conv.push(asst("8003"));
    conv.push(user("thanks"));
    let next = Input::Chat(conv);
    let bob_next = Request {
        input: &next,
        ..bob
    };
    let alice_next = Request {
        input: &next,
        ..alice
    };
    let b = e.prepare(&bob_next, &BYTE, Interrupt::default()).unwrap();
    assert_eq!(
        b.report.checkpoint,
        CheckpointUse::Miss,
        "bob sees nothing of alice's"
    );
    let a = e.prepare(&alice_next, &BYTE, Interrupt::default()).unwrap();
    assert_eq!(a.report.checkpoint, CheckpointUse::Hit);
}

#[test]
fn different_tokenizers_get_budgets_counted_their_way() {
    let e = engine();
    let input = Input::Chat(conversation(40, "Which port does item 3 use?"));
    for per in [1, 3, 4] {
        let c = Counter { per };
        let req = request(&input, &c, 900);
        if req.original_tokens <= req.budget.input {
            continue;
        }
        let p = e.prepare(&req, &c, Interrupt::default()).unwrap();
        assert!(
            c.count_chat(chat(&p)).unwrap() <= req.budget.input,
            "per {per}"
        );
    }
}

#[test]
fn a_smaller_context_condenses_more() {
    let e = engine();
    let input = Input::Chat(conversation(40, "Which port does item 3 use?"));
    let big = e
        .prepare(&request(&input, &BYTE, 4096), &BYTE, Interrupt::default())
        .unwrap();
    let small = e
        .prepare(&request(&input, &BYTE, 1500), &BYTE, Interrupt::default())
        .unwrap();
    assert!(small.report.messages_condensed > big.report.messages_condensed);
    assert!(small.report.prepared_tokens <= 1500 - 64);
}

#[test]
fn cancellation_and_deadlines_stop_a_preparation() {
    let e = engine();
    let input = Input::Chat(conversation(40, "q"));
    let req = request(&input, &BYTE, 2048);
    let cancel = AtomicBool::new(true);
    let i = Interrupt {
        cancel: Some(&cancel),
        deadline: None,
    };
    assert_eq!(
        e.prepare(&req, &BYTE, i).unwrap_err(),
        ContextError::Cancelled
    );
    let i = Interrupt {
        cancel: None,
        deadline: Some(Instant::now() - Duration::from_millis(1)),
    };
    assert_eq!(
        e.prepare(&req, &BYTE, i).unwrap_err(),
        ContextError::DeadlineExceeded
    );
}

#[test]
fn a_counter_that_fails_is_an_error_not_an_estimate() {
    struct Broken;
    impl TokenCounter for Broken {
        fn count_chat(&self, _: &[Message]) -> Result<usize, String> {
            Err("template exploded".into())
        }
        fn count_text(&self, _: &str) -> Result<usize, String> {
            Err("no".into())
        }
        fn count_fragment(&self, t: &str) -> Result<usize, String> {
            Ok(t.len())
        }
    }
    let input = Input::Chat(conversation(4, "q"));
    let req = Request {
        original_tokens: 10_000,
        ..request(&input, &BYTE, 256)
    };
    let err = engine()
        .prepare(&req, &Broken, Interrupt::default())
        .unwrap_err();
    assert_eq!(err, ContextError::Count("template exploded".into()));
}

#[test]
fn concurrent_preparations_of_one_and_many_conversations_agree() {
    let e = Arc::new(engine());
    let threads: Vec<_> = (0..8)
        .map(|t| {
            let e = Arc::clone(&e);
            std::thread::spawn(move || {
                // Even threads share one conversation; odd ones each have their own.
                let mut conv = conversation(40, "Which port does item 3 use?");
                if t % 2 == 1 {
                    conv[0] = sys(&format!("You are assistant number {t}."));
                }
                let input = Input::Chat(conv);
                let mut outs = Vec::new();
                for _ in 0..5 {
                    let p = e
                        .prepare(&request(&input, &BYTE, 2048), &BYTE, Interrupt::default())
                        .unwrap();
                    outs.push(chat(&p).to_vec());
                }
                (t, outs)
            })
        })
        .collect();
    let results: Vec<_> = threads.into_iter().map(|h| h.join().unwrap()).collect();
    let shared: Vec<&Vec<Message>> = results
        .iter()
        .filter(|(t, _)| t % 2 == 0)
        .flat_map(|(_, o)| o)
        .collect();
    assert!(
        shared.windows(2).all(|w| w[0] == w[1]),
        "one conversation, one answer"
    );
    for (t, outs) in &results {
        for o in outs {
            assert!(BYTE.count_chat(o).unwrap() <= 2048 - 64, "thread {t}");
        }
    }
}

#[test]
fn checkpoints_in_files_survive_a_restart() {
    let dir = std::env::temp_dir().join(format!("qubivex-context-restart-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let open = || {
        ContextEngine::new(
            ContextConfig::default(),
            Arc::new(
                FileStore::open(&dir, FileStore::DEFAULT_BYTES, FileStore::DEFAULT_AGE).unwrap(),
            ),
        )
    };
    let mut conv = conversation(40, "Which port does item 3 use?");
    let first = Input::Chat(conv.clone());
    let p1 = open()
        .prepare(&request(&first, &BYTE, 2048), &BYTE, Interrupt::default())
        .unwrap();
    assert_eq!(p1.report.checkpoint, CheckpointUse::Miss);
    // A new engine and store over the same directory: the restarted server.
    conv.push(asst("8003"));
    conv.push(user("And item 39?"));
    let second = Input::Chat(conv);
    let p2 = open()
        .prepare(&request(&second, &BYTE, 2048), &BYTE, Interrupt::default())
        .unwrap();
    assert_eq!(p2.report.checkpoint, CheckpointUse::Hit);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_full_store_costs_reuse_not_correctness() {
    let e = ContextEngine::new(ContextConfig::default(), Arc::new(MemoryStore::new(0)));
    let mut conv = conversation(40, "Which port does item 3 use?");
    let first = Input::Chat(conv.clone());
    e.prepare(&request(&first, &BYTE, 2048), &BYTE, Interrupt::default())
        .unwrap();
    conv.push(asst("8003"));
    conv.push(user("ok"));
    let second = Input::Chat(conv);
    let p = e
        .prepare(&request(&second, &BYTE, 2048), &BYTE, Interrupt::default())
        .unwrap();
    assert_eq!(p.report.checkpoint, CheckpointUse::Miss);
    assert!(p.report.prepared_tokens <= 2048 - 64);
}
