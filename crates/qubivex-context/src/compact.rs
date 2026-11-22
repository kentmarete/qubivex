//! The preparation itself (ADR-021 §2.3–§2.5, §2.7): reduce what must stay but is too
//! long, condense older turns, and count the whole result after every change.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::rank::{Bm25, normalized, salience};
use crate::store::{Checkpoint, NoteEntry, prefix_keys};
use crate::text::{Block, blocks, split_block, units};
use crate::{
    CheckpointUse, ContextEngine, ContextError, Fate, Input, Interrupt, Message, MessageFate,
    Prepared, Report, Request, TokenCounter,
};

/// Whole counts a preparation may take before it gives up.
const ROUNDS: usize = 12;
/// Checkpoint boundaries looked up, newest first.
const LOOKUPS: usize = 64;
/// The smallest a reduced message may be, in tokens: below this, refuse.
const MIN_REDUCED: usize = 16;

const NOTE_HEAD: &str = "[Earlier conversation, condensed to fit the model's context. \
                         Quoted excerpts only; nothing was added.]";

/// Counts and clocks of one preparation.
struct Work<'a> {
    counter: &'a dyn TokenCounter,
    interrupt: Interrupt<'a>,
    counts: usize,
    retrieval: Duration,
    compaction: Duration,
}

impl Work<'_> {
    fn chat(&mut self, m: &[Message]) -> Result<usize, ContextError> {
        self.interrupt.check()?;
        self.counts += 1;
        self.counter.count_chat(m).map_err(ContextError::Count)
    }

    fn text(&mut self, t: &str) -> Result<usize, ContextError> {
        self.interrupt.check()?;
        self.counts += 1;
        self.counter.count_text(t).map_err(ContextError::Count)
    }

    fn frag(&self, t: &str) -> Result<usize, ContextError> {
        self.counter.count_fragment(t).map_err(ContextError::Count)
    }
}

/// `share` (in [0, 1]) of `n`, rounded up.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn share_of(n: usize, share: f64) -> usize {
    (n as f64 * share.clamp(0.0, 1.0)).ceil() as usize
}

fn slack(over: usize, size: usize) -> usize {
    over + 2 + size / 64
}

pub(crate) fn prepare(
    engine: &ContextEngine,
    req: &Request<'_>,
    counter: &dyn TokenCounter,
    interrupt: Interrupt<'_>,
) -> Result<Prepared, ContextError> {
    let began = Instant::now();
    interrupt.check()?;
    let mut w = Work {
        counter,
        interrupt,
        counts: 0,
        retrieval: Duration::ZERO,
        compaction: Duration::ZERO,
    };
    let mut report = Report {
        budget: req.budget,
        original_tokens: req.original_tokens,
        prepared_tokens: 0,
        compacted: false,
        reduced: false,
        messages_condensed: 0,
        excerpts: 0,
        blocks_omitted: 0,
        blocks_kept: 0,
        checkpoint: CheckpointUse::None,
        counts: 0,
        messages: Vec::new(),
        retrieval_time: Duration::ZERO,
        compaction_time: Duration::ZERO,
        total_time: Duration::ZERO,
    };
    let input = match req.input {
        Input::Text(t) => Input::Text(prepare_text(t, req, &mut w, &mut report)?),
        Input::Chat(m) => Input::Chat(prepare_chat(engine, m, req, &mut w, &mut report)?),
    };
    report.counts = w.counts;
    report.retrieval_time = w.retrieval;
    report.compaction_time = w.compaction;
    report.total_time = began.elapsed();
    Ok(Prepared { input, report })
}

// ---------------------------------------------------------------------------------------
// Long messages: keep the relevant blocks verbatim.

struct Reduced {
    text: String,
    omitted: Vec<[usize; 2]>,
    kept: usize,
    dropped: usize,
}

fn marker(first: usize, last: usize) -> String {
    if first == last {
        format!("[… line {first} of this message omitted to fit the model's context …]")
    } else {
        format!("[… lines {first}–{last} of this message omitted to fit the model's context …]")
    }
}

/// `text` cut to at most `target` tokens (by fragment counts): its first and last blocks,
/// and the blocks most relevant to them, in their original order, with a marker for each
/// run left out.
fn reduce(text: &str, target: usize, w: &mut Work<'_>) -> Result<Reduced, ContextError> {
    let began = Instant::now();
    let max_block = (target / 8).max(MIN_REDUCED).min(target.max(1));
    let mut pieces: Vec<(Block, usize)> = Vec::new();
    for b in blocks(text) {
        w.interrupt.check()?;
        let cost = w.frag(&b.text)? + 1;
        if cost > max_block {
            let mut count = |s: &str| w.counter.count_fragment(s);
            pieces.extend(split_block(&b, max_block, &mut count).map_err(ContextError::Count)?);
        } else {
            pieces.push((b, cost));
        }
    }
    let n = pieces.len();
    let mk = w.frag(&marker(100_000, 100_000))? + 1;
    let last = n - 1;
    let mut kept = vec![false; n];
    let mut cost = 0usize;
    // Omitted runs, each costing a marker: at first, the whole text is one.
    let mut gaps = 1usize;
    let keep = |i: usize, kept: &mut Vec<bool>, cost: &mut usize, gaps: &mut usize| -> bool {
        let omitted = |j: Option<usize>| j.is_some_and(|j| j < n && !kept[j]);
        let after = match (omitted(i.checked_sub(1)), omitted(Some(i + 1))) {
            (true, true) => *gaps + 1,
            (false, false) => gaps.saturating_sub(1),
            _ => *gaps,
        };
        let next = *cost + pieces[i].1;
        if next + after * mk > target {
            return false;
        }
        kept[i] = true;
        *cost = next;
        *gaps = after;
        true
    };
    // The last block, then the first, when they fit.
    if !keep(last, &mut kept, &mut cost, &mut gaps) {
        return Err(ContextError::Unfit {
            tokens: 0,
            budget: crate::Budget {
                context: 0,
                reserve: 0,
                input: target,
            },
        });
    }
    if last > 0 {
        keep(0, &mut kept, &mut cost, &mut gaps);
    }
    // The rest by relevance to the first and last blocks.
    let query = format!("{}\n{}", pieces[0].0.text, pieces[last].0.text);
    let bm = Bm25::new(pieces.iter().map(|p| p.0.text.as_str()));
    let rel = normalized(&bm.scores(&query));
    let mut order: Vec<usize> = (1..last).collect();
    let score = |i: usize| rel[i] + 0.05 * salience(&pieces[i].0.text);
    order.sort_by(|&a, &b| score(b).total_cmp(&score(a)).then(a.cmp(&b)));
    for i in order {
        keep(i, &mut kept, &mut cost, &mut gaps);
    }
    // The text, with a marker per omitted run.
    let mut out = String::with_capacity(text.len().min(target * 8));
    let mut omitted = Vec::new();
    let mut i = 0;
    while i < n {
        if kept[i] {
            out.push_str(&pieces[i].0.text);
            out.push_str(pieces[i].0.sep);
            i += 1;
            continue;
        }
        let first = pieces[i].0.first_line;
        while i < n && !kept[i] {
            i += 1;
        }
        let lastl = pieces[i - 1].0.last_line;
        omitted.push([first, lastl]);
        out.push_str(&marker(first, lastl));
        out.push('\n');
    }
    let kept_n = kept.iter().filter(|k| **k).count();
    w.retrieval += began.elapsed();
    Ok(Reduced {
        text: out,
        omitted,
        kept: kept_n,
        dropped: n - kept_n,
    })
}

fn prepare_text(
    text: &str,
    req: &Request<'_>,
    w: &mut Work<'_>,
    report: &mut Report,
) -> Result<String, ContextError> {
    let budget = req.budget;
    let unfit = ContextError::Unfit {
        tokens: req.original_tokens,
        budget,
    };
    let frag = w.frag(text)?;
    let overhead = req.original_tokens.saturating_sub(frag);
    let mut target = budget.input.saturating_sub(overhead);
    for _ in 0..ROUNDS {
        if target < MIN_REDUCED {
            break;
        }
        let r = reduce(text, target, w).map_err(|e| fix_unfit(e, req))?;
        let n = w.text(&r.text)?;
        if n <= budget.input {
            report.reduced = true;
            report.prepared_tokens = n;
            report.blocks_kept = r.kept;
            report.blocks_omitted = r.dropped;
            report.messages = vec![MessageFate {
                index: 0,
                role: "prompt".into(),
                fate: Fate::Reduced {
                    omitted_lines: r.omitted,
                },
            }];
            return Ok(r.text);
        }
        target = target.saturating_sub(slack(n - budget.input, target));
    }
    Err(unfit)
}

/// `reduce`'s refusal carries no request numbers; these are the request's.
fn fix_unfit(e: ContextError, req: &Request<'_>) -> ContextError {
    match e {
        ContextError::Unfit { .. } => ContextError::Unfit {
            tokens: req.original_tokens,
            budget: req.budget,
        },
        e => e,
    }
}

// ---------------------------------------------------------------------------------------
// Conversations.

/// A selected excerpt set: the entries and, per message, (excerpts, units).
struct Note {
    entries: Vec<NoteEntry>,
}

fn entry_line(e: &NoteEntry) -> String {
    format!("#{} {}: {}", e.message + 1, e.role, e.text)
}

fn note_text(entries: &[NoteEntry], first: usize, last: usize) -> String {
    let range = if first == last {
        format!("message #{}", first + 1)
    } else {
        format!("messages #{}–#{}", first + 1, last + 1)
    };
    if entries.is_empty() {
        return format!("[Earlier {range} left out to fit the model's context.]");
    }
    let mut s = String::from(NOTE_HEAD);
    for e in entries {
        s.push('\n');
        s.push_str(&entry_line(e));
    }
    s.push_str(&format!("\n[End of earlier conversation: {range}.]"));
    s
}

/// Picks excerpts from `idxs` (older messages, in order) within `budget` tokens.
fn select(
    msgs: &[Message],
    idxs: &[usize],
    query: &str,
    tail: usize,
    budget: usize,
    w: &mut Work<'_>,
) -> Result<Note, ContextError> {
    struct U<'t> {
        msg: usize,
        ord: usize,
        text: &'t str,
        code: bool,
        cost: usize,
    }
    let mut us: Vec<U<'_>> = Vec::new();
    let mut prefix: HashMap<usize, usize> = HashMap::new();
    let mut totals: HashMap<usize, usize> = HashMap::new();
    for &i in idxs {
        w.interrupt.check()?;
        let m = &msgs[i];
        let found = units(&m.content);
        totals.insert(i, found.len());
        prefix.insert(i, w.frag(&format!("#{} {}: ", i + 1, m.role))? + 1);
        for (ord, u) in found.into_iter().enumerate() {
            let text = &m.content[u.start..u.end];
            us.push(U {
                msg: i,
                ord,
                text,
                code: u.code,
                cost: w.frag(text)? + 1,
            });
        }
    }
    let bm = Bm25::new(us.iter().map(|u| u.text));
    let rel = normalized(&bm.scores(query));
    let tail = tail.max(1) as f64;
    let score = |k: usize| {
        let u = &us[k];
        let first_user = u.ord == 0 && msgs[u.msg].role == "user";
        rel[k]
            + 0.6 * salience(u.text)
            + 0.4 * (u.msg + 1) as f64 / tail
            + if first_user { 0.3 } else { 0.0 }
    };
    let mut order: Vec<usize> = (0..us.len()).collect();
    order.sort_by(|&a, &b| score(b).total_cmp(&score(a)).then(a.cmp(&b)));
    let mut chosen = vec![false; us.len()];
    let mut started: HashMap<usize, bool> = HashMap::new();
    let mut used = 0usize;
    for k in order {
        let u = &us[k];
        if u.cost * 2 > budget {
            continue;
        }
        let extra = u.cost
            + if started.contains_key(&u.msg) {
                0
            } else {
                prefix[&u.msg]
            };
        if used + extra <= budget {
            used += extra;
            chosen[k] = true;
            started.insert(u.msg, true);
        }
    }
    // Entries in message order, excerpts in text order.
    let mut entries = Vec::new();
    let mut k = 0;
    while k < us.len() {
        let msg = us[k].msg;
        let mut text = String::new();
        let mut prev: Option<(usize, bool)> = None;
        let mut excerpts = 0;
        while k < us.len() && us[k].msg == msg {
            if chosen[k] {
                let u = &us[k];
                if let Some((p, pcode)) = prev {
                    let adjacent = p + 1 == u.ord;
                    text.push_str(match (adjacent, pcode || u.code) {
                        (true, false) => " ",
                        (true, true) => "\n",
                        (false, false) => " … ",
                        (false, true) => "\n…\n",
                    });
                } else if u.code {
                    text.push('\n');
                }
                text.push_str(u.text);
                prev = Some((u.ord, u.code));
                excerpts += 1;
            }
            k += 1;
        }
        if excerpts > 0 {
            entries.push(NoteEntry {
                message: msg,
                role: msgs[msg].role.clone(),
                text,
                excerpts,
                units: totals[&msg],
            });
        }
    }
    Ok(Note { entries })
}

/// The messages to send: instructions and kept messages in order, condensed ones out,
/// and the note in a user message in front of the first conversation message.
fn assemble(work: &[Message], condensed: &[bool], note: Option<&str>) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::with_capacity(work.len() + 1);
    let mut placed = note.is_none();
    for (i, m) in work.iter().enumerate() {
        if condensed[i] {
            continue;
        }
        if !placed && !m.is_instruction() {
            placed = true;
            let note = note.unwrap_or_default();
            if m.role == "user" {
                out.push(Message::new("user", format!("{note}\n\n{}", m.content)));
                continue;
            }
            out.push(Message::new("user", note));
        }
        out.push(m.clone());
    }
    if !placed {
        out.push(Message::new("user", note.unwrap_or_default()));
    }
    out
}

#[allow(clippy::too_many_lines)]
fn prepare_chat(
    engine: &ContextEngine,
    msgs: &[Message],
    req: &Request<'_>,
    w: &mut Work<'_>,
    report: &mut Report,
) -> Result<Vec<Message>, ContextError> {
    let budget = req.budget;
    let input = budget.input;
    let unfit = ContextError::Unfit {
        tokens: req.original_tokens,
        budget,
    };
    let n = msgs.len();
    if n == 0 {
        return Err(unfit);
    }
    // The current message: the last user message, and anything after it.
    let tail = msgs.iter().rposition(|m| m.role == "user").unwrap_or(n - 1);
    let pinned: Vec<bool> = (0..n)
        .map(|i| msgs[i].is_instruction() || i >= tail)
        .collect();
    let older: Vec<usize> = (0..tail).filter(|&i| !pinned[i]).collect();
    let mut work: Vec<Message> = msgs.to_vec();
    let mut fates: Vec<Fate> = vec![Fate::Kept; n];
    let minimal = |work: &[Message]| -> Vec<Message> {
        work.iter()
            .zip(&pinned)
            .filter(|(_, p)| **p)
            .map(|(m, _)| m.clone())
            .collect()
    };
    let mut min_tokens = w.chat(&minimal(&work))?;
    let share = engine.config().note_share.clamp(0.0, 1.0);

    // 1. Instructions and the current message must fit; reduce the current message if not.
    if min_tokens > input {
        let mut instr: Vec<Message> = msgs
            .iter()
            .filter(|m| m.is_instruction())
            .cloned()
            .collect();
        instr.push(Message::new("user", ""));
        let it = w.chat(&instr)?;
        if it > input {
            return Err(ContextError::Mandatory { tokens: it, budget });
        }
        let history = if older.is_empty() {
            0
        } else {
            share_of(input, share / 2.0)
        };
        let want = if input.saturating_sub(history) > it + MIN_REDUCED {
            input - history
        } else {
            input
        };
        let mut sizes: HashMap<usize, usize> = HashMap::new();
        for (i, m) in msgs.iter().enumerate().skip(tail) {
            if !m.is_instruction() {
                sizes.insert(i, w.frag(&m.content)?);
            }
        }
        let mut done = false;
        for _ in 0..ROUNDS {
            let Some((&i, &size)) = sizes.iter().max_by_key(|(i, s)| (**s, **i)) else {
                break;
            };
            let over = min_tokens - want;
            let left = size.saturating_sub(over);
            let target = left.saturating_sub(2 + left / 32);
            if target < MIN_REDUCED {
                break;
            }
            let r = reduce(&msgs[i].content, target, w).map_err(|e| fix_unfit(e, req))?;
            work[i].content = r.text;
            fates[i] = Fate::Reduced {
                omitted_lines: r.omitted,
            };
            report.blocks_kept = r.kept;
            report.blocks_omitted = r.dropped;
            sizes.insert(i, target);
            min_tokens = w.chat(&minimal(&work))?;
            if min_tokens <= want {
                done = true;
                break;
            }
        }
        if !done && min_tokens > input {
            return Err(unfit);
        }
        report.reduced = true;
    }

    // 2. Older turns.
    let finish = |out: Vec<Message>, tokens: usize, fates: Vec<Fate>, report: &mut Report| {
        report.prepared_tokens = tokens;
        report.messages = fates
            .into_iter()
            .enumerate()
            .map(|(index, fate)| MessageFate {
                index,
                role: msgs[index].role.clone(),
                fate,
            })
            .collect();
        out
    };
    if older.is_empty() {
        let out = minimal(&work);
        return Ok(finish(out, min_tokens, fates, report));
    }
    let began = Instant::now();
    let available = input.saturating_sub(min_tokens);
    let min_frag: usize = {
        let mut s = 0;
        for m in minimal(&work) {
            s += w.frag(&m.content)?;
        }
        s
    };
    let pinned_count = pinned.iter().filter(|p| **p).count().max(1);
    let per_message = min_tokens.saturating_sub(min_frag).div_ceil(pinned_count);
    let mut cost: HashMap<usize, usize> = HashMap::new();
    for &i in &older {
        cost.insert(i, w.frag(&msgs[i].content)? + per_message);
    }
    // Everything verbatim, if the reduction left room for it.
    if older.iter().map(|i| cost[i]).sum::<usize>() <= available {
        let n_tokens = w.chat(&work)?;
        if n_tokens <= input {
            w.compaction += began.elapsed();
            return Ok(finish(work, n_tokens, fates, report));
        }
    }
    // Turns: a user message and what follows it, up to the next.
    let mut turns: Vec<Vec<usize>> = Vec::new();
    for &i in &older {
        if turns.is_empty() || msgs[i].role == "user" {
            turns.push(vec![i]);
        } else if let Some(t) = turns.last_mut() {
            t.push(i);
        }
    }
    let note_max = share_of(available, share);
    let query = msgs[tail].content.clone();
    // A checkpoint on the longest condensed prefix this conversation had before.
    let keys = prefix_keys(req.namespace, req.scope, msgs, tail);
    let mut frozen: Option<Checkpoint> = None;
    let bounds = turns
        .iter()
        .map(|t| t[0])
        .chain(std::iter::once(tail))
        .rev()
        .take(LOOKUPS);
    for b in bounds {
        if let Ok(Some(cp)) = engine.store().get(&keys[b]) {
            if cp.covered == b && cp.entries.iter().all(|e| e.message < b) {
                frozen = Some(cp);
            }
            break;
        }
    }
    // The note's framing: its first line and the longest last line it can have.
    let head_cost = w.frag(NOTE_HEAD)?
        + w.frag("[End of earlier conversation: messages #100000–#100000.]")?
        + 2;
    // How much a stored note may grow per turn: an eighth of the share, so it lasts
    // several turns before it fills and is rebuilt.
    let step = note_max / 8;
    let mut forced = 0usize; // kept turns moved into the note after a failed count
    let mut shrink = 0usize;
    for _ in 0..ROUNDS {
        w.interrupt.check()?;
        let floor = frozen.as_ref().map_or(0, |c| c.covered);
        // Recent turns, newest first, in the room the note leaves.
        let room = available.saturating_sub(note_max);
        let mut kept_cost = 0;
        let mut first_kept = tail;
        let mut kept_turns = 0;
        for t in turns.iter().rev() {
            if t[0] < floor {
                break;
            }
            let c: usize = t.iter().map(|i| cost[i]).sum();
            if kept_cost + c > room {
                break;
            }
            kept_cost += c;
            first_kept = t[0];
            kept_turns += 1;
        }
        // After failed counts, fewer of them.
        let mut drop = forced.min(kept_turns);
        while drop > 0 {
            let t = turns.iter().find(|t| t[0] >= first_kept);
            let Some(t) = t else { break };
            kept_cost -= t.iter().map(|i| cost[i]).sum::<usize>();
            first_kept = turns
                .iter()
                .map(|t| t[0])
                .find(|&s| s > t[0])
                .unwrap_or(tail);
            kept_turns -= 1;
            drop -= 1;
        }
        let condensed_idx: Vec<usize> = older.iter().copied().filter(|&i| i < first_kept).collect();
        // The room the kept turns leave for the note's excerpts.
        let note_budget = available
            .saturating_sub(kept_cost)
            .saturating_sub(head_cost)
            .saturating_sub(shrink);
        let frozen_entries = frozen.as_ref().map_or(&[][..], |c| &c.entries[..]);
        let frozen_cost: usize = {
            let mut c = 0;
            for e in frozen_entries {
                c += w.frag(&entry_line(e))? + 1;
            }
            c
        };
        let fresh: Vec<usize> = condensed_idx
            .iter()
            .copied()
            .filter(|&i| i >= floor)
            .collect();
        if frozen.is_some() && frozen_cost + step.min(note_budget / 8) > note_budget {
            // The stored note leaves no room to add the newly condensed turns: rebuild.
            frozen = None;
            continue;
        }
        // A stored note grows by at most a step. A new one leaves room for later turns
        // to extend it without changing it: half the share, or more when the room left
        // is larger than the share.
        let fresh_budget = if frozen.is_some() {
            (note_budget - frozen_cost).min(step.max(1))
        } else {
            note_budget.saturating_sub((note_max / 2).min(note_budget / 2))
        };
        let picked = select(msgs, &fresh, &query, tail, fresh_budget, w)?;
        let mut fresh_cost = 0;
        for e in &picked.entries {
            fresh_cost += w.frag(&entry_line(e))? + 1;
        }
        let mut entries: Vec<NoteEntry> = frozen_entries.to_vec();
        entries.extend(picked.entries);
        let first_c = condensed_idx.first().copied().unwrap_or(0);
        let last_c = condensed_idx.last().copied().unwrap_or(0);
        let text = note_text(&entries, first_c, last_c);
        let mut condensed = vec![false; n];
        for &i in &condensed_idx {
            condensed[i] = true;
        }
        let out = assemble(&work, &condensed, Some(&text));
        let n_tokens = w.chat(&out)?;
        if n_tokens <= input {
            w.compaction += began.elapsed();
            let by_msg: HashMap<usize, &NoteEntry> =
                entries.iter().map(|e| (e.message, e)).collect();
            for &i in &condensed_idx {
                let (excerpts, units) = by_msg.get(&i).map_or_else(
                    || (0, units(&msgs[i].content).len()),
                    |e| (e.excerpts, e.units),
                );
                fates[i] = Fate::Condensed { excerpts, units };
            }
            report.compacted = !condensed_idx.is_empty();
            report.messages_condensed = condensed_idx.len();
            report.excerpts = entries.iter().map(|e| e.excerpts).sum();
            report.checkpoint = match (&frozen, condensed_idx.is_empty()) {
                (_, true) => CheckpointUse::None,
                (Some(_), false) => CheckpointUse::Hit,
                (None, false) => CheckpointUse::Miss,
            };
            if !condensed_idx.is_empty() && !entries.is_empty() {
                let cp = Checkpoint {
                    covered: first_kept,
                    entries,
                };
                // A store that fails costs the next turn a rebuild, never this request.
                let _ = engine.store().put(&keys[first_kept], &cp);
            }
            return Ok(finish(out, n_tokens, fates, report));
        }
        // Too long: a smaller note, then fewer verbatim turns, then no stored note.
        if fresh_cost > 0 {
            shrink += slack(n_tokens - input, note_budget);
            continue;
        }
        shrink = 0;
        if kept_turns > 0 {
            forced += 1;
        } else if frozen.is_some() {
            frozen = None;
        } else {
            break;
        }
    }
    Err(unfit)
}
