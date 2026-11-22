//! Cutting text into the units context management keeps or leaves out: blocks of a
//! document (paragraphs, fenced code blocks, slices of long ones) and the sentences of
//! a message. Every unit is a verbatim piece of the original; nothing here rewrites
//! text.

/// A run of a message's text that is kept or omitted whole.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Block {
    /// First line, counting from 1.
    pub(crate) first_line: usize,
    /// Last line, counting from 1.
    pub(crate) last_line: usize,
    /// The text, verbatim.
    pub(crate) text: String,
    /// What followed it in the original: `"\n"`, or nothing for the last block and for a
    /// slice of a line cut in the middle.
    pub(crate) sep: &'static str,
    /// A fenced code block, or a slice of one.
    pub(crate) code: bool,
}

/// Whether a line opens or closes a fenced code block.
fn is_fence(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("```") || t.starts_with("~~~")
}

/// The blocks of `text`: paragraphs (non-blank lines and the blank lines after them)
/// and fenced code blocks (fence to fence, and the blank lines after). Concatenating
/// every block's text and separator gives `text` back exactly.
pub(crate) fn blocks(text: &str) -> Vec<Block> {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let start = i;
        let code = is_fence(lines[i]);
        if code {
            i += 1;
            while i < lines.len() && !is_fence(lines[i]) {
                i += 1;
            }
            // The closing fence belongs to the block.
            i = (i + 1).min(lines.len());
        } else {
            while i < lines.len() && !lines[i].trim().is_empty() && !is_fence(lines[i]) {
                i += 1;
            }
        }
        // Trailing blank lines go with the block, so blocks tile the text.
        while i < lines.len() && lines[i].trim().is_empty() {
            i += 1;
        }
        let last = i == lines.len();
        out.push(Block {
            first_line: start + 1,
            last_line: i,
            text: lines[start..i].join("\n"),
            sep: if last { "" } else { "\n" },
            code,
        });
    }
    out
}

/// `block` cut at line boundaries into pieces of at most `max` tokens by `count`, and a
/// line longer than that cut at character boundaries. The pieces tile the block.
pub(crate) fn split_block(
    block: &Block,
    max: usize,
    count: &mut dyn FnMut(&str) -> Result<usize, String>,
) -> Result<Vec<(Block, usize)>, String> {
    let max = max.max(1);
    let lines: Vec<&str> = block.text.split('\n').collect();
    let mut out: Vec<(Block, usize)> = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    let mut cur_first = block.first_line;
    let mut cur_tokens = 0;
    let flush = |out: &mut Vec<(Block, usize)>, cur: &mut Vec<&str>, first: usize, tokens| {
        if cur.is_empty() {
            return;
        }
        out.push((
            Block {
                first_line: first,
                last_line: first + cur.len() - 1,
                text: cur.join("\n"),
                sep: "\n",
                code: block.code,
            },
            tokens,
        ));
        cur.clear();
    };
    for (k, line) in lines.iter().enumerate() {
        let n = block.first_line + k;
        let t = count(line)? + 1;
        if t > max {
            flush(&mut out, &mut cur, cur_first, cur_tokens);
            for (piece, tokens) in split_line(line, t, max, count)? {
                out.push((
                    Block {
                        first_line: n,
                        last_line: n,
                        text: piece,
                        sep: "",
                        code: block.code,
                    },
                    tokens,
                ));
            }
            // The line's end.
            if let Some(last) = out.last_mut() {
                last.0.sep = "\n";
            }
            cur_first = n + 1;
            cur_tokens = 0;
            continue;
        }
        if cur_tokens + t > max {
            flush(&mut out, &mut cur, cur_first, cur_tokens);
            cur_first = n;
            cur_tokens = 0;
        }
        cur.push(line);
        cur_tokens += t;
    }
    flush(&mut out, &mut cur, cur_first, cur_tokens);
    if let Some(last) = out.last_mut() {
        last.0.sep = block.sep;
    }
    Ok(out)
}

/// A line of `tokens` tokens cut at character boundaries into pieces of at most `max`.
fn split_line(
    line: &str,
    tokens: usize,
    max: usize,
    count: &mut dyn FnMut(&str) -> Result<usize, String>,
) -> Result<Vec<(String, usize)>, String> {
    let chars: Vec<char> = line.chars().collect();
    let mut out = Vec::new();
    // Characters per piece, from the line's own ratio; shrunk while a piece is too big.
    let mut step = (chars.len() * max / tokens.max(1)).max(1);
    let mut at = 0;
    while at < chars.len() {
        let end = (at + step).min(chars.len());
        let piece: String = chars[at..end].iter().collect();
        let t = count(&piece)?;
        if t > max && end - at > 1 {
            step = (step / 2).max(1);
            continue;
        }
        out.push((piece, t));
        at = end;
    }
    Ok(out)
}

/// A sentence or a code block of a message, by byte range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Unit {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) code: bool,
}

/// The units of a message: each fenced code block whole, and the prose between them cut
/// into sentences (after `.`, `!` or `?` followed by white space, and at line ends).
/// Ranges exclude surrounding white space; empty units are dropped.
pub(crate) fn units(text: &str) -> Vec<Unit> {
    let mut out = Vec::new();
    let mut pos = 0;
    let mut prose_start = 0;
    let mut in_code: Option<usize> = None;
    for line in text.split_inclusive('\n') {
        let line_start = pos;
        pos += line.len();
        if is_fence(line) {
            match in_code {
                None => {
                    sentences(text, prose_start, line_start, &mut out);
                    in_code = Some(line_start);
                }
                Some(s) => {
                    push_trimmed(text, s, pos, true, &mut out);
                    in_code = None;
                    prose_start = pos;
                }
            }
        }
    }
    match in_code {
        Some(s) => push_trimmed(text, s, text.len(), true, &mut out),
        None => sentences(text, prose_start, text.len(), &mut out),
    }
    out
}

fn push_trimmed(text: &str, start: usize, end: usize, code: bool, out: &mut Vec<Unit>) {
    let s = &text[start..end];
    let lead = s.len() - s.trim_start().len();
    let trail = s.len() - s.trim_end().len();
    if lead + trail < s.len() {
        out.push(Unit {
            start: start + lead,
            end: end - trail,
            code,
        });
    }
}

fn sentences(text: &str, from: usize, to: usize, out: &mut Vec<Unit>) {
    let s = &text[from..to];
    let mut start = 0;
    let mut prev: Option<char> = None;
    for (i, c) in s.char_indices() {
        let ends_before = c == '\n' || (c.is_whitespace() && matches!(prev, Some('.' | '!' | '?')));
        if ends_before {
            push_trimmed(text, from + start, from + i, false, out);
            start = i;
        }
        prev = Some(c);
    }
    push_trimmed(text, from + start, to, false, out);
}

/// Words that say nothing about relevance.
const STOP: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "for", "from", "has",
    "have", "how", "i", "if", "in", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or",
    "so", "that", "the", "their", "then", "there", "this", "to", "was", "we", "what", "when",
    "which", "with", "you", "your",
];

/// The search terms of `text`, lowercased: each word, and the parts of an identifier
/// (`reserve_kv` → `reserve_kv`, `reserve`, `kv`; `prepareManaged` → `preparemanaged`,
/// `prepare`, `managed`). Stop words and one-letter words are dropped.
pub(crate) fn terms(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let push = |w: &str, out: &mut Vec<String>| {
        let w = w.to_lowercase();
        if (w.chars().count() > 1 || w.chars().all(|c| c.is_ascii_digit()))
            && !STOP.contains(&w.as_str())
        {
            out.push(w);
        }
    };
    for word in text.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if word.is_empty() {
            continue;
        }
        push(word, &mut out);
        let mut cur = String::new();
        let mut prev_lower = false;
        let mut split: Vec<String> = Vec::new();
        for c in word.chars() {
            if c == '_' {
                if !cur.is_empty() {
                    split.push(std::mem::take(&mut cur));
                }
                prev_lower = false;
                continue;
            }
            if c.is_uppercase() && prev_lower && !cur.is_empty() {
                split.push(std::mem::take(&mut cur));
            }
            prev_lower = c.is_lowercase() || c.is_ascii_digit();
            cur.push(c);
        }
        if !cur.is_empty() {
            split.push(cur);
        }
        if split.len() > 1 {
            for p in &split {
                push(p, &mut out);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rebuild(b: &[Block]) -> String {
        b.iter().map(|b| format!("{}{}", b.text, b.sep)).collect()
    }

    #[test]
    fn blocks_tile_the_text_and_keep_code_blocks_whole() {
        let t = "Intro line\nsecond\n\n```rust\nfn a() {}\n\nfn b() {}\n```\n\nTail.\n";
        let b = blocks(t);
        assert_eq!(rebuild(&b), t);
        assert_eq!(b.len(), 3);
        assert!(b[1].code);
        assert!(b[1].text.contains("fn a() {}\n\nfn b() {}"));
        assert_eq!((b[1].first_line, b[1].last_line), (4, 9));
        for t in ["", "\n", "x", "\n\nx\n\n", "```\nopen fence"] {
            assert_eq!(rebuild(&blocks(t)), t, "{t:?}");
        }
    }

    #[test]
    fn long_blocks_split_at_lines_and_long_lines_at_characters() {
        let mut count = |s: &str| Ok(s.len());
        let b = Block {
            first_line: 10,
            last_line: 13,
            text: "aaaa\nbbbb\ncccc\n".to_owned() + &"x".repeat(40),
            sep: "\n",
            code: false,
        };
        let parts = split_block(&b, 12, &mut count).unwrap();
        let back: String = parts
            .iter()
            .map(|(b, _)| format!("{}{}", b.text, b.sep))
            .collect();
        assert_eq!(back, format!("{}\n", b.text));
        assert!(parts.iter().all(|(_, t)| *t <= 12));
        assert_eq!(parts[0].0.first_line, 10);
        assert_eq!(parts.last().unwrap().0.last_line, 13);
    }

    #[test]
    fn units_are_sentences_and_whole_code_blocks() {
        let t = "We decided on port 8080. Then what?\nNext line\n```\nlet x = 1. y\n```\nDone!";
        let u: Vec<&str> = units(t).iter().map(|u| &t[u.start..u.end]).collect();
        assert_eq!(
            u,
            [
                "We decided on port 8080.",
                "Then what?",
                "Next line",
                "```\nlet x = 1. y\n```",
                "Done!"
            ]
        );
        assert!(units("  \n ").is_empty());
    }

    #[test]
    fn terms_split_identifiers() {
        let t = terms("Call reserve_kv from prepareManaged, the HTTP layer");
        for w in [
            "call",
            "reserve_kv",
            "reserve",
            "kv",
            "preparemanaged",
            "prepare",
            "managed",
            "http",
            "layer",
        ] {
            assert!(t.contains(&w.to_owned()), "{w} in {t:?}");
        }
        assert!(!t.contains(&"the".to_owned()));
    }
}
