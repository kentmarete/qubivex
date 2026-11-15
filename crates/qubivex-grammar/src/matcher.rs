//! One generation's parse state, determinised lazily.
//!
//! A parse stack is a linked list of grammar positions, each the next element to match
//! in some alternative, the top being a byte class. Stacks are hash-consed into
//! *frames* (equal stacks have one id), so a set of stacks is a sorted list of ids and
//! is itself interned as a *state*. Advancing a state by a byte is computed once and
//! cached, so after warm-up it is a table lookup. A reference in tail position does not
//! push a frame, so right recursion (repetition) keeps stacks, and states, constant.

use std::collections::HashMap;
use std::sync::Arc;

use crate::grammar::{Grammar, Slot};
use crate::{ByteSet, GrammarError, TokenTrie};

/// The empty stack: the start rule is complete.
const ROOT: u32 = 0;
/// The state with no stacks.
const DEAD: u32 = 0;
const UNKNOWN: u32 = u32::MAX;
const MAX_FRAMES: usize = 1 << 22;
const MAX_STATES: usize = 1 << 16;

struct State {
    stacks: Box<[u32]>,
    accepts: bool,
    next: Option<Box<[u32; 256]>>,
    mask: Option<Arc<[u64]>>,
}

/// The parse state of one generation over a [`Grammar`].
pub struct Matcher {
    g: Arc<Grammar>,
    /// (position, parent frame); frame 0 is the empty stack.
    frames: Vec<(u32, u32)>,
    frame_ids: HashMap<(u32, u32), u32>,
    states: Vec<State>,
    state_ids: HashMap<Box<[u32]>, u32>,
    cur: u32,
}

impl std::fmt::Debug for Matcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Matcher")
            .field("frames", &self.frames.len())
            .field("states", &self.states.len())
            .field("accepts", &self.accepts())
            .finish_non_exhaustive()
    }
}

impl Matcher {
    /// A matcher at the start of `g`.
    ///
    /// # Errors
    /// [`GrammarError::TooLarge`] if the start state passes the matcher's limits.
    pub fn new(g: Arc<Grammar>) -> Result<Self, GrammarError> {
        let mut m = Self {
            g,
            frames: vec![(u32::MAX, u32::MAX)],
            frame_ids: HashMap::new(),
            states: Vec::new(),
            state_ids: HashMap::new(),
            cur: DEAD,
        };
        m.intern_state(Vec::new())?;
        let g = Arc::clone(&m.g);
        let mut out = Vec::new();
        for &a in &g.alt_starts[g.start as usize] {
            m.expand(a, ROOT, &mut out)?;
        }
        m.cur = m.intern_state(out)?;
        Ok(m)
    }

    /// Whether `bytes` would match in full from `g`'s start.
    ///
    /// # Errors
    /// [`GrammarError::TooLarge`] past the matcher's limits.
    pub fn matches(g: &Arc<Grammar>, bytes: &[u8]) -> Result<bool, GrammarError> {
        let mut m = Self::new(Arc::clone(g))?;
        Ok(m.advance_bytes(bytes)? && m.accepts())
    }

    /// Whether the text so far is a complete string of the grammar.
    #[must_use]
    pub fn accepts(&self) -> bool {
        self.states[self.cur as usize].accepts
    }

    /// Advances by one byte. Returns `false`, leaving the state as it was, if the byte
    /// cannot come next.
    ///
    /// # Errors
    /// [`GrammarError::TooLarge`] past the matcher's limits.
    pub fn advance(&mut self, b: u8) -> Result<bool, GrammarError> {
        let s = self.next(self.cur, b)?;
        if s == DEAD {
            return Ok(false);
        }
        self.cur = s;
        Ok(true)
    }

    /// Advances by `bytes`, all or nothing. Returns `false`, leaving the state as it
    /// was, if they cannot come next.
    ///
    /// # Errors
    /// [`GrammarError::TooLarge`] past the matcher's limits.
    pub fn advance_bytes(&mut self, bytes: &[u8]) -> Result<bool, GrammarError> {
        match self.walk(bytes)? {
            DEAD if !bytes.is_empty() => Ok(false),
            s => {
                self.cur = s;
                Ok(true)
            }
        }
    }

    /// The bytes that can come next.
    ///
    /// # Errors
    /// [`GrammarError::TooLarge`] past the matcher's limits.
    pub fn allowed_bytes(&mut self) -> Result<ByteSet, GrammarError> {
        let mut s = ByteSet::empty();
        for b in 0..=255u8 {
            if self.next(self.cur, b)? != DEAD {
                s = s.with(b);
            }
        }
        Ok(s)
    }

    /// Whether token `id` can come next: an end-of-sequence id when the text is
    /// complete, any other token when its bytes can come next.
    ///
    /// # Errors
    /// [`GrammarError::TooLarge`] past the matcher's limits.
    pub fn allows_token(&mut self, trie: &TokenTrie, id: u32) -> Result<bool, GrammarError> {
        if trie.is_eos(id) {
            return Ok(self.accepts());
        }
        match trie.bytes(id) {
            Some(b) => Ok(self.walk(b)? != DEAD),
            None => Ok(false),
        }
    }

    /// Takes token `id` (see [`Matcher::allows_token`]). Returns `false`, leaving the
    /// state as it was, if it cannot come next. An end-of-sequence id leaves the state
    /// unchanged.
    ///
    /// # Errors
    /// [`GrammarError::TooLarge`] past the matcher's limits.
    pub fn accept_token(&mut self, trie: &TokenTrie, id: u32) -> Result<bool, GrammarError> {
        if trie.is_eos(id) {
            return Ok(self.accepts());
        }
        match trie.bytes(id) {
            Some(b) => self.advance_bytes(b),
            None => Ok(false),
        }
    }

    /// The tokens that can come next, one bit per id (bit `id % 64` of word
    /// `id / 64`). Cached per state.
    ///
    /// # Errors
    /// [`GrammarError::TooLarge`] past the matcher's limits.
    pub fn mask(&mut self, trie: &TokenTrie) -> Result<Arc<[u64]>, GrammarError> {
        let cur = self.cur as usize;
        if let Some(m) = &self.states[cur].mask
            && m.len() == trie.vocab().div_ceil(64)
        {
            return Ok(Arc::clone(m));
        }
        let mut mask = vec![0u64; trie.vocab().div_ceil(64)];
        let mut at = vec![DEAD; trie.max_depth() + 1];
        at[0] = self.cur;
        let n = trie.nodes();
        let mut i = 1;
        while i < n {
            let (b, d, skip) = trie.node(i);
            let s = self.next(at[d - 1], b)?;
            if s == DEAD {
                i = skip;
                continue;
            }
            at[d] = s;
            for &t in trie.tokens_at(i) {
                mask[t as usize / 64] |= 1 << (t % 64);
            }
            i += 1;
        }
        if self.accepts() {
            for &t in trie.eos() {
                if let Some(w) = mask.get_mut(t as usize / 64) {
                    *w |= 1 << (t % 64);
                }
            }
        }
        let mask: Arc<[u64]> = mask.into();
        self.states[cur].mask = Some(Arc::clone(&mask));
        Ok(mask)
    }

    /// Distinct states and frames so far (memory use).
    #[must_use]
    pub fn size(&self) -> (usize, usize) {
        (self.states.len(), self.frames.len())
    }

    fn walk(&mut self, bytes: &[u8]) -> Result<u32, GrammarError> {
        let mut s = self.cur;
        for &b in bytes {
            s = self.next(s, b)?;
            if s == DEAD {
                break;
            }
        }
        Ok(s)
    }

    fn next(&mut self, s: u32, b: u8) -> Result<u32, GrammarError> {
        if s == DEAD {
            return Ok(DEAD);
        }
        if let Some(t) = &self.states[s as usize].next
            && t[b as usize] != UNKNOWN
        {
            return Ok(t[b as usize]);
        }
        let g = Arc::clone(&self.g);
        let stacks = self.states[s as usize].stacks.clone();
        let mut out = Vec::new();
        for &f in &stacks {
            if f == ROOT {
                continue;
            }
            let (pos, parent) = self.frames[f as usize];
            if let Slot::Bytes(set) = g.elems[pos as usize]
                && set.contains(b)
            {
                self.expand(pos + 1, parent, &mut out)?;
            }
        }
        let t = self.intern_state(out)?;
        self.states[s as usize]
            .next
            .get_or_insert_with(|| Box::new([UNKNOWN; 256]))[b as usize] = t;
        Ok(t)
    }

    /// Pushes onto `out` every stack whose top is a byte class (or the empty stack)
    /// reachable from position `pos` over `parent` without consuming a byte.
    fn expand(
        &mut self,
        mut pos: u32,
        mut parent: u32,
        out: &mut Vec<u32>,
    ) -> Result<(), GrammarError> {
        let g = Arc::clone(&self.g);
        loop {
            match g.elems[pos as usize] {
                Slot::End => {
                    if parent == ROOT {
                        out.push(ROOT);
                        return Ok(());
                    }
                    (pos, parent) = self.frames[parent as usize];
                }
                Slot::Bytes(_) => {
                    out.push(self.frame(pos, parent)?);
                    return Ok(());
                }
                Slot::Rule(r) => {
                    let cont = if g.elems[pos as usize + 1] == Slot::End {
                        parent
                    } else {
                        self.frame(pos + 1, parent)?
                    };
                    for &a in &g.alt_starts[r as usize] {
                        self.expand(a, cont, out)?;
                    }
                    return Ok(());
                }
            }
        }
    }

    fn frame(&mut self, pos: u32, parent: u32) -> Result<u32, GrammarError> {
        if let Some(&f) = self.frame_ids.get(&(pos, parent)) {
            return Ok(f);
        }
        if self.frames.len() >= MAX_FRAMES {
            return Err(GrammarError::TooLarge(format!(
                "the matcher passed {MAX_FRAMES} parse frames"
            )));
        }
        let f = u32::try_from(self.frames.len()).unwrap_or(u32::MAX);
        self.frames.push((pos, parent));
        self.frame_ids.insert((pos, parent), f);
        Ok(f)
    }

    fn intern_state(&mut self, mut stacks: Vec<u32>) -> Result<u32, GrammarError> {
        stacks.sort_unstable();
        stacks.dedup();
        let stacks: Box<[u32]> = stacks.into();
        if let Some(&s) = self.state_ids.get(&stacks) {
            return Ok(s);
        }
        if self.states.len() >= MAX_STATES {
            return Err(GrammarError::TooLarge(format!(
                "the matcher passed {MAX_STATES} parse states"
            )));
        }
        let s = u32::try_from(self.states.len()).unwrap_or(u32::MAX);
        self.states.push(State {
            accepts: stacks.contains(&ROOT),
            stacks: stacks.clone(),
            next: None,
            mask: None,
        });
        self.state_ids.insert(stacks, s);
        Ok(s)
    }
}
