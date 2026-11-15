//! The grammar IR: rules of alternatives of sequences of byte classes and rule
//! references.

use crate::GrammarError;

/// Most grammar elements (byte classes, references and alternative ends) together.
const MAX_ELEMS: usize = 1 << 22;

/// A set of bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ByteSet([u64; 4]);

impl ByteSet {
    /// No bytes.
    #[must_use]
    pub const fn empty() -> Self {
        Self([0; 4])
    }

    /// One byte.
    #[must_use]
    pub const fn byte(b: u8) -> Self {
        Self::empty().with(b)
    }

    /// The bytes `lo..=hi`.
    #[must_use]
    pub fn range(lo: u8, hi: u8) -> Self {
        let mut s = Self::empty();
        for b in lo..=hi {
            s = s.with(b);
        }
        s
    }

    /// Every byte in `bytes`.
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        bytes.iter().fold(Self::empty(), |s, &b| s.with(b))
    }

    /// This set with `b` added.
    #[must_use]
    pub const fn with(mut self, b: u8) -> Self {
        self.0[(b >> 6) as usize] |= 1 << (b & 63);
        self
    }

    /// This set with `b` removed.
    #[must_use]
    pub const fn without(mut self, b: u8) -> Self {
        self.0[(b >> 6) as usize] &= !(1 << (b & 63));
        self
    }

    /// The union.
    #[must_use]
    pub const fn union(self, o: Self) -> Self {
        Self([
            self.0[0] | o.0[0],
            self.0[1] | o.0[1],
            self.0[2] | o.0[2],
            self.0[3] | o.0[3],
        ])
    }

    /// Whether `b` is in the set.
    #[must_use]
    pub const fn contains(&self, b: u8) -> bool {
        self.0[(b >> 6) as usize] & (1 << (b & 63)) != 0
    }

    /// Whether the set has no bytes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.0[0] | self.0[1] | self.0[2] | self.0[3] == 0
    }

    /// The bytes in the set, ascending.
    pub fn iter(self) -> impl Iterator<Item = u8> {
        (0..=255u8).filter(move |&b| self.contains(b))
    }
}

impl std::fmt::Debug for ByteSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

/// A rule, by index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RuleId(pub(crate) u32);

/// One element of an alternative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    /// One byte from the set.
    Bytes(ByteSet),
    /// A string of the rule.
    Rule(RuleId),
}

/// An element in the flattened grammar: a byte class, a reference, or the end of an
/// alternative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Slot {
    Bytes(ByteSet),
    Rule(u32),
    End,
}

/// Builds a [`Grammar`]. Rules are declared, then defined, so they can refer to each
/// other (and to themselves) in any order.
#[derive(Debug, Default)]
pub struct GrammarBuilder {
    rules: Vec<Option<Vec<Vec<Elem>>>>,
    elems: usize,
}

impl GrammarBuilder {
    /// An empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A new rule, to be defined later with [`GrammarBuilder::define`].
    ///
    /// # Errors
    /// [`GrammarError::TooLarge`] past the rule limit.
    pub fn declare(&mut self) -> Result<RuleId, GrammarError> {
        let id = u32::try_from(self.rules.len())
            .ok()
            .filter(|&n| (n as usize) < MAX_ELEMS)
            .ok_or_else(|| GrammarError::TooLarge("too many grammar rules".into()))?;
        self.rules.push(None);
        Ok(RuleId(id))
    }

    /// Defines `rule` as a choice between `alts` (an empty sequence matches nothing).
    ///
    /// # Errors
    /// [`GrammarError::Invalid`] if `rule` is unknown or already defined;
    /// [`GrammarError::TooLarge`] past the size limit.
    pub fn define(&mut self, rule: RuleId, alts: Vec<Vec<Elem>>) -> Result<(), GrammarError> {
        let slot = self
            .rules
            .get_mut(rule.0 as usize)
            .ok_or_else(|| GrammarError::Invalid(format!("rule {} is not declared", rule.0)))?;
        if slot.is_some() {
            return Err(GrammarError::Invalid(format!(
                "rule {} is defined twice",
                rule.0
            )));
        }
        self.elems += alts.iter().map(|a| a.len() + 1).sum::<usize>();
        if self.elems > MAX_ELEMS {
            return Err(GrammarError::TooLarge(format!(
                "the grammar has more than {MAX_ELEMS} elements"
            )));
        }
        *slot = Some(alts);
        Ok(())
    }

    /// Declares and defines a rule in one step.
    ///
    /// # Errors
    /// As [`GrammarBuilder::define`].
    pub fn rule(&mut self, alts: Vec<Vec<Elem>>) -> Result<RuleId, GrammarError> {
        let r = self.declare()?;
        self.define(r, alts)?;
        Ok(r)
    }

    /// Checks the rules and freezes the grammar, starting at `start`.
    ///
    /// # Errors
    /// [`GrammarError::Invalid`] for an undefined rule, an empty byte class, a rule
    /// that cannot finish or left recursion.
    pub fn build(self, start: RuleId) -> Result<Grammar, GrammarError> {
        let n = self.rules.len();
        if start.0 as usize >= n {
            return Err(GrammarError::Invalid(
                "the start rule is not declared".into(),
            ));
        }
        let rules: Vec<Vec<Vec<Elem>>> = self
            .rules
            .into_iter()
            .enumerate()
            .map(|(i, r)| {
                r.ok_or_else(|| GrammarError::Invalid(format!("rule {i} is not defined")))
            })
            .collect::<Result<_, _>>()?;
        for (i, alts) in rules.iter().enumerate() {
            for e in alts.iter().flatten() {
                match e {
                    Elem::Bytes(s) if s.is_empty() => {
                        return Err(GrammarError::Invalid(format!(
                            "rule {i} has an empty byte class"
                        )));
                    }
                    Elem::Rule(r) if r.0 as usize >= n => {
                        return Err(GrammarError::Invalid(format!(
                            "rule {i} refers to undeclared rule {}",
                            r.0
                        )));
                    }
                    _ => {}
                }
            }
        }
        check(&rules)?;
        let mut elems = Vec::with_capacity(self.elems);
        let mut alt_starts = Vec::with_capacity(n);
        for alts in &rules {
            let mut starts = Vec::with_capacity(alts.len());
            for alt in alts {
                starts.push(
                    u32::try_from(elems.len())
                        .map_err(|_| GrammarError::TooLarge("grammar too large".into()))?,
                );
                elems.extend(alt.iter().map(|e| match *e {
                    Elem::Bytes(s) => Slot::Bytes(s),
                    Elem::Rule(r) => Slot::Rule(r.0),
                }));
                elems.push(Slot::End);
            }
            alt_starts.push(starts.into_boxed_slice());
        }
        Ok(Grammar {
            elems: elems.into_boxed_slice(),
            alt_starts: alt_starts.into_boxed_slice(),
            start: start.0,
        })
    }
}

/// Productivity, nullability and left recursion.
fn check(rules: &[Vec<Vec<Elem>>]) -> Result<(), GrammarError> {
    let n = rules.len();
    let fix = |pred: &dyn Fn(&Elem, &[bool]) -> bool| {
        let mut ok = vec![false; n];
        loop {
            let mut changed = false;
            for (i, alts) in rules.iter().enumerate() {
                if !ok[i] && alts.iter().any(|a| a.iter().all(|e| pred(e, &ok))) {
                    ok[i] = true;
                    changed = true;
                }
            }
            if !changed {
                return ok;
            }
        }
    };
    let productive = fix(&|e, ok| match e {
        Elem::Bytes(_) => true,
        Elem::Rule(r) => ok[r.0 as usize],
    });
    if let Some(i) = productive.iter().position(|p| !p) {
        return Err(GrammarError::Invalid(format!("rule {i} can never finish")));
    }
    let nullable = fix(&|e, ok| match e {
        Elem::Bytes(_) => false,
        Elem::Rule(r) => ok[r.0 as usize],
    });
    // Edges to the rules that can start a rule's string without consuming a byte.
    let left: Vec<Vec<usize>> = rules
        .iter()
        .map(|alts| {
            let mut v = Vec::new();
            for alt in alts {
                for e in alt {
                    match e {
                        Elem::Bytes(_) => break,
                        Elem::Rule(r) => {
                            v.push(r.0 as usize);
                            if !nullable[r.0 as usize] {
                                break;
                            }
                        }
                    }
                }
            }
            v
        })
        .collect();
    // Iterative depth-first search for a cycle: 0 unseen, 1 on the path, 2 done.
    let mut color = vec![0u8; n];
    for root in 0..n {
        if color[root] != 0 {
            continue;
        }
        let mut stack = vec![(root, 0usize)];
        color[root] = 1;
        while let Some(&mut (r, ref mut next)) = stack.last_mut() {
            if let Some(&s) = left[r].get(*next) {
                *next += 1;
                match color[s] {
                    0 => {
                        color[s] = 1;
                        stack.push((s, 0));
                    }
                    1 => {
                        return Err(GrammarError::Invalid(format!("rule {s} is left-recursive")));
                    }
                    _ => {}
                }
            } else {
                color[r] = 2;
                stack.pop();
            }
        }
    }
    Ok(())
}

/// A checked grammar over bytes: no left recursion, and every rule can finish.
#[derive(Debug, PartialEq, Eq)]
pub struct Grammar {
    /// Every alternative's elements, each alternative ended by [`Slot::End`].
    pub(crate) elems: Box<[Slot]>,
    /// For each rule, where its alternatives start in `elems`.
    pub(crate) alt_starts: Box<[Box<[u32]>]>,
    pub(crate) start: u32,
}

impl Grammar {
    /// Number of rules.
    #[must_use]
    pub fn rules(&self) -> usize {
        self.alt_starts.len()
    }
}
