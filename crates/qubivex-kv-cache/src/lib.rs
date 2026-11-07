//! Which retained KV a new prompt can reuse (ADR-016).
//!
//! A forward pass's keys and values depend only on the tokens before them, so a prompt
//! that starts with tokens a finished generation already ran can start from a copy of
//! that generation's KV, and its bits are the same (invariant 1). [`PrefixCache`] is the
//! index: it keeps finished values, finds the one sharing the longest prefix with a
//! prompt, and evicts the least recently used to stay within a budget.
//!
//! It knows nothing of models or tensors, so its rules are tested without either. The
//! value `V` is whatever the host retains, and costs are in the host's units. Paged KV
//! would change what `V` is and how a host copies from it, not this index.
//!
//! Values are immutable once inserted, and a [`Hit`] holds one by `Arc`. A host copies
//! what it needs out of the hit, so nothing a running sequence writes is ever shared. An
//! eviction during that copy only removes the value from the index, and the memory goes
//! when the hit is dropped.

use std::sync::Arc;

/// Fewest tokens worth keeping or reusing. An entry costs at least a whole KV chunk, so
/// prompts this short would fill the budget for the sake of a few passes.
pub const MIN_REUSE: usize = 32;

/// A retained value and the tokens that produced it.
struct Entry<V> {
    scope: Option<String>,
    tokens: Vec<u32>,
    cost: u64,
    value: Arc<V>,
    /// When it was last inserted or hit.
    used: u64,
}

/// What a lookup found.
#[derive(Debug)]
pub struct Hit<V> {
    /// The retained value. Copy from it: it is shared and must stay unchanged.
    pub value: Arc<V>,
    /// How many leading tokens of the prompt it holds.
    pub reuse: usize,
}

/// Counters since the cache was made, and what it holds now.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixStats {
    /// Lookups.
    pub lookups: u64,
    /// Lookups that found a prefix.
    pub hits: u64,
    /// Prompt tokens that did not have to run.
    pub tokens_reused: u64,
    /// Values kept.
    pub inserted: u64,
    /// Values dropped for room, or by [`PrefixCache::clear`]. A value replaced by a
    /// longer one with the same prefix is not counted.
    pub evicted: u64,
    /// Values held now.
    pub entries: usize,
    /// Cost held now.
    pub cost: u64,
}

/// Retained values by the tokens that produced them.
///
/// Lookups scan every entry. The budget bounds the entries and a scan compares a few
/// thousand integers at most, against a forward pass for each token it saves.
pub struct PrefixCache<V> {
    budget: u64,
    entries: Vec<Entry<V>>,
    clock: u64,
    stats: PrefixStats,
}

impl<V> PrefixCache<V> {
    /// An empty cache holding at most `budget` of cost.
    #[must_use]
    pub const fn new(budget: u64) -> Self {
        Self {
            budget,
            entries: Vec::new(),
            clock: 0,
            stats: PrefixStats {
                lookups: 0,
                hits: 0,
                tokens_reused: 0,
                inserted: 0,
                evicted: 0,
                entries: 0,
                cost: 0,
            },
        }
    }

    /// The most cost it will hold.
    #[must_use]
    pub const fn budget(&self) -> u64 {
        self.budget
    }

    /// Cost held now.
    #[must_use]
    pub const fn cost(&self) -> u64 {
        self.stats.cost
    }

    /// Counters and current contents.
    #[must_use]
    pub const fn stats(&self) -> PrefixStats {
        self.stats
    }

    const fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// The entry of `scope` sharing the longest prefix with `prompt`, or `None` below [`MIN_REUSE`] tokens. The last token of the prompt is never
    /// reused: its logits are what the caller needs, and only a pass produces them.
    pub fn lookup(&mut self, scope: Option<&str>, prompt: &[u32]) -> Option<Hit<V>> {
        self.stats.lookups += 1;
        let limit = prompt.len().checked_sub(1)?;
        let mut best: Option<(usize, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if e.scope.as_deref() != scope {
                continue;
            }
            let shared = common_prefix(&e.tokens, prompt).min(limit);
            if shared >= MIN_REUSE && best.is_none_or(|(n, _)| shared > n) {
                best = Some((shared, i));
            }
        }
        let (reuse, at) = best?;
        let used = self.tick();
        let e = &mut self.entries[at];
        e.used = used;
        let value = Arc::clone(&e.value);
        self.stats.hits += 1;
        self.stats.tokens_reused += reuse as u64;
        Some(Hit { value, reuse })
    }

    /// Keeps `value`, the result of running `tokens`, evicting the least recently used
    /// for room. Returns whether it was kept: it is not if it is too short, costs more
    /// than the budget, or an entry of the scope already covers it. Entries it covers
    /// are replaced, so a conversation holds one value and not one per turn.
    pub fn insert(&mut self, scope: Option<&str>, tokens: Vec<u32>, cost: u64, value: V) -> bool {
        if tokens.len() < MIN_REUSE || cost > self.budget {
            return false;
        }
        let used = self.tick();
        let same = |e: &Entry<V>| e.scope.as_deref() == scope;
        if let Some(e) = self
            .entries
            .iter_mut()
            .find(|e| same(e) && e.tokens.starts_with(&tokens))
        {
            e.used = used;
            return false;
        }
        let stats = &mut self.stats;
        self.entries.retain(|e| {
            let covered = same(e) && tokens.starts_with(&e.tokens);
            if covered {
                stats.cost -= e.cost;
                stats.entries -= 1;
            }
            !covered
        });
        while self.stats.cost + cost > self.budget && self.evict_oldest() {}
        self.entries.push(Entry {
            scope: scope.map(str::to_owned),
            tokens,
            cost,
            value: Arc::new(value),
            used,
        });
        self.stats.cost += cost;
        self.stats.entries += 1;
        self.stats.inserted += 1;
        true
    }

    /// Drops the least recently used entry. Returns whether there was one.
    pub fn evict_oldest(&mut self) -> bool {
        let oldest = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.used)
            .map(|(i, _)| i);
        let Some(i) = oldest else {
            return false;
        };
        let e = self.entries.swap_remove(i);
        self.stats.cost -= e.cost;
        self.stats.entries -= 1;
        self.stats.evicted += 1;
        true
    }

    /// Drops every entry.
    pub fn clear(&mut self) {
        while self.evict_oldest() {}
    }
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::{MIN_REUSE, PrefixCache};

    /// `n` tokens counting up from `from`.
    fn run(from: u32, n: usize) -> Vec<u32> {
        (from..).take(n).collect()
    }

    const N: usize = MIN_REUSE * 2;

    #[test]
    fn the_longest_shared_prefix_wins() {
        let mut c = PrefixCache::new(100);
        let long = run(0, N);
        let mut short = long[..N - 4].to_vec();
        short.push(9999);
        assert!(c.insert(None, short, 1, "short"));
        assert!(c.insert(None, long.clone(), 1, "long"));
        let mut prompt = long.clone();
        prompt.extend([7, 8]);
        let hit = c.lookup(None, &prompt).unwrap();
        assert_eq!((*hit.value, hit.reuse), ("long", N));
        let mut other = long[..N - 8].to_vec();
        other.extend([1, 2, 3]);
        assert_eq!(c.lookup(None, &other).unwrap().reuse, N - 8);
    }

    #[test]
    fn the_last_token_always_runs() {
        let mut c = PrefixCache::new(100);
        let t = run(0, N);
        c.insert(None, t.clone(), 1, ());
        assert_eq!(c.lookup(None, &t).unwrap().reuse, N - 1);
        assert_eq!(c.lookup(None, &t[..N - 3]).unwrap().reuse, N - 4);
        assert!(c.lookup(None, &[]).is_none());
    }

    #[test]
    fn short_matches_are_not_worth_a_copy() {
        let mut c = PrefixCache::new(100);
        assert!(!c.insert(None, run(0, MIN_REUSE - 1), 1, ()));
        assert_eq!(c.stats().entries, 0);
        c.insert(None, run(0, N), 1, ());
        let mut p = run(0, MIN_REUSE - 1);
        p.extend([9999, 9998]);
        assert!(c.lookup(None, &p).is_none());
        let mut p = run(0, MIN_REUSE);
        p.push(9999);
        assert_eq!(c.lookup(None, &p).unwrap().reuse, MIN_REUSE);
    }

    #[test]
    fn scopes_never_see_each_other() {
        let mut c = PrefixCache::new(100);
        let t = run(0, N);
        c.insert(Some("a"), t.clone(), 1, "a");
        assert!(c.lookup(Some("b"), &t).is_none());
        assert!(c.lookup(None, &t).is_none());
        // The same tokens in another scope are another entry, not a duplicate.
        assert!(c.insert(Some("b"), t.clone(), 1, "b"));
        assert!(c.insert(None, t.clone(), 1, "none"));
        assert_eq!(c.stats().entries, 3);
        assert_eq!(*c.lookup(Some("b"), &t).unwrap().value, "b");
        assert_eq!(*c.lookup(None, &t).unwrap().value, "none");
    }

    #[test]
    fn the_budget_holds_and_the_least_recently_used_goes_first() {
        let mut c = PrefixCache::new(3);
        let (a, b, d) = (run(0, N), run(1000, N), run(2000, N));
        c.insert(None, a.clone(), 1, 'a');
        c.insert(None, b.clone(), 1, 'b');
        c.insert(None, run(3000, N), 1, 'c');
        // Touching `a` leaves `b` the oldest.
        c.lookup(None, &a).unwrap();
        c.insert(None, d.clone(), 1, 'd');
        assert_eq!(c.cost(), 3);
        assert!(c.lookup(None, &b).is_none());
        assert!(c.lookup(None, &a).is_some());
        assert!(c.lookup(None, &d).is_some());
        // Room for a larger value takes as many as it needs.
        assert!(c.insert(None, run(4000, N), 3, 'e'));
        assert_eq!(c.stats().entries, 1);
        assert_eq!(c.stats().evicted, 4);
    }

    #[test]
    fn a_value_over_budget_is_refused_and_evicts_nothing() {
        let mut c = PrefixCache::new(3);
        let a = run(0, N);
        c.insert(None, a.clone(), 3, ());
        assert!(!c.insert(None, run(1000, N), 4, ()));
        assert!(c.lookup(None, &a).is_some());
        assert_eq!(c.stats().evicted, 0);
    }

    #[test]
    fn a_conversation_holds_one_entry() {
        let mut c = PrefixCache::new(10);
        let long = run(0, N + 10);
        assert!(c.insert(None, long[..N].to_vec(), 2, 1));
        // A longer turn replaces the shorter one, with no eviction.
        assert!(c.insert(None, long.clone(), 3, 2));
        assert_eq!((c.stats().entries, c.cost(), c.stats().evicted), (1, 3, 0));
        // A shorter run of what is held adds nothing, and counts as a use.
        assert!(!c.insert(None, long[..N].to_vec(), 2, 3));
        assert_eq!(*c.lookup(None, &long).unwrap().value, 2);
        // A branch is a new entry: it shares the prefix and does not cover the other.
        let mut branch = long[..N].to_vec();
        branch.push(9999);
        assert!(c.insert(None, branch, 2, 4));
        assert_eq!(c.stats().entries, 2);
    }

    #[test]
    fn a_replaced_value_makes_room_before_anything_older_is_evicted() {
        let mut c = PrefixCache::new(4);
        let (old, turn) = (run(1000, N), run(0, N + 4));
        c.insert(None, old.clone(), 2, ());
        c.insert(None, turn[..N].to_vec(), 2, ());
        assert!(c.insert(None, turn, 2, ()));
        assert!(c.lookup(None, &old).is_some());
    }

    #[test]
    fn a_hit_outlives_its_eviction() {
        let mut c = PrefixCache::new(10);
        let t = run(0, N);
        c.insert(None, t.clone(), 1, vec![1, 2, 3]);
        let hit = c.lookup(None, &t).unwrap();
        c.clear();
        assert_eq!(c.cost(), 0);
        assert_eq!(*hit.value, [1, 2, 3]);
    }

    #[test]
    fn counters_add_up() {
        let mut c = PrefixCache::new(10);
        let t = run(0, N);
        assert!(c.lookup(None, &t).is_none());
        c.insert(None, t.clone(), 2, ());
        c.lookup(None, &t).unwrap();
        let s = c.stats();
        assert_eq!(
            (s.lookups, s.hits, s.tokens_reused, s.inserted),
            (2, 1, N as u64 - 1, 1)
        );
        c.clear();
        assert_eq!(c.stats().evicted, 1);
        assert!(!c.evict_oldest());
    }
}
