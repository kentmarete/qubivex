//! Ranking units against the current request: BM25 over the request's own text (no
//! model, no index outside the request), plus salience and recency.

use std::collections::HashMap;

use crate::text::terms;

/// BM25 over a fixed set of documents.
pub(crate) struct Bm25 {
    docs: Vec<HashMap<String, u32>>,
    lens: Vec<usize>,
    avg_len: f64,
    df: HashMap<String, u32>,
}

const K1: f64 = 1.2;
const B: f64 = 0.75;

impl Bm25 {
    pub(crate) fn new<'a>(texts: impl IntoIterator<Item = &'a str>) -> Self {
        let mut docs = Vec::new();
        let mut lens = Vec::new();
        let mut df: HashMap<String, u32> = HashMap::new();
        for t in texts {
            let words = terms(t);
            lens.push(words.len());
            let mut tf: HashMap<String, u32> = HashMap::new();
            for w in words {
                *tf.entry(w).or_default() += 1;
            }
            for w in tf.keys() {
                *df.entry(w.clone()).or_default() += 1;
            }
            docs.push(tf);
        }
        let avg_len = if lens.is_empty() {
            1.0
        } else {
            (lens.iter().sum::<usize>() as f64 / lens.len() as f64).max(1.0)
        };
        Self {
            docs,
            lens,
            avg_len,
            df,
        }
    }

    /// Each document's score for `query`.
    pub(crate) fn scores(&self, query: &str) -> Vec<f64> {
        let mut q = terms(query);
        q.sort_unstable();
        q.dedup();
        let n = self.docs.len() as f64;
        self.docs
            .iter()
            .zip(&self.lens)
            .map(|(tf, &len)| {
                q.iter()
                    .filter_map(|w| {
                        let f = f64::from(*tf.get(w)?);
                        let df = f64::from(self.df.get(w).copied().unwrap_or(0));
                        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                        let norm = K1 * (1.0 - B + B * len as f64 / self.avg_len);
                        Some(idf * f * (K1 + 1.0) / (f + norm))
                    })
                    .sum()
            })
            .collect()
    }
}

/// Scores scaled to [0, 1] by the largest.
pub(crate) fn normalized(scores: &[f64]) -> Vec<f64> {
    let max = scores.iter().copied().fold(0.0_f64, f64::max);
    if max <= 0.0 {
        return vec![0.0; scores.len()];
    }
    scores.iter().map(|s| s / max).collect()
}

/// Words that mark a decision, an obligation or something to remember.
const DECISION: &[&str] = &[
    "agreed",
    "always",
    "chose",
    "decided",
    "decision",
    "deadline",
    "don't",
    "important",
    "must",
    "never",
    "note",
    "prefer",
    "remember",
    "require",
    "required",
    "requirement",
    "should",
    "todo",
    "use",
    "will",
];

/// How much a unit looks like a fact worth keeping, in [0, 1]: numbers, identifiers or
/// code, and decision words.
pub(crate) fn salience(text: &str) -> f64 {
    let mut s: f64 = 0.0;
    if text.chars().any(|c| c.is_ascii_digit()) {
        s += 0.3;
    }
    if text.contains('`')
        || text.contains("::")
        || text.contains("()")
        || text.split_whitespace().any(|w| {
            w.contains('_')
                || w.contains('/')
                || w.contains('.') && w.len() > 3 && !w.ends_with('.')
        })
    {
        s += 0.3;
    }
    let lower = text.to_lowercase();
    if lower
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .any(|w| DECISION.contains(&w))
    {
        s += 0.4;
    }
    s.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm25_prefers_documents_with_the_query_terms() {
        let docs = [
            "The scheduler queues requests by priority.",
            "KV reservations come from reserve_kv in the engine.",
            "Nothing relevant here at all.",
        ];
        let s = Bm25::new(docs).scores("where is reserve_kv called?");
        assert!(s[1] > s[0] && s[1] > s[2], "{s:?}");
        assert!(s[2].abs() < f64::EPSILON);
        let n = normalized(&s);
        assert!((n[1] - 1.0).abs() < 1e-12);
        assert_eq!(normalized(&[0.0, 0.0]), [0.0, 0.0]);
    }

    #[test]
    fn salience_sees_numbers_code_and_decisions() {
        assert!(salience("hello there") < 0.1);
        assert!(salience("We decided to use port 8080 for `serve`.") > 0.9);
        assert!(salience("see src/lib.rs") > 0.2);
    }
}
