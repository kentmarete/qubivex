//! Constrained decoding (ADR-020): which tokens may come next so that the output stays
//! in a language.
//!
//! Three layers, each usable on its own:
//!
//! - [`Grammar`]: rules of alternatives of sequences of byte classes and rule
//!   references, built with [`GrammarBuilder`]. Building checks that no rule is
//!   left-recursive and every rule can finish, so every live state can still complete.
//! - Front ends that produce grammars: [`json_schema`] (a JSON Schema subset; anything
//!   outside it is refused, never ignored) and [`json_object`] (JSON mode).
//! - [`Matcher`]: one generation's parse state, determinised lazily (equal parse stacks
//!   share an id, and each `(state, byte)` transition is computed once), with token
//!   masks over a [`TokenTrie`] cached per state.
//!
//! No I/O, no threads, no knowledge of models or tokenizers beyond token bytes.

mod grammar;
mod json;
mod matcher;
mod schema;
mod trie;

pub use grammar::{ByteSet, Elem, Grammar, GrammarBuilder, RuleId};
pub use json::Json;
pub use matcher::Matcher;
pub use schema::{json_object, json_schema};
pub use trie::TokenTrie;

/// Why a grammar could not be built, or a matcher could not go on.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GrammarError {
    /// The schema is not valid JSON Schema, or admits no value.
    #[error("invalid schema at {path}: {detail}")]
    Schema {
        /// JSON pointer to the offending subschema.
        path: String,
        /// What is wrong.
        detail: String,
    },
    /// The schema uses a keyword or form this compiler does not support.
    #[error("unsupported schema keyword {keyword:?} at {path}: {detail}")]
    Unsupported {
        /// The keyword.
        keyword: String,
        /// JSON pointer to the subschema.
        path: String,
        /// Why, or what to use instead.
        detail: String,
    },
    /// The grammar is malformed (left recursion, a rule that cannot finish, an empty
    /// byte class or an undefined rule).
    #[error("invalid grammar: {0}")]
    Invalid(String),
    /// The grammar, or a matcher's state, passed a size limit.
    #[error("limit exceeded: {0}")]
    TooLarge(String),
}
