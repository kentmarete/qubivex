//! A JSON value that keeps object keys in document order.
//!
//! The workspace's `serde_json` sorts object keys (no `preserve_order` feature, which
//! would change key order everywhere). A schema's property order is the order the
//! model writes properties in, so schemas are read into this type instead.

use std::fmt;

use serde::de::{Deserialize, Deserializer, Error as _, MapAccess, SeqAccess, Visitor};

/// A JSON value whose objects keep their keys in document order.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number.
    Number(serde_json::Number),
    /// A string.
    String(String),
    /// An array.
    Array(Vec<Self>),
    /// An object, keys in document order (a repeated key keeps every occurrence; the
    /// last one wins, as in `serde_json`).
    Object(Vec<(String, Self)>),
}

impl Json {
    /// The value at `key` of an object (the last occurrence).
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Self> {
        match self {
            Self::Object(o) => o.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The value at a JSON pointer (RFC 6901), `""` being the value itself.
    #[must_use]
    pub fn pointer(&self, ptr: &str) -> Option<&Self> {
        if ptr.is_empty() {
            return Some(self);
        }
        let rest = ptr.strip_prefix('/')?;
        rest.split('/').try_fold(self, |v, tok| {
            let tok = tok.replace("~1", "/").replace("~0", "~");
            match v {
                Self::Object(_) => v.get(&tok),
                Self::Array(a) => tok.parse::<usize>().ok().and_then(|i| a.get(i)),
                _ => None,
            }
        })
    }

    /// Compact JSON text.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        self.write(&mut s);
        s
    }

    fn write(&self, s: &mut String) {
        match self {
            Self::Null => s.push_str("null"),
            Self::Bool(b) => s.push_str(if *b { "true" } else { "false" }),
            Self::Number(n) => s.push_str(&n.to_string()),
            Self::String(t) => s.push_str(&quote(t)),
            Self::Array(a) => {
                s.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    v.write(s);
                }
                s.push(']');
            }
            Self::Object(o) => {
                s.push('{');
                for (i, (k, v)) in o.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    s.push_str(&quote(k));
                    s.push(':');
                    v.write(s);
                }
                s.push('}');
            }
        }
    }
}

/// A JSON string literal for `t`.
pub(crate) fn quote(t: &str) -> String {
    serde_json::Value::String(t.to_owned()).to_string()
}

impl From<&serde_json::Value> for Json {
    /// Converts a `serde_json` value (whose object keys are already sorted).
    fn from(v: &serde_json::Value) -> Self {
        match v {
            serde_json::Value::Null => Self::Null,
            serde_json::Value::Bool(b) => Self::Bool(*b),
            serde_json::Value::Number(n) => Self::Number(n.clone()),
            serde_json::Value::String(s) => Self::String(s.clone()),
            serde_json::Value::Array(a) => Self::Array(a.iter().map(Self::from).collect()),
            serde_json::Value::Object(o) => {
                Self::Object(o.iter().map(|(k, v)| (k.clone(), Self::from(v))).collect())
            }
        }
    }
}

impl std::str::FromStr for Json {
    type Err = serde_json::Error;

    /// Parses JSON text, keeping key order.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(JsonVisitor)
    }
}

struct JsonVisitor;

impl<'de> Visitor<'de> for JsonVisitor {
    type Value = Json;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_unit<E>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_none<E>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Json, D::Error> {
        Json::deserialize(d)
    }

    fn visit_bool<E>(self, b: bool) -> Result<Json, E> {
        Ok(Json::Bool(b))
    }

    fn visit_i64<E>(self, n: i64) -> Result<Json, E> {
        Ok(Json::Number(n.into()))
    }

    fn visit_u64<E>(self, n: u64) -> Result<Json, E> {
        Ok(Json::Number(n.into()))
    }

    fn visit_f64<E: serde::de::Error>(self, n: f64) -> Result<Json, E> {
        serde_json::Number::from_f64(n)
            .map(Json::Number)
            .ok_or_else(|| E::custom("a number must be finite"))
    }

    fn visit_str<E>(self, s: &str) -> Result<Json, E> {
        Ok(Json::String(s.to_owned()))
    }

    fn visit_string<E>(self, s: String) -> Result<Json, E> {
        Ok(Json::String(s))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Json, A::Error> {
        let mut v = Vec::new();
        while let Some(x) = a.next_element()? {
            v.push(x);
        }
        Ok(Json::Array(v))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Json, A::Error> {
        let mut v = Vec::new();
        while let Some(k) = a.next_key::<String>()? {
            let x: Json = a.next_value()?;
            v.push((k, x));
        }
        if v.len() > 1 << 20 {
            return Err(A::Error::custom("object too large"));
        }
        Ok(Json::Object(v))
    }
}
