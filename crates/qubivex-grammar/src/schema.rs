//! JSON Schema to grammar (ADR-020 §4), and the JSON-mode grammar.
//!
//! Every value rule matches the value alone; containers put bounded whitespace between
//! their tokens. Anything the compiler cannot guarantee is refused, so a grammar that
//! builds only produces text that parses as JSON and validates against the schema.

use std::collections::HashMap;
use std::sync::Arc;

use crate::json::{Json, quote};
use crate::{ByteSet, Elem, Grammar, GrammarBuilder, GrammarError, RuleId};

/// Most whitespace bytes in one gap between JSON tokens.
const MAX_WS: usize = 16;
/// Largest count unrolled for `maxLength`, `minLength`, `minItems`, `maxItems` and
/// `prefixItems`.
const MAX_COUNT: u64 = 1024;
/// Deepest schema nesting (each `$ref` target is compiled once, so recursion through
/// references does not count).
const MAX_DEPTH: usize = 64;
/// Integer digits (before any fraction or exponent): 1e18 fits every parser.
const MAX_INT_DIGITS: usize = 18;
/// Fraction digits.
const MAX_FRAC_DIGITS: usize = 15;
/// Exponent digits: with the limits above every number is finite as an f64.
const MAX_EXP_DIGITS: usize = 2;

/// Keywords that only annotate and never constrain.
const ANNOTATIONS: &[&str] = &[
    "$schema",
    "$id",
    "$comment",
    "$anchor",
    "$defs",
    "definitions",
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
];

/// Keywords the compiler enforces.
const SUPPORTED: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "prefixItems",
    "minItems",
    "maxItems",
    "enum",
    "const",
    "anyOf",
    "allOf",
    "$ref",
    "minLength",
    "maxLength",
];

const OBJECT_KEYS: &[&str] = &["properties", "required", "additionalProperties"];
const ARRAY_KEYS: &[&str] = &["items", "prefixItems", "minItems", "maxItems"];
const STRING_KEYS: &[&str] = &["minLength", "maxLength"];
const TYPES: &[&str] = &[
    "object", "array", "string", "number", "integer", "boolean", "null",
];

/// Why a keyword is refused.
fn refusal(k: &str) -> &'static str {
    match k {
        "oneOf" => "exactly-one-of cannot be enforced on the text; use anyOf",
        "not" => "negation cannot be enforced",
        "pattern" | "patternProperties" => "regular expressions are not supported",
        "format" => "formats are not enforced",
        "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum" | "multipleOf" => {
            "numeric bounds are not enforced"
        }
        "uniqueItems" | "contains" | "minContains" | "maxContains" => {
            "item uniqueness and containment are not enforced"
        }
        "minProperties" | "maxProperties" | "propertyNames" | "dependentRequired"
        | "dependentSchemas" | "dependencies" => "property-set constraints are not enforced",
        "if" | "then" | "else" => "conditional schemas are not supported",
        "unevaluatedProperties" | "unevaluatedItems" => "unevaluated keywords are not supported",
        "additionalItems" => "use prefixItems and items",
        "$dynamicRef" | "$recursiveRef" | "$dynamicAnchor" | "$recursiveAnchor" => {
            "dynamic references are not supported; use $ref"
        }
        _ => "unknown keyword",
    }
}

/// A grammar for JSON text that validates against `schema` (an order-preserving
/// [`Json`]; properties are written in the schema's order).
///
/// # Errors
/// [`GrammarError::Schema`] for an invalid or unsatisfiable schema,
/// [`GrammarError::Unsupported`] for a keyword outside the supported subset,
/// [`GrammarError::TooLarge`] past a size limit.
pub fn json_schema(schema: &Json) -> Result<Arc<Grammar>, GrammarError> {
    let mut c = Compiler::new(schema)?;
    let v = c.compile(schema, "#", 0)?;
    c.finish(v)
}

/// A grammar for any JSON object (JSON mode).
///
/// # Errors
/// None in practice; the grammar is fixed.
pub fn json_object() -> Result<Arc<Grammar>, GrammarError> {
    let root = Json::Bool(true);
    let mut c = Compiler::new(&root)?;
    let (_, obj) = c.any()?;
    c.finish(obj)
}

fn lit(bytes: &[u8]) -> Vec<Elem> {
    bytes
        .iter()
        .map(|&b| Elem::Bytes(ByteSet::byte(b)))
        .collect()
}

fn class(s: ByteSet) -> Elem {
    Elem::Bytes(s)
}

struct Compiler<'a> {
    b: GrammarBuilder,
    root: &'a Json,
    refs: HashMap<String, RuleId>,
    ws: RuleId,
    ch: RuleId,
    digit: RuleId,
    any: Option<(RuleId, RuleId)>,
    eps: RuleId,
    stars: HashMap<RuleId, RuleId>,
    string_any: Option<RuleId>,
}

impl<'a> Compiler<'a> {
    fn new(root: &'a Json) -> Result<Self, GrammarError> {
        let mut b = GrammarBuilder::new();
        let eps = b.rule(vec![vec![]])?;
        // ws_k: up to k whitespace bytes.
        let mut ws = eps;
        for _ in 0..MAX_WS {
            ws = b.rule(vec![
                vec![class(ByteSet::of(b" \t\n\r")), Elem::Rule(ws)],
                vec![],
            ])?;
        }
        let digit = b.rule(vec![vec![class(ByteSet::range(b'0', b'9'))]])?;
        let ch = string_char(&mut b)?;
        Ok(Self {
            b,
            root,
            refs: HashMap::new(),
            ws,
            ch,
            digit,
            any: None,
            eps,
            stars: HashMap::new(),
            string_any: None,
        })
    }

    fn finish(mut self, value: RuleId) -> Result<Arc<Grammar>, GrammarError> {
        let start = self
            .b
            .rule(vec![vec![Elem::Rule(self.ws), Elem::Rule(value)]])?;
        self.b.build(start).map(Arc::new).map_err(|e| match e {
            GrammarError::Invalid(d) => GrammarError::Schema {
                path: "#".into(),
                detail: format!(
                    "the schema has no finite values or refers to itself without progress ({d})"
                ),
            },
            e => e,
        })
    }

    fn w(&self) -> Elem {
        Elem::Rule(self.ws)
    }

    /// `x*`.
    fn star(&mut self, x: RuleId) -> Result<RuleId, GrammarError> {
        if let Some(&s) = self.stars.get(&x) {
            return Ok(s);
        }
        let s = self.b.declare()?;
        self.b
            .define(s, vec![vec![Elem::Rule(x), Elem::Rule(s)], vec![]])?;
        self.stars.insert(x, s);
        Ok(s)
    }

    /// `x{min,max}` (`max` `None`: unbounded).
    fn repeat(
        &mut self,
        x: RuleId,
        min: usize,
        max: Option<usize>,
    ) -> Result<Vec<Elem>, GrammarError> {
        let mut v = vec![Elem::Rule(x); min];
        match max {
            None => v.push(Elem::Rule(self.star(x)?)),
            Some(max) => {
                let mut r = self.eps;
                for _ in min..max {
                    r = self
                        .b
                        .rule(vec![vec![Elem::Rule(x), Elem::Rule(r)], vec![]])?;
                }
                v.push(Elem::Rule(r));
            }
        }
        Ok(v)
    }

    fn string(&mut self, min: usize, max: Option<usize>) -> Result<RuleId, GrammarError> {
        if min == 0
            && max.is_none()
            && let Some(s) = self.string_any
        {
            return Ok(s);
        }
        let mut v = lit(b"\"");
        v.extend(self.repeat(self.ch, min, max)?);
        v.extend(lit(b"\""));
        let r = self.b.rule(vec![v])?;
        if min == 0 && max.is_none() {
            self.string_any = Some(r);
        }
        Ok(r)
    }

    fn integer(&mut self) -> Result<Vec<Elem>, GrammarError> {
        let sign = self.b.rule(vec![lit(b"-"), vec![]])?;
        let mut nz = vec![class(ByteSet::range(b'1', b'9'))];
        nz.extend(self.repeat(self.digit, 0, Some(MAX_INT_DIGITS - 1))?);
        let int = self.b.rule(vec![lit(b"0"), nz])?;
        Ok(vec![Elem::Rule(sign), Elem::Rule(int)])
    }

    fn number(&mut self, integer: bool) -> Result<RuleId, GrammarError> {
        let mut v = self.integer()?;
        if !integer {
            let mut f = lit(b".");
            f.extend(self.repeat(self.digit, 1, Some(MAX_FRAC_DIGITS))?);
            let frac = self.b.rule(vec![f, vec![]])?;
            let mut e = vec![class(ByteSet::of(b"eE"))];
            let sign = self.b.rule(vec![vec![class(ByteSet::of(b"+-"))], vec![]])?;
            e.push(Elem::Rule(sign));
            e.extend(self.repeat(self.digit, 1, Some(MAX_EXP_DIGITS))?);
            let exp = self.b.rule(vec![e, vec![]])?;
            v.push(Elem::Rule(frac));
            v.push(Elem::Rule(exp));
        }
        self.b.rule(vec![v])
    }

    /// (any value, any object).
    fn any(&mut self) -> Result<(RuleId, RuleId), GrammarError> {
        if let Some(a) = self.any {
            return Ok(a);
        }
        let any = self.b.declare()?;
        self.any = Some((any, any));
        let s = self.string(0, None)?;
        let w = self.w();
        // member: string ws ':' ws any ws
        let mut m = vec![Elem::Rule(s), w];
        m.extend(lit(b":"));
        m.extend([w, Elem::Rule(any), w]);
        let member = self.b.rule(vec![m])?;
        let obj = self.list(b'{', b'}', member, member)?;
        let item = self.b.rule(vec![vec![Elem::Rule(any), w]])?;
        let arr = self.list(b'[', b']', item, item)?;
        let num = self.number(false)?;
        self.b.define(
            any,
            vec![
                vec![Elem::Rule(obj)],
                vec![Elem::Rule(arr)],
                vec![Elem::Rule(s)],
                vec![Elem::Rule(num)],
                lit(b"true"),
                lit(b"false"),
                lit(b"null"),
            ],
        )?;
        self.any = Some((any, obj));
        Ok((any, obj))
    }

    /// `open ws (first ("," ws rest)*)? close`, where `first` and `rest` end in ws.
    fn list(
        &mut self,
        open: u8,
        close: u8,
        first: RuleId,
        rest: RuleId,
    ) -> Result<RuleId, GrammarError> {
        let w = self.w();
        let mut more = lit(b",");
        more.extend([w, Elem::Rule(rest)]);
        let more = self.b.rule(vec![more])?;
        let more = self.star(more)?;
        let body = self
            .b
            .rule(vec![vec![Elem::Rule(first), Elem::Rule(more)], vec![]])?;
        let mut v = vec![class(ByteSet::byte(open)), w, Elem::Rule(body)];
        v.push(class(ByteSet::byte(close)));
        self.b.rule(vec![v])
    }

    fn compile(&mut self, s: &Json, path: &str, depth: usize) -> Result<RuleId, GrammarError> {
        if depth > MAX_DEPTH {
            return Err(GrammarError::TooLarge(format!(
                "schema nesting deeper than {MAX_DEPTH} at {path}"
            )));
        }
        let bad = |detail: &str| GrammarError::Schema {
            path: path.to_owned(),
            detail: detail.to_owned(),
        };
        let o = match s {
            Json::Bool(true) => return Ok(self.any()?.0),
            Json::Bool(false) => return Err(bad("the schema `false` admits no value")),
            Json::Object(o) => o,
            _ => return Err(bad("a schema must be an object or a boolean")),
        };
        for (k, _) in o {
            if !ANNOTATIONS.contains(&k.as_str()) && !SUPPORTED.contains(&k.as_str()) {
                return Err(GrammarError::Unsupported {
                    keyword: k.clone(),
                    path: path.to_owned(),
                    detail: refusal(k).into(),
                });
            }
        }
        let has = |k: &str| s.get(k).is_some();
        let constraining: Vec<&str> = SUPPORTED.iter().copied().filter(|k| has(k)).collect();
        let alone = |k: &str, allowed: &[&str]| -> Result<(), GrammarError> {
            match constraining
                .iter()
                .find(|&&x| x != k && !allowed.contains(&x))
            {
                Some(other) => Err(GrammarError::Unsupported {
                    keyword: k.into(),
                    path: path.to_owned(),
                    detail: format!("{k} together with {other} is not supported"),
                }),
                None => Ok(()),
            }
        };
        if let Some(r) = s.get("$ref") {
            alone("$ref", &[])?;
            let Json::String(r) = r else {
                return Err(bad("$ref must be a string"));
            };
            return self.reference(r, path);
        }
        if let Some(a) = s.get("anyOf") {
            alone("anyOf", &[])?;
            let Json::Array(a) = a else {
                return Err(bad("anyOf must be a non-empty array"));
            };
            if a.is_empty() {
                return Err(bad("anyOf must be a non-empty array"));
            }
            let alts = a
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    Ok(vec![Elem::Rule(self.compile(
                        x,
                        &format!("{path}/anyOf/{i}"),
                        depth + 1,
                    )?)])
                })
                .collect::<Result<Vec<_>, GrammarError>>()?;
            return self.b.rule(alts);
        }
        if let Some(a) = s.get("allOf") {
            alone("allOf", &[])?;
            return match a {
                Json::Array(a) if a.len() == 1 => {
                    self.compile(&a[0], &format!("{path}/allOf/0"), depth + 1)
                }
                _ => Err(GrammarError::Unsupported {
                    keyword: "allOf".into(),
                    path: path.to_owned(),
                    detail: "allOf is supported with exactly one schema".into(),
                }),
            };
        }
        let types = self.types(s, path)?;
        if has("enum") || has("const") {
            let key = if has("enum") { "enum" } else { "const" };
            alone(key, &["type", if key == "enum" { "const" } else { "enum" }])?;
            let mut values: Vec<&Json> = match s.get("enum") {
                Some(Json::Array(a)) => a.iter().collect(),
                Some(_) => return Err(bad("enum must be an array")),
                None => vec![],
            };
            if let Some(c) = s.get("const") {
                values = if has("enum") {
                    values.into_iter().filter(|v| *v == c).collect()
                } else {
                    vec![c]
                };
            }
            if let Some(t) = &types {
                values.retain(|v| t.iter().any(|t| is_type(v, t)));
            }
            if values.is_empty() {
                return Err(bad("no value satisfies enum, const and type together"));
            }
            let alts = values.iter().map(|v| lit(v.to_text().as_bytes())).collect();
            return self.b.rule(alts);
        }
        let Some(types) = types else {
            return Ok(self.any()?.0);
        };
        let mut alts = Vec::new();
        for t in &types {
            let r = match t.as_str() {
                "object" => self.object(s, path, depth)?,
                "array" => self.array(s, path, depth)?,
                "string" => {
                    let min = count(s, "minLength", path)?.unwrap_or(0);
                    let max = count(s, "maxLength", path)?;
                    if max.is_some_and(|m| m < min) {
                        return Err(bad("maxLength is below minLength"));
                    }
                    self.string(min, max)?
                }
                "number" => self.number(false)?,
                "integer" => self.number(true)?,
                "boolean" => self.b.rule(vec![lit(b"true"), lit(b"false")])?,
                _ => self.b.rule(vec![lit(b"null")])?,
            };
            alts.push(vec![Elem::Rule(r)]);
        }
        self.b.rule(alts)
    }

    /// The listed types, or those the keywords imply, or `None` for any value.
    fn types(&self, s: &Json, path: &str) -> Result<Option<Vec<String>>, GrammarError> {
        let bad = |detail: String| GrammarError::Schema {
            path: path.to_owned(),
            detail,
        };
        let check = |t: &str| {
            if TYPES.contains(&t) {
                Ok(t.to_owned())
            } else {
                Err(bad(format!("unknown type {t:?}")))
            }
        };
        let mut v = match s.get("type") {
            None => {
                let implied = |keys: &[&str]| keys.iter().any(|k| s.get(k).is_some());
                let mut v = Vec::new();
                for (keys, t) in [
                    (OBJECT_KEYS, "object"),
                    (ARRAY_KEYS, "array"),
                    (STRING_KEYS, "string"),
                ] {
                    if implied(keys) {
                        v.push(t.to_owned());
                    }
                }
                if v.is_empty() {
                    return Ok(None);
                }
                v
            }
            Some(Json::String(t)) => vec![check(t)?],
            Some(Json::Array(a)) if !a.is_empty() => a
                .iter()
                .map(|t| match t {
                    Json::String(t) => check(t),
                    _ => Err(bad("type must be a string or an array of strings".into())),
                })
                .collect::<Result<_, _>>()?,
            Some(_) => {
                return Err(bad(
                    "type must be a string or a non-empty array of strings".into()
                ));
            }
        };
        // "number" already admits every integer.
        if v.iter().any(|t| t == "number") {
            v.retain(|t| t != "integer");
        }
        v.dedup();
        Ok(Some(v))
    }

    fn reference(&mut self, r: &str, path: &str) -> Result<RuleId, GrammarError> {
        if let Some(&id) = self.refs.get(r) {
            return Ok(id);
        }
        let ptr = r
            .strip_prefix('#')
            .ok_or_else(|| GrammarError::Unsupported {
                keyword: "$ref".into(),
                path: path.to_owned(),
                detail: format!("only local references (\"#...\") are supported, not {r:?}"),
            })?;
        let target = self.root.pointer(ptr).ok_or_else(|| GrammarError::Schema {
            path: path.to_owned(),
            detail: format!("$ref {r:?} points nowhere"),
        })?;
        let id = self.b.declare()?;
        self.refs.insert(r.to_owned(), id);
        let body = self.compile(target, r, 0)?;
        self.b.define(id, vec![vec![Elem::Rule(body)]])?;
        Ok(id)
    }

    fn object(&mut self, s: &Json, path: &str, depth: usize) -> Result<RuleId, GrammarError> {
        let bad = |detail: &str| GrammarError::Schema {
            path: path.to_owned(),
            detail: detail.to_owned(),
        };
        let mut props: Vec<(String, Option<&Json>)> = match s.get("properties") {
            None => vec![],
            Some(Json::Object(o)) => {
                let mut v: Vec<(String, Option<&Json>)> = Vec::new();
                for (k, x) in o {
                    // A repeated key: the last occurrence wins, in its first position.
                    match v.iter_mut().find(|(n, _)| n == k) {
                        Some(e) => e.1 = Some(x),
                        None => v.push((k.clone(), Some(x))),
                    }
                }
                v
            }
            Some(_) => return Err(bad("properties must be an object")),
        };
        let required: Vec<String> = match s.get("required") {
            None => vec![],
            Some(Json::Array(a)) => a
                .iter()
                .map(|x| match x {
                    Json::String(x) => Ok(x.clone()),
                    _ => Err(bad("required must be an array of strings")),
                })
                .collect::<Result<_, _>>()?,
            Some(_) => return Err(bad("required must be an array of strings")),
        };
        let extra = match s.get("additionalProperties") {
            None | Some(Json::Bool(true)) => true,
            Some(Json::Bool(false)) => false,
            Some(Json::Object(o)) if o.iter().all(|(k, _)| ANNOTATIONS.contains(&k.as_str())) => {
                true
            }
            Some(Json::Object(_)) => {
                return Err(GrammarError::Unsupported {
                    keyword: "additionalProperties".into(),
                    path: path.to_owned(),
                    detail:
                        "a schema for additional properties is not supported; use false or true"
                            .into(),
                });
            }
            Some(_) => return Err(bad("additionalProperties must be a boolean or a schema")),
        };
        for r in &required {
            if !props.iter().any(|(n, _)| n == r) {
                if !extra {
                    return Err(bad(&format!(
                        "required property {r:?} is not in properties and additionalProperties is false"
                    )));
                }
                props.push((r.clone(), None));
            }
        }
        if props.is_empty() {
            if extra {
                return Ok(self.any()?.1);
            }
            let w = self.w();
            let mut v = lit(b"{");
            v.push(w);
            v.extend(lit(b"}"));
            return self.b.rule(vec![v]);
        }
        let w = self.w();
        let mut kv = Vec::with_capacity(props.len());
        for (name, schema) in &props {
            let value = match schema {
                Some(x) => self.compile(
                    x,
                    &format!("{path}/properties/{}", escape_ptr(name)),
                    depth + 1,
                )?,
                None => self.any()?.0,
            };
            let mut v = lit(quote(name).as_bytes());
            v.push(w);
            v.extend(lit(b":"));
            v.extend([w, Elem::Rule(value), w]);
            kv.push((self.b.rule(vec![v])?, required.contains(name)));
        }
        // after[i]: properties i.. once one has been written; first[i]: before any.
        let mut after = self.eps;
        let mut first = self.eps;
        for &(kv, req) in kv.iter().rev() {
            let mut a = lit(b",");
            a.extend([w, Elem::Rule(kv), Elem::Rule(after)]);
            let f = vec![Elem::Rule(kv), Elem::Rule(after)];
            let (na, nf) = if req {
                (self.b.rule(vec![a])?, self.b.rule(vec![f])?)
            } else {
                (
                    self.b.rule(vec![a, vec![Elem::Rule(after)]])?,
                    self.b.rule(vec![f, vec![Elem::Rule(first)]])?,
                )
            };
            after = na;
            first = nf;
        }
        let mut v = lit(b"{");
        v.extend([w, Elem::Rule(first)]);
        v.extend(lit(b"}"));
        self.b.rule(vec![v])
    }

    fn array(&mut self, s: &Json, path: &str, depth: usize) -> Result<RuleId, GrammarError> {
        let bad = |detail: &str| GrammarError::Schema {
            path: path.to_owned(),
            detail: detail.to_owned(),
        };
        let w = self.w();
        let prefix: Vec<&Json> = match s.get("prefixItems") {
            None => vec![],
            Some(Json::Array(a)) => a.iter().collect(),
            Some(_) => return Err(bad("prefixItems must be an array")),
        };
        if prefix.len() as u64 > MAX_COUNT {
            return Err(GrammarError::TooLarge(format!(
                "more than {MAX_COUNT} prefixItems at {path}"
            )));
        }
        let min = count(s, "minItems", path)?.unwrap_or(0);
        let mut max = count(s, "maxItems", path)?;
        let item = match s.get("items") {
            None | Some(Json::Bool(true)) => Some(self.any()?.0),
            Some(Json::Bool(false)) => None,
            Some(Json::Array(_)) => {
                return Err(GrammarError::Unsupported {
                    keyword: "items".into(),
                    path: path.to_owned(),
                    detail: "items as an array is draft 7 tuple form; use prefixItems".into(),
                });
            }
            Some(x) => Some(self.compile(x, &format!("{path}/items"), depth + 1)?),
        };
        let k = prefix.len();
        if item.is_none() {
            max = Some(max.map_or(k, |m| m.min(k)));
        }
        if max.is_some_and(|m| m < min) {
            return Err(bad(
                "no array length satisfies minItems, maxItems, prefixItems and items",
            ));
        }
        // e[j]: element j followed by whitespace.
        let mut elems = Vec::new();
        for (j, p) in prefix.iter().enumerate() {
            let r = self.compile(p, &format!("{path}/prefixItems/{j}"), depth + 1)?;
            elems.push(self.b.rule(vec![vec![Elem::Rule(r), w]])?);
        }
        let tail = match item {
            Some(i) => Some(self.b.rule(vec![vec![Elem::Rule(i), w]])?),
            None => None,
        };
        let elem = |j: usize| if j < k { Some(elems[j]) } else { tail };
        // a[j]: the rest once j elements are written. Past the prefix and the minimum,
        // an unbounded array repeats the tail.
        let last = max.unwrap_or(k.max(min));
        let mut a = match (max, tail) {
            (Some(_), _) | (None, None) => self.eps,
            (None, Some(t)) => {
                let mut more = lit(b",");
                more.extend([w, Elem::Rule(t)]);
                let more = self.b.rule(vec![more])?;
                self.star(more)?
            }
        };
        for j in (1..last).rev() {
            let e = elem(j).ok_or_else(|| bad("internal: no element schema"))?;
            let mut alt = lit(b",");
            alt.extend([w, Elem::Rule(e), Elem::Rule(a)]);
            a = if j >= min {
                self.b.rule(vec![alt, vec![]])?
            } else {
                self.b.rule(vec![alt])?
            };
        }
        let body = if last == 0 && max == Some(0) {
            self.eps
        } else {
            match elem(0) {
                Some(e) => {
                    // `a` is a[1] (or, with nothing bounded, the repetition).
                    let one = vec![Elem::Rule(e), Elem::Rule(a)];
                    if min == 0 {
                        self.b.rule(vec![one, vec![]])?
                    } else {
                        self.b.rule(vec![one])?
                    }
                }
                None => self.eps,
            }
        };
        let mut v = lit(b"[");
        v.extend([w, Elem::Rule(body)]);
        v.extend(lit(b"]"));
        self.b.rule(vec![v])
    }
}

/// A non-negative count keyword, at most [`MAX_COUNT`].
fn count(s: &Json, k: &str, path: &str) -> Result<Option<usize>, GrammarError> {
    match s.get(k) {
        None => Ok(None),
        Some(Json::Number(n)) => {
            let v = n
                .as_u64()
                .or_else(|| {
                    n.as_f64()
                        .filter(|f| f.fract() == 0.0 && *f >= 0.0)
                        .map(|_| u64::MAX)
                })
                .ok_or_else(|| GrammarError::Schema {
                    path: path.to_owned(),
                    detail: format!("{k} must be a non-negative integer"),
                })?;
            if v > MAX_COUNT {
                return Err(GrammarError::Unsupported {
                    keyword: k.into(),
                    path: path.to_owned(),
                    detail: format!("counts above {MAX_COUNT} are not supported"),
                });
            }
            Ok(usize::try_from(v).ok())
        }
        Some(_) => Err(GrammarError::Schema {
            path: path.to_owned(),
            detail: format!("{k} must be a non-negative integer"),
        }),
    }
}

fn is_type(v: &Json, t: &str) -> bool {
    match (v, t) {
        (Json::Null, "null")
        | (Json::Bool(_), "boolean")
        | (Json::String(_), "string")
        | (Json::Array(_), "array")
        | (Json::Object(_), "object")
        | (Json::Number(_), "number") => true,
        (Json::Number(n), "integer") => {
            n.is_i64() || n.is_u64() || n.as_f64().is_some_and(|f| f.fract() == 0.0)
        }
        _ => false,
    }
}

fn escape_ptr(s: &str) -> String {
    s.replace('~', "~0").replace('/', "~1")
}

/// One character of a JSON string: unescaped printable ASCII, a well-formed UTF-8
/// sequence, or an escape (`\u` escapes exclude surrogates, which parsers reject
/// unpaired).
fn string_char(b: &mut GrammarBuilder) -> Result<RuleId, GrammarError> {
    let cont = class(ByteSet::range(0x80, 0xBF));
    let r = |lo, hi| class(ByteSet::range(lo, hi));
    let ascii = ByteSet::range(0x20, 0x7F).without(b'"').without(b'\\');
    let hex = class(
        ByteSet::range(b'0', b'9')
            .union(ByteSet::range(b'a', b'f'))
            .union(ByteSet::range(b'A', b'F')),
    );
    let hex_not_d = class(
        ByteSet::range(b'0', b'9')
            .union(ByteSet::range(b'a', b'c'))
            .union(ByteSet::range(b'A', b'C'))
            .union(ByteSet::of(b"eEfF")),
    );
    let u = b.rule(vec![
        vec![hex_not_d, hex, hex, hex],
        vec![class(ByteSet::of(b"dD")), r(b'0', b'7'), hex, hex],
    ])?;
    let esc = b.rule(vec![
        vec![class(ByteSet::of(b"\"\\/bfnrt"))],
        vec![class(ByteSet::byte(b'u')), Elem::Rule(u)],
    ])?;
    b.rule(vec![
        vec![class(ascii)],
        vec![class(ByteSet::byte(b'\\')), Elem::Rule(esc)],
        vec![r(0xC2, 0xDF), cont],
        vec![class(ByteSet::byte(0xE0)), r(0xA0, 0xBF), cont],
        vec![r(0xE1, 0xEC), cont, cont],
        vec![class(ByteSet::byte(0xED)), r(0x80, 0x9F), cont],
        vec![r(0xEE, 0xEF), cont, cont],
        vec![class(ByteSet::byte(0xF0)), r(0x90, 0xBF), cont, cont],
        vec![r(0xF1, 0xF3), cont, cont, cont],
        vec![class(ByteSet::byte(0xF4)), r(0x80, 0x8F), cont, cont],
    ])
}
