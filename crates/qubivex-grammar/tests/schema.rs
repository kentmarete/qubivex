//! JSON Schema grammars: every text the matcher lets through parses and validates.
//!
//! Random walks pick allowed bytes (no model), so they reach corners a model rarely
//! would. Each finished text is parsed by `serde_json` and checked by the validator
//! below, which is written from the JSON Schema specification independently of the
//! compiler. With `QUBIVEX_GRAMMAR_SAMPLES=<file>` the walks are also written out as
//! JSON lines for `fixtures/oracle/validate_json.py` (Python `jsonschema`).

#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs)]

use std::io::Write as _;
use std::sync::Arc;

use qubivex_grammar::{Grammar, GrammarError, Json, Matcher, TokenTrie, json_object, json_schema};
use serde_json::Value;

/// The schema suite: (name, schema).
pub const SUITE: &[(&str, &str)] = &[
    ("string", r#"{"type":"string"}"#),
    ("integer", r#"{"type":"integer"}"#),
    ("number", r#"{"type":"number"}"#),
    ("boolean", r#"{"type":"boolean"}"#),
    ("null", r#"{"type":"null"}"#),
    ("any", r"{}"),
    ("true", r"true"),
    ("type list", r#"{"type":["string","integer","null"]}"#),
    (
        "bounded string",
        r#"{"type":"string","minLength":2,"maxLength":5}"#,
    ),
    ("enum", r#"{"enum":["red","green",3,null,{"a":[1,2]}]}"#),
    ("typed enum", r#"{"type":"string","enum":["a","b",1]}"#),
    ("const", r#"{"const":"fixed \"value\" ✓"}"#),
    (
        "person",
        r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"email":{"type":"string"}},"required":["name","age"],"additionalProperties":false}"#,
    ),
    (
        "all optional",
        r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"null"},"c":{"type":"integer"}}}"#,
    ),
    (
        "closed empty",
        r#"{"type":"object","additionalProperties":false}"#,
    ),
    ("open object", r#"{"type":"object"}"#),
    (
        "required not declared",
        r#"{"type":"object","properties":{"x":{"type":"integer"}},"required":["x","y"]}"#,
    ),
    ("array", r#"{"type":"array","items":{"type":"integer"}}"#),
    (
        "bounded array",
        r#"{"type":"array","items":{"type":"string","maxLength":3},"minItems":2,"maxItems":4}"#,
    ),
    (
        "tuple",
        r#"{"type":"array","prefixItems":[{"type":"string"},{"type":"integer"}],"items":false}"#,
    ),
    (
        "tuple then items",
        r#"{"type":"array","prefixItems":[{"const":"head"}],"items":{"type":"boolean"},"minItems":3}"#,
    ),
    ("empty array only", r#"{"type":"array","maxItems":0}"#),
    (
        "anyOf",
        r#"{"anyOf":[{"type":"object","properties":{"kind":{"const":"a"},"n":{"type":"integer"}},"required":["kind","n"],"additionalProperties":false},{"type":"array","items":{"type":"number"},"maxItems":2}]}"#,
    ),
    (
        "allOf one",
        r#"{"allOf":[{"type":"string","maxLength":1}]}"#,
    ),
    (
        "recursive tree",
        r##"{"$defs":{"node":{"type":"object","properties":{"value":{"type":"integer"},"children":{"type":"array","items":{"$ref":"#/$defs/node"},"maxItems":2}},"required":["value"],"additionalProperties":false}},"$ref":"#/$defs/node"}"##,
    ),
    (
        "definitions",
        r##"{"definitions":{"tag":{"type":"string","enum":["x","y"]}},"type":"object","properties":{"tags":{"type":"array","items":{"$ref":"#/definitions/tag"}}},"required":["tags"]}"##,
    ),
    (
        "nested",
        r#"{"type":"object","properties":{"user":{"type":"object","properties":{"id":{"type":"integer"},"roles":{"type":"array","items":{"enum":["admin","user"]}}},"required":["id","roles"],"additionalProperties":false},"active":{"type":"boolean"}},"required":["user","active"],"additionalProperties":false}"#,
    ),
    (
        "openai style",
        r#"{"type":"object","properties":{"steps":{"type":"array","items":{"type":"object","properties":{"explanation":{"type":"string"},"output":{"type":"string"}},"required":["explanation","output"],"additionalProperties":false}},"final_answer":{"type":"string"}},"required":["steps","final_answer"],"additionalProperties":false}"#,
    ),
    (
        "nullable via anyOf",
        r#"{"type":"object","properties":{"v":{"anyOf":[{"type":"string"},{"type":"null"}]}},"required":["v"],"additionalProperties":false}"#,
    ),
    (
        "implied object",
        r#"{"properties":{"k":{"type":"number"}},"required":["k"]}"#,
    ),
    (
        "awkward names",
        r#"{"type":"object","properties":{"a\"b":{"type":"integer"},"ü/~":{"type":"integer"},"":{"type":"null"}},"required":["a\"b","ü/~",""],"additionalProperties":false}"#,
    ),
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % n as u64).unwrap()
    }
}

/// A random text through `g`, or `None` if it grew past `cap` bytes.
fn walk(g: &Arc<Grammar>, rng: &mut Rng, cap: usize) -> Option<Vec<u8>> {
    let mut m = Matcher::new(Arc::clone(g)).unwrap();
    let mut out = Vec::new();
    loop {
        let allowed: Vec<u8> = m.allowed_bytes().unwrap().iter().collect();
        if m.accepts() && (allowed.is_empty() || rng.below(3) == 0) {
            return Some(out);
        }
        assert!(
            !allowed.is_empty(),
            "dead state after {:?}",
            String::from_utf8_lossy(&out)
        );
        if out.len() > cap {
            return None;
        }
        // Lean towards closing so walks finish.
        let closers: Vec<u8> = allowed
            .iter()
            .copied()
            .filter(|b| b"\"]}".contains(b))
            .collect();
        let b = if !closers.is_empty() && rng.below(3) == 0 {
            closers[rng.below(closers.len())]
        } else {
            allowed[rng.below(allowed.len())]
        };
        assert!(m.advance(b).unwrap());
        out.push(b);
    }
}

// ---- An independent validator for the supported subset (JSON Schema 2020-12). ----

fn validate(v: &Value, s: &Value, root: &Value) -> Result<(), String> {
    let o = match s {
        Value::Bool(true) => return Ok(()),
        Value::Bool(false) => return Err("false schema".into()),
        Value::Object(o) => o,
        _ => return Err("bad schema".into()),
    };
    if let Some(Value::String(r)) = o.get("$ref") {
        let t = root.pointer(&r[1..]).ok_or("dangling ref")?;
        validate(v, t, root)?;
    }
    if let Some(Value::Array(a)) = o.get("anyOf")
        && !a.iter().any(|x| validate(v, x, root).is_ok())
    {
        return Err(format!("{v} matches no anyOf branch"));
    }
    if let Some(Value::Array(a)) = o.get("allOf") {
        for x in a {
            validate(v, x, root)?;
        }
    }
    if let Some(t) = o.get("type") {
        let ts: Vec<&str> = match t {
            Value::String(t) => vec![t],
            Value::Array(a) => a.iter().filter_map(Value::as_str).collect(),
            _ => vec![],
        };
        if !ts.iter().any(|t| has_type(v, t)) {
            return Err(format!("{v} is not of type {t}"));
        }
    }
    if let Some(Value::Array(e)) = o.get("enum")
        && !e.iter().any(|x| json_eq(x, v))
    {
        return Err(format!("{v} not in enum"));
    }
    if let Some(c) = o.get("const")
        && !json_eq(c, v)
    {
        return Err(format!("{v} is not the const"));
    }
    if let Value::String(st) = v {
        let n = st.chars().count() as u64;
        if o.get("minLength")
            .and_then(Value::as_u64)
            .is_some_and(|m| n < m)
        {
            return Err("too short".into());
        }
        if o.get("maxLength")
            .and_then(Value::as_u64)
            .is_some_and(|m| n > m)
        {
            return Err("too long".into());
        }
    }
    if let Value::Object(obj) = v {
        let props = o.get("properties").and_then(Value::as_object);
        for r in o
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if !obj.contains_key(r.as_str().unwrap()) {
                return Err(format!("missing required {r}"));
            }
        }
        for (k, x) in obj {
            match props.and_then(|p| p.get(k)) {
                Some(ps) => validate(x, ps, root)?,
                None => {
                    if let Some(ap) = o.get("additionalProperties") {
                        validate(x, ap, root).map_err(|e| format!("additional {k}: {e}"))?;
                    }
                }
            }
        }
    }
    if let Value::Array(a) = v {
        let prefix = o.get("prefixItems").and_then(Value::as_array);
        let k = prefix.map_or(0, Vec::len);
        for (i, x) in a.iter().enumerate() {
            if i < k {
                validate(x, &prefix.unwrap()[i], root)?;
            } else if let Some(it) = o.get("items") {
                validate(x, it, root)?;
            }
        }
        let n = a.len() as u64;
        if o.get("minItems")
            .and_then(Value::as_u64)
            .is_some_and(|m| n < m)
        {
            return Err("too few items".into());
        }
        if o.get("maxItems")
            .and_then(Value::as_u64)
            .is_some_and(|m| n > m)
        {
            return Err("too many items".into());
        }
    }
    Ok(())
}

fn has_type(v: &Value, t: &str) -> bool {
    match t {
        "null" => v.is_null(),
        "boolean" => v.is_boolean(),
        "string" => v.is_string(),
        "array" => v.is_array(),
        "object" => v.is_object(),
        "number" => v.is_number(),
        "integer" => v.is_i64() || v.is_u64() || v.as_f64().is_some_and(|f| f.fract() == 0.0),
        _ => false,
    }
}

/// JSON equality, numbers by value (1 == 1.0).
fn json_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| json_eq(a, b))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.get(k).is_some_and(|w| json_eq(v, w)))
        }
        _ => a == b,
    }
}

fn compile(src: &str) -> Arc<Grammar> {
    json_schema(&src.parse::<Json>().unwrap()).unwrap_or_else(|e| panic!("{src}: {e}"))
}

#[test]
fn random_walks_parse_and_validate() {
    let mut dump =
        std::env::var_os("QUBIVEX_GRAMMAR_SAMPLES").map(|p| std::fs::File::create(p).unwrap());
    let mut rng = Rng(0x5eed_1234_abcd_0001);
    for (name, src) in SUITE {
        let g = compile(src);
        let schema: Value = serde_json::from_str(src).unwrap();
        let (mut done, mut capped) = (0, 0);
        while done < 400 {
            let Some(text) = walk(&g, &mut rng, 4096) else {
                capped += 1;
                assert!(capped < 2000, "{name}: walks never finish");
                continue;
            };
            done += 1;
            let v: Value = serde_json::from_slice(&text).unwrap_or_else(|e| {
                panic!(
                    "{name}: {:?} does not parse: {e}",
                    String::from_utf8_lossy(&text)
                )
            });
            if let Err(e) = validate(&v, &schema, &schema) {
                panic!(
                    "{name}: {:?} does not validate: {e}",
                    String::from_utf8_lossy(&text)
                );
            }
            if let Some(f) = dump.as_mut() {
                let line =
                    serde_json::json!({"schema": schema, "text": String::from_utf8(text).unwrap()});
                writeln!(f, "{line}").unwrap();
            }
        }
    }
}

#[test]
fn json_mode_walks_are_objects() {
    let g = json_object().unwrap();
    let mut rng = Rng(77);
    let mut done = 0;
    while done < 400 {
        if let Some(t) = walk(&g, &mut rng, 4096) {
            let v: Value = serde_json::from_slice(&t).unwrap();
            assert!(v.is_object(), "{v}");
            done += 1;
        }
    }
}

#[test]
fn accepts_and_rejects_known_texts() {
    let ok = |src: &str, t: &str| {
        assert!(
            Matcher::matches(&compile(src), t.as_bytes()).unwrap(),
            "{src} should accept {t}"
        )
    };
    let no = |src: &str, t: &str| {
        assert!(
            !Matcher::matches(&compile(src), t.as_bytes()).unwrap(),
            "{src} should reject {t}"
        )
    };
    let person = SUITE[12].1;
    ok(person, r#"{"name":"Ann","age":3}"#);
    ok(
        person,
        "{ \"name\" : \"Ann\" ,\n  \"age\": -0, \"email\": \"a@b\" }",
    );
    no(person, r#"{"age":3,"name":"Ann"}"#); // schema order
    no(person, r#"{"name":"Ann"}"#);
    no(person, r#"{"name":"Ann","age":3,"x":1}"#);
    no(person, r#"{"name":"Ann","age":3.5}"#);
    no(person, r#"{"name":"Ann","age":01}"#);
    ok(r#"{"type":"number"}"#, "-12.5e+07");
    no(r#"{"type":"number"}"#, "1e999");
    no(r#"{"type":"number"}"#, "1.");
    no(r#"{"type":"number"}"#, "+1");
    no(r#"{"type":"number"}"#, "1234567890123456789");
    ok(r#"{"type":"string"}"#, r#""tab\t é é 😀""#);
    no(r#"{"type":"string"}"#, r#""lone \ud800""#);
    no(r#"{"type":"string"}"#, "\"raw\nnewline\"");
    no(r#"{"type":"string"}"#, "\"bad utf8 \u{0}\"");
    no(r#"{"type":"string"}"#, "\"\u{fffd}\"\u{0}");
    ok(r#"{"type":"string","maxLength":2}"#, r#""\n😀""#);
    no(r#"{"type":"string","maxLength":2}"#, r#""abc""#);
    ok(SUITE[19].1, r#"["a", 1]"#);
    no(SUITE[19].1, r#"["a", 1, 2]"#);
    ok(SUITE[19].1, r#"["a"]"#); // prefixItems sets no minimum
    no(SUITE[19].1, r#"[1]"#);
    ok(
        SUITE[24].1,
        r#"{"value":1,"children":[{"value":2,"children":[]}]}"#,
    );
    // Whitespace is bounded per gap.
    ok(r#"{"type":"null"}"#, &format!("{}null", " ".repeat(16)));
    no(r#"{"type":"null"}"#, &format!("{}null", " ".repeat(17)));
    // Invalid UTF-8 never passes a string.
    assert!(!Matcher::matches(&compile(r#"{"type":"string"}"#), b"\"\xC0\xAF\"").unwrap());
    assert!(!Matcher::matches(&compile(r#"{"type":"string"}"#), b"\"\xED\xA0\x80\"").unwrap());
}

#[test]
fn refuses_what_it_cannot_enforce() {
    let refused = [
        (r#"{"oneOf":[{"type":"string"}]}"#, "oneOf"),
        (r#"{"type":"string","pattern":"^a$"}"#, "pattern"),
        (r#"{"type":"string","format":"email"}"#, "format"),
        (r#"{"type":"integer","minimum":3}"#, "minimum"),
        (r#"{"type":"array","uniqueItems":true}"#, "uniqueItems"),
        (r#"{"not":{"type":"null"}}"#, "not"),
        (
            r#"{"type":"object","additionalProperties":{"type":"string"}}"#,
            "additionalProperties",
        ),
        (r#"{"$ref":"https://example.com/s.json"}"#, "$ref"),
        (r#"{"type":"string","nullable":true}"#, "nullable"),
        (r#"{"allOf":[{"type":"string"},{"maxLength":2}]}"#, "allOf"),
        (r#"{"anyOf":[{"type":"string"}],"type":"string"}"#, "anyOf"),
        (r#"{"type":"string","maxLength":100000}"#, "maxLength"),
        (r#"{"type":"array","items":[{"type":"string"}]}"#, "items"),
    ];
    for (src, kw) in refused {
        match json_schema(&src.parse::<Json>().unwrap()) {
            Err(GrammarError::Unsupported { keyword, .. }) => assert_eq!(keyword, kw, "{src}"),
            other => panic!("{src}: expected {kw} refused, got {other:?}"),
        }
    }
    let invalid = [
        r"false",
        r#"{"type":"strin"}"#,
        r#"{"enum":[1,2],"type":"string"}"#,
        r##"{"$ref":"#/$defs/missing"}"##,
        r##"{"$ref":"#"}"##,
        r#"{"type":"object","properties":{},"required":["a"],"additionalProperties":false}"#,
        r#"{"type":"array","minItems":3,"maxItems":2}"#,
        r#"{"type":"array","prefixItems":[{}],"items":false,"minItems":2}"#,
        r#"{"type":"string","minLength":-1}"#,
    ];
    for src in invalid {
        assert!(
            matches!(
                json_schema(&src.parse::<Json>().unwrap()),
                Err(GrammarError::Schema { .. })
            ),
            "{src} should be invalid"
        );
    }
}

#[test]
fn properties_follow_schema_order() {
    // serde_json would sort these keys; the grammar must not.
    let g = compile(
        r#"{"type":"object","properties":{"zeta":{"type":"null"},"alpha":{"type":"null"}},"required":["zeta","alpha"]}"#,
    );
    assert!(Matcher::matches(&g, br#"{"zeta":null,"alpha":null}"#).unwrap());
    assert!(!Matcher::matches(&g, br#"{"alpha":null,"zeta":null}"#).unwrap());
}

/// A vocabulary like a byte-level BPE one: every single byte, plus multi-byte pieces
/// that cross JSON structure.
fn vocab() -> (Vec<Option<Vec<u8>>>, Vec<u32>) {
    let mut v: Vec<Option<Vec<u8>>> = (0..=255u8).map(|b| Some(vec![b])).collect();
    for p in [
        "\"", "\":", "\",", "\"}", "{\"", "name", "\"name\"", " \"", "}}", "]}", "\n  ", "  ",
        "true", "false", "null", "nul", "12", "1.5", "e+", "\\n", "\\u00", "😀", "é", "\"a\":1}",
        ",\"", "[{", "}]", "-0", "00", ":\"", " {", "age", "\"age\":", "\u{fffd}",
    ] {
        v.push(Some(p.as_bytes().to_vec()));
    }
    v.push(None); // a special token
    v.push(Some(vec![])); // a token with no bytes
    let eos = u32::try_from(v.len()).unwrap();
    v.push(Some(b"</s>".to_vec())); // eos: bytes are ignored
    (v, vec![eos, 1])
}

#[test]
fn masks_match_a_per_token_check() {
    let (bytes, eos) = vocab();
    let trie = TokenTrie::new(bytes.clone(), &eos);
    let mut rng = Rng(9);
    for (name, src) in SUITE {
        let g = compile(src);
        for _ in 0..6 {
            let Some(text) = walk(&g, &mut rng, 256) else {
                continue;
            };
            // Check the mask at every prefix of the walk.
            let mut m = Matcher::new(Arc::clone(&g)).unwrap();
            for cut in 0..=text.len() {
                let mask = m.mask(&trie).unwrap();
                for (id, b) in bytes.iter().enumerate() {
                    let id32 = u32::try_from(id).unwrap();
                    let want = if eos.contains(&id32) {
                        m.accepts()
                    } else {
                        b.as_ref().is_some_and(|b| {
                            !b.is_empty() && {
                                let mut c = Matcher::new(Arc::clone(&g)).unwrap();
                                c.advance_bytes(&text[..cut]).unwrap()
                                    && c.advance_bytes(b).unwrap()
                            }
                        })
                    };
                    let got = mask[id / 64] >> (id % 64) & 1 == 1;
                    assert_eq!(
                        got,
                        want,
                        "{name}: token {id} after {:?}",
                        String::from_utf8_lossy(&text[..cut])
                    );
                    assert_eq!(m.allows_token(&trie, id32).unwrap(), want);
                }
                if cut < text.len() {
                    assert!(m.advance(text[cut]).unwrap());
                }
            }
        }
    }
}

#[test]
fn rejected_bytes_leave_the_state_alone() {
    let g = compile(r#"{"type":"boolean"}"#);
    let mut m = Matcher::new(g).unwrap();
    assert!(m.advance_bytes(b"tr").unwrap());
    assert!(!m.advance_bytes(b"uX").unwrap());
    assert!(!m.advance(b'x').unwrap());
    assert!(m.advance_bytes(b"ue").unwrap());
    assert!(m.accepts());
    assert!(m.allowed_bytes().unwrap().is_empty());
}

#[test]
fn repetition_does_not_grow_the_state() {
    let g = compile(r#"{"type":"array","items":{"type":"string"}}"#);
    let mut m = Matcher::new(g).unwrap();
    assert!(m.advance_bytes(b"[\"").unwrap());
    for _ in 0..2000 {
        assert!(m.advance_bytes(b"ab").unwrap());
    }
    assert!(m.advance_bytes(b"\",\"cd\",\"").unwrap());
    let before = m.size();
    for _ in 0..2000 {
        assert!(m.advance_bytes(b"cd\",\"").unwrap());
    }
    assert_eq!(
        m.size(),
        before,
        "states and frames must not grow with repetition"
    );
}
