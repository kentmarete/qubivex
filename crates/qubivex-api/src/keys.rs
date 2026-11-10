//! Named API keys (ADR-013 amendment 7): each key belongs to a client, so the gateway
//! knows who a request is from. The server keeps only each key's SHA-256.

use sha2::{Digest, Sha256};

use crate::limits::RateLimit;

/// One client's API key.
#[derive(Clone, PartialEq, Eq)]
pub struct ClientKey {
    /// The client's name, as the log shows it: `[A-Za-z0-9._-]`, 1 to 64 bytes, unique
    /// among the keys.
    pub client: String,
    /// The SHA-256 of the key. The key itself is never stored.
    pub key_sha256: [u8; 32],
    /// This client's allowance. `None`: the server's
    /// [`rate_limit`](crate::ServerConfig::rate_limit), if any.
    pub rate_limit: Option<RateLimit>,
}

impl std::fmt::Debug for ClientKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientKey")
            .field("client", &self.client)
            .field("rate_limit", &self.rate_limit)
            .finish_non_exhaustive()
    }
}

impl ClientKey {
    /// A key for `client` from the key itself.
    #[must_use]
    pub fn new(client: impl Into<String>, key: &str, rate_limit: Option<RateLimit>) -> Self {
        Self {
            client: client.into(),
            key_sha256: digest(key),
            rate_limit,
        }
    }

    /// A key for `client` from its SHA-256 in hex, as `qubivex keys` prints it.
    ///
    /// # Errors
    /// A message if `hex` is not 64 hex digits.
    pub fn from_hex(
        client: impl Into<String>,
        hex: &str,
        rate_limit: Option<RateLimit>,
    ) -> Result<Self, String> {
        let client = client.into();
        let bad = || format!("the key_sha256 of client {client:?} is not 64 hex digits");
        let hex = hex.trim();
        if hex.len() != 64 || !hex.is_ascii() {
            return Err(bad());
        }
        let mut key_sha256 = [0u8; 32];
        for (i, b) in key_sha256.iter_mut().enumerate() {
            *b = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).map_err(|_| bad())?;
        }
        Ok(Self {
            client,
            key_sha256,
            rate_limit,
        })
    }
}

fn digest(key: &str) -> [u8; 32] {
    Sha256::digest(key.as_bytes()).into()
}

/// The SHA-256 of `key` in lowercase hex: what [`ClientKey::from_hex`] takes.
#[must_use]
pub fn hash_key(key: &str) -> String {
    use std::fmt::Write;
    digest(key)
        .iter()
        .fold(String::with_capacity(64), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

/// Whose key `bearer` is: an index into `keys`. Every digest is compared in full, in
/// time independent of where a match is.
pub(crate) fn identify(keys: &[ClientKey], bearer: Option<&str>) -> Option<usize> {
    let bearer = bearer?;
    if keys.is_empty() {
        return None;
    }
    let d = digest(bearer);
    let mut found = None;
    for (i, k) in keys.iter().enumerate() {
        let diff = k
            .key_sha256
            .iter()
            .zip(&d)
            .fold(0u8, |acc, (x, y)| acc | (x ^ y));
        if diff == 0 {
            found = Some(i);
        }
    }
    found
}

/// Checks the names, the limits and that no two keys are the same.
pub(crate) fn validate(keys: &[ClientKey]) -> Result<(), String> {
    for (i, k) in keys.iter().enumerate() {
        let c = &k.client;
        if c.is_empty()
            || c.len() > 64
            || !c
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        {
            return Err(format!(
                "client name {c:?} must be 1 to 64 letters, digits, '.', '_' or '-'"
            ));
        }
        if let Some(r) = k.rate_limit {
            r.check().map_err(|e| format!("client {c:?}: {e}"))?;
        }
        for other in &keys[..i] {
            if &other.client == c {
                return Err(format!("client {c:?} is named twice"));
            }
            if other.key_sha256 == k.key_sha256 {
                return Err(format!(
                    "clients {:?} and {c:?} have the same key",
                    other.client
                ));
            }
        }
    }
    Ok(())
}

/// Who a request is, once its key is checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Caller {
    /// No key, or the shared key: `user` is the client's word.
    Unnamed,
    /// A named key: an index into [`ServerConfig::client_keys`](crate::ServerConfig).
    Client(usize),
}

/// The fairness client and the prefix-cache scope for a request from `caller` with
/// `user`. A named client's requests are fair among clients whatever `user` says, and
/// its scope is its own; `user` only separates its users within it. Client names
/// cannot hold a newline, so no `user` value can reach another client's scope, and
/// unnamed scopes live under a different prefix.
pub(crate) fn identity(
    keys: &[ClientKey],
    caller: Caller,
    user: Option<&str>,
) -> (Option<String>, Option<String>) {
    match caller {
        Caller::Unnamed => (
            user.map(|u| format!("user:{u}")),
            user.map(|u| format!("user:{u}")),
        ),
        Caller::Client(i) => {
            let c = &keys[i].client;
            let scope = match user {
                Some(u) => format!("key:{c}\n{u}"),
                None => format!("key:{c}"),
            };
            (Some(format!("key:{c}")), Some(scope))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Caller, ClientKey, hash_key, identify, identity, validate};
    use crate::limits::RateLimit;

    #[test]
    fn a_key_is_found_by_its_hash_and_nothing_else_matches() {
        let keys = vec![
            ClientKey::new("a", "key-a", None),
            ClientKey::from_hex("b", &hash_key("key-b"), None).unwrap(),
        ];
        assert_eq!(identify(&keys, Some("key-a")), Some(0));
        assert_eq!(identify(&keys, Some("key-b")), Some(1));
        assert_eq!(identify(&keys, Some("key-c")), None);
        assert_eq!(identify(&keys, Some("")), None);
        assert_eq!(identify(&keys, None), None);
        assert_eq!(identify(&[], Some("key-a")), None);
    }

    #[test]
    fn hex_digests_are_checked() {
        let h = hash_key("x");
        assert_eq!(h.len(), 64);
        assert_eq!(
            ClientKey::from_hex("c", &h.to_uppercase(), None).unwrap(),
            ClientKey::new("c", "x", None)
        );
        for bad in [
            "",
            "ab",
            &h[..63],
            &format!("{}zz", &h[..62]),
            "é".repeat(32).as_str(),
        ] {
            assert!(ClientKey::from_hex("c", bad, None).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn names_limits_and_duplicates_are_validated() {
        assert!(validate(&[ClientKey::new("app-1.prod_x", "k", None)]).is_ok());
        for name in ["", "has space", "a/b", "line\nbreak", &"x".repeat(65)] {
            assert!(
                validate(&[ClientKey::new(name, "k", None)]).is_err(),
                "{name:?}"
            );
        }
        let zero = Some(RateLimit {
            per_minute: 0,
            burst: 1,
        });
        assert!(validate(&[ClientKey::new("a", "k", zero)]).is_err());
        let same_name = [
            ClientKey::new("a", "k1", None),
            ClientKey::new("a", "k2", None),
        ];
        assert!(validate(&same_name).unwrap_err().contains("twice"));
        let same_key = [
            ClientKey::new("a", "k", None),
            ClientKey::new("b", "k", None),
        ];
        assert!(validate(&same_key).unwrap_err().contains("same key"));
    }

    #[test]
    fn named_clients_own_their_scope_and_fairness() {
        let keys = vec![
            ClientKey::new("a", "ka", None),
            ClientKey::new("b", "kb", None),
        ];
        let (fa, sa) = identity(&keys, Caller::Client(0), Some("u1"));
        let (fa2, sa2) = identity(&keys, Caller::Client(0), Some("u2"));
        // Fairness ignores `user`; the scope keeps users apart.
        assert_eq!(fa, fa2);
        assert_ne!(sa, sa2);
        // Another client with the same user is apart in both.
        let (fb, sb) = identity(&keys, Caller::Client(1), Some("u1"));
        assert_ne!(fa, fb);
        assert_ne!(sa, sb);
        // No unnamed `user` reaches a named scope or fairness client.
        for u in ["key:a", "key:a\nu1", "a", "a\nu1"] {
            let (f, s) = identity(&keys, Caller::Unnamed, Some(u));
            assert_ne!(f, fa);
            assert_ne!(s, sa);
            assert_ne!(s, identity(&keys, Caller::Client(0), None).1);
        }
        // A named client without `user` still has its own scope.
        assert_eq!(
            identity(&keys, Caller::Client(0), None),
            (Some("key:a".into()), Some("key:a".into()))
        );
        assert_eq!(identity(&keys, Caller::Unnamed, None), (None, None));
    }
}
