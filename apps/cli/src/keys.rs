//! `qubivex keys`: named API keys for `qubivex serve` (ADR-013 amendment 7). The server
//! stores only each key's SHA-256, so the key is shown once, here.

use std::io::Read;

use qubivex_api::hash_key;

/// 32 random bytes from the operating system, as `qvx_` and 64 hex digits.
fn new_key() -> Result<String, String> {
    let mut b = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .map_err(|e| format!("cannot read random bytes from /dev/urandom: {e}"))?;
    Ok(b.iter().fold(String::from("qvx_"), |mut s, x| {
        s.push_str(&format!("{x:02x}"));
        s
    }))
}

/// The `[[serve.client_keys]]` table for `client` with the key's hash.
fn table(client: &str, hash: &str, rate_limit: Option<u32>, burst: Option<u32>) -> String {
    let mut t = format!("[[serve.client_keys]]\nclient = \"{client}\"\nkey_sha256 = \"{hash}\"\n");
    if let Some(r) = rate_limit {
        t.push_str(&format!("rate_limit = {r}\n"));
    }
    if let Some(b) = burst {
        t.push_str(&format!("rate_limit_burst = {b}\n"));
    }
    t
}

/// `keys new`: a fresh key for `client`, and the table to add to the config file.
pub(crate) fn new(
    client: &str,
    rate_limit: Option<u32>,
    burst: Option<u32>,
) -> Result<String, String> {
    if burst.is_some() && rate_limit.is_none() {
        return Err("--rate-limit-burst needs --rate-limit".into());
    }
    let probe = qubivex_api::ClientKey::new(client, "", None);
    qubivex_api::ServerConfig {
        client_keys: vec![probe],
        ..qubivex_api::ServerConfig::default()
    }
    .validate()?;
    let key = new_key()?;
    Ok(format!(
        "Key for {client} (shown once; give it to the client, keep no copy):\n\n  {key}\n\n\
         Add this to the serve config file (chmod 600), then restart the server:\n\n{}",
        table(client, &hash_key(&key), rate_limit, burst)
    ))
}

/// `keys hash`: the SHA-256 of a key read from standard input, for a key made
/// elsewhere. One trailing newline is not part of the key.
pub(crate) fn hash() -> Result<String, String> {
    let mut s = String::new();
    std::io::stdin()
        .read_to_string(&mut s)
        .map_err(|e| format!("cannot read the key from standard input: {e}"))?;
    let key = s
        .strip_suffix("\r\n")
        .or_else(|| s.strip_suffix('\n'))
        .unwrap_or(&s);
    if key.is_empty() {
        return Err("no key on standard input".into());
    }
    Ok(format!("{}\n", hash_key(key)))
}

#[cfg(test)]
mod tests {
    use super::{new, new_key, table};

    #[test]
    fn keys_are_random_and_long() {
        let (a, b) = (new_key().unwrap(), new_key().unwrap());
        assert_ne!(a, b);
        assert_eq!(a.len(), 4 + 64);
        assert!(a.starts_with("qvx_"));
    }

    #[test]
    fn the_printed_table_parses_back() {
        let out = new("app-1", Some(60), Some(5)).unwrap();
        let key = out.lines().nth(2).unwrap().trim();
        let toml_part = &out[out.find("[[serve").unwrap()..];
        assert!(toml_part.contains(&qubivex_api::hash_key(key)));
        assert!(toml_part.contains("rate_limit = 60\nrate_limit_burst = 5\n"));
        assert!(!toml_part.contains(key));
        assert_eq!(
            table("c", "h", None, None),
            "[[serve.client_keys]]\nclient = \"c\"\nkey_sha256 = \"h\"\n"
        );
        assert!(new("bad name", None, None).is_err());
        assert!(new("ok", None, Some(3)).is_err());
    }
}
