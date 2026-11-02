//! Small parsers shared by the probes. Inputs are untrusted text.

use std::collections::BTreeMap;

use crate::HardwareError;

/// Largest CPU index accepted from a range list (guards against huge ranges).
const MAX_CPU_INDEX: u32 = 1 << 16;

/// Parses a kernel CPU list such as `0-3,8,10-11` into sorted, unique indices.
pub(crate) fn cpu_list(what: &str, s: &str) -> Result<Vec<u32>, HardwareError> {
    let s = s.trim();
    let mut v = Vec::new();
    if s.is_empty() {
        return Ok(v);
    }
    for part in s.split(',') {
        let part = part.trim();
        let (lo, hi) = match part.split_once('-') {
            Some((a, b)) => (num(what, a)?, num(what, b)?),
            None => {
                let n = num(what, part)?;
                (n, n)
            }
        };
        if lo > hi || hi > MAX_CPU_INDEX {
            return Err(HardwareError::parse(what, format!("bad range {part:?}")));
        }
        v.extend(lo..=hi);
    }
    v.sort_unstable();
    v.dedup();
    Ok(v)
}

fn num(what: &str, s: &str) -> Result<u32, HardwareError> {
    s.trim()
        .parse()
        .map_err(|_| HardwareError::parse(what, format!("not a number: {s:?}")))
}

/// Parses an unsigned integer as the kernel writes it (digits, optional newline).
pub(crate) fn u64_value(what: &str, s: &str) -> Result<u64, HardwareError> {
    s.trim()
        .parse()
        .map_err(|_| HardwareError::parse(what, format!("not an unsigned integer: {:?}", s.trim())))
}

/// `key: value` lines (sysctl, /proc/cpuinfo, `sw_vers`). The first occurrence wins.
pub(crate) fn colon_map(s: &str) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    for line in s.lines() {
        if let Some((k, v)) = line.split_once(':') {
            m.entry(k.trim().to_owned())
                .or_insert_with(|| v.trim().to_owned());
        }
    }
    m
}

/// `KEY=value` lines with optional double quotes (os-release).
pub(crate) fn env_map(s: &str) -> BTreeMap<String, String> {
    s.lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim().to_owned(), v.trim().trim_matches('"').to_owned()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_lists() {
        assert_eq!(
            cpu_list("t", "0-3,8,10-11\n").unwrap(),
            vec![0, 1, 2, 3, 8, 10, 11]
        );
        assert_eq!(cpu_list("t", "5").unwrap(), vec![5]);
        assert_eq!(cpu_list("t", "").unwrap(), Vec::<u32>::new());
        assert!(cpu_list("t", "3-1").is_err());
        assert!(cpu_list("t", "0-4294967295").is_err());
        assert!(cpu_list("t", "a").is_err());
    }

    #[test]
    fn maps() {
        let m = colon_map("hw.ncpu: 11\nmodel name\t: X: Y\n");
        assert_eq!(m["hw.ncpu"], "11");
        assert_eq!(m["model name"], "X: Y");
        let e = env_map("NAME=\"Ubuntu\"\nVERSION_ID=24.04\n");
        assert_eq!(e["NAME"], "Ubuntu");
        assert_eq!(e["VERSION_ID"], "24.04");
    }
}
