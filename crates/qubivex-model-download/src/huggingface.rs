//! Hugging Face: `GET /api/models/{repo}/revision/{rev}` pins the commit, the paginated
//! `tree/{sha}` listing gives sizes and checksums, and `resolve/{sha}/{path}` serves the
//! bytes, usually through a redirect to a signed CDN URL.

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;
use ureq::http::{HeaderMap, StatusCode};

use crate::http::{Http, Reply, retry_after};
use crate::{Checksum, DownloadError, ModelProvider, Opened, RemoteFile, RepoListing, SearchHit};

/// The public hub.
pub const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// A transfer request's time budget. ureq's body timeout covers a whole body and cannot
/// see a stall, so a request ends here and the next continues from its offset. This is
/// also the longest a stalled connection can hold up a cancel.
pub const DEFAULT_BUDGET: Duration = Duration::from_secs(120);

/// The name in the registry.
const PROVIDER: &str = "huggingface";
/// Largest metadata reply read into memory.
const MAX_JSON: u64 = 64 << 20;
/// Pages of a tree listing (1000 entries each), so a server cannot page forever.
const MAX_PAGES: usize = 64;

/// An access token. It prints as `Token(redacted)` and is never put in an error, and
/// it has no command-line flag, because arguments show up in process listings.
#[derive(Clone, PartialEq, Eq)]
pub struct Token(String);

impl fmt::Debug for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Token(redacted)")
    }
}

impl Token {
    /// Wraps a token.
    ///
    /// # Errors
    /// [`DownloadError::InvalidToken`] if a header could not carry it, which also stops
    /// a token file from injecting a header.
    pub fn new(token: &str) -> Result<Self, DownloadError> {
        let t = token.trim();
        if t.is_empty() || !t.bytes().all(|b| b.is_ascii_graphic()) {
            return Err(DownloadError::InvalidToken);
        }
        Ok(Self(t.to_owned()))
    }

    /// The token the Hugging Face tools use: `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, then
    /// the file `hf auth login` writes.
    ///
    /// # Errors
    /// [`DownloadError::InvalidToken`] if the token found is unusable.
    pub fn from_environment() -> Result<Option<Self>, DownloadError> {
        let var = |n: &str| std::env::var(n).ok().filter(|v| !v.trim().is_empty());
        if let Some(t) = var("HF_TOKEN").or_else(|| var("HUGGING_FACE_HUB_TOKEN")) {
            return Self::new(&t).map(Some);
        }
        let file = std::env::var_os("HF_TOKEN_PATH")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HF_HOME").map(|h| Path::new(&h).join("token")))
            .or_else(|| {
                std::env::var_os("HOME").map(|h| Path::new(&h).join(".cache/huggingface/token"))
            });
        match file.and_then(|f| std::fs::read_to_string(f).ok()) {
            Some(t) if !t.trim().is_empty() => Self::new(&t).map(Some),
            _ => Ok(None),
        }
    }
}

/// A Hugging Face hub, or a mirror of one.
pub struct HuggingFace {
    http: Http,
}

impl fmt::Debug for HuggingFace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HuggingFace")
            .field("endpoint", &self.http.endpoint())
            .finish_non_exhaustive()
    }
}

#[derive(Deserialize)]
struct ModelInfo {
    sha: Option<String>,
}

#[derive(Deserialize)]
struct Lfs {
    oid: String,
}

#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    oid: Option<String>,
    lfs: Option<Lfs>,
}

#[derive(Deserialize)]
struct Hit {
    id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    gated: serde_json::Value,
}

/// Percent-encodes everything but RFC 3986 unreserved characters.
fn encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn encode_path(path: &str) -> String {
    path.split('/').map(encode).collect::<Vec<_>>().join("/")
}

/// `name` or `owner/name`, in the characters the hub allows.
fn valid_repository(repo: &str) -> bool {
    let part = |p: &str| {
        !p.is_empty()
            && p.len() <= 96
            && !p.contains("..")
            && !p.starts_with(['.', '-'])
            && !p.ends_with(['.', '-'])
            && p.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    };
    let parts: Vec<&str> = repo.split('/').collect();
    parts.len() <= 2 && parts.iter().all(|p| part(p))
}

fn is_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// The `next` target of a `Link` header.
fn next_link(headers: &HeaderMap) -> Option<String> {
    headers
        .get_all("link")
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .find(|l| l.contains("rel=\"next\""))
        .and_then(|l| Some(l.split_once('<')?.1.split_once('>')?.0.to_owned()))
}

/// A `Content-Range` header's start, checked against what was asked for.
fn range_start(headers: &HeaderMap) -> Option<u64> {
    let v = headers.get("content-range")?.to_str().ok()?;
    let (start, _) = v.strip_prefix("bytes ")?.split_once('-')?;
    start.trim().parse().ok()
}

impl HuggingFace {
    /// A client for `endpoint`. A transfer's request ends after `budget`, and continues
    /// in a new request. `http` is accepted only for a loopback endpoint.
    ///
    /// # Errors
    /// [`DownloadError::InvalidEndpoint`].
    pub fn new(
        endpoint: &str,
        token: Option<Token>,
        budget: Duration,
    ) -> Result<Self, DownloadError> {
        Ok(Self {
            http: Http::new(endpoint, token.map(|t| t.0), budget)?,
        })
    }

    /// Maps a reply to a final or retryable failure, or passes it through.
    fn check(&self, what: &str, reply: Reply) -> Result<Reply, DownloadError> {
        let status = reply.status;
        if status.is_success() {
            return Ok(reply);
        }
        let code = status.as_u16();
        Err(match status {
            StatusCode::NOT_FOUND => DownloadError::NotFound(what.to_owned()),
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => DownloadError::Unauthorized {
                what: what.to_owned(),
                status: code,
                hint: if self.http.has_token() {
                    "the token has no access; a gated model needs its terms accepted on its page"
                } else {
                    "it may not exist, or may be private or gated: set HF_TOKEN"
                },
            },
            StatusCode::REQUEST_TIMEOUT | StatusCode::TOO_MANY_REQUESTS => {
                DownloadError::Transient {
                    reason: format!("{what}: HTTP {code}"),
                    retry_after: retry_after(&reply.headers),
                }
            }
            s if s.is_server_error() => DownloadError::Transient {
                reason: format!("{what}: HTTP {code}"),
                retry_after: retry_after(&reply.headers),
            },
            _ => DownloadError::Protocol(format!("{what}: unexpected HTTP {code}")),
        })
    }

    fn json<T: for<'de> Deserialize<'de>>(
        what: &str,
        reply: Reply,
    ) -> Result<(HeaderMap, T), DownloadError> {
        let Reply {
            headers, mut body, ..
        } = reply;
        let bytes = body
            .with_config()
            .limit(MAX_JSON)
            .read_to_vec()
            .map_err(|_| DownloadError::transient(format!("{what}: the reply was cut short")))?;
        let value = serde_json::from_slice(&bytes)
            .map_err(|_| DownloadError::Protocol(format!("{what}: the reply is not usable")))?;
        Ok((headers, value))
    }

    fn list_tree(&self, repo: &str, sha: &str) -> Result<Vec<RemoteFile>, DownloadError> {
        let what = format!("{repo} at {sha}");
        let mut url = format!(
            "{}/api/models/{}/tree/{sha}?recursive=true",
            self.http.endpoint(),
            encode_path(repo)
        );
        let mut files = Vec::new();
        for _ in 0..MAX_PAGES {
            let reply = self.check(&what, self.http.get(&url, None)?)?;
            let (headers, page): (_, Vec<TreeEntry>) = Self::json(&what, reply)?;
            for e in page {
                if e.kind != "file" {
                    continue;
                }
                let checksum = match (&e.lfs, &e.oid) {
                    (Some(l), _) if is_hex(&l.oid, 64) => {
                        Some(Checksum::Sha256(l.oid.to_ascii_lowercase()))
                    }
                    (None, Some(o)) if is_hex(o, 40) => {
                        Some(Checksum::GitBlobSha1(o.to_ascii_lowercase()))
                    }
                    _ => None,
                };
                files.push(RemoteFile {
                    path: e.path,
                    size: e.size,
                    checksum,
                });
            }
            match next_link(&headers) {
                // The next request carries the token, so it stays on this origin.
                Some(next) if self.http.same_origin(&next) => url = next,
                Some(_) => {
                    return Err(DownloadError::Protocol(format!(
                        "{what}: the listing pointed to another host"
                    )));
                }
                None => return Ok(files),
            }
        }
        Err(DownloadError::Protocol(format!(
            "{what}: the listing has too many pages"
        )))
    }
}

impl ModelProvider for HuggingFace {
    fn name(&self) -> &str {
        PROVIDER
    }

    fn resolve(
        &self,
        repository: &str,
        revision: Option<&str>,
    ) -> Result<RepoListing, DownloadError> {
        if !valid_repository(repository) {
            return Err(DownloadError::InvalidRepository(repository.to_owned()));
        }
        let revision = revision.unwrap_or("main");
        let what = format!("{repository} at {revision}");
        let url = format!(
            "{}/api/models/{}/revision/{}",
            self.http.endpoint(),
            encode_path(repository),
            encode(revision)
        );
        let reply = self.check(&what, self.http.get(&url, None)?)?;
        let (_, info): (_, ModelInfo) = Self::json(&what, reply)?;
        let sha = info
            .sha
            .filter(|s| is_hex(s, 40))
            .ok_or_else(|| DownloadError::Protocol(format!("{what}: no usable commit id")))?
            .to_ascii_lowercase();
        let files = self.list_tree(repository, &sha)?;
        Ok(RepoListing {
            repository: repository.to_owned(),
            revision: sha,
            files,
        })
    }

    fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>, DownloadError> {
        let url = format!(
            "{}/api/models?search={}&limit={limit}&sort=downloads&direction=-1",
            self.http.endpoint(),
            encode(query)
        );
        let what = format!("the search for {query:?}");
        let reply = self.check(&what, self.http.get(&url, None)?)?;
        let (_, hits): (_, Vec<Hit>) = Self::json(&what, reply)?;
        Ok(hits
            .into_iter()
            .map(|h| SearchHit {
                formats: ["gguf", "safetensors"]
                    .into_iter()
                    .filter(|f| h.tags.iter().any(|t| t == f))
                    .map(str::to_owned)
                    .collect(),
                gated: !matches!(
                    h.gated,
                    serde_json::Value::Null | serde_json::Value::Bool(false)
                ),
                repository: h.id,
                downloads: h.downloads,
                likes: h.likes,
            })
            .collect())
    }

    fn open(
        &self,
        listing: &RepoListing,
        path: &str,
        offset: u64,
    ) -> Result<Opened, DownloadError> {
        let what = format!("{} in {}", path, listing.repository);
        let url = format!(
            "{}/{}/resolve/{}/{}",
            self.http.endpoint(),
            encode_path(&listing.repository),
            listing.revision,
            encode_path(path)
        );
        let range = (offset > 0).then_some(offset);
        let mut reply = self.http.get(&url, range)?;
        if reply.status == StatusCode::RANGE_NOT_SATISFIABLE {
            reply = self.http.get(&url, None)?;
        }
        let reply = self.check(&what, reply)?;
        let start = if reply.status == StatusCode::PARTIAL_CONTENT {
            let start = range_start(&reply.headers);
            if start != Some(offset) {
                return Err(DownloadError::Protocol(format!(
                    "{what}: the server sent a different range than the one asked for"
                )));
            }
            offset
        } else {
            0
        };
        Ok(Opened {
            start,
            body: Box::new(reply.body.into_reader()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repository_ids() {
        for ok in ["gpt2", "Acme/Model-0.6B", "a_b/c.d-e"] {
            assert!(valid_repository(ok), "{ok}");
        }
        for bad in [
            "", "/", "a/", "/a", "a/b/c", "../a", "a/..", "a/b?x", "a/b#c", "a b/c", ".a/b",
        ] {
            assert!(!valid_repository(bad), "{bad}");
        }
    }

    #[test]
    fn encoding_keeps_paths_and_escapes_the_rest() {
        assert_eq!(encode("refs/pr/1"), "refs%2Fpr%2F1");
        assert_eq!(encode_path("a b/c?d"), "a%20b/c%3Fd");
    }

    #[test]
    fn tokens_stay_out_of_debug_output() {
        let t = Token::new("hf_secret\n").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(t.0, "hf_secret");
        assert_eq!(format!("{t:?}"), "Token(redacted)");
        assert!(Token::new("a\r\nX-Evil: 1").is_err());
        assert!(Token::new("  ").is_err());
    }
}
