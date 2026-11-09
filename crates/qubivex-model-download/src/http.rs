//! Blocking HTTP with this crate's rules: redirects are followed here rather than in the
//! client, so the token policy and the scheme rules are in one place, and are testable.

use std::time::Duration;

use ureq::http::{HeaderMap, StatusCode, Uri};
use ureq::{Agent, Body};

use crate::DownloadError;

const MAX_REDIRECTS: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Origin {
    scheme: String,
    host: String,
    port: u16,
}

impl Origin {
    fn of(uri: &Uri) -> Option<Self> {
        let scheme = uri.scheme_str()?.to_ascii_lowercase();
        let port = uri.port_u16().or(match scheme.as_str() {
            "https" => Some(443),
            "http" => Some(80),
            _ => None,
        })?;
        Some(Self {
            host: uri.host()?.to_ascii_lowercase(),
            scheme,
            port,
        })
    }

    fn is_loopback(&self) -> bool {
        matches!(self.host.as_str(), "localhost" | "127.0.0.1" | "[::1]")
    }

    fn describe(&self) -> String {
        format!("{}://{}:{}", self.scheme, self.host, self.port)
    }
}

/// A reply with its status, after redirects.
pub(crate) struct Reply {
    pub(crate) status: StatusCode,
    pub(crate) headers: HeaderMap,
    pub(crate) body: Body,
}

/// The client for one endpoint.
pub(crate) struct Http {
    agent: Agent,
    endpoint: String,
    origin: Origin,
    /// Plain HTTP is allowed only for a loopback endpoint, which is how tests run.
    plain: bool,
    token: Option<String>,
}

impl Http {
    pub(crate) fn new(
        endpoint: &str,
        token: Option<String>,
        budget: Duration,
    ) -> Result<Self, DownloadError> {
        let bad = |why: &str| DownloadError::InvalidEndpoint(format!("endpoint: {why}"));
        let endpoint = endpoint.trim_end_matches('/').to_owned();
        let uri: Uri = endpoint.parse().map_err(|_| bad("not a URL"))?;
        let origin = Origin::of(&uri).ok_or_else(|| bad("not an http or https URL"))?;
        let plain = origin.scheme == "http";
        if plain && !origin.is_loopback() {
            return Err(bad("plain http is refused except on this machine"));
        }
        let config = Agent::config_builder()
            .max_redirects(0)
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(60)))
            .timeout_recv_body(Some(budget))
            .user_agent(concat!("qubivex/", env!("CARGO_PKG_VERSION")))
            .build();
        Ok(Self {
            agent: config.into(),
            endpoint,
            origin,
            plain,
            token,
        })
    }

    pub(crate) fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub(crate) fn has_token(&self) -> bool {
        self.token.is_some()
    }

    /// True if `url` is on the configured endpoint's origin.
    pub(crate) fn same_origin(&self, url: &str) -> bool {
        url.parse::<Uri>()
            .ok()
            .and_then(|u| Origin::of(&u))
            .is_some_and(|o| o == self.origin)
    }

    /// The next hop, or a refusal. Only an absolute URL or an absolute path is followed.
    fn next_hop(&self, from: &Origin, location: &str) -> Result<(String, Origin), DownloadError> {
        let refuse = || DownloadError::Protocol("the server sent an unusable redirect".to_owned());
        let url = if location.starts_with('/') && !location.starts_with("//") {
            let host = if (from.scheme == "https" && from.port == 443)
                || (from.scheme == "http" && from.port == 80)
            {
                from.host.clone()
            } else {
                format!("{}:{}", from.host, from.port)
            };
            format!("{}://{host}{location}", from.scheme)
        } else {
            location.to_owned()
        };
        let uri: Uri = url.parse().map_err(|_| refuse())?;
        let origin = Origin::of(&uri).ok_or_else(refuse)?;
        let allowed = match origin.scheme.as_str() {
            "https" => true,
            "http" => self.plain && origin.is_loopback(),
            _ => false,
        };
        if allowed {
            Ok((url, origin))
        } else {
            // The URL itself may be signed, so only its origin is named.
            Err(DownloadError::Protocol(format!(
                "refusing a redirect to {}",
                origin.describe()
            )))
        }
    }

    /// GET `url`, following redirects. The token goes only to the endpoint's origin, on
    /// every hop, so a redirect to a CDN or anywhere else never sees it.
    pub(crate) fn get(&self, url: &str, range_from: Option<u64>) -> Result<Reply, DownloadError> {
        let uri: Uri = url
            .parse()
            .map_err(|_| DownloadError::Protocol("a URL could not be built".to_owned()))?;
        let mut origin = Origin::of(&uri)
            .ok_or_else(|| DownloadError::Protocol("a URL could not be built".to_owned()))?;
        let mut url = url.to_owned();
        for _ in 0..=MAX_REDIRECTS {
            let mut request = self.agent.get(&url);
            if origin == self.origin
                && let Some(token) = &self.token
            {
                request = request.header("Authorization", format!("Bearer {token}"));
            }
            if let Some(from) = range_from {
                request = request.header("Range", format!("bytes={from}-"));
            }
            let response = request.call().map_err(|e| describe(&e))?;
            let status = response.status();
            if !status.is_redirection() || status == StatusCode::NOT_MODIFIED {
                let (parts, body) = response.into_parts();
                return Ok(Reply {
                    status,
                    headers: parts.headers,
                    body,
                });
            }
            let location = response
                .headers()
                .get("location")
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| {
                    DownloadError::Protocol("a redirect had no usable location".to_owned())
                })?;
            (url, origin) = self.next_hop(&origin, location)?;
        }
        Err(DownloadError::Protocol("too many redirects".to_owned()))
    }
}

/// A client error as a retryable failure, in words that cannot hold a URL: ureq's
/// messages name URLs, and redirect URLs are signed.
fn describe(e: &ureq::Error) -> DownloadError {
    let reason = match e {
        ureq::Error::Timeout(t) => format!("timed out: {t}"),
        ureq::Error::Io(io) => format!("connection error: {}", io.kind()),
        ureq::Error::HostNotFound => "host not found".to_owned(),
        ureq::Error::ConnectionFailed => "connection failed".to_owned(),
        ureq::Error::Tls(_) | ureq::Error::Rustls(_) => {
            return DownloadError::Protocol("the TLS connection failed".to_owned());
        }
        _ => "the request failed".to_owned(),
    };
    DownloadError::transient(reason)
}

/// Seconds in a `Retry-After` header. Its date form is ignored; backoff covers it.
pub(crate) fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    headers
        .get("retry-after")?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()
        .map(Duration::from_secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client(endpoint: &str) -> Http {
        Http::new(endpoint, None, Duration::from_secs(1)).unwrap_or_else(|e| panic!("{e}"))
    }

    fn origin(url: &str) -> Origin {
        url.parse::<Uri>()
            .ok()
            .and_then(|u| Origin::of(&u))
            .unwrap_or_else(|| panic!("{url}"))
    }

    #[test]
    fn https_never_downgrades() {
        let http = client("https://hub.example");
        let from = origin("https://hub.example");
        assert!(http.next_hop(&from, "https://cdn.example/x").is_ok());
        for bad in [
            "http://cdn.example/x",
            "http://127.0.0.1/x",
            "http://localhost:8080/admin",
            "file:///etc/passwd",
            // An explicit port gets past `Origin::of`, so the scheme rule has to hold.
            "ftp://cdn.example:21/x",
        ] {
            assert!(http.next_hop(&from, bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn plain_http_stays_on_this_machine() {
        let http = client("http://127.0.0.1:9");
        let from = origin("http://127.0.0.1:9");
        assert!(http.next_hop(&from, "http://localhost:9/x").is_ok());
        assert!(http.next_hop(&from, "http://example.com/x").is_err());
    }

    #[test]
    fn relative_redirects_stay_on_their_host() {
        let http = client("https://hub.example");
        let from = origin("https://cdn.example:8443");
        let (url, to) = http
            .next_hop(&from, "/a/b")
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(url, "https://cdn.example:8443/a/b");
        assert_eq!(to, from);
        // `//host/path` is another host, not a path.
        let (url, to) = http
            .next_hop(&from, "//evil.example/x")
            .unwrap_or_else(|_| (String::new(), from.clone()));
        assert!(url.is_empty(), "{url}");
        assert_eq!(to, from);
    }

    #[test]
    fn origins_compare_scheme_host_and_port() {
        assert_eq!(origin("https://A.example"), origin("https://a.example:443"));
        assert_ne!(origin("https://a.example"), origin("http://a.example"));
        assert_ne!(
            origin("https://a.example"),
            origin("https://a.example:8443")
        );
        assert_ne!(
            origin("https://a.example"),
            origin("https://a.example.evil")
        );
    }
}
