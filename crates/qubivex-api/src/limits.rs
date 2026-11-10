//! What the gateway lets a client do before a request reaches a model: a per-client rate
//! limit, CORS for browser clients, and the check that refuses to serve beyond loopback
//! without a key (spec §11). Request size limits live with the handlers, which know
//! the body and the prompt.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use kerniq_server::axum::extract::{ConnectInfo, Request, State};
use kerniq_server::axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use kerniq_server::axum::middleware::Next;
use kerniq_server::axum::response::{IntoResponse, Response};

use crate::error::ApiError;
use crate::{AppState, ServerConfig};

/// Clients the rate limiter remembers at most. Past this, the client seen longest ago
/// is forgotten (and starts again with a full allowance).
const MAX_CLIENTS: usize = 4096;

/// A per-client allowance of requests: a token bucket that holds `burst` requests and
/// refills at `per_minute`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RateLimit {
    /// Requests a client may make per minute, on average.
    pub per_minute: u32,
    /// Requests a client may make at once after being idle.
    pub burst: u32,
}

impl RateLimit {
    fn check(self) -> Result<(), String> {
        if self.per_minute == 0 || self.burst == 0 {
            return Err(
                "the rate limit needs at least 1 request per minute and a burst of at least 1"
                    .into(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct Bucket {
    tokens: f64,
    seen: Instant,
}

/// Token buckets by client address.
#[derive(Debug)]
pub(crate) struct RateLimiter {
    limit: RateLimit,
    buckets: Mutex<HashMap<IpAddr, Bucket>>,
}

/// Who a request counts against: its peer address, with IPv6 cut to its /64 (one host
/// usually holds a whole /64) and IPv4-mapped IPv6 read as IPv4. Without a peer address
/// (the router used without `serve`), every request counts against one client.
fn client_key(peer: Option<SocketAddr>) -> IpAddr {
    match peer.map(|p| p.ip()) {
        None => IpAddr::V6(Ipv6Addr::UNSPECIFIED),
        Some(IpAddr::V4(v4)) => IpAddr::V4(v4),
        Some(IpAddr::V6(v6)) => match v6.to_ipv4_mapped() {
            Some(v4) => IpAddr::V4(v4),
            None => {
                let s = v6.segments();
                IpAddr::V6(Ipv6Addr::new(s[0], s[1], s[2], s[3], 0, 0, 0, 0))
            }
        },
    }
}

impl RateLimiter {
    pub(crate) fn new(limit: RateLimit) -> Self {
        Self {
            limit,
            buckets: Mutex::new(HashMap::new()),
        }
    }

    /// Takes one request from `client`'s allowance at `now`, or says how long until it
    /// has one.
    pub(crate) fn take(&self, client: IpAddr, now: Instant) -> Result<(), Duration> {
        let per_sec = f64::from(self.limit.per_minute) / 60.0;
        let cap = f64::from(self.limit.burst);
        // A poisoned lock means another request panicked mid-update; the counts are
        // still usable.
        let mut b = self
            .buckets
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if b.len() >= MAX_CLIENTS && !b.contains_key(&client) {
            // Full buckets carry no state worth keeping.
            b.retain(|_, k| {
                k.tokens + now.saturating_duration_since(k.seen).as_secs_f64() * per_sec < cap
            });
            if b.len() >= MAX_CLIENTS
                && let Some(oldest) = b.iter().min_by_key(|(_, k)| k.seen).map(|(ip, _)| *ip)
            {
                b.remove(&oldest);
            }
        }
        let k = b.entry(client).or_insert(Bucket {
            tokens: cap,
            seen: now,
        });
        let refilled = now.saturating_duration_since(k.seen).as_secs_f64() * per_sec;
        k.tokens = (k.tokens + refilled).min(cap);
        k.seen = now;
        if k.tokens >= 1.0 {
            k.tokens -= 1.0;
            Ok(())
        } else {
            Err(Duration::from_secs_f64((1.0 - k.tokens) / per_sec))
        }
    }

    #[cfg(test)]
    fn clients(&self) -> usize {
        self.buckets.lock().map_or(0, |b| b.len())
    }
}

/// Middleware: refuses `/v1/*` requests past the client's allowance with a 429 and a
/// `retry-after`. Health and metrics stay open, so monitoring is never limited.
pub(crate) async fn rate_limit(
    State(st): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    let Some(limiter) = &st.limiter else {
        return next.run(req).await;
    };
    if !req.uri().path().starts_with("/v1/") {
        return next.run(req).await;
    }
    let peer = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| c.0);
    match limiter.take(client_key(peer), Instant::now()) {
        Ok(()) => next.run(req).await,
        Err(wait) => {
            let rid = st.request_id(req.headers());
            let mut resp = st.finish("rate_limit", &rid, Err(ApiError::rate_limited()));
            let secs = wait.as_secs() + u64::from(wait.subsec_nanos() > 0);
            resp.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(secs.max(1)));
            resp
        }
    }
}

/// Whether `origin` may call the API from a browser.
fn allowed<'a>(origins: &[String], headers: &'a HeaderMap) -> Option<&'a HeaderValue> {
    let origin = headers.get(header::ORIGIN)?;
    let s = origin.to_str().ok()?;
    origins
        .iter()
        .any(|o| o == "*" || o.eq_ignore_ascii_case(s))
        .then_some(origin)
}

/// Middleware: CORS for the configured origins. A preflight from an allowed origin is
/// answered here with a 204; one from any other origin gets a 403 with no CORS headers.
/// Other responses to an allowed origin carry `access-control-allow-origin`. No
/// cookies are involved, so credentials are never allowed: browsers send the bearer
/// token as an ordinary header.
pub(crate) async fn cors(State(st): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    let origins = &st.cfg.cors_origins;
    if origins.is_empty() {
        return next.run(req).await;
    }
    let any = origins.iter().any(|o| o == "*");
    let origin = allowed(origins, req.headers()).cloned();
    let preflight = req.method() == Method::OPTIONS
        && req
            .headers()
            .contains_key(header::ACCESS_CONTROL_REQUEST_METHOD);
    let mut resp = if preflight {
        if origin.is_none() {
            return StatusCode::FORBIDDEN.into_response();
        }
        let mut r = StatusCode::NO_CONTENT.into_response();
        let h = r.headers_mut();
        h.insert(
            header::ACCESS_CONTROL_ALLOW_METHODS,
            HeaderValue::from_static("GET, POST, DELETE"),
        );
        h.insert(
            header::ACCESS_CONTROL_ALLOW_HEADERS,
            HeaderValue::from_static("authorization, content-type, x-request-id"),
        );
        h.insert(
            header::ACCESS_CONTROL_MAX_AGE,
            HeaderValue::from_static("600"),
        );
        r
    } else {
        next.run(req).await
    };
    let h = resp.headers_mut();
    if !any {
        h.append(header::VARY, HeaderValue::from_static("origin"));
    }
    if let Some(origin) = origin {
        let value = if any {
            HeaderValue::from_static("*")
        } else {
            origin
        };
        h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, value);
        h.insert(
            header::ACCESS_CONTROL_EXPOSE_HEADERS,
            HeaderValue::from_static("x-request-id, retry-after"),
        );
    }
    resp
}

/// `scheme://host[:port]`, as a browser sends it in `Origin`.
fn is_origin(s: &str) -> bool {
    let Some((scheme, rest)) = s.split_once("://") else {
        return false;
    };
    ["http", "https"].contains(&scheme)
        && !rest.is_empty()
        && !rest.contains(['/', '?', '#', '@', ' '])
}

impl ServerConfig {
    /// Checks the settings that can be wrong on their own: CORS origins, the rate limit
    /// and the size limits.
    ///
    /// # Errors
    /// A message naming the setting and what it needs.
    pub fn validate(&self) -> Result<(), String> {
        for o in &self.cors_origins {
            if o != "*" && !is_origin(o) {
                return Err(format!(
                    "CORS origin {o:?} is not an origin; give scheme://host[:port], such as https://app.example.com, or *"
                ));
            }
        }
        if let Some(r) = self.rate_limit {
            r.check()?;
        }
        if self.max_body_bytes == 0 {
            return Err("the request body limit must be positive".into());
        }
        if self.max_prompt_tokens == Some(0) || self.max_generation_tokens == Some(0) {
            return Err("prompt and generation token limits must be positive".into());
        }
        if self
            .max_generation_tokens
            .is_some_and(|m| self.default_max_tokens > m)
        {
            return Err(format!(
                "the default max_tokens ({}) is above the generation limit ({})",
                self.default_max_tokens,
                self.max_generation_tokens.unwrap_or_default()
            ));
        }
        Ok(())
    }

    /// Refuses to listen on `addr` beyond loopback without an API key, unless
    /// [`ServerConfig::allow_unauthenticated_remote`] says to (spec §11: no
    /// unauthenticated inference server on the network by default).
    ///
    /// # Errors
    /// A message saying what to set.
    pub fn check_listen(&self, addr: SocketAddr) -> Result<(), String> {
        if addr.ip().is_loopback() || self.api_key.is_some() || self.allow_unauthenticated_remote {
            return Ok(());
        }
        Err(format!(
            "refusing to serve on {addr} without an API key: anyone who can reach it could use it. Set an API key, listen on 127.0.0.1, or allow unauthenticated remote access explicitly"
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
    use std::time::{Duration, Instant};

    use super::{MAX_CLIENTS, RateLimit, RateLimiter, client_key, is_origin};
    use crate::ServerConfig;

    fn ip(n: u32) -> IpAddr {
        IpAddr::V4(Ipv4Addr::from(n))
    }

    #[test]
    fn a_client_gets_its_burst_then_the_refill_rate() {
        let rl = RateLimiter::new(RateLimit {
            per_minute: 60,
            burst: 3,
        });
        let t = Instant::now();
        for _ in 0..3 {
            rl.take(ip(1), t).unwrap();
        }
        let wait = rl.take(ip(1), t).unwrap_err();
        assert_eq!(wait, Duration::from_secs(1));
        // Another client is not affected.
        rl.take(ip(2), t).unwrap();
        // One second refills one request, and no more than the burst.
        rl.take(ip(1), t + Duration::from_secs(1)).unwrap();
        assert!(rl.take(ip(1), t + Duration::from_secs(1)).is_err());
        let later = t + Duration::from_secs(3600);
        for _ in 0..3 {
            rl.take(ip(1), later).unwrap();
        }
        assert!(rl.take(ip(1), later).is_err());
    }

    #[test]
    fn a_refused_request_does_not_use_the_allowance() {
        let rl = RateLimiter::new(RateLimit {
            per_minute: 60,
            burst: 1,
        });
        let t = Instant::now();
        rl.take(ip(1), t).unwrap();
        for ms in [100, 200, 300, 400] {
            assert!(rl.take(ip(1), t + Duration::from_millis(ms)).is_err());
        }
        rl.take(ip(1), t + Duration::from_secs(1)).unwrap();
    }

    #[test]
    fn the_client_table_stays_bounded() {
        let rl = RateLimiter::new(RateLimit {
            per_minute: 1,
            burst: 1,
        });
        let t = Instant::now();
        let n = u32::try_from(MAX_CLIENTS).unwrap() + 100;
        for i in 0..n {
            rl.take(ip(i), t + Duration::from_millis(u64::from(i)))
                .unwrap();
        }
        assert!(rl.clients() <= MAX_CLIENTS);
        // The newest client is still limited.
        assert!(rl.take(ip(n - 1), t + Duration::from_secs(5)).is_err());
    }

    #[test]
    fn ipv6_clients_count_by_slash_64() {
        let a: SocketAddr = "[2001:db8:1:2:aaaa::1]:5000".parse().unwrap();
        let b: SocketAddr = "[2001:db8:1:2:bbbb::9]:6000".parse().unwrap();
        let c: SocketAddr = "[2001:db8:1:3::1]:6000".parse().unwrap();
        assert_eq!(client_key(Some(a)), client_key(Some(b)));
        assert_ne!(client_key(Some(a)), client_key(Some(c)));
        let mapped: SocketAddr = "[::ffff:10.0.0.1]:1".parse().unwrap();
        assert_eq!(
            client_key(Some(mapped)),
            IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1))
        );
        assert_eq!(client_key(None), IpAddr::V6(Ipv6Addr::UNSPECIFIED));
    }

    #[test]
    fn origins_are_checked() {
        for good in [
            "http://localhost:3000",
            "https://app.example.com",
            "http://[::1]:8080",
        ] {
            assert!(is_origin(good), "{good}");
        }
        for bad in [
            "app.example.com",
            "https://app.example.com/",
            "ftp://x",
            "https://",
            "https://a.com/path",
        ] {
            assert!(!is_origin(bad), "{bad}");
        }
        let cfg = ServerConfig {
            cors_origins: vec!["https://ok.example".into(), "nope".into()],
            ..ServerConfig::default()
        };
        assert!(cfg.validate().unwrap_err().contains("\"nope\""));
    }

    #[test]
    fn listening_beyond_loopback_needs_a_key_or_an_explicit_allowance() {
        let open: SocketAddr = "0.0.0.0:8080".parse().unwrap();
        let lan: SocketAddr = "192.168.1.5:8080".parse().unwrap();
        let local: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let local6: SocketAddr = "[::1]:8080".parse().unwrap();
        let cfg = ServerConfig::default();
        assert!(cfg.check_listen(local).is_ok());
        assert!(cfg.check_listen(local6).is_ok());
        assert!(cfg.check_listen(open).is_err());
        assert!(cfg.check_listen(lan).is_err());
        let keyed = ServerConfig {
            api_key: Some("k".into()),
            ..ServerConfig::default()
        };
        assert!(keyed.check_listen(open).is_ok());
        let allowed = ServerConfig {
            allow_unauthenticated_remote: true,
            ..ServerConfig::default()
        };
        assert!(allowed.check_listen(open).is_ok());
    }

    #[test]
    fn limits_are_validated() {
        let ok = ServerConfig::default();
        assert!(ok.validate().is_ok());
        let zero_rate = ServerConfig {
            rate_limit: Some(RateLimit {
                per_minute: 0,
                burst: 1,
            }),
            ..ServerConfig::default()
        };
        assert!(zero_rate.validate().is_err());
        let default_above_cap = ServerConfig {
            default_max_tokens: 512,
            max_generation_tokens: Some(256),
            ..ServerConfig::default()
        };
        assert!(default_above_cap.validate().is_err());
    }
}
