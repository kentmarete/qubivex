//! The admin download routes (ADR-013 amendment 8) over a fake [`ModelSource`]: search,
//! pull jobs, cancel and retry. Nothing here reaches a real model hub.

#![allow(clippy::unwrap_used)]

mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::{Resp, http, start};
use qubivex_api::{
    Gateway, LogFormat, ModelSource, PullOutcome, PullProgress, PullSpec, ServerConfig,
    SourceError, SourceErrorKind, SourceHit,
};
use serde_json::{Value, json};

const ADMIN: (&str, &str) = ("authorization", "Bearer admin-secret");
const FILE: u64 = 1000;

/// A source whose pulls send `steps` progress events `step` apart, and fail for a
/// repository whose name holds "broken".
#[derive(Debug)]
struct Fake {
    steps: u64,
    step: Duration,
    pulls: AtomicUsize,
    /// What the last pull of each repository got through, for resume checks.
    seen: Mutex<Vec<PullSpec>>,
}

impl Fake {
    fn new(steps: u64, step: Duration) -> Arc<Self> {
        Arc::new(Self {
            steps,
            step,
            pulls: AtomicUsize::new(0),
            seen: Mutex::new(Vec::new()),
        })
    }
}

impl ModelSource for Fake {
    fn search(&self, query: &str, limit: usize) -> Result<Vec<SourceHit>, SourceError> {
        if query == "down" {
            return Err(SourceError::new(
                SourceErrorKind::Upstream,
                "the hub did not answer",
            ));
        }
        Ok((0..limit.min(3))
            .map(|i| SourceHit {
                repository: format!("owner/{query}-{i}"),
                downloads: 100 - i as u64,
                likes: 1,
                formats: vec!["gguf".into()],
                gated: i == 2,
            })
            .collect())
    }

    fn pull(
        &self,
        spec: &PullSpec,
        cancel: &AtomicBool,
        progress: &(dyn Fn(PullProgress) + Sync),
    ) -> Result<PullOutcome, SourceError> {
        self.pulls.fetch_add(1, Ordering::Relaxed);
        self.seen.lock().unwrap().push(spec.clone());
        if spec.repository.contains("broken") {
            return Err(SourceError::new(
                SourceErrorKind::NotFound,
                format!("{} was not found", spec.repository),
            ));
        }
        progress(PullProgress::Planned {
            revision: "a".repeat(40),
            files: vec![("model.gguf".into(), FILE)],
        });
        for i in 1..=self.steps {
            if cancel.load(Ordering::Relaxed) {
                return Err(SourceError::new(SourceErrorKind::Cancelled, "cancelled"));
            }
            std::thread::sleep(self.step);
            progress(PullProgress::File {
                file: "model.gguf".into(),
                done: FILE * i / self.steps,
                total: FILE,
            });
        }
        progress(PullProgress::Validating);
        let id = spec.id.clone().unwrap_or_else(|| {
            spec.repository
                .rsplit('/')
                .next()
                .unwrap_or_default()
                .to_lowercase()
        });
        Ok(PullOutcome {
            id,
            warnings: Vec::new(),
            files: vec!["model.gguf".into()],
        })
    }

    fn installed(&self) -> Vec<(String, String)> {
        vec![("owner/qwen-0".into(), "qwen-0".into())]
    }
}

fn gateway(source: Option<Arc<Fake>>, admin: bool) -> Gateway {
    Gateway::new(ServerConfig {
        admin_token: admin.then(|| "admin-secret".into()),
        model_source: source.map(|s| s as Arc<dyn ModelSource>),
        log_format: LogFormat::Off,
        ..ServerConfig::default()
    })
}

fn pull(addr: std::net::SocketAddr, body: &Value) -> Resp {
    http(addr, "POST", "/v1/admin/models/pull", &[ADMIN], Some(body))
}

/// Polls job `id` until it finishes.
fn wait(addr: std::net::SocketAddr, id: &str) -> Value {
    let until = Instant::now() + Duration::from_secs(30);
    loop {
        let j = http(addr, "GET", &format!("/v1/admin/jobs/{id}"), &[ADMIN], None).json();
        let s = j["status"].as_str().unwrap().to_owned();
        if matches!(s.as_str(), "succeeded" | "failed" | "cancelled") {
            return j;
        }
        assert!(Instant::now() < until, "job {id} still {s}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn without_a_source_downloads_are_off_and_said_so() {
    let srv = start(&gateway(None, true));
    let sys = http(srv.addr, "GET", "/v1/system", &[], None).json();
    assert_eq!(sys["capabilities"]["model_downloads"], false);
    assert_eq!(sys["capabilities"]["admin"], true);
    let r = http(
        srv.addr,
        "GET",
        "/v1/admin/models/search?q=x",
        &[ADMIN],
        None,
    );
    assert_eq!(r.status, 404);
    assert_eq!(r.json()["error"]["code"], "downloads_disabled");
    let r = pull(srv.addr, &json!({ "repository": "a/b" }));
    assert_eq!(r.status, 404);
}

#[test]
fn the_routes_need_the_admin_token() {
    let srv = start(&gateway(Some(Fake::new(1, Duration::ZERO)), false));
    let r = http(srv.addr, "GET", "/v1/admin/models/search?q=x", &[], None);
    assert_eq!(r.status, 403);
    assert_eq!(r.json()["error"]["code"], "admin_disabled");

    let srv = start(&gateway(Some(Fake::new(1, Duration::ZERO)), true));
    let sys = http(srv.addr, "GET", "/v1/system", &[], None).json();
    assert_eq!(sys["capabilities"]["model_downloads"], true);
    let wrong = [("authorization", "Bearer nope")];
    for (method, path) in [
        ("GET", "/v1/admin/models/search?q=x"),
        ("POST", "/v1/admin/models/pull"),
        ("GET", "/v1/admin/jobs"),
        ("GET", "/v1/admin/jobs/job-1"),
        ("DELETE", "/v1/admin/jobs/job-1"),
        ("POST", "/v1/admin/jobs/job-1/retry"),
    ] {
        let r = http(srv.addr, method, path, &wrong, Some(&json!({})));
        assert_eq!(r.status, 401, "{method} {path}");
    }
}

#[test]
fn search_lists_hits_and_what_is_installed() {
    let srv = start(&gateway(Some(Fake::new(1, Duration::ZERO)), true));
    let r = http(
        srv.addr,
        "GET",
        "/v1/admin/models/search?q=qwen&limit=2",
        &[ADMIN],
        None,
    );
    assert_eq!(r.status, 200);
    let b = r.json();
    assert_eq!(b["query"], "qwen");
    let data = b["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["repository"], "owner/qwen-0");
    assert_eq!(data[0]["installed_as"], json!(["qwen-0"]));
    assert_eq!(data[1]["installed_as"], json!([]));

    for bad in [
        "/v1/admin/models/search",
        "/v1/admin/models/search?q=%20",
        "/v1/admin/models/search?q=x&limit=0",
        "/v1/admin/models/search?q=x&limit=101",
    ] {
        let r = http(srv.addr, "GET", bad, &[ADMIN], None);
        assert_eq!(r.status, 400, "{bad}");
    }
    let r = http(
        srv.addr,
        "GET",
        "/v1/admin/models/search?q=down",
        &[ADMIN],
        None,
    );
    assert_eq!(r.status, 502);
    assert_eq!(r.json()["error"]["code"], "upstream_error");
}

#[test]
fn a_pull_becomes_a_job_that_reports_progress_and_succeeds() {
    let srv = start(&gateway(
        Some(Fake::new(4, Duration::from_millis(30))),
        true,
    ));
    let r = pull(
        srv.addr,
        &json!({ "repository": "Owner/Model-GGUF", "id": "my-model" }),
    );
    assert_eq!(r.status, 202);
    let job = r.json();
    let id = job["id"].as_str().unwrap().to_owned();
    assert_eq!(
        r.header("location"),
        Some(format!("/v1/admin/jobs/{id}").as_str())
    );
    assert_eq!(job["kind"], "pull");
    assert_eq!(job["status"], "running");
    assert_eq!(job["model"], "my-model");
    assert_eq!(job["cancellable"], true);

    let done = wait(srv.addr, &id);
    assert_eq!(done["status"], "succeeded", "{done}");
    assert_eq!(done["result"]["id"], "my-model");
    assert_eq!(done["revision"], "a".repeat(40));
    assert_eq!(done["progress"]["bytes_done"], FILE);
    assert_eq!(done["progress"]["bytes_total"], FILE);
    assert_eq!(done["progress"]["files"][0]["name"], "model.gguf");
    assert_eq!(done["cancellable"], false);
    assert!(done["finished_at"].is_u64());

    let list = http(srv.addr, "GET", "/v1/admin/jobs", &[ADMIN], None).json();
    assert_eq!(list["data"][0]["id"], id.as_str());
}

#[test]
fn a_bad_pull_request_is_refused_before_a_job_starts() {
    let fake = Fake::new(1, Duration::ZERO);
    let srv = start(&gateway(Some(Arc::clone(&fake)), true));
    for body in [
        json!({}),
        json!({ "repository": "no-owner" }),
        json!({ "repository": "a/b/c" }),
        json!({ "repository": "a/b", "files": "x" }),
        json!({ "repository": "a/b", "set": [1] }),
        json!({ "repository": 5 }),
    ] {
        let r = pull(srv.addr, &body);
        assert_eq!(r.status, 400, "{body}");
    }
    assert_eq!(fake.pulls.load(Ordering::Relaxed), 0);
    let r = http(srv.addr, "GET", "/v1/admin/jobs/job-nope", &[ADMIN], None);
    assert_eq!(r.status, 404);
    assert_eq!(r.json()["error"]["code"], "job_not_found");
}

#[test]
fn a_failed_pull_says_why() {
    let srv = start(&gateway(Some(Fake::new(1, Duration::ZERO)), true));
    let id = pull(srv.addr, &json!({ "repository": "owner/broken" })).json()["id"]
        .as_str()
        .unwrap()
        .to_owned();
    let done = wait(srv.addr, &id);
    assert_eq!(done["status"], "failed");
    assert_eq!(done["error"]["code"], "not_found");
    assert_eq!(done["error"]["message"], "owner/broken was not found");
}

#[test]
fn pulls_are_limited_and_one_repository_pulls_once() {
    let srv = start(&gateway(
        Some(Fake::new(50, Duration::from_millis(20))),
        true,
    ));
    let a = pull(srv.addr, &json!({ "repository": "owner/a" }));
    assert_eq!(a.status, 202);
    let again = pull(srv.addr, &json!({ "repository": "OWNER/A" }));
    assert_eq!(again.status, 409);
    let same_id = pull(srv.addr, &json!({ "repository": "owner/x", "id": "a" }));
    assert_eq!(
        same_id.status, 202,
        "a pull without an id does not claim one"
    );
    let third = pull(srv.addr, &json!({ "repository": "owner/c" }));
    assert_eq!(third.status, 429);
    for j in [a.json(), same_id.json()] {
        let id = j["id"].as_str().unwrap();
        let r = http(
            srv.addr,
            "DELETE",
            &format!("/v1/admin/jobs/{id}"),
            &[ADMIN],
            None,
        );
        assert_eq!(r.status, 202);
        assert_eq!(wait(srv.addr, id)["status"], "cancelled");
    }
}

#[test]
fn a_cancelled_pull_can_be_retried() {
    let fake = Fake::new(100, Duration::from_millis(10));
    let srv = start(&gateway(Some(Arc::clone(&fake)), true));
    let id = pull(
        srv.addr,
        &json!({ "repository": "owner/m", "set": { "k": 1 } }),
    )
    .json()["id"]
        .as_str()
        .unwrap()
        .to_owned();
    std::thread::sleep(Duration::from_millis(50));
    let r = http(
        srv.addr,
        "DELETE",
        &format!("/v1/admin/jobs/{id}"),
        &[ADMIN],
        None,
    );
    assert_eq!(r.status, 202);
    let done = wait(srv.addr, &id);
    assert_eq!(done["status"], "cancelled");
    assert_eq!(done["error"]["code"], "cancelled");

    // Finished jobs cannot be cancelled; running ones cannot be retried.
    let r = http(
        srv.addr,
        "DELETE",
        &format!("/v1/admin/jobs/{id}"),
        &[ADMIN],
        None,
    );
    assert_eq!(r.status, 409);

    let r = http(
        srv.addr,
        "POST",
        &format!("/v1/admin/jobs/{id}/retry"),
        &[ADMIN],
        None,
    );
    assert_eq!(r.status, 202);
    let new = r.json()["id"].as_str().unwrap().to_owned();
    assert_ne!(new, id);
    let r = http(
        srv.addr,
        "POST",
        &format!("/v1/admin/jobs/{new}/retry"),
        &[ADMIN],
        None,
    );
    assert_eq!(r.status, 409);
    assert_eq!(wait(srv.addr, &new)["status"], "succeeded");
    let seen = fake.seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(
        seen[0], seen[1],
        "a retry pulls exactly what the first job asked for"
    );
}
