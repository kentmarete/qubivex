//! The models a gateway serves and how each is doing. One engine per model directory:
//! concurrent loads of a model share one load, and a generation that names a model
//! still loading waits for it. Requests hold an `Arc` of what they run on, so unloading
//! closes the queue at once and the weights go when the last running request ends.

use std::path::{Path, PathBuf};
use std::sync::{Arc, PoisonError, RwLockReadGuard, RwLockWriteGuard, Weak};
use std::time::Instant;

use kerniq_server::axum::http::StatusCode;
use qubivex_model::{ArchitectureRegistry, find};
use qubivex_runtime::scheduler::{Limits, Scheduler};
use qubivex_runtime::{Engine, KvAdmission};
use serde_json::{Value, json};
use tokio::sync::watch;

use crate::{ApiError, AppState, now_secs};

/// A load's result, once there is one.
type LoadResult = Option<Result<(), ApiError>>;

/// A loaded model.
pub(crate) struct Loaded {
    pub(crate) engine: Arc<Engine>,
    pub(crate) scheduler: Scheduler<KvAdmission>,
    pub(crate) created: u64,
    /// What the model was loaded as: a model id or a path.
    source: String,
    /// The model's directory, resolved: an instance's identity.
    dir: PathBuf,
}

/// A model's state in the serving process.
pub(crate) enum Entry {
    Loading {
        source: String,
        /// Set as soon as the load finds the model.
        dir: Option<PathBuf>,
        done: watch::Receiver<LoadResult>,
    },
    Ready(Arc<Loaded>),
    /// The last load failed; another attempt replaces this.
    Failed,
}

fn resolved(dir: &Path) -> PathBuf {
    dir.canonicalize().unwrap_or_else(|_| dir.to_owned())
}

impl AppState {
    pub(crate) fn read_models(
        &self,
    ) -> RwLockReadGuard<'_, std::collections::BTreeMap<String, Entry>> {
        self.models.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write_models(&self) -> RwLockWriteGuard<'_, std::collections::BTreeMap<String, Entry>> {
        self.models.write().unwrap_or_else(PoisonError::into_inner)
    }

    fn limits(&self) -> Limits {
        Limits {
            max_running: self.cfg.max_concurrent,
            max_queue: self.cfg.max_queue,
            aging: self.cfg.priority_aging,
            memory: self.cfg.memory_policy,
        }
    }

    fn entry_ready(&self, engine: Engine, source: String) -> Entry {
        let dir = resolved(&engine.metadata().dir);
        let engine = Arc::new(engine);
        Entry::Ready(Arc::new(Loaded {
            scheduler: Scheduler::new(KvAdmission(Arc::clone(&engine)), self.limits()),
            engine,
            created: now_secs(),
            source,
            dir,
        }))
    }

    /// Serves `engine` under `id`.
    pub(crate) fn insert(&self, id: &str, engine: Engine) -> Result<(), ApiError> {
        let entry = self.entry_ready(engine, id.to_owned());
        let mut m = self.write_models();
        if m.contains_key(id) {
            return Err(ApiError::conflict(format!(
                "the model {id:?} is already loaded"
            )));
        }
        m.insert(id.to_owned(), entry);
        drop(m);
        self.log.info("model_added", &[("model", id.into())]);
        Ok(())
    }

    /// The models ready to serve.
    pub(crate) fn ready(&self) -> Vec<(String, Arc<Loaded>)> {
        self.read_models()
            .iter()
            .filter_map(|(id, e)| match e {
                Entry::Ready(l) => Some((id.clone(), Arc::clone(l))),
                _ => None,
            })
            .collect()
    }

    /// Every known model and its state.
    pub(crate) fn states(&self) -> Vec<Value> {
        self.read_models()
            .iter()
            .map(|(id, e)| {
                let state = match e {
                    Entry::Loading { .. } => "loading",
                    Entry::Ready(_) => "loaded",
                    Entry::Failed => "failed",
                };
                json!({ "id": id, "state": state })
            })
            .collect()
    }

    /// The model `id`, if it is ready.
    pub(crate) fn model(&self, id: &str) -> Result<Arc<Loaded>, ApiError> {
        match self.read_models().get(id) {
            Some(Entry::Ready(l)) => Ok(Arc::clone(l)),
            _ => Err(ApiError::model_not_found(id)),
        }
    }

    /// The model `id`, waiting until `until` if it is loading.
    pub(crate) async fn model_ready(
        &self,
        id: &str,
        until: Instant,
    ) -> Result<Arc<Loaded>, ApiError> {
        let mut done = match self.read_models().get(id) {
            Some(Entry::Loading { done, .. }) => done.clone(),
            Some(Entry::Ready(l)) => return Ok(Arc::clone(l)),
            _ => return Err(ApiError::model_not_found(id)),
        };
        let wait = done.wait_for(Option::is_some);
        tokio::time::timeout_at(until.into(), wait)
            .await
            .map_err(|_| ApiError::overloaded(format!("the model {id:?} is still loading")))?
            .map_err(|_| ApiError::model_not_found(id))?;
        self.model(id)
    }

    /// Loads `what` as `id`, or joins the load already running. Returns the model and
    /// whether this call started the load. The load belongs to the server, not to the
    /// request: a client that goes away does not end it for the others.
    pub(crate) async fn load(
        self: &Arc<Self>,
        rid: &str,
        what: &str,
        id: &str,
    ) -> Result<(Arc<Loaded>, bool), ApiError> {
        let busy = |id: &str| {
            ApiError::conflict(format!("the model {id:?} is already loaded from elsewhere"))
        };
        let (mut done, started) = {
            let mut m = self.write_models();
            match m.get(id) {
                Some(Entry::Ready(l)) if l.source == what || id == what => {
                    return Ok((Arc::clone(l), false));
                }
                Some(Entry::Ready(_)) => return Err(busy(id)),
                Some(Entry::Loading { source, done, .. }) if source == what => {
                    (done.clone(), false)
                }
                Some(Entry::Loading { .. }) => return Err(busy(id)),
                Some(Entry::Failed) | None => {
                    let (tx, rx) = watch::channel(None);
                    m.insert(
                        id.to_owned(),
                        Entry::Loading {
                            source: what.to_owned(),
                            dir: None,
                            done: rx.clone(),
                        },
                    );
                    self.metrics.count_load();
                    let job = LoadJob {
                        st: Arc::clone(self),
                        rid: rid.to_owned(),
                        what: what.to_owned(),
                        id: id.to_owned(),
                    };
                    tokio::spawn(job.run(tx));
                    (rx, true)
                }
            }
        };
        let result = done
            .wait_for(Option::is_some)
            .await
            .map_err(|_| ApiError::internal(rid))?
            .clone();
        match result {
            Some(Ok(())) => self.model(id).map(|l| (l, started)),
            Some(Err(e)) => Err(e),
            None => Err(ApiError::internal(rid)),
        }
    }

    /// Claims `dir` for `id`, so no two ids load one model.
    fn claim_dir(&self, id: &str, dir: &Path) -> Result<(), ApiError> {
        let mut m = self.write_models();
        let holder = m.iter().find(|(other, e)| {
            other.as_str() != id
                && match e {
                    Entry::Loading { dir: Some(d), .. } => d == dir,
                    Entry::Ready(l) => l.dir == dir,
                    _ => false,
                }
        });
        if let Some((other, _)) = holder {
            return Err(ApiError::conflict(format!(
                "this model is already loaded as {other:?}"
            )));
        }
        if let Some(Entry::Loading { dir: d, .. }) = m.get_mut(id) {
            *d = Some(dir.to_owned());
        }
        Ok(())
    }

    /// Stops serving `id`. Returns how many requests are still running on it. Waiting
    /// requests end now.
    pub(crate) fn unload(&self, id: &str) -> Result<usize, ApiError> {
        let mut m = self.write_models();
        let entry = match m.get(id) {
            None => return Err(ApiError::model_not_found(id)),
            Some(Entry::Loading { .. }) => {
                return Err(ApiError::conflict(format!("the model {id:?} is loading")));
            }
            Some(_) => m.remove(id),
        };
        drop(m);
        let Some(Entry::Ready(l)) = entry else {
            return Ok(0);
        };
        l.scheduler.close();
        if let Ok(mut u) = self.unloaded.lock() {
            u.retain(|w| w.strong_count() > 0);
            u.push(Arc::downgrade(&l.engine));
        }
        Ok(l.scheduler.stats().running)
    }

    /// Unloaded models whose weights are still held by running requests.
    pub(crate) fn draining(&self) -> usize {
        self.unloaded.lock().map_or(0, |u| {
            u.iter()
                .filter(|w: &&Weak<Engine>| w.strong_count() > 0)
                .count()
        })
    }
}

/// One model load, owned by the server.
struct LoadJob {
    st: Arc<AppState>,
    rid: String,
    what: String,
    id: String,
}

impl LoadJob {
    /// Finds and loads the model, off the Tokio workers: loading reads weights for
    /// seconds.
    fn load(&self) -> Result<Engine, ApiError> {
        let st = &self.st;
        let (what, rid) = (&self.what, &self.rid);
        let reg = ArchitectureRegistry::with_builtins();
        let meta = find(what, &st.cfg.model_dirs, &reg).map_err(|e| {
            st.log.warn(
                "model_not_found",
                &[
                    ("request_id", rid.as_str().into()),
                    ("detail", e.to_string().into()),
                ],
            );
            ApiError::model_not_found(what)
        })?;
        st.claim_dir(&self.id, &resolved(&meta.dir))?;
        Engine::load(meta, st.cfg.engine).map_err(|e| ApiError::from_runtime(&e, rid, &st.log))
    }

    async fn run(self, done: watch::Sender<LoadResult>) {
        let job = Arc::new(self);
        let j = Arc::clone(&job);
        let st = &job.st;
        let ids = [
            ("request_id", job.rid.as_str().into()),
            ("model", job.id.as_str().into()),
        ];
        st.log.info("model_load_started", &ids);
        let began = Instant::now();
        let loaded = tokio::task::spawn_blocking(move || j.load())
            .await
            .unwrap_or_else(|_| Err(ApiError::internal(&job.rid)));
        let took = began.elapsed();
        let mut fields = ids.to_vec();
        fields.push(("duration_ms", crate::log::millis(took)));
        match &loaded {
            Ok(_) => {
                st.metrics.model(&job.id).load.observe_duration(took);
                st.log.info("model_loaded", &fields);
            }
            Err(e) => {
                fields.push(("error", e.code.into()));
                st.log.warn("model_load_failed", &fields);
            }
        }
        let (entry, result) = match loaded {
            Ok(engine) => (Some(st.entry_ready(engine, job.what.clone())), Ok(())),
            // Nothing was found to load, or another id serves it: no state to keep.
            Err(e) if e.status == StatusCode::NOT_FOUND || e.status == StatusCode::CONFLICT => {
                (None, Err(e))
            }
            Err(e) => (Some(Entry::Failed), Err(e)),
        };
        let mut m = st.write_models();
        match entry {
            Some(e) => m.insert(job.id.clone(), e),
            None => m.remove(&job.id),
        };
        drop(m);
        done.send_replace(Some(result));
    }
}

#[cfg(test)]
mod tests {
    use tokio::sync::watch;

    use super::Entry;
    use crate::{Gateway, ServerConfig};

    /// Unloading must not remove a load in progress: it would finish and serve the model
    /// anyway, with the caller told it was gone.
    #[test]
    fn a_model_still_loading_cannot_be_unloaded() {
        let gw = Gateway::new(ServerConfig::default());
        let (_done, rx) = watch::channel(None);
        gw.state.models.write().unwrap().insert(
            "m".to_owned(),
            Entry::Loading {
                source: "m".to_owned(),
                dir: None,
                done: rx,
            },
        );
        assert_eq!(gw.state.unload("m").unwrap_err().code, "conflict");
        assert!(gw.state.models.read().unwrap().contains_key("m"));
    }
}
