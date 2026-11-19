//! `GET /v1/admin/models/installed`: the models on disk under the server's model
//! directories, the ones `POST /v1/models/load` can name by id, joined with what the
//! server has loaded. A model that does not validate is listed with its error, so an
//! operator sees why it will not load.
//!
//! Like every response, this one holds no filesystem path: `dir` is the model's
//! directory name inside its models directory, and errors have paths cut out.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use kerniq_server::axum::Json;
use kerniq_server::axum::extract::State;
use kerniq_server::axum::http::HeaderMap;
use kerniq_server::axum::response::{IntoResponse, Response};
use qubivex_model::{ArchitectureRegistry, ModelMetadata, discover};
use serde_json::{Value, json};

use crate::models::resolved;
use crate::{ApiError, AppState};

pub(crate) async fn list(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let began = Instant::now();
    let rid = st.request_id(&headers);
    let r = async {
        st.check_admin(&headers)?;
        let roots = st.cfg.model_dirs.clone();
        let found = tokio::task::spawn_blocking(move || {
            let reg = ArchitectureRegistry::with_builtins();
            discover(&roots, &reg)
                .into_iter()
                .map(|d| {
                    let id = d.id();
                    let card = match &d.model {
                        Ok(m) => summary(m),
                        Err(e) => json!({
                            "status": "error",
                            "error": scrub(&format!("{e}"), &roots, &d.dir),
                        }),
                    };
                    (resolved(&d.dir), dir_name(&d.dir), id, card)
                })
                .collect::<Vec<_>>()
        })
        .await
        .map_err(|_| ApiError::internal(&rid))?;
        let served = st.served();
        let data: Vec<Value> = found
            .into_iter()
            .map(|(dir, name, id, mut card)| {
                let mine: Vec<_> = served
                    .iter()
                    .filter(|(sid, sdir, _)| match sdir {
                        Some(sd) => *sd == dir,
                        // A failed load never found its directory; match it by id.
                        None => id.as_deref() == Some(sid.as_str()),
                    })
                    .collect();
                let loaded = loaded_state(mine.iter().map(|(_, _, s)| *s));
                let o = card.as_object_mut().expect("a card is an object");
                o.insert("id".into(), json!(id));
                o.insert("dir".into(), json!(name));
                o.insert("install_state".into(), json!("installed"));
                o.insert("loaded".into(), loaded);
                o.insert(
                    "serve_ids".into(),
                    json!(
                        mine.iter()
                            .filter(|(_, _, s)| *s == "loaded")
                            .map(|(sid, _, _)| sid)
                            .collect::<Vec<_>>()
                    ),
                );
                card
            })
            .collect();
        Ok(Json(json!({ "object": "list", "data": data })).into_response())
    }
    .await;
    st.finish("admin_models_installed", &rid, began, r)
}

/// `true` when any instance is loaded, else the furthest state of the others
/// (`"loading"`, then `"failed"`), else `false`.
fn loaded_state<'a>(states: impl Iterator<Item = &'a str>) -> Value {
    let states: Vec<&str> = states.collect();
    for s in ["loaded", "loading", "failed"] {
        if states.contains(&s) {
            return if s == "loaded" { json!(true) } else { json!(s) };
        }
    }
    json!(false)
}

fn summary(m: &ModelMetadata) -> Value {
    let d = &m.config.dims;
    json!({
        "status": "ok",
        "error": null,
        "name": m.manifest.model.name,
        "architecture": format!("{} {}", m.manifest.architecture.id, m.manifest.architecture.version),
        "format": m.manifest.weights.format,
        "quantization": m.manifest.quantization,
        "source": m.manifest.source,
        "layers": d.layers,
        "hidden": d.hidden,
        "vocab": d.vocab,
        "experts": d.experts,
        "experts_per_token": d.experts_per_token,
        "max_context": m.manifest.context_length.or(d.max_context),
        "bytes": m.bytes(),
    })
}

fn dir_name(dir: &Path) -> String {
    dir.file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
}

/// `message` with the model's directory shown by name and each models directory cut
/// out, so no path reaches the client.
fn scrub(message: &str, roots: &[PathBuf], dir: &Path) -> String {
    let mut s = message.replace(&dir.display().to_string(), &dir_name(dir));
    for r in roots {
        let r = r.display().to_string();
        if !r.is_empty() {
            s = s.replace(&format!("{r}/"), "").replace(&r, "<models>");
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loaded_prefers_loaded_then_loading() {
        assert_eq!(loaded_state([].into_iter()), json!(false));
        assert_eq!(loaded_state(["failed"].into_iter()), json!("failed"));
        assert_eq!(
            loaded_state(["failed", "loading"].into_iter()),
            json!("loading")
        );
        assert_eq!(loaded_state(["loading", "loaded"].into_iter()), json!(true));
    }

    #[test]
    fn scrub_removes_paths() {
        let roots = [PathBuf::from("/srv/models")];
        let dir = Path::new("/srv/models/model-a");
        let s = scrub(
            "config.json in /srv/models/model-a is bad; also /srv/models/other/x and /srv/models",
            &roots,
            dir,
        );
        assert_eq!(
            s,
            "config.json in model-a is bad; also other/x and <models>"
        );
    }
}
