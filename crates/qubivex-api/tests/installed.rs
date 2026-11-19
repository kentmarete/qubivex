//! `GET /v1/admin/models/installed`: the models on disk, joined with the loaded ones,
//! with no filesystem path in the answer.

#![allow(clippy::unwrap_used)]

mod common;

use std::fs;

use common::{engine, http, root, start};
use qubivex_api::{Gateway, ServerConfig};
use serde_json::{Value, json};

const ADMIN: (&str, &str) = ("authorization", "Bearer admin-secret");

fn by_id<'a>(data: &'a [Value], id: &str) -> &'a Value {
    data.iter()
        .find(|m| m["id"] == json!(id))
        .unwrap_or_else(|| panic!("{id} not listed in {data:?}"))
}

#[test]
fn lists_models_on_disk_and_which_are_loaded() {
    let gw = Gateway::new(ServerConfig {
        admin_token: Some("admin-secret".into()),
        model_dirs: vec![root()],
        ..ServerConfig::default()
    });
    gw.add_model("dense-tiny", engine("dense-tiny")).unwrap();
    let s = start(&gw);

    let r = http(s.addr, "GET", "/v1/admin/models/installed", &[ADMIN], None);
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    let body = String::from_utf8_lossy(&r.body).into_owned();
    assert!(!body.contains(&root().display().to_string()), "{body}");
    let v = r.json();
    assert_eq!(v["object"], json!("list"));
    let data = v["data"].as_array().unwrap();
    for id in ["dense-tiny", "olmoe-tiny", "qwen2-tiny"] {
        let m = by_id(data, id);
        assert_eq!(m["status"], json!("ok"), "{m}");
        assert_eq!(m["install_state"], json!("installed"));
        assert_eq!(m["dir"], json!(id));
    }
    let dense = by_id(data, "dense-tiny");
    assert_eq!(dense["loaded"], json!(true));
    assert_eq!(dense["serve_ids"], json!(["dense-tiny"]));
    assert_eq!(dense["format"], json!("safetensors"));
    assert!(dense["bytes"]["total"].as_u64().unwrap() > 0);
    let olmoe = by_id(data, "olmoe-tiny");
    assert_eq!(olmoe["loaded"], json!(false));
    assert_eq!(olmoe["serve_ids"], json!([]));
}

#[test]
fn a_broken_model_is_listed_with_its_error_and_no_path() {
    let tmp = std::env::temp_dir().join(format!("qvx-installed-{}", std::process::id()));
    let broken = tmp.join("broken");
    fs::create_dir_all(&broken).unwrap();
    let manifest = fs::read_to_string(root().join("dense-tiny/manifest.json"))
        .unwrap()
        .replace("\"dense-tiny\"", "\"broken\"");
    fs::write(broken.join("manifest.json"), manifest).unwrap();

    let gw = Gateway::new(ServerConfig {
        admin_token: Some("admin-secret".into()),
        model_dirs: vec![tmp.clone()],
        ..ServerConfig::default()
    });
    let s = start(&gw);
    let r = http(s.addr, "GET", "/v1/admin/models/installed", &[ADMIN], None);
    fs::remove_dir_all(&tmp).ok();
    assert_eq!(r.status, 200);
    let body = String::from_utf8_lossy(&r.body).into_owned();
    assert!(!body.contains(&tmp.display().to_string()), "{body}");
    let v = r.json();
    let m = by_id(v["data"].as_array().unwrap(), "broken");
    assert_eq!(m["status"], json!("error"));
    assert!(m["error"].as_str().is_some_and(|e| !e.is_empty()), "{m}");
    assert_eq!(m["loaded"], json!(false));
}

#[test]
fn needs_the_admin_token() {
    let gw = Gateway::new(ServerConfig {
        admin_token: Some("admin-secret".into()),
        model_dirs: vec![root()],
        ..ServerConfig::default()
    });
    let s = start(&gw);
    assert_eq!(
        http(s.addr, "GET", "/v1/admin/models/installed", &[], None).status,
        401
    );
    let sys = http(s.addr, "GET", "/v1/system", &[ADMIN], None).json();
    assert_eq!(sys["capabilities"]["model_management"], json!(true));
    let off = Gateway::new(ServerConfig {
        model_dirs: vec![root()],
        ..ServerConfig::default()
    });
    let s = start(&off);
    let r = http(s.addr, "GET", "/v1/admin/models/installed", &[], None);
    assert_eq!(r.status, 403, "{}", String::from_utf8_lossy(&r.body));
    let sys = http(s.addr, "GET", "/v1/system", &[], None).json();
    assert_eq!(sys["capabilities"]["model_management"], json!(false));
}
