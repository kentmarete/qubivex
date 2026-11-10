//! Crate dependency-direction rules (QUBIVEX_DESIGN §3.1).
//!
//! Every workspace crate must appear in [`allowed_deps`]. A crate that is not listed
//! fails the check, so adding a crate forces a deliberate decision about its edges.

use serde::Deserialize;

/// The subset of `cargo metadata --no-deps` output this check reads.
#[derive(Debug, Deserialize)]
pub(crate) struct Metadata {
    pub(crate) packages: Vec<Package>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Package {
    pub(crate) name: String,
    pub(crate) dependencies: Vec<Dependency>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Dependency {
    pub(crate) name: String,
    /// `null` for normal dependencies, `"dev"` or `"build"` otherwise.
    pub(crate) kind: Option<String>,
}

const BACKENDS: &[&str] = &[
    "qubivex-backend-cpu",
    "qubivex-backend-cuda",
    "qubivex-backend-vulkan",
    "qubivex-backend-metal",
];

const APPS: &[&str] = &["qubivex-cli", "qubivex-server"];

/// Internal crates `krate` may depend on (normal and build dependencies), or `None`
/// if the crate is unknown to the rules.
fn allowed_deps(krate: &str) -> Option<Vec<&'static str>> {
    let mut v: Vec<&'static str> = match krate {
        "qubivex-core" => vec![],
        "qubivex-hardware" | "qubivex-tokenizer" => vec!["qubivex-core"],
        // The model crate never depends on compute.
        "qubivex-model" => vec!["qubivex-core"],
        // Installs and removes models; never executes them.
        "qubivex-model-registry" => vec!["qubivex-model"],
        // Fetches models and installs them through the registry; never executes them.
        "qubivex-model-download" => vec!["qubivex-model-registry", "qubivex-model"],
        // Decides what runs; knows nothing of models, HTTP or Tokio.
        "qubivex-scheduler" => vec![],
        // Decides which retained KV a prompt can reuse; knows nothing of tensors.
        "qubivex-kv-cache" => vec![],
        "qubivex-compute" => vec!["qubivex-core"],
        b if BACKENDS.contains(&b) => vec!["qubivex-compute", "qubivex-core"],
        "qubivex-arch" => vec!["qubivex-model", "qubivex-compute", "qubivex-core"],
        "qubivex-expert-stream" => vec!["qubivex-compute", "qubivex-core"],
        "qubivex-memory" => vec![
            "qubivex-model",
            "qubivex-hardware",
            "qubivex-compute",
            "qubivex-core",
        ],
        "qubivex-runtime" => {
            let mut v = vec![
                "qubivex-arch",
                "qubivex-memory",
                "qubivex-expert-stream",
                "qubivex-kv-cache",
                "qubivex-model",
                "qubivex-scheduler",
                "qubivex-tokenizer",
                "qubivex-hardware",
                "qubivex-compute",
                "qubivex-core",
            ];
            v.extend_from_slice(BACKENDS);
            v
        }
        "qubivex-api" => vec![
            "qubivex-runtime",
            "qubivex-memory",
            "qubivex-model",
            "qubivex-tokenizer",
            "qubivex-hardware",
            "qubivex-core",
        ],
        "qubivex-test-utils" => vec![
            "qubivex-model",
            "qubivex-compute",
            "qubivex-core",
            "qubivex-backend-cpu",
        ],
        a if APPS.contains(&a) => return Some(vec![]), // handled in `is_allowed`
        // Repository tooling: reads model containers to generate fixtures. Nothing
        // depends on xtask, so this edge cannot leak into product crates.
        "xtask" => vec!["qubivex-core", "qubivex-model"],
        _ => return None,
    };
    v.sort_unstable();
    Some(v)
}

fn is_allowed(from: &str, to: &str, dev: bool, allowed: &[&str]) -> bool {
    if APPS.contains(&to) {
        return false; // nothing depends on an application
    }
    if APPS.contains(&from) {
        return to.starts_with("qubivex-");
    }
    if dev && to == "qubivex-test-utils" && from != to {
        return true;
    }
    // The gateway's tests slow expert reads with the fault-injection source, so a
    // generation is still running when a test cancels it.
    if dev && from == "qubivex-api" && to == "qubivex-expert-stream" {
        return true;
    }
    // A GPU backend's parity tests use the CPU backend's kernels and the forward
    // pass's ops as their reference (ADR-007, ADR-011).
    if dev && (to == "qubivex-backend-cpu" || to == "qubivex-arch") && BACKENDS.contains(&from) {
        return true;
    }
    allowed.contains(&to)
}

/// Crates that would tie the runtime to a server, an async runtime or the CLI.
const NOT_IN_RUNTIME: &[&str] = &[
    "axum",
    "clap",
    "hyper",
    "kerniq-server",
    "qubivex-api",
    "qubivex-cli",
    "tokio",
    "tower",
];

/// The [`NOT_IN_RUNTIME`] crates in `cargo tree --prefix none` output.
pub(crate) fn server_deps(tree: &str) -> Vec<String> {
    let mut found: Vec<String> = tree
        .lines()
        .filter_map(|l| l.split_whitespace().next())
        .filter(|name| NOT_IN_RUNTIME.contains(name))
        .map(str::to_owned)
        .collect();
    found.sort_unstable();
    found.dedup();
    found
}

/// Returns one message per forbidden edge or unknown crate.
pub(crate) fn violations(meta: &Metadata) -> Vec<String> {
    let internal: Vec<&str> = meta.packages.iter().map(|p| p.name.as_str()).collect();
    let mut out = Vec::new();
    for pkg in &meta.packages {
        let Some(allowed) = allowed_deps(&pkg.name) else {
            out.push(format!(
                "{} is not in the dependency rules; add it to xtask/src/deps.rs",
                pkg.name
            ));
            continue;
        };
        for dep in &pkg.dependencies {
            if !internal.contains(&dep.name.as_str()) {
                continue; // third-party
            }
            let dev = dep.kind.as_deref() == Some("dev");
            if !is_allowed(&pkg.name, &dep.name, dev, &allowed) {
                let kind = dep.kind.as_deref().unwrap_or("normal");
                out.push(format!(
                    "{} -> {} ({kind} dependency) is forbidden",
                    pkg.name, dep.name
                ));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(dependency, kind)` pairs.
    type Deps<'a> = &'a [(&'a str, Option<&'a str>)];
    /// A crate and its dependencies.
    type Edge<'a> = (&'a str, Deps<'a>);

    fn meta(edges: &[Edge<'_>]) -> Metadata {
        Metadata {
            packages: edges
                .iter()
                .map(|(name, deps)| Package {
                    name: (*name).to_owned(),
                    dependencies: deps
                        .iter()
                        .map(|(d, k)| Dependency {
                            name: (*d).to_owned(),
                            kind: k.map(str::to_owned),
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    const ALL: &[&str] = &[
        "qubivex-core",
        "qubivex-hardware",
        "qubivex-model",
        "qubivex-compute",
        "qubivex-backend-cpu",
        "qubivex-arch",
        "qubivex-memory",
        "qubivex-expert-stream",
        "qubivex-runtime",
        "qubivex-api",
        "qubivex-test-utils",
        "qubivex-cli",
    ];

    fn with_all(extra: &[Edge<'_>]) -> Vec<String> {
        let mut edges: Vec<Edge<'_>> = ALL
            .iter()
            .filter(|n| !extra.iter().any(|(e, _)| e == *n))
            .map(|n| (*n, &[][..]))
            .collect();
        edges.extend_from_slice(extra);
        violations(&meta(&edges))
    }

    #[test]
    fn layered_graph_passes() {
        let v = with_all(&[
            ("qubivex-hardware", &[("qubivex-core", None)]),
            ("qubivex-model", &[("qubivex-core", None), ("serde", None)]),
            (
                "qubivex-runtime",
                &[("qubivex-arch", None), ("qubivex-backend-cpu", None)],
            ),
            (
                "qubivex-cli",
                &[("qubivex-runtime", None), ("qubivex-api", None)],
            ),
            ("qubivex-arch", &[("qubivex-test-utils", Some("dev"))]),
        ]);
        assert_eq!(v, Vec::<String>::new());
    }

    #[test]
    fn the_runtime_may_not_gain_a_server_dependency() {
        let clean = "qubivex-runtime v0.1.0 (/r)\nrayon v1.10.0\ntokio-util v0.7.0\n";
        assert!(server_deps(clean).is_empty());
        let dirty = "qubivex-runtime v0.1.0 (/r)\ntokio v1.40.0\naxum v0.8.0\ntokio v1.40.0 (*)\n";
        assert_eq!(server_deps(dirty), ["axum", "tokio"]);
    }

    #[test]
    fn forbidden_edges_from_the_design_fail() {
        for (from, to) in [
            ("qubivex-model", "qubivex-backend-cpu"),
            ("qubivex-model", "qubivex-compute"),
            ("qubivex-memory", "qubivex-arch"),
            ("qubivex-expert-stream", "qubivex-arch"),
            ("qubivex-backend-cpu", "qubivex-api"),
            ("qubivex-backend-cpu", "qubivex-runtime"),
            ("qubivex-core", "qubivex-hardware"),
            ("qubivex-runtime", "qubivex-cli"),
        ] {
            let deps = [(to, None)];
            let v = with_all(&[(from, &deps)]);
            assert_eq!(v.len(), 1, "{from} -> {to} should be forbidden: {v:?}");
            assert!(v[0].contains(&format!("{from} -> {to}")));
        }
    }

    #[test]
    fn test_utils_only_as_dev_dependency() {
        let v = with_all(&[("qubivex-arch", &[("qubivex-test-utils", None)])]);
        assert_eq!(v.len(), 1);
        assert!(v[0].contains("normal dependency"));
    }

    #[test]
    fn gpu_backends_use_the_cpu_backend_only_in_tests() {
        let v = with_all(&[(
            "qubivex-backend-metal",
            &[("qubivex-backend-cpu", Some("dev"))],
        )]);
        assert!(v.is_empty(), "{v:?}");
        let v = with_all(&[("qubivex-backend-metal", &[("qubivex-backend-cpu", None)])]);
        assert_eq!(v.len(), 1);
        let v = with_all(&[("qubivex-arch", &[("qubivex-backend-cpu", Some("dev"))])]);
        assert_eq!(v.len(), 1, "only backends get the exception");
    }

    #[test]
    fn apps_cannot_depend_on_apps() {
        let v = with_all(&[("qubivex-cli", &[("qubivex-server", None)])]);
        // qubivex-server is not a workspace member here, so it is treated as third-party.
        assert!(v.is_empty());
        let v = violations(&meta(&[
            ("qubivex-cli", &[("qubivex-server", None)]),
            ("qubivex-server", &[]),
        ]));
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn unknown_crate_fails() {
        let v = with_all(&[("qubivex-mystery", &[])]);
        assert_eq!(v.len(), 1);
        assert!(v[0].contains("not in the dependency rules"));
    }
}
