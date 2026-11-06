//! `qubivex models analyse`: what a model needs, and whether this machine has it.

use std::fmt::Write;

use qubivex_core::Bytes;
use qubivex_hardware::{BackendStatus, GpuInfo, Probed};
use qubivex_memory::{Assessment, ModelFootprint, Verdict, assess};
use qubivex_model::{ArchitectureRegistry, ModelMetadata, find};
use qubivex_runtime::cpu_pass_reason;
use serde::Serialize;
use serde_json::{Value, json};

use crate::chain;
use crate::models::{roots, source_line};
use crate::plan::{PlanArgs, host_memory};

#[derive(Serialize)]
struct Backend {
    name: String,
    /// The device, if the backend has one.
    device: Option<String>,
    /// Whether this model could use it, or why not.
    state: String,
}

#[derive(Serialize)]
struct Report<'a> {
    model: &'a str,
    source: Value,
    disk: Bytes,
    ram_total: Option<Bytes>,
    ram_available: Option<Bytes>,
    backends: Vec<Backend>,
    assessment: &'a Assessment,
    footprint: &'a ModelFootprint,
}

fn known(p: &Probed<Bytes>) -> Option<Bytes> {
    match p {
        Probed::Known(b) => Some(*b),
        Probed::Unknown(_) => None,
    }
}

fn shown(b: Option<Bytes>) -> String {
    b.map_or_else(|| "unknown".to_owned(), |b| b.to_string())
}

/// One entry per GPU backend found, after the CPU, which always runs.
fn backends(
    gpus: &[GpuInfo],
    m: &ModelMetadata,
    a: &Assessment,
    fp: &ModelFootprint,
) -> Vec<Backend> {
    let mut out = vec![Backend {
        name: "cpu".to_owned(),
        device: None,
        state: "available (the default)".to_owned(),
    }];
    // The engine runs whole layers on Metal only with routed experts and enough of them
    // resident, and otherwise the CPU pass. It says why, so this asks it.
    let fallback = a.plan.as_ref().and_then(|p| {
        cpu_pass_reason(
            &fp.layers,
            p.ram.expert_cache_start.bytes.0,
            m.config.dims.experts_per_token as usize,
        )
    });
    let on_gpu = || {
        if a.plan.is_none() {
            return "available, with --backend metal".to_owned();
        }
        if fp.layers.is_empty() {
            return "matmuls, with --backend metal".to_owned();
        }
        fallback
            .clone()
            .unwrap_or_else(|| "whole layers, with --backend metal".to_owned())
    };
    for g in gpus {
        let name = g.backend.as_str().to_owned();
        match &g.status {
            BackendStatus::Unavailable { reason } => out.push(Backend {
                name,
                device: None,
                state: format!("unavailable: {reason}"),
            }),
            BackendStatus::Available { devices } if devices.is_empty() => out.push(Backend {
                name,
                device: None,
                state: "no devices".to_owned(),
            }),
            BackendStatus::Available { devices } => out.extend(devices.iter().map(|d| Backend {
                name: name.clone(),
                device: Some(d.name.clone()),
                state: if name == "metal" {
                    on_gpu()
                } else {
                    "no backend built for it yet".to_owned()
                },
            })),
        }
    }
    out
}

pub(crate) fn run(a: &PlanArgs<'_>) -> Result<String, String> {
    let meta = find(
        a.model,
        &roots(a.dirs),
        &ArchitectureRegistry::with_builtins(),
    )
    .map_err(|e| chain(&e))?;
    let fp = ModelFootprint::from_metadata(&meta).map_err(|e| chain(&e))?;
    let (memory, host) = host_memory()?;
    let assessment = assess(&fp, &memory, &a.request()?).map_err(|e| chain(&e))?;
    let report = Report {
        model: &meta.manifest.model.id,
        source: json!(meta.manifest.source),
        disk: Bytes(meta.tensors.files().iter().map(|f| f.size).sum()),
        ram_total: known(&memory.total),
        ram_available: known(&memory.available),
        backends: backends(
            &qubivex_hardware::probe(&host).gpus,
            &meta,
            &assessment,
            &fp,
        ),
        assessment: &assessment,
        footprint: &fp,
    };
    if a.json {
        return crate::to_json(&report);
    }
    Ok(render(&meta, &report))
}

fn verdict_line(a: &Assessment) -> String {
    // The figures assume the context asked for, else the model's, so a plan with less
    // means the planner cut it.
    let wanted = a.requirements.kv_context;
    let shortened = a
        .plan
        .as_ref()
        .filter(|p| p.context < wanted)
        .map(|p| {
            format!(
                ", with the context shortened from {wanted} to {}",
                p.context
            )
        })
        .unwrap_or_default();
    match a.verdict {
        Verdict::Fits => format!("runs here, everything in RAM{shortened}"),
        Verdict::Streams => {
            format!("runs here, reading some experts from disk{shortened}")
        }
        Verdict::DoesNotFit => "does not fit".to_owned(),
    }
}

fn render(m: &ModelMetadata, r: &Report<'_>) -> String {
    let a = r.assessment;
    let need = &a.requirements;
    let mut s = String::new();
    let _ = writeln!(s, "Model    {} ({})", r.model, m.manifest.model.name);
    let _ = writeln!(s, "Source   {}", source_line(m.manifest.source.as_ref()));
    let _ = writeln!(s, "Verdict  {}", verdict_line(a));
    if let Some(why) = &a.refusal {
        let why = why.strip_prefix("does not fit: ").unwrap_or(why);
        let _ = writeln!(s, "         {why}");
    }
    let _ = writeln!(s, "\nNeeds");
    let _ = writeln!(s, "  {:<30}{:>11}", "on disk", r.disk.to_string());
    let _ = writeln!(
        s,
        "  {:<30}{:>11}",
        "RAM to start",
        need.minimum.to_string()
    );
    if need.at_context != need.minimum {
        let at = format!("RAM at context {}", need.kv_context);
        let _ = writeln!(s, "  {at:<30}{:>11}", need.at_context.to_string());
    }
    if need.in_memory != need.at_context {
        let _ = writeln!(
            s,
            "  {:<30}{:>11}",
            "RAM with every expert cached",
            need.in_memory.to_string()
        );
    }
    let _ = writeln!(s, "\nThis machine");
    let _ = writeln!(s, "  {:<30}{:>11}", "RAM total", shown(r.ram_total));
    let _ = writeln!(
        s,
        "  {:<30}{:>11}",
        "RAM available now",
        shown(r.ram_available)
    );
    if let Some(p) = &a.plan {
        let at = format!("budget ({} profile)", p.profile.name());
        let _ = writeln!(s, "  {at:<30}{:>11}", p.ram.budget.bytes.to_string());
    }
    let _ = writeln!(s, "\nBackends");
    for b in &r.backends {
        let device = b
            .device
            .as_deref()
            .map(|d| format!(" ({d})"))
            .unwrap_or_default();
        let _ = writeln!(s, "  {}{device}: {}", b.name, b.state);
    }
    if a.plan.is_some() {
        let _ = writeln!(s, "\n`qubivex plan {}` shows the placement.", r.model);
    }
    s
}
