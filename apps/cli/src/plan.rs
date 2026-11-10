//! `qubivex plan`: how a model would be placed in memory on this machine, and why.

use std::fmt::Write;
use std::path::PathBuf;

use qubivex_core::Bytes;
use qubivex_hardware::{Host, ResourceProbe, SystemProbe};
use qubivex_memory::{
    HostMemory, LayerSlots, Location, MemoryPlan, ModelFootprint, PlanRequest, Profile, Quantity,
    Source, VramRequest, plan,
};
use qubivex_model::{ArchitectureRegistry, find};
use serde::Serialize;

use crate::chain;
use crate::models::roots;

pub(crate) struct PlanArgs<'a> {
    pub(crate) model: &'a str,
    pub(crate) dirs: &'a [PathBuf],
    pub(crate) profile: &'a str,
    pub(crate) ram_fraction: Option<f64>,
    pub(crate) ram_budget: Option<&'a str>,
    pub(crate) context: Option<u64>,
    pub(crate) sequences: u32,
    /// A discrete GPU's memory to plan for (`--vram`).
    pub(crate) vram: Option<&'a str>,
    /// An explicit VRAM budget (`--vram-budget`).
    pub(crate) vram_budget: Option<&'a str>,
    pub(crate) json: bool,
}

impl PlanArgs<'_> {
    pub(crate) fn request(&self) -> Result<PlanRequest, String> {
        Ok(PlanRequest {
            profile: profile(self.profile, self.ram_fraction)?,
            ram_budget: self.ram_budget.map(parse_bytes).transpose()?,
            context: self.context,
            sequences: self.sequences,
            layer_weights: None,
            vram: match (self.vram, self.vram_budget) {
                (None, None) => None,
                (None, Some(_)) => return Err("--vram-budget needs --vram".to_owned()),
                (Some(v), b) => Some(VramRequest {
                    device: 0,
                    available: parse_bytes(v)?,
                    budget: b.map(parse_bytes).transpose()?,
                }),
            },
        })
    }
}

/// This machine's memory as a plan sees it, before anything is loaded.
pub(crate) fn host_memory() -> Result<(HostMemory, Host), String> {
    let host = Host::live();
    let memory = qubivex_hardware::probe_memory(&host);
    let now = SystemProbe::new(host.clone())
        .snapshot()
        .map_err(|e| chain(&e))?;
    Ok((HostMemory::before_load(&memory, &now), host))
}

#[derive(Serialize)]
struct Report<'a> {
    model: &'a str,
    footprint: &'a ModelFootprint,
    plan: &'a MemoryPlan,
}

/// Parses `8GiB`, `8G`, `512MiB`, `512M`, `4096KiB` or plain bytes.
pub(crate) fn parse_bytes(s: &str) -> Result<Bytes, String> {
    let t = s.trim();
    let split = t
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(t.len());
    let (num, unit) = t.split_at(split);
    let n: f64 = num
        .parse()
        .map_err(|_| format!("{s:?} is not a size (for example 8GiB or 512MiB)"))?;
    let mult: u64 = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "kib" | "kb" => 1 << 10,
        "m" | "mib" | "mb" => 1 << 20,
        "g" | "gib" | "gb" => 1 << 30,
        "t" | "tib" | "tb" => 1 << 40,
        u => return Err(format!("unknown size unit {u:?} in {s:?}")),
    };
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let b = (n * mult as f64).round() as u64;
    if b == 0 {
        return Err(format!("{s:?} is zero"));
    }
    Ok(Bytes(b))
}

pub(crate) fn profile(name: &str, fraction: Option<f64>) -> Result<Profile, String> {
    if let Some(f) = fraction {
        return Ok(Profile::Custom(f));
    }
    match name {
        "performance" => Ok(Profile::Performance),
        "balanced" => Ok(Profile::Balanced),
        "saver" => Ok(Profile::Saver),
        "custom" => Err("the custom profile needs --ram-fraction".into()),
        p => Err(format!(
            "unknown profile {p:?} (performance, balanced, saver, or --ram-fraction)"
        )),
    }
}

pub(crate) fn run(a: &PlanArgs<'_>) -> Result<String, String> {
    let meta = find(
        a.model,
        &roots(a.dirs),
        &ArchitectureRegistry::with_builtins(),
    )
    .map_err(|e| chain(&e))?;
    let fp = ModelFootprint::from_metadata(&meta).map_err(|e| chain(&e))?;
    let (host, _) = host_memory()?;
    let p = plan(&fp, &host, &a.request()?).map_err(|e| chain(&e))?;
    let id = meta.manifest.model.id.as_str();
    if a.json {
        return crate::to_json(&Report {
            model: id,
            footprint: &fp,
            plan: &p,
        });
    }
    Ok(render(id, &fp, &p))
}

fn tag(q: &Quantity) -> String {
    let s = match q.source {
        Source::Measured => "measured",
        Source::Estimated => "estimated",
        Source::Requested => "requested",
    };
    format!("{:>11}  ({s})", q.bytes.to_string())
}

/// `16 × 64 of 64` or `8 × 12, 8 × 11 of 64`: runs of equal slot counts.
fn slots(s: &[LayerSlots]) -> String {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for l in s {
        match runs.last_mut() {
            Some((n, v)) if *v == l.slots => *n += 1,
            _ => runs.push((1, l.slots)),
        }
    }
    let of = s.iter().map(|l| l.experts).max().unwrap_or(0);
    let body: Vec<String> = runs
        .iter()
        .map(|(n, v)| format!("{n} layer{} × {v}", if *n == 1 { "" } else { "s" }))
        .collect();
    format!("{} (of {of} experts per layer)", body.join(", "))
}

fn render(id: &str, fp: &ModelFootprint, p: &MemoryPlan) -> String {
    let mut s = String::new();
    let r = &p.ram;
    let _ = writeln!(
        s,
        "Memory plan for {id}: {} profile, context {}{}",
        p.profile.name(),
        p.context,
        if p.sequences > 1 {
            format!(" × {} sequences", p.sequences)
        } else {
            String::new()
        }
    );
    let _ = writeln!(s, "\nRAM");
    for (name, q) in [
        ("budget", &r.budget),
        ("dense weights", &r.dense),
        ("activations", &r.activations),
        ("shared room", &r.room),
        ("  KV at full context", &r.kv_full_context),
        ("  expert cache, empty KV", &r.expert_cache_start),
        ("  expert cache, full KV", &r.expert_cache_full_context),
        ("  cache floor", &r.cache_floor),
        ("all routed experts", &r.experts_total),
    ] {
        let _ = writeln!(s, "  {name:<26}{}", tag(q));
    }
    let _ = writeln!(s, "  {:<26}{}", "KV per token", tag(&r.kv_per_token));
    if p.vram.is_empty() {
        let _ = writeln!(s, "VRAM\n  none");
    }
    for v in &p.vram {
        let _ = writeln!(s, "VRAM, device {}", v.device);
        for (name, q) in [
            ("budget", &v.budget),
            ("dense weights", &v.dense),
            ("activations", &v.activations),
            ("expert cache", &v.expert_cache),
        ] {
            let _ = writeln!(s, "  {name:<26}{}", tag(q));
        }
    }
    let _ = writeln!(s, "\nPlacement");
    for pl in &p.placements {
        let loc = match pl.location {
            Location::Ram => "RAM",
            Location::RamCache => "RAM cache",
            Location::Disk => "disk",
            Location::Vram(_) => "VRAM",
            Location::VramCache(_) => "VRAM cache",
        };
        let _ = writeln!(
            s,
            "  {:<16}{:>11}  {loc:<10} {}",
            pl.what,
            pl.bytes.bytes.to_string(),
            pl.reason
        );
    }
    if !fp.layers.is_empty() {
        let _ = writeln!(s, "\nExpert slots per layer");
        let _ = writeln!(s, "  empty KV  {}", slots(&p.expert_cache.at_start));
        let _ = writeln!(s, "  full KV   {}", slots(&p.expert_cache.at_full_context));
    }
    let _ = writeln!(s, "\nWhy");
    for reason in &p.reasons {
        let _ = writeln!(s, "  - {reason}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_parse() {
        assert_eq!(parse_bytes("8GiB").unwrap(), Bytes(8 << 30));
        assert_eq!(parse_bytes("8g").unwrap(), Bytes(8 << 30));
        assert_eq!(parse_bytes("1.5G").unwrap(), Bytes(3 << 29));
        assert_eq!(parse_bytes("512 MiB").unwrap(), Bytes(512 << 20));
        assert_eq!(parse_bytes("4096").unwrap(), Bytes(4096));
        assert!(parse_bytes("8 parsecs").is_err());
        assert!(parse_bytes("0G").is_err());
        assert!(parse_bytes("GiB").is_err());
    }

    #[test]
    fn profiles_parse() {
        assert_eq!(profile("saver", None).unwrap(), Profile::Saver);
        assert_eq!(
            profile("performance", Some(0.6)).unwrap(),
            Profile::Custom(0.6)
        );
        assert!(profile("custom", None).is_err());
        assert!(profile("turbo", None).is_err());
    }
}
