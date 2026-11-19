//! `qubivex serve --detach | --status | --stop`: the gateway as a background process.
//!
//! A detached server is this binary started again with the same arguments, in its own
//! process group, with its log in `~/.qubivex/serve-<port>.log` and its pid in
//! `~/.qubivex/serve-<port>.pid`. One per port. `--detach` checks the settings and the
//! port first and returns once the server accepts connections (or has failed, with the
//! end of its log). `--stop` sends SIGTERM, which drains open requests, as Ctrl-C does.

use std::ffi::OsString;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::background;
use crate::serve::{self, ServeFlags};

/// How long `--detach` waits for the server to accept connections (models load first).
const START_WAIT: Duration = Duration::from_secs(300);
/// How long `--stop` waits for open requests to finish.
const STOP_WAIT: Duration = Duration::from_secs(60);
/// Lines of the log shown by `--status` and when a start fails.
const LOG_LINES: usize = 15;

/// The arguments of `qubivex serve`.
#[derive(Clone, Debug, clap::Args)]
pub(crate) struct ServeArgs {
    #[command(flatten)]
    flags: ServeFlags,
    /// Run the server in the background and return once it accepts connections. Its
    /// log goes to ~/.qubivex/serve-<port>.log; see --status and --stop.
    #[arg(long, conflicts_with_all = ["status", "stop"])]
    detach: bool,
    /// List the background servers, with the end of the log of one (--listen) or of
    /// the only one.
    #[arg(long, conflicts_with = "stop")]
    status: bool,
    /// Stop the background server on --listen's port, or all of them. Open requests
    /// finish first.
    #[arg(long)]
    stop: bool,
    /// Set on the background process `--detach` starts.
    #[arg(long, hide = true)]
    detached_child: bool,
}

/// Runs `qubivex serve` with `args`.
pub(crate) fn serve(args: &ServeArgs) -> Result<(), String> {
    if args.status {
        return status(args.flags.listen());
    }
    if args.stop {
        return stop(args.flags.listen());
    }
    let plan = serve::plan(args.flags.clone())?;
    if args.detach {
        return detach(plan.addr());
    }
    if !args.detached_child {
        return serve::serve(plan);
    }
    let pid_file = files(plan.addr().port())?.0;
    let result = serve::serve(plan);
    if let Err(e) = &result {
        eprintln!("qubivex: server failed: {e}");
    }
    background::release(&pid_file);
    result
}

/// The pid file and log of the background server on `port`.
fn files(port: u16) -> Result<(PathBuf, PathBuf), String> {
    let data = background::data_dir()?;
    Ok((
        data.join(format!("serve-{port}.pid")),
        data.join(format!("serve-{port}.log")),
    ))
}

/// The ports with a background server pid file, ascending.
fn ports() -> Result<Vec<u16>, String> {
    let data = background::data_dir()?;
    let entries =
        std::fs::read_dir(&data).map_err(|e| format!("cannot read {}: {e}", data.display()))?;
    let mut ports: Vec<u16> = entries
        .filter_map(Result::ok)
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            name.strip_prefix("serve-")?
                .strip_suffix(".pid")?
                .parse()
                .ok()
        })
        .collect();
    ports.sort_unstable();
    Ok(ports)
}

/// Where to connect to reach a server listening on `addr`.
fn probe_addr(addr: SocketAddr) -> SocketAddr {
    match addr.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => (Ipv4Addr::LOCALHOST, addr.port()).into(),
        IpAddr::V6(ip) if ip.is_unspecified() => (Ipv6Addr::LOCALHOST, addr.port()).into(),
        _ => addr,
    }
}

fn print_log(log: &Path) -> Result<(), String> {
    if let Some(lines) = background::tail(log, LOG_LINES)? {
        println!("log: {} (last {LOG_LINES} lines)", log.display());
        for line in &lines {
            println!("  {line}");
        }
    }
    Ok(())
}

fn detach(addr: SocketAddr) -> Result<(), String> {
    let port = addr.port();
    if port == 0 {
        return Err("--detach needs a fixed port; pass --listen ADDRESS:PORT".to_owned());
    }
    let (pid_file, log) = files(port)?;
    if let Some(pid) = background::running(&pid_file) {
        return Err(format!(
            "a background server is already running on port {port} (pid {pid}); stop it \
             with `qubivex serve --stop --listen {addr}`"
        ));
    }
    // Fail here, rather than after the models load, if the port is taken.
    drop(TcpListener::bind(addr).map_err(|e| format!("cannot listen on {addr}: {e}"))?);
    let mut child_args: Vec<OsString> = std::env::args_os()
        .skip(1)
        .filter(|a| a != "--detach")
        .collect();
    child_args.push("--detached-child".into());
    let mut child = background::spawn(
        &child_args,
        &log,
        &format!("qubivex serve started on {addr}"),
        &pid_file,
    )?;
    let pid = child.id();
    let start = Instant::now();
    let probe = probe_addr(addr);
    loop {
        if let Ok(Some(exit)) = child.try_wait() {
            // A pid file left behind only makes `--status` check a dead pid.
            let _ = std::fs::remove_file(&pid_file);
            print_log(&log)?;
            return Err(format!(
                "the background server exited while starting ({exit})"
            ));
        }
        if TcpStream::connect_timeout(&probe, Duration::from_millis(200)).is_ok() {
            break;
        }
        if start.elapsed() > START_WAIT {
            println!(
                "the server (pid {pid}) is still starting after {} s; follow it with \
                 `tail -f {}`",
                START_WAIT.as_secs(),
                log.display()
            );
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    println!("serving in the background on http://{addr} (pid {pid})");
    println!("  log:    {}", log.display());
    println!("  status: qubivex serve --status");
    println!("  stop:   qubivex serve --stop --listen {addr}");
    Ok(())
}

fn status(listen: Option<SocketAddr>) -> Result<(), String> {
    let ports = match listen {
        Some(a) => vec![a.port()],
        None => ports()?,
    };
    let mut shown = Vec::new();
    for &port in &ports {
        let (pid_file, log) = files(port)?;
        match background::running(&pid_file) {
            Some(pid) => {
                println!("port {port}: running (pid {pid}), log {}", log.display());
                shown.push(log);
            }
            None if pid_file.exists() => {
                println!(
                    "port {port}: not running (it stopped), log {}",
                    log.display()
                );
                // A stale pid file only repeats this line next time.
                let _ = std::fs::remove_file(&pid_file);
                shown.push(log);
            }
            None => println!("port {port}: no background server"),
        }
    }
    if ports.is_empty() {
        println!("no background server is running");
    }
    if let [log] = shown.as_slice() {
        print_log(log)?;
    }
    Ok(())
}

fn stop(listen: Option<SocketAddr>) -> Result<(), String> {
    let ports = match listen {
        Some(a) => vec![a.port()],
        None => ports()?,
    };
    let mut stopped = 0;
    let mut stuck = Vec::new();
    for port in ports {
        let (pid_file, _) = files(port)?;
        let Some(pid) = background::running(&pid_file) else {
            let _ = std::fs::remove_file(&pid_file);
            continue;
        };
        println!("stopping the server on port {port} (pid {pid}); open requests finish first");
        background::signal(pid, "-TERM");
        let start = Instant::now();
        while background::alive(pid) && start.elapsed() < STOP_WAIT {
            std::thread::sleep(Duration::from_millis(100));
        }
        if background::alive(pid) {
            stuck.push(format!("port {port} (pid {pid})"));
        } else {
            let _ = std::fs::remove_file(&pid_file);
            stopped += 1;
        }
    }
    if !stuck.is_empty() {
        return Err(format!(
            "still running after {} s: {}; force it with kill -9 PID",
            STOP_WAIT.as_secs(),
            stuck.join(", ")
        ));
    }
    if stopped == 0 {
        println!("no background server is running");
    } else {
        println!("stopped {stopped} server(s)");
    }
    Ok(())
}
