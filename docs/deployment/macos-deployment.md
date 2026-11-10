# macOS deployment

How to run `qubivex serve` on a Mac, by hand or as a launchd service. Apple Silicon is
the measured platform: every benchmark in this repository is from an M3 Pro. Intel Macs
build and run on the CPU, and are not tested in CI.

## Build and install

```sh
xcode-select --install          # the Command Line Tools, for the C compiler
cargo build --release --locked
sudo install -m 0755 target/release/qubivex /usr/local/bin/qubivex
qubivex system
```

`qubivex system` should show the CPU's performance and efficiency cores, the RAM, and a
`metal` line with the GPU. Full Xcode is not needed: the Metal kernels are compiled from
source by the system when a model loads.

## Models

```sh
qubivex models pull Qwen/Qwen3-0.6B
qubivex models analyse qwen3-0.6b
```

Models go in `~/.qubivex/models` unless `--dir` or `QUBIVEX_MODELS_DIRS` says otherwise.
Keep them on the internal SSD: experts that do not fit in RAM are read from disk during
generation, and an external or network volume makes every such read slower.
`qubivex doctor --storage --path ~/.qubivex/models` measures the disk.

## CPU or Metal

```sh
qubivex serve --model olmoe-1b-7b-0125-instruct                    # CPU (the default)
qubivex serve --model olmoe-1b-7b-0125-instruct --backend metal    # Apple GPU
```

Both give the same tokens, bit for bit. Which is faster depends on the model and on
memory:

| Case | Use | Why |
|---|---|---|
| A MoE model whose experts all fit in RAM | `--backend metal` | OLMoE-1B-7B Q4_K_M, one conversation: 60 tok/s on Metal against 41 on the CPU. Eight at once: 102 against 62 |
| A MoE model with a small expert cache | Either | When any layer's pool holds fewer than half its experts, the engine runs the CPU pass even under `--backend metal`, because the GPU pass stops paying while experts stream (ADR-011 amendment 6) |
| A dense model (`transformer_dense`) | CPU | On Metal its matmuls are placed one at a time: 33 tok/s against 69 on the CPU for Qwen3-0.6B |

`--metal-ops token` (the default with `--backend metal`) runs whole layers on the GPU with
routing on the GPU; `--metal-ops layers` routes on the host. They are close from two
concurrent conversations upwards. On Apple Silicon the GPU shares the system's RAM, so
Metal does not add memory: weights, KV and experts count in the one RAM budget.

## Memory on a Mac

- The planner budgets 0.88 of the RAM available when the model loads (`--profile
  performance`). On a machine you also work on, `--profile balanced` (0.75) or
  `--ram-budget 8GiB` leaves room for other applications.
- macOS answers a new allocation partly by compressing other memory, so available RAM
  falls by less than what another application allocates, and the pressure monitor sees
  less than it would on Linux (ADR-005 amendment 3). An explicit `--ram-budget` is the
  dependable control.
- Quit applications that hold several gigabytes (Docker Desktop, IDEs, browsers) before
  loading a model that needs most of the RAM. `qubivex models analyse` reports against
  the memory free at that moment.

## Run it as a service (launchd)

A per-user agent starts at login and restarts the server if it exits. Create the config
first:

```sh
mkdir -p ~/.qubivex ~/Library/Logs/qubivex
cat > ~/.qubivex/serve.toml <<'EOF'
[serve]
listen = "127.0.0.1:8080"
models = ["qwen3-0.6b"]
log_format = "json"
api_key = "replace-with-a-long-random-string"
EOF
chmod 600 ~/.qubivex/serve.toml
```

A file holding `api_key` or `admin_token` must be `chmod 600`, or the server refuses to
start. On loopback the key is optional; set it anyway if other users share the Mac.

`~/Library/LaunchAgents/com.qubivex.serve.plist` (replace `YOU` with your user name;
launchd does not expand `~`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.qubivex.serve</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/qubivex</string>
        <string>serve</string>
        <string>--config</string>
        <string>/Users/YOU/.qubivex/serve.toml</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>ExitTimeOut</key>
    <integer>45</integer>
    <key>ProcessType</key>
    <string>Interactive</string>
    <key>StandardErrorPath</key>
    <string>/Users/YOU/Library/Logs/qubivex/serve.log</string>
    <key>StandardOutPath</key>
    <string>/Users/YOU/Library/Logs/qubivex/serve.log</string>
</dict>
</plist>
```

```sh
plutil -lint ~/Library/LaunchAgents/com.qubivex.serve.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.qubivex.serve.plist
launchctl print gui/$(id -u)/com.qubivex.serve | head -20
curl -s http://127.0.0.1:8080/health
tail -f ~/Library/Logs/qubivex/serve.log
```

To stop and remove it:

```sh
launchctl bootout gui/$(id -u)/com.qubivex.serve
```

What the keys do:

- `KeepAlive` with `SuccessfulExit` false restarts the server after a crash, and not
  after a clean stop.
- `ExitTimeOut` is how long launchd waits after SIGTERM before it kills the process. The
  server drains for `--shutdown-grace` (30 s), so 45 leaves room.
- `ProcessType` `Interactive` gives the server an application's resource limits, which
  is none. Without it launchd applies light CPU and I/O limits to the job.
- launchd does not rotate the log file. Rotate it with `newsyslog`, or log less with
  `log_format = "off"` and rely on `/metrics`.

The plist above passes `plutil -lint`. Running the agent across a reboot has not been
tested for this document.

## Serving other machines

The server speaks plain HTTP. To serve beyond the Mac, keep `listen` on loopback and put
a TLS proxy (Caddy or nginx) in front, as in
[Linux deployment](linux-deployment.md#tls-and-the-public-network). Listening on
`0.0.0.0` directly needs an API key and sends it unencrypted, so use it only on a network
you trust.

A Mac that sleeps stops serving. `caffeinate -s qubivex serve ...` keeps it awake on
power while the server runs, or set it in System Settings.

## Docker on a Mac

Docker Desktop runs containers in a Linux virtual machine, so a container uses the CPU
backend and the VM's memory limit, and never Metal. It is the way to test the Linux image
on a Mac, not the way to serve from one.

## Not supported

- **Metal for dense models at full speed.** It works and is slower than the CPU.
- **Intel Mac GPUs.** `--backend metal` has only been run on Apple Silicon, where the
  GPU shares the system's memory. On a discrete GPU it is untested.
- **A system-wide daemon** (`/Library/LaunchDaemons`). Only the per-user agent above is
  described. A daemon has no login session, and Metal there is untested.
- **A signed or notarised binary, and Homebrew.** Build from source.
- **Peak GPU memory in `/metrics`.** Metal shares host memory, so there is no separate
  figure to report.
