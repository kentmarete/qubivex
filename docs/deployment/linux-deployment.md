# Linux deployment

How to run `qubivex serve` as a managed service on Linux x86-64 or arm64, either under
systemd or in a container. Day-to-day operation is in
[Production operations](production-operations.md).

On Linux QUBIVEX runs on the CPU. There is no CUDA or Vulkan backend yet (see
[Not supported](#not-supported)).

## Before you start

| Need | Why |
|---|---|
| A 64-bit CPU. On x86-64, AVX2 and FMA are used when present | The binary picks its kernels at run time, so one build runs on any CPU of its architecture |
| RAM for the model's dense weights plus one expert per layer, at least | `qubivex models analyse <ID>` gives the exact figures for a model on a machine |
| The models on local SSD or NVMe | Experts that are not in RAM are read from disk during generation. `qubivex doctor --storage --path <DIR>` measures the disk |
| A reverse proxy for TLS | The server speaks plain HTTP. It has no TLS of its own |

## Option 1: systemd

### Install the binary

Build on the target architecture, or on a machine of the same one:

```sh
cargo build --release --locked
sudo install -m 0755 target/release/qubivex /usr/local/bin/qubivex
```

The binary links only glibc and `libgcc_s`, which every glibc distribution has. A build
needs a C compiler (for `ring`); the installed binary needs nothing else.

### Create the user and directories

```sh
sudo useradd --system --home-dir /var/lib/qubivex --create-home --shell /usr/sbin/nologin qubivex
sudo install -d -o qubivex -g qubivex /var/lib/qubivex/models
sudo install -d -m 0750 -o root -g qubivex /etc/qubivex
```

### Install a model

```sh
sudo -u qubivex env QUBIVEX_MODELS_DIRS=/var/lib/qubivex/models \
  qubivex models pull Qwen/Qwen3-0.6B
sudo -u qubivex env QUBIVEX_MODELS_DIRS=/var/lib/qubivex/models \
  qubivex models analyse qwen3-0.6b
```

`analyse` says whether the model runs entirely in RAM, runs while reading some experts
from disk, or does not fit. A gated or private model needs `HF_TOKEN` in the environment
of the pull. To copy a model in by hand instead, see
[Model lifecycle](../model-management/model-lifecycle.md).

### Configuration

`/etc/qubivex/serve.toml`:

```toml
[serve]
listen = "127.0.0.1:8080"
models = ["qwen3-0.6b"]
dirs = ["/var/lib/qubivex/models"]
log_format = "json"
max_concurrent = 4
max_queue = 32
rate_limit = 120
```

Every `qubivex serve` flag is a key here, with `_` for `-`. Unknown keys are refused.
[HTTP API](../serving/http-api.md#server-settings) lists them all.

Keep the secrets out of that file, in `/etc/qubivex/serve.env`, readable only by root
(systemd reads it before dropping privileges):

```sh
sudo sh -c 'umask 077; cat > /etc/qubivex/serve.env' <<'EOF'
QUBIVEX_API_KEY=replace-with-a-long-random-string
QUBIVEX_ADMIN_TOKEN=replace-with-another
EOF
```

`openssl rand -hex 32` makes a suitable key. If you put `api_key` or `admin_token` in the
TOML file instead, the file must be `chmod 600` and owned by the `qubivex` user, or the
server refuses to start.

### The unit

`/etc/systemd/system/qubivex.service`:

```ini
[Unit]
Description=QUBIVEX inference server
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=qubivex
Group=qubivex
EnvironmentFile=/etc/qubivex/serve.env
ExecStart=/usr/local/bin/qubivex serve --config /etc/qubivex/serve.toml
Restart=on-failure
RestartSec=5

# SIGTERM starts the drain: running requests get --shutdown-grace (30 s) to finish.
# Leave systemd longer than that before it kills the process.
KillSignal=SIGTERM
TimeoutStopSec=45

# The memory planner reads this limit and plans inside it.
# MemoryMax=12G

NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
PrivateDevices=true
ProtectKernelTunables=true
ProtectControlGroups=true
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
ReadOnlyPaths=/var/lib/qubivex/models
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now qubivex
systemctl status qubivex
journalctl -u qubivex -f
curl -s http://127.0.0.1:8080/health
```

`ReadOnlyPaths` suits a server, which only reads models. It applies to the service
alone: `sudo -u qubivex qubivex models pull ...` from a shell can still write there.

With `MemoryMax` set, `qubivex system` run inside the same cgroup shows the limit as the
"cgroup limit", and the planner budgets against the cgroup's room, not the machine's free
memory.

### TLS and the public network

Keep `listen` on loopback and put a reverse proxy in front. An nginx location that
streams correctly:

```nginx
location /v1/ {
    proxy_pass http://127.0.0.1:8080;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;          # deliver server-sent events as they are produced
    proxy_read_timeout 700s;      # longer than --request-timeout (600 s)
    client_max_body_size 4m;      # the server's own limit is 4 MiB
}
```

Do not proxy `/metrics` to the public network: it needs no key. Behind a proxy every
request reaches the server from the proxy's address, so the server's own `rate_limit`
gives all clients one shared allowance. Rate-limit at the proxy instead, and leave
`rate_limit` unset.

## Option 2: Docker

The repository's `Dockerfile` builds a Debian image of about 160 MB (arm64) that runs as a
non-root user (uid 10001) and holds the binary and the tiny fixture models.

```sh
docker build -t qubivex .
```

One image per architecture: build on arm64 for arm64 and on x86-64 for x86-64, or use
`docker buildx build --platform linux/amd64,linux/arm64`.

### Smoke test

```sh
docker run --rm -p 8080:8080 -e QUBIVEX_API_KEY=sk-test qubivex \
  serve --listen 0.0.0.0:8080 --dir /usr/share/qubivex/models --model olmoe-tiny
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models -H 'Authorization: Bearer sk-test'
```

### With real models

Models live in a volume mounted at `/models` (the image sets
`QUBIVEX_MODELS_DIRS=/models`).

```sh
docker volume create qubivex-models

# Download into the volume. Add -e HF_TOKEN for a gated model.
docker run --rm -v qubivex-models:/models qubivex models pull Qwen/Qwen3-0.6B

# Serve.
docker run -d --name qubivex \
  -p 127.0.0.1:8080:8080 \
  -v qubivex-models:/models:ro \
  -e QUBIVEX_API_KEY="$(openssl rand -hex 32)" \
  --memory 12g \
  --stop-timeout 45 \
  --restart unless-stopped \
  qubivex serve --listen 0.0.0.0:8080 --model qwen3-0.6b --log-format json
```

Things that matter:

- **The key.** Inside a container the server must listen on `0.0.0.0`, and it refuses to
  listen beyond loopback without an API key. Pass `QUBIVEX_API_KEY`, or
  `--allow-unauthenticated` when something in front of the container authenticates.
- **`--memory`.** The planner reads the container's cgroup limit and plans inside it:
  under `--memory 2g` it budgeted 1.76 GiB. Without a limit it plans against the host's
  free memory, which other containers also use. The plan sets 512 MiB aside for
  activations, so even the tiny fixture models are refused under a 512 MiB limit.
- **`--stop-timeout`.** `docker stop` sends SIGTERM, and the server then gives running
  requests `--shutdown-grace` (30 s). Docker's default of 10 s would kill them first.
- **`-p 127.0.0.1:8080:8080`** publishes the port to the host only. Publish it wider only
  behind TLS.
- **A bind mount** instead of a volume must be readable by uid 10001, and writable by it
  for `models pull`.
- **Health.** The image has a `HEALTHCHECK` on `/health`, with a start period of 120 s to
  cover the model load. `docker ps` shows `healthy` once it answers.

### Compose

```yaml
services:
  qubivex:
    image: qubivex
    command: ["serve", "--listen", "0.0.0.0:8080", "--model", "qwen3-0.6b", "--log-format", "json"]
    environment:
      QUBIVEX_API_KEY: ${QUBIVEX_API_KEY:?set QUBIVEX_API_KEY}
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - qubivex-models:/models:ro
    mem_limit: 12g
    stop_grace_period: 45s
    restart: unless-stopped
    read_only: true
    cap_drop: [ALL]
    security_opt: ["no-new-privileges:true"]

volumes:
  qubivex-models:
    external: true
```

The server writes nothing to disk, so the root filesystem can be read-only.

### Kubernetes

Nothing here is specific to an orchestrator, and QUBIVEX has not been run on Kubernetes.
The facts a manifest needs:

| Setting | Value |
|---|---|
| Liveness and readiness probe | `GET /health` on the container port. It answers 200 once the models named at start have loaded, because the port opens only then |
| `startupProbe` | Allow for the model load: the dense weights are read before the port opens |
| `terminationGracePeriodSeconds` | More than `--shutdown-grace` (30 s): 45 |
| Memory limit | Set one. The planner reads the cgroup limit |
| Replicas | Each replica loads its own copy of the model and has its own queue and KV. There is no shared state, so any load balancer works, but prefix reuse only helps when a conversation returns to the same replica |
| Secrets | `QUBIVEX_API_KEY` and `QUBIVEX_ADMIN_TOKEN` as environment variables from a Secret |

## Verify the deployment

```sh
KEY=...   # the API key
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models -H "Authorization: Bearer $KEY"
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"Say hello."}],"max_tokens":16}'
curl -s http://127.0.0.1:8080/metrics | grep qubivex_generated_tokens_total
```

## Not supported

- **GPUs on Linux.** There is no CUDA backend (it needs NVIDIA hardware to build and
  measure on) and no Vulkan backend. `--backend metal` exists only on macOS.
- **TLS.** Terminate it at a proxy.
- **More than one API key.** One key serves every client, so clients cannot be told
  apart or limited separately.
- **`io_uring`.** Experts are read with `pread` on dedicated threads. `--direct-io`
  (`O_DIRECT`) is available on `qubivex run` and `bench`, not on `serve`.
- **Hand-written x86-64 kernels.** x86-64 runs the portable kernels compiled for AVX2 and
  FMA. They give the same bits as the arm64 NEON kernels, and they have not been
  benchmarked: every recorded number is from Apple Silicon.
- **Packages.** There is no `.deb`, `.rpm` or published image. Build from source.
