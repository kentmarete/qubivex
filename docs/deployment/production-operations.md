# Production operations

Runbooks for a running `qubivex serve`: what to watch, how to size it, and what to do
when it misbehaves. Installing it is covered in [Linux deployment](linux-deployment.md)
and [macOS deployment](macos-deployment.md).

Examples assume the server on `127.0.0.1:8080`, the API key in `$KEY` and the admin token
in `$ADMIN`.

## Contents

1. [Before going live](#before-going-live)
2. [Sizing](#sizing)
3. [Start, stop, restart](#start-stop-restart)
4. [Monitoring](#monitoring)
5. [Runbooks](#runbooks)
6. [Changing models](#changing-models)
7. [Upgrades and rollback](#upgrades-and-rollback)
8. [Backup](#backup)
9. [Known limits](#known-limits)

## Before going live

| Check | How |
|---|---|
| The API key is set and long | `QUBIVEX_API_KEY`; the server refuses to listen beyond loopback without one |
| The admin token is unset, or different from the API key | Unset, `POST /v1/models/load` and `/unload` answer 403 |
| TLS terminates in front of the server | The server speaks plain HTTP |
| `/metrics` is not reachable from the public network | It needs no key |
| A memory limit is set (cgroup, container or `--ram-budget`) | `qubivex system` shows the cgroup limit; `/v1/system` shows `plan_budget_bytes` |
| The model's files are verified | `qubivex models verify <ID>` |
| The model fits | `qubivex models analyse <ID>` |
| Logs go somewhere kept | `--log-format json` to the journal or the container log |
| The stop timeout exceeds `--shutdown-grace` | systemd `TimeoutStopSec`, `docker stop -t`, Kubernetes `terminationGracePeriodSeconds` |
| Request limits suit your clients | `--max-prompt-tokens`, `--max-generation-tokens`, `--max-body-bytes`, `--request-timeout` |

## Sizing

### Memory

A model needs its dense weights resident, at least one expert per layer, and KV for every
conversation running at once. Everything beyond that is expert cache, which is speed.

```sh
qubivex models analyse <ID> --ctx 8192
qubivex plan <ID> --ctx 8192 --sequences 4
```

`plan --sequences N` shows the placement with `N` conversations' KV reserved, which is
what `--max-concurrent N` can ask for. If the plan shortens the context or leaves the
expert cache near one slot per layer, lower `--max-concurrent` or `--context`, or give it
more RAM.

A request is admitted only when the plan can hold its KV. With `--memory-policy queue`
(the default) a request that does not fit waits; with `refuse` it gets a 503 at once.

Several models in one server each plan their own memory when they load, and each has its
own compute threads. Nothing arbitrates between them. Give each an explicit
`--ram-budget` that sums to less than the machine, or run one server per model.

### Concurrency

| Setting | Default | Raise it when | Lower it when |
|---|---|---|---|
| `--max-concurrent` | 4 | Slots are always busy and memory has room. Throughput grows with shared passes: on OLMoE-1B-7B, eight concurrent conversations gave 1.5× the tokens per second of eight run in turn on the CPU, 1.9× on Metal | Per-conversation speed matters more than total: each running conversation gets slower as the pass widens |
| `--max-queue` | 32 | Clients would rather wait than retry | Clients time out before the queue drains |
| `--queue-timeout` | 30 s | | Clients give up sooner than this |
| `--threads` | The CPU's performance cores | | Other work shares the machine |

### Disk

Experts outside the cache are read during generation. The rate of
`qubivex_expert_read_bytes_total` is the read throughput in use, and
`qubivex doctor --storage --path <models dir>` measures what the disk can deliver. If the
first is near the second, generation speed is bound by the disk: add RAM for a larger
cache before anything else.

## Start, stop, restart

| Action | systemd | Docker |
|---|---|---|
| Start | `systemctl start qubivex` | `docker start qubivex` |
| Stop, draining | `systemctl stop qubivex` | `docker stop -t 45 qubivex` |
| Restart | `systemctl restart qubivex` | `docker restart -t 45 qubivex` |
| Logs | `journalctl -u qubivex -f` | `docker logs -f qubivex` |

**Startup.** The server checks every setting, loads the models named at start, and only
then opens the port. A wrong setting or a model that does not load ends the process with
status 1 and a message, before anything listens. So a port that answers `/health` means
the starting models are loaded.

**Shutdown.** SIGTERM or SIGINT starts a drain:

1. The listener closes. Requests waiting in the queue get a 503 at once.
2. Running generations get `--shutdown-grace` (30 s) to finish.
3. After the grace, a stream still running gets an error event and `data: [DONE]`, and a
   non-streaming request gets a 503.
4. The process exits with status 0, at most 5 s after the grace.

The log shows `server_stopping` with the number of generations still active, then
`server_stopped`. SIGKILL skips all of this; the server holds no state on disk, so
nothing needs repair afterwards.

**A restart drops** every loaded model, every retained KV prefix and all counters. Models
loaded through the admin endpoint are not remembered: list every model you want in
`models` in the config.

## Monitoring

### Health

`GET /health` answers `{"models":1,"status":"ok"}` with no key. It says the process is
serving and how many models are loaded. It does not run a generation. For a deeper probe,
send a one-token completion with the API key from your monitoring system.

### Metrics

`GET /metrics` is Prometheus text. The full list is in
[HTTP API](../serving/http-api.md#get-metrics). The ones to put on a dashboard:

| Question | Expression |
|---|---|
| Is it serving? | `sum(rate(qubivex_requests_total[5m])) by (endpoint, status)` |
| Error rate | `sum(rate(qubivex_requests_total{status=~"5.."}[5m])) / sum(rate(qubivex_requests_total[5m]))` |
| Throughput | `rate(qubivex_generated_tokens_total[5m])` |
| Time to first token, p95 | `histogram_quantile(0.95, sum(rate(qubivex_time_to_first_token_seconds_bucket[5m])) by (le, model))` |
| Queue wait, p95 | `histogram_quantile(0.95, sum(rate(qubivex_queue_wait_seconds_bucket[5m])) by (le, model))` |
| Saturation | `qubivex_generation_slots_free`, `qubivex_queue_depth` |
| Refusals | `rate(qubivex_queue_rejected_total[5m])` |
| Expert cache hit rate | `rate(qubivex_expert_cache_hits_total[5m]) / (rate(qubivex_expert_cache_hits_total[5m]) + rate(qubivex_expert_cache_loads_total[5m]))` |
| Mean expert load time | `rate(qubivex_expert_load_wait_seconds_total[5m]) / rate(qubivex_expert_cache_loads_total[5m])` |
| Mean batch width | `rate(qubivex_batch_steps_total[5m]) / rate(qubivex_batch_passes_total[5m])` |
| Memory against the plan | `qubivex_memory_dense_bytes + qubivex_memory_expert_cache_bytes + qubivex_memory_kv_bytes` against `qubivex_memory_budget_bytes`; `qubivex_process_resident_bytes` |

### Alerts to start from

The thresholds are starting points, not measurements. Tune them to your model and
clients.

```yaml
groups:
  - name: qubivex
    rules:
      - alert: QubivexDown
        expr: up{job="qubivex"} == 0
        for: 2m
      - alert: QubivexErrors
        expr: sum(rate(qubivex_requests_total{status=~"5.."}[5m])) > 0.05
        for: 5m
      - alert: QubivexGenerationsFailing
        expr: rate(qubivex_generations_failed_total[5m]) > 0
        for: 5m
      - alert: QubivexQueueRejecting
        expr: sum(rate(qubivex_queue_rejected_total[5m])) by (model) > 0
        for: 10m
      - alert: QubivexSlowFirstToken
        expr: histogram_quantile(0.95, sum(rate(qubivex_time_to_first_token_seconds_bucket[5m])) by (le, model)) > 10
        for: 10m
      - alert: QubivexExpertReadTrouble
        expr: rate(qubivex_expert_read_retries_total[5m]) > 0 or rate(qubivex_expert_read_timeouts_total[5m]) > 0
        for: 5m
```

### Logs

One line per event on standard error. With `--log-format json` each line is one object:

```json
{"ts":"2026-11-10T01:14:46.414Z","level":"info","event":"generation_finished","request_id":"req-6af27006-1","model":"olmoe-tiny","sequence_id":1,"finish":"length","prompt_tokens":18,"cached_tokens":0,"completion_tokens":6,"ttft_ms":30.149,"tokens_per_second":1185.95,"duration_ms":34.387}
```

No line holds a prompt, a message or generated text. Every response carries an
`x-request-id` header, and the same id is on that request's log lines, so a client's
report can be matched to the log:

```sh
journalctl -u qubivex -o cat | grep req-6af27006-1
```

Events worth alerting on from the log: `internal_error`, `generation_panicked`,
`generation_failed`, `model_load_failed`.

### Live state

```sh
curl -s http://127.0.0.1:8080/v1/system -H "Authorization: Bearer $KEY"
```

shows, per model, the plan's budget, the ledger (dense, expert cache, KV), reserved KV
positions, free slots, the queue and the prefix cache, plus each model's state (loading,
loaded or failed) and how many unloaded models are still draining.

## Runbooks

### Clients get 429

There are two different 429s. The `code` in the error body tells them apart.

| `code` | Meaning | Action |
|---|---|---|
| `rate_limit_exceeded` (with a `retry-after` header) | One client address used its `--rate-limit` allowance | Expected under a limit. If every client shares it, the server is behind a proxy: limit at the proxy and unset `--rate-limit` |
| `too_many_requests` | Every slot was busy and the queue was full, or the request waited `--queue-timeout` | Check `qubivex_generation_slots_free` and `qubivex_queue_depth`. If slots are always 0, the server is at capacity: raise `--max-concurrent` if memory allows (see [Sizing](#sizing)), add a replica, or shorten generations with `--max-generation-tokens` |

### Clients get 503 `overloaded`

The message says which case:

- **"memory: ..."**: the request's KV does not fit the plan (with `--memory-policy
  refuse`), or the pressure monitor has reached its critical level (under 256 MiB
  available) and refuses new requests while running ones finish. Find what took the
  memory: `qubivex_process_resident_bytes` against the budget, and other processes on
  the host. Lower `--max-concurrent`, `--context` or `--prefix-cache-mb`, or set a
  `--ram-budget` that leaves room.
- **"storage: ..."**: a disk problem that is about capacity. Check the disk.
- **"the server is shutting down"** or **"the model is being unloaded"**: expected
  during a stop or an unload. Clients should retry against another replica.

### Clients get 504 `deadline_exceeded`

The request set its own `timeout` and it passed while the request waited in the queue.
The server is saturated for that client's patience: treat as
[429 `too_many_requests`](#clients-get-429).

### Clients get 500 `internal_error`

The body names only the request id. The detail is in the log:

```sh
journalctl -u qubivex -o cat | grep '"event":"internal_error"' | tail
```

The usual cause is an expert that could not be read: see the next runbook. A
`generation_panicked` event is a bug: the request failed, the server kept running, and
the log line plus the request are what a bug report needs.

### Expert reads fail or time out

Signs: `qubivex_expert_read_retries_total` or `qubivex_expert_read_timeouts_total` rising,
`internal_error` lines naming an expert, generations failing mid-stream.

A failed read fails only the requests waiting on that expert. Nothing wrong is emitted,
the memory is returned, and the next request reads the expert again.

1. Check the disk and the kernel log (`dmesg`, SMART) for I/O errors.
2. Check the files: `qubivex models verify <ID>`. It checks every file against the
   manifest's checksums and every expert against `experts.qvxsum`.
3. If a file is damaged, restore it from a copy or pull the model again
   (`qubivex models remove <ID> --yes`, then `qubivex models pull ...`), and restart or
   reload the model.
4. On a network or slow disk, timeouts without errors mean the disk cannot keep up. Move
   the model to local SSD, or give the model more RAM so fewer experts stream.

### The first token is slow

Compare the two histograms:

- `qubivex_queue_wait_seconds` high: requests wait for a slot or for memory. This is
  saturation: see [429](#clients-get-429).
- Queue wait low, `qubivex_time_to_first_token_seconds` high: the time is prefill. Long
  prompts cost prefill in proportion to their length. For chat clients, which resend the
  conversation every turn, `--prefix-cache-mb` skips the part already run: on
  OLMoE-1B-7B a 714-token second turn took 0.57 s instead of 6.59 s. Have clients send a
  stable `user` value, because prefixes are only reused within one `user`.
- Right after a start, the first requests read experts from a cold disk cache. This
  passes once they have been read.

### Generation is slow

1. `qubivex_generations_active`: every running conversation shares the cores. Total
   tokens per second rises with concurrency and each conversation's falls.
2. Expert cache hit rate (above): a low rate means experts are streaming. Check
   `qubivex_expert_cache_bytes` against the model's expert bytes
   (`qubivex models inspect <ID>`). The planner shrinks the cache as KV grows, so fewer
   or shorter conversations leave more cache.
3. Other load on the machine. The compute threads assume they own the performance cores.
4. Memory pressure: below 512 MiB available the monitor shrinks the expert cache. It
   regrows when memory returns.

### The process is killed for memory

The planner budgets a fraction of the memory available when the model loaded. It can be
exceeded when other processes grow afterwards, when several models each planned alone,
or when no cgroup limit told it the real bound.

1. Set a memory limit on the service or container, so the planner sees it.
2. Set `--ram-budget` explicitly, or use `--profile balanced` (0.75) or `saver` (0.50).
3. Run one model per server.

### A model does not load

At start the process exits with the reason. Through the admin endpoint the reason is in
the response and `/v1/system` lists the model as `failed`. Then:

```sh
qubivex models list                 # a broken model is listed with its error
qubivex models inspect <ID>
qubivex models analyse <ID>         # does it fit?
qubivex models verify <ID>          # are the files intact?
```

A model that "does not fit" needs more RAM, a smaller `--context`, or fewer
`--max-concurrent`.

### A download fails or fills the disk

`qubivex models pull` checks free space first, writes each file as `<name>.part`, and
installs the model only when every file passed its checksum. A failed or interrupted
pull keeps its files: run the same command to continue. To give up on it,
`qubivex models remove <ID> --yes` deletes the partial files too.

## Changing models

With an admin token, models load and unload without a restart:

```sh
curl -s http://127.0.0.1:8080/v1/models/load \
  -H "Authorization: Bearer $ADMIN" -H 'content-type: application/json' \
  -d '{"model":"qwen3-0.6b"}'
```

```json
{"created":1794294611,"id":"qwen3-0.6b","object":"model","owned_by":"qubivex"}
```

```sh
curl -s http://127.0.0.1:8080/v1/models/unload \
  -H "Authorization: Bearer $ADMIN" -H 'content-type: application/json' \
  -d '{"model":"qwen3-0.6b"}'
```

```json
{"deleted":true,"draining":0,"id":"qwen3-0.6b","object":"model"}
```

- The model is looked up in the server's `--dir` directories.
- Loading a model that is loading joins that load. Loading a loaded model answers 200.
- Unloading closes the model's queue (waiters get a 503) and lets running generations
  finish. `draining` is how many were still running; the weights are freed when the last
  ends.
- **Replacing a model's files:** unload it, replace the files, run
  `qubivex models verify <ID> --record` if you changed them on purpose, load it again.
  Do not change the files of a loaded model: the server reads experts from them while it
  runs.
- `qubivex models remove` does not check whether a server has the model loaded. Unload
  first.
- Add the model to `models` in the config too, or it is gone after the next restart.

## Upgrades and rollback

1. Build the new binary, or the new image, and keep the old one.
2. Read the release's notes for changed defaults and flags. Unknown config keys are
   refused at start, so a removed key stops the new version before it serves.
3. Check the config against the new binary without serving: start it on a spare port
   (`--listen 127.0.0.1:18080`) and stop it once `/health` answers.
4. Stop the old server (it drains), start the new one.
5. Roll back by starting the old binary. There is no on-disk state to migrate back.

Model directories are forward compatible: a 1.0 manifest loads as it is, and
`qubivex models migrate` is optional. A manifest newer than the binary is refused by
name.

With two or more replicas behind a load balancer, upgrade one at a time: remove it from
the balancer, stop it, start the new version, wait for `/health`, put it back.

## Backup

| What | Where | Note |
|---|---|---|
| Configuration and secrets | `/etc/qubivex/` | The only thing that cannot be recreated |
| Models | The models directory, with `registry.json` | Can be pulled again. `registry.json` and each `manifest.json` record the source and exact revision, so a pull of that revision gives the same files |
| Server state | None | Retained KV and counters live in memory only |

## Known limits

- **One API key.** Every client holds the same key. The `user` field is whatever the
  client sends, so fairness and prefix-cache separation hold only among clients that do
  not lie about it. `priority: "high"` is open to anyone with the key.
- **No per-key rate limits or quotas.** The rate limit counts peer addresses.
- **Models share nothing and arbitrate nothing.** Two loaded models can together exceed
  what either planned for.
- **Cancellation lands between tokens or prefill chunks,** not inside a forward pass.
- **No tracing.** There are metrics and logs, and no OpenTelemetry export.
- **Not measured:** peak GPU memory and memory fragmentation. On macOS the peak resident
  size is the highest value a scrape has read.
- **No hot reload of settings.** Changing any flag, including the API key, needs a
  restart.
