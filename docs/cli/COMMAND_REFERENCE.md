# Qubivex CLI Command Reference

A consolidated reference for the Qubivex command-line interface (CLI),
including command descriptions, syntax, examples, and common workflows.

> **Version note:** This document consolidates the command inventory and
> examples previously documented for Qubivex. CLI options can change
> between versions. Treat `qubivex --help` and the help output for each
> installed subcommand as authoritative for the binary you are using.
> Commands and flags in this document should be verified against the
> current repository before being treated as an exhaustive,
> release-validated specification.

## 1. General CLI help and version

``` bash
# Show the root command list and general usage
qubivex --help

# Show the installed version
qubivex --version

# Get help for individual command groups and commands
qubivex run --help
qubivex serve --help
qubivex bench --help
qubivex models --help
qubivex models pull --help
qubivex keys --help
qubivex doctor --help
```

Every command should be checked with `--help` before relying on optional
flags.

## 2. Model management

Model-management commands cover discovery, downloading, importing,
inspection, compatibility analysis, verification, migration, and
removal.

### 2.1 Search for models

Searches Hugging Face for models matching a query.

``` bash
qubivex models search "Qwen3"
qubivex models search "OLMoE"
qubivex models search "Llama 3" --limit 10
qubivex models search "embedding" --json
```

### 2.2 Pull or download a model

Downloads model files and installs the model in the Qubivex model
directory.

``` bash
# Download a model
qubivex models pull Qwen/Qwen3-0.6B

# Select a specific file from a repository
qubivex models pull owner/Model-GGUF \
  --file Model-Q4_K_M.gguf

# Select a repository revision
qubivex models pull Qwen/Qwen3-0.6B \
  --revision main

# Assign a local model ID and display name
qubivex models pull Qwen/Qwen3-0.6B \
  --id qwen3-0.6b \
  --name "Qwen3 0.6B"

# Override missing model configuration values
qubivex models pull owner/model \
  --set norm_topk_prob=false

# Choose an installation directory
qubivex models pull Qwen/Qwen3-0.6B \
  --dir ~/ai-models

# Limit concurrent downloads
qubivex models pull Qwen/Qwen3-0.6B \
  --concurrency 2
```

Options documented for this command:

  Option            Description
  ----------------- -----------------------------------------------
  `--file`          Select specific repository files
  `--revision`      Select a branch, tag, or commit
  `--id`            Set the local model ID
  `--name`          Set the display name
  `--set`           Supply or override model configuration values
  `--dir`           Set the model installation directory
  `--concurrency`   Control parallel downloads

### 2.3 List installed models

Lists models found in the configured model directory.

``` bash
qubivex models list

# JSON output
qubivex models list --json

# Use a custom model directory
qubivex models list --dir ~/ai-models
```

### 2.4 Inspect a model

Displays model metadata, architecture information, dimensions, tensor
counts, weight sizes, data types, and configuration.

``` bash
qubivex models inspect olmoe-1b-7b-0125-instruct

# Include tensor-level details
qubivex models inspect olmoe-tiny --tensors

# JSON output
qubivex models inspect olmoe-tiny --json

# Inspect a model in a custom directory
qubivex models inspect olmoe-tiny --dir fixtures/models
```

### 2.5 Analyse model compatibility

Assesses whether a model is compatible with the current machine and
estimates memory requirements.

The command may report whether the model fits in RAM, requires
disk-backed weight streaming, or exceeds the available resource budget.

``` bash
qubivex models analyse olmoe-1b-7b-0125-instruct

# Assess against a 2 GiB RAM budget
qubivex models analyse olmoe-1b-7b-0125-instruct \
  --ram-budget 2GiB

# Include an 8K context estimate
qubivex models analyse olmoe-1b-7b-0125-instruct \
  --ram-budget 2GiB \
  --ctx 8192

# JSON output
qubivex models analyse olmoe-1b-7b-0125-instruct \
  --ram-budget 2GiB \
  --ctx 8192 \
  --json
```

Aliases documented for this command:

``` bash
qubivex models analyze olmoe-tiny
```

### 2.6 Import a local model

Imports a local checkpoint and generates a Qubivex model manifest. The
documented formats include SafeTensors and GGUF.

``` bash
# Import a local checkpoint and write its manifest
qubivex models import ./models/my-model \
  --id my-model \
  --write

# Supply missing architecture configuration
qubivex models import ./models/my-model \
  --id my-model \
  --set norm_topk_prob=false \
  --write

# Set a display name
qubivex models import ./models/my-model \
  --id my-model \
  --name "My Local Model" \
  --write

# Replace an existing manifest
qubivex models import ./models/my-model \
  --id my-model \
  --write \
  --force
```

Options documented for this command:

  Option      Description
  ----------- -----------------------------------------------
  `--id`      Assign the local model identifier
  `--name`    Set the display name
  `--set`     Supply or override model configuration values
  `--write`   Write the manifest to disk
  `--force`   Replace an existing manifest

### 2.7 Verify model integrity

Checks model files against manifest checksums and performs the
documented integrity checks.

``` bash
qubivex models verify olmoe-tiny

# Configure verification thread count
qubivex models verify olmoe-tiny --threads 4

# Recalculate and record checksums
qubivex models verify olmoe-tiny --record
```

> **Caution:** `--record` changes the recorded checksums to match the
> files currently present. Use it only when intentionally accepting the
> current files as the trusted baseline.

Alias:

``` bash
qubivex models validate olmoe-tiny
```

### 2.8 Migrate a model manifest

Upgrades an older model manifest to the current schema.

``` bash
# Preview or inspect the migration
qubivex models migrate olmoe-tiny

# Write the migrated manifest
qubivex models migrate olmoe-tiny --write

# Migrate without generating checksums
qubivex models migrate olmoe-tiny \
  --write \
  --no-checksums
```

### 2.9 Remove an installed model

Removes a model from the configured model directory.

``` bash
# Invoke the removal command
qubivex models remove olmoe-tiny

# Skip interactive confirmation
qubivex models remove olmoe-tiny --yes

# Specify a model directory
qubivex models remove olmoe-tiny \
  --dir ~/ai-models \
  --yes
```

### 2.10 List supported architectures

Lists the model architectures supported by Qubivex and their relevant
implementation capabilities.

``` bash
qubivex models architectures
```

This is useful when checking whether a model family can be imported and
executed by an existing architecture implementation, or whether
additional architecture support is required.

## 3. Inference commands

### 3.1 Run a model

Runs local inference and writes generated text to the terminal.

``` bash
# Basic text generation
qubivex run \
  --model olmoe-1b-7b-0125-instruct \
  --prompt "Explain artificial intelligence"

# Limit generated tokens
qubivex run \
  --model olmoe-1b-7b-0125-instruct \
  --prompt "Explain neural networks" \
  --max-tokens 200

# Apply the model's chat template
qubivex run \
  --model olmoe-1b-7b-0125-instruct \
  --prompt "What is machine learning?" \
  --chat

# Set the expert cache budget
qubivex run \
  --model olmoe-1b-7b-0125-instruct \
  --prompt "Explain mixture of experts" \
  --expert-cache-mb 64

# Set the context length
qubivex run \
  --model olmoe-1b-7b-0125-instruct \
  --prompt "Explain transformers" \
  --context 8192

# Set the CPU thread count
qubivex run \
  --model olmoe-1b-7b-0125-instruct \
  --prompt "Explain inference" \
  --threads 8

# Configure sampling
qubivex run \
  --model olmoe-1b-7b-0125-instruct \
  --prompt "Write a story" \
  --temperature 0.7 \
  --top-p 0.9 \
  --seed 42

# Suppress performance statistics
qubivex run \
  --model olmoe-1b-7b-0125-instruct \
  --prompt "Hello" \
  --quiet
```

Options documented for this command:

  Option                Description
  --------------------- ------------------------------------
  `--model`             Model ID or model directory
  `--prompt`            Input prompt
  `--chat`              Apply the model's chat template
  `--max-tokens`        Maximum number of generated tokens
  `--temperature`       Sampling temperature
  `--top-p`             Nucleus sampling threshold
  `--seed`              Random seed
  `--expert-cache-mb`   Expert-cache memory budget
  `--context`           Context window size
  `--threads`           CPU compute thread count
  `--quiet`             Suppress performance statistics
  `--dir`               Additional model search directory

The expert cache controls the memory budget available for keeping
experts readily accessible. A smaller cache may increase disk reads and
inference latency for MoE models.

### 3.2 Start the inference server

Starts the Qubivex HTTP inference server.

``` bash
# Start a server with one model
qubivex serve \
  --model olmoe-1b-7b-0125-instruct

# Configure multiple models
qubivex serve \
  --model olmoe-1b-7b-0125-instruct \
  --model qwen3-0.6b

# Set the listening address and API key
qubivex serve \
  --listen 0.0.0.0:8080 \
  --model qwen3-0.6b \
  --api-key sk-local

# Start in the background
qubivex serve \
  --model qwen3-0.6b \
  --api-key sk-local \
  --detach

# Check background server status
qubivex serve --status

# Stop background servers
qubivex serve --stop

# Stop a server bound to a particular address
qubivex serve \
  --stop \
  --listen 127.0.0.1:8080

# Load server configuration from TOML
qubivex serve \
  --config ~/.qubivex/serve.toml
```

Documented server options:

  Option                      Description
  --------------------------- --------------------------------------------
  `--config`                  Load TOML configuration
  `--listen`                  Server bind address
  `--model`                   Model to serve; repeatable
  `--dir`                     Model directories
  `--backend`                 Compute backend
  `--api-key`                 Client authentication key
  `--admin-token`             Administrative authentication token
  `--detach`                  Run in the background
  `--status`                  Show background server status
  `--stop`                    Stop background server process(es)
  `--cors-origin`             Allowed browser origins
  `--rate-limit`              Requests-per-minute limit
  `--rate-limit-burst`        Burst request allowance
  `--max-concurrent`          Concurrent generations per model
  `--max-queue`               Maximum queued requests
  `--max-prompt-tokens`       Maximum prompt size
  `--max-generation-tokens`   Maximum generation size
  `--prefix-cache-mb`         Memory budget for reusable prefix/KV cache
  `--log-format`              Logging format
  `--no-metrics`              Disable Prometheus metrics
  `--no-context-management`   Refuse requests longer than the context (400)
  `--context-note-share`      Room kept for the note on older turns (0.25)
  `--context-store`           Keep context checkpoints in files in this directory
  `--context-store-mb`        Size of the context checkpoint store
  `--context-store-days`      Age limit of context checkpoint files

The previously documented HTTP endpoints are:

  Endpoint                           Purpose
  ---------------------------------- ---------------------------------
  `GET /health`                      Health check
  `GET /metrics`                     Prometheus metrics
  `GET /v1/models`                   List loaded models
  `GET /v1/models/{id}`              Inspect a loaded model
  `POST /v1/chat/completions`        Chat completion
  `POST /v1/completions`             Text completion
  `GET /v1/system`                   Hardware and memory information
  `POST /v1/models/load`             Load a model
  `POST /v1/models/unload`           Unload a model
  `GET /v1/admin/models/installed`   List installed models
  `GET /v1/admin/models/search`      Search for models
  `POST /v1/admin/models/pull`       Download a model
  `GET /v1/admin/jobs`               Inspect download jobs

Administrative endpoints require the configured administrative token.
Confirm exact paths, authentication rules, and endpoint availability
against the server implementation for the target release.

## 4. Benchmarking and hardware diagnostics

### 4.1 Benchmark performance

Runs the Qubivex benchmarking command.

``` bash
qubivex bench
```

Use the command's help output to discover the benchmark modes, model
selection options, warm-up controls, iteration counts, and output
formats supported by the installed version:

``` bash
qubivex bench --help
```

Do not assume that benchmark flags from another inference engine are
supported by Qubivex.

### 4.2 Inspect system hardware

Detects and reports hardware available to Qubivex.

``` bash
# Human-readable output
qubivex system

# Machine-readable output
qubivex system --json
```

The system report may include:

-   CPU model and core configuration.
-   Performance and efficiency cores.
-   SIMD capabilities.
-   NUMA configuration.
-   Available system RAM.
-   Container memory limits.
-   GPU devices and compute-backend availability.
-   Mounted storage devices and capacity.

### 4.3 Diagnose storage performance

Measures storage performance, which is particularly relevant to
disk-backed weight and MoE expert streaming.

``` bash
# Test the current filesystem
qubivex doctor --storage

# Test a specific model directory
qubivex doctor --storage \
  --path ~/.qubivex/models

# Increase the test file size
qubivex doctor --storage \
  --path /Volumes/AI-Models \
  --size-mib 2048

# JSON output
qubivex doctor --storage --json
```

The documented test areas include sequential reads, random reads,
buffered reads, and direct-I/O performance. Use the results to
understand whether the storage device may constrain expert loading and
inference throughput.

## 5. API key management

### 5.1 Generate a named client key

Generates an API key for a named client.

``` bash
qubivex keys new billing-app
```

Example with a client rate limit:

``` bash
qubivex keys new billing-app \
  --rate-limit 600
```

The key is intended to be displayed once. Store it securely; use the
generated hash in server configuration where required.

### 5.2 Hash an existing API key

Reads a key from standard input and outputs its SHA-256 hash.

``` bash
printf '%s' 'my-secret-client-key' | qubivex keys hash
```

Avoid placing real secrets directly in shell history. Prefer a secure
secret-input method when handling production credentials.

## 6. Installation, updates, and maintenance

### 6.1 Install Qubivex

The following installer command was documented for the repository:

``` bash
curl -fsSL https://raw.githubusercontent.com/kentmarete/qubivex/main/install.sh | bash
```

Review the installer script before executing it, particularly on
production or security-sensitive machines.

### 6.2 Update Qubivex

``` bash
# Check for updates
qubivex update --check

# Update
qubivex update

# Run update in the background
qubivex update --detach

# Check background update status
qubivex update --status

# Select a branch
qubivex update --branch main

# Limit build parallelism
qubivex update --jobs 4
```

### 6.3 Uninstall Qubivex

The following repository installer option was documented for
uninstalling the CLI:

``` bash
./install.sh --uninstall
```

The documented behaviour is to remove the CLI binary while retaining
model data and the `~/.qubivex` directory. Verify the current installer
implementation before relying on that behaviour.

## 7. Command inventory

The following is the consolidated command inventory covered by this
reference.

  --------------------------------------------------------------------------------
  Area                    Command                          Purpose
  ----------------------- -------------------------------- -----------------------
  General                 `qubivex --help`                 Show root help

  General                 `qubivex --version`              Show installed version

  Inference               `qubivex run`                    Run local model
                                                           inference

  Serving                 `qubivex serve`                  Start and manage the
                                                           HTTP inference server

  Benchmarking            `qubivex bench`                  Run performance
                                                           benchmarks

  Hardware                `qubivex system`                 Inspect system hardware
                                                           and resources

  Diagnostics             `qubivex doctor --storage`       Diagnose storage
                                                           performance

  Model discovery         `qubivex models search`          Search model
                                                           repositories

  Model download          `qubivex models pull`            Download and install a
                                                           model

  Model inventory         `qubivex models list`            List installed models

  Model inspection        `qubivex models inspect`         Inspect model metadata
                                                           and tensors

  Compatibility           `qubivex models analyse`         Analyse compatibility
                                                           and memory requirements

  Compatibility alias     `qubivex models analyze`         Alias for
                                                           `models analyse`

  Local import            `qubivex models import`          Import a local
                                                           checkpoint and write a
                                                           manifest

  Integrity               `qubivex models verify`          Verify model files and
                                                           checksums

  Integrity alias         `qubivex models validate`        Alias for
                                                           `models verify`

  Migration               `qubivex models migrate`         Migrate model manifests

  Removal                 `qubivex models remove`          Remove an installed
                                                           model

  Architecture            `qubivex models architectures`   List supported
                                                           architectures

  Authentication          `qubivex keys new`               Generate a named client
                                                           API key

  Authentication          `qubivex keys hash`              Hash a key read from
                                                           standard input

  Maintenance             `qubivex update`                 Update Qubivex
  --------------------------------------------------------------------------------

## 8. Common end-to-end workflows

### 8.1 Download and run a model

``` bash
# 1. Inspect hardware
qubivex system

# 2. Search for a model
qubivex models search "Qwen3"

# 3. Download the model
qubivex models pull Qwen/Qwen3-0.6B

# 4. List installed models
qubivex models list

# 5. Analyse compatibility
qubivex models analyse qwen3-0.6b

# 6. Inspect the model
qubivex models inspect qwen3-0.6b

# 7. Run inference
qubivex run \
  --model qwen3-0.6b \
  --prompt "Explain how a transformer works"
```

### 8.2 Import and run a local GGUF model

``` bash
# 1. Import the local checkpoint
qubivex models import ./my-model.gguf \
  --id my-model \
  --write

# 2. Inspect it
qubivex models inspect my-model

# 3. Verify integrity
qubivex models verify my-model

# 4. Analyse compatibility
qubivex models analyse my-model

# 5. Run inference
qubivex run \
  --model my-model \
  --prompt "Hello Qubivex"
```

### 8.3 Start an inference server

``` bash
# 1. Inspect hardware
qubivex system

# 2. Check storage performance
qubivex doctor --storage \
  --path ~/.qubivex/models

# 3. Start the server
qubivex serve \
  --model qwen3-0.6b \
  --listen 127.0.0.1:8080 \
  --api-key sk-local \
  --detach

# 4. Check server status
qubivex serve --status

# 5. Check health
curl http://127.0.0.1:8080/health

# 6. Stop the server
qubivex serve --stop
```

### 8.4 Assess a large MoE model

``` bash
# 1. Inspect hardware
qubivex system

# 2. Benchmark model storage
qubivex doctor --storage \
  --path ~/.qubivex/models

# 3. Download a model (example repository)
qubivex models pull Qwen/Qwen3-30B-A3B

# 4. Analyse memory requirements
qubivex models analyse qwen3-30b-a3b \
  --ram-budget 16GiB \
  --ctx 8192

# 5. Inspect model tensors
qubivex models inspect qwen3-30b-a3b \
  --tensors

# 6. Run with a constrained expert cache
qubivex run \
  --model qwen3-30b-a3b \
  --prompt "Explain mixture of experts" \
  --expert-cache-mb 512
```

Model IDs and repository names in examples are illustrative; use the
actual ID created by the pull/import operation.

## 9. Verifying the CLI inventory for a release

For a release-quality command reference, capture the actual help output
from the built binary and compare it with this document.

``` bash
qubivex --version
qubivex --help

qubivex run --help
qubivex serve --help
qubivex bench --help
qubivex system --help
qubivex doctor --help

qubivex models --help
qubivex models search --help
qubivex models pull --help
qubivex models list --help
qubivex models inspect --help
qubivex models analyse --help
qubivex models import --help
qubivex models verify --help
qubivex models migrate --help
qubivex models remove --help
qubivex models architectures --help

qubivex keys --help
qubivex keys new --help
qubivex keys hash --help

qubivex update --help
```

For repository maintenance, consider generating this reference from the
CLI's own command definitions or help output as part of CI. That reduces
the risk of documentation drifting from the implementation.

## 10. Repository placement

Suggested location:

``` text
docs/
└── cli/
    └── COMMAND_REFERENCE.md
```

Suggested link from the repository README:

``` markdown
## CLI documentation

See the [Qubivex CLI Command Reference](docs/cli/COMMAND_REFERENCE.md)
for command descriptions, options, examples, and common workflows.
```
