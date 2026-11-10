# MoE routing

A Mixture-of-Experts model replaces each layer's one large feed-forward network with
many small ones, the **experts**, and a **router** that picks a few per token. This page
explains how QUBIVEX runs that, and why it is what the rest of the design is built
around.

## The shape of a MoE model

OLMoE-1B-7B is the reference: 16 layers, 64 experts per layer, 8 chosen per token.

| | Dense part | Routed experts |
|---|---|---|
| What | Embedding, attention, norms, routers, output head | 1,024 experts (16 × 64) |
| Used by | Every token | Only the chosen ones: 8 of 64 per layer |
| Size (Q4_K_M file) | About 300 MiB | About 3.6 GiB |
| Kept | Always in memory | In a cache, read from disk on demand |

Each token touches all of the small part and an eighth of the large part. That is the
property QUBIVEX uses: the experts do not all have to be in memory.

## One token through one MoE layer

After attention, for each token:

1. **Score.** The router is one small matrix. Multiplying the token's hidden state by
   it gives one logit per expert.
2. **Select.** A softmax over the logits gives probabilities. The top K experts are
   chosen (K is `num_experts_per_tok` in the model's config). Ties go to the lower
   expert id, so the choice is deterministic.
3. **Weigh.** Each chosen expert's weight is its probability. If the model's
   `norm_topk_prob` is true, the K weights are renormalised to sum to 1 (Qwen3-MoE); if
   false, they are used as they are (OLMoE).
4. **Run.** Each chosen expert is a SwiGLU network of three matrices (gate, up, down).
   The token's hidden state goes through each one.
5. **Combine.** The outputs are summed in **router rank order**, weighted, and added to
   the residual stream.

```text
hidden ──► router ──► softmax ──► top-K ──► expert 17 ──┐
                                      ├───► expert 3  ──┼──► weighted sum, in rank order ──► + residual
                                      └───► expert 42 ──┘
```

## Routing and the expert cache

Step 4 is where a MoE layer meets the expert store. The layer asks for all of a pass's
chosen experts at once. Experts already in memory are leased immediately. The others
are read from disk by I/O threads while the layer computes the ones it has, and are used
as they arrive. See [Expert streaming](expert-streaming.md).

Experts can therefore arrive in any order, and the output must not depend on that
order. It does not, because each expert writes only its own result and the combination
in step 5 always runs in router rank order, after all of them are done.

## The rules routing keeps

**Placement decides speed, never bits.** Which experts are chosen depends only on the
hidden state. Whether an expert was in the cache, came from disk, or ran on the GPU
changes how long the token takes and nothing else. The same prompt gives the same
tokens with every expert in RAM and with one cache slot per layer.

**Prediction never routes.** Prefetching (below) reads router state to guess what will
be needed. It can never change which experts are used: the real router runs on the real
hidden state every time.

**Routing is per token.** In a pass over many tokens (a prompt, or several
conversations' steps), each token is routed on its own. The layer loads the union of
their experts once, and each token uses its own K.

## Prefetch

During decode, the next layer's experts can be guessed before the layer runs: the next
layer's router is applied early to the current hidden state, and the predicted experts
are read ahead. A wrong guess costs a read; it cannot change the output.

It is **off by default** and available on `qubivex run` and `qubivex bench` with
`--prefetch`. It only matters when experts are streaming from a disk slow enough for
reads to hold compute back, and with several conversations in a pass the reads already
overlap with compute, so nothing further is predicted for batches (ADR-017 §7). Related
flags: `--prefetch-width`, `--prefetch-confidence`, `--prefetch-smooth`,
`--prefetch-guard` and `--prefetch-throttle`.

## Routing on the GPU

With `--backend metal --metal-ops token` (the default on Metal) the router, the top-K
selection and the experts all run on the GPU while every chosen expert is resident, so a
whole token is one submission. `--metal-ops layers` routes on the host and submits once
per layer. Both give the CPU's bits.

## What a model must say

The router's behaviour comes from the model's config, with gaps filled by the manifest:

| Config value | Meaning |
|---|---|
| `num_experts` | Experts per layer |
| `num_experts_per_tok` | K |
| `norm_topk_prob` | Renormalise the top-K weights. GGUF files do not always record it; supply it with `--set norm_topk_prob=false` or `true` at import |
| `router_scoring` (manifest) | `softmax` |

## Observing routing

- `qubivex run` prints expert-cache hits, loads and evictions for the run.
- `/metrics` has `qubivex_expert_cache_{hits,loads,evictions}_total` per model.
- `qubivex models inspect <ID>` shows the expert bytes and the largest expert, which is
  the unit the cache moves.
- `qubivex plan <ID>` shows how many experts per layer the cache holds.

## Not supported

- **Sigmoid router scoring.** Parsed, and refused at load.
- **Shared experts** (an expert every token uses beside the routed ones, as in
  Qwen2-MoE and DeepSeek), and layers that mix dense and routed blocks. Refused at
  import with a message naming the feature.
- **Routing with auxiliary inputs,** noisy top-K and other training-time behaviour.
- **Per-token top-K that varies,** and expert capacity limits.
- **Learned pinning of hot experts across runs.** The cache keeps what is hot by use
  (LFRU); it does not persist routing statistics.
