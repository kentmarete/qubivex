# Multimodal runtime

- State (2026-11-15): **design only. Nothing here is implemented.** Qubivex takes text
  only; `image_url` parts are refused with a 400 (`qubivex-api/src/openai.rs`).
- Plan and acceptance criteria: [ROADMAP_GENERAL_PURPOSE_RUNTIME.md](ROADMAP_GENERAL_PURPOSE_RUNTIME.md)
  §2 D and §3 steps 7–10, 13. An ADR records each part before it is built.

## Goal

A reusable multimodal architecture: images, audio and video enter one input
representation, are processed by registered, configurable components, and reach the
same decoder that runs text. A vision-language model is a **configuration** of those
components. Qwen2.5-VL-7B is the first acceptance target; a second, structurally
different vision-language model is the proof that the components are general (no
family-specific branches).

## 1. Input representation

A prompt becomes one sequence of **items**:

| Item | Contents |
|---|---|
| Token run | Token ids, embedded through the model's table |
| Embedding span | `n × hidden` embeddings produced by an encoder and projector, a **content digest**, the modality, and its metadata (grid, timestamps) |

Each item also carries its **positions** under the model's position scheme (§3). The
decoder's forward pass takes items instead of only token ids; a sequence of token runs
alone is today's input and keeps today's bits.

The KV cache length and the position are separate quantities: after a span, the next
token's position is whatever the scheme says, not the count of items so far.

## 2. Components (registered, configured)

All components are registered in the Architecture Registry with an id, a version and a
config schema, like decoder architectures (ADR-003), and are cross-validated against
their tensors at import.

### Preprocessor (per modality)
A declarative pipeline read from the model's processor config:

| Stage | Options |
|---|---|
| Decode | image: PNG, JPEG, WebP, GIF (first frame); audio: WAV, FLAC, MP3, Ogg; video: frames from a container (later) |
| Validate | byte and pixel/sample limits, checked from headers before decoding |
| Colour / channels | to RGB with alpha composited on a configured background; audio to mono |
| Resize / resample | fixed size; aspect-preserving to a multiple of N within min/max pixels; tiling; filter (bicubic, bilinear, Lanczos) matching the reference library; audio resampling to the model rate |
| Normalise | rescale, per-channel mean/std |
| Featurise | image patches (`patch`, `temporal_patch`, ordering); log-mel spectrogram (window, hop, mel bins) |

### Encoder
A vision (or audio) transformer from configuration: patch or convolutional embedding;
position encoding (learned absolute, interpolated, or 2-D RoPE); norm (LayerNorm or
RMSNorm); attention pattern (full, windowed, which blocks are full); MLP (GELU or SwiGLU,
biases); output taps (last layer or a given one).

### Projector
Linear, MLP (activation configurable), or spatial merge followed by MLP.

### Placeholder expansion
Declared in the manifest: the start, pad and end tokens of each modality and the number
of pads per span (a function of the span's grid). The chat template places the media;
the runtime expands it.

## 3. Position schemes

| Scheme | Used by |
|---|---|
| 1-D RoPE | today's text models |
| Sectioned multi-axis RoPE (axes and section sizes from config) | Qwen2-VL / Qwen2.5-VL (M-RoPE: t, h, w with sections 16, 24, 24) |
| None inside spans (span positions advance as tokens) | many LLaVA-style models |

For text-only prompts every scheme reduces to 1-D RoPE with the same bits.

## 4. Rules

- **Prefix reuse is keyed by content.** A retained KV entry's key includes each span's
  digest of its preprocessed input. Same text, different image: no reuse. This is a
  required test, because otherwise identical placeholder ids would let one request reuse
  another's image KV.
- **Batch invariance holds** (ADR-017): a sequence with spans gives the same bits alone
  and in a mixed batch.
- **Untrusted input**: limits checked before decoding; decoding on a bounded pool, off
  the gateway's async runtime; fuzz targets for every decoder.
- **No remote media fetching** by default (SSRF): `data:` URIs and uploaded bytes only.
- **Memory**: encoder weights and activation peaks are planned like decoder ones;
  encoder attention is computed in blocks (full-attention blocks are quadratic in
  patches).
- **Embeddable**: `GenerationRequest` carries media, so an application gets multimodal
  input without the server.
- **No advertising before end to end**: a modality is listed as supported only when the
  whole path (decode → encoder → decoder → API) passes its reference tests.

## 5. Parity testing

- Encoder parity is tested with the reference's own preprocessed tensors as input, so it
  is separate from preprocessing parity.
- Preprocessing parity is tested against the reference processor within a stated
  tolerance per stage.
- End to end: logits and greedy tokens for fixed images and prompts.
- The reference oracle for Qwen2.5-VL drives transformers module by module so a 7B model
  runs in f32 within 18 GB (draft script on branch `qwen25-vl-adr`,
  `fixtures/oracle/qwen25_vl.py`; to be adapted when the vision work starts).

## 6. Qwen2.5-VL-7B as configuration

Checked against the checkpoint at revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`
(`config.json`, `preprocessor_config.json`, the safetensors index) and transformers
5.22.1's implementation:

| Component | Configuration |
|---|---|
| Decoder | `transformer_dense`: hidden 3584, 28 layers, 28 heads / 4 KV heads, FFN 18944 SwiGLU, RMSNorm 1e-6, **q/k/v biases**, untied embeddings, vocab 152064 |
| Position scheme | sectioned RoPE, θ 1e6, axes (t, h, w), sections (16, 24, 24), NeoX layout |
| Preprocessor | RGB on white; aspect-preserving resize to multiples of 28 within 3136…12 845 056 pixels; PIL bicubic; rescale 1/255; CLIP mean/std; 14 × 14 patches over 2 frames (a still image is duplicated) |
| Encoder | 32 blocks, hidden 1280, 16 heads, RMSNorm 1e-6, qkv and proj biases, SwiGLU 3420 with biases, 2-D RoPE (θ 10 000), window attention of 4 × 4 merged patches except blocks 7, 15, 23, 31 (full) |
| Projector | RMSNorm, 2 × 2 spatial merge (5120), Linear → GELU (erf) → Linear to 3584 |
| Placeholders | `<|vision_start|>` 151652, `<|image_pad|>` 151655 × (h·w/4), `<|vision_end|>` 151653 |
| Weights | text from GGUF Q4_K_M (what Ollama serves; fits 18 GB) or bf16 SafeTensors; vision from bf16 SafeTensors (`visual.*`) |

## 7. Audio and video (design to be completed before building)

- **Audio**: decode and resample to the model's rate; log-mel features per the model's
  config; an encoder component (conv front-end + transformer); a projector; spans with
  1-D positions. Speech output (a decoder producing audio tokens) is out of scope until
  an architecture needing it is chosen.
- **Video**: decode to frames; sample at a configured rate, keeping timestamps; frames go
  through the vision path with `temporal_patch` > 1; temporal positions per the scheme
  (Qwen2.5-VL scales the time axis by `tokens_per_second` and frame timing).
