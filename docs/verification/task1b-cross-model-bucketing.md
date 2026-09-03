# Task 1b Verification — Bucketing generalized: Wan / Flux / Qwen-Image

Branch: `feat/bucketed-artifacts` · Host: trn2.3xlarge tp4 bf16 · Scratch `$LOG = /home/ubuntu/.claude/jobs/e22aa8d9/tmp/`

The generic mechanism (`ShapeBucketedInputGenerator`, Task 1) is wired into every supported
model; this doc records the per-model device verification. HunyuanVideo's deep verification
(CPU-fp32 reference, WLO root-cause) is in `task1-bucketed-compile.md` — the cross-model bar here
is: one artifact, K routed NEFFs, ONE weight set, e2e generation at every member shape,
deterministic outputs, and a **frame-uniform** diff vs single-shape baselines (the WLO corruption
signature was a frame-0 concentration; uniformity is the discriminator).

## Per-model wiring notes

| model | what varies per bucket | special handling |
|---|---|---|
| HunyuanVideo DiT | latent (f,h,w) | none — RoPE derived in-graph from input shape (Task 1) |
| HunyuanVideo VAE | nothing (fixed tile) | K shapes dedupe to 1 bucket |
| Wan DiT | latent (f,h,w) | config `num_frames` semantics are LATENT frames; app converts pixel→latent per shape |
| Wan VAE | full latent per shape | genuinely K buckets; `--host-vae` remains the escape for long clips |
| Flux DiT | packed `num_patches` + the **host-computed RoPE input** `[num_patches+512, head_dim, 2]` | per-bucket RoPE example; runtime RoPE follows request `img_ids` (cache reset per generation) |
| Flux VAE | full latent (h/8, w/8) | K buckets via the same mixin |
| Qwen-Image DiT | packed `image_seq_len` | static-RoPE tables are per-bucket (`_StaticQwenImageRealRope` holds one (cos,sin) per packed shape, selected at trace time from the concrete seq len) |
| Qwen VAE (Wan decoder reused) | (h, w, 1) single-frame latents | image shapes mapped to 1-frame video shapes |

CLI: `--shapes` is wired for all four models (staged orchestrators for HunyuanVideo/Wan/Qwen,
DiffletPipeline path for Flux), with fail-fast membership checks in `generate` and Task-2 hash-dir
manifests covering the shape set.

## Wan (VERIFIED on device 2026-08-16 04:16 UTC)

Shapes: **9f = 480×832×9** (priority) and **5f = 480×832×5**, tp4, steps=2, seed 42, host-VAE decode.
Repro: `bash $LOG/wan_chain.sh` (runs baselines + bucketed + generations + router dump).

- Compiles (warm NCC cache, new hash-dir layout): base9 5 min, base5 4.5 min, bucketed 3 min.
- Bucketed artifact `~/.cache/difflet/wan_transformer/0da138c6be83d5ae/`:
  router shows **2 signatures** — latent `[1,16,3,60,104]`→bucket 0, `[1,16,2,60,104]`→bucket 1 —
  over **one** 4-shard weight set (hardlinked into the shared store).
- e2e outputs vs single-shape baselines (WLO) — small and **frame-uniform**:
  - 9f: max|Δ|=0.134, mean=0.0033, per-frame mean 0.0027–0.0036 (flat)
  - 5f: max|Δ|=0.120, mean=0.0033, per-frame mean 0.0026–0.0039 (flat)
- Determinism: bucketed 9f rerun with the same seed → tensor sha256 **identical**
  (`aab91bf5…423a`).
- Output tensors: `$LOG/wan_out_{base9,base5,bkt9,bkt5}.pt` (+ `_rep`), logs `$LOG/wan_*.log`.

Note: the first attempt of this chain (2026-08-15) died mid-compile from the shared-scratch
concurrency bug fixed in Task 2 (`runner.py` unique scratch dirs) — the rerun on the fixed code
passed cleanly.

## Flux (VERIFIED on device 2026-08-16 04:43 UTC)

Shapes: **1024×1024** (priority) + **512×512**, tp4, steps=4, seed 42.
Repro: `bash $LOG/flux_qwen_chain.sh` (flux section); outputs `$LOG/fq_flux_{1024,512}[_rep].png`.

- Bucketed compile (DiT + T5 + CLIP + VAE) in **20 min** cold; artifact
  `~/.cache/difflet/flux/372c8138ff876c68/` (schema-5 manifest with both shapes).
- Router maps: transformer 2 signatures — packed `[1,4096,64]` w/ RoPE `[4608,128,2]` → bucket 0,
  `[1,1024,64]` w/ RoPE `[1536,128,2]` → bucket 1 (the host-computed per-shape RoPE input works);
  decoder 2 signatures (`[1,16,128,128]` / `[1,16,64,64]`); CLIP/T5 1 each (shape-independent).
- Generations: both shapes render at the CORRECT dimensions (1024×1024 / 512×512 PNG) and
  same-seed reruns are **byte-identical** (no WLO-class nondeterminism).

## Qwen-Image (VERIFIED on device 2026-08-16 05:35 UTC)

Shapes: **1024×1024** (priority) + **512×512**, tp4, steps=4, seed 42.
Outputs `$LOG/fq_qwen_{1024,512}[_rep].png`.

- Bucketed DiT compile 21 min cold; artifact `~/.cache/difflet/qwen_image_dit/a4c671e252b424f3/`;
  router: `[1,4096,64]`→bucket 0, `[1,1024,64]`→bucket 1 (per-bucket static-RoPE tables selected
  at trace time). Qwen VAE (reused Wan decoder) bucketed via 1-frame shapes.
- Generations: correct dimensions at both shapes; same-seed reruns **byte-identical**.
- Two integration bugs found & fixed during this verification:
  1. the app pinned its pipeline shape to the LARGEST bucket, so a 512 request generated 1024
     latents — the app now keeps the REQUESTED shape for the pipeline and hands the shape set
     only to the compile config;
  2. `validate_qwen_image_dit_inputs` exact-matched the config's single `image_seq_len` — now a
     membership check over the compiled bucket seq lens (same pattern as HunyuanVideo).
  Plus the CLI VAE unpack assumed square grids (`sqrt(seq)`); it now derives the grid from the
  request shape.

## Out of scope (guarded)

HunyuanVideo 1.5 (`shapes` K>1 raises NotImplementedError), segmented runtimes, LTX-2. TeaCache
probes stay pinned to the largest shape. cp>1 / cfg-parallel with multi-shape untested (serving
uses cp=1).
