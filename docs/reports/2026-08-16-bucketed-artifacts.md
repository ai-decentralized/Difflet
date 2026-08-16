# Bucketed Compiled Artifacts: K Shapes, One Weight Copy — Report (2026-08-16)

**Branch:** `feat/bucketed-artifacts` (4 commits: `70c044e`, `2154dac`, `ff6ca6b`, `a1e9dfb`)
**Hardware:** trn2.3xlarge — 4 NeuronCores (LNC=2), 96 GiB device memory, tp=4, bf16
**Verified on:** HunyuanVideo, Wan 2.2, FLUX.1-dev, Qwen-Image (all on device, end to end)
**Detailed evidence:** `docs/verification/task{1,1b,2,3}-*.md`

---

## TL;DR

Trainium compiles ahead-of-time, so every request shape used to be its own artifact — and 99.8 %
of an artifact is tp-sharded weights that do **not** depend on shape. We now compile K request
shapes into **one** artifact: K small NEFFs sharing **one** device-resident weight copy, routed
per request by input shape. Measured on HunyuanVideo: serving **two shapes now takes LESS device
memory (50.0 GiB) than serving one shape did before (51.3 GiB)**, and one resident serving worker
answers both shapes in 12–21 s with zero reloads — where the CLI took ~4 min per generation.
Every additional shape costs ~2 GiB of scratch instead of a 27–43 GiB weight copy, per model.

---

## 1. What the feature is

```
before:   artifact(shape A) = NEFF_A (70 MB) + weights (44 GB)
          artifact(shape B) = NEFF_B (70 MB) + weights (44 GB)   ← same bytes, copied

after:    artifact({A,B})   = NEFF_A + NEFF_B + weights (44 GB, once)
                              runtime routes each request to its NEFF by exact input shape
```

Three deliverables, one per commit:

1. **Bucketed compile** (`70c044e`): a generic core mixin (`ShapeBucketedInputGenerator`) turns a
   `--shapes 320x512x61,320x512x33` list into one example-input set per shape; the vendor
   ModelBuilder then compiles one NEFF per shape bound to a single sharded-weight residency.
   Wired for all four models' DiT + VAE components (`a1e9dfb`). Text encoders need nothing —
   their padded sequence length never varies with the video/image shape.
2. **Shape-set identity** (`2154dac`): cache keys and artifact directories are now derived from
   the canonical (deduped, largest-first) shape **set** — pure-hash dirs with an authoritative
   `manifest.json`, plus `difflet cache ls` as the human-readable index. Set order can never
   fork the cache; adding/removing a shape correctly invalidates.
3. **Multi-shape serving** (`ff6ca6b`): `difflet serve --shapes …` runs **one** resident worker
   that owns one weight copy, warms every bucket at startup, serves any member shape, and
   strictly rejects out-of-set shapes with the allowed list.

## 2. Why: the memory wall

Weights depend only on (model, revision, dtype, tp-sharding). The compiled graph additionally
depends on (height, width, frames, batch). The old cache key coupled the two, so every
resolution/frame-count variant duplicated the full weight set on disk **and on device**. On a
96 GiB device where one HunyuanVideo profile is ~51 GiB resident, a second concurrent shape was
simply impossible (2 × 51.3 = 102.6 GiB) — the product could offer exactly one resolution per
box, and every shape switch meant a full worker restart plus a ~40 s weight reload.

## 3. Measured results

### 3.1 Device memory per model (2-shape bucketed artifact, measured via neuron-monitor)

| model | shapes compiled together | shareable weights (measured on disk) | peak device, 2 shapes bucketed (measured) | naive 2 shapes (derived) | saved |
|---|---|---|---|---|---|
| **HunyuanVideo** | 320×512×61 + 320×512×33 | DiT 43.3 + VAE 1.1 GiB | **50.0 GiB** (single-shape baseline: 51.3) | 102.6 GiB → **does not fit** | **52.6 GiB (51 %)** |
| **Wan 2.2 A14B** | 480×832×9 + 480×832×5 | DiT 27.9 + UMT5 10.6 GiB | **41.4 GiB** | ~79.9 GiB | **~38.5 GiB (48 %)** |
| **FLUX.1-dev** | 1024² + 512² | DiT 22.2 + T5 8.9 + CLIP 0.9 + VAE 0.4 GiB | **47.9 GiB** | ~80.2 GiB | **~32.3 GiB (40 %)** |
| **Qwen-Image** | 1024² + 512² | DiT 38.1 GiB | **38.1 GiB** | ~76.2 GiB | **~38.1 GiB (50 %)** |

- "Naive" = the measured bucketed peak plus one more copy of the shareable weights (what a second
  single-shape artifact would add). HunyuanVideo additionally has a directly measured single-shape
  baseline (51.29 GiB), so its row is fully measured.
- The general law: **every additional shape costs one NEFF (tens of MB) + ~2 GiB scratch instead
  of the model's full weight copy** (22–43 GiB). Savings scale linearly with K: at K shapes the
  naive plan needs K weight copies, bucketing needs one.
- HunyuanVideo's bucketed footprint is *below* its old single-shape footprint because disabling
  the weight-layout-optimization pass (see §5) also removes that pass's layout padding.

### 3.2 Capacity on this 96 GiB device (HunyuanVideo)

| plan | shapes resident simultaneously |
|---|---|
| one artifact per shape | **1** (a second does not fit) |
| bucketed | **~15–20** (50 GiB base + ~2.2 GiB per extra shape) |

### 3.3 Serving latency (one resident worker, both shapes, measured)

| request | shape | wall time |
|---|---|---|
| A, first | 320×512×61 | 20.9 s |
| B, first (shape switch, same worker) | 320×512×33 | 12.5 s |
| A, repeat | 320×512×61 | 20.9 s |
| B, repeat | 320×512×33 | 12.5 s |

Repeats equal first requests to the decisecond: startup warmup covers every bucket, and a shape
switch costs **zero** reload. The staged CLI takes ~4 min per generation (per-process weight
reloads) for comparison. Out-of-set shapes get an immediate 400 `profile_mismatch` naming the
allowed set.

### 3.4 Compile cost

Adding the second HunyuanVideo shape to a warm compiler cache took minutes (the bucketed
artifact's `model.pt` grows ~2 MB per extra NEFF); a cold 2-shape FLUX artifact (DiT + T5 + CLIP
+ VAE) compiled in 20 min. Disk stays deduplicated regardless: all artifacts hardlink their
shards from the shape-free shared weight store.

### 3.5 Correctness (the part that took the longest)

- HunyuanVideo bucketed outputs sit inside the normal bf16 envelope of a **CPU fp32 reference**
  (max|Δ| 0.065 at the DiT level, identical to each shape's standalone baseline), with no
  spatial error concentration, and all models' bucketed generations are **bit-deterministic**
  across same-seed reruns.
- Wan/Flux/Qwen: correct output dimensions at every member shape, byte-identical same-seed
  reruns, router maps showing exactly one NEFF per shape over one weight set.

## 4. How it works (one paragraph)

The vendor runtime already supported this: passing K example-input sets to `ModelBuilder.add`
compiles K NEFFs that the runtime binds to one weight object, dispatching by exact input-shape
signature. Our contribution is the difflet-side factorization: a per-model
`example_inputs_for_shape()` hook (only the latent/packed tensor varies; Flux also varies its
host-computed RoPE input; Qwen selects per-bucket static-RoPE tables at trace time), a canonical
shape order (largest first = layout-priority bucket), shape-set cache identity, and the serving
profile/validator work. Weight sharing itself needs no lookup: it is structural within an
artifact, and across artifacts the disk store keys shards by exactly the weights' true
dependency set — (source checkpoint, dtype, tp, world_size, cp/cfg/sp flags) — with **no shape**.

## 5. Two bugs found on the way (both worth knowing about)

1. **Vendor weight-layout-optimization (WLO) corrupts non-priority buckets.** With WLO on, bucket
   B's outputs were nondeterministic with errors 37× the bf16 envelope, localized to the first
   ~96 sequence tokens — the graph reads uninitialized device memory. Root-caused by a
   determinism probe + CPU-fp32 reference + a no-WLO isolation build (which is fully clean).
   Mitigation shipped: multi-bucket compiles force `priority_model_idx=None` (no WLO); cost is
   +38 % per DiT step ≈ 5–7 % e2e, accepted for correctness. Single-shape compiles keep WLO.
   TODO: file the minimal repro with the Neuron team (NxD 0.19.28492 / neuronx-cc 2.26).
2. **Concurrent compiles deleted each other's scratch.** `ModelBuilder.trace()` rmtree-s its
   workdir, which was keyed by component name only (`/tmp/nxd_model/transformer`) — a HunyuanVideo
   compile killed a running Wan compile mid-flight. Fixed: every stage gets a unique scratch dir.

## 6. How to use

```bash
# Compile once for a shape set (video: HxWxF; image: HxW)
difflet compile --model-id hunyuanvideo-community/HunyuanVideo --tp-degree 4 \
  --shapes 320x512x61,320x512x33

# Generate at any member shape (out-of-set fails fast with the allowed list)
difflet generate --model-id ... --shapes 320x512x61,320x512x33 \
  --height 320 --width 512 --num-frames 33 --prompt "..." --seed 42

# Serve every shape from ONE resident worker / one weight copy
difflet serve --model-id hunyuanvideo-community/HunyuanVideo \
  --shapes 320x512x61,320x512x33 --port 8091

# Inspect what's compiled
difflet cache ls
```

## 7. Limitations & next steps

- Frames/height/width are all sequence-length dimensions on an AOT target: every distinct
  (h, w, f) needs its own NEFF — bucketing makes that cheap, not free. The open research lever is
  **token-count canonicalization** (equal-token shapes sharing one NEFF, e.g. 320×512 vs
  512×320); deliberately deferred — its precondition (host-side position encodings) is unverified
  and will be settled by NEFF-hash measurement before any design.
- Multi-shape serving is HunyuanVideo-only for now; other models' serving adapters are a
  mechanical extension. Multi-worker shape routing is deferred until a profile outgrows one
  bucket's memory budget (~15–20 shapes).
- Out of scope, guarded with clear errors: HunyuanVideo 1.5 / segmented runtimes / LTX-2;
  batch bucketing; cp>1 with multi-shape. Re-enable WLO when the vendor bug is fixed to reclaim
  the ~5–7 % e2e cost.
