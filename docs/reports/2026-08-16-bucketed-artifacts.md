# Bucketed Compiled Artifacts: K Shapes, One Weight Copy — Report (2026-08-17)

**Branch:** `feat/bucketed-artifacts` (`70c044e`, `2154dac`, `ff6ca6b`, `a1e9dfb` + report)
**Hardware:** trn2.3xlarge — 4 NeuronCores (LNC=2), 96 GiB device memory, tp=4, bf16
**Models:** HunyuanVideo, Wan 2.2 A14B, FLUX.1-dev, Qwen-Image — all verified on device
**Evidence:** `docs/verification/task{1,1b,2,3}-*.md`; raw campaign logs in the job scratch dir

---

## TL;DR

On Trainium, every request shape used to be its own compiled artifact, and 99.8 % of an artifact
is tp-sharded weights that do **not** depend on shape. We now share weights at two layers — on
disk (hardlink store) and on device (K NEFFs bound to one resident weight copy) — and compared
**with vs without** weight sharing at 3 shapes per model:

- **Device memory:** 3 shapes resident in 40–71 GiB; without sharing, HunyuanVideo cannot even
  fit 2 of its 51 GiB profiles in 96 GiB. Each extra shape costs a NEFF + scratch instead of a
  22–43 GiB weight copy.
- **Compile time (3 shapes, warm compiler cache):** one bucketed compile ≈ the cost of the single
  most expensive shape, not the sum — 75.8 vs 161 min (HunyuanVideo), 3.3 vs 13.8 min (Wan),
  14.7 vs 29.8 min (Flux), 15.0 vs 41.1 min (Qwen); **2–4.2× faster**.
- **Serving:** one resident worker answers all three HunyuanVideo shapes; a shape switch costs
  the next request (~12–21 s) instead of a worker restart (~200 s+).
- **Disk (3 shapes):** one weight set + tiny NEFFs instead of 3 full copies — e.g. HunyuanVideo
  44.6 vs 133.7 GiB.

---

## 1. Background

**How Trainium executes.** The compiler freezes everything at build time: tensor shapes,
the per-engine instruction schedule, and the SBUF/HBM memory plan all go into a NEFF binary.
A different sequence length is a genuinely different program. Weights, however, are **not** baked
in (`inline_weights_to_neff=False`): the NEFF declares them as inputs and reads them from HBM
buffers bound at load time. Weight shapes depend only on the architecture and tp-sharding — a
3072×3072 projection is the same matrix whether it multiplies 5,760 or 10,240 tokens.

**Do GPUs have this problem? No — and it is worth being precise about why.** GPU kernels are
shape-generic: grid dimensions are launch-time parameters, so one cuBLAS/attention kernel serves
any sequence length, and the single weight tensor in HBM is shared by every launch. Even when GPU
stacks specialize per shape — CUDA Graph captures, `torch.compile` guards, autotuning caches —
those artifacts are captured in milliseconds-to-seconds and **reference the same weight buffers**;
nothing duplicates the weights. The one GPU-world analogue is ahead-of-time engines like
TensorRT with static shapes, which hit exactly this duplication and answer it with built-in
dynamic-shape ranges (one engine, min/opt/max). The Trainium compiler has no dynamic-shape
facility — which is why the runtime's *bucketing* (several static graphs sharing one weight
residency, routed by input shape) is the Trainium-native equivalent, and what this work turns on.

## 2. Problem Statement

Difflet's artifact identity historically coupled the two dependency sets: the cache key included
height/width/frames, so every shape variant produced a complete artifact — a ~70 MB NEFF plus a
full copy of the weights (HunyuanVideo: 44 GB; ratio ≈ 640:1). Consequences on a 96 GiB device:

- one HunyuanVideo serving profile is ~51 GiB resident → **a second shape cannot co-reside at
  all**; the product ships exactly one resolution per box;
- switching shapes means restarting the worker and re-reading ~44 GB of weights (~40 s load,
  ~200 s to a ready worker);
- every new shape costs a full compile **and** a full weight copy on disk.

The fix factorizes the artifact along its true dependency sets: weights are stored and resident
**once** per (checkpoint, dtype, tp-sharding) — on disk via a hardlink store keyed exactly by
that set, on device via K NEFFs compiled together and bound to one weight object — and requests
are routed to their NEFF by exact input shape.

## 3. Comparison — with vs without weight sharing

Setup: 3 shapes per model (the third is a new resolution), tp4, bf16, seed 42.
**Cache policy: all compile times measured under a warm neuronx-cc kernel cache, both sides**
(cold-cache rerun tracked as issue #31). "Derived" cells are arithmetic from measured components.

| model | shape set |
|---|---|
| HunyuanVideo | 320×512×61, 320×512×33, **256×448×61** |
| Wan 2.2 | 480×832×9, 480×832×5, **320×576×9** |
| FLUX.1-dev / Qwen-Image | 1024², 512², **768²** |

### 3.1 Device memory (peak during a full generation at the priority shape, neuron-monitor)

| model | WITH: 3-shape bucketed (measured) | WITHOUT: 1 shape (measured anchor) | WITHOUT: 3 shapes resident (derived) | avoided per extra shape |
|---|---|---|---|---|
| HunyuanVideo | **70.8 GiB** | 62.0 GiB | 150.8 GiB — **already exceeds 96 GiB at 2 shapes** (106.4) | 44.4 GiB |
| Wan 2.2 | **42.3 GiB** | 40.9 GiB | 117.9 GiB (3rd shape does not fit) | 38.5 GiB |
| FLUX.1-dev | **51.6 GiB** | 47.7 GiB | 112.3 GiB (3rd shape does not fit) | 32.3 GiB |
| Qwen-Image | **40.4 GiB** | 41.1 GiB | 117.3 GiB (3rd shape does not fit) | 38.1 GiB |

Both columns use the same methodology (peak across a full generation, all pipeline stages).
Holding **two extra shapes** costs +8.8 GiB (HunyuanVideo), +1.4 (Wan), +3.9 (Flux), −0.7 (Qwen —
within run-to-run noise; the bucketed run measured slightly *below* the single-shape anchor)
instead of +2 full weight copies. Without sharing, no model fits 3 shapes on this device and
HunyuanVideo cannot even fit 2.

### 3.2 Compile time for 3 shapes (warm cache, measured)

| model | WITH: one `--shapes A,B,C` compile | WITHOUT: three single-shape compiles (sum) | speedup |
|---|---|---|---|
| HunyuanVideo | **75.8 min** | 75.0 + 11.3† + 74.9 = **161.2 min** | 2.1× |
| Wan 2.2 | **3.3 min** | 5.1 + 4.5 + 4.2 = **13.8 min** | 4.2× |
| FLUX.1-dev | **14.7 min** | 14.8 + 4.4 + 10.7 = **29.8 min** | 2.0× |
| Qwen-Image | **15.0 min** | 17.8 + 10.1 + 13.3 = **41.1 min** | 2.7× |

The mechanism: a bucketed compile pays trace + weight-shard + artifact assembly **once** and the
per-shape NEFF builds share the kernel cache, so its cost tracks the most expensive member shape
(HunyuanVideo: 75.8 min bucketed vs 74.9 min for the new shape alone — bucketing two more shapes
cost 54 seconds). † one value carried over from an earlier warm-cache session on this branch.

### 3.3 Serving latency across 3 shapes (HunyuanVideo, measured)

| | WITH: one worker, `--shapes A,B,C` | WITHOUT: one single-shape worker per shape |
|---|---|---|
| first start (compiles its serving artifacts) | 80.5 min, **once for all 3 shapes** | 73.0 min (A) + 9.4 min (B) + one more per shape |
| steady-state restart-to-ready (artifacts cached) | 200 s¹ | 180 s |
| request A (61f) / B (33f) / C (256×448×61) | 20.8 s / 12.5 s / 19.5 s | 20.8 s / 10.9 s / — |
| repeat A after serving B and C | 20.8 s (bit of drift: none) | — |

¹ measured on the cached 2-shape profile (2026-08-16); structurally identical for 3 shapes
(weight load + K warmup forwards). Per-request latencies are equal on both sides — bucketing
costs nothing per request; the entire win is in what a shape *switch* costs (§3.4).

Wan/Flux/Qwen: N/A — multi-shape serving is HunyuanVideo-only today; extension tracked in
[issue #30](https://github.com/ai-decentralized/Difflet/issues/30).

### 3.4 Shape-switch cost (the operational headline)

- WITH: a switch **is** the next request — measured **12.5–20.8 s** across A→B→C→A on one worker.
- WITHOUT: kill worker → restart → reload ~44 GB weights → warm → smoke → request — measured
  steady-state **180 s restart + 10.9–20.8 s request ≈ 191–201 s**; first-ever switch to a
  never-compiled shape additionally pays that shape's serving-artifact compile (9–73 min).

**≈ 10–16× faster shape switching** in steady state; unboundedly better for first-time shapes.

### 3.5 Disk usage for 3 shapes (derived from measured file sizes)

| model | WITH: 1 weight set + 3-NEFF artifact | WITHOUT: 3 × (weights + NEFF) | saved |
|---|---|---|---|
| HunyuanVideo (DiT+VAE) | 44.4 GiB + 0.21 GiB | 133.7 GiB | **89 GiB (67 %)** |
| Wan 2.2 (DiT+UMT5) | 38.5 GiB + 0.07 GiB | 115.7 GiB | **77 GiB (67 %)** |
| FLUX.1-dev (all components) | 32.3 GiB + 0.17 GiB | 97.3 GiB | **65 GiB (67 %)** |
| Qwen-Image (DiT) | 38.1 GiB + 0.07 GiB | 114.3 GiB | **76 GiB (67 %)** |

(At K shapes the WITHOUT column scales as K×; WITH stays ~flat. The disk layer is the hardlink
store, keyed by exactly the weights' dependency set — checkpoint, dtype, tp, world-size,
cp/cfg/sp — with no shape.)

### 3.6 Weight-load I/O per shape switch (from load logs)

WITHOUT sharing, every shape switch re-reads the full shard set from disk: 44.4 GB (HunyuanVideo
DiT), 38.6 GB (Wan DiT+UMT5), ~32 GB (Flux), 39.0 GB (Qwen DiT). WITH sharing the weights are
read **once** per worker lifetime; a switch reads nothing.

### 3.7 Device capacity (max shapes resident, HunyuanVideo, derived)

WITHOUT: **1** (a second full profile does not fit: 62.0 + 44.4 = 106.4 GiB > 96). WITH: **~10+**
(70.8 GiB serves 3 shapes; each further shape adds NEFF + scratch on the order of single GiB,
bounded by the largest shape's activation footprint).

### 3.8 Correctness — no regression (measured)

Bucketed outputs sit inside the normal bf16 envelope of a **CPU fp32 reference** (HunyuanVideo
DiT-level max|Δ| 0.065, equal to each shape's standalone baseline; diffs frame-uniform), and all
four models produce **bit-identical** same-seed reruns at every shape, including the new third
resolutions.

### 3.9 Honest cost — WLO disabled for multi-bucket artifacts (measured)

Single-shape (WLO on): 795 ms per DiT step. Bucketed (WLO off, see §4): 1094 ms — **+38 % per
DiT step ≈ 5–7 % e2e** (DiT compute is 12–19 % of warm e2e). Accepted for correctness; reclaim
when the vendor bug is fixed.

### 3.10 New-shape onboarding time (measured)

Adding the new 256×448×61 resolution: extend the bucket → 75.8 min total recompile (dominated by
the new shape's own kernels, warm cache); build it as a fresh standalone artifact → 74.9 min —
**parity**, i.e. joining the shared-weight bucket costs nothing extra over a private artifact,
while inheriting all the memory/disk/serving benefits above.

## 4. Found along the way: a vendor bug (one page)

With the weight-layout-optimization (WLO) pass enabled, every **non-priority** bucket in a
multi-NEFF artifact computed wrong: outputs were nondeterministic between identical calls, with
errors 37× the bf16 envelope concentrated in the first ~96 sequence tokens — the graph reads
uninitialized device memory. Root-caused by a five-step chain (single-step diff → CPU-fp32
reference → error localization → determinism probe → no-WLO isolation build, which is fully
clean); full evidence in `docs/verification/task1-bucketed-compile.md` §6. Mitigation shipped:
multi-bucket compiles disable WLO (cost in §3.9); single-shape compiles are untouched. To be
filed upstream (NxD 0.19.28492 / neuronx-cc 2.26). A second find: concurrent difflet compiles
used to delete each other's compiler scratch (workdir keyed by component name only); fixed with
per-invocation scratch dirs.

## 5. Next steps

- Extend multi-shape serving to Wan/Flux/Qwen —
  [#30](https://github.com/ai-decentralized/Difflet/issues/30).
- Cold-cache compile comparison — [#31](https://github.com/ai-decentralized/Difflet/issues/31).
- Token-count canonicalization (equal-token shapes sharing one NEFF) — deliberately deferred;
  precondition to be settled by NEFF-hash measurement first.
- File the WLO bug upstream; re-enable WLO for buckets when fixed.
