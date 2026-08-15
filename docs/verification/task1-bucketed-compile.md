# Task 1 Verification — Bucketed compile: K shapes, one artifact, one weight copy

Branch: `feat/bucketed-artifacts` · Host: trn2.3xlarge (4 NeuronCores LNC=2, 96 GiB device memory)
Model: `hunyuanvideo-community/HunyuanVideo`, tp=4, bf16 · SDK: neuronx_distributed 0.19.28492, neuronx-cc 2.26
Shapes: **A = 320×512×61** (largest), **B = 320×512×33**
Fixed generation params: prompt `"a red cube on a white table"`, seed 42, steps 4.
All timestamps 2026-08-15 UTC. Scratch artifacts in `/home/ubuntu/.claude/jobs/e22aa8d9/tmp/` (`$LOG` below).

## Verdict summary

| claim | status |
|---|---|
| One artifact holds K NEFFs + ONE weight copy | ✅ (fs layout + router map, §2) |
| Both shapes generate correctly from one artifact | ✅ within healthy bf16 envelope vs CPU fp32 reference (§3, §6) |
| Second shape's marginal device memory ≈ NEFF+scratch, not a weight copy | ✅ bucketed 2-shape resident = **50.04 GiB**, LESS than one single-shape artifact (51.29 GiB) (§4) |
| Out-of-set shape strictly rejected | ✅ instant, clear error, no device work (§5) |
| K=1 path unchanged | ✅ unit tests + full suite green (§1) |
| **Found & mitigated a vendor bug**: WLO corrupts non-priority buckets | ✅ root-caused by measurement; K>1 now compiles without weight-layout optimization (§6) — perf cost ≈ +38 % per DiT step ≈ 5–7 % e2e (§7) |

## What changed (code)

- `difflet/backends/trainium/core/bucketing.py` (new): `canonicalize_shapes` (dedupe + sort descending
  by (h·w·f, h, w, f) — largest first), `resolve_compile_shapes`, `dedupe_example_inputs`,
  `ShapeBucketedInputGenerator` mixin. K example-input sets flow through the existing
  `ModelBuilder.add(example_inputs=...)` seam; NxD derives bucket_degree=K and binds one weight
  residency to all buckets; runtime routes by exact input shape.
- `core/application_base.py`: when a wrapper emits >1 example set, `priority_model_idx` is forced to
  None — disables the vendor weight-layout-optimization pass, which corrupts non-priority buckets (§6).
- HunyuanVideo backbone/VAE adopt the mixin (VAE's fixed tile dedupes K→1 bucket); per-shape config
  validation; `NeuronHunyuanVideoApplication(shapes=[...])`; DiT input validation = set membership.
- Wan backbone/VAE + application (`shapes` kwarg; backbone shapes carry LATENT frames), Flux backbone
  (per-shape num_patches + host-RoPE example), Qwen-Image transformer (per-bucket static-RoPE tables
  selected at trace time by concrete seq len) — code-wired + unit-tested; **device verification
  pending** (their weights finished downloading late; compiles are hours each — Task 1b follow-up).
- CLI `--shapes HxWxF,...` (HxW for image models); bucketed dirs `..._bkt<K>-<sethash6>` (interim
  naming until Task 2); fail-fast membership check in orchestrator `generate()` before any stage runs.

## 1. Unit tests (no device)

```bash
cd /home/ubuntu/Difflet
PATH=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:$PATH \
  /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python -m pytest tests/unit/backends/test_bucketing.py -q   # 17 passed
# full suite: 1523 passed, 25 skipped (1 deselected failure pre-exists on pristine main — verified in a temp worktree)
```

## 2. Artifact layout: K NEFFs, one weight set

Compile (baseA 77 min, baseB 11 min, bucketed 2 h 50 min):
```bash
difflet compile --model-id hunyuanvideo-community/HunyuanVideo --tp-degree 4 --height 320 --width 512 --num-frames 61
difflet compile ... --num-frames 33
difflet compile ... --shapes 320x512x61,320x512x33
```
Under `~/.cache/difflet/hunyuan_video_dit_tp4cp1_h320w512f61_bkt2-ed786a/`:
- `transformer/model.pt` ≈ 100 MB (single-shape: 72.5 MB) — two NEFFs in one artifact.
- `transformer/weights/` = ONE set: 4 × 11,634,459,808 B shards, hardlinked across all artifacts
  via the shared-weights store.
- Router map (`python $LOG/inspect_buckets.py <artifact>/transformer`):
  `[1,16,16,40,64]→bucket 0 (A)`, `[1,16,9,40,64]→bucket 1 (B)`; VAE: 1 signature (tile).

## 3. End-to-end outputs (4 denoise steps + VAE decode)

Final state (after the §6 fix; artifact = no-WLO transformer + VAE):

| run | artifact | shape | sha256 (tensor bytes) | vs baseline |
|---|---|---|---|---|
| baseA | single f61 (WLO) | A | `2f714ebc…ca0` | — |
| baseB | single f33 (WLO) | B | `214191b2…7d2` | — |
| fixed bktA | bucketed no-WLO | A | `a9421aa1…76e` | max\|Δ\|=0.066, mean=0.0016, frame0_mean=0.0017 ✅ |
| fixed bktB | bucketed no-WLO | B | `662e0fc9…278` | max\|Δ\|=0.055, mean=0.0021, frame0_mean=0.0022 ✅ |

Baselines keep WLO, the bucketed artifact doesn't (§6), so bit-identity is not expected; the
differences are uniform bf16 noise with NO frame-0 concentration (contrast §6's corrupted state:
frame0_mean was 4× the rest). Ground truth: at the single-DiT-step level, the fixed bucket-B output
sits at max|Δ|=0.065 from a CPU fp32 reference — the SAME envelope as its standalone baseline.

Repro: `bash $LOG/finalize_task1.sh` (idempotent; skips the swap if done) or any single run:
```bash
difflet generate --model-id hunyuanvideo-community/HunyuanVideo --tp-degree 4 \
  --shapes 320x512x61,320x512x33 --height 320 --width 512 --num-frames 33 \
  --prompt "a red cube on a white table" --seed 42 --steps 4 --output /tmp/b.pt
python $LOG/hash_pt.py /tmp/b.pt
```

## 4. Device memory (the core claim)

`neuron-monitor` during a resident hold; parse = `$LOG/verify_chain3.sh` tail:

| configuration | resident device memory |
|---|---|
| single-shape artifact (A, WLO) | 51.29 GiB |
| bucketed 2-shape (WLO build) | 53.51 GiB |
| **bucketed 2-shape (no-WLO, shipped)** | **50.04 GiB** |

Two shapes resident cost LESS than one WLO single-shape artifact (WLO-transformed weights carry
layout overhead). A second single-shape artifact would add ~51 GiB and cannot co-reside with
headroom on 96 GiB. This satisfies the memo §4 measurement requirement.

## 5. Negative test — strict rejection

Out-of-set request exits in <5 s before any stage/device work with:
`request shape 256x448x61 is not in the compiled bucket set ['320x512x61', '320x512x33']…`

## 6. Vendor bug found: WLO corrupts non-priority buckets (root-caused, mitigated)

Chain of evidence (scripts in `$LOG`, all re-runnable):
1. e2e B from the WLO bucketed build differed from baseline with diff 4× concentrated in video frame 0.
2. Single DiT step: bucket A bit-identical across artifacts; bucket B max|Δ|=2.40.
3. CPU fp32 reference (original diffusers DiT): baseB NEFF max|Δ|=0.065 (healthy bf16) vs
   WLO-bucketed-B max|Δ|=2.38 — 37× the baseline deviation. Not layout noise.
4. Localization: ALL |Δ|>0.5 elements in latent frame 0, rows 0–5 (first ~96 sequence tokens).
5. Determinism probe: WLO bucket-B forward twice in one process → different outputs (bucket A fully
   deterministic) ⇒ the non-priority bucket's graph reads uninitialized device memory.
6. Isolation: recompile with `priority_model_idx=None` (no WLO) → bucket B deterministic AND
   max|Δ|=0.0648 vs fp32 ref (= baseline envelope, frame-0 clean). **WLO is the cause.**

Mitigation shipped in `application_base.get_builder()`: any multi-bucket add compiles with
`priority_model_idx=None`. Single-bucket compiles keep WLO (zero regression). TODO: file the minimal
repro with the Neuron team (2-bucket HunyuanVideo DiT, tp4, NxD 0.19.28492 / neuronx-cc 2.26).

Note on the current bucketed artifact: its `transformer/` is the no-WLO build (`$LOG/compile_nowlo.py`),
swapped in place of the corrupt one (kept as `transformer.wlo-corrupt/` for the vendor report). A fresh
`difflet compile ... --shapes ...` with this branch reproduces the same thing via the new default.

## 7. Cost of the mitigation (measured)

DiT single-forward latency, shape A: WLO 795 ms → no-WLO 1094 ms (**+37.6 % per DiT step**). Per the
repo's measured breakdown (DiT compute = 12–19 % of warm e2e), expected e2e impact ≈ 5–7 %. Accepted
as correctness-first; revisit when the vendor bug is fixed.

## Artifacts index

- Compiled: `~/.cache/difflet/hunyuan_video_dit_tp4cp1_h320w512f61/`, `..._h320w512f33/`,
  `..._h320w512f61_bkt2-ed786a/` (fixed; `transformer.wlo-corrupt/` preserved), `..._bkt2-noWLO/`
- Outputs/logs: `$LOG/out_baseA|baseB.pt`, `out_fixed_A|B.pt`, `step_*.pt`, `ref_B_fp32.pt`,
  `gen_*.log`, `compile_*.log`, `nm_*.json`
- Scripts: `gen_chain.sh`, `step_chain.sh`, `verify_chain2.sh`, `verify_chain3.sh`, `verify_nowlo.sh`,
  `finalize_task1.sh`, `hash_pt.py`, `inspect_buckets.py`, `step_compare.py`, `cpu_ref.py`, `compile_nowlo.py`
