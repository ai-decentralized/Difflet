# attention_cte's softmax exp on ScalarE: Difflet's bf16 attention default on trn3

Plan `docs/plans/2026-11-16-attention-scalar-exp-default.md` (Tasks 3 and 4). Branch
`feat/attn-scalar-exp-default`: f6baeb04 makes `allow_vector_exp=False` the default of every flash
path of the native neuron backend's attention (unmasked, key-bounded, logsumexp); c2a75db8 drops
MiniMax-H3's own opt-in. Host: trn3.48xlarge, LNC=2, torch-neuronx 2.14.3, nkilib 2.14.

## What changes, and where it does not

- nkilib `attention_cte` (`allow_vector_exp`, default True) puts the softmax exp on the Vector
  engine's `nisa.exponential` only when `is_trn3_b1()` (NeuronCore gen4, sub-version 1). trn2 and
  trn3pre have no Vector exp and always run it on ScalarE (`nisa.activation`), so on them the
  change is a no-op (a new op name, so one recompile; not measured here: no trn2 host).
- On trn3 the Vector engine already evicts every QK^T tile from PSUM (scale and row max, plus a
  `range_select` on the bounded path), so with the exp there too it bounds the kernel while
  ScalarE idles (`docs/verification/2026-11-13-dit-block-profile.md` section 9 on
  `feat/mxfp8-sage-attention`: wan_81f VectorE 62.0 of 63.5 ms, ScalarE 1.6 ms).
- Unchanged: the trainium (NxD / torch-xla) backend (no torch_xla in the trn3 venv; trn2 only),
  the NKI Library ring kernel (no such argument), `DIFFLET_NEURON_ATTENTION=sdpa` (torch-neuronx's
  SDPA op, VectorE exp), and `attention(..., vector_exp=True)` (the previous kernels).

## Method

Before = `git archive d70faffc` (main), after = `git archive c2a75db8` (kernel probes: f6baeb04),
frozen under `/mnt/nvme/difflet-vexp/` (`TREES.txt`). `/mnt/nvme` is instance storage; the JSONs,
step logs and tables are packed on EBS at `/home/ubuntu/difflet-vexp-evidence-2026-11-16.tar.gz`
(248 files, sha256 aa04a3c0baf476a2...); the decoded outputs stay on NVMe only. Before every launch the lane's device had no
holder (`fuser`) and its CPUs were sampled (`mpstat`, logged per step); CPUs pinned with `taskset`,
`NEURON_NUMA_PIN=0`.

**Kernel** (`tests/manual/probe_neuron_attention_scalar_exp.py`): section 9's region and inputs
(`ops_impl.attention.attention` as the models call it, compiled; q, k, v ~ N(0, 1) bf16, seed 0),
one logical core, 3 worker processes per arm, 40 calls (flux, qwen) or 20; median and spread
(max - min) / median. Faster = ratio < 1 - J, J = max(2%, the VectorE arm's spread). Outputs are
compared bit for bit with section 9's saved outputs (`/mnt/nvme/difflet-mxattn/probe/rebalance/
{shipped,scalar_exp}/<shape>.out.pt`). Accuracy: rel L2 against fp32 attention per head (the
bounded reference rounds the pre-scaled q to bf16 as the device does, so it measures the kernel
alone and reads below section 9's bounded numbers; same output bits). Peaked: q and k times 2
(scores std ~4). LSE: `_attention_with_lse` (the ring / joint-ring partial) has no switch, so its
VectorE baseline is the same probe run in the before tree.

**Model**: `scripts/quant_bench/bench.sh` + `scripts/mx_attention/run_arms.sh` from
`feat/mxfp8-sage-attention` 5d5b83c5, copied identically into both trees with keys added
(`qwen_image_cp2`: tp2 cp2 ring, the joint ring's logsumexp op; `minimax_h3`; the tp2 control
`wan21_tp2`, `flux_tp2`) and a `gen1` step, arm bf16, seed 42, 50 steps (Wan 2.2 official: 40),
one compile cache for both trees (a hardlinked copy of an earlier campaign's; NEFF and HLO keys are
content hashes, and every after output differs from its before output, so no stale kernel ran).
Every lane first warmed both trees (2-step generate: the compile). Timing: two single-request
generate processes per tree, interleaved before gen1, after gen1, before gen2, after gen2, on one
device and CPU set; step time = median of the two processes' per-request step medians (the
orchestrator's `request: ... step median` line; HunyuanVideo-1.5's `denoise ... median`; MiniMax-H3's
`ms/DiT call`; LTX-2 and HunyuanVideo print none, so theirs is `(stage - load+warmup) / steps`, which
includes the prompt encode), spread = (max - min) / median, faster = ratio < 1 - J with
J = max(2%, before's spread); determinism = gen1 and gen2 identical within each tree (all are).
The `--requests` form (two requests in one process) was not usable on main: it goes through the DP
router, which assigns its workers cores 0-3 whatever `NEURON_RT_VISIBLE_CORES` says (fixed on
`fix/dp-router-visible-cores` 51cde90b, not on main); that attempt was discarded. Quality-only lanes
(the phase G short shapes, where CPU fp32 references exist): `warm` and `gen2` per tree. Quality:
after vs before (PSNR / SSIM of the decoded output, latent rel L2), and before and after vs the CPU
fp32 reference (`/home/ubuntu/fp8-bench/fp32_ref/<key>/`, same prompt, seed, shape and steps). The
user's rule: report; stop and ask if after-vs-fp32 is more than 0.5 dB below before-vs-fp32.

## Kernel results

One logical core per shape; devices and CPU sets as section 9's (device 4 held the LSE rows).
The LSE rows compare the before tree's logsumexp op (VectorE) with the after tree's (ScalarE).

| shape | device | VectorE ms (spread) | ScalarE ms (spread) | ratio | J | faster | bits vs section 9 (vector / scalar) | rel L2 vs fp32, vector / scalar (max head) | peaked (q, k x2) rel L2, vector / scalar |
|---|---|---|---|---|---|---|---|---|---|
| `flux_1024` | 4 | 0.9185 (7.84%) | 0.8217 (0.58%) | 0.895 | 8% | **yes** | identical / identical | 2.307e-3 / 2.307e-3 (2.316e-3 / 2.316e-3) | 1.784e-3 / 1.784e-3 |
| `qwen_1024` | 5 | 1.188 (3.68%) | 1.084 (0.48%) | 0.912 | 4% | **yes** | identical / identical | 2.318e-3 / 2.318e-3 (2.358e-3 / 2.358e-3) | 1.785e-3 / 1.785e-3 |
| `ltx2_video` | 7 | 2.066 (3.80%) | 1.773 (0.78%) | 0.858 | 4% | **yes** | identical / identical | 2.307e-3 / 2.306e-3 (2.327e-3 / 2.326e-3) | 1.784e-3 / 1.784e-3 |
| `wan_81f` | 1 | 63.55 (0.04%) | 51.98 (0.01%) | 0.818 | 2% | **yes** | identical / identical | 3.106e-3 / 3.106e-3 (3.214e-3 / 3.214e-3) | 2.677e-3 / 2.677e-3 |
| `hv15_121f` | 2 | 143.6 (0.01%) | 113.0 (0.05%) | 0.787 | 2% | **yes** | identical / identical | 3.833e-3 / 3.833e-3 (3.860e-3 / 3.859e-3) | 3.501e-3 / 3.501e-3 |
| `hv1_L` | 3 | 183.5 (0.01%) | 144.7 (0.01%) | 0.788 | 2% | **yes** | identical / identical | 4.190e-3 / 4.190e-3 (4.270e-3 / 4.270e-3) | 3.834e-3 / 3.834e-3 |
| `lse_qwen_cp2` (12 x 3072 x 2048) | 4 | 0.7078 (2.10%) | 0.5783 (2.51%) | 0.817 | 2% | **yes** | n/a | 2.302e-3 / 2.302e-3 (lse max err 4.8e-6 / 6.7e-6) | n/a |
| `lse_wan_cp2` (10 x 16380 x 16380) | 4 | 17.14 (1.74%) | 14.30 (3.79%) | 0.834 | 2% | **yes** | n/a | 2.594e-3 / 2.594e-3 (lse max err 9.5e-6 / 8.6e-6) | n/a |

Both arms reproduce section 9's saved outputs bit for bit, so the shipped kernels and section 9's
rebalanced kernel are exactly what Difflet's `attention()` now runs with `vector_exp=True` and by
default. The flux VectorE arm's spread (another session's host load) sets its J to 8%. `lse_wan_cp2`
is a size proxy: compiled Wan cp2 takes the NKI ring kernel at that shard length.
`tests/manual/check_neuron_compile_c9.py --expect cold` (device 6, tp4): all cases pass, the graph
holds one `difflet_neuron.flash_attention_scalar_exp` node.

## Model results: speed

| key | shape, layout | before ms/step (spread) | after ms/step (spread) | ratio | faster (J = 2%) |
|---|---|---|---|---|---|
| `hv15_official` | HunyuanVideo-1.5 480x848x121, tp4 | 9913.5 (0.01%) | 8268.5 (0.01%) | **0.834** | yes |
| `hunyuan_video_L` | HunyuanVideo 544x960x129, tp4 | 15912 (0.00%) | 13733 (0.04%) | **0.863** (stage - load) | yes |
| `wan22_official` | Wan 2.2 A14B 480x832x81, tp4, 40 steps | 10118.9 (0.05%) | 9085.7 (0.01%) | **0.898** | yes |
| `wan21_official` | Wan 2.1 14B 480x832x81, tp4 | 10106.9 (0.04%) | 9086.0 (0.11%) | **0.899** | yes |
| `flux` | FLUX.1-dev 1024x1024, tp4 | 223.6 (0.49%) | 216.0 (0.00%) | 0.966 | yes |
| `qwen_image` | Qwen-Image 1024x1024, tp4 | 304.2 (0.10%) | 296.9 (0.07%) | 0.976 | yes |
| `ltx_2` | LTX-2 512x768x121, tp4 | 1501 (1.47%) | 1468 (0.82%) | 0.978 (stage - load) | yes |
| `qwen_image_cp2` | Qwen-Image 1024x1024, tp2 cp2 ring | 688.3 (0.12%) | 675.5 (0.04%) | 0.981 | no (inside 2%) |
| `minimax_h3` | MiniMax-H3 256x448x124, tp4 | 580.0 (0.34%) | 580.5 (0.17%) | 1.001 | no |

The gain tracks attention's share of the step: 10-17% on the long video shapes, 2-3% on the
1024^2 images. MiniMax-H3 does not move: on main its key-bounded attention already ran the Scalar
exp (its own `attention_vector_exp` opt-in, now deleted), so only its few unbounded calls changed.

## Model results: quality

After vs before, and both vs CPU fp32 where a reference for the same config exists (seed 42):

| key | after vs before PSNR / SSIM | before vs fp32 (dB / SSIM) | after vs fp32 (dB / SSIM) | after - before (dB) |
|---|---|---|---|---|
| `wan21` (Wan 2.1, 480x832x9) | 36.00 / 0.9545 | 36.10 / 0.9554 | 34.89 / 0.9510 | **-1.22** |
| `flux` (1024^2) | 25.34 / 0.9361 | 23.36 / 0.9170 | 22.52 / 0.9010 | **-0.84** |
| `qwen_image` (1024^2) | 49.77 / 0.9976 | 36.34 / 0.9905 | 36.02 / 0.9900 | -0.32 |
| `ltx_2` (512x768x121) | 26.47 / 0.9167 | 16.55 / 0.4716 | 16.49 / 0.4662 | -0.06 |
| `qwen_image_cp2` (vs Qwen's reference) | 50.86 / 0.9977 | 36.42 / 0.9906 | 36.40 / 0.9906 | -0.02 |
| `hunyuan_video` (320x512x61) | 35.41 / 0.9543 | 21.77 / 0.8559 | 21.92 / 0.8590 | +0.15 |
| `wan22` (Wan 2.2, 480x832x9) | 36.76 / 0.9635 | 35.60 / 0.9593 | 35.88 / 0.9600 | +0.28 |
| `hunyuan_video_15` (480x848x9) | 28.87 / 0.8889 | 24.75 / 0.8459 | 25.07 / 0.8522 | +0.32 |

Long shapes (no fp32 reference): after vs before PSNR / SSIM `wan21_official` 27.56 / 0.846,
`wan22_official` 20.33 / 0.733, `hv15_official` 18.40 / 0.463, `hunyuan_video_L` 36.81 / 0.965;
MiniMax-H3 video latents rel L2 0.171 (cosine 0.985), audio 0.021. Clip statistics match (pixel
mean within 0.5, std within 0.2, mean frame-to-frame change and saturated share equal), so the
after clips are other samples of the same kind, not broken ones. The before outputs equal the
campaign's known main outputs where one exists (Qwen md5 5478687f, Wan 2.1 9-frame 39bac229,
FLUX 23.36 dB = phase G's device bf16). LTX-2's 16.5 dB is the known host-side bf16 state issue.

**The gate.** wan21 (-1.22 dB) and flux (-0.84 dB) crossed the 0.5 dB rule, so the work stopped and
the user asked for a control: the same keys at tp2, which changes only the reduction order.

| | vs fp32, tp4 | vs fp32, tp2 | tp2 vs tp4, same engine | ScalarE vs VectorE, same TP |
|---|---|---|---|---|
| Wan 2.1 9f, VectorE | 36.10 | 35.81 | 36.91 | |
| Wan 2.1 9f, ScalarE | 34.89 | 35.96 | 35.67 | tp4 36.00, tp2 36.77 |
| FLUX, VectorE | 23.36 | 23.27 | 35.65 | |
| FLUX, ScalarE | 22.52 | 22.50 | | tp4 25.34, tp2 25.54 |

For Wan the engine change is a perturbation of the same size as a TP change (both ~36-37 dB
apart), and at tp2 the ScalarE run is the closer one to fp32 (+0.15 dB): the -1.22 dB is one draw.
For FLUX a TP change barely moves the image (35.65 dB apart, -0.09 dB vs fp32) while the engine
change moves it to a different sample (25.3-25.5 dB apart) that is 0.8 dB further from fp32 at both
TP degrees; with one independent sample per engine this cannot be told apart from an unlucky basin.
Both FLUX results stay above FLUX's own CPU bf16 floor (21.47 dB vs fp32). Per call the two exps are
equally accurate (kernel table). **Decision (user, 2026-11-16): ScalarE exp is the default for
every model, FLUX included; no further seeds.**

## Reading

- Same math, different rounding: the Vector engine's `nisa.exponential` and the Scalar engine's
  `activation(exp)` round differently in the last bits, so the outputs differ while the per-call
  error against fp32 is identical. A 50-step trajectory carries that difference to a nearby but
  different sample; the 7 independent fp32 comparisons go both ways (-1.22 .. +0.32 dB, mean -0.24).
- Speed: every routed shape is 9-21% faster at the kernel; models gain in proportion to
  attention's share (HunyuanVideo-1.5 121f 0.834, Wan 81f 0.90, images 0.97-0.98).
- trn2 is unaffected in computation (no Vector exp there); the unmasked path's new op name means
  one recompile. Not measured on trn2 hardware.
- Found on the way: main's DP router ignores `NEURON_RT_VISIBLE_CORES` for `--requests`
  (`fix/dp-router-visible-cores`); a 67k-token key-bounded attention_cte's NKI trace can outlast the
  NKI trace cache's file lock (HunyuanVideo 544x960x129 warm-up: ranks broke the lock and retraced,
  ~65 min instead of ~30).
