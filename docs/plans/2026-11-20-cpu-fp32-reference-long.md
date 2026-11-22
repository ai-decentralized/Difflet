# CPU fp32 references at the const-max campaign's long shapes

**Goal:** score arms A (bf16), B (bf16 + `DIFFLET_EXP_ATTN_CONSTMAX=1`), C (`--quant mxfp8`) and
D (C + const-max) of Wan 2.1, Wan 2.2, HunyuanVideo 1.0 (HV1) and HunyuanVideo-1.5 (HV1.5)
against a diffusers fp32 run on the host CPU at the shapes and settings of the device runs under
`/mnt/nvme/difflet-exp/constmax-models/<model>/<arm>/`. The 2026-11-08 references
(`docs/plans/2026-11-08-cpu-fp32-reference.md`) are at the short shapes and do not match.

**What changed:** four sibling scripts, `scripts/fp32_ref/{wan21,wan22,hunyuan_video,
hunyuan_video_15}_long.py`, each the short script with the long settings below (the short scripts,
their settings and noise md5s are unchanged); `_wan.py` takes the model card's negative prompt
from bench.sh, checks an optional `noise_md5` and counts two DiT forwards per step under CFG;
`scripts/fp32_ref/run_long_queue.sh` runs the queue. (The campaign's scorer,
`scripts/attn_constmax/score.sh`, was removed with its other one-off scripts; it is at `618365c3`.)
`tests/unit/scripts/test_fp32_ref_long.py` checks every setting against each arm's command line
and request lines, the noise, Wan 2.2's expert split and the device latents' layout.

## Settings (only the dtype, fp32, and the device, CPU, differ)

Device files: `R = /mnt/nvme/difflet-exp/constmax-models`; `cmd` is the timed run's command line
(`R/wan2x/<arm>/timed.cmd.txt:1`; HV1 and HV1.5 `R/<model>/<arm>/timed.log`, the
`Command being timed` line), `req` is `R/<model>/<arm>/requests.jsonl:1-2`; all four arms agree.
The arm scripts cited below (`arms_wan.sh`, `hv1_arms.sh`, `hv15_arms.sh`, under
`scripts/attn_constmax/`) were removed with the campaign's one-off scripts; they are at `618365c3`.

| model | setting | value | device source |
|---|---|---|---|
| Wan 2.1 | model, revision | Wan-AI/Wan2.1-T2V-14B-Diffusers `38ec498c` | cmd; arms_wan.sh:35 |
| | shape | 480x832x81, latents (1, 16, 21, 60, 104) | cmd; arms_wan.sh:36 |
| | steps, seed | 50, 42 | cmd, req; steps arms_wan.sh:37 |
| | guidance | true CFG 5.0, cond + uncond forwards (100) | cmd; arms_wan.sh:37 |
| | prompt | "a cat walking through a garden" | req; arms_wan.sh:33 |
| | negative prompt | model card's (English), `WAN21_NEG` | req, cmd; bench.sh:64 |
| | text length | 512 | difflet/cli/orchestrators/wan.py:256 |
| | scheduler | snapshot UniPC, flow_shift 3.0 | difflet/models/wan/pipeline.py:577 |
| | fps | 16 | difflet/cli/orchestrators/wan.py:143 |
| Wan 2.2 | model, revision | Wan-AI/Wan2.2-T2V-A14B-Diffusers `5be7df96`, `--dual-expert` | cmd; arms_wan.sh:38 |
| | shape | 480x832x81, latents (1, 16, 21, 60, 104) | cmd; arms_wan.sh:39 |
| | steps, seed | 40, 42 | cmd, req; steps arms_wan.sh:40 |
| | guidance | CFG 4.0 (high noise) / 3.0 (low noise) | cmd; arms_wan.sh:40 |
| | experts | boundary 0.875: 13 steps t >= 875, 27 below (forwards 26 / 54) | snapshot model_index.json; difflet/models/wan/pipeline.py:496 |
| | prompt, negative | as Wan 2.1; card's Chinese `WAN22_NEG` | req; bench.sh:65 |
| | text, scheduler, fps | 512, UniPC flow_shift 3.0, 16 | as Wan 2.1 |
| HV1 | model, revision | hunyuanvideo-community/HunyuanVideo `e8c2aaa6` | cmd; hv1_arms.sh:42 |
| | shape | 544x960x129, latents (1, 16, 33, 68, 120) | cmd; hv1_arms.sh:43 |
| | steps, seed | 50, 42 | cmd, req; hv1_arms.sh:44,78 |
| | guidance | embedded 6.0, no negative, true CFG 1.0 | cmd, req; hv1_arms.sh:44 |
| | prompt | "a cinematic shot of a red fox running through a snowy forest" | req; hv1_arms.sh:45 |
| | text length | Llama 256 | difflet/cli/orchestrators/hunyuan_video.py:101 |
| | decode | tiles 256 px / 16 frames, stride 3/4 | difflet/backends/neuron/hunyuan_video/vae.py:68 |
| | fps | 24 | difflet/cli/orchestrators/hunyuan_video.py:137 |
| HV1.5 | model, revision | HunyuanVideo-1.5-Diffusers-480p_t2v `286be7ce` | cmd; hv15_arms.sh:25-26 |
| | shape | 480x848x121, latents (1, 32, 31, 30, 53) | cmd; hv15_arms.sh:26 |
| | steps, seed | 50, 42 | cmd, req; hv15_arms.sh:60 |
| | guidance | true CFG 6.0, the snapshot guider's (no flag) | guider/guider_config.json; orchestrators/hunyuan_video_15.py:566 |
| | prompt, negative | "a cat walking through a garden", "" | req; hv15_arms.sh:27; orchestrators/hunyuan_video_15.py:279 |
| | decode | tile 128 px | difflet/backends/neuron/hunyuan_video15/vae.py:28 |
| | fps | 24 | difflet/cli/orchestrators/hunyuan_video_15.py:78 |

The arms' own differences (const-max, MXFP8 targets: HV1.5's C/D set `DIFFLET_QUANT_TARGETS`) are
what the reference measures. The HV1 runs read the hub at `/mnt/nvme/difflet-quant-bench-hub`, the
reference `~/.cache/huggingface/hub`: the `e8c2aaa6` snapshots link the same 29 blobs.

## Initial noise

| model | law (device code) | shape | md5 |
|---|---|---|---|
| Wan 2.1, 2.2 | fp32 `torch.randn`, CPU generator seed 42 (wan/pipeline.py:161 via neuron_application.py:617, generator orchestrators/wan.py:734) | (1, 16, 21, 60, 104) | `fe29ee8e34bbe84ef350a10d25f1be01` |
| HV1 | fp32 `torch.randn` at `dit_input_shapes` (hunyuan_video/neuron_application.py:482, generator orchestrators/hunyuan_video.py:826) | (1, 16, 33, 68, 120) | `df925d6d08d0004d0c1c807cb1df6fea` |
| HV1.5 | bf16 `randn_tensor` on a CPU generator (hunyuan_video/pipeline15.py:221, generator orchestrators/hunyuan_video_15.py:575); hashed as bf16 | (1, 32, 31, 30, 53) | `56869f4e18d40091e5b7d3ad0c0acb63` |

Each `_long` script asserts its md5 before denoising. The test draws each tensor through the device
code itself (`WanOrchestrator.prepare_latents`; `_initial_latents` on a
`NeuronHunyuanVideoConfig(544, 960, 129)`; pipeline15's bf16 draw) and through diffusers'
`prepare_latents`, and all are bit-equal; HV1.5's bf16 draw also equals the fp32 draw rounded to
bf16. No device file holds the initial noise or its hash, so there is nothing more to compare with.

## Memory and time

- diffusers 0.38 runs attention through `F.scaled_dot_product_attention`, which on this CPU (torch
  2.14, fp32) takes the blocked flash path: no score matrix is materialized. Measured at 16,384
  tokens x 2 heads (`/mnt/nvme/difflet-exp/fp32_ref_long/_checks/sdpa_mem.py`): +0.03 GB with no
  mask (Wan) and with HV1's (B, 1, 1, N) key mask; +1.0 GB with HV1.5's (B, 1, N, N) bool mask,
  the mask's one float copy, not per head. At HV1.5's ~51k joint tokens that copy is ~10.5 GB
  (plus ~8 GB of bool masks), per attention call, freed after it. No fix is needed.
- Peak RSS: the 11-08 runs' (weights dominate: Wan 2.1 84 GB, Wan 2.2 144 GB, HV1 88 GB, HV1.5
  67 GB) plus the long activations (hidden states, FFN and QKV at 32-68k tokens, 10-20 GB) and the
  decode: about 100, 160, 110 and 95 GB. Any two fit easily in a NUMA node's 1 TB, so the two
  workers run concurrently.
- Time: the flash path runs at 0.88x the fp32 GEMM rate (`_checks/sdpa_rate.py`); scaling the 11-08
  step times by FLOPs gives about 20-21 min per step for Wan (two forwards), 23 min for HV1 and
  19 min for HV1.5: Wan 2.1 ~18 h, Wan 2.2 ~14 h, HV1 ~19 h, HV1.5 ~16 h, about 35 h of wall time
  on two workers at nice 19 on an idle host (longer when the other project's jobs run).

## Running

```bash
WT=/home/ubuntu/.claude/jobs/3bceca85/tmp/wt-models
$WT/scripts/fp32_ref/run_long_queue.sh setup       # src/ snapshot of HEAD, src.sha, queue.txt
nohup $WT/scripts/fp32_ref/run_long_queue.sh 0 > /mnt/nvme/difflet-exp/fp32_ref_long/worker0.out 2>&1 &
nohup $WT/scripts/fp32_ref/run_long_queue.sh 1 > /mnt/nvme/difflet-exp/fp32_ref_long/worker1.out 2>&1 &
```

The queue is HV1, Wan 2.1, Wan 2.2, HV1.5: the two workers take HV1 and Wan 2.1 first. Each run
writes `/mnt/nvme/difflet-exp/fp32_ref_long/<key>/` (`env.txt`, `run.log` with one line per step,
`/usr/bin/time -v`'s peak RSS at its end, `run.json` with `timings.steps_s`, `latents.pt`,
`out.mp4`, `out.pt`); `progress.txt` logs starts and exits.

## Scoring

Score a device run against a reference on the CPU with `scripts/compare_difflet_runs.py`:

```bash
R=/mnt/nvme/difflet-exp/fp32_ref_long/<key>
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/compare_difflet_runs.py \
  --ref-image $R/out.mp4 --ref-latents $R/latents.pt \
  --cand-image <run>/warm_2.mp4 --cand-latents <run>/work/worker_0/latents_req0001.pt \
  --out <run>_vs_fp32.json
```

with `<key>` = `wan21_long`, `wan22_long`, `hunyuan_video_long` or `hunyuan_video_15_long`, and
`<run>` a generate's work directory (its second request's output and final latents). It reports
video PSNR (frame mean and worst), SSIM, LPIPS, and latent cosine and rel L2. The const-max
campaign scored its arms A-D this way through `scripts/attn_constmax/score.sh` (at `618365c3`).
