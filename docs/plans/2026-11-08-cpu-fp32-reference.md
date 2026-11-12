# Phase G: the CPU fp32 reference for every model

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** For each of the seven models of the campaign, run diffusers in fp32 on the host CPU with exactly
the settings of that model's device runs, and score the device's bf16 and shipped-MXFP8 outputs against
it (PSNR / SSIM / LPIPS on the decoded output, rel L2 / cosine on the final latents).

**Architecture:** One reference script per model under `scripts/fp32_ref/`, sharing a small helper
(`scripts/fp32_ref/_common.py`: the device-noise law, saving latents and outputs the way the device run
did, a `run.json` record). Each script reproduces the device's initial noise on the CPU, injects it into
the diffusers fp32 pipeline (`latents=` or a `prepare_latents` replacement), runs 50 steps and saves the
final latents in the device's layout plus the decoded output in the device's format. A scoring script
(`scripts/fp32_ref/score.py`) compares the device files against the reference with
`scripts/compare_difflet_runs.py` and prints one row per model.

**Tech Stack:** diffusers 0.38, torch 2.14 (CPU, AVX-512, fp32), the host's hub snapshots.

**Spec:** the user's instruction (2026-11-08): "跑下 cpu fp32 ... 测试下每个 model 的 psnr/ssim, 从最快的跑起来,
用同一种 setting 来跑 cpu fp32, 和我们测试 on device 的每个 model 的 setting 要保持一致". The per-model research
(device settings with sources, device output paths, latents layouts, noise checks, recipes, CPU estimates,
risks) is `/home/ubuntu/fp8-bench/phase_g_research.json` (one object per model); every task reads its
model's object in full.

## Global Constraints

- Same settings as the device run: model id and snapshot, prompt, negative prompt, guidance (true CFG vs
  embedded, values), shape, 50 steps, seed 42, scheduler and sigmas, max sequence length, fps, Wan 2.2's
  `boundary_ratio` / second guidance; only the dtype (fp32) and the device (CPU) differ.
- **Noise:** the device's initial noise, bit for bit. FLUX, Qwen-Image, LTX-2 (video and audio) and
  HunyuanVideo-1.5 draw it in bf16 on a CPU generator; it equals diffusers' fp32 seed-42 draw rounded to
  bf16, so the reference injects `draw.to(bf16).float()`. Wan 2.1, Wan 2.2 and HunyuanVideo draw it in fp32
  exactly as diffusers does. Each script asserts the md5 of the noise it injects against the md5 its
  research object records (or computes the device law and asserts `torch.equal` with it).
- Short shapes: FLUX and Qwen-Image 1024x1024; LTX-2 512x768x121; Wan 2.1 and 2.2 480x832x9 (Wan 2.2 with
  both experts); HunyuanVideo 320x512x61; HunyuanVideo-1.5 480x848x9. The long shapes are 5-16 h each on the
  CPU and are out of scope.
- Never load a model on a Neuron device; never write under `~/.cache/difflet`; never run pytest beyond the
  new test files (real snapshots on this host; `tests/unit/conftest.py` guards bf16 copies only).
- Two references at a time at most, each pinned to one NUMA node (`taskset -c 0-47,96-143` /
  `taskset -c 48-95,144-191`), threads set to that node's 96 vCPUs.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; push every commit.

## Device outputs scored (bf16 / shipped MXFP8)

| model | bf16 (decoded / latents) | MXFP8 shipped default (decoded / latents) |
|---|---|---|
| FLUX | `flux_d_tp4/A1_gena.png` / none (tp4 decodes in process) | `flux_d_tp4/B1_gena.png` / none |
| Qwen-Image | `qwen_bf16_tp4_ref2` (research object) | `qwen_mx_hm0_tp4` |
| LTX-2 | `ltx2_tp4/bf16_gen50a.mp4` / `work_bf16_gen50a/latents.pt` | `ltx2_tp4/mx_gen50a.mp4` / `work_mx_gen50a/latents.pt` |
| Wan 2.1 | `wan_h3/h5/bf16_gen50a.mp4` (c7c2346) | `wan_h3/final/wan21_mx/gen50a.mp4` (90c648f) |
| Wan 2.2 (2 experts) | `wan_h3/final/wan22d_bf16/gen50a.mp4` | `wan_h3/final/wan22d_mx/gen50a.mp4` |
| HunyuanVideo | `hv1_tp4/bf16_gen50a.mp4` | `hv1_tp4/loo_V15_gen50a.mp4` (= fc38fb6's default) |
| HunyuanVideo-1.5 | `hv15/bf16_9/gena.mp4` | `hv15/nod4/mx_gena.mp4` (9f6aedc's default on the tree without D4, `0f75d11`; planned as `hv15/u3/gena.mp4`, D4's tree) |
(paths under `/home/ubuntu/fp8-bench/`; latents beside each in its work dir; the research object is the source.)
FLUX is scored on the decoded image only (no device latents at tp4).

## Review Focus

1. A setting that differs silently from the device (a pipeline default that difflet overrides, e.g.
   guidance, flow shift, max sequence length, negative prompt, Wan's CFG, Wan 2.2's boundary).
2. Noise injected in the wrong layout (packed vs unpacked, video vs audio for LTX-2) or dtype.
3. Latents saved in a layout `compare_difflet_runs.py` cannot match against the device's latents.pt.
4. The decoded output encoded differently from the device (fps, codec, frame count, value range), so the
   pixel metrics measure the encoder.
5. A pipeline component left in bf16 (the reference must be fp32 end to end, VAE included).

---

### Task G1: the shared helper, the scorer and their tests

**Files:** Create `scripts/fp32_ref/__init__.py` (empty), `scripts/fp32_ref/_common.py`,
`scripts/fp32_ref/score.py`, `tests/unit/scripts/test_fp32_ref_common.py`.

**Interfaces (`_common.py`):**
- `device_noise(shape, seed, *, bf16_draw: bool, generator_device="cpu") -> torch.Tensor` (fp32): diffusers'
  `randn_tensor` law with a CPU generator seeded `seed`; when `bf16_draw`, returns the bf16 draw as float32.
- `tensor_md5(t) -> str` (over the contiguous bytes of the tensor in its dtype).
- `set_threads(n: int) -> None` (torch intra-op threads; refuses n < 1).
- `save_run(out_dir, *, latents, output, settings: dict, noise_md5: str, timings: dict, fps=None)`: writes
  `latents.pt` (CPU tensor as given), the output (PIL image -> `out.png`; video frames -> `out.mp4` with the
  device's writer, `difflet`'s own export function, and the given fps), and `run.json`.
**Interfaces (`score.py`):** `python scripts/fp32_ref/score.py --model <key> --ref-dir <dir>` scores the
device bf16 and MXFP8 files of the table above (a dict `DEVICE_OUTPUTS` in the script) against
`<ref-dir>`, writes `<ref-dir>/score.json` and prints `model | bf16 vs fp32 psnr/ssim/lpips/cos | mx vs fp32 ...
| mx - bf16 (dB)`.

- [x] **Step 1: failing tests** (`tests/unit/scripts/test_fp32_ref_common.py`): `device_noise` with
  `bf16_draw=True` equals `torch.randn(shape, generator=g, dtype=torch.bfloat16).float()` and differs from the
  fp32 draw; with `False` equals the fp32 draw; `tensor_md5` is stable and dtype-sensitive; `save_run` writes
  the three files for an image and for a 4-frame video (tmp_path, no model); `score.py`'s `DEVICE_OUTPUTS`
  names every model key and every path exists on this host (skip the existence check when the bench dir is
  absent).
- [x] **Step 2-4:** run (fail), implement, run (pass): `pytest tests/unit/scripts/test_fp32_ref_common.py -q`.
- [x] **Step 5: Commit** `feat(scripts): phase G's CPU fp32 reference helper and scorer (the device-noise law, device-layout saves, one score row per model)`.

**Done** (`2e83328`). Deviation: `score.py`'s HunyuanVideo-1.5 MXFP8 entry named `hv15/u3/gena.mp4`,
the default on D4's tree; F7 reverted D4 (`dd3eed2`), so the shipped default's clip is
`hv15/nod4/mx_gena.mp4`. The runner's snapshot of `3821d3e` carried that one-line patch before the
repo fix `0f75d11` (evidence 10a, 10b).

### Tasks G2-G8: one reference script per model

`scripts/fp32_ref/<key>.py` for `flux`, `qwen_image`, `ltx_2`, `wan21`, `wan22`, `hunyuan_video`,
`hunyuan_video_15`. Each: `--out-dir`, `--threads` (default 96), `--steps` (default 50; a smoke run uses 1),
`--taskset` is the caller's job. The settings are module constants copied from the research object with a
comment naming each source; the noise is `device_noise(...)` with the model's law, asserted against the
research's md5 or law; the pipeline loads `torch_dtype=torch.float32`, every component fp32; latents saved
in the device layout; output saved as the device did. A unit test per script
(`tests/unit/scripts/test_fp32_ref_<key>.py`) pins the settings constants against the device values (no model
load) and the noise law at the device shape (cheap: a CPU randn).

- [x] **Step 1:** settings test (fails), **Step 2:** implement, **Step 3:** settings test passes,
  **Step 4:** a 1-step smoke run on the CPU at the device shape (`--steps 1`, pinned to one NUMA node),
  recording wall time and peak RSS; the files are written and `score.py` reads them.
- [x] **Step 5:** no per-task commit; G9 commits all scripts after review.

**Done** (committed in G9). The smoke runs are `/home/ubuntu/fp8-bench/fp32_ref_smoke/<key>/` (48
threads, one NUMA node, `smoke.log` under `time -v`): every one exited 0 and has its `score.json`.
Qwen-Image and LTX-2 ran `--steps 2` (Qwen-Image's one-step schedule gives NaN,
`qwen_image/smoke_steps1_nan.log`).

### Task G9: review, commit

Review every script against its research object and the Review Focus; fix; commit
`feat(scripts): phase G's CPU fp32 reference scripts for the seven models (device settings and noise, fp32 end to end)`; push.

- [x] **Done** (`3821d3e`, pushed).

### Task G10 (host, not the implementers'): the runs, fastest first

Order (research estimates at ~96 threads): FLUX (20-30 min), Qwen-Image (25-45 min), Wan 2.1 (25-45 min),
HunyuanVideo-1.5 (35-45 min), Wan 2.2 (30-50 min), HunyuanVideo (~1 h), LTX-2 (1-2 h). Two at a time, one per
NUMA node; after each, `score.py`, and the row goes to the status page.

- [x] **Done** (2026-11-08, 09:55:40 to 14:12:35): `/home/ubuntu/fp8-bench/fp32_ref/run_queue.sh`, two
  workers in this order, 96 threads each, from a snapshot of `3821d3e` with G1's `score.py` patch.
  Every reference exited 0. Wall times (`time -v`): FLUX 42:24, Qwen-Image 42:59, Wan 2.1 44:41,
  HunyuanVideo-1.5 56:08, Wan 2.2 49:53, HunyuanVideo 1:15:17, LTX-2 1:54:23. Deviation: FLUX is
  scored on its one image only (p0, seed 42: no device latents at tp4, and the 12-image set was not
  referenced). Follow-up beyond the plan: HunyuanVideo-1.5's two groups alone on the final tree
  (`hv15/final_g1`, `final_g3`) against the same reference. Scores and reading: evidence section 10.

### Task G11: evidence and docs

A section "CPU fp32 reference" in `docs/verification/2026-11-07-mxfp8-video-models.md`: the settings and
noise per model, the score table (bf16 vs fp32, MXFP8 vs fp32, their difference), the reading (MXFP8 judged
relative to bf16's own distance from fp32; single-sample caveat on chaotic models); README quality notes
updated where "pending phase G" stood.

- [x] **Done**: evidence section 10 (10a what ran, 10b the device files, 10c the scores, 10d the
  reading, 10e HunyuanVideo-1.5's single groups, 10f what phase G does not cover); every "pending
  phase G" in README, `docs/cli-staged-commands.md` and the evidence doc replaced by the measured
  result. MXFP8 - bf16 against fp32: Qwen-Image +1.02 dB, LTX-2 +0.29, HunyuanVideo -0.20, FLUX
  -2.06 (one image), Wan 2.1 -2.36, Wan 2.2 -2.93, HunyuanVideo-1.5 -7.47 (the image MLP alone
  -4.69). HunyuanVideo-1.5's default was not changed here; the user decided on 2026-11-09 that it has
  no MXFP8 default (Task F8 in `docs/plans/2026-11-07-mxfp8-hunyuan-video-15.md`).
