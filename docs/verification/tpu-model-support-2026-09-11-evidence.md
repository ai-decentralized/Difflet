# Trainium-verified models on the TPU backend — on-device evidence

Campaign: 2026-09-11, host Cloud TPU v5litepod-4 (4 chips, 16 GB HBM each, 188 GB host RAM),
branch `verify/tpu-models-2026-09-11` (on top of `tpu-teacache` @ `76a1f7b`).
Question asked: *do the models verified on Trainium (FLUX, Qwen-Image, Wan 2.2 / 2.1, HunyuanVideo,
LTX-2) also work on the TPU backend?* — for the paths TPU has: `difflet serve`, the benchmark
runners, and (negatively) the CLI. Raw logs live in `artifacts/verification-2026-09-11-tpu/`;
each phase ends with "How to inspect manually". The Qwen-Image and Wan 2.2 rows reuse the
TeaCache campaign of the same day (`docs/worklog/2026-09-11-tpu-teacache-device.md`).

Statuses: **PASS** (ran end-to-end here) · **N/A** (excluded by design — the registry lists no
`tpu` backend for the model; what is verified is that the exclusion is *fail-fast*) ·
**NOT MEASURED**.

Environment: `/mnt/models/tpuenv312` (Python 3.12.14, torch 2.9.0+cpu, torch_xla 2.9.0, jax 0.7.1),
`LD_LIBRARY_PATH=/mnt/models/uvpy/cpython-3.12.14-linux-x86_64-gnu/lib`, `HF_HOME=/mnt/models/hf`,
`DIFFLET_BACKEND=tpu`. No Neuron toolchain in the venv (so an unported model that "falls through"
to the Neuron path fails with a `ModuleNotFoundError` — on a host with both toolchains it would
instead start a Neuron compile on a TPU box).

## Campaign result (final)

(filled at the end)

## Phase plan

| phase | what | runner | status |
|---|---|---|---|
| 0 | code gates: which models declare `tpu` in `difflet/registry.py` | reading | done |
| 1 | serving fail mode for the three unported models (FLUX, HunyuanVideo, LTX-2) | `difflet serve … ` under `DIFFLET_BACKEND=tpu` | done — bug 1 |
| 2 | CLI fail mode (`difflet generate`) on TPU | `difflet generate …` | done — bug 2 |
| 3 | Wan 2.1 serving on TPU (registered for `tpu`, never measured) | `difflet serve` + `/v1/videos/sync` | running |
| 4 | controls: Qwen-Image, Wan 2.2 serving on TPU | from the TeaCache campaign | done |

## Phase 0 — what the code says

`difflet/registry.py` `backends=`: `wan` (Wan 2.2 **and** 2.1 share the entry) and `qwen_image` →
`("trainium", "tpu")`; `flux`, `hunyuan_video`, `ltx_2` → `("trainium",)`. `hunyuan_video_15` is a
download-only scaffold and was not included.

`entry.require_backend()` was enforced in exactly one place before this campaign:
`difflet/pipeline/difflet_pipeline.py:98` (`DiffletPipeline.from_pretrained`). Neither
`resolve_serving_model` nor the CLI's `compile/generate/run` dispatch consulted it.

## Phase 1 — serving fail mode, unported models (before the fix)

All under `DIFFLET_BACKEND=tpu`, `--tp-degree 4 --cp-degree 1`, 4 chips idle.

| model | command variant | what happened | time | evidence |
|---|---|---|---|---|
| hunyuan_video | pinned revision `e8c2aaa…`, online | resolved the snapshot, then **entered the Neuron compile path** — `serving/models/hunyuan_video.py:819 _compile_artifact → :876 _build_llama_app` → `ModuleNotFoundError: No module named 'neuronx_distributed_inference'` | 2 s | `serve_hunyuan_video_tpu_online.log` |
| flux | no revision, online, no HF token | went straight to `snapshot_download` (`orchestrators/flux.py:72 prepare_runtime`) and stopped at `GatedRepoError: 401` — no backend check before weight traffic; with a token it would download 24 GB first | 1 s | `serve_flux_tpu_online.log` |
| ltx_2 | pinned revision `47da56e…`, offline | `ValueError: model 'ltx_2' does not support backend 'tpu'; supported backends: trainium` from `registry.py:138 require_backend` via `serving/models/ltx_2.py:148 build_pipeline → DiffletPipeline.from_pretrained` — correct, but only because this adapter happens to build a `DiffletPipeline` | 1 s | `serve_ltx_2_tpu_rev.log` |
| (all three) | offline, no revision | `LocalEntryNotFoundError` inside `snapshot_download` — same point: weights before backend | 1 s | `serve_<model>_tpu.log` |

**Bug 1 → `d384b5c`** `fix(serving): gate difflet serve on the registry's backend list`:
`resolve_serving_model` now calls `entry.require_backend(current_backend())` right after
`resolve_model`, before any weight traffic. Pinned by
`test_serving_rejects_unported_models_on_tpu_before_touching_weights` (3 models) and
`test_serving_accepts_the_tpu_ported_models` (Qwen, Wan 2.2, Wan 2.1).

Re-verified on device (`serve_<model>_tpu_gated.log`), offline:

| model | result | time | `snapshot_download` reached? |
|---|---|---|---|
| flux | `ValueError: model 'flux' does not support backend 'tpu'; supported backends: trainium` | 0.17 s | no |
| hunyuan_video | `ValueError: model 'hunyuan_video' does not support backend 'tpu'; supported backends: trainium` | 0.16 s | no |
| ltx_2 | `ValueError: model 'ltx_2' does not support backend 'tpu'; supported backends: trainium` | 0.16 s | no |

The error is still presented as a Python traceback with exit 1 (the existing convention for
`difflet serve` option errors); the message itself is actionable.

**How to inspect manually:** `grep -n "does not support backend" artifacts/verification-2026-09-11-tpu/serve_*_gated.log`;
`grep -c snapshot_download` on the same files must be 0; compare with the `_online.log` /
`_rev.log` files where the traceback bottoms out in `huggingface_hub` or `_build_llama_app`.

## Phase 2 — CLI fail mode on TPU (before the fix)

`difflet generate --model-id … --tp-degree 4 --cp-degree 1 --steps 2`, `DIFFLET_BACKEND=tpu`:

| model | what happened | evidence |
|---|---|---|
| qwen_image (ported) | parent spawned `python -m difflet.cli.stage --stage text …`; stage died `ModuleNotFoundError: No module named 'neuronx_distributed_inference'`; parent reported only `CalledProcessError … returned non-zero exit status 1` | `generate_qwen_image_tpu.log` |
| wan2_1 (ported) | same via `--stage transformer`; `ModuleNotFoundError: No module named 'neuronx_distributed'` | `generate_wan2_1_tpu.log` |
| ltx_2 (unported) | `Error: model weights not found.` (offline) — never reached a backend check either | `generate_ltx_2_tpu.log` |

The plan for the TPU backend (`docs/plans/2026-08-16-tpu-backend-support.md`) says the CLI has
no TPU runtime and `TpuBackend.prepare_runtime` raises `NotImplementedError` — but the staged
orchestrators never reach it; they import Neuron directly in the stage subprocess.

**Bug 2 → `619d98b`** `fix(cli): refuse compile/generate/run on non-Trainium backends before
spawning Neuron stages`: `main()` exits 2 with
`'difflet generate' is not available on the 'tpu' backend (no CLI compile/generate runtime; only
Trainium has one). On TPU use 'difflet serve' (Qwen-Image, Wan) or the benchmark runners …`.
Re-verified: Qwen-Image and FLUX both stop in 0.07 s (`generate_*_tpu_gated.log`). Pinned by
`test_generate_refuses_the_tpu_backend_before_spawning_stages`.

## Phase 3 — Wan 2.1 serving on TPU

`difflet serve --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498c… --tp-degree 4
--cp-degree 1 --height 480 --width 832 --num-frames 9 --port 8096` (online: the local snapshot
lacked only `README.md`, which the resolver's default patterns include).

| item | measured |
|---|---|
| startup → `/ready` 200 | 191 s (single transformer, tp=4; startup smoke passed) |
| host RAM peak during load | 67 GB used of 188 (guarded at 30 GB free; never tripped) |
| request A: 832×480×9, 20 steps, guidance 1.0, seed 42 (`/v1/videos/sync`) | **HTTP 200 in 33.5 s**, 177,367 B mp4, frames finite — `serve_wan2_1_tpu.mp4` |
| request B: same, 40 steps, guidance 4.0 (the Trainium matrix's `wan2_1` settings) | **HTTP 200 in 70.4 s**, 224,899 B mp4 — `serve_wan2_1_tpu_g4_s40.mp4` |
| SIGTERM | clean in 10 s, chips released |

**Functionally PASS, but the output is visibly wrong.** Both videos show block/mosaic artifacts
(rectangular tiles of noise on the tree trunks and across the fox at request B; a mosaic column and
painterly smearing at request A). Wan 2.2 through the *same* serving code, VAE, text encoder and
settings as request A produced a clean photoreal frame the same day
(`/mnt/models/teacache_runs/wan22_serve_frame4.png` vs `wan21_frame4.png` / `wan21_g4_frame4.png`).
The two checkpoints' `transformer/config.json`, `scheduler_config.json` and `vae/config.json` differ
only in `_diffusers_version` fields and 2.2's `boundary_ratio` / `transformer_2` — same
architecture (40 layers, 40 heads × 128, 16 latent channels). So the difference is in what the TPU
path does with the 2.1 *weights*, not in the config. Oracle test (single forward vs. diffusers fp32
on CPU, the harness at `/mnt/models/oracle_{ref,cmp}.py` re-pointed at the 2.1 snapshot) in
progress — see below.

Status for the matrix: **BLOCKED (quality)** until the oracle result is in; the serving mechanics
themselves PASS.

## Phase 4 — controls (from the TeaCache campaign, same host, same day)

| model | path | result | evidence |
|---|---|---|---|
| qwen_image | `difflet serve` 1024², 20 steps | PASS — 200 in 7.6 s (cadence 2), bit-identical across restarts | `docs/worklog/2026-09-11-tpu-teacache-device.md` step 3 |
| qwen_image | `benchmark.bench --backend tpu` | PASS — denoise 10.12 s baseline | `benchmark/v5e-teacache-baseline/qwen_image.json` |
| wan (2.2) | `difflet serve` 832×480×9, 20 steps, `/v1/videos/sync` | PASS — 200 in 30.4 s | same worklog, step 3 |
| wan (2.2) | `benchmark/wan_tpu_run.py` | PASS — 12.18 s wall | `/mnt/models/teacache_runs/wan_baseline.json` |

## Bug ledger

| sha | symptom | fix |
|---|---|---|
| `d384b5c` | `difflet serve` on TPU for FLUX/HunyuanVideo resolved or downloaded weights, then died inside the Neuron compile path | backend gate in `resolve_serving_model` |
| `619d98b` | `difflet generate` on TPU died in a stage subprocess with a Neuron `ModuleNotFoundError` wrapped in `CalledProcessError` | backend check in CLI `main()` for compile/generate/run |
