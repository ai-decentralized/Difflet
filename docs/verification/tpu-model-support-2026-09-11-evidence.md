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

- **Qwen-Image, Wan 2.2, Wan 2.1: PASS on the TPU backend** (serving path; Qwen/Wan 2.2 also
  benchmark path). Wan 2.1 was never measured on TPU before this campaign; its serving works and
  its output matches upstream diffusers fp32 to 0.014/px — the visible artifacts are the model's own
  at the 9-frame smoke shape.
- **FLUX, HunyuanVideo, LTX-2: N/A by design at the start** (registry listed no `tpu` backend) — and
  the way they failed was a bug: serving had no backend gate and died inside the Neuron path after resolving or
  downloading weights (`d384b5c`); the CLI died in a stage subprocess with a Neuron import error
  even for the *ported* models (`619d98b`). Both now fail in < 0.2 s with a message naming the
  supported backends / the paths that do run on TPU.
- Porting plan for the three: `docs/plans/2026-09-11-tpu-port-hunyuan-ltx2-flux.md` — **all three
  are now ported** (Phases 5–7): HunyuanVideo and LTX-2 on 2026-09-11/12, FLUX.1-dev on 2026-09-12
  once a token for the gated repo was available. Every serving model runs on the TPU backend.
- Two commits, two bugs, both pinned by unit tests; 560/562 serving tests green in the TPU venv
  (the 2 failures reproduce on `main`, unrelated `test_video_storage` path checks).

## Support matrix — TPU backend (Cloud TPU v5e, tp=4, bf16)

| model | `difflet serve` | benchmark runner | CLI `generate` | TeaCache (probe-free) | notes |
|---|---|---|---|---|---|
| Qwen-Image | **PASS** · 7.6 s / 1024² 20 steps (cadence 2) | **PASS** · harness (Phase 8): resident 8.8 s, 277 ms/step | N/A · no CLI runtime on TPU (fail-fast, `619d98b`) | **PASS** cadence 2 = 0.75×; online-delta LIMIT (slower in natural basis) | |
| Wan 2.2 | **PASS** · 30.4 s / 832×480×9 20 steps | **PASS** · harness (Phase 8): resident 33.2 s, 610 ms/step | N/A | **PASS** cadence 2 = 0.75× | single expert resident ¹ |
| Wan 2.1 | **PASS** · 33.5 s (20 st) / 70.4 s (40 st, CFG) | **PASS** · harness (Phase 8): resident 33.1 s, 613 ms/step | N/A | NOT MEASURED (same controller as 2.2) | quality = upstream ² |
| FLUX.1-dev | **PASS** (port, `tpu-port-hunyuan`) · 10.1 / 9.3 s / 1024² 28 steps, bit-identical to the bench | **PASS** · harness (Phase 8): resident 9.1 s, **187 ms/step** | N/A | **PASS** cadence 2: 9/28 skipped, DiT 0.69×, 0.0041/px | ported 2026-09-12 (option (b): diffusers DiT sharded, Trainium fork untouched); first model faster per step than trn2 |
| HunyuanVideo | **PASS** (port, `tpu-port-hunyuan`) · 191 s / 512×320×61 20 steps (165 s of it host VAE) | **PASS** · 20.1 s denoise, 1.0 s/step (runner; harness row not re-run) | N/A | **PASS** cadence 2 = 0.75×, 0.0064/px | ported 2026-09-11; VAE on chip is the follow-up |
| LTX-2 | **PASS** (port, `tpu-port-hunyuan`) · 51 s / 704×480×49 20 steps | **PASS** · harness (Phase 8): resident 54 s, 1460 ms/step | N/A | **PASS** cadence 2: DiT 0.75×, 0.0088/px | 512×768×121 also fits (VAE 0.7 s on chip); ported 2026-09-12 |

¹ two experts do not fit 16 GB HBM; `benchmark/v5e/wan_2_2.md`. ² block artifacts at 9 frames /
guidance 1.0 are the model's own — reproduced with upstream diffusers fp32 on CPU.

### Feature blocks

**`difflet serve` on TPU**
| FLUX | Qwen-Image | Wan 2.2 | Wan 2.1 | HunyuanVideo | LTX-2 |
|---|---|---|---|---|---|
| PASS · 10.1 s (port) | PASS · 7.6 s | PASS · 30.4 s | PASS · 33.5 s | PASS · 191 s (port) | PASS · 51 s (port) |

**Backend gate (unported model refused before weights)** — new in this campaign
| FLUX | Qwen-Image | Wan 2.2 | Wan 2.1 | HunyuanVideo | LTX-2 |
|---|---|---|---|---|---|
| PASS · 0.17 s (before the port; now ported — the gate is pinned by a test that presents FLUX as Trainium-only) | (ported) | (ported) | (ported) | PASS · 0.16 s (before the port) | PASS · 0.16 s (before the port) |

**Numerical parity vs. upstream diffusers fp32 (single DiT forward on chip)**
| FLUX | Qwen-Image | Wan 2.2 | Wan 2.1 | HunyuanVideo | LTX-2 |
|---|---|---|---|---|---|
| PASS · cos 0.99946 @1024² (vs diffusers' own bf16 0.99974 @256²) | PASS (port acceptance, `oracle_qwen_cmp.log`) | PASS · cos 0.99861 | PASS · cos 0.99967 | PASS · cos 0.99949 | PASS · cos 0.99983 |

## Phase plan

| phase | what | runner | status |
|---|---|---|---|
| 0 | code gates: which models declare `tpu` in `difflet/registry.py` | reading | done |
| 1 | serving fail mode for the three unported models (FLUX, HunyuanVideo, LTX-2) | `difflet serve … ` under `DIFFLET_BACKEND=tpu` | done — bug 1 |
| 2 | CLI fail mode (`difflet generate`) on TPU | `difflet generate …` | done — bug 2 |
| 3 | Wan 2.1 serving on TPU (registered for `tpu`, never measured) | `difflet serve` + `/v1/videos/sync` + oracle + upstream CPU pipeline | done — PASS |
| 4 | controls: Qwen-Image, Wan 2.2 serving on TPU | from the TeaCache campaign | done |
| 5–7 | ports: HunyuanVideo, LTX-2, FLUX.1-dev (oracle + serve + bench + cadence-2 each) | `*_tpu_run.py`, `difflet serve`, `oracle_*_cmp.py` | done — PASS ×3 |
| 8 | every model through `benchmark.bench --backend tpu` on the trn2 protocol (2026-09-12) | `benchmark/adapters/tpu.py` + `tpu_models.py`, `/mnt/models/teacache_runs/bench_v5e/campaign.sh` | PASS ×5; HunyuanVideo not re-run (stopped) |

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

### Oracle: single transformer forward, difflet-on-TPU vs. diffusers fp32 on CPU

Harness: `/mnt/models/oracle_ref.py` / `oracle_cmp.py` (the ones the Wan 2.2 port was accepted
with), copied to `/mnt/models/teacache_runs/oracle_{ref,cmp}21.py` with `WAN` pointed at the 2.1
snapshot. Seeded inputs `[1,16,3,60,104]`, t=500, text `[1,512,4096]`. Logs:
`/mnt/models/teacache_runs/oracle_wan21_{ref,cmp}.log`.

| difflet TPU attention | cos vs fp32 | rel-L1 | cos vs diffusers bf16 | rel-L1 |
|---|---|---|---|---|
| fused (Pallas) | 0.99967456 | 2.47e-02 | 0.99974883 | 2.03e-02 |
| sdpa | 0.99967539 | 2.47e-02 | 0.99975008 | 2.03e-02 |
| control: diffusers bf16 vs its own fp32 | 0.99971920 | 2.30e-02 | | |

(For comparison the Wan 2.2 acceptance run measured 0.99861 / 5.26e-02 against a 0.99873 control.)
The TPU forward of the 2.1 checkpoint is as close to upstream as upstream's own bf16 is: **weights,
sharding, RoPE and both attention paths are correct for Wan 2.1**. CPU reference forward 56 s fp32 /
257 s bf16, host peak 29 GB; TPU comparison host peak recorded in the cmp log.

So the artifacts are not a forward-pass defect. Remaining candidates: (a) the pipeline layer on TPU
does something 2.1-specific wrong (scheduler / guidance handling — the code paths are shared with
2.2, which is clean), or (b) Wan 2.1 14B simply produces this at 9 frames / 480p (it was trained for
81 frames; the Trainium validation of 2.1 serving, `docs/design/t2v_serving/09_wan21_trn2_serving_validation.md`,
checked only that the MP4s decode, not what they look like). Decided by running the **upstream
diffusers `WanPipeline` on CPU fp32** at request A's settings.

### Upstream pipeline on CPU: the artifacts are the model's

`artifacts/verification-2026-09-11-tpu/wan21_cpu_ref_pipeline.py` — diffusers `WanPipeline`
from the same snapshot, fp32, CPU (112-thread EPYC), same prompt / 832×480×9 / 20 steps /
guidance 1.0 / seed 42. 20 steps at 53–56 s each = 1101 s; host peak 17 GB (mmap'd weights).
Log `wan21_cpu_ref.log`.

| | |
|---|---|
| frame 4, upstream CPU fp32 | `wan2_1_upstream_cpu_fp32_frame4.png` |
| frame 4, difflet TPU serving (request A) | `wan2_1_tpu_serve_frame4.png` |
| stacked | `wan2_1_cpu_vs_tpu_side.png` — **the same frame, mosaic column and all** |
| upstream vs TPU, 9 frames (TPU side read back through H.264) | mean abs **0.0141/px**, PSNR 32.5 dB, per-frame 0.013–0.017 |

That agreement is the bf16-vs-fp32 + codec gap (the Wan 2.2 TeaCache A/B measured 0.0126/px for a
5-step-skip change on the same shape), and the seed reproduces identically, so difflet's latent
RNG path matches diffusers too. **Conclusion: Wan 2.1 on TPU is correct end-to-end. The blocky
output is what Wan 2.1-T2V-14B produces at 9 frames / guidance 1.0 (it is an 81-frame model);
the Trainium serving validation ran the same settings and would have seen the same frames.**

Status for the matrix: **PASS** (serving + benchmark-grade parity). Operator note: for usable
Wan 2.1 output use its documented settings (guidance ~5, 81 frames), which are a different
HBM/latency budget than the 9-frame smoke shape.

## Phase 4 — controls (from the TeaCache campaign, same host, same day)

| model | path | result | evidence |
|---|---|---|---|
| qwen_image | `difflet serve` 1024², 20 steps | PASS — 200 in 7.6 s (cadence 2), bit-identical across restarts | `docs/worklog/2026-09-11-tpu-teacache-device.md` step 3 |
| qwen_image | `benchmark.bench --backend tpu` | PASS — denoise 10.12 s baseline | `benchmark/v5e-teacache-baseline/qwen_image.json` |
| wan (2.2) | `difflet serve` 832×480×9, 20 steps, `/v1/videos/sync` | PASS — 200 in 30.4 s | same worklog, step 3 |
| wan (2.2) | `benchmark/wan_tpu_run.py` | PASS — 12.18 s wall | `/mnt/models/teacache_runs/wan_baseline.json` |

## Phase 5 — HunyuanVideo ported to TPU (branch `tpu-port-hunyuan`, same day)

The plan's first port (`docs/plans/2026-09-11-tpu-port-hunyuan-ltx2-flux.md`), done after the
matrix above was written. Same host, 320×512×61, tp=4, bf16.

| check | result | evidence |
|---|---|---|
| single-forward parity vs diffusers fp32 (CPU) | **cos 0.99949 / rel-L1 3.08e-2** fused; 0.99949 / 3.09e-2 SDPA; control (diffusers bf16 vs fp32) 0.99946 / 3.15e-2 | `oracle_hunyuan_cmp2.log`, `oracle_hunyuan_ref.log` |
| first attempt, before `c7b9d09` | cos **0.9086** (rel-L1 0.57), fused == SDPA → TPU `RowParallelLinear` ignored `reduce_output`/`skip_bias_add` (double reduce, bias × tp in all 40 single blocks) | `oracle_hunyuan_cmp.log` |
| HBM | 10.84 GB resident, 12.5 GB peak of 15.75 (per-rank shard 5.8 B params, 3.5 B of them replicated adaLN linears) | oracle + bench logs |
| DiT step | **1.006 s** synced == natural (host-side Euler step syncs anyway); SDPA path 2.0 s | `hunyuan_baseline.json` |
| first compile | 100 s alone; 168–174 s under the 2-slot gate with 4 ranks; host RSS ~43 GB per compiling rank (peak 174 GB ungated → 118 GB gated in serving) | serve/bench logs |
| `difflet serve` (`--host-vae`) | ready in **372 s**; 20-step `POST /v1/videos/sync` **200 in 191 s**, 457 KB mp4, 61 finite frames, coherent motion | `serve_hunyuan_video_tpu_port.log`, `serve_hunyuan_video_tpu.mp4`, `hunyuan_video_tpu_frames_0_30_60.png` |
| benchmark runner | denoise 20.1 s (20 steps), Llama encode 4.1–7.0 s (fp32 on ordinal 0 + broadcast), **host VAE decode 167 s** | `hunyuan_baseline.{json,log}`, `hunyuan_video_tpu_bench_baseline.mp4` |
| TeaCache cadence 2 | **15 full / 5 skipped** every iteration; denoise **15.07 s vs 20.12 s = 0.749×**, per-full-step 1.004 s unchanged, HBM unchanged; video vs baseline **mean abs 0.0064/px, PSNR 36.9 dB** (per-frame 0.004–0.012), frame 30 side by side indistinguishable | `hunyuan_cadence2.{json,log}`, `hunyuan_video_tpu_bench_cadence2.mp4`, `hunyuan_video_tpu_baseline_vs_cadence2_f30.png` |

Where the 191 s request goes: ~20 s denoise + ~5 s text encode + ~165 s host VAE decode of 61
frames (`AutoencoderKLHunyuanVideo`, fp32, tiled, 112-thread EPYC). VAE-on-chip is the port's
biggest follow-up; the DiT itself is 1 s/step.

Bugs found by the port's on-device passes (each its own commit, each pinned):
`c7b9d09` RowParallelLinear kwargs · missing `await` in the TPU runner builder · non-primary
replicas validating a smoke file they never wrote · `8403351` benchmark ranks exiting while
rank 0 still decodes (TPU runtime kills the straggler, exit 1, no traceback).

LTX-2's (Phase 6): `5bca57d` unmasked text cross-attention · warmup via the bare module ·
prompt-encoder `dtype=` · `e03df45` bf16 host pipeline (emulated) · `f3aa906` device-VAE
wrapper without `latents_mean/std` · plus the two serving-shape facts: the 121-frame host
decode and the cold Gemma read both exceed the default 900 s startup budget.

## Phase 6 — LTX-2 ported to TPU (branch `tpu-port-hunyuan`, 2026-09-12)

Second port from the plan. 512×768×121 (the registry default), tp=4, bf16; 6144 video +
126 audio + 1024 text tokens.

| check | result | evidence |
|---|---|---|
| sharded build vs checkpoint | 3510 parameters map 1:1 to diffusers' keys; **4.98 B params / 9.3 GiB per rank** (4.63 B sharded, 0.35 B replicated) | dry run in the port commit |
| HBM fit (random inputs) | **9.28 GB resident, 9.36 GB peak** of 15.75; first compile 73 s; **1.68 s/step**; host peak 59 GB | `ltx2_fit_probe.log` |
| parity vs diffusers fp32 (CPU, 984/1024 text positions padded), first attempt | video cos **0.9932** / rel-L1 0.131, audio 0.946 — the shared TP attention processor runs text cross-attention *unmasked* (a Trainium kernel limitation) | `oracle_ltx2_cmp.log` |
| parity after honoring the mask on TPU (`5bca57d`) | video **cos 0.99983 / rel-L1 1.83e-2**, audio 0.99957; vs diffusers bf16 0.99985; control (diffusers bf16 vs fp32) 0.99983 / 1.78e-2 — indistinguishable from upstream's own bf16 | `oracle_ltx2_cmp2.log`, `oracle_ltx2_ref.log` (fp32 forward 86 s, bf16 control 413 s, 18.88 B params) |
| `difflet serve`, 512×768×121 | takes 1–3: one-line slips (warmup called the bare module positionally; prompt-encoder wrapper rejected diffusers' `dtype=`; missing `await` class); take 3 **timed out at 900 s** loading Gemma-3 (cold first read of the 46 GB checkpoint — 2.5 s once the page cache is warm); take 4 timed out at 1800 s *inside the smoke*: the fp32 host decode of 121 frames on every replica (standalone: not finished in 14 min) | `serve_ltx2_tpu.log` (overwritten per take) |
| video VAE on the chip | `AutoencoderKLLTX2Video` under torch_xla: **480×704×49 in 0.2 s** (54.6 s first compile, 2.98 GB), **512×768×121 in 0.7 s** (29 s compile, 3.99 GB), finite; numerics vs host fp32 on the real latents bf16 0.0048/px, fp32 0.0006/px | `ltx2_vae_device_probe*.log`, `ltx2_vae_ctor_probe.log` |
| the fp32-host finding | the host pipeline had been loaded in the DiT's bf16 → bf16 is *emulated* on the EPYC host: audio VAE decode **114 s**, vocoder **196 s**, connectors ~40 s per prompt (vs 0.1 s / 0.6 s / ~0 in fp32). e2e-to-latents 93.6 → 50.7 s | `e03df45`, runner decode split |
| the VAE-buffer finding | first videos off the chip: right structure, cyan cast + dithered grid; same latents decoded on the host: a proper red fox. `_denormalize_ltx_2_video_latents` reads `vae.latents_mean/latents_std` as *attributes* (buffers) and skips when absent — the device wrapper exposed only `config`. Device-vs-host on the same latents **0.1713 → 0.0024/px** | `f3aa906`, `ltx2_wrapper_probe{,2}.log`, `ltx_2_vae_buffer_bug_device_vs_host_f24.png` |
| **benchmark, 480×704×49 (the trn2 MATRIX row), 20 steps, guidance 1.0** | e2e-to-latents **49.6 / 48.2 s** synced, 49.5 s natural; **DiT 1453 ms/step** (29 s of the 49; the rest is Gemma-3 fp32 on 1024 tokens); decode: video VAE 0.2 s warm (51.8 s first compile), audio VAE 0.11 s, vocoder 0.56 s; HBM peak 12.3 GB (DiT + VAE on rank 0); latents finite | `ltx2_baseline.{json,log}`, `ltx_2_tpu_bench_baseline.mp4`, `ltx_2_tpu_frames_0_24_48.png` |
| TeaCache cadence 2 | **15 full / 5 skipped**; e2e-to-latents **42.4 / 43.3 / 42.0 s** (DiT 29 → 22 s = 0.75×; 0.86× overall because the encode is fixed); video vs baseline mean abs 0.0088/px, PSNR 37.9 dB | `ltx2_cadence2.{json,log}`, `ltx_2_tpu_bench_cadence2.mp4` |
| quality note | at guidance 1.0 / 20 steps (the timing row) the subject is muddy; at guidance 3.0 the host decode of the same pipeline's latents shows a clear red fox — the model at CFG-free settings, not the port (trn2 also only judged quality at 40 steps with CFG) | `ltx2_g3*` |
| **`difflet serve`, 480×704×49** (device VAE, primary-only decode, fp32 host) | **ready in 246 s** (warm cache; 595 s on the earlier take with the VAE compile in the smoke); `POST /v1/videos/sync` 20 steps → **200 in 51.6 s and 50.2 s**, 705 067 B mp4, 49 finite frames, **bit-identical across the two requests**; SIGTERM clean in 10 s | `serve_ltx_2_tpu_port.log`, `serve_ltx_2_tpu.mp4`, `ltx2_campaign.log` |

Structure: `difflet/models/ltx_2/tp_sharding.py` (the Trainium recipe, lifted unchanged, `365315b`),
`backends/tpu/ltx_2/{config,transformer}.py`, `models/ltx_2/tpu_application.py` (Gemma on
ordinal 0 + `collective_broadcast` of the packed [1, 1024, 188160] embeddings), serving adapter
TPU branch, registry flip. Venv fix: torchvision was the CUDA wheel (`torchvision::nms does not
exist` → Gemma3 import failed); now `0.24.0+cpu`.

## Phase 7 — FLUX.1-dev ported to TPU (branch `tpu-port-hunyuan`, 2026-09-12)

Third and last port. Option (b) of the plan: diffusers' `FluxTransformer2DModel` TP-sharded per
rank; the Trainium path (the legacy NxDI fork in `modeling_flux.py`) is untouched. 1024×1024,
tp=4, bf16; 4 096 image + 512 text tokens. The repo is gated: nothing ran on the chips with real
weights until the token arrived at 04:11, so the port was built and pinned weight-free first.

| check | result | evidence |
|---|---|---|
| module vs diffusers, CPU | at tp=1 the rewritten module (TP attention processor, split `proj_out`, parallel linears) equals diffusers' forward to 1e-5; the device Euler loop equals `FlowMatchEulerDiscreteScheduler.step` to 1e-6; packing / sigmas+mu / PIL postprocess equal diffusers' helpers; cadence 2 skips 5/20 | `tests/unit/models/flux/test_flux_{tp_sharding,tpu_application}.py`, `0389f74` |
| synthetic sharded checkpoint on 4 chips (no FLUX weights) | seeded 2+2-block model saved in the diffusers layout, loaded through the real sharded path incl. the `proj_out` `CheckpointSlice` windows: device tp=4 **fp32 vs CPU fp32 cos 0.9999995**; bf16 vs bf16 0.99991 (its own bf16-vs-fp32 control is 0.816 — a badly conditioned random model) | `flux_synth_parity.{json,log}` |
| full geometry, random weights (no FLUX weights) | **5.445 B params / 10.18 GB HBM per rank** at 1024²/tp=4 (2.4 B replicated: the adaLN modulation linears); first compile 44 s (2-slot gate, 62 GB host peak); **0.335 s per isolated forward** | `flux_synth_timing.{json,log}` |
| weights | `black-forest-labs/FLUX.1-dev` @ `3de623fc` (the pinned trn2 revision), 34 GB incl. T5-XXL, 04:11–04:13 | `flux_download.log` |
| parity vs diffusers fp32 (CPU, seeded random inputs) | 1024²/512: **cos 0.99946 / rel-L1 0.030** (vs diffusers' own bf16 **0.99986 / 0.014**; control bf16-vs-fp32 0.99946 / 0.029, i.e. identical to the device's distance from fp32); 256²/64: vs fp32 0.99019 / 0.135, **vs diffusers' own bf16 0.99974 / 0.022**, control bf16-vs-fp32 0.98974 / 0.138 — the device is closer to upstream bf16 than either is to fp32; sharded checkpoint load 7 s/rank | `oracle_flux_ref{,_small}.log`, `oracle_flux_cmp{,_small,_bf16}.log` |
| **benchmark, 1024², 28 steps, guidance 3.5** (the trn2 MATRIX row) | load 52–59 s; first request 36.6 s (loop-graph compile); **e2e-to-latents 8.72 s synced / 8.59 s natural** (T5-XXL fp32 3.3 s + 28 × 0.19 s); **DiT 187 ms/step** synced, 152 ms natural; VAE on chip: 117 s first compile, 0.31 s warm; HBM 10.18 resident / 10.45 GB peak; a clean red fox at golden hour | `flux_baseline.{json,log}`, `flux_1_dev_tpu_bench_baseline.png` |
| TeaCache cadence 2 | **19 full / 9 skipped** (window [5, 23) of 28); e2e-to-latents **7.63 / 7.09 s** (DiT 5.39 → 3.73 s = 0.69×); image vs baseline mean abs 0.0041/px, PSNR 40.7 dB | `flux_cadence2.{json,log}`, `flux_1_dev_tpu_bench_cadence2.png`, `flux_1_dev_tpu_baseline_vs_cadence2.png` |
| **`difflet serve`, 1024²** (TPU branch: no artifacts, eager app, VAE on the primary chip) | **ready in 216 s** (4 replicas incl. smoke with the VAE compile); `POST /v1/chat/completions` 28 steps → **200 in 10.1 s and 9.3 s**, 1 118 780 B PNGs **bit-identical to each other and to the benchmark's**; SIGTERM clean in 10 s | `serve_flux_tpu_port.log`, `serve_flux_tpu.png`, `flux_campaign_b.log` |
| bugs | none on device: the port ran first time on the real weights (the CPU-pinned tests caught the two slips — a rope-axes typo in a test config and a `DownloadPolicy` enum name — before any chip time) | — |

Structure: `difflet/models/flux/tp_sharding.py` (head-sharded attention with the per-head qk
RMSNorm and the shared per-position RoPE left local, column→row FFNs, single-block `proj_out`
split into two row-parallel halves reduced once — HunyuanVideo's split on diffusers' module),
`backends/tpu/flux/{config,transformer}.py`, `models/flux/tpu_application.py` (T5-XXL fp32 on
ordinal 0 + broadcast, CLIP-L per rank, `TpuDeviceImageVae`, device-resident loop with the
probe-free controller), the TPU branch of `serving/orchestrators/flux.py`, registry
`backends=("trainium","tpu")`, `--teacache-cadence/--teacache-online-delta` accepted for flux on
the TPU backend only, `benchmark/flux_tpu_run.py`.

## Phase 8 — every model through one harness on the trn2 protocol (2026-09-12)

Question: *are the v5e rows measured the way the trn2 rows are, and does every ported model
still run end to end?* Until this phase they were not: Qwen-Image went through
`benchmark.bench` (a Qwen-only TPU adapter), the other four through standalone runners with
no page-cache drop, no process restart for "warm", decode only on some, and FLUX on its own
prompt. `benchmark/adapters/tpu.py` is now model-generic (one driver per model type in
`benchmark/adapters/tpu_models.py`, each loading exactly what `difflet serve` loads), and
`benchmark/bench.py` carries the trn2 protocol for every backend: page cache dropped → cold
fresh process → 1 discarded warm-up → 3 warm fresh processes → 2 resident synced requests
(the per-step source) → 2 natural requests. Every run decodes on the primary replica and the
decoded tensor is range/finite-checked like trn2's `.pt` outputs; media go to
`artifacts/benchmark-v5e-2026-09-12/`.

Plumbing smoke first (`/mnt/models/teacache_runs/bench_v5e/smoke.py`, 2 steps, fresh + resident,
decoded; FLUX's and Qwen's resident 2-step requests here include the ~20–30 s natural-basis recompile that the `_sync` fix below removed): 

| model | fresh process (load + 2-step request, decoded) | load | resident 2-step request | encode / denoise / decode | HBM peak | decoded output |
|---|---:|---:|---:|---|---:|---|
| flux_1_dev | 207 s | 52 s | 28.4 s | 4.0 / 24.1 / 0.3 s | 10.5 GB | (1×3×1024×1024) finite=True |
| qwen_image | 165 s | 81 s | 35.7 s | 1.9 / 32.7 / 1.1 s | 11.1 GB | (1×3×1024×1024) finite=True |
| wan_2_1 | 158 s | 53 s | 22.0 s | 0.6 / 1.2 / 20.1 s | 7.5 GB | (1×3×9×480×832) finite=True |
| wan_2_2 | 174 s | 83 s | 22.0 s | 0.7 / 1.2 / 20.1 s | 7.5 GB | (1×3×9×480×832) finite=True |
| ltx_2 | 530 s | 206 s | 25.0 s | 20.7 / 3.4 / 0.9 s | 12.5 GB | (1×49×3×480×704) finite=True |
| hunyuan_video | 428 s | 227 s | 173.8 s | 4.6 / 2.0 / 167.1 s | 12.5 GB | (1×3×61×320×512) finite=True |

All six PASS (`/mnt/models/teacache_runs/bench_v5e/smoke_summary.json`, media in `smoke_out/`).

Campaign (`/mnt/models/teacache_runs/bench_v5e/campaign.sh`, logs `benchmark/v5e/logs/<slug>_bench.log`):

| model | v5e cold | v5e warm (n=3) | v5e resident request | v5e per-step (synced, n) | trn2 warm / warm−load / per-step | output (v5e) | media identical across the 9 runs |
|---|---:|---:|---:|---:|---|---|---|
| flux_1_dev | 261 s | 211.7 s | **9.1 s** | **187 ms** (n=54) | 35.3 s / 17.5 s / 268 ms | (1×3×1024×1024) finite | yes (md5 325f1405…) |
| qwen_image | 172 s | 92.4 s | **8.8 s** | **277 ms** (n=38) | 62.7 s / 32.9 s / 447 ms | (1×3×1024×1024) finite | yes (md5 287e425e…) |
| wan_2_1 | 197 s | 92.9 s | **33.1 s** | **613 ms** (n=38) | 56.4 s / 28.5 s / 555 ms | (1×3×9×480×832) finite | yes (md5 e0ec224d…) |
| wan_2_2 | 192 s | 93.8 s | **33.2 s** | **610 ms** (n=38) | 57.4 s / 28.4 s / 555 ms | (1×3×9×480×832) finite | yes (md5 ec46bca5…) |
| ltx_2 | 494 s | 268.8 s | **54.0 s** | **1460 ms** (n=38) | 58.5 s / 43.8 s / 438 ms | (1×49×3×480×704) finite | yes (md5 c94fa61c…) |
| hunyuan_video | — | — | — | — | 144.3 s / 101.5 s / 851 ms | (1×3×61×320×512) finite (smoke) | **not re-run**: stopped at the user's request 07:21 UTC during the cold process; port-era 1006 ms/step, 191 s served |

PASS ×5 on the unified protocol (`benchmark/v5e/<slug>.{json,md}`, commits `2582646` FLUX, `5c7d75d` Qwen, `1a9460b` Wan 2.1, `ada3a75` Wan 2.2, `197a782` LTX-2); HunyuanVideo NOT MEASURED on it. Cross-device tables: `benchmark/v5e/RESULTS.md` (generated by `benchmark/cross_device.py`).

Method differences that surfaced while unifying (all pinned or documented):

| finding | where | consequence |
|---|---|---|
| `snapshot_download(local_files_only=True)` refuses every checkpoint on this host (`IncompleteSnapshotError: .gitattributes missing`) because they were fetched with `allow_patterns` | `TpuAdapter._resolve_model_dir` | pinned revisions are resolved straight in the hub cache; `snapshot_download` only as fallback |
| a `mark_step` inside the per-step timer splits the step graph at the DiT output → a different executable from the natural/serving one → recompile on the first natural request (FLUX: 24 s denoise for 2 steps) | `tpu.py::_sync` | sync is `xm.wait_device_ops()` alone, as `RealLoopStepTimer` documents; every loop already `mark_step`s per step |
| the cold generate's per-step deltas on a lazy backend can still carry compiles past step 0 | `bench.py` | `step_latency` comes from the resident synced requests only; the cold deltas are the fallback for adapters without a resident mode (trn2, diffusers) |
| a Qwen DiT load read 79 s when the page cache held other checkpoints (8 s warm on 2026-08-24) | smoke | exactly why cold/warm are separate fresh-process rows |

## Bug ledger

| sha | symptom | fix |
|---|---|---|
| `d384b5c` | `difflet serve` on TPU for FLUX/HunyuanVideo resolved or downloaded weights, then died inside the Neuron compile path | backend gate in `resolve_serving_model` |
| `619d98b` | `difflet generate` on TPU died in a stage subprocess with a Neuron `ModuleNotFoundError` wrapped in `CalledProcessError` | backend check in CLI `main()` for compile/generate/run |
