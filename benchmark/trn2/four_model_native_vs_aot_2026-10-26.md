# Four models, native vs AoT at TP4 on trn2.3xlarge, 2026-10-26

**Status:** phases 1 and 2 of the plan in `docs/superpowers/plans/2026-10-26-four-model-native-vs-aot-metrics.md`. All eight device cells (four models, native and AoT) have numbers:
- **AoT**, all four models, measured on the latest `origin/main`: Qwen-Image and LTX-2 in phase 1, HunyuanVideo 1.0 and Wan2.1 in phase 2.
- **LTX-2 native:** a full cell under cold definition A, measured in phase 2.
- **Qwen-Image, HunyuanVideo and Wan2.1 native:** numbers reused from earlier cells on this host. Phase 2 confirmed them at the current branch HEAD by bit-identical outputs (plan D10, option A).

PSNR / SSIM: **Qwen-Image and LTX-2 have numbers** (phase 4, two 50-step CPU fp32 references each, sections "Phase 4: Qwen-Image CPU fp32 references" and "Phase 4: LTX-2 CPU fp32 references", collected with the tool sanity checks and a look at the references in "Quality vs CPU fp32 reference (part 1)"). Wan2.1 (both backends, one shared reference) and HunyuanVideo native have numbers too (section "Quality vs CPU fp32 reference (part 2)"). HunyuanVideo AoT has numbers too, from its AoT-noise reference run on 2026-10-27 on the replacement host (19.65 dB / 0.774, section "HunyuanVideo AoT vs its CPU fp32 reference"). All 8 cells now have PSNR / SSIM. Cold e2e is definition A in the tables above. Definition B (native without its weight store, Wan2.1 and LTX-2 only, plan phase 6) was measured on 2026-10-27 on a replacement host with the same root volume: Wan2.1 762.55 s and LTX-2 785.40 s, against A 533.89 s and 603.23 s on that host (section "2026-10-27, new host").

**Phase 3 supplement (2026-10-27/28, new host):** native Qwen-Image and Wan2.1 re-measured with the opt-in bf16 VAE (and bf16 umT5 for Wan) after every accuracy gate passed. Wan2.1 cold e2e A drops to 454.7 s (533.89 s fp32 on that host) and its peak HBM to 11.83 GiB per core, at 22.41 dB / 0.880 against the reference (fp32: 22.66 / 0.885). Qwen-Image's latents are bit-identical and its quality is unchanged (section "Phase 3: bf16 VAE and umT5"). The fp32 numbers above stay the headline, and bf16 is not a default.

**Device:** trn2.3xlarge: 1 device, 4 NeuronCores (LNC=2, 24 GiB HBM per core), 12 vCPU, 124.8 GiB RAM, one EBS root volume. The host is the same one used for `native_vs_aot_tp4_2026-10-25.md`.

**Raw evidence:** kept out of git, under `/home/ubuntu/bench-2026-10-26-4m/` on the benchmark host (`B` below).
- Each run has its own directory `B/<model>/<cell>/<run>/`, which holds:
  - `run.log`: the CLI output and `/usr/bin/time -v`, plus a header with the date, worktree, HEAD, `difflet.__file__`, the extra env and the exact command;
  - `summary.json`: the harness record (command, env, HEAD and dirty flag, meminfo, checkpoint page-cache residency, the pre-read record, the probe timeline, HLO modules, `difflet_file_precheck` and `difflet_files_seen`, `nmon_peak_gib`, `device_released`);
  - `probe/`: per-process events;
  - `nmon.jsonl`: neuron-monitor, one sample per second;
  - the outputs.
- `B/<model>/summary.jsonl` has one line per run.
- `B/<model>/<cell>/stats_gen1-3.txt` comes from `B/scripts/bench4m_stats.py`.
- `B/<model>/<cell>/hlo_allreduce.jsonl` comes from `B/scripts/hlo_allreduce.py`.

Every number below comes from a run made for this report and names its run directory, unless it is marked *reused*. Inferences are labelled.

## Summary: four models x two backends (TP4, 50 steps, seed 42)

All values were re-read from the run records (`summary.json`, `stats_gen1-3.txt`, `cell_record.json`) on 2026-10-26 after phase 2.
- **DiT step:** median of the per-run medians over gen1-3, step 0 excluded.
- **Warm e2e:** mean of gen1-3, with gen0 discarded.
- **Cold e2e A:** page cache dropped. Compile-time weight artifacts are allowed: the native weight store and the AoT pre-shards.
- **Compile:** a cold `difflet compile` on a fresh root, with the checkpoint already in the page cache.
- **Peak HBM:** neuron-monitor's per-core peak.
- **PSNR / SSIM:** the device q1 output against a 50-step CPU fp32 reference (diffusers) started from that backend's own initial latents. RGB uint8, SSIM per Wang 2004; latent cosine on the final packed latents. Not a native-vs-AoT comparison: each backend has its own noise and its own reference.

| model | backend | DiT step | warm e2e | cold e2e A | compile | peak HBM per core | PSNR / SSIM | date, commit, evidence |
|---|---|---:|---:|---:|---:|---|---|---|
| HunyuanVideo 1.0, 320x512x61, g6.0 | native | 808.2 ms | 126.7 s | 457.8 s | 274.5 s | 17.95 GiB on the decoder core, 12.98 on the other three | 20.78 dB / 0.869 (latent cosine 0.982), against the CPU fp32 reference from native's noise | *reused*: 2026-10-25/26 at `eadddec`, `/home/ubuntu/bench-2026-10-26/hunyuan_video/s4_tp4/`. Confirmed by hash on 2026-10-26 at `4f5d46a`: `B/hunyuan_video/native_tp4_hashcheck/` |
| | AoT | 801.8 ms | 133.0 s | 617.5 s | 4526.7 s | 15.53 GiB | 19.65 dB / 0.774 (latent cosine 0.969), against the CPU fp32 reference from AoT's noise (2026-10-27, new host) | 2026-10-26 at main `540e5f0`: `B/hunyuan_video/aot_main_tp4/` |
| Qwen-Image, 1024x1024, g4.0 | native | 359.0 ms | 82.3 s | 502.0 s | 187.0 s | 10.81 GiB on the DiT cores, 13.85 on the VAE core | 16.11 dB / 0.823 (latent cosine 0.872), against the CPU fp32 reference from native's noise | *reused*: 2026-10-25 at `4f6e1de`, `/home/ubuntu/bench-2026-10-25/qwen_image/native_tp4/`. Confirmed by hash on 2026-10-26 at `a87d89c`: `B/qwen_image/native_tp4_hashcheck/` |
| | AoT | 360.6 ms | 73.7 s | 515.6 s | 1209.9 s | 10.10 GiB | 21.48 dB / 0.912 (latent cosine 0.961), against the CPU fp32 reference from AoT's noise | 2026-10-26 at main `540e5f0`: `B/qwen_image/aot_main_tp4/` |
| Wan2.1 14B, 480x832x9, g5.0 (CFG) | native | 1063.6 ms | 128.5 s | 542.7 s (weight store) | 600.3 s | 15.15 GiB on the DiT cores, 18.40 on the VAE core | 22.66 dB / 0.885 (latent cosine 0.967), against the one CPU fp32 reference both backends share (same noise) | *reused*: 2026-10-25 at `8b2408c`, `/home/ubuntu/bench-2026-10-25/wan2_1/native_tp4/`. Confirmed by hash on 2026-10-26 at `4f5d46a`: `B/wan2_1/native_tp4_hashcheck/` |
| | AoT | 956.4 ms | **123.16 s** with `--host-vae` (A2) | **449.4 s** with `--host-vae` (A2) | 5935.7 s (the transformer stage alone: 430.5 s) | 10.34 GiB | 21.40 dB / 0.853 (latent cosine 0.954) for the A2 host fp32 decode, against the same shared reference. The device-VAE frames are corrupted: 11.72 dB / 0.321 against the reference, 12.41 dB against a host fp32 decode | 2026-10-26 at main `540e5f0`: `B/wan2_1/aot_main_tp4_twocore/` and `…_hostvae/` |
| LTX-2, 480x704x49, g1.0 | native | 350.5 ms | 92.46 s | 594.76 s (Gemma-3 weight store) | 327.59 s | 16.51-16.52 GiB | 18.06 dB / 0.811 (latent cosine 0.930 video, 0.989 audio), against the CPU fp32 reference from native's noise | 2026-10-26: compile at `999c434`, the other runs at `009d3dd`. `B/ltx_2/native_tp4/` |
| | AoT | 400.1 ms | 66.55 s | 778.1 s | 1052.2 s | 9.81 GiB | 16.21 dB / 0.660 (latent cosine 0.725 video, 0.980 audio), against the CPU fp32 reference from AoT's noise. Far below the 2026-10-24 single-core 20-step run (cosine 0.991). Not two-core attention: a two-core 20-step run on 2026-10-27 is byte-identical to that single-core run (cosine 0.9907, 35.45 dB / 0.979 against the 20-step reference), so the drop comes with the 50 steps (section "2026-10-27, new host") | 2026-10-26 at main `540e5f0`: `B/ltx_2/aot_main_tp4_twocore/` |

**Native vs AoT**, computed as (native − AoT) / AoT from the unrounded run values:

| model | DiT step | warm e2e | cold e2e A | compile | peak HBM per core |
|---|---:|---:|---:|---:|---|
| HunyuanVideo 1.0 | +0.8% | −4.7% | −25.9% | −93.9% | +15.6% on the decoder core, −16.4% on the others |
| Qwen-Image | −0.4% | +11.6% | −2.6% | −84.5% | +7.0% on the DiT cores |
| Wan2.1 | +11.2% | +4.4% (against `--host-vae`) | +20.8% (against `--host-vae`) | −89.9% (+39.5% against the transformer stage alone) | +46.5% on the DiT cores |
| LTX-2 | −12.4% | +38.9% | −23.6% | −68.9% | +68.4% |

**What each comparison includes (details in the model sections)**
- **Placements differ between the backends** (plan D8):
  - Native encodes the Qwen-Image and HunyuanVideo prompts on rank 0's host; AoT encodes them on the device.
  - LTX-2 native runs Gemma-3 on the device, which is why its HBM is higher; LTX-2 AoT runs Gemma-3 on the host.
  - The Wan2.1 AoT e2e figures use `--host-vae`, an fp32 host decode, because main's compiled bf16 device VAE corrupts its frames (Wan2.1 section). Its compile figure is main's default compile, which includes 5486.1 s for that device-VAE stage.
- **Kept in fp32 on native** (plan decision 3, second round): Wan umT5, the Wan VAE and the Qwen-Image VAE.
- **Noise:**
  - Qwen-Image, HunyuanVideo and LTX-2 start from different seed-42 noise on the two backends, so their images and videos are not comparable pixel for pixel.
    - Qwen-Image and LTX-2 draw the noise in bf16 on both backends, and torch 2.14 and torch 2.9.1 give different bf16 draws.
    - HunyuanVideo native draws the noise in fp32, while AoT draws it in bf16.
  - Wan2.1 starts from identical noise on both backends.

## Method

**Code under test**
- **AoT:** `origin/main` at `540e5f086ffa5dae8e957aa94b1b23616091a7bc`, checked out as a separate detached worktree at `/home/ubuntu/aot-main`. The tree was clean in every run (`head: 540e5f0… dirty=0` in each `run.log`).
  - The five bf16 all-reduce fixes are ancestors of that commit: `fc6a580`, `b4bdf35` (Qwen-Image), `4d4cc60` (LTX-2), `7fe5308` (HunyuanVideo) and `1c8a64e` (Wan umT5). Each was checked with `git merge-base --is-ancestor`, exit 0.
  - The NxD venv `/home/ubuntu/venvs/difflet-neuron` was used with `PYTHONPATH=B/probe:/home/ubuntu/aot-main` and cwd `/home/ubuntu/aot-main`.
  - Before every run, the harness imports `difflet` with the run's exact env and refuses to start if `difflet.__file__` is not under `/home/ubuntu/aot-main` (`difflet_file_precheck`). The probe also records `difflet.__file__` in every process (`difflet_files_seen`).
  - Both fields read `/home/ubuntu/aot-main/difflet/__init__.py` in all 33 completed AoT runs: 14 in phase 1, then 7 for HunyuanVideo, 7 for Wan2.1 and 5 for the Wan2.1 `--host-vae` supplement in phase 2 (`B/<model>/summary.jsonl`). That rules out the venv's editable install, which points at another worktree.
- **Native:** branch `feat/hunyuan-neuron` in `/home/ubuntu/Difflet/.claude/worktrees/hunyuan-neuron`, with the native venv `/home/ubuntu/venvs/native-312`.
  - Every run uses `PYTHONPATH=B/probe:<worktree>`, the venv's `bin` on PATH, and `DIFFLET_COMPILE_CACHE=$R` plus `--cache-dir $R`.
  - The same `difflet_file_precheck` / `difflet_files_seen` check applies. In every native run of phase 2, both fields read `/home/ubuntu/Difflet/.claude/worktrees/hunyuan-neuron/difflet/__init__.py`.
  - The HEAD of each run is in its `run.log`, and the tree was clean (`dirty=0`) in all of them:
    - **LTX-2 cell:** the compile ran at `999c434`; cold, gen0-3 and q1 ran at `009d3dd`. The only difference between the two commits is plan files (`git diff --stat 999c434 009d3dd`: two files under `docs/superpowers/plans/`). Both commits contain F12 `05614a2`, the fix for the LTX-2 text-encoder index (`git merge-base --is-ancestor`, exit 0).
    - **Hash checks:** Qwen-Image at `a87d89c`, HunyuanVideo and Wan2.1 at `4f5d46a`.
  - `DIFFLET_NEURON_PG_TIMEOUT_S` was not set in any run: it is absent from every `env_extra`. The CLI default applies, which is 3 h.
- The two backends run on different commits, and this cannot be avoided: `difflet/backends/neuron` does not exist on main, and the two branches conflict in 21 files.

**Workload**
- Seed 42, prompt "a cinematic shot of a red fox running through a snowy forest", `HF_HUB_OFFLINE=1`, pinned revisions.
- TP4 with `--cp-degree 1`.
- 50 steps, with `--steps 50` and the guidance always passed explicitly. Without them, the AoT CLI runs Qwen-Image and HunyuanVideo at 4 steps, and LTX-2 at 40 steps with guidance 3.5.

**Backend selection:** main's CLI has no `--backend` flag. It chooses the backend from `DIFFLET_BACKEND` or by auto-detection. Every run exports `DIFFLET_BACKEND=trainium`.

**bf16 per component (AoT on main)**
- Every component runs at main's bf16 default. No flag that falls back to fp32 was passed: no `--host-vae`, and `--tp-reduce-dtype` was left at its default.
- Qwen-Image components: text encoder (Qwen2.5-VL, TP4 on the device), DiT (TP4) and VAE (compiled, one core).
- LTX-2 components: the DiT runs at TP4 on the device. The host pipeline holds the Gemma-3 text encoder, the connectors, the video and audio VAEs and the vocoder.
- The HLO check (per-model sections) confirms the DiT all-reduces are bf16.

**Two-core attention**
- **Qwen-Image:** main's CLI sets two-core attention itself. `difflet/cli/orchestrators/qwen_image.py:34` reads `qwen_common.VIRTUAL_CORE_SIZE`, which is 2 (`difflet/common/orchestrators/qwen_image.py:23`). `run_stage` sets it with `setdefault` (`difflet/cli/runner.py:78`). The harness does not export the variable.
- **LTX-2:** main's CLI does not set it: `difflet/cli/orchestrators/ltx_2.py` has no reference to it. Only the serving path, `difflet/serving/models/ltx_2.py`, mentions it, and the serving path is not used here.
  - The cell exports `NEURON_RT_VIRTUAL_CORE_SIZE=2` for the compile and for every generate.
  - The AoT stage key does not include this variable, so the cell uses a fresh root.
  - The DiT HLO module was checked to differ from the single-core `MODULE_20da5719`.

**Runs per cell**
- **Compile cold:**
  - `difflet compile` on a fresh root `R = B/roots/<model>_<cell>`.
  - Every run of the cell sets `DIFFLET_SHARED_WEIGHTS_DIR=$R/_shared_weights`, `NEURON_COMPILE_CACHE_URL=$R/neuron-compile-cache` and `--cache-dir $R`. The compile therefore includes every NEFF compile and all of the weight pre-sharding.
  - Before the compile, the harness drops the page cache and reads the checkpoint twice. `mincore` residency is recorded in `preread_rec`.
- **Cold e2e (the approved definition, plan decision 2):**
  - `sync; echo 3 > /proc/sys/vm/drop_caches` through passwordless sudo, with meminfo `Cached` recorded before the run.
  - A fresh process and a fresh `--work-dir`, all 50 steps.
  - The text encoder, DiT and VAE weights are read from disk inside the timed run, and no stage output is reused.
  - Compiled artifacts may exist: NEFFs, and the AoT pre-shards in the cell's own root.
- **Warm:** one discarded warm generate (gen0), then gen1–gen3 timed, in the order compile, cold, gen0, gen1, gen2, gen3, q1.
- **Quality run (q1):** one more warm generate with lossless output: PNG for Qwen-Image, a frames tensor `out.pt` for the video models.
  - It also sets `DIFFLET_DUMP_LATENTS_DIR=<run>` and `DIFFLET_STEP_TIMES_FILE=<run>/steps.jsonl`. Both are supported on main: `difflet/utils/latent_dump.py` and `step_timing.py`, used in `qwen_image/pipeline.py` and `ltx_2/pipeline.py`.
  - Main has no latent dump for HunyuanVideo or Wan, and the native branch has none at all. For those cases, the q1 runs set the probe's opt-in `HV1_DUMP_LATENTS_DIR`.
    - On rank 0 (or in the stage process), it wraps the orchestrator's `_denoise`: `QwenImageOrchestrator`, `HunyuanVideoOrchestrator`, `WanOrchestrator` or `LTX2Orchestrator`.
    - It saves the latents entering the loop and the latents leaving it, and passes the arguments and results through unchanged.
    - The CPU self-tests in both venvs check it: `B/*/prep/probe_selftest_*.log`, all ok.
  - The dumped initial latents seed the CPU fp32 references (plan phase 4).
  - q1 is not part of the timing statistics.
- **Native cells:**
  - **LTX-2:** a full cell, run like the AoT cells: preread, compile on a fresh root `B/roots/ltx2_native_tp4`, cold, gen0, gen1-3, q1.
  - **Qwen-Image, HunyuanVideo and Wan2.1:** plan D10, option A. Each got one warm generate at the current HEAD, on a `cp -a` copy of the original cell's root. The copy was checked identical with `diff -r` before the run, and the original root was listed (size and mtime) before and after the run, unchanged.
    - **Pass rule:** `work/latents.pt` must be bit-identical to the reference cell. So must the PNG (Qwen-Image) or the mp4 re-encoded from the lossless `out.pt` with the native CLI's own writer (the video models).
    - If the rule passes, the reference cell's timings stand. The hash-check run's own wall time is not a timing figure (see the model sections).

**Metrics**
- **DiT step** = the median / p10 / p90 of the deltas between consecutive denoising steps, with step 0 excluded (49 deltas per run).
  - Qwen-Image: from the generate stage process (`QwenImageOrchestrator._scheduler_step` stamps).
  - LTX-2: from the generate process. One stamp per video+audio pair of `_scheduler_step` calls, checked in the probe self-test: 6 calls give 3 stamps.
  - The table value is the median of the per-run medians over gen1–3; p10 / p90 are pooled over gen1–3.
  - q1's `steps.jsonl`, written by main's own step timer, agrees: Qwen-Image 361.0 ms, LTX-2 400.8 ms (medians of 50 rows).
- **e2e** = the process wall of `python -m difflet.cli.main generate` (harness clock).
- **Cold-e2e breakdown** = probe spans in the cold run's `summary.json`.
- **Native DiT step:** from rank 0 of the generate launch, stamped the same way (one stamp per video+audio pair for LTX-2).
- **Peak HBM** = neuron-monitor's per-core peak of `memory_used` (the sum of its categories) from `nmon.jsonl`. The AoT stages are not torchrun ranks, so there is no `max_memory_allocated`. For native runs, rank 0's `torch_neuronx` `max_memory_allocated` is also recorded.
- **Outputs:** SHA-256 of every output and dump. The image or a frame strip was opened and checked against the prompt.

**Supervision**
- One device job at a time. Before each launch, the harness waited until `fuser /dev/neuron0` showed no process. After each run it confirmed the device was released: `device_released: true` in every `summary.json` of phases 1 and 2.
- Phase 1: the jobs ran under `scripts/supervise.sh` (Qwen-Image) and as a `systemd-run --user` unit (LTX-2 AoT).
- Phase 2: every job ran as a `systemd-run --user` unit, one after another:
  - `qwen_native_hashcheck`, 10:28 UTC;
  - `hv_aot_4m`, 10:46-12:29;
  - `wan_aot_4m`, 12:38-14:44;
  - `wan_aot_hostvae_4m`, from 14:51;
  - `hvwan_native_4m`, 15:20-15:41;
  - `ltx_native_4m`, 15:53:58-16:29:46, pre-read included.
- `DIFFLET_NEURON_PG_TIMEOUT_S` was not exported.

**Disk cleanup before phase 1 (plan decision 6, option A, approved):**

| root deleted | built at | size | freed |
|---|---|---:|---:|
| `bench-2026-10-25/roots/qwen_aot_tp4` | `4f6e1de` | 53G | 56 GB |
| `bench-2026-10-25/roots/wan_aot_tp4` (`--host-vae`) | `8b2408c` | 39G | 41 GB |
| `bench-2026-10-25/roots/wan_aot_tp4_twocore` (`--host-vae`, two-core env) | `8b2408c` | 39G | 41 GB |
| HunyuanVideo `aot_tp4` | `d6ffb9a` | 59G | 16 GB |

- All four roots were built before the bf16 fixes; their manifests have no reduce-dtype field.
- Free space on `/` went from 230.27 to 385.92 GB (`B/prep/delete_prefix_roots.log`).
- **Why the HunyuanVideo root freed only 16 GB:** its 47.95 GB of transformer, VAE and CLIP shards are hardlinked from `~/.cache/difflet/_shared_weights/hunyuanvideo-*` and from `~/.cache/difflet/hunyuan_video_{dit,clip}/…`.
- **What was kept:**
  - `~/.cache/difflet/_shared_weights` was not touched. The deleted Qwen-Image and Wan roots used their own internal `_shared_weights`. The HunyuanVideo entries are still linked from `~/.cache/difflet/hunyuan_video_*`, outside the approved list, so deleting them would free 0 bytes.
  - Nothing native, nothing in the HF cache and nothing else was deleted.
- After phase 1, the two new roots take 53 GB (Qwen-Image) and 38 GB (LTX-2) on disk, and `/` has 269 GB free.

## Qwen-Image

**Model:** `Qwen/Qwen-Image` at revision `75e0b4be04f60ec59a75f475837eced720f823b6`.  
**Workload:** 1024x1024, 50 steps, `--guidance-scale 4.0` (the DiT drops it: `guidance_embeds` is off), seed 42.  
**AoT cell:** `aot_main_tp4`, root `B/roots/qwen_image_aot_main_tp4`. Three stage processes run one after the other:
- the text stage: Qwen2.5-VL at TP4 on the device;
- the generate stage: the DiT at TP4;
- the vae stage: the compiled VAE on one core.

**Native cell:** *reused* from the 2026-10-25 cell `native_tp4`, measured at `4f6e1de` (`/home/ubuntu/bench-2026-10-25/qwen_image/native_tp4/`). Phase 2 confirmed it by hash at the current HEAD (plan D10, option A; details below).

| metric | AoT on main (`aot_main_tp4`) | native | native vs AoT | AoT before the bf16 fix (2026-10-25, `4f6e1de`, *reused*) |
|---|---|---|---|---|
| DiT step median / p10 / p90 | **360.6** / 360.2 / 361.8 ms (runs 360.6, 360.9, 360.5) | **359.0** / 358.2 / 360.1 ms (*reused*; the hash check at HEAD gave 358.6 / 357.7 / 360.7) | −0.4% | 418.2 ms (now −13.8%) |
| compile cold, checkpoint in the page cache | **1209.9 s** (text stage 128.7, generate stage 660.5, vae stage 401.1) | **187.0 s** (*reused*) | −84.5% | 1238.5 s (−2.3%) |
| e2e warm, n=3 (gen0 discarded) | **73.7 s** mean (74.16, 76.06, 70.98; median 74.2; gen0 72.9) | **82.3 s** (81.42, 82.13, 83.41; *reused*) | +11.6% | 75.4 s (−2.2%) |
| e2e cold, page cache dropped | **515.6 s** (breakdown below) | **502.0 s** (*reused*) | −2.6% | 504.2 s (+2.3%) |
| peak HBM per core (neuron-monitor) | **10.10 GiB** on all 4 cores in gen0–3 (10.04 in cold and q1) | **10.81 GiB** on the three DiT cores, 13.45–13.85 on the VAE core (*reused*) | +7.0% on the DiT cores | 10.29 GiB |
| output | PNG `506c09f4…`, `work/latents.pt` `35e26fb7…`, identical in cold, gen0–3 and q1 | PNG `b1906d53…`, latents `4da4c89e…` in the reused cell and in the HEAD hash check | different noise (torch 2.14 vs 2.9 bf16 draw) | PNG `e24798fe…` |
| logs | `B/qwen_image/aot_main_tp4/{compile,cold,gen0..gen3,q1}/` | `/home/ubuntu/bench-2026-10-25/qwen_image/native_tp4/`; hash check `B/qwen_image/native_tp4_hashcheck/` | | |

*Native vs AoT* is (native − AoT) / AoT, as in the 2026-10-25 report.

**The native numbers (*reused*, confirmed by hash at HEAD)**
- They come from `native_vs_aot_tp4_2026-10-25.md`, cell `native_tp4`, measured on 2026-10-25 05:36–05:59 UTC at `4f6e1de` on this host, with the native venv listed under Toolchain. Each figure was re-read from that cell's `summary.json`: compile 187.0 s, cold 501.99 s, gen1–3 81.42 / 82.13 / 83.41 s, peak HBM 10.81 GiB on cores 0, 1 and 3.
- The cell has no weight store (bf16 checkpoint, "no gain"), so it has one cold number, under definition A and B alike.
- **Hash check** (2026-10-26 10:28 UTC, HEAD `a87d89c`, clean; run dir `B/qwen_image/native_tp4_hashcheck/q1`, record `hashcheck.json`):
  - **How it ran:** one warm generate with the flags of the 2026-10-25 gen3 run, on a `cp -a` copy of the original root (`B/roots/qwen_image_native_tp4_copy1025`). The page cache was dropped and the checkpoint read twice first (100% resident).
  - **Result:** `out.png` `b1906d53…` and `work/latents.pt` `4da4c89e…` are bit-identical to every run of the reference cell, so the timings and HBM above stand. I opened the PNG: a red fox trotting through snow between snow-covered trees.
  - **Step:** the rank-0 step is 358.6 / 357.7 / 360.7 ms, against 359.0 / 358.2 / 360.1 ms in the reference cell.
  - **Its 117.2 s wall is not a warm figure.** One NEFF was compiled inside the run (`neff/39/1e/391e973e…`, neuronx-cc 20.2 s). That NEFF equals the existing `cde0c8b3…` except for ten backend debug blobs of the same length, so the compiled program is unchanged. Why the cache key changed was not found (*inference*: possibly the `attention.py` edit since `4f6e1de`).
  - **Noise:** the probe dumps `initial.pt` `95a4306b…`, packed (1,4096,64) bf16, drawn on rank 0's CPU. It equals the torch 2.14 CPU draw `pack(torch.randn((1,1,16,128,128), generator=Generator().manual_seed(42), dtype=bf16))`. `final.pt` `94d160cb…` equals `work/latents.pt`.
- Its DiT matches AoT's in two respects: it runs in bf16 (plan §1), and its NEFFs have the two-core attention split (`attention_cte_lnc0` / `lnc1` subgraphs, plan D1).
- It differs in three respects:
  - the VAE decodes in fp32;
  - Qwen2.5-VL encodes on rank 0's host, not on the device;
  - the decode runs in a launch of its own.

  These are placement and dtype differences (plan D2, D8), kept as decided for the device runs.

**Compile (`compile/run.log`)**
- **Text stage, 128.7 s:** build 101.0 s, of which the priority neuronx-cc compile took 58.5 s; sharding 7.2 s.
- **Generate stage, 660.5 s:** DiT build 293.4 s (module load 92.1, HLO 16.0, neuronx-cc 30.2), then pre-sharding 361.6 s (4 files, 38 987 MB).
- **VAE stage, 401.1 s:** neuronx-cc 386.2 s.
- **Pre-read:** the drop left `Cached` at 0.27 GiB and the checkpoint 0% resident. The checkpoint (53.74 GiB, 14 files) was then read twice: 439.1 s, then 20.6 s, 100% resident before the compile.

**Cold e2e breakdown (`cold/summary.json`, seconds from process start; `Cached` 0.26 GiB before)**

| part | start | duration |
|---|---:|---:|
| text stage | 0.25 | **140.2** |
| — text-encoder load (Qwen2.5-VL from disk) | 18.19 | 121.8 |
| generate stage | 144.26 | **353.1** |
| — DiT pre-shard load | 149.83 | 329.1 |
| — denoise, 50 steps | 479.0 | 18.4 |
| vae stage | 498.8 | **15.2** |
| — VAE load | 503.38 | 10.0 |
| — decode | 513.42 | 0.27 |
| gaps between stages, driver exit | | 7.1 |
| **process wall** | | **515.6** |

Against the pre-fix AoT cold run (504.2 s), the text stage is level (140.2 vs 140.7 s). The DiT load is 13.2 s longer (329.1 vs 315.9 s). The DiT load reads the cell's 39 GB of pre-shards from disk, so most of this e2e is EBS read time.

**Where the warm e2e goes (gen2, seconds; the 2026-10-25 pre-fix AoT gen2 in brackets, *reused*)**

| part | AoT on main (`gen2`) |
|---|---:|
| process wall | 76.06 [76.4] |
| text stage | 16.7 [16.4]; encoder load 11.9 |
| generate stage | 41.3 [41.2]; DiT load 18.0, denoise 18.4 [21.1] |
| vae stage | 13.8 [14.4]; VAE load 8.6, decode 0.27 |

- The denoise is 2.7 s shorter, in line with the 57.6 ms per-step drop from the bf16 all-reduce.
- gen2's DiT load is 3 s longer than gen1's and gen3's (15.0 and 14.9 s). The warm spread, 71.0–76.1 s, is mostly load time, not compute: the denoise is 18.2–18.4 s in every run.

**bf16 and the HLO modules (`compile/summary.json` `hlo_modules`; `hlo_allreduce.jsonl`)**
- DiT `MODULE_5421e02c…`: 240 all-reduces, all bf16 (120 on `[1,1024,3072]` and 120 on `[1,4096,3072]`).
- Text encoder `MODULE_9887b749…` and `MODULE_4804a098…`: 56 bf16 all-reduces each.
- VAE `MODULE_c08dfa24…`: none (one core).
- There is no f32 all-reduce in any module.

**Quality run q1**
- `out.png` (`506c09f4…`, lossless PNG) was opened: a red fox walking across snow in front of a snowy forest, on prompt.
- `initial.pt` (`c2d9d8b3…`) and `final.pt` (`a9c1f37e…`, packed latents) are kept for the CPU reference.
- `initial.pt` is byte-identical to `/home/ubuntu/campaign/bf16-reduce/runs/qwen/bf16/warm1/initial.pt`: the same seed-42 noise as the 20-step bf16 campaign.
- `steps.jsonl` has 50 rows, median 361.0 ms.

**Against the post-fix 20-step campaign** (`/home/ubuntu/campaign/bf16-reduce/runs/qwen/summary.json`, *reused*): step 361.9 → 360.6 ms (−0.4%). Its 60.6 s warm and 498.8 s cold e2e were measured at 20 steps, so they are not comparable.

## LTX-2

**Model:** `Lightricks/LTX-2` at revision `47da56e2ad66ce4125a9922b4a8826bf407f9d0a`.  
**Workload:** 480x704x49 (2310 video tokens, 51 audio frames), 50 steps, `--guidance-scale 1.0` (no CFG), seed 42.  
**AoT cell:** `aot_main_tp4_twocore`, root `B/roots/ltx_2_aot_main_tp4_twocore`, with `NEURON_RT_VIRTUAL_CORE_SIZE=2` exported for the compile and every generate (`env_extra` in every `run.log`).
- One process: the host pipeline (Gemma-3 text encoder, connectors, video and audio VAEs, vocoder) and the DiT at TP4 on the device.
- The prompt is encoded on the host.

**Native cell:** `native_tp4`, a full cell under definition A, measured in phase 2 (subsection "LTX-2 native" below). The earlier native measurements in `ltx_2_neuron.md` (2026-10-18/20) were taken on another instance with an older toolchain, and their raw evidence is not on this host, so they were not reused; they are compared below.

| metric | AoT on main, two-core (`aot_main_tp4_twocore`) | native | native vs AoT |
|---|---|---|---|
| DiT step median / p10 / p90 | **400.1** / 398.1 / 400.9 ms (runs 398.2, 400.3, 400.1) | **350.5** / 349.4 / 352.0 ms (runs 350.9, 350.5, 350.1) | −12.4% |
| compile cold, checkpoint in the page cache | **1052.2 s** (model build 542.2, pre-sharding 470.5) | **327.59 s** (DiT load 87.5, Gemma-3 load and store write 197.4) | −68.9% |
| e2e warm, n=3 (gen0 discarded) | **66.55 s** mean (66.00, 68.00, 65.64; median 66.0; gen0 66.1) | **92.46 s** mean (94.40, 92.19, 90.78; median 92.19; gen0 90.95) | +38.9% |
| e2e cold, page cache dropped | **778.1 s** (breakdown below) | **594.76 s**, definition A: Gemma-3 from its bf16 weight store (breakdown below); B later | −23.6% |
| peak HBM per core (neuron-monitor) | **9.81 GiB** on all 4 cores (cold, gen2, gen3, q1; 9.75 in gen0 and gen1) | **16.51–16.52 GiB** on all 4 cores in every run; rank 0 `max_memory_allocated` 15.66 GiB | +68.4%: Gemma-3 is on the device |
| output | `out.mp4` `5aab9e4e…`, identical in cold, gen0–3; q1 `out.pt` `1a875738…` | `out.mp4` `cd11731b…`, identical in cold, gen0–3; q1 `out.pt` `33fa80c2…` | different noise (torch 2.14 vs 2.9.1 bf16 draw) |
| logs | `B/ltx_2/aot_main_tp4_twocore/{compile,cold,gen0..gen3,q1}/` | `B/ltx_2/native_tp4/{compile,cold,gen0..gen3,q1}/` | |

**Compile (`compile/run.log`)**
- **Model build, 542.2 s:** DiT module load 92.4 s, HLO 22.7 s, priority neuronx-cc 80.1 s, and the weight-layout transformation.
- **Pre-sharding, 470.5 s:** 4 files, 38 011 MB.
- **Pre-read:** the drop left `Cached` at 0.27 GiB and the checkpoint 0% resident. The checkpoint (transformer, text_encoder `model-*-of-00011`, connectors, vae, audio_vae, vocoder: 85.71 GiB, 23 files) was then read twice: 701.1 s, then 32.8 s, 100% resident before the compile.

**Cold e2e breakdown (`cold/summary.json`, seconds from process start; `Cached` 0.26 GiB before)**

| part | start | duration |
|---|---:|---:|
| imports and start-up | 0 | 21.1 |
| host pipeline load (Gemma-3, connectors, VAEs, vocoder) | 21.13 | **373.8** |
| DiT pre-shard load | 394.89 | **309.7** |
| prompt encode (host) | 704.59 | 30.7 |
| denoise, 50 steps | 735.32 | 20.0 |
| decode | 755.27 | 15.2 |
| save video | 770.49 | 1.4 |
| process exit | 771.9 | |
| **harness wall** | | **778.1** |

- The cold run is 88% load: 683.5 s of the 778.1 s is host pipeline and DiT load.
- **Warm gen1 for comparison:** imports 4.7, host load 5.0, DiT load 11.6, encode 10.6, denoise 19.9, decode 9.4, save 1.2; process exit 62.5 s, wall 66.0 s.

**Two-core attention: verified two ways**
- The DiT HLO module is `MODULE_e1043f485ceeeb495915+9de06980` (`compile/summary.json` `hlo_modules`), not the single-core `MODULE_20da5719`.
- The NKI kernel configs were compared with the single-core module, which is still in `/var/tmp/neuron-compile-cache` (`B/ltx_2/aot_main_tp4_twocore/hlo_nki_attention_vs_20da5719.txt`, from `B/scripts/hlo_nki_diff.py`):
  - In this module, all 288 `attention_cte` calls (6 distinct kernels × 48) are `attention_cte_lnc0_*` with `"grid": 2`.
  - In `MODULE_20da5719`, all 288 are `attention_cte_*` with `"grid": 1`.
  - The two modules share no kernel config.

**bf16 (`hlo_allreduce.jsonl`)**
- The hidden-state all-reduces are bf16: 192 on `[1,2310,4096]` (video) and 192 on `[1,51,2048]` (audio).
- The f32 all-reduces are only on `[1,N,1]` shapes: 96 on `[1,1024,1]`, 240 on `[1,2310,1]` and 240 on `[1,51,1]`.
- These counts are the same as in the 2026-10-24 single-core bf16 module, which is `MODULE_20da5719` itself (`/home/ubuntu/campaign/bf16-reduce/runs/ltx2/hlo_allreduce.txt`). Its fp32 predecessor, `MODULE_85212230`, had the same five groups, with the hidden-state ones in f32.
- *Inference, not checked:* the `[1,N,1]` reductions are norm statistics.

**Quality run q1**
- `out.pt`: the decoded frames, shape (1,49,3,480,704), bf16 in [0,1], `1a875738…`.
- `final_video.pt` (`9c454431…`, (1,2310,128) bf16) and `final_audio.pt` (`648b6af8…`, (1,51,128) bf16) are kept.
- `initial_video.pt` (`4458e4cf…`) and `initial_audio.pt` (`8e52babb…`) are byte-identical to the 2026-10-24 campaign's dumps: the same seed-42 noise.
- `steps.jsonl` has 50 rows, median 400.8 ms.

**Content: only partly on prompt.**
- I opened the mid frame and a strip of frames 0, 12, 24, 36 and 48 (`q1/strip_0_12_24_36_48.png`).
- Every frame shows a snowy forest with snow-covered trees and snow on the ground.
- A red-orange and dark animal-like shape moves through it, but it is incoherent and cannot be recognised as a fox.
- The 2026-10-24 20-step output (`B/ltx_2/ref_2026-10-24/warm1_mid.png`) is also only loosely on prompt.
- `ltx_2_neuron.md` notes that guidance 1 (no CFG) gives low quality on both backends.
- *Inference:* this is the workload, not a regression. The 50-step CPU fp32 reference from the same noise (phase 4) will settle it.

**Against earlier AoT measurements (*reused*, not like for like)**
- **2026-10-20 two-core AoT, pre-fix (fp32 DiT all-reduce)** (`ltx_2_neuron.md`, 50 steps; raw evidence not on this host):
  - step 419.2 → 400.1 ms (−4.6%);
  - warm e2e 68.0 → 66.55 s (−2.1%);
  - cold e2e 781.3 → 778.1 s (−0.4%).
  - Compile, 576.9 vs 1052.2 s, is not comparable: the 2026-10-19 compile's pre-shard state is not documented, and today's compile includes 470.5 s of pre-sharding.
- **2026-10-24 single-core bf16 campaign** (`/home/ubuntu/campaign/bf16-reduce/runs/ltx2/`, 20 steps):
  - step 433.5 → 400.1 ms (−7.7%). Both use the bf16 all-reduce, so this is the two-core gain.
  - Its 54.8 s warm, 767.5 s cold and 555.9 s compile are not comparable: 20 steps, and that compile reused pre-shards.

### LTX-2 native (phase 2, cell `native_tp4`)

**Runs:** 2026-10-26, 15:53:58 to 16:29:46 UTC, pre-read included. The cell ran as the systemd user unit `ltx_native_4m` on the fresh root `B/roots/ltx2_native_tp4`.
- **HEAD:** the compile ran at `999c434`; cold, gen0-3 and q1 ran at `009d3dd`. The two commits differ only in plan docs, and both contain F12 (`05614a2`). Every `run.log` shows `dirty=0`.
- **Outcome:** all 7 runs returned rc 0 and released the device. Every process imported difflet from the hunyuan-neuron worktree.
- **No Difflet bug** came up, so there is no code commit.

**Placement:** the native default, recorded from the rank argv in `cold/probe/*-rank*.json`, which shows `--device-text-encoder`.
- **Gemma-3:** on the device in bf16, TP4 beside the DiT.
- **Connectors:** bf16 on rank 0's host.
- **Video VAE, audio VAE and vocoder:** decoded on the host, in bf16, in the driver process after the generate launch.
- **Command:** `--backend neuron --tp-degree 4`, plus the harness flags: 480x704x49, `--steps 50 --guidance-scale 1.0 --seed 42`, the prompt, and `HF_HUB_OFFLINE=1`.

**Pre-read before the compile** (`cell_record.json` `preread_rec`)
- The checkpoint is the same 85.71 GiB in 23 files as the AoT cell's.
- After the drop, `Cached` was 0.29 GiB and the checkpoint 0% resident.
- It was then read twice, in 702.3 s and 39.4 s. Afterwards it was 100% resident, with `Cached` at 86.16 GiB.

**Compile, 327.59 s** (`compile/run.log`; rank-0 spans and component phases in `cell_record.json`)
- **DiT load, 87.5 s:**
  - checkpoint 5.22 s, from the page cache;
  - warm-up forward 59.64 s, which includes the block NEFF compile.
  - The weight store logged `not written: no gain` for the DiT on every rank: its checkpoint is already bf16 (9.96 GB either way).
- **Gemma-3 load, 197.4 s:**
  - checkpoint read plus store write 181.64 s, then warm-up 15.24 s.
  - Every rank logged `[weight store] rank r/4 NeuronLtx2TextEncoderApplication: written 5.88 GB in 118.7–178.4 s` into `neuron/weights/Lightricks--LTX-2__text_encoder__47da56e2__bfloat16__tp4__979a2453cd7057f8`.
- **Blocks:** every rank logged `compiled_blocks=96`, with no load or generate fallbacks.

**F12 confirmed on the device** (store manifests `*.manifest.json` in that directory)
- The load report of every manifest has `missing: 0`.
- The store holds 626 tensors: `replicated` 289, plus 337 in each `tpN-of-4`. These cover the 624 layer tensors, `embed_tokens` and `norm`.
- The 439 checkpoint keys not loaded (`unexpected`) are `vision_tower` (437) and `multi_modal_projector` (2). These are the skipped prefixes.

**Cold e2e A, 594.76 s** (`cold/summary.json`; seconds from driver start)
- Before the run, `Cached` was 0.27 GiB and the checkpoint 0% resident.

| part | at | duration |
|---|---:|---:|
| driver start to rank start (torchrun launch) | 0 | 14.9 |
| rank imports and runtime set-up | 14.86 | 14.2 |
| DiT load (bf16 checkpoint read 286.55, build on meta 9.87, warm-up forward 12.86) | 29.01 | **310.2** |
| Gemma-3 load from the weight store (store read 178.78, warm-up 1.36; every rank logged `hit 5.88 GB` in 175.6–178.4 s) | 340.28 | **180.2** |
| prompt encode (connectors 24.95) | 520.55 | 25.3 |
| denoise, 50 steps | 545.88 | 17.56 |
| ranks exit / generate stage ends | 563.5 / 568.3 | |
| host decode (pipeline load 3.01, decode 19.3, save 0.73) | 568.31 | 25.41 |
| driver exit / **harness wall** | 593.73 | **594.76** |

- **Connectors:** they take 24.95 s here but 4.9 s in the warm runs, although their load span is only 0.06 s. *Inference:* their weights are read from disk on first use.
- **Against AoT's cold run (778.1 s):** AoT spends 373.8 s loading its host pipeline (Gemma-3 from its fp32 checkpoint, the connectors and the VAEs) and 309.7 s loading the DiT pre-shards. Native's DiT load takes about as long (310.2 s, of which the bf16 checkpoint read is 286.55 s). It reads Gemma-3 from its bf16 store, 5.88 GB per rank, in 180.2 s.

**Warm e2e, by part** (gen1, seconds from process start)

| part | native (`native_tp4/gen1`, wall 94.40) | AoT (`aot_main_tp4_twocore/gen1`, wall 66.00) |
|---|---|---|
| start-up | ranks start at 2.8; DiT load starts at 25.5 | imports 4.7 |
| text encoder load | Gemma-3 4.4 (store 3.0) | host pipeline 5.0 |
| DiT load | 17.9 (checkpoint 5.3, warm-up 9.2) | 11.6 |
| prompt encode | 5.3 | 10.6 |
| denoise | 17.55 | 19.9 |
| decode and save | stage ends at 74.5; host decode 19.0 (pipeline load 2.6, decode 13.6, save 0.5) | decode 9.4, save 1.2 |
| process exit | 93.5 | 62.5 |

- **Where native's 26 s go.** Native's denoise is 2.4 s faster, but it spends more on two things:
  - the launch: 22.7 s from rank start to the start of the DiT load. Across gen0-3 this was 19.2–22.7 s, against 14.2 s in the cold run. *Inference:* this is the torchrun launch jitter recorded in `neuron_final.md`.
  - the separate host decode: 19.0 s, against AoT's 10.6 s.

  This is the same split `ltx_2_neuron.md` reported.

**Two-core attention in the native NEFFs** (`B/ltx_2/native_tp4/neff_attn_scan.json`)
- 5 of the 90 NEFFs in the root contain `attention_cte`. All 5 split into `sg00:attention_cte_lnc0` and `sg01:attention_cte_lnc1`, and no attention NEFF has a single subgraph.
- The newest NEFF is from 16:11:43 UTC, before the cold run started (16:11:50). So nothing was compiled during the cold or warm runs.

**Outputs and q1** (`q1_checks/checks.json`)
- `out.mp4` `cd11731b…` is identical in cold and gen0-3, and `work/latents.pt` `b9c745bd…` is identical in all 6 generates.
- q1's `out.pt` `33fa80c2…` is (1,49,3,480,704) bf16 in [0,1]. Re-encoding it with the native `_save_video` gives `cd11731b…`, so q1 produced exactly the frames of the timed runs.
- **Probe dumps** (rank 0, `HV1_DUMP_LATENTS_DIR`, wrapper on `LTX2Orchestrator._denoise`):

  | dump | file sha256 | raw sha256 | shape |
  |---|---|---|---|
  | initial_video | `16e09464…` | `55b90133…` | (1,2310,128) bf16 |
  | initial_audio | `411c72d9…` | `24900b1b…` | (1,51,128) bf16 |
  | final_video | `c218a480…` | `887027a8…` | (1,2310,128) bf16 |
  | final_audio | `b868488a…` | `fac8052a…` | (1,51,128) bf16 |

  - The final dumps equal `work/latents.pt` bit for bit.
  - The initial noise is drawn on rank 0's CPU in bf16. A torch 2.14 CPU draw reproduces it bit for bit: `Generator().manual_seed(42)`, bf16 `randn` (1,128,7,15,22) packed, then (1,8,51,16) packed (`q1_checks/cpu_noise_seed42.pt`).
- **Noise against AoT:** native's noise differs from AoT's q1 noise (`initial_video` `4458e4cf…`, `initial_audio` `8e52babb…`).
  - The dtype and shapes are the same on both sides, so the difference is the torch version: 2.14.0 against 2.9.1.
  - The first video element is 1.9297 on native and −0.8086 on AoT, the values `ltx_2_neuron.md` recorded.
  - Final latents against AoT: cosine 0.363 (video) and 0.641 (audio), as expected from different noise.
- **Content:** I opened the mid frame and the strip of frames 0, 12, 24, 36 and 48 (`q1/strip_0_12_24_36_48.png`).
  - It shows a snowy conifer forest with a red-orange, fox-like animal running across the snow.
  - It is more recognisable than the AoT output, but blurred and distorted, with a cyan cast on the snow.
  - *Inference:* this is guidance 1 (no CFG), as for AoT. The CPU reference (phase 4) will settle it.

**Against the earlier native numbers** (`ltx_2_neuron.md`, 2026-10-19/20, *reused*: another instance, an older toolchain, no raw logs on this host)

| metric | earlier | now | change |
|---|---:|---:|---:|
| DiT step | 350.7 ms | 350.5 ms | −0.1% |
| warm e2e | 90.8 s | 92.46 s | +1.8% |
| cold e2e (with the weight store) | 588.3 s | 594.76 s | +1.1% |
| compile | 306.6 s | 327.59 s | +6.8% |

## HunyuanVideo 1.0

**Model:** `hunyuanvideo-community/HunyuanVideo` at revision `e8c2aaa66fe3742a32c11a6766aecbf07c56e773`.  
**Workload:** 320x512x61, 50 steps, `--guidance-scale 6.0` (embedded guidance: one DiT forward per step), seed 42.

**AoT cell:** `aot_main_tp4`, root `B/roots/hunyuan_video_aot_main_tp4`, run on 2026-10-26 from 10:46:43 to 12:29:13 UTC as unit `hv_aot_4m`, at main `540e5f0`, clean. Main's own placement uses three stage processes:
- CLIP on one core;
- Llama on the device at TP4;
- then the DiT at TP4, with the compiled bf16 VAE replicated on the four ranks.

**Native cell:** *reused* from `s4_tp4` (`hunyuan_video_neuron.md`, measured 2026-10-25 23:42 to 2026-10-26 00:25 UTC at `eadddec`, raw evidence under `/home/ubuntu/bench-2026-10-26/hunyuan_video/s4_tp4/`). Phase 2 confirmed it by hash at `4f5d46a` (below). Native's placement:
- Llama and CLIP encode on rank 0's host;
- the DiT runs at TP4;
- the compiled bf16 decoder runs on rank 0's core inside the generate launch.

| metric | AoT on main (`aot_main_tp4`) | native (`s4_tp4`, *reused*) | native vs AoT | AoT before the bf16 fix (2026-10-25, `d6ffb9a`, *reused*) |
|---|---|---|---|---|
| DiT step median / p10 / p90 | **801.8** / 800.9 / 802.5 ms (runs 801.8, 802.0, 801.2) | **808.2** / 807.0 / 810.7 ms (the hash check at HEAD gave 808.7 / 807.5 / 810.0) | +0.8% | 813.9 ms (now −1.5%) |
| compile cold, checkpoint in the page cache | **4526.7 s** (clip stage 41.4, llama stage 163.6, generate stage 4315.7) | **274.5 s** | −93.9% | 4347.7 s, a lower bound: it reused pre-shards |
| e2e warm, n=3 (gen0 discarded) | **133.0 s** mean (134.67, 131.73, 132.59; median 132.6; gen0 129.2) | **126.7 s** (129.18, 130.65, 120.26; gen0 130.5) | −4.7% | 132.9 s (n=2) |
| e2e cold, page cache dropped | **617.5 s** (breakdown below) | **457.8 s** (load 356.0, of which the DiT checkpoint 194.8 and host Llama/CLIP 123.3; includes a 6.4 s neuronx-cc on rank 3) | −25.9% | 617.3 s |
| peak HBM per core (neuron-monitor) | **15.53 GiB** on each of the 4 cores in every generate | **17.95 GiB** on the decoder core (DiT shard + decoder), 12.98 on the other three | +15.6% / −16.4% | 15.52 GiB |
| output | mp4 `334301fc…` in cold and gen0-3; q1 `out.pt` `33a8a130…` | mp4 `edf8d6be…`, latents `b62d0891…` | different noise (native fp32 draw, AoT bf16 draw) | `201e5ef7…` |
| logs | `B/hunyuan_video/aot_main_tp4/{compile,cold,gen0..gen3,q1}/` | `/home/ubuntu/bench-2026-10-26/hunyuan_video/s4_tp4/`; hash check `B/hunyuan_video/native_tp4_hashcheck/` | | `hunyuan_video_neuron.md` |

**AoT compile, 4526.7 s** (`compile/run.log`, `cell_record.json`)
- **Pre-read:** 39.01 GiB, read in 318.6 s and then 15.4 s, 100% resident.
- **Stages:** clip stage 41.4 s (build 20.8), llama stage 163.6 s (build 148.5, of which neuronx-cc 75.7), generate stage 4315.7 s.
- **Inside the generate stage:**
  - DiT build 421.5 s, of which neuronx-cc 171.3 s;
  - DiT pre-sharding 208.0 s (44.4 GB), the sharding the 2026-10-25 compile did not pay;
  - VAE build 3679.1 s, of which neuronx-cc 3571.4 s. **The VAE is 81% of the compile.**

**AoT cold e2e, 617.5 s** (`cold/summary.json`; `Cached` 0.27 GiB before)
- clip stage 26.4 s (load 7.0);
- llama stage 134.6 s (load 129.3);
- generate stage 448.3 s: DiT and VAE pre-shard load 385.0, denoise 40.3, VAE decode 17.05, save 0.96.

**AoT warm gen1, 134.67 s:** clip stage 12.2, llama stage 16.4 (load 11.0), generate stage 100.7 (app load 37.3, denoise 40.3, decode 17.05, save 0.76).

**Two-core attention and bf16** (`hlo_nki_grid.jsonl`, `hlo_allreduce.jsonl`)
- **Two-core:** main's CLI sets `_VIRTUAL_CORE_SIZE = 2` (`hunyuan_video.py:31`). A parse-only capture shows `NEURON_RT_VIRTUAL_CORE_SIZE=2` on the clip, llama and generate stages. In the DiT module `MODULE_f4b5a7e3…`, all 60 attention calls are `attention_cte_lnc0_de6532dd…` with grid 2.
- **bf16:** the DiT has 124 all-reduces, all bf16: 40 on `[1,10240,3072]`, 40 on `[1,10496,3072]` and 44 on `[1,256,3072]`. The two Llama graphs have 64 + 64 bf16 all-reduces. The VAE (`MODULE_f27c3584…`) and CLIP have none.
- **Native two-core:** native's NEFFs carry the `attention_cte_lnc0` / `lnc1` split (plan D1).

**AoT q1**
- `out.pt` `33a8a130…` is (1,3,61,320,512) bf16, the raw VAE output in [−1.27, 1.04]. Re-encoding it with main's `_save_video` gives `334301fc…`, the timed runs' mp4.
- **Probe dumps:** main has no HunyuanVideo latent dump, so the probe dump was used. It saved `initial.pt` `e13a3f46…` and `final.pt` `ec23d9af…`, both (1,16,16,40,64) bf16.
- **AoT noise on CPU:** in torch 2.9.1, `torch.manual_seed(42); torch.randn(1,16,16,40,64, dtype=torch.bfloat16)` equals `initial.pt` bit for bit (`q1_noise_check.json`).
- **Content:** I opened `cold/mid.png` and `q1/mid.png`. Both show a red fox running across snow, with a snowy conifer forest behind it.

**Native hash check** (2026-10-26 15:20 UTC, HEAD `4f5d46a`, clean; unit `hvwan_native_4m`; `B/hunyuan_video/native_tp4_hashcheck/`)
- **How it ran:** one warm generate on a `cp -a` copy of the `s4_tp4` root. `diff -r` showed it identical before the run, and the original root's listing was unchanged after.
- **Result:**
  - `work/latents.pt` `b62d0891…` is bit-identical to every `s4_tp4` run.
  - The lossless `out.pt` `a17c9714…` ((1,3,61,320,512) bf16) re-encodes with the native `_write_video` to `edf8d6be…`, the reference mp4.
  - So the reused numbers stand.
- **Noise:** the noise entering the loop is fp32 (1,16,16,40,64). It equals the CPU draw `torch.randn(…, generator=Generator().manual_seed(42), dtype=float32)` bit for bit. It differs from AoT's bf16 draw.
- **Step and HBM:** 808.7 / 807.5 / 810.0 ms; neuron-monitor 17.89 GiB on the decoder core and 12.92 on the others.
- **Its 156.1 s wall is not a warm figure.** Nothing was compiled: the root copy gained no files. Start-up and load were slower: the stage started 11.5 s later, and the app load took 56.3 s against 42.1 s. *Inference:* the page-cache drop before the pre-read also evicted venv and NEFF files that the reference warm runs had cached.

**Main's issue #110** (warm e2e up from 109.8 to 123.2 s after the bf16 fix, n=1) does not reproduce at this workload: warm is 133.0 s here, against 132.9 s before the fix.

## Wan2.1 14B

**Model:** `Wan-AI/Wan2.1-T2V-14B-Diffusers` at revision `38ec498cb3208fb688890f8cc7e94ede2cbd7f68` (DiT and umT5 checkpoints in fp32).  
**Workload:** 480x832x9, 50 steps, `--guidance-scale 5.0` (CFG: two DiT forwards per step), seed 42.

**AoT cell:** `aot_main_tp4_twocore`, root `B/roots/wan21_aot_main_tp4_twocore`, run on 2026-10-26 from 12:38 UTC as unit `wan_aot_4m`, at main `540e5f0`, clean.
- `NEURON_RT_VIRTUAL_CORE_SIZE=2` was exported for the compile and every generate:
  - main's Wan CLI sets `_VIRTUAL_CORE_SIZE = None`; only ring CP sets 2;
  - `run_stage` passes the exported value to both the transformer stage and the vae stage (parse capture, `B/wan2_1/prep/parse_check_vcs2.json`).
- **Placement** (main's default): a transformer stage runs umT5 and the DiT at TP4, then a vae stage runs the compiled bf16 VAE on one core.
- **Supplement A2** (`aot_main_tp4_twocore_hostvae`, unit `wan_aot_hostvae_4m`, from 14:51 UTC): the same root with `--host-vae`, no compile. The driver decodes on the host in fp32, which is the placement of the 2026-10-25 AoT cells.

**Native cell:** *reused* from `native_tp4` (`native_vs_aot_tp4_2026-10-25.md`, measured 2026-10-25 07:00–07:36 UTC at `8b2408c`, raw evidence `/home/ubuntu/bench-2026-10-25/wan2_1/native_tp4/`). Phase 2 confirmed it by hash at `4f5d46a` (below). Native's placement:
- umT5 runs in fp32 on the device, TP-sharded beside the DiT;
- the DiT is read from the bf16 weight store;
- the VAE decodes in fp32 in a one-core launch of its own.

| metric | AoT on main, device bf16 VAE (`aot_main_tp4_twocore`) | AoT A2, `--host-vae` (`…_hostvae`) | native (`native_tp4`, *reused*) | native vs AoT A2 | AoT two-core before the bf16 fix (2026-10-25, `8b2408c`, `--host-vae`, *reused*) |
|---|---|---|---|---|---|
| DiT step median / p10 / p90 | **956.4** / 955.3 / 958.0 ms (runs 956.4, 955.8, 956.9) | 956.8 / 956.0 / 958.2 ms | **1063.6** / 1061.5 / 1065.9 ms (the hash check at HEAD gave 1063.8 / 1062.2 / 1067.3) | +11.2% | 955.8 ms |
| compile cold, checkpoint in the page cache | **5935.7 s** (transformer stage 430.5, VAE stage 5486.1) | none: same root | **600.3 s** (generate launch 428.8, vae launch 169.7) | −89.9% (+39.5% against the transformer stage alone) | 445.1 s, no VAE stage |
| e2e warm, n=3 (gen0 discarded) | 105.99 s (106.97, 104.34, 106.65; gen0 105.61). **The frames are corrupted** | **123.16 s** (123.04, 123.43, 123.00; gen0 123.9) | **128.5 s** (125.98, 130.82, 128.83; gen0 129.4) | +4.4% | 120.0 s |
| e2e cold, page cache dropped (A) | 432.3 s. **The frames are corrupted** | **449.4 s** | **542.7 s** (DiT from the store 217.0, umT5 checkpoint 172.5) | +20.8% | 447.5 s |
| peak HBM per core (neuron-monitor) | **10.34 GiB** on each core | 10.34 GiB (10.37 in gen0) | **15.15 GiB** on the DiT cores; VAE core 18.12–18.40 | +46.5% on the DiT cores | 10.37 GiB |
| output | mp4 `941daa35…` in cold and gen0-3; latents `5ae9311e…` in all 6 generates | mp4 `60463ea2…`, latents `5ae9311e…` | mp4 `9b914e81…`, latents `3cafa271…` | same noise | mp4 `d2ecce24…` |
| logs | `B/wan2_1/aot_main_tp4_twocore/{compile,cold,gen0..gen3,q1}/` | `B/wan2_1/aot_main_tp4_twocore_hostvae/{cold,gen0..gen3}/` | `/home/ubuntu/bench-2026-10-25/wan2_1/native_tp4/`; hash check `B/wan2_1/native_tp4_hashcheck/` | | `native_vs_aot_tp4_2026-10-25.md` |

**Which AoT e2e figures are valid.** The headline AoT e2e figures are A2's (`--host-vae`), because the device-VAE frames are corrupted (below). With the device VAE, AoT is 17.2 s faster warm and 17.1 s faster cold, but it decodes wrong frames. The compile column is main's default compile, which builds that device VAE. The transformer stage alone takes 430.5 s, and A2 needs only that stage's artifacts.
- *Not checked:* whether `difflet compile --host-vae` on main skips the VAE stage.

**AoT compile, 5935.7 s** (`compile/run.log`, `cell_record.json`)
- **Pre-read:** 74.86 GiB in 18 files, read in 612.1 s and then 28.5 s, 100% resident.
- **Transformer stage, 430.5 s:**
  - umT5 build 35.8 s (neuronx-cc 6.8), pre-shard 5.7 s / 10.8 GB;
  - DiT build 252.7 s (neuronx-cc 62.0), pre-shard 114.2 s / 28.6 GB.
- **VAE stage, 5486.1 s:** neuronx-cc 5468.4 s.

**AoT cold breakdowns** (`Cached` 0.26 GiB before each)
- **Device VAE, 432.3 s:** transformer stage 394.8 (pre-shard load of umT5 + DiT 324.6, encode 0.64, denoise 48.0), vae stage 31.3 (load 24.15, decode 1.4, save 0.86).
- **A2, 449.4 s:** pre-shard load 321.9, denoise 48.0, host decode 50.2.

**AoT warm gen1 breakdowns**
- **Device VAE, 106.97 s:** transformer stage 72.1 (load 18.4, denoise 47.98), vae stage 31.43 (load 24.49, decode 1.40).
- **A2, 123.04 s:** transformer stage 70.2 (load 16.0, denoise 48.0), host decode 49.1, save 0.6.

**Two-core attention and bf16** (`hlo_nki_grid.jsonl`, `hlo_allreduce.jsonl`)
- **DiT module `MODULE_3c8d124d…`:**
  - All 80 attention calls are `attention_cte_lnc0_*` with grid 2.
  - It is neither the single-core stock module (`MODULE_c984433f`, by name from the 2026-10-25 logs; that root was deleted under decision 6) nor the 2026-10-25 two-core module (`MODULE_5ae08501`).
  - 120 bf16 all-reduces on `[1,4680,5120]`. The only f32 ones are on `[N,1]` shapes: 120 on `[1,4680,1]` and 40 on `[1,512,1]`. *Inference:* these are q/k norm statistics.
- **umT5 `MODULE_f0822b32…`:** 24 bf16 all-reduces, no f32.
- **VAE `MODULE_5dd36239…`:** no all-reduces and no NKI calls.

**Finding: main's compiled bf16 Wan VAE corrupts its frames on the device**
- **What it looks like:** the device-VAE output shows the fox only as a silhouette under coloured block noise (`cold/mid.png`). After frame 0, the frame std is 15–52 grey levels, against 62–81 for a host decode.
- **The latents are fine:** I decoded q1's `work/latents.pt` (`5ae9311e…`) on the CPU with diffusers' `AutoencoderKLWan`, in fp32 and in bf16. Both decodes are clean and on prompt (`q1_host_vae/host_{fp32,bf16}_mid.png`).
- **PSNR of the device frames** (`q1/out.pt` `4380f37c…`) against the fp32 host decode of the same latents (`q1_host_vae/host_fp32_vs_device.json`): mean 12.41 dB, worst 9.36 dB. Per frame: 21.4, 12.67, 10.99, 9.36, 11.05, 12.17, 11.29, 11.04, 11.71.
- **Runtime VCS is not the cause:** re-running only the vae stage on q1's latents gives `out.pt` `4380f37c…` bit for bit, with `NEURON_RT_VIRTUAL_CORE_SIZE=2` and with it unset (`vae_vcs_ab/`, `vae_vcs_ab.log`).
- **Compile-time VCS is inferred not to be the cause, not tested.** A test would need another 1.5 h compile. The evidence:
  - the VAE HLO has no NKI calls, and only the attention ops read the variable;
  - the 2026-10-24 Wan2.2 bf16 device-VAE run (`/home/ubuntu/campaign/bf16-reduce/runs/wan22/bf16/warm1`, no VCS export; mid frame copied to `ref_wan22_devvae/`) shows the same kind of corruption.
- **Status:** a defect in `origin/main`'s compiled Wan VAE (`trainium/wan/vae.py`), recorded as a follow-up. `aot-main` was not modified.
- **Native is not affected:** its VAE is a different, eager fp32 path.

**AoT q1 and noise**
- `out.pt` `4380f37c…` is (1,3,9,480,832) bf16, the corrupted device frames. Re-encoding it with main's `_save_video` gives `941daa35…`, the timed runs' mp4.
- **Probe dumps:** main has no Wan latent dump, so the probe dump was used. It saved `initial.pt` `03b6284b…` (fp32, raw sha `16aeaed7…`) and `final.pt` `c7332d35…`, which equals `work/latents.pt`.
- **One noise for both backends:** `initial.pt` equals the CPU draw `torch.randn((1,16,3,60,104), generator=Generator().manual_seed(42), dtype=float32)` in torch 2.9.1 and in torch 2.14 (`q1_noise/`). The native hash check's dump is the same file (`03b6284b…`). So both backends start from identical noise, and a single CPU reference covers both.
- **Final latents compared** (rel-L2 and cosine, `latents_vs_1025.json`):
  - against the 2026-10-25 AoT two-core run: rel-L2 0.178, cosine 0.984;
  - against native `3cafa271…`: rel-L2 0.216, cosine 0.977;
  - for scale, the 2026-10-25 native against the 2026-10-25 AoT: 0.245 / 0.970.
- **Content:** A2's mp4 `60463ea2…` is clean and shows a red fox in a snowy forest.

**Native hash check** (2026-10-26 15:30 UTC, HEAD `4f5d46a`, clean; unit `hvwan_native_4m`; `B/wan2_1/native_tp4_hashcheck/`)
- **How it ran:** one warm generate on a `cp -a` copy of the 2026-10-25 root (`B/roots/wan21_native_tp4_copy1025`). Before the run, the harness's new `--preread-files` read exactly what the run reads: the umT5 and VAE checkpoint files plus the copy root's store, 48.25 GiB, 100% resident.
- **Result:**
  - `work/latents.pt` `3cafa271…` is bit-identical to the reference cell.
  - The lossless `out.pt` `8d8a1779…` ((1,3,9,480,832) fp32) re-encodes with the native `_save_video` to `9b914e81…`, the reference mp4.
  - So the reused numbers stand.
- **Step and HBM:** 1063.8 / 1062.2 / 1067.3 ms; weight store hit 7.50 GB per rank; neuron-monitor 15.15 GiB on the DiT cores and 18.38 on the VAE core.
- **Its 209.9 s wall is not a warm figure.** Four NEFFs were compiled inside the run, one neuronx-cc per rank, which made the warm-up forward 73.9 s against 40.1 s.
  - Each new NEFF equals one existing per-rank DiT attention NEFF except for 10 debug blobs of the same length (`new_neffs_members.txt`), so the compiled program is unchanged.
  - This is the same pattern as the Qwen-Image hash check. The cause of the cache-key change was not established.

## Problems during phase 1 (harness, not Difflet)

1. **The first Qwen-Image compile exited in 0.1 s with rc=2.**
   - Cause: the harness passed `--backend trainium`, which main's CLI does not have.
   - Fix: the job scripts pass `DIFFLET_BACKEND=trainium` instead. Every compile and generate argument list for both models then passed a parse-only check against main's parser.
   - The failed run is kept as `B/qwen_image/aot_main_tp4/compile.failed-backend-flag`. The root had not been created, so it was still fresh for the rerun.
2. **The first Qwen-Image cold run was killed about 2 min in (08:56:42).**
   - Cause: the job had been started under `supervise.sh` from a Monitor task. It died when that Monitor reached its 30-minute limit, even with `setsid`.
   - The job quarantined the run as `cold.partial-1793005519` and reran cold after a fresh page-cache drop. All later runs are clean.
   - The LTX-2 cell ran as a `systemd-run --user` unit instead, which survived a Monitor expiry.

## Phase 2: notes and decisions (harness, not Difflet)

Phase 2 found no Difflet bug on the native branch, so it produced no code commits. `aot-main` stayed unmodified at `540e5f0`. The harness and probe changes below live outside the repo, in `B/scripts` and `B/probe`. Each change is additive, and the previous version is kept as `*.as-used-*.py`.

1. **Probe latent dumps for HunyuanVideo, Wan and native runs** (approved D9 tooling).
   - **Problem:** main's `DIFFLET_DUMP_LATENTS_DIR` exists only for Qwen-Image and LTX-2, and the native branch has none.
   - **Choice:** add an opt-in `HV1_DUMP_LATENTS_DIR` that wraps the orchestrators' `_denoise`, and set it only in q1 and the hash checks.
   - **Reason:** this gives the CPU references the exact device noise. The wrapper only observes, which the CPU self-tests check: HV AoT 15/15, Wan AoT 15/15, native Qwen 8/8, native HV/Wan 15/15, native LTX-2 23/23.
2. **Hash checks on root copies.**
   - **Problem:** a native hash check could write new cache entries into the original evidence root.
   - **Choice:** run each one on a `cp -a` copy, checked identical with `diff -r` before the run.
   - **Reason:** the originals keep their evidence. Two of the three runs did write NEFFs into the copy (Qwen-Image and Wan2.1).
3. **Hash-check wall times not used as timing figures.**
   - **Problem:** the hash-check walls were slower than the reference warm runs. Qwen-Image and Wan2.1 compiled NEFFs inside the run; HunyuanVideo's start-up and load were slower.
   - **Choice:** keep the reused numbers, which the pass rule (bit identity) allows, and record the walls as notes.
   - **Reason:** the step medians agree within 0.2–0.5 ms, and each in-run NEFF equals an existing one except for debug blobs.
4. **Wan2.1 AoT device-VAE corruption.**
   - **Problem:** the device-VAE frames are corrupted.
   - **Choice:**
     - isolate the cause without code changes (host fp32/bf16 decodes, plus a VAE-only rerun with VCS=2 and with VCS unset);
     - run the plan's optional A2 (`--host-vae`) on the same root, with no compile;
     - report A2's e2e as the valid AoT figure;
     - record the defect as a follow-up on main.
   - **Reason:** `aot-main` must not be modified, and a fix needs repeated 1.5 h VAE compiles.
5. **Harness additions.**
   - `bench4m.py` gained per-model entries for HunyuanVideo, Wan2.1 and LTX-2 native.
   - `root_prefix`: Wan2.1 uses `wan21`, so the root path matches the one the stage names.
   - `--preread-files`: the native Wan hash check pre-reads what the run actually reads, the umT5 and VAE checkpoints plus the bf16 store, instead of the unused fp32 transformer.
6. **LTX-2 native HEAD changed during the cell.**
   - **Problem:** plan commit `009d3dd` landed between the compile (`999c434`) and the cold run.
   - **Choice:** keep the runs, and record both HEADs.
   - **Reason:** `git diff --stat 999c434 009d3dd` touches only two files under `docs/superpowers/plans/`, so the difflet code is identical.
7. **Supervision.**
   - **Problem:** a job started from a Monitor dies when the Monitor expires (a phase-1 finding).
   - **Choice:** every phase-2 job ran as a `systemd-run --user` unit, watched with timeout-bounded loops.

**Disk:** the phase-2 roots take 60 GB (HunyuanVideo AoT), 40 GB (Wan2.1 AoT) and 22 GB (LTX-2 native). The native root copies take 27 GB (Wan2.1), 76 MB (HunyuanVideo) and 41 MB (Qwen-Image). `/` has 120 GB free after phase 2.

## Phase 4 preparation: CPU reference tooling for Qwen-Image and LTX-2

Tooling only, 2026-10-26 16:50–17:50 UTC. No 50-step reference has run yet, and the Neuron device stayed idle throughout.

**Scripts** (in `B/scripts`, outside the repo; main's originals are kept unmodified as `*.main-540e5f0.py`)
- `cpu_ref_qwen.py` and `cpu_ref_ltx2.py`, adapted from main's `scripts/cpu_fp32_reference.py`:
  - diffusers' own `QwenImagePipeline` / `LTX2Pipeline`, all components in fp32 on the CPU, at the model's pinned snapshot;
  - the prompt is encoded with only the text encoder loaded, then the encoder is freed (as on main);
  - the run starts from a device q1 dump: Qwen-Image through diffusers' `latents=` argument, LTX-2 by replacing `prepare_latents` / `prepare_audio_latents` (as on main; diffusers' own path would re-normalize the audio latents);
  - `--steps` defaults to 50.
- `quality_metrics.py`, adapted from main's `scripts/quality_vs_reference.py`. The metric definitions are main's (see Metrics below); it also reads lossless `.pt` frames and refuses mp4.
- `cpu_ref_common.py` and `cpu_only.py` hold shared helpers. `job_cpu_ref.sh` runs one reference as a `systemd-run --user` unit, under `/usr/bin/time -v`.
- Prepared but not started: `job_cpu_ref_queue_qwen_ltx2.sh` (the four 50-step references, one after the other) and `metrics_qwen_ltx2.sh` (the four comparisons).

**Environment**
- **Venv:** `/home/ubuntu/venvs/difflet-neuron`: torch 2.9.1, diffusers 0.38.0, transformers 4.57.6. The native venv has the same diffusers (0.38.0, torch 2.14).
  - Every reference uses this one venv, so a model's two references differ only in their noise.
  - It is also the venv of the 2026-10-24 20-step references.
- **Threads:** torch's default, 6 intra-op and 6 inter-op threads. The host has 6 physical cores (12 vCPU, Xeon 8488C).
- **No Neuron in the process:**
  - In this venv, diffusers, accelerate and transformers import `torch_xla` when it is installed. That initializes `torch_neuronx`, and the diffusers pipelines then call `xm.mark_step()` every step. The 2026-10-24 references ran that way.
  - In the native venv, `import torch` autoloads `torch_neuronx`.
  - `cpu_only.py` prevents both: it sets `USE_TORCH_XLA=0` and `TORCH_DEVICE_BACKEND_AUTOLOAD=0` before torch is imported, sets diffusers' `_torch_xla_available` to False, and refuses every import of the Neuron and XLA packages.
  - Each run records `cpu_only.report()`: Neuron modules loaded, open `/dev/neuron*` fds. A non-empty report fails the run.
  - In every smoke: no such module was loaded, no such fd was open, and the pipelines' `XLA_AVAILABLE` was False.

**Guidance, mirrored from the device runs**
- **Qwen-Image:** true CFG was off on both backends.
  - Both q1 runs passed `--guidance-scale 4.0` and no `--true-cfg-scale`.
  - Native runs true CFG only with `--true-cfg-scale > 1` (`_true_cfg_scale`). AoT on main has no negative branch.
  - The 4.0 is dropped by the DiT on both backends (`guidance_embeds` false).
  - The reference therefore passes `guidance_scale=4.0`, which diffusers ignores and logs, and `true_cfg_scale=1.0` with no negative prompt, so `do_true_cfg` is False.
- **LTX-2:** CFG was off on both backends at g1.0.
  - Native's `do_cfg` and AoT's `do_cfg` both require guidance > 1.
  - In diffusers, CFG needs `guidance_scale > 1` or `audio_guidance_scale > 1`, and the audio scale defaults to the video one.
- Both scripts count the DiT calls and fail unless there is exactly one per step.

**Injection is bit-exact**
- A forward pre-hook captures the transformer's first `hidden_states` (and, for LTX-2, `audio_hidden_states`): what actually enters the loop.
- That tensor must equal the fp32 upcast of the dump bit for bit (bf16 to fp32 is exact), and it must round back to the dump's bf16 bits.
- It held in every smoke, for both noises of both models:

  | dump (device q1) | dtype, shape | file / raw sha256 | first DiT input |
  |---|---|---|---|
  | Qwen AoT `initial.pt` | bf16 (1,4096,64) | `c2d9d8b3…` / `4dfaec64…` | bit-identical (`smoke_aot_cut2`) |
  | Qwen native `initial.pt` | bf16 (1,4096,64) | `95a4306b…` / `750e9824…` | bit-identical (`smoke_native_full2`, `smoke_native_cut2_final`) |
  | LTX-2 AoT `initial_video` / `initial_audio` | bf16 (1,2310,128) / (1,51,128) | `4458e4cf…` / `8e52babb…` (files) | both bit-identical (`smoke_aot_cut2`) |
  | LTX-2 native `initial_video` / `initial_audio` | bf16 (1,2310,128) / (1,51,128) | `16e09464…` / `411c72d9…` (files); raw `55b90133…` / `24900b1b…` | both bit-identical (`smoke_native_full2`, `smoke_native_cut2_final`) |

**Outputs** (in `B/<model>/cpu_ref/<name>/`)
- **Final latents:** the packed fp32 loop outputs, at the same point as the device dumps.
  - Qwen-Image: `final.pt` (1,4096,64), checked equal to the step callback's last latents.
  - LTX-2: `final_video.pt` (1,2310,128) and `final_audio.pt` (1,51,128), captured where diffusers unpacks and denormalizes them. Main recorded video only.
- **Decoded output, lossless:**
  - Qwen-Image: `out.pt`, fp32 (1,3,1024,1024) in [0,1], and `out.png` written from it with the device PNG rule;
  - LTX-2: `out.pt`, fp32 (1,49,3,480,704) in [0,1], the device `out.pt` layout. Also `audio.pt` (the vocoder waveform) and `mid.png`.
- **`reference.json`:** argv, versions, threads, script hashes, input and output hashes, the schedule, per-step and per-forward times, the time of each phase, VmRSS / RssAnon / RssFile peaks with the host's lowest MemAvailable, and the injection and CPU-only checks.

**Metrics (`quality_metrics.py`)**
- Latents: rel-L2 and cosine in float64 (main's definitions).
- Frames: RGB uint8, per-frame PSNR (mean and worst), and SSIM per Wang 2004: 11x11 Gaussian window, σ 1.5, K1 0.01, K2 0.03, valid region, per channel then averaged. These are main's definitions. Worst SSIM and per-channel SSIM are added.
- **Preprocessing:** float frames become uint8 by one rule, `uint8(clamp(x·255 + 0.5, 0, 255))` in float32. bf16 is upcast exactly first.
  - This is the rule of both device CLIs' PNG writers (torchvision `save_image` on AoT; native `_save_image_pil`, which reproduces it pixel for pixel).
  - The device PNGs (Qwen-Image) are read as they are.
- **Self-test passes:**
  - PSNR matches the closed form;
  - SSIM matches an independent scipy build to 1e-12;
  - main's script gives identical numbers on the Qwen device PNGs (12.1470 dB, SSIM 0.645899);
  - native vs AoT device latents for LTX-2 give cosine 0.363 / 0.641, as reported above.

**Smoke runs** (real shapes, 2 steps; `--cut-layers 2` builds a 2-block DiT)

| run | DiT | load (text encoder + encode / DiT and decoders) | step | decode | peak VmRSS | wall |
|---|---|---|---|---|---|---|
| Qwen `smoke_native_full2` | 60 blocks, 20.43 B params | 4.2 s (warm cache) / 301.2 s | 78.9, 79.0 s | 19.0 s | 85.2 GiB | 486 s |
| Qwen `smoke_aot_cut2` | 2 blocks | 127.3 s (cold) / 13.6 s | 2.8 s | 19.1 s | 33.1 GiB | 169 s |
| LTX-2 `smoke_native_full2` | 48 blocks, 18.88 B params | 29.7 s / 290.6 s | 79.6, 79.5 s (forward) | 26.4 s | 117.3 GiB | 515 s |
| LTX-2 `smoke_aot_cut2` | 2 blocks | 362 s (cold Gemma-3 fp32 read) / 59.3 s | 3.4 s | 25.3 s | 48.0 GiB | 462 s |

- **LTX-2 memory:** the 117.3 GiB VmRSS includes reclaimable file-backed pages.
  - An extra 1-step run (`smoke_aot_full1.failed-1step`) measured, after the DiT load: RssAnon 82.0 GiB, RssFile 20.4 GiB, host MemAvailable 39.9 GiB.
  - Gemma-3's fp32 checkpoint is mmapped as it is: 44 GiB RssFile during the encode, 1.6 GiB anonymous.
  - `job_cpu_ref.sh` refuses to start below 112 GiB MemAvailable and refuses a second concurrent reference.
- **That 1-step run failed** in diffusers' scheduler: `FlowMatchEulerDiscreteScheduler` with `shift_terminal` divides by zero on a one-sigma schedule (IndexError in `index_for_timestep`). Both scripts now refuse `--steps < 2`.

**Estimated 50-step wall time:**
- **Qwen-Image:** about 1.2–1.3 h each. Load about 7 min cold, plus 50 x 79 s, plus a 19 s decode. The 2026-10-24 20-step reference had six slow early steps (117–133 s), which would add about 4 min.
- **LTX-2:** about 1.3 h each. Encode about 6 min cold, DiT load about 5 min, 50 x 79.5 s, and a 26 s decode.
- **The four references in sequence:** about 5–5.5 h.

**Known differences, for the report:**
- **Qwen-Image text:** diffusers encodes the real prompt length (17 tokens) and drops the all-ones mask. Both device runs zero-pad to `TEXT_SEQ_LEN` and attend the padding unmasked.
- **Text encoders:** fp32 here, at the device runs' own dtypes and placements there (plan D2, D8; for example Qwen2.5-VL in bf16 at TP4 on the AoT device, Gemma-3 in bf16 on the native device).
- **VAEs:** fp32 diffusers VAEs here. On the device, Qwen-Image decodes with an fp32 VAE on native and a compiled bf16 VAE on AoT. LTX-2 decodes on the host on both backends (in bf16 on native).

## Phase 4: Qwen-Image CPU fp32 references (50 steps)

2026-10-26 17:49–20:12 UTC, CPU only: the Neuron device was idle throughout (`neuron_users` empty at each start, no Neuron module or `/dev/neuron*` fd in either process).

**How they ran**
- One `systemd-run --user` unit, `cpuref_qwen_queue`, ran `B/scripts/job_cpu_ref_queue_qwen.sh`: the two Qwen-Image entries of `job_cpu_ref_queue_qwen_ltx2.sh`, one after the other, each through `job_cpu_ref.sh` under `/usr/bin/time -v`. Queue log: `B/cpu_ref_queue_qwen.log`.
- Script `cpu_ref_qwen.py` (sha256 `90e537fe…`; `cpu_ref_common.py` `f88cce3f…`), unchanged since the tooling stage.
- diffusers' `QwenImagePipeline`, every component in fp32, snapshot `75e0b4be…`, 1024x1024, 50 steps, `guidance_scale=4.0` (ignored: `guidance_embeds` false), `true_cfg_scale=1.0` with no negative prompt. 50 DiT calls in each run (one per step), as on both device backends.
- **Environment:** venv `/home/ubuntu/venvs/difflet-neuron`, Python 3.12.3, torch 2.9.1, diffusers 0.38.0, transformers 4.57.6. torch threads: 6 intra-op, 6 inter-op (6 physical cores; `time -v` shows 545% and 563% CPU).

**Injection and schedule**
- In both runs, the transformer's first `hidden_states` is bit-identical to the fp32 upcast of the device dump and rounds back to its bf16 bits.
- `B/qwen_image/cpu_ref/schedule_check.json` (`B/scripts/schedule_check_qwen.py`) rebuilds the device schedule from the snapshot's scheduler config:
  - the formula is the one both device CLIs use: `sigmas = linspace(1, 1/50, 50)`, dynamic shift `mu` = 0.693548 at image_seq_len 4096 (main `difflet/cli/orchestrators/qwen_image.py:277-283`; native `difflet/models/qwen_image/neuron_application.py:661-672`);
  - both references' recorded timesteps and sigmas are identical to it (max |Δt| 0).

| reference | initial (device q1 dump) | text-encoder load + encode | DiT + VAE load | denoise, 50 steps | step median (min–max) | decode | wall | peak RSS (`time -v`) | lowest host MemAvailable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ref_aot_noise_50` | `B/qwen_image/aot_main_tp4/q1/initial.pt` `c2d9d8b3…` (raw `4dfaec64…`) | 2.8 + 1.5 s (in the page cache) | 298.6 s | 4026.4 s | 80.54 s (79.62–82.82) | 19.3 s | 4352.4 s (1:12:39) | 85.44 GiB | 36.4 GiB |
| `ref_native_noise_50` | `B/qwen_image/native_tp4_hashcheck/q1/initial.pt` `95a4306b…` (raw `750e9824…`) | 126.0 + 1.5 s (evicted by the previous run) | 24.9 s | 4073.1 s | 81.52 s (79.59–83.50) | 19.0 s | 4249.0 s (1:10:53) | 85.44 GiB | 36.5 GiB |

**Outputs** (in `B/qwen_image/cpu_ref/<reference>/`, with `run.log`, `rc` 0 and `reference.json`)
- `ref_aot_noise_50`: `final.pt` `6b8a5dea…` (fp32 (1,4096,64)), `out.pt` `a0a9edf3…` (fp32 (1,3,1024,1024), range 0.029–0.972), `out.png` `464155ec…`.
- `ref_native_noise_50`: `final.pt` `b357f08a…`, `out.pt` `2c8ff472…` (range 0.035–0.927), `out.png` `09035397…`.
- I opened both PNGs: both show a red fox running through snow in a forest, on prompt.

**Metrics** (`quality_metrics.py`, main's definitions; frames: the device `out.png` as it is against `to_uint8` of the reference's fp32 `out.pt`; latents: the device `final.pt` (bf16) against the reference's `final.pt`, both packed loop outputs)

| pair (same noise) | PSNR | SSIM (R / G / B) | latent rel-L2 | latent cosine | file |
|---|---:|---:|---:|---:|---|
| AoT device q1 vs `ref_aot_noise_50` | **21.48 dB** | **0.9115** (0.915 / 0.911 / 0.908) | 0.276 | 0.9614 | `B/qwen_image/cpu_ref/metrics_aot_vs_ref.json` |
| native device q1 vs `ref_native_noise_50` | **16.11 dB** | **0.8234** (0.813 / 0.828 / 0.829) | 0.522 | 0.8722 | `B/qwen_image/cpu_ref/metrics_native_vs_ref.json` |
| control: native device vs `ref_aot_noise_50` (mismatched noise) | 11.96 dB | 0.634 | 0.798 | 0.675 | `…/control_native_dev_vs_aotnoise_ref.json` |
| control: AoT device vs `ref_native_noise_50` (mismatched noise) | 11.47 dB | 0.586 | 0.912 | 0.607 | `…/control_aot_dev_vs_nativenoise_ref.json` |

Reading the numbers:
- Both matched pairs sit well above the mismatched-noise controls, so the pairing is right and each device run follows its reference's trajectory.
- The native image keeps its reference's composition (a fox facing right in the same place) but not its pose: the reference fox leaps, the device fox trots. A late-trajectory divergence of that kind costs far more PSNR than it changes the picture. The AoT pair keeps the composition and position; its leg pose differs too (the reference fox gallops, the device fox walks; see "Quality vs CPU fp32 reference").
- Compare with care: the two backends start from different noise, so 21.5 vs 16.1 dB measures how far each run drifts from its own fp32 trajectory. It is not a pixel comparison of the backends. Both runs pass through the same known differences from the reference (below), and a single seed cannot separate them. *Inference:* the text conditioning (padding, below) is the largest common difference.
- The 20-step AoT figure from the bf16-reduce campaign (24.72 dB, SSIM 0.957, cosine 0.981, same AoT noise `c2d9d8b3…`) is not comparable: more steps leave more room to diverge.

**Known differences from the device runs (carried into the metrics)**
- **Text:** diffusers encodes the real 17 tokens, (1,17,3584) fp32, and drops the all-ones mask. Both device runs zero-pad to `TEXT_SEQ_LEN` and attend the padding unmasked.
- **Text encoder:** fp32 here. On the device, Qwen2.5-VL runs in bf16: at TP4 on the device for AoT, on rank 0's host for native.
- **DiT:** fp32 here, bf16 at TP4 on both device backends.
- **VAE:** fp32 here. On the device, native decodes with an fp32 VAE and AoT with a compiled bf16 VAE.

## Phase 4: LTX-2 CPU fp32 references (50 steps)

The references ran 2026-10-26 20:15–22:51 UTC and a 20-step cross-check ran 22:57–23:38 UTC, all on the CPU. The Neuron device was idle throughout: `neuron_users` was empty at each start, and no process loaded a Neuron module or opened a `/dev/neuron*` fd.

**How they ran**
- One `systemd-run --user` unit, `cpuref_ltx2_queue`, ran `B/scripts/job_cpu_ref_queue_ltx2.sh`. The script holds the two LTX-2 entries of `job_cpu_ref_queue_qwen_ltx2.sh` and runs them one after the other, each through `job_cpu_ref.sh` under `/usr/bin/time -v`. Queue log: `B/cpu_ref_queue_ltx2.log`.
- Script `cpu_ref_ltx2.py` (sha256 `e9fd953b…`; `cpu_ref_common.py` `f88cce3f…`), unchanged since the tooling stage.
- diffusers' `LTX2Pipeline` at snapshot `47da56e2…`, with every component in fp32: Gemma-3, connectors, a 48-block DiT of 18.88 B parameters, the video VAE, the audio VAE and the vocoder.
  - Workload: 480x704x49 at 24 fps, 50 steps, `guidance_scale=1.0`, so CFG is off.
  - Each run made 50 DiT calls, one per step, as on both device backends.
  - Prompt embeds: (1,1024,188160) fp32 with 13 real tokens (`mask_sum` 13).
- **Environment:** venv `/home/ubuntu/venvs/difflet-neuron`, Python 3.12.3, torch 2.9.1, diffusers 0.38.0, transformers 4.57.6.
  - torch threads: 6 intra-op and 6 inter-op, on 6 physical cores.
  - `time -v` shows 511% and 524% CPU.

**Injection and schedule**
- In both runs, the transformer's first `hidden_states` and `audio_hidden_states` are bit-identical to the fp32 upcast of the device dumps, and both round back to the dumps' bf16 bits.
- `B/ltx_2/cpu_ref/schedule_check.json` (from `B/scripts/schedule_check_ltx2.py`) rebuilds the device schedule from the snapshot's scheduler config.
  - Both backends use the same formula: `sigmas = linspace(1, 1/50, 50)`, and `mu = ltx_2_scheduler_mu` at `max_image_seq_len` = 2.05, with `shift_terminal` 0.1 from the config.
    - Native: `difflet/models/ltx_2/pipeline.py:493-517,1172-1186`.
    - AoT: main `difflet/models/ltx_2/pipeline.py:511-517,1181-1195`.
  - diffusers' `calculate_shift` groups the `mu` expression differently but gives the same bits.
  - Both references' recorded timesteps and sigmas are identical to the rebuilt schedule (max |Δt| 0, max |Δσ| 0).

| reference | initial (device q1 dumps, file / raw sha256) | Gemma-3 load + encode | DiT + decoders load | denoise, 50 steps | step median (min–max) | decode | wall | peak RSS (`time -v`) | peak RssAnon | lowest host MemAvailable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ref_aot_noise_50` | `B/ltx_2/aot_main_tp4_twocore/q1`: video `4458e4cf…` / `1b2149d9…`, audio `8e52babb…` / `fb75005f…` | 1.9 + 360.0 s (cold) | 329.2 s | 3980.5 s | 79.49 s (79.04–85.20) | 25.9 s | 4701.3 s (1:18:28) | 117.24 GiB | 85.29 GiB | 36.54 GiB |
| `ref_native_noise_50` | `B/ltx_2/native_tp4/q1`: video `16e09464…` / `55b90133…`, audio `411c72d9…` / `24900b1b…` | 2.2 + 358.6 s (evicted by the previous run) | 184.5 s | 4023.7 s | 80.13 s (79.60–85.47) | 27.5 s | 4600.7 s (1:16:49) | 116.00 GiB | 85.21 GiB | 36.41 GiB |

- In both runs, the 117 GiB RSS peak includes the mmapped fp32 checkpoint pages: Gemma-3 during the encode, then the DiT. Anonymous memory peaked at 85.3 GiB, and the host never had less than 36.4 GiB available.

**Outputs** (in `B/ltx_2/cpu_ref/<reference>/`, beside `run.log`, `rc` 0 and `reference.json`)

| file | `ref_aot_noise_50` | `ref_native_noise_50` |
|---|---|---|
| `final_video.pt`, fp32 (1,2310,128), the packed loop output | `5d88c6a0…` | `5e239f4d…` |
| `final_audio.pt`, fp32 (1,51,128) | `c1f702ac…` | `82d883e9…` |
| `out.pt`, fp32 (1,49,3,480,704) in [0,1], lossless frames | `95f016cc…`, range 0–0.952 | `ed701374…`, range 0–1 |
| `audio.pt`, the vocoder waveform, fp32 (1,2,48240) | `054b4f53…` | `c8904bb7…` |
| `mid.png`, frame 24 | `957a0a24…` | `6e08645e…` |

- **Content:** I opened both mid frames and compared each with its device counterpart.
  - **Native noise:** a red fox runs across deep snow in a conifer forest, with the same cyan cast as the native device output. So the cyan cast and the soft fox come from the workload (guidance 1, no CFG), not from the device. The device frame has the same fox, in the same pose and place; only the background trees differ.
  - **AoT noise:** a snowy forest with pine branches and a blurred red-brown animal, only loosely on prompt. The AoT device frame shares the colour scheme but not the layout: it shows a large smeared shape where the reference has trees and the animal.

**Metrics** (`B/scripts/metrics_ltx2.sh`, which runs `quality_metrics.py` with main's definitions)
- Frames: the device q1 `out.pt` (bf16, upcast exactly) against the reference's fp32 `out.pt`. Both are turned into RGB uint8 by the one rule `uint8(clamp(x·255 + 0.5, 0, 255))`, and there is no other preprocessing. All 49 frames are compared.
- Latents: the device `final_video.pt` / `final_audio.pt` (bf16) against the reference's fp32 ones, all packed loop outputs.

| pair | PSNR mean (worst) | SSIM mean (worst); R / G / B | video latent rel-L2 / cosine | audio latent rel-L2 / cosine | file |
|---|---:|---:|---:|---:|---|
| **AoT device q1 vs `ref_aot_noise_50`** (same noise) | **16.21 dB** (15.45, frame 5) | **0.660** (0.555, frame 48); 0.687 / 0.666 / 0.627 | 0.786 / **0.725** | 0.335 / 0.980 | `B/ltx_2/cpu_ref/metrics_aot_vs_ref.json` |
| **native device q1 vs `ref_native_noise_50`** (same noise) | **18.06 dB** (16.49, frame 27) | **0.811** (0.743, frame 15); 0.820 / 0.808 / 0.807 | 0.414 / **0.930** | 0.229 / 0.989 | `B/ltx_2/cpu_ref/metrics_native_vs_ref.json` |
| control: native device vs `ref_aot_noise_50` (mismatched noise) | 6.88 dB | 0.437 | 1.504 / 0.198 | 1.070 / 0.595 | `…/control_native_dev_vs_aotnoise_ref.json` |
| control: AoT device vs `ref_native_noise_50` (mismatched noise) | 9.13 dB | 0.490 | 1.128 / 0.315 | 0.997 / 0.563 | `…/control_aot_dev_vs_nativenoise_ref.json` |
| control: the two references against each other (noise only) | 8.15 dB | 0.470 | 1.414 / 0.176 | 1.039 / 0.522 | `…/control_ref_native_vs_ref_aot.json` |

**Reading the numbers**
- Both matched pairs sit well above the three mismatched-noise controls, so the pairing is right.
- **Native** follows its fp32 trajectory closely: video latent cosine 0.930, and the same fox in the same pose.
- **AoT** follows its trajectory much more loosely: video latent cosine 0.725 and 16.2 dB, with a different layout in the frames. Its audio latents stay close (cosine 0.980).
- As for Qwen-Image, the two backends start from different noise, so this compares each run with its own fp32 trajectory. It is not a pixel comparison of the backends.

**The AoT gap against the 2026-10-24 run** (an open finding, not explained yet)
- **2026-10-24:** single-core attention (`MODULE_20da5719`), bf16 all-reduce, 20 steps, same noise `4458e4cf…`. Against its 20-step CPU reference, it scored video latent cosine 0.991 and 34.14 dB (`/home/ubuntu/campaign/bf16-reduce/runs/ltx2/quality_bf16_vs_cpu.json`, mp4 against mp4).
- **Today:** the two-core module (`MODULE_e1043f48…`, `attention_cte_lnc0` with grid 2), 50 steps, scored cosine 0.725.
- **The reference side is ruled out:**
  - A 20-step reference with today's script from the same AoT dumps (`B/ltx_2/cpu_ref/xcheck_aot_noise_20/`, rc 0, wall 2414 s, `final_video.pt` `5b953937…`) gives a `final_video` bit-identical to the 2026-10-24 CPU reference (`/home/ubuntu/campaign/bf16-reduce/runs/cpu_ref/ltx_2/final_video.pt`; `torch.equal`, max |Δ| 0).
  - Against it, the 2026-10-24 device output (`bf16/warm1/final_video.pt` `b426bc92…`) gives video cosine 0.9907 and audio cosine 0.9994 (`xcheck_dev20_2026-10-24_vs_ref20.json`, `xcheck_ref20_vs_ref20_2026-10-24.json`).
  - That 2026-10-24 reference ran with `torch_xla` loaded (`xm.mark_step()` every step) and today's does not. The identical result shows that this did not change the numbers.
- **What remains is the device side:** 50 against 20 steps, and two-core against single-core attention. A device run can separate them; the device stays idle in this phase.
  - *Comparison:* Qwen-Image AoT went from cosine 0.981 at 20 steps to 0.961 at 50 steps, and LTX-2 native at 50 steps reaches 0.930. So more steps alone would not obviously cost this much.
  - *Recommendation for the next device block:* run LTX-2 AoT two-core at 20 steps from the same seed and compare it with `xcheck_aot_noise_20` and with `b426bc92…`. If it stays near 0.99, the steps explain the gap; if it falls, two-core attention does. Optionally also run single-core at 50 steps against `ref_aot_noise_50`.

**Known differences from the device runs (carried into the metrics)**
- **Text encoder:** Gemma-3 runs in fp32 here. Native runs it in bf16 on the device at TP4. AoT runs it in bf16 on the host (main's host pipeline loads with `torch_dtype=bfloat16`, `difflet/models/ltx_2/application.py:425-445`; the orchestrator's `_dtype` is bf16).
- **Connectors:** fp32 here, bf16 on both backends.
- **DiT:** fp32 here, bf16 at TP4 on both backends.
- **Decoders:** the video VAE, audio VAE and vocoder run in fp32 here. Both backends decode on the host in bf16.

## Quality vs CPU fp32 reference (part 1: Qwen-Image and LTX-2)

This section collects the quality numbers for the two models that have references, recomputed from scratch on 2026-10-26 after both reference blocks had finished. CPU only, the Neuron device stayed idle. The run was one `systemd-run --user` unit, `quality_part1`, running `B/scripts/quality_part1.sh` (sha256 `9518c2b9…`); all outputs are in `B/quality_part1/`, and `summary.txt` there lists every number with its input hashes.

**Pairs.** Each device q1 output is compared with the 50-step CPU fp32 reference (diffusers, fp32 everywhere, CFG off, the same schedule) that started from **that run's own initial latents**, bit-identical at the first DiT call (sections "Phase 4: … CPU fp32 references"):
- native device q1 against `ref_native_noise_50`;
- AoT device q1 against `ref_aot_noise_50`.

The two backends start from different noise, so this measures how far each device run drifts from its own fp32 trajectory. It is **not** a native-vs-AoT pixel comparison.

**Definitions** (`B/scripts/quality_metrics.py`, sha256 `8f6e1b18…`, main's `scripts/quality_vs_reference.py` definitions at `540e5f0`):
- **Frames:** RGB uint8. PSNR per frame with peak 255; the table gives the mean over frames and the worst frame. SSIM per Wang 2004: 11x11 Gaussian window, sigma 1.5, K1 0.01, K2 0.03, L 255, valid region only, computed per channel and then averaged over R, G and B; mean and worst over frames.
- **Latents:** relative L2 `||cand − ref|| / ||ref||` and cosine, both in float64, on the packed loop-output latents (Qwen-Image `final.pt` (1,4096,64); LTX-2 `final_video.pt` (1,2310,128) and `final_audio.pt` (1,51,128)). The device latents are bf16 and are upcast exactly.
- **Preprocessing, and nothing else:**
  - Qwen-Image: the device `out.png` as written by the CLI (8-bit RGB, 1024x1024) against the reference's fp32 `out.pt` (1,3,1024,1024) turned into uint8 by `uint8(clamp(x·255 + 0.5, 0, 255))`. That is the rule both device CLIs use for their PNG, so the reference passes through the same rounding as the device output.
  - LTX-2: the device bf16 `out.pt` (1,49,3,480,704) against the reference's fp32 `out.pt`, both turned into uint8 by the same rule, all 49 frames. No mp4 is involved on either side. No resizing, cropping, colour conversion or frame alignment anywhere.

**Tool sanity checks** (all pass)

| check | expected | result | file |
|---|---|---|---|
| `--selftest`: closed-form PSNR, SSIM against an independent scipy build, the uint8 rule, latent cosine of scaled and negated copies | all ok | PASS, SSIM equal to 1e-12 | `selftest.txt` |
| identical pair: Qwen-Image reference `out.pt` and `final.pt` against themselves | inf dB, SSIM 1, rel-L2 0, cosine 1 | inf / 1.000000 / 0 / 1.000000 | `sanity_identical_qwen.json` |
| identical pair: LTX-2 reference, 49 frames, video and audio latents against themselves | inf dB, SSIM 1, rel-L2 0, cosine 1 | inf (worst inf) / 1.000000 / 0 / 1.000000 | `sanity_identical_ltx2.json` |
| the reference's `out.png` against its own fp32 `out.pt` (checks the uint8 rule end to end) | inf dB | inf dB, SSIM 1 | `sanity_png_rule_qwen.json` |
| known pair: Qwen-Image native device PNG against the AoT device PNG; main's unmodified script gave 12.1470 dB, SSIM 0.645899 at the tooling stage | the same | 12.1470 dB, SSIM 0.645899 | `sanity_known_qwen_devices.json` |
| known pair: the 2026-10-24 LTX-2 AoT device latents (`b426bc92…`) against the 20-step reference (`xcheck_aot_noise_20`) | video cosine 0.9907 (LTX-2 phase 4) | 0.990696 | `sanity_known_ltx2_xcheck20.json` |

The four recomputed pairs equal the earlier `B/<model>/cpu_ref/metrics_*_vs_ref.json` with the same input hashes. The one exception is the last digit of one float64 cosine (Qwen-Image AoT 0.9614398560993154 against …155), from a multi-threaded reduction.

**Results**

| model | pair (same noise) | PSNR mean (worst frame) | SSIM mean (worst frame); R / G / B | latent rel-L2 / cosine | audio latent rel-L2 / cosine |
|---|---|---:|---:|---:|---:|
| Qwen-Image 1024x1024 | native device q1 vs `ref_native_noise_50` | **16.11 dB** (1 frame) | **0.823**; 0.813 / 0.828 / 0.829 | 0.522 / **0.872** | n/a |
| Qwen-Image 1024x1024 | AoT device q1 vs `ref_aot_noise_50` | **21.48 dB** (1 frame) | **0.912**; 0.915 / 0.911 / 0.908 | 0.276 / **0.961** | n/a |
| LTX-2 480x704x49 | native device q1 vs `ref_native_noise_50` | **18.06 dB** (16.49, frame 27) | **0.811** (0.743, frame 15); 0.820 / 0.808 / 0.807 | video 0.414 / **0.930** | 0.229 / 0.989 |
| LTX-2 480x704x49 | AoT device q1 vs `ref_aot_noise_50` | **16.21 dB** (15.45, frame 5) | **0.660** (0.555, frame 48); 0.687 / 0.666 / 0.627 | video 0.786 / **0.725** | 0.335 / 0.980 |

Files: `B/quality_part1/{qwen,ltx2}_{native,aot}_vs_ref.json`. Mismatched-noise controls, which score far lower and confirm the pairing, are in the two phase 4 sections (Qwen-Image 11.5–12.0 dB; LTX-2 6.9–9.1 dB).

**What the references themselves show** (I opened the Qwen-Image PNGs and a strip of LTX-2 frames 0, 12, 24, 36 and 48 of each reference, `B/ltx_2/cpu_ref/ref_*_noise_50/strip_0_12_24_36_48.png`, made from the lossless `out.pt` by `B/scripts/ref_strip.py`; AoT `ba3e8b9c…`, native `e059c7dc…`)
- **Qwen-Image: both references show the prompt.**
  - AoT noise: a sharp, photographic red fox galloping right across snow, snow-covered trunks behind. The AoT device image has the same fox in the same place and framing, but walking with all four feet down rather than galloping. *(This corrects "the AoT pair keeps the pose" in the Qwen-Image phase 4 section: it keeps the composition and position, not the leg pose.)*
  - Native noise: a red fox leaping right through a snowy forest; the fox's head is malformed (a smeared muzzle). The native device image has a fox in the same place and direction, trotting, with a clean head.
- **LTX-2 at g1.0: the CPU fp32 reference is soft and colour-cast too**, so this comes from the workload (guidance 1.0, no CFG), not from the device.
  - Native noise: on prompt. A small red fox runs across deep snow in a conifer forest, past a large trunk, toward the camera. The whole video has a strong cyan cast, the fox is smeared, and the last frames show checkerboard-like texture on the fox. The native device video matches it closely: the same fox, path, trunk and cyan cast, frame for frame.
  - AoT noise: only loosely on prompt. Snow-laden pine branches in the foreground against a blue-violet sky, blowing snow, and a blurred red-brown animal that appears from frame 12 onward and is never clearly a fox.
  - The AoT device video is **worse than its reference**, not just different: the same blue palette, but the frames are dominated by crumpled, foil-like smeared shapes and a large red-brown blob; neither the branches nor the animal of the reference are recognisable. That matches its low latent cosine (0.725) and is the open AoT finding of the LTX-2 phase 4 section.

**Reading the numbers**
- **Qwen-Image:** AoT 21.5 dB / 0.912, native 16.1 dB / 0.823. Both device images are on prompt and keep their reference's composition. The PSNR gap mostly reflects how late in the trajectory each run diverges (pose), which a single seed cannot attribute to a backend. Both device runs share the same known differences from the reference: the padded, unmasked text conditioning (the largest common difference, *inference*), bf16 text encoder and DiT, and for AoT a compiled bf16 VAE.
- **LTX-2:** native 18.1 dB / 0.811 with video cosine 0.930 follows its reference closely. AoT 16.2 dB / 0.660 with video cosine 0.725 does not, and its frames show artefacts that its reference does not. Audio latents stay close on both (cosine 0.989 and 0.980).
- PSNR on these generative outputs is low in absolute terms: a small shift of a fox's legs or of background trees costs several dB. SSIM and latent cosine track what the frames show better here than PSNR.

## Phase 4 part 2 preparation: CPU reference tooling for Wan2.1 and HunyuanVideo

Tooling only, 2026-10-26 23:50 to 2026-10-27 00:50 UTC. No 50-step reference has run yet. Everything ran on the CPU, and the Neuron device stayed idle: `neuron_users` was empty at every start, and no process loaded a Neuron module or opened a `/dev/neuron*` fd.

**Scripts** (in `B/scripts`, outside the repo; same pattern as part 1)
- `cpu_ref_wan.py` (sha256 `02b0b9e3…`) uses diffusers' own `WanPipeline`. The call follows `origin/feat/torch-native-backend:tests/manual/wan_e2e_reference.py`, changed to Wan2.1 and this workload: 480x832x9, guidance 5.0, the campaign prompt.
- `cpu_ref_hv.py` (`a4c90338…`) uses diffusers' own `HunyuanVideoPipeline`. The prompt is encoded by the pipeline's own `encode_prompt` with Llama-3 8B and CLIP-L in fp32. Both text encoders are then freed before the fp32 DiT loads.
- `cpu_ref_common.py` (`f88cce3f…`, unchanged from part 1) and `cpu_only.py` are shared.
- `job_cpu_ref.sh` gained the `wan2_1` and `hunyuan_video` entries. The part-1 version is kept as `job_cpu_ref.as-used-part1-2026-10-26.sh`.
- `schedule_check_wan.py` and `schedule_check_hv.py` check the schedule. `hv_text_cmp.py` compares text tensors. `smoke_part2.sh` runs the smokes, and `ref_brief.py` summarises a run.
- Prepared but not started: `job_cpu_ref_queue_wan_hv.sh`, the three 50-step references one after the other (Wan2.1, then HunyuanVideo with native noise, then HunyuanVideo with AoT noise).
- **Environment:** the same as part 1. Venv `/home/ubuntu/venvs/difflet-neuron` with torch 2.9.1, diffusers 0.38.0 and transformers 4.57.6. torch uses 6 intra-op and 6 inter-op threads.

**Wan2.1: the negative prompt both device runs used is the empty string**
- Neither q1 command passed `--negative-prompt`. Main's CLI has no such flag, and the branch's flag applies only to Qwen-Image.
- Both backends' transformer stages call `app(prompt=…, guidance_scale=5.0, …)` with no negative prompt:
  - native: branch `difflet/cli/orchestrators/wan.py`, the transformer stage and `_stage_transformer_neuron`;
  - AoT: main `difflet/cli/orchestrators/wan.py:256-268`.
- That call reaches the same `WanOrchestrator` (`difflet/models/wan/pipeline.py`). The branch's file and main's differ only in a CUDA-only timestep-dtype hook.
- With `negative_prompt=None`, `_negative_prompt_embeds` encodes `""`. The empty string becomes one valid token (EOS), and the rest of the 512 positions are zero-padded.
- With guidance > 1, the pipeline runs two DiT forwards per step and combines them as `uncond + 5.0·(cond − uncond)`.
- So **native and AoT used the same negative prompt, `""`**, and nothing needs recording as a difference. It is also diffusers' default (`negative_prompt or ""`). The model card's long negative prompt was not used by either backend.
- The reference passes `negative_prompt=""` to diffusers' `encode_prompt` (`max_sequence_length` 512). `prompt_clean` leaves both strings unchanged. Encoding gives 16 valid tokens for the prompt and 1 for the negative.
- The script counts 2 DiT calls per step and checks the order: the first call's `encoder_hidden_states` is the prompt, the second's is the negative (`cfg.first_calls_text` = cond, uncond, cond, uncond).

**HunyuanVideo: the embedded guidance**
- **Device side:** both backends build the guidance input as `torch.full([1], 6.0 × 1000, dtype=bfloat16)`:
  - native: `_bundle_from_tensors` in `difflet/models/hunyuan_video/pipeline.py`, with the application dtype bf16, which is the `…__bfloat16__tp4__…` weight-store key in the run log;
  - AoT: main `difflet/cli/orchestrators/hunyuan_video.py:320-324`.
  - Each runs one DiT forward per step with no CFG.
- **diffusers:** builds `guidance = tensor([guidance_scale]) · 1000` in the transformer dtype. It runs true CFG only when `true_cfg_scale > 1` and a negative prompt is given.
- **The reference** passes `guidance_scale=6.0`, `true_cfg_scale=1.0` and no negative prompt. The DiT receives **6000.0** (recorded from its first call), with one call per step (counted).
- **Known difference:** in bf16, 6000 rounds to **6016.0**, so both device DiTs received 6016. The fp32 reference keeps diffusers' 6000.0, the nominal 6.0 × 1000 (`--guidance-scale 6.016` would mirror the bf16 value).
  - The device timesteps are bf16 as well (`_batch_timestep(…, model dtype)`), for example 875 becomes 876.
  - The Wan device DiT gets bf16 timesteps too (999 becomes 1000, 750 becomes 752), while diffusers passes the exact int64 timestep (`schedule_check_*.json`).
  - Part 1 did not mirror bf16 rounding of the device DiT inputs either.
- **Text conditioning, checked against the AoT device** (`hv_text_cmp.py`, AoT q1 `work/llama.pt` / `clip.pt` against the reference's `text_embeds.pt`; file `B/hunyuan_video/cpu_ref/smoke_aot_cut2/text_vs_aot_device.json`):
  - the masks are identical: 13 valid tokens, a prefix of the 256;
  - Llama cosine on the valid rows is 0.99997 (worst row 0.99995);
  - the CLIP pooled cosine is 0.99995.
  - The native device calls the same `encode_prompt`, in bf16 on rank 0's host.

**Injection is bit-exact.** In every smoke, the transformer's first `hidden_states` equals the fp32 upcast of the device dump bit for bit, and it rounds back to the dump's bits.

| dump (device q1) | dtype, shape | file / raw sha256 | smokes |
|---|---|---|---|
| Wan2.1 `aot_main_tp4_twocore/q1/initial.pt` (equal to the native dump) | fp32 (1,16,3,60,104) | `03b6284b…` / `16aeaed7…` | `smoke_cut2`, `smoke_full2` |
| HunyuanVideo native `native_tp4_hashcheck/q1/initial.pt` | fp32 (1,16,16,40,64) | `d4a20de5…` / `419eace4…` | `smoke_native_full2` |
| HunyuanVideo AoT `aot_main_tp4/q1/initial.pt` | bf16 (1,16,16,40,64) | `e13a3f46…` / `f70d76ae…` | `smoke_aot_cut2` |

**Schedules** (`schedule_check_wan.py`, `schedule_check_hv.py`; max |Δt| 0 and max |Δσ| 0 in every smoke)
- **Wan2.1:** the check loads both device copies of `difflet/models/wan/pipeline.py` (the branch's and `aot-main`'s) as standalone modules. It runs their own `_load_scheduler` and `set_timesteps(N)`: `UniPCMultistepScheduler` from the snapshot, `flow_shift` 3.0. The two copies agree with each other and with the reference.
- **HunyuanVideo:** the check rebuilds both device paths' formula, `FlowMatchEulerDiscreteScheduler` from the snapshot (shift 7.0) with `sigmas = linspace(1, 0, N+1)[:-1]`. It is identical to the reference, and the reference's DiT guidance equals 6.0 × 1000.

**Outputs** (in `B/<model>/cpu_ref/<name>/`)
- `final.pt` is the fp32 loop output, at the same point as the device `final.pt`.
- `out_raw.pt` is the raw fp32 VAE decode (1,3,F,H,W) in about [−1,1], captured inside diffusers' call. It has the layout of the device `out.pt` and of Wan's A2 `host_fp32_frames.pt`.
- `out.pt` is diffusers' `postprocess_video("pt")`: (1,F,3,H,W) in [0,1]. Each run checks that it equals `(out_raw/2 + 0.5).clamp(0,1)` bit for bit.
- Also written: `mid.png`, `strip.png`, `reference.json`, and for HunyuanVideo `text_embeds.pt`.
- Each run also checks that the VAE input equals the de-normalised final latents: Wan's `latents_mean` / `latents_std`, HunyuanVideo's `/ scaling_factor`.
- HunyuanVideo decodes with diffusers' tiled VAE in fp32: 256x256 spatial tiles, 16-frame temporal tiles, frame-wise decoding.

**Smoke runs** (real shapes, 2 steps; `--cut-layers 2` builds a 2-block DiT, or 2 dual + 2 single blocks for HunyuanVideo)

| run | DiT | text load + encode | DiT load | step (steady) | decode | peak VmRSS / RssAnon | lowest MemAvailable | wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Wan `smoke_cut2` | 2 blocks | 0.9 + 147.1 s (cold umT5 read) | 0.8 s (lazy mmap) | 15.4 s | 56.5 s | 18.7 / 8.9 GiB | 120.3 GiB | 264 s |
| Wan `smoke_full2` | 40 blocks, 14.29 B, fp32 | 0.5 + 14.7 s | 0.1 s; read during step 1 (first forward 414.0 s) | **295.4 s** (forwards 147.9 + 147.4) | 52.4 s | 62.7 / 8.8 GiB | 113.0 GiB | 930 s |
| HV `smoke_aot_cut2` | 2+2 blocks | 115.5 + 5.7 s (cold Llama read) | 20.3 s | 18.0 s | 414.7 s | 33.2 / 29.4 GiB | 92.6 GiB | 597 s |
| HV `smoke_native_full2` | 20+40 blocks, 12.82 B, fp32 | 2.9 + 5.7 s | 173.9 s | **269.8 s** (265.8 for step 1) | 433.1 s | 56.0 / 55.1 GiB | 66.8 GiB | 1158 s |

- **Wan2.1:** the fp32 checkpoint is mmapped, so the DiT stays file-backed (RssFile 53.8 GiB) and anonymous memory peaks at 8.8 GiB. The HunyuanVideo DiT checkpoint is bf16, so the fp32 upcast is anonymous memory (49 GiB).
- Peak memory stays well under the 112 GiB start guard of `job_cpu_ref.sh`. Two references never run together.
- The steady Wan forward, 148 s for 14.3 B parameters at 4680 tokens, is slower per parameter than part 1's Qwen-Image, which took 80 s per step for 20.4 B parameters at 4096 tokens. *Not investigated.* The thread settings were kept identical to part 1.
- **Content after 2 steps:** the Wan strip already shows a blurred fox running across snow in front of a forest. The HunyuanVideo strip is still mostly noise, a dark blob on a blue gradient, as expected at σ 0.875.

**Estimated 50-step wall time**
- **Wan2.1:** about 4.2 h. That is about 2.5 min cold encode, then 50 × 295 s, plus about 4.5 min of cold DiT read during step 1, plus a 52 s decode. The plan's inferred estimate was 2–2.5 h.
- **HunyuanVideo:** about 3.9–4.0 h each. That is about 2 min cold encode and 3–5 min DiT load, then 50 × 268 s, plus a 7 min tiled decode.
- **All three in sequence:** about 12 h.

**For the metrics stage:** the Wan and HunyuanVideo device `out.pt` files (and Wan's `host_fp32_frames.pt`) are raw decodes, (1,3,F,H,W) in about [−1,1]. `quality_metrics.py` reads (1,F,3,H,W) in [0,1]. So each frames input, device and reference alike, goes through the same map before the uint8 rule: diffusers' `(x/2 + 0.5).clamp(0,1)`, permuted to (1,F,3,H,W). The reference's `out.pt` is exactly that map of its `out_raw.pt`.

## Quality vs CPU fp32 reference (part 2: Wan2.1 and HunyuanVideo)

2026-10-27, CPU only; the Neuron device was not touched. **Wan2.1 is done (both backends). HunyuanVideo native is done**: its native-noise reference ran 05:33–09:40 UTC (unit `cpuref_chain_hv_native`). **HunyuanVideo AoT is pending**: its AoT-noise reference moves to the next machine because this host is terminated at about 11:00 UTC.

**Wan2.1 reference `B/wan2_1/cpu_ref/ref_50`** (unit `cpuref_wan_50`, `cpu_ref_wan.py` `02b0b9e3…`, diffusers `WanPipeline` in fp32, 480x832x9, 50 steps, guidance 5.0, negative prompt `""`)
- **Finished** 2026-10-27 05:33:10 UTC, `rc` 0. Wall 16856.6 s (4.68 h; `/usr/bin/time` 4:40:59), against the 4.2 h estimate.
  - Steady step 316.2 s (median), against 295 s in the smoke. Step 1 took 493.4 s (the cold read of the mmapped DiT).
  - Steps 17–21 (02:20–02:57 UTC) took 368–487 s. In that window `ref_50_contention.log` shows load up to 10.5, with other processes on the Neuron device and a `pytest` from another session. The reference is CPU only, so this changes the time, not the result.
- **Memory:** peak VmRSS 62.6 GiB (`ru_maxrss` 62.6), peak RssAnon 8.77 GiB (the fp32 checkpoint is file-backed), lowest MemAvailable 112.5 GiB.
- **Checks** (`reference.json`; `B/wan2_1/quality_part2/ref50_brief.txt`, `schedule_check_ref50.json`):
  - **Injection is bit-exact:** the first DiT `hidden_states` has raw sha256 `16aeaed7…`, equal to the fp32 dump shared by both backends (`03b6284b…`), and it round-trips to the dump's bits.
  - 100 DiT calls in the order cond, uncond, …; the first DiT timestep is 999 (int64).
  - **50-step schedule:** `schedule_check_wan.py --steps 50` runs both device copies of `difflet/models/wan/pipeline.py`. They agree with each other, and the reference's timesteps and sigmas are identical (n 50, max |Δt| 0). `ok: true`, CPU only.
  - `out.pt` equals `(out_raw/2 + 0.5).clamp(0,1)` bit for bit, both inside the run and again in `frames_map.py --check-equal`.
  - CPU only: no Neuron module loaded, no `/dev/neuron*` fd.
- **Outputs:** `final.pt` `a0ae2918…`, `out_raw.pt` `bfa6b76b…`, `out.pt` `13385b64…`, `strip.png` `f6889b47…` (frames 0, 2, 4, 6, 8).

**Metrics run.** `B/scripts/quality_part2_wan.sh` (`ad1bf58e…`; the version prepared at the tooling stage is kept as `quality_part2_wan.as-prepared-2026-10-27.sh`) ran in the foreground at 05:34 UTC with 4 threads, beside the HunyuanVideo reference. Its output is in `B/wan2_1/quality_part2/`.
- The definitions are part 1's, unchanged: `quality_metrics.py` `8f6e1b18…`, the same uint8 rule, PSNR per frame (mean and worst), SSIM Wang 2004, and latent rel-L2 / cosine in float64 on the loop-output `final.pt` (1,16,3,60,104).
- **The one added step:** each raw (1,3,F,H,W) decode goes through `frames_map.py` (`14ef6365…`), `(float32(x)/2 + 0.5).clamp(0,1)` permuted to (1,F,3,H,W). That covers the native `out.pt` `8d8a1779…`, the A2 `host_fp32_frames.pt` `7654a612…`, the AoT device-VAE `out.pt` `4380f37c…` and the reference's `out_raw.pt`. The map log is `wan_frames_map.jsonl`.
- **All 9 frames are compared**, with no resizing, cropping or alignment. Native and AoT start from the same noise, so one reference serves both backends, and a direct native-vs-AoT pixel comparison is valid for Wan (unlike the other three models).

**Sanity checks** (all pass)

| check | expected | result | file |
|---|---|---|---|
| `--selftest` | PASS | PASS | `selftest.txt` |
| identical pair: `ref_50` `out.pt` and `final.pt` against themselves | inf, SSIM 1, rel-L2 0, cosine 1 | inf (worst inf) / 1.000000 / 0 / 1.000000 | `sanity_identical_wan.json` |
| `frames_map` of the reference's `out_raw.pt` against its own `out.pt` | bit-identical | true | `wan_frames_map.jsonl` |
| known pair: AoT device-VAE frames against the A2 host fp32 decode (phase 2: mean 12.41 dB, worst 9.36 dB) | the same | 12.410 / 9.359 dB | `sanity_known_wan_devvae_vs_host.json` |

**Results** (Wan2.1 480x832x9, 50 steps, seed 42, one shared noise)

| pair | PSNR mean (worst frame) | SSIM mean (worst frame); R / G / B | latent rel-L2 / cosine | file |
|---|---:|---:|---:|---|
| native device q1 vs `ref_50` | **22.66 dB** (20.32, frame 0) | **0.885** (0.840, frame 0); 0.879 / 0.892 / 0.885 | 0.256 / **0.967** | `wan_native_vs_ref.json` |
| AoT q1, A2 host fp32 decode of the device latents, vs `ref_50` | **21.40 dB** (18.98, frame 0) | **0.853** (0.797, frame 0); 0.845 / 0.862 / 0.851 | 0.299 / **0.954** | `wan_aot_hostvae_vs_ref.json` |
| AoT q1, device bf16 VAE frames (corrupted), vs `ref_50` | 11.72 dB (9.24, frame 3) | 0.321 (0.189, frame 4); 0.300 / 0.356 / 0.307 | same latents as the A2 row | `wan_aot_devicevae_vs_ref.json` |
| AoT q1 (A2) vs native device q1, direct, native as the reference side | 25.20 dB (22.13, frame 0) | 0.908 (0.866, frame 0); 0.905 / 0.914 / 0.906 | 0.215 / 0.977 | `wan_aot_hostvae_vs_native.json` |

- **Per frame, PSNR:**
  - native vs ref: 20.32, 22.21, 22.65, 22.96, 23.05, 23.01, 23.15, 23.24, 23.38;
  - AoT A2 vs ref: 18.98, 20.76, 21.30, 21.65, 21.75, 21.99, 22.03, 22.06, 22.05;
  - AoT device VAE vs ref: 16.68, 12.30, 10.79, 9.24, 10.90, 12.01, 11.15, 10.91, 11.53.
- **Reference timing and memory**, for the table: wall 16856.6 s (4.68 h), peak RSS 62.6 GiB (8.77 GiB anonymous).
- **Latent rel-L2 is not symmetric.** Native vs AoT gives 0.2146 with native's latents in the denominator, and 0.2159 with AoT's, which is phase 2's figure (`latents_vs_1025.json`). The native probe `final.pt` (`c033e163…`) holds the same values as `work/latents.pt` (`3cafa271…`) (`torch.equal`); only the serialization differs.

**What the frames show.** I opened the reference `strip.png` and the four mapped strips `strip_wan_{ref50,native_q1,aot_q1_hostfp32,aot_q1_devicevae}_02468.png`.
- **The reference shows the prompt:** a sharp red fox trotting right across snow, in front of dark trunks in a snowy forest, in the same pose and place in frames 2–8.
- **Frame 0 of the reference is noisy too.** It has coloured blocky speckle over the background, and the fox is clean. The native and AoT A2 frame 0 look the same. So the speckle comes from the model or VAE on the first frame, not from the device, and it is why frame 0 is the worst frame in every row.
- **Native and AoT A2** both show the same fox in the same place, pose and framing as the reference, frame for frame. The differences are in the background trunks and snow texture.
- **The AoT device-VAE strip** is clean at frame 0. Frames 2–8 show only a grey-brown fox silhouette under coloured block noise, the corruption already recorded in the Wan2.1 section. Against the reference it scores 11.7 dB / SSIM 0.32. That puts a number on the defect: about 10 dB and 0.53 SSIM below the host decode of the same latents.

**Reading the numbers**
- **Both backends follow the fp32 trajectory closely:** latent cosine 0.967 for native and 0.954 for AoT, SSIM 0.885 and 0.853. Native is slightly closer on every metric (+1.3 dB, +0.032 SSIM, +0.013 cosine).
  - This is one seed, so the gap is small and should not be over-read. The two backends also differ in umT5 precision: native runs umT5 in fp32 on the device, AoT in bf16 (Method).
- **Native and AoT are closer to each other** (25.2 dB / 0.908 / cosine 0.977) than either is to the reference. *Inference:* they share the bf16 DiT and bf16 timestep rounding (999 → 1000), which the reference does not.
- **Compared with part 1**, Wan sits above Qwen-Image native and the LTX-2 rows in SSIM and latent cosine, and close to Qwen-Image AoT (0.912 / 0.961).

### HunyuanVideo native vs its CPU fp32 reference

**The native row was done on the first host; the AoT row was done on 2026-10-27 on the replacement host** (section "HunyuanVideo AoT vs its CPU fp32 reference" below). The paragraph that follows is the plan as written before the host change. The AoT-noise reference `ref_aot_noise_50` (from the AoT bf16 noise `B/hunyuan_video/aot_main_tp4/q1/initial.pt` `e13a3f46…`) did not run on this host: it needs about 4 h and the host is terminated at about 11:00 UTC. The command is `job_cpu_ref_queue_wan_hv.sh` run (3); the inputs and expected hashes are in "Phase 4 part 2 preparation". After it, the AoT row is `quality_part2_hv.sh` with the AoT q1 (`aot_main_tp4/q1/out.pt` `33a8a130…` and `final.pt`) against it. Caveat for that row: the AoT VAE tiling caps frame PSNR at about 42.5 dB (`hunyuan_video_neuron.md`).

**Reference `B/hunyuan_video/cpu_ref/ref_native_noise_50`** (unit `cpuref_chain_hv_native` → `job_chain_hv_native_after_wan.sh`, `cpu_ref_hv.py` `a4c90338…`, diffusers `HunyuanVideoPipeline` in fp32, 320x512x61, 50 steps, embedded guidance 6.0, `true_cfg_scale` 1.0, no negative prompt)
- **Ran** 2026-10-27 05:33:24 to 09:40:00 UTC, `rc` 0. Wall 14792.6 s (4.11 h; `/usr/bin/time` 4:06:36), against the 3.9–4.0 h estimate.
  - Text encoders load 89.9 s (cold Llama read) + encode 5.8 s; DiT and VAE load 176.9 s.
  - Steps: median 279.6 s, steady 278.9 s, step 1 331.2 s; steps 23–31 took 288–300 s. Denoise 14083.5 s, tiled fp32 decode and postprocess 429.7 s.
- **Memory:** peak VmRSS 55.95 GiB (`ru_maxrss` 55.95), peak RssAnon 55.07 GiB (the bf16 checkpoint is upcast to fp32 in anonymous memory), lowest MemAvailable 65.71 GiB.
- **Checks** (`reference.json`; `B/hunyuan_video/quality_part2/ref_brief.txt`, `schedule_check_ref.json`):
  - **Injection is bit-exact:** the first DiT `hidden_states` has raw sha256 `419eace4…`, equal to the fp32 upcast of the native dump `native_tp4_hashcheck/q1/initial.pt` (`d4a20de5…`, checked by `--expect-initial-file-sha`), and it round-trips to the dump's bits.
  - **Calls:** 50 DiT calls, one per step (no true CFG). The DiT received guidance 6000.0 (fp32), the nominal 6.0 × 1000; the device DiTs receive 6016 (bf16). The first DiT timestep is 1000.0. Prompt mask: 13 valid tokens, a prefix of the 256.
  - **50-step schedule:** `schedule_check_hv.py --steps 50` (`fa9887f2…`) rebuilds the device formula (FlowMatchEuler, shift 7.0, `linspace(1,0,51)[:-1]`). The reference's timesteps and sigmas are identical (n 50, max |Δt| 0). The device DiT sees 48 of the 50 timesteps rounded by bf16 (for example 997.09 → 996). `ok: true`, CPU only.
  - `out.pt` equals `(out_raw/2 + 0.5).clamp(0,1)` bit for bit, inside the run and again in `frames_map.py --check-equal`.
  - CPU only: no Neuron module loaded, no `/dev/neuron*` fd.
- **Outputs:** `final.pt` `5087164d…` (1,16,16,40,64) fp32, `out_raw.pt` `db057a37…` (1,3,61,320,512) in [−1.26, 1.12], `out.pt` `7b98e201…`, `strip.png` `c87a04c9…` (frames 0, 15, 30, 45, 60).

**Metrics run.** `B/scripts/quality_part2_hv.sh` (`3c982113…`) ran in the foreground at 09:40 UTC (4 threads, nothing else running), output in `B/hunyuan_video/quality_part2/`. It is the Wan script's pattern with part 1's definitions unchanged (`quality_metrics.py` `8f6e1b18…`, the uint8 rule, PSNR per frame, SSIM Wang 2004, latent rel-L2 / cosine in float64 on the loop-output `final.pt` (1,16,16,40,64)). Before the real run it was dry-run against the 2-step smoke `smoke_native_full2` (`B/hunyuan_video/quality_part2_dryrun_smoke/`), which exercised every step.
- **Frames map:** `frames_map.py` (`14ef6365…`) maps the native `out.pt` `a17c9714…` (bf16, (1,3,61,320,512)), the AoT `out.pt` `33a8a130…` (for the control) and the reference's `out_raw.pt` to (1,61,3,320,512) in [0,1]. Log: `hv_frames_map.jsonl`.
- **Latents:** the native probe `final.pt` (`1e11ad6a…`, fp32) holds the same values as `work/latents.pt` (`b62d0891…`) (`torch.equal`).
- **All 61 frames are compared**, with no resizing, cropping or alignment.

**Sanity checks** (all pass)

| check | expected | result | file |
|---|---|---|---|
| `--selftest` | PASS | PASS | `selftest.txt` |
| identical pair: reference `out.pt` and `final.pt` against themselves | inf, SSIM 1, rel-L2 0, cosine 1 | inf (worst inf) / 1.000000 / 0 / 1.000000 | `sanity_identical_hv.json` |
| `frames_map` of the reference's `out_raw.pt` against its own `out.pt` | bit-identical | true | `hv_frames_map.jsonl` |
| mismatched-noise control: AoT device q1 (bf16 noise) against the native-noise reference | far below the matched pair | 9.35 dB (worst 5.69, frame 1), SSIM 0.472, latent rel-L2 0.906 / cosine 0.549 | `control_hv_aot_vs_refnative.json` |

**Results** (HunyuanVideo 320x512x61, 50 steps, seed 42, embedded guidance 6.0)

| pair (same noise) | PSNR mean (worst frame) | SSIM mean (worst frame); R / G / B | latent rel-L2 / cosine | file |
|---|---:|---:|---:|---|
| native device q1 vs `ref_native_noise_50` | **20.78 dB** (17.49, frame 52) | **0.869** (0.796, frame 52); 0.870 / 0.871 / 0.865 | 0.193 / **0.982** | `hv_native_vs_ref.json` |
| AoT device q1 vs `ref_aot_noise_50` | **19.65 dB** (16.20, frame 30) | **0.774** (0.649, frame 36); 0.785 / 0.787 / 0.751 | 0.253 / **0.969** | `B/hunyuan_video/quality_part2_aot/hv_aot_vs_ref.json` |

- **Per frame, PSNR** (native vs ref): 22.07, 22.46, 22.40, 20.52, 21.03, 21.73, 19.75, 19.46, 19.91, 22.37, 23.43, 22.61, 21.47, 20.38, 20.44, 19.69, 18.83, 18.02, 19.09, 19.75, 21.09, 22.31, 22.83, 23.08, 23.05, 23.94, 23.60, 23.11, 23.34, 24.12, 24.17, 22.81, 21.52, 21.63, 20.90, 20.65, 20.18, 21.27, 21.52, 21.48, 21.85, 21.75, 22.04, 20.60, 21.19, 20.79, 19.83, 18.64, 18.55, 18.48, 18.67, 17.97, 17.49, 17.59, 17.67, 18.00, 18.25, 18.81, 18.46, 20.17, 18.96. SSIM per frame ranges 0.796–0.93, lowest in frames 47–56.
- **Reference timing and memory**, for the table: wall 14792.6 s (4.11 h), peak RSS 55.95 GiB (55.07 GiB anonymous).

**What the frames show.** I opened the reference `strip.png` and the mapped strips `strip_hv_{ref_native,native_q1,aot_q1}_0_15_30_45_60.png`.
- **The reference shows the prompt:** a red fox running right across snow, with a snowy conifer forest (dark trunks, green pines) behind it, the camera panning with it. The frames are sharp apart from motion blur on the background.
- **The native device video matches it closely:** the same fox, pose, position and framing in frames 0, 15, 30 and 45, and the same background trees. Frame 60 shows the same fox in the same place with slightly different legs and background. The worst frames (47–56) are where that late divergence sits.
- **The AoT device video** (different noise, so not comparable with this reference) is also on prompt: a larger, closer red fox bounding right across snow in front of a conifer forest. It confirms the control scores low because the content differs, not because of a tooling fault.

**Reading the numbers**
- **Native follows its fp32 trajectory closely:** latent cosine 0.982 is the highest of any device row so far (Wan native 0.967), and SSIM 0.869 is in line with Wan (0.885 / 0.853) and Qwen-Image AoT (0.912). PSNR 20.8 dB is lower than Wan's because the video is longer (61 frames) and the fox and camera move, so a small late shift in legs and background costs several dB in the last frames.
- The known differences from the reference all remain: bf16 DiT and VAE on the device, the DiT guidance 6016 against 6000, bf16 timesteps, and the native text encoders in bf16 on the host. One seed, so this is one sample, not a distribution.
- The matched pair is 11.4 dB and 0.40 SSIM above the mismatched-noise control, which confirms the pairing.

## 2026-10-27, new host (same root volume): LTX-2 AoT 20-step diagnostic and cold definitions A and B

**Host:** a replacement trn2.3xlarge, with the same instance type, LNC=2, aws-neuronx-runtime-lib 2.35.30.0 and venvs. The **same EBS root volume** was re-attached to it, so the data, venvs, compiled roots and HF cache are the ones used above. The volume is gp3 at 3000 IOPS / 125 MB/s, as before, and it is not a fresh snapshot restore (it was created on 2026-10-23), so there is no first-read penalty.

**Preflight** (17:10 UTC):
- `fuser /dev/neuron*` was empty, and stayed empty before every run; no wait was needed.
- `neuron-ls`: 1 device, 4 cores.
- 130 GB free on `/`, 122 GiB MemAvailable.

**Device phase:** one `systemd-run --user` unit, `device_1027`, ran `B/scripts/job_device_1027.sh` (`6331dda6…`), five runs one after the other, 17:12:53–18:10:24 UTC (log `B/device_1027.log`).
- All five runs returned rc 0 and released the device (`device_released: true`).
- The roots were listed before and after the phase (`B/device_1027_prep/roots_{before,after}.txt`, from `roots_snapshot_1027.sh`). The NEFF counts, file counts and newest file are unchanged in all three roots: no run compiled a NEFF or wrote to a weight store.

### LTX-2 AoT two-core at 20 steps (diagnostic, quality only)

**Question:** the 50-step two-core AoT q1 scored video latent cosine 0.725 against its CPU fp32 reference, while the 2026-10-24 single-core 20-step run scored 0.991 (LTX-2 phase 4 section). Is the drop caused by the 50 steps, or by two-core attention?

**Run:** `B/ltx_2/aot_main_tp4_twocore_steps20_1027/q1`, 17:12:54–17:25:30 UTC, main `540e5f0` (clean), `difflet` from `/home/ubuntu/aot-main` (precheck and probe).
- It is identical to the q1 run of `aot_main_tp4_twocore`, apart from the step count:
  - the same root `B/roots/ltx_2_aot_main_tp4_twocore` (`--root-cell`), loading its existing two-core NEFF (`MODULE_e1043f48…`, `attention_cte_lnc0` with grid 2);
  - `NEURON_RT_VIRTUAL_CORE_SIZE=2`, `DIFFLET_BACKEND=trainium`, `--tp-degree 4 --cp-degree 1`, 480x704x49, guidance 1.0, seed 42, the same prompt;
  - lossless `out.pt`, plus main's `DIFFLET_DUMP_LATENTS_DIR` and `DIFFLET_STEP_TIMES_FILE`.
- **How 20 steps were run:** `bench4m.py` hardcodes `--steps 50`. A copy, `B/scripts/bench4m_steps.py` (`3e48846e…`), sits beside it. Its only change is a `--steps N` option (default 50) that replaces the hardcoded value and is recorded as `steps_arg` in `summary.json`. `bench4m.py` (`6c52b12e…`) was left as used. A dry run showed the command equal to q1's apart from `--steps 20` and the run paths.
- **Timing (not a figure):** `steps.jsonl` has 20 rows, median 401.6 ms, the two-core speed (q1 at 50 steps: 400.8 ms; single-core 433.5 ms). Peak HBM 9.75 GiB per core. The 754.1 s wall is a first run on an empty page cache, so it is not used as a timing figure.

**Noise:** `initial_video.pt` `4458e4cf…` and `initial_audio.pt` `8e52babb…` are bit-identical (`torch.equal` and file sha256) to the 50-step two-core q1 dumps and to the 2026-10-24 single-core run (`B/ltx_2/diag20_1027/initial_noise_check.json`).

**Result: two-core attention is not the cause.**
- The run's `final_video.pt` (`b426bc92…`) and `final_audio.pt` (`512298a5…`) are **byte-identical** to the 2026-10-24 single-core 20-step run (`/home/ubuntu/campaign/bf16-reduce/runs/ltx2/bf16/warm1/`): same file sha256, and `torch.equal` is true for both.
- So at 20 steps, two-core and single-core attention give the same bits, and the 50-step gap comes from running 50 steps rather than from the attention split.

**Metrics** (`B/scripts/quality_ltx2_diag20_1027.sh`, `2ec1b904…`; `quality_metrics.py` `8f6e1b18…` with part 1's definitions, selftest PASS; output `B/ltx_2/diag20_1027/`)

| pair (same noise) | PSNR mean (worst frame) | SSIM mean (worst frame); R / G / B | video latent rel-L2 / cosine | audio latent rel-L2 / cosine | file |
|---|---:|---:|---:|---:|---|
| **AoT two-core, 20 steps (2026-10-27) vs CPU fp32 `xcheck_aot_noise_20`** | **35.45 dB** (33.44, frame 48) | **0.979** (0.970, frame 48); 0.980 / 0.978 / 0.978 | 0.140 / **0.9907** | 0.035 / **0.9994** | `aot_twocore20_vs_ref20.json` |
| known: 2026-10-24 single-core 20-step device vs the same reference | latents only | | 0.140 / 0.9907 | 0.035 / 0.9994 | `known_singlecore20_vs_ref20.json` |
| two-core 20 vs single-core 20, device vs device | | | 0 / 1.000 | 0 / 1.000 | `twocore20_vs_singlecore20.json` |
| for scale: AoT two-core q1, 50 steps vs `ref_aot_noise_50` (part 1) | 16.21 dB | 0.660 | 0.786 / 0.725 | 0.335 / 0.980 | `known_twocore50_vs_ref50.json` |
| sanity: `xcheck_aot_noise_20` vs itself | inf | 1.000 | 0 / 1.000 | 0 / 1.000 | `sanity_identical_ref20.json` |

- The 20-step frames are compared losslessly (`out.pt` `347910e6…` against the reference's fp32 `out.pt`, all 49 frames). The 34.14 dB recorded for 2026-10-24 was mp4 against mp4.
- **Content:** I opened the strips of frames 0, 12, 24, 36 and 48 (`strip_diag20_…png`, `strip_ref20_…png`). They match frame for frame: a snowy conifer forest under a blue sky, a dark animal shape, a red-brown blob and spraying snow. Both are only loosely on prompt, which is the guidance-1.0 workload.

**Conclusion.** At 20 steps, the two-core AoT run is near 0.99 (video cosine 0.9907, audio 0.9994, 35.5 dB, SSIM 0.979), and it is bit-identical to single-core. The 0.725 at 50 steps is therefore a property of the 50-step bf16 trajectory from this noise, not of two-core attention.
- *Inference, not tested:* a single-core 50-step run would give the same bits as the two-core one, since the two are bit-identical at 20 steps with the same kernels per step.
- What stays open is why this noise's 50-step bf16 trajectory drifts so far from fp32 (video cosine 0.725), while LTX-2 native at 50 steps reaches 0.930 from other noise. One seed per backend cannot separate the noise from the backend. The planned CPU bf16 vs fp32 baseline (plan, "Added 2026-10-27") measures exactly this: a CPU bf16 run from the same AoT noise against `ref_aot_noise_50`.

### Cold e2e, definitions A and B (native Wan2.1 and LTX-2, plan phase 6)

**Runs:**
- Native backend at HEAD `9d425d5` (clean). The difflet code is identical to `4f5d46a` and `009d3dd`, the commits of the Wan hash check and the LTX-2 cell (`git diff --stat … -- difflet` is empty).
- Flags as in the earlier cells: `--backend neuron --tp-degree 4`, plus the harness's 50 steps, guidance, seed 42 and prompt. The dtypes are unchanged: Wan umT5 and VAE in fp32, Gemma-3 in bf16 on the device.
- Each run is one cold generate through `bench4m.py` `--drop-cache`:
  - `sync; drop_caches`, which left `Cached` at 0.35 GiB with the checkpoint 0% resident in every run;
  - a fresh process and a fresh work dir;
  - all weights read from disk inside the timed run.
- **A:** the existing store (Wan: `B/roots/wan21_native_tp4_copy1025`, DiT bf16 store; LTX-2: `B/roots/ltx2_native_tp4`, Gemma-3 bf16 store).
- **B:** `DIFFLET_NEURON_WEIGHT_STORE=0` on the same roots, so the original checkpoints are read and nothing is written.
  - The B run logs have no `[weight store]` line, against 9 (Wan) and 7 (LTX-2) in A.
  - Wan B reads the fp32 transformer checkpoint and casts it to bf16. LTX-2 B reads the Gemma-3 checkpoint.
- New cells, so nothing existing was overwritten: `B/{wan2_1,ltx_2}/native_tp4_{A,B}_1027/cold/`. One run each; none looked anomalous (below).

| model | run | wall e2e | vs old-host A | B vs A | output | peak HBM per core |
|---|---|---:|---:|---:|---|---|
| Wan2.1 | **A** (2026-10-27) | **533.89 s** | −8.85 s (−1.6%) against 542.74 s | | mp4 `9b914e81…`, latents `3cafa271…` | 15.15 GiB DiT cores, 18.38 VAE core |
| Wan2.1 | **B** (2026-10-27) | **762.55 s** | | **+228.66 s (+42.8%)** | the same mp4 and latents | 15.15 / 18.06 GiB |
| LTX-2 | **A** (2026-10-27) | **603.23 s** | +8.47 s (+1.4%) against 594.76 s | | mp4 `cd11731b…`, latents `b9c745bd…` | 16.51–16.52 GiB |
| LTX-2 | **B** (2026-10-27) | **785.40 s** | | **+182.17 s (+30.2%)** | the same mp4 and latents | 16.52 GiB |

- **Outputs:**
  - Wan2.1 A and B both give the mp4 and `work/latents.pt` of the 2026-10-25 reference cell and of the 2026-10-26 hash check.
  - LTX-2 A and B both give those of every run of `native_tp4` (cold, gen0-3).
  - So B produces exactly the output of A for both models: reading the original checkpoint and casting on load gives the same bf16 weights as the store.
- **The hosts are comparable.** A on this host is within 1.6% of the old host's A, and the components match to the second:
  - Wan: DiT store read 217.41 vs 217.04 s, umT5 172.56 vs 172.52 s.
  - LTX-2: Gemma-3 store 178.78 vs 178.78 s, DiT checkpoint 287.34 vs 286.55 s.
  - The differences are start-up and one in-run compile (tables below).
- **Against AoT cold** (`*` = against Wan2.1 AoT A2 `--host-vae`, 449.4 s):

  | model | AoT cold | native A on this host | native B |
  |---|---:|---:|---:|
  | Wan2.1 | 449.4 s* | +18.8%* | **+69.7%*** |
  | LTX-2 | 778.1 s | −22.5% | **+0.9%** |

  - Under B, native LTX-2 cold is level with AoT's, and native Wan2.1 cold is 1.7x AoT's. AoT cold always reads its bf16 pre-shards; AoT has no B.

**Wan2.1 cold breakdown** (rank 0 probe spans and the app phase seconds, seconds from driver start; `B/device_1027_prep/breakdown_wan.json` from `breakdown_1027.py` `6558d7ab…`)

| part | A, old host (2026-10-25, `8b2408c`) | **A** (2026-10-27) | **B** (2026-10-27) |
|---|---:|---:|---:|
| driver start to rank start (torchrun) | 14.59 | 15.06 | 15.26 |
| rank imports and set-up, to the load | 24.77 | 18.28 | 27.52 |
| load (`app_load`) | 410.30 | 411.28 | **629.14** |
| — DiT weights | 217.04 (store) | 217.41 (store; every rank `hit 7.50 GB` in 216.0–217.4 s) | **435.07** (fp32 checkpoint, cast to bf16) |
| — umT5 checkpoint (fp32) | 172.52 | 172.56 | 172.53 |
| — warm-up forwards, DiT / umT5 | 13.44 / 4.88 | 13.95 / 4.86 | 13.91 / 4.90 |
| load end to encode (in-run NEFF compile) | 6.43 | 0.06 | 0.07 |
| encode (cond + uncond) | 2.23 + 0.09 | 1.95 + 0.09 | 1.97 + 0.10 |
| denoise, 50 steps (step median) | 53.25 (1062.4 ms) | 53.39 (1064.7 ms) | 53.40 (1065.5 ms) |
| generate ranks exit | 511.6 | 500.03 | 727.39 |
| VAE launch: load / decode / save (stage) | 9.65 / 7.77 / 0.50 (20.98) | 11.01 / 7.93 / 0.50 (22.72) | 10.20 / 8.47 / 0.54 (22.92) |
| **harness wall** | **542.74** | **533.89** | **762.55** |

**LTX-2 cold breakdown** (`B/device_1027_prep/breakdown_ltx_{A,B}.json`)

| part | A, old host (2026-10-26, `009d3dd`) | **A** (2026-10-27) | **B** (2026-10-27) |
|---|---:|---:|---:|
| driver start to rank start (torchrun) | 14.86 | 15.53 | 15.45 |
| rank imports and set-up, to the DiT load | 14.15 | 19.45 | 21.11 |
| DiT load: bf16 checkpoint / build on meta / warm-up (span) | 286.55 / 9.87 / 12.86 (310.18) | 287.34 / 10.61 / 13.26 (312.14) | 287.23 / 10.10 / 13.11 (311.34) |
| Gemma-3 load: weights / warm-up (span) | 178.78 store / 1.36 (180.21) | 178.78 store / 1.42 (180.29; rank 0 `hit 5.88 GB` in 177.7 s) | **358.32 checkpoint** / 1.44 (**359.84**) |
| prompt encode (connectors) | 25.31 (24.95) | 25.40 (25.04) | 25.53 (25.17) |
| denoise, 50 steps (step median) | 17.56 (350.8 ms) | 17.59 (351.6 ms) | 17.60 (351.8 ms) |
| generate stage ends | 568.30 | 576.80 | 757.75 |
| host decode: pipeline load / decode / save (total) | 3.01 / 19.30 / 0.73 (25.41) | 3.07 / 19.07 / 0.74 (25.36) | 3.31 / 19.56 / 0.78 (26.31) |
| **harness wall** | **594.76** | **603.23** | **785.40** |

- **Reading the breakdowns:**
  - B adds exactly the difference between reading the original checkpoint and reading the store: Wan DiT 435.07 − 217.41 = 217.7 s of the 228.7 s; LTX-2 Gemma-3 358.32 − 178.78 = 179.5 s of the 182.2 s. The rest is start-up spread.
  - The earlier estimates (Wan DiT 217 vs 435 s; Gemma-3 178.5 vs 358 s, plan addendum) are confirmed.
  - Denoise, encode and decode are the same under A and B.
- **Against the old host:**
  - Wan2.1 A is 8.85 s faster. The old run spent 6.43 s on a small NEFF compile between load and encode; that NEFF now exists in the root copy (the hash check wrote it on 2026-10-26), so this run compiled nothing. Its ranks also reached the load 6.5 s sooner.
  - LTX-2 A is 8.47 s slower:
    - 6.0 s of it comes before the DiT load: 19.45 vs 14.15 s from rank start, plus 0.7 s in the driver launch. This is the torchrun start-up jitter already seen in the LTX-2 warm runs (19.2–22.7 s).
    - The other 2.5 s is spread over the DiT load (+2.0 s) and the encode, denoise and decode, each within 0.3 s.
- No run was repeated: every component time agrees with the old host or with the expected checkpoint read, and the outputs are bit-identical.

**Decisions on this host** (problem, choice, reason)
1. **20 steps without editing the as-used harness.** Problem: `bench4m.py` hardcodes `--steps 50`, and a duplicate `--steps` in the cell flags would be overridden by the later one. Choice: a copy `bench4m_steps.py` with a `--steps` option that defaults to 50, plus a dry run to compare the command with q1's. Reason: the as-used harness stays byte-identical (`6c52b12e…`), and the change is one recorded option.
2. **The diagnostic ran first, on the empty page cache, without `--drop-cache`.** Reason: it is quality only, and its outputs depend only on the inputs. Its wall time is not reported as a figure.
3. **Metrics after the device phase.** The frame metrics ran only after the last timed run (18:11 UTC, as a user unit). Exception: a latent-only check of about 5 s ran at about 17:26, during Wan A's disk-bound load; its effect is negligible, and Wan A's load components equal the old host's to 0.4 s.
4. **A and B on the existing roots, in new cells.** Reason: definitions A and B differ only in the env switch. The roots were checked unchanged after the phase, and the earlier cells are untouched.
5. **One run per definition.** Reason: no run was anomalous (above).
6. **HunyuanVideo AoT CPU reference started right after the device phase** (18:12:52 UTC, before the 19:30 limit), after the diagnostic metrics had finished, so that the reference had the host to itself.

### HunyuanVideo AoT vs its CPU fp32 reference

**Reference `B/hunyuan_video/cpu_ref/ref_aot_noise_50`** (unit `cpuref_hv_aot` → `job_cpu_ref_queue_wan_hv.sh` run (3), `cpu_ref_hv.py` `a4c90338…`, same settings as `ref_native_noise_50`; started from the AoT bf16 noise `aot_main_tp4/q1/initial.pt` `e13a3f46…`)
- **Ran** 2026-10-27 from 18:12:52 UTC on the replacement host, `rc` 0. Wall 15387.4 s (4.27 h). Text encoders load 115.7 s + encode 6.1 s; DiT and VAE load 195.2 s; denoise 14590.7 s (step median 285.8 s); tiled fp32 decode and postprocess 470.0 s. Steps 5–9 were slower (325–389 s) while the phase-3 unit tests ran beside it; the result does not depend on speed.
- **Memory:** peak VmRSS 55.66 GiB, RssAnon 54.79 GiB, lowest MemAvailable 66.91 GiB.
- **Checks** (`B/hunyuan_video/quality_part2_aot/ref_brief.txt`, `schedule_check_ref.json`): injection bit-exact (`injection True True`), CPU only, 50 DiT calls with guidance 6000.0 (device 6016 in bf16), first timestep 1000.0, 13 valid prompt tokens; the 50-step schedule check is `ok: true`; `out.pt` equals the mapped `out_raw.pt` bit for bit.
- **Outputs:** `final.pt` `d6a1c808…`, `out_raw.pt` `bf2de541…`, `out.pt` `a3803c95…`, `strip.png` `3815f142…`.

**Metrics run.** `B/scripts/quality_part2_hv_aot.sh` (the native script's pattern for the AoT pair, same `quality_metrics.py`, `frames_map.py`, definitions and uint8 rule), output in `B/hunyuan_video/quality_part2_aot/`.

| check | expected | result | file |
|---|---|---|---|
| `--selftest` | PASS | PASS | `selftest.txt` |
| identical pair: reference against itself | inf, SSIM 1, rel-L2 0, cosine 1 | inf / 1.000 / 0 / 1.000 | `sanity_identical_hv_aot.json` |
| mismatched-noise control: native device q1 against the AoT-noise reference | far below the matched pair | 8.95 dB (worst 5.58), SSIM 0.464, latent rel-L2 1.052 / cosine 0.526 | `control_hv_native_vs_refaot.json` |

| pair (same noise) | PSNR mean (worst frame) | SSIM mean (worst frame); R / G / B | latent rel-L2 / cosine |
|---|---:|---:|---:|
| AoT device q1 vs `ref_aot_noise_50` | **19.65 dB** (16.20, frame 30) | **0.774** (0.649, frame 36); 0.785 / 0.787 / 0.751 | 0.253 / **0.969** |
| native device q1 vs `ref_native_noise_50` (above) | 20.78 dB (17.49, frame 52) | 0.869 (0.796, frame 52) | 0.193 / 0.982 |

- **Per frame, PSNR** (AoT vs ref): 24.48, 23.74, 23.56, 22.97, 22.81, 22.19, 21.29, 20.32, 20.08, 19.89, 19.47, 19.54, 19.55, 20.04, 20.93, 21.53, 22.22, 22.09, 21.59, 20.80, 19.88, 20.10, 20.03, 20.73, 19.47, 18.98, 18.73, 18.73, 18.05, 16.63, 16.20, 16.53, 17.07, 17.79, 17.45, 16.52, 16.34, 17.14, 17.45, 17.65, 17.71, 17.98, 18.19, 18.49, 18.19, 18.43, 18.23, 18.16, 17.65, 18.20, 18.31, 19.07, 20.24, 20.35, 21.35, 21.07, 20.83, 20.89, 21.28, 21.59, 21.59. SSIM per frame ranges 0.649–0.905.

**What the frames show.** The reference strip shows the prompt: a large red fox running right across snow in front of a conifer forest, close to the camera. The AoT device strip (`strip_hv_aot_q1_0_15_30_45_60.png`) matches it in fox, pose, position and framing at frames 0, 15, 30, 45 and 60; legs, tail and background branches differ in detail.

**Reading the numbers**
- **Both HunyuanVideo backends follow their fp32 trajectories well**, native somewhat closer: latent cosine 0.982 vs 0.969, SSIM 0.869 vs 0.774, PSNR 20.78 vs 19.65 dB. Each is measured against the reference from its own noise, so this is two single samples from different starting points, not a paired comparison.
- The matched AoT pair is 10.7 dB and 0.31 SSIM above its mismatched-noise control, which confirms the pairing.
- Unlike LTX-2 AoT (video cosine 0.725 at 50 steps), HunyuanVideo AoT shows no large 50-step drift.

## Phase 3: bf16 VAE and umT5 (2026-10-27, new host)

**A supplement.** The fp32 numbers in the summary table stay the headline. bf16 is **not** the default anywhere: every switch below is an opt-in flag, and making one the default needs the user's approval (plan, "Reporting and defaults").

**What ran.** The plan's "Phase 3 device validation runbook" was run on the replacement host from 2026-10-27 22:35 to 2026-10-28 00:00 UTC.
- **Code:** native backend at `eab8fc8` (clean), which contains the opt-in commits `e809595`, `c706e39` and `1832960`.
- **Preconditions** (22:33 UTC):
  - `fuser /dev/neuron*` was empty, and no `cpu_ref_` process was running. `ref_aot_noise_50` had finished with `rc` 0.
  - 122 GiB MemAvailable and 129 GB free on `/`.
  - Phase 6 (B) had already run, so nothing had to run before step 4.
- **Supervision:**
  - Each device step ran as a `systemd-run --user` unit, one at a time: `p3_gates2` for steps 1-2 and `p3_cells` for steps 4-5. The metrics ran afterwards as `p3_quality`, CPU only.
  - Every unit was watched with bounded wait loops.
  - Every run returned rc 0 and released the device (`device_released: true`).
- **Evidence:** `B/phase3/` (`gates.log`, `cells.log`, `f13/`, `vae_wan/`, `vae_qwen/`, `quality/`), plus the new cells `B/qwen_image/native_tp4_p3/` and `B/wan2_1/native_tp4_p3/` on the fresh roots `B/roots/{qwen_image,wan21}_native_tp4_p3`. No existing cell or root was written.

### Gates (runbook steps 1-3)

**Step 1, F13: umT5 bf16 vs fp32 at TP4.** Run with `f13_umt5_bf16_check.py` (`5f549677…`) through the native prompt path, with the weight store off; output in `B/phase3/f13/`. The embeddings were identical on all 4 ranks: **PASS**.

| prompt | valid tokens | cosine (gate ≥ 0.995) | worst token cosine | max abs | rel-L2 | after the bf16 cast the DiT sees: cosine / max abs |
|---|---:|---:|---:|---:|---:|---|
| `"a cinematic shot of a red fox running through a snowy forest"` | 16 | **0.999481** | 0.998515 | 3.11e-2 | 3.22e-2 | 0.999479 / 3.13e-2 |
| `""` (the CFG negative prompt) | 1 | **0.999951** | 0.999951 | 4.61e-3 | 1.01e-2 | 0.999953 / 3.91e-3 |

- **Padding:** the padded rows are exactly zero in both dtypes.
- **Per-rank bytes:** fp32 5.68 GB, bf16 2.84 GB.
- **Load:**
  - fp32: 192.5 s from a cold page cache (checkpoint read 172.4 s, warm-up 13.8 s).
  - bf16: 11.3 s, but it ran right after the fp32 load on a warm page cache, so it is **not** a cold figure. The cold bf16 figure is the Wan cell's store read below.

**Step 2, VAE gate: bf16 vs fp32 device decode of the same latents.** Run with `vae_bf16_check.py` (`b87a3338…`) on one core, with the phase-2 q1 latents. Both models **PASS**.

| model | bf16 vs fp32 device, PSNR mean (worst frame) / SSIM (gate ≥ 50 dB) | fp32 device vs diffusers fp32 host | bf16 device vs host | decode s, first / repeat: fp32 → bf16 |
|---|---|---|---|---|
| Wan2.1, 480x832x9 | **56.18 dB** (55.89, frame 0) / 0.99932 | 97.44 dB / 1.00000 | 56.18 dB / 0.99932 | 172.2 / 3.38 → 122.2 / 1.85 |
| Qwen-Image, 1024x1024 | **56.72 dB** / 0.99881 | 99.49 dB / 1.00000 | 56.72 dB / 0.99881 | 84.0 / 0.92 → 52.7 / 0.41 |

- Both pass at the level of HunyuanVideo's bf16 decoder (56.07 dB). Wan's per-frame range is 55.89-56.50 dB.
- The fp32 device decode matches the host decode to 97-99 dB, so the device path itself is faithful.
- **First decode:** it includes the eager per-op kernel compile into `p3_scratch`.
- **Peak HBM:** the reported peak (11.84 GiB for Wan, 9.30 for Qwen-Image) is a process-wide maximum over both dtypes, so it does not isolate bf16. The cells below do.

**Step 3, flags chosen** (every gate passed):
- Qwen-Image: `--vae-dtype bf16`.
- Wan2.1: `--vae-dtype bf16 --text-encoder-dtype bf16`.

### Re-measured cells (runbook steps 4 and 5)

The commands are the runbook's (`job_phase3_cells.sh`, `d3b1cb53…`; harness `bench4m.py` `6c52b12e…`, unchanged). Every run is at `eab8fc8` and clean, and `difflet` was imported from the worktree in every process (precheck and probe).
- The new stage manifests carry the new fields: `vae_dtype: bfloat16` on the Wan and Qwen-Image vae stages, `text_encoder_dtype: bfloat16` on the Wan transformer stage. All other key fields match the fp32 roots. The Qwen-Image DiT key is the same `f8267c1d…`.
- **Statistics:** `stats_gen1-3.txt` from `bench4m_stats.py`. The breakdowns come from `B/scripts/p3_breakdown.py` (`d39ea789…`, which reads `summary.json`).

| model | metric | fp32 native (summary table) | **p3 native (bf16 flags)** | p3 vs fp32 | AoT | p3 vs AoT |
|---|---|---|---|---:|---|---:|
| Qwen-Image | DiT step median / p10 / p90 | 359.0 / 358.2 / 360.1 ms | **360.4** / 359.6 / 361.9 ms | +0.4% | 360.6 ms | −0.1% |
| | warm e2e, mean of gen1-3 | 82.3 s (81.42, 82.13, 83.41) | **78.4 s** (77.06, 75.03, 83.08; gen0 86.06) | −4.8% | 73.7 s | +6.4% |
| | cold e2e A | 502.0 s (2026-10-25, old host) | **519.2 s** | +3.4% | 515.6 s | +0.7% |
| | compile | 187.0 s (generate launch 91.3, vae launch 93.8) | **194.2 s** (98.8 / 92.9) | +3.8% | 1209.9 s | −84.0% |
| | peak HBM per core (neuron-monitor) | 10.81 GiB DiT cores, 13.45-13.85 VAE core | **10.81 GiB on every core** (the VAE core no longer stands out; rank-0 VAE `max_memory_allocated` 9.30 → 5.22 GiB) | DiT cores 0%, VAE core at least −22% | 10.10 GiB | +7.0% |
| | PSNR / SSIM vs `ref_native_noise_50` | 16.11 dB / 0.823 | **16.11 dB / 0.823** (16.106 vs 16.105) | 0 | 21.48 / 0.912 (own noise) | |
| | output | PNG `b1906d53…`, latents `4da4c89e…` | PNG `dc302d76…`, latents **`4da4c89e…`** | latents identical | | |
| Wan2.1 | DiT step median / p10 / p90 | 1063.6 / 1061.5 / 1065.9 ms | **1064.9** / 1063.7 / 1069.0 ms | +0.1% | 956.4 ms | +11.3% |
| | warm e2e, mean of gen1-3 | 128.5 s (125.98, 130.82, 128.83) | **127.9 s** (131.14, 122.47, 130.19; gen0 122.63) | −0.5% | 123.16 s (A2, `--host-vae`) | +3.9% |
| | cold e2e A | 542.7 s (old host); **533.89 s on this host** | **454.7 s** | **−16.2%** (−14.8% on this host) | 449.4 s (A2) | **+1.2%** |
| | compile | 600.3 s (generate launch 428.8, vae launch 169.7) | **638.2 s** (457.3 / 178.4) | +6.3% | 5935.7 s (transformer stage alone 430.5 s) | −89.2% (+48.2% against the transformer stage) |
| | peak HBM per core (neuron-monitor) | 15.15 GiB DiT cores, 18.12-18.40 VAE core | **11.83 GiB** on every core (11.95 once, in gen2) | **−21.9%** DiT cores, at least **−35%** VAE core | 10.34 GiB | +14.4% |
| | PSNR / SSIM vs `ref_50` (shared noise) | 22.66 dB / 0.885 | **22.41 dB / 0.880** | −0.25 dB / −0.0054 | 21.40 / 0.853 (A2) | +1.0 dB / +0.027 |
| | output | mp4 `9b914e81…`, latents `3cafa271…` | mp4 `2da4edd4…`, latents `fbd0306e…`, the same in cold, gen0-3 and q1 | latents change (umT5) | | |

**Qwen-Image: what changed.** The output changes **only through the VAE**.
- q1's noise (`initial.pt` `95a4306b…`), loop output (`final.pt` `94d160cb…`) and `work/latents.pt` (`4da4c89e…`) are bit-identical (`torch.equal`) to the fp32 hash check (`B/phase3/quality/qwen_noise_latents_check.json`).
- The PNG differs: it is 56.72 dB / SSIM 0.9988 from the fp32-VAE PNG, which is the step-2 gate value reproduced through the CLI.
- I opened `out.png`: a red fox trotting across snow in a snowy forest, on prompt.
- **Timing:** the bf16 decoder is faster, but the effect on e2e is below the run-to-run spread.
  - The warm decode drops from 2.52 to 1.68 s. The vae launch as a whole is unchanged within noise: 18.8-20.8 s against 18.7-19.5 s.
  - The −3.9 s in the warm mean comes mostly from rank start-up spread. Rank start to `app_load` is 11.3-21.0 s here and 21.5-24.4 s in the 2026-10-25 cell.
  - **Cold:**
    - The vae stage is 0.7 s shorter (15.26 vs 15.93 s), and the DiT and text-encoder loads are equal (checkpoint 310.96 vs 310.88 s, text encoder 99.71 vs 99.62 s).
    - The +17.2 s comes from rank start-up: 27.0 s from rank start to `app_load`, against 13.3 s in the 2026-10-25 cold. This is the host/HEAD start-up spread already seen on this host for LTX-2 (section "Cold e2e, definitions A and B"), not the VAE. *Not measured:* an fp32 Qwen-Image cold run on this host.

**Wan2.1 cold A breakdown** (rank-0 probe spans and the app phase seconds, seconds from driver start)

| part | fp32 A on this host (`native_tp4_A_1027`) | **p3** (`native_tp4_p3/cold`) |
|---|---:|---:|
| driver start to rank start | 15.06 | 15.66 |
| rank imports and set-up, to the load | 18.28 | 19.05 |
| load (`app_load`) | 411.28 | **325.18** |
| — DiT weights from the store | 217.41 (every rank `hit 7.50 GB`) | 217.07 (every rank `hit 7.50 GB`, 215.9-217.0 s) |
| — umT5 | 172.56 (fp32 checkpoint) | **85.80** (bf16 store, every rank `hit 2.84 GB`, 85.1-85.8 s) |
| — warm-up forwards, DiT / umT5 | 13.95 / 4.86 | 14.27 / 5.39 |
| load end to encode | 0.06 | 6.43 (one in-run neuronx-cc compile, 6.07 s) |
| encode (cond + uncond) | 1.95 + 0.09 | 2.35 + 0.05 |
| denoise, 50 steps (step median) | 53.39 (1064.7 ms) | 53.37 (1065.3 ms) |
| generate ranks exit | 500.03 | 422.06 |
| VAE launch: load / decode / save (stage) | 11.01 / 7.93 / 0.50 (22.72) | 9.62 (bf16 store `hit 0.15 GB`) / 6.26 / 0.54 (20.09) |
| **harness wall** | **533.89** | **454.68** |

- **Where the −79.2 s comes from:**
  - umT5 read: −86.8 s.
  - VAE launch: −2.6 s.
  - The in-run compile: +6.4 s.
- **The in-run compile:** the 2026-10-25 fp32 cell's cold run had the same one (6.43 s). The 2026-10-27 fp32 A run skipped it only because the phase-2 hash check had written that NEFF into the root copy. So against the original fp32 cell (542.74 s), the comparison is like for like.
- **The plan's projection** (cold about 456 s, umT5 about 86 s) is met: 454.7 s and 85.8 s.
- **Store:**
  - The compile wrote the DiT entry (same key `73e38f6c…` as the fp32 root, 27 GB), the umT5 bf16 entry (11 GB, 2.84 GB per rank) and a 0.15 GB bf16 VAE entry (`replicated`). The VAE entry is not in the runbook; the fp32 VAE writes none ("no gain").
  - The root takes 38 GB, and `/` has 91 GB free.

**Wan2.1 warm.** Bf16 shortens the umT5 load from the page cache (2.31-2.38 → 1.25-1.52 s) and the VAE decode (7.25-7.29 → 5.62-5.85 s). The DiT warm-up forward is about 1.1 s longer (13.2-13.3 → 14.2-14.5 s). The net is within the start-up spread (rank start to `app_load` 13.7-22.4 s), so warm e2e is unchanged (−0.5%). The bf16 umT5 helps when the encoder is read from disk (cold), not from the page cache.

**Compile:**
- Wan2.1: +37.9 s.
  - Generate launch +28.5 s: the umT5 load, which now writes its bf16 store entry (88.55 vs 68.53 s), plus a +6.6 s longer warm-up.
  - Vae launch +8.7 s: the bf16 eager kernels (`vae_load` 170.39 vs 160.98 s).
- Qwen-Image: +7.2 s, all in the generate launch. The vae launch is level (92.9 vs 93.8 s).

**Peak HBM:**
- **Wan2.1:**
  - The DiT cores drop from 15.15 to 11.83 GiB per core, the umT5 shard halving. Rank 0's `max_memory_allocated` goes from 13.48 to 10.67 GiB: −2.81 GiB, the 2.84 GB umT5 difference.
  - The VAE core drops from 18.40 to at most 11.95 GiB. The VAE rank's `max_memory_allocated` goes from 11.84 to 6.24 GiB.
  - Native Wan2.1 is now 14% above AoT per core (10.34 GiB), instead of 46.5%.
- **Qwen-Image:** the VAE core's peak falls below the DiT cores' 10.81 GiB.

### Quality (runbook step 6, and the e2e half of F13)

The metrics ran on the CPU after the device phase (`B/scripts/quality_phase3.sh`, `85701049…`; output `B/phase3/quality/`).
- **Tools and definitions:** those of parts 1 and 2 (`quality_metrics.py` `8f6e1b18…`, selftest PASS; `frames_map.py` `14ef6365…`). The reference's `out_raw.pt` maps bit-identically to its `out.pt`.
- **Known pairs recomputed:** the fp32 native q1 rows reproduce parts 1 and 2: Wan 22.663 dB / 0.8854 / latent cosine 0.9668; Qwen-Image 16.105 dB / 0.8234 / 0.8722.

| pair (same noise) | PSNR mean (worst frame) | SSIM mean (worst frame); R / G / B | latent rel-L2 / cosine | file |
|---|---:|---:|---:|---|
| **Wan p3 q1 (bf16 umT5 + bf16 VAE) vs `ref_50`** | **22.41 dB** (20.33, frame 0) | **0.880** (0.840, frame 0); 0.875 / 0.887 / 0.879 | 0.258 / **0.966** | `wan_p3_vs_ref.json` |
| Wan fp32 native q1 vs `ref_50` (part 2) | 22.66 dB (20.32, frame 0) | 0.885 (0.840, frame 0); 0.879 / 0.892 / 0.885 | 0.256 / 0.967 | `known_wan_fp32q1_vs_ref.json` |
| Wan p3 latents decoded on the host in fp32 vs `ref_50` (umT5 alone) | 22.41 dB (20.33) | 0.880 (0.840) | same latents as p3 | `wan_p3hostfp32_vs_ref.json` |
| Wan p3 device bf16 VAE frames vs the host fp32 decode of the same latents (VAE alone) | 56.14 dB (55.82) | 0.9993 | | `wan_p3_devvae_vs_hostfp32.json` |
| Wan p3 q1 vs fp32 native q1, direct | 28.86 dB (25.70, frame 0) | 0.950 (0.926) | 0.148 / 0.989 | `wan_p3_vs_fp32q1.json` |
| **Qwen-Image p3 q1 (bf16 VAE) vs `ref_native_noise_50`** | **16.11 dB** | **0.823** | 0.522 / 0.872 (identical latents) | `qwen_p3_vs_ref.json` |
| Qwen-Image p3 q1 vs fp32 native q1, direct | 56.72 dB | 0.9988 | 0 / 1.000 | `qwen_p3_vs_fp32q1.json` |

- **Wan per frame, PSNR** (p3 vs ref): 20.33, 22.03, 22.28, 22.62, 22.71, 22.90, 22.91, 22.95, 23.00. The fp32 native q1 gave 20.32, 22.21, 22.65, 22.96, 23.05, 23.01, 23.15, 23.24, 23.38.
- **F13, 50-step e2e half: PASS.**
  - The rule (plan, recorded 2026-10-27) allows at most 0.5 dB PSNR and 0.01 SSIM below the fp32 native q1 against `ref_50`.
  - Wan p3 is 0.249 dB and 0.0054 SSIM below it, and the latent cosine moves from 0.9668 to 0.9664.
  - So `--text-encoder-dtype bf16` meets both F13 gates.
- **Which component costs what:**
  - All of the Wan drop comes from bf16 umT5. Its latents decoded on the host in fp32 score the same 22.41 dB / 0.880 as the device bf16 decode.
  - The bf16 VAE is 56 dB from an fp32 decode, which does not move the comparison with the reference (22.414 vs 22.412 dB).
  - bf16 umT5 changes the trajectory (direct p3 vs fp32: 28.9 dB, latent cosine 0.989) about as far from the fp32 native run as from the reference.
- **Noise:** Wan p3 q1's `initial.pt` is `03b6284b…`, the shared fp32 noise of both earlier Wan runs.
- **Frames:** I opened `strip_wan_{native_p3_q1,ref50}_02468.png`.
  - They show the same red fox trotting right through a snowy forest, in the same pose and place as the reference in frames 2-8.
  - Frame 0 has the reference's own coloured speckle.

### Decisions on this host (problem, choice, reason)

1. **The runbook's `$PY/torchrun` ran the wrong interpreter.**
   - **Problem:** the first gate attempt (22:35 UTC, `B/phase3/failed-attempt1-path/`) failed in seconds.
     - The console scripts of `/home/ubuntu/venvs/native-312` have the shebang `/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/python3`, the venv it was copied from. That venv has no diffusers: `ModuleNotFoundError` in both VAE checks.
     - A systemd unit's PATH has no venv `bin`, so neuronx-cc was not found: `posix_spawn failed` in F13.
   - **Choice:** launch with `$PY/python -m torch.distributed.run` and `PATH=$PY:$PATH`, exactly as the CLI (`difflet/cli/runner.py`) and `bench4m.py` do. Move attempt 1's 48 KB `p3_scratch` aside, so no failed-compile entry could be reused. Keep the attempt-1 job as `job_phase3_gates.as-used-attempt1-2026-10-27.sh`.
   - **Reason:** this is a harness and environment problem, not a Difflet bug. The rest of every command is the runbook's.
2. **`neuron-ls not found in PATH; NUMA optimizations disabled`.**
   - **Problem:** the unit's PATH has no `/opt/aws/neuron/bin`.
   - **Choice:** keep it as it was.
   - **Reason:** the phase-2 and 2026-10-27 unit runs had the same warning (for example `native_tp4_A_1027`, the hash checks). So the p3 cells run under the conditions of the same-host fp32 A run they are compared with. The 2026-10-25 reference cells, started from a login shell, did not have it.
3. **No same-host fp32 Qwen-Image control.**
   - **Problem:** the fp32 Qwen-Image native cell is from 2026-10-25 on the old host.
   - **Choice:** compare against it, and attribute each difference with the per-stage breakdown.
   - **Reason:** the runbook does not call for a control, and the breakdown separates the VAE (−0.7 s cold, −0.85 s warm decode) from the start-up spread.
4. **Metrics only after the device phase,** as on 2026-10-27, so that no CPU load overlaps a timed run.
5. **One added quality row:** a host fp32 decode of the p3 Wan latents (`wan_host_vae_decode.py`, the phase-2 A2 tool), to split umT5's effect from the VAE's. It runs on the CPU in about 1 min.
6. **The Wan in-run NEFF compile is left in the cold figure.**
   - **Reason:** the original fp32 cell's cold run, also on a fresh root, had the same compile. This is a fresh-root cell measured as the runbook defines it.
7. **One run per definition.** No run was anomalous: the outputs are identical across each cell's runs, and every component time is explained above.

## Last step: region-compiled Wan/Qwen VAE (phase 7, 2026-10-28)

**A supplement, opt-in.** The plan's "Phase 7" ("last step"): the Wan VAE decoder, which Qwen-Image also uses, compiled region by region on the native backend, after HunyuanVideo's compiled decoder (F1a, `1c10dfc`). The switch is **`--vae-exec-mode compile`** (Wan2.1 and Qwen-Image on `--backend neuron`; default `eager`, unchanged). It adds `vae_exec_mode: compile` to the vae stage key, so default stages keep their keys and manifests. **Not the default:** see the recommendation at the end of this section.

**Code** (branch `feat/hunyuan-neuron`):
- `796e81b` `WanVAERegionDecoder` (`difflet/models/wan/vae/regions.py`, backend-neutral).
- `338d73e` `NeuronWanVaeDecoderApplication(exec_mode="compile")`.
- `2cefb43` the CLI switch (Wan and Qwen-Image vae stage; stage key; forwarding to stage processes and DP workers; refused with `--host-vae`).

**Design** (how it differs from HunyuanVideo's):
- **The walk.** The decode goes one latent frame at a time, as diffusers' untiled `AutoencoderKLWan.decode` does (the host reference). Each causal convolution's two-frame cache is an explicit input and output of its region: the input is `cat(cache, x)`, and the next cache is the last two frames of that input. This reproduces diffusers' `feat_cache` rule in every case, so every latent frame after the first runs the same graphs.
- **The first latent frame** has its own graphs.
  - Its causal 3x3x3 convolutions see only zero padding before the frame, so each region convolves with the last temporal tap alone. That is the same sum without the zero terms, on a third of the input.
  - The temporal upsamplers skip their time convolution (diffusers' `"Rep"`) and seed a zero cache.
  - A decode of a single latent frame (Qwen-Image) returns no caches.
- **Graph count.** A decode of any length needs two sets of graphs: 46 NEFFs at Wan's 480x832, 24 at Qwen-Image's 1024x1024.
- **No spatial tiling** (decision 1 below). The frame is decoded whole, as the reference decodes it, so no tile blending is involved.
- **Regions follow HunyuanVideo's lessons:**
  - a resnet is four regions: norm + SiLU, convolution, norm + SiLU, convolution + shortcut + residual;
  - an upsampler's nearest repeat is one region and its `Conv2d` another;
  - the zero padding is done inside the convolution op, not with `F.pad`;
  - no `torch.where`;
  - caches are made contiguous, because a strided view triggers a recompile.
- **Precision** is phase 3's bf16 rules: the RMS norms normalise in fp32, the attention scores are fp32, and the output head and clamp run in fp32.
- **Host and device.** The latents and the decoded chunks stay on the host between region calls. The warm-up decodes at most two latent frames of zeros, which builds every graph.

**Tests (CPU):**
- New:
  - `tests/unit/models/wan/test_wan_vae_regions.py`: 12 tests. In fp32 the region decode equals diffusers' `AutoencoderKLWan.decode` and the eager model for 1, 2, 3 and 5 latent frames (atol 2e-5), including three resnets per block on a wide latent. In bf16 it is ≥ 50 dB from the eager bf16 decode. With a counting compile backend, the first two latent frames build every graph and later frames and decodes build none; a one-frame decode builds only cache-free graphs.
  - `tests/unit/backends/test_neuron_wan_vae_regions_app.py`: 8 tests. The application's compile mode against eager and diffusers' host decode; the warm-up frame count; no graphs after the warm-up.
  - New CLI tests in `tests/unit/cli/test_orchestrator_{wan,qwen_neuron}.py`. Among them, Qwen-Image's compiled vae stage writes the eager stage's image.
- **Totals:** `tests/unit/cli/` 836 passed.
- **The related model and backend suites:** 315 passed and 12 failed. The same 12 fail at `b421f82` (before this work): `neuronx_distributed` is missing from the native venv, and two `configure_compile_cache` HLO-directory tests.

### Device gate: the decoder alone (one core, the phase-3 q1 latents)

**Tool.** `B/scripts/vae_region_check.py`, run by `job_phase7_vae.sh` as `systemd-run --user` units, one at a time.
- Each model gets a fresh root `B/roots/p7_<tag>`.
- **`--phase compile`** loads with the warm-up, as the CLI compile stage does. On the fresh root this is the cold compile.
- **`--phase generate`** is a new process on the same root without the warm-up, as the CLI generate stage runs it. Its first decode loads the graphs.
- **Latents:** `B/wan2_1/native_tp4_p3/q1/final.pt` and `B/qwen_image/native_tp4_p3/q1/work/latents.pt`.
- **Reference:** diffusers' fp32 host decode of the same latents (`host_fp32.pt`).
- **Metrics:** RGB uint8 per frame, `quality_metrics.py`.
- **Evidence:** `B/phase7/{wan_bf16,qwen_bf16,wan_fp32,qwen_fp32}/`, at `2cefb43` with a clean worktree.

| model, dtype | vs fp32 host decode: PSNR mean (worst) / SSIM (gate ≥ 50 dB) | compile: warm-up decode on a fresh root | NEFFs | decode, steady (repeat in process) | decode, first in a warm process | peak HBM (`max_memory_allocated`) |
|---|---|---:|---:|---:|---:|---:|
| **Wan2.1 480x832x9, bf16** | **55.91 dB** (55.35, frame 0) / 0.99934 — **PASS** | 132.3 s | 46 | **1.02 s** | 9.58 s | 6.21 GiB |
| Wan2.1, fp32 | 97.47 dB (96.61) / 1.00000, max abs < 5e-5 | 203.4 s (*) | 46 | 2.45 s | 9.63 s | 12.38 GiB |
| **Qwen-Image 1024x1024, bf16** | **55.87 dB** / 0.99856 — **PASS** (56.05 dB / 0.99856 from the phase-3 eager bf16 frames) | 39.9 s | 24 | **0.15 s** | 3.67 s | 2.40 GiB |
| Qwen-Image, fp32 | 99.31 dB / 1.00000 | 51.6 s | 24 | 0.31 s | 3.64 s | 4.79 GiB |
| *eager bf16 (phase 3), for comparison:* Wan2.1 | 56.18 dB (other latents: the fp32-umT5 hash check) | vae launch 178.4 s (`vae_load` 170.4) | ~hundreds of per-op | 1.85 s | 5.62-5.85 s (cell warm), 6.26 s (cold A) | 6.24 GiB |
| *eager bf16 (phase 3):* Qwen-Image | 56.72 dB | vae launch 92.9 s | per-op | 0.41 s | 1.68 s (cell warm) | 5.22 GiB |

(*) The Wan fp32 compile overlapped a CPU unit-test run, so it is not a clean figure. The bf16 rows ran on an idle host.

**What the gate shows:**
- **Accuracy.** The fp32 compiled decode matches the host to 97.5 / 99.3 dB, the same as the eager fp32 device decode (97.44 / 99.49). So the regions are exact on the device, and bf16 lands where phase 3's eager bf16 landed. Qwen-Image's bf16 is 0.85 dB below eager bf16 against the host; both are 5.9-6.7 dB above the gate.
- **Steady-state decode:** 1.8x faster for Wan (1.02 vs 1.85 s) and 2.7x faster for Qwen-Image (0.15 vs 0.41 s). Peak HBM falls for Qwen-Image (2.40 vs 5.22 GiB).
- **The first decode in a fresh process is slower than eager's:** Wan 9.58 vs 5.6-5.9 s, Qwen-Image 3.67 vs 1.68 s. Every process re-traces each region with Dynamo and loads each NEFF.
  - The diagnostic (`B/phase7/diag/firstcall_wan.json`, `vae_region_firstcall_diag.py`) puts 5.6 s of Wan's first decode in Dynamo: `compile_inner` with bytecode tracing 1.1 s, AOT dispatch 2.2 s, torch-neuronx lowering 1.1 s and guards 0.9 s. The first region call in a process adds 1.3 s of one-time set-up. The rest is NEFF loads and the decode itself.
  - The CLI starts a fresh vae process for every generate, so it pays this cost on every request.

**Experiment: fewer, larger regions** (`B/phase7/wan_bf16_block/`, an uncommitted `split_resnets=False` option, reverted after the run). Each resnet and the output head as one region gave:
- 27 NEFFs instead of 46;
- compile 213.9 s instead of 132.3 s;
- steady decode 1.10 s instead of 1.02 s;
- first decode in a warm process 9.05 s instead of 9.58 s;
- the same 55.91 dB.

The first-decode cost scales with the ops traced, not the number of graphs, so the fine regions stay.

**Failed attempt, kept:** `B/phase7/qwen_bf16_attempt1_edited_during_run/`, with its root `roots/p7_qwen_bf16_attempt1`.
- **What happened:** I edited `regions.py` (the experiment above) while that run's generate phase was starting. The edit moved line numbers, and the warm process recompiled all 24 NEFFs: first decode 36.5 s.
- **What it shows:** the native NEFF cache key changes when the source changes, even when the default path is unchanged.
- **What was re-run:** the run was repeated on a fresh root with the committed file (the table above). The attempt's compile phase (37.6 s, 0.15 s, 55.87 dB) agrees with the re-run.

### Re-measured cells: `--vae-exec-mode compile` on top of the phase-3 flags

**Setup.**
- **Harness and definitions:** those of phase 3 (`bench4m.py`, definition A cold, gen0 discarded, statistics from gen1-3 with `bench4m_stats.py`, breakdowns from `p3_breakdown.py`).
- **Cells and roots:** new cells `native_tp4_p7` on fresh roots `B/roots/{qwen_image,wan21}_native_tp4_p7`.
- **Jobs:** `job_phase7_cells.sh`, then `job_phase7_wan_rest.sh`.
- **Code and runs:** every run is at `2cefb43`, clean, with `difflet` imported from the worktree, rc 0 and the device released.
- **Flags:**
  - Qwen-Image: `--vae-dtype bf16 --vae-exec-mode compile`.
  - Wan2.1: `--vae-dtype bf16 --text-encoder-dtype bf16 --vae-exec-mode compile`.
- **Stage keys:** the vae stage key gains `vae_exec_mode: compile`. The DiT keys are the phase-3 roots' keys.

| model | metric | p3 native (eager bf16 VAE) | **p7 native (region-compiled bf16 VAE)** | p7 vs p3 | AoT | p7 vs AoT |
|---|---|---|---|---:|---|---:|
| Qwen-Image | VAE compile (the vae launch in `compile`) | 93.8 s (`vae_load` 85.0) | **62.4 s** (`vae_load` 53.7) | **−33.5%** | (AoT compile 1209.9 s in all) | |
| | compile, total | 194.2 s (99.6 / 93.8) | **157.2 s** (94.0 / 62.4) | **−19.1%** | 1209.9 s | −87.0% |
| | decode, warm (first in the vae process) | 1.67-1.69 s | 3.71-3.90 s | +2.1 s | | |
| | vae launch, warm (start to exit) | 19.6-21.6 s (mean 20.67) | 20.7-22.3 s (mean 21.74) | +1.1 s | | |
| | DiT step median / p10 / p90 | 360.4 / 359.6 / 361.9 ms | 359.8 / 359.1 / 361.7 ms | −0.2% | 360.6 ms | −0.2% |
| | **warm e2e**, mean of gen1-3 | 78.4 s (77.06, 75.03, 83.08) | **81.2 s** (84.53, 76.56, 82.40; gen0 84.0) | **+3.5%** | 73.7 s | +10.1% |
| | **cold e2e A** | 519.2 s (vae stage 20.5, decode 1.92) | **517.9 s** (vae stage 27.4, decode 7.85) | −0.3% | 515.6 s | +0.4% |
| | peak HBM per core (neuron-monitor) | 10.81 GiB on every core | **10.81 GiB on every core**; VAE rank `max_memory_allocated` 5.22 → **2.40 GiB** | 0 | 10.10 GiB | +7.0% |
| | output | PNG `dc302d76…`, latents `4da4c89e…` | PNG `1a4d131d…`, latents **`4da4c89e…`** (the same in cold, gen0-3 and q1) | latents identical | | |
| Wan2.1 | VAE compile (the vae launch in `compile`) | 179.3 s (`vae_load` 170.4) | **139.8 s** (`vae_load` 131.4) | **−22.0%** | VAE 5468 s | −97.4% |
| | compile, total | 638.2 s (458.1 / 179.3) | **593.1 s** (452.6 / 139.8) | **−7.1%** | 5935.7 s | −90.0% |
| | decode, warm (first in the vae process) | 5.62-5.85 s | 9.08-9.38 s | +3.5 s | | |
| | vae launch, warm | 24.6-25.6 s (mean 25.00) | 27.4-28.9 s (mean 28.03) | +3.0 s | | |
| | DiT step median / p10 / p90 | 1064.9 / 1063.7 / 1069.0 ms | 1064.6 / 1063.4 / 1067.5 ms | 0.0% | 956.4 ms | +11.3% |
| | **warm e2e**, mean of gen1-3 | 127.9 s (131.14, 122.47, 130.19) | **129.1 s** (131.14, 131.87, 124.40; gen0 130.3) | **+1.0%** | 123.16 s (A2) | +4.9% |
| | **cold e2e A** | 454.7 s (vae stage 25.7, decode 6.26) | **464.4 s** (vae stage 28.4, decode 9.11) | +2.1% | 449.4 s (A2) | +3.3% |
| | peak HBM per core (neuron-monitor) | 11.83 GiB on every core | **11.83 GiB on every core**; VAE rank 6.24 → 6.21 GiB | 0 | 10.34 GiB | +14.4% |
| | output | mp4 `2da4edd4…`, latents `fbd0306e…` | mp4 `62823dab…` (the same in cold and gen0-3) | latents: see quality | | |

**Where the differences come from** (per-stage breakdowns):
- **Compile.** The region-compiled decoder compiles 31 s faster for Qwen-Image and 40 s faster for Wan: 46 and 24 small graphs, against the eager decoder's per-op kernels. The generate launches are level within 6 s.
- **Warm e2e.** The vae launch is 1.1 s longer for Qwen-Image and 3.0 s longer for Wan. All of it is the first decode in the fresh vae process: Dynamo re-traces each region and each NEFF is loaded (see the gate). The rest of the warm difference is the generate launch's rank start-up spread (rank start to `app_load`: Qwen-Image 18.6 → 21.4 s mean).
- **Cold.**
  - The vae stage is 7.0 s longer for Qwen-Image (decode 7.85 s, with the NEFFs read from a dropped page cache) and 2.7 s longer for Wan.
  - The generate launches move by −8.2 s (Qwen-Image) and +7.1 s (Wan), both from rank start-up (rank start to `app_load`: 42.6 → 36.2 s and 34.7 → 43.3 s).
  - This is the start-up spread phase 3 already saw on this host, not the VAE. With one run per definition it does not average out.
- **DiT and HBM** are unchanged. The VAE rank's own peak halves for Qwen-Image. The neuron-monitor peaks per core are the DiT cores' and do not move.

**Quality** (`B/scripts/quality_phase7.sh`, run on the CPU after the device work; output in `B/phase7/quality/`; `quality_metrics.py` selftest PASS):
- **Latents.** q1's noise, loop output and `work/latents.pt` are bit-identical to the p3 q1 runs for both models (Qwen-Image `4da4c89e…`, Wan `fbd0306e…`). The VAE is the only change.
- **Qwen-Image:**
  - p7 q1 vs `ref_native_noise_50`: **16.09 dB / 0.823** (p3: 16.11 / 0.823).
  - p7 vs p3 q1 directly: 56.05 dB / 0.9986.
- **Wan2.1:**
  - p7 q1 vs `ref_50`: **22.41 dB (worst 20.33, frame 0) / 0.880** (p3: 22.41 / 0.880).
  - p7 vs p3 q1 directly: 56.32 dB (worst 55.93) / 0.9994.
  - The p7 device decode against the host fp32 decode of the same latents: 55.91 dB / 0.9993 (p3's eager bf16: 56.14 dB).
- **Frames.** I opened `strip_wan_native_p7_q1_02468.png` and Qwen-Image's `out.png`. Both show the red fox in the snowy forest as before; Wan's frame 0 has the reference's own speckle.

### Supplement: decode time vs clip length (Wan2.1 480x832, bf16, one core)

**What ran.** `job_phase7_long.sh` with `vae_long_clip_timing.py`, random latents of 21 latent frames (81 video frames).
- Eager on a fresh root, `roots/p7_long_eager`: the first process fills its kernel cache, then two warm processes.
- The region decoder on `roots/p7_wan_bf16`, built at 9 frames: two processes. The clip length added no NEFF; the root still holds 46.
- Timing only (`B/phase7/long/`).

| 81 frames (21 latent) | first decode in a warm process | second decode | cold first decode (empty root) | peak HBM |
|---|---:|---:|---:|---:|
| eager bf16 | 21.1-21.5 s | 17.45-17.46 s | 172.9 s | 6.57 GiB |
| **region-compiled bf16** | **17.1-17.3 s** | **9.50-9.54 s** | (132.3 s warm-up at 9 frames, valid for every length) | 6.21 GiB |

**Per process and per frame.**
- **Set-up:** the region decoder's first decode in a process costs 7.7 s more than its second, against eager's 3.9 s.
- **Per latent frame:** about 0.47 s, against eager's 0.87 s (from the second decodes at 3 and 21 latent frames).
- **At Wan's benchmark length** (9 frames, 3 latent frames) the set-up dominates and the region decoder loses 3.5 s per request.
- **At 81 frames** it wins 4 s on the first decode of a process and 8 s on every decode after it.
- **Break-even,** by these numbers: near 12 latent frames (about 45 video frames).

### Decisions (problem, choice, reason)

1. **No spatial tiling.**
   - **Problem:** the plan's wording was "fixed-size tiles". The 50 dB gate's reference is diffusers' untiled decode, and diffusers' own tiled decode blends overlapping tiles, which makes it a different decode. An exact tiling would need halos as deep as the decoder's receptive field.
   - **Choice:** whole frames. The fixed shapes come from the frame-by-frame walk with explicit caches instead.
   - **Reason:** it is exact (fp32 matches the host to 97.5 / 99.3 dB), and the regions compiled at full frame within neuronx-cc's limits (46 / 24 graphs, 132 / 40 s).
2. **Opt-in, not the default.** HunyuanVideo's compiled decoder became its default only after the `74fc979` rule (accuracy plus TP2, TP4 and TP2xCP2 runs) and because it won 58 s per request. Here it loses 1-3.5 s per request at the benchmark shapes. The defaults for Wan and Qwen-Image are not changed (task rule).
3. **Fine regions kept** over whole-resnet regions, after the experiment above: those compile faster and decode faster, and the larger regions save only 0.5 s of set-up.
4. **The harness disk guard.**
   - **Problem:** `bench4m.py` stopped the Wan cell after its compile (rc 0): it wanted 80 GB free before every run, and the compile's 38 GB weight store left 55.8 GB.
   - **Choice:** ask 80 GB for a compile and 40 GB for the other runs, then run cold, gen0-3 and q1 on the same root with the same flags (`job_phase7_wan_rest.sh`). The earlier version is kept as `bench4m.as-used-p3-and-p7-qwen-wan-compile-2026-10-28.py`.
   - **Reason:** the other runs write only outputs and logs. No existing root was deleted to make room.
   - **Afterwards:** `/` has 52 GiB free.
5. **The in-process decoders** of the composed applications (`enable_vae_decoder=True`, which the CLI does not use) stay eager. Only the CLI vae stage takes the switch.
6. **One run per definition**, as in phase 3. Every difference is attributed with the stage breakdowns above.

### Recommendation

**Keep `--vae-exec-mode compile` opt-in for Wan2.1 and Qwen-Image; do not make it the default at the benchmark shapes.**
- It passes every gate: 55.9 dB, the same e2e quality, exact in fp32, and the CPU tests against diffusers.
- It compiles the VAE 22-34% faster.
- But the CLI starts a fresh vae process for every request, and its first decode in a process re-traces every region with Dynamo and loads every NEFF. That decode takes 3.7 s for Qwen-Image and 9.1-9.4 s for Wan, against eager's 1.7 s and 5.7 s, which is more than the region decoder saves on 1 image or 3 latent frames. Warm e2e is +1.0% (Wan) and +3.5% (Qwen-Image); cold is within the start-up spread.

It is the better choice for long clips (81 frames: 4 s faster per process; break-even near 45 frames) and for a persistent decoder process: 1.8x / 2.7x faster steady-state decodes, and Qwen-Image's VAE rank at 2.40 instead of 5.22 GiB.

Making it a default anywhere, for example for clips of about 45 frames or more, or for an in-process or serving decoder where the set-up is paid once, needs the user's approval and the `74fc979` runs.

## CPU bf16 baseline (2026-10-28)

**Question:** how much of the device-vs-CPU-fp32 gap does bf16 alone cause? Each CPU fp32 reference is re-run on the CPU in bf16, from the same injected dumps, and compared with its fp32 counterpart. If a device output is about as close to fp32 as CPU bf16 is, the device adds no error of its own. The first pair, the LTX-2 AoT-noise 50-step run, is the open question: video latent cosine 0.725 at 50 steps against 0.991 at 20.

**What runs in bf16:** the `cpu_ref_*.py` scripts have a new `--dtype` / `--vae-dtype` option. It defaults to float32, and with the defaults the scripts run the fp32 code path.
- The DiT and the text encoders run in bf16: Qwen2.5-VL, Gemma-3, umT5, and Llama-3 + CLIP-L. So do LTX-2's text connectors. They load through diffusers' own `torch_dtype=bfloat16`, which keeps Wan's `_keep_in_fp32_modules` in fp32.
- The VAEs stay fp32, and so do LTX-2's audio VAE and vocoder.
- **Why the VAE stays fp32:** the question is the denoising trajectory. The latent metrics do not depend on the VAE, and with an fp32 decode the frame PSNR reflects only the trajectory. The bf16 VAE was gated on its own in phase 3.
- **Unchanged from fp32:** the dumps (checked against the fp32 `reference.json` before each run), prompt, steps, schedule, guidance, negative prompt and diffusers' pipeline call.
- **Loop latents:** diffusers sets the dtype of the latents in the loop, and each `reference.json` records it.
  - Qwen-Image keeps them in bf16 (the prompt-embeds dtype), and its timestep reaches the DiT in bf16.
  - LTX-2 and Wan2.1 keep them in fp32 and cast the DiT input to bf16. LTX-2 also rounds the final latents to bf16 before its fp32 VAE.
  - HunyuanVideo starts in fp32. FlowMatchEuler casts back to the DiT output dtype, so its latents are bf16 from step 2 on. Its guidance input is bf16(6.0) x 1000 = 6016, the value both device orchestrators feed; the fp32 reference fed 6000.

**Smoke tests** (2 DiT blocks, 2 steps, CPU only, the Neuron modules blocked, all rc 0):
- **Injection:** the loop starts from the dump bit for bit in all five smokes: Qwen native, LTX-2 AoT, Wan2.1, HunyuanVideo native and HunyuanVideo AoT. The first DiT input equals the dump cast to bf16: lossless for the bf16 dumps, and the bf16 rounding for the fp32 dumps of Wan2.1 and HunyuanVideo native.
- **fp32 path unchanged:** fp32 re-runs of the edited Qwen-Image and Wan2.1 scripts reproduce the 2026-10-26 smokes bit for bit (`final.pt`, `out.pt`, PNG / `out_raw.pt`).
- **Dry run of the metrics script** on the smoke pairs, as a tool check only (2 blocks, 2 steps): bf16 vs fp32 cosine 1.0000 / 1.0000 / 0.9992 / 0.9986 for Qwen-Image / LTX-2 / Wan2.1 / HunyuanVideo AoT.

**Queue:** the systemd user unit `cpuref_bf16_queue` (`B/scripts/job_cpu_ref_queue_bf16.sh`) started at 03:03 UTC, one run at a time. Order:
1. LTX-2 AoT noise, 50 steps
2. LTX-2 AoT noise, 20 steps (`xcheck_aot_noise_20_bf16`)
3. LTX-2 native noise, 50 steps
4. Qwen-Image AoT noise
5. Qwen-Image native noise
6. Wan2.1
7. HunyuanVideo native noise
8. HunyuanVideo AoT noise

- **First run:** the injection check passed after step 1. Step 1 took 273.6 s, most of it reading the DiT weights lazily from disk; step 2 took 21.5 s, against 80 s in fp32.
- **Expected finish:** about 6-7.5 h in all, so roughly 09:00-10:30 UTC. The plan's 24 h assumed bf16 would be slower than fp32 on this CPU; it is about 3.7x faster here.
- **Outputs:** `B/<model>/cpu_ref/<name>_bf16/`.
- **Metrics:** after each run, `B/scripts/metrics_bf16_vs_fp32.py` writes `B/cpu_bf16_baseline/metrics/`: the pair check, `quality_metrics.py` (CPU fp32 = ref, CPU bf16 = candidate) and `summary.md`, which sets each pair beside the device-vs-fp32 numbers for the same noise.
- **Finished:** all 8 runs rc 0, 03:03–09:42 UTC (6.7 h); every pair check passed (same dumps, schedule, steps, guidance, DiT calls; both injection checks).

### Results: CPU bf16 vs CPU fp32, speed and quality

Steady step is the median after the first step (the first step reads the bf16 weights lazily). The fp32 references ran on the first host (HunyuanVideo AoT on the replacement host, same instance type), 6 torch threads on both sides. Quality: CPU fp32 = reference, CPU bf16 = candidate, same `quality_metrics.py` definitions as parts 1 and 2; VAEs are fp32 on both sides. Script: `B/scripts/bf16_vs_fp32_perf_table.py`; files `B/cpu_bf16_baseline/perf_table.{md,json}`.

| pair | steps | steady step fp32 → bf16 | denoise | wall | peak RSS | bf16 vs fp32 quality |
|---|---:|---:|---:|---:|---:|---|
| LTX-2 AoT noise | 50 | 79.5 → 20.9 s (x3.80) | 66.3 → 21.9 min | 1.31 → 0.42 h (x3.12) | 116.91 → 48.11 GiB | video 0.9912 / audio 0.9987; 32.76 dB / 0.978 |
| LTX-2 AoT noise, 20 steps | 20 | 80.6 → 21.5 s (x3.76) | 27.0 → 7.3 min | 0.67 → 0.20 h (x3.40) | 108.84 → 48.13 GiB | video 0.9962 / audio 0.9995; 37.24 dB / 0.990 |
| LTX-2 native noise | 50 | 80.1 → 22.9 s (x3.51) | 67.1 → 19.4 min | 1.28 → 0.34 h (x3.81) | 115.96 → 47.39 GiB | video 0.9908 / audio 0.9999; 30.82 dB / 0.961 |
| Qwen-Image AoT noise | 50 | 80.5 → 24.7 s (x3.26) | 67.1 → 25.4 min | 1.21 → 0.45 h (x2.69) | 85.44 → 47.09 GiB | latent 0.9735; 22.93 dB / 0.926 |
| Qwen-Image native noise | 50 | 81.5 → 24.5 s (x3.32) | 67.9 → 20.6 min | 1.18 → 0.36 h (x3.28) | 84.93 → 46.61 GiB | latent 0.9960; 32.24 dB / 0.985 |
| Wan2.1 (shared noise) | 50 | 316.2 → 123.5 s (x2.56) | 277.5 → 103.6 min | 4.68 → 1.91 h (x2.45) | 62.6 → 80.31 GiB | latent 0.9854; 27.60 dB / 0.943 |
| HunyuanVideo native noise | 50 | 278.9 → 90.4 s (x3.09) | 234.7 → 77.2 min | 4.11 → 1.44 h (x2.85) | 55.95 → 31.94 GiB | latent 0.9740; 19.77 dB / 0.853 |
| HunyuanVideo AoT noise | 50 | 286.2 → 92.0 s (x3.11) | 243.2 → 77.1 min | 4.27 → 1.41 h (x3.04) | 55.66 → 32.12 GiB | latent 0.9699; 19.77 dB / 0.778 |

**bf16 alone vs the device** (same noise, both against CPU fp32; `B/cpu_bf16_baseline/metrics/summary.md`):

| pair | steps | CPU bf16 vs CPU fp32 | device vs CPU fp32 (same noise) | pair check | bf16 step s (fp32) |
|---|---|---|---|---|---|
| ltx2_aot_50 | 50 | final_video cos 0.9912, final_audio cos 0.9987; 32.76 dB / 0.978 | device: final_video cos 0.7246, final_audio cos 0.9800; 16.21 dB / 0.660 | True | 20.9 (79.5) |
| ltx2_aot_20 | 20 | final_video cos 0.9962, final_audio cos 0.9995; 37.24 dB / 0.990 | device: final_video cos 0.9907, final_audio cos 0.9994; 35.45 dB / 0.979 | True | 21.5 (80.6) |
| ltx2_native_50 | 50 | final_video cos 0.9908, final_audio cos 0.9999; 30.82 dB / 0.961 | device: final_video cos 0.9304, final_audio cos 0.9894; 18.06 dB / 0.811 | True | 22.9 (80.1) |
| qwen_aot_50 | 50 | final cos 0.9735; 22.93 dB / 0.926 | device: final cos 0.9614; 21.48 dB / 0.912 | True | 24.7 (80.5) |
| qwen_native_50 | 50 | final cos 0.9960; 32.24 dB / 0.985 | device: final cos 0.8722; 16.11 dB / 0.823 | True | 24.5 (81.5) |
| wan_50 | 50 | final cos 0.9854; 27.60 dB / 0.943 | native: final cos 0.9668; 22.66 dB / 0.885; aot_hostvae: final cos 0.9545; 21.40 dB / 0.853 | True | 123.5 (316.2) |
| hv_native_50 | 50 | final cos 0.9740; 19.77 dB / 0.853 | device: final cos 0.9818; 20.78 dB / 0.869 | True | 90.4 (278.9) |
| hv_aot_50 | 50 | final cos 0.9699; 19.77 dB / 0.778 | device: final cos 0.9685; 19.65 dB / 0.774 | True | 92.0 (286.2) |

**Reading the numbers**
- **Speed:** on this CPU bf16 is 2.6–3.8x faster per step than fp32 and 2.5–3.8x faster end to end (the VAEs stay fp32; LTX-2 fp32 spent about 360 s reading Gemma-3 in fp32 against 7 s in bf16). Peak RSS drops to 30–57% of fp32 for LTX-2, Qwen-Image and HunyuanVideo. Wan2.1 rises (62.6 → 80.3 GiB): its fp32 checkpoint was memory-mapped (8.8 GiB anonymous), while the bf16 load converts it into anonymous memory.
- **The bf16 floor:** bf16 alone already moves the result: latent cosine 0.970–0.996, 19.8–37.2 dB. HunyuanVideo (19.8 dB, cosine 0.97) and Qwen-Image's AoT noise (22.9 dB) are the most sensitive; LTX-2 and Qwen-Image's native noise the least.
- **Devices at the bf16 floor:** HunyuanVideo native (0.982 vs floor 0.974) and AoT (0.969 vs 0.970), Qwen-Image AoT (0.961 vs 0.974, 21.5 vs 22.9 dB) and LTX-2 AoT at 20 steps (0.991 vs 0.996). Their gap to fp32 is about what bf16 alone gives.
- **Devices below the floor, with an extra error source:**
  - **LTX-2 AoT at 50 steps:** 0.725 vs 0.991. Not bf16 itself and not two-core attention (20 steps are fine). An AoT issue; next step a per-step latent comparison against CPU bf16.
  - **Qwen-Image native:** 0.872 / 16.1 dB vs 0.996 / 32.2 dB. Root cause found and fixed: the DiT attended the prompt's zero padding (not the text encode). See the next section.
  - **LTX-2 native:** 0.930 vs 0.991.
  - **Wan2.1:** native 0.967 / 22.7 dB and AoT (host VAE) 0.954 / 21.4 dB vs 0.985 / 27.6 dB: a moderate extra error on both backends.

## Qwen-Image native accuracy: root cause and fix (2026-10-28)

**Root cause:** every Difflet Qwen-Image backend zero-pads the prompt embeddings to `TEXT_SEQ_LEN` = 1024 rows and drops the mask, so the DiT's joint attention attends the padding. A zero row is not an absent key: `txt_norm` / `txt_in` turn it into `txt_in`'s bias, which then flows through every block as a real key and value. The fox prompt has 17 valid rows, so 1007 of the 1024 text keys were padding. diffusers (the CPU references) passes the 17 rows alone. The text encode is not the cause.

**Evidence** (CPU only unless noted; scratch and scripts in `B/nf_qwen_text/`, `B` = `/home/ubuntu/bench-2026-10-26-4m`):
1. **Text embeddings are fine** (`text_cmp.py`). Against diffusers' fp32 `encode_prompt` (17 x 3584), cosine 0.99988 for both the native host encoder (bf16, run through difflet's `_HostQwenTextEncoder._encode`) and diffusers in bf16, which match each other exactly. The AoT device's `text.pt` gives 0.99991. All three have 17 valid rows and the same 34-token template drop. The CPU bf16 baseline used these same bf16 embeddings and reached 0.996, so the text encode cannot explain 0.872.
2. **One DiT forward** (`dit_pad_cmp.py`, the real 20B DiT, native noise, the fp32 embeddings). Rel. L2 of the noise prediction against fp32 on the 17 rows:

   | input | t = 1.0 | t = 0.5 |
   |---|---:|---:|
   | fp32, padded to 1024, unmasked | 6.5 % | 2.1 % |
   | fp32, padded to 1024, masked | 3e-7 | 3e-7 |
   | bf16, 17 rows | 2.7 % | 1.2 % |
   | bf16, padded to 1024, unmasked | 6.8 % | 2.4 % |

   The padding alone moves the prediction more than bf16 does. With the mask, the padded call equals the unpadded one.
3. **50 steps on the CPU reproduce the device.** The CPU bf16 baseline with one change, the embeddings padded to 1024 and passed without a mask (`cpu_pad_run.py --mode pad_unmasked`), reaches **0.865 / 15.53 dB / 0.815** against the fp32 reference. The native device reached 0.872 / 16.11 dB / 0.823. The old native device latents are much closer to this padded CPU run (cosine 0.967, 21.6 dB) than to the unpadded CPU bf16 run (0.873).
4. **Ruled out by the steps above:** VAE (the latents already differ), text encode and template drop (1), timestep dtype, schedule and guidance (identical in the padded CPU run, which reproduces the gap).

**Fix (neuron backend only):** the DiT now receives the mask and attends only the valid text rows, as diffusers does. Shapes stay static.
- `_OnDeviceTransformer` sends the mask (int64, checked to be a prefix of valid rows) as the DiT's fourth input; guidance is still dropped.
- The trace module turns the mask into a valid-row count.
- `_QwenImageTrainiumAttnProcessor._masked_joint_attention` orders the keys `[image ‖ text]`, so the valid keys are the contiguous range `[0, image tokens + count)`. That range goes to the key-bounded flash kernel. Ring and Ulysses CP mask by the same count.
- The warm-up compiles with an all-ones mask. The count is a tensor, so a different prompt length does not compile a new graph.
- Trainium and TPU are unchanged: they pass no count and still attend the padding.

**Tests:**
- New: `test_padded_prompt_attends_only_its_valid_rows`, 8 topologies (tp1 eager and compile, tp2 compile, tp2+SP, CP2 gather_kv / ring / ulysses, tp2xCP2 compile). The run that wrote the fix saw all 8 fail before it (every element mismatched; that log was not kept); after it all 8 pass at 1e-5 against diffusers on the valid rows.
- Follow-up (2026-10-28 review): under ring CP, a mask with no valid row at all (all-False, reachable only through the tensor API: encoded prompts always keep the template's closing tokens) made the whole output NaN, because the text partial's softmax over no key gave `exp(-inf - -inf)`. `_masked_attention_with_lse` now returns the empty partial (output 0, logsumexp -inf, weight 0 in the merge), so ring attends the image keys alone, as gather_kv, Ulysses and diffusers do; unchanged (bit for bit) when a key is valid. `tests/unit/backends/test_neuron_joint_attention_cp.py` covers counts 0, 1 and more at cp 1, 2 and 4, eager and compiled.
- Updated to diffusers' masked semantics: the end-to-end and true-CFG / CFG-parallel references of q2 and q5, and the adapter and warm-up shape tests.

**Device** (TP4, 1024x1024, 50 steps, seed 42, `--vae-dtype bf16`; same `initial.pt` `95a4306b…` as `native_tp4_hashcheck/q1`; harness `scripts/bench4m_nfq.py`, which is `bench4m.py` with the native worktree taken from `BENCH4M_NATIVE_WT`):
- "before" is HEAD `549e6f7` from a snapshot of its `difflet/`.
- "after" is the fix.
- Each cell has its own hard-link copy of `roots/qwen_image_native_tp4_p3`.

| vs CPU fp32 `ref_native_noise_50` | latent cosine | PSNR | SSIM |
|---|---:|---:|---:|
| native before (`nfq_before/q1`; `final.pt` bit-identical to `native_tp4_hashcheck/q1` and `native_tp4_p3/q1`) | 0.8722 | 16.11 dB | 0.823 |
| **native after (`nfq_after/q1`)** | **0.9957** | **32.88 dB** | **0.985** |
| CPU bf16 floor (`ref_native_noise_50_bf16`) | 0.9960 | 32.24 dB | 0.985 |

- The fixed native run is at the bf16 floor. Against CPU bf16 it reaches cosine 0.9959 / 32.41 dB.
- Its image shows the reference's composition: the fox leaping right, including the reference's smeared muzzle.

**Speed:** both cells ran back to back on a quiet host, interleaved before / after, gen2 and gen3.

| | DiT step median (p10–p90) | warm e2e |
|---|---:|---:|
| before | 359.9 ms (359.2–361.1) | 86.5 / 88.2 s |
| after | 368.5 ms (367.5–369.6) | 86.7 / 86.0 s |

- The step is 8.6 ms slower (+2.4 %). *Inference, not profiled:* the key-bounded kernel path, with its separate q pre-scale, costs more than the plain flash kernel.
- Warm e2e does not change materially.
- The first run after the change compiles the new block graph once: load+warmup 67 s, against 37–49 s warm (`nfq_after/cold`, 127 s e2e).
- Scratch runs before the host was quiet are kept but not used for speed (`nfq_before/gen1`, `nfq_after/gen1`: 377 / 397 ms).

**Left open:**
- **Trainium (AoT) and TPU still attend the padding.**
  - On the AoT noise, the CPU bf16 run with the padding unmasked gives 0.938 / 18.6 dB, against 0.974 / 22.9 dB without.
  - The AoT device itself sits at 0.961. That is between the two, and closer to the unpadded CPU bf16 run (0.951) than to the padded one (0.923).
  - This seed's AoT-noise trajectory is too sensitive (bf16 alone 0.974) to attribute the AoT device's error on one run.
  - The AoT fix needs `attention_cte`'s bounds in the traced graph; it is not in this change.
- **Masked attention under CP and SP** is unit-tested on CPU only. On the device only TP4 was run.

## Wan2.1 native vs AoT DiT step: cause

Plan item F6a (2026-10-28). Native at `feat/native-followups` `fa3bc3e` with the phase-3 flags (`--vae-dtype bf16 --text-encoder-dtype bf16`); AoT at main `540e5f0`, two-core, `--host-vae`. Raw evidence in `B/nf_wan_step/` (`B` = `/home/ubuntu/bench-2026-10-26-4m`): scripts in `scripts/`, timings in `prof/`, device profiles in `ntff/`, run logs in `cells/` (also linked as `B/wan2_1/nfw_*`). All runs on a quiet host, one device job at a time.

**Result.**
- **Most of the gap is the block's own device time.** Native compiles each transformer block as its own graph, and that graph runs 12.31 ms on the device against about 10.95 ms per block in AoT's whole-model graph. Over 40 blocks that is 54 ms per forward, as large as the whole gap: 53.5 ms per forward, 107 ms per step.
- **The compiler version is not the cause.** AoT's own HLO, compiled with native's neuronx-cc 2.28, runs as fast as with 2.26.
- **About a sixth of the block gap is one compiler flag.** AoT compiles with `--tensorizer-options=--enable-ccop-compute-overlap`; native does not.
- **The rest, about 40 ms per forward, is not assigned to a measured cause.** *Inference:* it comes from the graph torch-native lowers. Measured: its lowering differs from torch-xla's in the LayerNorm, the GELU and the RoPE, and the native block leaves the matmul engine (PE) idle 4.6 ms per block, against 2.8 ms in AoT; but removing any one of the three never makes the block faster, so no measurement ties the remainder to them.
  - Removing the LayerNorm, the GELU or the RoPE one at a time does not make the block faster.
  - So no single op causes it. *Inference:* the cause is how the compiler schedules the native graph as a whole.
- **No code change in this commit.** The two measured knobs together give −26.8 ms per step (−2.5%), a quarter of the gap. Both change every native model's compiled graphs, and their effect on the other models is not measured (proposals below).

**1. The gap today** (`B/wan2_1/nfw_base`, `nfw_aot`; native and AoT runs interleaved gen0, gen1, gen2; DiT step median of 50 steps without step 0)

| | gen0 | gen1 | gen2 | median |
|---|---:|---:|---:|---:|
| native (`fa3bc3e`, phase-3 flags) | 1065.1 | 1064.9 | 1064.5 | **1064.9 ms** |
| AoT (main, two-core, `--host-vae`) | 957.9 | 959.2 | 957.8 | **957.9 ms** |

- Native is +11.2% (+107.0 ms per step), as on 2026-10-26.
- Native's latents (`fbd0306e…`) and mp4 (`2da4edd4…`) are bit-identical to the phase-3 cell. The current HEAD therefore keeps the phase-3 quality figures: 22.41 dB / 0.880, latent cosine 0.966.

**2. Where one native forward goes** (`scripts/prof_dit.py`, `prof/base.json`)
- **How it was measured:**
  - The DiT is built at TP4 as the CLI builds it: the same application, weight store and NEFF cache.
  - Each segment is timed between `torch.neuron.synchronize()` calls, median of 8.
  - The segments re-implement the model's forward; their output equals the full forward exactly (max abs 0).

| segment | ms |
|---|---:|
| the CLI's adapter call (host inputs → device → forward → sync → host) | 529.1 |
| full forward on device inputs | 525.0 |
| 40 compiled blocks, back to back | **520.1** (13.0 per block; repeats 519.6, 520.2) |
| eager top level before the blocks (RoPE table, patch-embedding Conv3d, condition embedder) | 3.7 |
| eager top level after the blocks (modulation, `norm_out`, `proj_out`, unpatchify) | 2.7 |
| host → device copies of the three inputs | 2.1 |
| one block alone, synced | 14.9 |
| host time to issue the 40 blocks, no sync | 9.7 |

- **Per step:** 2 × 529.1 ms of DiT forwards leaves an unmeasured remainder of 6.7 ms in the 1064.9 ms step (1064.9 − 2 × 529.1). *Inference:* the host-side scheduler step and the CFG combine.
- **Host launches are not the cause.** Issuing the 40 blocks takes 9.7 ms of host time, without a sync, against 520 ms of device time, so launches are asynchronous and hidden.
- **The eager top level and the input copies cost about 9 ms per forward.** AoT runs this work inside its single graph, where it cannot be timed separately.
- **CFG is the same on both backends.** Each runs two batch-1 forwards per step (AoT: `ModelWrapperWanBackbone` traces batch 1).

**3. Device profiles: block against block** (neuron-explorer 2.33 `capture`, 4 workers, worker 0 profiled, random inputs; `ntff/*summary.txt`)

| graph | device time | per block |
|---|---:|---:|
| native block NEFF (`neuron/neff/46/81/4681…`, neuronx-cc 2.28, `-O2`) | 12.31 ms | **12.31 ms** |
| AoT whole-DiT NEFF `MODULE_3c8d124d…` (neuronx-cc 2.26, `-O1` + ccop overlap) | 438.1 ms | **10.95 ms** (pre and post included) |
| the same AoT HLO compiled with native's neuronx-cc 2.28, AoT's flags (`xcc/aot_hlo_cc228/`) | 442.7 ms | 11.07 ms |

**PE and collectives per block on physical core 0** (`scripts/cc_overlap.py` over the parquet export)
- **Native:** PE busy 7.68 ms, idle 4.63 ms. Collectives run 1.07 ms while the PE is idle. Spill traffic is 1.13 GB.
- **AoT:** PE busy 8.11 ms, idle 2.84 ms. Collectives run 0.72 ms while the PE is idle. Spill traffic is about 0.58 GB.
- **Caveat on the AoT figures:** the AoT trace hit the profiler's instruction cap (8,388,078 rows), so it covers the first 319 ms of 438. Its per-block figures are scaled from that window (29.2 blocks).
- **Native's PE is idle during its dots.** The native timeline (`pq_native`) shows the q, k and v projections each taking 0.85–1.0 ms of wall time for 0.42 ms of PE work. The FFN takes 2.8 ms for 2.2 ms.
- **Each native block ends with an exposed tail.** The FFN all-reduce plus the residual add form a 1.0 ms tail that nothing overlaps.

**4. The two graphs** (native block StableHLO saved with `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1`, `hlo_native/block.mlir.txt`; AoT `model.hlo_module.pb`, `hlo_aot_hist.json`)
- **Compile flags:**
  - native: `neuronx-cc compile module.mlir --framework XLA --target trn2 --lnc 2 -O2 --model-type transformer --auto-cast=none` (torch-neuronx's default for `torch.compile` graphs);
  - AoT: `--target=trn2 --model-type=transformer -O1 --tensorizer-options=--enable-ccop-compute-overlap --auto-cast=none`.
- **The same in both graphs:**
  - every linear is `dot(x, transpose(W))` in bf16, ten dots per block;
  - three bf16 all-reduces of `[1,4680,5120]` per block, plus four fp32 `[N,1]` all-reduces for the q/k norm statistics;
  - attention is one `attention_cte` call on `[10,4680,128]` (and `[10,512,128]` for the cross-attention), two-core, with the same `tp_q`/`tp_k` layout;
  - the residual and modulation math is in fp32. Both backends run `modeling_wan.py`, each at its own commit (AoT at main `540e5f0`, native at `fa3bc3e`); the two copies differ by 59 lines (the `--tp-reduce-dtype` plumbing, a tp1 fused `rms_norm_fp32` path and a `.contiguous()` after the patch flatten), none of them on the TP4 path measured here.
- **Different:**

| op | AoT (torch-xla) | native (torch-neuronx) |
|---|---|---|
| LayerNorm (3 per block) | `batch-norm-training` (the vector engine's BN_STATS instructions) | reduce → divide → subtract → square → reduce → rsqrt → multiply, in fp32 over `[4680,5120]` |
| GELU (tanh) | `AwsNeuronGeluApprxTanh` custom call | seven bf16 element-wise ops, `power(x, 3)` and `tanh` among them |
| RoPE's interleaved write | `pad` + `select` | two `scatter` per tensor (q and k) |

**5. Experiments on the block** (40 blocks back to back, ms per forward; base 520.1; `prof/*.json`; each one recompiled)

| change | blocks | Δ |
|---|---:|---:|
| `NEURON_CC_FLAGS=--tensorizer-options=--enable-ccop-compute-overlap` (AoT's flag) | 511.0 | **−9.1** |
| `NEURON_COMPILER_OPT_LEVEL=-O1` (AoT's level) | 540.9 | +20.8 |
| 2 blocks per graph / 4 blocks per graph | 523.5 / 530.9 | +3.4 / +10.8 |
| the same, with the ccop flag (1 / 2 / 4 blocks per graph) | 510.6 / 511.7 / 522.6 | −9.5 / −8.4 / +2.5 |
| RoPE removed (ablation) | 554.7 | +34.6 |
| RoPE with `stack` + `flatten` instead of strided writes (bit-identical on CPU) | 516.6 | **−3.5** |
| GELU removed (ablation) / GELU as `x·sigmoid(1.5958·(x + 0.044715x³))` | 526.4 / 529.3 | +6.3 / +9.2 |
| LayerNorm removed (ablation) | 523.6 | +3.5 |
| ccop flag + scatter-free RoPE | 508.4 | **−11.7** |
| ccop flag + scatter-free RoPE + sigmoid GELU | 507.4 | −12.7 |

- **Graph boundaries are not the cost.** Two or four blocks per graph are slower, with or without the ccop flag. Neither per-graph launch overhead nor cross-block overlap explains the gap.
- **No single element-wise op is the cost.** Removing an op never speeds the block up, and removing RoPE slows it by 35 ms. *Inference:* these ops are not on the critical path, and the compiler's schedule for the torch-native graph is what differs.
- **-O1 makes native slower,** although AoT compiles at -O1.

**6. Both measured knobs in the real CLI** (cell `nfw_fixab`; a snapshot of `fa3bc3e`'s `difflet/` with the scatter-free RoPE in `backends/neuron/ops_impl/embeddings.py`, plus the ccop flag exported; not committed)
- **Step:** 1038.5 / 1038.0 / 1038.2 / 1038.0 ms for gen0, gen1, gen2 and q1. That is −26.8 ms (−2.5%) against 1064.9, and the gap to AoT drops from +11.2% to +8.4%.
- **HBM:** unchanged, 11.83 GiB per core.
- **Quality against CPU fp32 `ref_50`** (`B/nf_wan_step/quality/`, the same tools as phase 3):
  - 22.06 dB / 0.876, latent cosine 0.9648;
  - phase 3: 22.41 dB / 0.880, 0.9664;
  - against the phase-3 q1 directly: 31.9 dB / 0.976, cosine 0.9952.
- **The output changed (mp4 `ccba6698…`).** The −0.35 dB is within the 0.5 dB gate used for the bf16 changes. It comes from one seed, though. It is not attributed: the RoPE change is bit-identical on CPU, so *inference:* the ccop flag's collective and compute order changes the bf16 sums.

**7. Ruled out**, each with its evidence:
- **two-core attention:** both backends run `attention_cte` over both physical cores (F6a audit);
- **CFG batching:** two batch-1 forwards per step on both backends;
- **all-reduce dtype:** bf16 on both;
- **compiler version:** AoT's HLO with 2.28 runs 442.7 ms against 438.1 ms with 2.26;
- **per-block graph launches and host dispatch:** grouping blocks does not help, and dispatch is asynchronous;
- **-O level:** -O1 is worse on native.

**Proposed fixes, by measured gain.** As first written, each changed every native model's compiled graphs and needed the user's decision. Proposals 1 and 2 are now **implemented for Wan only** (section "Task 6" below); the other models keep their graphs.
1. **Give native graph compiles AoT's `--tensorizer-options=--enable-ccop-compute-overlap`:** −9.1 ms per forward, −18 ms per step.
   - The flag reaches neuronx-cc through `NEURON_CC_FLAGS`, which torch-neuronx reads for every graph and eager op. It changes every model's NEFF cache keys, so every root recompiles once.
   - Qwen-Image, HunyuanVideo and LTX-2 must be measured before it becomes the default. Wan's one-seed quality change must be checked against a second seed.
   - **Implemented for Wan only (task 6):** the Wan neuron applications add the flag to their own processes, default on, opt-out `DIFFLET_NEURON_WAN_CCOP_OVERLAP=0`. The second seed passed the gate. The other models were not measured with it and do not get it.
2. **Scatter-free RoPE in `difflet/backends/neuron/ops_impl/embeddings.py`:** −3.5 ms per forward. It is bit-identical on CPU. It also changes the Flux and HunyuanVideo native graphs (`difflet.ops.apply_rotary_emb`).
   - **Implemented for Wan only (task 6):** a separate `apply_rotary_emb_stacked` that only the Wan neuron DiT uses, default on, opt-out `DIFFLET_NEURON_WAN_SCATTER_FREE_ROPE=0`. The shared `apply_rotary_emb` is unchanged, so Flux, HunyuanVideo, the CPU backend and the AoT paths keep their graphs.
3. **Compile the top level** (condition embedder, final norm and projection) as graphs of its own: up to about 5 ms per forward of the 9 ms measured. Not tried.
4. **The remaining ~40 ms per forward** is in how torch-neuronx lowers the block and how neuronx-cc schedules it.
   - Lowering LayerNorm and GELU as AoT does would need torch-neuronx changes or NKI kernels, and the ablations above give no sign that either alone would pay.
   - The better route is a newer torch-neuronx, or a report to AWS with the two graphs in `B/nf_wan_step/` as the reproducer.


## Task 6: Wan-only native step speed-up (ccop overlap + scatter-free RoPE)

2026-10-29. This implements proposals 1 and 2 above for Wan2.1/Wan2.2 on the native neuron backend only. Both levers are **on by default**:

| lever | what changes | opt-out |
|---|---|---|
| ccop overlap flag | The Wan neuron applications append `--tensorizer-options=--enable-ccop-compute-overlap` to their own process's `NEURON_CC_FLAGS`, once and after any flags the user exported. These are the DiT, the device umT5 and the device VAE when built by Wan's callers. | `DIFFLET_NEURON_WAN_CCOP_OVERLAP=0` |
| scatter-free RoPE | The Wan neuron DiT's self-attentions use `apply_rotary_emb_stacked` (`stack` + `flatten`, bit-identical on CPU) | `DIFFLET_NEURON_WAN_SCATTER_FREE_ROPE=0` |

`off`, `false` and `no` also work for both switches. Each lever adds a field to the Wan neuron stage key while it is on: `neuron_cc_flags` on the transformer and vae stages, `rope` on the transformer stage. A manifest from an older compile is therefore not reused, and the stage asks for `difflet compile`. With both switches off, the keys and graphs are exactly those of `d1d2e5b`.

**Setup.**
- **Model and flags:** Wan2.1 14B, 480x832, 9 frames, TP4, 50 steps, phase-3 flags (`--vae-dtype bf16 --text-encoder-dtype bf16`).
- **Before:** a `git archive` of `d1d2e5b`. **After:** `d1d2e5b` plus the task-6 change.
- **Roots:** each side ran on its own hardlinked copy of the p3 root.
- **Runs:** interleaved; per side, 1 compile, 4 timed generates and 2 quality runs (seeds 42 and 7). The opt-out run used the after tree on the before root.
- **Evidence:** in `B/nf_wan6/` (`cells/`, `stats.json`, `quality/`, `logs/device/`). Unit-level scope checks are in the repo's tests.

**Speed** (DiT step, median of 50 steps without step 0)

| | before (`d1d2e5b`) | after (task 6) | Δ |
|---|---:|---:|---:|
| DiT step, timed gen1–3 (median of medians) | 1065.9 / 1064.8 / 1065.5 → **1065.5 ms** | 1038.5 / 1038.5 / 1038.0 → **1038.5 ms** | **−27.0 ms (−2.53%)** |
| all 6 generates per side | 1064.8–1065.9 ms | 1037.7–1039.0 ms | |
| step 0 | 1.111 s | 1.085 s | −26 ms |
| gap to AoT (957.9 ms, section 1 above) | +11.2% | +8.4% | |
| opt-out (both switches 0) | | 1065.6 ms, mp4 `2da4edd4` (the before bits) | back to before |

- The step matches the uncommitted `nfw_fixab` experiment (1038.1 ms), and seed 42's output is bit-identical to it (final latents `f52ee861…`, mp4 `ccba6698…`).
- The opt-out key matched `d1d2e5b`'s manifest, so that run needed no compile and wrote no new HLO.
- **e2e, CLI generate wall, gen1–3 median:** 129.6 s before, 127.3 s after. This number is noisy. Each side spreads over about 11 s (119.0–130.7 s before, 120.5–129.8 s after). The transformer-stage process alone takes either about 89 s or about 98 s, on both sides. *Inference:* the real e2e saving is the step saving, about 50 × 27 ms + 26 ms ≈ 1.4 s, which is smaller than this noise.
- **HBM:** unchanged. The neuron-monitor peak was 11.83 GiB per core on every after run; the before runs peaked at 11.83 GiB, with 11.98 GiB on two of them. The probe peaks were 6.24 / 10.67 GiB on both sides.
- **The flag reaches neuronx-cc:** all 299 neuronx-cc command lines of the after compile carry it. None of the 5 command lines from the before side and the opt-out run do.

**Compile, the one-time cost of the new keys**
- **Wall time:** the first compile with the levers took 988 s. Of that, 489 s is the 27 GB bf16 copy of the fp32 DiT checkpoint, which any compile at `d1d2e5b` writes into a root that has none (the feature from `1e183b4`, not the levers). That leaves about 499 s, against 138 s for the before side's compile, which reused its NEFFs: **about +360 s**.
- **Per stage:** transformer 296 s against 105 s, vae 173 s against 18.5 s.
- **Cause:** every Wan NEFF has the flag in its cache key, so all of them (about 300) recompile cold. The transformer stage also read its checkpoints more slowly: DiT 61 s against 4 s, umT5 87 s against 1 s. *Inference:* the page cache was cold after the copy.
- *Inference, not measured:* later compiles reuse the NEFFs, as the before side's compile did.
- **Disk:** upgrading a root made before `1e183b4` needs 27 GB free for that copy.
- **Moving the tree:** moving it to another path recompiles the 4 block NEFFs, because their debug info holds absolute source paths. This is torch-neuronx behaviour, not caused by this change.

**Quality against CPU fp32 references.** Gate: on each seed, after − before ≥ −0.5 dB mean PSNR and ≥ −0.01 mean SSIM.

| seed | reference | side | PSNR mean / worst (dB) | SSIM mean / worst | latent cosine | Δ PSNR / Δ SSIM | gate |
|---|---|---|---:|---:|---:|---:|---|
| 42 | `ref_50` | before | 22.414 / 20.332 | 0.8800 / 0.8399 | 0.96635 | | |
| 42 | `ref_50` | after | 22.060 / 19.963 | 0.8757 / 0.8340 | 0.96479 | −0.353 / −0.0043 | **PASS** |
| 7 | `ref_50_seed7` (new) | before | 29.517 / 27.154 | 0.9613 / 0.9511 | 0.98962 | | |
| 7 | `ref_50_seed7` (new) | after | 29.701 / 27.304 | 0.9626 / 0.9523 | 0.98963 | +0.183 / +0.0012 | **PASS** |

- **After against before directly:** seed 42 is 31.92 dB mean / 28.51 worst, SSIM 0.976, latent cosine 0.99516. Seed 7 is 36.24 / 33.42 dB, SSIM 0.990, cosine 0.99705.
- **The new seed-7 reference:** `B/wan2_1/cpu_ref/ref_50_seed7/`, a 50-step CPU fp32 diffusers run.
  - It took 4 h 31 min and peaked at 62.6 GiB RSS.
  - It starts from the device's seed-7 noise, bit-identical (raw `d51a24b0…`).
  - Schedule, prompt and negative-prompt embeddings, CFG settings and pipeline config are identical to `ref_50`.
- **Strips** (`B/nf_wan6/quality/strips/`): after and before show the same composition, pose and colour. The ×4 difference holds only fine texture and edge noise. Frame 0 is speckled in every row, the CPU reference included.
- *Inference:* the change of the output comes from the ccop flag, which reorders bf16 sums. The RoPE change is bit-identical on CPU. Each lever alone was not run on the device.

**Scope, other models unchanged**
- **Flux, HunyuanVideo 1.0/1.5 and MiniMax:** they keep the shared `apply_rotary_emb`. Checks on CPU, both trees:
  - HunyuanVideo 1.0's neuron generate-stage key and directory are identical.
  - `NEURON_CC_FLAGS` stays unset after building the HunyuanVideo 1.0 and Flux DiT applications for the neuron device.
  - Tiny HunyuanVideo 1.0 and Flux DiTs in compile mode give identical aten graphs and identical output bits.
- **Qwen-Image:** its native VAE reuses Wan's `NeuronWanVaeDecoderApplication`. A review found that this class added the flag in Qwen-Image processes too. Fix: the class now adds the flag only when its Wan callers pass `ccop_overlap=True`. Unit tests build Qwen-Image's CLI vae stage decoder and its composed application with the device VAE (with and without its DiT) for the neuron device, and check that `NEURON_CC_FLAGS` stays unset. Qwen-Image's stage keys are unchanged.
- **The AoT (trainium, TPU) and CPU backends:** keys and graphs unchanged. The shared modeling file keeps the old RoPE as its default.
- **In-process API callers:** the flag is held by the Wan applications that added it. When the last one is released (garbage collected, or `release_ccop_overlap`), `NEURON_CC_FLAGS` gets its previous value back. A model built after that in the same Python process compiles without the flag. A model built while a Wan application is still alive does get it. The CLI runs each Wan stage in a process of its own, and the stages keep their applications until they exit.
- **A user `--tensorizer-options` in `NEURON_CC_FLAGS` wins:** the flag is not added, a warning is logged, and the stage keys do not name the flag. This came from the review. Before the fix, the key recorded the flag although the stages compiled without it.

**Device behaviour after the review fixes.** The three review fixes do not change what a default Wan run compiles. Recomputed from the fixed tree for the device runs' command line:
- the default keys equal the after manifests (`6b2db266…` transformer, `af4a926a…` vae);
- the opt-out keys equal the before manifests (`41c1fae1…`, `a9a52a99…`).

The cells were therefore not re-run.

**Not done.**
- Wan2.2 A14B on the device. Its two experts are two instances of the same DiT application, so they get both levers. *Inference, not measured.*
- Each lever on its own on the device.


## Next

- **Phase 4 (CPU):** the 7 references, 50 steps each, in fp32, Qwen-Image and LTX-2 first. Inputs:
  - **Qwen-Image:**
    - AoT noise: `B/qwen_image/aot_main_tp4/q1`, `initial.pt` `c2d9d8b3…`;
    - native noise: `B/qwen_image/native_tp4_hashcheck/q1`, `initial.pt` `95a4306b…`.
  - **LTX-2:**
    - AoT noise: `B/ltx_2/aot_main_tp4_twocore/q1`;
    - native noise: `B/ltx_2/native_tp4/q1`, `initial_video` / `initial_audio` raw `55b90133…` / `24900b1b…`.
    - The reference needs a lossless frames output (plan D9): done, `cpu_ref_ltx2.py` writes `out.pt`.
  - Qwen-Image: **done** (section "Phase 4: Qwen-Image CPU fp32 references").
  - LTX-2: **done** (section "Phase 4: LTX-2 CPU fp32 references"). It leaves an open finding, the AoT gap against 2026-10-24, which needs a 20-step two-core AoT device run.
  - **HunyuanVideo:**
    - native noise: fp32, `B/hunyuan_video/native_tp4_hashcheck/q1/initial.pt`;
    - AoT noise: bf16, `B/hunyuan_video/aot_main_tp4/q1/initial.pt`.
    - Caveat: the AoT VAE tiling caps frame PSNR at about 42.5 dB (`hunyuan_video_neuron.md`).
  - HunyuanVideo and Wan2.1 tooling: **ready** (section "Phase 4 part 2 preparation").
  - Wan2.1 `ref_50`: **done** 2026-10-27 05:33 UTC (section "Quality vs CPU fp32 reference (part 2)").
  - HunyuanVideo `ref_native_noise_50`: **done** 2026-10-27 09:40 UTC (section "Quality vs CPU fp32 reference (part 2)"). `ref_aot_noise_50` **moves to the next machine** (host termination about 11:00 UTC): `job_cpu_ref_queue_wan_hv.sh` run (3), then `B/scripts/quality_part2_hv.sh` extended with the AoT pair.
  - **Wan2.1:** one reference serves both backends: noise `03b6284b…` (raw `16aeaed7…`).
    - Native frames: `B/wan2_1/native_tp4_hashcheck/q1/out.pt`.
    - AoT frames: use the fp32 host decode `B/wan2_1/aot_main_tp4_twocore/q1_host_vae/host_fp32_frames.pt`, which is the A2 placement, because the device-VAE frames are corrupted.
- **Phase 5:** PSNR / SSIM fill the last column of the summary table. Part 1 (Qwen-Image and LTX-2) is **done**: section "Quality vs CPU fp32 reference (part 1)". Part 2: Wan2.1 and HunyuanVideo native **done** (section "Quality vs CPU fp32 reference (part 2)"); HunyuanVideo AoT waits for `ref_aot_noise_50` on the next machine.
- **Phase 6:** definition B, native cold without the weight store, for Wan2.1 and LTX-2: **done** 2026-10-27 on the new host (section "2026-10-27, new host"), together with the LTX-2 AoT two-core 20-step diagnostic. `hunyuan_video/ref_aot_noise_50` started there at 18:12:52 UTC (unit `cpuref_hv_aot`).
- **Phase 3:** bf16 VAE and umT5 for native Qwen-Image and Wan2.1: **done** 2026-10-27/28 on the new host (section "Phase 3: bf16 VAE and umT5"); every gate passed. Making any of them the default needs the user's approval (the `74fc979` rule: accuracy plus TP2, TP4 and TP2xCP2 runs). Phase 7 (compiled-by-region Wan VAE) can build on the validated bf16 decoder.
- **Phase 7 (last step):** the region-compiled Wan/Qwen-Image VAE: **done** 2026-10-28 (section "Last step: region-compiled Wan/Qwen VAE"). It ships as the opt-in `--vae-exec-mode compile`: 55.9 dB, VAE compile −22% (Wan) and −34% (Qwen-Image), warm e2e +1.0% / +3.5% at the benchmark shapes, faster from about 45 frames. It is not the default, which would need the user's approval.
- **Follow-up on main:** the compiled bf16 Wan VAE (`trainium/wan/vae.py`) corrupts frames on the device. Wan2.2 on 2026-10-24 shows the same kind of corruption.

## Toolchain

- **AoT:** `/home/ubuntu/venvs/difflet-neuron`: Python 3.12.3, torch 2.9.1, torch-neuronx 2.9.0.2.15.32035+de43f57c, neuronx-cc 2.26.6360.0+6f180f47.
- **Native:** `/home/ubuntu/venvs/native-312`: Python 3.12.3, torch 2.14.0, torch-neuronx 2.14.3.0.1+b1b6c66, neuronx-cc 2.28.7631.0+8f7b0cf5 (re-read from the venv after phase 2; the same as in `native_vs_aot_tp4_2026-10-25.md`).
- **Host runtime:** aws-neuronx-runtime-lib 2.35.30.0, collectives 2.35.43.0 (re-read with `dpkg -l` after phase 2).
- The *reused* native cells ran on this host; their toolchains are listed in their own reports.
