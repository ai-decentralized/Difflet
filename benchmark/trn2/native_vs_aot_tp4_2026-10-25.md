# Native vs AoT at TP4 on trn2.3xlarge, re-measured 2026-10-25

**Backends:** neuron (TorchNeuron: torchrun, one process per NeuronCore, per-block `torch.compile`; "native") and the Trainium AoT backend (NxD; "AoT"), the same `difflet` CLI on the same host  
**Device:** trn2.3xlarge: 1 device, 4 NeuronCores (LNC=2, 24 GiB HBM per core), 12 vCPU, 124.8 GiB RAM, one EBS root volume  
**Code under test:** branch `feat/hunyuan-neuron`, clean tree in every run: Qwen-Image at `4f6e1de` (`head: 4f6e1de dirty=0` in each `run.log`), Wan2.1 at `8b2408c` (`head: 8b2408c dirty=0` in all 22 Wan `run.log`s; `8b2408c` only adds the Qwen-Image section of this doc)  
**Raw evidence:** kept out of git, under `/home/ubuntu/bench-2026-10-25/` on the benchmark host (`B` below). Each run has its own directory `B/<model>/<cell>/<run>/` with:
- `run.log`: the CLI output and `/usr/bin/time -v`;
- `summary.json`: the harness record (command, env, HEAD, meminfo, checkpoint page-cache residency, probe timeline, hashes, PNG check);
- `probe/`: the per-process events;
- `nmon.jsonl`: neuron-monitor, sampled every second;
- `out.png` (Qwen-Image) or `out.mp4` and its middle frame `mid.png` (Wan2.1), and `work/latents.pt`.

`B/<model>/summary.jsonl` has one line per run (`B/qwen_image/`, `B/wan2_1/`).

Every number below comes from a run made for this report and names its run directory, unless it is marked *reused*. Inferences are labelled. HunyuanVideo's TP4 native-vs-AoT comparison from the same day is in `hunyuan_video_neuron.md`.

## Summary: three models at TP4, native vs AoT

All three were measured on this host on 2026-10-25 with the same method (cold compile on a fresh root, cold e2e with the page cache dropped, gen0 discarded, n=3 timed warm runs, except HunyuanVideo's AoT cell with n=2: 133.0 and 132.9 s). The HunyuanVideo rows are *reused* from `hunyuan_video_neuron.md` (its "TP4: native vs AoT" table: native re-measured with the same method on 2026-10-26 after the F1/F2 VAE work, AoT not re-run); the other two are this report's sections below. Native vs AoT is (native − AoT) / AoT: negative means native is faster or smaller.

| model, workload | DiT step median: native / AoT | compile cold: native / AoT | e2e warm: native / AoT | e2e cold: native / AoT | peak HBM per DiT core (neuron-monitor): native / AoT |
|---|---|---|---|---|---|
| HunyuanVideo 1.0, 320x512x61, 50 steps, native after F1/F2 (2026-10-26, `eadddec`; *reused*, `hunyuan_video_neuron.md`; AoT the same 2026-10-25 cell) | 808.2 / 813.9 ms (−0.7%) | 274.5 / 4347.7 s (−93.7%; AoT a lower bound: pre-sharded weights reused) | 126.7 / 132.9 s (−4.7%) | 457.8 / 617.3 s (−25.8%) | 17.95 / 15.52 GiB on rank 0's core, which also holds the decoder (+15.7%); 12.98 GiB on the other three (−16%) |
| HunyuanVideo 1.0, same, native before F1/F2 (eager fp32 vae launch, 2026-10-25) | 808.3 / 813.9 ms (−0.7%) | 2877.2 / 4347.7 s (−34%) | 239.1 / 132.9 s (+80%) | 570.4 / 617.3 s (−7.6%) | 12.98 / 15.52 GiB (−16%) |
| Qwen-Image, 1024x1024, 50 steps | 359.0 / 418.2 ms (−14.2%; 331.0 ms with `--sp`, −20.9%) | 187.0 / 1238.5 s (−84.9%) | 82.3 / 75.4 s (+9.2%) | 502.0 / 504.2 s (−0.4%) | 10.81 / 10.29 GiB (+5.1%) |
| Wan2.1 14B, 480x832x9, 50 steps, AoT with two-core attention | 1063.6 / 955.8 ms (+11.3%; 1042.7 ms with `--sp`, +9.1%) | 600.3 / 445.1 s (+34.9%) | 128.5 / 120.0 s (+7.1%) | 542.7 / 447.5 s (+21.3%) | 15.15 / 10.37 GiB (+46%) |
| Wan2.1 14B, same, AoT stock | 1063.6 / 1150.8 ms (−7.6%) | 600.3 / 453.3 s (+32.4%) | 128.5 / 131.5 s (−2.3%) | 542.7 / 455.6 s (+19.1%) | 15.15 / 10.34 GiB (+47%) |

- **DiT step:** native is level with AoT on HunyuanVideo and 14% faster on Qwen-Image. On Wan2.1 it beats stock AoT by 7.6% but trails two-core AoT by 11.3%. Wan's two-core AoT traces the NKI attention for both physical cores of a logical core; HunyuanVideo's and Qwen-Image's AoT CLIs already do.
- **Warm e2e:** on 2026-10-25 AoT led on all three, counting Wan's two-core AoT (HunyuanVideo after F1/F2: native leads, below); native beats Wan's stock AoT by 3.0 s. The gap is 8.5 s on Wan2.1 and 6.9 s on Qwen-Image, and on both most of it is native's launch: 20–24 s from rank start to load start. On HunyuanVideo the gap was 106 s, almost all of it native's eager fp32 decode; after F1/F2 (2026-10-26: compiled bf16 decoder on rank 0's core in the generate launch) native leads by 6.2 s (126.7 vs 132.9 s).
- **Cold e2e:** native is level or ahead on HunyuanVideo (25.8% ahead after F1/F2) and Qwen-Image, and 21% behind on Wan2.1. Wan's cold load includes 172.5 s for the fp32 umT5, which the weight store does not cover.
- **Compile:** native compiles 85% faster on Qwen-Image and 34% faster on HunyuanVideo (94% after F1/F2), but 35% slower on Wan2.1. Wan's compile writes the bf16 weight store (192–215 s per rank), and its vae launch fills the eager kernel cache (170 s).
- **HBM:** native uses more HBM per DiT core only on Wan2.1, 4.8 GiB more, and, after F1/F2, on HunyuanVideo's rank 0 core, which now also holds the decoder (2.4 GiB more than AoT's cores). *Inference:* most of that is the umT5 on the device in fp32 (5.68 GB per rank) against AoT's bf16 umT5 (10.84 GB over 4 shards).

## Qwen-Image

**Model:** `Qwen/Qwen-Image` at revision `75e0b4be04f60ec59a75f475837eced720f823b6`.  
**Workload:** 1024x1024, 50 steps, seed 42, prompt "a cinematic shot of a red fox running through a snowy forest", `--guidance-scale 4.0` (the DiT drops it: `guidance_embeds` is off), `HF_HUB_OFFLINE=1`, the workload of `neuron_final.md`.  
**Campaign:** 2026-10-25, 05:36 – 06:50 UTC.

**Placements:**
- **Native:** the default placement. Qwen2.5-VL encodes on rank 0's host and the DiT runs at tp4 in the generate launch; the vae stage then decodes in fp32 in a one-core launch of its own.
- **AoT:** `--backend trainium` with the NxD venv, at its own default placement. Three stage processes run one after the other:
  - the text stage: Qwen2.5-VL at TP4 on the device;
  - the generate stage: the DiT at TP4, with the two-core attention already in Qwen-Image's AoT CLI (`_VIRTUAL_CORE_SIZE = 2`);
  - the vae stage: the compiled VAE on one core.

**Compile cold:** `difflet compile` on a fresh root, with the checkpoint in the page cache (see Method). For AoT, the fresh root also holds a fresh pre-sharded weight store and a fresh neuronx-cc cache, so the AoT compile includes the weight sharding and every NEFF compile.

| metric | native (`native_tp4`) | AoT (`aot_tp4`) | native vs AoT | 2026-10-20 values (`neuron_final.md`, *reused*): native / AoT / native vs AoT |
|---|---:|---:|---:|---|
| DiT step median / p10 / p90 | **359.0** / 358.2 / 360.1 ms (runs 359.1, 359.0, 358.8) | **418.2** / 417.2 / 418.9 ms (runs 418.2, 417.8, 418.3) | **−14.2%** | 352.6 / 418.4 ms / −15.7% |
| DiT step with `--sp` (`native_tp4_sp`, n=2) | **331.0** / 330.4 / 332.5 ms (runs 330.9, 331.0) | — | **−20.9%** | 329.9 / 418.4 ms / −21.1% |
| compile cold, checkpoint in the page cache | **187.0 s** (generate launch 91.2, vae launch 93.8) | **1238.5 s** (text stage 134.4, generate stage 665.2, vae stage 419.4) | **−84.9%** | 161.7 / 1480 s / −89% |
| e2e warm (n=3; gen0 discarded) | **82.3 s** (81.4, 82.1, 83.4; gen0 72.5) | **75.4 s** (74.2, 76.4, 75.7; gen0 75.0) | **+9.2%** | 79.6 / 76.9 s / +3.5% |
| e2e cold, page cache dropped | **502.0 s** (DiT checkpoint 310.9 s, host Qwen2.5-VL 99.6 s) | **504.2 s** (text stage 140.7 s, DiT load 315.9 s) | **−0.4%** | 496.1 / 505.9 s / −1.9% |
| peak HBM per DiT core: allocated / neuron-monitor | 10.15 / 10.81 GiB (vae core: 9.30 / 13.85 GiB) | — / 10.29 GiB (no `max_memory_allocated`: the AoT stages are not torchrun ranks) | +5.1% (neuron-monitor) | native 10.15 / 10.76 GiB, vae core 13.83 GiB; AoT not given |
| output (opened) | fox in a snowy forest; PNG `b1906d53…`, latents.pt `4da4c89e…` | fox walking in snow, a different composition (see "Outputs"); PNG `e24798fe…`, latents.pt `7b36ae14…` | | native PNG `a300a78e…` |
| logs | `B/qwen_image/native_tp4/{compile,cold,gen0..gen3}/` | `B/qwen_image/aot_tp4/{compile,cold,gen0..gen3}/` | | |

What the table says:

- **The DiT step reproduces the 2026-10-20 picture.**
  - Native is 14.2% faster than AoT per step, and 20.9% faster with `--sp`.
  - AoT's step is unchanged: 418.2 ms against 418.4 ms.
  - Native's step is 1.8% slower than on 2026-10-20 (see "Drift").
- **Warm e2e: AoT leads by 6.9 s (75.4 vs 82.3 s), up from 2.7 s.** Native's denoise is 3.3 s faster (18.0 vs 21.3 s), but two things cost it more (breakdown below):
  - its generate launch spends 21.5–24.4 s from rank start to load start in every timed run (launch jitter, open issue 1);
  - its eager fp32 vae launch takes 19.2–19.5 s, against 13.3–14.4 s for AoT's compiled vae stage.

  With a fast launch (11.7 s, as in native gen0), native's e2e is 72.5 s, which would beat AoT's.
- **Cold e2e: level (−0.4%).**
  - Native reads the bf16 checkpoint (310.9 s) and the host Qwen2.5-VL (99.6 s) in one load phase.
  - AoT reads its text encoder in the text stage (140.7 s), then its pre-sharded DiT (315.9 s).
- **Compile: native is 6.6x faster** (187.0 vs 1238.5 s).
  - AoT's time is the VAE's neuronx-cc compile (404.2 s), the DiT weight sharding (361.7 s), the DiT build (297.5 s, of which 85.2 s loading the module and 31.5 s compiling) and the text encoder build (106.2 s) (`aot_tp4/compile/run.log`).
  - Native writes no weight store ("no gain": the checkpoint is already bf16), the same as on 2026-10-20.
- **HBM: AoT uses 0.5 GiB less per DiT core** by neuron-monitor (10.29 vs 10.81 GiB). Native's vae core peaks at 13.85 GiB, in a launch of its own after the DiT launch has exited.

### Drift from 2026-10-20 (flag: more than 2%)

| cell, metric | now | 2026-10-20 (*reused*, `neuron_final.md`) | delta | flag | likely cause |
|---|---:|---:|---:|:---:|---|
| native tp4, step | 359.0 ms | 352.6 ms | +1.8% | — (below 2%) | See note 1. |
| native tp4, compile cold | 187.0 s | 161.7 s | **+15.6%** | **yes** | See note 2. |
| native tp4, e2e warm | 82.3 s | 79.6 s | **+3.4%** | **yes** | See note 3. |
| native tp4, e2e cold | 502.0 s | 496.1 s | +1.2% | — | The load phases match: DiT checkpoint 310.9 s now and on 2026-10-20, host Qwen2.5-VL 99.6 s now against 99.9 s. |
| native tp4, HBM alloc / nmon | 10.15 / 10.81 GiB | 10.15 / 10.76 GiB | 0 / +0.5% | — | |
| native `--sp`, step | 331.0 ms | 329.9 ms | +0.3% | — | |
| native `--sp`, e2e warm | 73.3 s (77.1, 69.6) | 75.0 s (76.2, 73.8) | **−2.3%** | **yes** | Launch jitter, in the other direction. Rank 0's start-to-load time was 19.1 s in gen1 and 11.9 s in gen2 (`native_tp4_sp/gen{1,2}`). n=2 is too few to average the jitter out. |
| native `--sp`, compile (page cache as left: 94% of the checkpoint resident) | 162.6 s (generate launch 69.3, vae launch 91.9) | 154.8 s (62.1 + 92.5) | **+5.0%** | **yes** | The generate launch is 7.2 s longer and the vae launch is level. *Inference:* the same toolchain change as note 1. The 2026-10-20 split below the launch level is not available to confirm it. |
| native `--sp`, HBM alloc | 10.00 GiB | 9.93 GiB | +0.7% | — | |
| AoT tp4, step | 418.2 ms | 418.4 ms | −0.05% | — | |
| AoT tp4, compile cold | 1238.5 s | 1480 s | **−16.3%** | **yes** | See note 4. |
| AoT tp4, e2e warm | 75.4 s | 76.9 s | −2.0% (−1.95%) | — (at the threshold) | |
| AoT tp4, e2e cold | 504.2 s | 505.9 s | −0.3% | — | |

Notes on the drift rows:

1. **Native tp4 step, +1.8%.** The 6.4 ms is a steady per-step offset: every warm step is between 358 and 361 ms. The output is no longer bit-identical to 2026-10-19: the PNG is `b1906d53…` against `a300a78e…`, so the compiled DiT itself changed. *Inference:* the cause is the toolchain and host change, not the Difflet code. Measured facts behind that:
   - This report runs on a different trn2.3xlarge instance. Its `/home/ubuntu` was created on 2026-10-19 at 22:11 UTC (`stat`), and the 2026-10-19 campaign's evidence directory, `/home/ubuntu/bench-neuron-final`, is not on it.
   - The native venv is different. Then: Python 3.14.8, neuronx-cc 2.28.4568.0, torch-neuronx 2.14.3.0.0. Now: Python 3.12.3, neuronx-cc 2.28.7631.0, torch-neuronx 2.14.3.0.1.
   - The Neuron runtime is different. Then: runtime 2.35.27.0, collectives 2.35.40.0. Now: 2.35.30.0, 2.35.43.0.
   - The earlier venv is not on this host, so no A/B was possible. `61022ec`, the commit tested on 2026-10-19, is not in this clone's history, so the code diff cannot be listed either.
2. **Native tp4 compile, +15.6%.** The whole extra time is in the generate launch: 91.2 s against 68.4 s. The vae launch is level (93.8 s against 93.0 s).
   - Within the generate launch, torchrun took 13.85 s to start rank 0, against 2.66 s in the `--sp` compile, and "build on meta" took 9.71 s against 2.77 s (`native_tp4/compile` vs `native_tp4_sp/compile`, `phase_seconds`).
   - *Inference:* the page-cache drop before the pre-read also evicted the venv's libraries, and the pre-read only brings back the checkpoint. The cold e2e, also run straight after a drop, shows the same 14.3 s rank start.
   - The `--sp` compile, with the page cache as left, took 162.6 s.
3. **Native tp4 warm e2e, +3.4%: launch jitter, plus 0.4 s of denoise.**
   - Rank 0's start-to-load time was in its slow mode in all three timed runs: 21.5, 22.4 and 24.4 s. In gen0 it was 11.7 s, and that run took 72.5 s.
   - Against the 2026-10-20 gen2 breakdown, the parts are within 1.5 s each (table below). The denoise is 18.0 s against 17.6 s.
4. **AoT compile, −16.3%.** The 1480 s is *reused* from an A/B at `03f3a84` (`neuron_feature_matrix.md`) and has no breakdown, so the difference cannot be attributed to a part.
   - This run is not a cache hit. It started with an empty pre-sharded store and an empty neuronx-cc cache in the fresh root, it wrote 39 GB of shards ("Done Sharding weights in 361.7"), and it compiled every NEFF, about 122 MB of new cache entries.
   - *Inference:* the checkpoint was in the page cache here (Method). The page-cache state of the 2026-10-17 run is unknown.

### Where the warm e2e goes (gen2, seconds)

Times come from the probe events in `B/qwen_image/<cell>/gen2/summary.json`; `B/scripts/qbreakdown.py` prints them. The brackets hold the 2026-10-20 native gen2 breakdown (*reused*, `neuron_final.md`).

| part | native (`native_tp4/gen2`) | AoT (`aot_tp4/gen2`) |
|---|---:|---|
| process wall | 82.1 [78.4] | 76.4 |
| text | (in the generate launch) | text stage 16.4 (Qwen2.5-VL TP4 on the device) |
| generate launch / stage, total | 61.6 [60.2] | 41.2 |
| — torchrun start to rank 0 start | 2.7 | — |
| — rank 0 start to load start (imports, runtime, process group) | 22.4 [21.7] | — |
| — load + warm-up | 16.1 [15.3]: build on meta 2.8, DiT checkpoint 3.7, warm-up forward 8.3, host Qwen2.5-VL load 0.9 | DiT load 15.2 |
| — host encode (rank 0) | 0.5 [0.5] | — |
| — denoise, 50 steps | 18.0 [17.6] | 21.1 |
| — rest (save latents, teardown; AoT: imports and setup) | 2.0 | 4.9 |
| vae launch / stage, total | 19.2 [18.0]: launch 2.5, imports 3.1, load 9.5, decode 2.5, PNG 0.3, exit 1.2 | 14.4 (compiled VAE, one core; PNG 0.3) |
| gaps between launches, driver exit | 1.3 | 4.4 |

- **Native's denoise and encode are faster; what it loses is launch time.** It spends 22.4 s from rank start to load start, and 5.6 s of its vae launch goes to launching and importing before the decoder loads.
- **AoT's text stage costs 16.4 s and native's host encode costs 0.5 s.** Native loads the host Qwen2.5-VL inside its load phase, in 0.9 s warm.

### Native per cell

| metric | `native_tp4` (headline) | `native_tp4_sp` |
|---|---:|---:|
| DiT step median / p10 / p90 (ms) | 359.0 / 358.2 / 360.1 (step 0: 0.36 s) | 331.0 / 330.4 / 332.5 |
| e2e warm (s) | 81.4, 82.1, 83.4 (mean 82.3; gen0 72.5) | 77.1, 69.6 (mean 73.3; gen0 79.6) |
| e2e cold, page cache dropped (s) | 502.0: generate launch 480.1 (rank start 14.3, imports 13.3, load + warm-up 429.6, of which DiT checkpoint 310.9 and host Qwen2.5-VL 99.6), vae launch 19.8 | not run (extra cell) |
| compile cold (s) | 187.0 (checkpoint pre-read: 100% resident) | 162.6 (page cache as left: 94% resident) |
| compile: generate launch / vae launch (s) | 91.2 / 93.8 (the vae warm-up decode fills the eager kernel cache: 85.2–86.6 s) | 69.3 / 91.9 |
| NEFFs in the root after compile | 135 | — |
| peak HBM allocated / neuron-monitor per DiT core; vae core (GiB) | 10.15 / 10.81; 9.30 / 13.85 | 10.00 / 10.63; 9.30 / 13.83 |
| weight store | not written ("no gain": 10.22 GB either way, bf16 checkpoint) | same |
| output | PNG `b1906d53…` and latents.pt `4da4c89e…` in the cold run and all 4 warm runs | PNG `cf5d5e02…` and latents.pt `fc6d04c4…` in all 3 runs |
| logs | `B/qwen_image/native_tp4/` | `B/qwen_image/native_tp4_sp/` |

SP takes 7.8% off the step (331.0 vs 359.0 ms) and 0.15 GiB off the allocated HBM. Its warm e2e is 9.0 s lower on average, but most of that is launch jitter: gen2 had the fast 11.9 s launch.

### Outputs

- **All three cells' outputs show the prompt's content.** One PNG was opened per cell (`native_tp4/cold`, `native_tp4_sp/gen1`, `aot_tp4/cold`): each shows a red fox in snow in front of a snowy forest.
  - The native PNG has pixel std 52.8 and mean neighbour difference 1.73 levels; AoT's has 70.3 and 1.67. These are natural images, not noise.
  - Within each cell, every run produced the same PNG and the same latents.pt, bit for bit (hashes in each `summary.json`).
- **The native image is no longer the 2026-10-19 golden.** That was PNG `a300a78e…` and latents.pt `aac25966…` (*reused*, `neuron_final.md` A1). Now: `b1906d53…` and `4da4c89e…`. See drift note 1.
- **AoT's composition differs from native's, and the seed-42 noise explains it.** AoT shows a fox walking across open snow; native shows one trotting between trees.
  - Measured on this host: `torch.randn((1, 1, 16, 128, 128), dtype=torch.bfloat16)` after seeding 42 gives different values in the two venvs. The first values are 1.930 in torch 2.14 and −0.809 in torch 2.9.
  - The fp32 draw is identical in both.
  - *Inference:* the pipeline's `prepare_latents` draws the noise in its bf16 dtype, so the two backends start from different noise. The SP image matches the tp4 image to the eye; its PNG differs because SP changes the reduction order.

## Wan2.1 14B

**Model:** `Wan-AI/Wan2.1-T2V-14B-Diffusers` at revision `38ec498cb3208fb688890f8cc7e94ede2cbd7f68` (DiT and umT5 checkpoints in fp32).  
**Workload:** 480x832x9, 50 steps, `--guidance-scale 5.0` (CFG: two DiT forwards per step), seed 42, prompt "a cinematic shot of a red fox running through a snowy forest", `HF_HUB_OFFLINE=1`. This is the workload of `neuron_final.md`.  
**Campaign:** 2026-10-25, 07:00 – 09:17 UTC, at `8b2408c`.

**Cells and placements:**
- **`native_tp4` (headline):** the neuron backend's default placement with the weight store. umT5 runs on the device in fp32, TP-sharded beside the DiT, in the generate launch. The VAE decodes in fp32 in a one-core launch of its own. Flags: `--backend neuron --tp-degree 4`.
- **`native_tp4_sp`:** the same with `--sp`. n=2, and no cold e2e (an extra cell).
- **`aot_tp4` (AoT stock):** `--backend trainium --tp-degree 4 --host-vae` with the NxD venv, from this branch. One transformer stage process runs umT5 and the DiT at TP4. The driver then decodes on the host.
  - `--host-vae` is the placement of the AoT numbers this compares against: `neuron_feature_matrix.md` lists that cell as "AoT tp4 (`--host-vae`)".
  - Its outputs equal the 2026-10-20 AoT hashes (below), so it is the same configuration.
- **`aot_tp4_twocore` (AoT, two-core attention):** the same command with `NEURON_RT_VIRTUAL_CORE_SIZE=2` exported in every run of the cell, so the trace picks `attention_cte[2]` (`difflet/backends/trainium/ops_impl/attention.py`). **Deviation from the plan:** the plan was to check out commit `5f62732` from `origin/feat/aot-bf16-allreduce` in a separate worktree. Neither exists on this host:
  - `git ls-remote origin` lists no `feat/aot-bf16-allreduce` branch.
  - `5f62732` is not an object in this clone (`unknown revision`).
  - No other clone on the host has it: `/home/ubuntu/campaign/*` are worktrees of the same repository.

  `bda3dde`'s message describes that commit as "NEURON_RT_VIRTUAL_CORE_SIZE=2 at trace time". `runner.run_stage` passes a shell-exported `NEURON_RT_VIRTUAL_CORE_SIZE` through to a staged AoT stage unchanged, because Wan's `_VIRTUAL_CORE_SIZE` is `None`. So the variable reproduces the change without a code edit. Three checks confirm that it took effect:
  - **The trace changed.** The DiT's HLO module is `MODULE_5ae08501…` against stock's `MODULE_c984433f…`, and umT5's is `579580ee…` against `8713aabf…` (`aot_tp4{,_twocore}/compile/run.log`).
  - **The outputs are bit-identical** to stock and to the 2026-10-20 two-core run (below).
  - **The step matches** 2026-10-20's two-core figure to 0.14%.

  No worktree was created, so there was none to remove.

**Compile cold:** `difflet compile` on a fresh root, with the checkpoint in the page cache (see "Method differences").
- **Native:** the compile writes the bf16 weight store into the root (`<root>/neuron/weights`).
- **AoT:** the fresh root also holds a fresh pre-sharded store and neuronx-cc cache, so the compile includes the weight sharding and every NEFF compile. The two AoT cells have separate roots.

| metric | native (`native_tp4`) | AoT two-core (`aot_tp4_twocore`) | native vs AoT two-core | AoT stock (`aot_tp4`) | native vs AoT stock | 2026-10-19/20 (`neuron_final.md`, *reused*): native / AoT two-core / AoT stock |
|---|---:|---:|---:|---:|---:|---|
| DiT step median / p10 / p90 | **1063.6** / 1061.5 / 1065.9 ms (runs 1063.6, 1063.8, 1063.2) | **955.8** / 954.8 / 957.0 ms (runs 955.8, 955.8, 955.9) | **+11.3%** | **1150.8** / 1149.9 / 1151.5 ms (runs 1150.8, 1150.8, 1150.7) | **−7.6%** | 1063.1 / 957.1 / 1151.5 ms |
| DiT step with `--sp` (`native_tp4_sp`, n=2) | **1042.7** / 1041.5 / 1045.8 ms (runs 1042.8, 1042.6) | | +9.1% | | −9.4% | 1043.7 ms |
| compile cold, checkpoint in the page cache | **600.3 s** (generate launch 428.8, vae launch 169.7) | **445.1 s** (umT5 build 35.8 + sharding 5.7; DiT build 251.3, of which NEFF compile 62.5, + sharding 113.3) | **+34.9%** | **453.3 s** (umT5 36.0 + 5.7; DiT 258.3, of which NEFF compile 72.1, + 113.8) | **+32.4%** | 518.9 / 303.2 / 482.0 s |
| e2e warm (n=3; gen0 discarded) | **128.5 s** (126.0, 130.8, 128.8; gen0 129.4) | **120.0 s** (120.4, 119.7, 120.0; gen0 119.5) | **+7.1%** | **131.5 s** (132.5, 131.1, 131.1; gen0 130.4) | **−2.3%** | 125.5 / 123.1 / 134.5 s |
| e2e cold, page cache dropped | **542.7 s** (DiT from the store 217.0 s, umT5 checkpoint 172.5 s) | **447.5 s** (load of the pre-sharded umT5 + DiT 323.1 s, host decode 48.3 s) | **+21.3%** | **455.6 s** (load 322.6 s, host decode 48.0 s) | **+19.1%** | 520.2 / 450.0 / 461.8 s |
| peak HBM per DiT core: allocated / neuron-monitor | 13.48 / 15.15 GiB (13.58 allocated in the compile's warm-up); vae core 11.84 / 18.40 GiB | — / 10.37 GiB | +46% (neuron-monitor) | — / 10.34 GiB | +47% | native 13.58 / 15.15, vae core 18.51 GiB; AoT stock 10.34 GiB (`neuron_feature_matrix.md`) |
| output (mid frame opened) | fox running through a snowy forest. mp4 `9b914e81…`, latents.pt `3cafa271…` in the cold run and all 4 warm runs: **the 2026-10-19 golden, bit for bit** | fox running through a snowy forest. mp4 `d2ecce24…`, latents.pt `f3dd7d2e…` in all 5 generates | | the same mp4 and latents as two-core, in all 5 generates | | the same hashes on 2026-10-19/20 |
| logs | `B/wan2_1/native_tp4/{compile,cold,gen0..gen3}/` | `B/wan2_1/aot_tp4_twocore/…` | | `B/wan2_1/aot_tp4/…` | | |

AoT's peak HBM comes from neuron-monitor only, because the AoT stages are not torchrun ranks. The SP cell's percentages are against the AoT step in the same column.

What the table says:

- **The step picture of 2026-10-20 holds.**
  - Two-core AoT is the fastest DiT: native is 11.3% slower per step (11.1% on 2026-10-20) and 9.1% slower with `--sp`.
  - Native is still 7.6% faster than stock AoT.
  - All three steps are within 0.14% of their 2026-10-19/20 values.
- **Warm e2e: two-core AoT leads by 8.5 s (120.0 vs 128.5 s); native beats stock AoT by 3.0 s.** In the breakdown below, native's denoise costs 5.3 s more than two-core AoT's, and its launch about 20 s more (rank start to load start). Its device decode (vae launch 23.2 s) saves about 24 s against AoT's host decode (47.0 s).
  - With a fast launch, native would be level: `native_tp4_sp/gen2` had a 12.1 s rank start-to-load and took 116.6 s.
- **Cold e2e: AoT leads by 95 s (two-core) and 87 s (stock).**
  - Native reads the DiT from its bf16 store (217.0 s) and the fp32 umT5 from the checkpoint (172.5 s): 389.6 s of reads.
  - AoT reads its pre-sharded bf16 umT5 and DiT in 323 s.
  - The two load figures match 2026-10-19's (217.4 and 172.5 s, `neuron_final.md`) to 0.4 s.
- **Compile: AoT is 26% faster (native +35%).**
  - Native pays the weight-store write: 191.7–215.2 s per rank, in the "load checkpoint" phase of 220.3 s. It also pays 169.7 s of vae launch, where the warm-up decode fills the eager kernel cache.
  - AoT pays 119 s of weight sharding.
- **HBM: native uses 4.8 GiB more per DiT core** by neuron-monitor (15.15 vs 10.37 GiB). *Inference:* most of it is umT5, which native holds in fp32 (5.68 GB per rank, the weight store's "not written: no gain" line in `native_tp4/compile/run.log`). AoT shards umT5 in bf16: 10.84 GB over 4 shards, about 2.7 GB per core.

### Drift from 2026-10-19/20 (flag: more than 2%)

| cell, metric | now | 2026-10-19/20 (*reused*, `neuron_final.md`) | delta | flag | likely cause |
|---|---:|---:|---:|:---:|---|
| native tp4, step | 1063.6 ms | 1063.1 ms | +0.05% | — | Output bit-identical (mp4 `9b914e81…`). Unlike Qwen-Image, the toolchain change did not change Wan's compiled DiT. |
| native tp4, compile cold | 600.3 s | 518.9 s | **+15.7%** | **yes** | See note 1. |
| native tp4, e2e warm | 128.5 s | 125.5 s | **+2.4%** | **yes** | Launch jitter, plus up to 2.7 s more vae launch. Rank 0's start-to-load time was 20.3, 23.3 and 22.0 s (`native_tp4/gen{1,2,3}`), against 21.7 s in the 2026-10-19 gen1 breakdown. The denoise is 53.26–53.29 s against 53.3 s. The vae launch took 23.2–24.9 s against 22.2 s. |
| native tp4, e2e cold | 542.7 s | 520.2 s | **+4.3%** | **yes** | See note 2. |
| native tp4, HBM alloc / nmon; vae core nmon | 13.48 (13.58 in compile) / 15.15; 18.40 GiB | 13.58 / 15.15; 18.51 GiB | 0 / 0; −0.6% | — | |
| native `--sp`, step | 1042.7 ms | 1043.7 ms | −0.1% | — | mp4 `4d7d84b2…`, the 2026-10-19 file |
| native `--sp`, e2e warm | 124.7 s (132.8, 116.6) | 125.2 s | −0.4% | — | gen1 read the DiT store partly from disk (17.4 s against 3.6 s); gen2 had a fast 12.1 s launch |
| native `--sp`, compile (page cache as left: 29% resident) | 955.7 s | 956.0 s | −0.03% | — | DiT load 653.3 s (cold read + store write), against 653.6 s |
| native `--sp`, HBM alloc / nmon | 13.48 / 14.70 GiB | 13.51 / 14.70 GiB | 0 | — | |
| AoT stock, step | 1150.8 ms | 1151.5 ms | −0.06% | — | bit-identical outputs |
| AoT stock, compile cold | 453.3 s | 482.0 s | **−6.0%** | **yes** | See note 3. |
| AoT stock, e2e warm | 131.5 s | 134.5 s | **−2.2%** | **yes** | See note 4. |
| AoT stock, e2e cold | 455.6 s | 461.8 s | −1.3% | — | |
| AoT two-core, step | 955.8 ms | 957.1 ms | −0.14% | — | bit-identical outputs |
| AoT two-core, compile cold | 445.1 s | 303.2 s | **+46.8%** | **yes** | See note 3. |
| AoT two-core, e2e warm | 120.0 s | 123.1 s | **−2.5%** | **yes** | See note 4. |
| AoT two-core, e2e cold | 447.5 s | 450.0 s | −0.6% | — | |

Notes on the drift rows:

1. **Native compile, +15.7%.** The DiT's "load checkpoint" phase, which includes the store write, is unchanged: 220.3 s against 220.2 s. The 2026-10-19 report gives no other split. In this run, the parts beyond the DiT load are:
   - torchrun took 13.7 s to start rank 0, and rank 0 took 26.9 s to reach its load (warm runs: 2.4–2.6 s and 20–24 s);
   - umT5 "load checkpoint" took 68.5 s, against 2.3 s warm;
   - the warm-up forwards took 70.1 s (DiT) and 11.3 s (umT5);
   - the vae launch took 169.7 s.

   *Inference:* the slow rank start is the effect the Qwen-Image section found: dropping the page cache before the pre-read also evicts the venv's libraries. And the 28 GB of store writes evicted part of the pre-read umT5: the checkpoint was 100% resident before the compile (`ckpt_resident_before`), yet umT5 read at 68.5 s. Without the 2026-10-19 split, neither can be confirmed as the cause of the difference.
2. **Native cold e2e, +4.3% (22.5 s).** The two load phases are the same as on 2026-10-19: DiT 217.0 s from the store (217.4 s then) and umT5 172.5 s (172.5 s). The extra time is outside the loads. torchrun took 13.8 s to start rank 0, against 2.4–2.6 s warm, and rank 0 took 24.8 s to reach its load. *Inference:* the same venv eviction by the page-cache drop as note 1. The 2026-10-19 launch split is not available to compare.
3. **AoT compile: stock −6.0%, two-core +46.8%.** Both runs started from an empty pre-sharded store and an empty neuronx-cc cache, so each wrote 39 GB of shards (119 s) and compiled every NEFF. The 2026-10-19/20 compiles have no breakdown, and their store and cache state is not recorded. Without the sharding, two-core's compile is 326 s, close to the 303.2 s. *Inference:* the 303.2 s run probably reused pre-sharded weights. Unconfirmed.
4. **AoT warm, −2.2% / −2.5%.** The step is unchanged.
   - The host decode now takes 47.0–47.5 s per warm run (`host_decode` span), against "49 s in the AoT venv" (*reused*, `neuron_feature_matrix.md` open issue 5). That accounts for about 1.5–2 s.
   - The 2026-10-17/20 AoT runs have no further breakdown, so the remaining ~1 s cannot be attributed.

### Where the warm e2e goes (gen1, seconds)

Times come from the probe events in `B/wan2_1/<cell>/gen1/summary.json`; `B/scripts/wbreakdown.py` prints them. The brackets hold the 2026-10-19 native gen1 breakdown (*reused*, `neuron_final.md`).

| part | native (`native_tp4/gen1`) | AoT two-core (`aot_tp4_twocore/gen1`) | AoT stock (`aot_tp4/gen1`) |
|---|---:|---:|---:|
| process wall | 126.0 [125.6] | 120.4 | 132.5 |
| generate launch / transformer stage, total | 101.5 [103.1] | 70.6 | 82.1 |
| — torchrun start to rank 0 start | 2.6 | — | — |
| — rank 0 start to load start (imports, runtime, process group) | 20.3 [21.7] | — | — |
| — load + warm-up | 22.2 [22.4]: DiT shards 3.6 from the store, warm-up 13.2; umT5 shards 2.3, warm-up 1.8 | load (umT5 + DiT, pre-sharded) 16.9 | 18.8 |
| — prompt encode (device umT5) | 1.0 [0.8] | (in "rest") | (in "rest") |
| — denoise, 50 steps | 53.3 [53.3] | 48.0 | 57.7 |
| — rest (native: latents save and teardown; AoT: imports, set-up, encode) | 2.1 | 5.7 | 5.6 |
| decode | vae launch 23.2 [22.2]: launch 2.4, imports 3.1, load 8.9, decode 7.3, mp4 0.2, exit 1.3 | host decode in the driver 47.0 (VAE load + decode; mp4 0.5) | 47.5 (mp4 0.5) |
| gaps between launches, driver start and exit | 1.3 | 2.8 | 2.9 |

- **Native's denoise is 5.3 s slower than two-core AoT's** (the step difference times 50, with CFG). Its launch costs 22.9 s before the load starts. Its device decode is 24 s faster than AoT's host decode.
- **AoT on its own default placement**, a compiled VAE stage instead of `--host-vae`, was not measured. It would change AoT's decode figure.

### Outputs

- **Every Wan run shows the prompt's content.** One `mid.png` (frame 4 of 9) was opened per cell: `native_tp4/cold`, `native_tp4_sp/gen1` and `aot_tp4/cold`. Each shows a red fox running through snow in a forest. `aot_tp4_twocore`'s `mid.png` is the same file as stock's (`671586bb…`).
  - Each mp4 has 9 frames, 480x832, with per-frame pixel std of at least 62 levels and a mean frame-to-frame difference of about 6 levels (`mp4` in each `summary.json`).
- **Within each cell, every run gave the same mp4 and latents.pt, bit for bit.**
- **Native and AoT show the same composition here,** unlike Qwen-Image. Their mp4s differ only slightly in the decoded statistics (mean level 110.8 vs 110.0).
  - The reason is in the code: Wan draws its seed-42 noise in fp32 on both backends. Native uses `TorchNeuronWanApplication._initial_latents` (`dtype=torch.float32`, drawn on rank 0 and broadcast); AoT uses `WanOrchestrator.__call__` (`prepare_latents(..., dtype=torch.float32)`). The Qwen-Image section measured that the fp32 draw is the same in torch 2.9 and 2.14; only the bf16 draw differs.
- **The native outputs are the 2026-10-19 goldens:** tp4 mp4 `9b914e81…` and latents.pt `3cafa271…`, and SP mp4 `4d7d84b2…`. So the toolchain change (drift note 1 of Qwen-Image) left Wan's numerics unchanged.
- **The AoT outputs equal 2026-10-20's:** mp4 `d2ecce24…` and latents `f3dd7d2e…`, for stock and two-core alike.

### Method differences (Wan2.1)

The method is the "Method" section below, with these differences:

- **Harness:** `B/scripts/wbench.py`, derived from `qbench.py`. It adds the shape `--height 480 --width 832 --num-frames 9`, `--steps 50 --guidance-scale 5.0` and `--output <run>/out.mp4`. It decodes each mp4 (`B/scripts/check_mp4.py`), saves the middle frame as `mid.png` and hashes the mp4, `mid.png` and latents.pt.
  - It also removes `DIFFLET_NEURON_WEIGHT_STORE`, `..._ALL` and `NEURON_RT_VIRTUAL_CORE_SIZE` from the environment, then adds the cell's `--env` values. Only the two-core cell has one: `NEURON_RT_VIRTUAL_CORE_SIZE=2`, recorded in each `run.log`'s `env_extra`.
- **Probe:** `B/probe/sitecustomize.py` was extended additively with Wan targets. The file as the Qwen-Image cells used it is kept as `B/probe/sitecustomize.qwen-as-used-2026-10-25.py`. The Wan targets:
  - `WanOrchestrator._scheduler_step` (a stamp after each step), `._denoise`, `._decode_latents` and `.encode_prompt`;
  - `TorchNeuronWanApplication.load`, with a read-only look at `phase_seconds`;
  - `_DeviceVaeDecoder.__call__` and `NeuronWanApplication.load` (AoT);
  - the CLI's `_write_frames`, `_decode_latents_host` and `_save_video`, and its `_stage_transformer` / `_stage_vae`.

  Before the runs, an import check in both venvs (`B/scripts/probe_selftest_wan.py`) confirmed that every target was wrapped.
- **Page-cache-warm compile for all three headline cells** (`native_tp4`, `aot_tp4`, `aot_tp4_twocore`): drop the page cache, then read the checkpoint twice. The checkpoint is the transformer, text_encoder and vae, 74.9 GiB.
  - The first read took 612 s, the second 28–29 s.
  - `mincore` showed 100% residency before each compile (`preread_rec`).
  - The `--sp` compile ran with the page cache as the previous run left it: 29% resident.
- **Cold e2e:** for the three headline cells, page cache dropped through passwordless sudo (`Cached` 0.15–0.22 GiB before).
- **One device job at a time:** `fuser /dev/neuron0` was idle before every run, and `device_released: true` in all 22 Wan `summary.json` files.
- **Disk:** the Wan roots stay on the host: `B/roots/wan_native_tp4` and `wan_native_tp4_sp` hold 27 GB each (the weight store), and the two AoT roots 39 GB each (the pre-sharded umT5 and DiT). 285 GB (decimal) were free after the last run (`df_free_gb_after`).

## Method

This section was written for the Qwen-Image cells; the Wan2.1 cells follow it with the differences listed in "Method differences (Wan2.1)". The method is `neuron_final.md`'s "Method", with these differences:

- **Observer probe.**
  - The campaign probe that report names (`scripts/probe/sitecustomize.py`) is not on this host. It is not in `/home/ubuntu/campaign`, in any git worktree, or in `git log --all -- '*sitecustomize*'`; this was checked for the HunyuanVideo stage, `hunyuan_video_neuron.md`.
  - The equivalent timestamp-only observer written then, `B/probe/sitecustomize.py`, was extended additively with Qwen-Image targets. The file as the HunyuanVideo cells used it is kept as `B/probe/sitecustomize.hv1-as-used-2026-10-25.py`.
  - Each wrapper calls the original with the same arguments and returns its result unchanged: no argument, tensor or control flow changes.
  - Qwen-Image targets:
    - `QwenImageOrchestrator._scheduler_step`: a stamp after each step, on the native ranks and in the AoT generate stage alike;
    - `._denoise`;
    - `TorchNeuronQwenImageApplication.load` (with a read-only look at its `phase_seconds`) and `.encode_prompt`;
    - `NeuronQwenImageApplication.load` (AoT);
    - the vae stage decoder's `load` and call;
    - the CLI's `_write_image`;
    - the CLI class's `_stage_text`, `_stage_generate` and `_stage_vae`.
  - At exit, each process writes its events and, on a neuron rank, `torch_neuronx.max_memory_allocated()`.
- **DiT step** = median / p10 / p90 of rank 0's inter-step deltas of the real loop, step 0 excluded (49 deltas per run). For AoT the deltas come from the generate stage process.
  - The table value is the median of the per-run medians of the timed runs; p10 / p90 are pooled over the timed runs.
- **Runs per cell.**
  - **Compile:** a cold `difflet compile` on a fresh root, `B/roots/qwen_<cell>`. For native it is also exported as `DIFFLET_COMPILE_CACHE`. For AoT, the harness also sets `DIFFLET_SHARED_WEIGHTS_DIR=<root>/_shared_weights` and `NEURON_COMPILE_CACHE_URL=<root>/neuron-compile-cache`, the same in every run of the cell.
    - Without those two settings, the AoT compile would have reused the Qwen-Image tp4 shards already in `~/.cache/difflet/_shared_weights/` and the entries in `/var/tmp/neuron-compile-cache`. That is what made the HunyuanVideo AoT compile a lower bound.
  - **Page-cache-warm compile (the two headline cells):** drop the page cache, read the checkpoint (transformer, text_encoder, vae: 53.7 GiB) twice, then compile.
    - The first read took 439 s, the second 23–26 s.
    - `mincore` showed 100% residency before the compile (`preread_rec` in each compile's `summary.json`).
    - The `--sp` compile ran with the page cache as the previous run left it: 94% of the checkpoint resident.
  - **Cold e2e (the two headline cells):** a generate with the page cache dropped first (`sync; echo 3 > /proc/sys/vm/drop_caches` through passwordless sudo; `Cached` 0.2 GiB before).
  - **Warm runs:** one discarded warm generate (gen0), then n timed warm generates: n=3 for the headline cells, n=2 for `--sp`.
- **e2e** = the process wall of `python -m difflet.cli.main generate` (harness clock; `/usr/bin/time -v` agrees to 0.01 s).
- **HBM** = `torch_neuronx.max_memory_allocated()` per rank (max over ranks, native only), plus neuron-monitor's per-core peak of `memory_used` (sum of its categories) from `nmon.jsonl`.
- **No exported timeout:** `DIFFLET_NEURON_PG_TIMEOUT_S` is unset in every run (the harness removes it).
- **One device job at a time.** Before each run, the harness waited until `fuser /dev/neuron0` showed no process. After each run, it checked that every process had released the device: `device_released: true` in all 16 `summary.json` files.

## Open issues

1. **Native launch jitter** (`neuron_final.md` open issue 2, unchanged). Rank 0's start-to-load time is bimodal, about 11–12 s or 19–24 s: 11.7, 21.5, 22.4 and 24.4 s in `native_tp4` gen0–3, and 19.1 and 11.9 s in `native_tp4_sp` gen1–2. At 79–82 s per run, that is the whole native vs AoT warm gap.
2. **Native's step is 1.8% slower than on 2026-10-19 and its output hash changed.** The suspected cause is the toolchain (neuronx-cc 2.28.4568 → 2.28.7631) and host change. It is not confirmed: the earlier venv is not on this host. An A/B with the two compilers on one host would settle it.
3. **The native vae launch costs 19 s for a 2.5 s decode** (gen2: launch 2.5, imports 3.1, load 9.5, exit 1.2). AoT's vae stage takes 13–14 s. A resident decoder or a lighter launch would close most of the remaining warm gap.
4. **Wan2.1: native trails two-core AoT by 11.3% per step.** Two-core AoT traces the NKI attention for both physical cores of a logical core (`attention_cte[2]`). The native per-block `torch.compile` path was not profiled for the same effect in this stage. Native's warm e2e also loses 20–23 s per run to launch time (open issue 1).
5. **Wan2.1 native cold e2e is still 21% behind AoT.** The fp32 umT5 is read from the checkpoint in 172.5 s on every cold launch, and the weight store does not cover it ("no gain"). The native compile also includes the 192–215 s store write and a 170 s vae launch (`neuron_final.md` open issue 1, unchanged).
6. **The AoT two-core change is not on any branch this host can reach.** Commit `5f62732` and `origin/feat/aot-bf16-allreduce` are absent. The cell was reproduced by exporting `NEURON_RT_VIRTUAL_CORE_SIZE=2`, and the outputs equal the 2026-10-20 two-core run's bit for bit. To make it the default, Wan's AoT CLI would set it for the transformer stage, as `_transformer_virtual_core_size` already does for ring CP.

## Reproduction

```bash
# native: the native venv's bin/ on PATH (neuronx-cc is spawned by name), this worktree on PYTHONPATH
R=<fresh root>
DIFFLET_COMPILE_CACHE=$R HF_HUB_OFFLINE=1 python -m difflet.cli.main compile --backend neuron \
  --model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 \
  --height 1024 --width 1024 --tp-degree 4 [--sp] --cache-dir $R
DIFFLET_COMPILE_CACHE=$R HF_HUB_OFFLINE=1 python -m difflet.cli.main generate --backend neuron \
  --model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 \
  --height 1024 --width 1024 --tp-degree 4 [--sp] --steps 50 --guidance-scale 4.0 --seed 42 \
  --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.png --cache-dir $R
# AoT: the NxD venv (/home/ubuntu/venvs/difflet-neuron), --backend trainium, no DIFFLET_COMPILE_CACHE;
#   DIFFLET_SHARED_WEIGHTS_DIR=$R/_shared_weights NEURON_COMPILE_CACHE_URL=$R/neuron-compile-cache for a
#   compile with nothing pre-sharded or cached; pass --steps 50 (the AoT Qwen-Image stage defaults to 4)

# Wan2.1, native (same set-up as above)
DIFFLET_COMPILE_CACHE=$R HF_HUB_OFFLINE=1 python -m difflet.cli.main compile --backend neuron \
  --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
  --height 480 --width 832 --num-frames 9 --tp-degree 4 [--sp] --cache-dir $R
DIFFLET_COMPILE_CACHE=$R HF_HUB_OFFLINE=1 python -m difflet.cli.main generate --backend neuron \
  --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
  --height 480 --width 832 --num-frames 9 --tp-degree 4 [--sp] --steps 50 --guidance-scale 5.0 --seed 42 \
  --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4 --cache-dir $R
# Wan2.1, AoT: the NxD venv, --backend trainium --tp-degree 4 --host-vae, the same shape and generate flags,
#   DIFFLET_SHARED_WEIGHTS_DIR / NEURON_COMPILE_CACHE_URL as for Qwen-Image; two-core: also export
#   NEURON_RT_VIRTUAL_CORE_SIZE=2 for the compile and every generate (fresh root: the AoT stage key does not
#   include it)
```

Harness: `B/scripts/qbench.py cell --cell <cell> --venv native|aot [--preread] --runs compile,cold,gen0,gen1,gen2[,gen3] -- <flags>`. It adds the model, the revision, the shape and `--cache-dir B/roots/qwen_<cell>`; for a generate it also adds the steps, guidance, seed, prompt, output and `--work-dir <run>/work --keep-work-dir`. `B/scripts/qbreakdown.py` prints the breakdowns and cell statistics. Wan2.1: `B/scripts/wbench.py` (same interface, plus `--env K=V`; roots `B/roots/wan_<cell>`) and `B/scripts/wbreakdown.py`.

## Toolchain

- **Native:** `/home/ubuntu/venvs/native-312`: Python 3.12.3, torch 2.14.0, torch-neuronx 2.14.3.0.1, neuronx-cc 2.28.7631.0.
- **AoT:** `/home/ubuntu/venvs/difflet-neuron`: torch 2.9.1, torch-neuronx 2.9.0.2.15.32035, neuronx-cc 2.26.6360.0.
- **Host runtime:** aws-neuronx-runtime-lib 2.35.30.0, collectives 2.35.43.0.
- **2026-10-19/20 (*reused*, `neuron_final.md`):** native Python 3.14.8, neuronx-cc 2.28.4568.0, torch-neuronx 2.14.3.0.0; runtime 2.35.27.0, collectives 2.35.40.0.
