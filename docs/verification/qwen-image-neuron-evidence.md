# Qwen-Image on the neuron backend: on-device evidence

Campaign: 2026-10-16, device launches 03:02-04:29 UTC. Host: `trn2.3xlarge`, 1 Neuron device, 4 logical
NeuronCores (logical config 2), 24 GiB HBM per logical core (96 GiB total), 12 vCPUs, 124 GB RAM.
Branch: `feat/qwen-image-neuron`.

This doc collects the evidence for the Qwen-Image port to the PyTorch-native neuron backend
(TorchNeuron, `torchrun`, one process per NeuronCore). There were three device steps:

* **V1:** the DiT at TP4 against a CPU fp32 reference.
* **V2:** the DiT under context parallelism.
* **V3:** end to end through the user-facing `difflet compile/generate --backend neuron`.

**V4** (this document) used no device. The curated logs, the per-rank `result*.json` files and
each step's `summary.json` are committed under `artifacts/verification-2026-10-16-qwen-neuron/`.
In this doc, `E/` stands for that directory. Every number below was copied from those files and
checked against them; a value marked *inference* is an explanation that no run measured directly.

Statuses: **PASS** (ran on device in this campaign and met its gate) · **LIMIT** (works with a
documented constraint) · **BLOCKED** (diagnosed toolchain issue) · **N/A** (excluded by design or
rejected in phase 1) · **NOT MEASURED** (not gated off, but this campaign did not run it).

## Summary

* **Outcome: PASS, no Difflet bug found.** All 4 ranks passed every gated cell in V1 (6 cells), V2
  (10 gated cells plus 1 LIMIT probe) and V3 (8 cells). No fix commits were needed. The bug ledger
  is empty.
* **DiT parity, TP4, all 60 layers, compiled.** Against a full 60-layer diffusers 0.38 fp32 CPU
  reference, cosine is 0.999889/0.999930 at t=1000/500. rel_l1 is 1.018x/0.891x the bf16 noise
  floor (a CPU bf16 control against the same fp32 reference). Eager mode is 0.904x/1.018x. Every CP
  mode at every depth stays within 1.044x/1.107x of the control.
* **Context parallelism, all three modes PASS:** gather_kv, ulysses and ring at tp2cp2 (4 and 60
  layers) and tp1cp4 (4 and 30 layers). Ring's padding path also passes at 1024x1008. Output
  sha256 is identical on every rank in every cell.
* **LIMIT:** 60 layers do not fit at tp1. That needs 38.05 GiB of weights per rank against 24 GiB
  per core, and the run hit a device OOM during the load. On 4 cores, 60-layer CP therefore means
  tp2cp2 (19.75 GiB/rank peak).
* **End to end (`difflet generate --backend neuron`), TP4, 1024x1024, 50 steps, warm cache:**
  * Process wall 82.4 s. Steps run at 352.4 ms median, so denoise takes 17.63 s; the host VAE
    decode takes 23.2 s.
  * 0 compilations and 0 fallbacks.
  * The image is coherent. The output is bit-identical across 5 runs, including one on a cache
    root compiled separately: latents `02613803…`, PNG `c643eefc…`.
* **Other end-to-end configurations:**
  * tp2cp2 Ulysses: 419.1 ms/step, wall 91.4 s.
  * TP4 eager: 1152 ms/step, wall 119.5 s.
  * A cold CLI compile costs 20.6 s of compile wall (51 NEFFs).
* **Fastest configuration on 4 cores:** TP4 compile. Its 60-layer DiT forward takes 348.6 ms,
  against 416.0 ms for Ulysses, 445.9 ms for gather_kv and 543.6 ms for ring, all at tp2cp2.
* **Not verified:**
  * The CLI with gather_kv or ring. The CLI ran only TP4 and tp2cp2 Ulysses.
  * Any CLI shape other than 1024x1024.
  * The device VAE, and serving.
  * Rejected in phase 1, so N/A: CFG-parallel, SP, DP, TeaCache, multi-shape buckets and batch >1.
* **Semantic difference from diffusers:** the text mask is dropped, as on Trainium and TPU. Prompts
  are zero-padded to 1024 rows and attended unmasked. There is also no true-CFG negative pass.

## Scope

**Verified on device:**

* The DiT (`NeuronQwenImageTransformerApplication`) at TP4 in compile mode and in eager mode. It
  ran at 4 and 60 layers against a first-N-blocks diffusers fp32 CPU reference built from the real
  checkpoint (V1).
* The DiT under CP gather_kv / ulysses / ring:
  * at tp2cp2 with 4 and 60 layers;
  * at tp1cp4 with 4 and 30 layers;
  * at 1024x1008, which exercises the ring padding path (V2).
* The tp1 HBM limit at 60 layers (V2).
* The CLI `difflet compile` and `difflet generate --backend neuron`: TP4 compile, TP4 eager and
  tp2cp2 Ulysses, at 50 and 30 steps, on a warm and a cold cache root (V3).
* Teacher-forced per-step DiT parity on the real CLI trajectory (V3).
* Host prompt-embedding parity against diffusers `_get_qwen_prompt_embeds` (V3, CPU only).

**Not verified:**

* The CLI with gather_kv or ring.
* The CLI at tp1cp4.
* The CLI at any shape other than 1024x1024.
* tp2 or tp1 without CP (cp=1 on fewer cores).
* The device VAE (`enable_vae_decoder=True`).
* Resident or serving use.
* A page-cache-cold run (no cache drop was done).
* Throttling and thermal behaviour.
* Any phase-1 rejection on device. Those rejections are code-gated and covered by unit tests only,
  and those tests were not run in this campaign.

## Environment

| Item | Value | Source |
|---|---|---|
| Instance | trn2.3xlarge, 4 logical NeuronCores (config 2), 24 GiB per core (`total_hbm` 25,769,803,776 B in the V2 OOM dump) | `E/v1/summary.json` `host`; `E/v2/summary.json` `derived.l60_tp1cp4_hbm_probe` |
| Kernel | 6.17.0-1019-aws | `uname -r` (V4) |
| Python | 3.14.8 (`/home/ubuntu/workspace/native_venv`) | venv interpreter (V4) |
| torch | 2.14.0+cpu | venv `importlib.metadata` (V4); `E/v1/summary.json` references `versions` |
| torch-neuronx | 2.14.3.0.0+unknown.dev | venv metadata (V4) |
| neuronx-cc | 2.28.4568.0+0163fa9e | venv metadata (V4) |
| diffusers / transformers / safetensors / huggingface-hub | 0.38.0 / 4.57.6 / 0.8.0 / 0.36.2 | venv metadata (V4) |
| Neuron system packages | aws-neuronx-runtime-lib 2.35.27.0, aws-neuronx-collectives 2.35.40.0, aws-neuronx-dkms 2.31.6.0, aws-neuronx-tools 2.33.32.0 | `dpkg -l` (V4) |
| Checkpoint | `Qwen/Qwen-Image` @ `75e0b4be04f60ec59a75f475837eced720f823b6`: DiT 60 layers, 24 heads x 128, about 41 GB bf16; Qwen2.5-VL 7B text encoder; VAE | task brief; V1 reference load (20.430B params) |
| Code under test | `difflet/` at `9aa4e3f`. It is byte-identical through `1c16390`: `git diff --stat 9aa4e3f 1c16390 -- difflet/` is empty | `git` (V4) |
| HEAD at launch | V1 `9aa4e3f`; V2 `ea6fe1b`/`eca572e`; V3 `317060e` (printed in every launch line of the logs) | `E/v*/…log` |
| Commits of this campaign | `98a081c`, `ea6fe1b` (V1); `eca572e`, `317060e` (V2); `3d2641c`, `1c16390` (V3); this doc (V4). None pushed. | `git log` |
| Compile cache | main root `/home/ubuntu/.cache/difflet-neuron-qwen`; scratch roots `…-qwen-l4` (V1 4-layer) and `…-qwen-v3cold` (V3 cold) | launch lines |

Launch environment for every device run (`E/v1/scripts/dev.sh`, `E/v3/scripts/cli.sh`):
`PATH=<venv>/bin:$PATH` (neuronx-cc is spawned from there), `HF_HUB_OFFLINE=1`,
`DIFFLET_NO_JEMALLOC=1`, `DIFFLET_NEURON_PG_TIMEOUT_S=7200` and `DIFFLET_COMPILE_CACHE=<root>`.
The preflight runs also set `OMP_NUM_THREADS=3`, `DIFFLET_DISABLE_PREWARM=1` and the
`TORCH_NEURONX_*` cache dirs; the CLI runs set `PYTHONSAFEPATH=1`.

## How the numbers are defined

* **Parity.** Metrics come from `tests/numerical/neuron_parity.py`. The *control* is diffusers run
  in bf16 on the CPU, compared against the same fp32 CPU reference: that is the bf16 noise floor.
  "1.018x the control" means device rel_l1 divided by control rel_l1. The gate is the shared
  noise-floor gate. Each pair of values is for t=1000/500, which is DiT timestep 1.0/0.5.
* **Reference inputs.** V1 and V2 use real Qwen2.5-VL bf16 prompt embeddings for the prompt *"A
  cozy coffee shop storefront on a rainy evening, a chalkboard sign by the door reads "Fresh
  Bread", warm light through the windows, photorealistic"*. That gives 36 valid rows, zero-padded
  to 1024. Other inputs: `encoder_hidden_states_mask=None`, `img_shapes=[[(1,64,64)]]`.
* **Forward (ms).** Preflight `bench_median_ms_per_rank`, on device-resident inputs, rank 0.
* **Step (ms).** CLI: the inter-step delta of the real loop. Step 0 is excluded, n=49, rank 0.
  This is the `[v3probe]` observer, and a probe-free run was bit-identical.
* **Warm-up.** The rank-max `warmup forward` phase. **Cold compile wall** is the cold warm-up minus
  the warm warm-up on the same graphs.
* **HBM.** `torch_neuronx.max_memory_allocated` per rank. The sysfs `device_mem` peak is not a
  per-process figure: it read about 16.2 GB on every even physical core in both the 4- and 60-layer
  runs.
* **Process wall.** The CLI process start to exit, which includes torchrun spawn, the ranks and the
  driver's host VAE decode.

## Phase plan

| Step | Feature group | Runner | Status |
|---|---|---|---|
| V1 | DiT TP4, eager and compile, 4 and 60 layers, cold and warm cache | `tests/manual/check_neuron_qwen_preflight.py` (torchrun x4) + `tests/manual/qwen_image_dit_reference.py` (CPU) | **PASS** (6/6) |
| V2 | DiT CP gather_kv / ulysses / ring at tp2cp2 and tp1cp4, 4/30/60 layers, ring padding, tp1 HBM probe | same preflight with `--cp-degree/--cp-mode` | **PASS** (10/10 gated), 1 **LIMIT** |
| V3 | `difflet compile/generate --backend neuron` end to end, text-embedding parity, teacher-forced trajectory parity | `python -m difflet.cli.main`, `tests/manual/check_qwen_image_prompt_embeds.py`, observer probe | **PASS** (8/8) |
| V4 | This evidence doc | no device | done |

## V1: DiT at TP4 vs CPU fp32 (1024x1024, text 1024)

Evidence: `E/v1/`; aggregate in `E/v1/summary.json`. The launcher is `E/v1/scripts/dev.sh
<cache_root> <preflight args>`, a 4-rank torchrun of `tests/manual/check_neuron_qwen_preflight.py`
with `--model-dir <snapshot>`. The reference script is `E/v1/scripts/ref.sh <layers> <out.pt>`.
Both ran with the work dir `/home/ubuntu/qwen-v1-work`, where the `.pt` inputs and outputs live
(not committed).

| Cell | Outcome | Key metrics | Evidence | Reproduction |
|---|---|---|---|---|
| V1-ref: CPU references (diffusers 0.38, first N blocks of the real checkpoint) | **PASS** | 4 layers: 1.400B params, missing 0, unexpected 1792 (blocks 4-59, expected); fp32 forward 6.7 s; control rel_l1 1.230e-2/4.525e-3. 60 layers: 20.430B params, missing 0, unexpected 0; load 258.9 s; fp32 forward 100 s per timestep; control rel_l1 2.457e-2/1.641e-2, cosine 0.99986449/0.99990324. Prompt embeddings: absmean 2.89, absmax 170, sha256 `b5260c2e…` | `E/v1/ref-l4.log`, `E/v1/ref-l60.log`, `E/v1/summary.json` `references`, `prompt_embeds` | `bash E/v1/scripts/ref.sh 4 /home/ubuntu/qwen-v1-work/ref-l4.pt`; `… ref.sh 60 …/ref-l60.pt` |
| V1a: 4 layers TP4 eager | **PASS** | cosine 0.999968/0.999989; rel_l1 1.249e-2/4.668e-3 = 1.015x/1.031x the control; rel_l2 1.381e-2/4.742e-3. Forward median 81.0 ms (p10 79.8, p90 82.2, n=10). Warm-up 33.6 s (104 eager-op NEFFs, fresh scratch root). HBM 1.40 GiB/rank. Fallbacks [], unwarmed [], 0 compiles in the forwards | `E/v1/a-l4-eager.log`, `E/v1/a-l4-eager/result*.json`; first try without PATH: `E/v1/a-l4-eager.try1-nopath.log` | `bash E/v1/scripts/dev.sh /home/ubuntu/.cache/difflet-neuron-qwen-l4 --reference /home/ubuntu/qwen-v1-work/ref-l4.pt --num-layers 4 --exec-mode eager --save /home/ubuntu/qwen-v1-work/a-l4-eager/out.pt --out-json /home/ubuntu/qwen-v1-work/a-l4-eager/result.json` |
| V1b: 4 layers TP4 compile (cold block NEFF) | **PASS** | vs fp32: cosine 0.999967/0.999990, rel_l1 1.041x/0.964x the control. Vs eager: cosine 0.999991/0.999990, rel_l1 3.5e-3/3.8e-3 (0.285x/0.830x the control). One Dynamo graph, 0 breaks, 1 block NEFF `a0654ef7f432` (2,580,116 B, compiled by rank 1 in 13.9 s). Warm-up 24.1 s. Forward 28.3 ms (n=10), 2.9x faster than eager. HBM 0.96 GiB/rank | `E/v1/b-l4-compile-cold.log`, `E/v1/b-l4-compile-cold/result*.json` | `… dev.sh /home/ubuntu/.cache/difflet-neuron-qwen-l4 --reference …/ref-l4.pt --num-layers 4 --exec-mode compile --expect cold --compare eager=…/a-l4-eager/out.pt --save … --out-json …` |
| V1c-cold: 60 layers TP4 compile, empty main cache root | **PASS** | cosine 0.999889/0.999930; rel_l1 2.502e-2/1.462e-2 = 1.018x/0.891x the control; rel_l2 2.519e-2/1.479e-2. Load checkpoint 53.4 s, prepare_runtime 20.1 s, warm-up 29.3 s, launch wall 123.9 s. 51 new .neff; one graph for 60 blocks, same block key `a0654ef7f432` (2,582,059 B, rank 2, 16.5 s). CompilationTime sum r0 157.1 / r1 1.1 / r2 33.6 / r3 3.3 s (a sum over concurrent compiles, not a wall). Forward 349.0 ms (p10 348.2, p90 349.5, n=10). HBM 10.12-10.15 GiB/rank (weights 9.76 GiB) | `E/v1/c-l60-compile-cold.log`, `E/v1/c-l60-compile-cold/result*.json` | `… dev.sh /home/ubuntu/.cache/difflet-neuron-qwen --reference …/ref-l60.pt --num-layers 60 --exec-mode compile --expect cold --bench-iters 5 --bench-rounds 2 --save … --out-json …` |
| V1c-warm: 60 layers TP4 compile, second launch | **PASS** | 0 compilations, 0 new .neff on every rank. Outputs `torch.equal` to the cold launch. Warm-up 8.1 s, so the cold compile wall is 21.2 s. Load checkpoint 3.4 s, prepare_runtime 22.0 s, launch wall 54.3 s. Forward 348.6 ms (p10 347.9, p90 350.3, n=30). HBM 10.15 GiB/rank | `E/v1/c-l60-compile-warm.log`, `E/v1/c-l60-compile-warm/result*.json`, `E/v1/summary.json` `derived.l60_cold_compile_wall_s` | `… dev.sh /home/ubuntu/.cache/difflet-neuron-qwen --reference …/ref-l60.pt --num-layers 60 --exec-mode compile --expect warm --equal-to …/c-l60-compile-cold/out.pt --bench-iters 10 --bench-rounds 3 --save … --out-json …` |
| V1d: 60 layers TP4 eager | **PASS** | vs fp32: cosine 0.999911/0.999905, rel_l1 0.904x/1.018x the control. Vs compile: cosine 0.999980/0.999967, rel_l1 0.258x/0.477x the control. Forward 1155.3 ms (n=3); compile is 3.31x faster. Warm-up 23.2 s (54 new eager-op NEFFs). HBM 10.14-10.48 GiB/rank. No compiled block | `E/v1/d-l60-eager.log`, `E/v1/d-l60-eager/result*.json` | `… dev.sh /home/ubuntu/.cache/difflet-neuron-qwen --reference …/ref-l60.pt --num-layers 60 --exec-mode eager --compare compile=…/c-l60-compile-cold/out.pt --bench-warmup 1 --bench-iters 3 --bench-rounds 1 --save … --out-json …` |

Timing notes:

* The 53.4 s first 60-layer load, against 3.4 s later, is an *inference*: the 82 GB fp32 CPU
  reference that ran just before pushed shards out of the page cache.
* The 0.67-1.5 s first forwards on ranks 1-3 are also an *inference*: those ranks waited in the
  first collective for rank 0, which finished warm-up last.

## V2: DiT context parallelism (1024x1024, text 1024, compile mode)

Evidence: `E/v2/`; aggregate in `E/v2/summary.json`. The launcher is `E/v2/scripts/cell.sh <name>
<cp> <mode> <layers> [extra args]`. It wraps `dev.sh` on the main cache root with `--exec-mode
compile --cp-degree <cp> --cp-mode <mode>` and picks the reference by depth. Its work dir is
`/home/ubuntu/qwen-v2-work`, and a later `--reference`/`--width` overrides the default. TP is set
to 4/cp, so these runs are tp2cp2 or tp1cp4.

Every gated cell had `ranks_identical` true, meaning the same output sha256 on all 4 ranks. Every
gated cell also had fallbacks [], 0 compiles in the forwards and one Dynamo graph.

| Cell | Outcome | Key metrics | Evidence | Reproduction |
|---|---|---|---|---|
| V2a: 4L tp2cp2 gather_kv | **PASS** | cosine 0.999967/0.999991; rel_l1 1.039x/0.958x the control. cp_scatter: 2048 image tokens per rank. Forward 34.7 ms warm, 35.1 ms cold. Cold warm-up 27.1 s with 1 block NEFF (`faa1a546aa0a`, rank 0, 16.7 s); warm 9.2 s. HBM 1.51 GiB | `E/v2/l4-tp2cp2-gather_kv.log`, `E/v2/l4-tp2cp2-gather_kv/`; cold first launch, run before the cp_scatter hook existed: `E/v2/l4-tp2cp2-gather_kv.try1-nohook.log` | `bash E/v2/scripts/cell.sh l4-tp2cp2-gather_kv 2 gather_kv 4` |
| V2b: 4L tp2cp2 ulysses | **PASS** | cosine 0.999967/0.999991; 1.039x/0.959x. Forward 32.9 ms. Cold warm-up 28.1 s with 2 block NEFFs, one per CP coordinate (`702b26e1e07f`, `9b9c0cfaa045`, 19.6/19.8 s). HBM 1.51 GiB | `E/v2/l4-tp2cp2-ulysses.log`, `E/v2/l4-tp2cp2-ulysses/` | `cell.sh l4-tp2cp2-ulysses 2 ulysses 4` |
| V2c: 4L tp2cp2 ring | **PASS** | cosine 0.999967/0.999991; 1.039x/0.957x. Forward 40.9 ms. Cold warm-up 32.0 s with 2 block NEFFs (`40b84c825c2d`, `206b7f20df5e`, 23.8/23.7 s). Ring query 1024+2048 = 3072, no padding. HBM 1.51 GiB | `E/v2/l4-tp2cp2-ring.log`, `E/v2/l4-tp2cp2-ring/` | `cell.sh l4-tp2cp2-ring 2 ring 4` |
| V2d: 4L tp1cp4 ulysses | **PASS** | cosine 0.999968/0.999991; 1.033x/0.910x. 1024 image tokens per rank; 6 heads per rank after the all-to-all. Forward 31.7 ms. Cold warm-up 42.0 s with 4 block NEFFs (32.6-33.0 s each). HBM 2.73 GiB | `E/v2/l4-tp1cp4-ulysses.log`, `E/v2/l4-tp1cp4-ulysses/` | `cell.sh l4-tp1cp4-ulysses 4 ulysses 4` |
| V2e: 4L tp1cp4 ring | **PASS** | cosine 0.999968/0.999992; 1.033x/0.910x. Forward 60.1 ms (1.89x Ulysses). Cold warm-up 59.4 s with 4 block NEFFs (44.6-53.1 s each). Ring query 2048, no padding. HBM 2.73 GiB | `E/v2/l4-tp1cp4-ring.log`, `E/v2/l4-tp1cp4-ring/` | `cell.sh l4-tp1cp4-ring 4 ring 4` |
| V2f: 4L tp1cp4 gather_kv | **PASS** | cosine 0.999968/0.999991; 1.034x/0.911x. Forward 40.6 ms. Cold warm-up 35.5 s with 1 block NEFF (`242efaca9490`, 23.6 s; the graph does not depend on the rank). HBM 2.73 GiB | `E/v2/l4-tp1cp4-gather_kv.log`, `E/v2/l4-tp1cp4-gather_kv/` | `cell.sh l4-tp1cp4-gather_kv 4 gather_kv 4` |
| V2g: 60L tp2cp2 ulysses vs the 60-layer fp32 reference | **PASS** | weights 19.50 GiB/rank, peak 19.75 GiB. cosine 0.999888/0.999933; rel_l1 2.480e-2/1.435e-2 = 1.009x/0.875x. Forward 416.2 ms (n=10). Warm relaunch: 0 compilations, `torch.equal` to the first launch, 416.0 ms (n=30, p10 415.3, p90 416.8). Vs V1 TP4: cosine 0.999989/0.999990, rel_l1 0.168x/0.228x the control. Block NEFF reused from 4 layers (same key `702b26e1e07f`). Load 6.6 s, warm-up 8.5 s, launch wall 50.0 s | `E/v2/l60-tp2cp2-ulysses.log`, `E/v2/l60-tp2cp2-ulysses-warm.log` and dirs | `cell.sh l60-tp2cp2-ulysses 2 ulysses 60 --bench-iters 5 --bench-rounds 2`; then `cell.sh l60-tp2cp2-ulysses-warm 2 ulysses 60 --bench-iters 10 --bench-rounds 3 --expect warm --equal-to /home/ubuntu/qwen-v2-work/l60-tp2cp2-ulysses/out.pt --compare tp4=/home/ubuntu/qwen-v1-work/c-l60-compile-cold/out.pt` |
| V2h: 60L tp2cp2 gather_kv and ring | **PASS** | gather_kv: cosine 0.999871/0.999929, 1.044x/0.888x, forward 445.9 ms (1.07x Ulysses). Ring: cosine 0.999890/0.999933, 1.007x/0.872x, forward 543.6 ms (1.31x Ulysses). 19.75 GiB/rank, 0 compiles. Vs V1 TP4: gather_kv 0.145x/0.226x, ring 0.154x/0.231x the control. Vs Ulysses: at most 0.187x | `E/v2/l60-tp2cp2-gather_kv.log`, `E/v2/l60-tp2cp2-ring.log` and dirs | `cell.sh l60-tp2cp2-{gather_kv,ring} 2 {mode} 60 --bench-iters 5 --bench-rounds 2 --compare tp4=/home/ubuntu/qwen-v1-work/c-l60-compile-cold/out.pt --compare ulysses=/home/ubuntu/qwen-v2-work/l60-tp2cp2-ulysses/out.pt` |
| V2i: 30L tp1cp4, all 3 modes, vs a 30-layer fp32 reference made in V2 | **PASS** | Reference: 10.235B params; control cosine 0.99941163/0.99997140, rel_l1 4.013e-2/1.671e-2. Ulysses: cosine 0.999389/0.999974, 1.016x/1.105x, 217.8 ms. gather_kv: 1.013x/1.107x, 283.8 ms. Ring: 1.012x/1.101x, 430.3 ms. Weights 19.07 GiB, peak 19.24 GiB/rank, 0 compiles | `E/v2/ref-l30.log`, `E/v2/l30-tp1cp4-*.log` and dirs | `bash E/v1/scripts/ref.sh 30 /home/ubuntu/qwen-v2-work/ref-l30.pt`; `cell.sh l30-tp1cp4-{mode} 4 {mode} 30 --bench-iters 5 --bench-rounds 2 [--compare ulysses=…]` |
| V2j: ring padding at 1024x1008 (4032 image tokens), 4L, vs a new 1024x1008 reference | **PASS** | Ring tp2cp2: query 1024+2016 = 3040, padded to 3072. cosine 0.999968/0.999991, 1.014x/0.963x, 50.5 ms. Ring tp1cp4: query 2032, padded to 2048. 1.031x/0.911x, 74.4 ms. Ulysses tp2cp2 control: 1.016x/0.963x, 34.0 ms; vs ring rel_l1 0.120x/0.308x the control. At this shape ring is 1.23x slower than at 1024x1024 and Ulysses 1.03x slower | `E/v2/ref-l4-1024x1008.log`, `E/v2/l4w1008-*.log` and dirs | `ref.sh 4 /home/ubuntu/qwen-v2-work/ref-l4-1024x1008.pt --width 1008`; `cell.sh l4w1008-tp2cp2-ring 2 ring 4 --reference /home/ubuntu/qwen-v2-work/ref-l4-1024x1008.pt --width 1008` (likewise tp1cp4 ring and tp2cp2 ulysses `--compare ring=…`) |
| V2k: 60L tp1cp4 HBM probe | **LIMIT** | tp1 needs 38.05 GiB/rank of weights against 24 GiB per core, and the device OOMed during the weight load. The first runtime OOM came 23 s after launch (a 113,246,208 B request); the allocator stopped at 25,643,949,568 of 25,769,803,776 B. Ranks 2 and 3 raised `NRT EXECUTION FAILED: lazy::AllocBind: NRT_RESOURCE; no pending ops on stream to wait for (cannot defer), Failed to allocate resource` at `load_checkpoint_into` `target.copy_`; torchrun SIGTERMed ranks 0 and 1. The error surfaced only after about 16 min of allocator retries (2812 OOM warnings; launch wall 996.9 s, rc=1). Afterwards the device was healthy: a warm 4L tp2cp2 Ulysses rerun passed, `torch.equal` to V2b, with 0 compiles | `E/v2/l60-tp1cp4-ulysses-probe.log` (curated), `E/v2/l4-tp2cp2-ulysses-postprobe.log`, `E/v2/summary.json` `derived.l60_tp1cp4_hbm_probe` | `cell.sh l60-tp1cp4-ulysses-probe 4 ulysses 60 --bench-iters 0`; then `cell.sh l4-tp2cp2-ulysses-postprobe 2 ulysses 4 --expect warm --equal-to /home/ubuntu/qwen-v2-work/l4-tp2cp2-ulysses/out.pt` |

Shape rules, held in every cell (`E/v2/summary.json` `derived.shape_rules`):

* Image tokens per rank = image_seq_len / cp; text is replicated.
* The ring query is text + image/cp, padded by the flash-lse op to a multiple of 128.
* Ulysses heads per rank = 24 / tp / cp.

The block NEFF key does not depend on depth: it is the same for 4, 30 and 60 layers of a topology.
So the 30- and 60-layer CP runs were warm, and no cold 60-layer CP compile was measured.

## V3: end to end through `difflet compile/generate --backend neuron`

Evidence: `E/v3/`; aggregate in `E/v3/summary.json`. Wrapper scripts:

* `E/v3/scripts/cli.sh <run_dir> <probe 0|1> <subcommand + args>` runs `python -m
  difflet.cli.main` with the launch environment above and counts NEFF and HLO files before and
  after.
* `E/v3/scripts/gen.sh <run_dir> <probe> <steps> <tp> <cp> <cp_mode>` runs `generate --backend
  neuron --exec-mode compile --model-id Qwen/Qwen-Image --revision
  75e0b4be04f60ec59a75f475837eced720f823b6 --height 1024 --width 1024 --seed 42 --prompt "A red fox
  sitting in a snowy forest at sunrise, highly detailed photograph" --cache-dir
  /home/ubuntu/.cache/difflet-neuron-qwen --work-dir <run>/work --keep-work-dir`.

The probe (`E/v3/scripts/probe`, a sitecustomize enabled by `DIFFLET_V3_PROBE=1`) only observes.
Generate 3 ran without it and gave the same PNG and latents.

| Cell | Outcome | Key metrics | Evidence | Reproduction |
|---|---|---|---|---|
| V3a: `difflet compile`, TP4, main cache (warm from V1) | **PASS** | Process wall 40.9 s. CLI `load+warmup` 34.2 s (includes prepare_runtime). App load 15.0 s: checkpoint 3.76 s, warm-up 8.16 s, text encoder load 1.2 s. TotalCompilations 0 on every rank (PersistentHits 46/51/51/51), 0 new .neff and HLO files. 60 compiled blocks; fallbacks [], unwarmed []. HBM 10.12 GiB/rank | `E/v3/tp4-compile.log`, `E/v3/tp4-compile.result.json`; offline failure without `--revision`: `E/v3/tp4-compile.try1-norevision.log` | `bash E/v3/scripts/cli.sh <run> 1 compile --backend neuron --exec-mode compile --model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 --tp-degree 4 --cp-degree 1 --height 1024 --width 1024 --cache-dir /home/ubuntu/.cache/difflet-neuron-qwen` |
| V3b: `difflet generate`, TP4, 50 steps, seed 42 (first generate after compile) | **PASS** | Process wall 179.6 s. Rank-0 prompt encode 98.4 s (19 valid rows); denoise 17.62 s; step median 352.1 ms (p10 351.4, p90 354.0, n=49); forward 351.2 ms; host VAE decode 23.9 s (fp32, 6 threads). Host VmHWM: rank 0 13.66 GiB, other ranks 1.37-1.57 GiB, driver 9.27 GiB. HBM 10.12-10.15 GiB/rank. 0 compilations, 0 new .neff; fallbacks [], unwarmed []. Latents sha256 `02613803…` on all 4 ranks, finite, std 0.427. The PNG (`c643eefc…`) is a coherent photo: a red fox in deep snow facing the camera, with sharp fur and amber eyes, snowy trunks and low sun; no artifacts | `E/v3/tp4-s50-gen1.log`, `E/v3/tp4-s50-gen1.result.json`, `E/v3/png/tp4-s50-seed42.png` | `bash E/v3/scripts/gen.sh <run> 1 50 4 1 gather_kv` |
| V3c: second generate (warm) and a third without the probe | **PASS** | Generate 2: wall 82.4 s. Start to ranks loaded 37.6 s (22.9 s of spawn, imports and prepare_runtime, then app load 14.7 s). Encode 0.57 s, denoise 17.63 s, step 352.4 ms (p90 353.5), ranks done to decode 2.7 s, host decode 23.2 s. Rank-0 VmHWM 13.67 GiB. 0 compilations. Bit-identical to generate 1. Generate 3 without the probe: wall 81.4 s, same PNG and latents | `E/v3/tp4-s50-gen2.log`, `E/v3/tp4-s50-gen2.result.json`, `E/v3/tp4-s50-gen3-noprobe.log`, `E/v3/summary.json` `timelines` | `gen.sh <run> 1 50 4 1 gather_kv`; `gen.sh <run> 0 50 4 1 gather_kv` |
| V3d: cold CLI compile on an empty root, then generate on it | **PASS** | Compile wall 67.2 s. Warm-up 28.73 s rank-max against 8.16 s warm, so the cold compile wall is 20.6 s. 51 new .neff, 152 HLO files; TotalCompilations r0 36 / r1 11 / r2 1 / r3 3. The generate on that root: wall 81.5 s, 0 compilations, same PNG and latents as the main cache | `E/v3/cold-compile.log`, `E/v3/cold-gen.log`, `E/v3/cold-compile.result.json`, `E/v3/cold-gen.result.json`, `E/v3/summary.json` `derived` | `bash E/v3/scripts/cold-batch.sh` (`cli-cold.sh`/`gen-cold.sh` on `/home/ubuntu/.cache/difflet-neuron-qwen-v3cold`) |
| V3e: TP4, 30 steps | **PASS** | Wall 75.2 s, denoise 10.57 s, step 352.1 ms, decode 23.0 s, 0 compiles, ranks identical (latents `61e0eac9…`). Coherent: the same scene and fox, tail curled forward. Vs 50 steps: PSNR 22.4 dB, mean abs diff 8.35/255 | `E/v3/tp4-s30.log`, `E/v3/tp4-s30.result.json`, `E/v3/png/tp4-s30-seed42.png`, `E/v3/compare.json` | `gen.sh <run> 1 30 4 1 gather_kv` |
| V3f: tp2cp2 Ulysses, 50 steps | **PASS** | Wall 91.4 s, CLI load+warmup 39.3 s, denoise 20.96 s, step 419.1 ms (p90 420.9). HBM 19.72-19.75 GiB/rank. 0 compiles (V2 left the NEFFs warm), fallbacks [], ranks identical (`a2b0d982…`). Coherent: the same composition with small differences in the hind leg and tail. Vs TP4: PSNR 28.8 dB, mean abs diff 2.76/255, latents cosine 0.99353, rel_l1 4.78%. Vs TP4 eager: PSNR 28.5 dB | `E/v3/tp2cp2-ulysses-s50.log`, `E/v3/tp2cp2-ulysses-s50.result.json`, `E/v3/png/tp2cp2-ulysses-s50-seed42.png`, `E/v3/compare.json` | `gen.sh <run> 1 50 2 2 ulysses` |
| V3g: TP4 eager control + teacher-forced per-step DiT parity on the real CLI trajectory | **PASS** | TP4 eager, 50 steps: wall 119.5 s, step 1152 ms, image coherent. Vs TP4 compile: PSNR 45.1 dB, latents rel_l1 1.04%. Teacher-forced, with the TP4-compile trajectory's inputs fed to each config: TP4 compile replayed into itself is `torch.equal` at all 50 steps. Noise-prediction rel_l1 vs TP4 compile: tp2cp2 Ulysses 0.37-1.70% (mean 0.47%); TP4 eager 0.53-3.11% (mean 0.86%). The CP/eager ratio is 0.42-0.77 at every step. Free-running latents vs TP4 compile, CP vs eager: 0.054% vs 0.071% at step 10; 1.03% vs 0.38% at step 30; 4.7% vs 1.0% final. *Inference:* this is chaotic amplification of bf16-level per-step differences, not a CP defect | `E/v3/tp4-eager-s50.log`, `E/v3/tp4-eager-s50.result.json`, `E/v3/png/tp4-eager-s50-seed42.png`, `E/v3/diag/*.log`, `E/v3/diag/traj_compare.json`, `E/v3/summary.json` `teacher_forced_and_free_run` | `bash E/v3/scripts/gen-eager.sh <run> 1 50 4 1 gather_kv`; `bash E/v3/scripts/diag-batch.sh`; `python E/v3/scripts/traj_cmp.py` |
| V3h: text-embedding parity (CPU): `encode_prompt` vs diffusers `_get_qwen_prompt_embeds` | **PASS** | The templates are identical; both give 19 valid rows. Ours is `torch.equal` to diffusers bf16 (max abs 0), and the padding rows are exactly zero. The embeddings rank 0 broadcast on device are `torch.equal` to both. The bf16-vs-fp32 floor is cosine 0.99988 and rel_l1 1.48%, the same for both implementations. Encode time: ours 0.51 s, diffusers 7.8 s | `E/v3/embeds_parity.json`, `E/v3/embeds_parity.log` | `bash E/v3/scripts/embeds.sh` (`tests/manual/check_qwen_image_prompt_embeds.py --fp32 --device-embeds <rank-0 dump>`) |

Golden output for regressions: seed 42, the fox prompt, TP4 and 50 steps give these, reproduced 5
times, including on the separately compiled V3d root:

* latents sha256 `02613803dae28a5172db5b43597f6a6c2d37ca4934799d20df11f2c7bd1decdb`;
* PNG sha256 `c643eefc7c2820dfb2d299b4935e7ef8b0582888561761b3823e7c4222766ba9`.

**Transcription note.** The V3 step report rounded two V3g ranges differently from its summary
file. It gave TP4 eager per-step rel_l1 as 0.58-3.11% and the CP/eager ratio as 0.43-0.69. This doc
uses the values in `E/v3/summary.json` `teacher_forced_and_free_run`: minimum 0.5256% (0.578% is
step 0, not the minimum) and ratio 0.419-0.769. The conclusion does not change: CP is closer to TP4
compile than eager is at every step.

## Support matrix: Qwen-Image on the neuron backend (1024x1024, text 1024, bf16, 4 NeuronCores)

| Feature | DiT preflight (V1/V2) | CLI end to end (V3) | Status | Measurement / reason |
|---|---|---|---|---|
| TP4, compile (`--exec-mode compile`) | PASS, 4L and 60L | PASS | **PASS** | 60L forward 348.6 ms; CLI 352.4 ms/step, warm wall 82.4 s, 10.15 GiB/rank |
| TP4, eager | PASS, 4L and 60L | PASS | **PASS** | 60L forward 1155.3 ms; CLI 1152 ms/step, wall 119.5 s |
| tp2 or tp1 with cp=1 (fewer cores) | not run | not run | **NOT MEASURED** | — |
| CP gather_kv (`--cp-mode gather_kv`) | PASS: tp2cp2 4L/60L, tp1cp4 4L/30L | not run | **PASS** (DiT) · CLI **NOT MEASURED** | 60L tp2cp2 445.9 ms; 1 block NEFF for all ranks |
| CP ulysses | PASS: tp2cp2 4L/60L, tp1cp4 4L/30L, tp2cp2 4L @1024x1008 | PASS (tp2cp2) | **PASS** | 60L tp2cp2 416.0 ms; CLI 419.1 ms/step, wall 91.4 s, 19.75 GiB/rank; 1 block NEFF per CP coordinate |
| CP ring | PASS: tp2cp2 4L/60L, tp1cp4 4L/30L, padding at 1024x1008 | not run | **PASS** (DiT) · CLI **NOT MEASURED** | 60L tp2cp2 543.6 ms (1.31x Ulysses); slowest mode everywhere |
| 60 layers at tp1 (tp1cp4) | OOM during the load | — | **LIMIT** | 38.05 GiB/rank of weights > 24 GiB per core; use tp ≥ 2 (19.50 GiB at tp2). 30 layers at tp1cp4 PASS |
| CFG-parallel (`--cfg-parallel`) | — | — | **N/A** | Qwen-Image is guidance-distilled: one forward per step. Rejected by the config, the application and the driver |
| SP (`--sp`) | — | — | **N/A** | Not implemented on the neuron backend: `check_parallel_supported` and the driver reject it |
| DP (`dp_degree>1`, `--requests/--dp`) | — | — | **N/A** | Not implemented on the neuron backend: rejected by `check_parallel_supported` and the driver |
| TeaCache (any flag) | — | — | **N/A** (phase 1) | Rejected in `_reject_unsupported` and by the driver before launch; unit-tested, not exercised on device |
| Multi-shape buckets (`--shapes` with >1 shape) | — | — | **N/A** (phase 1) | One static shape per process; the driver rejects it. A single non-default shape works at the DiT level (1024x1008, V2j); the CLI at another shape is NOT MEASURED |
| batch > 1 | — | — | **N/A** (phase 1) | `batch_size=1` only; rejected in `_reject_unsupported` |
| Device VAE (`enable_vae_decoder=True`) | — | not used | **NOT MEASURED** | The code path exists (reuses `NeuronWanVaeDecoderApplication`; unit tests only). The CLI decodes on the host: fp32, 23 s, 28% of the warm wall |
| Host VAE (CLI default) | — | PASS | **PASS** | 23.0-23.9 s at 6 threads; images coherent |
| Device text encoder | — | — | **N/A** | By design the neuron application runs Qwen2.5-VL on rank 0's host and broadcasts the embeddings |
| Host text encoder (Qwen2.5-VL bf16) | — | PASS | **PASS** | `torch.equal` to diffusers; 0.57 s warm encode; 98.4 s on the first, page-cache-cold encode |
| Serving | — | — | **N/A** (not wired) | `difflet/serving/orchestrators/qwen_image.py` builds the Trainium `NeuronQwenImageApplication`; there is no torchrun neuron serving path |
| Persistent NEFF cache reuse | PASS | PASS | **PASS** | Warm launches: 0 compilations and 0 new .neff; outputs `torch.equal` / bit-identical to cold |
| Rank determinism | PASS | PASS | **PASS** | Identical output and latents sha256 on all 4 ranks in every cell |

### Performance at a glance (60 layers, 1024x1024, warm)

| Config | DiT forward, preflight (ms) | CLI step median (ms) | CLI process wall, 50 steps (s) | Peak HBM (GiB/rank) | Cold block NEFF compile |
|---|---:|---:|---:|---:|---|
| TP4 compile | 348.6 (n=30) | 352.4 | 82.4 | 10.15 | 20.6 s CLI / 21.2 s preflight compile wall; 1 block NEFF |
| tp2cp2 ulysses | 416.0 (n=30) | 419.1 | 91.4 | 19.75 | 2 block NEFFs, 19.6/19.8 s (measured at 4L) |
| tp2cp2 gather_kv | 445.9 | NOT MEASURED | NOT MEASURED | 19.75 | 1 block NEFF, 16.7 s (4L) |
| tp2cp2 ring | 543.6 | NOT MEASURED | NOT MEASURED | 19.75 | 2 block NEFFs, 23.8/23.7 s (4L) |
| TP4 eager | 1155.3 (n=3) | 1152 | 119.5 | 10.48 (preflight) | no block NEFF; 54 eager-op NEFFs |

On 4 cores at this shape, TP4 is faster than any CP mode. *Inference:* CP adds collectives without
cutting per-rank weight traffic the way tp4 does. The recommended CP mode is Ulysses, then
gather_kv.

## Known semantic differences from diffusers

1. **The text mask is dropped, as on Trainium and TPU.** Prompt embeddings are zero-padded to
   `text_seq_len` (1024) and attended unmasked. `encoder_hidden_states_mask` is accepted and deleted
   (`difflet/backends/neuron/qwen_image/transformer.py` module docstring). This matches diffusers
   only where the mask is all ones. The V1/V2 references were therefore run with
   `encoder_hidden_states_mask=None` on the same padded embeddings. Masked attention is a follow-up,
   not a phase-1 feature.
2. **There is no true-CFG negative pass.** The CLI accepts `--guidance-scale` and drops it. The
   checkpoint has `guidance_embeds=false`, and `guidance_embeds=True` is rejected by
   `NeuronQwenImageConfig.validate`. Each step is one forward, as on the Trainium path. Images are
   still clean (V3b-g).
3. **The latents and the FlowMatch Euler scheduler stay on the host** in the application dtype, as
   on Trainium. Every rank runs the orchestrator on identical inputs.
4. **The timestep is t/1000 rounded to bf16** (the orchestrator's value). The CPU references use
   the same rounding.

## Bug ledger

No Difflet bug was found. There are no fix commits. Commits from this campaign (tests and
evidence only):

| SHA | Kind | What |
|---|---|---|
| `98a081c` | test | Qwen-Image DiT device preflight and first-N-blocks CPU reference |
| `ea6fe1b` | evidence | V1 TP4, 4 and 60 layers, eager and compile |
| `eca572e` | test | Preflight checks rank identity (`ranks_identical`) and the CP scatter (`cp_scatter`) |
| `317060e` | evidence | V2 context parallelism (gather_kv, ulysses, ring) |
| `3d2641c` | test | Prompt-embedding parity against diffusers on the host |
| `1c16390` | evidence | V3 end to end through `difflet compile/generate --backend neuron` |

Environment traps, not Difflet bugs (the logs are kept):

* **The venv's `bin/` is not on PATH.** `prepare_runtime` fails with `COMPILATION FAILED:
  posix_spawn failed: No such file or directory` because neuronx-cc is spawned from the venv
  (`E/v1/a-l4-eager.try1-nopath.log`). This is the known gotcha. Fix: put
  `PATH=/home/ubuntu/workspace/native_venv/bin:$PATH` in the launch environment.
* **There is no `refs/main` in the local HF cache.** With `HF_HUB_OFFLINE=1`, a launch without a
  revision fails on every rank in `snapshot_download` (`LocalEntryNotFoundError`) about 25 s after
  start (`E/v3/tp4-compile.try1-norevision.log`). Fix: pass `--revision
  75e0b4be04f60ec59a75f475837eced720f823b6`.

## Open issues and follow-ups

* **A tp1 OOM fails slowly.** A device OOM during the load raises only after about 16 min of
  torch-neuronx lazy-allocator retries (V2k). Suggestion, not implemented: a fail-fast HBM budget
  check in `TorchNeuronApplicationBase.load` before `materialize_meta_`, comparing the estimated
  weight bytes per rank against per-core HBM and giving tp guidance. Estimated at about 20 lines
  plus a unit test, with no cache-key impact.
* **Ring is the slowest CP mode:**
  * 1.24x Ulysses at 4L tp2cp2 and 1.89x at 4L tp1cp4;
  * 1.31x at 60L tp2cp2 and 1.98x at 30L tp1cp4;
  * another 1.23x slower at 1024x1008.

  `collectives.ring_shift` is an `all_to_all_single` carrying zero slots. Its docstring says this
  costs bandwidth at cp > 2. *Inference:* the padded query or the 2016-token key blocks are slow
  paths in the flash-lse kernel.
* **Ulysses and ring compile one block NEFF per CP coordinate,** because the CP coordinate is baked
  into the graph: 2 at tp2cp2 and 4 at tp1cp4. gather_kv compiles 1. This is expected for the
  one-process-per-rank design, but it multiplies cold compile work by cp.
* **The first request after a page-cache eviction is slow.** The rank-0 prompt encode took 98.4 s,
  against 0.57 s warm. The text encoder's `from_pretrained` is lazy (0.7 s), so the 16 GB read
  lands in the first encode, and the CLI's `load+warmup` line does not show it. *Inference:* this is
  a disk read; free buff/cache went from 55 to 68 GiB, but no cache drop was done.
* **The host VAE decode takes about 23 s,** 28% of the warm 82 s wall. Candidates are the device
  VAE or more threads; not investigated.
* **The CLI prints no stage clocks** beyond the rank `load+warmup / stage` line. The encode,
  denoise, step and decode times above come from the V3 observer probe.
* **Optional:** the driver could resolve the model path once and fail fast before torchrun spawns
  4 ranks (see the `refs/main` trap).
* **Not committed by design:**
  * `E/prompt_embeds.pt` (7.3 MB; its sha256 is in `E/v1/summary.json`).
  * Every reference `.pt`, device `out.pt`, `latents.pt` and trajectory dump, under
    `/home/ubuntu/qwen-v{1,2,3}-work`.
  * The scratch cache roots `…-qwen-l4` and `…-qwen-v3cold`. They can be removed with user
    approval.

## How to inspect manually

* **Per cell, preflight (V1/V2):**
  * The log's first line, `launch <ts> head=<sha> cache=<root> args: …`, is the exact argv; the last
    line is `EXIT rc=0 launch_wall_s=…`.
  * Each rank prints its check verdicts (`checks_rank0`, plus `failed_checks_any_rank` in V2, which
    is `[]` in every gated cell).
  * `result.json` (rank 0) and `result.rank{1,2,3}.json` hold parity, phase seconds, counters,
    block NEFF keys, HBM and fallbacks per rank.
* **Per cell, CLI (V3):**
  * The log's `cmd:` line is the CLI argv; `EXIT rc=0 process_wall_s=… new_neff=… new_hlo_files=…`
    closes it.
  * The `[qwen_image]` rank lines show `compiled_blocks=60`, `fallbacks_load=[]`,
    `fallbacks_generate=[]` and `unwarmed_shapes=[]`.
  * `<cell>.result.json` holds per-rank latents sha256, step series, VmHWM and HBM.
* **Parallelism fingerprints:**
  * CP: `shape_rules` and the `cp_scatter` detail in the V2 results (image tokens per rank = 4096 /
    cp).
  * Ulysses and ring: distinct block NEFF keys per CP coordinate (`block_neffs_per_rank`);
    gather_kv: one key shared by all ranks.
  * tp1 vs tp2: `weights_gib_per_rank` (2.61 vs 1.34 GiB at 4L; 19.07 GiB at 30L tp1; 19.50 GiB at
    60L tp2).
* **Images:** `E/v3/png/*.png`. Pixel and latent comparisons are in `E/v3/compare.json`.
* **Re-run the golden check (warm cache):**
  `bash artifacts/verification-2026-10-16-qwen-neuron/v3/scripts/gen.sh /tmp/qwen-golden 0 50 4 1
  gather_kv`, then check that `sha256sum /tmp/qwen-golden/out.png` is `c643eefc…`. `gen.sh` execs
  `/home/ubuntu/qwen-v3-work/cli.sh`, which is present on this host. On another host, copy
  `E/v3/scripts/` there first or edit that path.
