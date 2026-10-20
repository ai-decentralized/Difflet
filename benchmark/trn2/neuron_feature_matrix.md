# Neuron backend feature matrix — Wan2.1 14B and Qwen-Image on trn2.3xlarge

**Backend:** neuron (TorchNeuron: torchrun, one process per NeuronCore, per-block `torch.compile`), with a same-host AoT (Trainium/NxD) baseline  
**Device:** trn2.3xlarge, 1 device, 4 NeuronCores (LNC=2, 24 GiB HBM per core), 12 vCPU, 124 GB RAM  
**Campaign:** 2026-10-17 21:26 UTC – 2026-10-18 07:53 UTC; code under test `c255cb6` plus the fix commits `b78f508`, `7941447`, `486ca61`, `6d38344` (branch `feat/neuron-wan-qwen-gaps`)  
**Raw evidence:** kept out of git, under `/home/ubuntu/bench-neuron-matrix/` on the benchmark host (full report: `report.txt`; per-cell logs, results, images, scripts)

Every number below comes from this campaign. The exception is the Qwen-Image AoT row, which is reused from the same-host A/B at commit `03f3a84` (2026-10-17) and is marked. Inferences are labelled.

## Method (identical on every cell)

- **Cells and runs:**
  - Each cell runs on a fresh compile-cache root.
  - `difflet compile` runs cold.
  - One warm generate is discarded, then two warm generates are timed.
- **DiT step:** median / p10 / p90 of the rank-0 inter-step deltas of the real 50-step loop, step 0 excluded, n=49.
  - Measured by an observer probe that wraps the orchestrator's `_scheduler_step` (shared by both backends).
  - With the probe on, both goldens reproduced bit for bit.
- **e2e:** process wall of `python -m difflet.cli.main generate`.
  - *Cold* = page cache dropped first, NEFF cache warm.
- **Compile:** the cold `difflet compile` process wall.
- **HBM:** `torch_neuronx.max_memory_allocated`, plus neuron-monitor's per-core peak.
- **Workload:**
  - Wan: 480x832x9, 50 steps, guidance 5.0.
  - Qwen: 1024x1024, 50 steps.
  - Both: seed 42, prompt "a cinematic shot of a red fox running through a snowy forest".

## Headline: native vs AoT

**Wan2.1-T2V-14B** (480x832x9, g=5; both venvs import difflet from the same worktree)

| metric | native | AoT | native vs AoT |
|---|---:|---:|---:|
| DiT step, tp4 | 1061.7 ms | 1151.5 ms | −7.8% |
| DiT step, tp2cfg2 | 931.3 ms | 1071.8 ms | −13.1% |
| compile cold, tp4 | 314.2 s | 482.0 s | −35% |
| compile cold, tp2cfg2 | 295.2 s | 1468.9 s | −80% |
| e2e warm, tp4 | 125.8 s (default: device umT5 + device VAE) | 134.5 s | −6.5% |
| e2e warm, tp4, same placement as AoT (`--host-vae`) | 165.2 s | 134.5 s | +23% |
| e2e warm, tp2cfg2 | 130.4 s | 140.1 s | −6.9% |
| e2e cold, tp4 | 746.2 s | 461.8 s | +62% |

*Inference:* the native cold start reads the fp32 safetensors (57 GB DiT + 22.7 GB umT5) and casts them, while AoT reads its pre-sharded bf16 weights.

**Qwen-Image** (1024x1024, TP4)

| metric | native | AoT (reused, `03f3a84`) | native vs AoT |
|---|---:|---:|---:|
| DiT step | 351.4 ms; 328.9 ms with `--sp` | 418.4 ms | −16.0%; −21.4% |
| compile cold | 155.2 s (with the default VAE stage) | 1480 s | −90% |
| e2e warm | 79.4 s (new default, mean of 4) | 76.9 s | +3.3% |
| e2e cold | 492.0 s | 505.9 s | −2.7% |

Why the Qwen AoT step is slower: NxD's `RowParallelLinear` all-reduces in fp32 by default. Forcing bf16 takes AoT from 417 to 360 ms per forward (profiling, 2026-10-17).

## Wan2.1 14B matrix (native unless noted)

| cell | outcome | step median / p10 / p90 (ms) | e2e warm (s) | compile cold (s) | HBM alloc / peak (GiB per core) |
|---|---|---|---|---|---|
| tp4, default (device umT5 + device VAE) | PASS | 1061.7 / 1060.7 / 1064.8 | 125.8, 125.8 | 314.2 | 13.58 / 15.15 |
| tp4 `--host-text-encoder --host-vae` | PASS | 1062.0 / 1060.5 / 1064.6 | 161.8, 161.3 | 490.4 | 8.12 / 9.55 |
| tp4 eager | PASS | 7369.8 / 7352.9 / 7392.6 | 440.3, 440.8 | 262.4 | 13.77 / 15.30 |
| tp4 `--sp` | PASS | 1042.1 / 1041.1 / 1044.4 | 129.1, 126.8 | 291.1 | 13.48 / 14.68 |
| tp2cfg2 | PASS | 931.3 / 930.2 / 934.5 | 129.7, 131.2 | 295.2 | 14.13 / 15.59 |
| tp2cfg2 `--sp` | PASS | 944.8 / 944.0 / 948.9 | 133.2, 132.2 | 296.0 | 13.86 / 15.23 |
| tp2cp2 gather_kv | PASS | 1191.0 / 1190.1 / 1193.7 | 134.3, 137.1 | 265.5 | 13.89 / 15.14 |
| tp2cp2 ulysses | PASS | 1154.0 / 1152.9 / 1156.5 | 133.9, 134.6 | 262.0 | 13.89 / 14.95 |
| tp2cp2 ring | PASS | 1304.4 / 1303.4 / 1307.8 | 128.3, 136.9 | 292.7 | 13.91 / 15.13 |
| dp2tp2, 4 requests | PASS | 1852.1 (per tp2 replica, serial CFG) | 294.1, 295.2 for 4 requests (49 requests/h) | 280.7 | 14.12 |
| tp4, 480x832x81, default placement | LIMIT | — | device OOM at the first denoise step (23.81 GiB per core) | 1616.4 | — |
| tp4, 480x832x81, `--host-text-encoder` | PASS (n=1) | 14535.8 (ramps 11.65 → 14.9 s; *inference:* throttling) | 1213.5, 1307.4 | — | 12.18 / 21.88 |
| AoT tp4 (`--host-vae`) | PASS | 1151.5 / 1150.7 / 1153.7 | 134.4, 134.6 | 482.0 | 10.34 |
| AoT tp2cfg2 | PASS | 1071.8 / 1071.2 / 1073.0 | 140.4, 139.9 | 1468.9 | 19.85 |

At tp2 (CFG, CP, DP) the driver moves umT5 to the host on its own. It prints, for example: "umT5 on the device needs at least 23.9 GiB per core beside the DiT at tp=2, over the 22.8 GiB budget".

## Qwen-Image matrix (native)

| cell | outcome | step median / p10 / p90 (ms) | e2e warm (s) | compile cold (s) | HBM (GiB per core) |
|---|---|---|---|---|---|
| tp4, host decode (`--host-vae`) | PASS | 351.4 / 350.8 / 352.7 | 71.5–87.0 (6 runs) | 352.5 | 10.15 |
| tp4, device VAE (**now the default**) | PASS | 351.6 / 350.8 / 352.4 | 75.2–81.0 (6 runs) | 155.2 | 10.15; VAE core 13.45 |
| tp4 `--device-text-encoder` | PASS | 351.3 | 84.5, 80.6 | 83.5 | 13.40 |
| tp4, device VAE + device text encoder | PASS | 351.5 | 77.5, 80.1 | 168.3 | 13.40 |
| tp4 `--sp` | PASS | 328.9 / 328.2 / 329.9 | 78.7, 80.0 | 68.6 | 9.93 |
| tp2cp2 gather_kv | PASS | 448.1 / 447.7 / 449.0 | 89.2, 92.0 | 77.0 | 19.75 |
| tp2cp2 ulysses | PASS | 418.4 / 417.9 / 419.7 | 87.9, 79.5 | 77.5 | 19.75 |
| tp2cp2 ring | PASS | 546.4 / 546.0 / 547.4 | 93.5, 95.3 | 71.9 | 19.75 |
| dp2tp2, 4 requests | PASS | 615.5 (per tp2 replica) | 149.0, 147.2 for 4 requests (about 97 requests/h) | 72.4 | 19.81 |
| true CFG tp4 (`--true-cfg-scale 4.0`) | PASS | 703.4 / 702.1 / 705.2 | 90.2, 101.8 | — | — |
| true CFG tp2cfg2 | PASS | 621.4 / 620.5 / 623.3 | 103.6, 105.5 | 79.5 | 19.81 |
| tp4 eager | PASS | 1153.5 / 1143.9 / 1170.5 | 119.1, 118.9 | 62.7 | 10.48 |
| dp4tp1 | refused in 0.50 s (HBM check: 38.1 GiB per core needed) | — | — | — | — |

## Feature verification (Part A)

**Wan2.1**

- **4-layer preflight vs the CPU fp32 oracle:** PASS in all 6 cells: tp4, tp2cp2 gather_kv / ulysses / ring, tp1cp4 ulysses, tp4 `--sp`.
  - Cosine ≥ 0.99995; rel_l1 0.83–0.96x the bf16 control.
  - Ranks identical; one compiled graph per cell.
- **HBM budget:** neuron-ls reports 24 GiB per core. The estimate runs 0–7% below the measured allocation, and no working configuration was rejected.
- **Host umT5 + host VAE:** reproduces the W4 golden bit for bit (latents `8e5eaffe…`, mp4 `a9f9004f…`).
- **Default device umT5:** latents `9f198762…`, cosine 0.9956 to the golden after 50 steps.
- **tp2cfg2:** equals the serial-CFG tp2 run bit for bit.

**Qwen-Image**

- **Device VAE vs host decode:** fp32 image cosine 1 − 3e-13; PNGs at most 1 level apart.
- **`--sp` vs non-SP:** DiT cosine 0.999988.
- **DP:** dp2tp2 equals dp1tp2 per request, bit for bit.
- **Device text encoder vs the host bf16 encoder:** cosine 0.99983, 0 fallbacks, one graph for the 28 layers. At tp2 the driver falls back to the host with a note.
- **True CFG:** tp2cfg2 equals serial tp2 bit for bit.
- **Default path:** latents unchanged (`02613803…`).

## Bugs fixed during the campaign

| commit | problem → fix |
|---|---|
| `b78f508` | The DP router's HBM check ignored `--revision`, so with no `refs/main` it skipped the check. It now resolves the pinned revision, as the workers do. |
| `7941447` | Qwen `--device-vae` failed at 1024x1024. The Wan VAE attention's fused SDPA lowers to nkilib `attention_cte`, which asserts out-of-bounds at ≥ 8192 key tokens. On neuron devices it now uses the explicit softmax attention (cosine 1.0). Trainium/TPU keep the fused call. |
| `6d38344` | One dp2tp2 request in 4 launches gave different latents. A fresh process's first bf16 Qwen2.5-VL encode on the host was occasionally different under CPU contention (cause inside the CPU kernels not identified). Fix: one discarded encode at load; afterwards 48 of 48 encodes were stable. |

**Default changed (`486ca61`):** Qwen-Image now decodes in a 1-core device VAE stage.
- Gain: decode 19.2 s vs 23.6 s; e2e 79.4 s vs 83.2 s.
- Correctness: latents unchanged; image within 1 level of the host decode.
- Cost: about 90 s more cold compile.
- `--host-vae` opts out.

## Pre-sharded weight store (2026-10-19)

`difflet/backends/neuron/core/weight_store.py` (commits `4a56b85`, `7a151de`). Every device component's load writes this rank's slices, in the load dtype, under `<DIFFLET_COMPILE_CACHE>/neuron/weights/`. Later loads read those files instead of the checkpoint. An entry is written only when it is at least 10% smaller than what the checkpoint load reads (an fp32 checkpoint loaded in bf16). `DIFFLET_NEURON_WEIGHT_STORE=0` turns the store off, and `DIFFLET_NEURON_WEIGHT_STORE_ALL=1` writes every component.

**Method:** the same host and workloads as the matrix above. Each model ran at tp4 with the default placement, on a fresh compile-cache root.
- `difflet compile` ran with the page cache dropped first, and wrote the store.
- Then one store-cold generate and three store-warm generates, then one direct (store off) cold generate and three direct warm generates. The first warm generate of each series is discarded.
- *Cold* means the page cache was dropped; the NEFF cache is warm.
- The "load" columns give the component's "load checkpoint" phase (max over ranks).
- The disk is the EBS root volume: `dd` read it at 132–135 MB/s, with one reader or four.

| model, component | load: direct cold / store cold | load: direct warm / store warm | e2e cold: direct / store | e2e warm: direct / store | store size | write cost (compile load phase) |
|---|---:|---:|---:|---:|---:|---:|
| Wan2.1 14B, DiT (fp32 → bf16) | 435.0 / 217.0 s | 5.0–5.1 / 3.5–4.7 s | 739.4 / 524.5 s | 122.9, 126.0 / 125.0, 123.4 s | 28.58 GB | 652.9 s vs 435.0 s direct cold |
| LTX-2, device Gemma-3 (fp32 → bf16) | 358.0 / 178.5 s | 3.4–3.5 / 2.6–2.8 s | 745.1 / 574.3 s | 77.8, 78.3 / 76.6, 77.7 s | 23.53 GB | 537.5 s vs 358.0 s direct cold |
| Qwen-Image, DiT (bf16 checkpoint, written before `7a151de`) | 311.0 / 311.0 s | 3.6–3.7 / 3.6 s | 497.9 / 497.2 s | 77.7, 76.5 / 76.6, 78.9 s | 40.86 GB | 622.4 s vs 311.0 s direct cold |

- **Bit-identity.** Every store-loaded generate equals the direct-loaded one bit for bit:
  - Wan: latents `9f198762…`, the matrix golden above;
  - Qwen-Image: latents `1fea84b1…`, PNG `a300a78e…`;
  - LTX-2: latents `95ef7d53…`.
- **Qwen-Image.** The store saves nothing when the checkpoint is already in the load dtype, so `7a151de` stops writing such entries. The same applies to the fp32 umT5 and the VAEs. After that change, a Qwen run on an empty store writes nothing ("no gain").
- **The key separates TP layouts.** Wan tp2cfg2 on the same root wrote a second entry (`…__tp2__685ea74c…`, 28.58 GB) beside the tp4 one.
  - Rank 0 wrote the replicated and tp0 files and rank 3 wrote tp1. The two CFG replicas skipped them, because another process held the lock.
  - Warm DiT load: 9.4 s from the store vs 8.8–8.9 s direct. Warm e2e: 133.2 vs 133.5 / 134.9 s.
  - Store and direct are bit-identical (latents `3717d503…`).
- **Lazy population.** A first Wan generate on an empty store root, with no compile, wrote the entry. Its DiT load phase took 220.4 s, of which 215.2 s were store writes. The next generate read the entry.

## Open issues

1. **Wan 81 frames, default placement:** device OOM. The umT5 placement check counts weights only and not the activations at 32,760 tokens.
2. **Wan 81 frames, warm-up and step time:** the warm-up takes 427 s per launch even with warm NEFFs. The step time ramps (*inference:* throttling).
3. **Native cold start (Wan):** 62% slower than AoT. The bf16 pre-sharded weight store (below) takes the Wan cold e2e from 739.4 to 524.5 s.
4. **Per-launch jitter:** 23–34 s of spawn and `prepare_runtime` jitter. The device VAE adds a second torchrun launch.
5. **Wan host VAE decode:** 59–61 s in the native venv vs 49 s in the AoT venv (not investigated).
6. **Not measured:** Wan 1.3B and Wan 2.2 (no weights on the host). The Qwen AoT numbers were reused, not re-measured.
7. **Cold loads the weight store does not cover:**
   - The fp32 umT5 on the device: 172.3 s per cold Wan launch, the same bytes either way.
   - The host encoders: Qwen2.5-VL took 99.8 s per cold Qwen launch. The host umT5 and the host Gemma-3 at tp2 were not measured.
   - A store on the instance's local NVMe (`DIFFLET_NEURON_WEIGHT_STORE` plus `..._ALL=1`) would also speed up bf16 checkpoints. It was not measured.

## Toolchain

| | native_venv | aot_venv |
|---|---|---|
| Python | 3.14.8 | 3.12.3 |
| torch | 2.14.0+cpu | 2.9.1 |
| torch-neuronx | 2.14.3.0.0 | 2.9.0.2.15 |
| neuronx-cc | 2.28.4568.0 | 2.26.6360.0 |
| nki | 0.7.0 | 0.5.0 |
| NxD | — | 0.19.28492 |
| diffusers / transformers | 0.38.0 / 4.57.6 | 0.38.0 / 4.57.6 |

Host runtime: aws-neuronx-runtime-lib 2.35.27.0, collectives 2.35.40.0.
