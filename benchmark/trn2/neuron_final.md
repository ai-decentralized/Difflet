# Neuron backend final validation: Wan2.1 14B, Qwen-Image and LTX-2 on trn2.3xlarge

**Backend:** neuron (TorchNeuron: torchrun, one process per NeuronCore, per-block `torch.compile`)  
**Device:** trn2.3xlarge, 1 device, 4 NeuronCores (LNC=2, 24 GiB HBM per core), 12 vCPU, 124 GB RAM, one EBS root volume (~130 MB/s)  
**Campaign:** 2026-10-19 16:20 – 22:19 UTC, branch `feat/ltx2-neuron`, code under test `61022ec` (clean tree in every run: `dirty=0` in each launch line)  
**What changed since the earlier campaigns:** the code-convergence refactor (`c241cc7..ddf7acc`, 27 commits: `TorchNeuronModelApplication`, `config_checks`, the component HBM budget, the torchrun stage helpers, fixes C3, S3, S6, S7, L1) and the bf16 pre-sharded weight store (`4a56b85..61022ec`)  
**Raw evidence:** kept out of git, under `/home/ubuntu/bench-neuron-final/` on the benchmark host (per-run logs, probe events, outputs, contact sheets, `summary.jsonl`, the scripts)

Every number below comes from a run made for this report, unless it is marked *reused*. The "before" columns are reused from the earlier reports on this host: `neuron_feature_matrix.md` (Wan, Qwen-Image and its weight-store section) and `ltx_2_neuron.md`. All AoT numbers are reused (the AoT venv was not run). Inferences are labelled.

## Verdict

- **No regression in outputs.** Every default-path golden, every CFG-parallel, CP, SP and DP case that has a pre-refactor hash reproduced it bit for bit: 17 cells (26 latents hashes, counting each DP request), plus the PNG and mp4 hashes wherever the decode path is the same as before.
- **No regression in device time.** The DiT step is within −0.3% to +0.3% of the earlier campaigns in all 15 cells, and peak HBM is unchanged.
- **The folded-in fixes hold on the device:** L1, C3, S3, S6 and the weight store (hit, "no gain", a tp2 entry beside the tp4 one).
- **No bug found, so no fix commit.**
- **One cost to know about (open issue 1):** a cold compile on a fresh root now writes the weight store. On this disk that adds about 200 s even when the checkpoint is in the page cache: Wan tp4 compiles in 518.9 s against 314.2 s before, and LTX-2 in 306.6 s against 119.3 s. In exchange, every later cold start is faster: Wan cold e2e is 520.2 s against 746.2 s.

## Method

The method is the earlier campaigns', with three differences, all listed here:

- **Observer probe.** One probe (`scripts/probe/sitecustomize.py`) merges the matrix probe's step timer, the LTX-2 probe's audio-step stamp and the weight-store probe's load events. It is active with `DIFFLET_AB_PROBE=1` and changes no argument, tensor or control flow.
  - It adds three events: `rank_env` (what each rank sees for `DIFFLET_NEURON_PG_TIMEOUT_S` at `init_process_group`), `stage_cmd` / `stage_env` (each torchrun launch), and `model_load` (composed phase keys, compiled blocks, unwarmed shapes).
  - **DiT step** = median / p10 / p90 of the rank-0 inter-step deltas of the real loop, step 0 excluded (n=49 per run for Wan and Qwen, n=19 for LTX-2). The table value is the median of the per-run medians of the timed runs. p10 / p90 are pooled over the timed runs.
- **No exported timeout.** `DIFFLET_NEURON_PG_TIMEOUT_S` is not exported anywhere, except in the one S3 override run. The earlier campaigns exported 7200.
- **Compile and the page cache.**
  - The earlier campaigns measured compile with the checkpoint in the page cache. Here each cell compiles on a fresh root with the page cache as the previous cell left it, and the load phases say whether the read was cold.
  - For the three tp4 headline cells there is also a **page-cache-warm compile**: drop the page cache, read the checkpoint twice, check residency with `mincore`, then compile on a fresh root.
  - A first attempt that read the files once, without dropping first, did not stay resident. Those runs are reported but not used.

Unchanged from before:

- **Runs per cell:** a cold `difflet compile` on a fresh root `~/.cache/final-<model>-<cell>`. For the tp4 headline cells, a generate with the page cache dropped (cold e2e). Then one discarded warm generate (gen0), then n timed warm generates: n=3 for tp4 headline, n=2 elsewhere.
- **e2e:** the process wall of `python -m difflet.cli.main generate`.
- **HBM:** `torch_neuronx.max_memory_allocated` (max over ranks), plus neuron-monitor's per-core peak.
- **Workloads:**
  - Wan: 480x832x9, 50 steps, g=5.
  - Qwen-Image: 1024x1024, 50 steps.
  - LTX-2: 480x704x49, 20 steps, g=1.
  - All: seed 42, prompt "a cinematic shot of a red fox running through a snowy forest", pinned revisions, `HF_HUB_OFFLINE=1`.

One output image or frame per cell was opened and checked (contact sheets in `sheets/`). Every seed-42 output shows a red fox in a snowy forest, in all three models.
- **Wan 9-frame mp4:** it shows the same coloured speckle in frame 0 that the matrix report noted for the golden; it is the same file, bit for bit.
- **LTX-2 seed-45 DP request (req3):** at 20 steps and g=1 it shows a figure in an orange jacket in the snow rather than a fox. That mp4 is byte-identical to the LTX campaign's.

## Part A: regression and correctness

| check | outcome | evidence (`F` = `/home/ubuntu/bench-neuron-final`) |
|---|---|---|
| A1 Wan tp4, default placement (device umT5 + device VAE) | **PASS, bit-identical** | latents `9f198762…`, latents.pt `3cafa271…`, mp4 `9b914e81…` in the cold run and all 4 warm runs. These equal the matrix's and the weight-store run's hashes. `F/cells/wan/tp4/` |
| A1 Wan `--host-text-encoder --host-vae` (W4 golden) | **PASS, bit-identical** | latents `8e5eaffe…`, latents.pt `e639f314…`, mp4 `a9f9004f…`, the W4 golden. `F/cells/wan/hostte-hostvae/` |
| A1 Qwen-Image tp4 default (device VAE), V3 golden command | **PASS, bit-identical** | latents `02613803…` on all 4 ranks, latents.pt `a3972ad1…`, PNG `789553f0…`. That PNG is exactly the post-flip golden, which is within 1 level of the host-decode `c643eefc…`. `F/cells/qwen/golden-v3.log` |
| A1 Qwen-Image tp4 default, campaign prompt | **PASS, bit-identical** | latents `1fea84b1…`, latents.pt `aac25966…`, PNG `a300a78e…` in the cold run and 4 warm runs. These equal the matrix's device-VAE cell and the weight-store run. `F/cells/qwen/tp4/` |
| A1 LTX-2 tp4 default (device Gemma-3, host decode) | **PASS, bit-identical** | video / audio latents `95ef7d53…` / `7b27d951…`, latents.pt `75a85964…`, mp4 `7328fe9e…` in the cold run and 4 warm runs. `F/cells/ltx/tp4/` |
| A1 CFG parallel: Wan tp2cfg2 | **PASS, bit-identical** | latents `3717d503…`, mp4 `b725b513…`, the matrix's |
| A1 CFG parallel: Qwen true CFG tp2cfg2 (`--true-cfg-scale 4.0`) | **PASS, bit-identical latents** | latents `a8ae41a9…`, latents.pt `e1d8b0bf…`, the matrix's. PNG `b6316f2e…` is a **new reference**: the matrix cell decoded on the host, before the device-VAE default |
| A1 CFG parallel: LTX-2 tp2cfg2, g=4 | **PASS, bit-identical** | latents `b1ce08d3…` / `4423cbd7…`, mp4 `dcf1fb67…`, the LTX campaign's |
| A1 CP: Wan tp2cp2 ulysses | **PASS, bit-identical** | latents `1ce63b49…`, mp4 `ebe0d297…` |
| A1 CP: Qwen tp2cp2 ulysses | **PASS, bit-identical latents** | latents `76d6d0d7…`, latents.pt `1804e81d…`. PNG `73d756b2…` is a new reference (the matrix cell decoded on the host) |
| A1 CP: LTX-2 tp2cp2 ulysses | **PASS, bit-identical** | latents `eb35b537…` / `122bd987…`, mp4 `cbe44656…` |
| A1 extra cells | **PASS, bit-identical** | Wan tp4 `--sp`: `f414ca27…`, mp4 `4d7d84b2…`. Qwen tp4 `--sp`: `4914cd37…` (PNG `33a15872…` is new, device decode). LTX-2 tp2cp2 gather_kv: `f59e2ff5…` / `4ef0ceae…`, mp4 `8ff32cdb…`. DP below |
| A1 DP, 4 requests (seeds 42–45) | **PASS, bit-identical per request** | Wan dp2tp2: the 4 latents (`3717d503…`, `6c6ef6dc…`, `d202a9d2…`, `e98b94d8…`) and the 4 mp4s equal the matrix's. Qwen dp2tp2: the 4 latents and 4 PNGs equal the post-`6d38344` run (`dp2tp2-postfix`). LTX-2 dp2tp2: the 4 latents and 4 mp4s equal the LTX campaign's. Every request was identical in every run |
| A2 L1: LTX-2 4-block preflight (`Fp32GELU` module instead of the class swap) | **PASS, bit-identical** | `l4-eager` and `l4-compile` `out.pt`: `torch.equal` on all 4 tensors against the LTX campaign's outputs. The gates read the same values: eager 1.035x / 1.053x video, 1.000x / 1.053x audio; compile 0.987x / 1.023x and 1.004x / 1.020x. One graph, 4 of 4 blocks, 0 fallbacks, ranks identical. The block NEFF key differs (`442d04d4…`, 2,924,318 B, vs `31a6f895…`, 2,925,280 B). *Inference:* the module change alters the traced HLO's structure or metadata, not its numerics. `F/preflight/` |
| A3 C3: a vae stage launched by `python -m difflet.cli.stage` outside `run_stage` | **PASS** | The Qwen-Image vae stage ran in compile mode as one plain process (no torchrun) on a fresh `DIFFLET_COMPILE_CACHE`, with no `TORCH_NEURONX_*` or `NEURON_*` cache variable in its env. It wrote 82 NEFFs, all under `$DIFFLET_COMPILE_CACHE/neuron/neff`, and **0** new files under `/tmp/neff_cache` (checked against a marker file's mtime). `F/partA/c3.log` |
| A4 S3: the process-group timeout default | **PASS** | In every torchrun launch, every rank's `rank_env` shows `DIFFLET_NEURON_PG_TIMEOUT_S=10800` and `process_group_timeout()` = 10800 s, with the variable unset in the shell: 332 of 336 rank events (for example `F/cells/wan/tp4/cold.log`). The other 4 are the override run. With `DIFFLET_NEURON_PG_TIMEOUT_S=7200` exported (LTX-2 tp2cp2 ulysses gen1), all 4 ranks show 7200 and the stage env carries 7200. These cold launches without the export all succeeded: Wan tp4 (page cache dropped, 520.2 s), Qwen-Image and LTX-2 tp4 cold, and LTX-2 dp2tp2 (host Gemma-3 on two replicas, 256–466 s) |
| A5 S6: Qwen-Image and LTX-2 tp4 load and warm up normally | **PASS** | Qwen: 60 compiled blocks, one warm-up per rank, `unwarmed_shapes=[]`. LTX-2: 96 blocks (48 DiT + 48 Gemma-3), 2 warm-ups per rank, `unwarmed_shapes=[]`. Mesh-less build: `tests/unit/backends/test_neuron_safe_tp_size.py::test_the_neuron_backend_without_a_mesh_fails_loudly` passed (the file: 6 passed, `F/tests/s6.log`) |
| A6 weight store: a hit prints the evidence line | **PASS** | Wan, every rank: `[weight store] rank 1/4 NeuronWanTransformerApplication: hit 7.50 GB in 216.8 s (…__transformer__38ec498c__bfloat16__tp4__63f8c2b16dc7025d)` (cold) and 3.3–4.8 s warm. LTX-2: `… NeuronLtx2TextEncoderApplication: hit 5.88 GB in 178.7 s` cold, 1.6–2.8 s warm |
| A6 weight store: Qwen-Image writes nothing | **PASS** | Fresh root: `… NeuronQwenImageTransformerApplication: not written: no gain: the checkpoint holds these tensors in the load dtype (10.22 GB either way; …)` on 4 ranks, the same for the VAE (0.29 GB); `store_entries` empty. Also "no gain": the Wan umT5 (fp32) and VAE, and the LTX-2 bf16 DiT |
| A6 weight store: a tp2 layout gets its own entry | **PASS** | Wan tp2cfg2, pointed at the tp4 root's store, wrote `…__tp2__685ea74cae68e0fe` beside `…__tp4__63f8c2b16dc7025d`. Rank 0 wrote replicated+tp0 (14.52 GB), rank 3 wrote tp1 (14.05 GB), ranks 1 and 2 printed "being written by another process". Its warm runs hit the entry (6.1–9.1 s). The keys equal the weight-store campaign's |
| A7 stage evidence lines | **PASS** | See the table below. All 272 evidence lines (every rank of every compile and generate) read `unwarmed_shapes=[] fallbacks: load=[] generate=[]`; 0 graph breaks; no traceback in any log |

**Stage evidence per model (tp4 default, warm generate):**

| model | evidence line, rank 0 | compiled blocks | warm-ups per rank (generate; compile adds the vae stage's) | composed `phase_seconds` keys |
|---|---|---|---|---|
| Wan2.1 | `[wan] neuron rank 0/4: exec_mode=compile compiled_blocks=64 unwarmed_shapes=[] fallbacks: load=[] generate=[] load+warmup 42.1s, stage 96.3s` | 64 = 40 DiT + 24 umT5 | 2 (DiT, umT5); compile: rank 0 +1 (VAE) | runtime init, build on meta, load checkpoint, eval, compile, warmup forward, and the same six prefixed `text_encoder` |
| Qwen-Image | `[qwen_image] neuron rank 0/4: … compiled_blocks=60 unwarmed_shapes=[] … load+warmup 37.1s, stage 55.2s` | 60 | 1 (DiT); compile: +1 (VAE) | the six DiT phases + `text encoder load` (host Qwen2.5-VL) |
| LTX-2 | `[ltx_2] neuron rank 0/4: … compiled_blocks=96 unwarmed_shapes=[] … load+warmup 46.6s, stage 59.0s` | 96 = 48 DiT + 48 Gemma-3 | 2 (DiT, Gemma-3) | the six DiT phases + the six `text_encoder` phases |

Under tp2 (CFG, CP, DP) the text encoders move to the host as before, and the counts drop to the DiT's: 40 Wan, 60 Qwen, 48 LTX-2, one warm-up per rank.

## Part B: performance

The "before" column is reused from the earlier reports; the "delta" column is now − before.

### Wan2.1 14B (480x832x9, 50 steps, g=5)

| cell | metric | before (earlier campaign) | now | delta |
|---|---|---:|---:|---:|
| tp4 default + weight store | DiT step median / p10 / p90 (ms) | 1061.7 / 1060.7 / 1064.8 | 1063.1 / 1062.2 / 1066.3 (runs 1063.6, 1063.1, 1063.1) | +0.13% |
| | e2e warm (s) | 125.8, 125.8 | 125.6, 129.2, 121.8 (mean 125.5) | −0.2% |
| | e2e cold, page cache dropped (s) | 746.2 (matrix, direct load); 524.5 (weight-store run, store hit) | **520.2** (store hit; DiT load 217.4 s, umT5 172.5 s) | −30.3% vs the matrix; −0.8% vs the store run |
| | compile, checkpoint in the page cache (s) | 314.2 (no store then) | 518.9 (DiT load 220.2 s, of which store writes 195–215 s per rank) | +65% (open issue 1) |
| | compile, page cache as left (s) | — | 1115.1 (DiT read cold + store write: load 653.3 s; umT5 172.7 s) | |
| | HBM alloc / nmon per DiT core; VAE core (GiB) | 13.58 / 15.15; 18.44 | 13.58 / 15.15; 18.51 | 0 |
| tp4 `--sp` | step (ms) | 1042.1 / 1041.1 / 1044.4 | 1043.7 / 1042.5 / 1046.3 | +0.15% |
| | e2e warm (s) | 129.1, 126.8 (128.0) | 124.3, 126.0 (125.2) | −2.2% |
| | compile (s) | 291.1 | 956.0 (DiT load 653.6 s: cold read + store write) | not like-for-like |
| | HBM (GiB) | 13.48 / 14.68 | 13.51 / 14.70 | 0 |
| tp2cfg2 | step (ms) | 931.3 / 930.2 / 934.5 | 932.5 / 931.3 / 935.9 | +0.13% |
| | e2e warm (s) | 129.7, 131.2 (130.4) | 133.7, 135.7 (134.7) | +3.3% (*inference:* launch jitter, see below) |
| | compile (s) | 295.2 | 561.8 (DiT load 278.2 s incl. writing the tp2 entry) | not like-for-like |
| | HBM (GiB) | 14.13 / 15.59 | 14.13 / 15.59 | 0 |
| tp2cp2 ulysses | step (ms) | 1154.0 / 1152.9 / 1156.5 | 1155.4 / 1154.3 / 1158.8 | +0.12% |
| | e2e warm (s) | 133.9, 134.6 (134.3) | 132.5, 133.5 (133.0) | −0.9% |
| | compile (s) | 262.0 | 752.8 (DiT load 480.1 s) | not like-for-like |
| | HBM (GiB) | 13.89 / 14.95 | 13.87 / 14.94 | 0 |
| dp2tp2, 4 requests | step per tp2 replica (ms) | 1852.1 | 1854.6 (per-request medians 1851.1–1859.7) | +0.1% |
| | e2e for 4 requests (s) | 294.1, 295.2 (294.6; 49 requests/h) | 288.0, 292.9 (290.4; 49.6 requests/h) | −1.4% |
| | compile (s) | 280.7 | 509.2 (DiT load 228.2 s) | not like-for-like |
| | HBM (GiB) | 14.12 | 14.12 | 0 |
| tp4 `--host-text-encoder --host-vae` (golden placement) | e2e (s) | 161.8, 161.3 | 152.5 (n=1: gen0, right after the tp4 cell) | — |

**Where the warm e2e goes** (tp4, gen1, seconds; the matrix's breakdown in brackets, *reused*):

| part | seconds |
|---|---|
| process wall | 125.6 [125.8] |
| transformer stage | 103.1 [103.3] |
| — rank 0, python start to load start | 21.7 [22.2] |
| — load + warm-up | 22.4 [22.4] (DiT shards 4.9 from the store, warm-up 12.9; umT5 shards 2.4, warm-up 0.7) |
| — prompt encode (device umT5) | 0.8 [0.75] |
| — denoise | 53.3 [53.2] |
| vae stage | 22.2 [22.2] |

### Qwen-Image (1024x1024, 50 steps)

| cell | metric | before | now | delta |
|---|---|---:|---:|---:|
| tp4 default (device VAE) | step (ms) | 351.6 / 350.8 / 352.4 (the host-decode headline: 351.4) | 352.6 / 351.9 / 354.0 (runs 352.6, 352.4, 352.6) | +0.3% |
| | e2e warm (s) | 79.4 (mean of 4) | 82.2, 78.4, 78.3 (mean 79.6) | +0.3% |
| | e2e cold (s) | 492.0 | 496.1 (DiT load 310.9 s, host Qwen2.5-VL 99.9 s) | +0.8% |
| | compile, checkpoint in the page cache (s) | 155.2 | 161.7 (generate stage 68.4 + vae stage 93.0) | +4.2% |
| | compile, page cache as left (s) | — | 451.7 (DiT load 199.0 s) | |
| | weight store | — | nothing written ("no gain", bf16 checkpoint) | |
| | HBM alloc / nmon DiT core; VAE core (GiB) | 10.15 / 10.76; 13.45 | 10.15 / 10.76; 13.83 | 0 / 0; +0.38 (*inference:* 1 s sampling of the eager decode) |
| tp4 `--sp` | step (ms) | 328.9 / 328.2 / 329.9 | 329.9 / 329.3 / 331.0 | +0.3% |
| | e2e warm (s) | 78.7, 80.0 (79.4) | 76.2, 73.8 (75.0) | −5.5% |
| | compile (s) | 68.6 (no vae stage then) | 154.8 (62.1 + vae 92.5) | generate stage −9.5% |
| | HBM (GiB) | 9.93 | 9.93 | 0 |
| tp2cp2 ulysses | step (ms) | 418.4 / 417.9 / 419.7 | 419.5 / 419.0 / 420.7 | +0.3% |
| | e2e warm (s) | 87.9, 79.5 (83.7) | 84.0, 85.5 (84.8) | +1.3% |
| | compile (s) | 77.5 (no vae stage) | 163.8 (71.6 + 92.0) | generate stage −7.6% |
| | HBM (GiB) | 19.75 | 19.72 | 0 |
| true CFG tp2cfg2 | step (ms) | 621.4 / 620.5 / 623.3 | 622.4 / 621.6 / 624.3 | +0.2% |
| | e2e warm (s) | 103.6, 105.5 (104.6) | 101.4, 101.8 (101.6) | −2.8% |
| | compile (s) | 79.5 (no vae stage) | 175.2 (82.5 + 92.5) | generate stage +3.8% |
| | HBM (GiB) | 19.81 | 19.81 | 0 |
| dp2tp2, 4 requests | step per replica (ms) | 615.5 | 615.9 (615.6–616.4) | +0.1% |
| | e2e for 4 requests (s) | 149.0, 147.2 (148.1; ~97 requests/h) | 162.7, 160.7 (161.7; 89 requests/h) | **+9.2%** (*inference:* launch jitter, see below) |
| | compile (s) | 72.4 (no vae stage) | 159.7 (67.4 + 92.0) | generate stage −6.9% |
| | HBM (GiB) | 19.81 | 19.81 | 0 |

The vae stage compile (about 92 s) is in every "now" compile because the device VAE has been the default since `486ca61`, which predates the refactor. The matrix's SP, CP, true-CFG and DP cells ran before that flip.

**The dp2tp2 +9.2% is launch time, not device time.** The step, the load (31–32 s per replica) and the denoise (30.8 s per request) are unchanged. What moved is the time from a rank's python start to its load start:

- **Before:** one replica took 9.4–9.9 s and the other 20.3–20.5 s. The fast replica finished 15 s earlier, so the two vae stages did not overlap.
- **Now:** both replicas took 19.7–21.3 s, and the two vae stages overlapped: they took 17.5 and 28.2 s, against 17.7 and 18.1 s before.

The same 9-to-21 s bimodality appears in the pre-refactor runs: in `dp2tp2-postflip/gen2` (`486ca61`, 153.0 s) both replicas took 20.4 s. It also appears in single-replica cells now: LTX-2 tp4 gen1 took 27.0 s against 19.8 s in gen3. This is the matrix's open issue 4 (per-launch jitter). *Inference:* not caused by the refactor.

**Where the warm e2e goes** (tp4, gen2, seconds; the matrix's breakdown in brackets, *reused*):

| part | seconds |
|---|---|
| process wall | 78.4 [79.5] |
| generate stage | 60.2 [60.7] |
| — rank 0, start to load | 21.7 [23.4] |
| — load + warm-up | 15.3 [14.3] |
| — encode | 0.5 [0.50] |
| — denoise | 17.6 [17.6] |
| vae stage | 18.0 [18.6] |

### LTX-2 (480x704x49, 20 steps)

| cell | metric | before | now | delta |
|---|---|---:|---:|---:|
| tp4 default (device Gemma-3, host decode) + weight store | step (ms) | 350.1 / 349.1 / 351.5 (the host-Gemma headline: 349.3) | 349.7 / 348.7 / 350.9 (runs 349.7, 350.0, 349.6) | −0.1% |
| | e2e warm (s) | 77.4, 78.8 (78.1) | 84.8, 80.4, 77.2 (mean 80.8) | +3.5% (the 84.8 s run: rank start-to-load 27.0 s against 19.8–22.9 s; launch jitter) |
| | e2e cold, page cache dropped (s) | 758.0 (`--device-text-encoder --device-vae`, direct load); 745.1 / 574.3 (weight-store run: direct / store) | **576.9** (Gemma-3 from the store, 178.7 s; DiT 287.3 s) | −23.9% vs 758.0; +0.5% vs the store run |
| | compile, checkpoint in the page cache (s) | 119.3 (no store then) | 306.6 (Gemma-3 load 181.6 s, of which store writes 142–178 s per rank) | +157% (open issue 1) |
| | compile, page cache as left (s) | — | 935.9 (DiT 287.3 s cold, Gemma-3 538.1 s cold + write) | |
| | HBM alloc / nmon peak (GiB) | 15.66 / 16.82 | 15.66 / 16.82 | 0 |
| tp2cfg2, g=4 | step (ms) | 611.6 / 610.1 / 613.1 | 611.2 / 609.3 / 612.7 | −0.07% |
| | e2e warm (s) | 142.0, 143.6 (142.8) | 139.0, 141.4 (140.2) | −1.8% |
| | compile (s) | 140.0 | 497.6 (host Gemma-3 read from disk: 386.4 s against 29.4 s before) | not like-for-like |
| | HBM (GiB) | 18.15 / 18.75 | 18.15 / 18.75 | 0 |
| tp2cp2 gather_kv | step (ms) | 416.8 / 415.7 / 417.7 | 417.0 / 416.1 / 418.9 | +0.04% |
| | e2e warm (s) | 112.4, 114.1 (113.2) | 107.6, 119.0 (113.3) | +0.1% |
| | compile (s) | 139.8 | 131.0 | −6.3% |
| | HBM (GiB) | 18.04 / 18.63 | 18.04 / 18.63 | 0 |
| tp2cp2 ulysses (Part A cell) | step (ms) | 412.7 / 411.9 / 414.3 | 411.6 / 410.8 / 413.2 (gen1; gen0 412.1) | −0.3% |
| | e2e warm (s) | 119.3, 118.4 (118.8) | 113.2 (n=1; gen0 117.4) | −4.8% |
| | compile (s) | 128.9 | 133.4 | +3.5% |
| dp2tp2, 4 requests | step per replica (ms) | 601.8–602.3 | 602.1 (601.7–602.6) | 0 |
| | e2e for 4 requests (s) | 751.3 (n=1; gen0 747.5) | 423.3, 466.1 (444.7; gen0 256.6) | −41% (*inference:* page-cache state, see below) |
| | compile at tp2 (s) | 674.6 (the dp1tp2 cell) | 130.3 | page-cache state |
| | HBM (GiB) | 18.13 | 18.13 / 18.69 | 0 |

**LTX-2 dp2tp2 is host-memory and page-cache bound, not code bound.** Two replicas each hold the host Gemma-3 (61 GB tree RSS peak, the same as before), which leaves about 68 GB of page cache (`Cached` at the end of each run) for 80.6 GiB of DiT and Gemma-3 shards. What gets evicted decides the run time:

- **Before:** both replicas read Gemma-3 from disk, 418 s each.
- **Now, gen0:** 33 / 68 s.
- **Now, gen1:** Gemma-3 63–65 s, DiT 142–154 s.
- The step and the per-request outputs are identical.

*Inference:* the faster runs reflect what earlier cells left in the cache, not the code.

**Where the warm e2e goes** (tp4, gen2, seconds; the LTX campaign's device-Gemma breakdown in brackets, *reused*):

| part | seconds |
|---|---|
| process wall | 80.4 [77.4] |
| generate stage | 61.5 [58.1] |
| — rank 0, start to load | 22.9 [18.7] |
| — load + warm-up | 21.2: DiT 5.4 + 8.4 [5.3 + 8.7]; Gemma-3 2.9 from the store + 1.0 [4.5 in all] |
| — encode | 5.4 [5.5] |
| — denoise | 7.0 [7.0] |
| host decode in the driver | 17.8 (decode 13.8) [18.2; decode 13.9] |

### Final: native vs AoT

Updated 2026-10-20. The AoT column is now AoT with `NEURON_RT_VIRTUAL_CORE_SIZE=2` traced in: profiling found that Wan2.1's and LTX-2's AoT CLIs traced `attention_cte` on one of the two physical cores (`difflet/backends/trainium/ops_impl/attention.py` picks the two-core kernel only with that variable set); Qwen-Image's AoT CLI already set it. The fix (commits `5f62732` Wan, `62e3475` LTX-2) lives on the AoT experiment branch `feat/aot-bf16-allreduce`, not in this PR. Its outputs are **bit-identical** to stock AoT (Wan mp4 `d2ecce24…`, latents `f3dd7d2e…`; LTX-2 latents `e597cc5d…`, mp4 `396b8947…`). LTX-2 now runs **50 steps** (the CLI default is 40; the earlier 20 steps copied the AoT benchmark), native included.

Measured 2026-10-20 on this host with the same probe and method: AoT Wan two-core and both LTX-2 AoT cells at commits `62e3475` (two-core) and `026376a` (stock), native LTX-2 at `026376a`. Compile times for the two-core cells are from fresh roots on 2026-10-19 (same code); native compile and the native Wan and Qwen-Image rows are this report's runs; Qwen-Image AoT is reused from the A/B at `03f3a84`. Native is the default placement with the weight store. "Compile" is the cold compile with the checkpoint in the page cache.

| model | metric | native | AoT, two-core attention | native vs AoT | AoT stock |
|---|---|---:|---:|---:|---:|
| Wan2.1 14B, tp4, 480x832x9, 50 steps | DiT step | 1063.1 ms | 957.1 ms | +11.1% | 1151.5 ms |
| | compile cold | 518.9 s | 303.2 s | +71% | 482.0 s |
| | e2e warm | 125.5 s | 123.1 s | +2.0% | 134.5 s |
| | e2e cold | 520.2 s | 450.0 s | +15.6% | 461.8 s |
| Qwen-Image, tp4, 1024x1024, 50 steps | DiT step | 352.6 ms (329.9 with `--sp`) | 418.4 ms (reused; already two-core) | −15.7% (−21.1%) | same |
| | compile cold | 161.7 s | 1480 s | −89% | same |
| | e2e warm | 79.6 s | 76.9 s | +3.5% | same |
| | e2e cold | 496.1 s | 505.9 s | −1.9% | same |
| LTX-2, tp4, 480x704x49, 50 steps, g=1 | DiT step | 350.7 ms | 419.2 ms | −16.3% | 461.4 ms |
| | compile cold | 306.6 s | 576.9 s | −47% | 1838.9 s (939.1 s in another run) |
| | e2e warm | 90.8 s | 68.0 s | +33.5% | 71.8 s |
| | e2e cold | 588.3 s | 781.3 s | −24.7% | 785.5 s |

What the table says:

- **Two-core attention takes 16.9% off AoT Wan's step (1151.5 → 957.1 ms) and 9.1% off LTX-2's (461.4 → 419.2 ms), bit for bit.** AoT Wan is now 11.1% faster than native per step; native stays 16% faster for LTX-2 and Qwen-Image. Profiling attributes the remaining LTX-2 / Qwen-Image gap to AoT's older compiler (2.26), its fp32 all-reduce and per-block work not yet explained (`ltx_2_neuron.md`, the Qwen-Image profile); Wan's AoT modeling already all-reduces in bf16.
- **The weight store fixed Wan's cold start** (746 → 520 s) and cut LTX-2's (758 → 588 s at 50 steps). Native Wan is still 15.6% behind two-core AoT cold; 172.5 s of its cold load is the fp32 umT5 off disk, which the store does not cover because an fp32 store entry saves no bytes.
- **The store's one-time write moved compile.** Wan's native compile is now slower than AoT's (open issue 1). Qwen-Image is unaffected (nothing written).
- **Warm e2e:** native trails AoT on LTX-2 (+22.8 s at 50 steps) and Qwen-Image (+2.7 s) for the host-side and launch reasons the earlier reports attributed, and is within 2% on Wan.
- **LTX-2 image quality at guidance 1 is weak on both backends** (no CFG; the setting copies the AoT benchmark). Guidance 3–4 is the practical setting; per-step time doubles with CFG on both.

## Bugs fixed

None. No check failed, so this campaign made no fix commit.

## Unit tests

Native venv, with the device idle: `tests/unit/backends tests/unit/benchmark tests/unit/cli -k "neuron or wan or qwen or ltx"`: **1340 passed**, 503 deselected, 0 failed (870 s; `F/tests/unit.log`). The S6 file: 6 passed.

## Open issues

1. **A cold compile on a fresh root now pays the weight-store write, even when the checkpoint is in the page cache.**
   - **Cost on this host (EBS at ~130 MB/s, writes disk-bound):**
     - Wan tp4: the DiT's 28.6 GB took 195–215 s per rank (4 ranks in parallel, ~140 MB/s aggregate), so compile went from 314.2 to 518.9 s.
     - LTX-2 tp4: Gemma-3's 23.5 GB took 142–178 s, so compile went from 119.3 to 306.6 s.
     - Each tp2 layout writes another entry: Wan tp2cfg2 "written 14.52 GB in 195.4 s".
   - **What it buys:** it is paid once per (model, revision, TP layout, store root). Every later cold launch then reads half the bytes (Wan cold e2e 746 → 520 s).
   - **What makes it worse:** the 28.6 GB of written pages also evict the checkpoint from the page cache. When the checkpoint was read only once beforehand (`tp4-pcwarm`, not used in the tables), the DiT load took 644.6 s and umT5 172.7 s, about the cold time.
   - **Options** (not tried; *inference*):
     - write the entry after the warm-up, or in the background;
     - write it on the first generate rather than in `compile`;
     - document `DIFFLET_NEURON_WEIGHT_STORE=0` for one-shot compiles.
   - Not a correctness bug, so not fixed here.
2. **Launch jitter of 9–27 s from a rank's start to its load start** (the matrix's open issue 4). It accounts for the +9.2% Qwen-Image dp2tp2 and the LTX-2 84.8 s warm run. It shows the same 9-to-21 s bimodality in pre-refactor runs.
3. **LTX-2 dp2tp2 depends on the page-cache state** (256–466 s for 4 requests now, 747–751 s before), because two host Gemma-3 copies leave too little page cache for the shards. A device Gemma-3 does not fit at tp2 (28.5 GiB per core).
4. **Compile numbers depend on the page-cache state** (Wan 519–1115 s, LTX-2 307–936 s on fresh roots). A page-cache-warm compile has to be set up deliberately on this 124 GB host: drop the cache, then read the checkpoint twice. Two of the like-for-like rows above used exactly that set-up.
5. **The fp32 umT5 cold load (172.5 s) and the host encoders** (Qwen2.5-VL 99.9 s cold) are still not covered by the store. Wan cold e2e is still +12.6% against AoT.
6. **Still open from the earlier reports, not re-run:**
   - Wan 81 frames (OOM at the default placement; warm-up time);
   - the native warm e2e gap to AoT on LTX-2 (+24 s, host-side);
   - Wan host VAE decode speed;
   - Wan 1.3B and 2.2 (no weights on the host).
7. **AoT was not re-measured on `61022ec`.** All AoT numbers are reused.
8. **Disk.** The fresh roots of this campaign hold about 282 GB, almost all of it weight-store entries:
   - Wan: `final-wan-tp4` 54G (tp4 + tp2 entries), `final-wan-tp4-sp`, `-tp2cp2-ulysses`, `-dp2tp2`, `-tp4-pcwarm`, `-tp4-pcwarm2` 27G each;
   - LTX-2: `final-ltx-tp4`, `-tp4-pcwarm`, `-tp4-pcwarm2` 22G each.
   - Free space at the end: 346G of 1.5T.
   - Nothing was deleted. Candidates if space is needed: the `-pcwarm*` roots and the tp2 Wan roots (`tp2cp2-ulysses`, `dp2tp2`), plus the weight-store campaign's scratch roots (`sstore-wan-lazy` 27G, `sstore-qwen` 39G).

## Reproduction

```bash
# per cell (native venv on PATH, this worktree on PYTHONPATH); DIFFLET_NEURON_PG_TIMEOUT_S unset
DIFFLET_COMPILE_CACHE=<fresh root> HF_HUB_OFFLINE=1 python -m difflet.cli.main compile --backend neuron \
  --model-id <id> --revision <rev> <shape> <cell flags> --cache-dir <fresh root>
DIFFLET_COMPILE_CACHE=<same root> HF_HUB_OFFLINE=1 python -m difflet.cli.main generate --backend neuron \
  --model-id <id> --revision <rev> <shape> <cell flags> --steps <50|20> --guidance-scale <5.0|4.0|1.0> \
  --seed 42 --prompt "a cinematic shot of a red fox running through a snowy forest" --output <out> \
  --cache-dir <same root>
```

- **Shapes:** Wan `--height 480 --width 832 --num-frames 9`; Qwen-Image `--height 1024 --width 1024`; LTX-2 `--height 480 --width 704 --num-frames 49`.
- **Cell flags:**
  - `--tp-degree 4`;
  - `--tp-degree 4 --sp`;
  - `--tp-degree 2 --cfg-parallel` (Qwen-Image also `--true-cfg-scale 4.0` on generate; LTX-2 `--guidance-scale 4.0`);
  - `--tp-degree 2 --cp-degree 2 --cp-mode ulysses|gather_kv`;
  - `--tp-degree 2 --dp 2 --requests <jsonl>` (LTX-2 compiles at `--tp-degree 2`).
- **Harness:** `/home/ubuntu/bench-neuron-final/scripts/` (`cell.sh`, `cli.sh`, `q1.sh` … `q7.sh`, the probe, `table.py`, `parse.py`, `breakdown.py`, `pagecache.py`, `c3.sh`, `preflight.sh`).

## Toolchain

The native venv, unchanged from the earlier reports:

- Python 3.14.8, torch 2.14.0+cpu, torch-neuronx 2.14.3.0.0, neuronx-cc 2.28.4568.0, nki 0.7.0;
- diffusers 0.38.0, transformers 4.57.6;
- host runtime: aws-neuronx-runtime-lib 2.35.27.0, collectives 2.35.40.0.
