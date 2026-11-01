# Feature catalog — what each cell means, how it runs, what to expect

Everything here was device-verified on trn2.3xlarge (4 NeuronCores, LNC=2, 96 GiB HBM) with
neuronx-cc 2.26. Model keys are `scripts/verify_cli.py` keys; CLI model ids in parentheses.

## Verified support snapshot (2026-08-31; HunyuanVideo 1.5 column 2026-11-01; use as the default expectation per cell)

| Feature | FLUX | Qwen-Image | Wan 2.2 | HunyuanVideo | HunyuanVideo 1.5 | LTX-2 |
|---|---|---|---|---|---|---|
| TP | PASS | PASS | PASS | PASS | PASS (TP2, TP4) | PASS |
| CP gather_kv | PASS | PASS | PASS | BLOCKED (NCC_INLA001 behind %128 bounds) | PASS (watch: 3-step 1.25x) | N/A |
| CP ring | PASS | PASS | LIMIT (%128; PASS @512×512×9) | N/A (mask) | N/A (mask) | N/A |
| CP ulysses | PASS | PASS | PASS | PASS (bounds route) | PASS | N/A |
| SP | PASS | N/A (deferred) | PASS | PASS | PASS (refused @480×848×121: N%4) | N/A |
| CFG-parallel | N/A (distilled) | N/A | PASS | N/A | PASS (default TP2×CFG2) | PASS |
| DP router | PASS | PASS | PASS | BLOCKED (HBM @121f) | PASS | PASS |
| DP bit-identity | PASS | NOT MEASURED | NOT MEASURED | BLOCKED | PASS (4/4) | NOT MEASURED |
| Bucketed `--shapes` compile | PASS | PASS | PASS | PASS (earlier) | N/A (one shape; refused) | LIMIT (flag ignored) |
| Shared weight store | PASS (proof) | PASS | PASS (proof) | PASS | NOT MEASURED | PASS |
| Serving | PASS chat | PASS chat (cp=1) | PASS videos | PASS videos | N/A (not wired) | PASS videos (host VAE) |
| Multi-shape serving | PASS | PASS | PASS | PASS (earlier) | N/A | N/A |
| TeaCache fixed cadence | PASS | PASS | N/A (CLI) | N/A (CLI) | PASS (parity with native) | N/A (CLI) |
| TeaCache adaptive | PASS | LIMIT (weak signal) | N/A (CLI) | N/A (CLI) | LIMIT (weak signal; not recommended) | N/A (CLI) |
| TAEF1 | PASS | N/A | N/A | N/A | N/A | N/A |

Anything that lands differently from this table in a new campaign is a finding: either a
regression, a fix that upgraded a cell, or a NOT MEASURED cell finally measured — call it out.

The table above is the Trainium AoT backend (NxD stages). The PyTorch-native **neuron** backend
(`--backend neuron`, torchrun, one process per NeuronCore, per-block `torch.compile`, neuronx-cc
2.28) is a separate runtime with its own rows below; do not carry a cell over between the two.

### Trainium AOT backend: HunyuanVideo 1.5 (verified 2026-10-30 to 2026-11-01, trn2.3xlarge)

`difflet compile|generate --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`
(trainium backend) runs three stages: `encode` (host: Qwen2.5-VL + ByT5, `embeds.pt`),
`aot_transformer` (one NxD process on all `tp × cp (×2 with CFG)` cores, `latents.pt`) and
`aot_vae` (host decode by default, diffusers' tiled fp32 decoder; `--device-vae` opts into the
compiled `vae15`, which does not build on neuronx-cc 2.26). True CFG at 6.0 on every step; default
layout TP2×CFG2. Evidence: `docs/verification/hv15-aot-evidence.md` (Summary at the top).

| Feature | AOT backend, HunyuanVideo 1.5 (320×512×61, 50 steps unless stated) |
|---|---|
| TP2 / TP2×CFG2 / TP4 / TP4+SP | PASS (1-step cos vs native 0.99912-0.99916; 2690 replica / 1573 / 1731 / 1601 ms per step; native TP2×CFG2 1325 ms) |
| CP ulysses / gather_kv (TP2×CP2) | PASS (1783 / 1858 ms per step). gather_kv watch item: 3-step distance to diffusers 1.25x native's (gate 1.15x), deterministic |
| CP ring | N/A (mask): refused before encode (`--cp-mode ring`) |
| DP / JSONL batch | PASS (`verify_dp_correctness.py` 4/4 bit-identical; `--requests` loads the DiT once, each request bit-identical to its single run) |
| TeaCache cadence 2 / online-delta 0.6 | PASS (20 / 17 skipped, -41% / -35% denoise; cos 0.939 / 0.918 vs no TeaCache; native cadence 2 is 0.932, so 0.99 is not reachable on either stack) |
| TeaCache adaptive | LIMIT, not recommended (calibration R² 0.41, Pearson −0.12, visibly degraded; the probe doubles per-core HBM and does not fit at 480×848×121). `--requests` refuses TeaCache flags |
| Default shape 480×848×121 (TP2×CFG2) | PASS: compile 3970 s (host RAM peak 116.5/124 GiB), 13072 ms per step (native 11178); `--tp-degree 4 --sp` refused before compile (N % 4 = 2) |
| VAE | host decode LIMIT (409-431 s @61f, 2699 s @121f vs native device 31 / 89 s); device `vae15` BLOCKED (NCC_EVRF007 / NCC_IBIR229 / NCC_EXTP003) |

Gotchas: the default-shape DiT compiles only without `--enable-ccop-compute-overlap` (NCC_IRPX901);
Difflet drops the option above 16384 image tokens and keys the artifact on the compiler args.
The AOT and native stacks draw different seed-42 noise (torch 2.9.1 vs 2.14), so cross-stack
latents are not comparable from CLI runs; gate parity with `scripts/hv15_aot_denoise_from_inputs.py`
on the native run's `embeds.pt` and noise. Native reference runs need the `native-312` venv (the
AOT venv lacks the `neuron` process-group backend).

### Neuron backend: HunyuanVideo (verified 2026-10-24/25, trn2.3xlarge)

`difflet generate --backend neuron --model-id hunyuanvideo-community/HunyuanVideo` runs two
launches: `generate` on tp×cp cores (Llama/CLIP encode on rank 0's host, denoise, latents.pt),
then a one-core `vae` launch (diffusers' decoder, fp32, eager, tiled; the default since
`d6ffb9a`; `--host-vae` decodes on the host instead). Floor TP2 (tp1 is refused before torchrun:
the 23.9 GiB DiT does not fit one core). Evidence: `docs/verification/hunyuan-video-neuron-evidence.md`
(parity) and `benchmark/trn2/hunyuan_video_neuron.md` (timings).

| Feature | neuron backend, HunyuanVideo |
|---|---|
| TP2 / TP4 | PASS (CLI 320×512×61, 50 steps: 1461.8 / 808.3 ms per step, e2e 300.9 / 239.1 s before `f4d9ac1`; TP4 222.5 s after) |
| CP gather_kv (TP2×CP2) | PASS (CLI: 905.1 ms per step, e2e 250.0 s; e2e parity 1.021x) |
| CP ulysses / ring (TP2×CP2) | PASS at DiT level (truncated-depth parity, compiled mask bounds); CLI NOT MEASURED. Ring takes the valid-text count as `text_valid_len` |
| SP | PASS (TP2+SP DiT parity at truncated depth; at full depth on the t=900 oracle 1.190x, like every compiled layout; TP4 `--sp` CLI: no gain, 809.3 ms) |
| CFG-parallel | N/A (guidance-distilled; refused) |
| DP / serving / multi-shape | NOT MEASURED (out of the plan's scope) |
| TeaCache fixed cadence | PASS (`--teacache-cadence 2`: 20/50 steps skipped, latents cos 0.99984) |
| TeaCache adaptive | NOT MEASURED (no HunyuanVideo calibration exists) |
| DiT parity, full depth | LIMIT: compiled layouts 1.19x the bf16 control on the t=900 / 20-token input (gate 1.15x), 0.84–1.05x on 7 other inputs; eager 1.006x; gate decision pending |
| e2e parity (3 steps, latents) | PASS (TP2 1.003x, TP2×CP2 gather_kv 1.021x; host encoder equal to diffusers') |
| Device VAE | PASS fp32 (default; 3.5e-7 vs host fp32); bf16 BLOCKED by accuracy (20.8 dB, banding) |
| 720×1280×129 (TP4) | BLOCKED on trn2.3xlarge: after `f4d9ac1`/`15b503f`/`ce9d212` the block graph's neuronx-cc compile is OOM-killed at 109.8 GiB of the host's 124.8 GiB RAM; no forward ran |

Gotchas: a `pytest` process in the native venv held `/dev/neuron0` in this campaign (*inference:*
importing torch-neuronx opens it), so never run unit tests while a device job runs or is about to
start. The venv's `bin/` must be on PATH (neuronx-cc is spawned by name: `posix_spawn failed`
otherwise). `difflet compile` with the device vae costs about 48 min cold (29 min eager decoder
compile, then a 927 s exit); the vae stage's NEFFs are layout-independent, so a root that already
holds them only compiles the DiT (about 3 min with the checkpoint in the page cache). At long
sequences the bounded attention kernel's NKI trace dominates a cold compile (at 119k joint tokens:
44 min with 6 heads per call, about 5 min with 1); the NKI trace cache (`DIFFLET_NKI_TRACE_CACHE`,
on for HunyuanVideo since `f4d9ac1`) keeps it for later launches and also takes 23 s off every
320×512×61 launch. Editing `ops_impl/attention.py` recompiles every cached block graph once.

## Models and their verification shapes (verify_cli.py MODELS)

| key | model id | shape | steps | pipeline |
|---|---|---|---|---|
| `flux` | black-forest-labs/FLUX.1-dev | 1024×1024 | 28 | single process |
| `qwen_image` | Qwen/Qwen-Image | 1024×1024 | 50 | staged: text → generate → vae |
| `wan` | Wan-AI/Wan2.2-T2V-A14B-Diffusers | 480×832×9f | 40 | staged: transformer → vae |
| `wan2_1` | Wan-AI/Wan2.1-T2V-14B-Diffusers | same as wan | 40 | staged (same runtime) |
| `hunyuan_video` | hunyuanvideo-community/HunyuanVideo | 320×512×121f | 50 | staged: clip → llama → generate |
| `hunyuan_video_15` | …HunyuanVideo-1.5-Diffusers-480p_t2v | 320×512×61f | 50 | staged: encode → aot_transformer → aot_vae (host) |
| `ltx_2` | Lightricks/LTX-2 | 256×384×121f | 40 | Neuron DiT, host prompt/decode |

Staged models report *staged CLI wall time* (three process loads + denoise + VAE), which is not
comparable to a single-process generate. Say which kind of number you are quoting.

## Parallelism configs (verify_cli.py PARALLEL_CONFIGS, all 4 cores)

`tp4` · `tp2cp2` (CP gather_kv) · `tp2cp2ring` · `tp2cp2ulysses` · `tp2cfg` (CFG-parallel, world
= 2×tp) · `tp4sp` (SP reuses the TP group) · `dp2tp2` (2 replicas × tp2, compile-once-load-twice).

Skip / expected-fail rules live in `verify_cli.py` and are pinned by
`tests/unit/cli/test_verify_cli.py`; when a fix changes support, edit both. Sets as of the campaign:

- `DISTILLED` = flux, qwen_image, hunyuan_video → `tp2cfg` SKIP (no CFG branch). HunyuanVideo 1.5
  runs true CFG (TP2×CFG2 is its default).
- `SP_SUPPORTED` = flux, qwen_image, wan, wan2_1, hunyuan_video, hunyuan_video_15 → others SKIP `tp4sp`.
- `CP_UNSUPPORTED` = ltx_2 → all CP configs SKIP.
- `RING_UNSUPPORTED` = hunyuan_video, hunyuan_video_15 (always carry an attention mask; the Trainium
  ring kernel has no mask path). No ring config is in `PARALLEL_CONFIGS`, so this is a catalog
  rule, not a code set.
- `ULYSSES_UNSUPPORTED` = ∅ since the masked-text bounds route (HV ulysses PASSes).
- `EXPECTED_FAIL_CELLS`: none for hv15 (all six configs PASS on device); hunyuan `tp2cp2` (neuronx-cc SBUF bug
  `NCC_INLA001` behind the %128 bounds constraint); hunyuan `dp2tp2` (121-frame VAE exceeds a
  2-core replica's HBM); wan/wan2_1 `tp2cp2ring` at the default shape (see ring rule).

### Per-axis expectations and fingerprints

- **TP**: shards = `total/tp` bytes each; `attn.to_q.weight` [3072,3072] → [768,3072] at tp4.
- **CP**: weights split by tp only while 2×tp ranks run (`tp2w4-cp` store entries hold 4 shards of
  total/2); manifest `cp_degree`/`cp_mode`. Ring rule: per-rank latent tokens % 128 == 0
  (nkilib `ring_attention_spmd_fwd`); Wan 480×832×9 → 2340/rank fails fast, 512×512×9 → 1536 passes.
  Ring also needs `NEURON_RT_VIRTUAL_CORE_SIZE=2` on stage subprocesses (wan orchestrator sets it
  for ring only). Ulysses needs heads % (tp·cp) == 0.
- **CFG-parallel**: manifest `cfg_parallel_enabled: true`; true-CFG models only (wan, ltx_2,
  hunyuan_video_15).
- **SP**: weights byte-identical to tp4 — SP re-shards activations; proof is the separate store entry
  / artifact hash and the generate-time profile.
- **DP**: `[dp-router] worker 0: cores 0-1` / `worker 1: cores 2-3` in `step_generate.log`; one
  output per replica; `requests.jsonl` + `.claim`/`.done` files; `verify_dp_correctness.py` proves
  byte-identity.
- **HunyuanVideo CP**: only ulysses works (padded Llama text mask expressed as attention_cte bounds).
  gather_kv at cp=2 stacks two blockers: bounds-path seqlen_q % 128 (10176 = 79.5×128 at 121f; now
  fail-fast) and a genuine `NCC_INLA001` SBUF error behind it (control at 125f reaches it).

## Weight sharing / bucketed compile (Phase 3a)

`difflet compile --model-id M --tp-degree 4 --shapes 1024x1024,768x768` (video: `HxWxF,...`), then
`difflet generate ... --shapes <same set> --height H --width W` at each member. The `_shared_weights`
store dedupes pre-sharded checkpoints by (source, dtype, topology, layout tag); bucketed artifacts
hardlink the same files. Proof: `stat -c %h` on `shard0.safetensors` before/after (2 → 3).
Caveat found on device: probe apps (TeaCache) use a nested key namespace and need their own store
entry (`weights_layout_tag`, additive-only key field).

## Multi-shape serving (Phase 3b)

`difflet serve --model-id M --tp-degree 4 --shapes A,B --port 8091`; wait for `/ready`.
Gate: `_MULTI_SHAPE_SERVING_MODELS` = flux, qwen_image (image), wan, hunyuan_video (video); ltx_2 is
single-shape (pinned tp4, host VAE). Requests:

- image: `POST /v1/chat/completions` JSON `{"model", "messages":[{"role":"user","content":"..."}],
  "extra_body": {"height","width","steps","seed"}}` → base64 data URL when no S3 store is configured.
- video: `POST /v1/videos/sync` **multipart form** fields `model, prompt, height, width, num_frames,
  num_inference_steps, seed` → mp4 bytes.
- off-set shape → `400 {"code":"profile_mismatch", "message":"request shape ... is not in the serving
  profile's compiled shape set [...]"}` — that message is the evidence the gate works.

Serving rejects `--cfg-parallel` for all models, `--sp` outside flux/wan/hunyuan, ulysses in
`--cp-mode`, and all TeaCache for video models; qwen serving requires cp = 1.

## TeaCache — two features

**Fixed cadence** (`--teacache-cadence N`, probe-free, no calibration): CLI-wired for flux and qwen
(post-fixes); the pipeline prints `[teacache] stats: {'full_steps': .., 'skipped_steps': .., ...}`.
Expected at cadence 2 with warmup/cooldown 5: 9/28 skips (flux), 20/50 (qwen, −32.7% e2e).
These kwargs are runtime-only (excluded from the compile-cache key) — a warm cache must hit.

**Adaptive** (`--teacache-speedup S --teacache-calibration file.json`): needs the fused probe NEFF and
a calibration. Harnesses do compile → record-only signal gate → fit → same-seed A/B:

- flux: `PYTHONPATH=$PWD python scripts/run_flux_teacache_e2e.py` → Pearson ≈ 0.68, ~1.88×,
  cosine ≈ 0.875, writes `cclogs/m9-teacache/calibration_flux_1024_28step.json`.
- qwen: first `scripts/cache_qwen_calibration_bundles.py --prompts-file
  scripts/data/m9_teacache_prompts.tsv --output-dir .difflet-cache/qwen_image_dit_inputs/m9_calib_50step
  --model-id <HF snapshot root>`, then `run_qwen_teacache_e2e.py --model-dir <snapshot ROOT, not
  /transformer> --transformer-cache ~/.cache/difflet/qwen_image_dit`. With the committed 16-prompt
  set the fit is weak (R² 0.125) and the controller correctly skips 0 — report that as
  "functional, not beneficial; use cadence".

Wan/LTX-2 TeaCache is Python-API-only on main (no CLI wiring); HV's CLI disables its probe.
HunyuanVideo 1.5's AOT CLI wires all three modes (`--teacache-cadence`, `--teacache-online-delta`,
`--teacache-speedup` + a `scripts/calibrate_teacache.py --model hunyuan_video_15` calibration);
use cadence 2, adaptive is not recommended for it (weak signal, second weight copy in HBM).

## TAEF1 (flux only)

`--taef1 --taef1-path madebyollin/taef1` on compile and generate (separate decoder artifact).
Reference: 41.1 s → 34.7 s per 28-step generate at 1024².

## DP correctness (Phase 5)

`PYTHONPATH=$PWD python scripts/verify_dp_correctness.py --model-id black-forest-labs/FLUX.1-dev
--dp 2 --tp-degree 2 --steps 4` → `4/4 identical`. Needs the tp2 artifact of the model.
