# Feature catalog — what each cell means, how it runs, what to expect

Everything here was device-verified on trn2.3xlarge (4 NeuronCores, LNC=2, 96 GiB HBM) with
neuronx-cc 2.26. Model keys are `scripts/verify_cli.py` keys; CLI model ids in parentheses.

## Models and their verification shapes (verify_cli.py MODELS)

| key | model id | shape | steps | pipeline |
|---|---|---|---|---|
| `flux` | black-forest-labs/FLUX.1-dev | 1024×1024 | 28 | single process |
| `qwen_image` | Qwen/Qwen-Image | 1024×1024 | 50 | staged: text → generate → vae |
| `wan` | Wan-AI/Wan2.2-T2V-A14B-Diffusers | 480×832×9f | 40 | staged: transformer → vae |
| `wan2_1` | Wan-AI/Wan2.1-T2V-14B-Diffusers | same as wan | 40 | staged (same runtime) |
| `hunyuan_video` | hunyuanvideo-community/HunyuanVideo | 320×512×121f | 50 | staged: clip → llama → generate |
| `hunyuan_video_15` | …HunyuanVideo-1.5-Diffusers-480p_t2v | — | — | download-only scaffold (XFAIL) |
| `ltx_2` | Lightricks/LTX-2 | 256×384×121f | 40 | Neuron DiT, host prompt/decode |

Staged models report *staged CLI wall time* (three process loads + denoise + VAE), which is not
comparable to a single-process generate. Say which kind of number you are quoting.

## Parallelism configs (verify_cli.py PARALLEL_CONFIGS, all 4 cores)

`tp4` · `tp2cp2` (CP gather_kv) · `tp2cp2ring` · `tp2cp2ulysses` · `tp2cfg` (CFG-parallel, world
= 2×tp) · `tp4sp` (SP reuses the TP group) · `dp2tp2` (2 replicas × tp2, compile-once-load-twice).

Skip / expected-fail rules live in `verify_cli.py` and are pinned by
`tests/unit/cli/test_verify_cli.py`; when a fix changes support, edit both. Sets as of the campaign:

- `DISTILLED` = flux, qwen_image, hunyuan_video, hunyuan_video_15 → `tp2cfg` SKIP (no CFG branch).
- `SP_SUPPORTED` = flux, wan, wan2_1, hunyuan_video → others SKIP `tp4sp` (Qwen deferred by design).
- `CP_UNSUPPORTED` = ltx_2, hunyuan_video_15 → all CP configs SKIP.
- `RING_UNSUPPORTED` = hunyuan_video (always carries an attention mask; ring kernel has no mask path).
- `ULYSSES_UNSUPPORTED` = ∅ since the masked-text bounds route (HV ulysses PASSes).
- `EXPECTED_FAIL_CELLS`: hv15 `tp4`/`dp2tp2` (scaffold); hunyuan `tp2cp2` (neuronx-cc SBUF bug
  `NCC_INLA001` behind the %128 bounds constraint); hunyuan `dp2tp2` (121-frame VAE exceeds a
  2-core replica's HBM); wan/wan2_1 `tp2cp2ring` at the default shape (see ring rule).

### Per-axis expectations and fingerprints

- **TP**: shards = `total/tp` bytes each; `attn.to_q.weight` [3072,3072] → [768,3072] at tp4.
- **CP**: weights split by tp only while 2×tp ranks run (`tp2w4-cp` store entries hold 4 shards of
  total/2); manifest `cp_degree`/`cp_mode`. Ring rule: per-rank latent tokens % 128 == 0
  (nkilib `ring_attention_spmd_fwd`); Wan 480×832×9 → 2340/rank fails fast, 512×512×9 → 1536 passes.
  Ring also needs `NEURON_RT_VIRTUAL_CORE_SIZE=2` on stage subprocesses (wan orchestrator sets it
  for ring only). Ulysses needs heads % (tp·cp) == 0.
- **CFG-parallel**: manifest `cfg_parallel_enabled: true`; true-CFG models only (wan, ltx_2).
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

## TAEF1 (flux only)

`--taef1 --taef1-path madebyollin/taef1` on compile and generate (separate decoder artifact).
Reference: 41.1 s → 34.7 s per 28-step generate at 1024².

## DP correctness (Phase 5)

`PYTHONPATH=$PWD python scripts/verify_dp_correctness.py --model-id black-forest-labs/FLUX.1-dev
--dp 2 --tp-degree 2 --steps 4` → `4/4 identical`. Needs the tp2 artifact of the model.
