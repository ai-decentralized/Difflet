# Task 3 Verification — One resident worker serving K shapes

**Status: VERIFIED on device 2026-08-16 03:51 UTC.**

Branch: `feat/bucketed-artifacts` · Host: trn2.3xlarge · Model: HunyuanVideo tp4 bf16
Profile under test: `--shapes 320x512x61,320x512x33` (A = 61f priority, B = 33f; Neuron VAE,
`--request-timeout 900`)
Scratch: `/home/ubuntu/.claude/jobs/e22aa8d9/tmp/` (`$LOG`; logs `t3_*.log`, server log `t3_serve.log`)

## What changed

- `ServingProfile.shapes` (Task 2) is now driven end-to-end: `difflet serve --shapes HxWxF,...`
  (CSV, same syntax as compile) → `ServeOptions.shapes` → `build_serving_profile` (parses,
  canonicalizes largest-first, pins profile h/w/f to the largest shape; HunyuanVideo-video-only
  for now) → `_build_denoiser`/`_build_vae_decoder` construct the app with the shape set → ONE
  bucketed denoiser artifact, ONE resident weight copy.
- **Strict membership validation**: `validate()` replaces the exact-profile equality check with
  membership in `profile.shape_set()`; misses get `profile_mismatch` naming the allowed set
  (gating BEFORE the NxD router, which would raise a raw ValueError on unknown shapes).
- **All-bucket warmup**: multi-shape profiles load the denoiser with warmup enabled — the
  application's warmup loop runs one forward per bucket at startup, so the first request at ANY
  profile shape sees steady-state latency. Single-shape profiles keep the previous skip-warmup
  behavior (zero regression).
- `_validate_profile` checks every shape in the set (divisible-by-16, 4n+1 frames) and enforces
  the "profile h/w/f == largest shape" invariant.
- Runtime dispatch needs no new code: the denoiser stage already builds latents at the request
  shape and the NxD router selects the bucket by exact input signature.

## Unit tests

`tests/unit/serving/test_multi_shape_serving.py` (8 tests): canonical set helpers, per-shape
profile validation + largest-pin invariant, validator accepts every member shape / rejects
out-of-set with the allowed list, denoiser identity covers the set (order/dup invariant; llama
identity unchanged across sets), `--shapes` rejected for non-HunyuanVideo serving.

## On-device verification (`bash $LOG/task3_verify.sh`)

| check | expectation | result |
|---|---|---|
| server start | one worker; warmup covers BOTH buckets; one weight load per component | ✅ healthy in 200 s (artifacts cached); denoiser weights loaded once (40.2 s), llama once (13.5 s) |
| request A (61f) then B (33f) | both succeed from the same worker, no restart/reload between | ✅ HTTP 200 / 200 |
| repeat A and B | warm latencies; shape switch costs no reload | ✅ latencies identical to the decisecond (below) |
| out-of-set 320×512×45 | HTTP 4xx `profile_mismatch` naming the allowed set | ✅ 400: `request shape 320x512x45 is not in the serving profile's compiled shape set ['320x512x61', '320x512x33']` |
| outputs | mp4 payloads for both shapes | ✅ `$LOG/t3_{a1,b1,a2,b2}.out` (A: 40,581 B; B: 23,594 B; same-seed repeats byte-identical in size) |

### Latency table (seed 42, 4 steps)

| request | shape | wall time | note |
|---|---|---|---|
| a1 | 320×512×61 | **20.9 s** | first A after warmup |
| b1 | 320×512×33 | **12.5 s** | first B — shape switch, same worker, no reload |
| a2 | 320×512×61 | **20.9 s** | warm repeat — equals a1 ⇒ a1 paid no cold cost |
| b2 | 320×512×33 | **12.5 s** | warm repeat — equals b1 |

Context: the staged CLI takes ~4 min per generation (per-process weight reloads); the resident
multi-shape worker serves either shape in 12–21 s. This is the end-to-end payoff of one weight
residency + K NEFFs.

### Issues found & fixed during verification

1. The multipart transport normalizer had its own exact-profile shape check
   (`serving_video.py`), which 400-ed every non-largest member shape before the adapter validator
   ran — replaced with membership against `profile.shape_set()` (single-shape profiles keep
   exact-match; profile doubles without the helper degrade gracefully).
2. First run used `--host-vae`: CPU fp32 decode of 61 frames blows the 300 s request timeout
   (worker pinned in `stage=decoder` for minutes). Multi-shape serving verification uses the
   Neuron tile decoder; if you must host-decode, raise `--request-timeout`.
3. A killed serve can orphan its worker subprocess, which keeps holding the NeuronCores and makes
   the next start fail with "PyTorch Neuron Runtime could not be initialized" — check
   `neuron-ls --json-output` for stale processes and kill them (the verify script now does a
   pre-flight check and a hard cleanup).

## Manual repro

```bash
cd /home/ubuntu/Difflet
PATH=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:$PATH \
difflet serve --model-id hunyuanvideo-community/HunyuanVideo \
  --shapes 320x512x61,320x512x33 --host-vae --port 8091
# other terminal:
curl -s -X POST http://127.0.0.1:8091/v1/videos/sync \
  -F model=hunyuanvideo-community/HunyuanVideo -F "prompt=a red cube on a white table" \
  -F height=320 -F width=512 -F num_frames=33 -F fps=24 \
  -F seed=42 -F num_inference_steps=4 -F guidance_scale=6.0 -o /tmp/b.mp4 -w '%{http_code}\n'
# out-of-set rejection:
curl -s -X POST ... -F num_frames=45 ...   # expect 4xx profile_mismatch with the allowed set
```
