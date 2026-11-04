> Port-era notes (2026-08/09, standalone runner). The measured row on the shared
> cold/warm/per-step protocol is [`../flux_1_dev.md`](../flux_1_dev.md) (2026-09-12,
> `benchmark.bench --backend tpu`); the cross-device tables are in
> [`../RESULTS.md`](../RESULTS.md). Kept for the TPU-specific detail
> (what is on the chip, parity, TeaCache A/B, profiling) that the generated report does not carry.

# Benchmark — black-forest-labs/FLUX.1-dev

**Status:** ok
**Backend:** tpu
**Device:** Cloud TPU v5litepod-4 / 4 chips / topology 2x2 / 16 GB HBM per chip
**Timestamp:** 2026-09-12 04:30 UTC

> Configuration: tp=4, bf16, eager. Same `benchmark/models.py::MATRIX` row as
> `benchmark/trn2/flux_1_dev.md` (1024×1024, 28 steps, guidance 3.5, seed 42).
> Runner: `benchmark/flux_tpu_run.py`.

## Configuration

| key | value |
|---|---|
| model type | flux |
| HF revision (pinned) | `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | 1024 × 1024 |
| steps | 28 |
| guidance scale | 3.5 (guidance-distilled, embedded — no CFG) |
| seed | 42 |
| tokens | 4 096 image (128×128 latent, 2×2 packed) + 512 text |

### What is TPU-specific

| key | value |
|---|---|
| execution | eager (lazy XLA), one worker process per chip |
| DiT | diffusers' `FluxTransformer2DModel`, TP-sharded per rank (`difflet/models/flux/tp_sharding.py`; the Trainium path keeps its NxDI fork): **5.445 B params / 10.18 GB per chip** (3.0 B sharded, 2.4 B replicated — the adaLN modulation linears are not sharded); single-block `proj_out` split into two row-parallel halves reduced once |
| text encoders | host, fp32: T5-XXL on XLA ordinal 0 only (its [1, 512, 4096] embeds are broadcast), CLIP-L pooled on every rank |
| VAE | **on the chip** (primary replica only): 1024² decode 0.31 s warm, 117 s first compile |
| denoise loop | device-resident (per-step sigmas/timesteps as device tensors, Euler update in fp32 on the chip); probe-free TeaCache controller keeps its residual on the chip |
| first compile | 44–47 s per rank (DiT, 2-slot gate), host peak 62–73 GB for 4 ranks |

## Results — 1024×1024, 28 steps, tp=4, bf16

| metric | value |
|---|---|
| load (checkpoint → chips 6–7 s, warmup compile 44–47 s, host encoders 0.3–4.9 s) | 52–59 s |
| first request (loop-graph compile) | 36.6 s |
| **e2e to latents** (T5 encode + 28 DiT steps) | **8.72 s** synced, **8.59 s** natural |
| **DiT per-step** (synced, n=27) | **187 ms** |
| DiT per-step (natural) | 152 ms |
| T5-XXL fp32 encode (512 tokens, one rank) / CLIP-L | 3.3 s (the rest of the e2e is 28 × 0.19 s) |
| decode: VAE on chip | 0.31 s warm (117 s first compile) |
| HBM resident / peak (rank 0, DiT + VAE) | 10.18 / 10.45 GB of 15.75 |
| output | 1024² PNG, a red fox in a snowy forest at golden hour (`artifacts/verification-2026-09-11-tpu/flux_1_dev_tpu_bench_baseline.png`); **bit-identical across two separate benchmark runs and both served requests** |
| `difflet serve` request (`/v1/chat/completions`, same settings) | **200 in 10.1 s / 9.3 s**; PNGs bit-identical to each other and to the benchmark's; ready in 216 s |

### TeaCache (probe-free, cadence 2)

| mode | e2e to latents (s) | skipped / full | per full step | image vs. baseline |
|---|---|---|---|---|
| baseline | 8.72 synced / 8.59 natural | 0 / 28 | 187 ms | — |
| `--teacache-cadence 2` | **7.63 synced / 7.09 natural** | 9 / 19 | 188 ms | mean abs 0.0041/px, PSNR 40.7 dB |

28 steps at cadence 2 skip 9 (window [5, 23)); the DiT part goes 5.39 → 3.73 s (0.69×), the
fixed 3.3 s encode dilutes the e2e to 0.83–0.88×. Side by side:
`artifacts/verification-2026-09-11-tpu/flux_1_dev_tpu_baseline_vs_cadence2.png`.

## Cross-device (same MATRIX row)

| device | e2e warm | DiT per-step | notes |
|---|---|---|---|
| trn2 (4 cores) | 35 s | 267.6 ms | `benchmark/trn2/flux_1_dev.md` |
| **v5e x4** | **10 s** (served, incl. encode + decode) | **187 ms** | the first model where the v5e is faster per step (1.43×): 4 608 tokens of dense matmul, small attention |

## Numerical parity

Single forward vs. diffusers fp32 on CPU, seeded random inputs:

| shape | device bf16 (tp=4) vs fp32 | vs diffusers' own bf16 | control: diffusers bf16 vs fp32 |
|---|---|---|---|
| 1024²/512 tok | cos **0.99946** / rel-L1 0.030 | cos **0.99986** / rel-L1 0.014 | cos 0.99946 / rel-L1 0.029 |
| 256²/64 tok | cos 0.99019 / rel-L1 0.135 | cos **0.99974** / rel-L1 0.022 | cos 0.98974 / rel-L1 0.138 |

The device output is closer to diffusers' bf16 than either is to fp32 — the deviation is
bf16, not the port. Before any weights were on the host, a seeded synthetic checkpoint loaded
through the same sharded path matched its CPU fp32 reference at cos 0.9999995 (device fp32).

## Reproduce

```bash
export LD_LIBRARY_PATH=/mnt/models/uvpy/cpython-3.12.14-linux-x86_64-gnu/lib
HF_HOME=/mnt/models/hf DIFFLET_BACKEND=tpu \
  /mnt/models/tpuenv312/bin/python benchmark/flux_tpu_run.py --iters 2 --natural-iters 1 \
      --out /mnt/models/flux_tpu_out                # + --teacache-cadence 2 for the A/B
HF_HOME=/mnt/models/hf DIFFLET_BACKEND=tpu \
  difflet serve --model-id black-forest-labs/FLUX.1-dev --revision 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 \
      --tp-degree 4 --cp-degree 1 --height 1024 --width 1024 --port 8099
curl -X POST :8099/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"black-forest-labs/FLUX.1-dev",
  "messages":[{"role":"user","content":"a cinematic photo of a red fox in a snowy forest at golden hour, highly detailed"}],
  "extra_body":{"num_inference_steps":28,"guidance_scale":3.5,"seed":42}}'
# FLUX.1-dev is a gated repo: put a token with access in $HF_HOME/token first.
```

Evidence: `docs/verification/tpu-model-support-2026-09-11-evidence.md` (Phase 7),
`artifacts/verification-2026-09-11-tpu/flux_*`, `oracle_flux_*`, `serve_flux_tpu_port.log`.
