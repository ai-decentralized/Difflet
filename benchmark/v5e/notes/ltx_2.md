> Port-era notes (2026-08/09, standalone runner). The measured row on the shared
> cold/warm/per-step protocol is [`../ltx_2.md`](../ltx_2.md) (2026-09-12,
> `benchmark.bench --backend tpu`); the cross-device tables are in
> [`../RESULTS.md`](../RESULTS.md). Kept for the TPU-specific detail
> (what is on the chip, parity, TeaCache A/B, profiling) that the generated report does not carry.

# Benchmark — Lightricks/LTX-2

**Status:** ok
**Backend:** tpu
**Device:** Cloud TPU v5litepod-4 / 4 chips / topology 2x2 / 16 GB HBM per chip
**Timestamp:** 2026-09-12 03:10 UTC

> Configuration: tp=4, bf16, eager. Same `benchmark/models.py::MATRIX` row as
> `benchmark/trn2/ltx_2.md` (480×704×49, 20 steps, guidance 1.0, seed 42).
> Runner: `benchmark/ltx2_tpu_run.py`.

## Configuration

| key | value |
|---|---|
| model type | ltx_2 |
| HF revision (pinned) | `47da56e2ad66ce4125a9922b4a8826bf407f9d0a` |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 480, 'width': 704, 'num_frames': 49} |
| steps | 20 |
| guidance scale | 1.0 (no CFG; the quality row is 3.0 — see below) |
| seed | 42 |
| latent grid | 7 x 15 x 22 → 2 310 video tokens + 51 audio tokens + 1 024 text |

### What is TPU-specific

| key | value |
|---|---|
| execution | eager (lazy XLA), one worker process per chip |
| DiT | diffusers' `LTX2VideoTransformer3DModel`, TP-sharded with the recipe lifted from the Trainium wrapper (`difflet/models/ltx_2/tp_sharding.py`): 4.98 B params / 9.3 GiB per chip; text cross-attention **masked** on TPU (key-window bounds into the fused kernel), which Trainium cannot |
| text encoder | host: Gemma-3 12B **fp32** on XLA ordinal 0, packed hidden states [1, 1024, 188160] broadcast to the other ranks; connectors fp32 on every rank |
| video VAE | **on the chip** (primary replica only): 0.2 s at this shape, 0.7 s at 512×768×121 — the fp32 host decode of 121 frames did not finish in 14 minutes |
| audio VAE + vocoder | host, fp32 (0.1 s + 0.6 s; in bf16 they took 114 s + 196 s: bf16 is emulated on the EPYC host) |
| first compile | 73 s alone; 140–144 s with the 2-slot compile gate under serving |

## Results — 480×704×49, 20 steps, tp=4, bf16

| metric | value |
|---|---|
| load (checkpoint → chips, warmup compile, host pipeline, VAE to chip) | 152 s |
| **e2e to latents** (Gemma encode + 20 DiT steps) | **49.6 / 48.2 s** synced, **49.5 s** natural |
| **DiT per-step** (synced, n=19) | **1453 ms** |
| DiT per-step (natural) | 1457 ms |
| Gemma-3 fp32 encode (1024 tokens, one rank) | ≈ 20 s (the e2e minus 20 × 1.45 s) |
| decode: video VAE (chip) / audio VAE / vocoder | 0.2 s warm (51.8 s first compile) / 0.11 s / 0.56 s |
| HBM after load / peak (rank 0, DiT + VAE) | 11.6 / 12.3 GB of 15.75 |
| output | latents finite; 49 frames 480×704 |
| `difflet serve` request (`/v1/videos/sync`, same settings) | **200 in 51.6 s / 50.2 s**, bit-identical; ready in 246 s |

### TeaCache (probe-free, cadence 2)

| mode | e2e to latents (s) | skipped / full | per full step | video vs. baseline |
|---|---|---|---|---|
| baseline | 49.6 / 48.2 | 0 / 20 | 1453 ms | — |
| `--teacache-cadence 2` | **42.4 / 43.3 / 42.0** | 5 / 15 | 1451 ms | mean abs 0.0088/px, PSNR 37.9 dB |

The DiT part goes 29 → 22 s (0.75×, the same as every other model here); the fixed ~20 s
encode dilutes the e2e to 0.86×.

### Quality note

At guidance 1.0 / 20 steps (this timing row) the subject is muddy; the same pipeline at
guidance 3.0 produces a clear red fox (`artifacts/verification-2026-09-11-tpu/ltx_2_vae_buffer_bug_device_vs_host_f24.png`,
right half). trn2's row is the same timing-only configuration; its visual check used 40 steps
with CFG.

## Cross-device (same MATRIX row)

| device | e2e warm | DiT per-step | notes |
|---|---|---|---|
| trn2 (4 cores) | 58 s | 441.8 ms | `benchmark/trn2/ltx_2.md` |
| **v5e x4** | **51 s** (served, incl. encode + decode) | **1453 ms** | per-step 3.3× slower than trn2, e2e faster: the decode is on the chip and the encode on one rank |

## Numerical parity

Single forward vs. diffusers fp32 on CPU, 512×768×121, 984/1024 text positions padded:
video cos **0.99983** / rel-L1 1.83e-2, audio 0.99957; control (diffusers bf16 vs its own fp32)
0.99983 / 1.78e-2. Before the masked cross-attention fix (`5bca57d`): 0.9932 / 0.131.

## Reproduce

```bash
export LD_LIBRARY_PATH=/mnt/models/uvpy/cpython-3.12.14-linux-x86_64-gnu/lib
HF_HOME=/mnt/models/hf DIFFLET_BACKEND=tpu \
  /mnt/models/tpuenv312/bin/python benchmark/ltx2_tpu_run.py --steps 20 --iters 2 --natural-iters 1 \
      --out /mnt/models/ltx2_tpu_out               # + --teacache-cadence 2 for the A/B
HF_HOME=/mnt/models/hf DIFFLET_BACKEND=tpu \
  difflet serve --model-id Lightricks/LTX-2 --tp-degree 4 --cp-degree 1 \
      --height 480 --width 704 --num-frames 49 --host-vae --port 8098
# cold page cache: the first read of the 46 GB Gemma-3 checkpoint alone can exceed the
# default 900 s startup budget -> --worker-restart-timeout 1800 for the first start.
```

Evidence: `docs/verification/tpu-model-support-2026-09-11-evidence.md` (Phase 6),
`artifacts/verification-2026-09-11-tpu/ltx2_*`, `ltx_2_*`.
