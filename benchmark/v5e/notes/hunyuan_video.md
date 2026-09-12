> Port-era notes (2026-08/09, standalone runner). The measured row on the shared
> cold/warm/per-step protocol is [`../hunyuan_video.md`](../hunyuan_video.md) (2026-09-12,
> `benchmark.bench --backend tpu`); the cross-device tables are in
> [`../RESULTS.md`](../RESULTS.md). Kept for the TPU-specific detail
> (what is on the chip, parity, TeaCache A/B, profiling) that the generated report does not carry.

# Benchmark — hunyuanvideo-community/HunyuanVideo

**Status:** ok
**Backend:** tpu
**Device:** Cloud TPU v5litepod-4 / 4 chips / topology 2x2 / 16 GB HBM per chip
**Timestamp:** 2026-09-11 23:20 UTC

> Configuration: tp=4, bf16, eager. Same `benchmark/models.py::MATRIX` row as
> `benchmark/trn2/hunyuan_video.md` (320×512×61, 20 steps, guidance 6.0, seed 42).
> Runner: `benchmark/hunyuan_tpu_run.py` (the TPU adapter is Qwen-specific).

## Configuration

| key | value |
|---|---|
| model type | hunyuan_video |
| HF revision (pinned) | `e8c2aaa66fe3742a32c11a6766aecbf07c56e773` |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 320, 'width': 512, 'num_frames': 61} |
| steps | 20 |
| guidance scale | 6.0 (guidance-distilled: embedded, one forward per step) |
| seed | 42 |
| latent grid | 16 x 40 x 64 → 10 240 patch tokens (+256 text) |

### What is TPU-specific

| key | value |
|---|---|
| execution | eager (lazy XLA), one worker process per chip |
| DiT | difflet's backend-neutral `modeling_hunyuan_video` sharded per rank (`backends/tpu/hunyuan_video`), 5.8 B params / 10.84 GB per chip (3.5 B of them replicated adaLN linears) |
| attention | Pallas flash attention with the joint text mask as key-window segment ids (`attention(bound_min, bound_max)`); SDPA path measured at 2.0 s/step for comparison |
| text encoders | host: Llama-3 8B fp32 on XLA ordinal 0 + `collective_broadcast` (hidden_states[30] = Trainium's `layers.29` capture), CLIP-L fp32 on every rank |
| VAE | host, fp32, tiled — **167 s for 61 frames**; on-chip decode is the open follow-up |
| first compile | 100 s alone; 168–174 s with the 2-slot compile gate (`DIFFLET_TPU_COMPILE_SLOTS`) under serving |

## Results — 320×512×61, 20 steps, tp=4, bf16

| metric | value |
|---|---|
| load (checkpoint → chips, incl. warmup compile) | 180 s |
| text encode (Llama fp32 + broadcast + CLIP) | 6.3 s cold / 4.1–4.6 s warm |
| **denoise, 20 steps** | **20.12 / 20.14 s** synced, **20.11 s** natural |
| **DiT per-step** (synced, n=19) | **1006 ms** |
| DiT per-step (natural) | 1005 ms (host-side Euler step syncs every step anyway) |
| host VAE decode, 61 frames | 167 s |
| HBM after load / peak | 10.84 / 12.5 GB of 15.75 |
| output | latents finite, mean −0.0713 std 0.973; 61 frames, coherent motion |
| `difflet serve` request (`/v1/videos/sync`, same settings) | 200 in **191 s** (≈ 20 s denoise + 5 s encode + 165 s host decode) |

### TeaCache (probe-free, cadence 2)

| mode | denoise (s) | skipped / full | per full step | video vs. baseline |
|---|---|---|---|---|
| baseline | 20.12 | 0 / 20 | 1006 ms | — |
| `--teacache-cadence 2` | **15.07 / 15.08 / 15.19** (0.749×) | 5 / 15 | 1004 ms | mean abs 0.0064/px, PSNR 36.9 dB, visually identical |

Same 0.75× as Qwen-Image and Wan on this backend.

## Cross-device (same MATRIX row)

| device | e2e warm | DiT per-step | notes |
|---|---|---|---|
| trn2 (4 cores) | 144 s | 850.6 ms | `benchmark/trn2/hunyuan_video.md` (Neuron VAE / host VAE per profile) |
| **v5e x4** | **191 s** (served) / 20.1 s denoise | **1006 ms** | 165 s of the e2e is the host VAE; denoise alone is 7× faster than the trn2 e2e |

## Numerical parity

Single forward vs. diffusers fp32 on CPU (`/mnt/models/teacache_runs/oracle_hunyuan_cmp2.log`):
fused cos **0.99949** / rel-L1 3.08e-2, SDPA 0.99949 / 3.09e-2; control (diffusers bf16 vs its
own fp32) 0.99946 / 3.15e-2.

## Reproduce

```bash
export LD_LIBRARY_PATH=/mnt/models/uvpy/cpython-3.12.14-linux-x86_64-gnu/lib
HF_HOME=/mnt/models/hf DIFFLET_BACKEND=tpu \
  /mnt/models/tpuenv312/bin/python benchmark/hunyuan_tpu_run.py --steps 20 --iters 2 --natural-iters 1 \
      --out /mnt/models/hunyuan_tpu_out            # + --teacache-cadence 2 for the A/B
HF_HOME=/mnt/models/hf DIFFLET_BACKEND=tpu \
  difflet serve --model-id hunyuanvideo-community/HunyuanVideo --tp-degree 4 --cp-degree 1 \
      --height 320 --width 512 --num-frames 61 --host-vae --port 8097
```

Evidence: `docs/verification/tpu-model-support-2026-09-11-evidence.md` (Phase 5),
`artifacts/verification-2026-09-11-tpu/hunyuan_*`.
