# Original HunyuanVideo on CUDA

Experimental single-GPU BF16 support for the original
`hunyuanvideo-community/HunyuanVideo` Diffusers checkpoint, using Difflet's transformer
and the shared GPU application, Python API, CLI and resident HTTP server.

## Usage

```bash
DIFFLET_BACKEND=cuda difflet run \
  --model-id hunyuanvideo-community/HunyuanVideo \
  --height 320 --width 512 --num-frames 61 \
  --steps 50 --guidance-scale 6 --seed 0 \
  --prompt "a cat walking across a sunlit garden" --output video.mp4

DIFFLET_BACKEND=cuda difflet serve \
  --model-id hunyuanvideo-community/HunyuanVideo \
  --height 320 --width 512 --num-frames 61
```

Add `--torch-compile` for opt-in regional compilation of the 60 dual-stream and
single-stream blocks; the token refiner remains eager. In the Python factory, set
`application_kwargs={"torch_compile": True}` for the same option.
`pipe(prompt=..., num_inference_steps=50, guidance_scale=6, seed=0)` returns CPU FP32
video in `(B,C,F,H,W)` layout, scaled to `[-1,1]`.

Each loaded application has one shape and generates one prompt at a time. Height and
width must be positive multiples of 16; frames must be positive and `4n+1`. Defaults
are 320×512×61, 50 steps, guidance 6 and 24 fps. Llama retains up to 256 tokens after
the reference template crop; longer prompts are refused. CLIP uses the reference
77-token truncation. The CLI and server share one MP4 encoder.

HunyuanVideo 1.5, image-to-video, negative prompts, TP/CP/SP, CFG-parallel, TeaCache,
host VAE, TAEF1 and shape buckets are unsupported. The resident video service uses one
worker; replica scaling and smaller GPUs have not been qualified.

## Numerical implementation

Strict checkpoint loading restores the reference's concatenated single-stream projection
and biased linear rounding at TP=1. Compact joint attention preserves padded text masks.
The adapter preserves the reference driver's BF16-rounded latents and timesteps; upcasting
a received timestep does not recover lost precision. Regional compilation retains native
linear and normalization operations at compiler boundaries to preserve reference rounding.
Other block operations compile through Inductor. Shared Trainium defaults remain unchanged.

## Verified scope

Qualification on 2026-10-30 used an RTX PRO 6000 Blackwell 96 GB, checkpoint revision
`e8c2aaa66fe3742a32c11a6766aecbf07c56e773`, 320×512×61, 50 steps, guidance 6, seeds 0–2
and prompt `a cat walking across a sunlit garden`, with dense BF16 and no quantization,
cache acceleration, offload or tiling.

Nine stock BF16 eager/compiled and contract-FP32 captures established per-seed limits
before candidate execution. The FP32 control used BF16-rounded weights and the unchanged
BF16 driver. Maximum error, MAE, relative L1/L2 and RMSE passed for final video and all
50 latent states in all 12 cases: three each for Difflet eager/compiled, unmodified Omni
eager and the supplementary modified compiled control. Independent rescoring reproduced
the calibration and outcomes. All 24 timed outputs exactly matched their own accepted
quality captures. Limits were unchanged after candidate results.

On the supported runtime (torch 2.11.0+cu128, cuDNN 9.19, Diffusers 0.38.0,
Transformers 5.20.1), actual public Python, CLI and HTTP routes passed in both modes:
pre-encoding float identity, decoded CLI/HTTP frame identity, repeated A/B/A requests in
one resident application, stable graphs, 512×320 output with 61 frames at 24 fps, and
valid long-prompt/invalid-request checks. Traces distinguished eager from compiled execution.
These identity checks do not transfer numerical calibration between runtimes.

## Controlled performance comparison

The comparison used Python 3.12.13, torch 2.13.0+cu130, CUDA 13.0, cuDNN 9.20,
Diffusers 0.40.0, Transformers 5.19.0 and vLLM 0.32.0. It deliberately overrides
Difflet's default dependencies. The unmodified vLLM-Omni Diffusers adapter was pinned at
`cbf35fb0ccf46c5d7ff740f74ebb6201e33c23a3`; its relevant files matched upstream
`5b234955e0b81765020eae3a76a312a390e6955d`.

Each mode used six matched pairs in Difflet/Omni/Omni/Difflet order, with three seeds per
cell, full warmups and separate traces. No new graphs appeared during timed requests.
Generation includes text encoding, DiT, VAE and conversion to CPU FP32 video. It excludes
loading, compilation, warmup, prepared-noise creation, HTTP/IPC/server overhead and writes.

| Comparator | Difflet median | Comparator median | Median paired ratio | Ratio 95% interval |
|---|---|---|---|---|
| Unmodified Omni adapter, eager | 65.626 s | 67.300 s | 0.975147 | [0.974129, 0.975709] |
| Modified compiled control (supplementary) | 61.440 s | 61.118 s | 1.005334 | [1.004813, 1.005985] |

Ratios are Difflet/comparator. Intervals use a paired bootstrap median, 2,000 draws and
seed 0. Both generation and
denoising met the predeclared 1.05 ratio limit. Eager generation was 2.485% lower latency;
compiled generation was 0.533% higher against the modified control. The latter compiles
exactly 60 denoiser blocks using `max-autotune-no-cudagraphs`, `dynamic=False` and
`fullgraph=False`, matching the stock control's scope and options.

Official Omni compile discovery produced CUDA execution with zero compiled DiT regions
for this pinned adapter route, so an official compiled baseline remains unavailable.
The modified control is supplementary. These results cover one GPU, prompt and shape;
they establish no default-installation, HTTP or official compiled speedup, broader
perceptual equivalence, or Trainium device qualification. Subsequent integration with main
was checked through source comparison and CPU replay, without a new GPU measurement.

## Verification tools

The weight-free attention probe checks dispatch and allocation at the requested geometry:

```bash
DIFFLET_BACKEND=cuda python tests/numerical/gpu_hunyuan_video_preflight.py \
  --height 320 --width 512 --num-frames 61 --mode both \
  --output-dir /tmp/hunyuan-preflight
```

The full-pipeline harness compares Difflet with stock Diffusers. Use a local snapshot whose
directory name matches the required checkpoint revision:

```bash
DIFFLET_BACKEND=cuda python tests/numerical/gpu_hunyuan_video_parity.py \
  --model-path /path/to/snapshots/CHECKPOINT_COMMIT \
  --checkpoint-revision CHECKPOINT_COMMIT \
  --height 320 --width 512 --num-frames 61 --steps 50 \
  --seeds 0,1,2 --mode both --output-dir /tmp/hunyuan-controls
```

Without independently established `--calibration-limits`, quality is `uncalibrated` and
the harness exits 2. Limits must contain separate `video` and `trajectory` objects for
`max_abs`, `mean_abs`, `relative_l1`, `relative_l2` or `rmse`; establish them from known-correct
controls before candidates. This stock-Diffusers harness does not reproduce the separate
Omni comparison. Keep generated reports, tensors, videos and traces outside the repository.
