# HunyuanVideo 1.5 on CUDA

Experimental single-GPU BF16 text-to-video for
`hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`, through Difflet's GPU
application, Python API, CLI and resident HTTP server. It is separate from the Trainium
HunyuanVideo 1.5 path and from the original HunyuanVideo; support does not carry over
between them.

## Usage

Install the CUDA stack as in the [README](../../README.md#install), then:

```bash
DIFFLET_BACKEND=cuda difflet run \
  --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
  --height 480 --width 848 --num-frames 121 \
  --steps 50 --guidance-scale 6 --seed 0 \
  --prompt "a cat walking across a sunlit garden" --output video.mp4

DIFFLET_BACKEND=cuda difflet serve \
  --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
  --height 480 --width 848 --num-frames 121 --request-timeout 1800
```

A 121-frame request takes about seven minutes on an RTX PRO 6000, longer than the server's
default 300 s request timeout, hence `--request-timeout`.

Add `--torch-compile` to compile the 54 transformer blocks; the token refiner, text encoders
and VAE stay eager. In Python, pass `parallel=DiffletParallelConfig(tp_degree=1)`: without it
the registry's Trainium layout (TP2 × CFG2) is used and refused. Set
`application_kwargs={"torch_compile": True}` to compile.
`pipe(prompt=..., num_inference_steps=50, guidance_scale=6, seed=0)` returns CPU FP32 video in
`(B, C, F, H, W)` layout, scaled to `[-1, 1]`.

Each loaded application has one shape and generates one prompt at a time. Height and width
must be multiples of 16, and frames `4n+1`. Defaults are 480×848×121, 50 steps, guidance 6
and 24 fps. Classifier-free guidance runs its two branches one after the other on the same
GPU. Negative prompts work through `difflet run --negative-prompt`, `--requests` batches,
`--dp` runs, the Python API and the server.

Image-to-video, the 720p checkpoints, CFG-parallel, TP/CP/SP, TeaCache, host VAE, TAEF1 and
shape buckets are not supported. DP replicas have not been measured.

## How it runs

- **Transformer.** Diffusers' HunyuanVideo 1.5 building blocks keep the checkpoint layout,
  normalization and cast boundaries; Difflet replaces the projections and attention. The
  checkpoint loads strictly.
- **Eager attention.** When the model loads eagerly in BF16 on a GPU with cuDNN attention and
  Triton, each block of a one-sample request projects Q, K and V with one GEMM per stream,
  trims the text context to its valid tokens, applies Q/K RMSNorm and RoPE in one fused Triton
  op (`difflet_hv15::fused_joint_qk_norm_rope`, written for Difflet) and attends with cuDNN
  SDPA. Otherwise the blocks use the prefix attention below.
- **Prefix attention.** Compiled blocks, and eager blocks that cannot take the path above,
  attend each sample's valid token prefix, so one static-shape graph serves every prompt
  length. In compiled blocks, Inductor's reductions are deterministic.
- **Conditioning.** Diffusers' driver encodes the prompts. Difflet recomputes the Qwen rotary
  frequencies on the GPU and turns cuDNN SDPA off while prompts are encoded, which matches
  vLLM-Omni's conditioning bit for bit.
- **Decoding.** Diffusers' native VAE tiles: 256 px with 25% overlap, which keeps a 121-frame
  decode on one GPU. Tiled and untiled decodes differ slightly; compare outputs with the same
  decoder settings.

## Matched comparison with vLLM-Omni

### Setup

- **Code:** the HunyuanVideo 1.5 source as committed with this guide.
- **Reference:** vLLM-Omni `08177216` (vLLM 0.32.0), unmodified.
- **Runtime, shared by both:** Python 3.12, Torch 2.13.0+cu130, Triton 3.7.1,
  Diffusers 0.40.0 and Transformers 5.19.0.
- **Hardware:** one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB).
- **Procedure:** every engine and mode ran in a fresh process. The whole set ran twice, the
  second time in reverse order.

The two requests:

- **G**, vLLM-Omni's own test settings: 832×480, 33 frames, 50 steps, guidance 6, flow shift 5,
  seed 42, prompt "A cat walks on the grass, realistic style."
- **P**, the request in [Usage](#usage): 848×480, 121 frames, 50 steps, guidance 6, seed 0.

Both engines start from bit-identical initial latents. The decoders differ: Difflet decodes
BF16 latents with a BF16 VAE in native tiles; vLLM-Omni decodes FP32 latents with an FP32 VAE,
untiled for G (its default; the decode peaks at 92.9 GiB of the card's 96 GB) and with its VAE
tiling on for P.

### Output agreement

Videos were written with Diffusers `export_to_video` at 24 fps and scored with ffmpeg SSIM and
PSNR through vLLM-Omni's own accuracy-test helper.

The acceptance bar is relative. On G, for each mode, Difflet against vLLM-Omni must score at
least what vLLM-Omni's own eager and compiled outputs score against each other. P is reported
but does not gate. vLLM-Omni's own accuracy test uses a fixed bar of SSIM ≥ 0.94 and PSNR ≥
28 dB: Difflet's eager G meets it, and compiled G (0.9181) does not.

| Request | Mode | Difflet vs vLLM-Omni (SSIM / PSNR) | vLLM-Omni eager vs compiled | Gate |
|---|---|---|---|---|
| G | eager | 0.9711 / 39.95 dB | 0.8412 / 27.51 dB | pass |
| G | compiled | 0.9181 / 31.65 dB | 0.8412 / 27.51 dB | pass |
| P | eager | 0.9555 / 35.46 dB | 0.4999 / 18.04 dB | reported |
| P | compiled | 0.6465 / 21.16 dB | 0.4999 / 18.04 dB | reported |

**Eager.** The denoiser is bit-identical to vLLM-Omni's: for the first step, in both guidance
branches, all 54 blocks and the final output match exactly. The remaining difference comes from
the VAE decode. Decoding vLLM-Omni's latents with Difflet's decoder reproduces Difflet's video
exactly, and rounding vLLM-Omni's FP32 latents to BF16 before its own decoder already scores
SSIM 0.9747 / 40.59 dB against its own video.

**Compiled.** The two engines compile differently, so their compiled trajectories differ from
block 0: vLLM-Omni compiles regions with dynamic shapes, Difflet compiles each block with static
shapes and deterministic reductions. Each engine's eager and compiled outputs differ from each
other by a similar amount: Difflet 0.8615 / 28.35 dB, vLLM-Omni 0.8412 / 27.51 dB on G.

An earlier run of the same code, on the same card model, gave
byte-identical Difflet videos in both modes and for both requests. Only vLLM-Omni's compiled
videos changed, which moved its eager-vs-compiled G score to 0.8343 / 27.46 dB and Difflet's
compiled G score against it to 0.9041 / 30.20 dB: the bar passed there too.

**Repeatability.** Each engine and mode reproduced its first-pass denoiser digests exactly in
the second pass, in both guidance branches, and gave the same video scores.

**Fused op.** Difflet's fused Q/K op was compared directly with vLLM-Omni's fused op on the GPU,
at the G and P sequence lengths, with 13 and 6 context tokens and three seeds each: Q, K and V
were bit-identical in 12 of 12 cases.

### Latency, start-up and memory

Warm latency is CUDA-synchronized from prompt to decoded frames: the median of three G
requests, and one P request after a warm-up. The speed bar is Difflet / vLLM-Omni ≤ 1.10.

| Mode | Request | Difflet (s), pass 1 / 2 | vLLM-Omni (s), pass 1 / 2 | Ratio, pass 1 / 2 |
|---|---|---|---|---|
| eager | G | 58.83 / 58.85 | 58.81 / 58.85 | 1.000 / 1.000 |
| eager | P | 412.2 / 412.4 | 418.5 / 418.8 | 0.985 / 0.985 |
| compiled | G | 58.58 / 58.65 | 54.11 / 54.14 | 1.083 / 1.083 |
| compiled | P | 408.1 / 407.8 | 398.4 / 397.5 | 1.024 / 1.026 |

Start-up and memory are recorded but do not gate:

- **Load time** (process start to loaded model): Difflet 15–18 s eager and 17–26 s compiled,
  vLLM-Omni 25–28 s.
- **Compilation** (time inside TorchDynamo, pass 1 with empty compile caches / pass 2): Difflet
  5.4 / 0.8 s, while loading; vLLM-Omni 9.4 / 3.3 s, during its first request.
- **Peak device memory**, over the G and P requests:

| Peak memory | Difflet | vLLM-Omni |
|---|---|---|
| Device (NVML) | 62.9 GiB | 92.9 GiB |
| Torch allocated | 45.3 GiB | 72.5 GiB |

### The pinned CUDA environment

vLLM 0.32.0 needs Torch 2.13, so the comparison above uses that runtime. On the same card,
Difflet was also run in the pinned environment of `requirements-cuda.lock` (the README's Torch
2.11/cu128 stack): Python 3.12, Torch 2.11.0+cu128, Triton 3.6.0, Diffusers 0.38.0 and
Transformers 5.20.1.

- `tests/unit/backends/test_cuda_hunyuan15_joint_attention.py` passed on the GPU, 20 of 20,
  including the fused kernel against its reference.
- In eager mode, `difflet run` wrote a 33-frame G video at 24 fps (59.4 s to generate), and
  `difflet serve` returned a G video over HTTP (startup 26 s, inference 59.4 s).
- Warm latency was within 3% of the Torch 2.13 runs: eager G 59.13 s and P 413.6 s, compiled G
  60.24 s and P 420.3 s.
- Initial latents were identical to the Torch 2.13 run, but the Qwen prompt encoder's output
  differs between the two stacks, so the videos differ: eager G scores 0.6880 / 22.32 dB and
  compiled G 0.8140 / 27.00 dB against Difflet's Torch 2.13 videos.

These results cover this checkpoint, these two requests and this card. They do not establish
broader model support.
