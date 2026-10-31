<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/yotta-logo-white.svg">
  <img alt="Yotta Labs" src="docs/assets/yotta-logo-black.svg" width="260">
</picture>
</div>

# Difflet

**Run diffusion transformers on NVIDIA GPUs, AWS Trainium and (experimentally) Cloud TPU: FLUX, Qwen-Image, Wan, HunyuanVideo, LTX-2 etc., from one CLI, one Python API, and one OpenAI-compatible server.**

MiniMax-H3 also supports text-to-video with audio through the staged CLI.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.8%2B-76B900.svg)
![Neuron](https://img.shields.io/badge/Neuron-neuronx--cc%202.26-232F3E.svg)

```bash
DIFFLET_BACKEND=cuda difflet run --model-id black-forest-labs/FLUX.1-dev \
  --prompt "a photorealistic cat sitting in a sunlit garden" --output cat.png
```

On an NVIDIA GPU, that one command downloads the weights, loads the model onto the card, and
writes a 1024×1024 image. There is no compile step. On Trainium, drop `DIFFLET_BACKEND` and add
`--tp-degree 4`: the first run AOT-compiles the model for four NeuronCores and caches the
artifact, and later runs skip straight to load and denoise.

[Latest news](#latest-news) · [Hardware](#hardware-support) · [Models](#supported-models) · [Feature support](#feature-support) ·
[Quick start](#quick-start) · [Choose a topology](#choose-a-topology) · [Serving](#serving) ·
[Go further](#go-further) · [CLI reference](#cli-reference) · [Troubleshooting](#troubleshooting) ·
[Developer guide](DEVELOPER.md)

## Latest News

- [09/30] **MiniMax-H3 text-to-video with audio is verified on Trainium2.** The four-stage CLI generated a 256×448, 124-frame MP4 with stereo audio on `trn2.3xlarge` using TP4. See the [end-to-end verification](docs/design/minimax_h3/01_e2e_reverification_2026-09-30.md).
- [09/10] **Difflet 1.0 is released.**

## What Difflet does

Difflet handles the full lifecycle of a diffusion model: model download, compilation where the
hardware needs it, a content-addressed artifact cache, and multi-device execution. Model code
is written once against a backend-neutral op surface (`difflet.ops`), and each accelerator
supplies its own implementation:

- **NVIDIA GPUs**: eager PyTorch on one GPU per model copy, with optional `--torch-compile`.
- **Trainium**: ahead-of-time (AOT) compilation and SPMD execution across NeuronCores. Text
  encoders, the DiT backbone, and the VAE all run on device for the supported models, with no
  CPU fallbacks in the hot path.
- **Cloud TPU** (experimental): tensor-parallel serving across TPU chips.

See [Hardware support](#hardware-support) for what each backend runs today.

Three entry points share one engine:

- **`difflet` CLI** — `run` for one-shot generation, or `download → compile → generate` staged.
- **`difflet serve`** — a resident, OpenAI-compatible HTTP server for image and video models.
- **`DiffletPipeline`** — a Python API that mirrors `diffusers`.

## Hardware support

| Backend | `DIFFLET_BACKEND` | Status | Models | Parallelism | Details |
|---|---|---|---|---|---|
| NVIDIA GPUs | `cuda` | ✅ Supported (subset) | FLUX.1-dev, Qwen-Image, Wan 2.2 | DP, CFG-parallel (Wan) | [GPU features](#nvidia-gpus) |
| AWS Trainium2 | `trainium` | ✅ Supported | All [supported models](#supported-models) | TP, CP, SP, CFG-parallel, DP | [Trainium features](#trainium) |
| Cloud TPU | `tpu` | 🧪 Experimental | Qwen-Image, Wan (`difflet serve` only) | TP | [v5e benchmarks](benchmark/v5e/RESULTS.md) |
| AMD ROCm | `rocm` | Planned | — | — | — |

Without `DIFFLET_BACKEND`, Difflet picks the backend it detects on the host. It checks Trainium,
then TPU, then CUDA, then ROCm, which only matters on a host with more than one accelerator.
Set `DIFFLET_BACKEND` to choose one explicitly. A CPU reference backend (`cpu`) exists for
numerical tests and CI. It is not meant for generation.

Cloud TPU work in progress, including FLUX, HunyuanVideo and LTX-2 ports, TeaCache in serving,
and a move from `torch_xla` to TorchTPU, is tracked in open pull requests
[#60–#65](https://github.com/ai-decentralized/Difflet/pulls?q=is%3Apr+is%3Aopen+TPU) and
[#119](https://github.com/ai-decentralized/Difflet/pull/119). Until those land, `difflet run`
and the Python API do not run on TPU.

## Supported models

| Model | Type | Default shape | Notes |
|---|---|---|---|
| [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Text-to-image | 1024×1024 | Single-process |
| [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) | Text-to-image | 1024×1024 | 3-stage (text → generate → vae) |
| [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | Text-to-video | 480×832×9 | 2-stage (transformer → vae); true CFG |
| [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) | Text-to-video | 480×832×9 | Same runtime as Wan 2.2 |
| [hunyuanvideo-community/HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) | Text-to-video | 320×512×61 | 3-stage (clip → llama → generate) |
| [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) | Text-to-video | 512×768×121 | Single-process; TP only; exports `.mp4` |
| [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | Text-to-video with audio | 768×1344×124 ⚠️ ¹¹ | Verified at **256×448×124**; TP4; 4-stage CLI (text → generate → video_vae → audio_vae); `.mp4` with stereo audio |

## Feature support

Which features each model supports today. ✅ = supported · ⚠️ = supported with a caveat
(see note) · ❌ = not supported · — = not applicable.

**Feature legend**

- **TP** — tensor parallelism (`--tp-degree`). Splits each layer across devices (NeuronCores on Trainium).
- **CP — all-gather** — context parallelism with gather-KV attention (`--cp-degree N --cp-mode gather_kv`, the default). Splits the sequence across ranks.
- **CP — ring** — context parallelism with ring attention (`--cp-degree N --cp-mode ring`). Lower memory than all-gather for long sequences.
- **CP — ulysses** — context parallelism with head-sharded attention (`--cp-mode ulysses`). Needs the model's head count divisible by `tp_degree × cp_degree`.
- **SP** — Megatron-style sequence parallelism (`--sp`). Shards the norm/modulation/residual regions along the sequence axis across the existing tensor-parallel group; `world_size` is unchanged. Mutually exclusive with `--cp-degree > 1`.
- **CFG-parallel** — splits the conditional/unconditional CFG passes across 2 data-parallel ranks (`--cfg-parallel`). Only meaningful for true two-pass classifier-free guidance, not working for distilled guidance.
- **DP** — data-parallel replicas (`--dp N`). A router spawns N full model copies on disjoint core ranges or GPUs and distributes requests across them; use `--mode throughput` or `--mode mixed` for a preset.
- **CLI generation** — generate media with `difflet run` or `download → compile → generate`. MiniMax-H3 exports video and stereo audio together in one MP4.
- **Multi-shape compile** — one bucketed artifact covering several request shapes (`--shapes 320x512x61,320x512x33`), sharing a single weight copy on device. `difflet serve --shapes` serves all of them from one resident worker.
- **TeaCache (adaptive)** — calibration-driven step-skipping (`--teacache-speedup` / `--teacache-online-delta`, with `--teacache-calibration`).
- **TeaCache (fixed cadence)** — blind skip-every-N-steps (`--teacache-cadence N`, no calibration needed).
- **Serving** — resident `difflet serve` worker. Image models answer `/v1/chat/completions`; video models answer the [Videos API](docs/serving/videos_api.md).
- **Batch (JSONL)** — `--requests FILE` runs one request per line (prompt, output, seed, optional negative prompt, guidance scale, steps) through one loaded model.

### NVIDIA GPUs

`DIFFLET_BACKEND=cuda`. Each model copy runs on one GPU at tp=1, with no ahead-of-time compile
step. Verified on RTX PRO 6000 Blackwell (96 GB) and A100 80 GB.

| Model | CLI generation | Serving | `--torch-compile` | TeaCache (fixed cadence) | DP | CFG-parallel | TP / CP / SP | Multi-shape |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| FLUX.1-dev | ✅ image | ✅ image | ✅ | ❌ | ✅ ² | — ¹ | ❌ | ❌ |
| Qwen-Image | ✅ image | ⚠️ image ³ | ✅ | ✅ | ✅ | — ⁴ | ❌ | ❌ |
| Wan 2.2 | ✅ video | ✅ video | ✅ | ❌ | ⚠️ ⁵ | ✅ ⁶ | ❌ | ❌ |

An option the GPU backend does not support is refused with a message naming it, not ignored.
TeaCache (adaptive), TAEF1, and `--host-vae` are refused on GPUs for all three models.

1. Guidance-distilled model (the guidance scale is baked into the timestep embedding), so there
   is no second CFG branch to split.
2. Two replicas verified on 2 × A100 (`--dp 2 --requests` and `serve --dp-degree 2`), with
   output byte-identical to one card.
3. `difflet serve` handles one request at a time. A second concurrent request gets `429`
   after the 30 s queue timeout. Use `serve --dp-degree N` for concurrency.
4. Qwen-Image runs one transformer pass per step, so there is no CFG branch to split.
5. `--dp N --requests` works, with output bit-identical to one GPU. `serve --dp-degree` does
   not apply, because the video service runs one job at a time. Wan 2.1 is routed to the GPU
   flow but has not been run on a GPU.
6. Two GPUs under `torchrun`, `difflet run` only. Output is bit-identical to one GPU.

Measured timings against stock diffusers, and the full pass/limit matrix, are in
[docs/verification/gpu-backend-results.md](docs/verification/gpu-backend-results.md).

### Trainium

**Parallelism**

| Model | TP | CP — all-gather | CP — ring | CP — ulysses | SP | CFG-parallel | DP |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| FLUX.1-dev | ✅ | ✅ | ✅ | ✅ | ✅ | — ⁷ | ✅ |
| Qwen-Image | ✅ | ✅ | ✅ | ✅ | ✅ | — ⁷ | ✅ |
| Wan 2.2 / 2.1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| HunyuanVideo | ✅ | ⚠️ ⁸ | ✅ | ❌ | ✅ | — ⁷ | ⚠️ ⁹ |
| LTX-2 | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| MiniMax-H3 | ✅ TP4 | ❌ | ❌ | ❌ | ❌ | — ⁷ | ❌ |

**Runtime features**

| Model | CLI generation | Multi-shape compile | TeaCache (adaptive) | TeaCache (fixed cadence) | Serving | Batch (JSONL) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| FLUX.1-dev | ✅ image | ✅ | ✅ | ✅ | ✅ image | ✅ |
| Qwen-Image | ✅ image | ✅ | ✅ | ✅ | ✅ image | ✅ |
| Wan 2.2 / 2.1 | ✅ video | ✅ | ✅ | ✅ | ✅ video ¹⁰ | ✅ |
| HunyuanVideo | ✅ video | ✅ | ✅ | ✅ | ✅ video | ✅ |
| LTX-2 | ✅ video | ❌ | ✅ | ✅ | ✅ video | ✅ |
| MiniMax-H3 | ✅ video + stereo audio | ❌ | ❌ | ❌ | ❌ | ❌ |

**FP8 on Trainium3** (PyTorch-native `neuron` backend, `--quant mxfp8`): Qwen-Image's DiT
runs its attention and MLP linears in OCP MXFP8 through nkilib's `matmul_mxfp8` NKI kernel,
no calibration needed. Measured on trn3.48xlarge at tp4, 1024×1024: DiT step 267 ms against
299 ms in bf16 (0.89×), PSNR 29.2 dB / SSIM 0.973 / LPIPS 0.04 against the bf16 image, half
the DiT weight bytes. Refused on the AoT `trainium` backend, on trn2 and for other models;
see [docs/cli-staged-commands.md](docs/cli-staged-commands.md#qwen-image-in-fp8-pytorch-native-neuron-backend-trainium3-only)
and the evidence in
[docs/verification/2026-10-31-qwen-image-fp8-native-trn3.md](docs/verification/2026-10-31-qwen-image-fp8-native-trn3.md).

**Notes**

7. Guidance-distilled model (single forward pass with the guidance scale baked into the timestep embedding) — there is no second CFG branch to split.
8. `tp2 cp2` with gather-KV hits a `neuronx-cc` internal error (`NCC_INLA001` / `NCC_IBIR243`) on the CP-degree-2 DiT graph; ring CP and `tp4 --sp` are the working multi-core paths. See [DEVELOPER.md](DEVELOPER.md).
9. DP works, but on a 4-core `trn2.3xlarge` each 2-core replica runs out of HBM loading the compiled VAE at the default 320×512×61 shape. Use a smaller shape or a host with more cores per replica.
10. Wan 2.1 is the qualified serving checkpoint. Wan 2.2 can be started for experiments but its dual-transformer path has not passed resident-serving acceptance.
11. MiniMax-H3 supports the Trainium four-stage CLI at TP4 and CP1: text and DiT use four cores, and each VAE uses one core. The verified end-to-end shape is **256×448×124**; pass `--height 256 --width 448 --num-frames 124`. The registry default **768×1344×124** is not qualified on `trn2.3xlarge`: the standard DiT graph exceeds device memory, and compilation with precomputed AdaLN exceeded host memory. The 640×1152×124 DiT compiled successfully but has not been verified end to end. Python API generation, resident serving, DP, JSONL batching, multi-shape compilation, and TeaCache are not supported for H3. See the [verification](docs/design/minimax_h3/01_e2e_reverification_2026-09-30.md) and [bring-up handoff](docs/design/minimax_h3/00_trainium_bringup_handoff.md).

Context parallelism (`--cp-degree > 1`) and CFG-parallel both consume the data-parallel lanes, so they are mutually exclusive (and each is mutually exclusive with `--sp`). `world_size = dp × (2 if cfg-parallel else 1) × cp_degree × tp_degree`; `--sp` leaves it unchanged. `difflet plan --model-id <id>` lists the combinations your host can run.

## Quick start

### Prerequisites

- An accelerator, one of:
  - **An NVIDIA GPU** with a CUDA 12.8+ driver and enough memory for the whole model on one
    card: FLUX.1-dev and Qwen-Image fit in 80 GB; Wan 2.2 keeps both experts resident and
    was verified on a 96 GB card.
  - **AWS Trainium2.** Validated on `trn2.3xlarge`: one Trainium2 chip presented as 4
    logical NeuronCores under the default `LNC=2`. Other Trn2 shapes should work; the
    tensor-parallel degree must divide the number of visible cores.
- Python 3.10 or newer.
- A Hugging Face account with access to the gated FLUX.1-dev repo, if you want Flux.
- Disk: weights are large (LTX-2 alone is 86 GB). On Trainium, compiled artifacts add tens of
  GB.

### Install

**On NVIDIA GPUs:**

```bash
git clone git@github.com:ai-decentralized/Difflet.git
cd Difflet
python3 -m venv .venv && source .venv/bin/activate
# torch must come from the CUDA 12.8 index (Blackwell needs a 12.8+ wheel).
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
pip install -e ".[cuda]"
huggingface-cli login         # for gated repos such as FLUX.1-dev
export DIFFLET_BACKEND=cuda
```

`requirements-cuda.lock` has the full pinned environment, and
[DEVELOPER.md](DEVELOPER.md#gpu-backends-cuda) covers the GPU internals.

**On Trainium:**

```bash
git clone git@github.com:ai-decentralized/Difflet.git
cd Difflet
./scripts/setup_env.sh        # builds .venv from requirements-neuron.lock
source .venv/bin/activate     # required: the Neuron runtime needs .venv/bin on PATH
huggingface-cli login         # for gated repos such as FLUX.1-dev
```

`setup_env.sh` installs the pinned Neuron toolchain (`neuronx-cc`, `torch-neuronx`,
`neuronx-distributed`, `nki`) plus the `difflet` CLI. Recent Neuron DLAMI releases no
longer ship the old `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference` venv, so the repo
builds its own.

Sanity-check without touching the device (either backend):

```bash
PYTHON_BIN=$PWD/.venv/bin/python ./scripts/check_quick.sh   # import gate + CPU-only unit tests
difflet --help
```

### First image

```bash
difflet run --model-id black-forest-labs/FLUX.1-dev \
  --height 1024 --width 1024 \
  --prompt "a photorealistic cat sitting in a sunlit garden" \
  --output cat.png
```

On a GPU there is no compile step: the run loads the weights and generates. Add
`--torch-compile` for faster steps after a warm-up, or `--dp N` for one replica per GPU.

On Trainium, add `--tp-degree 4`. The first run AOT-compiles the model, which takes tens of
minutes. Artifacts are cached under `~/.cache/difflet/`, and every later run with the same
model, shape, parallel config, and toolchain skips straight to load and denoise.

### First video

```bash
difflet run --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --height 480 --width 832 --num-frames 9 \
  --steps 40 --guidance-scale 4.0 --seed 42 \
  --prompt "a cat walking through a garden" \
  --output cat.mp4
```

On a GPU, Wan 2.2 keeps both experts on one card (verified on 96 GB). Add `--cfg-parallel` to
split the guidance passes across two GPUs; `difflet run` relaunches itself under `torchrun`.
On Trainium, add `--tp-degree 2 --cp-degree 2`.

### Python API

```python
from difflet import DiffletPipeline, DiffletParallelConfig

pipe = DiffletPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    parallel=DiffletParallelConfig(),  # one GPU; on Trainium: DiffletParallelConfig(tp_degree=4)
    height=1024,
    width=1024,
)

image = pipe(
    prompt="a photorealistic cat sitting in a sunlit garden",
    num_inference_steps=28,
).images[0]
image.save("out.png")
```

Pass `parallel` explicitly on GPUs: a model's registry default is its Trainium layout (for
example `tp_degree=8` for Flux), which the GPU backend refuses. On GPUs, the Python API is
covered by unit tests; hardware verification so far is through the CLI and `difflet serve`.

## Choose a topology

This section applies to Trainium. On GPUs, each model copy uses one card; scale out with
`--dp N` (see [NVIDIA GPUs](#nvidia-gpus)).

Every configuration must use exactly the cores you have:

```
world_size = dp × (2 if cfg-parallel else 1) × cp_degree × tp_degree
```

`--sp` reshards within the TP group and leaves `world_size` unchanged. Context parallelism
and CFG-parallel both consume the data-parallel lanes, so they are mutually exclusive, and
each is mutually exclusive with `--sp`.

On a 4-core host the useful presets are:

| Goal | Flags | Works for |
|---|---|---|
| Lowest latency, one request | `--tp-degree 2 --cp-degree 2` or `--tp-degree 4 --sp` | Flux, Qwen-Image, Wan, HunyuanVideo |
| Lowest latency, true-CFG model | `--tp-degree 2 --cfg-parallel` | Wan, LTX-2 |
| Lowest latency, LTX-2 | `--tp-degree 4` | LTX-2 |
| Highest throughput, many requests | `--tp-degree 2 --dp 2` or `--tp-degree 1 --dp 4` | Flux, Qwen-Image, Wan, LTX-2 |
| Balanced | `--mode mixed` | all |

`--mode latency|throughput|mixed` picks `dp`, `cfg`, and `cp` for the model class; explicit
flags override individual fields. When in doubt, ask the planner. It reads the host's
`neuron-ls`, scores every legal config for your model and shape, and marks which ones are
already compiled:

```bash
difflet plan --model-id black-forest-labs/FLUX.1-dev --objective latency
difflet plan --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --objective throughput --serving
```

## Serving

`difflet serve` loads one model with one immutable profile into a resident worker. HTTP
requests never trigger compilation or reloading, so the request shape must be one the server
was started with.

### Start

On NVIDIA GPUs:

```bash
DIFFLET_BACKEND=cuda difflet serve \
  --model-id Qwen/Qwen-Image \
  --host 0.0.0.0 --port 8092
```

The worker handles one request at a time. Add `--dp-degree N` to serve from N GPUs, with one
replica each, or `--torch-compile` for faster steps after a warm-up.

On Trainium:

```bash
difflet serve \
  --model-id black-forest-labs/FLUX.1-dev \
  --tp-degree 4 --cp-degree 1 \
  --height 1024 --width 1024 \
  --host 0.0.0.0 --port 8092
```

On Trainium, the first start downloads weights, compiles artifacts, loads the worker, and
runs a real generation smoke test before `/ready` opens. Warm restarts reuse the cache. One
serving profile owns all four cores on a `trn2.3xlarge`, so you cannot run two servers, or a
server and a CLI generation, at the same time.

### Check readiness

```bash
curl http://127.0.0.1:8092/health
curl http://127.0.0.1:8092/ready
curl http://127.0.0.1:8092/v1/models
```

### Generate an image

```bash
curl -sS -X POST http://127.0.0.1:8092/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "black-forest-labs/FLUX.1-dev",
    "messages": [{"role": "user", "content": "a small red sailboat on a calm blue lake"}],
    "extra_body": {"height": 1024, "width": 1024, "num_inference_steps": 20,
                   "guidance_scale": 3.5, "seed": 42}
  }' | jq -r '.choices[0].message.content[0].image_url.url' \
    | cut -d',' -f2- | base64 -d > output.png
```

`height` and `width` must match the server's profile. Steps must be between 1 and 50. Without
S3 configured the URL is a Base64 data URL; with S3 it is an expiring presigned URL.

### Generate a video

Start a video model, then submit a job and download the result when it completes:

```bash
difflet serve --model-id hunyuanvideo-community/HunyuanVideo \
  --tp-degree 4 --height 320 --width 512 --num-frames 61 \
  --host 0.0.0.0 --port 8092
```

```bash
video_id=$(curl -sS -X POST http://127.0.0.1:8092/v1/videos \
  -F 'model=hunyuanvideo-community/HunyuanVideo' \
  -F 'prompt=a cinematic mountain landscape at sunrise' \
  -F 'size=512x320' -F 'num_frames=61' -F 'fps=24' \
  -F 'num_inference_steps=4' -F 'guidance_scale=6.0' | jq -r '.id')

curl -sS "http://127.0.0.1:8092/v1/videos/${video_id}" | jq .status
curl -fL "http://127.0.0.1:8092/v1/videos/${video_id}/content" -o "${video_id}.mp4"
```

`POST /v1/videos/sync` returns the raw video bytes in one call instead. Lifecycle, retention,
listing, deletion, and error codes are in the [Videos API reference](docs/serving/videos_api.md).

### Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Process and worker health |
| `GET /ready` | Model readiness after load and smoke test |
| `GET /v1/models` | The model served by this process |
| `POST /v1/chat/completions` | Text-to-image generation |
| `POST /v1/videos` | Create an asynchronous video job |
| `POST /v1/videos/sync` | Generate a video and return the bytes |
| `GET /v1/videos`, `GET /v1/videos/{id}` | List jobs, read one job's status |
| `GET /v1/videos/{id}/content` | Download a completed video |
| `DELETE /v1/videos/{id}` | Cancel a queued job or delete a finished one |

Logs go to the console and `./logs/`, capped at 5 MiB per file and rotated on size or date.
Stop the server with `Ctrl+C` or `SIGTERM`.

### Observability

The server logs each stage's start and completion with its duration (`stage.start`,
`stage.complete`), not the device memory each stage keeps resident. A per-request stage
breakdown and per-stage memory (from those logs plus `neuron-monitor` on Trainium or
`nvidia-smi` on GPUs) are planned as a skill under `.claude/skills/`, so the serving code
itself carries no instrumentation.

### Hardening

**API key.** Authentication is off until a key is set with `--api-key` or `DIFFLET_API_KEY`;
the flag wins. With a key, every `/v1` route requires `Authorization: Bearer <key>` and
rejects bad credentials with `401` before the body is parsed. `/health` and `/ready` stay
open. This is one shared service key, not per-user authorization, so still put a remotely
exposed port behind a security group or a TLS-terminating reverse proxy.

**S3 output.** To upload generated bytes to a private bucket and return an expiring presigned
URL instead of a data URL, copy `.env.example` to `.env` and set the bucket and region. IAM
roles, S3-compatible providers, and URL lifetime are covered in
[docs/serving/s3.md](docs/serving/s3.md).

## Go further

**Serve several shapes from one worker.** `--shapes` compiles one bucketed artifact that
shares a single weight copy on device. Requests outside the set are rejected with
`profile_mismatch`.

```bash
difflet serve --model-id hunyuanvideo-community/HunyuanVideo --tp-degree 4 \
  --shapes 320x512x61,320x512x33 --host 0.0.0.0 --port 8092
```

**Batch many prompts through one loaded model.** `--requests` takes a JSONL file with one
request per line. Pair it with `--dp` to spread the batch across replicas.

```bash
printf '%s\n' \
  '{"prompt": "a red fox in snow", "output": "fox.png", "seed": 1}' \
  '{"prompt": "a blue whale at dusk", "output": "whale.png", "seed": 2, "steps": 20}' \
  > batch.jsonl
difflet generate --model-id black-forest-labs/FLUX.1-dev --tp-degree 2 --dp 2 \
  --height 1024 --width 1024 --requests batch.jsonl
```

Optional per-line keys: `negative_prompt`, `guidance_scale`, `steps`.

**Skip denoise steps with TeaCache.** Fixed cadence needs no setup; adaptive mode uses a
calibration file to hit a target speedup.

```bash
difflet run ... --teacache-cadence 2                                  # skip every other step
difflet run ... --teacache-speedup 1.5 --teacache-calibration cal.json # adaptive
```

**Run the stages separately.** `download`, `compile`, and `generate` can each be run on
their own to isolate a failure or inspect intermediate tensors. Per-model recipes, artifact
paths, and stage core counts are in [docs/cli-staged-commands.md](docs/cli-staged-commands.md).

## CLI reference

| Command | Purpose |
|---|---|
| `difflet run` | `download` + `compile` + `generate` in one shot |
| `difflet download` | Fetch model weights from Hugging Face |
| `difflet compile` | AOT-compile the model and cache it on disk |
| `difflet generate` | Run inference against a prior `compile` |
| `difflet serve` | Start the OpenAI-compatible image and video server |
| `difflet plan` | Rank the parallel configurations this host and model allow |
| `difflet cache ls` | List compiled artifacts with their shapes, TP, and dtype |
| `difflet clean` | Delete Neuron compiler scratch from the working directory |

### Flags

| Flag | Applies to | Description |
|---|---|---|
| `--model-id` | all | Hugging Face model id (see [Supported models](#supported-models)) |
| `--revision REV` | download, compile, generate, run, serve | Pin a Hub revision |
| `--tp-degree N` | compile, generate, run, serve | Tensor-parallel degree (default: registry default) |
| `--cp-degree N` | compile, generate, run, serve | Context-parallel degree (default 1) |
| `--cp-mode {gather_kv,ring,ulysses}` | compile, generate, run, serve | Context-parallel attention strategy |
| `--cfg-parallel` | compile, generate, run, serve | Split the uncond/cond CFG passes across 2 ranks (true-CFG models only) |
| `--sp` | compile, generate, run, serve | Megatron-style sequence parallelism over the TP group (Flux, Qwen-Image, Wan, HunyuanVideo) |
| `--dp N` | compile, generate, run | Data-parallel replicas; a router spreads requests across them |
| `--dp-schedule {round_robin,least_loaded}` | compile, generate, run | Request-to-replica schedule for `--dp > 1` |
| `--mode {latency,throughput,mixed}` | compile, generate, run | Preset that picks `dp`, `cfg`, and `cp` for the model class |
| `--total-cores N` | compile, generate, run | Core budget for validation (default: `NEURON_RT_NUM_CORES`) |
| `--height/--width/--num-frames` | compile, generate, run, serve | Output shape (default: the model's registry shape) |
| `--shapes HxWxF[,...]` | compile, generate, run, serve | Compile or serve several shapes from one bucketed artifact |
| `--teacache-cadence N` | generate, run | Skip every N-th denoise step |
| `--teacache-speedup F`, `--teacache-calibration PATH` | generate, run, serve | Adaptive TeaCache target and calibration file |
| `--teacache-online-delta ALPHA` | generate, run | Probe-free online-delta TeaCache |
| `--prompt` | generate, run | Text prompt |
| `--requests FILE` | generate, run | JSONL batch file, one request per line |
| `--output PATH` | generate, run | Output file (`.png` or `.mp4`) |
| `--steps N`, `--guidance-scale F`, `--seed N` | generate, run | Sampler settings (seed default 42) |
| `--cache-dir PATH` | compile, generate, run, serve, cache | Compiled-artifact cache root (default `~/.cache/difflet/`) |
| `--force` | compile, generate, run, serve | Recompile even if a valid cache entry exists |
| `--work-dir PATH`, `--keep-work-dir` | generate, run | Inter-stage tensor directory for staged models |
| `--host`, `--port`, `--api-key` | serve | Bind address and optional shared API key |
| `--clip-placement {host,neuron}` | serve | HunyuanVideo CLIP placement |

`difflet <command> --help` is the authoritative list.

### Where things land

| What | Where |
|---|---|
| Downloaded weights | `~/.cache/huggingface/hub/models--<org>--<name>/` |
| Compiled artifacts | `~/.cache/difflet/` (override with `--cache-dir` or `DIFFLET_COMPILE_CACHE`) |
| Inter-stage tensors | `--work-dir` (default `~/.cache/difflet/work/<model>/`) |
| Serving logs | `./logs/` |
| Compiler scratch | the working directory; remove with `difflet clean` |

## Troubleshooting

**On GPUs**

- **"`DIFFLET_BACKEND=cuda` selected but `torch.cuda.is_available()` is False".** The
  installed torch is a CPU build, or a CUDA build without a matching driver. Reinstall torch
  from the `cu128` index as shown in [Install](#install).

**On Trainium**

- **`ImportError` or a missing `libneuronpjrt-path` binary.** Activate the venv. The Neuron
  runtime shells out to helpers in `.venv/bin`, so calling `.venv/bin/difflet` by absolute
  path without `source .venv/bin/activate` fails at import.
- **`NCC_ISMP902` internal compiler error on the Flux text encoder.** You upgraded
  `neuronx-cc` to 2.27. The lock pins 2.26.6360.0; reinstall from
  `requirements-neuron.lock`.
- **`NCC_INLA001` or `NCC_IBIR243` on HunyuanVideo.** The `tp2 cp2` gather-KV graph crashes
  the compiler. Use `--cp-mode ring` or `--tp-degree 4 --sp`.
- **`NRT` allocation failure while loading a DP replica.** HunyuanVideo does not fit on
  2-core replicas at its default shape. Use `--tp-degree 4` without `--dp` or a smaller
  shape.
- **"Cores are already held by another process".** A previous run or server is still
  resident. Find it with `neuron-ls` or `neuron-top`, stop it, then retry. `difflet plan`
  reports busy cores.
- **A `--backend neuron` launch times out in a collective while loading.** A cold launch can
  keep ranks waiting on a peer for a long time (rank 0 reading a host text encoder, a cold
  warm-up compile). The CLI gives its torchrun ranks a 3-hour process-group timeout; torch's
  own default is 30 minutes. Export `DIFFLET_NEURON_PG_TIMEOUT_S=<seconds>` to choose another
  one; an exported value always wins.
- **Compile finished but generate recompiles.** Any change to shape, parallel flags,
  or toolchain version changes the cache key. `difflet cache ls` shows what is cached.
- **Disk filling up.** Compiler scratch accumulates in the working directory; run
  `difflet clean --dry-run` then `difflet clean`. The artifact cache itself is under
  `~/.cache/difflet/` and can be deleted by hand.

## Developer guide

Architecture, the backend abstraction, the compile cache, parallelism internals, the runtime
protocol, and how to port a new model are in **[DEVELOPER.md](DEVELOPER.md)**.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, checks, and the
PR checklist, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Report security issues privately
as described in [SECURITY.md](SECURITY.md).

## License

Apache License 2.0. See [`LICENSE`](LICENSE). Difflet incorporates code derived from
third-party Apache-2.0 projects; attributions are in [`NOTICE`](NOTICE) and at the top of
each derived file.
