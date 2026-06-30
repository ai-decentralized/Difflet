# Difflet

**A focused inference engine for diffusion transformers (DiTs) on AWS Trainium.**

[Installation](#installation) · [Quick start](#quick-start) · [CLI reference](#cli-reference) · [Developer guide](DEVELOPER.md)

## About

Difflet runs image and video diffusion models on AWS Trainium with a single, consistent
interface. It handles the full lifecycle — model download, ahead-of-time (AOT) compilation,
on-disk artifact caching, and SPMD multi-core execution — so you can go from a Hugging Face
model id to a generated image or video in one command.

Two entry points expose the same engine:

- **`difflet` CLI** — `download → compile → generate`, or `difflet run` to do all three at once.
- **`DiffletPipeline`** — a Python API mirroring `diffusers` for use inside your own scripts.

Core capabilities:

- **One engine, many models** — Flux, Wan 2.2, HunyuanVideo, Qwen-Image, and LTX-2 behind a
  single CLI and registry.
- **Fully on-device** — text encoders, the DiT backbone, and the VAE all run on Trainium for
  the supported models (no CPU fallbacks in the hot path).
- **Content-addressed compile cache** — AOT artifacts are hashed by model, parallel config,
  shape, and toolchain versions, so a warm cache skips straight to load + denoise.
- **Tensor + context + CFG parallelism** — scale a single generation across NeuronCores with
  `tp`, `cp`, and CFG-parallel modes.
- **Latency tooling** — optional TeaCache step-skipping for faster denoise.

## Supported models

| Model | Type | Resolution (default) | Notes |
|---|---|---|---|
| [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Text-to-image | 1024×1024 | Single-process; CFG-parallel available |
| [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) | Text-to-image | 1024×1024 | 3-stage (text → generate → vae) |
| [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | Text-to-video | 480×832×9 | 2-stage (transformer → vae); CFG-parallel |
| [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) | Text-to-video | 480×832×9 | Same runtime as Wan 2.2 |
| [hunyuanvideo-community/HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) | Text-to-video | 320×512×61 | 3-stage (clip → llama → generate) |
| [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) | Text-to-video | 512×768×121 | Single-process; CP not supported (use `tp=4`) |

## Feature support

Which acceleration features each model supports. ✅ = supported, ❌ = not supported.

| Model | TP | CP — all-gather | CP — ring | TeaCache (adaptive) | TeaCache (fixed cadence) | CFG-parallel |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| FLUX.1-dev | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ ¹ |
| Qwen-Image | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ ¹ |
| Wan 2.2 / 2.1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| HunyuanVideo | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ ¹ |
| LTX-2 | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |

**Feature legend**

- **TP** — tensor parallelism (`--tp-degree`). Splits each layer across NeuronCores.
- **CP — all-gather** — context parallelism with gather-KV attention (`--cp-degree N --cp-mode gather_kv`, the default). Splits the sequence across ranks.
- **CP — ring** — context parallelism with ring attention (`--cp-degree N --cp-mode ring`). Lower memory than all-gather for long sequences.
- **TeaCache (adaptive)** — calibration-driven step-skipping (`--teacache-speedup` / `--teacache-online-delta`, with `--teacache-calibration`).
- **TeaCache (fixed cadence)** — blind skip-every-N-steps (`--teacache-cadence N`, no calibration needed).
- **CFG-parallel** — splits the conditional/unconditional CFG passes across 2 data-parallel ranks (`--cfg-parallel`). Only meaningful for true two-pass classifier-free guidance.

**Notes**

1. Guidance-distilled model (single forward pass with the guidance scale baked into the timestep embedding) — there is no second CFG branch to split.

Context parallelism (`--cp-degree > 1`) and CFG-parallel both consume the data-parallel lanes, so they are mutually exclusive. `world_size = tp_degree × cp_degree` (or `tp_degree × 2` with CFG-parallel).

## Installation

### Prerequisites

- **Instance** — AWS Trainium v2 (validated on `trn2.3xlarge`: one Trainium2 chip, 96 GiB HBM,
  presented as 4 logical NeuronCores under the Trn2 default `LNC=2`). Other Trn2 shapes should
  work; the tensor-parallel degree must divide the number of visible NeuronCores.
- **Runtime** — a Neuron PyTorch 2.9 environment with `neuronx-cc`, `neuronx-distributed`, `nki`,
  `nkilib`, `torch-neuronx`, and `libneuronxla`. Pinned versions live in `pyproject.toml`. The
  reference development image bundles all of these at
  `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`.
- **Python** — 3.10+.

### Install

```bash
git clone git@github.com:ai-decentralized/Difflet.git
cd Difflet
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/active
pip install diffusers==0.38.0
pip install imageio-ffmpeg
pip install -e . --no-deps
```

This installs the `difflet` CLI on your `PATH`.

Authenticate with Hugging Face for gated checkpoints such as `black-forest-labs/FLUX.1-dev`:

```bash
huggingface-cli login
```

## Quick start

Generate an image or video in a single command with `difflet run`. It downloads the weights,
AOT-compiles the model (cached on first run), and generates — end to end.

```bash
# Text-to-image — Flux at 1024×1024
difflet run --model-id black-forest-labs/FLUX.1-dev \
  --tp-degree 2 --cp-degree 2 \
  --height 1024 --width 1024 \
  --prompt "a photorealistic cat sitting in a sunlit garden" \
  --output cat.png
```

```bash
# Text-to-video — Wan 2.2 at 480×832, 9 frames
difflet run --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --tp-degree 2 --cp-degree 2 \
  --height 480 --width 832 --num-frames 9 \
  --steps 50 --guidance-scale 1.0 --seed 42 \
  --prompt "a cat walking through a garden" \
  --output cat.mp4
```

The first run triggers AOT compilation (~10–15 minutes for Flux on 4 cores; the larger video
DiTs take longer). Compiled artifacts are cached under `~/.cache/difflet/`; subsequent runs hit
the cache and skip straight to load + denoise.

> **Parallelism cheat-sheet.** Flux, Wan, HunyuanVideo, and Qwen-Image support `--tp-degree 2
> --cp-degree 2` (world size 4) on a 4-core host. LTX-2 do not support
> context parallelism.

### Python API

The same engine is available as a library:

```python
from difflet import DiffletPipeline, DiffletParallelConfig

pipe = DiffletPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    parallel=DiffletParallelConfig(tp_degree=4),
    height=1024,
    width=1024,
)

image = pipe(
    prompt="a photorealistic cat sitting in a sunlit garden",
    num_inference_steps=28,
).images[0]
image.save("out.png")
```

## CLI reference

The CLI has four subcommands. `run` is the one-shot path; the other three let you run and verify
each step independently — useful for debugging compilation or inspecting intermediate artifacts.

| Command | Purpose |
|---|---|
| `difflet download` | Fetch model weights from Hugging Face |
| `difflet compile` | AOT-compile the model NEFFs and cache them on disk |
| `difflet generate` | Run inference (requires a prior `compile`) |
| `difflet run` | `download` + `compile` + `generate` in one shot |

### Common flags

| Flag | Applies to | Description |
|---|---|---|
| `--model-id` | all | Hugging Face model id (see [Supported models](#supported-models)) |
| `--tp-degree N` | compile, generate, run | Tensor-parallel degree (default: registry default) |
| `--cp-degree N` | compile, generate, run | Context-parallel degree (default: 1) |
| `--cp-mode {gather_kv,ring}` | compile, generate, run | Context-parallel attention strategy |
| `--cfg-parallel` | compile, generate, run | Split the uncond/cond CFG passes across 2 ranks (true-CFG models only) |
| `--height/--width/--num-frames` | compile, generate, run | Output shape (defaults to the model's registry shape) |
| `--cache-dir PATH` | compile, generate, run | Compiled-artifact cache root (default `~/.cache/difflet/`) |
| `--force` | compile, generate, run | Recompile even if a valid cache entry exists |
| `--prompt` | generate, run | Text prompt (required) |
| `--output PATH` | generate, run | Output file (`.png` or `.mp4`) (required) |
| `--steps N` | generate, run | Inference steps |
| `--guidance-scale F` | generate, run | Classifier-free guidance scale |
| `--seed N` | generate, run | RNG seed (default 42) |
| `--work-dir PATH` | generate, run | Directory for inter-stage tensors (staged models) |
| `--keep-work-dir` | generate, run | Keep the work-dir after a successful run |

TeaCache step-skipping flags (`--teacache-cadence`, `--teacache-online-delta`,
`--teacache-speedup`, `--teacache-calibration`) are available on `generate` and `run`.

### Staged usage

Run `download → compile → generate` separately to verify each step before proceeding. The
examples below mirror the validated configurations; capture logs with `tee` so a failed step is
easy to inspect.

```bash
mkdir -p /tmp/logs
```

> Override the compile-cache root with `DIFFLET_COMPILE_CACHE=<path>` or `--cache-dir <path>`.

#### Flux (single-process image model)

```bash
difflet download --model-id black-forest-labs/FLUX.1-dev \
  2>&1 | tee /tmp/logs/flux-download.log

difflet compile --model-id black-forest-labs/FLUX.1-dev \
  --tp-degree 2 --cp-degree 2 --height 1024 --width 1024 \
  2>&1 | tee /tmp/logs/flux-compile.log

difflet generate --model-id black-forest-labs/FLUX.1-dev \
  --tp-degree 2 --cp-degree 2 --height 1024 --width 1024 \
  --prompt "a cat sitting on a bench" --output flux.png \
  2>&1 | tee /tmp/logs/flux-generate.log
```

#### LTX-2 (single-process video model, CP not supported)

```bash
difflet download --model-id Lightricks/LTX-2 \
  2>&1 | tee /tmp/logs/ltx2-download.log

difflet compile --model-id Lightricks/LTX-2 \
  --tp-degree 4 --height 512 --width 768 --num-frames 121 \
  2>&1 | tee /tmp/logs/ltx2-compile.log

difflet generate --model-id Lightricks/LTX-2 \
  --tp-degree 4 --height 512 --width 768 --num-frames 121 \
  --prompt "a cat walking through a garden" --output ltx2.mp4 \
  2>&1 | tee /tmp/logs/ltx2-generate.log
```

#### Wan 2.2 (2-stage: transformer → vae)

`compile` spawns two subprocess stages (transformer @ `tp×cp` cores, VAE @ 1 core); `generate`
spawns the same stages in inference mode and passes a latent tensor between them.

```bash
difflet download --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  2>&1 | tee /tmp/logs/wan-download.log

difflet compile --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --tp-degree 2 --cp-degree 2 --height 480 --width 832 --num-frames 9 \
  2>&1 | tee /tmp/logs/wan-compile.log

difflet generate --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --tp-degree 2 --cp-degree 2 --height 480 --width 832 --num-frames 9 \
  --steps 50 --guidance-scale 1.0 --seed 42 \
  --prompt "a cat walking through a garden" --output wan.mp4 \
  --work-dir /tmp/logs/wan-work --keep-work-dir \
  2>&1 | tee /tmp/logs/wan-generate.log
```

#### HunyuanVideo (3-stage: clip → llama → generate)

`compile` spawns: clip (1 core), llama (`tp×cp` cores), generate (`tp×cp` cores) — all with
`NEURON_RT_VIRTUAL_CORE_SIZE=2`, set automatically by the orchestrator.

```bash
difflet download --model-id hunyuanvideo-community/HunyuanVideo \
  2>&1 | tee /tmp/logs/hv-download.log

difflet compile --model-id hunyuanvideo-community/HunyuanVideo \
  --tp-degree 2 --cp-degree 2 --height 320 --width 512 --num-frames 61 \
  2>&1 | tee /tmp/logs/hv-compile.log

difflet generate --model-id hunyuanvideo-community/HunyuanVideo \
  --tp-degree 2 --cp-degree 2 --height 320 --width 512 --num-frames 61 \
  --steps 50 --guidance-scale 6.0 --seed 42 \
  --prompt "a cat sitting on a bench" --output hunyuan.mp4 \
  --work-dir /tmp/logs/hv-work --keep-work-dir \
  2>&1 | tee /tmp/logs/hv-generate.log
```

#### Qwen-Image (3-stage: text → generate → vae)

```bash
difflet download --model-id Qwen/Qwen-Image \
  2>&1 | tee /tmp/logs/qwen-download.log

difflet compile --model-id Qwen/Qwen-Image \
  --tp-degree 2 --cp-degree 2 --height 1024 --width 1024 \
  2>&1 | tee /tmp/logs/qwen-compile.log

difflet generate --model-id Qwen/Qwen-Image \
  --tp-degree 2 --cp-degree 2 --height 1024 --width 1024 \
  --steps 50 --guidance-scale 7.5 --seed 42 \
  --prompt "a cat sitting on a bench" --output qwen.png \
  --work-dir /tmp/logs/qwen-work --keep-work-dir \
  2>&1 | tee /tmp/logs/qwen-generate.log
```

### Artifact locations

| Step | Where artifacts land |
|---|---|
| `download` | `~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<hash>/` |
| `compile` (single-process: flux, ltx-2) | `~/.cache/difflet/<model_name>/<hash>/` |
| `compile` (staged models) | `~/.cache/difflet/<stage-specific-dir>/` |
| `generate` inter-stage tensors | `--work-dir` path (default `~/.cache/difflet/work/<model>/`) |
| `generate` final output | `--output` path |

## Project status

| Milestone | State | Notes |
|---|---|---|
| M1 — Flux end-to-end | Done | `FLUX.1-dev` at 1024² in 28 steps |
| M2 — Wan 2.2 T2V | Done | 480×832×9 video tensor, TP=4 |
| M3 — HunyuanVideo | Done | T2V at 320×512×61, fully on-device |
| M4a — Qwen-Image | Done | Text-to-image fully on-device |
| M6 — LTX-2 | Done | Dual-stream segmented DiT, decoded end-to-end |
| M6a — HunyuanVideo 1.5 | In progress | Registered; `download` only |
| Context / CFG parallelism | Done | `cp_degree` and CFG-parallel for the supported models |
| Planned | — | 720p / longer-frame / I2V, TP refactor, NKI masked attention, Z-Image |

## Developer guide

Architecture, the backend abstraction, the compile cache, parallelism internals, the runtime
protocol, and instructions for porting a new model live in **[DEVELOPER.md](DEVELOPER.md)**.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

Difflet incorporates code derived from third-party Apache-2.0 projects; attributions and
modification banners are in [`NOTICE`](NOTICE) and at the top of each derived file.
