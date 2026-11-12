<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/difflet-logo-white.svg">
  <img alt="Difflet" src="docs/assets/difflet-logo-black.svg" width="260">
</picture>
</div>

# 
**Run diffusion transformers on NVIDIA GPUs, AWS Trainium and (experimentally) Cloud TPU: FLUX, Qwen-Image, Wan, HunyuanVideo, LTX-2 and MiniMax-H3 (video with audio), from one CLI, one Python API, and one OpenAI-compatible server.**

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
- [2026-11-02] **HunyuanVideo 1.5 generates video end to end on Trainium.** The 480p text-to-video model runs on both the AOT and the native backend with true CFG, TP, CFG-parallel, SP, CP (Ulysses, all-gather), DP, JSONL batching and TeaCache, at the default 480×848×121 shape; the VAE decodes on the host by default. See the [AOT verification](docs/verification/hv15-aot-evidence.md).
- [2026-10-31] **The CUDA backend grows into a shared GPU platform.** Every GPU model now goes through one application, serving and CLI path, with data parallelism (`--dp`, `serve --dp-degree`; two replicas on 2 × A100, output identical to one card), CFG-parallel Wan 2.2 on two GPUs, `--torch-compile`, and experimental single-GPU HunyuanVideo. See the [GPU backend results](docs/verification/gpu-backend-results.md).
- [2026-10-29] **Difflet runs on Trainium through native PyTorch.** The new `neuron` backend executes models with torch-neuronx, eagerly or through `torch.compile`, without ahead-of-time tracing, for all seven Trainium models (FLUX.1-dev, Qwen-Image, Wan 2.1/2.2, HunyuanVideo 1.0/1.5, LTX-2) with TP, SP, CP, CFG-parallel and DP layouts, on Trainium2 and Trainium3. See the [native backend results](docs/verification/neuron-backend-results.md).
- [2026-09-30] **MiniMax-H3 text-to-video with audio is verified on Trainium2.** The four-stage CLI generated a 256×448, 124-frame MP4 with stereo audio on `trn2.3xlarge` using TP4. See the [end-to-end verification](docs/design/minimax_h3/01_e2e_reverification_2026-09-30.md).
- [2026-09-10] **First public release.**

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
| NVIDIA GPUs | `cuda` | ✅ Supported (subset) | FLUX.1-dev, Qwen-Image, Wan 2.2; HunyuanVideo and LTX-2 experimental | DP, CFG-parallel (Wan) | [GPU features](#nvidia-gpus) |
| AWS Trainium2 (AOT) | `trainium` | ✅ Supported | All [supported models](#supported-models) | TP, CP, SP, CFG-parallel, DP | [Trainium features](#trainium) |
| AWS Trainium2 / Trainium3 (native PyTorch) | `neuron` | ✅ Supported | FLUX.1-dev, Qwen-Image, Wan 2.1/2.2, HunyuanVideo 1.0/1.5, LTX-2 | TP, CP, SP, CFG-parallel, DP; FP8 (`--quant mxfp8`) on Trainium3 | [Native backend results](docs/verification/neuron-backend-results.md) |
| Cloud TPU | `tpu` | 🧪 Experimental | FLUX.1-dev, Qwen-Image, Wan, HunyuanVideo, LTX-2 (`difflet serve` and the benchmark runners) | TP | [v5e benchmarks](benchmark/v5e/RESULTS.md) |
| AMD ROCm | `rocm` | Planned | — | — | — |

Without `DIFFLET_BACKEND` (or `--backend`), Difflet picks the backend it detects on the host. It
checks the native Neuron stack (`torch_neuronx` without `torch_xla`), then Trainium (AOT), then
TPU, then CUDA, then ROCm, which only matters on a host with more than one accelerator. With
no accelerator detected it falls back to `trainium`, so on a GPU host whose torch build lacks
CUDA, set `DIFFLET_BACKEND=cuda` to get a clear error instead of a Neuron import failure.
A CPU reference backend (`cpu`) exists for numerical tests and CI. It is not meant for
generation.

On Cloud TPU the runtime is TorchTPU (`torch_tpu`), one process per chip, eager or `torch.compile`.
Probe-free TeaCache (`--teacache-cadence` / `--teacache-online-delta`) works in serving there.
`difflet compile/generate/run` and the Python API do not run on TPU.

## Supported models

| Model | Type | Default shape | Notes |
|---|---|---|---|
| [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Text-to-image | 1024×1024 | Single-process |
| [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) | Text-to-image | 1024×1024 | 3-stage (text → generate → vae) |
| [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | Text-to-video | 480×832×9 | 2-stage (transformer → vae); true CFG |
| [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) | Text-to-video | 480×832×9 | Same runtime as Wan 2.2 |
| [hunyuanvideo-community/HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) | Text-to-video | 320×512×61 | 3-stage (clip → llama → generate) |
| [hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v) | Text-to-video | 480×848×121 | 3-stage (encode → transformer → vae); true CFG (6.0), default `--tp-degree 2 --cfg-parallel`; on Trainium the VAE decodes on the host ¹⁶ |
| [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) | Text-to-video | 512×768×121 | Single-process; TP only; exports `.mp4` |
| [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | Text-to-video with audio | 768×1344×124 ⚠️ ¹³ | Verified at **256×448×124**; TP4; 4-stage CLI (text → generate → video_vae → audio_vae); `.mp4` with stereo audio |

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
step. Hardware coverage includes RTX PRO 6000 Blackwell (96 GB), A100 80 GB and B200 180GB;
see the per-model limits below.

| Model | CLI generation | Serving | `--torch-compile` | TeaCache (fixed cadence) | DP | CFG-parallel | TP / CP / SP | Multi-shape |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| FLUX.1-dev | ✅ image | ✅ image | ✅ | ❌ | ✅ ² | — ¹ | ❌ | ❌ |
| Qwen-Image | ✅ image | ⚠️ image ³ | ✅ | ✅ | ✅ | — ⁴ | ❌ | ❌ |
| Wan 2.2 | ✅ video | ✅ video | ✅ | ❌ | ⚠️ ⁵ | ✅ ⁶ | ❌ | ❌ |
| HunyuanVideo (original) | ⚠️ experimental ⁷ | ⚠️ experimental ⁷ | ⚠️ execution observed ⁷ | ❌ | Not measured | — ¹ | ❌ | ❌ |
| LTX-2 | ⚠️ video, CPU checks ⁸ | ⚠️ video, CPU checks ⁸ | ⚠️ Python qualified ⁸ | ❌ | Not measured | ❌ | ❌ | ❌ |

An option the GPU backend does not support is refused with a message naming it, not ignored.
TeaCache (adaptive), TAEF1, and `--host-vae` are refused on GPUs for these models.

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
7. Experimental single-GPU BF16 support for `hunyuanvideo-community/HunyuanVideo`.
   Eager and compiled Python, CLI and HTTP generation were verified at 320×512×61,
   50 steps on a 96 GB GPU. See the [HunyuanVideo CUDA guide](docs/verification/hunyuanvideo-cuda.md)
   for usage, accuracy and benchmark limits. HunyuanVideo 1.5 is a separate model.
8. LTX-2 single-GPU BF16 Python audiovisual generation passes eager/compiled correctness
   and performance checks on B200. CLI/HTTP video-only paths have CPU coverage; GPU frontend
   validation is not claimed. The opt-in profile requires explicit TP=1 and a pre-CUDA startup
   setting; see the [LTX-2 CUDA guide](docs/verification/ltx2-cuda.md).

Measured timings against stock diffusers, and the full pass/limit matrix, are in
[docs/verification/gpu-backend-results.md](docs/verification/gpu-backend-results.md).

### Trainium

**Parallelism**

| Model | TP | CP — all-gather | CP — ring | CP — ulysses | SP | CFG-parallel | DP |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| FLUX.1-dev | ✅ | ✅ | ✅ | ✅ | ✅ | — ⁹ | ✅ |
| Qwen-Image | ✅ | ✅ | ✅ | ✅ | ✅ | — ⁹ | ✅ |
| Wan 2.2 / 2.1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| HunyuanVideo | ✅ | ⚠️ ¹⁰ | — ¹⁴ | ✅ | ✅ | — ⁹ | ⚠️ ¹¹ |
| HunyuanVideo 1.5 | ✅ | ✅ | — ¹⁴ | ✅ | ⚠️ ¹⁵ | ✅ | ✅ |
| LTX-2 | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| MiniMax-H3 | ✅ TP4 | ❌ | ❌ | ❌ | ❌ | — ⁹ | ❌ |

**Runtime features**

| Model | CLI generation | Multi-shape compile | TeaCache (adaptive) | TeaCache (fixed cadence) | Serving | Batch (JSONL) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| FLUX.1-dev | ✅ image | ✅ | ✅ | ✅ | ✅ image | ✅ |
| Qwen-Image | ✅ image | ✅ | ✅ | ✅ | ✅ image | ✅ |
| Wan 2.2 / 2.1 | ✅ video | ✅ | ✅ | ✅ | ✅ video ¹² | ✅ |
| HunyuanVideo | ✅ video | ✅ | ✅ | ✅ | ✅ video | ✅ |
| HunyuanVideo 1.5 | ⚠️ video ¹⁶ | ❌ | ⚠️ ¹⁷ | ✅ | ❌ | ✅ |
| LTX-2 | ✅ video | ❌ | ✅ | ✅ | ✅ video | ✅ |
| MiniMax-H3 | ✅ video + stereo audio | ❌ | ❌ | ❌ | ❌ | ❌ |

**FP8 on Trainium3** (PyTorch-native `neuron` backend, `--quant mxfp8`): Qwen-Image's DiT
runs its attention q/k/v and MLP linears in OCP MXFP8 through nkilib's `matmul_mxfp8` NKI kernel,
no calibration needed. Measured on trn3.48xlarge at tp4, 1024×1024: DiT step 239 ms against
299 ms in bf16 (0.80×); against the bf16 image, PSNR 26.8–39.2 dB over 4 prompts × 3 seeds
(mean 32.5), SSIM 0.955–0.990, LPIPS 0.014–0.092 (bf16-vs-bf16 at another TP degree: 39–46 dB); and
the DiT weights at 60% of their bf16 bytes. FLUX.1-dev takes the same flag (quality 31.7 dB against bf16) but gains no step time at
tp4 / 1024×1024 yet: its single-stream blocks' fused output projection stays bf16. Refused on the
AoT `trainium` backend, on trn2 and for other models;
see [docs/cli-staged-commands.md](docs/cli-staged-commands.md#qwen-image-in-fp8-pytorch-native-neuron-backend-trainium3-only)
and the evidence in
[docs/verification/2026-10-31-qwen-image-fp8-native-trn3.md](docs/verification/2026-10-31-qwen-image-fp8-native-trn3.md).

**Notes**

9. Guidance-distilled model (single forward pass with the guidance scale baked into the timestep embedding) — there is no second CFG branch to split.
10. `tp2 cp2` with gather-KV hits a `neuronx-cc` internal error (`NCC_INLA001` / `NCC_IBIR243`) on the CP-degree-2 DiT graph; ulysses CP and `tp4 --sp` are the working multi-core paths. See [DEVELOPER.md](DEVELOPER.md).
11. DP works, but on a 4-core `trn2.3xlarge` each 2-core replica runs out of HBM loading the compiled VAE at the default 320×512×61 shape. Use a smaller shape or a host with more cores per replica.
12. Wan 2.1 is the qualified serving checkpoint. Wan 2.2 can be started for experiments but its dual-transformer path has not passed resident-serving acceptance.
13. MiniMax-H3 supports the Trainium four-stage CLI at TP4 and CP1: text and DiT use four cores, and each VAE uses one core. The verified end-to-end shape is **256×448×124**; pass `--height 256 --width 448 --num-frames 124`. The registry default **768×1344×124** is not qualified on `trn2.3xlarge`: the standard DiT graph exceeds device memory, and compilation with precomputed AdaLN exceeded host memory. The 640×1152×124 DiT compiled successfully but has not been verified end to end. Python API generation, resident serving, DP, JSONL batching, multi-shape compilation, and TeaCache are not supported for H3. See the [verification](docs/design/minimax_h3/01_e2e_reverification_2026-09-30.md) and [bring-up handoff](docs/design/minimax_h3/00_trainium_bringup_handoff.md).
14. Not applicable: the model always carries a padded-text attention mask, and the Trainium ring-attention kernel has no text-mask path. `--cp-mode ring` is refused (for HunyuanVideo 1.5 before any stage launches); use `--cp-mode ulysses` (or `gather_kv` for HunyuanVideo 1.5).
15. `--tp-degree 4 --sp` needs the image token count divisible by 4. It works at 320×512×61; at the default 480×848×121 it is refused before compile, so use the default `--tp-degree 2 --cfg-parallel` there.
16. The DiT runs on the NeuronCores; the VAE decodes on the host by default (diffusers' tiled decoder in fp32), because the compiled decoder does not build on neuronx-cc 2.26 (`--device-vae` opts into it). The host decode takes about 7 minutes at 320×512×61 and about 45 minutes at 480×848×121. Verified on device: TP2, TP2 × CFG2, TP4, TP4 + SP, TP2 × CP2 (ulysses and gather_kv), DP, JSONL batches and the default shape; see [docs/verification/hv15-aot-evidence.md](docs/verification/hv15-aot-evidence.md).
17. Available but not recommended for HunyuanVideo 1.5: the calibration signal is weak and the output is visibly degraded, and the probe loads a second copy of the DiT weights, so it does not fit at the default shape. Use `--teacache-cadence 2` (20 of 50 steps skipped, about 40% less denoise time).

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
  - **AWS Trainium3** through the native PyTorch backend (`--backend neuron`), which also
    runs on Trainium2; FP8 (`--quant mxfp8`) is Trainium3-only. The AOT `trainium` backend
    is qualified on Trainium2.
- Python 3.11 or newer.
- A Hugging Face account with access to the gated FLUX.1-dev repo, if you want Flux.
- Disk: weights are large (LTX-2 alone is 86 GB). On Trainium, compiled artifacts add tens of
  GB.

### Install

**On NVIDIA GPUs:**

```bash
git clone git@github.com:ai-decentralized/Difflet.git
cd Difflet
python3 -m venv .venv && source .venv/bin/activate
# torch must come from the CUDA 12.8 index (Blackwell needs a 12.8+ wheel);
# the version is the one in requirements-cuda.lock.
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

LTX-2 also has an experimental single-GPU CUDA path with joint audiovisual Python output
and an opt-in native-compatible BF16 profile. See the [LTX-2 CUDA guide](docs/verification/ltx2-cuda.md)
for its separate runtime, explicit TP=1/startup requirements and qualified eager/compiled results.

For HunyuanVideo CUDA, pass `parallel=DiffletParallelConfig(tp_degree=1)` explicitly.
Omitting `parallel` preserves the model-specific registry default, including HunyuanVideo's
Trainium topology. GPU Python API paths have unit coverage; see the
[per-model results](docs/verification/gpu-backend-results.md) for hardware coverage.

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
| Lowest latency, true-CFG model | `--tp-degree 2 --cfg-parallel` | Wan, LTX-2, HunyuanVideo 1.5 |
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
`stage.complete`). It does not report device memory; read that from `neuron-monitor` on
Trainium or `nvidia-smi` on GPUs while a request runs.

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
| `--backend NAME` | compile, generate, run | Hardware backend, one of the `DIFFLET_BACKEND` values (default: `$DIFFLET_BACKEND`, else auto-detected); `serve` reads `DIFFLET_BACKEND` |
| `--tp-degree N` | compile, generate, run, serve | Tensor-parallel degree (default: registry default) |
| `--cp-degree N` | compile, generate, run, serve | Context-parallel degree (default 1) |
| `--cp-mode {gather_kv,ring,ulysses}` | compile, generate, run, serve | Context-parallel attention strategy |
| `--cfg-parallel` | compile, generate, run, serve | Split the uncond/cond CFG passes across 2 ranks (true-CFG models only) |
| `--sp` | compile, generate, run, serve | Megatron-style sequence parallelism over the TP group (Flux, Qwen-Image, Wan, HunyuanVideo, HunyuanVideo 1.5) |
| `--dp N` | compile, generate, run | Data-parallel replicas; a router spreads requests across them |
| `--dp-schedule {round_robin,least_loaded}` | compile, generate, run | Request-to-replica schedule for `--dp > 1` |
| `--dp-degree N` | serve | CUDA backend only: resident replicas, one GPU each; requests go to the least loaded |
| `--mode {latency,throughput,mixed}` | compile, generate, run | Preset that picks `dp`, `cfg`, and `cp` for the model class |
| `--total-cores N` | compile, generate, run | Core budget for validation (default: `NEURON_RT_NUM_CORES`) |
| `--height/--width/--num-frames` | compile, generate, run, serve | Output shape (default: the model's registry shape) |
| `--shapes HxWxF[,...]` | compile, generate, run, serve | Compile or serve several shapes from one bucketed artifact |
| `--teacache-cadence N` | generate, run | Skip every N-th denoise step |
| `--teacache-speedup F`, `--teacache-calibration PATH` | generate, run, serve | Adaptive TeaCache target and calibration file |
| `--teacache-online-delta ALPHA` | generate, run | Probe-free online-delta TeaCache |
| `--torch-compile` | generate, run, serve | CUDA backend only: `torch.compile` the DiT blocks (eager is the default) |
| `--quant mxfp8` | compile, generate, run | `--backend neuron` on Trainium3: MXFP8 attention q/k/v and MLP linears (Qwen-Image, FLUX.1-dev) |
| `--prompt` | generate, run | Text prompt |
| `--negative-prompt TEXT`, `--true-cfg-scale F` | generate, run | Qwen-Image on `--backend neuron`: opt-in true classifier-free guidance |
| `--requests FILE` | generate, run | JSONL batch file, one request per line |
| `--output PATH` | generate, run | Output file (`.png` or `.mp4`) |
| `--steps N`, `--guidance-scale F`, `--seed N` | generate, run | Sampler settings (seed default 42) |
| `--cache-dir PATH` | compile, generate, run, serve, cache | Compiled-artifact cache root (default `~/.cache/difflet/`) |
| `--force` | compile, generate, run, serve | Recompile even if a valid cache entry exists |
| `--work-dir PATH`, `--keep-work-dir` | generate, run | Inter-stage tensor directory for staged models |
| `--host-vae`, `--device-vae` | compile, generate, run; `--host-vae` also serve | VAE placement: decode on the host CPU, or in a device VAE stage of its own (model-specific; see `--help`) |
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
  the compiler. Use `--cp-mode ulysses` or `--tp-degree 4 --sp`.
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
