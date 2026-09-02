# Difflet

**A focused inference engine for diffusion transformers (DiTs) on AWS Trainium.**

[Installation](#installation) · [Quick start](#quick-start) · [Serving](#serving) · [Python API](#python-api) · [CLI reference](#cli-reference) · [Verified matrix](#verified-parallelism-matrix) · [Developer guide](DEVELOPER.md)

## About

Difflet runs image and video diffusion models on AWS Trainium with a single, consistent
interface. It handles the full lifecycle — model download, ahead-of-time (AOT) compilation,
on-disk artifact caching, and SPMD multi-core execution — so you can go from a Hugging Face
model id to a generated image or video in one command.

Three entry points expose the same engine:

- **`difflet serve`** — a resident, OpenAI-compatible HTTP server for image and video models.
- **`difflet` CLI** — `download → compile → generate`, or `difflet run` to do all three at once.
- **`DiffletPipeline`** — a Python API mirroring `diffusers` for use inside your own scripts.

Core capabilities:

- **One engine, many models** — Flux, Wan 2.2, HunyuanVideo, Qwen-Image, and LTX-2 behind a
  single CLI and registry.
- **Fully on-device** — text encoders, the DiT backbone, and the VAE all run on Trainium for
  the supported models (no CPU fallbacks in the hot path).
- **Content-addressed compile cache** — AOT artifacts are hashed by model, parallel config,
  shape, and toolchain versions, so a warm cache skips straight to load + denoise.
- **Tensor + context + CFG + sequence parallelism** — scale a single generation across
  NeuronCores with `tp`, `cp`, CFG-parallel, and Megatron-style sequence-parallel (`--sp`)
  modes; every mode is exercised on-device by the verification matrix
  (`scripts/verify_cli.py`).
- **Latency tooling** — optional TeaCache step-skipping for faster denoise.

## Supported models

| Model | Type | Resolution (default) | Notes |
|---|---|---|---|
| [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Text-to-image | 1024×1024 | Single-process; CFG-parallel available |
| [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) | Text-to-image | 1024×1024 | 3-stage (text → generate → vae) |
| [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | Text-to-video | 480×832×9 | 2-stage (transformer → vae); CFG-parallel |
| [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) | Text-to-video | 480×832×9 | Same runtime as Wan 2.2 |
| [hunyuanvideo-community/HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) | Text-to-video | 320×512×61 | 3-stage (clip → llama → generate) |
| [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) | Text-to-video | 512×768×121 | Single-process; CP/SP not supported (use `tp=4`); exports `.mp4` |

## Feature support

Which acceleration features each model supports. ✅ = supported, ❌ = not supported.

| Model | TP | CP — all-gather | CP — ring | SP | TeaCache (adaptive) | TeaCache (fixed cadence) | CFG-parallel |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| FLUX.1-dev | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ ¹ |
| Qwen-Image | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ ¹ |
| Wan 2.2 / 2.1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| HunyuanVideo | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ ¹ |
| LTX-2 | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |

**Feature legend**

- **TP** — tensor parallelism (`--tp-degree`). Splits each layer across NeuronCores.
- **CP — all-gather** — context parallelism with gather-KV attention (`--cp-degree N --cp-mode gather_kv`, the default). Splits the sequence across ranks.
- **CP — ring** — context parallelism with ring attention (`--cp-degree N --cp-mode ring`). Lower memory than all-gather for long sequences.
- **SP** — Megatron-style sequence parallelism (`--sp`). Shards the norm/modulation/residual regions along the sequence axis across the existing tensor-parallel group; `world_size` is unchanged. Mutually exclusive with `--cp-degree > 1`.
- **TeaCache (adaptive)** — calibration-driven step-skipping (`--teacache-speedup` / `--teacache-online-delta`, with `--teacache-calibration`).
- **TeaCache (fixed cadence)** — blind skip-every-N-steps (`--teacache-cadence N`, no calibration needed).
- **CFG-parallel** — splits the conditional/unconditional CFG passes across 2 data-parallel ranks (`--cfg-parallel`). Only meaningful for true two-pass classifier-free guidance.

**Notes**

1. Guidance-distilled model (single forward pass with the guidance scale baked into the timestep embedding) — there is no second CFG branch to split.

Context parallelism (`--cp-degree > 1`) and CFG-parallel both consume the data-parallel lanes, so they are mutually exclusive (and each is mutually exclusive with `--sp`). `world_size = tp_degree × cp_degree` (or `tp_degree × 2` with CFG-parallel; `--sp` leaves it unchanged).

## Installation

### Prerequisites

- **Instance** — AWS Trainium v2 (validated on `trn2.3xlarge`: one Trainium2 chip, 96 GiB HBM,
  presented as 4 logical NeuronCores under the Trn2 default `LNC=2`). Other Trn2 shapes should
  work; the tensor-parallel degree must divide the number of visible NeuronCores.
- **Runtime** — a Neuron PyTorch 2.9 environment with `neuronx-cc`, `neuronx-distributed`, `nki`,
  `nkilib`, `torch-neuronx`, and `libneuronxla`. The reference development image bundles all of these at `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`.
- **Python** — 3.10+.

### Install

```bash
git clone git@github.com:ai-decentralized/Difflet.git
cd Difflet
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
# This installs the `difflet` CLI on your `PATH`.
pip install -e .
# Setup your huggingface credential
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
  --steps 40 --guidance-scale 4.0 --seed 42 \
  --prompt "a cat walking through a garden" \
  --output cat.mp4
```

The first run triggers AOT compilation (~10–15 minutes for Flux on 4 cores; the larger video
DiTs take longer). Compiled artifacts are cached under `~/.cache/difflet/`; subsequent runs hit
the cache and skip straight to load + denoise.

> **Parallelism cheat-sheet.** Flux, Wan, HunyuanVideo, and Qwen-Image support `--tp-degree 2
> --cp-degree 2` (world size 4) on a 4-core host; Flux, Wan, and HunyuanVideo also support
> `--tp-degree 4 --sp`. LTX-2 does not support context or sequence parallelism (use
> `--tp-degree 4`, optionally with `--cfg-parallel` at `--tp-degree 2`).

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

## Serving

`difflet serve` keeps one image or video model loaded in a resident Trainium
worker. S3 is not required: without an S3 bucket configuration, generated images
are returned as Base64 data URLs in the OpenAI-style Chat Completions response,
while completed videos remain available through the Videos content endpoint.

For asynchronous and synchronous video generation, request fields, lifecycle,
download, deletion, retention, and S3 behavior, see the
[Videos API reference](docs/serving/videos_api.md).

Start Flux on a four-core `trn2.3xlarge`:

```bash
difflet serve \
  --model-id black-forest-labs/FLUX.1-dev \
  --tp-degree 4 \
  --cp-degree 1 \
  --height 1024 \
  --width 1024 \
  --host 0.0.0.0 \
  --port 8092
```

Generate and save an image locally on the client:

```bash
curl -sS -X POST http://127.0.0.1:8092/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "black-forest-labs/FLUX.1-dev",
    "messages": [
      {"role": "user", "content": "a small red sailboat on a calm blue lake"}
    ],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 20,
      "guidance_scale": 3.5,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' \
    | cut -d',' -f2- | base64 -d > output.png
```

When private S3 storage is configured, the same `image_url.url` field contains
an expiring presigned URL instead of a data URL.

### Optional API-key authentication

API authentication is disabled when no key is configured. Set one key either
with `--api-key` or through `DIFFLET_API_KEY`; the CLI flag takes precedence:

```bash
difflet serve \
  --model-id black-forest-labs/FLUX.1-dev \
  --tp-degree 4 --cp-degree 1 \
  --height 1024 --width 1024 \
  --host 0.0.0.0 --port 8092 \
  --api-key 'replace-with-a-secret'
```

When enabled, every `/v1` request requires a Bearer token:

```bash
curl http://127.0.0.1:8092/v1/models \
  -H 'Authorization: Bearer replace-with-a-secret'
```

Missing or incorrect credentials return `401 {"error":"Unauthorized"}` before
the request body is parsed or generation capacity is reserved. `/health` and
`/ready` intentionally remain unauthenticated. This is a single shared service
key, not tenant isolation or per-user authorization.

### Optional S3 artifact storage

With no S3 variables, Difflet returns the generated PNG inline as a Base64 data
URL. To upload generated PNG bytes to a private S3 bucket and return an expiring
`image_url`, create an environment file:

```bash
cp .env.example .env
```

Configure the AWS S3 bucket and region:

```dotenv
DIFFLET_S3_BUCKET=difflet
DIFFLET_S3_REGION=ap-southeast-4
DIFFLET_S3_PREFIX=difflet
```

Run the server from the directory containing `.env`. If the file exists, it is
loaded automatically without overriding variables already exported by the shell.
Do not commit `.env` or credentials. Boto3 uses its standard credential provider
chain; on EC2, attach an IAM role to the instance instead of storing access keys
in `.env`. The role needs `s3:PutObject` and `s3:GetObject` access to
`arn:aws:s3:::difflet/difflet/*`.

Difflet always returns an S3 presigned URL whose access lifetime is controlled by
the server-owned `artifact_ttl_seconds` setting (currently 3600 seconds). Keep S3
Block Public Access enabled. A lifecycle rule may delete expired objects later;
URL expiry and object deletion are independent.

For non-AWS S3-compatible providers, set `DIFFLET_S3_ENDPOINT_URL` explicitly.
AWS S3 does not require this setting; boto3 derives the endpoint from
`DIFFLET_S3_REGION`. Providers that do not use the boto3 default credential chain
may also set `DIFFLET_S3_ACCESS_KEY_ID` and `DIFFLET_S3_SECRET_ACCESS_KEY`;
`DIFFLET_S3_SESSION_TOKEN` is optional for temporary credentials. The access key
and secret key must either both be present or both be absent. Difflet uses SigV4
and virtual-hosted addressing for AWS presigned URLs. Compatible providers that
require path-style URLs may set `DIFFLET_S3_ADDRESSING_STYLE=path`.

### Qwen-Image and startup behavior

The other supported serving model is Qwen-Image. Start it with the same fixed
four-core profile:

```bash
difflet serve \
  --model-id Qwen/Qwen-Image \
  --tp-degree 4 \
  --cp-degree 1 \
  --height 1024 \
  --width 1024 \
  --host 0.0.0.0 \
  --port 8092
```

The first startup downloads missing Hugging Face weights, compiles missing serving artifacts,
loads the resident worker, and runs a real generation smoke test before readiness opens. Warm
restarts reuse the immutable compile cache. A four-core host cannot run these two profiles, or a
CLI generation and one of these servers, at the same time because each profile owns all four
NeuronCores.

### Check readiness and generate

```bash
curl http://127.0.0.1:8092/health
curl http://127.0.0.1:8092/ready
curl http://127.0.0.1:8092/v1/models
```

Send an image-generation request:

```bash
curl -sS -X POST http://127.0.0.1:8092/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "black-forest-labs/FLUX.1-dev",
    "messages": [
      {"role": "user", "content": "a small red sailboat on a calm blue lake"}
    ],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 20,
      "guidance_scale": 3.5,
      "seed": 42
    }
  }'
```

Request `height` and `width` must match the server's startup profile. Inference steps must be
between 1 and 50. The response image is available at
`choices[0].message.content[0].image_url.url`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Process and worker health |
| `GET /ready` | Model readiness after load and smoke |
| `GET /v1/models` | The model served by this process |
| `POST /v1/chat/completions` | Text-to-image generation |

Serving logs are written to both the console and `./logs/`. Log files are capped at 5 MiB and
rotated on size or when the date changes. Stop the server with `Ctrl+C` or `SIGTERM`. The API
supports the optional shared key described above; still protect a remotely exposed port with a
security group, TLS-terminating reverse proxy, or equivalent network access control.

## CLI reference

The CLI has four model subcommands. `run` is the one-shot path; the other three let you run and
verify each step independently — useful for debugging compilation or inspecting intermediate
artifacts. `difflet clean` is a housekeeping command that takes no `--model-id`.

| Command | Purpose |
|---|---|
| `difflet download` | Fetch model weights from Hugging Face |
| `difflet compile` | AOT-compile the model NEFFs and cache them on disk |
| `difflet generate` | Run inference (requires a prior `compile`) |
| `difflet run` | `download` + `compile` + `generate` in one shot |
| `difflet clean` | Delete Neuron compiler scratch from the working directory |

### Common flags

| Flag | Applies to | Description |
|---|---|---|
| `--model-id` | all | Hugging Face model id (see [Supported models](#supported-models)) |
| `--tp-degree N` | compile, generate, run | Tensor-parallel degree (default: registry default) |
| `--cp-degree N` | compile, generate, run | Context-parallel degree (default: 1) |
| `--cp-mode {gather_kv,ring}` | compile, generate, run | Context-parallel attention strategy |
| `--cfg-parallel` | compile, generate, run | Split the uncond/cond CFG passes across 2 ranks (true-CFG models only) |
| `--sp` | compile, generate, run | Megatron-style sequence parallelism over the TP group (Flux, Wan, HunyuanVideo) |
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
  --steps 40 --guidance-scale 4.0 --seed 42 \
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

### Cleaning up compiler scratch

Device compiles leave scratch in the process working directory: per-kernel cache directories
named with a 16-hex-char hash, one `neuronxcc-<id>/` work directory per compiler invocation, and
the `log-neuron-cc.txt` / `global_metric_store.json` / `PostSPMDPassesExecutionDuration.txt`
diagnostic files. They are gitignored but accumulate across runs.

```bash
difflet clean --dry-run     # list what would go
difflet clean               # delete it
difflet clean --dir PATH    # sweep somewhere other than the cwd
```

Only direct children of the target directory are touched, symlinks are never followed, and a
hash-named directory holding anything other than compiler output is reported and left in place.
This does **not** touch the compiled-artifact cache under `~/.cache/difflet/` — remove that by
hand (`rm -rf ~/.cache/difflet`) or recompile over it with `--force`.

## Verified parallelism matrix

`scripts/verify_cli.py` runs the full CLI (`download → compile → timed generate`) for every
model × parallel-config cell on a 4-core `trn2.3xlarge`, with per-cell logs and a
machine-readable `results.json`. Each config uses exactly 4 NeuronCores
(`world_size = (2 if cfg else 1) × cp × tp`). Latest full run:

| Model | `tp4` | `tp2cp2` | `tp2cfg` | `tp4sp` |
|---|:---:|:---:|:---:|:---:|
| flux | ✅ | ✅ | — ¹ | ✅ |
| qwen_image | ✅ | ✅ | — ¹ | — ² |
| ltx_2 | ✅ | — ³ | ✅ | — ² |
| wan (2.2) | ✅ | ✅ | ✅ | ✅ |
| wan2_1 | ✅ | ✅ | ✅ | ✅ |
| hunyuan_video | ✅ | ✗ ⁴ | — ¹ | ✅ |
| hunyuan_video_15 | ✗ ⁵ | — ³ | — ¹ | — ² |

✅ compile + generate pass with output artifact · — auto-skipped (unsupported combination) ·
✗ expected failure (known gap). ¹ guidance-distilled, no CFG branch. ² SP not supported.
³ CP not supported. ⁴ `NCC_INLA001` compiler crash (see [DEVELOPER.md](DEVELOPER.md)). ⁵ HunyuanVideo 1.5 is a scaffold
(`download` only).

```bash
python scripts/verify_cli.py                     # full matrix
python scripts/verify_cli.py --models wan ltx_2  # subset
```

## Developer guide

Architecture, the backend abstraction, the compile cache, parallelism internals, the runtime
protocol, and instructions for porting a new model live in **[DEVELOPER.md](DEVELOPER.md)**.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

Difflet incorporates code derived from third-party Apache-2.0 projects; attributions and
modification banners are in [`NOTICE`](NOTICE) and at the top of each derived file.
