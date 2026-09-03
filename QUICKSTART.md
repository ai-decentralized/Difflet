# Quickstart: environment setup, CLI, and serving

Step-by-step guide for building the Difflet Neuron environment and running
image/video generation through `difflet run` and `difflet serve`. Every command
below was verified end-to-end on a `trn2.3xlarge` (4 logical NeuronCores).

## 1. Build the environment

Recent Neuron DLAMI releases no longer ship the prebuilt
`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference` venv. The environment is now
built from the committed recipe:

```bash
git clone git@github.com:ai-decentralized/Difflet.git
cd Difflet
./scripts/setup_env.sh          # creates <repo>/.venv from requirements-neuron.lock
source .venv/bin/activate       # puts difflet (and the Neuron toolchain) on PATH
huggingface-cli login           # FLUX.1-dev is a gated repo
```

Notes:

- Always activate the venv (or otherwise put `.venv/bin` on `PATH`) — the
  Neuron runtime shells out to helper binaries like `libneuronpjrt-path` that
  live in the venv's `bin/`, so calling `.venv/bin/difflet` by absolute path
  without activation fails at import time.
- The lock pins `neuronx-cc==2.26.6360.0`. Do not "upgrade" to 2.27.5334.0 —
  it fails with an internal compiler error (`NCC_ISMP902`) on the Flux CLIP
  text encoder.
- Repo scripts (`scripts/*.sh`) honor `PYTHON_BIN`; with the `/opt` venv gone,
  run them as `PYTHON_BIN=$PWD/.venv/bin/python ./scripts/<script>.sh`
  (the flux smoke scripts fall back to `<repo>/.venv` automatically).

Alternatively, build the same environment as a container image:

```bash
docker build -t difflet .
docker run --rm --device /dev/neuron0 \
  -v $HOME/.cache/difflet:/root/.cache/difflet \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  difflet run --model-id black-forest-labs/FLUX.1-dev --tp-degree 4 \
    --height 1024 --width 1024 --prompt "a cat" --output /out/cat.png
```

## 2. Sanity-check without touching the device

```bash
PYTHON_BIN=$PWD/.venv/bin/python ./scripts/check_quick.sh   # imports gate + unit tests, CPU-only
difflet --help
```

## 3. Generate with the CLI

`difflet run` downloads weights, AOT-compiles (cached under `~/.cache/difflet`),
and generates in one command. First runs pay the cold compile (~11 min for
Flux; ~80 min for Wan, dominated by the VAE); later runs with the same shape,
parallelism, and toolchain reuse the cache and start in about a minute.

Image — FLUX.1-dev at 1024×1024 on all 4 cores:

```bash
difflet run --model-id black-forest-labs/FLUX.1-dev \
  --tp-degree 4 \
  --height 1024 --width 1024 \
  --steps 28 --seed 42 \
  --prompt "a photorealistic cat sitting in a sunlit garden" \
  --output cat.png
```

Video — Wan2.2 at 480×832, 9 frames (keep 9 frames on-device; use `--host-vae`
for longer clips):

```bash
difflet run --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --tp-degree 2 --cp-degree 2 \
  --height 480 --width 832 --num-frames 9 \
  --steps 40 --guidance-scale 4.0 --seed 42 \
  --prompt "a cat walking through a garden" \
  --output cat.mp4
```

`--tp-degree × --cp-degree` must not exceed the visible NeuronCores (4 on
`trn2.3xlarge`); Flux's registry default is `tp_degree=8`, so pass `--tp-degree`
explicitly on a 4-core host. Inspect or prune compiled artifacts with
`difflet cache ls` and `difflet clean`.

## 4. Serve over HTTP

One serve profile owns all 4 cores — stop any running serve/CLI generation
first. Startup loads weights, compiles if the cache is cold, and runs a real
generation smoke before opening readiness; with a warm cache expect a few
minutes to `/ready`.

Image serving (Flux):

```bash
difflet serve --model-id black-forest-labs/FLUX.1-dev \
  --tp-degree 4 --cp-degree 1 \
  --height 1024 --width 1024 \
  --host 0.0.0.0 --port 8092
```

Video serving (Wan). Note: resident Wan serving requires `tp_degree=4` —
`tp2/cp2` works for `difflet run` but is rejected by `difflet serve`. A tp4
serve cannot reuse a tp2/cp2 CLI compile cache, so its first start pays a fresh
DiT compile:

```bash
TOKENIZERS_PARALLELISM=false difflet serve \
  --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --tp-degree 4 --cp-degree 1 \
  --height 480 --width 832 --num-frames 9 \
  --host 0.0.0.0 --port 8092 \
  --request-timeout 1800 --worker-restart-timeout 2400
```

Health and readiness (always unauthenticated; `/v1/*` requires
`Authorization: Bearer <key>` only when `--api-key`/`DIFFLET_API_KEY` is set):

```bash
curl -s http://127.0.0.1:8092/health     # {"status":"ok"} once the app is up
curl -s http://127.0.0.1:8092/ready      # {"status":"ready",...} once generation works
curl -s http://127.0.0.1:8092/v1/models
```

Request an image (OpenAI-style chat completions; the image comes back as a
base64 data URL):

```bash
curl -sS -X POST http://127.0.0.1:8092/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"black-forest-labs/FLUX.1-dev",
       "messages":[{"role":"user","content":"a small red sailboat on a calm blue lake"}],
       "extra_body":{"height":1024,"width":1024,"num_inference_steps":28,"guidance_scale":3.5,"seed":42}}' \
  | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > out.png
```

Request a video (synchronous multipart endpoint):

```bash
curl -fL -X POST http://127.0.0.1:8092/v1/videos/sync \
  -F 'model=Wan-AI/Wan2.2-T2V-A14B-Diffusers' \
  -F 'prompt=a cat walking through a garden' \
  -F 'size=832x480' -F 'num_frames=9' -F 'fps=16' \
  -F 'num_inference_steps=40' -F 'guidance_scale=4.0' \
  -o out.mp4
```

Serve logs go to the console and `./logs/` in the working directory.

## 5. Sharing the environment with teammates

Do not copy or commit `.venv` — it is ~10 GB of machine-specific binaries with
absolute paths baked in. Share the recipe instead: `requirements-neuron.lock`
plus `scripts/setup_env.sh` (or the `Dockerfile`) rebuilds an identical
environment anywhere. Because the compile-cache key includes the toolchain
versions, identical envs can also share `~/.cache/difflet` artifacts across
hosts (set `DIFFLET_COMPILE_CACHE`, and `DIFFLET_SHARED_WEIGHTS_DIR` on the
same filesystem, to use a common mount). After any deliberate toolchain change,
regenerate the lock:

```bash
.venv/bin/pip freeze --exclude-editable > requirements-neuron.lock
```
