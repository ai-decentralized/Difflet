# Difflet

A focused inference engine for diffusion transformers (DiTs) on AWS Trainium.

Difflet provides a single Python entry point — `DiffletPipeline` — that handles model
download, ahead-of-time compilation, on-disk artifact caching, and SPMD
multi-core execution for image and video diffusion models on Trainium v3.

## Status

| Milestone | State | Notes |
|---|---|---|
| M0 — repository foundation | Done | Core inference primitives, layer library, registry, compile cache |
| M1 — Flux end-to-end | Done | `FLUX.1-dev` at 1024² in 28 steps, cache-hit baseline below |
| M2 — Wan 2.2 T2V (spike) | Done | Prompt → UMT5 → DiT (TP=4) → VAE → `(1, 3, 9, 480, 832)` video tensor at 480×832×9. Sequential 4-core split via `scripts/wan_smoke.sh`. |
| M2.5 — Wan numerical alignment | Done (component) | UMT5 / DiT / VAE NEFF-vs-CPU all PASS (cosine ≥ 0.995). Full denoise trajectory parity vs HF diffusers still open. |
| Phase B — backend abstraction | Done | `difflet/core/` and 4 Trainium-only `difflet/utils/*` files relocated under `difflet/backends/trainium/`; compatibility shims removed during M3. Models import only via `difflet.ops`. |
| M3 — HunyuanVideo | Done | T2V at `320x512x61`, TP=4. Now **fully on-device** — Llama 3 + CLIP text encoders, the DiT, and the 16-segment NEFF VAE decoder all run on Trainium (`examples/hunyuan_video_example.py`). 4-step DiT trajectory cosine min `0.999896` vs HF. The original M3 v0 hybrid path (HF CPU text/VAE) is still available via `scripts/hunyuan_smoke.sh`. |
| M3.x — VAE on Trainium | Done | 16-segment NEFF decoder bypasses a `neuronx-cc` `GroupNorm+SiLU → causal-Conv3D` same-graph lowering bug (`cclogs/m3-hunyuan/38`). Full tiled parity cosine `1.0022` vs HF (`≥ 0.999` gate); decode 91.4 s vs HF CPU 149.7 s (~1.6×). |
| M4a — Qwen-Image | Done | Text-to-image **fully on-device** — Qwen2.5-VL + DiT + VAE on Trainium (`examples/qwen_image_example.py`); the VAE reuses Difflet's Wan VAE decoder port. |
| M6 — LTX-2 | Done | Dual-stream segmented DiT runtime, decoded end-to-end. |
| M6a — HunyuanVideo 1.5 | Done | Registered; segmented DiT + VAE runtime. |
| Context parallelism (Wan) | Done | `cp_degree` sequence parallelism for the Wan DiT (gather-KV self-attention); `world_size = tp_degree × cp_degree`. |
| Planned | — | 720p / longer-frame / I2V, TP refactor, NKI masked attention, Z-Image. |

## Hardware and software prerequisites

- **Instance**: AWS Trainium v3 (validated on `trn3pd98.3xlarge`, 4 NeuronCores,
  144 GB device memory). Other trn3 shapes should work; the tensor-parallel
  degree must match the number of visible NeuronCores.
- **Runtime image**: a Neuron PyTorch 2.9 environment with `neuronx-cc`,
  `neuronx-distributed`, `nki`, `nkilib`, `torch-neuronx`, and `libneuronxla`.
  The current pinned versions live in `pyproject.toml`.
- **Python**: 3.10+.

The reference development image bundles all of the above at
`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`.

## Install

```bash
git clone git@github.com:ai-decentralized/Difflet.git
cd Difflet
pip install -e .
```

For development tooling (`pytest`, `black`, `ruff`):

```bash
pip install -e ".[dev]"
```

Hugging Face authentication is required for gated checkpoints such as
`black-forest-labs/FLUX.1-dev`:

```bash
huggingface-cli login
```

## Quick start — Flux

Run a 28-step 1024×1024 generation on 4 NeuronCores:

```bash
NEURON_RT_NUM_CORES=4 python examples/flux_example.py \
    --model black-forest-labs/FLUX.1-dev \
    --tp-degree 4 \
    --num-inference-steps 28 \
    --prompt "a photorealistic cat sitting in a sunlit garden" \
    --output out.png
```

The first run triggers ahead-of-time compilation (~10–15 minutes for Flux on
4 cores). Compiled artifacts land in `~/.cache/difflet/flux/<key>/`; subsequent
runs hit the cache and skip directly to load + denoise.

Equivalent helper script:

```bash
./scripts/flux_baseline_28.sh
```

## Quick start — Wan 2.2

Run the M2 spike smoke at 480×832×9 frames. The 4-core `trn3pd98.3xlarge`
cannot fit TP=4 transformer + TP=1 VAE in one process, so the smoke script
splits into two sequential stages (text + DiT, then VAE decode):

```bash
./scripts/wan_smoke.sh
```

Defaults: prompt `a cat walking`, 480×832, 9 frames, TP=4 transformer,
TP=1 VAE, 2 inference steps. Stage 1 writes a latent tensor to
`.difflet-cache/wan_smoke_latents.pt`; stage 2 reads it back and decodes.
The final `(1, 3, 9, 480, 832)` bf16 video tensor is saved to
`/tmp/wan_smoke.pt` (MP4 export is best-effort and requires
`imageio-ffmpeg`).

Single-process Wan CLI (mirrors `examples/flux_example.py` and works on
larger Trainium instances with enough cores):

```bash
NEURON_RT_NUM_CORES=4 python examples/wan_example.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --tp-degree 4 --skip-warmup \
    --num-frames 9 --height 480 --width 832 \
    --num-inference-steps 2 \
    --prompt "a cat walking" \
    --output /tmp/wan_smoke.mp4
```

Use `--download-weights` on the first run to fetch transformer / text
encoder / tokenizer / VAE shards from HF.

## Quick start — HunyuanVideo (on-device)

Fully on-device text-to-video: Llama 3 + CLIP text encoders, the DiT, and the
VAE decoder all run on Trainium. Capacity forces staging — the 8B Llama encoder
and the 13B DiT cannot co-fit on one 4-core card — so each stage is its own
process and passes tensors through `--work-dir` files. Every NEFF compiles on
first run and is cached: CLIP/Llama in their stages, and the DiT + VAE decoder in
the `generate` stage (the 13B DiT compile is ~30 min the first time).

First download the weights (~42 GB, public model):

```bash
huggingface-cli download hunyuanvideo-community/HunyuanVideo
```

Then run the three stages (each resolves the local HF snapshot automatically):

```bash
# Stage 1 — CLIP pooled projections (1 core)
NEURON_RT_NUM_CORES=1 NEURON_RT_VIRTUAL_CORE_SIZE=2 \
    python examples/hunyuan_video_example.py --stage clip \
        --prompt "a cat walking in a sunlit garden"

# Stage 2 — Llama 3 prompt embeddings (TP=4)
NEURON_RT_NUM_CORES=4 NEURON_RT_VIRTUAL_CORE_SIZE=2 \
    python examples/hunyuan_video_example.py --stage llama \
        --prompt "a cat walking in a sunlit garden"

# Stage 3 — DiT denoise + on-device VAE decode → video (TP=4)
#   first run compiles the DiT + VAE NEFF into .difflet-cache/hunyuan_ondevice/
NEURON_RT_NUM_CORES=4 NEURON_RT_VIRTUAL_CORE_SIZE=2 \
    python examples/hunyuan_video_example.py --stage generate \
        --num-inference-steps 4 --output /tmp/hunyuan.mp4
```

Pass `--cpu-vae` to stage 3 to decode the VAE on the HF CPU reference instead
(the earlier M3 v0 hybrid path; `scripts/hunyuan_smoke.sh` also drives it). The
generated video is saved as a `.pt` tensor (the `--output` `.mp4` suffix is a
label; MP4 encoding is not yet wired). Validated end-to-end on
`trn3pd98.3xlarge`: a `(1, 3, 61, 320, 512)` video tensor in 4 steps.

## Quick start — Qwen-Image (on-device)

Fully on-device text-to-image: the Qwen2.5-VL text encoder, the DiT, and the VAE
all run on Trainium (the VAE reuses Difflet's Wan VAE decoder port — the Qwen-Image
VAE config is identical to Wan's). Staged like HunyuanVideo; every NEFF compiles
on first run (encoder in `text`, DiT in `generate`, VAE in `vae`).

First download the weights:

```bash
huggingface-cli download Qwen/Qwen-Image
```

Then:

```bash
# Stage 1 — Qwen2.5-VL prompt embeddings (TP=4)
NEURON_RT_NUM_CORES=4 NEURON_RT_VIRTUAL_CORE_SIZE=2 \
    python examples/qwen_image_example.py --stage text \
        --prompt "a small red cabin beside a lake, crisp morning light"

# Stage 2 — DiT denoise → packed latents (TP=4); first run compiles the DiT NEFF
NEURON_RT_NUM_CORES=4 NEURON_RT_VIRTUAL_CORE_SIZE=2 \
    python examples/qwen_image_example.py --stage generate --num-inference-steps 4

# Stage 3 — VAE decode → image (1 core)
NEURON_RT_NUM_CORES=1 NEURON_RT_VIRTUAL_CORE_SIZE=2 \
    python examples/qwen_image_example.py --stage vae --output /tmp/qwen.png
```

### Library API

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

## Performance baselines

### Flux.1-dev — 1024×1024, 28 steps, bf16, cache hit

| Stage | M1 closure | After Phase B (current) | Gate |
|---|---:|---:|---:|
| `from_pretrained` (load + shard + NRT init) | 36.7 s | 64.6 s | not gated |
| 28-step forward | 7.8 s | 7.8 s | ≤ 8.5 s |
| Denoise throughput (steady state) | 3.78 it/s | 3.78 it/s | not gated |
| Compile cache size on disk | 113 MB | 113 MB | n/a |
| First-time AOT compile (cold) | ~683 s | ~683 s | n/a |

Measured on `trn3pd98.3xlarge` with `NEURON_RT_NUM_CORES=4`,
`--tp-degree 4`, `--skip-warmup`. The remaining ~65 s after the
Phase B fix is Neuron sharding + traced weight init and is the next
optimization target (pre-sharded component cache).

### Wan 2.2 T2V — 480×832, 9 frames, bf16, cache hit, 2 inference steps

| Stage | Time |
|---|---:|
| Stage 1 load (text encoder + DiT, TP=4) | 19.4 s |
| Stage 1 forward (UMT5 + 2 denoise steps) | 3.7 s |
| Stage 2 load (VAE decoder, TP=1) | 15.3 s |
| Stage 2 forward (single decode) | 1.0 s |
| Total wall clock (`wan_smoke.sh`) | ~53 s |

Spike baseline; Phase B reproduced these numbers bit-identically.

## Runtime protocol — important constraints

Difflet diffusion artifacts have hard constraints that differ from typical
LLM serving setups. Violating them produces silent SIGSEGVs or
`global communicator` errors.

1. **Use one Python process with multiple visible NeuronCores.**
   Set `NEURON_RT_NUM_CORES=N` and run `python examples/flux_example.py`.
   Do **not** launch with `torchrun --nproc_per_node=N` for the current
   diffusion path — `torchrun`'s MPMD model assigns one core per process,
   which is incompatible with the way these traced artifacts initialize
   the runtime communicator.

2. **All components in a pipeline must share the same `world_size`.**
   For Flux on 4 cores: T5 and the transformer run as `TP=4, DP=1`; CLIP
   and the VAE decoder run as `TP=1, DP=4`. All four components have
   `world_size=4`. Mixing `world_size=1` and `world_size=4` artifacts in
   one process will crash during weight initialization.

3. **Component load order matters.** The first component loaded fixes the
   process-wide NeuronCore communicator. Tensor-parallel components must
   load before replicated ones. Flux load order:
   `text_encoder_2 → transformer → text_encoder → decoder`.

These rules are enforced in `difflet/models/flux/application.py`; new model
ports should follow the same pattern.

## Architecture

```
User script
   │
   ▼
DiffletPipeline             single public entry; resolves registry, manages
   │  from_pretrained,   compile cache, dispatches to <Model>Application
   │  precompile, __call__
   ▼
ModelEntry (registry.py) per-model metadata: factory, default parallel +
   │                     shape, download patterns, allowed backends
   ▼
<Model>Application       composes encoder / DiT / VAE sub-apps; exposes
   │                     compile() + load() + __call__()
   ▼
difflet.ops                 backend-neutral op surface (attention / linear /
   │                     norm / collectives / embeddings / platform).
   │                     Frozen v1; additive only; dispatch frozen at
   │                     first import per process.
   ▼
difflet/backends/<hw>/ops_impl/   per-hardware implementations.
                               trainium = the real backend; cpu = pure
                               torch numerical reference; cuda/rocm stubs.
```

`DiffletPipeline` only invokes `compile() / load() / __call__()` on an
application. Model code imports only from `difflet.ops` (no direct
`neuronx_distributed` / `nkilib` / `torch_neuronx`); backend-specific
implementations live entirely under `difflet/backends/<hw>/ops_impl/`.

### Repository layout

```
difflet/
├── pipeline/        Difflet-authored public API (DiffletPipeline, compile cache,
│                    parallel config, HF path resolver)
├── registry.py      @register_model + ModelEntry
├── ops/             Backend-neutral op surface (frozen v1)
├── backends/
│   ├── trainium/    Real backend (NXD + nkilib + torch_neuronx)
│   │   ├── core/    AOT base classes, attention, custom_calls
│   │   ├── utils/   compile_env, runtime_env, distributed, snapshot
│   │   ├── ops_impl/    Trainium impls of difflet.ops
│   │   ├── nki_kernels/    NKI custom kernels (MX microscaling, etc.)
│   │   └── flux/ wan/ hunyuan_video/ qwen_image/ ltx_2/    per-model wrappers
│   ├── cpu/         Pure-torch numerical reference
│   └── cuda/ rocm/  Stubs
├── utils/           Hardware-neutral utilities (HF / diffusers adapters)
├── layers/          Diffusion-specific layers; import only via difflet.ops
└── models/          flux/ wan/ hunyuan_video/ qwen_image/ ltx_2/ — each:
                     modeling, pipeline, application, entry, checkpoint
                     (where needed). HunyuanVideo 1.5 is served from
                     hunyuan_video/ (model_version="1.5").
```

`difflet/backends/trainium/{core,modules}` follows upstream Neuron coding
style so periodic rebases are clean; `difflet/{pipeline, ops, registry.py,
models/<new>}` is Difflet-authored and formatted with Black.

## Compile cache

Difflet maintains a content-addressed cache of AOT-compiled artifacts.

```
~/.cache/difflet/<model>/<sha256-prefix>/
├── manifest.json
├── text_encoder/   model.pt + neuron_config.json
├── text_encoder_2/ model.pt + neuron_config.json
├── transformer/    model.pt + neuron_config.json
└── decoder/        model.pt + neuron_config.json
```

The cache key hashes:
- model id, registry name, revision
- parallel configuration (`tp_degree`, `cp_degree`, `cfg_parallel_enabled`)
- dtype (normalized — `"bf16"`, `"bfloat16"`, `torch.bfloat16` collapse to one key)
- shape (`height`, `width`, `num_frames`)
- toolchain versions (Python major.minor, torch, neuronx-cc, neuronx-distributed,
  nki, libneuronxla, torch-neuronx, torch-xla, diffusers, transformers)

`model_path` and Python patch version are recorded in the manifest for
debugging but excluded from the key, so caches are portable across hosts
and survive Python patch upgrades.

Override the cache root with the `DIFFLET_COMPILE_CACHE` environment
variable or `compile_cache_dir=` in `from_pretrained`. Pass
`force_compile=True` to bypass a valid cache hit.

## Parallel modes

`DiffletParallelConfig` exposes three parallelism axes:

- `tp_degree` — tensor parallel degree (must divide visible NeuronCore count).
- `cp_degree` — context parallel degree (1 = disabled); `world_size` becomes
  `tp_degree * cp_degree`.
- `cfg_parallel_enabled` — splits the CFG conditional/unconditional batch;
  doubles `world_size` to `tp_degree * 2`. Mutually exclusive with `cp_degree > 1`.

```python
DiffletParallelConfig(tp_degree=4)                          # world_size=4
DiffletParallelConfig(tp_degree=4, cfg_parallel_enabled=1)  # world_size=8
DiffletParallelConfig(tp_degree=4, cp_degree=2)             # world_size=8
DiffletParallelConfig(tp_degree=4, cp_degree=4)             # world_size=16
```

## Adding a new model

To port a diffusion model, add three things:

1. **`difflet/models/<name>/`** — implementation: `application.py` composing
   the encoder / backbone / decoder sub-applications, `pipeline.py` (or a
   thin orchestrator like `difflet/models/wan/pipeline.py`), plus
   `modeling_<name>.py` for the DiT backbone.
2. **`difflet/models/<name>/entry.py`** — a factory
   `create_<name>_application(model_path, parallel, dtype, shape, **kwargs)`.
3. **`difflet/registry.py`** — a `@register_model` entry pointing to the
   factory by string (lazy import) plus default parallel config, shape,
   HF download patterns, and supported `backends=("trainium", ...)`.

Hard rule for new modeling code (enforced by the repo-wide import guard in
`scripts/test_imports.sh`):

- model files import **only** from `difflet.ops`, `torch`, stdlib,
  `diffusers`, and `transformers`;
- no direct `neuronx_distributed`, `torch_neuronx`, `nkilib`, `difflet.core`,
  or old `difflet.utils.{compile_env,runtime_env,distributed,snapshot}`
  imports;
- if a primitive is missing, add it to `difflet.ops` first (with at least
  the Trainium implementation under `difflet/backends/trainium/ops_impl/`,
  ideally also a CPU reference under `difflet/backends/cpu/ops_impl/`).

`DiffletPipeline` itself does not change. Multi-component models extend the shared
`MultiComponentApplication` base (`difflet/backends/trainium/core/`), which provides
race-safe compile with `model.pt` markers and SPMD barriers, and ordered load
with the biggest-TP component first. All current model applications (Flux, Wan,
HunyuanVideo, Qwen-Image, LTX-2) build on it.

## Development

Project-local helper scripts:

```bash
./scripts/check_quick.sh                      # imports + unit tests
./scripts/test_unit.sh                        # pytest tests/unit -q
./scripts/test_imports.sh                     # smoke import difflet + key submodules
./scripts/flux_smoke.sh                       # 1-step Flux smoke (load + 1 forward)
./scripts/flux_baseline_28.sh                 # 28-step Flux baseline gate

# Wan 2.2 spike + alignment
./scripts/wan_smoke.sh                        # M2 spike e2e (text + DiT + VAE, 4-core sequential)
./scripts/wan_e2e_smoke.sh                    # single-process Wan e2e (component-level debug)
./scripts/wan_e2e_sequential_smoke.sh         # explicit two-stage split, no env-flag magic
./scripts/wan_backbone_compile_smoke.sh       # AOT compile DiT only
./scripts/wan_text_encoder_compile_smoke.sh   # AOT compile UMT5 only
./scripts/wan_vae_compile_smoke.sh            # AOT compile VAE only (~78 min @ 480×832×9)

# M2.5 numerical alignment gates
./scripts/wan_m25_numerical_alignment.sh      # M2.5-A: VAE NEFF vs CPU
./scripts/wan_m25b_text_dit_alignment.sh      # M2.5-B: UMT5 / DiT real-weight CPU parity
./scripts/wan_m25c_neff_cpu_alignment.sh      # M2.5-C: UMT5 / DiT NEFF vs CPU
./scripts/wan_vae_real_alignment.sh           # VAE real-weight CPU + NEFF parity (used by M2.5-A)

# Wan utilities
./scripts/wan_convert_checkpoint.sh           # HF → Difflet state-dict conversion CLI
```

All scripts auto-set the Neuron venv on `PATH`, project on `PYTHONPATH`,
and `NEURON_RT_NUM_CORES`. M2.5 scripts use a 115 GB peak-RSS gate
(`DIFFLET_M25{B,C}_PEAK_RSS_MAX_GB`) to avoid OOM on the 4-core spike host.

Run unit tests directly:

```bash
PYTHONPATH=. pytest tests/unit -q
```

Format Difflet-authored code:

```bash
black difflet/pipeline difflet/registry.py examples tests
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

Difflet incorporates code derived from third-party Apache-2.0 projects;
attributions and modification banners are in [`NOTICE`](NOTICE) and at
the top of each derived file.
