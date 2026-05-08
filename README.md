# Nova

A focused inference engine for diffusion transformers (DiTs) on AWS Trainium.

Nova provides a single Python entry point — `NovaPipeline` — that handles model
download, ahead-of-time compilation, on-disk artifact caching, and SPMD
multi-core execution for image and video diffusion models on Trainium v3.

## Status

| Milestone | State | Notes |
|---|---|---|
| M0 — repository foundation | Done | Core inference primitives, layer library, registry, compile cache |
| M1 — Flux end-to-end | Done | `FLUX.1-dev` at 1024² in 28 steps, cache-hit baseline below |
| M2 — Wan 2.2 T2V/I2V | Planned | Architecture spike next |
| M3 — HunyuanVideo + 1.5 | Planned | |
| M4 — Qwen-Image, LTX-2, Z-Image | Planned | |

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
git clone git@github.com:binkma-v/Nova.git
cd Nova
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
4 cores). Compiled artifacts land in `~/.cache/nova/flux/<key>/`; subsequent
runs hit the cache and skip directly to load + denoise.

Equivalent helper script:

```bash
./scripts/flux_baseline_28.sh
```

### Library API

```python
from nova import NovaPipeline, NovaParallelConfig

pipe = NovaPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    parallel=NovaParallelConfig(tp_degree=4),
    height=1024,
    width=1024,
)

image = pipe(
    prompt="a photorealistic cat sitting in a sunlit garden",
    num_inference_steps=28,
).images[0]
image.save("out.png")
```

## Performance baseline

Flux.1-dev, 1024×1024, 28 steps, bf16, single instance, cache hit:

| Stage | Time |
|---|---:|
| `from_pretrained` (load + weight shard + NRT init) | 36.7 s |
| 28-step forward | 7.8 s |
| Denoise throughput (steady state) | ~3.86 it/s |
| Compile cache size on disk | 113 MB |
| First-time AOT compile (cold) | ~683 s |

Measured on `trn3pd98.3xlarge` with `NEURON_RT_NUM_CORES=4`,
`--tp-degree 4`, `--skip-warmup`. See `cclogs/06-M1-closure.md` for the
detailed run.

## Runtime protocol — important constraints

Nova diffusion artifacts have hard constraints that differ from typical
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

These rules are enforced in `nova/models/flux/application.py`; new model
ports should follow the same pattern.

## Architecture

```
                     User script
                          │
                          ▼
              ┌──────────────────────┐
              │     NovaPipeline     │   single public entry point
              │  from_pretrained,    │   resolves registry, manages
              │  precompile, __call__│   cache, dispatches to app
              └──────────┬───────────┘
                         │ resolve(model_id) via registry
                         ▼
              ┌──────────────────────┐
              │      ModelEntry      │   per-model metadata: factory,
              │   (registry.py)      │   default parallel + shape,
              │                      │   download patterns
              └──────────┬───────────┘
                         │ lazy factory
                         ▼
              ┌──────────────────────┐
              │   <Model>Application │   composes Neuron sub-apps for
              │   (e.g. Flux: 4 sub- │   text encoders / DiT / VAE
              │    apps composed)    │   compile() + load() + __call__
              └──────────┬───────────┘
                         │ per component
                         ▼
              ┌──────────────────────┐
              │ NeuronApplicationBase│   AOT compile to NEFF, SPMD
              │  (core/)             │   load, weight sharding via
              │                      │   neuronx-distributed parallel
              │                      │   layers and nkilib kernels
              └──────────────────────┘
```

`NovaPipeline` only invokes three methods on an application:
`compile()`, `load()`, `__call__()`. Internal Neuron-side abstractions
can evolve without breaking the public API.

### Repository layout

```
nova/
├── pipeline/             # Public API — Nova-authored, formatted with black
│   ├── nova_pipeline.py  # NovaPipeline.from_pretrained / precompile / __call__
│   ├── parallel_config.py
│   ├── compile_cache.py  # cache_key + manifest schema
│   └── path_resolver.py  # local path / HF snapshot_download with allow_patterns
├── registry.py           # @register_model decorator + ModelEntry
├── core/                 # Inference base classes (application_base, config,
│   │                     # model_wrapper, modules/{attention,custom_calls,...})
│   └── modules/
├── layers/               # Diffusion-specific layers (embeddings, normalization,
│                         # activations, padder)
├── models/
│   └── flux/             # Flux application + pipeline + DiT + CLIP + T5 + VAE
└── utils/                # HF / diffusers adapters, distributed helpers,
                          # runtime + compile env setup
```

`nova/{core, layers, utils, models/<existing>}` follow upstream coding
style (Black/isort skipped in `pyproject.toml`) so that periodic syncs
with the upstream Neuron infrastructure produce clean diffs.
`nova/{pipeline, registry.py, models/<new>}` are Nova-authored and
formatted with Black.

## Compile cache

Nova maintains a content-addressed cache of AOT-compiled artifacts.

```
~/.cache/nova/<model>/<sha256-prefix>/
├── manifest.json
├── text_encoder/   model.pt + neuron_config.json
├── text_encoder_2/ model.pt + neuron_config.json
├── transformer/    model.pt + neuron_config.json
└── decoder/        model.pt + neuron_config.json
```

The cache key hashes:
- model id, registry name, revision
- parallel configuration (`tp_degree`, `cp_enabled`, `cfg_parallel_enabled`)
- dtype (normalized — `"bf16"`, `"bfloat16"`, `torch.bfloat16` collapse to one key)
- shape (`height`, `width`, `num_frames`)
- toolchain versions (Python major.minor, torch, neuronx-cc, neuronx-distributed,
  nki, libneuronxla, torch-neuronx, torch-xla, diffusers, transformers)

`model_path` and Python patch version are recorded in the manifest for
debugging but excluded from the key, so caches are portable across hosts
and survive Python patch upgrades.

Override the cache root with the `NOVA_COMPILE_CACHE` environment
variable or `compile_cache_dir=` in `from_pretrained`. Pass
`force_compile=True` to bypass a valid cache hit.

## Parallel modes

`NovaParallelConfig` exposes three parallelism axes:

- `tp_degree` — tensor parallel degree (must divide visible NeuronCore count).
- `cp_enabled` — context parallel; doubles `world_size` to `tp_degree * 2`.
- `cfg_parallel_enabled` — splits the CFG conditional/unconditional batch;
  doubles `world_size` to `tp_degree * 2`. Mutually exclusive with `cp_enabled`.

```python
NovaParallelConfig(tp_degree=4)                          # world_size=4
NovaParallelConfig(tp_degree=4, cfg_parallel_enabled=1)  # world_size=8
NovaParallelConfig(tp_degree=4, cp_enabled=True)         # world_size=8
```

## Adding a new model

To port a diffusion model, add three things:

1. **`nova/models/<name>/`** — implementation: `application.py` composing
   the encoder / backbone / decoder sub-applications, `pipeline.py`
   subclassing the corresponding `diffusers` pipeline, plus
   `modeling_<name>.py` for the DiT backbone.
2. **`nova/models/<name>/entry.py`** — a factory
   `create_<name>_application(model_path, parallel, dtype, shape, **kwargs)`.
3. **`nova/registry.py`** — a `@register_model` entry pointing to the
   factory by string (lazy import) plus default parallel config, shape,
   and HF download patterns.

`NovaPipeline` itself does not change. The pattern that Flux establishes
(four sub-applications, race-safe compile with `model.pt` markers and
SPMD barriers, ordered load) generalizes to multi-component diffusion
pipelines and should be reused. A `MultiComponentApplication` base class
is planned to factor this out during M2 (Wan).

## Development

Project-local helper scripts:

```bash
./scripts/check_quick.sh       # imports + unit tests
./scripts/test_unit.sh         # pytest tests/unit -q
./scripts/test_imports.sh      # smoke import nova + key submodules
./scripts/flux_smoke.sh        # 1-step Flux smoke (verifies load + 1 forward)
./scripts/flux_baseline_28.sh  # 28-step Flux baseline
```

All scripts auto-set the Neuron venv on `PATH`, project on `PYTHONPATH`,
and `NEURON_RT_NUM_CORES=4`.

Run unit tests directly:

```bash
PYTHONPATH=. pytest tests/unit -q
```

Format Nova-authored code:

```bash
black nova/pipeline nova/registry.py examples tests
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

Nova incorporates code derived from third-party Apache-2.0 projects;
attributions and modification banners are in [`NOTICE`](NOTICE) and at
the top of each derived file.
