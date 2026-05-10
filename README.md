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
| M2 — Wan 2.2 T2V (spike) | Done | Prompt → UMT5 → DiT (TP=4) → VAE → `(1, 3, 9, 480, 832)` video tensor at 480×832×9. Sequential 4-core split via `scripts/wan_smoke.sh`. See `cclogs/m2-wan/17-M2-wan-spike-retro.md`. |
| M2.5 — Wan numerical alignment | Done (component) | UMT5 / DiT / VAE NEFF-vs-CPU all PASS (cosine ≥ 0.995). Full denoise trajectory parity vs HF diffusers still open. See `cclogs/m2-wan/{18,19,20}`. |
| Phase B — backend abstraction | Done | `nova/core/` and 4 Trainium-only `nova/utils/*` files relocated under `nova/backends/trainium/`. Models import only via `nova.ops`. See `cclogs/backend/23-phase-b-closure.md`. |
| M3 — HunyuanVideo + 1.5 | Planned | First model written entirely against the frozen `nova.ops` v1 surface. |
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

## Quick start — Wan 2.2

Run the M2 spike smoke at 480×832×9 frames. The 4-core `trn3pd98.3xlarge`
cannot fit TP=4 transformer + TP=1 VAE in one process, so the smoke script
splits into two sequential stages (text + DiT, then VAE decode):

```bash
./scripts/wan_smoke.sh
```

Defaults: prompt `a cat walking`, 480×832, 9 frames, TP=4 transformer,
TP=1 VAE, 2 inference steps. Stage 1 writes a latent tensor to
`.nova-cache/wan_smoke_latents.pt`; stage 2 reads it back and decodes.
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
`--tp-degree 4`, `--skip-warmup`. The `from_pretrained` regression vs M1
closure is documented in `cclogs/backend/23-phase-b-closure.md` §2; the
remaining ~65 s is Neuron sharding + traced weight init and is the next
optimization target (pre-sharded component cache).

### Wan 2.2 T2V — 480×832, 9 frames, bf16, cache hit, 2 inference steps

| Stage | Time |
|---|---:|
| Stage 1 load (text encoder + DiT, TP=4) | 19.4 s |
| Stage 1 forward (UMT5 + 2 denoise steps) | 3.7 s |
| Stage 2 load (VAE decoder, TP=1) | 15.3 s |
| Stage 2 forward (single decode) | 1.0 s |
| Total wall clock (`wan_smoke.sh`) | ~53 s |

Spike baseline. See `cclogs/m2-wan/21-M2.5D-current-wan-smoke-baseline.md`
for the full run record (latent / video sha256, tensor stats); Phase B
reproduced these numbers bit-identically.

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
              │                      │   download patterns, allowed backends
              └──────────┬───────────┘
                         │ lazy factory
                         ▼
              ┌──────────────────────┐
              │   <Model>Application │   composes sub-apps for text
              │   (Flux: 4 sub-apps) │   encoders / DiT / VAE
              │   (Wan:  up to 4   ) │   compile() + load() + __call__
              └──────────┬───────────┘
                         │ model code uses ONLY:
                         ▼
              ┌──────────────────────┐
              │       nova.ops       │   backend-neutral op surface
              │ attention, RMSNorm,  │   (linear, attention, norm,
              │ Column/Row/Embedding │   collectives, embeddings,
              │ collectives, plat-   │   platform). Frozen v1; additive
              │ form helpers         │   only. Dispatch frozen at first
              │                      │   import per process.
              └──────────┬───────────┘
                         │ NOVA_BACKEND ∈ {trainium, cpu, …}
                         ▼
              ┌──────────────────────────────────────┐
              │            nova/backends/            │
              │ ┌─────────────┐  ┌─────────────────┐ │
              │ │ trainium/   │  │ cpu/            │ │
              │ │  ops_impl/  │  │  ops_impl/      │ │
              │ │  core/      │  │  runtime.py     │ │
              │ │  utils/     │  │ (numerical ref) │ │
              │ │  runtime.py │  └─────────────────┘ │
              │ └─────────────┘  ┌─────────────────┐ │
              │                  │ cuda/ rocm/     │ │
              │                  │  (stubs)        │ │
              │                  └─────────────────┘ │
              └──────────────────────────────────────┘
```

`NovaPipeline` only invokes three methods on an application:
`compile()`, `load()`, `__call__()`. Model code imports only from
`nova.ops` (no direct `neuronx_distributed` / `nkilib` / `torch_neuronx`);
backend-specific implementations live entirely under
`nova/backends/<hw>/ops_impl/`. CUDA / ROCm backends remain stubs.

### Repository layout

```
nova/
├── pipeline/             # Public API — Nova-authored, formatted with black
│   ├── nova_pipeline.py  # NovaPipeline.from_pretrained / precompile / __call__
│   ├── parallel_config.py
│   ├── compile_cache.py  # cache_key + manifest schema
│   └── path_resolver.py  # local path / HF snapshot_download with allow_patterns
├── registry.py           # @register_model decorator + ModelEntry
├── ops/                  # Backend-neutral op surface (frozen v1)
│   ├── attention.py / linear.py / norm.py / collectives.py
│   ├── embeddings.py / platform.py
│   └── _dispatch.py      # lazy backend selection (frozen on first import)
├── backends/             # Per-hardware implementations
│   ├── base.py / registry.py
│   ├── trainium/         # NXD + nkilib + torch_neuronx
│   │   ├── core/         # AOT inference base classes (application_base,
│   │   │                 # config, modules/{attention,custom_calls,...})
│   │   ├── utils/        # compile_env, runtime_env, distributed, snapshot
│   │   ├── ops_impl/     # Trainium impls of nova.ops
│   │   └── wan/          # Trainium-side Wan wrappers
│   ├── cpu/              # Pure-torch numerical reference
│   │   └── ops_impl/
│   ├── cuda/  rocm/      # Stubs
│   └── runtime.py        # backend probe + selection
├── core/                 # Compatibility shims → nova/backends/trainium/core/
├── utils/                # Hardware-neutral utilities (HF / diffusers adapters);
│                         # the four Trainium-flavored files here are shims
├── layers/               # Diffusion-specific layers; import only via nova.ops
├── models/
│   ├── flux/             # Flux application + pipeline + DiT + CLIP + T5 + VAE
│   └── wan/              # Wan application + orchestrator + DiT + UMT5 + VAE
│       ├── modeling_wan.py / pipeline.py / application.py
│       ├── checkpoint/   # HF → Nova state-dict conversion
│       ├── umt5/  vae/   # encoder + decoder modeling
│       └── entry.py      # registered factory
```

`nova/backends/trainium/{core,modules}` follows upstream coding style so
periodic syncs with the upstream Neuron infrastructure produce clean
diffs. `nova/{pipeline, ops, registry.py, models/<new>}` are Nova-authored
and formatted with Black.

The compatibility shims at `nova/core/` and the four Trainium-named files
in `nova/utils/` re-export from `nova/backends/trainium/`. They exist
only to give external callers a migration window and will be removed at
M3 closure (see `cclogs/backend/23-phase-b-closure.md` §4).

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
   the encoder / backbone / decoder sub-applications, `pipeline.py` (or a
   thin orchestrator like `nova/models/wan/pipeline.py`), plus
   `modeling_<name>.py` for the DiT backbone.
2. **`nova/models/<name>/entry.py`** — a factory
   `create_<name>_application(model_path, parallel, dtype, shape, **kwargs)`.
3. **`nova/registry.py`** — a `@register_model` entry pointing to the
   factory by string (lazy import) plus default parallel config, shape,
   HF download patterns, and supported `backends=("trainium", ...)`.

Hard rule for new modeling code (enforced grep-statically and by the
`forbidden_prefixes` guard in `tests/unit/test_modeling_*.py`):

- model files import **only** from `nova.ops`, `torch`, stdlib,
  `diffusers`, and `transformers`;
- no direct `neuronx_distributed`, `torch_neuronx`, `nkilib`, `nova.core`
  (shim), or old `nova.utils.{compile_env,runtime_env,distributed,snapshot}`
  imports;
- if a primitive is missing, add it to `nova.ops` first (with at least
  the Trainium implementation under `nova/backends/trainium/ops_impl/`,
  ideally also a CPU reference under `nova/backends/cpu/ops_impl/`).

`NovaPipeline` itself does not change. The pattern that Flux and Wan
establish (multiple sub-applications, race-safe compile with `model.pt`
markers and SPMD barriers, ordered load with biggest-TP component first)
should be reused. Lifting this into a shared
`MultiComponentApplication` base class is a deferred Phase B/M3 cleanup
(retro proposal in `cclogs/m2-wan/17-M2-wan-spike-retro.md` §2.2).

## Development

Project-local helper scripts:

```bash
./scripts/check_quick.sh                      # imports + 76 unit tests
./scripts/test_unit.sh                        # pytest tests/unit -q
./scripts/test_imports.sh                     # smoke import nova + key submodules
./scripts/flux_smoke.sh                       # 1-step Flux smoke (load + 1 forward)
./scripts/flux_baseline_28.sh                 # 28-step Flux baseline gate

# Wan 2.2 spike + alignment
./scripts/wan_smoke.sh                        # M2 spike e2e (text + DiT + VAE, 4-core sequential)
./scripts/wan_e2e_smoke.sh                    # single-process Wan e2e (component-level debug)
./scripts/wan_e2e_sequential_smoke.sh         # explicit two-stage split, no env-flag magic
./scripts/wan_backbone_compile_smoke.sh       # AOT compile DiT only
./scripts/wan_text_encoder_compile_smoke.sh   # AOT compile UMT5 only
./scripts/wan_vae_compile_smoke.sh            # AOT compile VAE only (~78 min @ 480×832×9)

# M2.5 numerical alignment gates (per cclogs/m2-wan/{18,19,20})
./scripts/wan_m25_numerical_alignment.sh      # M2.5-A: VAE NEFF vs CPU
./scripts/wan_m25b_text_dit_alignment.sh      # M2.5-B: UMT5 / DiT real-weight CPU parity
./scripts/wan_m25c_neff_cpu_alignment.sh      # M2.5-C: UMT5 / DiT NEFF vs CPU
./scripts/wan_vae_real_alignment.sh           # VAE real-weight CPU + NEFF parity (used by M2.5-A)

# Wan utilities
./scripts/wan_convert_checkpoint.sh           # HF → Nova state-dict conversion CLI
```

All scripts auto-set the Neuron venv on `PATH`, project on `PYTHONPATH`,
and `NEURON_RT_NUM_CORES`. M2.5 scripts use a 115 GB peak-RSS gate
(`NOVA_M25{B,C}_PEAK_RSS_MAX_GB`) to avoid OOM on the 4-core spike host.

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
