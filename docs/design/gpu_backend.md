# GPU backends: platforms, model specs and the generic GPU flow

How Difflet runs on GPUs (CUDA today, ROCm later) without a per-model copy of the application,
serving and CLI code, and without backend-name checks spread through the codebase.

## Layout

```
difflet/
├── platforms/                  # one Platform per backend
│   ├── base.py                 #   PlatformCapabilities, Platform
│   ├── gpu.py                  #   the GPU family shared by CUDA and ROCm
│   └── trainium.py  tpu.py  cuda.py  rocm.py  cpu.py
├── ops/                        # op names; each backend implements them in ops_impl/
├── models/<model>/
│   ├── modeling_<model>.py     # difflet.ops only; no backend names, no environment variables
│   └── spec.py                 # the model's ModelSpec (torch-free)
└── backends/
    ├── <backend>/ops_impl/     # per-backend op implementations
    ├── gpu/                    # the generic GPU flow, written once
    │   ├── application.py      #   GpuComponent (one transformer), GpuApplication (the pipeline contract)
    │   ├── cli.py              #   the one-process CLI orchestrator
    │   ├── serving.py          #   the resident serving adapter
    │   └── policy.py           #   which platform, which requests are refused (torch-free)
    ├── cuda/<model>/           # only what differs per model on CUDA
    └── trainium/ …             # unchanged
```

## Rules

1. **Ask the platform, not the backend name.** Code outside `difflet/platforms` and
   `difflet/backends` reads `current_platform().capabilities` or calls a platform method. The
   remaining backend-name checks outside those packages are the Trainium/TPU dispatch in the
   model entry points and two TPU serving checks, all older than the GPU flow.
2. **Model code does not know the backend.** Options reach a model through constructor
   arguments or configuration, not environment variables.
3. **Three levels of ops.** Base ops live in `difflet.ops`. A fusion shared by two or more
   models joins `difflet.ops` once both use it. A fusion for one model on one backend is a
   load-time module swap under `backends/<backend>/<model>/` and never edits a shared model
   file.
4. **One generic flow.** Loading, compile, the `DiffletPipeline` contract, serving and the CLI
   are written once in `backends/gpu/`. A model supplies its spec and its real differences.
5. **Evidence before defaults.** An optimisation is on by default only with model-level
   measurements; a test that claims a fast path checks that it ran.

## Platforms

`PlatformCapabilities` keeps the three original fields (`requires_aot`,
`single_process_multi_core`, `supports_torchrun_mpmd`) and adds:

| Field | Meaning |
|---|---|
| `flow` | which application, CLI and serving path the platform runs: `neuron`, `tpu`, `gpu`, `cpu` |
| `parallel_modes` | the parallel modes the platform's collectives and attention implement |
| `supports_torch_compile` | `--torch-compile` is an option |
| `device_unit` | what one device is called in messages |

`Platform` methods cover what callers used to branch on: `detect()`, `prepare_runtime()`,
`default_tp_degree()`, `toolchain_versions()` (compile-cache key), `layer_markers()`, and for
GPU platforms `device_count()`, `device_memory_bytes()` and `visible_devices()` /
`worker_env()` (pinning a worker to its GPUs). `difflet.backends.base` and
`difflet.backends.registry` keep their names as aliases.

Parallel support is checked in one place: `platform.unsupported_parallel_modes(parallel)`, plus
CFG-parallel for a model without a conditional/unconditional pair (`policy.unsupported_parallel`).
The application, the CLI and serving all call it and only format the message differently.

## Model specs

`ModelSpec` (`difflet/models/spec.py`) is data the CLI, serving and the application read:
checkpoints the CLI routes and serving accepts, default shape and steps, the guidance kind
(`none`, `distilled`, `cfg`) and defaults, refused CLI flags, serving options, video fields and
application kwargs, prompt budgets per tokenizer, the output kind, and the application class.
It imports no torch, because the serving parent process reads it. Fields another model will need
(audio, encoder residency, several prompt budgets) are added with that model. Spatial multiples
and a frame stride are declarative fields: original HunyuanVideo requires H/W multiples of 16
and `4n+1` frames. One torch-free validator applies these rules before application loading,
CLI download and serving worker startup; other models keep their existing defaults.

## The generic GPU flow

- **`GpuComponent`**: build on meta → materialise → `load_weights` → device →
  `requires_grad_(False)` → `eval` → load-time swaps → per-block `torch.compile` of
  `repeated_blocks` (dynamic when the model says so). `compile()` writes a manifest; there is no
  ahead-of-time step.
- **`GpuApplication`**: the reflective `DiffletPipeline` contract (`compile`, `load`,
  `has_compiled_artifacts`, `__call__`), refusal of unsupported kwargs and parallel modes, shape
  and dtype handling, one artifact directory per transformer for models with several experts.
  A model's subclass implements `_configure`, `_build_transformers`, `load_companions`,
  `generate` and `dit_input_contract`.
- **`GpuOrchestrator`** and the **serving adapter** read the spec for defaults, refusals,
  guidance rules, prompt budgets and output encoding; they share `gpu_pipeline_kwargs`, so a
  served model loads exactly as `difflet run` loads it.

`ModelEntry.create_application` sends every model on a GPU platform to
`create_gpu_application`; `difflet run` and `difflet serve` choose the generic orchestrator and
serving adapter from the platform's `flow`.

## What a model writes

1. `difflet/models/<model>/spec.py` and its line in `models/spec.py`'s table.
2. `backends/cuda/<model>/transformer.py`: a `GpuComponent` subclass with `build_module`,
   `load_weights`, `get_example_inputs` and `repeated_blocks`.
3. `backends/cuda/<model>/application.py`: the `GpuApplication` subclass wiring its encoders,
   decoder and denoise driver.
4. Tests for its own differences; the generic flow is tested once.

## What a GPU platform writes

`platforms/<name>.py` (a `GpuPlatform` subclass: detection, runtime checks, toolchain versions,
the visible-devices variable) and an `ops_impl` package. Torch on ROCm exposes its devices under
the `cuda` device type, so ROCm's `ops_impl` can re-export CUDA's and the generic flow runs
unchanged.

## Parallelism

The model files already express TP, SP and CP through `difflet.ops`, and the resident serving
worker already runs one process per device when `supports_torchrun_mpmd` is set. Adding a mode
on CUDA is implementing its collectives (or ring/Ulysses attention) in `backends/cuda/ops_impl/`
and adding it to the platform's `parallel_modes`. On cards without NVLink, CFG-parallel and
context parallelism come before tensor parallelism, which all-reduces every layer over PCIe.

## Design choices

- **Eager torch**, with per-block `torch.compile` as an opt-in: multi-shape compilation would
  cost recompiles for no measured win.
- **SDPA for attention**, no `flash-attn` dependency; inputs are lifted to 4-D so SDPA can pick
  a fused kernel.
- **Every component on the device in one process** (encoders, DiT, VAE): a 96 GB card removes
  the reason other backends stage them.
- **bf16 only**; MX ops raise rather than silently changing precision.
- **The oracle is upstream diffusers**, never another Difflet backend.
