# Difflet CLI Design Spec

**Date:** 2026-06-23
**Status:** Approved

---

## Goal

Replace the current workflow of running per-model example scripts with a unified `difflet` CLI that handles download, compilation, and inference for all supported models. Staged models (HunyuanVideo, Qwen-Image, Wan) require sequential subprocess launches with different `NEURON_RT_NUM_CORES` values; the CLI orchestrates this transparently.

---

## Commands

Four top-level commands, all flag-based (`--model` selects the model):

```
difflet download --model <name> [--revision REV]

difflet compile  --model <name>
                 [--tp-degree N] [--cp-degree N]
                 [--height H] [--width W] [--num-frames F]
                 [--cache-dir PATH] [--force]

difflet generate --model <name>
                 --prompt "..."
                 --output PATH
                 [--tp-degree N] [--cp-degree N]
                 [--height H] [--width W] [--num-frames F]
                 [--steps N] [--guidance-scale F] [--seed N]
                 [--cache-dir PATH] [--work-dir PATH] [--keep-work-dir]
                 [--teacache-cadence N |
                  --teacache-online-delta ALPHA |
                  --teacache-speedup X --teacache-calibration PATH]

difflet run      --model <name>
                 --prompt "..."
                 --output PATH
                 (all flags from compile + generate combined)
```

**Model names:** `flux`, `wan`, `hunyuan-video`, `qwen-image`, `ltx-2`

**`run` semantics:** download (skip if weights exist) → compile (skip if cache hit) → generate. Chains the three steps; each step's skip logic reuses the existing `DiffletPipeline` and HF hub cache checks.

**`generate` with no compiled cache** exits immediately:
```
Error: no compiled artifacts found for flux at ~/.cache/difflet/flux/<key>/.
Run: difflet compile --model flux --tp-degree 4
```

---

## File Layout

```
difflet/cli/
├── __init__.py            # exports main()
├── main.py                # top-level argparse, validates flags, routes to commands
├── stage.py               # internal subprocess dispatcher (~15 lines)
├── runner.py              # env-var builder + subprocess.run() wrapper
└── orchestrators/
    ├── __init__.py
    ├── base.py            # ModelOrchestrator ABC: download / compile / generate / run
    ├── flux.py            # single-process via DiffletPipeline
    ├── wan.py             # 2 subprocess stages: text+transformer / vae
    ├── hunyuan_video.py   # 3 subprocess stages: clip / llama / generate
    ├── qwen_image.py      # 3 subprocess stages: text / generate / vae
    └── ltx_2.py           # single-process via DiffletPipeline
```

**Entry point** added to `pyproject.toml`:
```toml
[project.scripts]
difflet = "difflet.cli:main"
```

---

## Architecture

### Base class (`orchestrators/base.py`)

```python
class ModelOrchestrator(ABC):
    def __init__(self, args: argparse.Namespace) -> None: ...
    @abstractmethod
    def download(self) -> None: ...
    @abstractmethod
    def compile(self) -> None: ...
    @abstractmethod
    def generate(self) -> None: ...
    def run(self) -> None:
        self.download()
        self.compile()
        self.generate()
```

### Single-process models (Flux, LTX-2)

Orchestrators call `DiffletPipeline.precompile()` and `DiffletPipeline.from_pretrained()` directly in-process. No subprocesses.

### Staged models (HunyuanVideo, Qwen-Image, Wan)

`compile` for staged models is itself a multi-stage subprocess sequence — each stage's NEFF is compiled in its own subprocess with the correct core counts. For example, `difflet compile --model hunyuan-video` spawns three subprocesses in order: CLIP compile (1 core), Llama compile (tp×cp cores), DiT+VAE compile (tp×cp cores). Each subprocess runs the stage in compile-only mode and exits; no inference is performed. The `generate` command then assumes all stage NEFFs are cached.

Each stage runs as a separate subprocess because `NEURON_RT_NUM_CORES` must be fixed before any Neuron runtime initializes. The orchestrator calls `runner.run_stage()` once per stage with the correct env vars.

**HunyuanVideo stages:**

| Stage | `NEURON_RT_NUM_CORES` | `NEURON_RT_VIRTUAL_CORE_SIZE` | Produces |
|---|---|---|---|
| `clip` | 1 | 2 | `{work_dir}/clip.pt` |
| `llama` | `tp_degree × cp_degree` | 2 | `{work_dir}/llama.pt` |
| `generate` | `tp_degree × cp_degree` | 2 | output video tensor |

**Qwen-Image stages:**

| Stage | `NEURON_RT_NUM_CORES` | `NEURON_RT_VIRTUAL_CORE_SIZE` | Produces |
|---|---|---|---|
| `text` | `tp_degree × cp_degree` | 2 | `{work_dir}/text.pt` |
| `generate` | `tp_degree × cp_degree` | 2 | `{work_dir}/latents.pt` |
| `vae` | 1 | 2 | output image |

**Wan stages:**

| Stage | `NEURON_RT_NUM_CORES` | `NEURON_RT_VIRTUAL_CORE_SIZE` | Produces |
|---|---|---|---|
| `transformer` | `tp_degree × cp_degree` | — | `{work_dir}/latents.pt` |
| `vae` | 1 | — | output video |

### `runner.py`

Single public function:

```python
def run_stage(
    orchestrator: str,
    stage: str,
    num_cores: int,
    virtual_core_size: int | None,
    cli_args: list[str],
) -> None:
    env = os.environ.copy()
    env.setdefault("NEURON_RT_NUM_CORES", str(num_cores))
    if virtual_core_size is not None:
        env.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", str(virtual_core_size))
    cmd = [sys.executable, "-m", "difflet.cli.stage",
           "--orchestrator", orchestrator, "--stage", stage, *cli_args]
    subprocess.run(cmd, env=env, check=True)
```

`env.setdefault` is used so a user-set `NEURON_RT_NUM_CORES` in the shell always wins.

### `stage.py`

Thin dispatcher (~15 lines). Parses `--orchestrator` and `--stage`, imports the right orchestrator class, calls `_run_stage_internal(stage, args)`. This is an internal entry point; users never call it directly.

### Stage logic migration

Per-stage functions (`stage_clip`, `stage_llama`, `stage_generate`, etc.) move from `examples/` into each orchestrator's `_run_stage_internal()`. The example scripts become thin wrappers that instantiate the orchestrator and call the same method — so they continue to work standalone without duplicating logic.

---

## Env Var Handling

| Env var | Who sets it | Value |
|---|---|---|
| `NEURON_RT_NUM_CORES` | `runner.py` (via `setdefault`) | `tp_degree × cp_degree` for full stages; `1` for single-core stages |
| `NEURON_RT_VIRTUAL_CORE_SIZE` | `runner.py` (via `setdefault`) | Model-specific constant baked into each orchestrator (`2` for HunyuanVideo and Qwen-Image; unset for others) |
| `DIFFLET_BACKEND` | Orchestrator before subprocess | `"trainium"` |

User shell values always take precedence (`setdefault` never overwrites).

If `--tp-degree` is omitted, the CLI falls back to the registry default for that model (same as `DiffletPipeline.from_pretrained()` today).

---

## Work-dir Lifecycle

- Created by the orchestrator before the first stage subprocess.
- Default path: `~/.cache/difflet/work/<model>/` (configurable via `--work-dir`).
- Passed as `--work-dir` into each stage subprocess so they can read/write intermediate tensors.
- Deleted automatically after a successful `generate` or `run`.
- **Preserved on failure** so users can inspect intermediate tensors.
- `--keep-work-dir` flag suppresses cleanup even on success.

---

## TeaCache Flags

Three mutually exclusive modes, validated in `main.py` before any subprocess:

| Flag(s) | Mode | Calibration needed |
|---|---|---|
| `--teacache-cadence N` | Fixed cadence — skip every N-th step | No |
| `--teacache-online-delta ALPHA` | Online delta — skip flat steps using measured output delta | No |
| `--teacache-speedup X --teacache-calibration PATH` | Adaptive probe — target speedup multiplier with per-model calibration | Yes |

`--teacache-speedup` without `--teacache-calibration` exits with:
```
Error: --teacache-speedup requires --teacache-calibration PATH.
```

TeaCache flags are forwarded only to the DiT `generate` stage. Text encoder and VAE stages never receive them.

For single-process models, flags map directly to `DiffletPipeline.from_pretrained()` kwargs:
- `teacache_speedup`, `teacache_calibration_path` — existing kwargs
- `cadence` and `online_delta_alpha` — passed via `application_kwargs`

---

## Error Handling

| Situation | Behavior |
|---|---|
| `generate` with no compiled cache | Exit: `"No compiled artifacts found for {model} at {path}.\nRun: difflet compile --model {model} --tp-degree {N}"` |
| `generate` but weights missing | Exit: `"Model weights not found.\nRun: difflet download --model {model}"` |
| Stage subprocess exits non-zero | Re-raise: `"Stage '{stage}' failed (exit code {N}). Work dir preserved at {path} for inspection."` |
| Conflicting `--teacache-*` flags | Exit early: `"--teacache-cadence and --teacache-online-delta are mutually exclusive."` |
| Unknown `--model` value | Exit: `"Unknown model '{name}'. Valid models: flux, wan, hunyuan-video, qwen-image, ltx-2."` |

---

## What Does Not Change

- `DiffletPipeline` public API — untouched.
- `examples/` scripts — become thin wrappers into orchestrators; still work standalone.
- Registry, compile cache, `difflet.ops` — untouched.
- No new model ports — this spec covers CLI scaffolding only.
