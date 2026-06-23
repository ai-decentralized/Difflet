# Replace `--model` with `--model-id` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `--model` flag (short names) with `--model-id` (HuggingFace model IDs) across the entire CLI — no aliases, only HF IDs accepted.

**Architecture:** Four touch-points: `main.py` (parser + routing), `stage.py` (subprocess dispatcher), staged orchestrators' `_shared_cli_args` (subprocess arg forwarding), and two single-process orchestrators' error messages. Each change is independent except that `stage.py` must accept the same `--model-id` values that `main.py` now forwards.

**Tech Stack:** Python stdlib `argparse`; no new dependencies.

## Global Constraints

- Python ≥ 3.10; Black line-length 100
- Flag renamed: `--model` → `--model-id` (argparse `dest` becomes `model_id`)
- Short names (`flux`, `wan`, etc.) are removed entirely — no aliases
- Valid HF IDs (case-sensitive): `black-forest-labs/FLUX.1-dev`, `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, `hunyuanvideo-community/HunyuanVideo`, `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`, `Qwen/Qwen-Image`, `Lightricks/LTX-2`

---

## File Map

**Modified:**
- `difflet/cli/main.py` — `VALID_MODELS`, `_MODEL_TYPE`, flag name, `_get_orchestrator`, `main()`
- `difflet/cli/stage.py` — `_ORCHESTRATOR_MAP` keys, `--model-id` in stage parser
- `difflet/cli/orchestrators/wan.py` — `_shared_cli_args` passes `--model-id`
- `difflet/cli/orchestrators/hunyuan_video.py` — `_shared_cli_args` passes `--model-id`
- `difflet/cli/orchestrators/qwen_image.py` — `_shared_cli_args` passes `--model-id`
- `difflet/cli/orchestrators/flux.py` — error messages use `--model-id`
- `difflet/cli/orchestrators/ltx_2.py` — error messages use `--model-id`
- `tests/unit/test_cli_main.py` — `--model` → `--model-id`, short names → HF IDs
- `tests/unit/test_cli_runner.py` — `--model` → `--model-id` in cli_args assertion
- `tests/unit/test_cli_flux_orchestrator.py` — error message assertions updated
- `tests/unit/test_cli_ltx2_orchestrator.py` — error message assertion updated
- `docs/cli-staged-commands.md` — all `--model` → `--model-id` with HF IDs

---

## Task 1: `main.py` — rename flag, swap VALID_MODELS and routing to HF IDs

**Files:**
- Modify: `difflet/cli/main.py`
- Modify: `tests/unit/test_cli_main.py`

**Interfaces:**
- Produces: `VALID_MODELS: set[str]` — HF IDs
- Produces: `_MODEL_TYPE: dict[str, str]` — keyed by HF ID
- Produces: `main(argv)` — reads `args.model_id` (not `args.model`)
- Produces: `_get_orchestrator(args)` — keyed by HF ID

- [ ] **Step 1: Update `tests/unit/test_cli_main.py` to use `--model-id` and HF IDs**

Replace the entire file:

```python
from __future__ import annotations
import importlib
import pytest


def _cli() -> object:
    return importlib.import_module("difflet.cli.main")


def test_unknown_model_id_exits(capsys):
    cli = _cli()
    with pytest.raises(SystemExit) as exc:
        cli.main(["compile", "--model-id", "badmodel"])
    assert exc.value.code == 1
    assert "Unknown model-id 'badmodel'" in capsys.readouterr().err


def test_teacache_cadence_and_online_delta_mutually_exclusive(monkeypatch, capsys):
    cli = _cli()
    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: None)
    with pytest.raises(SystemExit) as exc:
        cli.main([
            "generate", "--model-id", "black-forest-labs/FLUX.1-dev",
            "--prompt", "x", "--output", "out.png",
            "--teacache-cadence", "2", "--teacache-online-delta", "0.6",
        ])
    assert exc.value.code == 1
    assert "mutually exclusive" in capsys.readouterr().err


def test_teacache_speedup_requires_calibration(monkeypatch, capsys):
    cli = _cli()
    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: None)
    with pytest.raises(SystemExit) as exc:
        cli.main([
            "generate", "--model-id", "black-forest-labs/FLUX.1-dev",
            "--prompt", "x", "--output", "out.png",
            "--teacache-speedup", "1.5",
        ])
    assert exc.value.code == 1
    assert "--teacache-calibration" in capsys.readouterr().err


def test_valid_generate_routes_to_orchestrator(monkeypatch):
    cli = _cli()
    calls = []
    class FakeOrch:
        def generate(self): calls.append("generate")
    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: FakeOrch())
    cli.main(["generate", "--model-id", "black-forest-labs/FLUX.1-dev",
              "--prompt", "x", "--output", "o.png"])
    assert calls == ["generate"]
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
PYTHONPATH=. pytest tests/unit/test_cli_main.py -v
```
Expected: 4 failures — `error: unrecognized arguments: --model-id` or `--model` still required.

- [ ] **Step 3: Update `difflet/cli/main.py`**

Replace the top of the file (through `_get_orchestrator`):

```python
from __future__ import annotations

import argparse
import sys

VALID_MODELS = {
    "black-forest-labs/FLUX.1-dev",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "hunyuanvideo-community/HunyuanVideo",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    "Qwen/Qwen-Image",
    "Lightricks/LTX-2",
}

_MODEL_TYPE: dict[str, str] = {
    "black-forest-labs/FLUX.1-dev": "flux",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "wan",
    "hunyuanvideo-community/HunyuanVideo": "hunyuan_video",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v": "hunyuan_video_15",
    "Qwen/Qwen-Image": "qwen_image",
    "Lightricks/LTX-2": "ltx_2",
}


def _add_model_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model-id",
        required=True,
        dest="model_id",
        help="HuggingFace model ID. One of:\n  " + "\n  ".join(sorted(VALID_MODELS)),
    )


def _add_parallel_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tp-degree", type=int, default=None,
                   help="Tensor-parallel degree (default: registry default)")
    p.add_argument("--cp-degree", type=int, default=1,
                   help="Context-parallel degree (default: 1)")


def _add_shape_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)


def _add_cache_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cache-dir", default=None,
                   help="Compiled artifact cache root (default: ~/.cache/difflet/)")
    p.add_argument("--force", action="store_true",
                   help="Recompile even if a valid cache entry exists")


def _add_generate_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", required=True, help="Output file path (.png or .mp4)")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work-dir", default=None,
                   help="Directory for inter-stage tensors (staged models only)")
    p.add_argument("--keep-work-dir", action="store_true",
                   help="Do not delete work-dir after successful generation")
    p.add_argument("--teacache-cadence", type=int, default=None,
                   metavar="N", help="Skip every N-th DiT step (fixed cadence, no calibration)")
    p.add_argument("--teacache-online-delta", type=float, default=None,
                   metavar="ALPHA", help="Online-delta TeaCache alpha (no calibration)")
    p.add_argument("--teacache-speedup", type=float, default=None,
                   metavar="X", help="Adaptive TeaCache target speedup (requires --teacache-calibration)")
    p.add_argument("--teacache-calibration", default=None,
                   metavar="PATH", help="Path to TeaCache calibration JSON")


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="difflet",
                                   description="Difflet — diffusion inference on Trainium")
    sub = root.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Download model weights from HuggingFace")
    _add_model_flag(dl)
    dl.add_argument("--revision", default=None)

    cp_cmd = sub.add_parser("compile", help="AOT-compile model NEFFs and cache on disk")
    _add_model_flag(cp_cmd)
    _add_parallel_flags(cp_cmd)
    _add_shape_flags(cp_cmd)
    _add_cache_flags(cp_cmd)

    gen = sub.add_parser("generate", help="Run inference (requires prior compile)")
    _add_model_flag(gen)
    _add_parallel_flags(gen)
    _add_shape_flags(gen)
    _add_cache_flags(gen)
    _add_generate_flags(gen)

    run_cmd = sub.add_parser("run", help="Download + compile + generate in one shot")
    _add_model_flag(run_cmd)
    _add_parallel_flags(run_cmd)
    _add_shape_flags(run_cmd)
    _add_cache_flags(run_cmd)
    _add_generate_flags(run_cmd)

    return root


def _validate_teacache(args: argparse.Namespace) -> None:
    cadence = getattr(args, "teacache_cadence", None)
    online = getattr(args, "teacache_online_delta", None)
    speedup = getattr(args, "teacache_speedup", None)
    calib = getattr(args, "teacache_calibration", None)

    active = [
        ("--teacache-cadence", cadence is not None),
        ("--teacache-online-delta", online is not None),
        ("--teacache-speedup", speedup is not None),
    ]
    active_names = [name for name, on in active if on]
    if len(active_names) > 1:
        print(f"Error: {active_names[0]} and {active_names[1]} are mutually exclusive.",
              file=sys.stderr)
        raise SystemExit(1)
    if speedup is not None and calib is None:
        print("Error: --teacache-speedup requires --teacache-calibration PATH.",
              file=sys.stderr)
        raise SystemExit(1)


def _get_orchestrator(args: argparse.Namespace):
    from difflet.cli.orchestrators.flux import FluxOrchestrator
    from difflet.cli.orchestrators.ltx_2 import LTX2Orchestrator
    from difflet.cli.orchestrators.wan import WanOrchestrator
    from difflet.cli.orchestrators.hunyuan_video import HunyuanVideoOrchestrator
    from difflet.cli.orchestrators.hunyuan_video_15 import HunyuanVideo15Orchestrator
    from difflet.cli.orchestrators.qwen_image import QwenImageOrchestrator

    mapping = {
        "black-forest-labs/FLUX.1-dev": FluxOrchestrator,
        "Lightricks/LTX-2": LTX2Orchestrator,
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers": WanOrchestrator,
        "hunyuanvideo-community/HunyuanVideo": HunyuanVideoOrchestrator,
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v": HunyuanVideo15Orchestrator,
        "Qwen/Qwen-Image": QwenImageOrchestrator,
    }
    return mapping[args.model_id](args)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.model_id not in VALID_MODELS:
        print(
            f"Error: Unknown model-id '{args.model_id}'. Valid model IDs:\n"
            + "\n".join(f"  {m}" for m in sorted(VALID_MODELS)),
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.command in ("generate", "run"):
        _validate_teacache(args)

    orchestrator = _get_orchestrator(args)
    getattr(orchestrator, args.command)()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
PYTHONPATH=. pytest tests/unit/test_cli_main.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add difflet/cli/main.py tests/unit/test_cli_main.py
git commit -m "feat: rename --model to --model-id, accept HuggingFace model IDs"
```

---

## Task 2: `stage.py` — re-key orchestrator map, rename flag in stage parser

**Files:**
- Modify: `difflet/cli/stage.py`

**Interfaces:**
- Consumes: `--model-id <hf-id>` forwarded by orchestrators' `_shared_cli_args`
- Produces: `_ORCHESTRATOR_MAP` keyed by HF ID

No new tests — stage.py is an internal subprocess entry point covered by the staged orchestrator integration tests in Task 3.

- [ ] **Step 1: Update `difflet/cli/stage.py`**

Replace the entire file:

```python
"""Internal subprocess dispatcher — not a user-facing entry point.

Called by runner.run_stage() as:
    python -m difflet.cli.stage --orchestrator <hf-id> --stage <stage> [forwarded args]
"""
from __future__ import annotations

import argparse
import importlib
import sys

_ORCHESTRATOR_MAP: dict[str, str] = {
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "difflet.cli.orchestrators.wan.WanOrchestrator",
    "hunyuanvideo-community/HunyuanVideo": (
        "difflet.cli.orchestrators.hunyuan_video.HunyuanVideoOrchestrator"
    ),
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v": (
        "difflet.cli.orchestrators.hunyuan_video_15.HunyuanVideo15Orchestrator"
    ),
    "Qwen/Qwen-Image": "difflet.cli.orchestrators.qwen_image.QwenImageOrchestrator",
}


def _load_orchestrator_class(name: str):
    cls_path = _ORCHESTRATOR_MAP[name]
    module_path, cls_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _build_stage_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="difflet.cli.stage", add_help=False)
    p.add_argument("--orchestrator", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--stage-mode", default="generate", choices=["compile", "generate"])
    p.add_argument("--model-id", dest="model_id", default=None)
    p.add_argument("--tp-degree", type=int, default=None)
    p.add_argument("--cp-degree", type=int, default=1)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--teacache-cadence", type=int, default=None)
    p.add_argument("--teacache-online-delta", type=float, default=None)
    p.add_argument("--teacache-speedup", type=float, default=None)
    p.add_argument("--teacache-calibration", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_stage_parser()
    args, _ = parser.parse_known_args(argv)
    cls = _load_orchestrator_class(args.orchestrator)
    orchestrator = cls(args)
    orchestrator._run_stage_internal(args.stage, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Commit**

```bash
git add difflet/cli/stage.py
git commit -m "feat: re-key stage orchestrator map to HF IDs, rename --model to --model-id"
```

---

## Task 3: Staged orchestrators — forward `--model-id` in `_shared_cli_args`

**Files:**
- Modify: `difflet/cli/orchestrators/wan.py`
- Modify: `difflet/cli/orchestrators/hunyuan_video.py`
- Modify: `difflet/cli/orchestrators/qwen_image.py`
- Modify: `tests/unit/test_cli_runner.py`
- Modify: `tests/unit/test_cli_staged_orchestrators.py`

**Interfaces:**
- Each orchestrator's `_shared_cli_args` now produces `["--model-id", "<hf-id>", ...]`
- `stage.py` (Task 2) reads `--model-id` and `--orchestrator` independently; orchestrator key matches `_HF_MODEL_ID`

- [ ] **Step 1: Update `tests/unit/test_cli_runner.py`**

In `test_builds_correct_command`, change the `cli_args` and the assertion:

```python
def test_builds_correct_command(monkeypatch):
    captured = {}
    monkeypatch.setattr("subprocess.run", lambda cmd, env, check: captured.update({"cmd": cmd}))
    from difflet.cli.runner import run_stage
    run_stage("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "transformer", num_cores=4,
               virtual_core_size=None,
               cli_args=["--model-id", "Wan-AI/Wan2.2-T2V-A14B-Diffusers", "--tp-degree", "4"])
    cmd = captured["cmd"]
    assert cmd[0] == sys.executable
    assert "-m" in cmd
    assert "difflet.cli.stage" in cmd
    assert "--orchestrator" in cmd
    assert "Wan-AI/Wan2.2-T2V-A14B-Diffusers" in cmd
    assert "--stage" in cmd
    assert "transformer" in cmd
    assert "--model-id" in cmd
```

- [ ] **Step 2: Run the runner test — verify it fails**

```bash
PYTHONPATH=. pytest tests/unit/test_cli_runner.py::test_builds_correct_command -v
```
Expected: FAIL — `--model-id` not in cmd (old code still passes `--model`).

- [ ] **Step 3: Update `_shared_cli_args` in `difflet/cli/orchestrators/wan.py`**

In `_shared_cli_args`, replace `"--model", "wan"` with `"--model-id", _HF_MODEL_ID`:

```python
    def _shared_cli_args(self, stage_mode: str, work_dir: str | None = None) -> list[str]:
        a = self.args
        parts = [
            "--model-id", _HF_MODEL_ID,
            "--tp-degree", str(a.tp_degree or 4),
            "--cp-degree", str(a.cp_degree or 1),
            "--height", str(a.height or 480),
            "--width", str(a.width or 832),
            "--num-frames", str(a.num_frames or 9),
            "--steps", str(a.steps or 2),
            "--guidance-scale", str(a.guidance_scale or 1.0),
            "--seed", str(a.seed),
            "--stage-mode", stage_mode,
        ]
        if a.prompt:
            parts += ["--prompt", a.prompt]
        if a.output:
            parts += ["--output", a.output]
        if a.cache_dir:
            parts += ["--cache-dir", a.cache_dir]
        if work_dir:
            parts += ["--work-dir", work_dir]
        return parts
```

- [ ] **Step 4: Update `_shared_cli_args` in `difflet/cli/orchestrators/hunyuan_video.py`**

In `_shared_cli_args`, replace `"--model", "hunyuan-video"` with `"--model-id", _HF_MODEL_ID`:

```python
    def _shared_cli_args(self, stage_mode: str, work_dir: str | None = None) -> list[str]:
        a = self.args
        parts = [
            "--model-id", _HF_MODEL_ID,
            "--tp-degree", str(a.tp_degree or 4),
            "--cp-degree", str(a.cp_degree or 1),
            "--height", str(a.height or 320),
            "--width", str(a.width or 512),
            "--num-frames", str(a.num_frames or 61),
            "--steps", str(a.steps or 4),
            "--guidance-scale", str(a.guidance_scale or 6.0),
            "--seed", str(a.seed),
            "--stage-mode", stage_mode,
        ]
        if a.prompt:
            parts += ["--prompt", a.prompt]
        if a.output:
            parts += ["--output", a.output]
        if a.cache_dir:
            parts += ["--cache-dir", a.cache_dir]
        if work_dir:
            parts += ["--work-dir", work_dir]
        return parts
```

- [ ] **Step 5: Update `_shared_cli_args` in `difflet/cli/orchestrators/qwen_image.py`**

In `_shared_cli_args`, replace `"--model", "qwen-image"` with `"--model-id", _HF_MODEL_ID`:

```python
    def _shared_cli_args(self, stage_mode: str, work_dir: str | None = None) -> list[str]:
        a = self.args
        parts = [
            "--model-id", _HF_MODEL_ID,
            "--tp-degree", str(a.tp_degree or 4),
            "--cp-degree", str(a.cp_degree or 1),
            "--height", str(a.height or 1024),
            "--width", str(a.width or 1024),
            "--steps", str(a.steps or 4),
            "--guidance-scale", str(a.guidance_scale or 4.0),
            "--seed", str(a.seed),
            "--stage-mode", stage_mode,
        ]
        if a.prompt:
            parts += ["--prompt", a.prompt]
        if a.output:
            parts += ["--output", a.output]
        if a.cache_dir:
            parts += ["--cache-dir", a.cache_dir]
        if work_dir:
            parts += ["--work-dir", work_dir]
        return parts
```

- [ ] **Step 6: Update `tests/unit/test_cli_staged_orchestrators.py` — `model=` in namespace helpers**

In `_wan_args`, `_hv_args`, and `_qwen_args`, rename `model=` to `model_id=` and use the HF ID:

```python
def _wan_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers", tp_degree=4, cp_degree=1,
        height=480, width=832, num_frames=9,
        cache_dir=None, force=False, revision=None,
        prompt="a cat walking", output="/tmp/wan.mp4",
        steps=2, guidance_scale=1.0, seed=42,
        work_dir=None, keep_work_dir=False,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _hv_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="hunyuanvideo-community/HunyuanVideo", tp_degree=4, cp_degree=1,
        height=320, width=512, num_frames=61,
        cache_dir=None, force=False, revision=None,
        prompt="a cat", output="/tmp/hv.mp4",
        steps=4, guidance_scale=6.0, seed=42,
        work_dir=None, keep_work_dir=False,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _qwen_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="Qwen/Qwen-Image", tp_degree=4, cp_degree=1,
        height=1024, width=1024, num_frames=None,
        cache_dir=None, force=False, revision=None,
        prompt="a cat", output="/tmp/qwen.png",
        steps=4, guidance_scale=4.0, seed=42,
        work_dir=None, keep_work_dir=False,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)
```

- [ ] **Step 7: Run all updated tests**

```bash
PYTHONPATH=. pytest tests/unit/test_cli_runner.py tests/unit/test_cli_staged_orchestrators.py -v
```
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add difflet/cli/orchestrators/wan.py \
        difflet/cli/orchestrators/hunyuan_video.py \
        difflet/cli/orchestrators/qwen_image.py \
        tests/unit/test_cli_runner.py \
        tests/unit/test_cli_staged_orchestrators.py
git commit -m "feat: forward --model-id in staged orchestrator subprocess args"
```

---

## Task 4: Single-process orchestrator error messages

**Files:**
- Modify: `difflet/cli/orchestrators/flux.py`
- Modify: `difflet/cli/orchestrators/ltx_2.py`
- Modify: `tests/unit/test_cli_flux_orchestrator.py`
- Modify: `tests/unit/test_cli_ltx2_orchestrator.py`

**Interfaces:**
- Error messages now read: `difflet download --model-id black-forest-labs/FLUX.1-dev`

- [ ] **Step 1: Update `tests/unit/test_cli_flux_orchestrator.py`**

Change the two error message assertions:

```python
def test_generate_exits_when_weights_missing(monkeypatch, capsys):
    from difflet.cli.orchestrators.flux import FluxOrchestrator

    def fake_resolve(model_id, *, revision=None, local_files_only=False, allow_patterns=None):
        if local_files_only:
            raise OSError("not cached")
        return "/fake/path"

    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path", fake_resolve)

    with pytest.raises(SystemExit) as exc:
        FluxOrchestrator(_flux_args()).generate()
    assert exc.value.code == 1
    assert "difflet download --model-id black-forest-labs/FLUX.1-dev" in capsys.readouterr().err


def test_generate_exits_when_no_compiled_cache(monkeypatch, capsys):
    from difflet.cli.orchestrators.flux import FluxOrchestrator

    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *a, **kw: "/fake/path",
    )
    monkeypatch.setattr(
        "difflet.pipeline.compile_cache.has_valid_manifest",
        lambda *a, **kw: False,
    )

    with pytest.raises(SystemExit) as exc:
        FluxOrchestrator(_flux_args()).generate()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "difflet compile --model-id black-forest-labs/FLUX.1-dev" in err
```

Also update `_flux_args` to use `model_id` instead of `model`:

```python
def _flux_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="black-forest-labs/FLUX.1-dev", tp_degree=4, cp_degree=1,
        height=1024, width=1024, num_frames=None,
        cache_dir=None, force=False, revision=None,
        prompt="a cat", output="/tmp/out.png",
        steps=28, guidance_scale=3.5, seed=42,
        work_dir=None, keep_work_dir=False,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)
```

- [ ] **Step 2: Update `tests/unit/test_cli_ltx2_orchestrator.py`**

Change the error message assertion and `_ltx2_args`:

```python
def _ltx2_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="Lightricks/LTX-2", tp_degree=4, cp_degree=1,
        height=512, width=768, num_frames=121,
        cache_dir=None, force=False, revision=None,
        prompt="a cat", output="/tmp/out.mp4",
        steps=40, guidance_scale=3.5, seed=42,
        work_dir=None, keep_work_dir=False,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_generate_exits_when_no_compiled_cache(monkeypatch, capsys):
    from difflet.cli.orchestrators.ltx_2 import LTX2Orchestrator

    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *a, **kw: "/fake/path",
    )
    monkeypatch.setattr(
        "difflet.pipeline.compile_cache.has_valid_manifest",
        lambda *a, **kw: False,
    )
    with pytest.raises(SystemExit) as exc:
        LTX2Orchestrator(_ltx2_args()).generate()
    assert exc.value.code == 1
    assert "difflet compile --model-id Lightricks/LTX-2" in capsys.readouterr().err
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
PYTHONPATH=. pytest tests/unit/test_cli_flux_orchestrator.py tests/unit/test_cli_ltx2_orchestrator.py -v
```
Expected: FAIL on the error message assertions.

- [ ] **Step 4: Update error messages in `difflet/cli/orchestrators/flux.py`**

Replace the two `print(...)` error strings in `generate()`:

```python
        except OSError:
            print(
                f"Error: model weights not found.\n"
                f"Run: difflet download --model-id {_HF_MODEL_ID}",
                file=sys.stderr,
            )
            raise SystemExit(1)
```

```python
        if not has_valid_manifest(compiled, spec):
            print(
                f"Error: no compiled artifacts found for {_HF_MODEL_ID} at {compiled}.\n"
                f"Run: difflet compile --model-id {_HF_MODEL_ID} --tp-degree {parallel.tp_degree}",
                file=sys.stderr,
            )
            raise SystemExit(1)
```

- [ ] **Step 5: Update error messages in `difflet/cli/orchestrators/ltx_2.py`**

Same pattern as flux.py — replace both `print(...)` error strings in `generate()`:

```python
        except OSError:
            print(
                f"Error: model weights not found.\n"
                f"Run: difflet download --model-id {_HF_MODEL_ID}",
                file=sys.stderr,
            )
            raise SystemExit(1)
```

```python
        if not has_valid_manifest(compiled, spec):
            print(
                f"Error: no compiled artifacts found for {_HF_MODEL_ID} at {compiled}.\n"
                f"Run: difflet compile --model-id {_HF_MODEL_ID} --tp-degree {parallel.tp_degree}",
                file=sys.stderr,
            )
            raise SystemExit(1)
```

- [ ] **Step 6: Run tests — verify they pass**

```bash
PYTHONPATH=. pytest tests/unit/test_cli_flux_orchestrator.py tests/unit/test_cli_ltx2_orchestrator.py -v
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add difflet/cli/orchestrators/flux.py \
        difflet/cli/orchestrators/ltx_2.py \
        tests/unit/test_cli_flux_orchestrator.py \
        tests/unit/test_cli_ltx2_orchestrator.py
git commit -m "feat: update orchestrator error messages to use --model-id with HF IDs"
```

---

## Task 5: Update docs

**Files:**
- Modify: `docs/cli-staged-commands.md`

No tests — docs only.

- [ ] **Step 1: Replace all `--model` with `--model-id <hf-id>` in `docs/cli-staged-commands.md`**

Apply these substitutions throughout the file:

| Old | New |
|---|---|
| `--model flux` | `--model-id black-forest-labs/FLUX.1-dev` |
| `--model ltx-2` | `--model-id Lightricks/LTX-2` |
| `--model wan` | `--model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers` |
| `--model hunyuan-video` | `--model-id hunyuanvideo-community/HunyuanVideo` |
| `--model hunyuan-video-1.5` | `--model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` |
| `--model qwen-image` | `--model-id Qwen/Qwen-Image` |

- [ ] **Step 2: Commit**

```bash
git add docs/cli-staged-commands.md
git commit -m "docs: update staged commands reference to use --model-id with HF IDs"
```

---

## Final verification

- [ ] **Run the full CLI unit test suite**

```bash
PYTHONPATH=. pytest tests/unit/test_cli_main.py \
                    tests/unit/test_cli_runner.py \
                    tests/unit/test_cli_base.py \
                    tests/unit/test_cli_flux_orchestrator.py \
                    tests/unit/test_cli_ltx2_orchestrator.py \
                    tests/unit/test_cli_staged_orchestrators.py -v
```
Expected: all pass.

- [ ] **Smoke-test the help output**

```bash
difflet --help
difflet compile --help
```
Expected: `--model-id` appears, valid HF IDs listed in help text.
