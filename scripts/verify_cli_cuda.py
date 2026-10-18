"""The CUDA column of the CLI matrix (CUDA plan F3).

``scripts/verify_cli.py`` sizes every cell to the four NeuronCores of a trn2.3xlarge. On one
GPU the matrix has three cells — Qwen-Image (F3), FLUX.1-dev (F5) and Wan 2.2 T2V-A14B (F1)
at ``tp=1`` — grown by F2 (a second card). Per cell: ``difflet download`` →
``difflet compile`` → TIMED ``difflet generate``, each a subprocess with
``DIFFLET_BACKEND=cuda``, logs and ``results.json`` under a fresh run dir. GPU-only (the
unit tests stub the subprocesses).

Each cell generates at its model's default step count (``CELL_STEPS``: Qwen-Image 50,
FLUX.1-dev 28, Wan 2.2 40); ``--steps N`` overrides every cell. A generate that exits 0 but
leaves no output at the cell's path (``.png``, or ``.mp4`` for Wan — ``CELL_OUTPUT_SUFFIX``)
fails the cell.

    python scripts/verify_cli_cuda.py --seed 0 [--steps N] [--run-root ~/.cache/difflet/verify-cli-cuda]

Exit code 0 only when every cell passes.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from verify_cli import _DIFFLET_CMD, Status, StepResult, _kill_process_group, allocate_run_dir  # noqa: E402

CELLS: list[tuple[str, str]] = [
    ("Qwen/Qwen-Image", "tp1"),
    ("black-forest-labs/FLUX.1-dev", "tp1"),
    ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "tp1"),
]
# Per-model generate steps when --steps is not given: diffusers' pipeline defaults (Wan's 40 is
# WanPipeline's; the registry's serving default_steps=2 is a Trainium smoke value).
CELL_STEPS: dict[str, int] = {
    "Qwen/Qwen-Image": 50,
    "black-forest-labs/FLUX.1-dev": 28,
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": 40,
}
# The generate output's suffix per model (default ".png"): a video model writes an MP4.
CELL_OUTPUT_SUFFIX: dict[str, str] = {"Wan-AI/Wan2.2-T2V-A14B-Diffusers": ".mp4"}
DEFAULT_RUN_ROOT = pathlib.Path.home() / ".cache" / "difflet" / "verify-cli-cuda"
DEFAULT_PROMPT = "a photorealistic cat sitting in a sunlit garden"


def cell_output(run_dir: pathlib.Path, model_id: str, config: str) -> pathlib.Path:
    slug = model_id.split("/")[-1].lower()
    return run_dir / f"{slug}_{config}{CELL_OUTPUT_SUFFIX.get(model_id, '.png')}"


def check_output(result: StepResult, output: pathlib.Path) -> StepResult:
    """A passing generate that left no file at ``output`` is a failure."""
    if result.status is not Status.PASS or output.is_file():
        return result
    reason = f"exit 0 but no output at {output}"
    return StepResult(Status.FAIL, result.duration, reason, cmd=result.cmd)


def cuda_env() -> dict[str, str]:
    return {**os.environ, "DIFFLET_BACKEND": "cuda"}


def build_cmds(
    model_id: str, *, prompt: str, output: str, steps: int, seed: int, cache_dir: str | None
) -> dict[str, list[str]]:
    common = ["--model-id", model_id, "--tp-degree", "1"]
    if cache_dir:
        common += ["--cache-dir", cache_dir]
    return {
        "download": _DIFFLET_CMD + ["download", "--model-id", model_id],
        "compile": _DIFFLET_CMD + ["compile", *common],
        "generate": _DIFFLET_CMD
        + ["generate", *common, "--prompt", prompt, "--output", output, "--steps", str(steps), "--seed", str(seed)],
    }


def run_step(cmd: list[str], log_fh, *, timeout: float, env: dict[str, str]) -> StepResult:
    """verify_cli.run_step with an explicit environment (it has none)."""
    log_fh.write(f"\n{'=' * 60}\nCMD: {' '.join(cmd)}\nSTARTED: {datetime.datetime.now().isoformat()}\n{'=' * 60}\n")
    log_fh.flush()
    start = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        duration = time.monotonic() - start
        _kill_process_group(proc)
        log_fh.write(f"\nTIMEOUT after {duration:.0f}s (process group killed)\n")
        return StepResult(Status.FAIL, duration, f"timeout after {timeout:.0f}s", cmd=cmd)
    duration = time.monotonic() - start
    log_fh.write(f"\nFINISHED in {duration:.1f}s (exit {proc.returncode})\n")
    log_fh.flush()
    if proc.returncode != 0:
        return StepResult(Status.FAIL, duration, f"exit code {proc.returncode}", cmd=cmd)
    return StepResult(Status.PASS, duration, cmd=cmd)


def cell_outcome(results: dict[str, StepResult]) -> Status:
    return Status.PASS if all(r.status is Status.PASS for r in results.values()) else Status.FAIL


def _device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # torch missing or broken: still a valid (failing) run
        pass
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        help="generate steps for every cell (default: CELL_STEPS)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds per step")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args(argv)

    run_dir = allocate_run_dir(pathlib.Path(args.run_root).expanduser(), datetime.datetime.now())
    env = cuda_env()
    cells = []
    for model_id, config in CELLS:
        slug = model_id.split("/")[-1].lower()
        steps = args.steps if args.steps is not None else CELL_STEPS[model_id]
        output = cell_output(run_dir, model_id, config)
        cmds = build_cmds(
            model_id,
            prompt=args.prompt,
            output=str(output),
            steps=steps,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )
        results: dict[str, StepResult] = {}
        with (run_dir / f"{slug}_{config}.log").open("w") as log_fh:
            for step in ("download", "compile", "generate"):
                results[step] = run_step(cmds[step], log_fh, timeout=args.timeout, env=env)
                if step == "generate":
                    results[step] = check_output(results[step], output)
                if results[step].status is not Status.PASS:
                    break
        outcome = cell_outcome(results)
        timing = ", ".join(f"{k} {r.duration:.1f} s" for k, r in results.items() if r.duration is not None)
        print(f"{model_id} {config}: {outcome.value} ({timing})", flush=True)
        cells.append(
            {
                "model_id": model_id,
                "config": config,
                "outcome": outcome.value,
                "generate_steps": steps,
                "steps": {
                    k: {"status": r.status.value, "duration_s": r.duration, "reason": r.reason, "cmd": r.cmd}
                    for k, r in results.items()
                },
            }
        )
    payload = {
        "backend": "cuda",
        "device": _device_name(),
        "generate_steps": args.steps,  # the --steps override; None = per-cell CELL_STEPS
        "seed": args.seed,
        "run_dir": str(run_dir),
        "cells": cells,
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"results: {run_dir / 'results.json'}")
    return 0 if all(c["outcome"] == "PASS" for c in cells) else 1


if __name__ == "__main__":
    sys.exit(main())
