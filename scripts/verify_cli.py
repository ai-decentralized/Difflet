"""On-device parallelism-config matrix for the difflet CLI.

Runs ``difflet download`` once per model, then per supported parallel config
``difflet compile`` (warms the compile cache) followed by a TIMED
``difflet generate`` (pure inference on the warm cache). Every config is sized
to exactly the 4 NeuronCores of a trn2.3xlarge:
world_size = (2 if cfg else 1) * cp * tp.

Design: docs/superpowers/specs/2026-07-05-verify-cli-parallelism-matrix-design.md
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum

_HF_CACHE = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
_DIFFLET_CACHE = pathlib.Path.home() / ".cache" / "difflet"

# Invoke the CLI as a module of THIS checkout (cwd on sys.path), not the
# `difflet` console script: the editable install may point at a different
# checkout that lacks the flags under test (e.g. --dp on a feature branch).
_DIFFLET_CMD = [sys.executable, "-m", "difflet.cli.main"]


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    XFAIL = "XFAIL"  # expected failure (documented known gap)
    XPASS = "XPASS"  # expected failure that unexpectedly passed


@dataclass
class StepResult:
    status: Status
    duration: float | None = None
    reason: str = ""
    cmd: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ParallelConfig:
    key: str
    flags: tuple[str, ...]
    world_size: int


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    shape_flags: tuple[str, ...]
    # Sample-grade generate-only settings (steps/guidance) so outputs are
    # prompt-faithful and human-verifiable; compile never sees these.
    generate_flags: tuple[str, ...]
    prompt: str
    output_name: str            # value passed to --output (inside the cell dir)
    artifact_names: tuple[str, ...]  # accepted output artifacts, in preference order
    staged: bool
    cache_dir: str | None = None  # per-model --cache-dir override

    @property
    def download_glob(self) -> str:
        org, name = self.model_id.split("/")
        return str(_HF_CACHE / f"models--{org}--{name}" / "snapshots" / "*")


PARALLEL_CONFIGS: dict[str, ParallelConfig] = {
    "tp4": ParallelConfig("tp4", ("--tp-degree", "4"), 4),
    "tp2cp2": ParallelConfig("tp2cp2", ("--tp-degree", "2", "--cp-degree", "2"), 4),
    # --cfg-parallel doubles world_size: 2 (tp) x 2 (cfg) = 4
    "tp2cfg": ParallelConfig("tp2cfg", ("--tp-degree", "2", "--cfg-parallel"), 4),
    # SP reuses the TP group; world_size unchanged
    "tp4sp": ParallelConfig("tp4sp", ("--tp-degree", "4", "--sp"), 4),
    # DP: 2 replicas x tp2 = 4 cores. Generate runs a 2-request batch through
    # the router (one request per replica); compile resolves to the same dp=1
    # tp2 artifact both replicas load (compile-once-load-k).
    "dp2tp2": ParallelConfig("dp2tp2", ("--tp-degree", "2", "--dp", "2"), 4),
}

MODELS: dict[str, ModelSpec] = {
    "flux": ModelSpec(
        key="flux", model_id="black-forest-labs/FLUX.1-dev",
        shape_flags=("--height", "1024", "--width", "1024"),
        generate_flags=("--steps", "28"),
        prompt="a cat sitting on a bench",
        output_name="flux.png", artifact_names=("flux.png",), staged=False,
    ),
    "qwen_image": ModelSpec(
        key="qwen_image", model_id="Qwen/Qwen-Image",
        shape_flags=("--height", "1024", "--width", "1024"),
        generate_flags=("--steps", "50"),
        prompt="a cat sitting on a bench",
        # torchvision PNG write falls back to a .pt tensor on failure
        output_name="qwen.png", artifact_names=("qwen.png", "qwen.pt"), staged=True,
    ),
    "ltx_2": ModelSpec(
        key="ltx_2", model_id="Lightricks/LTX-2",
        shape_flags=("--height", "256", "--width", "384", "--num-frames", "121"),
        generate_flags=("--steps", "40"),
        prompt="a cat walking through a garden",
        # MP4 export with a .pt tensor fallback on codec failure
        output_name="ltx2.mp4", artifact_names=("ltx2.mp4", "ltx2.pt"), staged=False,
    ),
    "wan": ModelSpec(
        key="wan", model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        # 9 frames: device attention fidelity degrades on long sequences
        # (f81 = ~33k joint tokens -> washed/blank output; f9 = ~4.7k -> real
        # content, one-step parity 0.9986). Long-clip support is blocked on the
        # toolchain attention kernel, not difflet.
        shape_flags=("--height", "480", "--width", "832", "--num-frames", "9"),
        generate_flags=("--steps", "40", "--guidance-scale", "4.0"),
        prompt="a cat walking through a garden",
        # export_to_video falls back to a .pt tensor on failure
        output_name="wan.mp4", artifact_names=("wan.mp4", "wan.pt"), staged=True,
    ),
    "wan2_1": ModelSpec(
        key="wan2_1", model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        shape_flags=("--height", "480", "--width", "832", "--num-frames", "9"),
        generate_flags=("--steps", "40", "--guidance-scale", "4.0"),
        prompt="a cat walking through a garden",
        output_name="wan21.mp4", artifact_names=("wan21.mp4", "wan21.pt"), staged=True,
        # Isolate wan2_1's compiled artifacts: pre-fix difflet installs shared
        # staged compiled-dir names between Wan 2.2 and 2.1 (fixed by
        # wan.py _cache_prefix); harmless belt-and-braces on fixed installs.
        cache_dir=str(_DIFFLET_CACHE / "wan2_1"),
    ),
    "hunyuan_video": ModelSpec(
        key="hunyuan_video", model_id="hunyuanvideo-community/HunyuanVideo",
        shape_flags=("--height", "320", "--width", "512", "--num-frames", "121"),
        generate_flags=("--steps", "50"),
        prompt="a cat sitting on a bench",
        output_name="hunyuan.mp4", artifact_names=("hunyuan.mp4", "hunyuan.pt"), staged=True,
    ),
    "hunyuan_video_15": ModelSpec(
        key="hunyuan_video_15",
        model_id="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        # scaffold: compile/generate raise NotImplementedError before shape matters
        shape_flags=(), generate_flags=(), prompt="a cat sitting on a bench",
        output_name="hunyuan15.mp4", artifact_names=("hunyuan15.mp4", "hunyuan15.pt"),
        staged=False,
    ),
}

# Skip-rule sets. DISTILLED and SP_SUPPORTED mirror difflet.cli.main
# (_DISTILLED_MODELS / _SP_SUPPORTED_MODELS); unit tests pin them to the source
# of truth. CP_UNSUPPORTED lives in the model entries (ltx_2/entry.py raises for
# cp_degree>1; hunyuan_video/entry.py for HunyuanVideo 1.5), not in main.py.
DISTILLED = frozenset({"flux", "qwen_image", "hunyuan_video", "hunyuan_video_15"})
SP_SUPPORTED = frozenset({"flux", "wan", "wan2_1", "hunyuan_video"})
CP_UNSUPPORTED = frozenset({"ltx_2", "hunyuan_video_15"})

# Documented known gaps:
# - hunyuan_video_15 is a scaffold: compile/generate raise NotImplementedError
#   (difflet/cli/orchestrators/hunyuan_video_15.py). Its dp cell fails the
#   same way (DP inherits automatically once generate lands).
# - hunyuan_video tp2cp2: neuronx-cc 2.25.3371.0 dies with INTERNAL_ERROR
#   NCC_INLA001 (SBUF alloc out of bound) on the CP-degree-2 DiT graph —
#   deterministic, reproduced twice; HLO repro preserved for an
#   aws-neuron-sdk ticket. On 2.26.6360.0 the signature drifted to
#   NCC_IBIR243 "Access pattern out of bounds" — same cell, still expected.
# - hunyuan_video dp2tp2: per-replica HBM capacity, not a DP bug. The generate
#   stage dies loading the compiled VAE (torch.jit.load, NRT "status=4
#   Allocation Failure") at f121 on a 2-core replica; a lone pinned dp=1 tp2
#   replica with the rest of the device idle fails identically (verified
#   2026-07-12, NRT 2.33.10 / neuronx-cc 2.26.6360.0), while 4-core tp4
#   passes. hunyuan_video has never fit any 2-core-per-shard config.
EXPECTED_FAIL_CELLS = frozenset({
    ("hunyuan_video_15", "tp4"),
    ("hunyuan_video_15", "dp2tp2"),
    ("hunyuan_video", "tp2cp2"),
    ("hunyuan_video", "dp2tp2"),
})


def skip_reason(model_key: str, config_key: str) -> str | None:
    """Why (model, config) cannot run, or None if it is supported."""
    cfg = PARALLEL_CONFIGS[config_key]
    if "--cfg-parallel" in cfg.flags and model_key in DISTILLED:
        return "distilled"
    if "--sp" in cfg.flags and model_key not in SP_SUPPORTED:
        return "no-SP"
    if "--cp-degree" in cfg.flags and model_key in CP_UNSUPPORTED:
        return "no-CP"
    return None


@dataclass(frozen=True)
class Cell:
    model_key: str
    config_key: str
    skip_reason: str | None
    expected_fail: bool


def plan_cells(model_keys: list[str], config_keys: list[str]) -> list[Cell]:
    return [
        Cell(
            model_key=m, config_key=c,
            skip_reason=skip_reason(m, c),
            expected_fail=(m, c) in EXPECTED_FAIL_CELLS,
        )
        for m in model_keys
        for c in config_keys
    ]


# ---------------------------------------------------------------- commands

def build_download_cmd(spec: ModelSpec) -> list[str]:
    return _DIFFLET_CMD + ["download", "--model-id", spec.model_id]


def _common_flags(spec: ModelSpec, cfg: ParallelConfig) -> list[str]:
    flags = list(cfg.flags) + list(spec.shape_flags)
    if spec.cache_dir:
        flags += ["--cache-dir", spec.cache_dir]
    return flags


def build_compile_cmd(spec: ModelSpec, cfg: ParallelConfig) -> list[str]:
    return _DIFFLET_CMD + ["compile", "--model-id", spec.model_id] + _common_flags(spec, cfg)


def _dp_variants(name: str, replica: int) -> list[str]:
    """Per-request artifact preference list, e.g. wan.mp4 -> [wan_dp1.mp4, wan_dp1.pt]."""
    stem, suffix = name.rsplit(".", 1)
    variants = [f"{stem}_dp{replica}.{suffix}"]
    if suffix != "pt":
        variants.append(f"{stem}_dp{replica}.pt")
    return variants


def build_generate_cmd(
    spec: ModelSpec, cfg: ParallelConfig, cell_dir: pathlib.Path,
) -> tuple[list[str], list[list[pathlib.Path]]]:
    """Returns (cmd, artifact_groups): each group is a preference list of paths,
    at least one of which must exist for the cell to pass."""
    cmd = _DIFFLET_CMD + ["generate", "--model-id", spec.model_id] + _common_flags(spec, cfg)
    cmd += list(spec.generate_flags)
    if "--dp" in cfg.flags:
        # DP cell: a 2-request batch routed across the replicas.
        dp = int(cfg.flags[cfg.flags.index("--dp") + 1])
        stem, suffix = spec.output_name.rsplit(".", 1)
        requests_file = cell_dir / "requests.jsonl"
        cell_dir.mkdir(parents=True, exist_ok=True)
        requests_file.write_text(
            "\n".join(
                json.dumps({
                    "prompt": f"{spec.prompt} (variant {r})",
                    "output": str(cell_dir / f"{stem}_dp{r}.{suffix}"),
                    "seed": 42 + r,
                })
                for r in range(dp)
            ) + "\n"
        )
        cmd += ["--requests", str(requests_file)]
        artifact_groups = [
            [cell_dir / v for v in _dp_variants(spec.output_name, r)] for r in range(dp)
        ]
    else:
        cmd += ["--prompt", spec.prompt, "--output", str(cell_dir / spec.output_name)]
        artifact_groups = [[cell_dir / name for name in spec.artifact_names]]
    if spec.staged or "--dp" in cfg.flags:
        cmd += ["--work-dir", str(cell_dir / "work"), "--keep-work-dir"]
    return cmd, artifact_groups


# ---------------------------------------------------------------- execution

def run_step(cmd: list[str], log_fh, *, timeout: float) -> StepResult:
    ts = datetime.datetime.now().isoformat()
    log_fh.write(f"\n{'=' * 60}\nCMD: {' '.join(cmd)}\nSTARTED: {ts}\n{'=' * 60}\n")
    log_fh.flush()

    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired:
        duration = time.monotonic() - start
        log_fh.write(f"\nTIMEOUT after {duration:.0f}s\n")
        return StepResult(status=Status.FAIL, duration=duration,
                          reason=f"timeout after {timeout:.0f}s", cmd=cmd)
    duration = time.monotonic() - start
    log_fh.write(f"\nFINISHED in {duration:.1f}s (exit {proc.returncode})\n")
    log_fh.flush()

    if proc.returncode != 0:
        return StepResult(status=Status.FAIL, duration=duration,
                          reason=f"exit code {proc.returncode}", cmd=cmd)
    return StepResult(status=Status.PASS, duration=duration, cmd=cmd)


def run_cell(
    spec: ModelSpec, cfg: ParallelConfig, cell_dir: pathlib.Path, *, timeout: float,
) -> dict[str, StepResult]:
    cell_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, StepResult] = {}

    with open(cell_dir / "step_compile.log", "w") as fh:
        results["compile"] = run_step(build_compile_cmd(spec, cfg), fh, timeout=timeout)

    if results["compile"].status != Status.PASS:
        results["generate"] = StepResult(status=Status.SKIP, reason="compile failed")
        return results

    gen_cmd, artifact_groups = build_generate_cmd(spec, cfg, cell_dir)
    with open(cell_dir / "step_generate.log", "w") as fh:
        result = run_step(gen_cmd, fh, timeout=timeout)
    missing = [g for g in artifact_groups if not any(p.exists() for p in g)]
    if result.status == Status.PASS and missing:
        result = StepResult(
            status=Status.FAIL, duration=result.duration,
            reason="artifact not found: "
            + " ; ".join(" | ".join(str(p) for p in g) for g in missing),
            cmd=gen_cmd,
        )
    results["generate"] = result
    return results


def cell_outcome(compile_r: StepResult, generate_r: StepResult, *, expected_fail: bool) -> Status:
    passed = compile_r.status == Status.PASS and generate_r.status == Status.PASS
    if expected_fail:
        return Status.XPASS if passed else Status.XFAIL
    return Status.PASS if passed else Status.FAIL


def compute_exit_code(outcomes) -> int:
    return 1 if any(o in (Status.FAIL, Status.XPASS) for o in outcomes) else 0


# ---------------------------------------------------------------- reporting

def _cell_label(cell_result: dict) -> str:
    outcome: Status = cell_result["outcome"]
    if outcome == Status.SKIP:
        return f"SKIP {cell_result.get('reason', '')}".strip()
    gen = cell_result.get("generate")
    if outcome in (Status.PASS, Status.XPASS) and gen and gen.duration is not None:
        return f"{outcome.value} {gen.duration:.1f}s"
    return outcome.value


def format_summary(
    results: dict, model_keys: list[str], config_keys: list[str], run_dir: str,
) -> str:
    name_w, col_w = 18, 17
    lines = ["", "=" * 60, "=== MATRIX SUMMARY ===", ""]

    header = f"{'Model':<{name_w}}" + "".join(f"  {c:<{col_w - 2}}" for c in config_keys)
    lines.append(header)
    lines.append("-" * name_w + ("  " + "-" * (col_w - 2)) * len(config_keys))
    for m in model_keys:
        row = f"{m:<{name_w}}"
        for c in config_keys:
            cell = results["cells"].get((m, c))
            row += f"  {_cell_label(cell) if cell else '-':<{col_w - 2}}"
        lines.append(row)

    lines += ["", "DOWNLOADS:"]
    for m in model_keys:
        r = results["downloads"].get(m)
        if r is None:
            continue
        dur = f" {r.duration:.0f}s" if r.duration is not None else ""
        reason = f"  ({r.reason})" if r.reason else ""
        lines.append(f"  {m:<{name_w}} {r.status.value}{dur}{reason}")

    unexpected = []
    for m in model_keys:
        for c in config_keys:
            cell = results["cells"].get((m, c))
            if not cell or cell["outcome"] not in (Status.FAIL, Status.XPASS):
                continue
            for step_name in ("compile", "generate"):
                step = cell.get(step_name)
                if step is not None and step.status == Status.FAIL:
                    unexpected.append((m, c, step_name, step))
    if unexpected:
        lines += ["", "FAILED (unexpected):"]
        for m, c, step_name, step in unexpected:
            lines += [
                f"  [{m}/{c}] {step_name}",
                f"    cmd:  {' '.join(step.cmd)}",
                f"    why:  {step.reason}",
                f"    log:  {run_dir}/{m}/{c}/step_{step_name}.log",
            ]

    lines += ["", f"Artifacts and per-step logs under: {run_dir}/<model>/<config>/", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- main

def _disk_free_gb() -> float:
    return shutil.disk_usage(pathlib.Path.home()).free / 1e9


def _results_json(results: dict) -> dict:
    def step(r: StepResult | None):
        if r is None:
            return None
        return {"status": r.status.value, "duration_s": r.duration,
                "reason": r.reason, "cmd": r.cmd}

    return {
        "downloads": {m: step(r) for m, r in results["downloads"].items()},
        "cells": {
            f"{m}/{c}": {
                "outcome": cell["outcome"].value,
                "reason": cell.get("reason", ""),
                "compile": step(cell.get("compile")),
                "generate": step(cell.get("generate")),
            }
            for (m, c), cell in results["cells"].items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify_cli",
        description="Parallelism-config matrix for the difflet CLI "
                    "(7 models x 5 four-core configs).",
    )
    parser.add_argument("--models", nargs="*", choices=list(MODELS), default=list(MODELS),
                        metavar="MODEL",
                        help="Models to verify (default: all). Choices: " + ", ".join(MODELS))
    parser.add_argument("--configs", nargs="*", choices=list(PARALLEL_CONFIGS),
                        default=list(PARALLEL_CONFIGS), metavar="CONFIG",
                        help="Parallel configs (default: all). Choices: "
                             + ", ".join(PARALLEL_CONFIGS))
    parser.add_argument("--step-timeout", type=float, default=14400,
                        help="Per-step subprocess timeout in seconds (default: 14400)")
    args = parser.parse_args(argv)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pathlib.Path(f"/tmp/logs/verify_matrix_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)
    main_log = run_dir / "main.log"
    print(f"Run dir: {run_dir}")

    results: dict = {"downloads": {}, "cells": {}}
    cells = plan_cells(args.models, args.configs)

    with open(main_log, "w") as log_fh:
        log_fh.write(f"verify_cli matrix started {datetime.datetime.now().isoformat()}\n")
        log_fh.write(f"models: {args.models}\nconfigs: {args.configs}\n")

        for model_key in args.models:
            spec = MODELS[model_key]
            log_fh.write(f"\n{'#' * 60}\n# MODEL: {model_key} ({spec.model_id})\n"
                         f"# disk free: {_disk_free_gb():.0f} GB\n{'#' * 60}\n")
            log_fh.flush()

            print(f"[{model_key}] download ...")
            dl = run_step(build_download_cmd(spec), log_fh, timeout=args.step_timeout)
            if dl.status == Status.PASS and not list(
                pathlib.Path(spec.download_glob).parent.glob(
                    pathlib.Path(spec.download_glob).name)
            ):
                dl = StepResult(status=Status.FAIL, duration=dl.duration,
                                reason=f"snapshot not found: {spec.download_glob}",
                                cmd=dl.cmd)
            results["downloads"][model_key] = dl
            print(f"[{model_key}] download {dl.status.value}")

            for cell in [c for c in cells if c.model_key == model_key]:
                key = (cell.model_key, cell.config_key)
                if cell.skip_reason:
                    results["cells"][key] = {"outcome": Status.SKIP,
                                             "reason": cell.skip_reason}
                    continue
                if dl.status != Status.PASS:
                    results["cells"][key] = {"outcome": Status.SKIP,
                                             "reason": "download failed"}
                    continue

                print(f"[{model_key}/{cell.config_key}] compile + timed generate ...")
                cell_dir = run_dir / model_key / cell.config_key
                steps = run_cell(spec, PARALLEL_CONFIGS[cell.config_key], cell_dir,
                                 timeout=args.step_timeout)
                outcome = cell_outcome(steps["compile"], steps["generate"],
                                       expected_fail=cell.expected_fail)
                results["cells"][key] = {"outcome": outcome, **steps}
                gen = steps["generate"]
                timing = f" ({gen.duration:.1f}s generate)" if (
                    outcome in (Status.PASS, Status.XPASS) and gen.duration is not None
                ) else ""
                print(f"[{model_key}/{cell.config_key}] {outcome.value}{timing}")

                (run_dir / "results.json").write_text(
                    json.dumps(_results_json(results), indent=2) + "\n")

        summary = format_summary(results, args.models, args.configs, str(run_dir))
        log_fh.write(summary)
        print(summary)

    (run_dir / "results.json").write_text(json.dumps(_results_json(results), indent=2) + "\n")

    download_statuses = [r.status for r in results["downloads"].values()]
    outcomes = [c["outcome"] for c in results["cells"].values()]
    exit_code = compute_exit_code(outcomes) or (
        1 if Status.FAIL in download_statuses else 0)
    print(f"Done. Log: {main_log} (exit {exit_code})")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
