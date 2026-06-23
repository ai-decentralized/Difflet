from __future__ import annotations

import argparse
import datetime
import pathlib
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum

STEPS = ["download", "compile", "generate"]

_HF_CACHE = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
_DC = pathlib.Path.home() / ".cache" / "difflet"


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class StepResult:
    status: Status
    reason: str = ""
    cmd: list[str] = field(default_factory=list)


MODEL_CONFIGS: dict[str, dict] = {
    "flux": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "compile_args": ["--tp-degree", "2", "--cp-degree", "2",
                         "--height", "1024", "--width", "1024"],
        "generate_args": ["--tp-degree", "2", "--cp-degree", "2",
                          "--height", "1024", "--width", "1024",
                          "--prompt", "a cat sitting on a bench"],
        "output_filename": "flux.png",
        "staged": False,
        "download_glob": str(_HF_CACHE / "models--black-forest-labs--FLUX.1-dev" / "snapshots" / "*"),
        "compile_globs": [str(_DC / "flux" / "*")],
    },
    "ltx_2": {
        "model_id": "Lightricks/LTX-2",
        "compile_args": ["--tp-degree", "4",
                         "--height", "256", "--width", "384", "--num-frames", "9"],
        "generate_args": ["--tp-degree", "4",
                          "--height", "256", "--width", "384", "--num-frames", "9",
                          "--prompt", "a cat walking through a garden"],
        "output_filename": "ltx2.mp4",
        "staged": False,
        "download_glob": str(_HF_CACHE / "models--Lightricks--LTX-2" / "snapshots" / "*"),
        "compile_globs": [str(_DC / "ltx_2" / "*")],
    },
    "wan": {
        "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "compile_args": ["--tp-degree", "2", "--cp-degree", "2",
                         "--height", "480", "--width", "832", "--num-frames", "9"],
        "generate_args": ["--tp-degree", "2", "--cp-degree", "2",
                          "--height", "480", "--width", "832", "--num-frames", "9",
                          "--steps", "50", "--guidance-scale", "1.0", "--seed", "42",
                          "--prompt", "a cat walking through a garden"],
        "output_filename": "wan.mp4",
        "staged": True,
        "download_glob": str(_HF_CACHE / "models--Wan-AI--Wan2.2-T2V-A14B-Diffusers" / "snapshots" / "*"),
        "compile_globs": [
            str(_DC / "wan_transformer_tp2cp2_*"),
            str(_DC / "wan_vae_*"),
        ],
    },
    "hunyuan_video": {
        "model_id": "hunyuanvideo-community/HunyuanVideo",
        "compile_args": ["--tp-degree", "2", "--cp-degree", "2",
                         "--height", "320", "--width", "512", "--num-frames", "61"],
        "generate_args": ["--tp-degree", "2", "--cp-degree", "2",
                          "--height", "320", "--width", "512", "--num-frames", "61",
                          "--steps", "50", "--guidance-scale", "6.0", "--seed", "42",
                          "--prompt", "a cat sitting on a bench"],
        "output_filename": "hunyuan.mp4",
        "staged": True,
        "download_glob": str(_HF_CACHE / "models--hunyuanvideo-community--HunyuanVideo" / "snapshots" / "*"),
        "compile_globs": [
            str(_DC / "hunyuan_video_clip"),
            str(_DC / "hunyuan_video_llama_*"),
            str(_DC / "hunyuan_video_dit_*"),
        ],
    },
    "qwen_image": {
        "model_id": "Qwen/Qwen-Image",
        "compile_args": ["--tp-degree", "2", "--cp-degree", "2",
                         "--height", "1024", "--width", "1024"],
        "generate_args": ["--tp-degree", "2", "--cp-degree", "2",
                          "--height", "1024", "--width", "1024",
                          "--steps", "50", "--guidance-scale", "7.5", "--seed", "42",
                          "--prompt", "a cat sitting on a bench"],
        "output_filename": "qwen.png",
        "staged": True,
        "download_glob": str(_HF_CACHE / "models--Qwen--Qwen-Image" / "snapshots" / "*"),
        "compile_globs": [
            str(_DC / "qwen_image_enc_*"),
            str(_DC / "qwen_image_dit_*"),
            str(_DC / "qwen_image_vae_*"),
        ],
    },
}


def _check_glob(glob_str: str) -> bool:
    """Return True if glob_str matches at least one existing path."""
    p = pathlib.Path(glob_str).expanduser()
    if "*" not in glob_str:
        return p.exists()
    return bool(list(p.parent.glob(p.name)))


def run_step(
    cmd: list[str],
    artifact_globs: list[str],
    log_fh,
    *,
    _check_artifact=None,
) -> StepResult:
    checker = _check_artifact or _check_glob
    ts = datetime.datetime.now().isoformat()
    log_fh.write(f"\n{'='*60}\nCMD: {' '.join(cmd)}\nSTARTED: {ts}\n{'='*60}\n")
    log_fh.flush()

    proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    log_fh.flush()

    if proc.returncode != 0:
        return StepResult(status=Status.FAIL, reason=f"exit code {proc.returncode}", cmd=cmd)

    for glob_str in artifact_globs:
        if not checker(glob_str):
            return StepResult(
                status=Status.FAIL,
                reason=f"artifact not found: {glob_str}",
                cmd=cmd,
            )

    return StepResult(status=Status.PASS, cmd=cmd)


def _build_cmd(
    step: str,
    model_key: str,
    cfg: dict,
    work_dir: pathlib.Path,
) -> tuple[list[str], list[str]]:
    model_id = cfg["model_id"]
    base = ["difflet", step, "--model-id", model_id]

    if step == "download":
        return base, [cfg["download_glob"]]

    if step == "compile":
        return base + cfg["compile_args"], cfg["compile_globs"]

    # generate
    output_path = work_dir / cfg["output_filename"]
    cmd = base + cfg["generate_args"] + ["--output", str(output_path)]
    if cfg["staged"]:
        cmd += ["--work-dir", str(work_dir), "--keep-work-dir"]
    return cmd, [str(output_path)]


def _run_model(
    model_key: str,
    cfg: dict,
    log_fh,
    *,
    _check_artifact=None,
) -> dict[str, StepResult]:
    work_dir = pathlib.Path(f"/tmp/logs/verify_{model_key}")
    work_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, StepResult] = {}
    failed = False

    for step in STEPS:
        if failed:
            results[step] = StepResult(status=Status.SKIP)
            continue
        cmd, artifact_globs = _build_cmd(step, model_key, cfg, work_dir)
        print(f"  [{model_key}] {step} ...")
        result = run_step(cmd, artifact_globs, log_fh, _check_artifact=_check_artifact)
        results[step] = result
        if result.status == Status.FAIL:
            failed = True

    return results


def format_summary(
    results: dict[str, dict[str, StepResult]],
    active_configs: dict[str, dict],
    log_path: str,
) -> str:
    NAME_W, COL_W = 18, 10
    sep = "=" * 60

    lines: list[str] = [f"\n{sep}", "=== SUMMARY ===", ""]

    header = f"{'Model':<{NAME_W}}" + "".join(f"  {s:<{COL_W - 2}}" for s in STEPS)
    lines.append(header)
    lines.append("-" * NAME_W + ("  " + "-" * (COL_W - 2)) * len(STEPS))

    for model_key in active_configs:
        row = f"{model_key:<{NAME_W}}"
        for step in STEPS:
            r = results.get(model_key, {}).get(step)
            status = r.status.value if r else Status.SKIP.value
            row += f"  {status:<{COL_W - 2}}"
        lines.append(row)

    failures = [
        (model_key, step, r)
        for model_key in active_configs
        for step in STEPS
        if (r := results.get(model_key, {}).get(step)) and r.status == Status.FAIL
    ]

    if failures:
        lines += ["", "FAILED COMMANDS:"]
        for model_key, step, r in failures:
            lines += [
                f"  [{model_key}] {step}",
                f"    cmd:  {' '.join(r.cmd)}",
                f"    why:  {r.reason}",
                f"    log:  {log_path}",
                "",
            ]

    lines += ["", "ARTIFACT LOCATIONS:"]
    for model_key, cfg in active_configs.items():
        lines.append(f"  {model_key:<14} download   {cfg['download_glob']}")
        for g in cfg["compile_globs"]:
            lines.append(f"  {model_key:<14} compile    {g}")
        out = f"/tmp/logs/verify_{model_key}/{cfg['output_filename']}"
        lines.append(f"  {model_key:<14} generate   {out}")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="verify_cli",
        description="Verify difflet CLI commands for all supported models.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(MODEL_CONFIGS.keys()),
        default=list(MODEL_CONFIGS.keys()),
        metavar="MODEL",
        help="Models to verify (default: all). Choices: " + ", ".join(MODEL_CONFIGS.keys()),
    )
    args = parser.parse_args(argv)

    pathlib.Path("/tmp/logs").mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"/tmp/logs/verify_cli_{ts}.log"

    print(f"Log: {log_path}")

    active_configs = {k: MODEL_CONFIGS[k] for k in args.models}
    all_results: dict[str, dict[str, StepResult]] = {}

    with open(log_path, "w") as log_fh:
        log_fh.write(f"verify_cli started {datetime.datetime.now().isoformat()}\n")
        log_fh.write(f"models: {args.models}\n")

        for model_key in args.models:
            cfg = MODEL_CONFIGS[model_key]
            log_fh.write(f"\n{'#'*60}\n# MODEL: {model_key}\n{'#'*60}\n")
            log_fh.flush()

            all_results[model_key] = _run_model(model_key, cfg, log_fh)

        summary = format_summary(all_results, active_configs, log_path)
        log_fh.write(summary)
        print(summary)

    print(f"\nDone. Log: {log_path}")


if __name__ == "__main__":
    main()
