#!/usr/bin/env python3
"""On-device FP8 PTQ A/B for Wan: quantize -> compile both -> generate both -> compare.

One command produces the evidence the verification plan asks for, per arm
(bf16, fp8): quantize time, compile time, per-run e2e wall time and weight-load
time, DiT per-step ms (real loop, step 0 excluded), and per run pair the
latent-level error (cosine / MSE / SNR of the DiT output latents) plus PSNR /
SSIM / LPIPS of the decoded videos. Same prompt, seed, shape and steps on both
arms; the bf16 run-to-run pair is a determinism control.

    PYTHONPATH=$PWD python scripts/ptq_fp8_ab.py --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers \\
        --tp-degree 4 --height 480 --width 832 --num-frames 9 --steps 20 --guidance-scale 1.0 \\
        --seed 42 --runs 2 --out-dir artifacts/ptq/wan21-ab [--quant-granularity tensor] \\
        [--quant-act dynamic] [--cache-dir ~/.cache/difflet] [--skip-compile] [--drop-caches]

``--dry-run`` prints every command without running anything. Logs of every
subprocess land under ``<out-dir>/logs``; the summary is ``<out-dir>/ab_summary.json``
and ``<out-dir>/ab_summary.md``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = ("bf16", "fp8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    p.add_argument("--revision", default=None)
    p.add_argument("--tp-degree", type=int, default=4)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=9)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default="a cinematic shot of a red fox running through a snowy forest")
    p.add_argument("--runs", type=int, default=2, help="generate runs per arm (run 0 is the cold one)")
    p.add_argument("--quant-granularity", choices=["tensor", "channel"], default="tensor")
    p.add_argument("--quant-act", choices=["dynamic", "none"], default="dynamic")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--only", choices=["bf16", "fp8", "both"], default="both")
    p.add_argument("--skip-quantize", action="store_true")
    p.add_argument("--skip-compile", action="store_true")
    p.add_argument("--drop-caches", action="store_true",
                   help="sudo-drop the OS page cache before run 0 of each arm (true cold e2e)")
    p.add_argument("--no-lpips", action="store_true")
    p.add_argument("--host-vae", action="store_true", help="decode on the host (needed beyond ~9 frames)")
    p.add_argument("--dry-run", action="store_true")
    return p


def _difflet(args, command: str, *extra: str) -> list[str]:
    cmd = [sys.executable, "-m", "difflet.cli.main", command, "--model-id", args.model_id]
    if args.revision:
        cmd += ["--revision", args.revision]
    if args.cache_dir:
        cmd += ["--cache-dir", args.cache_dir]
    return cmd + list(extra)


def _quant_flags(args) -> list[str]:
    return ["--quant", "fp8", "--quant-granularity", args.quant_granularity, "--quant-act", args.quant_act]


def _shape_flags(args) -> list[str]:
    return ["--tp-degree", str(args.tp_degree), "--height", str(args.height),
            "--width", str(args.width), "--num-frames", str(args.num_frames)]


def _arm_flags(args, arm: str) -> list[str]:
    return _quant_flags(args) if arm == "fp8" else []


def _drop_caches() -> bool:
    return subprocess.run(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"]).returncode == 0


class Runner:
    def __init__(self, args):
        self.args = args
        self.logs = args.out_dir / "logs"
        if not args.dry_run:
            self.logs.mkdir(parents=True, exist_ok=True)

    def run(self, name: str, cmd: list[str]) -> tuple[float, str, int]:
        printable = " ".join(shlex.quote(c) for c in cmd)
        print(f"[ab] {name}: {printable}", flush=True)
        if self.args.dry_run:
            return 0.0, "", 0
        log = self.logs / f"{name}.log"
        started = time.perf_counter()
        with log.open("w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ))
        wall = time.perf_counter() - started
        text = log.read_text(errors="ignore")
        if proc.returncode != 0:
            tail = "\n".join(text.splitlines()[-30:])
            raise RuntimeError(f"{name} failed ({proc.returncode}); see {log}\n{tail}")
        return wall, text, proc.returncode


def _step_stats(step_seconds: list[float]) -> dict | None:
    if not step_seconds:
        return None
    ms = [s * 1000.0 for s in step_seconds]
    return {"n": len(ms), "mean": statistics.fmean(ms), "median": statistics.median(ms),
            "min": min(ms), "max": max(ms)}


def _latents_path(work_dir: Path) -> Path | None:
    found = sorted(glob.glob(str(work_dir / "**" / "latents*.pt"), recursive=True))
    return Path(found[0]) if found else None


def main() -> int:
    args = build_parser().parse_args()
    arms = [a for a in ARMS if args.only in ("both", a)]
    runner = Runner(args)
    summary: dict = {
        "model_id": args.model_id, "shape": {"height": args.height, "width": args.width,
                                             "num_frames": args.num_frames},
        "steps": args.steps, "guidance_scale": args.guidance_scale, "seed": args.seed,
        "prompt": args.prompt, "tp_degree": args.tp_degree,
        "quant": {"format": "fp8", "weight_granularity": args.quant_granularity,
                  "activation": args.quant_act},
        "arms": {arm: {"runs": []} for arm in arms}, "compare": {},
    }

    from benchmark.adapters.trainium import parse_dit_step_seconds
    from benchmark.parse_generate import parse as parse_generate_log

    # 1. quantize (CPU, once per granularity)
    if "fp8" in arms and not args.skip_quantize:
        wall, text, _ = runner.run("quantize", _difflet(
            args, "quantize", "--quant", "fp8", "--quant-granularity", args.quant_granularity))
        summary["arms"]["fp8"]["quantize_seconds"] = round(wall, 3)

    # 2. compile both arms
    for arm in arms:
        if args.skip_compile:
            continue
        cmd = _difflet(args, "compile", *_shape_flags(args), *_arm_flags(args, arm))
        if args.host_vae:
            cmd.append("--host-vae")
        wall, text, _ = runner.run(f"compile_{arm}", cmd)
        summary["arms"][arm]["compile_seconds"] = round(wall, 3)
        summary["arms"][arm]["compile_cache_hit"] = "already compiled" in text

    # 3. generate: N runs per arm, same seed, latents kept
    for arm in arms:
        for run in range(max(1, args.runs)):
            if run == 0 and args.drop_caches and not args.dry_run:
                summary["arms"][arm]["page_cache_dropped"] = _drop_caches()
            out = args.out_dir / f"{arm}_run{run}.mp4"
            work = args.out_dir / f"work_{arm}_run{run}"
            cmd = _difflet(args, "generate", *_shape_flags(args), *_arm_flags(args, arm),
                           "--prompt", args.prompt, "--seed", str(args.seed),
                           "--steps", str(args.steps), "--guidance-scale", str(args.guidance_scale),
                           "--output", str(out), "--work-dir", str(work), "--keep-work-dir")
            if args.host_vae:
                cmd.append("--host-vae")
            wall, text, _ = runner.run(f"generate_{arm}_run{run}", cmd)
            breakdown = parse_generate_log(text, wall) if text else {}
            record = {
                "run": run,
                "e2e_wall_seconds": round(wall, 3),
                "weights_load_total_seconds": breakdown.get("weights_load_total_s"),
                "compute_and_overhead_seconds": breakdown.get("compute_and_overhead_s"),
                "dit_step_ms": _step_stats(parse_dit_step_seconds(text)),
                "output": str(out),
                "latents": str(_latents_path(work)) if not args.dry_run else str(work / "latents.pt"),
            }
            summary["arms"][arm]["runs"].append(record)

    # 4. compare outputs and latents
    if not args.dry_run:
        from difflet.quant.metrics import compare_latents, compare_outputs

        lpips_net = None if args.no_lpips else "alex"

        def compare(tag, ref, test):
            entry: dict = {}
            try:
                entry["output"] = compare_outputs(ref["output"], test["output"], lpips_net=lpips_net)
            except Exception as exc:  # decoded outputs may be .pt on export failure
                entry["output_error"] = f"{type(exc).__name__}: {exc}"
            if ref.get("latents") and test.get("latents") and Path(ref["latents"]).exists() \
                    and Path(test["latents"]).exists():
                entry["latents"] = compare_latents(ref["latents"], test["latents"])
            summary["compare"][tag] = entry

        bf16_runs = summary["arms"].get("bf16", {}).get("runs", [])
        fp8_runs = summary["arms"].get("fp8", {}).get("runs", [])
        for i, (ref, test) in enumerate(zip(bf16_runs, fp8_runs)):
            compare(f"fp8_vs_bf16_run{i}", ref, test)
        if len(bf16_runs) >= 2:
            compare("bf16_run1_vs_run0_control", bf16_runs[0], bf16_runs[1])
        if len(fp8_runs) >= 2:
            compare("fp8_run1_vs_run0_control", fp8_runs[0], fp8_runs[1])

    # 5. report
    if args.dry_run:
        print("[ab] dry run: nothing executed")
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ab_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / "ab_summary.md").write_text(render_markdown(summary))
    print((args.out_dir / "ab_summary.md").read_text())
    print(f"[ab] wrote {args.out_dir / 'ab_summary.json'}")
    return 0


def render_markdown(summary: dict) -> str:
    def fmt(value, digits=1):
        return "—" if value is None else f"{value:.{digits}f}"

    lines = [
        f"# FP8 PTQ A/B — {summary['model_id']}",
        "",
        f"shape {summary['shape']['height']}x{summary['shape']['width']}x{summary['shape']['num_frames']}, "
        f"{summary['steps']} steps, guidance {summary['guidance_scale']}, seed {summary['seed']}, "
        f"tp{summary['tp_degree']}; quant {summary['quant']}",
        "",
        "| arm | quantize s | compile s (cache hit) | run | e2e wall s | weights load s | DiT step ms mean / median (n) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, data in summary["arms"].items():
        for record in data["runs"]:
            step = record.get("dit_step_ms") or {}
            lines.append(
                f"| {arm} | {fmt(data.get('quantize_seconds'))} | "
                f"{fmt(data.get('compile_seconds'))} ({data.get('compile_cache_hit')}) | {record['run']} | "
                f"{fmt(record['e2e_wall_seconds'])} | {fmt(record.get('weights_load_total_seconds'))} | "
                f"{fmt(step.get('mean'))} / {fmt(step.get('median'))} ({step.get('n', 0)}) |"
            )
    lines += ["", "| pair | PSNR dB | SSIM | LPIPS | latent cosine | latent MSE | latent SNR dB |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for tag, entry in summary["compare"].items():
        out = entry.get("output") or {}
        lat = entry.get("latents") or {}
        lpips = out.get("lpips")
        lpips_str = "n/a" if lpips is None else f"{lpips:.4f}"
        mse_str = "—" if lat.get("mse") is None else f"{lat['mse']:.3e}"
        lines.append(
            f"| {tag} | {fmt(out.get('psnr_db'), 2)} | {fmt(out.get('ssim'), 4)} | "
            f"{lpips_str} | {fmt(lat.get('cosine'), 6)} | {mse_str} | {fmt(lat.get('snr_db'), 2)} |"
        )
        if entry.get("output_error"):
            lines.append(f"| {tag} (output error) | {entry['output_error']} | | | | | |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
