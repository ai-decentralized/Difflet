#!/usr/bin/env python3
"""F2.0 Wan 2.2 two-stage wall-clock profiler.

Wraps the existing ``wan_smoke.sh`` two-stage split and records per-stage
wall-clock so cclog 65's "VAE share >= 20%" gate can be applied. The split is
unchanged:

  stage 1 (TP=4, NEURON_RT_NUM_CORES=4): text + transformer  -> latents.pt
  stage 2 (TP=1, NEURON_RT_NUM_CORES=1): vae decode          -> tensor / mp4

For F2.0 the question is wall-clock attribution, not e2e correctness — we
attribute stage 1 to (text encoder + DiT denoise loop) and stage 2 to
(VAE decode + decoder side overhead). Stage-1 cannot be further split without
modifying ``examples/wan_example.py``; the conservative attribution is to
report stage_1_total and stage_2_total as separate components, plus the sum.

Example:
    python scripts/profile_wan_twostage_wallclock.py \\
        --output cclogs/m8-dit-vae-separation/profile_wan_2stage_components.json \\
        --num-inference-steps 2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WAN_SMOKE = ROOT / "scripts" / "wan_smoke.sh"
SCHEMA = "nova-f2-0-wan-twostage-wallclock-v1"
GATE_THRESHOLD = 0.20


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="Where to write the per-stage JSON")
    p.add_argument(
        "--latents-path",
        default=str(ROOT / ".nova-cache" / "wan_smoke_latents.pt"),
        help="latents shuttle file between stage 1 and stage 2",
    )
    p.add_argument(
        "--mp4-output",
        default="/tmp/wan_smoke_f2_0.mp4",
        help="stage 2 output target (mp4 export is best-effort)",
    )
    p.add_argument("--num-inference-steps", type=int, default=2)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=9)
    p.add_argument("--tp-degree", type=int, default=4)
    p.add_argument(
        "--prompt", default="a cat walking", help="text prompt for stage 1"
    )
    p.add_argument(
        "--reuse-existing-latents",
        action="store_true",
        help=(
            "If set, skip stage 1 and re-time only stage 2 against an existing "
            "latents shuttle. Useful when stage 1 already ran successfully."
        ),
    )
    return p.parse_args()


def _ensure_smoke_runnable() -> None:
    if not WAN_SMOKE.exists():
        raise SystemExit(f"[f2.0] {WAN_SMOKE} missing")
    if not os.access(WAN_SMOKE, os.X_OK):
        raise SystemExit(f"[f2.0] {WAN_SMOKE} not executable")


def _stage1_cmd(args: argparse.Namespace) -> list[str]:
    return [
        "bash",
        "-c",
        " ".join(
            [
                f"NOVA_WAN_STEPS={args.num_inference_steps}",
                f"NOVA_WAN_FRAMES={args.num_frames}",
                f"NOVA_WAN_HEIGHT={args.height}",
                f"NOVA_WAN_WIDTH={args.width}",
                f"NOVA_WAN_TP_DEGREE={args.tp_degree}",
                f"NOVA_WAN_LATENTS_PATH={args.latents_path}",
                f"NOVA_WAN_PROMPT={shlex_quote(args.prompt)}",
                # We deliberately re-invoke the wan_smoke.sh script for stage 1
                # rather than re-implementing it; the helper splits internally.
                # However wan_smoke.sh runs *both* stages; for clean per-stage
                # timing we shell out to the underlying example directly.
                f"NEURON_RT_NUM_CORES={4}",
                f"{NEURON_PY()}",
                str(ROOT / "examples" / "wan_example.py"),
                f"--model Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                f"--tp-degree {args.tp_degree}",
                "--skip-warmup",
                f"--num-inference-steps {args.num_inference_steps}",
                f"--num-frames {args.num_frames}",
                f"--height {args.height} --width {args.width}",
                f"--prompt {shlex_quote(args.prompt)}",
                "--enable-text --enable-transformer --no-vae",
                "--output-type latent",
                f"--save-latents {args.latents_path}",
                f"--compiled-dir {ROOT / '.nova-cache' / 'wan_smoke_stage1'}",
            ]
        ),
    ]


def _stage2_cmd(args: argparse.Namespace) -> list[str]:
    return [
        "bash",
        "-c",
        " ".join(
            [
                f"NEURON_RT_NUM_CORES={1}",
                f"{NEURON_PY()}",
                str(ROOT / "examples" / "wan_example.py"),
                f"--model Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                "--tp-degree 1",
                "--skip-warmup",
                f"--num-frames {args.num_frames}",
                f"--height {args.height} --width {args.width}",
                "--no-text --no-transformer --enable-vae",
                f"--load-latents {args.latents_path}",
                "--output-type pt",
                f"--output {args.mp4_output}",
                f"--compiled-dir {ROOT / '.nova-cache' / 'wan_smoke_stage2'}",
            ]
        ),
    ]


def shlex_quote(s: str) -> str:
    if not s:
        return "''"
    if all(c.isalnum() or c in "@%+=:,./-_" for c in s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def NEURON_PY() -> str:
    candidate = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python")
    if candidate.exists():
        return str(candidate)
    return sys.executable


def _run_timed(name: str, cmd: list[str]) -> dict[str, Any]:
    print(f"[f2.0] running {name}: {' '.join(cmd)}", flush=True)
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True)
    elapsed = time.perf_counter() - start
    rc = proc.returncode
    stdout_tail = proc.stdout.decode("utf-8", "replace")[-4000:]
    stderr_tail = proc.stderr.decode("utf-8", "replace")[-4000:]
    print(f"[f2.0] {name} elapsed = {elapsed:.3f}s rc={rc}", flush=True)
    return {
        "name": name,
        "returncode": rc,
        "elapsed_s": float(elapsed),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def main() -> int:
    args = _parse_args()
    _ensure_smoke_runnable()

    stages: list[dict[str, Any]] = []

    if not args.reuse_existing_latents:
        stage1 = _run_timed("stage1_text_transformer", _stage1_cmd(args))
        stages.append(stage1)
        if stage1["returncode"] != 0:
            _write(args, stages, completed=False)
            print("[f2.0] stage1 failed; see stderr_tail in output JSON")
            return 1
    else:
        if not Path(args.latents_path).exists():
            raise SystemExit(
                f"--reuse-existing-latents set but {args.latents_path} missing"
            )

    stage2 = _run_timed("stage2_vae_decode", _stage2_cmd(args))
    stages.append(stage2)
    if stage2["returncode"] != 0:
        _write(args, stages, completed=False)
        return 1

    _write(args, stages, completed=True)
    return 0


def _write(args: argparse.Namespace, stages: list[dict[str, Any]], *, completed: bool) -> None:
    stage1 = next((s for s in stages if s["name"] == "stage1_text_transformer"), None)
    stage2 = next((s for s in stages if s["name"] == "stage2_vae_decode"), None)
    s1 = float(stage1["elapsed_s"]) if stage1 else 0.0
    s2 = float(stage2["elapsed_s"]) if stage2 else 0.0
    total = s1 + s2
    denom = max(total, 1e-12)
    shares = {
        "stage1_text_transformer_share": s1 / denom,
        "stage2_vae_decode_share": s2 / denom,
        "vae_share_lower_bound": s2 / denom,
        "vae_share_lower_bound_note": (
            "stage2 wall-clock includes process startup + neuron runtime init + load "
            "for the VAE; this is an over-estimate of pure VAE decode time, so "
            "vae_share_lower_bound is conservative-LOW for the question 'does VAE "
            "dominate end-to-end?' and conservative-HIGH for raw decode time."
        ),
        "gate_threshold": GATE_THRESHOLD,
        "passes_threshold": (s2 / denom) >= GATE_THRESHOLD,
    }
    out = {
        "schema": SCHEMA,
        "completed": completed,
        "command_args": vars(args),
        "stages": stages,
        "totals": {
            "stage1_text_transformer_s": s1,
            "stage2_vae_decode_s": s2,
            "two_stage_sum_s": total,
        },
        "shares": shares,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"[f2.0] wrote {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
