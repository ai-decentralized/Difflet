"""Layer 1 of the TPU profile: where does one warm request's time go?

Reads the per-model JSON that ``benchmark.bench --backend tpu`` already writes
(``stage_seconds`` = text_encode / denoise / vae_decode, ``step_latency``) and
prints one markdown row per run. No device needed -- this answers "should we be
looking at the DiT at all" from runs that already exist.

    python -m benchmark.tpu_profile.stage_table runs/*/wan_2_1.json runs/*/qwen_image.json

The MFU column is filled only where the per-forward FLOPs are known: pass
``--flops wan_2_1=133.5`` (TFLOP per DiT forward) and, for true-CFG models,
``--forwards qwen_image=2`` (forwards per step). Wan's 133.5 is the hand count
in benchmark/v5e/notes/wan_2_2.md at 480x832x9 -- it is shape-specific.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: v5e bf16 peak per chip.
PEAK_FLOPS = 197e12

KNOWN_TFLOP_PER_FORWARD = {"wan_2_1": 133.5, "wan_2_2": 133.5}


def _pairs(items: list[str], kind):
    out = {}
    for item in items:
        name, _, value = item.partition("=")
        out[name] = kind(value)
    return out


def row(path: Path, flops: dict[str, float], forwards: dict[str, int]) -> str:
    d = json.loads(path.read_text())
    slug = d["config_slug"]
    stages = d.get("stage_seconds") or {}
    total = sum(stages.values()) or 1.0
    step = (d.get("step_latency") or {}).get("mean") or 0.0
    runtime = "torch_xla" if d["toolchain"].get("torch-xla") not in (None, "absent") else "torch_tpu"
    chips = int(d["parallel"]["tp_degree"]) * int(d["parallel"].get("cp_degree") or 1)
    mfu = ""
    tflop = flops.get(slug, KNOWN_TFLOP_PER_FORWARD.get(slug))
    if tflop and step:
        mfu = f"{tflop * 1e12 * forwards.get(slug, 1) / step / (chips * PEAK_FLOPS):.0%}"
    cells = [f"{stages.get(k, 0.0):.2f} s ({stages.get(k, 0.0) / total:.0%})"
             for k in ("text_encode", "denoise", "vae_decode")]
    return (f"| {slug} | {runtime} | {path.parent.name} | {step * 1000:.0f} ms | {mfu} | "
            + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--flops", nargs="*", default=[], help="slug=TFLOP per DiT forward")
    parser.add_argument("--forwards", nargs="*", default=[], help="slug=forwards per step")
    args = parser.parse_args()
    flops, forwards = _pairs(args.flops, float), _pairs(args.forwards, int)
    print("| model | runtime | run | step | denoise MFU | text encode | denoise | VAE decode |")
    print("|---|---|---|---|---|---|---|---|")
    for path in args.runs:
        print(row(path, flops, forwards))


if __name__ == "__main__":
    main()
