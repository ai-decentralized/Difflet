#!/usr/bin/env python3
"""Score one model's device bf16 and shipped-MXFP8 outputs against its CPU fp32 reference.

The reference directory is what ``scripts/fp32_ref/<model>.py`` wrote through ``save_run``
(``latents.pt``, ``out.png`` or ``out.mp4``); the device files are ``DEVICE_OUTPUTS`` (phase G's
plan table, from ``/home/ubuntu/fp8-bench/phase_g_research.json``). Each arm is compared with
``scripts/compare_difflet_runs.py``: PSNR / SSIM / LPIPS on the decoded output, rel L2 / cosine
on the final latents where the device kept them. MXFP8 is read relative to bf16's own distance
from fp32, hence the last column.

    python scripts/fp32_ref/score.py --model wan21 --ref-dir /path/to/ref

writes ``<ref-dir>/score.json`` and prints
``model | bf16 vs fp32 psnr/ssim/lpips/cos | mx vs fp32 psnr/ssim/lpips/cos | mx - bf16 (dB)``.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = Path("/home/ubuntu/fp8-bench")

#: The device files scored, relative to BENCH, per arm: the decoded output and the final
#: latents.pt (None where the run kept none).
DEVICE_OUTPUTS = {
    # tp4 decodes in process, so neither arm wrote latents.pt (research: bf16_latents).
    "flux": {
        "bf16": {"output": "flux_d_tp4/A1_gena.png", "latents": None},
        "mx": {"output": "flux_d_tp4/B1_gena.png", "latents": None},
    },
    # qwen_bf16_tp4_ref2, not qwen_bf16_tp4: the latter predates the text-mask change c9f4009.
    "qwen_image": {
        "bf16": {"output": "qwen_bf16_tp4_ref2/gen50a.png",
                 "latents": "qwen_bf16_tp4_ref2/work_gen50a/latents.pt"},
        "mx": {"output": "qwen_mx_hm0_tp4/gen50a.png",
               "latents": "qwen_mx_hm0_tp4/work_gen50a/latents.pt"},
    },
    "ltx_2": {
        "bf16": {"output": "ltx2_tp4/bf16_gen50a.mp4",
                 "latents": "ltx2_tp4/work_bf16_gen50a/latents.pt"},
        "mx": {"output": "ltx2_tp4/mx_gen50a.mp4", "latents": "ltx2_tp4/work_mx_gen50a/latents.pt"},
    },
    # bf16 at c7c2346 (stride-2 rotary); MXFP8 rerun at 90c648f's default.
    "wan21": {
        "bf16": {"output": "wan_h3/h5/bf16_gen50a.mp4",
                 "latents": "wan_h3/h5/work_bf16_gen50a/latents.pt"},
        "mx": {"output": "wan_h3/final/wan21_mx/gen50a.mp4",
               "latents": "wan_h3/final/wan21_mx/work_a/latents.pt"},
    },
    # Both experts, both arms rerun at 90c648f (the 1a8382f runs predate c7c2346's rotary).
    "wan22": {
        "bf16": {"output": "wan_h3/final/wan22d_bf16/gen50a.mp4",
                 "latents": "wan_h3/final/wan22d_bf16/work_a/latents.pt"},
        "mx": {"output": "wan_h3/final/wan22d_mx/gen50a.mp4",
               "latents": "wan_h3/final/wan22d_mx/work_a/latents.pt"},
    },
    # MXFP8 arm V15 = fc38fb6's default set.
    "hunyuan_video": {
        "bf16": {"output": "hv1_tp4/bf16_gen50a.mp4",
                 "latents": "hv1_tp4/work_bf16_gen50a/latents.pt"},
        "mx": {"output": "hv1_tp4/loo_V15_gen50a.mp4",
               "latents": "hv1_tp4/work_loo_V15_gen50a/latents.pt"},
    },
    # 9 frames; MXFP8 arm u3 = 9f6aedc's default set.
    "hunyuan_video_15": {
        "bf16": {"output": "hv15/bf16_9/gena.mp4", "latents": "hv15/bf16_9/work_a/latents.pt"},
        "mx": {"output": "hv15/u3/gena.mp4", "latents": "hv15/u3/work_a/latents.pt"},
    },
}


@functools.cache
def _cdr():
    spec = importlib.util.spec_from_file_location(
        "compare_difflet_runs", REPO / "scripts" / "compare_difflet_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_arm(arm: dict, ref_dir: Path) -> dict:
    """compare_difflet_runs' metrics of one device arm against the reference (reference first):
    the decoded output always, the latents when the device kept them."""
    output = BENCH / arm["output"]
    latents = BENCH / arm["latents"] if arm["latents"] else None
    ref_latents = ref_dir / "latents.pt" if latents else None
    return _cdr().compare_runs(ref_latents, latents, ref_dir / f"out{output.suffix}", output)


def _decoded(result: dict) -> dict:
    return result.get("image") or result["video"]


def _cosine(latents: dict | None) -> str:
    """The latents cosine, one per tensor for a dict (LTX-2's video and audio)."""
    if latents is None:
        return "-"
    if "cosine" in latents:
        return f"{latents['cosine']:.4f}"
    return ",".join(f"{key}={_cosine(value)}" for key, value in latents.items())


def _cell(result: dict) -> str:
    decoded = _decoded(result)
    lpips = "-" if decoded["lpips"] is None else f"{decoded['lpips']:.3f}"
    return (f"{decoded['psnr']:.2f}/{decoded['ssim']:.3f}/{lpips}/"
            f"{_cosine(result.get('latents'))}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(DEVICE_OUTPUTS))
    p.add_argument("--ref-dir", required=True, type=Path)
    args = p.parse_args(argv)
    arms = DEVICE_OUTPUTS[args.model]
    result = {"model": args.model, "ref_dir": str(args.ref_dir)}
    for name in ("bf16", "mx"):
        result[name] = score_arm(arms[name], args.ref_dir)
    result["mx_minus_bf16_db"] = _decoded(result["mx"])["psnr"] - _decoded(result["bf16"])["psnr"]
    (args.ref_dir / "score.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"{args.model} | bf16 vs fp32 {_cell(result['bf16'])} | mx vs fp32 {_cell(result['mx'])}"
          f" | mx - bf16 {result['mx_minus_bf16_db']:+.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
