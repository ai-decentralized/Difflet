"""The quality side of the before/after bench (evidence sections 11 and 12).

Per key under ROOT:
  * determinism: md5 of the decoded outputs of cold_1, cold_2 and both warm requests, per arm
    (equal: the four runs produced one sample, so one quality number covers them all); an
    output whose step's log does not end in ``exit 0`` counts as missing;
  * provenance, for the keys phase G scored: the same md5 against the campaign's scored device
    output (scripts/fp32_ref/score.py DEVICE_OUTPUTS, under its BENCH), where that host has it;
  * MXFP8 against bf16 on the device: scripts/compare_difflet_runs.py on cold_1, bf16 first
    (PSNR mean and worst frame, SSIM, LPIPS);
  * with ``--floor key=<file>`` (the key's bf16 output at another TP degree): the floor against
    bf16 the same way, and the campaign's floor-relative screen on MXFP8 against bf16:
    PSNR >= min(30, floor - 4), SSIM >= min(0.97, floor - 0.10), LPIPS <= max(0.05, 2 x floor).
Writes ROOT/quality.json and prints one line per key. An empty root gives no rows.

usage: quality.py [--root ROOT] [--floor key=FILE ...] [key ...]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KEYS = ["flux", "qwen_image", "ltx_2", "wan21", "wan22", "hunyuan_video", "hunyuan_video_15",
        "wan21_official", "wan22_official", "hv15_official"]
IMAGE_KEYS = {"flux", "qwen_image"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def md5(path: Path | None) -> str | None:
    """The md5 of a file's bytes (the first 8 hex digits), None when there is no file."""
    if path is None or not path.exists():
        return None
    return hashlib.md5(path.read_bytes()).hexdigest()[:8]


def step_ok(arm: Path, step: str) -> bool:
    """True when the step's log (``cold_1.log``; ``warm.log`` for a warm request) ends in
    ``exit 0``: an output another run left behind, or a failed step's partial one, is not
    the arm's."""
    log = arm / f"{'warm' if step.startswith('warm') else step}.log"
    exits = [ln for ln in log.read_text(errors="replace").splitlines()
             if ln.startswith("exit ")] if log.exists() else []
    return bool(exits) and exits[-1].strip() == "exit 0"


def decoded(result: dict | None) -> dict:
    """compare_runs' decoded-output metrics (under ``image`` or ``video``)."""
    result = result or {}
    return result.get("image") or result.get("video") or {}


def screen(mx: dict, floor: dict) -> dict:
    """The campaign's floor-relative screen on MXFP8 against bf16, given the bf16 floor."""
    psnr_min = min(30.0, floor["psnr"] - 4)
    ssim_min = min(0.97, floor["ssim"] - 0.10)
    lpips_max = max(0.05, 2 * floor["lpips"]) if floor.get("lpips") is not None else None
    checks = {"psnr": mx["psnr"] >= psnr_min, "ssim": mx["ssim"] >= ssim_min}
    if lpips_max is not None and mx.get("lpips") is not None:
        checks["lpips"] = mx["lpips"] <= lpips_max
    return {"psnr_min": psnr_min, "ssim_min": ssim_min, "lpips_max": lpips_max,
            "checks": checks, "pass": all(checks.values())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path,
                   default=Path(os.environ.get("BENCH_ROOT", "/mnt/nvme/difflet-bench/results")))
    p.add_argument("--floor", nargs="*", default=[], metavar="KEY=FILE",
                   help="the key's bf16 output at another TP degree")
    p.add_argument("keys", nargs="*")
    args = p.parse_args()
    floors = {k: Path(v) for k, v in (item.split("=", 1) for item in args.floor)}
    keys = args.keys or [k for k in KEYS if (args.root / k).is_dir()]
    if not keys:
        print("no results under", args.root)
        if args.root.is_dir():
            (args.root / "quality.json").write_text("{}\n")
        return 0
    cdr = _load("compare_difflet_runs", REPO / "scripts" / "compare_difflet_runs.py")
    score = _load("fp32_score", REPO / "scripts" / "fp32_ref" / "score.py")
    out = {}
    for key in keys:
        if not (args.root / key).is_dir():
            continue
        res: dict = {}
        ext = "png" if key in IMAGE_KEYS else "mp4"
        for arm in ("bf16", "mx"):
            d = args.root / key / arm
            hashes = {k: md5(d / f"{k}.{ext}") if step_ok(d, k) else None
                      for k in ("cold_1", "cold_2", "warm_1", "warm_2")}
            camp = (score.DEVICE_OUTPUTS.get(key) or {}).get(arm, {}).get("output")
            res[arm] = {"md5": hashes, "campaign_md5": md5(score.BENCH / camp) if camp else None,
                        "deterministic": all(hashes.values()) and len(set(hashes.values())) == 1}
            res[arm]["matches_campaign"] = (res[arm]["campaign_md5"] is not None
                                            and res[arm]["campaign_md5"] == hashes["cold_1"])
        bf16, mx = args.root / key / "bf16" / f"cold_1.{ext}", args.root / key / "mx" / f"cold_1.{ext}"
        ok = step_ok(bf16.parent, "cold_1") and step_ok(mx.parent, "cold_1")
        if ok and bf16.exists() and mx.exists():
            res["mx_vs_bf16"] = cdr.compare_runs(None, None, bf16, mx)
        if key in floors and bf16.exists():
            res["floor"] = str(floors[key])
            res["floor_vs_bf16"] = cdr.compare_runs(None, None, bf16, floors[key])
            if res.get("mx_vs_bf16"):
                res["screen"] = screen(decoded(res["mx_vs_bf16"]), decoded(res["floor_vs_bf16"]))
        out[key] = res
        dec, fl = decoded(res.get("mx_vs_bf16")), decoded(res.get("floor_vs_bf16"))
        line = (f"{key}: bf16 md5 {res['bf16']['md5']['cold_1']} det={res['bf16']['deterministic']}"
                f" campaign={res['bf16']['matches_campaign']} | mx md5 {res['mx']['md5']['cold_1']}"
                f" det={res['mx']['deterministic']} campaign={res['mx']['matches_campaign']} | mx vs"
                f" bf16 psnr {dec.get('psnr')} (worst {dec.get('psnr_worst')}) ssim {dec.get('ssim')}"
                f" lpips {dec.get('lpips')}")
        if fl:
            line += (f" | floor psnr {fl.get('psnr')} ssim {fl.get('ssim')} lpips {fl.get('lpips')}"
                     f" screen {'pass' if res.get('screen', {}).get('pass') else 'FAIL'}")
        print(line, flush=True)
    (args.root / "quality.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
