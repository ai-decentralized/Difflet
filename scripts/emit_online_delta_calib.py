"""Emit a TeaCache calibration for a model (cclog 91/92).

Two modes:
  * default: force the generic 0-per-model **online_delta** method (self-tuning at
    inference; just needs alpha + shape). Replaces blind fixed-cadence for models whose
    output trajectory is smooth (skip flat steps via the measured noise_pred delta).
  * --from-pairs PATH: read a saved (signal, delta) trajectory and let the unified gate
    AUTO-SELECT adaptive / online_delta / cadence (the cclog-91 decision).

Examples:
  python scripts/emit_online_delta_calib.py --model wan --shape 832x480x13 --num-steps 50
  python scripts/emit_online_delta_calib.py --model qwen_image --shape 1024x1024 \
      --from-pairs cclogs/m9-teacache/pairs_qwen_image_1024_50step.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from difflet.pipeline.teacache import TeaCacheCalibration
from difflet.pipeline.teacache_gate import build_calibration


def _load_pairs(path: str):
    s = json.load(open(path))["samples"]
    byb = defaultdict(list)
    for x in s:
        byb[x.get("bundle", "_")].append(x)
    traj = max(byb.values(), key=len)
    traj = sorted(traj, key=lambda r: r.get("step_index", 0))
    return ([r["mod_input_diff_norm"] for r in traj], [r["noise_pred_diff_norm"] for r in traj])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shape", required=True, help="shape_label, e.g. 832x480x13")
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--cooldown", type=int, default=2)
    ap.add_argument("--from-pairs", default=None, help="saved (signal,delta) trajectory → auto-select")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.from_pairs:
        sig, dlt = _load_pairs(args.from_pairs)
        cal = build_calibration(
            model=args.model, shape_label=args.shape, num_steps=args.num_steps,
            signals=sig, deltas=dlt, warmup=args.warmup, cooldown=args.cooldown, alpha=args.alpha,
        )
        method = "adaptive" if cal.accumulate else "online_delta" if cal.online_delta_alpha > 0 else "cadence"
        print(f"[emit] {args.model}: auto-selected '{method}' from {args.from_pairs}", flush=True)
    else:
        cal = TeaCacheCalibration(
            model=args.model, shape_label=args.shape, num_steps=args.num_steps,
            poly_coef=(0.0,), threshold=0.0,
            warmup_steps=args.warmup, cooldown_steps=args.cooldown,
            online_delta_alpha=args.alpha,
        )
        print(f"[emit] {args.model}: forced online_delta (alpha={args.alpha}) — self-tuning at inference",
              flush=True)

    out = args.out or f"cclogs/m9-teacache/teacache_calib_{args.model}_online.json"
    Path(out).write_text(json.dumps(cal.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"[emit] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
