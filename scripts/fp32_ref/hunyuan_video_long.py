#!/usr/bin/env python3
"""HunyuanVideo's CPU fp32 reference at the const-max campaign's shape (544x960x129):
``hunyuan_video.py`` with the shape, latents layout and noise md5 of the device runs
``/mnt/nvme/difflet-exp/constmax-models/hunyuan_video/{A,B,C,D}``
(arm script ``scripts/attn_constmax/hv1_arms.sh``, at 618365c3) and their prompt. Everything
else (revision, 50 steps, seed 42, embedded guidance 6.0, Llama at 256 tokens, the 256 / 16
decode tiles, 24 fps) is the short script's and equals the device's; the table is
docs/plans/2026-11-20-cpu-fp32-reference-long.md.

    DIFFLET_BACKEND=cpu HF_HUB_OFFLINE=1 PYTHONPATH=. taskset -c 0-47,96-143 \\
        python scripts/fp32_ref/hunyuan_video_long.py --out-dir R --threads 96

``--prompt TEXT`` makes the reference of another prompt at the same settings. It is written to
R_<the first 8 hex digits of the prompt's md5>, never to R, so it cannot replace the pinned
prompt's reference; ``run.json`` records the prompt (``settings.prompt``).
"""

from __future__ import annotations

import argparse
import hashlib
import sys

from scripts.fp32_ref import hunyuan_video as base

#: the arms' --height / --width / --num-frames, the device's latents (B, C, (F - 1) / 4 + 1,
#: H / 8, W / 8) and the md5 of the fp32 seed-42 draw there (difflet's ``_initial_latents`` and
#: diffusers' ``prepare_latents`` give the same tensor).
#: The prompt is pinned too (--prompt's default): the device runs and the 2026-11-21 reference
#: used the short script's prompt of then, which #170 (7859fc26) changed to the bench card's.
LONG = {"HEIGHT": 544, "WIDTH": 960, "NUM_FRAMES": 129, "LATENT_SHAPE": (1, 16, 33, 68, 120),
        "NOISE_MD5": "df925d6d08d0004d0c1c807cb1df6fea",
        "PROMPT": "a cinematic shot of a red fox running through a snowy forest"}


def main(argv=None) -> int:
    # abbreviations stay on: base.main takes --out for --out-dir, which would skip the suffix
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--prompt", default=LONG["PROMPT"])
    p.add_argument("--out-dir")
    args, rest = p.parse_known_args(argv)  # base.main parses (and requires) the rest
    for name, value in {**LONG, "PROMPT": args.prompt}.items():
        setattr(base, name, value)  # base.run() and initial_noise() read these module globals
    if args.out_dir and args.prompt != LONG["PROMPT"]:
        args.out_dir = (f"{args.out_dir.rstrip('/')}_"
                        f"{hashlib.md5(args.prompt.encode()).hexdigest()[:8]}")
        print(f"[fp32-ref hunyuan_video_long] --prompt: writing to {args.out_dir}", flush=True)
    return base.main(rest + (["--out-dir", args.out_dir] if args.out_dir else []))


if __name__ == "__main__":
    sys.exit(main())
