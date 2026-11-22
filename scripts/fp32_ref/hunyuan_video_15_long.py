#!/usr/bin/env python3
"""HunyuanVideo-1.5's CPU fp32 reference at the const-max campaign's shape (480x848x121):
``hunyuan_video_15.py`` with the frames, latents layout and noise md5 of the device runs
``/mnt/nvme/difflet-exp/constmax-models/hunyuan_video_15/{A,B,C,D}``
(arm script ``scripts/attn_constmax/hv15_arms.sh``, at c72c0485). Everything else (revision,
480x848, 50 steps, seed 42, the prompt, true CFG 6.0 against "", the 128-pixel decode tiles,
24 fps) is the short script's and equals the device's; the table is
docs/plans/2026-11-20-cpu-fp32-reference-long.md.

    DIFFLET_BACKEND=cpu HF_HUB_OFFLINE=1 PYTHONPATH=. taskset -c 48-95,144-191 \\
        python scripts/fp32_ref/hunyuan_video_15_long.py --out-dir R --threads 96
"""

from __future__ import annotations

import sys

from scripts.fp32_ref import hunyuan_video_15 as base

#: the arms' --num-frames 121, the device's latents (B, 32, (F - 1) / 4 + 1, H / 16, W / 16)
#: and the md5 of its bf16 seed-42 draw there (hashed as bf16; equal to pipeline15's draw, to
#: diffusers' bf16 ``prepare_latents`` and to the fp32 draw rounded to bf16).
LONG = {"NUM_FRAMES": 121, "LATENT_SHAPE": (1, 32, 31, 30, 53),
        "NOISE_MD5": "56869f4e18d40091e5b7d3ad0c0acb63"}


def main(argv=None) -> int:
    for name, value in LONG.items():
        setattr(base, name, value)  # base.run() and initial_noise() read these module globals
    return base.main(argv)


if __name__ == "__main__":
    sys.exit(main())
