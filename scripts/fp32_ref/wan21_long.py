#!/usr/bin/env python3
"""CPU fp32 reference for Wan 2.1-T2V-14B at the const-max campaign's shape (480x832x81, CFG 5.0
with the model card's negative prompt): ``wan21.py`` with the settings of the device runs
``/mnt/nvme/difflet-exp/constmax-models/wan21/{A,B,C,D}`` (bench.sh's ``wan21_official``; their
arm script ``scripts/attn_constmax/arms_wan.sh`` is at 618365c3); the table is
docs/plans/2026-11-20-cpu-fp32-reference-long.md. Writes what ``wan21.py`` writes (score a run
against it with ``scripts/compare_difflet_runs.py``):

    DIFFLET_BACKEND=cpu HF_HUB_OFFLINE=1 PYTHONPATH=. taskset -c 48-95,144-191 \\
        python scripts/fp32_ref/wan21_long.py --out-dir R --threads 96
"""

from __future__ import annotations

import sys

from scripts.fp32_ref import _wan, wan21

SETTINGS = {
    **wan21.SETTINGS,  # model, revision, 480x832, 50 steps, seed 42, prompt, text 512, flow shift 3
    "num_frames": 81,
    "guidance_scale": 5.0,
    "negative_prompt": _wan.card_negative_prompt("WAN21_NEG"),
    "noise_shape": (1, 16, 21, 60, 104),
    #: md5 of the fp32 seed-42 draw at noise_shape (equal to WanOrchestrator.prepare_latents').
    "noise_md5": "fe29ee8e34bbe84ef350a10d25f1be01",
}

#: One expert, CFG: a cond and an uncond forward per step.
DIT_FORWARDS = {"transformer": 2 * SETTINGS["steps"]}


def main(argv=None) -> int:
    return _wan.main("wan21_long", SETTINGS, DIT_FORWARDS, argv)


if __name__ == "__main__":
    sys.exit(main())
