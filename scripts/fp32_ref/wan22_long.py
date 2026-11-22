#!/usr/bin/env python3
"""CPU fp32 reference for Wan 2.2-T2V-A14B (both experts) at the const-max campaign's shape
(480x832x81, 40 steps, CFG 4.0 / 3.0 with the model card's negative prompt): ``wan22.py`` with
the settings of the device runs ``/mnt/nvme/difflet-exp/constmax-models/wan22/{A,B,C,D}``
(bench.sh's ``wan22_official``; their arm script ``scripts/attn_constmax/arms_wan.sh`` is at
c72c0485); the table is docs/plans/2026-11-20-cpu-fp32-reference-long.md. Writes what
``wan22.py`` writes (score a run against it with ``scripts/compare_difflet_runs.py``):

    DIFFLET_BACKEND=cpu HF_HUB_OFFLINE=1 PYTHONPATH=. taskset -c 48-95,144-191 \\
        python scripts/fp32_ref/wan22_long.py --out-dir R --threads 96
"""

from __future__ import annotations

import sys

from scripts.fp32_ref import _wan, wan22

SETTINGS = {
    **wan22.SETTINGS,  # model, revision, 480x832, seed, prompt, boundary 0.875, text 512, shift 3
    "num_frames": 81,
    "steps": 40,
    "guidance_scale": 4.0,  # the high-noise expert's (t >= 875)
    "guidance_scale_2": 3.0,  # the low-noise expert's
    "negative_prompt": _wan.card_negative_prompt("WAN22_NEG"),
    "noise_shape": (1, 16, 21, 60, 104),
    #: md5 of the fp32 seed-42 draw at noise_shape (equal to WanOrchestrator.prepare_latents').
    "noise_md5": "fe29ee8e34bbe84ef350a10d25f1be01",
}

#: Of the snapshot's 40 UniPC timesteps, 13 are >= 875 (the high-noise ``transformer``) and 27
#: below (``transformer_2``); two forwards (cond, uncond) per step.
DIT_FORWARDS = {"transformer": 26, "transformer_2": 54}


def main(argv=None) -> int:
    return _wan.main("wan22_long", SETTINGS, DIT_FORWARDS, argv)


if __name__ == "__main__":
    sys.exit(main())
