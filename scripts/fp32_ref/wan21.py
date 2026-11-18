#!/usr/bin/env python3
"""Phase G's CPU fp32 reference for Wan 2.1-T2V-14B at the device's short shape (480x832x9).

diffusers' WanPipeline with every component in fp32 on the host CPU, from the device's own
fp32 initial noise, with the settings of the scored device runs (bf16:
``wan_h3/verify_h5.sh``; shipped MXFP8: ``wan_h3/final_reruns.sh wan21_mx``); writes
``latents.pt`` (the device's layout), ``out.mp4`` (the device's writer and fps), ``out.pt`` (the
frames, lossless) and ``run.json`` into ``--out-dir`` for ``score.py --model wan21``. Pin it to
one NUMA node:

    DIFFLET_BACKEND=cpu HF_HUB_OFFLINE=1 PYTHONPATH=. taskset -c 48-95,144-191 \\
        python scripts/fp32_ref/wan21.py --out-dir R --threads 96
"""

from __future__ import annotations

import sys

from scripts.fp32_ref import _wan

#: The device runs' settings, from Wan 2.1's object in /home/ubuntu/fp8-bench/phase_g_research.json
#: (``device_settings`` unless named otherwise); only the dtype (fp32) and device (CPU) differ.
SETTINGS = {
    "model_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    # revision: refs/main, which the device resolved (it passed no --revision); the h5 log's
    # weight-store keys name it.
    "revision": "38ec498cb3208fb688890f8cc7e94ede2cbd7f68",
    "height": 480,
    "width": 832,
    "num_frames": 9,
    "steps": 50,
    "seed": 42,
    "prompt": "a cat walking through a garden",
    # negative_prompt: the neuron request never passes one, and guidance 1.0 never encodes it.
    "negative_prompt": None,
    # guidance: the CLI's 1.0 (no --guidance-scale), so no CFG; diffusers' default is 5.0.
    "guidance_scale": 1.0,
    # guidance: none for a second expert (no --dual-expert; the snapshot has no boundary_ratio).
    "guidance_scale_2": None,
    # model: model_index.json has no boundary_ratio, a single-expert model.
    "boundary_ratio": None,
    # other: text_seq_len 512 (the CLI's _TORCHRUN_APPLICATION_KWARGS), padded to max_length.
    "max_sequence_length": 512,
    # other: the snapshot's UniPCMultistepScheduler, flow_shift 3.0, which both sides load.
    "flow_shift": 3.0,
    # latents_layout: (B, C, (9 - 1) // 4 + 1, H / 8, W / 8), the noise and the final latents.
    "noise_shape": (1, 16, 3, 60, 104),
}

#: DiT forwards per expert over the 50 steps: one expert, one forward per step (no CFG).
DIT_FORWARDS = {"transformer": 50}


def main(argv=None) -> int:
    return _wan.main("wan21", SETTINGS, DIT_FORWARDS, argv)


if __name__ == "__main__":
    sys.exit(main())
