#!/usr/bin/env python3
"""Phase G's CPU fp32 reference for Wan 2.2-T2V-A14B with both experts at the device's short
shape (480x832x9).

diffusers' WanPipeline with every component in fp32 on the host CPU (both experts, umT5 upcast
from its bf16 checkpoint, the VAE), from the device's own fp32 initial noise, with the settings
of the scored device runs (``wan_h3/final_reruns.sh wan22d_bf16`` and ``wan22d_mx``, both
``--dual-expert``); writes ``latents.pt`` (the device's layout), ``out.mp4`` (the device's
writer and fps), ``out.pt`` (the frames, lossless) and ``run.json`` (with the forwards each
expert ran) into ``--out-dir`` for ``score.py --model wan22``. Pin it to one NUMA node:

    DIFFLET_BACKEND=cpu HF_HUB_OFFLINE=1 PYTHONPATH=. taskset -c 48-95,144-191 \\
        python scripts/fp32_ref/wan22.py --out-dir R --threads 96
"""

from __future__ import annotations

import sys

from scripts.fp32_ref import _wan

#: The device runs' settings, from Wan 2.2's object in /home/ubuntu/fp8-bench/phase_g_research.json
#: (``device_settings`` unless named otherwise); only the dtype (fp32) and device (CPU) differ.
SETTINGS = {
    "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    # revision: the only local snapshot; the device log's weight-store paths name it.
    "revision": "5be7df9619b54f4e2667b2755bc6a756675b5cd7",
    "height": 480,
    "width": 832,
    "num_frames": 9,
    "steps": 50,
    "seed": 42,
    "prompt": "a cat walking through a garden",
    # negative_prompt: none given, and guidance 1.0 never encodes one.
    "negative_prompt": None,
    # guidance: the CLI's 1.0 (no --guidance-scale), so no CFG; diffusers' default is 5.0.
    "guidance_scale": 1.0,
    # guidance: no --guidance-scale-2, so the device's low-noise expert runs at guidance_scale
    # too; explicit here because diffusers would otherwise take guidance_scale's value as well.
    "guidance_scale_2": 1.0,
    # other: model_index.json's boundary_ratio 0.875, the boundary timestep 875 on both sides.
    "boundary_ratio": 0.875,
    # other: max_sequence_length / text_seq_len 512.
    "max_sequence_length": 512,
    # other: the snapshot's UniPCMultistepScheduler, flow_shift 3.0, which both sides load.
    "flow_shift": 3.0,
    # latents_layout: (B, C, (9 - 1) // 4 + 1, H / 8, W / 8), the noise and the final latents.
    "noise_shape": (1, 16, 3, 60, 104),
}

#: DiT forwards per expert over the 50 steps (other: steps 0..15 at t >= 875 on the high-noise
#: ``transformer``, 16..49 on the low-noise ``transformer_2``; one forward per step, no CFG).
DIT_FORWARDS = {"transformer": 16, "transformer_2": 34}


def main(argv=None) -> int:
    return _wan.main("wan22", SETTINGS, DIT_FORWARDS, argv)


if __name__ == "__main__":
    sys.exit(main())
