"""Benchmark matrix: the "best performing version" config for each supported model.

Each entry encodes the knobs that currently give the best end-to-end performance
on the target accelerator. Shapes are chosen to fit a single trn2.3xlarge (1 Neuron
device, 4 cores x 24 GB); ``tp=4`` is the max on that box (FLUX's registry default is
tp=8, overridden to 4 here). Adjust ``shape``/``tp``/``steps`` to retune.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# The Neuron inference venv all Trainium runs use.
NXD_VENV = "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"


@dataclass
class BenchConfig:
    model_id: str
    model_type: str
    tp: int = 4
    cp: int = 1
    dtype: str = "bf16"
    height: Optional[int] = None
    width: Optional[int] = None
    num_frames: Optional[int] = None
    steps: int = 20
    guidance_scale: Optional[float] = None
    prompt: str = "a cinematic shot of a red fox running through a snowy forest"
    output_kind: str = "video"               # video | image
    extra_generate_flags: list[str] = field(default_factory=list)
    config_label: str = ""                   # human description of the best-perf knobs
    notes: str = ""

    def shape_flags(self) -> list[str]:
        f: list[str] = []
        if self.height is not None:
            f += ["--height", str(self.height)]
        if self.width is not None:
            f += ["--width", str(self.width)]
        if self.num_frames is not None:
            f += ["--num-frames", str(self.num_frames)]
        return f


# Keyed by a short slug used for the report filename (benchmark/<slug>.md).
MATRIX: dict[str, BenchConfig] = {
    "ltx_2": BenchConfig(
        model_id="Lightricks/LTX-2",
        model_type="ltx_2",
        tp=4, height=480, width=704, num_frames=49, steps=20, guidance_scale=1.0,
        output_kind="video",
        config_label="tp=4, bf16, TP-sharded transformer + attention_cte self-attn, "
                     "guidance=1.0 (batch-1 NEFF)",
        notes="Default registry shape 512x768x121 also compiles; 480x704x49 used here "
              "as the representative fast shape. CFG (guidance>1) needs a batch-2 NEFF.",
    ),
    "wan_2_1": BenchConfig(
        model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        model_type="wan",
        tp=4, height=480, width=832, num_frames=9, steps=20, guidance_scale=1.0,
        output_kind="video",
        config_label="tp=4, bf16, single-transformer (no MoE), attention_cte, 2-stage "
                     "(transformer + VAE) subprocess pipeline",
    ),
    "wan_2_2": BenchConfig(
        model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        model_type="wan",
        tp=4, height=480, width=832, num_frames=9, steps=20, guidance_scale=1.0,
        output_kind="video",
        config_label="tp=4, bf16, A14B (high/low-noise experts), attention_cte",
    ),
    "flux_1_dev": BenchConfig(
        model_id="black-forest-labs/FLUX.1-dev",
        model_type="flux",
        tp=4, height=1024, width=1024, num_frames=None, steps=28, guidance_scale=3.5,
        output_kind="image",
        config_label="tp=4 (registry default tp=8 -> 4 on trn2.3xlarge), bf16, attention_cte",
    ),
    "qwen_image": BenchConfig(
        model_id="Qwen/Qwen-Image",
        model_type="qwen_image",
        tp=4, height=1024, width=1024, num_frames=None, steps=20, guidance_scale=4.0,
        output_kind="image",
        config_label="tp=4, bf16, joint attention via attention_cte",
    ),
    "hunyuan_video": BenchConfig(
        model_id="hunyuanvideo-community/HunyuanVideo",
        model_type="hunyuan_video",
        tp=4, height=320, width=512, num_frames=61, steps=20, guidance_scale=6.0,
        output_kind="video",
        config_label="tp=4, bf16, attention_cte",
    ),
    "hunyuan_video_15": BenchConfig(
        model_id="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        model_type="hunyuan_video_15",
        tp=4, height=480, width=848, num_frames=121, steps=20, guidance_scale=6.0,
        output_kind="video",
        config_label="tp=4, bf16, attention_cte + MX precision ops",
    ),
}
