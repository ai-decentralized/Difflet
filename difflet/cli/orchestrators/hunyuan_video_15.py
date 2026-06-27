"""HunyuanVideo 1.5 T2V orchestrator — scaffold only.

Stage internals (text encoding, DiT generate) are not yet implemented.
HunyuanVideo 1.5 requires Qwen2.5-VL, ByT5 glyph, and image-semantic embeddings
(HunyuanVideo15DiTInputBundle) rather than the CLIP + Llama3 pipeline used by 1.0.
CP is not supported (tp=4 only).

Default HF model: hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
Registry defaults: height=480, width=848, num_frames=121, tp_degree=4
"""
from __future__ import annotations

import argparse
import sys

from difflet.cli.orchestrators.base import ModelOrchestrator

_HF_MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
_MODEL_TYPE = "hunyuan_video_15"
_CLI_NAME = "hunyuan-video-1.5"


class HunyuanVideo15Orchestrator(ModelOrchestrator):

    def download(self) -> None:
        from difflet.pipeline.path_resolver import resolve_model_path
        resolve_model_path(_HF_MODEL_ID, local_files_only=False)
        print(f"[difflet] weights ready for {_HF_MODEL_ID}")

    def compile(self) -> None:
        raise NotImplementedError(
            f"difflet compile --model-id {_HF_MODEL_ID} is not yet implemented.\n"
            "HunyuanVideo 1.5 requires Qwen2.5-VL, ByT5 glyph, and image-semantic\n"
            "text encoders (HunyuanVideo15DiTInputBundle) — stage logic TBD."
        )

    def generate(self) -> None:
        raise NotImplementedError(
            f"difflet generate --model-id {_HF_MODEL_ID} is not yet implemented.\n"
            "HunyuanVideo 1.5 requires Qwen2.5-VL, ByT5 glyph, and image-semantic\n"
            "text encoders (HunyuanVideo15DiTInputBundle) — stage logic TBD."
        )

    # ------------------------------------------------------------ helpers

    def _shared_cli_args(self, stage_mode: str, work_dir: str | None = None) -> list[str]:
        """Shared CLI args for HunyuanVideo 1.5 stages (forward-looking; stages TBD)."""
        a = self.args
        parts = [
            "--model-id", _HF_MODEL_ID,
            "--tp-degree", str(a.tp_degree or 4),
            "--cp-degree", str(a.cp_degree or 1),
            "--cp-mode", str(getattr(a, "cp_mode", "gather_kv")),
            "--height", str(a.height or 480),
            "--width", str(a.width or 848),
            "--num-frames", str(a.num_frames or 121),
            "--steps", str(getattr(a, "steps", None) or 4),
            "--guidance-scale", str(getattr(a, "guidance_scale", None) or 6.0),
            "--seed", str(getattr(a, "seed", 42)),
            "--stage-mode", stage_mode,
        ]
        if getattr(a, "prompt", None):
            parts += ["--prompt", a.prompt]
        if getattr(a, "output", None):
            parts += ["--output", a.output]
        if a.cache_dir:
            parts += ["--cache-dir", a.cache_dir]
        if work_dir:
            parts += ["--work-dir", work_dir]
        return parts
