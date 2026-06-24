"""Internal subprocess dispatcher — not a user-facing entry point.

Called by runner.run_stage() as:
    python -m difflet.cli.stage --orchestrator <hf-id> --stage <stage> [forwarded args]
"""
from __future__ import annotations

import argparse
import importlib
import sys

_ORCHESTRATOR_MAP: dict[str, str] = {
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "difflet.cli.orchestrators.wan.WanOrchestrator",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": "difflet.cli.orchestrators.wan.WanOrchestrator",
    "hunyuanvideo-community/HunyuanVideo": (
        "difflet.cli.orchestrators.hunyuan_video.HunyuanVideoOrchestrator"
    ),
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v": (
        "difflet.cli.orchestrators.hunyuan_video_15.HunyuanVideo15Orchestrator"
    ),
    "Qwen/Qwen-Image": "difflet.cli.orchestrators.qwen_image.QwenImageOrchestrator",
}


def _load_orchestrator_class(name: str):
    if name not in _ORCHESTRATOR_MAP:
        sys.exit(f"[difflet.cli.stage] unknown orchestrator: {name!r}")
    cls_path = _ORCHESTRATOR_MAP[name]
    module_path, cls_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _build_stage_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="difflet.cli.stage", add_help=False)
    p.add_argument("--orchestrator", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--stage-mode", default="generate", choices=["compile", "generate"])
    p.add_argument("--model-id", dest="model_id", default=None)
    p.add_argument("--tp-degree", type=int, default=None)
    p.add_argument("--cp-degree", type=int, default=1)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--teacache-cadence", type=int, default=None)
    p.add_argument("--teacache-online-delta", type=float, default=None)
    p.add_argument("--teacache-speedup", type=float, default=None)
    p.add_argument("--teacache-calibration", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_stage_parser()
    args, _ = parser.parse_known_args(argv)
    cls = _load_orchestrator_class(args.orchestrator)
    orchestrator = cls(args)
    orchestrator._run_stage_internal(args.stage, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
