from __future__ import annotations

import argparse
import sys

VALID_MODELS = {
    "black-forest-labs/FLUX.1-dev",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "hunyuanvideo-community/HunyuanVideo",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    "Qwen/Qwen-Image",
    "Lightricks/LTX-2",
}

_MODEL_TYPE: dict[str, str] = {
    "black-forest-labs/FLUX.1-dev": "flux",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "wan",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": "wan",
    "hunyuanvideo-community/HunyuanVideo": "hunyuan_video",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v": "hunyuan_video_15",
    "Qwen/Qwen-Image": "qwen_image",
    "Lightricks/LTX-2": "ltx_2",
}


def _add_model_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model-id",
        required=True,
        dest="model_id",
        help="HuggingFace model ID. One of:\n  " + "\n  ".join(sorted(VALID_MODELS)),
    )


def _add_parallel_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tp-degree", type=int, default=None,
                   help="Tensor-parallel degree (default: registry default)")
    p.add_argument("--cp-degree", type=int, default=1,
                   help="Context-parallel degree (default: 1)")
    p.add_argument("--cp-mode", choices=["gather_kv", "ring"], default="gather_kv",
                   help="Context-parallel attention strategy (default: gather_kv)")


def _add_shape_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)


def _add_cache_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cache-dir", default=None,
                   help="Compiled artifact cache root (default: ~/.cache/difflet/)")
    p.add_argument("--force", action="store_true",
                   help="Recompile even if a valid cache entry exists")


def _add_generate_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", required=True, help="Output file path (.png or .mp4)")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work-dir", default=None,
                   help="Directory for inter-stage tensors (staged models only)")
    p.add_argument("--keep-work-dir", action="store_true",
                   help="Do not delete work-dir after successful generation")
    p.add_argument("--teacache-cadence", type=int, default=None,
                   metavar="N", help="Skip every N-th DiT step (fixed cadence, no calibration)")
    p.add_argument("--teacache-online-delta", type=float, default=None,
                   metavar="ALPHA", help="Online-delta TeaCache alpha (no calibration)")
    p.add_argument("--teacache-speedup", type=float, default=None,
                   metavar="X", help="Adaptive TeaCache target speedup (requires --teacache-calibration)")
    p.add_argument("--teacache-calibration", default=None,
                   metavar="PATH", help="Path to TeaCache calibration JSON")


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="difflet",
                                   description="Difflet — diffusion inference on Trainium")
    sub = root.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Download model weights from HuggingFace")
    _add_model_flag(dl)
    dl.add_argument("--revision", default=None)

    cp_cmd = sub.add_parser("compile", help="AOT-compile model NEFFs and cache on disk")
    _add_model_flag(cp_cmd)
    cp_cmd.add_argument("--revision", default=None)
    _add_parallel_flags(cp_cmd)
    _add_shape_flags(cp_cmd)
    _add_cache_flags(cp_cmd)

    gen = sub.add_parser("generate", help="Run inference (requires prior compile)")
    _add_model_flag(gen)
    gen.add_argument("--revision", default=None)
    _add_parallel_flags(gen)
    _add_shape_flags(gen)
    _add_cache_flags(gen)
    _add_generate_flags(gen)

    run_cmd = sub.add_parser("run", help="Download + compile + generate in one shot")
    _add_model_flag(run_cmd)
    run_cmd.add_argument("--revision", default=None)
    _add_parallel_flags(run_cmd)
    _add_shape_flags(run_cmd)
    _add_cache_flags(run_cmd)
    _add_generate_flags(run_cmd)

    return root


def _validate_teacache(args: argparse.Namespace) -> None:
    cadence = getattr(args, "teacache_cadence", None)
    online = getattr(args, "teacache_online_delta", None)
    speedup = getattr(args, "teacache_speedup", None)
    calib = getattr(args, "teacache_calibration", None)

    active = [
        ("--teacache-cadence", cadence is not None),
        ("--teacache-online-delta", online is not None),
        ("--teacache-speedup", speedup is not None),
    ]
    active_names = [name for name, on in active if on]
    if len(active_names) > 1:
        print(f"Error: {active_names[0]} and {active_names[1]} are mutually exclusive.",
              file=sys.stderr)
        raise SystemExit(1)
    if speedup is not None and calib is None:
        print("Error: --teacache-speedup requires --teacache-calibration PATH.",
              file=sys.stderr)
        raise SystemExit(1)


def _get_orchestrator(args: argparse.Namespace):
    from difflet.cli.orchestrators.flux import FluxOrchestrator
    from difflet.cli.orchestrators.ltx_2 import LTX2Orchestrator
    from difflet.cli.orchestrators.wan import WanOrchestrator
    from difflet.cli.orchestrators.hunyuan_video import HunyuanVideoOrchestrator
    from difflet.cli.orchestrators.hunyuan_video_15 import HunyuanVideo15Orchestrator
    from difflet.cli.orchestrators.qwen_image import QwenImageOrchestrator

    mapping = {
        "black-forest-labs/FLUX.1-dev": FluxOrchestrator,
        "Lightricks/LTX-2": LTX2Orchestrator,
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers": WanOrchestrator,
        "Wan-AI/Wan2.1-T2V-14B-Diffusers": WanOrchestrator,
        "hunyuanvideo-community/HunyuanVideo": HunyuanVideoOrchestrator,
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v": HunyuanVideo15Orchestrator,
        "Qwen/Qwen-Image": QwenImageOrchestrator,
    }
    return mapping[args.model_id](args)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.model_id not in VALID_MODELS:
        print(
            f"Error: Unknown model-id '{args.model_id}'. Valid model IDs:\n"
            + "\n".join(f"  {m}" for m in sorted(VALID_MODELS)),
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.command in ("generate", "run"):
        _validate_teacache(args)

    orchestrator = _get_orchestrator(args)
    getattr(orchestrator, args.command)()


if __name__ == "__main__":
    main()
