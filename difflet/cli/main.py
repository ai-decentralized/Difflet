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

SERVE_VALID_MODELS = {
    "black-forest-labs/FLUX.1-dev",
    "Qwen/Qwen-Image",
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


def _add_serve_model_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model-id",
        required=True,
        dest="model_id",
        help="HuggingFace model ID. One of:\n  " + "\n  ".join(sorted(SERVE_VALID_MODELS)),
    )


def _add_parallel_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--tp-degree",
        type=int,
        default=None,
        help="Tensor-parallel degree (default: registry default)",
    )
    p.add_argument("--cp-degree", type=int, default=1, help="Context-parallel degree (default: 1)")
    p.add_argument(
        "--cp-mode",
        choices=["gather_kv", "ring"],
        default="gather_kv",
        help="Context-parallel attention strategy (default: gather_kv)",
    )
    p.add_argument(
        "--cfg-parallel",
        dest="cfg_parallel",
        action="store_true",
        help="Split the uncond/cond CFG passes across 2 data-parallel "
        "ranks (doubles world_size). Mutually exclusive with "
        "--cp-degree>1. Only for true-CFG models (Flux, Wan, LTX-2).",
    )
    p.add_argument(
        "--sp",
        dest="sp_enabled",
        action="store_true",
        help="Enable Megatron-style sequence parallelism: shard the "
        "norm/modulation/residual regions along the sequence axis "
        "across the tensor-parallel group (reduce-scatter replaces "
        "the row-parallel all-reduce; world_size unchanged). "
        "Mutually exclusive with --cp-degree>1. Supported: Flux, "
        "Wan, HunyuanVideo.",
    )


def _add_shape_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)


def _add_cache_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Compiled artifact cache root (default: ~/.cache/difflet/)",
    )
    p.add_argument(
        "--force", action="store_true", help="Recompile even if a valid cache entry exists"
    )
    p.add_argument(
        "--host-vae",
        dest="host_vae",
        action="store_true",
        help="Decode the VAE on host CPU via diffusers instead of a "
        "compiled Neuron VAE. Required for Wan clips beyond ~9 "
        "frames: the single-shot Neuron VAE graph exceeds the "
        "compiler instruction limit (NCC_EVRF007).",
    )


def _add_serve_profile_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--tp-degree",
        type=int,
        default=None,
        help="Tensor-parallel degree (default: registry default)",
    )
    p.add_argument(
        "--cp-degree",
        type=int,
        default=None,
        help="Context-parallel degree (default: registry default)",
    )
    p.add_argument(
        "--cp-mode",
        choices=["gather_kv", "ring"],
        default=None,
        help="Context-parallel attention strategy (default: registry default)",
    )
    cfg = p.add_mutually_exclusive_group()
    cfg.add_argument(
        "--cfg-parallel",
        dest="cfg_parallel",
        action="store_true",
        help="Enable CFG-parallel startup topology",
    )
    cfg.add_argument(
        "--no-cfg-parallel",
        dest="cfg_parallel",
        action="store_false",
        help="Disable CFG-parallel startup topology",
    )
    p.set_defaults(cfg_parallel=None)
    sp = p.add_mutually_exclusive_group()
    sp.add_argument(
        "--sp",
        dest="sp_enabled",
        action="store_true",
        help="Enable sequence parallelism for the resident model profile",
    )
    sp.add_argument(
        "--no-sp",
        dest="sp_enabled",
        action="store_false",
        help="Disable sequence parallelism for the resident model profile",
    )
    p.set_defaults(sp_enabled=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Compiled artifact cache root (default: ~/.cache/difflet/)",
    )
    p.add_argument(
        "--force", action="store_true", help="Recompile even if a valid cache entry exists"
    )
    p.add_argument(
        "--host-vae",
        dest="host_vae",
        action="store_true",
        help="Request host VAE decode (rejected by current image serving)",
    )
    p.add_argument(
        "--teacache-cadence",
        type=int,
        default=None,
        metavar="N",
        help="Request fixed-cadence TeaCache (not supported by serving)",
    )
    p.add_argument(
        "--teacache-online-delta",
        type=float,
        default=None,
        metavar="ALPHA",
        help="Request online-delta TeaCache (not supported by serving)",
    )
    p.add_argument(
        "--teacache-speedup",
        type=float,
        default=None,
        metavar="X",
        help="Adaptive TeaCache target speedup",
    )
    p.add_argument(
        "--teacache-calibration", default=None, metavar="PATH", help="TeaCache calibration JSON"
    )


def _add_generate_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", required=True, help="Output file path (.png or .mp4)")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--work-dir", default=None, help="Directory for inter-stage tensors (staged models only)"
    )
    p.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Do not delete work-dir after successful generation",
    )
    p.add_argument(
        "--teacache-cadence",
        type=int,
        default=None,
        metavar="N",
        help="Skip every N-th DiT step (fixed cadence, no calibration)",
    )
    p.add_argument(
        "--teacache-online-delta",
        type=float,
        default=None,
        metavar="ALPHA",
        help="Online-delta TeaCache alpha (no calibration)",
    )
    p.add_argument(
        "--teacache-speedup",
        type=float,
        default=None,
        metavar="X",
        help="Adaptive TeaCache target speedup (requires --teacache-calibration)",
    )
    p.add_argument(
        "--teacache-calibration",
        default=None,
        metavar="PATH",
        help="Path to TeaCache calibration JSON",
    )


def _add_serve_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument(
        "--worker-heartbeat-interval",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Worker heartbeat interval in seconds (default: 30)",
    )


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="difflet", description="Difflet — diffusion inference on Trainium"
    )
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

    serve = sub.add_parser("serve", help="Start OpenAI-compatible T2I serving")
    _add_serve_model_flag(serve)
    serve.add_argument("--revision", default=None)
    _add_serve_profile_flags(serve)
    _add_serve_flags(serve)

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
        print(
            f"Error: {active_names[0]} and {active_names[1]} are mutually exclusive.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if speedup is not None and not calib:
        print("Error: --teacache-speedup requires --teacache-calibration PATH.", file=sys.stderr)
        raise SystemExit(1)


# Guidance-distilled models run a single forward pass with the guidance scale
# baked into the timestep embedding, so CFG-parallel has no second branch to
# split via the CLI. Hunyuan/Qwen have no true-CFG path at all; Flux is also
# guidance-distilled and *does* have an opt-in true-CFG path, but we don't expose
# its --true-cfg-scale/--negative-prompt knobs through the CLI, so cfg-parallel is
# rejected here too. The staged CLI path builds the app directly (bypassing each
# model's entry.py guard), so reject before dispatch.
_DISTILLED_MODELS = {
    "black-forest-labs/FLUX.1-dev",
    "hunyuanvideo-community/HunyuanVideo",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    "Qwen/Qwen-Image",
}


# Models whose backbone wires Megatron-style sequence parallelism, device-verified
# (dense-vs-SP cosine >= 0.999). Qwen-Image is deferred: its forward monkey-patches
# the upstream diffusers transformer, where the SPMDRank per-rank id used by the
# sequence scatter is not a live/loaded graph input, so every rank reads rank 0
# (tracked follow-up — needs reimplementing Qwen's forward like the others). LTX-2
# (tri-stream, no CP foundation) and HunyuanVideo-1.5 (segmented runtime) are also
# out of scope for this increment.
_SP_SUPPORTED_MODELS = {
    "black-forest-labs/FLUX.1-dev",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "hunyuanvideo-community/HunyuanVideo",
}


def _validate_sp(args: argparse.Namespace) -> None:
    if not getattr(args, "sp_enabled", False):
        return
    if (getattr(args, "cp_degree", 1) or 1) > 1:
        print(
            "Error: --sp and --cp-degree>1 are mutually exclusive (SP shards the "
            "sequence over the tensor-parallel group, CP over the data-parallel "
            "group).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.model_id not in _SP_SUPPORTED_MODELS:
        print(
            f"Error: {args.model_id} does not support --sp. Sequence parallelism "
            "is available for Flux, Wan, and HunyuanVideo.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _validate_cfg_parallel(args: argparse.Namespace) -> None:
    if not getattr(args, "cfg_parallel", False):
        return
    if (getattr(args, "cp_degree", 1) or 1) > 1:
        print(
            "Error: --cfg-parallel and --cp-degree>1 are mutually exclusive "
            "(both consume the data-parallel lanes).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.model_id in _DISTILLED_MODELS:
        print(
            f"Error: {args.model_id} is guidance-distilled (single forward pass "
            "with the guidance scale baked into the timestep embedding); "
            "CFG-parallel requires true two-pass classifier-free guidance and "
            "does not apply.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _get_orchestrator(args: argparse.Namespace):
    from difflet.cli.orchestrators.flux import FluxOrchestrator
    from difflet.cli.orchestrators.hunyuan_video import HunyuanVideoOrchestrator
    from difflet.cli.orchestrators.hunyuan_video_15 import HunyuanVideo15Orchestrator
    from difflet.cli.orchestrators.ltx_2 import LTX2Orchestrator
    from difflet.cli.orchestrators.qwen_image import QwenImageOrchestrator
    from difflet.cli.orchestrators.wan import WanOrchestrator

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

    valid_models = SERVE_VALID_MODELS if args.command == "serve" else VALID_MODELS
    if args.model_id not in valid_models:
        print(
            f"Error: Unknown model-id '{args.model_id}'. Valid model IDs:\n"
            + "\n".join(f"  {m}" for m in sorted(valid_models)),
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.command in ("compile", "generate", "run"):
        _validate_cfg_parallel(args)
        _validate_sp(args)

    if args.command in ("generate", "run"):
        _validate_teacache(args)

    if args.command == "serve":
        from difflet.serving.cli.serve import run, validate_serve_args

        validate_serve_args(args)
        run(args)
    else:
        orchestrator = _get_orchestrator(args)
        getattr(orchestrator, args.command)()


if __name__ == "__main__":
    main()
