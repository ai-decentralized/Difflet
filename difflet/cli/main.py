from __future__ import annotations

import argparse
import math
import os
import sys

from difflet.pipeline.parallel_config import CP_MODES

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
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "hunyuanvideo-community/HunyuanVideo",
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
    p.add_argument(
        "--cp-degree",
        type=int,
        default=None,
        help="Context-parallel degree (default: 1)",
    )
    p.add_argument(
        "--cp-mode",
        choices=list(CP_MODES),
        default="gather_kv",
        help="Context-parallel attention strategy (default: gather_kv). "
        "'ring' rotates the K,V shards; 'ulysses' all-to-alls the "
        "sequence shard into a head shard. Both need --cp-degree > 1; "
        "'ulysses' additionally needs the model's head count divisible "
        "by tp_degree * cp_degree.",
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
    p.add_argument(
        "--dp",
        type=int,
        default=None,
        help="Data-parallel replica count. The router spawns N workers, "
        "each a full dp=1 model copy on its own core range; requests "
        "are distributed across them (default: 1)",
    )
    p.add_argument(
        "--mode",
        choices=["latency", "throughput", "mixed"],
        default=None,
        help="Runtime mode preset selecting dp/cfg/cp per model class "
        "(explicit parallelism flags override individual fields)",
    )
    p.add_argument(
        "--dp-schedule",
        choices=["round_robin", "least_loaded"],
        default="round_robin",
        help="Request-to-replica schedule for --dp>1 (default: round_robin)",
    )
    p.add_argument(
        "--total-cores",
        type=int,
        default=None,
        help="Total NeuronCores available for dp*cfg*cp*tp validation "
        "(default: NEURON_RT_NUM_CORES when set, else unchecked)",
    )


def _add_shape_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument(
        "--shapes",
        default=None,
        metavar="HxWxF[,HxWxF...]",
        help="Compile a bucketed artifact covering several request shapes "
        "(e.g. 320x512x61,320x512x33; HxW for image models). All shapes "
        "share one weight copy on device. For generate, --height/--width/"
        "--num-frames select the request shape, which must be in this set.",
    )


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
    p.add_argument(
        "--taef1",
        dest="taef1",
        action="store_true",
        help="Replace the standard VAE decoder with the lightweight TAEF1 "
        "decoder (Flux only). Requires --taef1-path.",
    )
    p.add_argument(
        "--taef1-path",
        default=None,
        metavar="REPO_ID",
        help="HuggingFace repo id of the tiny VAE (e.g. madebyollin/taef1). "
        "Implies --taef1.",
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
        help="Decode the video VAE on the host CPU instead of the model's "
        "default serving placement",
    )
    p.add_argument(
        "--clip-placement",
        choices=["host", "neuron"],
        default=None,
        help="HunyuanVideo CLIP placement for this immutable serving profile",
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
    p.add_argument("--prompt", required=False, default=None)
    p.add_argument("--output", required=False, default=None, help="Output file path (.png or .mp4)")
    p.add_argument(
        "--requests",
        default=None,
        help="JSONL batch file: one request per line with prompt/output/"
        "seed and optional negative_prompt/guidance_scale/steps",
    )
    p.add_argument("--requests-dir", default=None, help=argparse.SUPPRESS)  # worker mode
    p.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
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
        "--api-key",
        type=_api_key,
        default=None,
        metavar="KEY",
        help=(
            "Bearer API key required by /v1 endpoints "
            "(default: DIFFLET_API_KEY; disabled when unset)"
        ),
    )
    p.add_argument(
        "--max-queued-requests",
        type=_nonnegative_int,
        default=8,
        metavar="COUNT",
        help="Maximum requests waiting behind the active request (default: 8; 0 disables queuing)",
    )
    p.add_argument(
        "--queue-timeout",
        type=_positive_finite_float,
        default=None,
        metavar="SECONDS",
        help=(
            "Maximum time a request may wait in the queue "
            "(default: 30 for image, 86400 for video)"
        ),
    )
    p.add_argument(
        "--request-timeout",
        type=_positive_finite_float,
        default=300.0,
        metavar="SECONDS",
        help=(
            "Maximum request execution time after queue admission for video, "
            "or total request time for image (default: 300)"
        ),
    )
    p.add_argument(
        "--artifact-store-timeout",
        type=_positive_finite_float,
        default=60.0,
        metavar="SECONDS",
        help="Maximum time for each artifact upload or URL operation (default: 60)",
    )
    p.add_argument(
        "--worker-cancel-timeout",
        type=_positive_finite_float,
        default=10.0,
        metavar="SECONDS",
        help="Time to wait for cooperative worker cancellation before restart (default: 10)",
    )
    p.add_argument(
        "--worker-restart-timeout",
        type=_positive_finite_float,
        default=900.0,
        metavar="SECONDS",
        help="Maximum time for worker restart and readiness recovery (default: 900)",
    )
    p.add_argument(
        "--worker-heartbeat-interval",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Worker heartbeat interval in seconds, 5-120 inclusive (default: 30)",
    )
    p.add_argument(
        "--validation-workers",
        type=_positive_int,
        default=4,
        metavar="COUNT",
        help="CPU request-validation threads (default: 4)",
    )
    p.add_argument(
        "--validation-max-waiting",
        type=_nonnegative_int,
        default=32,
        metavar="COUNT",
        help="Maximum validation submissions waiting for a thread (default: 32)",
    )
    p.add_argument(
        "--validation-timeout",
        type=_positive_finite_float,
        default=30.0,
        metavar="SECONDS",
        help="Maximum validation wait plus execution time (default: 30)",
    )
    p.add_argument(
        "--video-retention-seconds",
        type=_positive_int,
        default=25 * 60 * 60,
        metavar="SECONDS",
        help="Terminal in-process video job retention (default: 90000, 25 hours)",
    )
    p.add_argument(
        "--video-max-jobs",
        type=_positive_int,
        default=4096,
        metavar="COUNT",
        help="Maximum retained asynchronous video job records (default: 4096)",
    )
    p.add_argument(
        "--video-sweep-interval",
        type=_positive_finite_float,
        default=5 * 60.0,
        metavar="SECONDS",
        help="Expired-video sweep interval (default: 300)",
    )


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _api_key(value: str) -> str:
    if not value or any(character.isspace() for character in value):
        raise argparse.ArgumentTypeError("must be non-empty and cannot contain whitespace")
    return value


def _positive_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


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

    clean = sub.add_parser(
        "clean",
        help="Remove Neuron compiler scratch (hash dirs, neuronxcc-*/, "
        "log-neuron-cc.txt) from a directory",
    )
    clean.add_argument(
        "--dir",
        default=".",
        help="Directory to sweep (default: current working directory)",
    )
    clean.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="List what would be removed without deleting anything",
    )

    serve = sub.add_parser("serve", help="Start OpenAI-compatible image/video serving")
    _add_serve_model_flag(serve)
    serve.add_argument("--revision", default=None)
    _add_serve_profile_flags(serve)
    _add_serve_flags(serve)

    cache = sub.add_parser("cache", help="Inspect the compiled-artifact cache")
    cache.add_argument(
        "cache_action",
        choices=["ls"],
        help="ls: list artifacts (hash dir -> shapes/tp/dtype) from their manifests",
    )
    cache.add_argument(
        "--cache-dir",
        default=None,
        help="Compiled artifact cache root (default: ~/.cache/difflet/)",
    )
    cache.add_argument("--json", action="store_true", help="Emit JSON instead of a table")

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


def _validate_taef1(args: argparse.Namespace) -> None:
    if getattr(args, "taef1_path", None) is not None:
        setattr(args, "taef1", True)  # --taef1-path implies --taef1
    if not getattr(args, "taef1", False):
        return
    if not args.taef1_path:
        print(
            "Error: --taef1 requires --taef1-path REPO_ID (e.g. madebyollin/taef1).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.model_id != "black-forest-labs/FLUX.1-dev":
        print(
            f"Error: {args.model_id} does not support --taef1. The lightweight "
            "TAEF1 VAE is only wired into the Flux application.",
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


def _replica_cores(args: argparse.Namespace) -> int:
    from difflet.registry import resolve_model

    entry = resolve_model(args.model_id, model_type=_MODEL_TYPE[args.model_id])
    tp = args.tp_degree or entry.default_parallel.tp_degree
    cfg = 2 if getattr(args, "cfg_parallel", False) else 1
    return tp * (args.cp_degree or 1) * cfg


def _validate_dp(args: argparse.Namespace) -> None:
    batch = getattr(args, "requests", None) is not None or (getattr(args, "dp", None) or 1) > 1
    worker = getattr(args, "requests_dir", None) is not None
    if args.command in ("generate", "run") and not worker:
        if not batch and not (args.prompt and args.output):
            print(
                "Error: --prompt and --output are required (or use --requests FILE).",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if batch and args.prompt and args.requests:
            print("Error: --prompt and --requests are mutually exclusive.", file=sys.stderr)
            raise SystemExit(1)
    if batch and any(
        getattr(args, name, None) is not None
        for name in ("teacache_cadence", "teacache_online_delta", "teacache_speedup")
    ):
        print(
            "Error: TeaCache flags are not supported in batch/DP mode "
            "(per-request controller reset is a follow-up).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if (getattr(args, "dp", None) or 1) > 1 and not worker:
        total = getattr(args, "total_cores", None)
        if total is None and os.environ.get("NEURON_RT_NUM_CORES"):
            total = int(os.environ["NEURON_RT_NUM_CORES"])
        needed = args.dp * _replica_cores(args)
        if total is not None and needed > total:
            print(
                f"Error: dp*cfg*cp*tp = {needed} cores exceeds available cores ({total}).",
                file=sys.stderr,
            )
            raise SystemExit(1)


def _dispatch_dp(args: argparse.Namespace) -> None:
    """Route batch/DP generate runs through the router; exits the process."""
    from difflet.cli.dp import router
    from difflet.cli.dp.requests_io import RequestSpec, load_requests_jsonl

    if args.requests:
        requests = load_requests_jsonl(args.requests)
    else:
        requests = [
            RequestSpec(
                index=0,
                prompt=args.prompt,
                output=args.output,
                seed=args.seed,
                guidance_scale=args.guidance_scale,
                steps=args.steps,
            )
        ]
    dp = args.dp or 1
    if dp > 1 and len(requests) == 1:
        print("Warning: --dp > 1 with a single request leaves replicas idle.", file=sys.stderr)
    if args.command == "run":
        orch = _get_orchestrator(args)
        orch.download()
        orch.compile()
    raise SystemExit(router.run_router(args, requests, replica_cores=_replica_cores(args)))


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


def _ensure_jemalloc() -> None:
    """Re-exec once with jemalloc preloaded.

    The concurrent per-rank weight load (``torch.ops.neuron._parallel_load``,
    one thread per rank) spends most of its CPU in glibc ``calloc``/``free`` +
    page faults; the rank threads contend on the malloc arena lock / mmap_lock.
    jemalloc's per-thread arenas remove that contention — ~17% faster warm
    weight load, bit-identical output (validated trn3/FLUX).

    jemalloc ships inside ``torch_neuronx`` but its preload is disabled upstream
    (``torch_neuronx/__init__.py``: ``# _add_lib_preload("jemalloc")``).
    ``LD_PRELOAD`` must be set before the process starts, so we re-exec once.
    Opt out with ``DIFFLET_NO_JEMALLOC=1``.
    """
    if os.environ.get("DIFFLET_NO_JEMALLOC"):
        return
    if "libjemalloc" in os.environ.get("LD_PRELOAD", ""):
        return  # already preloaded / re-exec'd — avoid an exec loop
    import importlib.util

    spec = importlib.util.find_spec("torch_neuronx")  # locate without importing
    if spec is None or spec.origin is None:
        return
    lib = os.path.join(os.path.dirname(spec.origin), "lib", "libjemalloc.so")
    if not os.path.exists(lib):
        return
    os.environ["LD_PRELOAD"] = os.pathsep.join(
        p for p in (lib, os.environ.get("LD_PRELOAD", "")) if p
    )
    os.execv(sys.executable, [sys.executable, "-m", "difflet.cli.main", *sys.argv[1:]])


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "clean":
        from difflet.cli.clean import run as run_clean

        run_clean(args)
        return

    if args.command == "cache":
        from difflet.cli.cache_cmd import run_cache_command

        raise SystemExit(run_cache_command(args))

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
        _validate_taef1(args)
        from difflet.cli.modes import resolve_mode

        mode_cfg = resolve_mode(args.model_id, getattr(args, "mode", None), args)
        if mode_cfg is not None:
            args.dp = mode_cfg.dp
            args.cfg_parallel = mode_cfg.cfg_parallel
            args.cp_degree = mode_cfg.cp_degree
            print(
                f"[difflet] parallel: dp={args.dp or 1} "
                f"cfg={2 if args.cfg_parallel else 1} cp={args.cp_degree or 1} "
                f"(mode={args.mode})",
                flush=True,
            )
            _validate_cfg_parallel(args)  # re-run with resolved flags
            _validate_sp(args)

    if args.command in ("generate", "run"):
        _validate_teacache(args)
        _validate_dp(args)
        if args.requests_dir is None and (args.requests is not None or (args.dp or 1) > 1):
            _dispatch_dp(args)  # raises SystemExit

    # Only for the weight-loading commands, where jemalloc pays off. This list
    # is NOT the safety boundary for compilation: `run` also compiles when the
    # cache is cold, and neuronx-cc aborts under jemalloc. The compile path
    # strips the preload itself for exactly that reason -- see
    # difflet/backends/trainium/utils/compile_allocator.py.
    if argv is None and args.command in ("generate", "run"):
        _ensure_jemalloc()

    if args.command == "serve":
        from difflet.cli.serve import run, validate_serve_args

        validate_serve_args(args)
        run(args)
    else:
        orchestrator = _get_orchestrator(args)
        getattr(orchestrator, args.command)()


if __name__ == "__main__":
    main()
