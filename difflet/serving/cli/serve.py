"""`difflet serve` command implementation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from difflet.serving.factory import build_serving_stack
from difflet.serving.openai.api_server import create_app
from difflet.serving.options import CompilePolicy, DownloadPolicy, ServeOptions


def options_from_args(args: argparse.Namespace) -> ServeOptions:
    validate_serve_args(args)
    compile_policy = CompilePolicy.FORCE if getattr(args, "force", False) else CompilePolicy.AUTO
    download_policy = DownloadPolicy.AUTO
    return ServeOptions(
        model_id=args.model_id,
        revision=args.revision,
        host=args.host,
        port=args.port,
        tp_degree=args.tp_degree,
        cp_degree=args.cp_degree,
        cp_mode=args.cp_mode,
        cfg_parallel=getattr(args, "cfg_parallel", None),
        sp_enabled=getattr(args, "sp_enabled", None),
        height=args.height,
        width=args.width,
        num_frames=getattr(args, "num_frames", None),
        cache_dir=args.cache_dir,
        host_vae=getattr(args, "host_vae", False),
        teacache_cadence=getattr(args, "teacache_cadence", None),
        teacache_online_delta=getattr(args, "teacache_online_delta", None),
        teacache_speedup=getattr(args, "teacache_speedup", None),
        teacache_calibration=getattr(args, "teacache_calibration", None),
        download_policy=download_policy,
        compile_policy=compile_policy,
        worker_heartbeat_interval=getattr(args, "worker_heartbeat_interval", 30.0),
    )


def validate_serve_args(args: argparse.Namespace) -> None:
    """Validate universal serve-process settings before adapter selection."""

    if getattr(args, "worker_heartbeat_interval", 30.0) <= 0:
        print(
            "Error: --worker-heartbeat-interval must be greater than 0.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def run(args: argparse.Namespace) -> None:
    _load_serving_environment()
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("uvicorn is required for `difflet serve`") from exc

    options = options_from_args(args)
    stack = build_serving_stack(options)
    app = create_app(
        options=options,
        resolved_model=stack.resolved_model,
        engine=stack.engine,
        request_validator=stack.request_validator,
    )
    uvicorn.run(app, host=options.host, port=options.port, workers=1)


def _load_serving_environment() -> None:
    dotenv_path = Path.cwd() / ".env"
    if not dotenv_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("python-dotenv is required to load .env for `difflet serve`") from exc
    load_dotenv(dotenv_path=dotenv_path, override=False)
