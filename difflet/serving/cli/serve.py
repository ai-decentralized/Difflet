"""`difflet serve` command implementation."""

from __future__ import annotations

import argparse
import sys

from difflet.serving.factory import build_serving_stack
from difflet.serving.openai.api_server import create_app
from difflet.serving.options import CompilePolicy, DownloadPolicy, ServeOptions


def options_from_args(args: argparse.Namespace) -> ServeOptions:
    _validate_p0_serve_args(args)
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
        height=args.height,
        width=args.width,
        num_frames=getattr(args, "num_frames", None),
        cache_dir=args.cache_dir,
        download_policy=download_policy,
        compile_policy=compile_policy,
    )


def _validate_p0_serve_args(args: argparse.Namespace) -> None:
    if getattr(args, "cfg_parallel", False):
        print("Error: P0 serving does not support --cfg-parallel.", file=sys.stderr)
        raise SystemExit(1)
    if getattr(args, "sp_enabled", False):
        print("Error: P0 serving does not support --sp.", file=sys.stderr)
        raise SystemExit(1)
    if getattr(args, "num_frames", None) is not None:
        print(
            "Error: --num-frames is reserved for future video serving; "
            "Qwen/Flux P0 image serving requires it to be omitted.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def run(args: argparse.Namespace) -> None:
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
