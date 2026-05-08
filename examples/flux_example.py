"""Flux text-to-image inference on Trainium via Nova.

Single-instance launch (one image):

    torchrun --nproc_per_node=8 examples/flux_example.py \\
        --model black-forest-labs/FLUX.1-dev \\
        --tp-degree 8 \\
        --prompt "A photorealistic cat sitting in a sunlit garden" \\
        --output out.png

Smaller dev box (e.g. trn3pd98.3xlarge has 4 NeuronCores):

    torchrun --nproc_per_node=4 examples/flux_example.py \\
        --model black-forest-labs/FLUX.1-dev \\
        --tp-degree 4 \\
        --prompt "..." --output out.png

CFG-parallel (doubles world_size; faster denoising at the cost of one more
data-parallel rank per device group):

    torchrun --nproc_per_node=8 examples/flux_example.py \\
        --model black-forest-labs/FLUX.1-dev \\
        --tp-degree 4 --cfg-parallel \\
        --prompt "..." --negative-prompt "blurry, low quality" \\
        --output out.png

Precompile only — useful for warming the cache before a benchmark or for CI:

    torchrun --nproc_per_node=4 examples/flux_example.py \\
        --model black-forest-labs/FLUX.1-dev \\
        --tp-degree 4 --precompile-only

Cache key composition:
    (model_id, revision, tp/cp/cfg-parallel, dtype, height, width, toolchain)

The first run with a given combination triggers AOT compile (single-digit
to tens of minutes). Subsequent runs hit the on-disk cache; default
location ``~/.cache/nova/`` or ``$NOVA_COMPILE_CACHE``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from nova import NovaParallelConfig, NovaPipeline


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ----- model selection -----
    p.add_argument("--model", required=True,
                   help="HF model id (e.g. black-forest-labs/FLUX.1-dev) or local path")
    p.add_argument("--model-type", default="flux",
                   help="Override registry detection (default: flux)")
    p.add_argument("--revision", default=None,
                   help="HF revision / commit pin")

    # ----- parallelism (None means: fall through to registry default) -----
    p.add_argument("--tp-degree", type=int, default=None,
                   help="Tensor-parallel degree (registry default: 8)")
    p.add_argument("--cp-enabled", action="store_true",
                   help="Enable context parallel (mutually exclusive with --cfg-parallel)")
    p.add_argument("--cfg-parallel", action="store_true",
                   help="Enable CFG parallel (mutually exclusive with --cp-enabled)")

    # ----- shape / dtype -----
    p.add_argument("--height", type=int, default=None, help="default 1024")
    p.add_argument("--width", type=int, default=None, help="default 1024")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])

    # ----- inference params -----
    p.add_argument("--prompt", default="A photorealistic cat sitting in a sunlit garden")
    p.add_argument("--negative-prompt", default=None,
                   help="Required for true CFG (with --cfg-parallel)")
    p.add_argument("--num-inference-steps", type=int, default=28)
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--true-cfg-scale", type=float, default=1.0,
                   help=">1.0 enables true CFG (must pair with --negative-prompt)")
    p.add_argument("--max-sequence-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)

    # ----- output / cache / debugging -----
    p.add_argument("--output", default="out.png")
    p.add_argument("--compile-cache-dir", default=None,
                   help="Default: $NOVA_COMPILE_CACHE or ~/.cache/nova/")
    p.add_argument("--precompile-only", action="store_true",
                   help="AOT compile and exit; skip load + forward")
    p.add_argument("--debug-compile", action="store_true",
                   help="Pass debug=True into app.compile() if supported")
    p.add_argument("--force-compile", action="store_true",
                   help="Recompile even if a valid cached artifact exists")

    return p.parse_args(argv)


def _build_parallel_config(args: argparse.Namespace) -> NovaParallelConfig | None:
    """Return None when user didn't specify any parallel flag (use registry default)."""
    if args.tp_degree is None and not args.cp_enabled and not args.cfg_parallel:
        return None
    return NovaParallelConfig(
        tp_degree=args.tp_degree if args.tp_degree is not None else 1,
        cp_enabled=args.cp_enabled,
        cfg_parallel_enabled=args.cfg_parallel,
    )


def _is_main_process() -> bool:
    """True on rank 0 / single-process launches. We only save / log there."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    import os
    rank_env = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    return rank_env is None or rank_env == "0"


def _log(msg: str) -> None:
    if _is_main_process():
        print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    parallel = _build_parallel_config(args)
    dtype = getattr(torch, args.dtype)

    _log(f"[nova] model           = {args.model}")
    _log(f"[nova] parallel        = {parallel or 'registry-default'}")
    _log(f"[nova] dtype           = {dtype}")
    _log(f"[nova] shape           = h={args.height} w={args.width}")
    _log(f"[nova] cache_dir       = {args.compile_cache_dir or '$NOVA_COMPILE_CACHE / ~/.cache/nova'}")

    common_kwargs = dict(
        model_type=args.model_type,
        parallel=parallel,
        dtype=dtype,
        height=args.height,
        width=args.width,
        compile_cache_dir=args.compile_cache_dir,
        revision=args.revision,
        force_compile=args.force_compile,
        debug_compile=args.debug_compile,
    )

    t0 = time.monotonic()
    if args.precompile_only:
        NovaPipeline.precompile(args.model, **common_kwargs)
        _log(f"[nova] precompile done in {time.monotonic() - t0:.1f}s")
        return 0

    pipe = NovaPipeline.from_pretrained(args.model, **common_kwargs)
    _log(f"[nova] from_pretrained done in {time.monotonic() - t0:.1f}s "
         f"(compile cache at {pipe.compiled_path})")

    # FluxPipeline accepts a torch.Generator. Per-rank seeding is fine; the
    # text encoder forward is replicated and the diffusion noise is broadcast
    # from rank 0 internally by the diffusers scheduler.
    generator = torch.Generator().manual_seed(args.seed)

    forward_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        true_cfg_scale=args.true_cfg_scale,
        max_sequence_length=args.max_sequence_length,
        height=args.height or 1024,
        width=args.width or 1024,
        generator=generator,
    )

    t1 = time.monotonic()
    output = pipe(**forward_kwargs)
    _log(f"[nova] forward done in {time.monotonic() - t1:.1f}s")

    # Save only on the main process — every rank holds a copy of the image
    # tensor / PIL.Image after VAE decode, but writing the file once is enough.
    if _is_main_process():
        images = getattr(output, "images", None) or output[0]
        if isinstance(images, list):
            image = images[0]
        else:
            image = images
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        image.save(args.output)
        _log(f"[nova] image saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
