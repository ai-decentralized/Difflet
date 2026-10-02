"""``difflet quantize`` — build the FP8 checkpoint copies of a model's DiT on the CPU.

Runs the offline quantizer (``difflet.quant.checkpoint``) for every transformer
subfolder of the downloaded model. ``difflet compile --quant …`` does the same
implicitly; this command exists so the (CPU-only, minutes-long) step can be run
and inspected on its own.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from difflet.quant.checkpoint import quantize_checkpoint_dir, quantized_checkpoint_dir
from difflet.quant.spec import QuantSpec

# Model types whose backbone is wired for FP8 PTQ (QuantApplicationMixin +
# quantize_traced_model_ in the backbone; see difflet/quant/targets.py for the
# per-model layer sets). HunyuanVideo 1.5 and the segmented runtimes are not.
QUANT_MODEL_TYPES: frozenset[str] = frozenset(
    {"wan", "flux", "qwen_image", "hunyuan_video", "ltx_2"}
)
_TRANSFORMER_SUBFOLDERS = ("transformer", "transformer_2")


def transformer_subfolders(model_dir: str) -> list[str]:
    return [
        name
        for name in _TRANSFORMER_SUBFOLDERS
        if os.path.isfile(os.path.join(model_dir, name, "config.json"))
    ]


def run(args: argparse.Namespace) -> int:
    from difflet.pipeline.path_resolver import resolve_model_path

    spec = QuantSpec.from_args(args)
    if spec is None:
        print("Error: --quant is required (e.g. --quant fp8).", file=sys.stderr)
        return 1
    try:
        model_dir = resolve_model_path(args.model_id, revision=args.revision, local_files_only=True)
    except OSError:
        print(
            f"Error: model weights not found.\nRun: difflet download --model-id {args.model_id}",
            file=sys.stderr,
        )
        return 1
    subfolders = transformer_subfolders(model_dir)
    if not subfolders:
        print(f"Error: no transformer/config.json under {model_dir}", file=sys.stderr)
        return 1
    for subfolder in subfolders:
        source = os.path.join(model_dir, subfolder)
        dest = quantized_checkpoint_dir(args.cache_dir, source, spec)
        started = time.perf_counter()
        manifest = quantize_checkpoint_dir(source, dest, spec, force=bool(args.force))
        report = manifest.get("report", {})
        print(
            f"[quantize] {subfolder}: {spec.checkpoint_label()} -> {dest}\n"
            f"[quantize]   linears quantized: {report.get('num_quantized')}  "
            f"bytes {report.get('bytes_before')} -> {report.get('bytes_after')}  "
            f"({time.perf_counter() - started:.1f}s)",
            flush=True,
        )
    return 0
