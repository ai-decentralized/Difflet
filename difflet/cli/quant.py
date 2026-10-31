"""The FP8 quantization flags of the CLI (``--quant``, ``--quant-calibration``,
``--quant-compute``): their parser, their validation, the spec they describe, their cache-key
fields and their forwarding to the stage and DP worker launches. Torch-free: the driver runs
these before torchrun starts a rank.

What they mean: ``--quant fp8`` runs the DiT's attention and MLP linears in FP8 (W8A8, static
per-tensor scales; ``difflet/quant``) on the neuron backend on Trainium3, from a quantized
checkpoint that ``difflet compile`` writes beside the NEFF cache and ``generate`` only reads.
``--quant-calibration`` is the activation calibration file (``scripts/qwen_image_fp8_cpu_reference.py``
produces one); ``--quant-compute`` picks the matmul path (``nki``, the FP8 kernel, the default;
``dequant``, fp8 weights with bf16 matmuls, memory only). The models with FP8 target layers are
``difflet.quant.targets.TARGETS_BY_MODEL``.
"""

from __future__ import annotations

import argparse
import os
import sys

from difflet.quant.spec import COMPUTE_MODES, DEFAULT_COMPUTE, FORMATS, QuantSpec, file_sha256
from difflet.quant.targets import TARGETS_BY_MODEL, default_spec

#: CLI model ids with FP8 target layers, by their ``difflet.quant.targets`` key.
QUANT_MODELS: dict[str, str] = {"Qwen/Qwen-Image": "qwen_image"}


def add_quant_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--quant",
        dest="quant",
        choices=list(FORMATS),
        default=None,
        help="--backend neuron on Trainium3: run the DiT's attention and MLP linears in FP8 "
        "from a quantized checkpoint that `compile` writes under the cache dir; the "
        "modulation, embedders and output projection stay bf16. 'mxfp8': OCP microscaling, "
        "32-value block scales for weights and activations (in the kernel), no calibration. "
        "'fp8': per-tensor weights and static per-tensor activation scales from "
        "--quant-calibration. Qwen-Image only.",
    )
    p.add_argument(
        "--quant-calibration",
        dest="quant_calibration",
        default=None,
        metavar="PATH",
        help="The activation calibration JSON the FP8 input scales come from "
        "(scripts/qwen_image_fp8_cpu_reference.py writes one). Its sha256 is part of the "
        "artifact identity.",
    )
    p.add_argument(
        "--quant-compute",
        dest="quant_compute",
        choices=list(COMPUTE_MODES),
        default=None,
        help=f"How the FP8 linears multiply (default: {DEFAULT_COMPUTE}): 'nki', the FP8 "
        "tensor-engine kernel; 'dequant', fp8 weights in HBM with bf16 matmuls (memory only, "
        "for debugging).",
    )


def _fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def validate_quant(args: argparse.Namespace) -> None:
    """The driver's refusals, before anything is downloaded or launched."""
    quant = getattr(args, "quant", None)
    calibration = getattr(args, "quant_calibration", None)
    compute = getattr(args, "quant_compute", None)
    if quant is None:
        for flag, value in (("--quant-calibration", calibration), ("--quant-compute", compute)):
            if value is not None:
                _fail(f"{flag} needs --quant fp8.")
        return
    backend = getattr(args, "backend", None) or os.environ.get("DIFFLET_BACKEND")
    if backend != "neuron":
        _fail("--quant runs on the PyTorch-native neuron backend only (pass --backend neuron).")
    model_id = getattr(args, "model_id", None)
    if model_id not in QUANT_MODELS:
        _fail(f"--quant is not available for {model_id!r}; models with FP8 target layers: "
              f"{', '.join(sorted(QUANT_MODELS))}.")
    exec_mode = getattr(args, "exec_mode", None) or os.environ.get("DIFFLET_EXEC_MODE")
    if exec_mode == "eager":
        _fail("--quant needs the compiled DiT (--exec-mode compile, the default): eager "
              "quantizes every activation op by op.")
    if quant == "fp8" and calibration is None:
        _fail("--quant fp8 needs --quant-calibration PATH (the activation calibration file); "
              "--quant mxfp8 needs none.")
    if calibration is not None and not os.path.isfile(calibration):
        _fail(f"--quant-calibration {calibration}: no such file.")
    if getattr(args, "teacache_speedup", None) is not None:
        _fail("--quant and adaptive TeaCache (--teacache-speedup) are mutually exclusive; use "
              "--teacache-cadence instead.")
    if os.environ.get("UNSAFE_FP8FNCAST"):
        _fail("--quant: unset UNSAFE_FP8FNCAST; it makes neuronx-cc mix the legacy fp8 type "
              "with OCP float8_e4m3fn (NCC_EOCP001).")


#: Semicolon-separated regexes replacing the model's default target layers (experiments on
#: which layers to quantize; see ``difflet/quant/targets.py`` for the defaults).
QUANT_TARGETS_ENV = "DIFFLET_QUANT_TARGETS"


def quant_spec(args: argparse.Namespace) -> QuantSpec | None:
    """The spec the flags describe, with the calibration file's sha256, or None when off."""
    quant = getattr(args, "quant", None)
    if quant is None:
        return None
    model = QUANT_MODELS[args.model_id]
    if model not in TARGETS_BY_MODEL:
        raise ValueError(f"no FP8 targets for {model}")
    compute = getattr(args, "quant_compute", None) or DEFAULT_COMPUTE
    calibration = getattr(args, "quant_calibration", None)
    sha = file_sha256(calibration) if (calibration and quant == "fp8") else None
    # The row-parallel weights of mxfp8 are quantized per rank K range: the tp the launch uses
    # (the Qwen-Image driver's default is 4, cli/orchestrators/qwen_image.py).
    tp = int(getattr(args, "tp_degree", None) or 4)
    overrides = {}
    targets = os.environ.get(QUANT_TARGETS_ENV)
    if targets:
        # An experiment knob, not a flag: the layer set is part of the recipe, and the spec's
        # targets_sha256 keys the quantized checkpoint and the stage cache, so two settings
        # never share an artifact.
        overrides["targets"] = tuple(t for t in targets.split(";") if t)
    return default_spec(model, format=quant, compute=compute, calibration_sha256=sha,
                        tp_degree=tp, **overrides)


def quant_cli_args(args: argparse.Namespace) -> list[str]:
    """The flags to forward to a stage or DP worker launch, only when given (a default command
    line stays byte-identical)."""
    parts: list[str] = []
    if getattr(args, "quant", None):
        parts += ["--quant", args.quant]
    if getattr(args, "quant_calibration", None):
        parts += ["--quant-calibration", args.quant_calibration]
    if getattr(args, "quant_compute", None):
        parts += ["--quant-compute", args.quant_compute]
    return parts


__all__ = ["QUANT_MODELS", "add_quant_flags", "quant_cli_args", "quant_spec", "validate_quant"]
