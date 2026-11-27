"""The MXFP8 quantization flags of the CLI (``--quant``, ``--quant-compute``): their parser, their
validation (and that of ``--attention mx_attention``, the MXFP8 attention kernel), the spec they
describe, their cache-key fields, their forwarding to stage and DP-worker launches, and the
driver's quantized-checkpoint step (``prepare_quant``). Torch-free at import, so the driver
validates before torchrun starts a rank; only ``prepare_quant`` loads the checkpoint writer, and
torch with it, when it runs.

``--quant mxfp8`` quantizes the DiT's target linears (``difflet/quant/targets.py``) to OCP
MXFP8 from a quantized checkpoint directory that ``compile`` writes under the cache dir
(``difflet/quant/checkpoint.py``; ``generate`` only reads it). No calibration: the weights'
block scales are computed offline, the activations' inside the kernel. ``--quant-compute``
picks the matmul path (``nki``, the MXFP8 kernel, the default; ``dequant``, fp8 weights in HBM
with bf16 matmuls, memory only).

``--attention mx_attention`` (``difflet/cli/main.py``), independent of ``--quant``, runs the DiT's
self/joint attention with QK^T and PV on the MXFP8 attention kernel (``cte``, ``const-max`` or no
flag: bf16 attention). It travels as ``DIFFLET_ATTN_QUANT=mxfp8``, which ``ops_impl/attention.py``
reads once at import, so ``main._apply_backend_flags`` exports it before anything imports a
backend; stages and DP workers inherit it, as they do ``--attention const-max``'s variable, and
the DiT stage keys name it (``quant_cache_fields``). ``validate_quant`` holds its launch checks
beside ``--quant``'s.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

from difflet.quant.spec import (
    ATTN_QUANT_ENV,
    COMPUTE_MODES,
    DEFAULT_COMPUTE,
    FORMAT,
    QuantSpec,
    attn_quant_mode,
)
from difflet.quant.targets import TARGETS_BY_MODEL, default_spec

#: CLI model ids with MXFP8 target layers, by their ``difflet.quant.targets`` key.
QUANT_MODELS: dict[str, str] = {
    "Qwen/Qwen-Image": "qwen_image",
    "black-forest-labs/FLUX.1-dev": "flux",
    "Lightricks/LTX-2": "ltx_2",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": "wan",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "wan",
    "hunyuanvideo-community/HunyuanVideo": "hunyuan_video",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v": "hunyuan_video_15",
}


def add_quant_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--quant",
        dest="quant",
        choices=[FORMAT],
        default=None,
        help="--backend neuron on Trainium3: run the DiT's attention q/k/v and MLP linears in "
        "OCP MXFP8 (32-value block scales for weights, offline, and for activations, in the "
        "kernel; no calibration) from a quantized checkpoint that `compile` writes under the "
        "cache dir; the output projections, modulation, embedders and the output layer stay "
        "bf16 (except the single-stream output projection of FLUX and HunyuanVideo, quantized "
        "as its attention and MLP halves; LTX-2: its audio stream too; Wan: its self-attention "
        "q/k/v too; HunyuanVideo: its image q/k/v too). "
        "Qwen-Image, FLUX.1-dev, LTX-2, Wan 2.1, Wan 2.2-A14B (both experts), HunyuanVideo and "
        "HunyuanVideo-1.5, which has no default set: it needs DIFFLET_QUANT_TARGETS (see the "
        "evidence).",
    )
    p.add_argument(
        "--quant-compute",
        dest="quant_compute",
        choices=list(COMPUTE_MODES),
        default=None,
        help=f"How the MXFP8 linears multiply (default: {DEFAULT_COMPUTE}): 'nki', the MXFP8 "
        "tensor-engine kernel; 'dequant', fp8 weights in HBM with bf16 matmuls (memory only, "
        "for debugging).",
    )


def _fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


#: ``--attention``'s MXFP8 kernel, as the refusals name it.
MX_ATTENTION_FLAG = "--attention mx_attention"


def validate_quant(args: argparse.Namespace) -> None:
    """The driver's refusals, before anything is downloaded or launched. ``--quant`` and
    ``--attention mx_attention`` each need the neuron backend, a model with MXFP8 target layers,
    the compiled DiT and no UNSAFE_FP8FNCAST; neither needs the other. Runs after
    ``main._apply_backend_flags``, so an exported ``DIFFLET_ATTN_QUANT`` is ``--attention``'s
    when the flag is given, and the ambient value otherwise."""
    quant = getattr(args, "quant", None)
    if quant is None and getattr(args, "quant_compute", None) is not None:
        _fail("--quant-compute needs --quant mxfp8.")
    try:
        exported = attn_quant_mode()
    except ValueError as exc:
        _fail(str(exc))
    if exported != "off" and os.environ.get("DIFFLET_EXP_ATTN_CONSTMAX") == "1":
        # Only an ambient pair gets here (--attention clears the other kernel's variable); the
        # neuron attention would refuse it at its import, in every rank.
        _fail(f"{ATTN_QUANT_ENV}={exported} and DIFFLET_EXP_ATTN_CONSTMAX=1 select different "
              "attention kernels: at most one may be set (--attention picks one: mx_attention "
              "or const-max).")
    if exported != "off" and getattr(args, "model_id", None) == "MiniMaxAI/MiniMax-H3":
        _fail(f"{ATTN_QUANT_ENV}={exported} ({MX_ATTENTION_FLAG}) is not available for "
              "'MiniMaxAI/MiniMax-H3': it has no MXFP8 target layers; unset it.")
    if quant is not None:
        _validate_mxfp8_launch(args, "--quant")
    if getattr(args, "attention", None) == "mx_attention":
        _validate_mxfp8_launch(args, MX_ATTENTION_FLAG)
    if quant is not None and getattr(args, "teacache_speedup", None) is not None:
        _fail("--quant and adaptive TeaCache (--teacache-speedup) are mutually exclusive; use "
              "--teacache-cadence instead.")


def _validate_mxfp8_launch(args: argparse.Namespace, flag: str) -> None:
    backend = getattr(args, "backend", None) or os.environ.get("DIFFLET_BACKEND")
    if backend != "neuron":
        _fail(f"{flag} runs on the PyTorch-native neuron backend only (pass --backend neuron).")
    model_id = getattr(args, "model_id", None)
    if model_id not in QUANT_MODELS:
        _fail(f"{flag} is not available for {model_id!r}; models with MXFP8 target layers: "
              f"{', '.join(sorted(QUANT_MODELS))}.")
    exec_mode = getattr(args, "exec_mode", None) or os.environ.get("DIFFLET_EXEC_MODE")
    if exec_mode == "eager":
        _fail(f"{flag} needs the compiled DiT (--exec-mode compile, the default): eager "
              "quantizes every activation op by op.")
    if os.environ.get("UNSAFE_FP8FNCAST"):
        _fail(f"{flag}: unset UNSAFE_FP8FNCAST; it makes neuronx-cc mix the legacy fp8 type "
              "with OCP float8_e4m3fn (NCC_EOCP001).")
    cp_mode = getattr(args, "cp_mode", None)
    if flag == MX_ATTENTION_FLAG and (getattr(args, "cp_degree", None) or 1) > 1 and cp_mode in (
            "ring", "ulysses"):
        _fail(f"{flag} with --cp-mode {cp_mode}: ring and Ulysses context parallelism keep "
              "every attention call bf16, so nothing would be quantized; use --cp-mode gather_kv "
              "or no context parallelism.")


#: Semicolon-separated regexes replacing the model's default target layers (experiments on
#: which layers to quantize; see ``difflet/quant/targets.py`` for the defaults).
QUANT_TARGETS_ENV = "DIFFLET_QUANT_TARGETS"


def quant_spec(args: argparse.Namespace, *, default_tp_degree: int = 4) -> QuantSpec | None:
    """The spec the flags describe, or None when off. The row-parallel weights are quantized
    per rank K range, so the spec carries the tp the launch uses: ``--tp-degree``, else
    ``default_tp_degree`` (a driver whose default tp is not 4 passes its own). The default is a
    fallback, never an override: the flag wins on the driver as it does on the stage line the
    driver launches, so both sides build the same spec."""
    quant = getattr(args, "quant", None)
    if quant is None:
        return None
    model = QUANT_MODELS[args.model_id]
    if model not in TARGETS_BY_MODEL:
        raise ValueError(f"no MXFP8 targets for {model}")
    compute = getattr(args, "quant_compute", None) or DEFAULT_COMPUTE
    tp = int(getattr(args, "tp_degree", None) or default_tp_degree)
    overrides = {}
    targets = os.environ.get(QUANT_TARGETS_ENV)
    if targets:
        # An experiment knob, not a flag: the layer set is part of the recipe, and the spec's
        # targets_sha256 keys the quantized checkpoint and the stage cache, so two settings
        # never share an artifact.
        overrides["targets"] = tuple(t for t in targets.split(";") if t)
    return default_spec(model, compute=compute, tp_degree=tp, **overrides)


def prepare_quant(args: argparse.Namespace, component_dirs: dict[str, str], *, create: bool,
                  tag: str, default_tp_degree: int = 4,
                  fail: Callable[[str], None] = _fail) -> dict[str, str] | None:
    """The driver's ``--quant`` step, before any rank starts: for every ``name -> source_dir``
    (a diffusers component directory, ``transformer``), the quantized directory under the
    compile cache as ``name -> path``, written when ``create`` (``compile``) and only resolved
    otherwise (``generate``: a missing or stale one is refused with the fix, so no rank writes).
    Every rank resolves it again, read only, from the spec the stage flags carry, so the stage
    line names the tp this step quantized for (``--tp-degree``, flag or default, as every
    driver's does). One line per component, ``[tag] MXFP8 name (compute): path``.

    A refusal (a stale or, on ``generate``, missing directory; a component without safetensors,
    a partial download; a target pattern matching no linear weight of the component, which the
    writer names before quantizing anything) names the component and goes through ``fail``:
    ``_fail`` (exit 1)
    unless the driver passes the helper its other refusals use (``refuse``, exit 2), so a
    refused ``--quant`` exits the way the rest of that driver's refusals do.

    None when ``--quant`` is off, and None without a line when no source directory is there
    (none given, or one missing: the weights are not downloaded yet, and the ranks report
    that).
    """
    spec = quant_spec(args, default_tp_degree=default_tp_degree)
    if spec is None:
        return None
    if not component_dirs or not all(os.path.isdir(d) for d in component_dirs.values()):
        return None
    from difflet.quant.checkpoint import prepare_quantized_checkpoint

    prepared: dict[str, str] = {}
    for name, source_dir in component_dirs.items():
        try:
            quantized = prepare_quantized_checkpoint(source_dir, spec, create=create)
        except (ValueError, FileNotFoundError) as exc:
            fail(f"--quant: {name}: {exc}")
        prepared[name] = str(quantized)
        print(f"[{tag}] MXFP8 {name} ({spec.compute}): {quantized}", flush=True)
    return prepared


def quant_cli_args(args: argparse.Namespace) -> list[str]:
    """The flags to forward to a stage or DP worker launch, only when given (a default command
    line stays byte-identical)."""
    parts: list[str] = []
    if getattr(args, "quant", None):
        parts += ["--quant", args.quant]
    if getattr(args, "quant_compute", None):
        parts += ["--quant-compute", args.quant_compute]
    return parts


__all__ = ["MX_ATTENTION_FLAG", "QUANT_MODELS", "QUANT_TARGETS_ENV", "add_quant_flags",
           "prepare_quant", "quant_cli_args", "quant_spec", "validate_quant"]
