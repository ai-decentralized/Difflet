"""Offline FP8 quantization of a HuggingFace transformer checkpoint directory.

Runs on CPU (no Neuron dependency). Produces a directory the vendored NxDI
checkpoint loader consumes as ``NeuronConfig.quantized_checkpoints_path``:

    <cache_dir>/quantized/<source-slug>/<subfolder>/<fp8-<granularity>-<hash8>>/
        model.safetensors (or sharded + index)   fp8 weights + float32 weight_scale
        config.json                              copied from the source
        difflet_quant.json                       manifest (spec, source, report)

For every target linear ``<prefix>.weight`` becomes ``float8_e4m3fn`` and
``<prefix>.weight_scale`` is added (``[1]`` per-tensor, ``[out, 1]`` per
channel). ``NeuronApplicationBase.get_state_dict`` renames ``.weight_scale`` to
``.scale`` (the NxD quantized-layer parameter) and ``checkpoint_loader_fn``
leaves fp8 tensors and scales uncast. Everything else is copied verbatim.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from difflet.quant.fp8 import quantize_weight
from difflet.quant.spec import QuantSpec

MANIFEST_FILENAME = "difflet_quant.json"
MANIFEST_SCHEMA_VERSION = 1
_MODEL_FILES = ("model.safetensors", "model.safetensors.index.json")


# HF (diffusers) spellings of targets whose Difflet module name differs; the
# calibration run names layers the Difflet way (the CPU model), the checkpoint
# keys are the HF ones.
_HF_TO_DIFFLET_NAME = (
    (".ffn.net.0.proj", ".ffn.net_in"),
    (".ffn.net.2", ".ffn.net_out"),
)


def calibrated_amax(layers: dict[str, float], prefix: str) -> float:
    """Input absmax for the HF-named target ``prefix`` from a calibration map.

    Tries the HF name, its Difflet rename, and — for FLUX / HunyuanVideo's fused
    single-block ``proj_out`` — the max over the two device halves.
    """
    candidates = [prefix] + [prefix.replace(hf, ours) for hf, ours in _HF_TO_DIFFLET_NAME if hf in prefix]
    for name in candidates:
        if name in layers:
            return layers[name]
    if prefix.endswith(".proj_out"):
        halves = [layers[n] for n in (prefix + "_attn", prefix + "_mlp") if n in layers]
        if halves:
            return max(halves)
    raise ValueError(f"calibration has no input absmax for target {prefix!r}")


def quantize_state_dict(
    state_dict: dict[str, Any], spec: QuantSpec
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Quantize the target 2-D ``.weight`` tensors; pass everything else through.

    With ``spec.calibration`` every quantized target also gets a float32 ``[1]``
    ``.input_scale`` (= calibrated absmax * STATIC_ACT_MARGIN / 240) that NxD's
    static activation path loads next to the weight scale.
    """
    from difflet.quant.fp8 import FP8_MAX, STATIC_ACT_MARGIN

    layers = spec.calibration_layers() if spec.calibration else None
    out: dict[str, Any] = {}
    quantized: list[str] = []
    bytes_before = 0
    bytes_after = 0
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            bytes_before += value.numel() * value.element_size()
        if (
            key.endswith(".weight")
            and torch.is_tensor(value)
            and value.ndim == 2
            and spec.matches(key[: -len(".weight")])
        ):
            prefix = key[: -len(".weight")]
            weight_fp8, scale = quantize_weight(value, spec.weight_granularity)
            out[key] = weight_fp8
            out[prefix + ".weight_scale"] = scale
            if layers is not None:
                amax = calibrated_amax(layers, prefix)
                out[prefix + ".input_scale"] = torch.tensor(
                    [amax * STATIC_ACT_MARGIN / FP8_MAX], dtype=torch.float32
                )
                bytes_after += 4
            quantized.append(prefix)
            bytes_after += weight_fp8.numel() * weight_fp8.element_size()
            bytes_after += scale.numel() * scale.element_size()
        else:
            out[key] = value
            if torch.is_tensor(value):
                bytes_after += value.numel() * value.element_size()

    report: dict[str, Any] = {
        "num_quantized": len(quantized),
        "quantized": quantized,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "fp8_max": FP8_MAX,
    }
    if layers is not None:
        report["static_activation_scales"] = len(quantized)
        report["static_margin"] = STATIC_ACT_MARGIN
        report["calibration_sha"] = spec.calibration_sha()
    return out, report


def split_fused_proj_out(
    state_dict: dict[str, Any],
    prefix: str,
    *,
    attn_name: str,
    mlp_name: str,
    cols: int,
) -> None:
    """Split ``<prefix>.weight`` (``[out, in]``) at column ``cols`` into two linears, in place.

    FLUX / HunyuanVideo single blocks store one fused ``proj_out`` over
    ``cat([attn, mlp])`` that the device runs as two row-parallel linears. The
    attn half keeps the bias (the device adds it once, after the reduce); the
    fp8 ``scale`` — per-tensor ``[1]`` or per-channel ``[out, 1]`` — is copied to
    both halves, which is exact because the split is along the input dim. A
    bf16 checkpoint (no ``.scale``) takes the same path without a scale.
    """
    w = state_dict.pop(f"{prefix}.weight")
    state_dict[f"{attn_name}.weight"] = w[:, :cols].clone().contiguous()
    state_dict[f"{mlp_name}.weight"] = w[:, cols:].clone().contiguous()
    bias = state_dict.pop(f"{prefix}.bias", None)
    if bias is not None:
        state_dict[f"{attn_name}.bias"] = bias.clone().contiguous()
    scale = state_dict.pop(f"{prefix}.scale", None)
    if scale is not None:
        state_dict[f"{attn_name}.scale"] = scale.clone()
        state_dict[f"{mlp_name}.scale"] = scale.clone()
    # Static activation scale: calibrated over the fused cat([attn, mlp]) input, so
    # the same (conservative) constant serves both halves.
    input_scale = state_dict.pop(f"{prefix}.input_scale", None)
    if input_scale is not None:
        state_dict[f"{attn_name}.input_scale"] = input_scale.clone()
        state_dict[f"{mlp_name}.input_scale"] = input_scale.clone()


def _source_slug(source: str) -> str:
    """Human-readable dir token for a resolved source: the HF ``models--org--name``
    cache component when present, else ``<parent>_<name>``."""
    for part in reversed(Path(source).parts):
        if part.startswith("models--"):
            return part[len("models--") :]
    path = Path(source)
    return f"{path.parent.name}_{path.name}" if path.parent.name else path.name


def quantized_checkpoint_dir(
    cache_dir: str | os.PathLike[str] | None,
    source_dir: str | os.PathLike[str],
    spec: QuantSpec,
) -> Path:
    """Deterministic location of the quantized copy of ``source_dir`` (a
    ``transformer/`` folder). Keyed by the resolved source path (so a new HF
    snapshot never reuses stale weights) and the weight-affecting spec fields."""
    if cache_dir is None:
        from difflet import envs

        root = Path(envs.DIFFLET_COMPILE_CACHE)
    else:
        root = Path(cache_dir)
    source = os.path.realpath(str(source_dir))
    return (
        root.expanduser()
        / "quantized"
        / _source_slug(source)
        / Path(source).name
        / f"{spec.checkpoint_label()}-{spec.checkpoint_hash(source)}"
    )


def read_manifest(checkpoint_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    path = Path(checkpoint_dir) / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def is_valid_quantized_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    spec: QuantSpec,
    source_dir: str | os.PathLike[str],
) -> bool:
    manifest = read_manifest(checkpoint_dir)
    if manifest is None or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return False
    if manifest.get("checkpoint_identity") != spec.checkpoint_identity():
        return False
    if manifest.get("source_realpath") != os.path.realpath(str(source_dir)):
        return False
    return any((Path(checkpoint_dir) / name).is_file() for name in _MODEL_FILES)


def quantize_checkpoint_dir(
    source_dir: str | os.PathLike[str],
    dest_dir: str | os.PathLike[str],
    spec: QuantSpec,
    *,
    force: bool = False,
    max_shard_size: str = "10GB",
) -> dict[str, Any]:
    """Quantize ``source_dir`` into ``dest_dir``; returns the manifest. A valid
    existing checkpoint is reused unless ``force``."""
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    if not force and is_valid_quantized_checkpoint(dest_dir, spec, source_dir):
        return read_manifest(dest_dir) or {}

    # Import-safe on CPU: the vendored loader/saver only need torch + safetensors.
    from difflet.backends.trainium.core.modules.checkpoint import (
        load_state_dict,
        save_state_dict_safetensors,
    )

    started = time.perf_counter()
    state_dict = load_state_dict(str(source_dir))
    load_s = time.perf_counter() - started
    quantized, report = quantize_state_dict(state_dict, spec)
    del state_dict
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    save_state_dict_safetensors(quantized, str(dest_dir), max_shard_size=max_shard_size)
    config = source_dir / "config.json"
    if config.is_file():
        shutil.copy2(config, dest_dir / "config.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "spec": spec.to_dict(),
        "checkpoint_identity": spec.checkpoint_identity(),
        "source_dir": str(source_dir),
        "source_realpath": os.path.realpath(str(source_dir)),
        "report": report,
        "load_seconds": round(load_s, 3),
        "total_seconds": round(time.perf_counter() - started, 3),
        "torch": torch.__version__,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (dest_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def ensure_quantized_checkpoint(
    source_dir: str | os.PathLike[str],
    dest_dir: str | os.PathLike[str],
    spec: QuantSpec,
    *,
    create: bool,
    force: bool = False,
) -> Path:
    """Return ``dest_dir`` once it holds a valid quantized copy of ``source_dir``.

    ``create=True`` (compile / ``difflet quantize``) builds it when missing;
    ``create=False`` (generate / serve load) raises with the command to run.
    """
    dest = Path(dest_dir)
    if not force and is_valid_quantized_checkpoint(dest, spec, source_dir):
        return dest
    if not create:
        raise FileNotFoundError(
            f"quantized checkpoint ({spec.checkpoint_label()}) for {source_dir} is missing "
            f"at {dest}; run `difflet quantize --model-id <id> --quant fp8 "
            f"--quant-granularity {spec.weight_granularity}` or `difflet compile --quant fp8 ...` first."
        )
    quantize_checkpoint_dir(source_dir, dest, spec, force=force)
    return dest


__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "ensure_quantized_checkpoint",
    "is_valid_quantized_checkpoint",
    "quantize_checkpoint_dir",
    "quantize_state_dict",
    "quantized_checkpoint_dir",
    "read_manifest",
]
