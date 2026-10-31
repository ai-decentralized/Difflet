"""The offline quantized checkpoint: a diffusers-layout directory with fp8 target weights.

Why a directory and not quantize-at-load: the native backend loads each rank's shards straight
from safetensors onto the device (``difflet/backends/neuron/core/checkpoint.py``) and caches them
in its pre-sharded weight store; both key on the files. A quantized directory with the same
layout and key names as the source lets that machinery load fp8 weights unchanged, and makes the
quantized bytes an artifact the cache keys can name (the bf16 copies that
``core/weights_bf16.py`` makes for Wan are the precedent).

Layout of ``<cache>/quantized/<source label>/<component>/fp8-<digest>/``:

* ``diffusion_pytorch_model-0000i-of-0000n.safetensors`` + ``.index.json``: every tensor of the
  source component. A target ``<prefix>.weight`` is stored as ``float8_e4m3fn`` with its fp32
  ``[1]`` ``<prefix>.weight_scale`` and ``<prefix>.input_scale`` beside it; everything else
  (biases, the bf16 modulation and embedder weights, norms) is copied byte for byte.
* ``config.json`` and the other non-tensor files of the source, copied.
* ``difflet_quant.json``: the spec, the source fingerprint, the calibration's sha256 and
  provenance, and per-tensor counts. :func:`check_quantized_dir` refuses a directory whose
  manifest does not match what the caller asks for.

The digest hashes the spec's weights identity, the source fingerprint (file names, sizes and
Hugging Face blob names, which are content hashes) and the calibration sha256, so another
checkpoint, another target set or another calibration is another directory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from difflet.quant.calibration import Calibration
from difflet.quant.fp8 import FP8_DTYPE, quantize_weight
from difflet.quant.spec import QuantSpec, strip_prefix

MANIFEST = "difflet_quant.json"
#: Shard size of the written safetensors files (diffusers' default is 10 GB; smaller shards keep
#: the writer's host memory and the loader's per-file reads bounded).
SHARD_BYTES = 4 * 1024**3
_SHARD = "diffusion_pytorch_model-{i:05d}-of-{n:05d}.safetensors"
_INDEX = "diffusion_pytorch_model.safetensors.index.json"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


# ----------------------------------------------------------------------- identity


def source_fingerprint(model_dir: str | Path) -> list[list[Any]]:
    """The source's identity without reading its data: every safetensors file's name, size and
    resolved blob name (a Hugging Face blob is named by its content hash), and every index
    file's sha256. No mtime: a copied checkpoint is the same checkpoint."""
    model_dir = Path(model_dir)
    entries: list[list[Any]] = []
    for path in sorted(model_dir.glob("*.safetensors")):
        entries.append([path.name, path.stat().st_size,
                        os.path.realpath(path).rsplit(os.sep, 1)[-1]])
    for path in sorted(model_dir.glob("*.safetensors.index.json")):
        entries.append([path.name, hashlib.sha256(path.read_bytes()).hexdigest()])
    if not entries:
        raise FileNotFoundError(f"no .safetensors under {model_dir}")
    return entries


def quantized_digest(spec: QuantSpec, fingerprint: list[list[Any]]) -> str:
    body = {"spec": spec.weights_identity(), "source": fingerprint}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
                          ).hexdigest()[:16]


def quantized_dir(cache_root: str | Path, model_dir: str | Path, spec: QuantSpec) -> Path:
    """Where the quantized copy of ``model_dir`` built with ``spec`` lives (exists or not)."""
    model_dir = Path(model_dir)
    digest = quantized_digest(spec, source_fingerprint(model_dir))
    label = _SAFE.sub("-", _source_label(model_dir)).strip("-_")[:120] or "checkpoint"
    return Path(cache_root).expanduser() / "quantized" / label / model_dir.name / f"{spec.format}-{digest}"


def _source_label(model_dir: Path) -> str:
    """``<repo>__<snapshot8>`` for a Hugging Face cache path, else the parent directory's name."""
    parts = model_dir.resolve().parts
    for i, part in enumerate(parts):
        if part.startswith("models--") and i + 2 < len(parts) and parts[i + 1] == "snapshots":
            return f"{part.removeprefix('models--')}__{parts[i + 2][:8]}"
    return model_dir.resolve().parent.name


# ------------------------------------------------------------------------ writing


def _tensor_key_prefix(key: str) -> str | None:
    return key[: -len(".weight")] if key.endswith(".weight") else None


def iter_source_tensors(model_dir: Path) -> Iterator[tuple[str, torch.Tensor, str]]:
    """``(key, tensor, file name)`` for every tensor of the component, file by file, in the
    index's order when there is one."""
    from safetensors import safe_open

    from difflet.backends._torch_megatron.checkpoint import build_weight_map

    weight_map = build_weight_map(model_dir)
    by_file: dict[str, list[str]] = {}
    for key, path in weight_map.items():
        by_file.setdefault(path, []).append(key)
    for path in sorted(by_file):
        with safe_open(path, framework="pt") as handle:
            for key in sorted(by_file[path]):
                yield key, handle.get_tensor(key), Path(path).name


def quantize_tensors(model_dir: Path, spec: QuantSpec, calibration: Calibration
                     ) -> Iterator[tuple[str, torch.Tensor, bool]]:
    """Every tensor of the quantized component in write order: ``(key, tensor, quantized)``.
    A target weight yields three entries (``.weight`` fp8, ``.weight_scale``, ``.input_scale``)."""
    for key, tensor, _ in iter_source_tensors(model_dir):
        prefix = _tensor_key_prefix(key)
        if prefix is not None and spec.matches(prefix) and tensor.dim() == 2:
            weight_q, weight_scale = quantize_weight(tensor, fp8_max=spec.fp8_max)
            yield key, weight_q, True
            yield f"{prefix}.weight_scale", weight_scale, True
            yield f"{prefix}.input_scale", calibration.input_scale(prefix, spec), True
        else:
            yield key, tensor, False


def source_bytes(model_dir: str | Path) -> int:
    """Bytes of every tensor in the component's safetensors files, from the headers only."""
    import struct

    from safetensors import safe_open  # noqa: F401  (the files are safetensors; headers read raw)

    total = 0
    for path in sorted(Path(model_dir).glob("*.safetensors")):
        with open(path, "rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(length))
        for name, info in header.items():
            if name != "__metadata__":
                lo, hi = info["data_offsets"]
                total += hi - lo
    return total


class _ShardWriter:
    """Writes tensors into ``shard_bytes``-sized safetensors shards as they arrive (host memory
    holds one shard at a time), then names them ``-0000i-of-0000n`` and writes the index."""

    def __init__(self, out_dir: Path, shard_bytes: int) -> None:
        self.out_dir = out_dir
        self.shard_bytes = shard_bytes
        self.pending: dict[str, torch.Tensor] = {}
        self.pending_bytes = 0
        self.written: list[tuple[Path, list[str], int]] = []

    def add(self, key: str, tensor: torch.Tensor) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        if self.pending and self.pending_bytes + nbytes > self.shard_bytes:
            self._flush()
        self.pending[key] = tensor.contiguous()
        self.pending_bytes += nbytes

    def _flush(self) -> None:
        from safetensors.torch import save_file

        if not self.pending:
            return
        path = self.out_dir / f"shard-{len(self.written):05d}.part"
        save_file(self.pending, str(path), metadata={"format": "pt"})
        self.written.append((path, list(self.pending), self.pending_bytes))
        self.pending, self.pending_bytes = {}, 0

    def finish(self) -> tuple[dict[str, str], int]:
        """Name the shards, write the index; returns ``(weight_map, total bytes)``."""
        self._flush()
        n = len(self.written)
        weight_map: dict[str, str] = {}
        total = 0
        for i, (path, keys, nbytes) in enumerate(self.written, start=1):
            name = _SHARD.format(i=i, n=n)
            path.rename(self.out_dir / name)
            for key in keys:
                weight_map[key] = name
            total += nbytes
        index = {"metadata": {"total_size": total}, "weight_map": weight_map}
        (self.out_dir / _INDEX).write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        return weight_map, total


def write_quantized_dir(model_dir: str | Path, out_dir: str | Path, spec: QuantSpec,
                        calibration: Calibration, *, shard_bytes: int = SHARD_BYTES
                        ) -> dict[str, Any]:
    """Quantize ``model_dir`` into ``out_dir`` (created; must not already hold a manifest) and
    return the manifest. Written to a temporary sibling and renamed, so a reader never sees a
    half-written directory."""
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    if spec.calibration_sha256 is not None and spec.calibration_sha256 != calibration.sha256:
        raise ValueError("spec.calibration_sha256 does not match the calibration file")
    if (out_dir / MANIFEST).exists():
        raise FileExistsError(f"{out_dir} already holds a quantized checkpoint")
    tmp = out_dir.with_name(f".{out_dir.name}.tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        writer = _ShardWriter(tmp, shard_bytes)
        quantized, kept = [], 0
        for key, tensor, is_q in quantize_tensors(model_dir, spec, calibration):
            writer.add(key, tensor)
            if is_q:
                if key.endswith(".weight"):
                    quantized.append(strip_prefix(key[: -len(".weight")]))
            else:
                kept += 1
        weight_map, total = writer.finish()
        for path in sorted(model_dir.iterdir()):
            if path.is_file() and not path.name.endswith(".safetensors") \
                    and not path.name.endswith(".safetensors.index.json"):
                shutil.copy2(path, tmp / path.name)
        manifest = {
            "schema": spec.schema,
            "spec": spec.to_dict(),
            "source": {"path": str(model_dir), "fingerprint": source_fingerprint(model_dir),
                       "bytes": source_bytes(model_dir)},
            "calibration": {"sha256": calibration.sha256, "path": str(calibration.path),
                            **calibration.provenance},
            "quantized_linears": quantized,
            "tensors": {"quantized": len(quantized), "kept": kept, "total": len(weight_map)},
            "bytes": total,
            "files": sorted(set(weight_map.values())),
            "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (tmp / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return manifest


def read_manifest(quantized: str | Path) -> dict[str, Any]:
    return json.loads((Path(quantized) / MANIFEST).read_text())


def check_quantized_dir(quantized: str | Path, model_dir: str | Path, spec: QuantSpec,
                        calibration_sha256: str | None) -> dict[str, Any]:
    """The manifest of ``quantized`` when it was built from ``model_dir`` with ``spec`` and this
    calibration; raises ``ValueError`` naming the first mismatch otherwise."""
    quantized = Path(quantized)
    try:
        manifest = read_manifest(quantized)
    except FileNotFoundError:
        raise ValueError(f"{quantized}: no {MANIFEST}; the quantized checkpoint was never "
                         "written (run `difflet compile` with the same --quant flags)") from None
    if manifest.get("schema") != spec.schema:
        raise ValueError(f"{quantized}: schema {manifest.get('schema')} != {spec.schema}")
    if QuantSpec.from_dict(manifest["spec"]).weights_identity() != spec.weights_identity():
        raise ValueError(f"{quantized}: built with another quantization spec")
    if manifest["source"]["fingerprint"] != source_fingerprint(model_dir):
        raise ValueError(f"{quantized}: built from another checkpoint than {model_dir}")
    if calibration_sha256 is not None and manifest["calibration"]["sha256"] != calibration_sha256:
        raise ValueError(f"{quantized}: built with another calibration file")
    for name in manifest["files"] + [_INDEX]:
        if not (quantized / name).is_file():
            raise ValueError(f"{quantized}: {name} is missing")
    return manifest


def ensure_quantized_dir(cache_root: str | Path, model_dir: str | Path, spec: QuantSpec,
                         calibration: Calibration, *, create: bool) -> Path:
    """The valid quantized directory for ``model_dir``/``spec``, written when ``create`` and
    absent (``difflet compile`` creates; ``generate`` only resolves)."""
    target = quantized_dir(cache_root, model_dir, spec)
    try:
        check_quantized_dir(target, model_dir, spec, calibration.sha256)
        return target
    except ValueError as exc:
        if not create:
            raise ValueError(f"{exc}; run `difflet compile` with the same --quant flags first"
                             ) from None
        if target.exists():
            shutil.rmtree(target)
    write_quantized_dir(model_dir, target, spec, calibration)
    return target


def quantized_linear_names(quantized: str | Path) -> list[str]:
    return list(read_manifest(quantized)["quantized_linears"])


def default_cache_root() -> Path:
    """Where quantized directories live by default: beside the neuron NEFF cache and the
    weight store, under ``DIFFLET_COMPILE_CACHE`` (the CLI's ``--cache-dir``) or
    ``~/.cache/difflet``."""
    from difflet import envs

    cache = envs.DIFFLET_COMPILE_CACHE or "~/.cache/difflet"
    return Path(os.path.abspath(os.path.expanduser(cache)))


def prepare_quantized_checkpoint(model_dir: str | Path, spec: QuantSpec,
                                 calibration_path: str | Path, *, create: bool,
                                 cache_root: str | Path | None = None
                                 ) -> tuple[Path, QuantSpec]:
    """The quantized directory for ``model_dir`` and the spec completed with the calibration's
    sha256: what an application needs. ``create`` writes the directory when it is missing (the
    CLI's ``compile``, on the driver); without it a missing directory is an error naming the
    fix (every rank of a ``generate`` resolves, none writes)."""
    from difflet.quant.calibration import read_calibration

    calibration = read_calibration(calibration_path)
    if spec.calibration_sha256 is None:
        spec = dataclasses.replace(spec, calibration_sha256=calibration.sha256)
    elif spec.calibration_sha256 != calibration.sha256:
        raise ValueError(f"{calibration_path}: the calibration file changed since the spec "
                         "was made (sha256 mismatch)")
    root = default_cache_root() if cache_root is None else Path(cache_root)
    return ensure_quantized_dir(root, model_dir, spec, calibration, create=create), spec


def is_fp8(tensor: torch.Tensor) -> bool:
    return tensor.dtype == FP8_DTYPE


__all__ = [
    "MANIFEST",
    "SHARD_BYTES",
    "check_quantized_dir",
    "default_cache_root",
    "ensure_quantized_dir",
    "prepare_quantized_checkpoint",
    "iter_source_tensors",
    "quantize_tensors",
    "quantized_digest",
    "quantized_dir",
    "quantized_linear_names",
    "read_manifest",
    "source_fingerprint",
    "write_quantized_dir",
]
