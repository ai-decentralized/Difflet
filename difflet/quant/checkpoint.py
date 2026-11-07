"""The offline quantized checkpoint: a diffusers-layout directory with mxfp8 target weights.

Why a directory and not quantize-at-load: the native backend loads each rank's shards straight
from safetensors onto the device (``difflet/backends/neuron/core/checkpoint.py``) and caches them
in its pre-sharded weight store; both key on the files. A quantized directory with the same
layout and key names as the source lets that machinery load fp8 weights unchanged, and makes the
quantized bytes an artifact the cache keys can name (the bf16 copies that
``core/weights_bf16.py`` makes for Wan are the precedent).

Layout of ``<cache>/quantized/<source label>/<component>/mxfp8-<digest>/``:

* ``diffusion_pytorch_model-0000i-of-0000n.safetensors`` + ``.index.json``: every tensor of the
  source component. A target ``<prefix>.weight`` is stored as ``float8_e4m3fn`` ``[N, K]`` with
  its uint8 E8M0 block scales ``<prefix>.weight_scale`` ``[N, K/32]`` beside it (a row-parallel
  target quantized per rank K range, ``mx.py``); everything else (biases, the bf16 modulation
  and embedder weights, norms) is copied byte for byte. A linear the spec splits
  (``QuantSpec.split_linears``: the single-stream ``proj_out`` ``[N, K]`` over
  ``cat([attn, mlp])``) is not stored fused: ``<prefix>_attn.weight`` holds its input columns
  ``[0, N)`` and ``<prefix>_mlp.weight`` the rest, the parameters the neuron model runs, and
  each half is a target of its own (quantized per rank K range of the half when row-parallel)
  or the source's column range in the source's dtype. ``<prefix>.bias`` is copied as is.
* ``config.json`` and the other non-tensor files of the source, copied; a bf16 copy's own
  manifest excepted, since the quantized directory is no bf16 copy of anything.
* ``difflet_quant.json``: the spec, the source fingerprint and per-tensor counts, and for a
  source that is a bf16 copy the fp32 directory it was made from (``source.bf16_copy_of``), so
  the manifest on device says which hub snapshot the directory came from.
  :func:`check_quantized_dir` refuses a directory whose manifest does not match what the caller
  asks for.

A spec with a target pattern that matches no 2-D weight of the source (a split one counting as
its two halves), or a split pattern that matches none, is refused before anything is written,
naming the patterns (:func:`write_quantized_dir`): the directory would hold less than the spec
claims.

The digest hashes the spec's weights identity and the source fingerprint (file names, sizes and
Hugging Face blob names, which are content hashes), so another checkpoint, another target set
or another TP degree is another directory. A source that is a bf16 copy (Wan's fp32 hub DiTs)
adds its copy manifest's sha256: the copy's files carry no blob name, and the manifest is what
names the fp32 snapshot the copy, and so the quantized directory, were built from.
"""

from __future__ import annotations

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

from difflet.quant import mx
from difflet.quant.spec import QuantSpec, strip_prefix

MANIFEST = "difflet_quant.json"
#: The manifest a bf16 copy (``core/weights_bf16.py``, which takes the name from here) writes
#: beside its files. Named here, torch-free, so that fingerprinting a copy on every rank and on
#: the driver imports no backend package (the neuron package's ``__init__`` loads the runtime).
BF16_MANIFEST = "difflet-bf16-manifest.json"
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
    file's sha256. No mtime of the files themselves: a copied checkpoint is the same checkpoint.

    A bf16 copy (``core/weights_bf16.py``) ends with its manifest's sha256: the copy writes its
    own safetensors, so their names say nothing about the hub snapshot, and the manifest (the
    fp32 source's resolved path and its files' names, sizes and mtimes) is the copy's own tie to
    it. A remade copy is another manifest, so the quantized directory built from the old one is
    another checkpoint's; this is where mtimes do enter: a source moved or re-stamped (an rsync
    without ``-t``) remakes the copy at the next compile and the quantized directory with it,
    and the old one stays on disk at its old digest. A directory without a manifest (every hub
    checkpoint) keeps the fingerprint it had, so every digest written before the entry existed
    still resolves."""
    model_dir = Path(model_dir)
    entries: list[list[Any]] = []
    for path in sorted(model_dir.glob("*.safetensors")):
        entries.append([path.name, path.stat().st_size,
                        os.path.realpath(path).rsplit(os.sep, 1)[-1]])
    for path in sorted(model_dir.glob("*.safetensors.index.json")):
        entries.append([path.name, hashlib.sha256(path.read_bytes()).hexdigest()])
    if not entries:
        raise FileNotFoundError(f"no .safetensors under {model_dir}")
    manifest = model_dir / BF16_MANIFEST
    if manifest.is_file():
        entries.append([manifest.name, hashlib.sha256(manifest.read_bytes()).hexdigest()])
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


def _bf16_copy_source(model_dir: Path) -> Path | None:
    """The fp32 directory a bf16 copy (``core/weights_bf16.py``) was made from, from its
    manifest; None for anything else (every hub checkpoint)."""
    manifest = model_dir / BF16_MANIFEST
    if not manifest.is_file():
        return None
    return Path(json.loads(manifest.read_text())["source"])


def _source_label(model_dir: Path) -> str:
    """``<repo>__<snapshot8>`` for a Hugging Face cache path, else the parent directory's name.
    A bf16 copy is labelled by its fp32 source: the copy's own path,
    ``weights-bf16/<snapshot>/<component>``, names no repo, and the label is what the driver's
    ``[<tag>] MXFP8 <component> (<compute>): <dir>`` lines and the cache layout show."""
    model_dir = _bf16_copy_source(model_dir) or model_dir
    parts = model_dir.resolve().parts
    for i, part in enumerate(parts):
        if part.startswith("models--") and i + 2 < len(parts) and parts[i + 1] == "snapshots":
            return f"{part.removeprefix('models--')}__{parts[i + 2][:8]}"
    return model_dir.resolve().parent.name


# ------------------------------------------------------------------------ writing


def _tensor_key_prefix(key: str) -> str | None:
    return key[: -len(".weight")] if key.endswith(".weight") else None


def _keys_by_file(model_dir: Path) -> dict[str, list[str]]:
    """The component's tensor keys by the safetensors file holding them (the index's, when
    there is one)."""
    from difflet.backends._torch_megatron.checkpoint import build_weight_map

    by_file: dict[str, list[str]] = {}
    for key, path in build_weight_map(model_dir).items():
        by_file.setdefault(path, []).append(key)
    return by_file


def iter_source_tensors(model_dir: Path) -> Iterator[tuple[str, torch.Tensor, str]]:
    """``(key, tensor, file name)`` for every tensor of the component, file by file, in the
    index's order when there is one."""
    from safetensors import safe_open

    by_file = _keys_by_file(model_dir)
    for path in sorted(by_file):
        with safe_open(path, framework="pt") as handle:
            for key in sorted(by_file[path]):
                yield key, handle.get_tensor(key), Path(path).name


def _weight_shapes(model_dir: Path) -> dict[str, list[int]]:
    """The shape of every 2-D ``<prefix>.weight`` of the component by its prefix, from the
    safetensors headers alone (no tensor data is read)."""
    from safetensors import safe_open

    by_file = _keys_by_file(model_dir)
    shapes: dict[str, list[int]] = {}
    for path in sorted(by_file):
        with safe_open(path, framework="pt") as handle:
            for key in sorted(by_file[path]):
                prefix = _tensor_key_prefix(key)
                shape = handle.get_slice(key).get_shape()
                if prefix is not None and len(shape) == 2:
                    shapes[prefix] = list(shape)
    return shapes


def _split_parts(prefix: str, shape) -> list[tuple[str, int, int]]:
    """``(name, first input column, width)`` of the two halves of the split linear ``prefix``
    ``[N, K]``: the attention half reads the first N input columns, the MLP half the rest."""
    n, k = (int(d) for d in shape)
    if k <= n:
        raise ValueError(f"{prefix}: a split linear needs K > N (its first N input columns are "
                         f"the attention half), got [{n}, {k}]")
    return [(f"{prefix}_attn", 0, n), (f"{prefix}_mlp", n, k - n)]


def _quantizable_prefixes(shapes: dict[str, list[int]], spec: QuantSpec) -> list[str]:
    """The key prefixes ``quantize_tensors`` can quantize under ``spec``: every 2-D weight, a
    split one as its two halves (the fused name is not written, so no target can match it)."""
    prefixes: list[str] = []
    for prefix, shape in shapes.items():
        if spec.is_split(prefix):
            prefixes.extend(part for part, _, _ in _split_parts(prefix, shape))
        else:
            prefixes.append(prefix)
    return prefixes


def _quantize_weight(prefix: str, weight: torch.Tensor, spec: QuantSpec
                     ) -> Iterator[tuple[str, torch.Tensor, bool]]:
    k_shards = spec.k_shards(prefix)
    try:
        weight_q, weight_scale = mx.quantize_mx_sharded(weight, k_shards)
    except ValueError as exc:
        # The writer runs on the driver before any rank builds a layer, so a K the MX layout
        # cannot take at this tp is refused here, naming the layer.
        split = f" (K={weight.shape[-1]} split over tp={k_shards})" if k_shards > 1 else ""
        raise ValueError(f"{prefix}{split}: {exc}") from None
    yield f"{prefix}.weight", weight_q, True
    yield f"{prefix}.weight_scale", weight_scale, True


def quantize_tensors(model_dir: Path, spec: QuantSpec) -> Iterator[tuple[str, torch.Tensor, bool]]:
    """Every tensor of the quantized component in write order: ``(key, tensor, quantized)``.
    A target weight yields ``.weight`` (fp8) and ``.weight_scale`` (uint8 E8M0 ``[N, K/32]``, a
    row-parallel target quantized per rank K range). A split linear yields its halves in its
    place, each quantized on its own (per rank K range of the half: its own shard count) when a
    target names it, else the source's column range, uncast."""
    for key, tensor, _ in iter_source_tensors(model_dir):
        prefix = _tensor_key_prefix(key)
        if prefix is not None and tensor.dim() == 2 and spec.is_split(prefix):
            for part, start, width in _split_parts(prefix, tensor.shape):
                half = tensor[:, start:start + width]
                if spec.matches(part):
                    yield from _quantize_weight(part, half, spec)
                else:
                    yield f"{part}.weight", half, False
        elif prefix is not None and tensor.dim() == 2 and spec.matches(prefix):
            yield from _quantize_weight(prefix, tensor, spec)
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


def write_quantized_dir(model_dir: str | Path, out_dir: str | Path, spec: QuantSpec, *,
                        shard_bytes: int = SHARD_BYTES) -> dict[str, Any]:
    """Quantize ``model_dir`` into ``out_dir`` (created; must not already hold a manifest) and
    return the manifest. Written to a temporary sibling and renamed, so a reader never sees a
    half-written directory."""
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    if (out_dir / MANIFEST).exists():
        raise FileExistsError(f"{out_dir} already holds a quantized checkpoint")
    shapes = _weight_shapes(model_dir)
    dead = spec.unmatched_splits(shapes)
    if dead:
        # Before the target check: a dead split also leaves every target on its halves dead,
        # and the split is the pattern to name.
        raise ValueError(f"split patterns match no linear weight in {model_dir}: "
                         f"{'; '.join(dead)}")
    dead = spec.unmatched_targets(_quantizable_prefixes(shapes, spec))
    if dead:
        # Refused here, on the driver, before anything is quantized: the directory would hold
        # less than the spec claims, and every rank would refuse it (``quantize_module_``) only
        # after the driver spent minutes writing it. The patterns verbatim, separated as
        # ``DIFFLET_QUANT_TARGETS`` separates them.
        raise ValueError(f"target patterns match no linear weight in {model_dir}: "
                         f"{'; '.join(dead)}")
    tmp = out_dir.with_name(f".{out_dir.name}.tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        writer = _ShardWriter(tmp, shard_bytes)
        quantized, kept = [], 0
        for key, tensor, is_q in quantize_tensors(model_dir, spec):
            writer.add(key, tensor)
            if is_q:
                if key.endswith(".weight"):
                    quantized.append(strip_prefix(key[: -len(".weight")]))
            else:
                kept += 1
        weight_map, total = writer.finish()
        for path in sorted(model_dir.iterdir()):
            if path.is_file() and path.name != BF16_MANIFEST \
                    and not path.name.endswith(".safetensors") \
                    and not path.name.endswith(".safetensors.index.json"):
                shutil.copy2(path, tmp / path.name)
        source = {"path": str(model_dir), "fingerprint": source_fingerprint(model_dir),
                  "bytes": source_bytes(model_dir)}
        fp32_source = _bf16_copy_source(model_dir)
        if fp32_source is not None:
            # The copy's manifest is not carried over (it would mark this directory as a bf16
            # copy of the fp32 source, which it is not); the snapshot it names is kept here.
            source["bf16_copy_of"] = str(fp32_source)
        manifest = {
            "schema": spec.schema,
            "spec": spec.to_dict(),
            "source": source,
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


def check_quantized_dir(quantized: str | Path, model_dir: str | Path, spec: QuantSpec
                        ) -> dict[str, Any]:
    """The manifest of ``quantized`` when it was built from ``model_dir`` with ``spec``; raises
    ``ValueError`` naming the first mismatch otherwise."""
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
    for name in manifest["files"] + [_INDEX]:
        if not (quantized / name).is_file():
            raise ValueError(f"{quantized}: {name} is missing")
    return manifest


def ensure_quantized_dir(cache_root: str | Path, model_dir: str | Path, spec: QuantSpec, *,
                         create: bool) -> Path:
    """The valid quantized directory for ``model_dir``/``spec``, written when ``create`` and
    absent (``difflet compile`` creates; ``generate`` only resolves)."""
    target = quantized_dir(cache_root, model_dir, spec)
    try:
        check_quantized_dir(target, model_dir, spec)
        return target
    except ValueError as exc:
        if not create:
            raise ValueError(f"{exc}; run `difflet compile` with the same --quant flags first"
                             ) from None
        if target.exists():
            shutil.rmtree(target)
    write_quantized_dir(model_dir, target, spec)
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


def prepare_quantized_checkpoint(model_dir: str | Path, spec: QuantSpec, *, create: bool,
                                 cache_root: str | Path | None = None) -> Path:
    """The quantized directory for ``model_dir``: what an application needs. ``create`` writes
    the directory when it is missing (the CLI's ``compile``, on the driver); without it a
    missing directory is an error naming the fix (every rank of a ``generate`` resolves, none
    writes)."""
    root = default_cache_root() if cache_root is None else Path(cache_root)
    return ensure_quantized_dir(root, model_dir, spec, create=create)


def is_fp8(tensor: torch.Tensor) -> bool:
    return tensor.dtype == mx.FP8_DTYPE


__all__ = [
    "BF16_MANIFEST",
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
