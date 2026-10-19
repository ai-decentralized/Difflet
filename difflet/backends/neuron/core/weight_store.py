"""Pre-sharded weight store of the neuron backend (TorchNeuron, one process per NeuronCore).

Why: without it every launch reads the diffusers safetensors and casts them on the host. For
Wan2.1 14B that is the 57 GB fp32 DiT, read off a ~130 MB/s EBS volume and cast to bf16 by
every rank, slice by slice. The AoT backend reads pre-sharded bf16 files instead
(``difflet/backends/trainium/core/shared_weights.py``); this is the same idea for the
per-rank lazy loader (``core/checkpoint.py``).

What it stores: for one component (a checkpoint directory loaded by one
``TorchNeuronApplicationBase``), exactly the host tensors the loader copies onto the device,
in their load dtype (bf16 for a bf16 DiT from an fp32 checkpoint; an fp32 umT5 would stay
fp32, but is not written by default, see the flow below):

* ``replicated.safetensors``: the tensors every rank holds whole (no shard declaration, or
  tp=1), written once and read by every rank, so replicated tensors cost their bytes once;
* ``tp<r>-of-<n>.safetensors``: TP rank ``r``'s slices of the sharded tensors. It is keyed by
  the TP rank, not the global rank: CP, CFG and DP replicas of one TP rank load the same bytes
  (the loader slices by TP rank only), so they share the file.

Each data file has a ``<file>.manifest.json`` beside it: the store format, the entry digest,
the tensors (name, dtype, shape) in file order, the byte size, the sha256 of the safetensors
header and of 64 KiB samples of the data, and the load report of the checkpoint load that
wrote it.

Key: the entry directory is ``<label>__<digest>``, where the digest hashes the store format,
the checkpoint's content fingerprint (every ``*.safetensors`` and index file in the component
directory: name, size, mtime and the resolved blob name, which in a Hugging Face cache is the
file's content hash; the index files' sha256), the load dtype, the TP degree, and the layout of
the meta-built module: every persistent tensor's name, checkpoint key, per-rank shape, dtype
and shard dim. So a num_layers truncation, another shard declaration, another rename or
another TP degree is another entry. The label (model, component, revision, dtype, tp) only
makes ``ls`` readable.

Location: ``$DIFFLET_NEURON_WEIGHT_STORE`` if set to a path; off when set to ``0``, ``off``,
``false`` or ``none``; else ``<DIFFLET_COMPILE_CACHE>/neuron/weights`` (beside the NEFF
cache, which the CLI roots at ``--cache-dir``).

Flow (``load_with_store``, called by ``TorchNeuronApplicationBase.load``):

1. hit: every file this rank needs has a manifest that matches the module and the file
   (size, header, samples): the tensors are read in file order, in large sequential reads
   with one read ahead, and copied onto the device. No checkpoint read, no cast.
2. miss: the checkpoint loads as before. When the entry is at least ``MIN_SAVING`` smaller than
   what the checkpoint load reads (an fp32 checkpoint loaded in bf16; or any component with
   ``DIFFLET_NEURON_WEIGHT_STORE_ALL=1``, for a store on a faster disk than the checkpoint),
   each host tensor the loader copies to the device is also written to the store at its
   precomputed offset. The files are written to a temp
   name, fsynced and renamed, then the manifest is written the same way. A rank writes a file
   only while it holds the file's ``flock`` (a concurrent launch or a CP/DP replica of the same
   TP rank skips it), only when the disk keeps ``RESERVE_BYTES`` free after the whole entry,
   and only after re-checking under the lock that nobody wrote it meanwhile.

Any store problem (an invalid manifest, a failed read or write) falls back to the checkpoint
load and, for a bad entry, rewrites it; the store never fails a load. Both ``difflet compile``
and the first ``generate`` without one populate the store, since both run the full load.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import struct
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from difflet import envs

logger = logging.getLogger(__name__)

__all__ = [
    "RESERVE_BYTES",
    "STORE_ENV",
    "STORE_FORMAT",
    "StoreResult",
    "load_with_store",
    "store_root",
]

#: Bump when the key inputs or the file layout change: old entries are then never matched.
STORE_FORMAT = 1
STORE_ENV = "DIFFLET_NEURON_WEIGHT_STORE"
_OFF_VALUES = ("0", "off", "false", "none", "no")
ALL_ENV = "DIFFLET_NEURON_WEIGHT_STORE_ALL"
#: Free disk a write must leave behind, after the whole entry.
RESERVE_BYTES = 10 * 1024**3
#: An entry is written only when it is at least this much smaller than what the checkpoint
#: load reads for the same tensors (unless DIFFLET_NEURON_WEIGHT_STORE_ALL is set).
MIN_SAVING = 0.1
_SAMPLES = 8
_SAMPLE_BYTES = 64 * 1024
_REPLICATED = "replicated"

_DTYPES: dict[torch.dtype, str] = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
_ITEMSIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
             "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def store_root() -> Path | None:
    """Where the store lives, or None when ``DIFFLET_NEURON_WEIGHT_STORE`` switches it off."""
    raw = (envs.DIFFLET_NEURON_WEIGHT_STORE or "").strip()
    if raw.lower() in _OFF_VALUES:
        return None
    if raw:
        return Path(os.path.abspath(os.path.expanduser(raw)))
    cache = envs.DIFFLET_COMPILE_CACHE or "~/.cache/difflet"
    return Path(os.path.abspath(os.path.expanduser(cache))) / "neuron" / "weights"


# ------------------------------------------------------------------------------------ specs


@dataclass(frozen=True)
class _Spec:
    name: str  # module name
    key: str  # checkpoint key
    shape: tuple[int, ...]
    dtype: torch.dtype  # the dtype the loader copies in (the target's, after materialisation)
    shard_dim: int | None  # None: replicated at this tp

    @property
    def nbytes(self) -> int:
        numel = 1
        for n in self.shape:
            numel *= n
        return numel * self.dtype.itemsize


def _module_specs(module: nn.Module, *, dtype: torch.dtype, tp_size: int, prefix: str,
                  rename: Callable[[str], str] | None) -> list[_Spec]:
    """The tensors ``load_checkpoint_into`` fills, in its order, as they are on the device:
    meta parameters in ``dtype`` (``materialize_meta_``), everything else in its own dtype."""
    from difflet.backends.tpu.core.weights import shard_dim

    persistent = set(module.state_dict().keys())
    specs = []
    for name, tensor in [*module.named_parameters(), *module.named_buffers()]:
        if name not in persistent:
            continue
        key = name[len(prefix):] if prefix and name.startswith(prefix) else name
        if rename is not None:
            key = rename(key)
        is_param = isinstance(tensor, nn.Parameter)
        target_dtype = dtype if (is_param and tensor.is_meta) else tensor.dtype
        dim = shard_dim(module, name) if tp_size > 1 else None
        specs.append(_Spec(name, key, tuple(int(n) for n in tensor.shape), target_dtype, dim))
    return specs


def _checkpoint_bytes(model_dir: Path, specs: list[_Spec]) -> int | None:
    """The bytes the checkpoint load reads for ``specs`` (this rank's slices, in the
    checkpoint's own dtypes), from the safetensors headers; None when a key or dtype is not
    found there."""
    from difflet.backends.tpu.core.checkpoint import build_weight_map

    weight_map = build_weight_map(model_dir)
    headers: dict[str, dict[str, Any]] = {}
    total = 0
    for spec in specs:
        path = weight_map.get(spec.key)
        if path is None:
            return None
        if path not in headers:
            with open(path, "rb") as f:
                (length,) = struct.unpack("<Q", f.read(8))
                headers[path] = json.loads(f.read(length))
        itemsize = _ITEMSIZE.get((headers[path].get(spec.key) or {}).get("dtype"))
        if itemsize is None:
            return None
        total += spec.nbytes // spec.dtype.itemsize * itemsize
    return total


def _write_everything() -> bool:
    return (envs.DIFFLET_NEURON_WEIGHT_STORE_ALL or "").strip().lower() not in ("", *_OFF_VALUES)


def _fingerprint(model_dir: Path) -> list[list[Any]]:
    """The checkpoint's identity without reading its data: every safetensors file (name, size,
    mtime, resolved blob name: a Hugging Face blob is named by its content hash) and every
    index file's sha256."""
    entries: list[list[Any]] = []
    for path in sorted(model_dir.glob("*.safetensors")):
        stat = path.stat()
        entries.append([path.name, stat.st_size, stat.st_mtime_ns, os.path.realpath(path).rsplit(
            os.sep, 1)[-1]])
    for path in sorted(model_dir.glob("*.safetensors.index.json")):
        entries.append([path.name, hashlib.sha256(path.read_bytes()).hexdigest()])
    return entries


_SAFE_LABEL = re.compile(r"[^A-Za-z0-9._-]+")


def _label(model_dir: Path, dtype: torch.dtype, tp_size: int) -> str:
    from difflet.backends.trainium.core.shared_weights import _describe_source

    bits = [*_describe_source(str(model_dir)), str(dtype).removeprefix("torch."), f"tp{tp_size}"]
    label = _SAFE_LABEL.sub("-", "__".join(str(b) for b in bits if b)).strip("-_")
    return label[:180] or "component"


# ------------------------------------------------------------------------- safetensors file


def _header_bytes(specs: list[_Spec]) -> tuple[bytes, int]:
    """A safetensors header for ``specs`` laid out back to back in order, padded to 8 bytes,
    with its 8-byte length prefix; and the total file size."""
    header: dict[str, Any] = {"__metadata__": {"format": "pt", "difflet_store": str(STORE_FORMAT)}}
    offset = 0
    for spec in specs:
        header[spec.name] = {"dtype": _DTYPES[spec.dtype], "shape": list(spec.shape),
                             "data_offsets": [offset, offset + spec.nbytes]}
        offset += spec.nbytes
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    prefix = struct.pack("<Q", len(raw)) + raw
    return prefix, len(prefix) + offset


def _read_header(fd: int) -> tuple[bytes, dict[str, Any]]:
    head = os.pread(fd, 8, 0)
    if len(head) != 8:
        raise ValueError("file shorter than a safetensors header")
    (length,) = struct.unpack("<Q", head)
    if length > 100 * 1024**2:
        raise ValueError(f"implausible safetensors header length {length}")
    raw = os.pread(fd, length, 8)
    if len(raw) != length:
        raise ValueError("truncated safetensors header")
    return head + raw, json.loads(raw)


def _samples_digest(fd: int, data_start: int, size: int) -> str:
    """sha256 of ``_SAMPLES`` evenly spaced 64 KiB windows of the data, and of its last one."""
    digest = hashlib.sha256()
    span = max(size - data_start, 0)
    starts = sorted({data_start + (span * i) // _SAMPLES for i in range(_SAMPLES)}
                    | {max(data_start, size - _SAMPLE_BYTES)})
    for start in starts:
        digest.update(os.pread(fd, min(_SAMPLE_BYTES, size - start), start))
    return digest.hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> memoryview:
    flat = tensor.detach().contiguous().reshape(-1)
    return memoryview(flat.view(torch.uint8).numpy())


def _pwrite_all(fd: int, data: memoryview, offset: int) -> None:
    while len(data):
        written = os.pwrite(fd, data, offset)
        data, offset = data[written:], offset + written


def _fsync_dir(directory: Path) -> None:
    with contextlib.suppress(OSError):
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _write_json_atomic(path: Path, body: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(body, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
    _fsync_dir(path.parent)


# -------------------------------------------------------------------------------- one file


class _StoreFile:
    """One data file of an entry and its manifest."""

    def __init__(self, directory: Path, stem: str, specs: list[_Spec], digest: str) -> None:
        self.directory = directory
        self.path = directory / f"{stem}.safetensors"
        self.manifest = directory / f"{stem}.safetensors.manifest.json"
        self.lock = directory / f"{stem}.safetensors.lock"
        self.specs = specs
        self.digest = digest
        self.header, self.size = _header_bytes(specs)

    def expected_tensors(self) -> list[list[Any]]:
        return [[s.name, _DTYPES[s.dtype], list(s.shape)] for s in self.specs]

    def check(self) -> tuple[dict[str, Any] | None, str]:
        """``(manifest, "")`` when the file is complete and matches the module, else
        ``(None, why)``."""
        try:
            body = json.loads(self.manifest.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None, "no entry"
        except (OSError, ValueError) as exc:
            return None, f"unreadable manifest ({exc})"
        if body.get("format") != STORE_FORMAT or body.get("digest") != self.digest:
            return None, "manifest of another store format or key"
        if body.get("tensors") != self.expected_tensors():
            return None, "manifest tensors differ from the module's"
        try:
            fd = os.open(self.path, os.O_RDONLY)
        except OSError as exc:
            return None, f"data file unreadable ({exc.strerror})"
        try:
            size = os.fstat(fd).st_size
            if size != body.get("bytes") or size != self.size:
                return None, f"data file is {size} bytes, manifest says {body.get('bytes')}"
            header = os.pread(fd, len(self.header), 0)  # the module's header, byte for byte
            if header != self.header:
                return None, "data file header differs"
            if hashlib.sha256(header).hexdigest() != body.get("header_sha256"):
                return None, "header checksum mismatch"
            if _samples_digest(fd, len(header), size) != body.get("samples_sha256"):
                return None, "data checksum mismatch"
        except (OSError, ValueError) as exc:
            return None, f"data file unreadable ({exc})"
        finally:
            os.close(fd)
        return body, ""

    def read_into(self, targets: dict[str, torch.Tensor]) -> None:
        """Copy every tensor of the file into ``targets[name]``, reading in file order with
        one tensor read ahead while the previous one is copied."""
        fd = os.open(self.path, os.O_RDONLY)
        try:
            with contextlib.suppress(AttributeError, OSError):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
            offset = len(self.header)
            jobs = []
            for spec in self.specs:
                jobs.append((spec, offset))
                offset += spec.nbytes

            def read(job):
                spec, start = job
                buf = torch.empty(spec.nbytes, dtype=torch.uint8)
                view = memoryview(buf.numpy())
                got = 0
                while got < spec.nbytes:
                    n = os.preadv(fd, [view[got:]], start + got)
                    if n <= 0:
                        raise OSError(f"{self.path}: short read at {start + got}")
                    got += n
                return buf.view(spec.dtype).reshape(spec.shape)

            with ThreadPoolExecutor(max_workers=1) as pool, torch.no_grad():
                pending = pool.submit(read, jobs[0]) if jobs else None
                for i, (spec, _) in enumerate(jobs):
                    tensor = pending.result()
                    pending = pool.submit(read, jobs[i + 1]) if i + 1 < len(jobs) else None
                    targets[spec.name].copy_(tensor)
        finally:
            os.close(fd)

    @contextlib.contextmanager
    def locked(self) -> Iterator[bool]:
        """Hold the file's ``flock`` (True), or yield False when another process holds it."""
        fd = os.open(self.lock, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class _FileWriter:
    """Writes one ``_StoreFile`` while the checkpoint loads: each tensor at its offset."""

    def __init__(self, store_file: _StoreFile) -> None:
        self.file = store_file
        self.offsets: dict[str, tuple[int, _Spec]] = {}
        offset = len(store_file.header)
        for spec in store_file.specs:
            self.offsets[spec.name] = (offset, spec)
            offset += spec.nbytes
        self.pending = set(self.offsets)
        # Stale temp files of a writer that died: we hold the lock, nobody else writes them.
        for stale in store_file.directory.glob(f".{store_file.path.name}.tmp-*"):
            with contextlib.suppress(OSError):
                stale.unlink()
        with contextlib.suppress(FileNotFoundError):
            store_file.manifest.unlink()  # an old entry is never valid once a rewrite starts
        self.tmp = store_file.directory / (
            f".{store_file.path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
        self.fd = os.open(self.tmp, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            with contextlib.suppress(AttributeError):
                os.posix_fallocate(self.fd, 0, store_file.size)  # ENOSPC now, not mid-write
            _pwrite_all(self.fd, memoryview(store_file.header), 0)
        except BaseException:
            self.abort()
            raise

    def add(self, name: str, tensor: torch.Tensor) -> None:
        offset, spec = self.offsets[name]
        if tuple(tensor.shape) != spec.shape or tensor.dtype != spec.dtype:
            raise ValueError(f"{name}: got {tuple(tensor.shape)} {tensor.dtype}, expected "
                             f"{spec.shape} {spec.dtype}")
        _pwrite_all(self.fd, _tensor_bytes(tensor), offset)
        self.pending.discard(name)

    def commit(self, extra: dict[str, Any]) -> None:
        if self.pending:
            raise ValueError(f"{len(self.pending)} tensors never written, e.g. "
                             f"{sorted(self.pending)[:3]}")
        os.fsync(self.fd)
        header, _ = _read_header(self.fd)
        samples = _samples_digest(self.fd, len(header), self.file.size)
        os.close(self.fd)
        self.fd = -1
        os.replace(self.tmp, self.file.path)
        _fsync_dir(self.file.directory)
        _write_json_atomic(self.file.manifest, {
            "format": STORE_FORMAT,
            "digest": self.file.digest,
            "file": self.file.path.name,
            "bytes": self.file.size,
            "header_sha256": hashlib.sha256(header).hexdigest(),
            "samples_sha256": samples,
            "tensors": self.file.expected_tensors(),
            **extra,
        })

    def abort(self) -> None:
        if self.fd >= 0:
            with contextlib.suppress(OSError):
                os.close(self.fd)
            self.fd = -1
        with contextlib.suppress(OSError):
            self.tmp.unlink()


# --------------------------------------------------------------------------------- the entry


@dataclass
class StoreResult:
    """What the store did for one component load on this rank."""

    status: str  # "hit", "written", "not written", "off"
    detail: str = ""
    directory: str | None = None
    bytes: int = 0  # bytes this rank read from (hit) or wrote to (written) the store
    seconds: float = 0.0  # time in store reads (hit) or in store writes and commit (written)

    def line(self, what: str, rank: int, world: int) -> str:
        size = f" {self.bytes / 1e9:.2f} GB in {self.seconds:.1f} s" if self.bytes else ""
        detail = f": {self.detail}" if self.detail else ""
        where = f" ({self.directory})" if self.directory else ""
        return f"[weight store] rank {rank}/{world} {what}: {self.status}{size}{detail}{where}"


class _Entry:
    def __init__(self, root: Path, model_dir: Path, specs: list[_Spec], *, dtype: torch.dtype,
                 tp_size: int, tp_rank: int, extra_key: dict[str, Any]) -> None:
        layout = hashlib.sha256(json.dumps(
            [[s.name, s.key, list(s.shape), str(s.dtype), s.shard_dim] for s in specs],
            separators=(",", ":")).encode()).hexdigest()
        self.key_inputs = {
            "format": STORE_FORMAT,
            "source": _fingerprint(model_dir),
            "component": model_dir.name,
            "dtype": str(dtype),
            "tp_degree": tp_size,
            "layout_sha256": layout,
            **extra_key,
        }
        self.digest = hashlib.sha256(json.dumps(
            self.key_inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
        self.directory = root / f"{_label(model_dir, dtype, tp_size)}__{self.digest}"
        replicated = [s for s in specs if s.shard_dim is None]
        sharded = [s for s in specs if s.shard_dim is not None]
        self.files: list[_StoreFile] = []
        if replicated:
            self.files.append(_StoreFile(self.directory, _REPLICATED, replicated, self.digest))
        if sharded:
            self.files.append(_StoreFile(self.directory, f"tp{tp_rank}-of-{tp_size}", sharded,
                                         self.digest))
        rank_bytes = sum(s.nbytes for s in sharded)
        #: Bytes of the whole entry on disk: the replicated file once, every TP rank's file.
        self.entry_bytes = sum(s.nbytes for s in replicated) + tp_size * rank_bytes

    @property
    def rank_bytes(self) -> int:
        return sum(f.size for f in self.files)


def _targets(module: nn.Module) -> dict[str, torch.Tensor]:
    persistent = set(module.state_dict().keys())
    return {name: t for name, t in [*module.named_parameters(), *module.named_buffers()]
            if name in persistent}


def load_with_store(
    module: nn.Module,
    model_dir: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rename: Callable[[str], str] | None,
    load_checkpoint: Callable[..., dict[str, list[str]]],
    extra_key: dict[str, Any] | None = None,
    root: Path | None = None,
) -> tuple[dict[str, list[str]], StoreResult]:
    """Load this rank's weights into the meta-built ``module`` from the store, or from the
    checkpoint with ``load_checkpoint(module, model_dir, device=, dtype=, rename=, on_tensor=)``
    (``core/checkpoint.load_sharded_checkpoint``) while writing the store. Returns the load
    report (the checkpoint's, as recorded at write time on a hit) and what the store did."""
    from difflet.backends.neuron.core.checkpoint import (
        _meta_computed_buffers,
        materialize_meta_,
    )
    from difflet.backends.neuron.ops_impl.collectives import get_tp_rank, get_tp_size

    root = store_root() if root is None else root

    def direct(on_tensor=None):
        return load_checkpoint(module, model_dir, device=device, dtype=dtype, rename=rename,
                               on_tensor=on_tensor)

    if root is None:
        return direct(), StoreResult("off")
    tp_size = get_tp_size()
    tp_rank = get_tp_rank() if tp_size > 1 else 0
    model_dir = Path(model_dir)
    try:
        specs = _module_specs(module, dtype=dtype, tp_size=tp_size, prefix="", rename=rename)
        unsupported = sorted({str(s.dtype) for s in specs if s.dtype not in _DTYPES})
        if unsupported or _meta_computed_buffers(module):
            why = f"dtypes {unsupported}" if unsupported else "meta computed buffers"
            return direct(), StoreResult("not written", f"unsupported module ({why})")
        entry = _Entry(root, model_dir, specs, dtype=dtype, tp_size=tp_size, tp_rank=tp_rank,
                       extra_key=dict(extra_key or {}))
        checks = [f.check() for f in entry.files]
    except (OSError, ValueError) as exc:
        return direct(), StoreResult("not written", f"store unavailable ({exc})")
    where = str(entry.directory)

    if all(body is not None for body, _ in checks):
        start = time.perf_counter()
        try:
            materialize_meta_(module, device=device, dtype=dtype)
            targets = _targets(module)
            for f in entry.files:
                f.read_into(targets)
            module.to(device)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("weight store read failed (%s); loading the checkpoint", exc)
            return direct(), StoreResult("not written", f"store read failed ({exc})", where)
        report = checks[0][0].get("load_report") or {"missing": [], "unexpected": []}
        return ({"missing": list(report["missing"]), "unexpected": list(report["unexpected"])},
                StoreResult("hit", "", where, entry.rank_bytes, time.perf_counter() - start))

    reasons = sorted({why for body, why in checks if body is None})
    if not _write_everything():
        try:
            read = _checkpoint_bytes(model_dir, specs)
        except (OSError, ValueError) as exc:
            return direct(), StoreResult("not written", f"checkpoint headers unreadable ({exc})",
                                         where)
        stored = sum(s.nbytes for s in specs)
        if read is not None and stored > read * (1 - MIN_SAVING):
            # The checkpoint already holds these tensors in the load dtype: the store would read
            # the same bytes (measured on trn2: Qwen-Image's bf16 DiT loads in 311 s cold and
            # 3.6 s warm either way) and cost their bytes again on disk and a write.
            return direct(), StoreResult(
                "not written", f"no gain: the checkpoint holds these tensors in the load dtype "
                f"({stored / 1e9:.2f} GB either way; {ALL_ENV}=1 writes it anyway)", where)
    with contextlib.ExitStack() as stack:
        writers: list[_FileWriter] = []
        note = ""
        try:
            entry.directory.mkdir(parents=True, exist_ok=True)
            missing = [f for f, (body, _) in zip(entry.files, checks) if body is None]
            free = shutil.disk_usage(entry.directory).free
            # The whole entry, the same on every rank, so the TP ranks decide alike.
            needed = entry.entry_bytes
            if free < needed + RESERVE_BYTES:
                note = (f"needs {needed / 1e9:.1f} GB + {RESERVE_BYTES / 1e9:.0f} GB reserve, "
                        f"{free / 1e9:.1f} GB free")
            else:
                busy = False
                for f in missing:
                    if not stack.enter_context(f.locked()):
                        busy = True  # another process is writing it
                        continue
                    if f.check()[0] is not None:
                        continue  # written by another process since the check above
                    writers.append(_FileWriter(f))
                if not writers:
                    note = ("being written by another process" if busy
                            else "written by another process meanwhile")
        except OSError as exc:
            for writer in writers:
                writer.abort()
            writers, note = [], f"store unwritable ({exc.strerror or exc})"

        if not writers:
            report = direct()
            return report, StoreResult("not written", f"{', '.join(reasons)}; {note}", where)

        spent = [0.0]
        by_name = {spec.name: w for w in writers for spec in w.file.specs}
        failed: list[str] = []

        def on_tensor(name: str, tensor: torch.Tensor) -> None:
            writer = by_name.get(name)
            if writer is None:
                return
            start = time.perf_counter()
            try:
                writer.add(name, tensor)
            except (OSError, ValueError) as exc:  # a store write never fails the load
                writer.abort()
                for other in [n for n, w in by_name.items() if w is writer]:
                    del by_name[other]
                writers.remove(writer)
                failed.append(f"{writer.file.path.stem}: {exc}")
            spent[0] += time.perf_counter() - start

        try:
            report = direct(on_tensor)
        except BaseException:
            for writer in writers:
                writer.abort()
            raise
        start = time.perf_counter()
        written = 0
        try:
            if failed:
                raise OSError("; ".join(failed))
            if report["missing"]:
                raise ValueError(f"{len(report['missing'])} tensors missing from the checkpoint")
            for writer in writers:
                writer.commit({"load_report": report, "key": entry.key_inputs,
                               "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                written += writer.file.size
        except (OSError, ValueError) as exc:
            for writer in writers:
                writer.abort()
            return report, StoreResult("not written", f"write failed ({exc})", where)
        spent[0] += time.perf_counter() - start
        detail = ", ".join(reasons) if reasons != ["no entry"] else ""
        files = "+".join(w.file.path.stem for w in writers)
        return report, StoreResult("written", f"{files}{'; was: ' + detail if detail else ''}",
                                   where, written, spent[0])


def print_result(result: StoreResult, what: str, rank: int, world: int) -> None:
    """One evidence line per component and rank, on stderr."""
    print(result.line(what, rank, world), file=sys.stderr, flush=True)
