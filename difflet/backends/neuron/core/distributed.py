"""Rank-0 host I/O and failure propagation for the neuron backend (one process per core).

Under torchrun every rank must issue the same collectives in the same order. Host-side work
(reading inputs, building example tensors, writing files) runs on rank 0 only and its result is
broadcast; a rank that raises *before* a collective would leave the others blocked in it until
the process-group timeout. So every fallible, collective-free step runs inside
``collective_phase``, which ends with an int32 all-reduce(MAX) of a status code: when any rank
failed, every rank raises (the failing one its own error, the others ``RankFailureError``). This
generalises the encode-status broadcast of ``difflet/models/wan/tpu_application.py:35-40,92-120``.

A phase body must not issue collectives itself. If it did, a rank failing before one of them
would send its status all-reduce while its peers sit in the body's collective, and the two would
pair up. So a model forward never runs inside a phase: a rank that fails mid-forward raises and
exits, and the launcher (torchrun) tears down the peers blocked in the forward's collectives.

Neuron process-group constraints (torch_neuronx):

* it serves ``neuron`` tensors only (``distributed/backend.py:46-50``), so ``device`` is the
  rank's compute device -- gloo tests pass ``"cpu"``;
* ``barrier`` works on the default group only (``NeuronBackend.cpp:1177-1178``), object
  collectives force graph breaks (``distributed/backend.py:58-92``) and there is no functional
  broadcast (``distributed/ops/functional_collectives.py:202-226``), so this module uses eager
  ``dist.all_reduce``/``dist.broadcast`` on the default group, outside compiled regions only;
* ``broadcast`` is a zero-fill on non-roots plus all-reduce(SUM) (``NeuronBackend.cpp:1110-1141``),
  exact for finite values (``-0.0`` arrives as ``+0.0``); collectives run int64 as int32
  (``csrc/core/dispatch/DispatchUtils.cpp:284-302``), so on the neuron device an int64 payload
  outside the int32 range is rejected rather than silently narrowed, and one inside it is sent
  as int32 and widened on arrival (the group's own narrowing falls back to the CPU).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

import torch
import torch.distributed as dist

__all__ = [
    "RankFailureError",
    "broadcast_tensor",
    "collective_phase",
    "is_rank0",
    "rank0_call",
    "sync_status",
    "world_info",
]

T = TypeVar("T")

#: Dtypes ``broadcast_tensor`` carries; the header stores the index into this tuple.
_BROADCAST_DTYPES: tuple[torch.dtype, ...] = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
)
_MAX_NDIM = 8
_HEADER_LEN = 2 + _MAX_NDIM  # [dtype code, ndim, dim_0 .. dim_7]
_FAILED_CODE = -1
_STATUS_DTYPE = torch.int32
_INT32_MIN, _INT32_MAX = -(2**31), 2**31 - 1


class RankFailureError(RuntimeError):
    """Another rank failed in a collective phase; this rank stops at the same point."""


def world_info() -> tuple[int, int]:
    """``(rank, world_size)`` of the default process group; ``(0, 1)`` without one."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def is_rank0() -> bool:
    return world_info()[0] == 0


def sync_status(ok: bool, *, device, what: str) -> None:
    """Agree on success across ranks; healthy ranks raise when any rank failed.

    Each rank contributes 0 (ok) or ``rank + 1`` (failed); the all-reduce(MAX) result names the
    highest failing rank. The failing rank itself does not raise here: its caller re-raises
    the original error. A no-op on one rank.
    """
    rank, world = world_info()
    if world == 1:
        return
    flag = torch.tensor([0 if ok else rank + 1], dtype=_STATUS_DTYPE).to(device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    failed = int(flag.cpu()[0])
    if ok and failed:
        raise RankFailureError(
            f"{what}: failed on rank {failed - 1}; rank {rank} stops too "
            f"(see rank {failed - 1}'s log)"
        )


@contextmanager
def collective_phase(what: str, *, device) -> Iterator[None]:
    """Run a collective-free body, then sync its outcome so every rank leaves together.

    Every exit path issues exactly one status all-reduce, including a ``RankFailureError``
    raised by a nested phase. That keeps the collective count equal on all ranks. The body
    must not issue collectives (see the module docstring).
    """
    try:
        yield
    except Exception:
        sync_status(False, device=device, what=what)
        raise
    sync_status(True, device=device, what=what)


def rank0_call(fn: Callable[..., T], *args: Any, device, what: str, **kwargs: Any) -> T | None:
    """``fn(*args, **kwargs)`` on rank 0 inside a status-synced phase; None on other ranks."""
    result = None
    with collective_phase(what, device=device):
        if is_rank0():
            result = fn(*args, **kwargs)
    return result


def broadcast_tensor(tensor: torch.Tensor | None, *, device, src: int = 0) -> torch.Tensor:
    """Send ``src``'s tensor to every rank, on ``device``; other ranks pass None.

    An int32 header (dtype code, ndim, dims) goes first so receivers can allocate. The source
    validates and stages its payload on ``device`` before sending the header; if anything there
    fails it sends a failed header and raises its own error, and receivers raise
    ``RankFailureError`` instead of waiting for a payload that never comes. The payload travels
    flat, and where collectives narrow int64 (the neuron device) an int64 payload must fit int32
    and travels as int32 (see ``_wire_dtype``).
    """
    rank, world = world_info()
    if world == 1:
        _encode_header(tensor)  # same validation as the multi-rank path
        _check_payload(tensor, device)
        return tensor.detach().contiguous().to(device)
    if rank == src:
        try:
            header = _encode_header(tensor)
            _check_payload(tensor, device)
            dtype, shape = tensor.dtype, tuple(tensor.shape)
            # contiguous on the host first: a strided host view moved to the device is
            # restrided there through a CPU round trip.
            flat = tensor.detach().contiguous().reshape(-1)
            wire = flat.to(_wire_dtype(dtype, device)).to(device)
            error = None
        except Exception as exc:  # noqa: BLE001 - re-raised below, after the receivers know
            header, wire, error = _failed_header(), None, exc
        dist.broadcast(header.to(device), src=src)
        if error is not None:
            raise error
    else:
        header = torch.zeros(_HEADER_LEN, dtype=torch.int32).to(device)
        dist.broadcast(header, src=src)
        dtype, shape = _decode_header(header.cpu(), src=src)
        wire = torch.empty(math.prod(shape), dtype=_wire_dtype(dtype, device), device=device)
    if wire.numel():
        dist.broadcast(wire, src=src)
    return wire.to(dtype).reshape(shape)


def _narrows_int64(device) -> bool:
    """Whether collectives on ``device`` carry int64 as int32 (the neuron process group does)."""
    return torch.device(device).type == "neuron"


def _wire_dtype(dtype: torch.dtype, device) -> torch.dtype:
    """The payload dtype inside the collective. On the neuron device int64 goes as int32
    (``_check_payload`` has checked the range): the process group's own int64 -> int32
    narrowing records an ``aten::_to_copy`` CPU fallback, while widening a flat int32 tensor
    on the device does not (a 0-dim one does, hence the flat wire)."""
    return torch.int32 if dtype == torch.int64 and _narrows_int64(device) else dtype


def _check_payload(tensor: torch.Tensor, device) -> None:
    if tensor.dtype != torch.int64 or not tensor.numel() or not _narrows_int64(device):
        return
    low, high = (int(v) for v in torch.aminmax(tensor.detach()))
    if low < _INT32_MIN or high > _INT32_MAX:
        raise ValueError(
            f"broadcast_tensor: int64 values in [{low}, {high}] are outside the int32 range, "
            f"and the {torch.device(device).type} process group carries int64 as int32, so "
            "they would arrive narrowed; split them into int32 parts before broadcasting"
        )


def _encode_header(tensor: Any) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            "broadcast_tensor: the source rank must pass a tensor, got "
            f"{type(tensor).__name__}"
        )
    if tensor.dtype not in _BROADCAST_DTYPES:
        names = ", ".join(str(d) for d in _BROADCAST_DTYPES)
        raise TypeError(f"broadcast_tensor: unsupported dtype {tensor.dtype}; supported: {names}")
    if tensor.dim() > _MAX_NDIM:
        raise ValueError(f"broadcast_tensor: at most {_MAX_NDIM} dims, got {tensor.dim()}")
    header = torch.zeros(_HEADER_LEN, dtype=torch.int32)
    header[0] = _BROADCAST_DTYPES.index(tensor.dtype)
    header[1] = tensor.dim()
    for i, size in enumerate(tensor.shape):
        header[2 + i] = size
    return header


def _failed_header() -> torch.Tensor:
    header = torch.zeros(_HEADER_LEN, dtype=torch.int32)
    header[0] = _FAILED_CODE
    return header


def _decode_header(header: torch.Tensor, *, src: int) -> tuple[torch.dtype, tuple[int, ...]]:
    values = [int(v) for v in header.tolist()]
    code, ndim = values[0], values[1]
    if code == _FAILED_CODE:
        raise RankFailureError(
            f"broadcast_tensor: source rank {src} failed before sending; see its log"
        )
    if not 0 <= code < len(_BROADCAST_DTYPES) or not 0 <= ndim <= _MAX_NDIM:
        raise RuntimeError(f"broadcast_tensor: corrupt header {values}")
    return _BROADCAST_DTYPES[code], tuple(values[2 : 2 + ndim])
