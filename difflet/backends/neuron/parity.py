"""Parity harness for the neuron backend.

Two kinds of check:

* ``check``: a module or function on the device against the same target on CPU
  in fp32. In fp32 the device must match to ``fp32_rtol`` of the reference's
  peak magnitude. In bf16/fp16 the criterion is relative to rounding: the device
  error against fp32 may be at most ``low_precision_ratio`` times the error of
  the same target run on CPU in the same dtype, so no per-op threshold is
  needed. Any CPU fallback on the device fails the check.
* ``compare_to_reference``: a device output against outputs saved by another
  backend or machine (``save_reference``), e.g. the AoT path at identical inputs.

Every run uses a fresh copy of a module target: casting a module to bf16 rounds
its weights, which would contaminate a later fp32 run of the same object. All
metrics are computed in fp64.
"""

from __future__ import annotations

import contextlib
import copy
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from difflet.backends.neuron.buildinfo import build_info

LOW_PRECISION = (torch.bfloat16, torch.float16)


@dataclass(frozen=True)
class Metrics:
    max_abs: float
    mean_abs: float
    max_rel: float  # max_abs relative to the reference's peak magnitude
    cosine: float

    def __str__(self) -> str:
        return (f"max_abs={self.max_abs:.2e} mean_abs={self.mean_abs:.2e} "
                f"max_rel={self.max_rel:.2e} cos={self.cosine:.6f}")


@dataclass
class ParityResult:
    name: str
    dtype: str
    device: Metrics
    baseline: Metrics | None = None  # CPU in the same low-precision dtype, against fp32
    fallbacks: list[str] = field(default_factory=list)
    passed: bool = False
    reason: str = ""

    def __str__(self) -> str:
        line = f"{'PASS' if self.passed else 'FAIL'}  {self.name:32s} [{self.dtype}] device: {self.device}"
        if self.baseline is not None:
            line += f" | cpu {self.dtype}: mean_abs={self.baseline.mean_abs:.2e}"
        if self.fallbacks:
            line += f" | fallbacks={self.fallbacks}"
        if not self.passed and self.reason:
            line += f" | {self.reason}"
        return line


def compare(out: torch.Tensor, ref: torch.Tensor) -> Metrics:
    out = out.detach().to("cpu", torch.float64).flatten()
    ref = ref.detach().to("cpu", torch.float64).flatten()
    if out.shape != ref.shape:
        raise ValueError(f"shape mismatch: {tuple(out.shape)} vs {tuple(ref.shape)}")
    err = (out - ref).abs()
    peak = ref.abs().max().item() if ref.numel() else 0.0
    norms = out.norm() * ref.norm()
    cosine = 1.0 if norms.item() == 0.0 else (out @ ref / norms).item()
    max_abs = err.max().item() if err.numel() else 0.0
    return Metrics(
        max_abs=max_abs,
        mean_abs=err.mean().item() if err.numel() else 0.0,
        max_rel=max_abs / peak if peak > 0 else max_abs,
        cosine=cosine,
    )


def check(
    name: str,
    target: nn.Module | Callable[..., Any],
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "neuron",
    prepare: Callable[[nn.Module], Any] | None = None,
    output: Callable[[Any], torch.Tensor] | None = None,
    fp32_rtol: float = 1e-4,
    low_precision_ratio: float = 1.5,
) -> ParityResult:
    """Run ``target`` on ``device`` in ``dtype`` and judge it against CPU fp32.

    ``prepare`` runs on the device copy of a module before the call (e.g.
    ``compile_blocks``); ``output`` selects the tensor to compare from the result.
    """
    kwargs = dict(kwargs or {})
    select = output or _first_tensor
    ref = select(_run(target, args, kwargs, "cpu", torch.float32, None))
    with _track_fallbacks(device) as fallbacks:
        dev = select(_run(target, args, kwargs, device, dtype, prepare))
    metrics = compare(dev, ref)

    result = ParityResult(name=name, dtype=_dtype_name(dtype), device=metrics, fallbacks=list(fallbacks))
    if dtype in LOW_PRECISION:
        base = select(_run(target, args, kwargs, "cpu", dtype, None))
        result.baseline = compare(base, ref)
        limit = low_precision_ratio * result.baseline.mean_abs + 1e-7
        ok, result.reason = metrics.mean_abs <= limit, f"mean_abs above {low_precision_ratio}x CPU {result.dtype}"
    else:
        ok, result.reason = metrics.max_rel <= fp32_rtol, f"max_rel above {fp32_rtol:g}"
    if result.fallbacks:
        ok, result.reason = False, "CPU fallbacks on the device"
    result.passed = ok
    return result


def save_reference(path: str | Path, tensors: Mapping[str, torch.Tensor], meta: Mapping[str, Any] | None = None) -> None:
    """Save tensors (safetensors) and metadata plus the build (json) beside each other."""
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().to("cpu").contiguous() for k, v in tensors.items()}, str(path.with_suffix(".safetensors")))
    record = {"meta": dict(meta or {}), "build": build_info()}
    path.with_suffix(".json").write_text(json.dumps(record, indent=2, sort_keys=True))


def load_reference(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from safetensors.torch import load_file

    path = Path(path)
    tensors = load_file(str(path.with_suffix(".safetensors")))
    record = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").exists() else {}
    return tensors, record


def compare_to_reference(name: str, out: torch.Tensor, ref: torch.Tensor, *, cosine_min: float = 0.999) -> ParityResult:
    """Judge a device output against a saved reference from another backend (cosine gate)."""
    metrics = compare(out, ref)
    passed = metrics.cosine >= cosine_min
    return ParityResult(name=name, dtype=_dtype_name(out.dtype), device=metrics, passed=passed,
                        reason="" if passed else f"cosine below {cosine_min}")


def report(results: Sequence[ParityResult]) -> int:
    """Print results and the build; return a process exit code (0 when all pass)."""
    for result in results:
        print(result, flush=True)
    passed = sum(r.passed for r in results)
    print(f"{passed}/{len(results)} passed | build: {build_info()}", flush=True)
    return 0 if passed == len(results) else 1


def results_json(results: Sequence[ParityResult]) -> str:
    return json.dumps({"build": build_info(), "results": [asdict(r) for r in results]}, indent=2)


# ---- internals -----------------------------------------------------------------------


def _run(target, args, kwargs, device, dtype, prepare):
    if isinstance(target, nn.Module):
        fn = copy.deepcopy(target).to(device=device, dtype=dtype).eval()
        if prepare is not None:
            prepare(fn)
    else:
        fn = target
    with torch.no_grad():
        out = fn(*_cast(args, device, dtype), **_cast(kwargs, device, dtype))
    if device == "neuron":
        torch.neuron.synchronize()
    return out


def _cast(value, device, dtype):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
    if isinstance(value, Mapping):
        return {k: _cast(v, device, dtype) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cast(v, device, dtype) for v in value)
    return value


def _first_tensor(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, Mapping):
        out = list(out.values())
    if isinstance(out, (list, tuple)):
        for item in out:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"no tensor in output of type {type(out).__name__}; pass output=")


@contextlib.contextmanager
def _track_fallbacks(device: str) -> Iterator[list[str]]:
    if device != "neuron":
        yield []
        return
    from difflet.backends.neuron.runtime import track_fallbacks

    with track_fallbacks() as ops:
        yield ops


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


__all__ = [
    "Metrics",
    "ParityResult",
    "check",
    "compare",
    "compare_to_reference",
    "load_reference",
    "report",
    "results_json",
    "save_reference",
]
