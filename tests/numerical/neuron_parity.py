"""Shared parity metrics and device evidence for model bring-up on the neuron backend.

Every gate that compares a device output with a CPU reference and has to prove "the blocks
compiled and nothing fell back" goes through this module, so the metrics, the thresholds and
the blind spot of the fallback check live in one place:

* ``metrics(actual, reference, control=...)``: the numerical-oracle dict, computed in fp64,
  plus ``err_cos`` (how the result's error lines up with the bf16 control's) given a control.
* ``check(metrics, control=...)``: the TPU numerical gate's three thresholds as defaults, the
  same-dtype one gated only when the result rounds like the bf16 control (D68), where it is
  a diagnostic plus a nan detector rather than an independent gate (D69);
  ``same_dtype_gated`` / ``same_dtype_label`` say which, for callers' logs.
* ``device_evidence(app, fallbacks)``: what the run itself shows about compilation and memory.
* ``run_parity(name, target, args, ...)``: runs a module or function on the device and on CPU
  (fp32 reference, same-dtype control) and judges the result with ``metrics`` and ``check``.
* ``save_reference`` / ``load_reference``: tensors plus the build, to compare against outputs
  produced on another stack (the AoT backend) at identical inputs; ``report`` prints results.

Pure torch: it imports on a CPU-only host and never imports ``torch_neuronx`` or starts the
Neuron runtime (``device_evidence`` reads ``torch_neuronx`` only if it is already loaded).
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import json
import math
import os
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tests.numerical.buildinfo import build_info

from difflet.backends.neuron.compile import NEFF_CACHE_ENV, default_neff_cache_dir

__all__ = [
    "COUNTER_NAMES",
    "MAX_RATIO_VS_CONTROL",
    "MAX_SAME_DTYPE_RATIO",
    "MIN_COSINE",
    "NEFF_CACHE_ENV",
    "SAME_DTYPE_GATED_MIN_ERR_COS",
    "check",
    "device_evidence",
    "format_result",
    "load_reference",
    "metrics",
    "report",
    "run_parity",
    "same_dtype_gated",
    "same_dtype_label",
    "save_reference",
]

#: How much error difflet may add on top of what bf16 alone costs: difflet's rel_l1 against the
#: fp32 reference, divided by the bf16 control's rel_l1 against the same reference
#: (tests/numerical/test_tpu_vs_diffusers.py:71, measured 1.042x for Wan).
MAX_RATIO_VS_CONTROL = 1.15
#: Distance to the same-dtype upstream result, as a multiple of the control's distance from
#: fp32 (test_tpu_vs_diffusers.py:76, measured 0.91 for Wan).
MAX_SAME_DTYPE_RATIO = 1.2
#: Decision D68: ``MAX_SAME_DTYPE_RATIO`` is gated only when the result's error vector has at
#: least this cosine with the bf16 control's (``err_cos``, i.e. the result rounds like
#: upstream bf16); below it, the same-dtype distance is reported, not gated, and ``check``
#: notes the decorrelation (D69). See ``check``.
SAME_DTYPE_GATED_MIN_ERR_COS = 0.5
#: Absolute cosine floor, a break detector and not the gate: upstream's own bf16 scores
#: 0.99873 against fp32 for Wan (test_tpu_vs_diffusers.py:81).
MIN_COSINE = 0.995

#: The ``CompilationCache.*`` counters ``device_evidence`` reports (torch_neuronx.metrics).
COUNTER_NAMES = ("TotalCompilations", "PersistentHits", "InMemoryHits")
_COUNTER_PREFIX = "CompilationCache."
_GIB = 2**30


def metrics(
    actual: torch.Tensor, reference: torch.Tensor, *, control: torch.Tensor | None = None
) -> dict[str, float]:
    """Distance of ``actual`` from ``reference`` over the flattened tensors, in fp64.

    Keys: ``cosine``; ``rel_l1`` (mean |a - r| / mean |r|); ``rel_l2`` (||a - r|| / ||r||);
    ``max_abs`` (max |a - r|); ``ref_absmean`` (mean |r|), the keys of
    tests/numerical/tpu_oracle_compare.py; and ``rel_max`` (max |a - r| / max |r|), the ratio
    tests/manual/check_neuron_launch_c10.py gates on. Tensors on the device are copied to the
    host first. Shapes must match exactly: equal element counts under different shapes are a
    layout bug, not a comparison.

    With ``control`` (the same computation in bf16 upstream, e.g. diffusers bf16) the dict also
    holds ``err_cos``: the cosine between the two error vectors ``actual - reference`` and
    ``control - reference``. Near 1 the result rounds like the control; near 0 its error is
    independent of the control's. nan when either error is exactly zero (no direction).
    ``check`` reads it to decide whether the same-dtype distance is gated (D68).
    """
    for name, other in (("reference", reference), ("control", control)):
        if other is not None and actual.shape != other.shape:
            raise ValueError(
                f"parity.metrics: shape mismatch, actual {tuple(actual.shape)} vs "
                f"{name} {tuple(other.shape)}"
            )
    a = actual.detach().cpu().to(torch.float64).flatten()
    r = reference.detach().cpu().to(torch.float64).flatten()
    diff = (a - r).abs()
    ref_abs = r.abs()
    result = {
        "cosine": float(torch.nn.functional.cosine_similarity(a, r, dim=0)),
        "rel_l1": float(diff.mean() / ref_abs.mean()),
        "rel_l2": float((a - r).norm() / r.norm()),
        "max_abs": float(diff.max()),
        "ref_absmean": float(ref_abs.mean()),
        "rel_max": float(diff.max() / ref_abs.max().clamp_min(1e-12)),
    }
    if control is not None:
        err = a - r
        err_control = control.detach().cpu().to(torch.float64).flatten() - r
        norms = float(err.norm()) * float(err_control.norm())
        # cosine_similarity clamps a zero norm to eps and returns 0; an exact result has no
        # error direction, so say so with nan instead.
        result["err_cos"] = float(err @ err_control) / norms if norms > 0 else math.nan
    return result


def same_dtype_gated(metrics: Mapping[str, Any]) -> bool:
    """Whether ``check`` gates the same-dtype distance of this result (decision D68).

    True when ``metrics["err_cos"]`` is at least ``SAME_DTYPE_GATED_MIN_ERR_COS``, and when it is
    missing or nan: without it, whether the result rounds like the control is unknown, so the
    limit applies.
    """
    err_cos = metrics.get("err_cos")
    return err_cos is None or not err_cos < SAME_DTYPE_GATED_MIN_ERR_COS


def same_dtype_label(metrics: Mapping[str, Any], control: Mapping[str, Any]) -> str:
    """One log phrase for the same-dtype distance and how ``check`` treats it, e.g.
    ``"same-dtype 1.201x, reported, not gated (err_cos 0.231 < 0.5, D68)"``."""
    same_dtype = metrics.get("vs_bf16")
    if same_dtype is None:
        return "same-dtype not measured"
    ratio = _ratio(same_dtype["rel_l1"], control["rel_l1"])
    err_cos = metrics.get("err_cos")
    limit = SAME_DTYPE_GATED_MIN_ERR_COS
    if err_cos is None:
        why = "err_cos not recorded"
    elif math.isnan(err_cos):
        why = "err_cos nan"
    else:
        why = f"err_cos {err_cos:.3f} {'>=' if err_cos >= limit else '<'} {limit}"
    verdict = "gated" if same_dtype_gated(metrics) else "reported, not gated"
    return f"same-dtype {ratio:.3f}x, {verdict} ({why}, D68)"


def _ratio(value: float, control: float) -> float:
    """``value / control``; nan stays nan, and a zero-error control only admits zero error."""
    if math.isnan(value) or math.isnan(control):
        return math.nan
    if control > 0:
        return value / control
    return 0.0 if value == 0 else math.inf


def check(
    metrics: Mapping[str, Any],
    *,
    control: Mapping[str, Any] | None = None,
    max_ratio_vs_control: float = MAX_RATIO_VS_CONTROL,
    max_same_dtype_ratio: float = MAX_SAME_DTYPE_RATIO,
    min_cosine: float = MIN_COSINE,
    notes: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Judge one result against the TPU numerical gate; returns ``(ok, reasons)``.

    ``metrics`` is ``metrics(device_output, fp32_reference)``, optionally with the same-dtype
    comparison ``metrics(device_output, bf16_reference)`` under the key ``"vs_bf16"`` (the layout
    of the TPU oracle's ``results[...]`` entries). ``control`` is ``metrics(bf16_reference,
    fp32_reference)``: what bf16 alone costs. Checks, each failing with a reason that names its
    threshold:

    * ``cosine`` (and the same-dtype cosine) at least ``min_cosine``: the direction gate;
    * with ``control``: ``rel_l1 / control["rel_l1"]`` at most ``max_ratio_vs_control``: the
      magnitude gate, "difflet adds little on top of what bf16 costs";
    * with ``control`` and ``"vs_bf16"``: the same-dtype ratio ``vs_bf16 rel_l1 /
      control["rel_l1"]`` is never nan, and is at most ``max_same_dtype_ratio`` when
      ``same_dtype_gated(metrics)`` holds: ``metrics["err_cos"]`` (from ``metrics(...,
      control=...)``) is at least ``SAME_DTYPE_GATED_MIN_ERR_COS`` (0.5), or is missing or
      nan (decision D68).

    What the same-dtype check is (decision D69): a reported diagnostic plus a nan detector, not
    an independent gate. With ``e`` the result's error from fp32 and ``e_c`` the control's, the
    result lies ``|e - e_c|`` from the control, so the same-dtype ratio is
    ``sqrt(r^2 + 1 - 2 r err_cos)`` for a result ``r`` times the control from fp32: exactly in
    the L2 norm, and to within a few per cent in the rel_l1 used here (W3: predicted 1.196,
    measured 1.198). It is therefore fixed by ``r`` and ``err_cos``. Inside the gated regime
    (``err_cos >= 0.5``) a result that passes the magnitude gate (``r <= 1.15``) sits at most
    ``sqrt(1.15^2 - 1.15 + 1) = 1.083`` from the control, so the 1.2 limit cannot fail it;
    it can only fail alongside the magnitude gate, or when ``err_cos`` is unknown. Below 0.5
    the errors are decorrelated and the distance is set by their sizes alone (1.24x at
    ``r = 1.0`` and ``err_cos`` 0.23, whatever the result's correctness), so it is reported,
    not gated (D68). The real "rounds like upstream bf16" signal is ``err_cos`` itself: when
    it is below 0.5 a note saying so is appended to ``notes`` (when the caller passes a list),
    whatever the same-dtype magnitude, so a decorrelation regression, e.g. an eager path that
    stops rounding like diffusers, is visible in the logs without failing the gate. Such a
    regression passes as long as it stays within 1.15x, the bar every mode has.

    Measured in W3: the per-block compiled 14B Wan DiT is 0.929x / 0.832x the control from
    fp32 at t=996 / 500 but 1.201x / 1.097x it from diffusers bf16, with err_cos 0.23 / 0.30:
    neuronx-cc keeps the residual stream in fp32 inside each compiled block although the
    StableHLO it receives rounds it to bf16 as diffusers does. Eager rounds like diffusers
    (err_cos 0.57 / 0.66; W2: 1.025x vs fp32, same-dtype 0.931x). D68 replaced D65, which gated
    the same-dtype distance only above 1.0x the control from fp32: with decorrelated errors
    that made 1.0x, not 1.15x, the compiled path's effective limit (W3 review).

    Limits are inclusive, and a nan never passes: a nan ratio fails the magnitude gate, a nan
    same-dtype ratio fails whether gated or not, and a nan ``err_cos`` gates the same-dtype
    limit. Without ``control`` only the cosine floor is judged. The defaults are the TPU gate's
    values; loosening one here loosens every caller.
    """
    reasons: list[str] = []
    same_dtype = metrics.get("vs_bf16")
    if not metrics["cosine"] >= min_cosine:
        reasons.append(f"cosine {metrics['cosine']:.8f} is below min_cosine {min_cosine}")
    if same_dtype is not None and not same_dtype["cosine"] >= min_cosine:
        reasons.append(
            f"same-dtype (vs_bf16) cosine {same_dtype['cosine']:.8f} is below "
            f"min_cosine {min_cosine}"
        )
    if control is not None:
        ratio = _ratio(metrics["rel_l1"], control["rel_l1"])
        if not ratio <= max_ratio_vs_control:
            reasons.append(
                f"rel_l1 {metrics['rel_l1']:.3e} is {ratio:.3f}x the control's "
                f"{control['rel_l1']:.3e}, above max_ratio_vs_control {max_ratio_vs_control}"
            )
        if same_dtype is not None:
            same_ratio = _ratio(same_dtype["rel_l1"], control["rel_l1"])
            if math.isnan(same_ratio):
                reasons.append(
                    f"same-dtype (vs_bf16) rel_l1 {same_dtype['rel_l1']:.3e} against the "
                    f"control's {control['rel_l1']:.3e} gives a nan ratio"
                )
            elif not same_ratio <= max_same_dtype_ratio and same_dtype_gated(metrics):
                reasons.append(
                    f"same-dtype (vs_bf16) rel_l1 {same_dtype['rel_l1']:.3e} is "
                    f"{same_ratio:.3f}x the control's {control['rel_l1']:.3e}, above "
                    f"max_same_dtype_ratio {max_same_dtype_ratio}"
                )
    err_cos = metrics.get("err_cos")
    if notes is not None and err_cos is not None and err_cos < SAME_DTYPE_GATED_MIN_ERR_COS:
        distance = ("" if control is None or same_dtype is None else
                    f"; same-dtype {_ratio(same_dtype['rel_l1'], control['rel_l1']):.3f}x the "
                    f"control's distance from fp32, reported, not gated")
        notes.append(
            f"err_cos {err_cos:.3f} < {SAME_DTYPE_GATED_MIN_ERR_COS}: the result's error is "
            f"decorrelated from the bf16 control's, so it does not round like upstream bf16"
            f"{distance} (D68/D69)"
        )
    return not reasons, reasons


def _neff_count() -> int:
    """``.neff`` files under the NEFF cache this process uses: the launcher's
    ``TORCH_NEURONX_NEFF_CACHE_DIR`` if set (``configure_compile_cache`` keeps it verbatim),
    else ``default_neff_cache_dir()``."""
    directory = Path(os.environ.get(NEFF_CACHE_ENV) or default_neff_cache_dir()).expanduser()
    return sum(1 for _ in directory.rglob("*.neff")) if directory.is_dir() else 0


def device_evidence(app: Any, fallbacks: list[str]) -> dict[str, Any]:
    """What one rank's run shows about compilation, shapes and memory, as plain JSON types.

    ``app`` is a ``TorchNeuronApplicationBase``; ``fallbacks`` is what
    ``difflet.backends.neuron.runtime.track_fallbacks()`` collected around the forwards.

    Keys: ``fallbacks`` (a copy), ``unwarmed_shapes`` (forward shapes the warm-up never compiled,
    as nested lists; must be ``[]``), ``compiled_blocks``, ``phase_seconds``, ``counters``
    (``CompilationCache.{TotalCompilations,PersistentHits,InMemoryHits}``, ``None`` read as 0),
    ``neff_count`` (``.neff`` files in the NEFF cache, see ``_neff_count``) and
    ``peak_device_mem_gb`` (``torch_neuronx.max_memory_allocated()`` in GiB, 2**30 bytes, to
    compare with the 18 GiB shape-fallback threshold; ``None`` off the device).

    The counters and the peak are read only when the app's device is ``neuron`` and
    ``torch_neuronx`` is already in ``sys.modules``; otherwise the counters are 0 and the peak is
    ``None``. The counters stay 0 as well if the caller never enabled ``torch_neuronx.metrics``.

    Blind spot, read before trusting an empty ``fallbacks``: ``track_fallbacks`` lists only eager
    ops that torch_neuronx dispatched to the CPU. Inside a compiled graph nothing can "fall back"
    (with ``fallback_execution`` off a lowering failure raises), so an empty list proves only that
    the eager top level ran on the device. That the blocks really compiled is shown by the
    compilation counters, ``compiled_blocks`` and the graph count; a block that lowers to a slow
    decomposition is invisible to every field here, and speed is proven only by the timers.
    """
    counters = dict.fromkeys(COUNTER_NAMES, 0)
    peak_device_mem_gb = None
    on_device = getattr(getattr(app, "device", None), "type", None) == "neuron"
    if on_device and "torch_neuronx" in sys.modules:
        counter_value = importlib.import_module("torch_neuronx.metrics").get_counter_value
        counters = {
            name: counter_value(_COUNTER_PREFIX + name) or 0 for name in COUNTER_NAMES
        }
        peak_device_mem_gb = sys.modules["torch_neuronx"].max_memory_allocated() / _GIB
    return {
        "fallbacks": list(fallbacks),
        "unwarmed_shapes": [[list(shape) for shape in key] for key in app.unwarmed_shapes],
        "compiled_blocks": list(app.compiled_blocks),
        "phase_seconds": dict(app.phase_seconds),
        "counters": counters,
        "neff_count": _neff_count(),
        "peak_device_mem_gb": peak_device_mem_gb,
    }


# ---- running a target against the CPU references ------------------------------------------

LOW_PRECISION = (torch.bfloat16, torch.float16)
#: fp32 has no control to judge against, so a device result must match CPU fp32 to this
#: ``rel_max``; the C2/C3 op checks measured 1e-6 to 3e-7.
FP32_MAX_REL = 1e-4


def run_parity(
    name: str,
    target: nn.Module | Callable[..., Any],
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "neuron",
    prepare: Callable[[nn.Module], Any] | None = None,
    output: Callable[[Any], torch.Tensor] | None = None,
    fp32_max_rel: float = FP32_MAX_REL,
) -> dict[str, Any]:
    """Run ``target`` on ``device`` in ``dtype`` and judge it against CPU with ``check``.

    Runs, each on a fresh copy of a module target (casting a module to bf16 rounds its weights,
    which would leak into a later run of the same object): CPU fp32, the reference; ``device``
    in ``dtype``, the result, with CPU fallbacks tracked; and for bf16/fp16 CPU in ``dtype``,
    the control (what the dtype alone costs). A low-precision result goes through ``check`` with
    ``control``, ``vs_bf16`` and ``err_cos``. An fp32 result has no control, so it must match the
    reference to ``fp32_max_rel`` (``rel_max``) besides the cosine floor. A CPU fallback fails
    either way. ``prepare`` runs on the device copy of a module before the call (e.g.
    ``compile_blocks``); ``output`` selects the tensor to compare.

    Returns a JSON-able dict: ``name``, ``dtype``, ``metrics``, ``control``, ``fallbacks``,
    ``ok``, ``reasons``, ``notes``.
    """
    kwargs = dict(kwargs or {})
    select = output or _first_tensor
    reference = select(_run(target, args, kwargs, "cpu", torch.float32, None))
    with _fallback_tracking(device) as fallbacks:
        actual = select(_run(target, args, kwargs, device, dtype, prepare))
    notes: list[str] = []
    control = None
    if dtype in LOW_PRECISION:
        control_out = select(_run(target, args, kwargs, "cpu", dtype, None))
        result_metrics = metrics(actual, reference, control=control_out)
        result_metrics["vs_bf16"] = metrics(actual, control_out)
        control = metrics(control_out, reference)
        ok, reasons = check(result_metrics, control=control, notes=notes)
    else:
        result_metrics = metrics(actual, reference)
        ok, reasons = check(result_metrics)
        if not result_metrics["rel_max"] <= fp32_max_rel:
            ok = False
            reasons.append(
                f"rel_max {result_metrics['rel_max']:.3e} is above fp32_max_rel {fp32_max_rel}"
            )
    if fallbacks:
        ok = False
        reasons.append(f"CPU fallbacks on the device: {list(fallbacks)}")
    return {
        "name": name,
        "dtype": str(dtype).removeprefix("torch."),
        "metrics": result_metrics,
        "control": control,
        "fallbacks": list(fallbacks),
        "ok": ok,
        "reasons": reasons,
        "notes": notes,
    }


def save_reference(
    path: str | os.PathLike[str], tensors: Mapping[str, torch.Tensor], meta: Mapping[str, Any] | None = None
) -> None:
    """Save tensors (safetensors) and ``{"meta", "build"}`` (json) under ``path``'s stem.

    Readable on another stack: the AoT backend's outputs at identical inputs travel this way.
    """
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.detach().to("cpu").contiguous() for key, value in tensors.items()},
        str(path.with_suffix(".safetensors")),
    )
    record = {"meta": dict(meta or {}), "build": build_info()}
    path.with_suffix(".json").write_text(json.dumps(record, indent=2, sort_keys=True))


def load_reference(path: str | os.PathLike[str]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from safetensors.torch import load_file

    path = Path(path)
    tensors = load_file(str(path.with_suffix(".safetensors")))
    record_path = path.with_suffix(".json")
    return tensors, json.loads(record_path.read_text()) if record_path.exists() else {}


def format_result(result: Mapping[str, Any]) -> str:
    m = result["metrics"]
    line = (f"{'PASS' if result['ok'] else 'FAIL'}  {result['name']:32s} [{result['dtype']}] "
            f"cos={m['cosine']:.6f} rel_l1={m['rel_l1']:.2e} rel_max={m['rel_max']:.2e}")
    if result.get("control"):
        line += f" | {_ratio(m['rel_l1'], result['control']['rel_l1']):.3f}x the control"
    for text in [*result.get("reasons", []), *result.get("notes", [])]:
        line += f"\n      {text}"
    return line


def report(results: Sequence[Mapping[str, Any]], json_path: str | os.PathLike[str] | None = None) -> int:
    """Print the results and the build; optionally write both as JSON. Returns an exit code."""
    for result in results:
        print(format_result(result), flush=True)
    build = build_info()
    passed = sum(bool(result["ok"]) for result in results)
    print(f"{passed}/{len(results)} passed | build: {build}", flush=True)
    if json_path is not None:
        Path(json_path).write_text(json.dumps({"build": build, "results": list(results)}, indent=2))
    return 0 if passed == len(results) else 1


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
        return {key: _cast(item, device, dtype) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cast(item, device, dtype) for item in value)
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
def _fallback_tracking(device: str) -> Iterator[list[str]]:
    """``track_fallbacks`` on the neuron device; nothing to track elsewhere."""
    if device != "neuron":
        yield []
        return
    from difflet.backends.neuron.runtime import track_fallbacks

    with track_fallbacks() as ops:
        yield ops
