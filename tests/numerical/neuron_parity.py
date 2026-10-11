"""Shared parity metrics and device evidence for model bring-up on the neuron backend.

Every gate that compares a device output with a CPU reference and has to prove "the blocks
compiled and nothing fell back" goes through this module, so the metrics, the thresholds and
the blind spot of the fallback check live in one place:

* ``metrics(actual, reference, control=...)``: the numerical-oracle dict, computed in fp64,
  plus ``err_cos`` (how the result's error lines up with the bf16 control's) given a control.
* ``check(metrics, control=...)``: the TPU numerical gate's three thresholds as defaults, the
  same-dtype one gated only when the result rounds like the bf16 control (D68);
  ``same_dtype_gated`` / ``same_dtype_label`` say which, for callers' logs.
* ``device_evidence(app, fallbacks)``: what the run itself shows about compilation and memory.

Pure torch: it imports on a CPU-only host and never imports ``torch_neuronx`` or starts the
Neuron runtime (``device_evidence`` reads ``torch_neuronx`` only if it is already loaded).
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

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
    "metrics",
    "same_dtype_gated",
    "same_dtype_label",
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
#: upstream bf16); below it, the same-dtype distance is reported, not gated. See ``check``.
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

    * ``cosine`` (and the same-dtype cosine) at least ``min_cosine``;
    * with ``control``: ``rel_l1 / control["rel_l1"]`` at most ``max_ratio_vs_control``;
    * with ``control`` and ``"vs_bf16"``: the same-dtype ``rel_l1 / control["rel_l1"]`` is never
      nan, and is at most ``max_same_dtype_ratio`` when ``same_dtype_gated(metrics)``: the
      result's error has cosine ``metrics["err_cos"]`` (from ``metrics(..., control=...)``) of
      at least ``SAME_DTYPE_GATED_MIN_ERR_COS`` (0.5) with the control's, or no recorded
      ``err_cos`` (decision D68). Below 0.5 the same-dtype distance is reported, not gated,
      and a note saying so is appended to ``notes`` when the caller passes a list and the
      distance is over the limit.

    Why D68: the same-dtype distance asks "does the result round like upstream bf16?", which
    presumes that it does: then its distance from upstream bf16 is a small residual and 1.2x
    the control's distance from fp32 flags a real difference. That premise is measurable, as
    the cosine between the two error vectors. When the errors are decorrelated, the distance
    between result and control is set by their sizes alone, sqrt(r^2 + 1 - 2 r err_cos) times
    the control's for a result r times the control from fp32, about 1.24x at r = 1.0 and
    err_cos 0.23, whatever the result's correctness; the main gate (r <= 1.15) then judges it,
    as every mode. Measured in W3: the per-block compiled 14B Wan DiT is 0.929x / 0.832x the
    control from fp32 at t=996 / 500 but 1.201x / 1.097x it from diffusers bf16, with err_cos
    0.23 / 0.30: neuronx-cc keeps the residual stream in fp32 inside each compiled block
    although the StableHLO it receives rounds it to bf16 as diffusers does. Eager rounds like
    diffusers (err_cos 0.57 / 0.66) and stays gated (W2: 1.025x vs fp32, same-dtype 0.931x).
    D68 replaces D65, which gated the same-dtype distance only above 1.0x the control from
    fp32: with decorrelated errors that made 1.0x, not 1.15x, the compiled path's effective
    limit (W3 review).

    Limits are inclusive, and a nan never passes: a nan ratio fails the main gate, a nan
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
            elif not same_ratio <= max_same_dtype_ratio:
                if same_dtype_gated(metrics):
                    reasons.append(
                        f"same-dtype (vs_bf16) rel_l1 {same_dtype['rel_l1']:.3e} is "
                        f"{same_ratio:.3f}x the control's {control['rel_l1']:.3e}, above "
                        f"max_same_dtype_ratio {max_same_dtype_ratio}"
                    )
                elif notes is not None:
                    notes.append(
                        f"same-dtype (vs_bf16) rel_l1 {same_dtype['rel_l1']:.3e} is "
                        f"{same_ratio:.3f}x the control's, reported, not gated: the result's "
                        f"error has cosine {metrics['err_cos']:.3f} with the control's (below "
                        f"{SAME_DTYPE_GATED_MIN_ERR_COS}), so it does not round like upstream "
                        f"bf16 and the distance measures the control's own rounding (D68)"
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
