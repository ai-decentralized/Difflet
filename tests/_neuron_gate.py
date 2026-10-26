"""The Neuron gate: how the suite behaves on a host without the Neuron SDK.

Difflet's reference test environment is a Trainium host where ``neuronx_distributed``,
``torch_neuronx``, ``torch_xla`` and ``nkilib`` import. On a laptop, a GPU host or CI they
do not, and many tests would be red for exactly that reason. Those tests are not wrong,
they are Trainium-only; this module lets them *skip with a reason* there, by exact
signature, and does nothing at all when the SDK is present.

Two mechanisms, both reviewable:

* ``NEURON_COLLECT_IGNORE`` — test modules whose *import* needs the SDK (they import a
  Trainium application or ``torch_xla`` at module level). Listed explicitly, relative to
  ``tests/``, so a stale entry is a test failure (``test_neuron_gate.py``) rather than a
  silently vanished module. Remove a line when its module becomes importable.
* ``neuron_only_failure`` — the three exception signatures a Neuron-less host produces at
  run time. Only these convert a failure into a skip; everything else stays red.

Loaded by ``tests/conftest.py`` by path (pytest runs with ``--import-mode=importlib``, so
there is no package to import it from).
"""

from __future__ import annotations

import importlib.util
import re

NEURON_MODULE_PREFIXES = (
    "neuronx_distributed",
    "neuronx_distributed_inference",
    "torch_neuronx",
    "torch_xla",
    "nki",
    "nkilib",
    "neuronxcc",
    "libneuronxla",
)

TRAINIUM_RUNTIME_MESSAGE = "Trainium runtime dependencies are unavailable"

# difflet.registry.ModelEntry.require_backend's exact wording.
_TRAINIUM_ONLY_BACKEND = re.compile(
    r"does not support backend 'cpu'; supported backends: [^;]*\btrainium\b"
)

# Node ids convert_report turned into skips in this session. tests/conftest.py dumps
# them to $NEURON_GATE_REPORT at session end; the CPU gate diffs that file against its
# committed neuron-skips.txt, so a *new* Neuron-only skip (a regression hiding as a
# skip) fails the gate instead of only nudging the skipped count.
CONVERTED: list[str] = []

NEURON_COLLECT_IGNORE = [
    "unit/layers/test_normalization.py",
    "unit/models/flux/test_flux_application.py",
    "unit/models/flux/test_flux_clip.py",
    "unit/models/flux/test_flux_sp.py",
    "unit/models/flux/test_flux_t5.py",
    "unit/models/flux/test_flux_teacache_probe_keys.py",
    "unit/models/flux/test_flux_tp_reduce_dtype.py",
    "unit/models/flux/test_flux_vae.py",
    "unit/models/hunyuan_video/test_hunyuan_video_teacache_probe_keys.py",
    "unit/models/ltx_2/test_ltx2_tp_reduce_dtype.py",
    "unit/models/ltx_2/test_ltx_2_application.py",
    "unit/models/ltx_2/test_ltx_2_entry.py",
    "unit/models/ltx_2/test_ltx_2_pipeline.py",
    "unit/models/ltx_2/test_ltx_2_pipeline_orchestrator.py",
    "unit/models/minimax_h3/test_trainium_transformer.py",
    "unit/models/minimax_h3/test_vae.py",
    "unit/models/qwen_image/test_qwen_application.py",
    "unit/models/qwen_image/test_qwen_entry.py",
    "unit/models/qwen_image/test_qwen_image_pipeline_orchestrator.py",
    "unit/models/qwen_image/test_qwen_pipeline.py",
    "unit/models/qwen_image/test_qwen_teacache_probe_free.py",
    "unit/models/qwen_image/test_qwen_teacache_probe_keys.py",
    "unit/models/qwen_image/test_qwen_tp_reduce_dtype.py",
    "unit/models/wan/test_wan_application.py",
    "unit/pipeline/test_wan_ltx2_device_probe.py",
    "unit/utils/test_diffusers_adapter.py",
    "unit/utils/test_kv_cache_reconstruct_utils.py",
    "unit/utils/test_tensor_replacement_registry.py",
]


def neuron_available() -> bool:
    return importlib.util.find_spec("neuronx_distributed") is not None


def neuron_only_failure(exc: BaseException) -> str | None:
    """Return a skip reason when ``exc`` is one a Neuron-less host produces, else None."""
    if isinstance(exc, ModuleNotFoundError):
        name = exc.name or ""
        if any(name == p or name.startswith(p + ".") for p in NEURON_MODULE_PREFIXES):
            return f"needs Neuron SDK: module {name!r} not installed"
        return None
    if isinstance(exc, RuntimeError) and str(exc).startswith(TRAINIUM_RUNTIME_MESSAGE):
        return "needs Neuron SDK: Trainium runtime dependencies are unavailable"
    if isinstance(exc, ValueError) and _TRAINIUM_ONLY_BACKEND.search(str(exc)):
        return "needs Neuron SDK: model is registered for the trainium backend only"
    return None


def convert_report(report, excinfo, *, neuron_available: bool) -> bool:
    """Turn a failed test report into a skip when the failure is Neuron-only.

    Returns True when the report was converted. Never converts when the SDK is
    available: on a Neuron host every one of these signatures is a real failure.
    """
    if neuron_available or excinfo is None or getattr(report, "when", "call") != "call":
        return False
    if report.outcome != "failed":
        return False
    reason = neuron_only_failure(excinfo.value)
    if reason is None:
        return False
    report.outcome = "skipped"
    report.longrepr = (str(getattr(report, "fspath", "")), 0, reason)
    CONVERTED.append(str(getattr(report, "nodeid", "")))
    return True


def dump_conversions(path) -> None:
    """Write the converted node ids, sorted and unique, one per line."""
    lines = sorted(set(CONVERTED))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(f"{line}\n" for line in lines))
