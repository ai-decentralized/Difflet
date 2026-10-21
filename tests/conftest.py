"""Pytest configuration for all tests.

Real ``torch`` (and the rest of the validated Neuron stack) is importable in the
reference test environment, so tests use the real library. Tests that need to
avoid the heavy Neuron runtime (``torch_xla`` / ``neuronx_distributed``) should
mock only those modules locally rather than replacing ``torch`` globally.

On a host **without** the Neuron SDK (a laptop, the CUDA box, CI) the Neuron gate
in ``tests/_neuron_gate.py`` applies: test modules that cannot even import without
the SDK are not collected (an explicit, reviewed list), and a test that fails with
one of the three exact Neuron-less signatures is reported as skipped with the reason
``needs Neuron SDK: ...``. When the SDK is present the gate does nothing, so a
Trainium host sees every failure. Nothing is hidden on either kind of host: the
skip count and reasons are in the summary (``-rs``), and the gate prints its state
in the header. Introduced by the CUDA backend plan (Phase 1, task 1.10).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


def _load_neuron_gate():
    path = Path(__file__).with_name("_neuron_gate.py")
    spec = importlib.util.spec_from_file_location("difflet_tests_neuron_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_gate = _load_neuron_gate()
NEURON_AVAILABLE = _gate.neuron_available()

if not NEURON_AVAILABLE:
    collect_ignore = list(_gate.NEURON_COLLECT_IGNORE)


def pytest_report_header(config):
    if NEURON_AVAILABLE:
        return "neuron gate: inactive (neuronx_distributed importable)"
    return (
        f"neuron gate: active — {len(_gate.NEURON_COLLECT_IGNORE)} Trainium-only test modules "
        "not collected; Neuron-only failures reported as skips (see tests/_neuron_gate.py)"
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    _gate.convert_report(report, call.excinfo, neuron_available=NEURON_AVAILABLE)


def pytest_sessionfinish(session, exitstatus):
    # The CPU gate sets NEURON_GATE_REPORT and diffs the result against its committed
    # neuron-skips.txt; a conversion not in that file is a regression hiding as a skip.
    path = os.environ.get("NEURON_GATE_REPORT")
    if path:
        _gate.dump_conversions(path)
