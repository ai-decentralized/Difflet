"""Pytest configuration for all tests.

Real ``torch`` (and the rest of the validated Neuron stack) is importable in the
reference test environment, so tests use the real library. Tests that need to
avoid the heavy Neuron runtime (``torch_xla`` / ``neuronx_distributed``) should
mock only those modules locally rather than replacing ``torch`` globally.

On a host **without** the Neuron SDK (a laptop, a GPU host, CI) the Neuron gate
in ``tests/_neuron_gate.py`` applies: test modules that cannot even import without
the SDK are not collected (an explicit, reviewed list), and a test that fails with
one of the three exact Neuron-less signatures is reported as skipped with the reason
``needs Neuron SDK: ...``. When the SDK is present the gate does nothing, so a
Trainium host sees every failure. Nothing is hidden on either kind of host: the
skip count and reasons are in the summary (``-rs``), and the gate prints its state
in the header.

With ``CPU_GATE_SHARD=k/n`` only shard k of the ``tests/unit`` files is collected
(``tests/_shard.py``); CI runs the trainium-native suite as three shards.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


def _load(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_gate = _load("_neuron_gate.py", "difflet_tests_neuron_gate")
NEURON_AVAILABLE = _gate.neuron_available()
_shard = _load("_shard.py", "difflet_tests_shard")
_SHARD_FILES = _shard.shard_files(os.environ.get("CPU_GATE_SHARD"))

if not NEURON_AVAILABLE:
    collect_ignore = list(_gate.NEURON_COLLECT_IGNORE)


def pytest_report_header(config):
    if NEURON_AVAILABLE:
        return "neuron gate: inactive (neuronx_distributed importable)"
    return (
        f"neuron gate: active — {len(_gate.NEURON_COLLECT_IGNORE)} Trainium-only test modules "
        "not collected; Neuron-only failures reported as skips (see tests/_neuron_gate.py)"
    )


def pytest_ignore_collect(collection_path, config):
    # CPU_GATE_SHARD=k/n (scripts/cpu-gate.sh, CI): only shard k's tests/unit files are
    # collected, so the others are not even imported (tests/_shard.py).
    if _SHARD_FILES is None or not collection_path.name.startswith("test_"):
        return None
    try:
        name = collection_path.resolve().relative_to(_shard.ROOT).as_posix()
    except ValueError:
        return None
    if collection_path.suffix != ".py" or not name.startswith("tests/unit/"):
        return None
    return True if name not in _SHARD_FILES else None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    _gate.convert_report(report, call.excinfo, neuron_available=NEURON_AVAILABLE)


def pytest_sessionfinish(session, exitstatus):
    # The CPU gate sets NEURON_GATE_REPORT and diffs the result against its committed
    # neuron-skips.txt; a conversion not in that file is a regression hiding as a skip.
    # Under pytest-xdist each worker hands its conversions to the controller, which writes.
    _gate.finish_session(session.config, os.environ.get("NEURON_GATE_REPORT"))


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    # pytest-xdist controller hook: a worker finished; keep its Neuron-gate conversions.
    _gate.collect_worker_conversions(node)
