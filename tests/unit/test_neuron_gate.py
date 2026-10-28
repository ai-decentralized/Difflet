"""The Neuron gate (tests/_neuron_gate.py, wired in tests/conftest.py).

The gate must never fire when the SDK is present, and it
must only convert the three exact Neuron-less signatures, never a look-alike.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_TESTS_DIR = Path(__file__).resolve().parents[1]


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "_neuron_gate_under_test", _TESTS_DIR / "_neuron_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def _mnf(name: str) -> ModuleNotFoundError:
    exc = ModuleNotFoundError(f"No module named {name!r}")
    exc.name = name
    return exc


def _failed_report(exc: BaseException):
    report = SimpleNamespace(outcome="failed", when="call", fspath="tests/unit/x.py", longrepr=None)
    return report, SimpleNamespace(value=exc)


def test_neuron_gate_converts_known_signatures():
    assert gate.neuron_only_failure(_mnf("neuronx_distributed.parallel_layers")) is not None
    assert gate.neuron_only_failure(_mnf("torch_xla")) is not None
    assert gate.neuron_only_failure(_mnf("nkilib")) is not None
    assert gate.neuron_only_failure(_mnf("nki")) is not None
    assert gate.neuron_only_failure(_mnf("nki.language")) is not None
    assert (
        gate.neuron_only_failure(RuntimeError("Trainium runtime dependencies are unavailable."))
        is not None
    )
    assert (
        gate.neuron_only_failure(
            ValueError(
                "model 'wan' does not support backend 'cpu'; supported backends: trainium, tpu"
            )
        )
        is not None
    )


def test_neuron_gate_ignores_other_errors():
    assert gate.neuron_only_failure(ValueError("trainium is mentioned but this is a bug")) is None
    assert gate.neuron_only_failure(_mnf("requests")) is None
    assert gate.neuron_only_failure(_mnf("torch_xla_lookalike")) is None
    assert gate.neuron_only_failure(RuntimeError("something else about Trainium")) is None
    assert gate.neuron_only_failure(AssertionError("x")) is None
    # a model that lists cpu among its backends is not trainium-only
    assert (
        gate.neuron_only_failure(
            ValueError(
                "model 'q' does not support backend 'cuda'; supported backends: cpu, trainium"
            )
        )
        is None
    )


def test_convert_report_marks_neuron_only_failure_skipped():
    report, excinfo = _failed_report(RuntimeError("Trainium runtime dependencies are unavailable."))
    assert gate.convert_report(report, excinfo, neuron_available=False) is True
    assert report.outcome == "skipped"
    assert report.longrepr[2].startswith("needs Neuron SDK")


def test_neuron_gate_inactive_when_sdk_present():
    report, excinfo = _failed_report(RuntimeError("Trainium runtime dependencies are unavailable."))
    assert gate.convert_report(report, excinfo, neuron_available=True) is False
    assert report.outcome == "failed"


def test_convert_report_leaves_other_failures_alone():
    report, excinfo = _failed_report(AssertionError("real bug"))
    assert gate.convert_report(report, excinfo, neuron_available=False) is False
    assert report.outcome == "failed"


def test_convert_report_only_touches_the_call_phase():
    report, excinfo = _failed_report(_mnf("torch_neuronx"))
    report.when = "setup"
    assert gate.convert_report(report, excinfo, neuron_available=False) is False


@pytest.mark.parametrize("relpath", gate.NEURON_COLLECT_IGNORE)
def test_collect_ignore_entries_exist(relpath):
    # A stale entry would hide a module that no longer exists — or no longer needs the SDK.
    assert (_TESTS_DIR / relpath).is_file(), relpath


def test_convert_report_records_the_converted_nodeid(tmp_path):
    gate.CONVERTED.clear()
    report, excinfo = _failed_report(_mnf("torch_neuronx"))
    report.nodeid = "tests/unit/x.py::test_a"
    gate.convert_report(report, excinfo, neuron_available=False)
    assert gate.CONVERTED == ["tests/unit/x.py::test_a"]


def test_dump_conversions_writes_sorted_unique_nodeids(tmp_path):
    gate.CONVERTED.clear()
    gate.CONVERTED.extend(["tests/unit/b.py::t", "tests/unit/a.py::t", "tests/unit/b.py::t"])
    out = tmp_path / "neuron-skips.txt"
    gate.dump_conversions(out)
    assert out.read_text().splitlines() == ["tests/unit/a.py::t", "tests/unit/b.py::t"]
