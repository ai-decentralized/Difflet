"""W0 unit tests (CPU only): the shared parity metrics and device-evidence helper.

``difflet/backends/neuron/core/parity.py`` is what every Wan bring-up gate (W2-W6) compares
device output with and what proves "the blocks compiled and nothing fell back". A threshold that
drifts here loosens every later gate silently, so the defaults are pinned by value and against
the TPU numerical gate they come from.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from difflet.backends.neuron.core import parity  # noqa: E402
from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.unit.backends._neuron_toy import (  # noqa: E402
    TOY_DIM,
    ToyLaunchApplication,
    prepare_toy_work_dir,
    toy_launch_inputs,
)

METRIC_KEYS = {"cosine", "rel_l1", "rel_l2", "max_abs", "rel_max", "ref_absmean"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    parallel_mesh.destroy_parallel_mesh()
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    monkeypatch.delenv(parity.NEFF_CACHE_ENV, raising=False)
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path / "difflet_cache"))
    yield
    parallel_mesh.destroy_parallel_mesh()


# ---------------------------------------------------------------------------- metrics


def test_metrics_identical_tensors_are_perfect():
    x = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(0))
    m = parity.metrics(x, x.clone())
    assert set(m) == METRIC_KEYS
    assert m["cosine"] == pytest.approx(1.0, abs=1e-12)
    assert (m["rel_l1"], m["rel_l2"], m["max_abs"], m["rel_max"]) == (0.0, 0.0, 0.0, 0.0)
    assert all(isinstance(v, float) for v in m.values())


def test_metrics_sign_flipped_tensors_have_cosine_minus_one():
    x = torch.randn(5, 7, generator=torch.Generator().manual_seed(1))
    m = parity.metrics(-x, x)
    assert m["cosine"] == pytest.approx(-1.0, abs=1e-12)
    assert m["rel_l1"] == pytest.approx(2.0)
    assert m["rel_l2"] == pytest.approx(2.0)
    assert m["rel_max"] == pytest.approx(2.0)


def test_metrics_known_values_in_fp64_across_dtypes():
    ref = [1.0, -2.0, 3.0, -4.0]
    act = [1.5, -2.0, 3.0, -5.0]
    diff = [a - r for a, r in zip(act, ref)]
    dot = sum(a * r for a, r in zip(act, ref))
    norm_a, norm_r = (math.sqrt(sum(v * v for v in vs)) for vs in (act, ref))
    # bf16 actual against an fp32 reference: every value above is exact in bf16, so the
    # numbers must come out as the hand computation, not as a bf16-accumulated approximation.
    m = parity.metrics(torch.tensor(act, dtype=torch.bfloat16).reshape(2, 2),
                       torch.tensor(ref).reshape(2, 2))
    assert m["cosine"] == pytest.approx(dot / (norm_a * norm_r), abs=1e-12)
    assert m["rel_l1"] == pytest.approx(sum(map(abs, diff)) / sum(map(abs, ref)))
    assert m["rel_l2"] == pytest.approx(math.sqrt(sum(d * d for d in diff)) / norm_r)
    assert m["max_abs"] == 1.0
    assert m["ref_absmean"] == 2.5
    assert m["rel_max"] == 0.25  # max |a - r| / max |r| = 1 / 4


def test_metrics_rejects_shapes_that_only_match_when_flattened():
    with pytest.raises(ValueError, match="shape"):
        parity.metrics(torch.zeros(2, 6), torch.zeros(3, 4))
    with pytest.raises(ValueError, match="shape"):
        parity.metrics(torch.zeros(2, 6), torch.zeros(12))


# ----------------------------------------------------------------------------- check


def test_check_accepts_the_recorded_tpu_oracle_numbers():
    # tests/numerical/test_tpu_vs_diffusers.py: Wan scored 1.042x the bf16 control, whose own
    # cosine against fp32 is 0.99873 (upstream bf16); same-dtype distance ratio 0.91.
    control = {"rel_l1": 1.0e-2, "cosine": 0.99873}
    m = {"rel_l1": 1.042e-2, "cosine": 0.99873, "vs_bf16": {"rel_l1": 0.91e-2, "cosine": 0.9999}}
    assert parity.check(m, control=control) == (True, [])
    assert parity.check({"rel_l1": 1.042e-2, "cosine": 0.99873}, control=control) == (True, [])


def test_check_fails_with_a_reason_naming_the_threshold():
    control = {"rel_l1": 1.0e-2, "cosine": 0.99873}
    ok, reasons = parity.check({"rel_l1": 1.3e-2, "cosine": 0.99873}, control=control)
    assert ok is False
    assert len(reasons) == 1
    assert "max_ratio_vs_control" in reasons[0] and "1.15" in reasons[0] and "1.3" in reasons[0]
    # The limit is inclusive, and a caller-supplied limit replaces the default.
    assert parity.check({"rel_l1": 1.15, "cosine": 1.0}, control={"rel_l1": 1.0})[0] is True
    assert parity.check({"rel_l1": 1.3e-2, "cosine": 1.0}, control=control,
                        max_ratio_vs_control=1.5) == (True, [])


def test_check_gates_the_same_dtype_distance_and_the_cosine_floor():
    control = {"rel_l1": 1.0e-2, "cosine": 0.99873}
    near = {"rel_l1": 1.0e-2, "cosine": 0.999}
    ok, reasons = parity.check({**near, "vs_bf16": {"rel_l1": 1.25e-2, "cosine": 0.999}},
                               control=control)
    assert not ok and len(reasons) == 1
    assert "max_same_dtype_ratio" in reasons[0] and "1.2" in reasons[0]
    # Without a control only the absolute floor can be judged: on the headline cosine and on
    # the same-dtype one.
    ok, reasons = parity.check({"rel_l1": 1.0, "cosine": 0.5})
    assert not ok and len(reasons) == 1 and "min_cosine" in reasons[0] and "0.995" in reasons[0]
    ok, reasons = parity.check({"rel_l1": 1.0, "cosine": 0.999, "vs_bf16": {"cosine": 0.5}})
    assert not ok and len(reasons) == 1 and "same-dtype" in reasons[0]
    assert parity.check({"rel_l1": 1.0, "cosine": 0.995}) == (True, [])
    # A NaN metric is never a pass, and a zero-error control cannot be divided by.
    assert parity.check({"rel_l1": math.nan, "cosine": math.nan}, control=control)[0] is False
    assert parity.check({"rel_l1": 1e-3, "cosine": 1.0}, control={"rel_l1": 0.0})[0] is False
    assert parity.check({"rel_l1": 0.0, "cosine": 1.0}, control={"rel_l1": 0.0})[0] is True


def _tpu_gate_module():
    path = Path(__file__).resolve().parents[2] / "numerical" / "test_tpu_vs_diffusers.py"
    spec = importlib.util.spec_from_file_location("_tpu_gate_thresholds", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_default_thresholds_are_pinned_to_the_tpu_gate():
    defaults = {
        name: p.default for name, p in inspect.signature(parity.check).parameters.items()
    }
    assert defaults == {
        "metrics": inspect.Parameter.empty, "control": None,
        "max_ratio_vs_control": 1.15, "max_same_dtype_ratio": 1.2, "min_cosine": 0.995,
    }
    assert (parity.MAX_RATIO_VS_CONTROL, parity.MAX_SAME_DTYPE_RATIO, parity.MIN_COSINE) == (
        1.15, 1.2, 0.995)
    tpu = _tpu_gate_module()  # test_tpu_vs_diffusers.py:71,76,81
    assert tpu.MAX_ERROR_RATIO_VS_CONTROL == defaults["max_ratio_vs_control"]
    assert tpu.MAX_SAME_DTYPE_DISTANCE_RATIO == defaults["max_same_dtype_ratio"]
    assert tpu.MIN_ABSOLUTE_COSINE == defaults["min_cosine"]


# ------------------------------------------------------------------- device_evidence

EVIDENCE_KEYS = {"fallbacks", "unwarmed_shapes", "compiled_blocks", "phase_seconds",
                 "counters", "neff_count", "peak_device_mem_gb"}
COUNTERS_ZERO = {"TotalCompilations": 0, "PersistentHits": 0, "InMemoryHits": 0}


@pytest.fixture(scope="module")
def toy_ckpt(tmp_path_factory):
    pytest.importorskip("safetensors.torch")
    pytest.importorskip("accelerate")
    return prepare_toy_work_dir(tmp_path_factory.mktemp("w0_toy"))


def _toy_app(ckpt, exec_mode):
    return ToyLaunchApplication(model_path=ckpt, parallel=DiffletParallelConfig(),
                                dtype=torch.float32, exec_mode=exec_mode, device="cpu")


class _TripwireNeuronx(types.ModuleType):
    """A torch_neuronx that fails on any use: the CPU path must not touch it."""

    def __getattr__(self, name):
        raise AssertionError(f"device_evidence touched torch_neuronx.{name} on a CPU app")


def test_device_evidence_on_a_cpu_toy_app_reads_nothing_from_the_device(
    toy_ckpt, monkeypatch
):
    monkeypatch.setitem(sys.modules, "torch_neuronx", _TripwireNeuronx("torch_neuronx"))
    app = _toy_app(toy_ckpt, "eager")
    evidence = parity.device_evidence(app, [])
    assert set(evidence) == EVIDENCE_KEYS
    assert evidence["fallbacks"] == [] and evidence["unwarmed_shapes"] == []
    assert evidence["compiled_blocks"] == [] and evidence["phase_seconds"] == {}
    assert evidence["counters"] == COUNTERS_ZERO
    assert evidence["neff_count"] == 0 and evidence["peak_device_mem_gb"] is None
    # The caller's fallback list is reported as given, and as a copy.
    seen = ["aten::foo"]
    assert parity.device_evidence(app, seen)["fallbacks"] == ["aten::foo"]
    assert evidence["fallbacks"] is not seen


def test_device_evidence_reports_compiled_blocks_phases_and_unwarmed_shapes(toy_ckpt):
    app = _toy_app(toy_ckpt, "compile")
    app.load()
    evidence = parity.device_evidence(app, [])
    assert evidence["compiled_blocks"] == [f"blocks.{i}" for i in range(3)]
    assert evidence["unwarmed_shapes"] == []
    assert evidence["phase_seconds"]["warmup forward"] > 0.0
    assert set(evidence["phase_seconds"]) >= {"build on meta", "load checkpoint", "compile"}
    # A forward at a shape the warm-up never saw is the silent-recompile hazard; it must show.
    app(toy_launch_inputs()[:, : 128])
    evidence = parity.device_evidence(app, [])
    assert evidence["unwarmed_shapes"] == [[[1, 128, TOY_DIM]]]
    import json

    assert json.loads(json.dumps(evidence)) == evidence  # plain JSON types throughout
    # Eager mode compiles nothing, and says so.
    eager = _toy_app(toy_ckpt, "eager")
    eager.load()
    assert parity.device_evidence(eager, [])["compiled_blocks"] == []


def _neuron_app():
    return types.SimpleNamespace(
        device=types.SimpleNamespace(type="neuron"), unwarmed_shapes=[],
        compiled_blocks=["blocks.0"], phase_seconds={"warmup forward": 1.5},
    )


def test_device_evidence_reads_counters_and_peak_memory_only_once_torch_neuronx_is_loaded(
    monkeypatch,
):
    app = _neuron_app()
    monkeypatch.delitem(sys.modules, "torch_neuronx", raising=False)
    monkeypatch.delitem(sys.modules, "torch_neuronx.metrics", raising=False)
    evidence = parity.device_evidence(app, [])
    assert evidence["counters"] == COUNTERS_ZERO and evidence["peak_device_mem_gb"] is None
    assert "torch_neuronx" not in sys.modules  # reading evidence must not start the runtime

    values = {"CompilationCache.TotalCompilations": 2, "CompilationCache.PersistentHits": None,
              "CompilationCache.InMemoryHits": 5}
    fake = types.ModuleType("torch_neuronx")
    fake.max_memory_allocated = lambda: 3 * 2**30 + 2**29  # 3.5 GiB
    fake_metrics = types.ModuleType("torch_neuronx.metrics")
    fake_metrics.get_counter_value = values.get  # None until the counter's first increment
    fake.metrics = fake_metrics
    monkeypatch.setitem(sys.modules, "torch_neuronx", fake)
    monkeypatch.setitem(sys.modules, "torch_neuronx.metrics", fake_metrics)
    evidence = parity.device_evidence(app, ["aten::foo"])
    assert evidence["counters"] == {"TotalCompilations": 2, "PersistentHits": 0, "InMemoryHits": 5}
    assert evidence["peak_device_mem_gb"] == 3.5
    assert evidence["fallbacks"] == ["aten::foo"]
    assert evidence["compiled_blocks"] == ["blocks.0"]
    assert evidence["phase_seconds"] == {"warmup forward": 1.5}


def test_device_evidence_counts_neffs_in_the_cache_the_run_uses(tmp_path, monkeypatch):
    app = types.SimpleNamespace(device=types.SimpleNamespace(type="cpu"), unwarmed_shapes=[],
                                compiled_blocks=[], phase_seconds={})
    default = tmp_path / "difflet_cache" / "neuron" / "neff"  # default_neff_cache_dir()
    (default / "a" / "b").mkdir(parents=True)
    for name in ("a/one.neff", "a/b/two.neff", "a/other.txt", "root.neff"):
        (default / name).write_text("x")
    assert parity.device_evidence(app, [])["neff_count"] == 3
    # A launcher's TORCH_NEURONX_NEFF_CACHE_DIR wins, as configure_compile_cache keeps it.
    chosen = tmp_path / "launcher_cache"
    chosen.mkdir()
    (chosen / "x.neff").write_text("x")
    monkeypatch.setenv(parity.NEFF_CACHE_ENV, str(chosen))
    assert parity.device_evidence(app, [])["neff_count"] == 1
