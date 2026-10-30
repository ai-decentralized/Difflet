"""W0 unit tests (CPU only): the shared parity metrics and device-evidence helper.

``tests/numerical/neuron_parity.py`` is what every Wan bring-up gate (W2-W6) compares
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

from tests.numerical import neuron_parity as parity  # noqa: E402
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
    with pytest.raises(ValueError, match="shape"):
        parity.metrics(torch.zeros(2, 6), torch.zeros(2, 6), control=torch.zeros(12))


def test_metrics_with_a_control_reports_the_cosine_between_the_two_errors():
    # D68: err_cos = cos(actual - reference, control - reference), in fp64.
    ref = torch.tensor([1.0, -2.0, 3.0, -4.0])
    control = ref + torch.tensor([1.0, 1.0, 0.0, 0.0])
    m = parity.metrics(ref + torch.tensor([1.0, 0.0, 0.0, 0.0]), ref, control=control)
    assert set(m) == METRIC_KEYS | {"err_cos"}
    assert m["err_cos"] == pytest.approx(1 / math.sqrt(2), abs=1e-12)
    # The other keys do not depend on the control.
    assert {k: v for k, v in m.items() if k != "err_cos"} == parity.metrics(
        ref + torch.tensor([1.0, 0.0, 0.0, 0.0]), ref)
    # Same rounding pattern -> 1, the opposite one -> -1; bf16 inputs (exact values here) are
    # compared in fp64, not in bf16.
    same = parity.metrics(control.to(torch.bfloat16), ref, control=control)
    assert same["err_cos"] == pytest.approx(1.0, abs=1e-12)
    opposite = parity.metrics(2 * ref - control, ref, control=control.to(torch.bfloat16))
    assert opposite["err_cos"] == pytest.approx(-1.0, abs=1e-12)
    # An exact result has no error direction: nan, which check() treats as gated.
    assert math.isnan(parity.metrics(ref.clone(), ref, control=control)["err_cos"])


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
    # No err_cos recorded: whether the result rounds like the control is unknown, so the
    # same-dtype limit applies (D68). (1.05x and 1.25x imply err_cos 0.257: a decorrelated
    # result that, recorded with its err_cos, would be reported instead.)
    near = {"rel_l1": 1.05e-2, "cosine": 0.999}
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


def _same_dtype_ratio(r: float, err_cos: float) -> float:
    """Where a result ``r`` times the control from fp32, whose error has cosine ``err_cos``
    with the control's, lies from the control, in multiples of the control's distance (D69)."""
    return math.sqrt(r * r + 1 - 2 * r * err_cos)


def _result(r: float, err_cos: float | None, *, control_rel_l1: float = 1.0e-2,
            same_dtype: float | None = None) -> dict:
    """Metrics of a geometrically consistent result: its same-dtype distance follows from ``r``
    and ``err_cos`` unless given (needed only when ``err_cos`` is unknown)."""
    if same_dtype is None:
        same_dtype = _same_dtype_ratio(r, err_cos)
    m = {"rel_l1": r * control_rel_l1, "cosine": 0.999,
         "vs_bf16": {"rel_l1": same_dtype * control_rel_l1, "cosine": 0.999}}
    if err_cos is not None:
        m["err_cos"] = err_cos
    return m


def test_same_dtype_distance_follows_from_the_ratio_and_the_error_cosine():
    # D69: |actual - control| is the distance between the two error vectors, so the same-dtype
    # ratio is sqrt(r^2 + 1 - 2 r err_cos), exactly in L2. Built here from vectors.
    gen = torch.Generator().manual_seed(3)
    ref = torch.randn(4096, generator=gen, dtype=torch.float64)
    e_c = torch.randn(4096, generator=gen, dtype=torch.float64) * 1e-2
    other = torch.randn(4096, generator=gen, dtype=torch.float64)
    u = e_c / e_c.norm()
    w = other - (other @ u) * u
    w = w / w.norm()
    for r, c in ((1.0, 0.23), (0.929, 0.231), (1.15, 0.5), (1.025, 0.575), (2.5, 0.9)):
        e = r * e_c.norm() * (c * u + math.sqrt(1 - c * c) * w)
        m = parity.metrics(ref + e, ref, control=ref + e_c)
        control = parity.metrics(ref + e_c, ref)
        same = parity.metrics(ref + e, ref + e_c)
        assert m["err_cos"] == pytest.approx(c, abs=1e-12)
        assert m["rel_l2"] / control["rel_l2"] == pytest.approx(r, rel=1e-12)
        # |actual - control| / |control - ref|, from the two rel_l2 values: exact.
        l2 = same["rel_l2"] * float((ref + e_c).norm()) / (control["rel_l2"] * float(ref.norm()))
        assert l2 == pytest.approx(_same_dtype_ratio(r, c), rel=1e-9)
        # The rel_l1 ratio check() reads follows it to within a few per cent.
        assert same["rel_l1"] / control["rel_l1"] == pytest.approx(_same_dtype_ratio(r, c),
                                                                     rel=0.05)
    # Inside the gated regime (err_cos >= 0.5) a result within the magnitude gate (r <= 1.15)
    # is at most 1.083 from the control: the 1.2 limit cannot fail it.
    worst = max(_same_dtype_ratio(r / 1000, c / 1000)
                for r in range(0, 1151, 5) for c in range(500, 1001, 5))
    assert worst == pytest.approx(math.sqrt(1.15**2 - 1.15 + 1), abs=1e-12)
    assert worst < 1.0833 < parity.MAX_SAME_DTYPE_RATIO
    control = {"rel_l1": 1.0e-2, "cosine": 0.99873}
    for r in (0.0, 0.5, 0.9, 1.0, 1.1, 1.15):
        for c in (0.5, 0.6, 0.8, 1.0):
            assert parity.check(_result(r, c), control=control) == (True, [])


def test_check_gates_the_same_dtype_distance_only_when_the_errors_are_correlated():
    # D68: the same-dtype limit is gated only when the result's error points the way the
    # control's does (err_cos >= 0.5); D69: there it can only fail together with the magnitude
    # gate. Eager rounds like diffusers (W3: err_cos 0.57 / 0.66); the compiled 14B DiT does
    # not, because neuronx-cc keeps the residual stream in fp32 (0.23 / 0.30). Every case here
    # derives its same-dtype distance from r and err_cos.
    control = {"rel_l1": 1.0e-2, "cosine": 0.99873}
    # Correlated errors and a result far from fp32 (2.5x): both gates fail, no note.
    notes: list[str] = []
    ok, reasons = parity.check(_result(2.5, 0.9), control=control, notes=notes)
    assert not ok and len(reasons) == 2
    assert "max_ratio_vs_control" in reasons[0] and "max_same_dtype_ratio" in reasons[1]
    assert "1.658x" in reasons[1] and notes == []
    # Correlated and within the magnitude gate: nothing to report.
    assert parity.check(_result(1.0, 0.9), control=control, notes=notes) == (True, [])
    assert notes == []
    # Decorrelated (W3's compiled DiT): the same-dtype distance (1.198x here) is reported, and
    # the decorrelation is noted (D69), naming err_cos, the distance and the rule.
    ok, reasons = parity.check(_result(0.929, 0.231), control=control, notes=notes)
    assert (ok, reasons) == (True, [])
    assert len(notes) == 1
    assert "err_cos 0.231" in notes[0] and "1.197x" in notes[0] and "not gated" in notes[0]
    assert "D69" in notes[0]
    # The note appears whatever the distance: a decorrelated result 0.975x from the control.
    notes = []
    assert parity.check(_result(0.5, 0.3), control=control, notes=notes) == (True, [])
    assert len(notes) == 1 and "err_cos 0.300" in notes[0] and "0.975x" in notes[0]
    # No cliff at 1.0x: a decorrelated result 1.10x from fp32 (1.305x from the control) passes,
    # where D65 gated it ...
    assert parity.check(_result(1.10, 0.23), control=control) == (True, [])
    # ... and the magnitude gate still judges it on its own.
    ok, reasons = parity.check(_result(1.2, 0.23), control=control)
    assert not ok and len(reasons) == 1 and "max_ratio_vs_control" in reasons[0]
    # The boundary is inclusive: at 1.6x from fp32 (1.400x from the control), err_cos 0.5 is
    # gated (two reasons), 0.4999 reported (the magnitude gate alone, plus the note).
    ok, reasons = parity.check(_result(1.6, 0.5), control=control)
    assert not ok and len(reasons) == 2 and "max_same_dtype_ratio" in reasons[1]
    notes = []
    ok, reasons = parity.check(_result(1.6, 0.4999), control=control, notes=notes)
    assert not ok and len(reasons) == 1 and "max_ratio_vs_control" in reasons[0]
    assert len(notes) == 1
    # Unknown error direction (not recorded, or nan): gated. 1.05x from fp32 and 1.25x from
    # the control imply err_cos 0.257, but without err_cos the limit applies.
    ok, reasons = parity.check(_result(1.05, None, same_dtype=1.25), control=control)
    assert not ok and len(reasons) == 1 and "max_same_dtype_ratio" in reasons[0]
    ok, reasons = parity.check({**_result(1.05, None, same_dtype=1.25), "err_cos": math.nan},
                               control=control)
    assert not ok and len(reasons) == 1 and "max_same_dtype_ratio" in reasons[0]
    # A nan same-dtype distance fails in either regime (W3 review), naming the nan.
    for err_cos in (0.2, 0.9):
        nan_same = {**_result(1.0, err_cos), "vs_bf16": {"rel_l1": math.nan, "cosine": 0.999}}
        ok, reasons = parity.check(nan_same, control=control)
        assert not ok and len(reasons) == 1
        assert "same-dtype" in reasons[0] and "nan" in reasons[0]
    # The same-dtype cosine floor applies in either regime.
    for err_cos in (0.2, 0.9):
        low_cos = _result(1.0, err_cos)
        low_cos["vs_bf16"]["cosine"] = 0.5
        ok, reasons = parity.check(low_cos, control=control)
        assert not ok and len(reasons) == 1 and "same-dtype (vs_bf16) cosine" in reasons[0]
    assert parity.SAME_DTYPE_GATED_MIN_ERR_COS == 0.5
    assert not hasattr(parity, "SAME_DTYPE_GATED_ABOVE_RATIO")  # D65's rule is gone


def test_same_dtype_helpers_state_the_rule_check_applies():
    # Callers print "gated / reported" from these instead of re-deriving the rule. The values
    # are W2 eager's (1.0x, err_cos 0.575) and W3 compile's (0.929x, 0.231) at t=996, with the
    # same-dtype distance derived from them.
    control = {"rel_l1": 1.0e-2, "cosine": 0.99873}
    gated = _result(1.0, 0.575)
    reported = _result(0.929, 0.231)
    assert parity.same_dtype_gated(gated) is True
    assert parity.same_dtype_gated(reported) is False
    assert parity.same_dtype_gated({"rel_l1": 1.0, "cosine": 1.0}) is True
    assert parity.same_dtype_gated({**gated, "err_cos": math.nan}) is True
    assert parity.same_dtype_label(gated, control) == (
        "same-dtype 0.922x, gated (err_cos 0.575 >= 0.5, D68)")
    assert parity.same_dtype_label(reported, control) == (
        "same-dtype 1.197x, reported, not gated (err_cos 0.231 < 0.5, D68)")
    assert parity.same_dtype_label({"rel_l1": 1.0, "cosine": 1.0}, control) == (
        "same-dtype not measured")
    assert parity.same_dtype_label({**gated, "err_cos": None}, control) == (
        "same-dtype 0.922x, gated (err_cos not recorded, D68)")


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
        "notes": None,
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
