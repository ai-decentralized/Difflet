from __future__ import annotations

import json
import os

import pytest

from difflet.serving.engines.stage_observability import (
    CudaDeviceMemoryProbe,
    DeviceMemorySample,
    NeuronDeviceMemoryProbe,
    NullDeviceMemoryProbe,
    RequestStageBreakdown,
    StageLoadRecorder,
    StageTiming,
    format_bytes,
    record_stage_load,
    recording_stage_loads,
    select_device_memory_probe,
)


class _ScriptedProbe:
    """A probe that returns a fixed sequence of resident-byte readings."""

    def __init__(self, readings, *, scope: str = "process", reason: str | None = None) -> None:
        self.name = "scripted"
        self._readings = list(readings)
        self._scope = scope
        self._reason = reason
        self.samples = 0

    def sample(self):
        self.samples += 1
        if not self._readings:
            return None
        value = self._readings.pop(0)
        if value is None:
            return None
        return DeviceMemorySample(int(value), self._scope)

    @property
    def unavailable_reason(self):
        return self._reason


def test_recorder_attributes_each_stage_its_own_delta():
    # baseline, (before, after) per stage, final
    probe = _ScriptedProbe([0, 0, 10, 10, 40, 40, 45, 45])
    recorder = StageLoadRecorder(probe, backend="cuda")

    with recording_stage_loads(recorder):
        with record_stage_load("text"):
            pass
        with record_stage_load("generate"):
            pass
        with record_stage_load("vae"):
            pass

    report = recorder.report()

    assert report.attribution == "per_stage"
    assert report.scope == "process"
    assert [stage.stage_id for stage in report.stages] == ["text", "generate", "vae"]
    assert [stage.device_bytes for stage in report.stages] == [10, 30, 5]
    assert report.total_device_bytes == 45
    assert report.attributed_device_bytes == 45
    assert report.unattributed_device_bytes == 0
    assert all(stage.load_seconds >= 0.0 for stage in report.stages)
    assert report.reason is None


def test_recorder_reports_bytes_no_stage_claimed():
    # A load that happens outside any marked boundary shows up as unattributed, not as
    # one of the stages' residency.
    probe = _ScriptedProbe([0, 0, 10, 100])
    recorder = StageLoadRecorder(probe, backend="cuda")

    with recording_stage_loads(recorder):
        with record_stage_load("pipeline"):
            pass

    report = recorder.report()

    assert report.attributed_device_bytes == 10
    assert report.total_device_bytes == 100
    assert report.unattributed_device_bytes == 90


def test_recorder_says_total_only_when_no_stage_marks_its_load():
    probe = _ScriptedProbe([0, 70])
    recorder = StageLoadRecorder(probe, backend="trainium")

    with recording_stage_loads(recorder):
        pass

    report = recorder.report()

    assert report.attribution == "total_only"
    assert report.total_device_bytes == 70
    assert report.attributed_device_bytes is None
    assert report.unattributed_device_bytes is None
    assert "no stage marked its load boundary" in (report.reason or "")


def test_recorder_says_unavailable_rather_than_guessing():
    probe = _ScriptedProbe([], reason="neuron-monitor not on PATH")
    recorder = StageLoadRecorder(probe, backend="trainium")

    with recording_stage_loads(recorder):
        with record_stage_load("text"):
            pass

    report = recorder.report()

    assert report.attribution == "unavailable"
    assert report.total_device_bytes is None
    assert report.stages[0].device_bytes is None
    assert report.reason == "neuron-monitor not on PATH"


def test_recorder_marks_the_whole_report_host_scoped_if_any_sample_is():
    probe = _ScriptedProbe([0, 0, 10, 10], scope="host")
    recorder = StageLoadRecorder(probe, backend="trainium")

    with recording_stage_loads(recorder):
        with record_stage_load("pipeline"):
            pass

    assert recorder.report().scope == "host"


def test_one_boundary_can_cover_several_stages_without_splitting_the_bytes():
    probe = _ScriptedProbe([0, 0, 24, 24])
    recorder = StageLoadRecorder(probe, backend="trainium")

    with recording_stage_loads(recorder):
        with record_stage_load("prompt_encoder", "generation"):
            pass

    record = recorder.report().stages[0]

    assert record.stage_ids == ("prompt_encoder", "generation")
    assert record.stage_id == "prompt_encoder+generation"
    assert record.device_bytes == 24
    assert record.as_event()["stage_ids"] == ["prompt_encoder", "generation"]


def test_record_stage_load_requires_a_stage_id():
    recorder = StageLoadRecorder(_ScriptedProbe([0]), backend="cuda")

    with recording_stage_loads(recorder):
        with pytest.raises(ValueError, match="at least one stage"):
            with record_stage_load():
                pass


def test_record_stage_load_outside_a_worker_startup_is_inert():
    # The CLI loads the same adapters; an unwrapped call must not raise or record.
    with record_stage_load("text"):
        value = 1

    assert value == 1


def test_recorder_records_the_stage_even_when_its_load_raises():
    probe = _ScriptedProbe([0, 0, 10, 10])
    recorder = StageLoadRecorder(probe, backend="cuda")

    with recording_stage_loads(recorder):
        with pytest.raises(RuntimeError):
            with record_stage_load("text"):
                raise RuntimeError("weights missing")

    report = recorder.report()

    assert [stage.stage_id for stage in report.stages] == ["text"]
    assert report.stages[0].device_bytes == 10


def test_neuron_probe_reads_this_process_device_bytes():
    payload = {
        "neuron_runtime_data": [
            {
                "pid": os.getpid(),
                "report": {
                    "memory_used": {
                        "neuron_runtime_used_bytes": {"host": 11, "neuron_device": 73_401_942_984}
                    }
                },
            },
            {
                "pid": os.getpid() + 1,
                "report": {"memory_used": {"neuron_runtime_used_bytes": {"neuron_device": 5}}},
            },
        ]
    }
    probe = NeuronDeviceMemoryProbe(read_sample=lambda: json.dumps(payload))

    sample = probe.sample()

    assert sample == DeviceMemorySample(73_401_942_984, "process")
    assert probe.unavailable_reason is None


def test_neuron_probe_falls_back_to_host_scope_when_no_runtime_is_this_pid():
    payload = {
        "neuron_runtime_data": [
            {
                "pid": os.getpid() + 1,
                "report": {"memory_used": {"neuron_runtime_used_bytes": {"neuron_device": 7}}},
            }
        ]
    }
    probe = NeuronDeviceMemoryProbe(read_sample=lambda: json.dumps(payload))

    assert probe.sample() == DeviceMemorySample(7, "host")


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"neuron_runtime_data": []}, "no Neuron runtimes"),
        ({"system_data": {}}, "no Neuron runtimes"),
        ({"neuron_runtime_data": [{"pid": 1, "report": {}}]}, "neuron_device"),
    ],
)
def test_neuron_probe_reports_unavailable_instead_of_zero(payload, expected):
    probe = NeuronDeviceMemoryProbe(read_sample=lambda: json.dumps(payload))

    assert probe.sample() is None
    assert expected in (probe.unavailable_reason or "")


def test_neuron_probe_survives_a_missing_binary():
    def _explode() -> str:
        raise FileNotFoundError("neuron-monitor")

    probe = NeuronDeviceMemoryProbe(read_sample=_explode)

    assert probe.sample() is None
    assert "neuron-monitor unavailable" in (probe.unavailable_reason or "")


def test_neuron_probe_survives_unparseable_output():
    probe = NeuronDeviceMemoryProbe(read_sample=lambda: "not json")

    assert probe.sample() is None
    assert "unparseable JSON" in (probe.unavailable_reason or "")


def test_probe_selection_is_by_backend():
    assert isinstance(select_device_memory_probe("cuda"), CudaDeviceMemoryProbe)
    assert isinstance(select_device_memory_probe("trainium"), NeuronDeviceMemoryProbe)
    cpu_probe = select_device_memory_probe("cpu")
    assert isinstance(cpu_probe, NullDeviceMemoryProbe)
    assert cpu_probe.sample() is None
    assert "'cpu'" in (cpu_probe.unavailable_reason or "")


def test_cuda_probe_without_a_device_is_unavailable_not_zero():
    probe = CudaDeviceMemoryProbe()
    sample = probe.sample()

    if sample is None:
        assert probe.unavailable_reason
    else:
        # A machine that really has a CUDA device: the reading must still be a count.
        assert sample.scope == "process"
        assert sample.resident_bytes >= 0


def test_request_breakdown_summary_and_event():
    breakdown = RequestStageBreakdown(
        request_id="r1",
        model="fake/model",
        outcome="completed",
        stages=(StageTiming("text", 1.0), StageTiming("generate", 9.0)),
        total_seconds=10.0,
    )

    assert breakdown.stage_seconds == {"text": 1.0, "generate": 9.0}
    assert breakdown.summary() == "text=1.000s(10.0%) generate=9.000s(90.0%)"
    assert breakdown.as_event() == {
        "request_id": "r1",
        "model": "fake/model",
        "outcome": "completed",
        "stages": [
            {"stage_id": "text", "seconds": 1.0},
            {"stage_id": "generate", "seconds": 9.0},
        ],
        "total_seconds": 10.0,
    }


def test_empty_breakdown_summary_is_explicit():
    breakdown = RequestStageBreakdown("r1", "fake/model", "error", (), 0.0)

    assert breakdown.summary() == "none"


def test_format_bytes_says_unknown_rather_than_zero():
    assert format_bytes(None) == "unknown"
    assert format_bytes(2 * 1024**3) == "2147483648 (2.00 GiB)"
