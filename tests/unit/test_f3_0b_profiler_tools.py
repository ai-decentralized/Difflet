import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gate_audit_keeps_partial_evidence_gated(tmp_path):
    audit_mod = _load_script("audit_f3_0b_gate")
    combined = tmp_path / "profile_qwen_image_1024_f3_0b_combined.json"
    combined.write_text(
        """
        {
          "schema": "difflet-f3-denoise-loop-neuron-inspect-summary-v1",
          "steps": [
            {
              "neff_execution_time_s": 0.45,
              "neff_execution_time_source": "neuron_profile:nrt_execute_max_worker_duration"
            },
            {
              "neff_execution_time_s": 0.44,
              "neff_execution_time_source": "neuron_profile:nrt_execute_max_worker_duration"
            },
            {
              "neff_execution_time_s": 0.43,
              "neff_execution_time_source": "neuron_profile:nrt_execute_max_worker_duration"
            },
            {
              "neff_execution_time_s": 0.42,
              "neff_execution_time_source": "neuron_profile:nrt_execute_max_worker_duration"
            }
          ],
          "summary": {
            "num_steps": 4,
            "hardware_timed_steps": 4,
            "mean_neff_execution_time_s": 0.45,
            "steady_state_max_cpu_round_trip_share_after_step0": 0.012,
            "mean_cpu_round_trip_share": 0.008,
            "total_host_device_transfer_bytes": 1024,
            "total_host_device_transfer_count": 2
          }
        }
        """,
        encoding="utf-8",
    )

    result = audit_mod.audit(
        [combined],
        required_labels=["qwen_image", "ltx_2", "flux"],
        threshold=0.10,
    )

    assert result["can_unlock_f3_1"] is False
    assert result["can_write_negative_closeout"] is False
    assert result["decision"] == "remain_gated_partial_evidence"
    assert result["measured_labels"] == ["qwen_image"]
    assert result["missing_labels"] == ["ltx_2", "flux"]


def test_gate_audit_unlocks_on_any_measured_model_over_threshold(tmp_path):
    audit_mod = _load_script("audit_f3_0b_gate")
    combined = tmp_path / "profile_hunyuan_video_n4_f3_0b_combined.json"
    combined.write_text(
        """
        {
          "schema": "difflet-f3-denoise-loop-neuron-inspect-summary-v1",
          "steps": [
            {
              "neff_execution_time_s": 0.60,
              "neff_execution_time_source": "neuron_profile:nrt_execute_max_worker_duration"
            },
            {
              "neff_execution_time_s": 0.59,
              "neff_execution_time_source": "neuron_profile:nrt_execute_max_worker_duration"
            },
            {
              "neff_execution_time_s": 0.58,
              "neff_execution_time_source": "neuron_profile:nrt_execute_max_worker_duration"
            },
            {
              "neff_execution_time_s": 0.57,
              "neff_execution_time_source": "neuron_profile:nrt_execute_max_worker_duration"
            }
          ],
          "summary": {
            "num_steps": 4,
            "hardware_timed_steps": 4,
            "mean_neff_execution_time_s": 0.60,
            "steady_state_max_cpu_round_trip_share_after_step0": 0.125,
            "mean_cpu_round_trip_share": 0.11,
            "total_host_device_transfer_bytes": 2048,
            "total_host_device_transfer_count": 4
          }
        }
        """,
        encoding="utf-8",
    )

    result = audit_mod.audit(
        [combined],
        required_labels=["hunyuan_video", "qwen_image"],
        threshold=0.10,
    )

    assert result["can_unlock_f3_1"] is True
    assert result["can_write_negative_closeout"] is False
    assert result["decision"] == "unlock_f3_1"
    assert result["unlocking_labels"] == ["hunyuan_video"]


def test_gate_audit_ignores_fallback_or_incomplete_json_for_decisions(tmp_path):
    audit_mod = _load_script("audit_f3_0b_gate")
    combined = tmp_path / "profile_qwen_image_f3_0b_combined.json"
    combined.write_text(
        """
        {
          "schema": "difflet-f3-denoise-loop-profile-v2",
          "summary": {
            "num_steps": 4,
            "steady_state_max_cpu_round_trip_share_after_step0": 0.25,
            "mean_cpu_round_trip_share": 0.20,
            "total_host_device_transfer_bytes": null,
            "total_host_device_transfer_count": null
          }
        }
        """,
        encoding="utf-8",
    )

    result = audit_mod.audit(
        [combined],
        required_labels=["qwen_image"],
        threshold=0.10,
    )

    row = result["rows"][0]
    assert row["hardware_measured"] is False
    assert row["passes_threshold"] is False
    assert result["measured_labels"] == []
    assert result["missing_labels"] == ["qwen_image"]
    assert result["can_unlock_f3_1"] is False


def test_gate_audit_rejects_unknown_step_neff_source(tmp_path):
    audit_mod = _load_script("audit_f3_0b_gate")
    combined = tmp_path / "profile_qwen_image_f3_0b_combined.json"
    combined.write_text(
        """
        {
          "schema": "difflet-f3-denoise-loop-neuron-inspect-summary-v1",
          "steps": [
            {
              "neff_execution_time_s": 0.75,
              "neff_execution_time_source": "python_transformer_call_boundary_fallback"
            }
          ],
          "summary": {
            "num_steps": 1,
            "hardware_timed_steps": 1,
            "mean_neff_execution_time_s": 0.75,
            "steady_state_max_cpu_round_trip_share_after_step0": 0.25,
            "mean_cpu_round_trip_share": 0.25,
            "total_host_device_transfer_bytes": 2048,
            "total_host_device_transfer_count": 4
          }
        }
        """,
        encoding="utf-8",
    )

    result = audit_mod.audit(
        [combined],
        required_labels=["qwen_image"],
        threshold=0.10,
    )

    row = result["rows"][0]
    assert row["hardware_measured"] is False
    assert row["passes_threshold"] is False
    assert "step_neff_execution_source_not_hardware_counter" in row["invalid_reasons"]
    assert result["missing_labels"] == ["qwen_image"]


def test_profile_xla_sampler_uses_direct_metric_data_fallback():
    profile = _load_script("profile_denoise_loop")
    before = profile._XlaMetricSnapshot(
        True,
        {
            "TransferToServerTime__TotalSamples": 10,
            "TransferToServerTime__Accumulator_sec": 1.0,
            "InboundData__TotalSamples": 10,
            "InboundData__Accumulator_bytes": 1000,
            "ExecuteReplicatedTime__TotalSamples": 3,
            "ExecuteReplicatedTime__Accumulator_sec": 6.0,
            "aten::_to_copy__Value": 20,
        },
        0.001,
    )
    after = profile._XlaMetricSnapshot(
        True,
        {
            "TransferToServerTime__TotalSamples": 12,
            "TransferToServerTime__Accumulator_sec": 1.25,
            "InboundData__TotalSamples": 12,
            "InboundData__Accumulator_bytes": 1600,
            "ExecuteReplicatedTime__TotalSamples": 4,
            "ExecuteReplicatedTime__Accumulator_sec": 6.5,
            "aten::_to_copy__Value": 23,
        },
        0.002,
    )

    sampler = object.__new__(profile._XlaMetricsSampler)
    delta = sampler.delta(before, after)

    assert delta["to_device_transfer_time_s"] == pytest.approx(0.25)
    assert delta["host_device_transfer_count"] == 2
    assert delta["host_device_transfer_bytes"] == 600
    assert delta["neff_execution_time_s"] == pytest.approx(0.5)
    assert delta["neff_execution_count"] == 1
    assert delta["neff_execution_time_source"] == "xla_metric:ExecuteReplicatedTime"
    assert delta["xla_aten_transfer_counter_delta"] == 3


def test_requirement_coverage_distinguishes_qwen_profiler_from_gate():
    coverage_mod = _load_script("audit_f3_0b_requirement_coverage")
    direct = {
        "schema": "difflet-f3-denoise-loop-profile-v2",
        "model": "qwen-image",
        "height": 1024,
        "width": 1024,
        "steps": [
            {
                "scheduler_time_s": 0.001,
                "scheduler_tensor_bytes": 128,
                "mark_step_time_s": 0.002,
                "xla_metrics_available": True,
                "xla_metric_names_available": ["DeviceLockWait"],
                "xla_counter_names_available": ["MarkStep"],
                "xla_parsed_metric_keys": ["DeviceLockWait__Accumulator_sec"],
                "host_device_transfer_count": None,
                "host_device_transfer_bytes": None,
                "neff_execution_time_source": "python_transformer_call_boundary_fallback",
            }
        ],
        "summary": {
            "num_steps": 1,
            "xla_metrics_steps": 1,
            "hardware_timed_steps": 0,
            "mean_scheduler_time_s": 0.001,
            "mean_mark_step_time_s": 0.002,
            "total_host_device_transfer_count": None,
            "total_host_device_transfer_bytes": None,
        },
    }
    combined = {
        "schema": "difflet-f3-denoise-loop-neuron-inspect-summary-v1",
        "steps": [
            {
                "non_neff_time_s": 0.01,
                "neff_execution_time_s": 0.90,
                "neff_execution_time_source": "neuron_profile:nrt_execute_max_worker_duration",
                "host_device_transfer_count": 2,
                "host_device_transfer_bytes": 1024,
            }
        ],
        "summary": {
            "num_steps": 1,
            "hardware_timed_steps": 1,
            "mean_neff_execution_time_s": 0.90,
            "mean_cpu_round_trip_share": 0.01,
            "steady_state_max_cpu_round_trip_share_after_step0": 0.01,
            "total_host_device_transfer_count": 2,
            "total_host_device_transfer_bytes": 1024,
            "total_host_device_transfer_time_s": 0.0005,
        },
    }
    gate = {
        "decision": "remain_gated_partial_evidence",
        "can_unlock_f3_1": False,
        "can_write_negative_closeout": False,
        "measured_labels": ["qwen_image"],
        "missing_labels": ["ltx_2", "flux"],
    }
    artifacts = {
        "ready_labels": ["qwen_image"],
        "missing_labels": ["ltx_2", "flux"],
        "disk_available_gb": 34.68,
    }

    result = coverage_mod.audit_requirement_coverage(
        direct_profile=direct,
        combined_profile=combined,
        gate_audit=gate,
        artifact_inventory=artifacts,
    )

    assert result["qwen_profiler_requirement_satisfied"] is True
    assert result["production_gate_complete"] is False
    assert result["neuron_profile_fallback_required_for_qwen"] is True
    assert result["overall_status"] == "qwen_profiler_satisfied_gate_partial_evidence"
    by_id = {item["id"]: item for item in result["requirements"]}
    assert by_id["direct_xla_transfer_bytes_count_available"]["satisfied"] is False
    assert by_id["direct_xla_transfer_bytes_count_available"]["status"] == "attempted_unavailable"
    assert by_id["neuron_profile_transfer_bytes_count_measured"]["satisfied"] is True


def test_artifact_inventory_reports_ready_and_missing_labels(tmp_path):
    artifacts_mod = _load_script("audit_f3_0b_artifacts")
    (tmp_path / "ready_source").write_text("source\n", encoding="utf-8")
    (tmp_path / "ready_model.pt").write_text("compiled\n", encoding="utf-8")
    (tmp_path / "ready_bundle.safetensors").write_text("bundle\n", encoding="utf-8")
    result = artifacts_mod.audit_artifacts(
        tmp_path,
        {
            "ready": {
                "source": "ready_source",
                "compiled": "ready_model.pt",
                "bundle": "ready_bundle.safetensors",
            },
            "missing": {
                "source": "missing_source",
                "compiled": "missing_model.pt",
                "bundle": None,
            },
        },
    )

    assert result["ready_labels"] == ["ready"]
    assert result["missing_labels"] == ["missing"]
    assert result["disk_available_bytes"] > 0
    assert result["disk_available_gb"] > 0
    missing_row = next(row for row in result["rows"] if row["label"] == "missing")
    assert missing_row["ready_for_f3_0b"] is False
    assert missing_row["missing"] == ["source", "compiled", "bundle"]


def test_summarize_neuron_inspect_aligns_execution_and_transfer_events():
    summarize = _load_script("summarize_neuron_inspect")
    denoise_steps = [
        {"step_index": 0, "host_step_time_s": 1.0, "device_step_time_s": 0.95},
        {"step_index": 1, "host_step_time_s": 2.0, "device_step_time_s": 1.95},
    ]
    events = [
        {
            "name": "nrt_execute",
            "model_name": "/tmp/TestModel/graph.neff",
            "timestamp": 1_000_000_000,
            "duration": 900_000_000,
            "worker_gid": 0,
        },
        {
            "name": "nrt_execute",
            "model_name": "/tmp/TestModel/graph.neff",
            "timestamp": 1_000_010_000,
            "duration": 910_000_000,
            "worker_gid": 1,
        },
        {
            "name": "nrt_tensor_write",
            "timestamp": 990_000_000,
            "duration": 1_000_000,
            "size": 128,
        },
        {
            "name": "nrt_execute",
            "model_name": "/tmp/TestModel/graph.neff",
            "timestamp": 3_000_000_000,
            "duration": 1_800_000_000,
            "worker_gid": 0,
        },
        {
            "name": "nrt_execute",
            "model_name": "/tmp/TestModel/graph.neff",
            "timestamp": 3_000_010_000,
            "duration": 1_810_000_000,
            "worker_gid": 1,
        },
        {
            "name": "nrt_tensor_read",
            "timestamp": 4_810_000_000,
            "duration": 2_000_000,
            "size": 256,
        },
    ]

    clusters = summarize._cluster_execute_events(
        events,
        model_name_contains="TestModel",
        gap_ns=10_000_000,
    )
    windows = summarize._window_bounds(clusters)
    combined = [
        summarize._summarize_step(step, cluster, events, window=window)
        for step, cluster, window in zip(denoise_steps, clusters, windows)
    ]
    summary = summarize._summarize_combined(combined)

    assert len(combined) == 2
    assert combined[0]["neff_execution_time_s"] == 0.91
    assert combined[0]["neff_execution_time_source"].startswith("neuron_profile:")
    assert combined[0]["hardware_execution_time_s"] == 0.91
    assert combined[0]["host_device_transfer_bytes"] == 128
    assert combined[1]["hardware_execution_time_s"] == 1.81
    assert combined[1]["host_device_transfer_bytes"] == 256
    assert summary["total_host_device_transfer_bytes"] == 384
    assert summary["steady_state_max_cpu_round_trip_share_after_step0"] == pytest.approx(0.095)
