#!/usr/bin/env python3
"""Adjudicate the H1d request-context staging architecture gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _tag(event: dict[str, Any]) -> str:
    return Path(str(event["model_name"])).parent.parent.name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-json", required=True, type=Path)
    parser.add_argument("--mechanism-result", required=True, type=Path)
    parser.add_argument("--profile-result", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--failed-single-slot-result", required=True, type=Path)
    parser.add_argument("--h1c-boundary-result", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = {
        name: path.expanduser().resolve()
        for name, path in {
            "inspect_json": args.inspect_json,
            "mechanism_result": args.mechanism_result,
            "profile_result": args.profile_result,
            "protocol": args.protocol,
            "failed_single_slot_result": args.failed_single_slot_result,
            "h1c_boundary_result": args.h1c_boundary_result,
        }.items()
    }
    inspect = _load(paths["inspect_json"])
    mechanism = _load(paths["mechanism_result"])
    profile = _load(paths["profile_result"])
    failed_single_slot = _load(paths["failed_single_slot_result"])
    h1c_boundary = _load(paths["h1c_boundary_result"])
    events = inspect.get("trace_event")
    memory_rows = inspect.get("device_mem_usage")
    if not isinstance(events, list) or not isinstance(memory_rows, list):
        raise ValueError("inspect JSON lacks trace_event or device_mem_usage")

    expected_tags = [
        "transformer",  # backbone load-time weight layout, not request workload
        "request_context_stage_slot_0",
        "cache_initialize",
        "NeuronFluxTransformer2DModel",
        "cache_anchor_step",
        "NeuronFluxTransformer2DModel",
        "cache_anchor_step",
        "cache_predict",
        "cache_scheduler_step",
        "NeuronFluxTransformer2DModel",
        "cache_anchor_step",
        "cache_finalize",
    ]
    execute_events = [event for event in events if event.get("name") == "nrt_execute"]
    workers = sorted({int(event["worker_gid"]) for event in execute_events})
    if workers != [0, 1, 2, 3]:
        raise ValueError(f"expected TP4 workers, got {workers}")

    calls: list[dict[str, Any]] = []
    for worker in workers:
        worker_executes = sorted(
            (event for event in execute_events if int(event["worker_gid"]) == worker),
            key=lambda event: int(event["timestamp"]),
        )
        if [_tag(event) for event in worker_executes] != expected_tags:
            raise ValueError(f"worker {worker} execution sequence mismatch")
        # The capture assigns rank workload DMAs to thread worker+1.  Runtime
        # initialization uses thread 0 and can have the same tensor sizes; it
        # must not be charged to request staging.
        workload_thread = worker + 1
        worker_events = [
            event
            for event in events
            if int(event.get("worker_gid", -1)) == worker
            and int(event.get("thread_id", -1)) == workload_thread
        ]
        for ordinal, execute in enumerate(worker_executes):
            start = int(execute["timestamp"])
            low = (
                (int(worker_executes[ordinal - 1]["timestamp"]) + start) // 2
                if ordinal
                else start - 100_000_000
            )
            high = (
                (start + int(worker_executes[ordinal + 1]["timestamp"])) // 2
                if ordinal + 1 < len(worker_executes)
                else start + 100_000_000
            )
            selected = [
                event
                for event in worker_events
                if low <= int(event.get("timestamp", 0)) < high
            ]
            calls.append(
                {
                    "worker": worker,
                    "workload_thread": workload_thread,
                    "ordinal": ordinal,
                    "tag": _tag(execute),
                    "execute_time_us": int(execute.get("duration") or 0) / 1000.0,
                    "write_sizes": sorted(
                        int(event.get("size") or 0)
                        for event in selected
                        if event.get("name") == "nrt_tensor_write"
                    ),
                    "read_sizes": sorted(
                        int(event.get("size") or 0)
                        for event in selected
                        if event.get("name") == "nrt_tensor_read"
                    ),
                }
            )

    expected_write_sizes_by_ordinal = {
        0: [],
        1: [2, 4, 1536, 2359296, 4194304],
        2: [16, 524288],
        3: [2],
        4: [8],
        5: [2],
        6: [8],
        7: [8],
        8: [4],
        9: [2],
        10: [8],
        11: [12],
    }
    exact_transfer_sequence = all(
        call["write_sizes"] == expected_write_sizes_by_ordinal[int(call["ordinal"])]
        and call["read_sizes"] == []
        for call in calls
    )
    stage_calls = [call for call in calls if int(call["ordinal"]) == 1]
    post_stage_calls = [call for call in calls if 2 <= int(call["ordinal"]) <= 11]
    data_plane_calls = [call for call in calls if 3 <= int(call["ordinal"]) <= 10]
    stage_writes = Counter(
        size for call in stage_calls for size in call["write_sizes"]
    )
    post_stage_writes = Counter(
        size for call in post_stage_calls for size in call["write_sizes"]
    )
    data_plane_writes = Counter(
        size for call in data_plane_calls for size in call["write_sizes"]
    )
    post_stage_reads = Counter(
        size for call in post_stage_calls for size in call["read_sizes"]
    )

    context_sizes = mechanism["request_context_bytes_per_rank"]
    invariant_sizes = [
        int(context_sizes["encoder_hidden_states"]),
        int(context_sizes["pooled_projections"]),
        int(context_sizes["rotary_embedding"]),
    ]
    invariant_staged_once = (
        all(stage_writes[size] == 4 for size in invariant_sizes)
        and stage_writes[2] == 4  # guidance
        and all(post_stage_writes[size] == 0 for size in invariant_sizes)
    )
    timestep_only_at_backbone = all(
        call["write_sizes"] == [2]
        for call in calls
        if int(call["ordinal"]) in (3, 5, 9)
    )
    tensor_bytes = int(mechanism["latent_noise_tensor_bytes"])
    latent_noise_boundary = (
        data_plane_writes[tensor_bytes] == 0
        and sum(call["read_sizes"].count(tensor_bytes) for call in data_plane_calls)
        == 0
    )

    checks_by_name = {row["name"]: row for row in mechanism["checks"]}
    correctness_passed = (
        mechanism["status"] == "mechanism_passed"
        and float(mechanism["maximum_absolute_error"]) == 0.0
        and float(mechanism["maximum_relative_l2_error"]) == 0.0
    )
    isolation_names = (
        "backbone_direct_vs_staged_A_before_interleave",
        "backbone_direct_vs_staged_B",
        "backbone_A_survives_B_and_cache_execution",
    )
    isolation_passed = all(
        float(checks_by_name[name]["maximum_absolute_error"]) == 0.0
        and float(checks_by_name[name]["relative_l2_error"]) == 0.0
        for name in isolation_names
    )
    failed_checks = {
        row["name"]: row
        for row in failed_single_slot["checks"]
        if float(row["maximum_absolute_error"]) != 0.0
        or float(row["relative_l2_error"]) != 0.0
    }
    single_slot_failure_reproduced = (
        failed_single_slot["status"] == "mechanism_failed"
        and list(failed_checks) == ["backbone_A_survives_B_and_cache_execution"]
    )

    request_context_bytes = int(mechanism["request_context_total_bytes_tp4"])
    slot_token_bytes = 4 * 4
    one_time_stage_bytes = request_context_bytes + slot_token_bytes
    per_backbone_timestep_bytes = 2 * 4
    previous_per_backbone_bytes = request_context_bytes + per_backbone_timestep_bytes

    timing = {}
    for tag in (
        "request_context_stage_slot_0",
        "NeuronFluxTransformer2DModel",
        "cache_initialize",
        "cache_anchor_step",
        "cache_predict",
        "cache_scheduler_step",
        "cache_finalize",
    ):
        values = [
            float(call["execute_time_us"])
            for call in calls
            if call["tag"] == tag
        ]
        timing[tag] = {
            "rank_execution_count": len(values),
            "median_per_rank_execute_time_us": median(values),
            "maximum_per_rank_execute_time_us": max(values),
        }

    memory_fields = ("total_bytes", "tensors_bytes", "weights_bytes", "io_bytes")
    per_core_peaks: dict[str, dict[str, int]] = {}
    for row in memory_rows:
        core = str(row["hbm_idx"])
        peaks = per_core_peaks.setdefault(core, {key: 0 for key in memory_fields})
        for key in memory_fields:
            peaks[key] = max(peaks[key], int(row.get(key) or 0))
    h1c_per_core_peaks = h1c_boundary["device_memory_peak"]["per_core_bytes"]
    memory_delta_per_core = {
        core: {
            key: per_core_peaks[core][key] - int(h1c_per_core_peaks[core][key])
            for key in memory_fields
        }
        for core in sorted(per_core_peaks)
    }

    boundary_passed = (
        exact_transfer_sequence
        and invariant_staged_once
        and timestep_only_at_backbone
        and latent_noise_boundary
        and not post_stage_reads
    )
    passed = correctness_passed and isolation_passed and boundary_passed
    payload = {
        "schema": "difflet-flux-h1d-request-context-staging-architecture-result",
        "schema_revision": 1,
        "study_id": "flux-h1d-request-context-staging-20260812",
        "status": (
            "request_context_boundary_passed_full_loop_required"
            if passed
            else "request_context_boundary_failed"
        ),
        "serving_claim": False,
        "architecture_speed_claim": False,
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "D1_numerical_contract": {
            "check_count": len(mechanism["checks"]),
            "maximum_absolute_error": mechanism["maximum_absolute_error"],
            "maximum_relative_l2_error": mechanism["maximum_relative_l2_error"],
            "passed": correctness_passed,
        },
        "D2_D3_lifetime_and_isolation": {
            "single_slot_failure_reproduced": single_slot_failure_reproduced,
            "single_slot_failed_checks": failed_checks,
            "two_slot_interleave_checks": {
                name: checks_by_name[name] for name in isolation_names
            },
            "proof_slot_capacity": mechanism["request_slot_pool"]["proof_capacity"],
            "allocation_contract": mechanism["request_slot_pool"][
                "allocation_policy"
            ],
            "passed": isolation_passed and single_slot_failure_reproduced,
        },
        "D4_boundary_accounting": {
            "exact_transfer_sequence_passed": exact_transfer_sequence,
            "workload_thread_mapping": {
                str(worker): worker + 1 for worker in workers
            },
            "request_context_staged_once": invariant_staged_once,
            "backbone_receives_only_timestep_scalar": timestep_only_at_backbone,
            "latent_noise_host_write_bytes_between_initialize_finalize": data_plane_writes[
                tensor_bytes
            ]
            * tensor_bytes,
            "latent_noise_host_read_bytes_between_initialize_finalize": sum(
                call["read_sizes"].count(tensor_bytes) for call in data_plane_calls
            )
            * tensor_bytes,
            "post_stage_host_read_bytes": sum(
                size * count for size, count in post_stage_reads.items()
            ),
            "stage_write_size_counts": {
                str(size): count for size, count in sorted(stage_writes.items())
            },
            "post_stage_write_size_counts": {
                str(size): count for size, count in sorted(post_stage_writes.items())
            },
            "calls": calls,
            "passed": boundary_passed,
        },
        "D5_amortization": {
            "request_context_tensor_bytes_tp4": request_context_bytes,
            "slot_token_bytes_tp4": slot_token_bytes,
            "one_time_stage_host_write_bytes": one_time_stage_bytes,
            "per_backbone_timestep_host_write_bytes": per_backbone_timestep_bytes,
            "h1c_previous_per_backbone_host_write_bytes": previous_per_backbone_bytes,
            "projected_a12_context_and_timestep_bytes_before": previous_per_backbone_bytes
            * 12,
            "projected_a12_context_and_timestep_bytes_after": one_time_stage_bytes
            + per_backbone_timestep_bytes * 12,
            "projected_full50_context_and_timestep_bytes_before": previous_per_backbone_bytes
            * 50,
            "projected_full50_context_and_timestep_bytes_after": one_time_stage_bytes
            + per_backbone_timestep_bytes * 50,
            "claim_boundary": "Measured byte/count inventory and projections only; no end-to-end speedup claim.",
        },
        "D6_runtime_and_memory": {
            "compile_seconds": mechanism["compile_seconds"],
            "load_seconds": mechanism["load_seconds"],
            "request_context_artifact": mechanism["artifacts"]["request_context"],
            "execution_diagnostics": timing,
            "device_memory_peak_per_core_bytes": per_core_peaks,
            "delta_vs_h1c_per_core_bytes": memory_delta_per_core,
            "memory_comparison_note": "Same backbone/cache artifacts and profiler method; delta includes both retained slot outputs and runtime graph/input buffers, so it is not interpreted as payload size alone.",
        },
        "decision": {
            "resident_request_state_classes": {
                "mutable_request_state": "anchors and latent use aliased device state",
                "immutable_request_state": "conditioning uses request-owned ranked slot outputs",
                "step_state": "timestep, predictor coefficients, and delta sigma remain scalar host control",
            },
            "new_runtime_requirement": "A request owns a device context slot until release; re-executing one staging entry point overwrites its output buffers.",
            "next": "Run a complete 50-step real-prompt A12 loop and a materializing-vs-resident end-to-end A/B before any speed or image-quality claim.",
        },
    }
    _write(args.out.expanduser().resolve(), payload)
    print(
        f"{payload['status']}: one-time={one_time_stage_bytes}B "
        f"per-backbone={per_backbone_timestep_bytes}B; slots=2"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
