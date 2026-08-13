#!/usr/bin/env python3
"""Summarize Neuron runtime transfers for the FLUX H1a boundary spike."""

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-json", required=True, type=Path)
    parser.add_argument("--mechanism-result", required=True, type=Path)
    parser.add_argument("--profile-result", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    inspect_path = args.inspect_json.expanduser().resolve()
    mechanism_path = args.mechanism_result.expanduser().resolve()
    profile_path = args.profile_result.expanduser().resolve()
    inspect = _load(inspect_path)
    mechanism = _load(mechanism_path)
    profile = _load(profile_path)
    events = inspect.get("trace_event")
    if not isinstance(events, list):
        raise ValueError("inspect JSON is missing trace_event")
    memory_rows = inspect.get("device_mem_usage")
    if not isinstance(memory_rows, list) or not memory_rows:
        raise ValueError("inspect JSON is missing device_mem_usage")
    tensor_bytes = int(mechanism["tensor_bytes"])
    execute_events = [event for event in events if event.get("name") == "nrt_execute"]
    workers = sorted({int(event["worker_gid"]) for event in execute_events})
    if workers != [0, 1, 2, 3]:
        raise ValueError(f"expected TP4 workers, got {workers}")
    starts = sorted(
        int(event["timestamp"])
        for event in execute_events
        if int(event["worker_gid"]) == 0
    )
    expected_calls = (
        len(profile["sequence"])
        + int(profile["latency"]["warmup_calls"])
        + int(profile["latency"]["measured_calls"])
    )
    if len(starts) != expected_calls:
        raise ValueError(
            f"expected {expected_calls} invocation clusters, found {len(starts)}"
        )

    calls = []
    for index, start in enumerate(starts):
        low = (starts[index - 1] + start) // 2 if index else start - 20_000_000
        high = (
            (start + starts[index + 1]) // 2
            if index + 1 < len(starts)
            else start + 20_000_000
        )
        selected = [
            event for event in events if low <= int(event["timestamp"]) < high
        ]
        writes = [event for event in selected if event.get("name") == "nrt_tensor_write"]
        reads = [event for event in selected if event.get("name") == "nrt_tensor_read"]
        executes = [event for event in selected if event.get("name") == "nrt_execute"]
        write_sizes = Counter(int(event.get("size") or 0) for event in writes)
        read_sizes = Counter(int(event.get("size") or 0) for event in reads)
        calls.append(
            {
                "call_index": index,
                "worker_execute_count": len(executes),
                "large_tensor_write_count": write_sizes[tensor_bytes],
                "large_tensor_write_bytes": write_sizes[tensor_bytes] * tensor_bytes,
                "large_tensor_read_count": read_sizes[tensor_bytes],
                "large_tensor_read_bytes": read_sizes[tensor_bytes] * tensor_bytes,
                "scalar_write_bytes": sum(
                    int(event.get("size") or 0)
                    for event in writes
                    if int(event.get("size") or 0) != tensor_bytes
                ),
                "scalar_read_bytes": sum(
                    int(event.get("size") or 0)
                    for event in reads
                    if int(event.get("size") or 0) != tensor_bytes
                ),
                "aggregate_write_event_time_us": sum(
                    int(event.get("duration") or 0) for event in writes
                )
                / 1000.0,
                "aggregate_read_event_time_us": sum(
                    int(event.get("duration") or 0) for event in reads
                )
                / 1000.0,
                "maximum_rank_execute_time_us": max(
                    int(event.get("duration") or 0) for event in executes
                )
                / 1000.0,
            }
        )

    invariant_keys = (
        "worker_execute_count",
        "large_tensor_write_count",
        "large_tensor_write_bytes",
        "large_tensor_read_count",
        "large_tensor_read_bytes",
    )
    invariant = {
        key: sorted({int(call[key]) for call in calls}) for key in invariant_keys
    }
    stable_signature = all(len(values) == 1 for values in invariant.values())
    big_write = invariant["large_tensor_write_count"] == [0]
    big_read = invariant["large_tensor_read_count"] == [0]
    boundary_passed = stable_signature and big_write and big_read
    per_call_large_bytes = int(
        median(
            call["large_tensor_write_bytes"] + call["large_tensor_read_bytes"]
            for call in calls
        )
    )
    mechanism_passed = bool(mechanism["mechanism_gate"]["passed"])
    memory_fields = ("total_bytes", "tensors_bytes", "weights_bytes", "io_bytes")
    per_core_memory_peaks: dict[str, dict[str, int]] = {}
    for row in memory_rows:
        core = str(row["hbm_idx"])
        peaks = per_core_memory_peaks.setdefault(core, {field: 0 for field in memory_fields})
        for field in memory_fields:
            peaks[field] = max(peaks[field], int(row.get(field) or 0))
    if sorted(per_core_memory_peaks) != ["0", "1", "2", "3"]:
        raise ValueError("device-memory trace does not cover four HBM/core indices")
    payload = {
        "schema": "difflet-flux-h1a-resident-state-boundary-result",
        "schema_revision": 1,
        "study_id": "flux-h1a-post-gather-resident-state-boundary-20260812",
        "status": (
            "mechanism_passed_boundary_passed"
            if mechanism_passed and boundary_passed
            else "mechanism_passed_boundary_failed"
            if mechanism_passed
            else "mechanism_failed"
        ),
        "serving_claim": False,
        "architecture_speed_claim": False,
        "artifacts": {
            "inspect_json": {
                "path": str(inspect_path),
                "sha256": _sha256(inspect_path),
            },
            "mechanism_result": {
                "path": str(mechanism_path),
                "sha256": _sha256(mechanism_path),
            },
            "profile_result": {
                "path": str(profile_path),
                "sha256": _sha256(profile_path),
            },
        },
        "invocation_count": len(calls),
        "tp_degree": 4,
        "tensor_bytes": tensor_bytes,
        "static_signature_invariant": stable_signature,
        "per_invocation_invariant_values": invariant,
        "per_invocation_large_tensor_transfer_bytes": per_call_large_bytes,
        "per_invocation_large_tensor_transfer_mib": per_call_large_bytes / 1048576,
        "device_memory_peak": {
            "per_core_bytes": per_core_memory_peaks,
            "sum_of_per_core_peak_total_bytes": sum(
                row["total_bytes"] for row in per_core_memory_peaks.values()
            ),
            "sum_of_per_core_peak_tensor_bytes": sum(
                row["tensors_bytes"] for row in per_core_memory_peaks.values()
            ),
            "note": "These are per-core category maxima from the inspect trace; the tensor peak includes aliased state and transient graph IO, not just the two anchors.",
        },
        "timing_diagnostics": {
            "median_aggregate_write_event_time_us": median(
                call["aggregate_write_event_time_us"] for call in calls
            ),
            "median_aggregate_read_event_time_us": median(
                call["aggregate_read_event_time_us"] for call in calls
            ),
            "median_maximum_rank_execute_time_us": median(
                call["maximum_rank_execute_time_us"] for call in calls
            ),
            "note": "Write/read event times are sums across workers and may overlap; bytes and counts are the primary boundary evidence.",
        },
        "mechanism_gate": mechanism["mechanism_gate"],
        "boundary_gate": {
            "requires_zero_large_tensor_host_writes_on_skip": True,
            "requires_zero_large_tensor_host_reads_on_skip": True,
            "large_tensor_host_writes_absent": big_write,
            "large_tensor_host_reads_absent": big_read,
            "passed": boundary_passed,
        },
        "decision": {
            "h1a_alias_mechanism_established": mechanism_passed,
            "h1a_is_accelerator_resident_data_plane": boundary_passed,
            "post_gather_standalone_predictor_is_better_than_host": False,
            "next": "H1b must compose predictor, scheduler update, and latent state so a skipped step has scalar-only host control; the backbone-to-state inter-NEFF handoff remains a separate gate.",
        },
        "calls": calls,
    }
    _write(args.out.expanduser().resolve(), payload)
    print(
        f"{payload['status']}: calls={len(calls)} "
        f"large_transfer={per_call_large_bytes / 1048576:.3f}MiB/call "
        f"boundary_passed={boundary_passed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
