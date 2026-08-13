#!/usr/bin/env python3
"""Adjudicate H1b shared-state and ranked-I/O gates from Neuron inspect."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def _model_tag(event: dict[str, Any]) -> str:
    return Path(str(event["model_name"])).parent.parent.name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-json", required=True, type=Path)
    parser.add_argument("--mechanism-result", required=True, type=Path)
    parser.add_argument("--compile-result", required=True, type=Path)
    parser.add_argument("--profile-result", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = {
        key: value.expanduser().resolve()
        for key, value in {
            "inspect_json": args.inspect_json,
            "mechanism_result": args.mechanism_result,
            "compile_result": args.compile_result,
            "profile_result": args.profile_result,
            "protocol": args.protocol,
        }.items()
    }
    inspect = _load(paths["inspect_json"])
    mechanism = _load(paths["mechanism_result"])
    compile_result = _load(paths["compile_result"])
    profile = _load(paths["profile_result"])
    events = inspect.get("trace_event")
    memory_rows = inspect.get("device_mem_usage")
    if not isinstance(events, list) or not isinstance(memory_rows, list):
        raise ValueError("inspect JSON lacks trace_event or device_mem_usage")

    tensor_bytes = int(mechanism["tensor_bytes"])
    pair_count = int(profile["details"]["warmup_skip_pairs"]) + int(
        profile["details"]["measured_skip_pairs"]
    )
    expected_tags = [
        "resident_reset",
        "resident_anchor_update",
        "resident_anchor_update",
        *[tag for _ in range(pair_count) for tag in ("resident_predict", "resident_consume")],
    ]
    execute_events = [event for event in events if event.get("name") == "nrt_execute"]
    workers = sorted({int(event["worker_gid"]) for event in execute_events})
    if workers != [0, 1, 2, 3]:
        raise ValueError(f"expected TP4 workers, got {workers}")

    calls = []
    for worker in workers:
        worker_executes = sorted(
            (
                event
                for event in execute_events
                if int(event["worker_gid"]) == worker
            ),
            key=lambda event: int(event["timestamp"]),
        )
        tags = [_model_tag(event) for event in worker_executes]
        if tags != expected_tags:
            raise ValueError(f"worker {worker} invocation sequence mismatch: {tags}")
        worker_events = [
            event for event in events if int(event.get("worker_gid", -1)) == worker
        ]
        for index, execute in enumerate(worker_executes):
            start = int(execute["timestamp"])
            low = (
                (int(worker_executes[index - 1]["timestamp"]) + start) // 2
                if index
                else start - 20_000_000
            )
            high = (
                (start + int(worker_executes[index + 1]["timestamp"])) // 2
                if index + 1 < len(worker_executes)
                else start + 20_000_000
            )
            selected = [
                event
                for event in worker_events
                if low <= int(event.get("timestamp", 0)) < high
            ]
            writes = [event for event in selected if event.get("name") == "nrt_tensor_write"]
            reads = [event for event in selected if event.get("name") == "nrt_tensor_read"]
            calls.append(
                {
                    "worker": worker,
                    "ordinal": index,
                    "tag": _model_tag(execute),
                    "execute_time_us": int(execute.get("duration") or 0) / 1000.0,
                    "write_sizes": sorted(int(event.get("size") or 0) for event in writes),
                    "read_sizes": sorted(int(event.get("size") or 0) for event in reads),
                }
            )

    skip_calls = [call for call in calls if int(call["ordinal"]) >= 3]
    skip_writes = [size for call in skip_calls for size in call["write_sizes"]]
    skip_reads = [size for call in skip_calls for size in call["read_sizes"]]
    large_write_bytes = sum(size for size in skip_writes if size >= tensor_bytes)
    large_read_bytes = sum(size for size in skip_reads if size >= tensor_bytes)
    scalar_write_bytes = sum(size for size in skip_writes if size < tensor_bytes)
    scalar_read_bytes = sum(size for size in skip_reads if size < tensor_bytes)
    boundary_passed = large_write_bytes == 0 and large_read_bytes == 0
    mechanism_passed = bool(mechanism["details"]["mechanism_gate"]["passed"])

    seed_calls = [call for call in calls if int(call["ordinal"]) in (1, 2)]
    seed_large_write_bytes = sum(
        size
        for call in seed_calls
        for size in call["write_sizes"]
        if size >= tensor_bytes
    )
    per_core_peaks: dict[str, dict[str, int]] = {}
    memory_fields = ("total_bytes", "tensors_bytes", "weights_bytes", "io_bytes")
    for row in memory_rows:
        core = str(row["hbm_idx"])
        peaks = per_core_peaks.setdefault(core, {key: 0 for key in memory_fields})
        for key in memory_fields:
            peaks[key] = max(peaks[key], int(row.get(key) or 0))

    timing_by_tag = {}
    for tag in ("resident_predict", "resident_consume"):
        values = [float(call["execute_time_us"]) for call in skip_calls if call["tag"] == tag]
        timing_by_tag[tag] = {
            "event_count": len(values),
            "median_per_rank_execute_time_us": median(values),
            "maximum_per_rank_execute_time_us": max(values),
        }

    artifacts = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    payload = {
        "schema": "difflet-flux-h1b-cross-graph-ranked-io-result",
        "schema_revision": 1,
        "study_id": "flux-h1b-cross-graph-state-ranked-io-20260812",
        "status": (
            "feasibility_passed_integration_pending"
            if mechanism_passed and boundary_passed
            else "state_passed_boundary_failed"
            if mechanism_passed
            else "state_failed"
        ),
        "serving_claim": False,
        "architecture_speed_claim": False,
        "artifacts": artifacts,
        "tp_degree": 4,
        "tensor_bytes": tensor_bytes,
        "compiled_artifact": compile_result["compiled_artifact"],
        "compile_seconds": compile_result["compile_seconds"],
        "mechanism_gate": {
            **mechanism["details"]["mechanism_gate"],
            "checksum_absolute_error_diagnostic": mechanism["details"][
                "checksum_absolute_error"
            ],
            "checks": mechanism["details"]["checks"],
        },
        "boundary_gate": {
            "profiled_skip_pairs": pair_count,
            "profiled_skip_entry_point_invocations": pair_count * 2,
            "profiled_skip_rank_executions": len(skip_calls),
            "large_host_write_bytes": large_write_bytes,
            "large_host_read_bytes": large_read_bytes,
            "scalar_host_write_bytes": scalar_write_bytes,
            "scalar_host_read_bytes": scalar_read_bytes,
            "scalar_host_write_bytes_per_pair": scalar_write_bytes / pair_count,
            "scalar_host_read_bytes_per_pair": scalar_read_bytes / pair_count,
            "expected_control_plane_per_pair": "TP4 replication of 8-byte coefficients + 4-byte step scale = 48 write bytes; one rank-0 FP32 checksum = 4 read bytes",
            "passed": boundary_passed,
        },
        "anchor_update_diagnostic": {
            "profiled_anchor_updates": 2,
            "large_host_write_bytes": seed_large_write_bytes,
            "expected_bytes": 2 * 4 * tensor_bytes,
            "note": "Full anchor ingress remains on anchor steps in this spike; the zero-copy FLUX backbone-to-state connection is the next integration gate.",
        },
        "host_observed_skip_pair_latency": {
            "mechanism_run_p50_ms": mechanism["details"]["latency"][
                "host_observed_pair_p50_ms"
            ],
            "mechanism_run_p95_ms": mechanism["details"]["latency"][
                "host_observed_pair_p95_ms"
            ],
            "profile_run_p50_ms": profile["details"]["latency"][
                "host_observed_pair_p50_ms"
            ],
            "profile_run_p95_ms": profile["details"]["latency"][
                "host_observed_pair_p95_ms"
            ],
        },
        "neff_execution_diagnostics": timing_by_tag,
        "device_memory_peak": {
            "per_core_bytes": per_core_peaks,
            "sum_of_per_core_peak_total_bytes": sum(
                row["total_bytes"] for row in per_core_peaks.values()
            ),
            "sum_of_per_core_peak_tensor_bytes": sum(
                row["tensors_bytes"] for row in per_core_peaks.values()
            ),
        },
        "decision": {
            "shared_mutable_state_across_entry_points": mechanism_passed,
            "ranked_inter_neff_tensor_handoff_without_host_materialization": boundary_passed,
            "selected_graph_boundary": "one multi-entry NxDModel with shared alias state and ranked device I/O",
            "what_is_not_yet_proven": [
                "real scheduler and latent update composition",
                "zero-copy FLUX backbone anchor ingress",
                "TP-sharded rather than post-gather replicated anchor placement",
                "end-to-end speedup or serving isolation",
            ],
            "next": "Implement the real resident cache-step entry point, then feed it ranked FLUX backbone output on anchor steps and profile the complete denoise-loop boundary.",
        },
    }
    _write(args.out.expanduser().resolve(), payload)
    print(
        f"{payload['status']}: pairs={pair_count} "
        f"large_write={large_write_bytes} large_read={large_read_bytes} "
        f"scalar={scalar_write_bytes / pair_count:.0f}B write + "
        f"{scalar_read_bytes / pair_count:.0f}B read/pair"
    )
    return 0 if mechanism_passed and boundary_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
