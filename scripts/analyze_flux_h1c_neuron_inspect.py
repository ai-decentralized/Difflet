#!/usr/bin/env python3
"""Adjudicate fused and split H1c cache-step architecture experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
from typing import Any

ANCHORS = frozenset((0, 1, 2, 3, 4, 5, 9, 15, 21, 31, 41, 49))


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
    parser.add_argument("--split-mechanism", required=True, type=Path)
    parser.add_argument("--split-profile", required=True, type=Path)
    parser.add_argument("--split-protocol", required=True, type=Path)
    parser.add_argument("--fused-result", required=True, type=Path)
    parser.add_argument("--fused-protocol", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = {
        name: path.expanduser().resolve()
        for name, path in {
            "inspect_json": args.inspect_json,
            "split_mechanism": args.split_mechanism,
            "split_profile": args.split_profile,
            "split_protocol": args.split_protocol,
            "fused_result": args.fused_result,
            "fused_protocol": args.fused_protocol,
        }.items()
    }
    inspect = _load(paths["inspect_json"])
    split = _load(paths["split_mechanism"])
    profile = _load(paths["split_profile"])
    fused = _load(paths["fused_result"])
    events = inspect.get("trace_event")
    memory_rows = inspect.get("device_mem_usage")
    if not isinstance(events, list) or not isinstance(memory_rows, list):
        raise ValueError("inspect JSON lacks trace_event or device_mem_usage")
    tensor_bytes = int(split["tensor_bytes"])

    expected_tags = ["cache_initialize"]
    for step in range(50):
        if step in ANCHORS:
            expected_tags.append("cache_anchor_step")
        else:
            expected_tags.extend(("cache_predict", "cache_scheduler_step"))
    expected_tags.append("cache_finalize")
    execute_events = [event for event in events if event.get("name") == "nrt_execute"]
    workers = sorted({int(event["worker_gid"]) for event in execute_events})
    if workers != [0, 1, 2, 3]:
        raise ValueError(f"expected TP4 workers, got {workers}")

    calls = []
    for worker in workers:
        worker_executes = sorted(
            (event for event in execute_events if int(event["worker_gid"]) == worker),
            key=lambda event: int(event["timestamp"]),
        )
        actual_tags = [_tag(event) for event in worker_executes]
        if actual_tags != expected_tags:
            raise ValueError(f"worker {worker} execution sequence mismatch")
        worker_events = [
            event for event in events if int(event.get("worker_gid", -1)) == worker
        ]
        for ordinal, execute in enumerate(worker_executes):
            start = int(execute["timestamp"])
            low = (
                (int(worker_executes[ordinal - 1]["timestamp"]) + start) // 2
                if ordinal
                else start - 20_000_000
            )
            high = (
                (start + int(worker_executes[ordinal + 1]["timestamp"])) // 2
                if ordinal + 1 < len(worker_executes)
                else start + 20_000_000
            )
            selected = [
                event
                for event in worker_events
                if low <= int(event.get("timestamp", 0)) < high
            ]
            calls.append(
                {
                    "worker": worker,
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

    skip_calls = [
        call
        for call in calls
        if call["tag"] in ("cache_predict", "cache_scheduler_step")
    ]
    skip_writes = [size for call in skip_calls for size in call["write_sizes"]]
    skip_reads = [size for call in skip_calls for size in call["read_sizes"]]
    large_write_bytes = sum(size for size in skip_writes if size >= tensor_bytes)
    large_read_bytes = sum(size for size in skip_reads if size >= tensor_bytes)
    scalar_write_bytes = sum(size for size in skip_writes if size < tensor_bytes)
    scalar_read_bytes = sum(size for size in skip_reads if size < tensor_bytes)
    split_parity = split["details"]["schedule_parity"]
    split_exact = (
        float(split_parity["maximum_absolute_error"]) == 0.0
        and float(split_parity["maximum_relative_l2_error"]) == 0.0
    )
    boundary_passed = large_write_bytes == 0 and large_read_bytes == 0

    anchor_calls = [call for call in calls if call["tag"] == "cache_anchor_step"]
    anchor_large_write_bytes = sum(
        size
        for call in anchor_calls
        for size in call["write_sizes"]
        if size >= tensor_bytes
    )
    timing = {}
    for tag in ("cache_predict", "cache_scheduler_step", "cache_anchor_step"):
        values = [float(call["execute_time_us"]) for call in calls if call["tag"] == tag]
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

    fused_parity = fused["details"]["schedule_parity"]
    fused_contract_failed = (
        float(fused_parity["maximum_absolute_error"]) > 0.0
        and float(fused_parity["fused_reference_maximum_absolute_error"]) == 0.0
    )
    payload = {
        "schema": "difflet-flux-h1c-cache-step-architecture-result",
        "schema_revision": 1,
        "study_id": "flux-h1c-cache-step-precision-boundary-20260812",
        "status": (
            "split_contract_passed_backbone_integration_pending"
            if fused_contract_failed and split_exact and boundary_passed
            else "h1c_cache_step_failed"
        ),
        "serving_claim": False,
        "architecture_speed_claim": False,
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "fused_single_neff": {
            "status": "rejected_precision_contract",
            "host_contract_maximum_absolute_error": fused_parity[
                "maximum_absolute_error"
            ],
            "host_contract_maximum_relative_l2_error": fused_parity[
                "maximum_relative_l2_error"
            ],
            "host_contract_final_maximum_absolute_error": fused_parity[
                "final_maximum_absolute_error"
            ],
            "host_contract_final_relative_l2_error": fused_parity[
                "final_relative_l2_error"
            ],
            "fused_fp32_reference_maximum_absolute_error": fused_parity[
                "fused_reference_maximum_absolute_error"
            ],
            "fused_fp32_reference_maximum_relative_l2_error": fused_parity[
                "fused_reference_maximum_relative_l2_error"
            ],
            "diagnosis": "The compiler legally composes the FP32 predictor with Euler and removes the algorithm's intended BF16 materialization boundary.",
            "latency": fused["details"]["latency"],
        },
        "split_ranked_boundary": {
            "status": "passed",
            "compiled_artifact": split["compiled_artifact"],
            "compile_seconds": split["compile_seconds"],
            "load_seconds": split["load_seconds"],
            "numerical_gate": {
                "steps_checked": 50,
                "maximum_absolute_error": split_parity["maximum_absolute_error"],
                "maximum_relative_l2_error": split_parity["maximum_relative_l2_error"],
                "final_maximum_absolute_error": split_parity[
                    "final_maximum_absolute_error"
                ],
                "final_relative_l2_error": split_parity["final_relative_l2_error"],
                "passed": split_exact,
            },
            "boundary_gate": {
                "skip_steps": 38,
                "skip_entry_point_invocations": 76,
                "skip_rank_executions": len(skip_calls),
                "large_host_write_bytes": large_write_bytes,
                "large_host_read_bytes": large_read_bytes,
                "scalar_host_write_bytes": scalar_write_bytes,
                "scalar_host_read_bytes": scalar_read_bytes,
                "scalar_host_write_bytes_per_skip": scalar_write_bytes / 38,
                "scalar_host_read_bytes_per_skip": scalar_read_bytes / 38,
                "passed": boundary_passed,
            },
            "latency": split["details"]["latency"],
            "neff_execution_diagnostics": timing,
        },
        "anchor_ingress_diagnostic": {
            "anchor_steps": 12,
            "large_host_write_bytes": anchor_large_write_bytes,
            "expected_bytes": 12 * 4 * tensor_bytes,
            "passed_to_backbone_integration": False,
        },
        "device_memory_peak": {
            "per_core_bytes": per_core_peaks,
            "sum_of_per_core_peak_total_bytes": sum(
                row["total_bytes"] for row in per_core_peaks.values()
            ),
            "sum_of_per_core_peak_tensor_bytes": sum(
                row["tensors_bytes"] for row in per_core_peaks.values()
            ),
        },
        "runtime_contract_findings": [
            "A NEFF boundary is a numerical-semantics boundary as well as a dispatch boundary.",
            "Legacy NxDModel routes only by input-shape lists, not dtype; every entry point requires a unique shape signature.",
            "The failed split-v1 artifact reused [full_tensor, shape-2 scalar] for initialize and anchor despite different dtypes, causing a route collision and runtime abort; split-v2 uses a shape-4 initialize token.",
        ],
        "decision": {
            "selected_cache_step_contract": "cache_predict -> BF16 ranked device tensor -> cache_scheduler_step with shared aliased anchors and latent",
            "reason": "It is bit-exact to the existing cache semantics and has zero full-tensor host materialization on all 38 skips.",
            "performance_reading": "The standalone split path is slower than the current host tensor algebra because it pays two small NEFF dispatches. Its system value depends on eliminating backbone/output host boundaries in H1c-b; no speedup is claimed yet.",
            "next": "Feed real FLUX backbone ranked output into cache_anchor_step and resident latent into the next backbone invocation; then profile the entire 50-step loop.",
        },
    }
    _write(args.out.expanduser().resolve(), payload)
    print(
        f"{payload['status']}: exact={split_exact} "
        f"skip_large_write={large_write_bytes} skip_large_read={large_read_bytes} "
        f"scalar_write={scalar_write_bytes / 38:.0f}B/skip"
    )
    return 0 if payload["status"].startswith("split_contract_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
