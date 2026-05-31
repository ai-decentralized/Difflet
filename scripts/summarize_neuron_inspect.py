#!/usr/bin/env python3
"""Summarize a Neuron Profile inspect trace against a denoise-loop JSON.

`neuron-profile inspect` captures runtime and hardware events for a full user
script. This helper aligns the main model execution clusters with the per-step
records from `scripts/profile_denoise_loop.py` and emits the F3.0b hardware
breakdown needed before deciding whether F3.1 is justified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


NS_PER_S = 1_000_000_000


def _load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _event_time_s(event: dict[str, Any]) -> float:
    return float(event.get("duration") or 0) / NS_PER_S


def _event_bytes(event: dict[str, Any]) -> int:
    return int(event.get("size") or 0)


def _cluster_execute_events(
    events: list[dict[str, Any]],
    *,
    model_name_contains: str,
    gap_ns: int,
) -> list[list[dict[str, Any]]]:
    execute_events = sorted(
        [
            event
            for event in events
            if event.get("name") == "nrt_execute"
            and model_name_contains in str(event.get("model_name") or "")
        ],
        key=lambda event: int(event["timestamp"]),
    )
    clusters: list[list[dict[str, Any]]] = []
    for event in execute_events:
        if not clusters or int(event["timestamp"]) - int(clusters[-1][-1]["timestamp"]) > gap_ns:
            clusters.append([event])
        else:
            clusters[-1].append(event)
    return clusters


def _window_bounds(clusters: list[list[dict[str, Any]]]) -> list[tuple[int, int]]:
    starts = [min(int(event["timestamp"]) for event in cluster) for cluster in clusters]
    ends = [
        max(int(event["timestamp"]) + int(event.get("duration") or 0) for event in cluster)
        for cluster in clusters
    ]
    windows: list[tuple[int, int]] = []
    for index, start in enumerate(starts):
        if index == 0:
            lo = start - 20_000_000
        else:
            lo = (starts[index - 1] + start) // 2
        if index == len(starts) - 1:
            hi = ends[index] + 20_000_000
        else:
            hi = (start + starts[index + 1]) // 2
        windows.append((lo, hi))
    return windows


def _summarize_events(
    events: list[dict[str, Any]],
    *,
    name: str,
    lo: int,
    hi: int,
) -> dict[str, Any]:
    selected = [
        event
        for event in events
        if event.get("name") == name and lo <= int(event["timestamp"]) < hi
    ]
    return {
        "count": len(selected),
        "bytes": sum(_event_bytes(event) for event in selected),
        "time_s": sum(_event_time_s(event) for event in selected),
    }


def _summarize_step(
    denoise_step: dict[str, Any],
    cluster: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    window: tuple[int, int],
) -> dict[str, Any]:
    lo, hi = window
    execute_times = [_event_time_s(event) for event in cluster]
    execute_start = min(int(event["timestamp"]) for event in cluster)
    execute_end = max(
        int(event["timestamp"]) + int(event.get("duration") or 0) for event in cluster
    )
    hardware_execution_time_s = max(execute_times) if execute_times else None
    neff_execution_time_source = (
        "neuron_profile:nrt_execute_max_worker_duration"
        if hardware_execution_time_s is not None
        else None
    )
    host_step_time_s = float(denoise_step["host_step_time_s"])
    non_neff_time_s = (
        max(host_step_time_s - hardware_execution_time_s, 0.0)
        if hardware_execution_time_s is not None
        else None
    )
    cpu_round_trip_share = (
        non_neff_time_s / host_step_time_s
        if non_neff_time_s is not None and host_step_time_s > 0
        else None
    )
    tensor_write = _summarize_events(events, name="nrt_tensor_write", lo=lo, hi=hi)
    tensor_read = _summarize_events(events, name="nrt_tensor_read", lo=lo, hi=hi)
    dmem_copyin = _summarize_events(events, name="dmem_buf_copyin", lo=lo, hi=hi)
    dmem_copyout = _summarize_events(events, name="dmem_buf_copyout", lo=lo, hi=hi)
    host_device_transfer_count = tensor_write["count"] + tensor_read["count"]
    host_device_transfer_bytes = tensor_write["bytes"] + tensor_read["bytes"]
    host_device_transfer_time_s = tensor_write["time_s"] + tensor_read["time_s"]
    return {
        "step_index": int(denoise_step["step_index"]),
        "host_step_time_s": host_step_time_s,
        "device_step_time_s": float(denoise_step["device_step_time_s"]),
        "scheduler_time_s": float(denoise_step.get("scheduler_time_s") or 0.0),
        "mark_step_time_s": float(denoise_step.get("mark_step_time_s") or 0.0),
        "neff_execution_time_s": hardware_execution_time_s,
        "neff_execution_time_source": neff_execution_time_source,
        "hardware_execution_time_s": hardware_execution_time_s,
        "hardware_execution_event_count": len(cluster),
        "hardware_execution_window_time_s": (execute_end - execute_start) / NS_PER_S,
        "hardware_execution_worker_gids": sorted(
            {int(event["worker_gid"]) for event in cluster if event.get("worker_gid") is not None}
        ),
        "non_neff_time_s": non_neff_time_s,
        "cpu_round_trip_share": cpu_round_trip_share,
        "host_device_transfer_count": host_device_transfer_count,
        "host_device_transfer_bytes": host_device_transfer_bytes,
        "host_device_transfer_time_s": host_device_transfer_time_s,
        "nrt_tensor_write": tensor_write,
        "nrt_tensor_read": tensor_read,
        "dmem_buf_copyin": dmem_copyin,
        "dmem_buf_copyout": dmem_copyout,
    }


def _summarize_combined(steps: list[dict[str, Any]]) -> dict[str, Any]:
    shares = [float(step["cpu_round_trip_share"]) for step in steps]
    neff = [float(step["neff_execution_time_s"]) for step in steps]
    transfer_count_known = all(
        step.get("host_device_transfer_count") is not None for step in steps
    )
    transfer_bytes_known = all(
        step.get("host_device_transfer_bytes") is not None for step in steps
    )
    return {
        "num_steps": len(steps),
        "hardware_timed_steps": len(
            [
                step
                for step in steps
                if step.get("neff_execution_time_source")
                == "neuron_profile:nrt_execute_max_worker_duration"
            ]
        ),
        "mean_neff_execution_time_s": mean(neff) if neff else None,
        "mean_hardware_execution_time_s": mean(neff) if neff else None,
        "mean_cpu_round_trip_share": mean(shares) if shares else None,
        "max_cpu_round_trip_share": max(shares) if shares else None,
        "steady_state_max_cpu_round_trip_share_after_step0": (
            max(shares[1:]) if len(shares) > 1 else None
        ),
        "steady_state_mean_cpu_round_trip_share_after_step0": (
            mean(shares[1:]) if len(shares) > 1 else None
        ),
        "total_host_device_transfer_count": sum(
            int(step["host_device_transfer_count"]) for step in steps
        )
        if transfer_count_known
        else None,
        "total_host_device_transfer_bytes": sum(
            int(step["host_device_transfer_bytes"]) for step in steps
        )
        if transfer_bytes_known
        else None,
        "total_host_device_transfer_time_s": sum(
            float(step["host_device_transfer_time_s"]) for step in steps
        ),
        "clears_10pct_gate": bool(shares and max(shares[1:] or shares) >= 0.10),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--denoise-json", required=True)
    parser.add_argument("--inspect-json", required=True)
    parser.add_argument("--model-name-contains", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cluster-gap-ms", type=float, default=10.0)
    args = parser.parse_args()

    denoise = _load_json(args.denoise_json)
    inspect = _load_json(args.inspect_json)
    events = inspect["trace_event"]
    clusters = _cluster_execute_events(
        events,
        model_name_contains=args.model_name_contains,
        gap_ns=int(args.cluster_gap_ms * 1_000_000),
    )
    steps = denoise["steps"]
    if len(clusters) != len(steps):
        raise ValueError(
            f"expected {len(steps)} execution clusters, found {len(clusters)} "
            f"for model substring {args.model_name_contains!r}"
        )
    windows = _window_bounds(clusters)
    combined_steps = [
        _summarize_step(step, cluster, events, window=window)
        for step, cluster, window in zip(steps, clusters, windows)
    ]
    result = {
        "schema": "nova-f3-denoise-loop-neuron-inspect-summary-v1",
        "denoise_json": args.denoise_json,
        "inspect_json": args.inspect_json,
        "model_name_contains": args.model_name_contains,
        "step_window_note": (
            "Transfer counters are assigned to the nearest execution cluster using "
            "midpoints between cluster starts; first/last windows use a 20 ms edge margin."
        ),
        "steps": combined_steps,
        "summary": _summarize_combined(combined_steps),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
