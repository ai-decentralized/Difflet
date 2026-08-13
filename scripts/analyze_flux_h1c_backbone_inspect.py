#!/usr/bin/env python3
"""Adjudicate real FLUX backbone/cache-step ranked-I/O boundary trace."""

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
        }.items()
    }
    inspect = _load(paths["inspect_json"])
    mechanism = _load(paths["mechanism_result"])
    profile = _load(paths["profile_result"])
    events = inspect.get("trace_event")
    memory_rows = inspect.get("device_mem_usage")
    if not isinstance(events, list) or not isinstance(memory_rows, list):
        raise ValueError("inspect JSON lacks trace_event or device_mem_usage")
    tensor_bytes = int(mechanism["tensor_bytes"])
    expected_tags = [
        "transformer",  # load-time weight-layout transformer, not workload
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

    calls = []
    for worker in workers:
        worker_executes = sorted(
            (event for event in execute_events if int(event["worker_gid"]) == worker),
            key=lambda event: int(event["timestamp"]),
        )
        if [_tag(event) for event in worker_executes] != expected_tags:
            raise ValueError(f"worker {worker} execution sequence mismatch")
        worker_events = [
            event for event in events if int(event.get("worker_gid", -1)) == worker
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

    # Ordinals 2..9 are the actual bidirectional chain.  Input DMA for the
    # next backbone can begin before its execute event and land in the prior
    # call's midpoint window, so boundary/accounting uses the whole interval.
    workload_calls = [call for call in calls if 2 <= int(call["ordinal"]) <= 9]
    writes = [size for call in workload_calls for size in call["write_sizes"]]
    reads = [size for call in workload_calls for size in call["read_sizes"]]
    write_sizes = Counter(writes)
    read_sizes = Counter(reads)
    latent_noise_write_bytes = write_sizes[tensor_bytes] * tensor_bytes
    latent_noise_read_bytes = read_sizes[tensor_bytes] * tensor_bytes
    correctness_passed = (
        mechanism["status"] == "mechanism_passed"
        and float(mechanism["maximum_absolute_error"]) == 0.0
        and float(mechanism["maximum_relative_l2_error"]) == 0.0
    )
    boundary_passed = latent_noise_write_bytes == 0 and latent_noise_read_bytes == 0

    context_sizes = mechanism["context_input_bytes_per_rank"]
    context_expected_counts = {
        int(context_sizes["encoder_hidden_states"]): 3 * 4,
        int(context_sizes["pooled_projections"]): 3 * 4,
        int(context_sizes["rotary_embedding"]): 3 * 4,
        2: 3 * 4 * 2,
    }
    context_counts_match = all(
        write_sizes[size] == count for size, count in context_expected_counts.items()
    )
    context_write_bytes = sum(size * count for size, count in context_expected_counts.items())

    timing = {}
    for tag in (
        "NeuronFluxTransformer2DModel",
        "cache_anchor_step",
        "cache_predict",
        "cache_scheduler_step",
    ):
        values = [
            float(call["execute_time_us"])
            for call in workload_calls
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

    payload = {
        "schema": "difflet-flux-h1c-backbone-ranked-loop-boundary-result",
        "schema_revision": 1,
        "study_id": "flux-h1c-real-backbone-ranked-loop-20260812",
        "status": (
            "ranked_backbone_boundary_passed_context_residency_required"
            if correctness_passed and boundary_passed and context_counts_match
            else "ranked_backbone_boundary_failed"
        ),
        "serving_claim": False,
        "architecture_speed_claim": False,
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "correctness_gate": {
            "checks": mechanism["checks"],
            "final_maximum_absolute_error": mechanism[
                "final_maximum_absolute_error"
            ],
            "final_relative_l2_error": mechanism["final_relative_l2_error"],
            "passed": correctness_passed,
        },
        "bidirectional_ranked_boundary_gate": {
            "real_backbone_calls": 3,
            "resident_skip_steps": 1,
            "latent_noise_tensor_bytes": tensor_bytes,
            "latent_noise_host_write_bytes": latent_noise_write_bytes,
            "latent_noise_host_read_bytes": latent_noise_read_bytes,
            "passed": boundary_passed,
        },
        "remaining_host_input_inventory": {
            "counts_match_frozen_shapes": context_counts_match,
            "three_backbone_calls_total_bytes": context_write_bytes,
            "per_backbone_call_bytes": context_write_bytes / 3,
            "per_backbone_call_mib": context_write_bytes / 3 / 1048576,
            "projected_a12_12_anchor_bytes": context_write_bytes / 3 * 12,
            "projected_a12_12_anchor_mib": context_write_bytes / 3 * 12 / 1048576,
            "projected_full_50_step_bytes": context_write_bytes / 3 * 50,
            "projected_full_50_step_mib": context_write_bytes / 3 * 50 / 1048576,
            "per_call_per_rank": context_sizes,
            "observed_workload_write_size_counts": {
                str(size): count for size, count in sorted(write_sizes.items())
            },
            "note": "Encoder hidden states, pooled projections, and rotary embeddings are invariant within a request but the current backbone signature recopies them to all four ranks on every real anchor.",
        },
        "execution_diagnostics": timing,
        "load_seconds": mechanism["load_seconds"],
        "device_memory_peak": {
            "per_core_bytes": per_core_peaks,
            "sum_of_per_core_peak_total_bytes": sum(
                row["total_bytes"] for row in per_core_peaks.values()
            ),
            "sum_of_per_core_peak_tensor_bytes": sum(
                row["tensors_bytes"] for row in per_core_peaks.values()
            ),
            "sum_of_per_core_peak_weight_bytes": sum(
                row["weights_bytes"] for row in per_core_peaks.values()
            ),
        },
        "decision": {
            "separate_nxd_models_preserve_ranked_handles": boundary_passed,
            "backbone_cache_step_bidirectional_dataflow": correctness_passed
            and boundary_passed,
            "new_primary_bottleneck": "request-invariant backbone conditioning is recopied on every anchor",
            "next": "Add a request initialization/context-state entry point so encoder hidden states, pooled projections, guidance, and rotary embeddings become resident; then run the full 50-step A12 loop and end-to-end comparison.",
            "claim_boundary": "This is a real-model executable boundary result, not a prompt/image quality or end-to-end speed result.",
        },
    }
    _write(args.out.expanduser().resolve(), payload)
    print(
        f"{payload['status']}: latent/noise write={latent_noise_write_bytes} "
        f"read={latent_noise_read_bytes}; context={context_write_bytes / 3 / 1048576:.3f}MiB/backbone"
    )
    return 0 if payload["status"].startswith("ranked_backbone_boundary_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
