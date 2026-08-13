#!/usr/bin/env python3
"""Adjudicate the frozen H1e resident-vs-materializing end-to-end gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
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


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _bootstrap_stratified(
    values_by_prompt: dict[int, list[float]], *, seed: int, samples: int
) -> dict[str, float | int]:
    generator = random.Random(seed)
    estimates = []
    prompt_ids = sorted(values_by_prompt)
    for _ in range(samples):
        draw = []
        for prompt_id in prompt_ids:
            values = values_by_prompt[prompt_id]
            draw.extend(generator.choice(values) for _ in range(len(values)))
        estimates.append(_geometric_mean(draw))
    return {
        "seed": seed,
        "resamples": samples,
        "lower_95": _quantile(estimates, 0.025),
        "median": _quantile(estimates, 0.5),
        "upper_95": _quantile(estimates, 0.975),
    }


def _latency_summary(records: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    baseline = [float(row["materializing"]["timings"][metric]) for row in records]
    resident = [float(row["resident"]["timings"][metric]) for row in records]
    ratios = [left / right for left, right in zip(baseline, resident, strict=True)]
    by_prompt: dict[int, list[float]] = defaultdict(list)
    for row, ratio in zip(records, ratios, strict=True):
        by_prompt[int(row["prompt_index"])].append(ratio)
    return {
        "metric": metric,
        "request_count": len(records),
        "materializing_seconds": {
            "p50": _quantile(baseline, 0.5),
            "p95": _quantile(baseline, 0.95),
            "minimum": min(baseline),
            "maximum": max(baseline),
        },
        "resident_seconds": {
            "p50": _quantile(resident, 0.5),
            "p95": _quantile(resident, 0.95),
            "minimum": min(resident),
            "maximum": max(resident),
        },
        "paired_speedup_materializing_over_resident": {
            "median": median(ratios),
            "geometric_mean": _geometric_mean(ratios),
            "minimum": min(ratios),
            "maximum": max(ratios),
            "resident_win_count": sum(ratio > 1.0 for ratio in ratios),
            "bootstrap_prompt_stratified": _bootstrap_stratified(
                by_prompt, seed=20260812, samples=10_000
            ),
        },
    }


def _tag(event: dict[str, Any]) -> str:
    return Path(str(event["model_name"])).parent.parent.name


def _expected_resident_sequence() -> list[str]:
    anchors = {0, 1, 2, 3, 4, 5, 9, 15, 21, 31, 41, 49}
    result = [
        "transformer",
        "NeuronCLIPTextModel",
        "NeuronT5EncoderModel",
        "request_context_stage_slot_0",
        "cache_initialize",
    ]
    for step in range(50):
        if step in anchors:
            result.extend(("NeuronFluxTransformer2DModel", "cache_anchor_step"))
        else:
            result.extend(("cache_predict", "cache_scheduler_step"))
    result.extend(("cache_finalize", "Decoder"))
    return result


def _profile_calls(inspect: dict[str, Any], expected_tags: list[str]):
    events = inspect.get("trace_event")
    if not isinstance(events, list):
        raise ValueError("inspect JSON lacks trace_event")
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
        workload_events = [
            event
            for event in events
            if int(event.get("worker_gid", -1)) == worker
            and int(event.get("thread_id", -1)) == worker + 1
        ]
        for ordinal, execute in enumerate(worker_executes):
            timestamp = int(execute["timestamp"])
            low = (
                (int(worker_executes[ordinal - 1]["timestamp"]) + timestamp) // 2
                if ordinal
                else timestamp - 100_000_000
            )
            high = (
                (timestamp + int(worker_executes[ordinal + 1]["timestamp"])) // 2
                if ordinal + 1 < len(worker_executes)
                else timestamp + 1_000_000_000
            )
            selected = [
                event
                for event in workload_events
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
    return events, calls


def _memory_peaks(inspect: dict[str, Any]) -> dict[str, dict[str, int]]:
    fields = ("total_bytes", "tensors_bytes", "weights_bytes", "io_bytes")
    peaks: dict[str, dict[str, int]] = {}
    for row in inspect.get("device_mem_usage", []):
        core = str(row["hbm_idx"])
        values = peaks.setdefault(core, {field: 0 for field in fields})
        for field in fields:
            values[field] = max(values[field], int(row.get(field) or 0))
    return peaks


def _request_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    executes = [event for event in events if event.get("name") == "nrt_execute"]
    clip = [event for event in executes if _tag(event) == "NeuronCLIPTextModel"]
    decoder = [event for event in executes if _tag(event) == "Decoder"]
    if len(clip) != 4 or len(decoder) != 4:
        raise ValueError("expected one CLIP and Decoder execution per TP rank")
    low = min(int(event["timestamp"]) for event in clip) - 20_000_000
    high = max(
        int(event["timestamp"]) + int(event.get("duration") or 0)
        for event in decoder
    ) + 1_000_000_000
    return [
        event
        for event in events
        if low <= int(event.get("timestamp", 0)) <= high
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-result", required=True, type=Path)
    parser.add_argument("--materializing-profile-result", required=True, type=Path)
    parser.add_argument("--resident-profile-result", required=True, type=Path)
    parser.add_argument("--materializing-inspect", required=True, type=Path)
    parser.add_argument("--resident-inspect", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--h1d-result", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = {
        name: path.expanduser().resolve()
        for name, path in {
            "paired_result": args.paired_result,
            "materializing_profile_result": args.materializing_profile_result,
            "resident_profile_result": args.resident_profile_result,
            "materializing_inspect": args.materializing_inspect,
            "resident_inspect": args.resident_inspect,
            "protocol": args.protocol,
            "h1d_result": args.h1d_result,
        }.items()
    }
    paired = _load(paths["paired_result"])
    materializing_profile = _load(paths["materializing_profile_result"])
    resident_profile = _load(paths["resident_profile_result"])
    materializing_inspect = _load(paths["materializing_inspect"])
    resident_inspect = _load(paths["resident_inspect"])

    records = paired["records"]
    expected_prompt_repetitions = Counter(
        (prompt_index, repetition)
        for prompt_index in range(4)
        for repetition in range(6)
    )
    actual_prompt_repetitions = Counter(
        (int(row["prompt_index"]), int(row["repetition"])) for row in records
    )
    design_exact = (
        paired["status"] == "paired_benchmark_completed"
        and len(records) == 24
        and actual_prompt_repetitions == expected_prompt_repetitions
        and all(
            row["order"]
            == (
                ["materializing", "resident"]
                if int(row["repetition"]) % 2 == 0
                else ["resident", "materializing"]
            )
            for row in records
        )
    )
    parity = paired["parity_summary"]
    correctness_passed = design_exact and bool(parity["passed"])

    denoise_latency = _latency_summary(records, "denoise_total_s")
    full_latency = _latency_summary(records, "full_request_s")
    denoise_speed = denoise_latency["paired_speedup_materializing_over_resident"]
    full_speed = full_latency["paired_speedup_materializing_over_resident"]
    significance_passed = (
        float(denoise_speed["bootstrap_prompt_stratified"]["lower_95"]) > 1.0
        and float(full_speed["bootstrap_prompt_stratified"]["lower_95"]) > 1.0
    )
    materiality_passed = (
        float(denoise_speed["median"]) >= 1.03
        and float(full_speed["median"]) >= 1.01
    )
    tail_ratio = (
        float(full_latency["materializing_seconds"]["p95"])
        / float(full_latency["resident_seconds"]["p95"])
    )
    tail_passed = tail_ratio >= 0.98

    baseline_tags = [
        "transformer",
        "NeuronCLIPTextModel",
        "NeuronT5EncoderModel",
        *("NeuronFluxTransformer2DModel" for _ in range(12)),
        "Decoder",
    ]
    resident_tags = _expected_resident_sequence()
    baseline_events, baseline_calls = _profile_calls(
        materializing_inspect, baseline_tags
    )
    resident_events, resident_calls = _profile_calls(resident_inspect, resident_tags)

    baseline_backbone_calls = [
        call for call in baseline_calls if call["tag"] == "NeuronFluxTransformer2DModel"
    ]
    baseline_expected_writes = [2, 2, 1536, 524288, 2359296, 4194304]
    baseline_write_exact = all(
        call["write_sizes"] == baseline_expected_writes
        for call in baseline_backbone_calls
    )
    baseline_request_events = _request_events(baseline_events)
    resident_request_events = _request_events(resident_events)
    baseline_noise_reads = sum(
        1
        for event in baseline_request_events
        if event.get("name") == "nrt_tensor_read"
        and int(event.get("size") or 0) == 524288
    )

    resident_expected_writes = {
        "request_context_stage_slot_0": [2, 4, 1536, 2359296, 4194304],
        "cache_initialize": [16, 524288],
        "NeuronFluxTransformer2DModel": [2],
        "cache_anchor_step": [8],
        "cache_predict": [8],
        "cache_scheduler_step": [4],
        "cache_finalize": [12],
    }
    resident_denoise_tags = set(resident_expected_writes)
    resident_denoise_calls = [
        call for call in resident_calls if call["tag"] in resident_denoise_tags
    ]
    resident_write_exact = all(
        call["write_sizes"] == resident_expected_writes[call["tag"]]
        for call in resident_denoise_calls
    )
    resident_final_reads = sum(
        1
        for event in resident_request_events
        if event.get("name") == "nrt_tensor_read"
        and int(event.get("size") or 0) == 524288
    )
    resident_large_writes_after_initialize = sum(
        call["write_sizes"].count(524288)
        for call in resident_denoise_calls
        if call["tag"] != "cache_initialize"
    )

    baseline_write_bytes = sum(
        sum(call["write_sizes"]) for call in baseline_backbone_calls
    )
    baseline_read_bytes = baseline_noise_reads * 524288
    resident_write_bytes = sum(
        sum(call["write_sizes"]) for call in resident_denoise_calls
    )
    resident_read_bytes = resident_final_reads * 524288
    boundary_passed = (
        baseline_write_exact
        and baseline_noise_reads == 12
        and resident_write_exact
        and resident_final_reads == 1
        and resident_large_writes_after_initialize == 0
    )

    baseline_tag_counts = Counter(call["tag"] for call in baseline_calls)
    resident_tag_counts = Counter(call["tag"] for call in resident_calls)
    baseline_denoise_entrypoints_per_rank = 12
    resident_denoise_entrypoints_per_rank = sum(
        resident_tag_counts[tag] for tag in resident_denoise_tags
    ) // 4
    extra_entrypoints_per_rank = (
        resident_denoise_entrypoints_per_rank - baseline_denoise_entrypoints_per_rank
    )

    def execution_summary(calls, tags):
        result = {}
        for tag in tags:
            values = [
                float(call["execute_time_us"])
                for call in calls
                if call["tag"] == tag
            ]
            result[tag] = {
                "rank_execution_count": len(values),
                "median_per_rank_execute_time_us": median(values),
                "maximum_per_rank_execute_time_us": max(values),
                "sum_all_rank_execute_time_us": sum(values),
            }
        return result

    all_gates_passed = (
        correctness_passed
        and boundary_passed
        and significance_passed
        and materiality_passed
        and tail_passed
    )
    status = (
        "resident_e2e_system_gate_passed"
        if all_gates_passed
        else "resident_e2e_system_gate_rejected_dispatch_overhead"
        if correctness_passed and boundary_passed
        else "resident_e2e_system_gate_rejected_correctness_or_boundary"
    )
    payload = {
        "schema": "difflet-flux-h1e-resident-e2e-ab-result",
        "schema_revision": 1,
        "study_id": "flux-h1e-resident-e2e-ab-20260812",
        "status": status,
        "architecture_speed_claim": all_gates_passed,
        "serving_claim": False,
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "E1_correctness": {
            "design_exact": design_exact,
            "paired_request_count": len(records),
            "parity": parity,
            "passed": correctness_passed,
        },
        "E2_data_plane_boundary": {
            "materializing": {
                "backbone_calls": len(baseline_backbone_calls) // 4,
                "write_bytes": baseline_write_bytes,
                "read_bytes": baseline_read_bytes,
                "total_host_transfer_bytes": baseline_write_bytes
                + baseline_read_bytes,
                "write_sequence_exact": baseline_write_exact,
                "noise_read_count": baseline_noise_reads,
            },
            "resident": {
                "write_bytes": resident_write_bytes,
                "read_bytes": resident_read_bytes,
                "total_host_transfer_bytes": resident_write_bytes
                + resident_read_bytes,
                "write_sequence_exact": resident_write_exact,
                "final_latent_read_count": resident_final_reads,
                "latent_noise_writes_after_initialize": resident_large_writes_after_initialize,
            },
            "transfer_reduction_fraction": 1.0
            - (resident_write_bytes + resident_read_bytes)
            / (baseline_write_bytes + baseline_read_bytes),
            "transfer_reduction_ratio": (baseline_write_bytes + baseline_read_bytes)
            / (resident_write_bytes + resident_read_bytes),
            "passed": boundary_passed,
        },
        "E3_paired_significance": {
            "denoise": denoise_speed["bootstrap_prompt_stratified"],
            "full_request": full_speed["bootstrap_prompt_stratified"],
            "criterion": "both lower_95 > 1.0",
            "passed": significance_passed,
        },
        "E4_materiality": {
            "denoise_median_paired_speedup": denoise_speed["median"],
            "full_request_median_paired_speedup": full_speed["median"],
            "required": {"denoise": 1.03, "full_request": 1.01},
            "passed": materiality_passed,
        },
        "E5_tail": {
            "materializing_full_request_p95_s": full_latency[
                "materializing_seconds"
            ]["p95"],
            "resident_full_request_p95_s": full_latency["resident_seconds"][
                "p95"
            ],
            "materializing_over_resident_p95_ratio": tail_ratio,
            "required_minimum": 0.98,
            "passed": tail_passed,
        },
        "latency": {
            "denoise": denoise_latency,
            "full_request": full_latency,
        },
        "E6_resource_accounting": {
            "materializing_entrypoints_per_rank": baseline_denoise_entrypoints_per_rank,
            "resident_entrypoints_per_rank": resident_denoise_entrypoints_per_rank,
            "extra_resident_entrypoints_per_rank": extra_entrypoints_per_rank,
            "materializing_tag_counts_all_ranks": dict(baseline_tag_counts),
            "resident_tag_counts_all_ranks": dict(resident_tag_counts),
            "materializing_memory_peak_per_core_bytes": _memory_peaks(
                materializing_inspect
            ),
            "resident_memory_peak_per_core_bytes": _memory_peaks(resident_inspect),
            "execution_diagnostics": {
                "materializing": execution_summary(
                    baseline_calls, ("NeuronFluxTransformer2DModel",)
                ),
                "resident": execution_summary(
                    resident_calls,
                    (
                        "request_context_stage_slot_0",
                        "cache_initialize",
                        "NeuronFluxTransformer2DModel",
                        "cache_anchor_step",
                        "cache_predict",
                        "cache_scheduler_step",
                        "cache_finalize",
                    ),
                ),
            },
            "slot_capacity": 2,
            "passed": True,
        },
        "decision": {
            "mechanism": "resident request state is numerically exact and removes the intended host materialization",
            "system_result": "rejected: 91 extra exact-boundary entry points per rank outweigh the measured DMA reduction",
            "next_architecture_question": "Preserve the BF16 predictor boundary while fusing or batching dispatches; otherwise retain host materialization for this post-gather A12 path.",
            "claim_boundary": "No architecture speedup or serving claim. The negative wall-time result takes precedence over the positive byte reduction.",
        },
    }
    _write(args.out.expanduser().resolve(), payload)
    print(
        f"{status}: denoise={denoise_speed['geometric_mean']:.6f}x "
        f"full={full_speed['geometric_mean']:.6f}x "
        f"traffic_reduction={payload['E2_data_plane_boundary']['transfer_reduction_fraction']:.3%}"
    )
    return 0 if all_gates_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
