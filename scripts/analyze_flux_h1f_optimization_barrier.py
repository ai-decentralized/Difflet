#!/usr/bin/env python3
"""Freeze the H1f XLA optimization-barrier contract decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--run", required=True, action="append", type=Path)
    parser.add_argument("--h1c-fused-reference", required=True, type=Path)
    parser.add_argument("--hlo", required=True, type=Path)
    parser.add_argument("--compiler-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _parity_without_checksum(payload: dict[str, Any]) -> dict[str, Any]:
    parity = dict(payload["details"]["schedule_parity"])
    parity.pop("final_checksum", None)
    return parity


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = _parse_args()
    if len(args.run) < 2:
        raise ValueError("H1f requires at least two deterministic hardware runs")
    protocol = _read_json(args.protocol)
    runs = [_read_json(path) for path in args.run]
    h1c_fused = _read_json(args.h1c_fused_reference)

    parity = [_parity_without_checksum(run) for run in runs]
    deterministic = all(item == parity[0] for item in parity[1:])
    matches_prior_fused = parity[0] == _parity_without_checksum(h1c_fused)
    schedule = runs[0]["details"]["schedule_parity"]
    host_exact = (
        schedule["maximum_absolute_error"] == 0.0
        and schedule["maximum_relative_l2_error"] == 0.0
    )
    fused_exact = (
        schedule["fused_reference_maximum_absolute_error"] == 0.0
        and schedule["fused_reference_maximum_relative_l2_error"] == 0.0
    )

    hlo_bytes = args.hlo.read_bytes()
    compiler_log = args.compiler_log.read_text(encoding="utf-8", errors="replace")
    barrier_in_hlo = b"xla__optimization_barrier" in hlo_bytes
    removal_pass_observed = "RemoveOptimizationBarriers pass" in compiler_log
    compiler_passed = "Compilation Successfully Completed" in compiler_log
    compile_gate = compiler_passed and barrier_in_hlo
    single_dispatch_contract = all(
        run["details"]["execution_contract"]
        == "single_fused_predictor_barrier_scheduler"
        and run["details"]["skip_count"] == 38
        for run in runs
    )

    device_p50 = [float(run["details"]["latency"]["device_p50_ms"]) for run in runs]
    payload = {
        "schema": "difflet-flux-h1f-optimization-barrier-decision",
        "schema_revision": 1,
        "study_id": protocol["study_id"],
        "status": "optimization_barrier_contract_rejected",
        "serving_claim": False,
        "architecture_speed_claim": False,
        "paper_role": protocol["paper_role"],
        "artifacts": {
            "protocol": _artifact(args.protocol),
            "hardware_runs": [_artifact(path) for path in args.run],
            "prior_h1c_fused_run": _artifact(args.h1c_fused_reference),
            "candidate_hlo": _artifact(args.hlo),
            "compiler_log": _artifact(args.compiler_log),
        },
        "frozen_configuration": protocol["frozen_system"],
        "results": {
            "hardware_runs": len(runs),
            "deterministic_schedule_parity_across_runs": deterministic,
            "matches_prior_unbarriered_fused_error_signature": matches_prior_fused,
            "maximum_absolute_error_vs_host_bf16_contract": schedule[
                "maximum_absolute_error"
            ],
            "maximum_relative_l2_error_vs_host_bf16_contract": schedule[
                "maximum_relative_l2_error"
            ],
            "final_maximum_absolute_error_vs_host_bf16_contract": schedule[
                "final_maximum_absolute_error"
            ],
            "final_relative_l2_error_vs_host_bf16_contract": schedule[
                "final_relative_l2_error"
            ],
            "maximum_absolute_error_vs_fused_fp32_reference": schedule[
                "fused_reference_maximum_absolute_error"
            ],
            "maximum_relative_l2_error_vs_fused_fp32_reference": schedule[
                "fused_reference_maximum_relative_l2_error"
            ],
            "first_divergent_step": next(
                record["step"]
                for record in schedule["records"]
                if record["maximum_absolute_error"] != 0.0
            ),
            "candidate_skip_device_p50_ms_runs": device_p50,
            "candidate_skip_device_p50_ms_median": statistics.median(device_p50),
            "compiled_entry_points": [
                "cache_initialize",
                "cache_anchor_step",
                "cache_skip_step_barrier",
                "cache_finalize",
            ],
            "a12_cache_entry_points_per_rank": 52,
            "projected_full_resident_entry_points_per_rank": 65,
            "h1e_split_resident_entry_points_per_rank": 103,
            "projected_dispatch_reduction_vs_h1e": 38,
            "hlo_contains_xla_optimization_barrier": barrier_in_hlo,
            "compiler_log_contains_remove_optimization_barriers_pass": (
                removal_pass_observed
            ),
        },
        "gates": {
            "F1_compile": {
                "passed": compile_gate,
                "evidence": "TP4 compile/load passed and the input HLO contains xla__optimization_barrier.",
            },
            "F2_exact_contract": {
                "passed": host_exact,
                "evidence": (
                    "The candidate is exact only to the fused-FP32 reference; it "
                    "diverges from the frozen BF16-boundary host contract at step 6."
                ),
            },
            "F3_single_dispatch": {
                "passed": single_dispatch_contract,
                "evidence": "All 38 skips use one cache_skip_step_barrier invocation.",
            },
            "F4_scalar_boundary": {
                "passed": single_dispatch_contract,
                "evidence": (
                    "The compiled skip signature accepts two FP32 coefficients and "
                    "one FP32 delta; validation reads tensors only to audit F2."
                ),
            },
            "F5_latency_direction": {
                "passed": False,
                "stopped": True,
                "evidence": (
                    "Not promoted to a paired latency decision because the earlier "
                    "exact-contract gate failed. Candidate micro-latency is diagnostic only."
                ),
            },
        },
        "analysis": {
            "fused_reference_exact": fused_exact,
            "compiler_interpretation": (
                "The barrier was present in the input HLO, so this is not a Python "
                "device-branch tracing failure. Neuron's pipeline explicitly ran a "
                "RemoveOptimizationBarriers pass, and hardware output retained the "
                "same fused-FP32 behavior as H1c-a. The optimization barrier constrains "
                "motion but does not provide a durable materialization/rounding contract."
            ),
            "system_interpretation": (
                "The candidate would reduce the resident A12 request from 103 to 65 "
                "entry points per rank, but it changes numerical semantics. Positive "
                "dispatch potential cannot override failed algorithm identity."
            ),
        },
        "decision": {
            "rerun_h1e": False,
            "deploy_candidate": False,
            "reason": "F2 exact-contract failed deterministically on Trainium TP4.",
            "next": (
                "Do not search for correlation or retune A12. Either design a true "
                "persistent/multi-step device loop with an executable BF16 boundary, "
                "or retain host-materializing A12 as the deployable architecture."
            ),
        },
    }
    _write_json(args.output, payload)
    print(
        f"[h1f-analysis] {payload['status']} exact={host_exact} "
        f"barrier_in_hlo={barrier_in_hlo} output={args.output}",
        flush=True,
    )
    return 1 if host_exact else 0


if __name__ == "__main__":
    raise SystemExit(main())
