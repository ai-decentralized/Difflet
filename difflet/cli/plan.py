"""``difflet plan`` -- show the parallel configurations this host and model allow.

Read-only and offline: it probes the host with ``neuron-ls`` and reads the
registry, but never loads weights, claims a NeuronCore, or compiles anything. Run
it before a 25-minute AOT compile, not after.
"""

from __future__ import annotations

import argparse
import json
import sys

from difflet.planner.feasibility import FeasibilityReport, enumerate_candidates
from difflet.planner.hardware import HardwareProfile, detect_hardware


def build_report(args: argparse.Namespace) -> tuple[HardwareProfile, FeasibilityReport]:
    from difflet.cli.main import _MODEL_TYPE
    from difflet.registry import resolve_model

    hardware = detect_hardware(allocated_cores_override=getattr(args, "total_cores", None))
    entry = resolve_model(args.model_id, model_type=_MODEL_TYPE[args.model_id])
    report = enumerate_candidates(
        model_name=entry.name,
        capabilities=entry.require_capabilities(),
        cores=hardware.allocated_cores,
        serving=bool(getattr(args, "serving", False)),
    )
    return hardware, report


def run(args: argparse.Namespace) -> None:
    hardware, report = build_report(args)
    if getattr(args, "json", False):
        json.dump(_as_dict(args, hardware, report), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    _print_text(args, hardware, report)


def _as_dict(
    args: argparse.Namespace, hardware: HardwareProfile, report: FeasibilityReport
) -> dict:
    return {
        "model_id": args.model_id,
        "model": report.model_name,
        "hardware": {
            "instance_type": hardware.instance_type,
            "platform_target": hardware.platform_target,
            "num_devices": hardware.num_devices,
            "cores_per_device": hardware.cores_per_device,
            "allocated_cores": hardware.allocated_cores,
            "hbm_bytes_per_device": hardware.hbm_bytes_per_device,
            "source": hardware.source,
        },
        "feasible": [
            {
                "label": candidate.label,
                "flags": _flags(candidate.parallel),
                "parallel": {
                    "tp_degree": candidate.parallel.tp_degree,
                    "cp_degree": candidate.parallel.cp_degree,
                    "cp_mode": candidate.parallel.cp_mode,
                    "cfg_parallel_enabled": candidate.parallel.cfg_parallel_enabled,
                    "sp_enabled": candidate.parallel.sp_enabled,
                    "dp_degree": candidate.parallel.dp_degree,
                },
                "world_size": candidate.world_size,
            }
            for candidate in report.feasible
        ],
        "rejected": [
            {"label": r.label, "kind": r.kind, "reason": r.reason} for r in report.rejected
        ],
    }


def _flags(parallel) -> str:
    """The CLI flags that reproduce a configuration -- copy-pasteable output."""

    parts = ["--tp-degree", str(parallel.tp_degree)]
    if parallel.cp_degree > 1:
        parts += ["--cp-degree", str(parallel.cp_degree)]
        if parallel.cp_mode != "gather_kv":
            parts += ["--cp-mode", parallel.cp_mode]
    if parallel.cfg_parallel_enabled:
        parts.append("--cfg-parallel")
    if parallel.sp_enabled:
        parts.append("--sp")
    if parallel.dp_degree > 1:
        parts += ["--dp", str(parallel.dp_degree)]
    return " ".join(parts)


def _print_text(
    args: argparse.Namespace, hardware: HardwareProfile, report: FeasibilityReport
) -> None:
    print(f"model    {args.model_id}  ({report.model_name})")
    print(f"hardware {hardware.describe()}")
    if not hardware.source.startswith("neuron-ls"):
        print(
            "         note: the Neuron driver could not be queried; core count is "
            "a per-platform assumption. Pass --total-cores to plan for a specific host."
        )
    if hardware.busy_cores:
        busy = ",".join(str(core) for core in hardware.busy_cores)
        print(f"         note: cores {busy} are already held by another process")
    print(f"cores    {hardware.allocated_cores} (every configuration below uses all of them)")
    print()

    if report.feasible:
        width = max(len(candidate.label) for candidate in report.feasible)
        print(f"{len(report.feasible)} feasible:")
        for candidate in report.feasible:
            print(f"  {candidate.label:<{width}}  {_flags(candidate.parallel)}")
    else:
        print("no feasible configuration -- every candidate was rejected below.")

    if report.rejected:
        print()
        width = max(len(rejection.label) for rejection in report.rejected)
        print(f"{len(report.rejected)} rejected:")
        for rejection in report.rejected:
            print(f"  {rejection.label:<{width}}  [{rejection.kind}] {rejection.reason}")
