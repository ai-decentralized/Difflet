"""``difflet plan`` -- rank the parallel configurations this host and model allow.

Read-only and offline: it probes the host with ``neuron-ls``, reads the registry,
and consults packaged measurements, but never loads weights, claims a NeuronCore,
or compiles anything. Run it before a 25-minute AOT compile, not after.
"""

from __future__ import annotations

import argparse
import json
import sys

from difflet.planner.planner import Plan, plan as build_plan


def run(args: argparse.Namespace) -> None:
    from difflet.cli.main import _MODEL_TYPE

    result = build_plan(
        args.model_id,
        model_type=_MODEL_TYPE[args.model_id],
        height=getattr(args, "height", None),
        width=getattr(args, "width", None),
        num_frames=getattr(args, "num_frames", None),
        steps=getattr(args, "steps", None),
        objective=getattr(args, "objective", "latency"),
        total_cores=getattr(args, "total_cores", None),
        serving=bool(getattr(args, "serving", False)),
        cache_dir=getattr(args, "cache_dir", None),
        survivors=getattr(args, "survivors", None),
        topology_aware=not getattr(args, "no_topology", False),
    )
    if getattr(args, "json", False):
        json.dump(_as_dict(result), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    _print_text(result)


def flags_for(parallel) -> str:
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


def placement_lines(entry) -> list[str]:
    """Human-readable stage-2 notes for one ranked entry.

    Says which axis order the runtime uses today, where each collective family
    physically lands under it, and what the best alternative order would buy --
    with the caveat that only the default order is executable, so an
    alternative is a recommendation, not something to paste after
    ``difflet compile``.
    """

    choice = entry.placement
    if choice is None:
        return []
    lines = [f"placement: order {choice.default.order_text} (runtime default)"]
    for family in choice.default.families:
        if family.collective.barrier:
            lines.append(
                f"  {family.collective.family:>14} on {family.bottleneck_tier} "
                f"(barrier, {family.collective.axis} axis)"
            )
    if not choice.default_is_best:
        delta = 1.0 - choice.best.step_seconds / choice.default.step_seconds
        lines.append(
            f"  best order {choice.best.order_text} would cut the predicted step by "
            f"{delta:.1%} -- needs a runtime mesh-order flag (not executable today)"
        )
    total_contention = sum(family.contention_seconds for family in choice.default.families)
    if total_contention > 0:
        lines.append(
            f"  shared-link contention costs ~{total_contention * 1000:.1f} ms/step "
            "at the default order"
        )
    return lines


def _as_dict(result: Plan) -> dict:
    hardware = result.hardware
    return {
        "model_id": result.model_id,
        "model": result.model_name,
        "objective": result.objective,
        "steps": result.steps,
        "shape": result.shape,
        "sequence": {"image": result.sequence.image, "text": result.sequence.text},
        "hardware": {
            "instance_type": hardware.instance_type,
            "platform_target": hardware.platform_target,
            "num_devices": hardware.num_devices,
            "cores_per_device": hardware.cores_per_device,
            "allocated_cores": hardware.allocated_cores,
            "hbm_bytes_per_device": hardware.hbm_bytes_per_device,
            "source": hardware.source,
        },
        "calibration": {
            "kind": result.calibration.kind,
            "anchors": list(result.calibration.anchor_labels),
            "bandwidth_is_assumed": result.calibration.bandwidth_is_assumed,
            "compute_seconds_single_core": result.calibration.compute_seconds_single_core,
            "bandwidth_bytes_per_second": result.calibration.bandwidth_bytes_per_second,
        },
        "ranked": [
            {
                "rank": index,
                "label": entry.label,
                "flags": flags_for(entry.parallel),
                "step_seconds": entry.step_seconds,
                "request_seconds": entry.request_seconds,
                "requests_per_second": entry.throughput,
                "score": entry.score,
                "evidence": entry.prediction.evidence,
                "cached": entry.cached,
                "weight_bytes": entry.weight_bytes,
                "weights_over_budget": entry.weights_over_budget,
                "comm_bytes_per_step": entry.prediction.comm.total,
                "measurement_source": (
                    entry.prediction.measurement.source if entry.prediction.measurement else None
                ),
                "stage2_step_seconds": entry.stage2_step_seconds,
                "placement": _placement_dict(entry),
            }
            for index, entry in enumerate(result.ranked, start=1)
        ],
        "survivors": list(result.survivors),
        "rejected": [
            {"label": r.label, "kind": r.kind, "reason": r.reason}
            for r in result.feasibility.rejected
        ],
    }


def _placement_dict(entry) -> dict | None:
    choice = entry.placement
    if choice is None:
        return None

    def score_dict(score) -> dict:
        return {
            "order": score.order_text,
            "is_default": score.placement.is_default,
            "step_seconds": score.step_seconds,
            "barrier_seconds": score.barrier_seconds,
            "overlappable_seconds": score.overlappable_seconds,
            "families": [
                {
                    "family": family.collective.family,
                    "axis": family.collective.axis,
                    "barrier": family.collective.barrier,
                    "multiplier": family.multiplier,
                    "bottleneck_tier": family.bottleneck_tier,
                    "contention_seconds": family.contention_seconds,
                }
                for family in score.families
            ],
        }

    return {
        "best": score_dict(choice.best),
        "default": score_dict(choice.default),
        "alternates": [score_dict(score) for score in choice.alternates],
    }


def _print_text(result: Plan) -> None:
    hardware = result.hardware
    shape = " x ".join(
        str(result.shape[key]) for key in ("height", "width", "num_frames") if result.shape.get(key)
    )
    print(f"model      {result.model_id}  ({result.model_name})")
    print(f"hardware   {hardware.describe()}")
    print(
        f"workload   {shape}, {result.steps} steps, "
        f"{result.sequence.image} image + {result.sequence.text} text tokens"
    )
    print(f"objective  {result.objective}")
    print(f"calibrated {_calibration_note(result)}")
    if not hardware.source.startswith("neuron-ls"):
        print(
            "           note: the Neuron driver could not be queried; the core count is "
            "a per-platform assumption. Pass --total-cores to plan for a specific host."
        )
    if hardware.busy_cores:
        busy = ",".join(str(core) for core in hardware.busy_cores)
        print(f"           note: cores {busy} are already held by another process")
    print()

    if result.ranked:
        width = max(len(entry.label) for entry in result.ranked)
        print(
            f"{'#':>2}  {'config':<{width}}  {'step':>8}  {'request':>9}  {'req/s':>7}  "
            f"{'weights':>9}  {'evidence':<22}  flags"
        )
        for index, entry in enumerate(result.ranked, start=1):
            marker = " *cached" if entry.cached else ""
            weights = f"{entry.weight_bytes / 1e9:.0f}GB" if entry.weight_bytes else "-"
            if entry.weights_over_budget:
                weights += "!"
            print(
                f"{index:>2}  {entry.label:<{width}}  {entry.step_seconds:>7.3f}s  "
                f"{entry.request_seconds:>8.1f}s  {entry.throughput:>7.3f}  "
                f"{weights:>9}  {entry.prediction.evidence + marker:<22}  "
                f"{flags_for(entry.parallel)}"
            )
        print()
        print("  *cached = already compiled on this host; switching costs a full AOT recompile")
        if any(entry.weights_over_budget for entry in result.ranked):
            print(
                "  weights! = upper-bound device HBM for weights exceeds the budget. "
                "ADVISORY ONLY -- cp/cfg/dp each replicate the tp-sharded copy, but "
                "staged models never hold every component at once, so this over-counts. "
                "Not yet validated against measured peak HBM."
            )
        _print_placements(result.ranked)
    else:
        print("no feasible configuration -- every candidate was rejected below.")

    rejected = result.feasibility.rejected
    if rejected:
        print()
        width = max(len(r.label) for r in rejected)
        print(f"{len(rejected)} rejected:")
        for rejection in rejected:
            print(f"  {rejection.label:<{width}}  [{rejection.kind}] {rejection.reason}")


def _print_placements(ranked) -> None:
    """Stage-2 detail block: where each survivor's collectives physically land.

    Top five entries only -- the block is supporting detail for the ranking,
    not a second table; ``--json`` carries every entry's placement.
    """

    with_placement = [entry for entry in ranked if entry.placement is not None][:5]
    if not with_placement:
        return
    print()
    print("placements (stage 2, topology-aware, top 5):")
    for entry in with_placement:
        for line in placement_lines(entry):
            prefix = f"  [{entry.label}] "
            print(prefix + line if not line.startswith("  ") else prefix + line[2:])


def _calibration_note(result: Plan) -> str:
    calibration = result.calibration
    if calibration.kind == "measured-fit":
        anchors = ", ".join(calibration.anchor_labels)
        return (
            f"fitted to {len(calibration.anchor_labels)} measurements ({anchors}); "
            "both compute and bandwidth are measured"
        )
    if calibration.kind == "measured-anchor":
        anchors = ", ".join(calibration.anchor_labels)
        return (
            f"anchored on 1 measurement ({anchors}) with an assumed collective bandwidth. "
            "A second measured config at this shape would fit the bandwidth too"
        )
    return (
        "UNCALIBRATED -- no measurement for this model at this shape on this host, so "
        "times are order-of-magnitude only; trust the ordering, not the numbers"
    )
