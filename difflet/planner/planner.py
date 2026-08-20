"""Rank the feasible configurations for a (host, model, shape, objective).

Enumeration (``feasibility``) says what is legal; this module says which of those
to pick. Three inputs decide it: the analytic cost model, any real measurement
that matches, and whether a candidate is already compiled -- the last mattering
because switching configurations means a fresh AOT compile, which for Flux is
about 25 minutes of wall clock before a single image appears.

Since the AoiZora port the ranking is two-stage (arXiv 2606.17566, §4.5):

- **Stage 1** scores *every* feasible candidate with the placement-oblivious
  cost model and cuts the field to a survivor set (``survivors``; the default
  keeps all of them because Difflet's search space is a handful of configs,
  not the thousands that motivated the paper's pruning).
- **Stage 2** (``difflet.planner.topology``) enumerates each survivor's
  physical placements over the Trainium core/chip hierarchy and rescores the
  communication terms topology-aware: which axis's traffic lands on
  intra-chip links, which crosses the chip-to-chip NeuronLink, and which
  concurrent groups contend on a shared link.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.planner import cost_model
from difflet.planner.cost_model import Calibration, Prediction
from difflet.planner.feasibility import (
    Candidate,
    FeasibilityReport,
    enumerate_candidates,
)
from difflet.planner.hardware import HardwareProfile, detect_hardware
from difflet.planner.measurements import MeasurementStore, load_store
from difflet.planner.model_profile import (
    ModelProfile,
    SequenceLengths,
    device_weight_bytes,
    load_profile,
)
from difflet.planner.topology import PlacementChoice, choose_placement

OBJECTIVES = ("latency", "throughput", "balanced")


@dataclass(frozen=True)
class RankedConfig:
    candidate: Candidate
    prediction: Prediction
    step_seconds: float
    # Seconds of denoise per completed request, ignoring replica count.
    request_seconds: float
    # Requests per second with all dp replicas busy.
    throughput: float
    score: float
    cached: bool
    # Upper-bound weight bytes resident device-wide. Advisory: see
    # model_profile.WeightFootprint for why this does not gate feasibility.
    weight_bytes: int = 0
    weights_over_budget: bool = False
    # Stage-2 result: every distinct physical placement of this candidate's
    # mesh, ranked. None when topology ranking is off, the candidate did not
    # survive stage 1, or the config admits only one placement.
    placement: PlacementChoice | None = None
    # The step time stage 2 ranked this candidate by -- the paper's Q2 at the
    # chosen placement. Equal to `step_seconds` whenever stage 2 ran and the
    # entry was predicted rather than measured.
    stage2_step_seconds: float | None = None

    @property
    def label(self) -> str:
        return self.candidate.label

    @property
    def parallel(self) -> DiffletParallelConfig:
        return self.candidate.parallel


@dataclass(frozen=True)
class Plan:
    model_id: str
    model_name: str
    hardware: HardwareProfile
    objective: str
    steps: int
    shape: dict[str, int | None]
    sequence: SequenceLengths
    calibration: Calibration
    ranked: tuple[RankedConfig, ...]
    feasibility: FeasibilityReport
    # Labels that reached stage 2. Empty when topology ranking is off.
    survivors: tuple[str, ...] = ()

    @property
    def best(self) -> RankedConfig | None:
        return self.ranked[0] if self.ranked else None

    @property
    def evidence_summary(self) -> str:
        if not self.ranked:
            return "no candidates"
        measured = sum(1 for entry in self.ranked if entry.prediction.evidence == "measured")
        return f"{measured} measured, {len(self.ranked) - measured} predicted"


DEFAULT_STEPS = 20

# Share of device HBM assumed available for weights.
WEIGHT_BUDGET_FRACTION = 0.85


def plan(
    model_id: str,
    *,
    model_type: str | None = None,
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
    steps: int | None = None,
    objective: str = "latency",
    hardware: HardwareProfile | None = None,
    total_cores: int | None = None,
    serving: bool = False,
    store: MeasurementStore | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    survivors: int | None = None,
    topology_aware: bool = True,
    rank_by_best_placement: bool = False,
) -> Plan:
    """Two-stage ranking: analytic prune, then topology-aware placement.

    ``survivors`` is the stage-1 cut -- how many candidates stage 2 scores.
    ``None`` keeps all of them: AoiZora needs aggressive pruning because its
    stage 2 compiles every survivor, while this stage 2 is analytic and costs
    microseconds, so the default keeps the machinery exercised on the whole
    (tiny) field.

    ``rank_by_best_placement`` selects what stage 2's ranking means. The
    default (False) ranks by the Q2 at the *default* axis order -- the only
    placement the runtime can execute today -- and reports better orders as
    advisory deltas. True ranks by the best placement outright, which is the
    paper's semantics and the right switch once the runtime grows a mesh-order
    flag.
    """

    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")

    hardware = hardware or detect_hardware(allocated_cores_override=total_cores)
    profile = load_profile(model_id, model_type=model_type)
    shape = profile.resolve_shape(height=height, width=width, num_frames=num_frames)
    sequence = profile.sequence_lengths(height=height, width=width, num_frames=num_frames)
    resolved_steps = steps or DEFAULT_STEPS

    # Headroom for activations, KV and the runtime's own allocations. For
    # models whose whole footprint is co-resident (not staged) this budget is
    # also a hard feasibility rule: the dp*cfg*cp replicated copies are a
    # physical upper bound on device HBM (D35). Staged models keep the
    # budget advisory-only (D22).
    weight_budget = int(
        hardware.num_devices * hardware.hbm_bytes_per_device * WEIGHT_BUDGET_FRACTION
    )
    feasibility = enumerate_candidates(
        model_name=profile.name,
        capabilities=profile.capabilities,
        cores=hardware.allocated_cores,
        serving=serving,
        weight_bytes=profile.weights.total_bytes,
        weight_budget=weight_budget,
        staged=profile.weights.staged,
    )

    store = store if store is not None else load_store()
    shape_key = (shape.get("height"), shape.get("width"), shape.get("num_frames"))
    by_label = {candidate.label: candidate.parallel for candidate in feasibility.feasible}
    calibration = cost_model.calibrate(
        store.anchors(
            model=profile.name,
            instance_type=hardware.instance_type,
            shape=shape_key,
            model_id=model_id,
        ),
        profile=profile,
        seq=sequence,
        parallel_of=by_label.get,
    )

    # Surviving candidates keep the budget as an advisory flag, never a drop.
    compiled = _compiled_labels(cache_dir, profile.name)
    stage1 = [
        _rank(
            candidate,
            profile=profile,
            sequence=sequence,
            calibration=calibration,
            store=store,
            hardware=hardware,
            shape_key=shape_key,
            steps=resolved_steps,
            compiled=compiled,
            model_id=model_id,
            weight_budget=weight_budget,
        )
        for candidate in feasibility.feasible
    ]

    # ------------------------------------------------------------- stage 2
    survivor_labels: tuple[str, ...] = ()
    if topology_aware and stage1:
        ordered = _scored(stage1, objective)  # stage-1 order decides the cut
        cut = ordered[:survivors] if survivors is not None else ordered
        survivor_labels = tuple(entry.label for entry in cut)
        by_label_stage1 = {entry.label: entry for entry in stage1}
        stage2 = {
            label: _stage2(
                by_label_stage1[label],
                profile=profile,
                sequence=sequence,
                calibration=calibration,
                hardware=hardware,
                steps=resolved_steps,
                rank_by_best_placement=rank_by_best_placement,
            )
            for label in survivor_labels
        }
        merged = [stage2.get(entry.label, entry) for entry in stage1]
    else:
        merged = stage1

    return Plan(
        model_id=model_id,
        model_name=profile.name,
        hardware=hardware,
        objective=objective,
        steps=resolved_steps,
        shape=shape,
        sequence=sequence,
        calibration=calibration,
        ranked=tuple(_scored(merged, objective)),
        feasibility=feasibility,
        survivors=survivor_labels,
    )


def _rank(
    candidate: Candidate,
    *,
    profile: ModelProfile,
    sequence: SequenceLengths,
    calibration: Calibration,
    store: MeasurementStore,
    hardware: HardwareProfile,
    shape_key: tuple,
    steps: int,
    compiled: frozenset[str],
    model_id: str,
    weight_budget: int,
) -> RankedConfig:
    measurement = store.lookup(
        model=profile.name,
        label=candidate.label,
        instance_type=hardware.instance_type,
        shape=shape_key,
        steps=steps,
        model_id=model_id,
    )
    prediction = cost_model.predict(
        candidate.parallel,
        profile=profile,
        seq=sequence,
        calibration=calibration,
        measurement=measurement,
    )
    request_seconds = prediction.step_seconds * steps
    throughput = candidate.parallel.dp_degree / request_seconds if request_seconds > 0 else 0.0
    weights = device_weight_bytes(profile, candidate.parallel)
    return RankedConfig(
        candidate=candidate,
        prediction=prediction,
        step_seconds=prediction.step_seconds,
        request_seconds=request_seconds,
        throughput=throughput,
        score=0.0,  # assigned by _scored, which needs the whole set to normalize
        cached=candidate.label in compiled,
        weight_bytes=weights,
        weights_over_budget=bool(weight_budget and weights > weight_budget),
    )


def _stage2(
    entry: RankedConfig,
    *,
    profile: ModelProfile,
    sequence: SequenceLengths,
    calibration: Calibration,
    hardware: HardwareProfile,
    steps: int,
    rank_by_best_placement: bool,
) -> RankedConfig:
    """Topology-aware rescoring of one survivor.

    Measured entries keep their measured step time: a measurement was taken at
    the default axis order, and scaling it by the predicted ratio for another
    order would launder an assumption into a number the user reads as ground
    truth. Predicted entries adopt the stage-2 Q2, which at the default
    placement differs from stage 1 only by the paper's overlap structure
    (communication that hides under compute stops being charged serially).
    """

    from difflet.planner import topology as topology_module

    parallel = entry.parallel
    compute_seconds = calibration.compute_seconds_single_core * cost_model.compute_share(parallel)
    comm_by_axis = topology_module.comm_seconds_by_axis(
        parallel,
        profile=profile,
        seq=sequence,
        bandwidth_bytes_per_second=calibration.bandwidth_bytes_per_second,
    )
    choice = choose_placement(
        parallel,
        topology=topology_module.PhysicalTopology.from_hardware(hardware),
        profile=profile,
        seq=sequence,
        compute_seconds=compute_seconds,
        comm_seconds_by_axis=comm_by_axis,
    )
    if choice is None:
        return entry

    if entry.prediction.evidence == "measured":
        stage2_seconds = entry.step_seconds  # measured, at the default order
    else:
        picked = choice.best if rank_by_best_placement else choice.default
        stage2_seconds = picked.step_seconds
    request_seconds = stage2_seconds * steps
    return replace(
        entry,
        step_seconds=stage2_seconds,
        request_seconds=request_seconds,
        throughput=parallel.dp_degree / request_seconds if request_seconds > 0 else 0.0,
        placement=choice,
        stage2_step_seconds=stage2_seconds,
    )


def _scored(ranked: list[RankedConfig], objective: str) -> list[RankedConfig]:
    """Turn latency and throughput into one comparable score per objective.

    ``latency`` and ``throughput`` each optimize a single quantity, so the score
    is that quantity normalized against the best candidate -- 1.0 for the winner.
    ``balanced`` is their geometric mean, which refuses to trade one to zero for
    the other the way an arithmetic mean would.
    """

    if not ranked:
        return []
    best_latency = min(entry.request_seconds for entry in ranked if entry.request_seconds > 0)
    best_throughput = max(entry.throughput for entry in ranked)

    out: list[RankedConfig] = []
    for entry in ranked:
        latency_score = best_latency / entry.request_seconds if entry.request_seconds else 0.0
        throughput_score = entry.throughput / best_throughput if best_throughput else 0.0
        if objective == "latency":
            score = latency_score
        elif objective == "throughput":
            score = throughput_score
        else:
            score = (latency_score * throughput_score) ** 0.5
        out.append(replace(entry, score=score))
    out.sort(key=lambda entry: (-entry.score, entry.label))
    return out


def _compiled_labels(cache_dir, model_name: str) -> frozenset[str]:
    """Which candidates already have a compiled artifact on this host.

    Reads each manifest's recorded ``parallel`` block rather than recomputing
    cache keys, so it stays correct across the additive elisions in
    ``DiffletParallelConfig.to_cache_dict`` (cp_mode at gather_kv, sp when off,
    dp at 1 are all omitted). A partial entry -- a compile that died before
    writing its manifest -- reads as not compiled, which is what it is.
    """

    from difflet.planner.feasibility import config_label

    root = Path(cache_dir) if cache_dir else _default_cache_dir()
    model_root = root / model_name
    if not model_root.is_dir():
        return frozenset()

    labels: set[str] = set()
    for manifest_path in sorted(model_root.glob("*/manifest.json")):
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        parallel = (manifest.get("cache_inputs") or {}).get("parallel")
        if not isinstance(parallel, dict):
            continue
        try:
            config = DiffletParallelConfig(
                tp_degree=int(parallel.get("tp_degree", 1)),
                cp_degree=int(parallel.get("cp_degree", 1)),
                cp_mode=str(parallel.get("cp_mode", "gather_kv")),
                cfg_parallel_enabled=bool(parallel.get("cfg_parallel_enabled", False)),
                sp_enabled=bool(parallel.get("sp_enabled", False)),
                dp_degree=int(parallel.get("dp_degree", 1)),
            )
        except (TypeError, ValueError):
            continue
        labels.add(config_label(config))
    return frozenset(labels)


def _default_cache_dir() -> Path:
    from difflet.pipeline.compile_cache import _default_cache_dir as difflet_cache_dir

    return difflet_cache_dir()
