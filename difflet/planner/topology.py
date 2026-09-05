"""Topology-aware placement ranking -- the AoiZora stage-2 port.

AoiZora (arXiv 2606.17566) makes one observation this module ports: a logical
sharding says *which ranks* talk, not *which physical links* that traffic lands
on. The same ``tp=2 cp=2`` puts its tensor-parallel all-reduce groups on two
cores of one chip, or splits them across the chip-to-chip link, depending
purely on the order the mesh axes are laid onto core IDs. On a topology where
links are not interchangeable, that changes the cost of an otherwise identical
program.

Their two-stage design maps onto this planner as:

- **Stage 1 (pre-compilation pruning)** stays what ``cost_model`` already is: a
  placement-oblivious, calibration-anchored score over *every* feasible
  candidate, used to cut the field to a survivor set.
- **Stage 2 (this module)** takes each survivor, enumerates the distinct
  physical placements of its mesh, and ranks them with a topology-aware
  communication model over the Trainium core/chip/device hierarchy.

Where AoiZora parses compiled HLO to recover the concrete collective graph,
this prototype derives it analytically. That substitution is honest here for
two reasons: every collective a Difflet configuration issues is statically
determined by the config (there is no compiler rewrite between plan and run),
and the AoiZora alternative -- compile each survivor -- costs ~25 minutes per
candidate on this stack, which is exactly the cost the planner exists to avoid.
The analytic schedule is kept consistent with ``cost_model.comm_bytes`` by test,
so the two views cannot drift.

Scoring follows the paper's engine model rather than a flat sum:

    Q2 = C_comp + barrier_comm + max(0, overlappable_comm - C_comp)

Compute and overlappable communication run on separate engines and hide under
one another; the tensor-parallel all-reduce is the one collective pipelining
cannot hide (each block's residual add consumes the full reduced output, a
hard barrier on the loop-carried recurrence). Contention enters as a
physical-link term: collectives whose concurrent groups share a link serialize
on it, which is the paper's shared-link-contention mechanism rather than a
hop-count penalty.

Everything here is a *rank objective*, not a latency prediction: absolute
seconds are anchored to the stage-1 calibration at the default placement, and
only ratios between placements rest on the topology constants below.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import NamedTuple

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.pipeline.parallel_mesh import MeshSpec
from difflet.planner.cost_model import BF16_BYTES, RING_OVERLAP_RETENTION
from difflet.planner.hardware import HardwareProfile
from difflet.planner.model_profile import ModelProfile, SequenceLengths

# Trainium2 public geometry: one chip exposes two NeuronCores, and a
# NeuronDevice is two chips. neuron-ls reports the flat core list only, so the
# chip boundary is reconstructed from these constants -- core pairs (0,1) and
# (2,3) land on separate chips of one device on a trn2.3xlarge.
CORES_PER_CHIP = 2

# Effective bytes/second a collective achieves per physical tier. These are
# ORDERING ASSUMPTIONS, not measurements -- only their ratios decide which
# placement wins, and a calibration hook can replace them once a multi-tier
# measurement exists. The defaults keep the intra-chip : intra-device :
# inter-device ratio at 4 : 2 : 1, matching the shape of Trainium2's fabric
# (on-chip, then NeuronLink within a device, then NeuronLink across devices).
TIER_BANDWIDTH_BYTES_PER_SECOND: dict[str, float] = {
    "intra-chip": 400e9,
    "intra-device": 200e9,
    "inter-device": 100e9,
}

# The canonical mesh-axis order the runtime lays onto core IDs today
# (``MeshSpec``: tp innermost, dp outermost). Placements are scored relative to
# it because it is the only order a compiled artifact has ever run.
DEFAULT_AXIS_ORDER: tuple[str, ...] = ("tp", "cp", "cfg", "dp")


class CoreCoord(NamedTuple):
    """Physical coordinates of one logical NeuronCore."""

    device: int
    chip: int
    core: int  # within the chip


@dataclass(frozen=True)
class Link:
    """A physical interconnect segment between two cores.

    Intra-chip pairs get a chip-local link; cross-chip pairs a device-local
    NeuronLink; cross-device pairs an inter-device NeuronLink. Torus
    dimensions on multi-device hosts are deliberately flattened to one tier --
    ``neuron-ls`` does not expose them, and pretending to model a routing we
    cannot see would be false precision.
    """

    tier: str
    a: CoreCoord
    b: CoreCoord

    @property
    def bandwidth(self) -> float:
        return TIER_BANDWIDTH_BYTES_PER_SECOND[self.tier]


class PhysicalTopology:
    """The core/chip/device hierarchy a mesh placement lands on."""

    def __init__(self, cores: int, *, cores_per_device: int):
        if cores < 1:
            raise ValueError(f"cores must be >= 1, got {cores}")
        self.cores = cores
        self.cores_per_device = cores_per_device
        self._coords = [self.coord_of(core) for core in range(cores)]

    @classmethod
    def from_hardware(cls, hardware: HardwareProfile) -> "PhysicalTopology":
        return cls(hardware.allocated_cores, cores_per_device=hardware.cores_per_device)

    def coord_of(self, core_id: int) -> CoreCoord:
        if not 0 <= core_id < self.cores:
            raise ValueError(f"core {core_id} outside [0, {self.cores})")
        within_device = core_id % self.cores_per_device
        return CoreCoord(
            device=core_id // self.cores_per_device,
            chip=within_device // CORES_PER_CHIP,
            core=within_device % CORES_PER_CHIP,
        )

    def link_between(self, a: int, b: int) -> Link:
        if a == b:
            raise ValueError("a link needs two distinct cores")
        ca, cb = self._coords[a], self._coords[b]
        if (ca.device, ca.chip) == (cb.device, cb.chip):
            tier = "intra-chip"
        elif ca.device == cb.device:
            tier = "intra-device"
        else:
            tier = "inter-device"
        return Link(tier, ca, cb)

    def link_key(self, a: int, b: int) -> tuple:
        """Identity of the physical segment two cores communicate over.

        Two core pairs that sit on the same pair of chips share one physical
        interconnect -- cores {0,2} and {1,3} both cross the single
        chip-to-chip NeuronLink of a 4-core device -- so contention is counted
        per chip pair, not per core pair. Intra-chip traffic is identified by
        the chip itself: both core pairs of one chip share its fabric.
        """

        ca, cb = self._coords[a], self._coords[b]
        chip_a, chip_b = (ca.device, ca.chip), (cb.device, cb.chip)
        if chip_a == chip_b:
            return ("intra-chip", chip_a)
        return (self.link_between(a, b).tier, min(chip_a, chip_b), max(chip_a, chip_b))


@dataclass(frozen=True)
class Placement:
    """One way of laying the mesh axes onto the physical core list.

    ``order`` lists the active axes innermost-first: the first axis varies
    fastest in the rank-to-core mapping, so its groups occupy adjacent core
    IDs. The runtime's fixed layout (:data:`DEFAULT_AXIS_ORDER`) is the
    placement a compiled artifact actually realizes; every other order is a
    recommendation the runtime cannot execute yet, which the caller is
    expected to surface.
    """

    order: tuple[str, ...]
    mesh: MeshSpec

    @property
    def is_default(self) -> bool:
        return self.order == tuple(
            axis for axis in DEFAULT_AXIS_ORDER if self.mesh.axis_size(axis) > 1
        )

    def core_of_rank(self, rank: int) -> int:
        """Map a logical rank to a core ID by mixed-radix decomposition.

        Innermost axis = stride 1, mirroring how ``MeshSpec.rank_of`` composes
        ranks but with the axis roles permuted by this placement's order.
        """

        if not 0 <= rank < self.mesh.world_size:
            raise ValueError(f"rank {rank} outside [0, {self.mesh.world_size})")
        core, scale = 0, 1
        for axis in self.order:
            size = self.mesh.axis_size(axis)
            core += (rank % size) * scale
            rank //= size
            scale *= size
        return core

    def group_cores(self, axis: str) -> list[list[int]]:
        """Physical core IDs of each axis group, ordered by axis coordinate."""

        size = self.mesh.axis_size(axis)
        strides: dict[str, int] = {}
        scale = 1
        for active in self.order:
            strides[active] = scale
            scale *= self.mesh.axis_size(active)
        stride = strides.get(axis, 0)
        if stride == 0:
            raise ValueError(f"axis {axis!r} is not active in placement {self.order}")

        groups: dict[int, list[int]] = {}
        for rank in range(self.mesh.world_size):
            coord = (rank // stride) % size
            key = rank - coord * stride  # this axis zeroed out
            groups.setdefault(key, [0] * size)[coord] = self.core_of_rank(rank)
        return [group for _, group in sorted(groups.items())]


def enumerate_placements(mesh: MeshSpec, *, cores_per_device: int) -> tuple[Placement, ...]:
    """Distinct axis orders of the active axes, deduplicated by physical effect.

    Two orders that give every axis the same set of physical groups are the
    same placement -- AoiZora's symmetry dedup -- so only orders that move some
    axis's groups across a chip or device boundary survive. The default order
    is returned first so callers can present it as the status quo.
    """

    active = tuple(axis for axis in DEFAULT_AXIS_ORDER if mesh.axis_size(axis) > 1)
    seen: set[tuple] = set()
    placements: list[Placement] = []
    for order in itertools.permutations(active):
        placement = Placement(order=order, mesh=mesh)
        signature = placement_signature(placement, cores_per_device=cores_per_device)
        if signature in seen:
            continue
        seen.add(signature)
        placements.append(placement)
    placements.sort(key=lambda p: not p.is_default)  # default first, rest stable
    return tuple(placements)


def placement_signature(placement: Placement, *, cores_per_device: int) -> tuple:
    """The physical fingerprint of a placement, for dedup.

    Per active axis: the multiset of its groups' chip-membership. Orders that
    agree on this agree on every quantity the cost model can see.
    """

    def chip_of(core: int) -> tuple[int, int]:
        return (core // cores_per_device, (core % cores_per_device) // CORES_PER_CHIP)

    parts = []
    for axis in placement.order:
        groups = tuple(
            tuple(sorted(chip_of(core) for core in cores)) for cores in placement.group_cores(axis)
        )
        parts.append((axis, tuple(sorted(groups))))
    return tuple(parts)


# --------------------------------------------------------------------- schedule


@dataclass(frozen=True)
class Collective:
    """One collective family a configuration issues once per denoise step.

    ``per_rank_bytes`` is what ONE rank moves per occurrence -- the same
    accounting ``cost_model.comm_bytes`` uses, so the schedule's total matches
    the flat model's total by construction (pinned by test). ``barrier`` marks
    the collectives pipelining cannot hide behind compute: the TP all-reduce
    feeding a residual add, and the end-of-step CFG gather. Everything else
    (KV gathers, ring rotations, ulysses all-to-alls) is overlappable in the
    paper's engine model.
    """

    family: str  # all_reduce | all_gather | all_to_all
    axis: str  # mesh axis the group spans
    per_rank_bytes: float
    occurrences: int  # per denoise step
    barrier: bool
    note: str = ""


def collective_schedule(
    parallel: DiffletParallelConfig,
    *,
    profile: ModelProfile,
    seq: SequenceLengths,
    latent_channels: int = 64,
) -> tuple[Collective, ...]:
    """The concrete collectives one denoise step issues, derived from the config.

    Mirrors ``cost_model.comm_bytes`` term for term: two TP all-reduces per
    block over ``[B, S/cp, hidden]``; per-attention CP traffic by mode (KV
    all-gathers, ring rotations, or four ulysses all-to-alls); one CFG output
    gather per step. SP swaps the all-reduce for a reduce-scatter +
    all-gather pair of the same total volume.
    """

    dims = profile.dims
    tp, cp = parallel.tp_degree, parallel.cp_degree
    tokens_per_rank = seq.joint / cp
    out: list[Collective] = []

    if tp > 1:
        activation = tokens_per_rank * dims.hidden_size * BF16_BYTES
        if parallel.sp_enabled:
            out.append(
                Collective(
                    family="reduce_scatter",
                    axis="tp",
                    per_rank_bytes=(tp - 1) / tp * activation,
                    occurrences=dims.total_blocks * 2,
                    barrier=True,
                    note="SP splits each all-reduce into RS + AG",
                )
            )
            out.append(
                Collective(
                    family="all_gather",
                    axis="tp",
                    per_rank_bytes=(tp - 1) / tp * activation,
                    occurrences=dims.total_blocks * 2,
                    barrier=True,
                    note="SP pair",
                )
            )
        else:
            out.append(
                Collective(
                    family="all_reduce",
                    axis="tp",
                    per_rank_bytes=2.0 * (tp - 1) / tp * activation,
                    occurrences=dims.total_blocks * 2,
                    barrier=True,
                    note="attn-out and MLP-down, per block",
                )
            )

    if cp > 1:
        heads_per_rank = dims.num_attention_heads / tp
        local_kv = tokens_per_rank * heads_per_rank * dims.attention_head_dim * BF16_BYTES
        if parallel.cp_mode == "ulysses":
            out.append(
                Collective(
                    family="all_to_all",
                    axis="cp",
                    per_rank_bytes=(cp - 1) / cp * local_kv,
                    occurrences=dims.total_blocks * 4,
                    barrier=False,
                    note="q,k,v out and the inverse on the attention output",
                )
            )
        else:
            note = (
                "K,V shards rotate around the cp ring"
                if parallel.cp_mode == "ring"
                else "gather full K and V"
            )
            out.append(
                Collective(
                    family="all_gather",
                    axis="cp",
                    per_rank_bytes=(cp - 1) * local_kv,
                    occurrences=dims.total_blocks * 2,
                    barrier=False,
                    note=note,
                )
            )

    if parallel.cfg_parallel_enabled:
        out.append(
            Collective(
                family="all_gather",
                axis="cfg",
                per_rank_bytes=seq.image * latent_channels * BF16_BYTES,
                occurrences=1,
                barrier=True,
                note="combine the output latent",
            )
        )

    return tuple(out)


# --------------------------------------------------------------------- stage 2


@dataclass(frozen=True)
class FamilyScore:
    """Physical cost of one collective family under one placement."""

    collective: Collective
    # Multiplier vs the default placement: 1.0 on the default order, <1.0 when
    # this placement gives the family faster (or less contended) links.
    multiplier: float
    bottleneck_tier: str = ""
    # Extra cost this family pays because concurrent groups share a link.
    contention_seconds: float = 0.0


@dataclass(frozen=True)
class PlacementScore:
    """The stage-2 rank objective for one (candidate, placement) pair.

    ``step_seconds`` composes the calibrated compute term with physically
    rescaled communication under the paper's overlap structure; at the default
    placement every multiplier is 1.0, so it reduces exactly to the stage-1
    prediction and the two stages can be compared on one axis.
    """

    placement: Placement
    step_seconds: float
    compute_seconds: float
    barrier_seconds: float
    overlappable_seconds: float
    families: tuple[FamilyScore, ...] = field(default_factory=tuple)

    @property
    def order_text(self) -> str:
        return ".".join(self.placement.order)


def score_placement(
    *,
    placement: Placement,
    topology: PhysicalTopology,
    schedule: tuple[Collective, ...],
    compute_seconds: float,
    comm_seconds_by_axis: dict[str, float],
) -> PlacementScore:
    """Rank one placement of one candidate.

    ``comm_seconds_by_axis`` supplies the calibrated logical cost of each
    axis's traffic (stage 1's split of ``comm_bytes``). The topology model
    produces per-family *multipliers* -- physical seconds under this placement
    divided by physical seconds under the default placement -- so absolute
    numbers stay anchored to the calibration and only the ratios between
    placements rest on the tier constants.
    """

    default = Placement(
        order=tuple(axis for axis in DEFAULT_AXIS_ORDER if placement.mesh.axis_size(axis) > 1),
        mesh=placement.mesh,
    )

    # Split each axis's calibrated seconds over its collectives by byte share,
    # so the SP reduce-scatter + all-gather pair together cost exactly what
    # the all-reduce they replaced costs.
    axis_bytes: dict[str, float] = {}
    for collective in schedule:
        volume = collective.per_rank_bytes * collective.occurrences
        axis_bytes[collective.axis] = axis_bytes.get(collective.axis, 0.0) + volume

    barrier = 0.0
    overlappable = 0.0
    families: list[FamilyScore] = []

    for collective in schedule:
        volume = collective.per_rank_bytes * collective.occurrences
        share = volume / axis_bytes[collective.axis] if axis_bytes[collective.axis] else 0.0
        calibrated = comm_seconds_by_axis.get(collective.axis, 0.0) * share

        shared = _family_seconds(collective, placement, topology, share_links=True)
        alone = _family_seconds(collective, placement, topology, share_links=False)
        baseline = _family_seconds(collective, default, topology, share_links=True)
        if baseline <= 0:
            continue
        multiplier = shared / baseline
        cost = calibrated * multiplier
        # The fraction of this family's cost caused by concurrent groups
        # loading the same link -- the paper's R_cont, per family.
        contention = cost * max(0.0, 1.0 - alone / shared) if shared > 0 else 0.0

        families.append(
            FamilyScore(
                collective=collective,
                multiplier=multiplier,
                bottleneck_tier=_bottleneck_tier(collective, placement, topology),
                contention_seconds=contention,
            )
        )
        if collective.barrier:
            barrier += cost
        else:
            overlappable += cost

    # The paper's Q1/Q2 composition: overlappable communication hides under
    # compute; barrier communication cannot.
    hidden = max(0.0, overlappable - compute_seconds)
    return PlacementScore(
        placement=placement,
        step_seconds=compute_seconds + barrier + hidden,
        compute_seconds=compute_seconds,
        barrier_seconds=barrier,
        overlappable_seconds=overlappable,
        families=tuple(families),
    )


def _ring_links(cores: list[int]) -> list[tuple[int, int]]:
    """The undirected hops a ring collective over this group pays.

    Deduplicated: a 2-member ring's there-and-back hops are two directions of
    one exchange over one segment, and charging both against the segment's
    bandwidth would double-count a full-duplex link.
    """

    ordered = sorted(cores)
    hops = {frozenset((a, b)) for a, b in zip(ordered, ordered[1:] + ordered[:1])}
    return [tuple(hop) for hop in hops]


def _family_seconds(
    collective: Collective,
    placement: Placement,
    topology: PhysicalTopology,
    *,
    share_links: bool,
) -> float:
    """Seconds the family's concurrent groups need, as a max over link loads.

    Every byte a rank receives in a ring collective crosses its adjacent ring
    link, so each distinct ring link carries exactly ``per_rank_bytes`` per
    group -- the same volume the flat model charges a rank, now placed on a
    concrete tier. Groups of one axis execute simultaneously (each rank is in
    exactly one group), so with ``share_links=True`` their loads sum on any
    link they share and serialize there; with ``share_links=False`` every group
    is costed alone, which isolates contention for the breakdown. All-to-all
    traffic is approximated by the same rule -- a rank-objective stand-in for
    the per-pair routes, which the placement cannot reorder anyway.
    """

    groups = placement.group_cores(collective.axis)
    if not groups or collective.per_rank_bytes <= 0:
        return 0.0

    if share_links:
        loads: dict[tuple, float] = {}
        bandwidths: dict[tuple, float] = {}
        for cores in groups:
            for a, b in _ring_links(cores):
                key = topology.link_key(a, b)
                loads[key] = loads.get(key, 0.0) + collective.per_rank_bytes
                bandwidths[key] = topology.link_between(a, b).bandwidth
        return max(loads[key] / bandwidths[key] for key in loads)

    # Each group alone: its own hops still sum on a shared link (a ring may
    # cross the same physical segment twice), but nothing from other groups.
    worst = 0.0
    for cores in groups:
        group_loads: dict[tuple, float] = {}
        group_bandwidths: dict[tuple, float] = {}
        for a, b in _ring_links(cores):
            key = topology.link_key(a, b)
            group_loads[key] = group_loads.get(key, 0.0) + collective.per_rank_bytes
            group_bandwidths[key] = topology.link_between(a, b).bandwidth
        worst = max(
            worst,
            max(group_loads[key] / group_bandwidths[key] for key in group_loads),
        )
    return worst


def _bottleneck_tier(
    collective: Collective, placement: Placement, topology: PhysicalTopology
) -> str:
    """The slowest tier any of the family's groups crosses."""

    tiers: set[str] = set()
    for cores in placement.group_cores(collective.axis):
        for a, b in _ring_links(cores):
            tiers.add(topology.link_between(a, b).tier)
    if not tiers:
        return ""
    return min(tiers, key=lambda tier: TIER_BANDWIDTH_BYTES_PER_SECOND[tier])


@dataclass(frozen=True)
class PlacementChoice:
    """Stage-2 outcome for one survivor: the winning placement and the field."""

    best: PlacementScore
    default: PlacementScore
    alternates: tuple[PlacementScore, ...]

    @property
    def default_is_best(self) -> bool:
        return self.best.placement.is_default


def choose_placement(
    parallel: DiffletParallelConfig,
    *,
    topology: PhysicalTopology,
    profile: ModelProfile,
    seq: SequenceLengths,
    compute_seconds: float,
    comm_seconds_by_axis: dict[str, float],
) -> PlacementChoice | None:
    """Run the stage-2 search for one candidate: placements in, ranking out.

    Returns ``None`` only when the config issues no collectives at all (pure
    data parallelism) -- there is nothing to place. A config with a single
    distinct placement still gets a choice whose ``best`` and ``default`` are
    the same score: the Q2 composition (overlap-aware comm rescaling) applies
    to it exactly as to multi-placement configs, so a one-axis candidate and a
    two-axis candidate are ranked by the same objective.
    """

    schedule = collective_schedule(parallel, profile=profile, seq=seq)
    if not schedule:
        return None

    placements = enumerate_placements(
        parallel.mesh_spec, cores_per_device=topology.cores_per_device
    )
    scored = tuple(
        score_placement(
            placement=placement,
            topology=topology,
            schedule=schedule,
            compute_seconds=compute_seconds,
            comm_seconds_by_axis=comm_seconds_by_axis,
        )
        for placement in placements
    )
    ranked = sorted(scored, key=lambda score: score.step_seconds)
    default = next((score for score in scored if score.placement.is_default), ranked[0])
    return PlacementChoice(
        best=ranked[0],
        default=default,
        alternates=tuple(score for score in ranked if score is not ranked[0]),
    )


def comm_seconds_by_axis(
    parallel: DiffletParallelConfig,
    *,
    profile: ModelProfile,
    seq: SequenceLengths,
    bandwidth_bytes_per_second: float,
) -> dict[str, float]:
    """Split the calibrated communication seconds per mesh axis.

    Stage 2 rescales each axis's traffic independently -- tp and cp traffic
    land on different links under different axis orders -- so the flat
    ``comm_bytes`` total is divided by axis using the same schedule. Ring-CP
    keeps the flat model's overlap discount (``RING_OVERLAP_RETENTION``): the
    bytes move, but only that share sits on the critical path. The discount is
    applied here rather than in the schedule so the schedule stays a pure
    byte-volume view and sums to ``comm_bytes`` except for exactly this
    documented term.
    """

    schedule = collective_schedule(parallel, profile=profile, seq=seq)
    ring = parallel.cp_degree > 1 and parallel.cp_mode == "ring"
    out: dict[str, float] = {}
    for collective in schedule:
        seconds = collective.per_rank_bytes * collective.occurrences / bandwidth_bytes_per_second
        if ring and collective.axis == "cp":
            seconds *= RING_OVERLAP_RETENTION
        out[collective.axis] = out.get(collective.axis, 0.0) + seconds
    return out
