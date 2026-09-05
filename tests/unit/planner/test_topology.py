"""Stage 2 of the AoiZora port: placements, physical embedding, and Q2.

These tests pin three things the rest of the planner leans on:

- the **schedule** is the same accounting as ``cost_model.comm_bytes`` (the
  analytic collective graph must not drift from the flat model it anchors to);
- the **physical model** behaves like the links it stands for -- intra-chip is
  the fast tier, chip-pair links are shared resources, and a strided group
  pays for both the slower tier and the contention of concurrent groups;
- the **Q2 composition** implements the paper's engine model: overlappable
  communication hides under compute, barrier communication never does.
"""

from __future__ import annotations

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.planner import cost_model, topology as topo
from difflet.planner.hardware import HardwareProfile
from difflet.planner.model_profile import load_profile
from difflet.planner.planner import plan
from difflet.planner.topology import (
    PhysicalTopology,
    choose_placement,
    collective_schedule,
    comm_seconds_by_axis,
    enumerate_placements,
)

FLUX = "black-forest-labs/FLUX.1-dev"

TRN2 = HardwareProfile(
    instance_type="trn2.3xlarge",
    platform_target="trn2",
    num_devices=1,
    cores_per_device=4,
    hbm_bytes_per_device=103079215104,
    lnc=2,
    allocated_cores=4,
    source="neuron-ls",
)


def _flux_profile():
    return load_profile(FLUX, model_type="flux")


def _flux_seq():
    return _flux_profile().sequence_lengths(height=1024, width=1024)


FOUR_CORES = PhysicalTopology(4, cores_per_device=4)


# ------------------------------------------------------------------- geometry


def test_trainium2_hierarchy_two_chips_per_device():
    # Public geometry: chip = 2 cores, device = 2 chips. Core pairs (0,1) and
    # (2,3) sit on separate chips of the one device a trn2.3xlarge exposes.
    assert FOUR_CORES.coord_of(0).chip == FOUR_CORES.coord_of(1).chip == 0
    assert FOUR_CORES.coord_of(2).chip == FOUR_CORES.coord_of(3).chip == 1
    assert FOUR_CORES.coord_of(0).device == FOUR_CORES.coord_of(3).device


def test_link_tiers_and_chip_pair_identity():
    assert FOUR_CORES.link_between(0, 1).tier == "intra-chip"
    assert FOUR_CORES.link_between(0, 2).tier == "intra-device"
    # Cores {0,2} and {1,3} cross the same physical chip pair, so their
    # traffic contends on one segment even though the core pairs differ.
    assert FOUR_CORES.link_key(0, 2) == FOUR_CORES.link_key(1, 3)
    # ...while the two intra-chip pairs live on different chips' fabrics.
    assert FOUR_CORES.link_key(0, 1) != FOUR_CORES.link_key(2, 3)


def test_multi_device_topology_uses_inter_device_tier():
    eight = PhysicalTopology(8, cores_per_device=4)
    assert eight.coord_of(5).device == 1
    assert eight.link_between(1, 5).tier == "inter-device"


# ----------------------------------------------------------------- placements


def test_two_active_axes_yield_two_distinct_placements():
    mesh = DiffletParallelConfig(tp_degree=2, cp_degree=2).mesh_spec
    placements = enumerate_placements(mesh, cores_per_device=4)
    assert [p.order for p in placements] == [("tp", "cp"), ("cp", "tp")]
    assert placements[0].is_default


def test_single_active_axis_has_one_placement():
    mesh = DiffletParallelConfig(tp_degree=4).mesh_spec
    placements = enumerate_placements(mesh, cores_per_device=4)
    assert len(placements) == 1
    assert placements[0].is_default


def test_placement_maps_ranks_innermost_axis_contiguous():
    # Innermost axis varies fastest: with tp innermost, tp group 0 is ranks
    # {0,1} -> cores {0,1} (one chip); with cp innermost, cp group 0 is the
    # same cores and tp group 0 becomes the cross-chip pair {0,2}.
    mesh = DiffletParallelConfig(tp_degree=2, cp_degree=2).mesh_spec
    tp_first = topo.Placement(("tp", "cp"), mesh)
    cp_first = topo.Placement(("cp", "tp"), mesh)
    assert tp_first.group_cores("tp") == [[0, 1], [2, 3]]
    assert cp_first.group_cores("tp") == [[0, 2], [1, 3]]
    assert cp_first.group_cores("cp") == [[0, 1], [2, 3]]


# ------------------------------------------------------------------- schedule


@pytest.mark.parametrize(
    "parallel",
    [
        DiffletParallelConfig(tp_degree=4),
        DiffletParallelConfig(tp_degree=4, sp_enabled=True),
        DiffletParallelConfig(tp_degree=2, cp_degree=2),
        DiffletParallelConfig(tp_degree=2, cp_degree=2, cp_mode="ulysses"),
        DiffletParallelConfig(tp_degree=1, cp_degree=4, cp_mode="ulysses"),
    ],
)
def test_schedule_bytes_match_the_flat_cost_model(parallel):
    # The stage-2 collective graph and stage 1's comm_bytes are two views of
    # one program; if these totals ever diverge the stages rank different
    # realities and the comparison is meaningless. (Ring-CP is excluded here
    # and covered below: the flat model discounts its bytes for overlap.)
    profile, seq = _flux_profile(), _flux_seq()
    schedule = collective_schedule(parallel, profile=profile, seq=seq)
    total = sum(c.per_rank_bytes * c.occurrences for c in schedule)
    flat = cost_model.comm_bytes(parallel, profile=profile, seq=seq).total
    assert total == pytest.approx(flat, rel=1e-9)


@pytest.mark.parametrize(
    "parallel",
    [
        DiffletParallelConfig(tp_degree=4),
        DiffletParallelConfig(tp_degree=2, cp_degree=2),
        DiffletParallelConfig(tp_degree=2, cp_degree=2, cp_mode="ring"),
        DiffletParallelConfig(tp_degree=2, cp_degree=2, cp_mode="ulysses"),
    ],
)
def test_axis_split_reproduces_the_flat_total_seconds(parallel):
    # The calibrated seconds handed to stage 2 must sum to exactly what stage
    # 1 charged, ring's overlap discount included, so the Q2 anchor never
    # double-counts or drops traffic when it redistributes bytes per axis.
    profile, seq = _flux_profile(), _flux_seq()
    bandwidth = 100e9
    by_axis = comm_seconds_by_axis(
        parallel, profile=profile, seq=seq, bandwidth_bytes_per_second=bandwidth
    )
    flat = cost_model.comm_bytes(parallel, profile=profile, seq=seq).total
    assert sum(by_axis.values()) == pytest.approx(flat / bandwidth, rel=1e-9)


def test_tp_all_reduce_is_a_barrier_cp_traffic_is_overlappable():
    schedule = collective_schedule(
        DiffletParallelConfig(tp_degree=2, cp_degree=2),
        profile=_flux_profile(),
        seq=_flux_seq(),
    )
    by_axis = {c.axis: c for c in schedule}
    assert by_axis["tp"].barrier is True
    assert by_axis["cp"].barrier is False


# ------------------------------------------------------------- physical model


def _choice(parallel, compute_seconds=0.1, bandwidth=100e9):
    profile, seq = _flux_profile(), _flux_seq()
    return choose_placement(
        parallel,
        topology=FOUR_CORES,
        profile=profile,
        seq=seq,
        compute_seconds=compute_seconds,
        comm_seconds_by_axis=comm_seconds_by_axis(
            parallel,
            profile=profile,
            seq=seq,
            bandwidth_bytes_per_second=bandwidth,
        ),
    )


def test_dp2tp2_default_order_keeps_all_reduce_on_the_chip():
    choice = _choice(DiffletParallelConfig(tp_degree=2, dp_degree=2))
    default = choice.default
    tp_family = next(f for f in default.families if f.collective.axis == "tp")
    assert tp_family.bottleneck_tier == "intra-chip"
    assert tp_family.multiplier == pytest.approx(1.0)
    assert tp_family.contention_seconds == pytest.approx(0.0)
    assert choice.default_is_best


def test_dp2tp2_flipped_order_pays_tier_and_contention():
    # Both replicas' all-reduces cross the single chip-to-chip link: the
    # slower tier (2x) and two concurrent groups on one segment (2x) compound
    # to a 4x multiplier -- the paper's shared-link contention, quantified.
    choice = _choice(DiffletParallelConfig(tp_degree=2, dp_degree=2))
    flipped = choice.alternates[0]
    assert flipped.placement.order == ("dp", "tp")
    tp_family = next(f for f in flipped.families if f.collective.axis == "tp")
    tier_ratio = (
        topo.TIER_BANDWIDTH_BYTES_PER_SECOND["intra-chip"]
        / topo.TIER_BANDWIDTH_BYTES_PER_SECOND["intra-device"]
    )
    assert tp_family.multiplier == pytest.approx(4.0)  # 2 (tier) x 2 (sharing)
    assert tp_family.multiplier == pytest.approx(tier_ratio * 2.0)
    assert tp_family.contention_seconds > 0.0
    assert flipped.step_seconds > choice.default.step_seconds


def test_tp4_ring_pays_the_intra_device_tier_without_contention():
    # A single tp=4 group must cross the chip boundary (a 4-core device has
    # two chips), but one group cannot contend with itself.
    choice = _choice(DiffletParallelConfig(tp_degree=4))
    assert len(choice.alternates) == 0
    tp_family = choice.default.families[0]
    assert tp_family.bottleneck_tier == "intra-device"
    assert tp_family.contention_seconds == pytest.approx(0.0)


def test_q2_hides_overlappable_comm_under_compute():
    # tp1cp4: all communication is overlappable CP traffic, so when compute
    # dominates, Q2 must equal compute + max(0, comm - compute) -- less than
    # the stage-1 serial sum of the two.
    parallel = DiffletParallelConfig(cp_degree=4)
    profile, seq = _flux_profile(), _flux_seq()
    compute = 0.5
    by_axis = comm_seconds_by_axis(
        parallel, profile=profile, seq=seq, bandwidth_bytes_per_second=100e9
    )
    choice = choose_placement(
        parallel,
        topology=FOUR_CORES,
        profile=profile,
        seq=seq,
        compute_seconds=compute,
        comm_seconds_by_axis=by_axis,
    )
    overlappable = choice.default.overlappable_seconds
    assert choice.default.barrier_seconds == pytest.approx(0.0)
    assert choice.default.step_seconds == pytest.approx(compute + max(0.0, overlappable - compute))
    assert choice.default.step_seconds < compute + overlappable


def test_barrier_comm_is_never_hidden():
    parallel = DiffletParallelConfig(tp_degree=4)
    profile, seq = _flux_profile(), _flux_seq()
    compute = 10.0  # compute dwarfs communication
    by_axis = comm_seconds_by_axis(
        parallel, profile=profile, seq=seq, bandwidth_bytes_per_second=100e9
    )
    choice = choose_placement(
        parallel,
        topology=FOUR_CORES,
        profile=profile,
        seq=seq,
        compute_seconds=compute,
        comm_seconds_by_axis=by_axis,
    )
    assert choice.default.step_seconds == pytest.approx(compute + choice.default.barrier_seconds)


def test_pure_dp_has_no_placement_decision():
    # dp issues no collectives; there is nothing to place or rescore.
    assert _choice(DiffletParallelConfig(dp_degree=4)) is None


# --------------------------------------------------------------- integration


def _plan(**kwargs):
    kwargs.setdefault("model_type", "flux")
    kwargs.setdefault("steps", 28)
    kwargs.setdefault("hardware", TRN2)
    kwargs.setdefault("cache_dir", "/nonexistent-cache")
    return plan(kwargs.pop("model_id", FLUX), **kwargs)


def test_plan_runs_stage2_and_records_survivors():
    result = _plan()
    assert result.survivors  # default cut keeps the whole (tiny) field
    multi_axis = [
        entry
        for entry in result.ranked
        if entry.label in ("tp2cp2", "dp2tp2") and entry.placement is not None
    ]
    assert multi_axis
    for entry in multi_axis:
        assert entry.stage2_step_seconds == entry.step_seconds


def test_survivors_cut_leaves_the_rest_at_stage1():
    result = _plan(survivors=2)
    staged = {entry.label for entry in result.ranked if entry.stage2_step_seconds is not None}
    assert set(result.survivors) == staged
    assert len(staged) == 2


def test_no_topology_reproduces_the_single_stage_behavior():
    legacy = _plan(topology_aware=False)
    staged = _plan()
    assert not legacy.survivors
    assert all(entry.placement is None for entry in legacy.ranked)
    # Stage 2 only ever *hides* communication under compute; it cannot invent
    # time. Every refined prediction must be <= its stage-1 prediction.
    legacy_by_label = {entry.label: entry for entry in legacy.ranked}
    for entry in staged.ranked:
        if entry.prediction.evidence != "measured":
            assert entry.step_seconds <= legacy_by_label[entry.label].step_seconds


def test_measured_entries_keep_their_measured_step_time():
    from difflet.planner.measurements import Measurement, MeasurementStore

    store = MeasurementStore(
        (
            Measurement(
                instance_type="trn2.3xlarge",
                model="flux",
                model_id=FLUX,
                label="tp2cp2",
                height=1024,
                width=1024,
                num_frames=None,
                steps=28,
                step_latency_seconds=0.3,
                e2e_warm_seconds=None,
                compile_seconds=None,
                source="test",
            ),
        )
    )
    result = _plan(store=store)
    entry = next(e for e in result.ranked if e.label == "tp2cp2")
    assert entry.prediction.evidence == "measured"
    assert entry.step_seconds == pytest.approx(0.3)
    assert entry.placement is not None  # ranked, but not re-timed
