"""Rank <-> (dp, cfg, cp, tp) mapping and axis-subgroup math for MeshSpec."""

import pytest

from difflet.pipeline.parallel_mesh import AXES, MeshCoords, MeshSpec


def test_axes_order_outer_to_inner():
    assert AXES == ("dp", "cfg", "cp", "tp")


@pytest.mark.parametrize("axis", AXES)
def test_axis_must_be_positive(axis):
    with pytest.raises(ValueError):
        MeshSpec(**{axis: 0})


def test_world_size_is_product():
    assert MeshSpec(dp=2, cfg=2, cp=3, tp=4).world_size == 48


@pytest.mark.parametrize(
    "spec",
    [
        MeshSpec(dp=1, cfg=2, cp=2, tp=2),
        MeshSpec(dp=1, cfg=1, cp=4, tp=2),
        MeshSpec(dp=4, cfg=1, cp=1, tp=2),
        MeshSpec(dp=2, cfg=2, cp=1, tp=2),
    ],
)
def test_rank_roundtrip_all_required_combos(spec):
    for rank in range(spec.world_size):
        c = spec.coords_of(rank)
        assert spec.rank_of(dp=c.dp, cfg=c.cfg, cp=c.cp, tp=c.tp) == rank


def test_rank_formula_tp_innermost_dp_outermost():
    spec = MeshSpec(dp=2, cfg=2, cp=2, tp=2)
    # rank = tp + T*(cp + C*(cfg + G*dp))
    assert spec.rank_of(dp=0, cfg=0, cp=0, tp=1) == 1
    assert spec.rank_of(dp=0, cfg=0, cp=1, tp=0) == 2
    assert spec.rank_of(dp=0, cfg=1, cp=0, tp=0) == 4
    assert spec.rank_of(dp=1, cfg=0, cp=0, tp=0) == 8
    assert spec.coords_of(13) == MeshCoords(dp=1, cfg=1, cp=0, tp=1)


def test_coords_and_rank_validate_ranges():
    spec = MeshSpec(cfg=2, tp=2)
    with pytest.raises(ValueError):
        spec.coords_of(4)
    with pytest.raises(ValueError):
        spec.coords_of(-1)
    with pytest.raises(ValueError):
        spec.rank_of(cfg=2)


def test_axis_groups_partition_and_vary_only_that_axis():
    spec = MeshSpec(dp=2, cfg=2, cp=2, tp=2)
    for axis in AXES:
        groups = spec.axis_groups(axis)
        flat = sorted(r for g in groups for r in g)
        assert flat == list(range(spec.world_size))  # exact partition
        for group in groups:
            assert len(group) == spec.axis_size(axis)
            base = spec.coords_of(group[0])
            for i, rank in enumerate(group):
                c = spec.coords_of(rank)
                assert getattr(c, axis) == i  # increasing axis coord
                for other in AXES:
                    if other != axis:
                        assert getattr(c, other) == getattr(base, other)


def test_legacy_cfg2_mesh_matches_nxd_dp_columns():
    # (dp=1, cfg=2, cp=1, tp=T): cfg groups must be [[j, j+T] for j in range(T)]
    spec = MeshSpec(cfg=2, tp=4)
    assert spec.axis_groups("cfg") == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_legacy_cp4_mesh_matches_nxd_dp_columns():
    # (dp=1, cfg=1, cp=4, tp=2): cp groups must be [[j, j+T, j+2T, j+3T]]
    spec = MeshSpec(cp=4, tp=2)
    assert spec.axis_groups("cp") == [[0, 2, 4, 6], [1, 3, 5, 7]]


def test_axis_rank_matches_coords():
    spec = MeshSpec(dp=1, cfg=2, cp=2, tp=2)
    for rank in range(spec.world_size):
        c = spec.coords_of(rank)
        for axis in AXES:
            assert spec.axis_rank(rank, axis) == getattr(c, axis)


def test_unknown_axis_rejected():
    with pytest.raises(ValueError):
        MeshSpec().axis_groups("pp")
    with pytest.raises(ValueError):
        MeshSpec().axis_stride("pp")
    with pytest.raises(ValueError):
        MeshSpec().axis_size("pp")


def test_bool_axis_size_rejected():
    with pytest.raises(ValueError):
        MeshSpec(cfg=True)
