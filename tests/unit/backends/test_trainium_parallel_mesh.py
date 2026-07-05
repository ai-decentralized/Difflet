"""ProcessGroupManager: spec derivation, per-axis group construction, SPMD ranks."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pm = pytest.importorskip("difflet.backends.trainium.core.parallel_mesh")

from difflet.pipeline.parallel_mesh import MeshSpec  # noqa: E402


class _FakeWorldGroup:
    def __init__(self, n):
        self._n = n

    def size(self):
        return self._n


@pytest.fixture(autouse=True)
def _reset_mesh(monkeypatch):
    pm.destroy_parallel_mesh()
    created = []

    def fake_new_group(ranks, pg_options=None):
        created.append((list(ranks), pg_options))
        return SimpleNamespace(ranks=list(ranks), pg_options=pg_options)

    monkeypatch.setattr(pm.torch.distributed, "new_group", fake_new_group)
    yield created
    pm.destroy_parallel_mesh()


def _patch_world(monkeypatch, tp, world):
    monkeypatch.setattr(pm, "_nxd_tp_size", lambda: tp)
    monkeypatch.setattr(pm, "_nxd_world_group", lambda: _FakeWorldGroup(world))


def _config(**kwargs):
    return SimpleNamespace(**kwargs)


def test_spec_cfg_parallel(monkeypatch):
    _patch_world(monkeypatch, tp=2, world=4)
    spec = pm.mesh_spec_from_config(_config(cfg_parallel_enabled=True))
    assert spec == MeshSpec(dp=1, cfg=2, cp=1, tp=2)


def test_spec_cp_inferred_from_world(monkeypatch):
    _patch_world(monkeypatch, tp=2, world=8)
    spec = pm.mesh_spec_from_config(_config(context_parallel_enabled=True))
    assert spec == MeshSpec(dp=1, cfg=1, cp=4, tp=2)


def test_spec_product_must_match_world(monkeypatch):
    _patch_world(monkeypatch, tp=2, world=8)  # no cp/cfg flag but world > tp
    with pytest.raises(ValueError):
        pm.mesh_spec_from_config(_config())


def test_spec_indivisible_world_rejected(monkeypatch):
    _patch_world(monkeypatch, tp=2, world=3)
    with pytest.raises(ValueError):
        pm.mesh_spec_from_config(_config(cfg_parallel_enabled=True))


def test_init_builds_only_nontrivial_axes(monkeypatch):
    _patch_world(monkeypatch, tp=2, world=4)
    pm.init_parallel_mesh(_config(cfg_parallel_enabled=True))
    assert pm.get_mesh_spec() == MeshSpec(cfg=2, tp=2)
    assert pm.get_cfg_group() is not None
    with pytest.raises(AssertionError):
        pm.get_cp_group()
    with pytest.raises(AssertionError):
        pm.get_dp_group()


def test_getters_require_init():
    with pytest.raises(AssertionError):
        pm.get_cfg_group()
    with pytest.raises(AssertionError):
        pm.get_mesh_spec()


def test_cfg_group_mesh_matches_legacy_nxd_dp_mesh(monkeypatch, _reset_mesh):
    _patch_world(monkeypatch, tp=2, world=4)
    pm.init_parallel_mesh(_config(cfg_parallel_enabled=True))
    ((ranks, pg_options),) = _reset_mesh
    assert ranks == [0, 2]
    assert pg_options == {"xla_pg_options": {"mesh": [[0, 2], [1, 3]]}}


def test_init_idempotent_and_conflict_raises(monkeypatch, _reset_mesh):
    _patch_world(monkeypatch, tp=2, world=4)
    pm.init_parallel_mesh(_config(cfg_parallel_enabled=True))
    pm.init_parallel_mesh(_config(cfg_parallel_enabled=True))  # same spec: no-op
    assert len(_reset_mesh) == 1  # no duplicate group construction
    with pytest.raises(RuntimeError):
        pm.init_parallel_mesh(_config(context_parallel_enabled=True))


def test_cp_mesh_accessor(monkeypatch):
    _patch_world(monkeypatch, tp=2, world=8)
    pm.init_parallel_mesh(_config(context_parallel_enabled=True))
    assert pm.get_cp_mesh() == [[0, 2, 4, 6], [1, 3, 5, 7]]


def test_spmd_rank_helpers_match_mesh_coords(monkeypatch):
    _patch_world(monkeypatch, tp=2, world=4)
    pm.init_parallel_mesh(_config(cfg_parallel_enabled=True))
    ranks = torch.arange(4)
    cfg_ranks = pm.get_cfg_rank_spmd(ranks)
    assert cfg_ranks.dtype == torch.int32
    assert cfg_ranks.tolist() == [0, 0, 1, 1]
    assert pm.get_cp_rank_spmd(ranks).tolist() == [0, 0, 0, 0]


def test_spmd_cp_rank_legacy_equivalence(monkeypatch):
    # (cfg=1, cp=4, tp=2): cp_rank must equal legacy rank // tp
    _patch_world(monkeypatch, tp=2, world=8)
    pm.init_parallel_mesh(_config(context_parallel_enabled=True))
    ranks = torch.arange(8)
    assert pm.get_cp_rank_spmd(ranks).tolist() == [r // 2 for r in range(8)]
