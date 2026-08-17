"""Unit tests for the TPU component lifecycle (plan Phase 3).

No TPU needed: the artifact layout, manifest validation and weight-sharding
logic are all pure host code. The on-device round-trip (compile in 4 procs →
load in 4 fresh procs → forward matches a full-weight oracle) is verified
separately on real hardware.
"""

import json

import pytest
import torch
import torch.nn as nn

from difflet.backends.tpu.core import application_base as ab
from difflet.backends.tpu.core import weights as W
from difflet.backends.tpu.ops_impl import linear as L
from difflet.backends.tpu.ops_impl import parallel_mesh as pm
from difflet.pipeline.parallel_mesh import MeshSpec


@pytest.fixture(autouse=True)
def _clean_mesh():
    pm.destroy_parallel_mesh()
    yield
    pm.destroy_parallel_mesh()


# ------------------------------------------------------------------ contract


def test_signatures_match_what_the_pipeline_reflects_on():
    # difflet/pipeline/difflet_pipeline.py inspects these parameter NAMES and
    # only passes the ones it finds. A rename here silently drops arguments.
    import inspect

    compile_params = inspect.signature(ab.TpuApplicationBase.compile).parameters
    assert "debug" in compile_params

    load_params = inspect.signature(ab.TpuApplicationBase.load).parameters
    for name in ("start_rank_id", "local_ranks_size", "skip_warmup"):
        assert name in load_params


def test_forward_before_load_is_an_explicit_error():
    app = ab.TpuApplicationBase()
    with pytest.raises(RuntimeError, match="not loaded"):
        app(torch.zeros(1))


def test_forward_rejects_kwargs():
    app = ab.TpuApplicationBase()
    app.graph_module = lambda *a: a  # stand in for a loaded graph
    with pytest.raises(TypeError, match="positional inputs only"):
        app(torch.zeros(1), foo=1)


# ------------------------------------------------------------------ manifest


def _app_with_mesh(spec):
    pm.init_parallel_mesh(spec)
    return ab.TpuApplicationBase()


def test_has_compiled_artifacts_requires_every_rank(tmp_path):
    app = _app_with_mesh(MeshSpec(tp=4))
    app._write_manifest(tmp_path, app._mesh())
    assert app.has_compiled_artifacts(tmp_path) is False

    for rank in range(4):
        (tmp_path / ab.RANK_DIR_TEMPLATE.format(rank=rank)).mkdir()
    assert app.has_compiled_artifacts(tmp_path) is True


def test_has_compiled_artifacts_is_false_without_a_manifest(tmp_path):
    app = _app_with_mesh(MeshSpec(tp=1))
    assert app.has_compiled_artifacts(tmp_path) is False


def test_has_compiled_artifacts_survives_a_corrupt_manifest(tmp_path):
    app = _app_with_mesh(MeshSpec(tp=1))
    (tmp_path / ab.MANIFEST_FILE_NAME).write_text("{not json")
    # A half-written manifest must read as "not ready", not crash the caller.
    assert app.has_compiled_artifacts(tmp_path) is False


def test_manifest_round_trips_the_mesh(tmp_path):
    app = _app_with_mesh(MeshSpec(dp=1, cfg=2, cp=1, tp=2))
    app._write_manifest(tmp_path, app._mesh())
    body = json.loads((tmp_path / ab.MANIFEST_FILE_NAME).read_text())
    assert body["mesh"] == {"dp": 1, "cfg": 2, "cp": 1, "tp": 2}
    assert body["world_size"] == 4
    app._validate_manifest(tmp_path, app._mesh())  # must not raise


def test_loading_an_artifact_compiled_for_another_mesh_is_rejected(tmp_path):
    app = _app_with_mesh(MeshSpec(tp=4))
    app._write_manifest(tmp_path, app._mesh())
    # Same world size, different factorization — the collectives would be
    # wired differently, so this must not be accepted.
    with pytest.raises(ValueError, match="compiled for mesh"):
        app._validate_manifest(tmp_path, MeshSpec(tp=2, cp=2))


def test_validate_manifest_missing_file(tmp_path):
    app = _app_with_mesh(MeshSpec(tp=1))
    with pytest.raises(FileNotFoundError, match="manifest"):
        app._validate_manifest(tmp_path, app._mesh())


# ------------------------------------------------------------- weight shards


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = L.ColumnParallelLinear(8, 16, bias=True, gather_output=False)
        self.down = L.RowParallelLinear(16, 8, bias=True, input_is_parallel=True)


def _full_weights():
    return {
        "up.weight": torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8),
        "up.bias": torch.arange(16, dtype=torch.float32),
        "down.weight": torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16),
        "down.bias": torch.arange(8, dtype=torch.float32),
    }


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_shard_state_dict_splits_on_the_declared_axis(rank):
    pm.init_parallel_mesh(MeshSpec(tp=4))
    block = _Block()
    full = _full_weights()
    got = W.shard_state_dict(block, full, tp_size=4, tp_rank=rank)

    # Column-parallel: output dim (0) split; the rank's slice of the rows.
    assert torch.equal(got["up.weight"], full["up.weight"][rank * 4 : (rank + 1) * 4])
    assert torch.equal(got["up.bias"], full["up.bias"][rank * 4 : (rank + 1) * 4])
    # Row-parallel: input dim (1) split, bias replicated.
    assert torch.equal(got["down.weight"], full["down.weight"][:, rank * 4 : (rank + 1) * 4])
    assert torch.equal(got["down.bias"], full["down.bias"])


def test_shards_load_into_the_module():
    pm.init_parallel_mesh(MeshSpec(tp=4))
    block = _Block()
    W.load_sharded_state_dict(block, _full_weights(), tp_size=4, tp_rank=2)
    assert torch.equal(block.up.weight, _full_weights()["up.weight"][8:12])


def test_already_sharded_weights_are_not_split_twice():
    # Re-loading an already-sharded checkpoint must be a no-op, not a
    # quarter-of-a-quarter model — a silent corruption if it went wrong.
    pm.init_parallel_mesh(MeshSpec(tp=4))
    block = _Block()
    once = W.shard_state_dict(block, _full_weights(), tp_size=4, tp_rank=1)
    twice = W.shard_state_dict(block, once, tp_size=4, tp_rank=1)
    for name, tensor in once.items():
        assert torch.equal(twice[name], tensor)


def test_mismatched_shape_without_a_shard_axis_is_rejected():
    pm.init_parallel_mesh(MeshSpec(tp=4))
    block = _Block()
    bad = _full_weights()
    bad["down.bias"] = torch.zeros(99)  # replicated param, wrong shape
    with pytest.raises(ValueError, match="no shard axis"):
        W.shard_state_dict(block, bad, tp_size=4, tp_rank=0)


def test_indivisible_weight_is_rejected():
    pm.init_parallel_mesh(MeshSpec(tp=4))
    with pytest.raises(ValueError, match="not divisible"):
        W.narrow_to_rank(torch.zeros(10, 4), dim=0, tp_size=4, tp_rank=0)


def test_unexpected_checkpoint_entries_are_reported():
    pm.init_parallel_mesh(MeshSpec(tp=4))
    block = _Block()
    extra = _full_weights() | {"nope.weight": torch.zeros(4)}
    with pytest.raises(ValueError, match="no matching"):
        W.shard_state_dict(block, extra, tp_size=4, tp_rank=0)


def test_tp1_sharding_is_identity():
    pm.init_parallel_mesh(MeshSpec(tp=1))
    block = _Block()
    full = {
        "up.weight": torch.randn(16, 8),
        "up.bias": torch.randn(16),
        "down.weight": torch.randn(8, 16),
        "down.bias": torch.randn(8),
    }
    got = W.shard_state_dict(block, full, tp_size=1, tp_rank=0)
    for name, tensor in full.items():
        assert torch.equal(got[name], tensor)
