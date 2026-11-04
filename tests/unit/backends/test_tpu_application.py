"""Unit tests for the TPU component lifecycle.

No TPU needed: the contract (no ahead-of-time artifact on TorchTPU) and the
weight-sharding logic are pure host code.
"""

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


def test_compile_and_load_refuse_with_the_reason():
    app = ab.TpuApplicationBase()
    for call in (lambda: app.compile("/tmp/x"), lambda: app.load("/tmp/x")):
        with pytest.raises(NotImplementedError, match="no ahead-of-time artifact"):
            call()
    assert app.has_compiled_artifacts("/tmp/x") is False


def test_forward_before_the_module_is_built_is_an_explicit_error():
    app = ab.TpuApplicationBase()
    with pytest.raises(RuntimeError, match="not built"):
        app(torch.zeros(1))


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


# --- compile slots -------------------------------------------------------------


def test_warmup_eager_goes_through_the_subclass_forward(monkeypatch):
    """LTX-2's application maps the positional bundle onto diffusers' keyword
    arguments in its forward; calling the bare module positionally instead
    misfiled them (v5e: `arange() ... end=NoneType`)."""
    from difflet.backends.tpu.core.application_base import TpuApplicationBase
    from difflet.backends.tpu.ops_impl import device_runtime

    monkeypatch.setattr(device_runtime, "wait", lambda: None)
    seen = {}

    class App(TpuApplicationBase):
        def __init__(self):
            super().__init__(config=None)
            self.module = None

        def get_example_inputs(self):
            return (torch.ones(2), torch.zeros(2))

        def forward(self, a, b):
            seen["called_with"] = (a, b)
            return self.module(a=a, b=b)

    app = App()
    module = lambda *, a, b: a + b  # keyword-only: a positional call would raise
    app.warmup_eager(module, device="cpu")
    assert app.module is module
    assert torch.equal(seen["called_with"][0], torch.ones(2))
