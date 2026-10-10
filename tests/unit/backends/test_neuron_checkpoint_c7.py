"""Unit tests for the neuron backend's lazy per-rank checkpoint loader (C7), CPU only.

``device="cpu"`` runs the same code path the device takes up to the final
host-to-device copy; tests/manual/check_neuron_checkpoint_c7.py covers that on
hardware. The gloo tests run four real ranks that take their tp rank from the
neuron parallel mesh.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
import torch.nn as nn

pytest.importorskip("safetensors.torch")
pytest.importorskip("accelerate")

from difflet.backends.neuron.core import checkpoint as ckpt  # noqa: E402
from difflet.backends.neuron.ops_impl import parallel_mesh as pm  # noqa: E402
from difflet.backends.neuron.ops_impl.linear import ColumnParallelLinear  # noqa: E402
from difflet.backends.tpu.core import checkpoint as tpu_checkpoint  # noqa: E402
from difflet.backends.tpu.core import weights as tpu_weights  # noqa: E402
from difflet.pipeline.parallel_mesh import MeshSpec  # noqa: E402
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_toy import (  # noqa: E402
    ToyTPMLP,
    toy_full_weights,
    write_toy_checkpoint,
)
from tests.unit.backends._neuron_workers import (  # noqa: E402
    c7_checkpoint_worker,
    c7_peak_rss_worker,
)

REPO = Path(__file__).resolve().parents[3]
DIM, HIDDEN, TP = 16, 64, 4
RSS_LAYERS, RSS_DIM, RSS_HIDDEN = 16, 512, 4096
MiB = 2**20


@pytest.fixture(autouse=True)
def _clean_mesh():
    pm.destroy_parallel_mesh()
    yield
    pm.destroy_parallel_mesh()


def _toy():
    return ToyTPMLP(DIM, HIDDEN)


def _expected(module, name, full, rank, tp=TP):
    axis = ckpt.shard_dim(module, name)
    return full if axis is None else full.chunk(tp, dim=axis)[rank]


class _WithBuffers(nn.Module):
    """A sharded layer, a computed (non-persistent) buffer and a stored fp32 buffer."""

    def __init__(self):
        super().__init__()
        self.up = ColumnParallelLinear(DIM, HIDDEN, bias=True, gather_output=False)
        self.register_buffer("rope", torch.linspace(0.0, 1.0, 8), persistent=False)
        self.register_buffer("scale", torch.zeros(4), persistent=True)


def _with_buffers_weights():
    g = torch.Generator().manual_seed(0)
    return {
        "up.weight": torch.randn(HIDDEN, DIM, generator=g),
        "up.bias": torch.randn(HIDDEN, generator=g),
        "scale": torch.full((4,), 1.0 + 2.0**-12),  # exact in fp32, rounds to 1.0 in bf16
    }


def test_reexports_are_the_tpu_objects_not_copies():
    for name in ("build_weight_map", "load_checkpoint_into"):
        assert getattr(ckpt, name) is getattr(tpu_checkpoint, name)
    for name in (
        "load_sharded_state_dict",
        "materialize_meta_",
        "narrow_to_rank",
        "shard_dim",
        "shard_state_dict",
    ):
        assert getattr(ckpt, name) is getattr(tpu_weights, name)
    assert sorted(ckpt.__all__) == [
        "build_on_meta",
        "build_weight_map",
        "load_checkpoint_into",
        "load_sharded_checkpoint",
        "load_sharded_state_dict",
        "materialize_meta_",
        "narrow_to_rank",
        "shard_dim",
        "shard_state_dict",
    ]


_IMPORT_GUARD = textwrap.dedent(
    """
    import importlib.abc
    import sys

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.startswith(("torch_xla", "neuronx_distributed")):
                raise ImportError(f"blocked import of {name}")
            return None

    sys.meta_path.insert(0, _Blocker())
    import difflet.backends.neuron.core.checkpoint  # noqa: F401

    leaked = sorted(m for m in sys.modules if m.startswith(("torch_xla", "neuronx_distributed")))
    assert not leaked, leaked
    print("IMPORT-OK")
    """
)


def test_importing_the_loader_pulls_no_xla_or_nxd():
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO),
        "DIFFLET_BACKEND": "neuron",
        "DIFFLET_DISABLE_PREWARM": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_GUARD],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert "IMPORT-OK" in proc.stdout


def test_build_on_meta_puts_parameters_on_meta_and_keeps_computed_buffers_real():
    module = ckpt.build_on_meta(_WithBuffers)
    assert all(p.is_meta for p in module.parameters())
    assert not module.rope.is_meta
    assert torch.equal(module.rope, torch.linspace(0.0, 1.0, 8))


@pytest.mark.parametrize("num_files", [1, 2])
def test_tp4_shards_tile_the_checkpoint_exactly(tmp_path, num_files):
    # The invariant: the shards tile the original with no gap and no overlap.
    pm.init_parallel_mesh(MeshSpec(tp=TP))
    full = toy_full_weights(DIM, HIDDEN, seed=0)
    write_toy_checkpoint(tmp_path, full, num_files=num_files)
    shards = []
    for rank in range(TP):
        module = ckpt.build_on_meta(_toy)
        report = ckpt.load_sharded_checkpoint(
            module, tmp_path, device="cpu", dtype=torch.float32, tp_size=TP, tp_rank=rank
        )
        assert report == {"missing": [], "unexpected": []}
        shards.append(module.state_dict())
    dims = {name: ckpt.shard_dim(module, name) for name in full}
    assert {0, 1} <= set(dims.values()), dims  # the toy shards both ways
    for name, tensor in full.items():
        if dims[name] is None:
            assert all(torch.equal(sd[name], tensor) for sd in shards), name
        else:
            whole = torch.cat([sd[name] for sd in shards], dim=dims[name])
            assert torch.equal(whole, tensor), name


def test_fp32_checkpoint_is_cast_to_the_load_dtype(tmp_path):
    pm.init_parallel_mesh(MeshSpec(tp=TP))
    full = toy_full_weights(DIM, HIDDEN, seed=0)
    write_toy_checkpoint(tmp_path, full)
    module = ckpt.build_on_meta(_toy)
    ckpt.load_sharded_checkpoint(
        module, tmp_path, device="cpu", dtype=torch.bfloat16, tp_size=TP, tp_rank=2
    )
    state = module.state_dict()
    for name, tensor in full.items():
        assert state[name].dtype == torch.bfloat16, name
        assert torch.equal(state[name], _expected(module, name, tensor, 2).to(torch.bfloat16))


def test_buffers_keep_their_dtype_and_computed_buffers_are_untouched(tmp_path):
    # tp defaults: no mesh, world 1 -> tp 1, rank 0.
    weights = _with_buffers_weights()
    write_toy_checkpoint(tmp_path, weights)
    module = ckpt.build_on_meta(_WithBuffers)
    report = ckpt.load_sharded_checkpoint(module, tmp_path, device="cpu", dtype=torch.bfloat16)
    assert report == {"missing": [], "unexpected": []}
    assert module.up.weight.dtype == torch.bfloat16
    assert module.scale.dtype == torch.float32
    assert torch.equal(module.scale, weights["scale"])  # not rounded through bf16
    assert torch.equal(module.rope, torch.linspace(0.0, 1.0, 8))


def test_default_tp_comes_from_the_mesh_and_needs_a_process_group_at_tp4(tmp_path):
    pm.init_parallel_mesh(MeshSpec(tp=TP))  # spec-only: no process group
    write_toy_checkpoint(tmp_path, toy_full_weights(DIM, HIDDEN, seed=0))
    module = ckpt.build_on_meta(_toy)
    with pytest.raises(RuntimeError):
        ckpt.load_sharded_checkpoint(module, tmp_path, device="cpu")
    assert all(p.is_meta for p in module.parameters())  # failed before materialising


def test_rank_out_of_range_is_rejected(tmp_path):
    pm.init_parallel_mesh(MeshSpec(tp=TP))
    write_toy_checkpoint(tmp_path, toy_full_weights(DIM, HIDDEN, seed=0))
    with pytest.raises(ValueError, match="tp_rank 4 is out of range"):
        ckpt.load_sharded_checkpoint(
            ckpt.build_on_meta(_toy), tmp_path, device="cpu", tp_size=TP, tp_rank=TP
        )


def test_rename_maps_module_names_to_checkpoint_keys(tmp_path):
    pm.init_parallel_mesh(MeshSpec(tp=TP))
    full = toy_full_weights(DIM, HIDDEN, seed=0)

    def rename(name):
        return "ffn." + name.replace("up.", "net.0.proj.").replace("down.", "net.2.")

    write_toy_checkpoint(tmp_path, {rename(k): v for k, v in full.items()})
    module = ckpt.build_on_meta(_toy)
    report = ckpt.load_sharded_checkpoint(
        module, tmp_path, device="cpu", dtype=torch.float32, tp_size=TP, tp_rank=1, rename=rename
    )
    assert report == {"missing": [], "unexpected": []}
    state = module.state_dict()
    for name, tensor in full.items():
        assert torch.equal(state[name], _expected(module, name, tensor, 1)), name


def test_missing_entries_fail_strict_and_are_reported_otherwise(tmp_path):
    pm.init_parallel_mesh(MeshSpec(tp=TP))
    full = toy_full_weights(DIM, HIDDEN, seed=0)
    dropped = sorted(k for k in full if k.startswith("down."))
    write_toy_checkpoint(tmp_path, {k: v for k, v in full.items() if k not in dropped})
    with pytest.raises(ValueError, match="had no checkpoint entry"):
        ckpt.load_sharded_checkpoint(
            ckpt.build_on_meta(_toy), tmp_path, device="cpu", tp_size=TP, tp_rank=0
        )
    report = ckpt.load_sharded_checkpoint(
        ckpt.build_on_meta(_toy), tmp_path, device="cpu", tp_size=TP, tp_rank=0, strict=False
    )
    assert sorted(report["missing"]) == dropped


def test_meta_computed_buffers_are_refused(tmp_path):
    # Built under torch.device("meta") the computed buffer is on meta too; nothing
    # could ever fill it, so the loader must refuse rather than leave garbage.
    write_toy_checkpoint(tmp_path, _with_buffers_weights())
    with torch.device("meta"):
        module = _WithBuffers()
    with pytest.raises(ValueError, match="non-persistent buffers on meta"):
        ckpt.load_sharded_checkpoint(module, tmp_path, device="cpu")


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_gloo_tp4_each_rank_loads_its_shards_and_they_reassemble(tmp_path, dtype_name):
    write_toy_checkpoint(tmp_path, toy_full_weights(DIM, HIDDEN, seed=0), num_files=2)
    results = run_ranks(
        c7_checkpoint_worker, str(tmp_path), DIM, HIDDEN, dtype_name, world_size=TP
    )
    for rank, res in enumerate(results):
        assert res["rank"] == rank and res["tp_rank"] == rank, res
        assert res["missing"] == [] and res["unexpected"] == [], res
        assert res["shards_exact"], res
        assert res["reassembled_exact"], res
        if dtype_name == "float32":
            assert res["forward_err"] <= 1e-5, res


def test_gloo_tp4_peak_host_rss_stays_near_one_rank_shard(tmp_path):
    weights = {
        f"{i}.{name}": tensor
        for i in range(RSS_LAYERS)
        for name, tensor in toy_full_weights(RSS_DIM, RSS_HIDDEN, seed=i).items()
    }
    write_toy_checkpoint(tmp_path, weights, num_files=4)
    ckpt_bytes = sum(t.numel() * t.element_size() for t in weights.values())
    largest = max(t.numel() * t.element_size() for t in weights.values())
    del weights
    results = run_ranks(
        c7_peak_rss_worker, str(tmp_path), RSS_LAYERS, RSS_DIM, RSS_HIDDEN, world_size=TP
    )
    if any(res["peak_delta"] is None for res in results):
        pytest.skip("cannot reset the peak RSS counter (/proc/self/clear_refs)")
    for res in results:
        assert res["missing"] == [], res
        # module shards + two full fp32 tensors' mapped pages (a dim-1 slice touches every
        # page) + one fp32 shard and its bf16 copy in flight + allocator/page slack
        bound = (
            res["module_bytes"] + 2 * largest + largest // TP + largest // (2 * TP) + 48 * MiB
        )
        assert bound < ckpt_bytes // 2, "test too small to tell a lazy load from a full one"
        assert res["peak_delta"] <= bound, (
            f"rank {res['rank']}: load peak +{res['peak_delta'] / MiB:.1f} MiB > bound "
            f"{bound / MiB:.1f} MiB (checkpoint {ckpt_bytes / MiB:.0f} MiB)"
        )
