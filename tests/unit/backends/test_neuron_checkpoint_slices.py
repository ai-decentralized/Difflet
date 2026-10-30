"""Sliced parameters in the neuron checkpoint loader: a parameter that is a range of a fused
checkpoint tensor (Flux's single-stream proj_out split into proj_out_attn and proj_out_mlp), read
per TP shard within that range."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from difflet.backends.neuron.core.checkpoint import CheckpointSlice, load_sharded_checkpoint
from difflet.backends.neuron.ops_impl import collectives as C
from difflet.backends.neuron.ops_impl.linear import RowParallelLinear
from difflet.pipeline.parallel_mesh import MeshSpec

ATTN, MLP, OUT = 8, 16, 4


@pytest.fixture(autouse=True)
def _reset_mesh():
    C.destroy_parallel_mesh()
    yield
    C.destroy_parallel_mesh()


class Fused(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = RowParallelLinear(ATTN, OUT, bias=True)
        self.mlp = RowParallelLinear(MLP, OUT, bias=False)


@pytest.fixture
def checkpoint(tmp_path):
    weight = torch.arange(OUT * (ATTN + MLP), dtype=torch.float32).reshape(OUT, ATTN + MLP)
    bias = torch.arange(OUT, dtype=torch.float32) + 100
    save_file({"proj_out.weight": weight, "proj_out.bias": bias}, str(tmp_path / "model.safetensors"))
    return tmp_path, weight, bias


SLICES = {"attn.weight": CheckpointSlice("proj_out.weight", 1, 0, ATTN),
          "mlp.weight": CheckpointSlice("proj_out.weight", 1, ATTN, MLP)}


def _rename(name):
    return {"attn.bias": "proj_out.bias"}.get(name, name)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_each_rank_reads_its_shard_of_each_slice(checkpoint, tp):
    model_dir, weight, bias = checkpoint
    C.init_parallel_mesh(MeshSpec(tp=tp))
    for rank in range(tp):
        module = Fused()
        report = load_sharded_checkpoint(module, model_dir, device="cpu", dtype=torch.float32,
                                         tp_size=tp, tp_rank=rank, rename=_rename, slices=SLICES)
        a, m = ATTN // tp, MLP // tp
        torch.testing.assert_close(module.attn.weight, weight[:, rank * a:(rank + 1) * a])
        torch.testing.assert_close(module.mlp.weight, weight[:, ATTN + rank * m:ATTN + (rank + 1) * m])
        torch.testing.assert_close(module.attn.bias, bias)  # row-parallel bias: replicated
        assert report == {"missing": [], "unexpected": []}


def test_a_parameter_neither_in_the_checkpoint_nor_sliced_is_still_missing(checkpoint):
    model_dir, _, _ = checkpoint
    C.init_parallel_mesh(MeshSpec(tp=1))
    with pytest.raises(ValueError, match="had no checkpoint entry: \\['mlp.weight'\\]"):
        load_sharded_checkpoint(Fused(), model_dir, device="cpu", dtype=torch.float32,
                                rename=_rename, slices={"attn.weight": SLICES["attn.weight"]})


def test_a_slice_past_the_tensor_is_rejected(checkpoint):
    model_dir, _, _ = checkpoint
    C.init_parallel_mesh(MeshSpec(tp=1))
    bad = dict(SLICES, **{"mlp.weight": CheckpointSlice("proj_out.weight", 1, ATTN, MLP + 1)})
    with pytest.raises(ValueError, match="range exceeds"):
        load_sharded_checkpoint(Fused(), model_dir, device="cpu", dtype=torch.float32,
                                rename=_rename, slices=bad)


def test_no_slices_is_the_plain_loader(checkpoint, tmp_path):
    # Wan and Qwen-Image declare none: unchanged behaviour, including strict on a real gap.
    model_dir, _, _ = checkpoint
    C.init_parallel_mesh(MeshSpec(tp=1))
    with pytest.raises(ValueError, match="had no checkpoint entry"):
        load_sharded_checkpoint(Fused(), model_dir, device="cpu", dtype=torch.float32, rename=_rename)


def test_two_complete_indexes_resolve_from_the_one_holding_the_module_keys(checkpoint):
    # LTX-2's text_encoder/ ships transformers' and diffusers' indexes, both complete; the
    # diffusers one sorts first and holds none of the module's keys.
    model_dir, weight, bias = checkpoint
    (model_dir / "model.safetensors").rename(model_dir / "model-00001-of-00001.safetensors")
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        k: "model-00001-of-00001.safetensors" for k in ("proj_out.weight", "proj_out.bias")}}))
    save_file({"connectors.weight": torch.zeros(2)},
              str(model_dir / "diffusion_pytorch_model-00001-of-00001.safetensors"))
    (model_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"connectors.weight": "diffusion_pytorch_model-00001-of-00001.safetensors"}}))
    C.init_parallel_mesh(MeshSpec(tp=1))
    module = Fused()
    report = load_sharded_checkpoint(module, model_dir, device="cpu", dtype=torch.float32,
                                     rename=_rename, slices=SLICES)
    assert report == {"missing": [], "unexpected": []}
    torch.testing.assert_close(module.attn.weight, weight[:, :ATTN])
    torch.testing.assert_close(module.mlp.weight, weight[:, ATTN:])
    torch.testing.assert_close(module.attn.bias, bias)
