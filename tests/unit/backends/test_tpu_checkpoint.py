"""Unit tests for the lazy safetensors shard loader (plan Phase 4).

The point of this loader is that a rank reads only its own slice off disk —
four ranks each materializing Qwen-Image's 38 GiB transformer would need
~152 GiB of host RAM. These tests check the slicing is correct; the memory
behaviour is a property of safetensors' slice API, exercised for real on the
full checkpoint.
"""

import json

import pytest
import torch
import torch.nn as nn

from difflet.backends.tpu.core import checkpoint as ckpt
from difflet.backends.tpu.ops_impl import linear as L
from difflet.backends.tpu.ops_impl import parallel_mesh as pm
from difflet.pipeline.parallel_mesh import MeshSpec

safetensors_torch = pytest.importorskip("safetensors.torch")


@pytest.fixture(autouse=True)
def _clean_mesh():
    pm.destroy_parallel_mesh()
    yield
    pm.destroy_parallel_mesh()


FULL = {
    "up.weight": torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8),
    "up.bias": torch.arange(16, dtype=torch.float32),
    "down.weight": torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16),
    "down.bias": torch.arange(8, dtype=torch.float32),
}


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = L.ColumnParallelLinear(8, 16, bias=True, gather_output=False)
        self.down = L.RowParallelLinear(16, 8, bias=True, input_is_parallel=True)


def _write_single(tmp_path):
    safetensors_torch.save_file(FULL, str(tmp_path / "model.safetensors"))
    return tmp_path


def _write_sharded(tmp_path):
    a = {k: v for k, v in FULL.items() if k.startswith("up.")}
    b = {k: v for k, v in FULL.items() if k.startswith("down.")}
    safetensors_torch.save_file(a, str(tmp_path / "model-00001-of-00002.safetensors"))
    safetensors_torch.save_file(b, str(tmp_path / "model-00002-of-00002.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"total_size": 0},
            "weight_map": {
                **{k: "model-00001-of-00002.safetensors" for k in a},
                **{k: "model-00002-of-00002.safetensors" for k in b},
            },
        })
    )
    return tmp_path


def test_weight_map_from_a_single_file(tmp_path):
    got = ckpt.build_weight_map(_write_single(tmp_path))
    assert set(got) == set(FULL)


def test_weight_map_from_an_index(tmp_path):
    got = ckpt.build_weight_map(_write_sharded(tmp_path))
    assert set(got) == set(FULL)
    assert got["up.weight"].endswith("model-00001-of-00002.safetensors")
    assert got["down.weight"].endswith("model-00002-of-00002.safetensors")


def test_missing_checkpoint_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="no .safetensors"):
        ckpt.build_weight_map(tmp_path)


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("layout", ["single", "sharded"])
def test_each_rank_loads_exactly_its_own_slice(tmp_path, rank, layout):
    pm.init_parallel_mesh(MeshSpec(tp=4))
    directory = _write_single(tmp_path) if layout == "single" else _write_sharded(tmp_path)
    block = _Block()
    ckpt.load_checkpoint_into(block, directory, tp_size=4, tp_rank=rank)

    # Column-parallel splits the output dim; row-parallel the input dim; the
    # row-parallel bias is replicated because it is added after the all-reduce.
    assert torch.equal(block.up.weight, FULL["up.weight"][rank * 4 : (rank + 1) * 4])
    assert torch.equal(block.up.bias, FULL["up.bias"][rank * 4 : (rank + 1) * 4])
    assert torch.equal(block.down.weight, FULL["down.weight"][:, rank * 4 : (rank + 1) * 4])
    assert torch.equal(block.down.bias, FULL["down.bias"])


def test_ranks_together_reconstruct_the_full_weight(tmp_path):
    # The real invariant: the shards must tile the original exactly, with no
    # gap and no overlap. Checking one rank in isolation would not catch an
    # off-by-one in the stride.
    pm.init_parallel_mesh(MeshSpec(tp=4))
    directory = _write_single(tmp_path)
    pieces = []
    for rank in range(4):
        block = _Block()
        ckpt.load_checkpoint_into(block, directory, tp_size=4, tp_rank=rank)
        pieces.append(block.up.weight.clone())
    assert torch.equal(torch.cat(pieces, dim=0), FULL["up.weight"])


def test_prefix_is_stripped_to_form_checkpoint_keys(tmp_path):
    # difflet's trace modules nest the diffusers model under `transformer.`,
    # so the checkpoint keys have no such prefix.
    pm.init_parallel_mesh(MeshSpec(tp=1))

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = _Block()

    ckpt.load_checkpoint_into(
        Wrapper(), _write_single(tmp_path), tp_size=1, tp_rank=0, prefix="transformer."
    )


def test_missing_parameters_are_reported_not_silently_skipped(tmp_path):
    pm.init_parallel_mesh(MeshSpec(tp=1))
    partial = {k: v for k, v in FULL.items() if not k.startswith("down.")}
    safetensors_torch.save_file(partial, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="no checkpoint entry"):
        ckpt.load_checkpoint_into(_Block(), tmp_path, tp_size=1, tp_rank=0)

    report = ckpt.load_checkpoint_into(
        _Block(), tmp_path, tp_size=1, tp_rank=0, strict=False
    )
    assert sorted(report["missing"]) == ["down.bias", "down.weight"]


def test_shape_mismatch_is_an_error(tmp_path):
    pm.init_parallel_mesh(MeshSpec(tp=4))
    bad = dict(FULL)
    bad["up.weight"] = torch.zeros(20, 8)  # not the module's 16x8
    safetensors_torch.save_file(bad, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="!= parameter"):
        ckpt.load_checkpoint_into(_Block(), tmp_path, tp_size=4, tp_rank=0)


def test_indivisible_weight_is_rejected(tmp_path):
    pm.init_parallel_mesh(MeshSpec(tp=4))

    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = L.ColumnParallelLinear(8, 16, bias=False, gather_output=False)

    bad = {"up.weight": torch.zeros(18, 8)}
    safetensors_torch.save_file(bad, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="not divisible"):
        ckpt.load_checkpoint_into(Odd(), tmp_path, tp_size=4, tp_rank=0)


def test_dtype_conversion_on_load(tmp_path):
    pm.init_parallel_mesh(MeshSpec(tp=1))
    block = _Block().to(torch.bfloat16)
    ckpt.load_checkpoint_into(block, _write_single(tmp_path), tp_size=1, tp_rank=0)
    assert block.up.weight.dtype == torch.bfloat16


def test_non_persistent_buffers_are_not_expected_in_the_checkpoint(tmp_path):
    # `persistent=False` means "not part of the state dict" — such a buffer is
    # computed, never stored, so its absence from a checkpoint is correct and
    # must not be reported as missing. Qwen-Image's static RoPE is exactly this.
    pm.init_parallel_mesh(MeshSpec(tp=1))

    class WithComputedBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = L.ColumnParallelLinear(8, 16, bias=True, gather_output=False)
            self.down = L.RowParallelLinear(16, 8, bias=True, input_is_parallel=True)
            self.register_buffer("rope", torch.ones(4), persistent=False)
            self.register_buffer("kept", torch.zeros(2), persistent=True)

    safetensors_torch.save_file(
        {**FULL, "kept": torch.ones(2)}, str(tmp_path / "model.safetensors")
    )
    module = WithComputedBuffer()
    report = ckpt.load_checkpoint_into(module, tmp_path, tp_size=1, tp_rank=0)
    assert report["missing"] == []
    # the computed buffer keeps its computed value, untouched by the load
    assert torch.equal(module.rope, torch.ones(4))
    assert torch.equal(module.kept, torch.ones(2))


def test_rename_maps_module_names_to_checkpoint_keys(tmp_path):
    """Wan's FFN attribute names diverge from diffusers; the hook bridges it.

    The rename runs module -> checkpoint, so a parameter that exists has
    exactly one place to come from.
    """
    renamed = {f"ffn.net.0.proj.{leaf}": FULL[f"up.{leaf}"] for leaf in ("weight", "bias")}
    renamed.update({f"ffn.net.2.{leaf}": FULL[f"down.{leaf}"] for leaf in ("weight", "bias")})
    safetensors_torch.save_file(renamed, str(tmp_path / "model.safetensors"))

    class _Ffn(nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = nn.Module()
            self.ffn.net_in = L.ColumnParallelLinear(8, 16, bias=True, gather_output=False)
            self.ffn.net_out = L.RowParallelLinear(16, 8, bias=True, input_is_parallel=True)

    pm.init_parallel_mesh(MeshSpec(tp=4))

    def rename(name):
        return name.replace(".net_in.", ".net.0.proj.").replace(".net_out.", ".net.2.")

    module = _Ffn()
    report = ckpt.load_checkpoint_into(
        module, tmp_path, tp_size=4, tp_rank=1, rename=rename
    )
    assert report == {"missing": [], "unexpected": []}
    # Sharding still follows _difflet_shard, not the renamed key.
    assert torch.equal(module.ffn.net_in.weight, FULL["up.weight"][4:8])
    assert torch.equal(module.ffn.net_out.weight, FULL["down.weight"][:, 4:8])


def test_without_rename_the_same_module_reports_every_key_missing(tmp_path):
    """The hook is load-bearing: without it nothing resolves, loudly."""
    renamed = {f"ffn.net.0.proj.{leaf}": FULL[f"up.{leaf}"] for leaf in ("weight", "bias")}
    renamed.update({f"ffn.net.2.{leaf}": FULL[f"down.{leaf}"] for leaf in ("weight", "bias")})
    safetensors_torch.save_file(renamed, str(tmp_path / "model.safetensors"))

    class _Ffn(nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = nn.Module()
            self.ffn.net_in = L.ColumnParallelLinear(8, 16, bias=True, gather_output=False)

    pm.init_parallel_mesh(MeshSpec(tp=4))
    with pytest.raises(ValueError, match="had no checkpoint entry"):
        ckpt.load_checkpoint_into(_Ffn(), tmp_path, tp_size=4, tp_rank=0)


# --- CheckpointSlice: one checkpoint tensor feeding two row-parallel params ---


class _FusedOut(nn.Module):
    """HunyuanVideo's single-stream block shape: upstream stores one
    ``proj_out`` over cat([attn, mlp]); the modeling keeps two row-parallel
    projections, one per input stream, so each matches its own sharding."""

    def __init__(self):
        super().__init__()
        self.proj_out_attn = L.RowParallelLinear(8, 4, bias=True, input_is_parallel=True)
        self.proj_out_mlp = L.RowParallelLinear(16, 4, bias=False, input_is_parallel=True)


_FUSED = {
    "proj_out.weight": torch.arange(4 * 24, dtype=torch.float32).reshape(4, 24),
    "proj_out.bias": torch.arange(4, dtype=torch.float32),
}


def _fused_rename(name: str):
    if name == "proj_out_attn.weight":
        return ckpt.CheckpointSlice("proj_out.weight", dim=1, start=0, stop=8)
    if name == "proj_out_mlp.weight":
        return ckpt.CheckpointSlice("proj_out.weight", dim=1, start=8, stop=None)
    if name == "proj_out_attn.bias":
        return "proj_out.bias"
    return name


@pytest.mark.parametrize("rank", range(4))
def test_checkpoint_slice_windows_then_shards(tmp_path, rank):
    pm.init_parallel_mesh(MeshSpec(tp=4))
    safetensors_torch.save_file(_FUSED, str(tmp_path / "model.safetensors"))
    block = _FusedOut()
    ckpt.load_checkpoint_into(block, tmp_path, tp_size=4, tp_rank=rank, rename=_fused_rename)

    full = _FUSED["proj_out.weight"]
    # attn half: columns 0..8, this rank's 2 of them; mlp half: 8..24, 4 of them.
    assert torch.equal(block.proj_out_attn.weight, full[:, rank * 2 : (rank + 1) * 2])
    assert torch.equal(block.proj_out_mlp.weight, full[:, 8 + rank * 4 : 8 + (rank + 1) * 4])
    assert torch.equal(block.proj_out_attn.bias, _FUSED["proj_out.bias"])


def test_checkpoint_slice_tiles_the_source_exactly(tmp_path):
    pm.init_parallel_mesh(MeshSpec(tp=4))
    safetensors_torch.save_file(_FUSED, str(tmp_path / "model.safetensors"))
    attn, mlp = [], []
    for rank in range(4):
        block = _FusedOut()
        ckpt.load_checkpoint_into(block, tmp_path, tp_size=4, tp_rank=rank, rename=_fused_rename)
        attn.append(block.proj_out_attn.weight.clone())
        mlp.append(block.proj_out_mlp.weight.clone())
    assert torch.equal(torch.cat(attn + mlp, dim=1), _FUSED["proj_out.weight"])
