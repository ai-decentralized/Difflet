"""A split linear in the quantized checkpoint: the single-stream blocks' fused ``proj_out`` stored
as its attention and MLP column ranges, each quantized per rank K range of its own when a target
names it and kept in the source's dtype otherwise, and the halves what the neuron loader's
slices read from the fused weight."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from difflet.quant import mx
from difflet.quant.checkpoint import write_quantized_dir
from difflet.quant.spec import QuantSpec
from difflet.quant.targets import SINGLE_BLOCK_SPLIT_LINEARS

SPLIT = SINGLE_BLOCK_SPLIT_LINEARS
HALVES = (r"single_transformer_blocks\.\d+\.proj_out_(attn|mlp)",)
MLP_UP = (r"single_transformer_blocks\.\d+\.proj_mlp",)
P = "single_transformer_blocks.0"


def _source(tmp_path, dtype=torch.bfloat16):
    """One single-stream block's checkpoint tensors in ``dtype``: the fused ``proj_out``
    ``[256, 1280]`` (256 attention + 1024 MLP input columns), its bias and the MLP-up."""
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(0)
    W = torch.randn(256, 1280, generator=g).to(dtype)
    b = torch.randn(256, generator=g).to(dtype)
    up = torch.randn(1024, 256, generator=g).to(dtype)
    src = tmp_path / "src"
    src.mkdir()
    save_file({f"{P}.proj_out.weight": W, f"{P}.proj_out.bias": b, f"{P}.proj_mlp.weight": up},
              str(src / "diffusion_pytorch_model.safetensors"))
    return src, W, b, up


def _tensors(directory) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    tensors: dict[str, torch.Tensor] = {}
    for path in sorted(Path(directory).glob("*.safetensors")):
        tensors.update(load_file(str(path)))
    return tensors


def test_a_split_linear_is_written_as_its_two_column_ranges(tmp_path):
    """Each half is quantized per rank K range of the half: rank r of a row-parallel
    ``proj_out_attn`` reads its columns ``[r K/tp, (r+1) K/tp)`` and the scales of exactly those
    columns, which slicing a per-rank quantization of the fused weight would not give."""
    src, W, b, up = _source(tmp_path)
    spec = QuantSpec(targets=HALVES, row_parallel=HALVES, split_linears=SPLIT, tp_degree=2)
    manifest = write_quantized_dir(src, tmp_path / "q", spec)
    t = _tensors(tmp_path / "q")
    assert f"{P}.proj_out.weight" not in t
    assert torch.equal(t[f"{P}.proj_out.bias"], b) and torch.equal(t[f"{P}.proj_mlp.weight"], up)
    for half, cols in (("proj_out_attn", W[:, :256]), ("proj_out_mlp", W[:, 256:])):
        q, s = t[f"{P}.{half}.weight"], t[f"{P}.{half}.weight_scale"]
        assert q.dtype == torch.float8_e4m3fn and q.shape == cols.shape
        k = cols.shape[1] // 2
        for r in range(2):  # what rank r's FP8RowParallelLinear reads (shard dim 1, both tensors)
            eq, es = mx.quantize_mx(cols[:, r * k:(r + 1) * k])
            assert torch.equal(q[:, r * k:(r + 1) * k], eq)
            assert torch.equal(s[:, r * k // 32:(r + 1) * k // 32], es)
    assert manifest["quantized_linears"] == [f"{P}.proj_out_attn", f"{P}.proj_out_mlp"]
    # quantizing the fused weight per rank K range and slicing gives another block membership
    assert not torch.equal(t[f"{P}.proj_out_attn.weight_scale"][:, :4],
                           mx.quantize_mx_sharded(W, 2)[1][:, :4])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_a_split_linear_that_no_target_names_keeps_the_source_dtype(tmp_path, dtype):
    """A target set without the halves still splits: they are stored as the fused weight's
    column ranges, uncast, so the loader reads them by name as it would have sliced them."""
    src, W, b, up = _source(tmp_path, dtype)
    spec = QuantSpec(targets=MLP_UP, split_linears=SPLIT)
    manifest = write_quantized_dir(src, tmp_path / "q", spec)
    t = _tensors(tmp_path / "q")
    attn, mlp = t[f"{P}.proj_out_attn.weight"], t[f"{P}.proj_out_mlp.weight"]
    assert attn.dtype == mlp.dtype == dtype
    assert torch.equal(attn, W[:, :256]) and torch.equal(mlp, W[:, 256:])
    assert not any(key.startswith(f"{P}.proj_out") and key.endswith(".weight_scale") for key in t)
    assert f"{P}.proj_out.weight" not in t and torch.equal(t[f"{P}.proj_out.bias"], b)
    assert manifest["quantized_linears"] == [f"{P}.proj_mlp"]
    assert manifest["tensors"] == {"quantized": 1, "kept": 3, "total": 5}


def test_a_split_pattern_that_matches_no_linear_weight_is_refused_before_writing(tmp_path):
    """A dead split pattern, a split linear without the MLP columns after the attention ones,
    and a target on the fused name of a split linear are each refused before anything is
    written: the directory would hold another layout than the spec claims."""
    src, *_ = _source(tmp_path)
    with pytest.raises(ValueError, match="split patterns match no linear weight") as exc:
        write_quantized_dir(src, tmp_path / "q",
                            QuantSpec(targets=MLP_UP, split_linears=(r"nothing\.proj_out",)))
    assert str(exc.value).endswith(r": nothing\.proj_out")
    assert list(tmp_path.iterdir()) == [src]  # nothing written, no temporary sibling
    with pytest.raises(ValueError, match="proj_mlp: a split linear needs K > N"):
        write_quantized_dir(src, tmp_path / "q", QuantSpec(targets=MLP_UP, split_linears=MLP_UP))
    assert list(tmp_path.iterdir()) == [src]
    fused = (r"single_transformer_blocks\.\d+\.proj_out",)
    with pytest.raises(ValueError, match="target patterns match no linear weight") as exc:
        write_quantized_dir(src, tmp_path / "q", QuantSpec(targets=fused, split_linears=SPLIT))
    assert str(exc.value).endswith(f": {fused[0]}")
    assert list(tmp_path.iterdir()) == [src]


@pytest.mark.parametrize("model", ["flux", "hunyuan_video"])
def test_the_split_law_is_what_the_neuron_loader_slices(model, tmp_path):
    """The halves the writer stores are, byte for byte, the column ranges the neuron FLUX and
    HunyuanVideo loaders slice from the fused ``proj_out.weight`` (``CheckpointSlice``), so a
    split directory loads them by name in place of the slices."""
    pytest.importorskip("diffusers")
    from difflet.backends._torch_megatron.checkpoint import build_weight_map
    from difflet.backends.neuron.core.checkpoint import (
        _read_checkpoint_slice,
        single_block_proj_out_slices,
    )

    if model == "flux":
        from tests.unit.backends._neuron_flux_toy import TINY_FP8, write_tiny_flux

        config, src = TINY_FP8, write_tiny_flux(tmp_path, config=TINY_FP8)
    else:
        from tests.unit.backends._neuron_hv1_toy import TINY_FP8_HV1, write_tiny_hv1

        config, src = TINY_FP8_HV1, write_tiny_hv1(tmp_path, config=TINY_FP8_HV1)
    num_single = config["num_single_layers"]
    inner = config["num_attention_heads"] * config["attention_head_dim"]
    write_quantized_dir(src, tmp_path / "q", QuantSpec(targets=MLP_UP, split_linears=SPLIT))
    stored, src_map = _tensors(tmp_path / "q"), build_weight_map(src)
    slices = single_block_proj_out_slices(num_single, inner, 4 * inner)
    assert len(slices) == 2 * num_single
    for name, sl in slices.items():
        expect = _read_checkpoint_slice(src_map[sl.key], sl, shard=None, tp_size=1, tp_rank=0)
        assert torch.equal(stored[name], expect), name
    for i in range(num_single):
        assert f"single_transformer_blocks.{i}.proj_out.bias" in stored
        assert f"single_transformer_blocks.{i}.proj_out.weight" not in stored
