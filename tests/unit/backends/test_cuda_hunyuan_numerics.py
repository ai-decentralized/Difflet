"""Model-local BF16 rounding follows the original HunyuanVideo reference."""

import pytest
import torch


@pytest.fixture(autouse=True)
def isolated_cuda_mesh(monkeypatch):
    from difflet.backends.cuda.ops_impl import parallel_mesh

    monkeypatch.setattr(parallel_mesh, "_MESH_SPEC", None)
    monkeypatch.setattr(parallel_mesh, "_AXIS_GROUPS", {})


def test_existing_diffusers_norm_already_preserves_cast_before_affine():
    from diffusers.models.normalization import RMSNorm

    x = torch.tensor([[1.0078125, -0.2080078125]], dtype=torch.bfloat16)
    norm = RMSNorm(2, eps=1e-6).to(dtype=torch.bfloat16)
    norm.weight = torch.nn.Parameter(
        torch.tensor([0.74609375, 1.2890625], dtype=torch.bfloat16), requires_grad=False
    )
    expected = torch.tensor([[1.03125, -0.3671875]], dtype=torch.bfloat16)
    torch.testing.assert_close(norm(x), expected, atol=0, rtol=0)


@pytest.mark.parametrize(("backend", "expected_biased_rows"), [("cpu", 0), ("cuda", 6)])
def test_component_keeps_reference_norm_and_backend_specific_linears(
    tmp_path, backend, expected_biased_rows
):
    import os
    from pathlib import Path
    import subprocess
    import sys
    import textwrap

    # Tensor placement does not select the op bindings; isolate each binding in
    # a fresh interpreter so the fixture also works on an actual CUDA host.
    code = textwrap.dedent("""
        from pathlib import Path
        import sys
        from diffusers.models.normalization import RMSNorm
        from tests.unit.backends.test_cuda_hunyuan_transformer import _component

        _, component = _component(Path(sys.argv[1]))
        assert type(component.module.transformer_blocks[0].attn.norm_q) is RMSNorm
        assert all(p.device.type == 'cpu' and not p.is_meta
                   for p in component.module.parameters())
        assert component.numerics_swaps == {
            'projections': 1, 'biased_rows': int(sys.argv[2])}
    """)
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), str(expected_biased_rows)],
        env={**os.environ, "DIFFLET_BACKEND": backend, "PYTHONPATH": str(root)},
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_single_projection_restores_one_rounding_and_both_output_streams():
    from difflet.backends.cuda.hunyuan_video.numerics import fuse_single_projection
    from difflet.backends.cuda.ops_impl import parallel_mesh
    from difflet.models.hunyuan_video.modeling_hunyuan_video import (
        HunyuanVideoSingleTransformerBlock,
    )
    from difflet.pipeline.parallel_mesh import MeshSpec

    class EchoNorm(torch.nn.Module):
        def forward(self, x, emb):
            return x, torch.ones(x.shape[0], x.shape[-1], dtype=x.dtype)

    class EchoAttention(torch.nn.Module):
        def forward(self, hidden_states, encoder_hidden_states, **kwargs):
            return hidden_states, encoder_hidden_states

    parallel_mesh.init_parallel_mesh(MeshSpec(tp=1))
    block = HunyuanVideoSingleTransformerBlock(1, 2, mlp_ratio=1).to(torch.bfloat16).eval()
    block.norm, block.attn = EchoNorm(), EchoAttention()
    block.proj_mlp, block.act_mlp = torch.nn.Identity(), torch.nn.Identity()
    with torch.no_grad():
        block.proj_out_attn.weight.copy_(torch.tensor([[256.0, 1.0], [256.0, 1.0]]))
        block.proj_out_attn.bias.zero_()
        block.proj_out_mlp.weight.copy_(torch.tensor([[-256.0, 0.0], [-256.0, 0.0]]))
    x = torch.ones(1, 2, 2, dtype=torch.bfloat16)
    context = torch.ones(1, 1, 2, dtype=torch.bfloat16)
    temb = torch.zeros(1, 2, dtype=torch.bfloat16)
    full_weight = torch.cat([block.proj_out_attn.weight, block.proj_out_mlp.weight], 1)
    expected = torch.ones(1, 3, 2, dtype=torch.bfloat16) + torch.nn.functional.linear(
        torch.ones(1, 3, 4, dtype=torch.bfloat16), full_weight
    )
    with torch.no_grad():
        old = torch.cat(block(x, context, temb), 1)
        assert not torch.equal(old, expected)
        assert fuse_single_projection(block, tp_degree=1)
        actual = torch.cat(block(x, context, temb), 1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(block.proj_out.weight, full_weight, atol=0, rtol=0)
    assert block.proj_out_attn is None and block.proj_out_mlp is None
    assert not fuse_single_projection(block, tp_degree=1)


def test_projection_refuses_tp_before_any_mutation():
    import pytest
    from difflet.backends.cuda.hunyuan_video.numerics import fuse_single_projection

    block = torch.nn.Module()
    before = dict(vars(block))
    with pytest.raises(NotImplementedError, match="tp=1"):
        fuse_single_projection(block, tp_degree=2)
    assert vars(block) == before


def test_biased_row_projection_matches_one_bf16_rounding_and_retains_parameters():
    from difflet.backends.cuda.hunyuan_video.numerics import restore_reference_row_bias
    from difflet.backends.cuda.ops_impl import parallel_mesh
    from difflet.backends.cuda.ops_impl.linear import RowParallelLinear
    from difflet.pipeline.parallel_mesh import MeshSpec

    parallel_mesh.init_parallel_mesh(MeshSpec(tp=1))
    row = RowParallelLinear(2, 1, bias=True, input_is_parallel=True).to(torch.bfloat16).eval()
    with torch.no_grad():
        row.weight.copy_(torch.tensor([[256.0, 1.0]]))
        row.bias.fill_(-256.0)
    root = torch.nn.Sequential(row)
    weight, bias = row.weight, row.bias
    x = torch.ones(1, 2, dtype=torch.bfloat16)
    expected = torch.tensor([[1.0]], dtype=torch.bfloat16)
    assert not torch.equal(root(x), expected)
    assert restore_reference_row_bias(root, tp_degree=1) == 1
    torch.testing.assert_close(root(x), expected, atol=0, rtol=0)
    assert root[0].weight is weight and root[0].bias is bias
    assert root[0].training is False
    assert restore_reference_row_bias(root, tp_degree=1) == 0


def test_row_bias_swap_keeps_deferred_bias_and_partial_reduce_contracts():
    from difflet.backends.cuda.hunyuan_video.numerics import restore_reference_row_bias
    from difflet.backends.cuda.ops_impl import parallel_mesh
    from difflet.backends.cuda.ops_impl.linear import RowParallelLinear
    from difflet.pipeline.parallel_mesh import MeshSpec

    parallel_mesh.init_parallel_mesh(MeshSpec(tp=1))
    layers = [
        RowParallelLinear(2, 2, bias=False),
        RowParallelLinear(2, 2, skip_bias_add=True),
        RowParallelLinear(2, 2, reduce_output=False),
    ]
    root = torch.nn.Sequential(*layers)
    assert restore_reference_row_bias(root, tp_degree=1) == 0
    assert all(root[i] is original for i, original in enumerate(layers))
    assert isinstance(root[1](torch.ones(1, 2)), tuple)


def test_row_bias_refuses_tp_before_mutating_any_layer():
    import pytest
    from difflet.backends.cuda.hunyuan_video.numerics import restore_reference_row_bias

    root = torch.nn.Sequential(torch.nn.Linear(2, 2))
    original = root[0]
    with pytest.raises(NotImplementedError, match="tp=1"):
        restore_reference_row_bias(root, tp_degree=2)
    assert root[0] is original


def test_row_bias_swap_preserves_shared_module_aliases():
    from difflet.backends.cuda.hunyuan_video.numerics import restore_reference_row_bias
    from difflet.backends.cuda.ops_impl import parallel_mesh
    from difflet.backends.cuda.ops_impl.linear import RowParallelLinear
    from difflet.pipeline.parallel_mesh import MeshSpec

    parallel_mesh.init_parallel_mesh(MeshSpec(tp=1))
    row = RowParallelLinear(2, 1).to(torch.bfloat16)
    with torch.no_grad():
        row.weight.copy_(torch.tensor([[256.0, 1.0]]))
        row.bias.fill_(-256.0)
    root = torch.nn.Module()
    root.left = row
    root.right = row
    root.nested = torch.nn.Sequential(row)
    assert restore_reference_row_bias(root, tp_degree=1) == 1
    assert root.left is root.right is root.nested[0]
    x = torch.ones(1, 2, dtype=torch.bfloat16)
    for layer in (root.left, root.right, root.nested[0]):
        torch.testing.assert_close(layer(x), torch.ones(1, 1, dtype=x.dtype), rtol=0, atol=0)


def test_cuda_bound_bf16_transformer_matches_stock_with_matched_attention(tmp_path):
    """Fresh CUDA bindings on CPU expose bias rounding hidden by the CPU oracle's Linear."""
    import os
    from pathlib import Path
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        from pathlib import Path
        import sys, torch
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from difflet.backends.cuda.hunyuan_video.transformer import CudaHunyuanVideoTransformerApplication
        from tests.unit.backends.test_cuda_hunyuan_transformer import _checkpoint, _config, _inputs
        torch.set_num_threads(2)
        root = Path(sys.argv[1])
        ref, directory = _checkpoint(root)
        ref.to(torch.bfloat16)
        component = CudaHunyuanVideoTransformerApplication(
            model_path=str(directory), config=_config(torch_dtype=torch.bfloat16,
            joint_attention_mask_mode='compact'), device='cpu')
        component.compile(str(root/'manifest'))
        component.load(str(root/'manifest'), skip_warmup=True)
        for valid in (1, 3, 5):
            inputs = tuple(t.to(dtype=torch.bfloat16 if i not in (1,3) else t.dtype)
                           for i,t in enumerate(_inputs(valid)))
            with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                expected = ref(*inputs, return_dict=False)[0]
                actual = component(*inputs, return_dict=False)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert component.numerics_swaps == {'projections': 1, 'biased_rows': 6}
    """)
    root = Path(__file__).resolve().parents[3]
    env = {**os.environ, "DIFFLET_BACKEND": "cuda", "PYTHONPATH": str(root)}
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
