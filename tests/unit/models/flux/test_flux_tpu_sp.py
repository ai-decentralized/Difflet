"""CPU checks of the FLUX sequence-parallel forwards (backends/tpu/flux/sp.py).

The collective matmuls are JAX rings that only run on the chips; here they are
replaced by their tp=1 meaning (all-gather = identity, reduce-scatter = sum of the
matmuls), so the rewritten model / block forwards must reproduce diffusers' own
forward exactly. The single-block token order for tp>1 is checked on its own.
"""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn.functional as F

from difflet.backends.tpu.flux import sp
from difflet.backends.tpu.flux.config import TpuFluxConfig
from difflet.backends.tpu.ops_impl import collective_matmul as cm
from difflet.backends.tpu.ops_impl import parallel_mesh as pm
from difflet.models.flux.tp_sharding import build_flux_transformer
from difflet.pipeline.parallel_mesh import MeshSpec

from .test_flux_tp_sharding import _TINY, _inputs


@pytest.fixture(autouse=True)
def _mesh():
    # Resolved now, not at collection: sp.install / the SP forward import
    # parallel_mesh when they run, and another test in the same worker may have
    # dropped difflet.backends.tpu.* from sys.modules in between
    # (test_torch_megatron.py::test_weights_and_checkpoint_import_without_tpu),
    # leaving the module bound above a stale, never-initialized copy.
    mesh = importlib.import_module("difflet.backends.tpu.ops_impl.parallel_mesh")
    for module in {mesh, pm}:
        module.destroy_parallel_mesh()
        module.init_parallel_mesh(MeshSpec(tp=1))
    yield
    for module in {mesh, pm}:
        module.destroy_parallel_mesh()


def test_single_order_interleaves_each_ranks_text_then_image():
    assert sp.single_order(4, 8, 2).tolist() == [0, 1, 4, 5, 6, 7, 2, 3, 8, 9, 10, 11]
    assert sp.single_order(4, 8, 1).tolist() == list(range(12))


def _fake_rings(monkeypatch):
    def ag(x, linears):
        return [F.linear(x, l.weight, l.bias) for l in linears]

    def rs(x, linear):
        return F.linear(x, linear.weight, linear.bias)

    def rs2(x0, l0, x1, l1, bias=None):
        out = F.linear(x0, l0.weight) + F.linear(x1, l1.weight)
        return out if bias is None else out + bias

    monkeypatch.setattr(cm, "all_gather_matmul", ag)
    monkeypatch.setattr(cm, "matmul_reduce_scatter", rs)
    monkeypatch.setattr(cm, "matmul2_reduce_scatter", rs2)
    monkeypatch.setattr(cm, "prepare", lambda ops=None: None)
    monkeypatch.setattr(cm, "supports", lambda seq, tp: True)


@pytest.mark.parametrize("split_rope", [False, True])
def test_sp_forward_reproduces_diffusers_at_tp1(monkeypatch, split_rope):
    from difflet.backends.tpu.flux.rope import permute_flux_rope_layout_
    from difflet.models.flux import tp_sharding as S

    _fake_rings(monkeypatch)
    cfg = TpuFluxConfig(_TINY, height=32, width=48, tp_degree=1, torch_dtype=torch.float32)
    ref = build_flux_transformer(cfg).eval()
    model = build_flux_transformer(cfg).eval()
    model.load_state_dict(ref.state_dict())
    # The TP surgery shard_flux_transformer does at tp>1 (it is a no-op at tp=1).
    for block in model.single_transformer_blocks:
        dim = block.proj_out.out_features
        w, b = block.proj_out.weight.data.clone(), block.proj_out.bias.data.clone()
        block.proj_out_attn = S.RowParallelLinear(dim, dim, bias=True, input_is_parallel=True,
                                                  reduce_output=False, skip_bias_add=True)
        block.proj_out_mlp = S.RowParallelLinear(int(block.mlp_hidden_dim), dim, bias=False,
                                                 input_is_parallel=True, reduce_output=False)
        block.proj_out_attn.weight.data.copy_(w[:, :dim])
        block.proj_out_attn.bias.data.copy_(b)
        block.proj_out_mlp.weight.data.copy_(w[:, dim:])
        del block.proj_out
        block.attn.processor = S.FluxTPAttnProcessor()
    for block in model.transformer_blocks:
        block.attn.processor = S.FluxTPAttnProcessor()
    if split_rope:
        permute_flux_rope_layout_(model)
    hidden, text, pooled, t, img_ids, txt_ids, guidance = _inputs(cfg)
    assert sp.install(model, image_seq_len=hidden.shape[1], text_seq_len=text.shape[1])
    with torch.no_grad():
        expected = ref(hidden, text, pooled, t, img_ids, txt_ids, guidance, return_dict=False)[0]
        actual = model(hidden_states=hidden, encoder_hidden_states=text, pooled_projections=pooled,
                       timestep=t, img_ids=img_ids, txt_ids=txt_ids, guidance=guidance,
                       return_dict=False)[0]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_install_refuses_a_sequence_that_does_not_split(monkeypatch):
    monkeypatch.setattr(cm, "prepare", lambda ops=None: pytest.fail("must not build rings"))
    cfg = TpuFluxConfig(_TINY, height=32, width=48, tp_degree=1, torch_dtype=torch.float32)
    model = build_flux_transformer(cfg)
    before = model.forward
    assert not sp.install(model, image_seq_len=7, text_seq_len=5)
    assert model.forward == before
