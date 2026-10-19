"""The Flux DiT on the neuron backend (CPU, eager) against diffusers' FluxTransformer2DModel."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")

from difflet.backends.neuron.flux.config import NeuronFluxConfig  # noqa: E402
from difflet.backends.neuron.flux.transformer import (  # noqa: E402
    NeuronFluxTransformerApplication,
    checkpoint_key,
)
from difflet.backends.neuron.ops_impl import collectives as C  # noqa: E402
from tests.unit.backends._neuron_flux_toy import (  # noqa: E402
    TINY_SHAPE,
    diffusers_reference,
    first,
    tiny_inputs,
    write_tiny_flux,
)


@pytest.fixture(autouse=True)
def _reset_mesh():
    C.destroy_parallel_mesh()
    yield
    C.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return write_tiny_flux(tmp_path_factory.mktemp("flux"))


def test_checkpoint_mapping():
    assert checkpoint_key("single_transformer_blocks.3.proj_out_attn.bias") == \
        "single_transformer_blocks.3.proj_out.bias"
    assert checkpoint_key("transformer_blocks.0.attn.to_q.weight") == "transformer_blocks.0.attn.to_q.weight"
    app = NeuronFluxTransformerApplication.__new__(NeuronFluxTransformerApplication)
    app.config = NeuronFluxConfig()
    slices = app.checkpoint_slices()
    assert len(slices) == 2 * 38
    attn = slices["single_transformer_blocks.0.proj_out_attn.weight"]
    mlp = slices["single_transformer_blocks.0.proj_out_mlp.weight"]
    assert (attn.key, attn.dim, attn.start, attn.length) == ("single_transformer_blocks.0.proj_out.weight", 1, 0, 3072)
    assert (mlp.start, mlp.length) == (3072, 4 * 3072)


def test_config_rejects_what_flux_does_not_support():
    with pytest.raises(NotImplementedError, match="guidance-distilled"):
        NeuronFluxConfig(cfg_parallel_enabled=True).validate()
    with pytest.raises(ValueError, match="does not divide tp=5"):
        NeuronFluxConfig(tp_degree=5).validate()
    with pytest.raises(ValueError, match="mutually exclusive"):
        NeuronFluxConfig(sp_enabled=True, context_parallel_enabled=True).validate()
    NeuronFluxConfig().validate()


def test_nxd_explicit_tp_group_api():
    # Flux's single-stream blocks all-reduce with process_group=layer.tensor_parallel_group.
    from difflet.backends.neuron.ops_impl.linear import ColumnParallelLinear, RowParallelLinear
    from difflet.pipeline.parallel_mesh import MeshSpec

    C.init_parallel_mesh(MeshSpec(tp=1))
    for layer in (ColumnParallelLinear(4, 4), RowParallelLinear(4, 4)):
        assert layer.tensor_parallel_group is C.get_tp_group()  # None at TP1: "the TP group"
    t = torch.randn(3)
    assert C.reduce_from_tensor_model_parallel_region(t, process_group=RowParallelLinear(4, 4).tensor_parallel_group) is t


def test_tp1_matches_diffusers(tiny):
    config = NeuronFluxConfig.from_pretrained(tiny, **TINY_SHAPE)
    app = NeuronFluxTransformerApplication(model_path=tiny, config=config, dtype=torch.float32,
                                           exec_mode="eager", device="cpu")
    app.load()
    assert app.load_report == {"missing": [], "unexpected": []}
    out = first(app(**tiny_inputs(config)))
    torch.testing.assert_close(out, diffusers_reference(tiny, config), atol=1e-4, rtol=1e-4)


def test_tp4_on_four_ranks_matches_diffusers(tiny):
    from tests.unit.backends._neuron_gloo import run_ranks
    from tests.unit.backends._neuron_workers import flux_dit_worker

    config = NeuronFluxConfig.from_pretrained(tiny, **TINY_SHAPE)
    reference = diffusers_reference(tiny, config)
    results = run_ranks(flux_dit_worker, tiny, world_size=4)
    for r in results:
        assert r["report"] == {"missing": [], "unexpected": []}
        torch.testing.assert_close(torch.tensor(r["out"]), reference, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("tp, cp, cp_mode, sp", [
    (2, 2, "gather_kv", False), (2, 2, "ring", False), (2, 2, "ulysses", False),
    (4, 1, "gather_kv", True), (2, 1, "gather_kv", True),
], ids=["tp2-cp2-gather_kv", "tp2-cp2-ring", "tp2-cp2-ulysses", "tp4-sp", "tp2-sp"])
def test_parallel_layouts_match_diffusers(tiny, tp, cp, cp_mode, sp):
    """CP shards both streams ([text || image] per rank) and attends with gather_kv, ring or
    ulysses; SP shards the image stream across the TP group through the dual-stream blocks."""
    from tests.unit.backends._neuron_gloo import run_ranks
    from tests.unit.backends._neuron_workers import flux_dit_parallel_worker

    config = NeuronFluxConfig.from_pretrained(tiny, **TINY_SHAPE)
    reference = diffusers_reference(tiny, config)
    results = run_ranks(flux_dit_parallel_worker, tiny, tp, cp, cp_mode, sp, world_size=tp * cp)
    for r in results:
        assert r["report"] == {"missing": [], "unexpected": []}
        torch.testing.assert_close(torch.tensor(r["out"]), reference, atol=1e-4, rtol=1e-4)


def test_cp_needs_both_streams_divisible():
    with pytest.raises(ValueError, match="text_seq_len=510 is not divisible by cp_degree=4"):
        NeuronFluxConfig(text_seq_len=510, cp_degree=4, context_parallel_enabled=True).validate()
