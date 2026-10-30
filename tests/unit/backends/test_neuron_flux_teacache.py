"""Flux TeaCache on the neuron backend (CPU, eager): the probe reproduces the Trainium probe's
signal, the probe path leaves outputs unchanged, and a probe-free mode skips steps."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")

from difflet.backends.neuron.flux.config import NeuronFluxConfig  # noqa: E402
from difflet.backends.neuron.flux.teacache import NeuronFluxTeaCacheProbe  # noqa: E402
from difflet.backends.neuron.flux.transformer import NeuronFluxTransformerApplication  # noqa: E402
from difflet.backends.neuron.ops_impl import collectives as C  # noqa: E402
from difflet.models.flux.neuron_application import TorchNeuronFluxApplication  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.unit.backends._neuron_flux_toy import (  # noqa: E402
    PIPE_HW,
    TINY,
    TINY_SHAPE,
    diffusers_pipeline_reference,
    pipeline_kwargs,
    write_tiny_flux,
    write_tiny_pipeline,
)


@pytest.fixture(autouse=True)
def _reset_mesh():
    C.destroy_parallel_mesh()
    yield
    C.destroy_parallel_mesh()


def _probe_inputs(step):
    g = torch.Generator().manual_seed(10 + step)
    seq = (TINY_SHAPE["height"] // 16) * (TINY_SHAPE["width"] // 16)
    return (torch.randn(1, seq, TINY["in_channels"], generator=g), torch.tensor([1.0 - 0.2 * step]),
            torch.randn(1, TINY["pooled_projection_dim"], generator=g), torch.tensor([3.5]))


def test_probe_signal_matches_diffusers_modules(tmp_path):
    from diffusers import FluxTransformer2DModel

    path = write_tiny_flux(tmp_path)
    config = NeuronFluxConfig.from_pretrained(path, **TINY_SHAPE)
    app = NeuronFluxTransformerApplication(model_path=path, config=config, dtype=torch.float32,
                                           exec_mode="eager", device="cpu")
    app.load()
    probe = NeuronFluxTeaCacheProbe(app)
    ref = FluxTransformer2DModel.from_pretrained(path).eval()
    prev = None
    for step in range(3):
        x, t, pooled, g = _probe_inputs(step)
        ours = float(probe.teacache_delta(x, t, pooled, g))
        with torch.no_grad():
            temb = ref.time_text_embed(t * 1000, g * 1000, pooled)
            m = ref.transformer_blocks[0].norm1(ref.x_embedder(x), emb=temb)[0]
        p = prev if prev is not None else torch.zeros_like(m)
        expected = float((m - p).abs().mean() / (p.abs().mean() + 1e-8))
        if step == 0:  # against zeros: a huge number; only its scale is meaningful
            assert ours == pytest.approx(expected, rel=1e-4)
        else:
            assert ours == pytest.approx(expected, rel=1e-4, abs=1e-6)
        prev = m


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return write_tiny_pipeline(tmp_path_factory.mktemp("fluxtc"))


def _app(root, **options):
    a = TorchNeuronFluxApplication(model_path=root, parallel=DiffletParallelConfig(tp_degree=1),
                                   dtype=torch.float32, shape={"height": PIPE_HW, "width": PIPE_HW},
                                   device="cpu", exec_mode="eager", text_seq_len=8,
                                   clip_dtype=torch.float32, vae_dtype=torch.float32, **options)  # the fp32 oracle
    a.load()
    return a


def test_probe_without_a_controller_never_skips_and_matches_diffusers(root):
    app = _app(root, teacache_fused=True)
    assert isinstance(app.pipeline.teacache_probe, NeuronFluxTeaCacheProbe)
    assert app.pipeline.teacache_controller is None
    ours = app(**pipeline_kwargs("latent")).images
    torch.testing.assert_close(ours, diffusers_pipeline_reference(root, "latent"), atol=1e-4, rtol=1e-4)


def test_probe_free_cadence_skips_steps(root):
    app = _app(root, teacache_cadence=2)
    assert app.pipeline.teacache_probe is None
    controller = app.pipeline.teacache_controller
    kwargs = dict(pipeline_kwargs("latent"), num_inference_steps=12)
    out = app(**kwargs).images
    assert torch.isfinite(out).all() and controller.skipped_steps > 0


def test_adaptive_and_probe_free_modes_are_exclusive(root):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _app(root, teacache_fused=True, teacache_cadence=2)
