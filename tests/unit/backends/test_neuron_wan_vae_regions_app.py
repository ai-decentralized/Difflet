"""NeuronWanVaeDecoderApplication's compiled decoder (``exec_mode="compile"``) on the CPU.

The compile mode decodes through ``WanVAERegionDecoder`` (``models/wan/vae/regions.py``) with
each region compiled on its own; here with ``aot_eager`` or a counting backend, on one CPU rank,
from the tiny diffusers checkpoint of the W1 tests. The default stays the eager decoder.
"""

from __future__ import annotations

import math
import os

import pytest

torch = pytest.importorskip("torch")

from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from tests.unit.backends._neuron_wan_toy import (  # noqa: E402
    TINY_LATENT_SHAPE,
    _write_vae,
    tiny_wan_host_decode,
)

#: The tiny VAE's un-normalisation (TINY_VAE's latents_mean / latents_std).
_MEAN = torch.tensor([0.1, -0.2, 0.05, 0.3]).view(1, 4, 1, 1, 1)
_STD = torch.tensor([1.5, 0.8, 1.2, 2.0]).view(1, 4, 1, 1, 1)


@pytest.fixture(autouse=True)
def _clean_mesh(monkeypatch):
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def vae_dir(tmp_path_factory):
    pytest.importorskip("safetensors.torch")
    root = tmp_path_factory.mktemp("tiny_wan_vae")
    _write_vae(root)
    return root


def _app(vae_dir, latent_shape=TINY_LATENT_SHAPE, **kwargs):
    from difflet.backends.neuron.wan.vae import NeuronWanVaeDecoderApplication

    parallel_mesh.destroy_parallel_mesh()
    return NeuronWanVaeDecoderApplication(
        model_path=os.path.join(vae_dir, "vae"), latent_shape=latent_shape,
        parallel=DiffletParallelConfig(), device="cpu", **kwargs)


def _latents(shape=TINY_LATENT_SHAPE, seed=11):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _psnr(actual, reference):
    mse = float(((actual.double() - reference.double()) ** 2).mean())
    return math.inf if mse == 0 else 10 * math.log10(4.0 / mse)


def test_exec_mode_defaults_to_eager_and_refuses_unknown_modes(vae_dir):
    assert _app(vae_dir).exec_mode == "eager"
    assert _app(vae_dir, exec_mode=None).exec_mode == "eager"
    assert _app(vae_dir, exec_mode="compile").exec_mode == "compile"
    with pytest.raises(ValueError, match="eager or compile"):
        _app(vae_dir, exec_mode="aot")


def test_eager_mode_keeps_the_whole_model_as_its_module(vae_dir):
    from difflet.models.wan.vae.modeling_vae import WanVAEDecoderModel

    app = _app(vae_dir)
    app.load(skip_warmup=True)
    assert type(app.module) is WanVAEDecoderModel


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_decode_equals_the_eager_decode_and_the_host_decode(vae_dir, dtype):
    """fp32: the compiled regions decode as the eager decoder and diffusers' host decode do.
    bf16: fp32 frames (the output head) within rounding of the eager bf16 decode."""
    from difflet.models.wan.vae.regions import WanVAERegionDecoder

    latents = _latents()
    z = (latents * _STD + _MEAN).to(dtype)
    eager = _app(vae_dir, dtype=dtype)
    eager.load(skip_warmup=True)
    want = eager(z)
    app = _app(vae_dir, dtype=dtype, exec_mode="compile", compile_backend="aot_eager")
    app.load(skip_warmup=True)
    assert isinstance(app.module, WanVAERegionDecoder)
    assert {p.dtype for p in app.module.parameters()} == {dtype}
    assert app.load_report["missing"] == [] and app.load_report["unexpected"] == []
    got = app(z)
    assert got.device.type == "cpu" and got.dtype == torch.float32
    assert got.shape == want.shape == (1, 3, 5, 32, 48)
    if dtype is torch.float32:
        torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-4)
        torch.testing.assert_close(got, tiny_wan_host_decode(vae_dir, latents), atol=1e-4,
                                   rtol=1e-4)
    else:
        assert _psnr(got, want) >= 50.0


@pytest.mark.parametrize("frames, warm", [(1, 1), (2, 2), (5, 2)])
def test_the_warmup_decodes_at_most_two_latent_frames_and_builds_every_graph(vae_dir, frames,
                                                                              warm):
    """The first latent frame and one later frame build every graph a decode of any length runs,
    so the warm-up decodes those (zeros), not the whole clip; a decode at the request's shape
    after it builds none and is not reported as an unwarmed shape."""
    import torch._dynamo

    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    shape = (1, 4, frames, 4, 6)
    app = _app(vae_dir, latent_shape=shape, exec_mode="compile", compile_backend=backend)
    assert app.get_example_inputs()[0].shape == (1, 4, warm, 4, 6)
    app.load()
    assert app.warmup_shapes == [(1, 4, warm, 4, 6)]
    built = len(graphs)
    assert built > 0
    out = app(_latents(shape))
    assert out.shape[2] == 1 + 4 * (frames - 1)
    assert len(graphs) == built and app.unwarmed_shapes == []


def test_compile_mode_raises_dynamos_recompile_limit_for_the_region_shapes(vae_dir):
    import torch._dynamo

    with torch._dynamo.config.patch(recompile_limit=8):
        app = _app(vae_dir, exec_mode="compile", compile_backend="aot_eager")
        app.load(skip_warmup=True)
        assert torch._dynamo.config.recompile_limit >= app.RECOMPILE_LIMIT >= 32
