"""Unit tests for difflet.models.wan.vae.regions: the Wan VAE decoder as fixed-shape regions.

The region decoder must decode exactly as diffusers' untiled ``AutoencoderKLWan.decode`` (the
host reference of the neuron decoder's 50 dB gate) and as ``WanVAEDecoderModel`` (the eager
neuron path), for every latent frame count, with one set of graphs for the first latent frame
and one for every later frame. fp32 on the CPU; the compiled cases use a counting backend that
runs the traced graph as is. Nothing touches the device.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]

#: The real stage layout at toy width: the mid block's attention, two temporal and three spatial
#: upsamples; a (1, 4, T, 4, 6) latent decodes to 1 + 4 (T - 1) frames of 32 x 48.
TINY = dict(base_dim=8, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=1,
            temperal_downsample=[False, True, True])
LATENT_HW = (4, 6)


def _diffusers_vae(seed: int, **overrides):
    from diffusers import AutoencoderKLWan

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        vae = AutoencoderKLWan(**{**TINY, "attn_scales": [], **overrides}).eval()
        # diffusers initialises the RMS-norm gammas to one; random ones exercise the weights.
        with torch.no_grad():
            for name, p in vae.named_parameters():
                if name.endswith("gamma"):
                    p.copy_(1 + 0.2 * torch.randn_like(p))
    return vae


def _model_from(vae, *, fp32_output_head: bool = True, dtype=torch.float32):
    from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig, WanVAEDecoderModel

    config = WanVAEDecoderConfig.from_diffusers_dict(dict(vae.config))
    model = WanVAEDecoderModel(config, fp32_output_head=fp32_output_head).eval()
    state = {k: v for k, v in vae.state_dict().items()
             if k.startswith(("decoder.", "post_quant_conv."))}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected
    return model.to(dtype)


def _latents(frames: int, seed: int = 1) -> torch.Tensor:
    return torch.randn(1, TINY["z_dim"], frames, *LATENT_HW,
                       generator=torch.Generator().manual_seed(seed))


def _psnr(actual: torch.Tensor, reference: torch.Tensor) -> float:
    mse = float(((actual.double() - reference.double()) ** 2).mean())
    return math.inf if mse == 0 else 10 * math.log10(4.0 / mse)


def test_regions_imports_only_from_allowed_modules():
    tree = ast.parse((_REPO_ROOT / "difflet/models/wan/vae/regions.py").read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            roots.add((node.module or "").split(".", 1)[0])
    assert roots <= {"__future__", "torch", "typing", "collections", "difflet"}, roots
    assert not {"neuronx_distributed", "nkilib", "torch_neuronx"} & roots


@pytest.mark.parametrize("frames", [1, 2, 3, 5])
def test_region_decode_equals_diffusers_untiled_decode(frames):
    """fp32: the regions' decode is diffusers' ``AutoencoderKLWan.decode`` (untiled, the host
    reference) and the eager model's, for one latent frame (Qwen-Image) and several (Wan)."""
    from difflet.models.wan.vae.regions import WanVAERegionDecoder

    vae = _diffusers_vae(0)
    model = _model_from(vae)
    regions = WanVAERegionDecoder(model)
    z = _latents(frames)
    with torch.no_grad():
        want = vae.decode(z, return_dict=False)[0]
        eager = model(z)
        got = regions(z)
    assert got.shape == want.shape == (1, 3, 1 + 4 * (frames - 1), 32, 48)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(got, eager, atol=2e-5, rtol=1e-4)
    assert got.abs().max() <= 1.0


def test_region_decode_with_two_resnets_per_block_and_a_wide_frame():
    """Wan's real ``num_res_blocks`` (2, three resnets per up block, the first with a shortcut)
    on a latent wider than it is tall."""
    from difflet.models.wan.vae.regions import WanVAERegionDecoder

    vae = _diffusers_vae(3, num_res_blocks=2)
    model = _model_from(vae)
    z = torch.randn(1, 4, 3, 3, 7, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        want = vae.decode(z, return_dict=False)[0]
        got = WanVAERegionDecoder(model)(z)
    torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("seed", [0, 1])
def test_bf16_region_decode_is_the_eager_bf16_decode_within_rounding(seed):
    """bf16 weights and activations with the fp32 output head, as the neuron opt-in runs: the
    regions return fp32 frames within bf16 rounding of the eager bf16 decode (they differ only in
    the order of a few sums) and within bf16's error of the fp32 decode."""
    from difflet.models.wan.vae.regions import WanVAERegionDecoder

    vae = _diffusers_vae(seed)
    fp32 = _model_from(vae)
    bf16 = _model_from(vae, dtype=torch.bfloat16)
    z = _latents(3, seed=seed + 20)
    with torch.no_grad():
        reference = fp32(z)
        eager = bf16(z.to(torch.bfloat16))
        got = WanVAERegionDecoder(bf16)(z.to(torch.bfloat16))
    assert got.dtype == eager.dtype == torch.float32
    assert _psnr(got, eager) >= 50.0
    assert _psnr(got, reference) >= 40.0


def test_regions_lists_the_head_then_every_part_in_call_order():
    from difflet.models.wan.vae import regions as R

    model = _model_from(_diffusers_vae(0))
    decoder = R.WanVAERegionDecoder(model)
    regions = decoder.regions()
    assert regions[0] is decoder.head
    n_resnets = sum(isinstance(s, R._Resnet) for s in decoder.steps)
    n_parts = sum(len(s.regions()) for s in decoder.steps)
    assert n_resnets == 2 + 4 * 2  # the mid block's two, two per up block
    assert len(regions) == 1 + n_parts
    assert all(isinstance(r, torch.nn.Module) for r in regions)
    # the weights are the model's own: the decoder adds no parameters
    assert {id(p) for p in decoder.parameters()} == {id(p) for p in model.parameters()}


def _compiled(decoder, graphs):
    import torch._dynamo

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
    for region in decoder.regions():
        region.compile(backend=backend, fullgraph=True, dynamic=False)
    return decoder


def test_compiled_regions_build_their_graphs_in_the_first_two_latent_frames():
    """Compiled region by region: the first latent frame builds the first-frame graphs, the
    second the graphs every later frame reuses; frames 3-5 and a second decode build none, and
    the decode is diffusers'. Regions of one class at one shape share a graph."""
    from difflet.models.wan.vae.regions import WanVAERegionDecoder

    vae = _diffusers_vae(0)
    graphs: list = []
    decoder = _compiled(WanVAERegionDecoder(_model_from(vae)), graphs)
    after_each: list[int] = []
    z = _latents(5)
    with torch.no_grad():
        got = decoder(z, on_chunk=lambda i: after_each.append(len(graphs)))
        again = decoder(_latents(4, seed=3), on_chunk=lambda i: after_each.append(len(graphs)))
        want = vae.decode(z, return_dict=False)[0]
    torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-4)
    assert again.shape[2] == 13
    first, steady = after_each[0], after_each[1]
    assert 0 < first < steady
    assert after_each[1:] == [steady] * 8
    assert steady < 2 * len(decoder.regions())  # shared graphs: fewer than one per region per set


def test_a_single_latent_frame_builds_only_the_first_frame_graphs_and_keeps_no_cache():
    """Qwen-Image decodes one latent frame: the first-frame graphs without the cache outputs, and
    no later-frame graphs."""
    from difflet.models.wan.vae.regions import WanVAERegionDecoder

    vae = _diffusers_vae(0)
    graphs: list = []
    decoder = _compiled(WanVAERegionDecoder(_model_from(vae)), graphs)
    z = _latents(1)
    with torch.no_grad():
        got = decoder(z)
        want = vae.decode(z, return_dict=False)[0]
    torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-4)
    single = len(graphs)
    for gm in graphs:  # no graph returns more than its decoded tensor (no caches)
        outputs = [n for n in gm.graph.nodes if n.op == "output"][0].args[0]
        flat = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
        assert sum(o is not None for o in flat) == 1
    with torch.no_grad():
        decoder(_latents(2))
    assert len(graphs) > single


def test_region_decode_accepts_the_latents_on_another_device_and_returns_host_frames():
    """The walk moves each latent frame to the weights' device and each decoded chunk back to the
    host (the CPU here, so only the contract is checked: a host tensor in the input's frames)."""
    from difflet.models.wan.vae.regions import WanVAERegionDecoder

    model = _model_from(_diffusers_vae(0))
    with torch.no_grad():
        out = WanVAERegionDecoder(model)(_latents(2))
    assert out.device.type == "cpu" and out.shape[2] == 5
