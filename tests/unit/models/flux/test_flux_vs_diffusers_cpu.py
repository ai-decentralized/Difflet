"""The neutral Flux DiT against diffusers' ``FluxTransformer2DModel`` on CPU.

D13 proxy: with no trn2 available, the Trainium "same cosine before/after the split"
proof is replaced by pinning the neutral modeling, driven by ``FluxDiTConfig`` and fed
weights through ``convert_flux_hf_state_dict``, to the diffusers reference in fp32. A
strict load proves the conversion produces exactly the module's parameter names.

The single-stream block relies on two Trainium op contracts the CPU stand-ins do not
honour (``RowParallelLinear(skip_bias_add=True)`` returning ``(out, bias)``, and
``reduce_from_tensor_model_parallel_region(..., process_group=)``); the test bridges
them exactly as ``test_flux_sp.py`` does. The CUDA ops (``_torch_megatron``) have the
same two gaps, so the CUDA port has to close them for real.
"""

import torch
import torch.nn.functional as F
from diffusers import FluxPipeline, FluxTransformer2DModel

import difflet.models.flux.modeling_flux as mf
from difflet.layers.embeddings import FluxPosEmbed
from difflet.models.flux.dit_config import FluxDiTConfig
from difflet.models.flux.modeling_flux import (
    NeuronFluxTransformer2DModel,
    convert_flux_hf_state_dict,
)

# Rope axes must be even (cos/sin pairs) and sum to attention_head_dim (8).
AXES_DIMS_ROPE = (2, 2, 4)


class _FaithfulRowParallelLinear(mf.RowParallelLinear):
    """CPU ``RowParallelLinear`` that returns ``(x @ W^T, bias)`` under ``skip_bias_add``."""

    def __init__(self, input_size, output_size, bias=True, skip_bias_add=False, **kwargs):
        super().__init__(input_size, output_size, bias=bias, **kwargs)
        self._skip_bias_add = skip_bias_add
        self.tensor_parallel_group = None

    def forward(self, x):
        if self._skip_bias_add:
            return F.linear(x, self.weight), self.bias
        return super().forward(x)


def _cpu_reduce(tensor, process_group=None):
    del process_group
    return tensor


def test_neutral_flux_dit_matches_diffusers_on_cpu(monkeypatch):
    monkeypatch.setattr(mf, "RowParallelLinear", _FaithfulRowParallelLinear)
    monkeypatch.setattr(mf, "reduce_from_tensor_model_parallel_region", _cpu_reduce)
    torch.manual_seed(0)
    reference = FluxTransformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        pooled_projection_dim=8,
        guidance_embeds=True,
        axes_dims_rope=AXES_DIMS_ROPE,
    ).eval()

    config = FluxDiTConfig(
        patch_size=1,
        in_channels=4,
        out_channels=None,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        pooled_projection_dim=8,
        guidance_embeds=True,
        axes_dims_rope=AXES_DIMS_ROPE,
        height=64,
        width=64,
        torch_dtype=torch.float32,
    )
    model = NeuronFluxTransformer2DModel(config).eval()
    converted = convert_flux_hf_state_dict(
        reference.state_dict(),
        num_single_layers=config.num_single_layers,
        inner_dim=config.inner_dim,
        world_size=config.neuron_config.world_size,
    )
    # global_rank.rank is the Trainium SPMDRank parameter; off Trainium SPMDRank is a
    # plain rank holder with no state, so that one key has nothing to land on.
    assert "global_rank.rank" not in model.state_dict()
    converted.pop("global_rank.rank")
    model.load_state_dict(converted, strict=True)

    hidden = torch.randn(1, 16, 4)
    text = torch.randn(1, 6, 16)
    pooled = torch.randn(1, 8)
    timestep = torch.tensor([0.5])
    guidance = torch.tensor([3.5])
    # 64x64 pixels -> 8x8 VAE latent -> 4x4 packed tokens = 16 image tokens. FluxPipeline
    # passes the packed grid (latent // 2) to _prepare_latent_image_ids.
    img_ids = FluxPipeline._prepare_latent_image_ids(1, 4, 4, "cpu", torch.float32)
    txt_ids = torch.zeros(6, 3)
    assert img_ids.shape == (16, 3) and config.image_seq_len == 16

    rope = torch.stack(
        FluxPosEmbed(10000, AXES_DIMS_ROPE)(torch.cat((txt_ids, img_ids), dim=0)), dim=2
    )

    with torch.no_grad():
        expected = reference(
            hidden_states=hidden,
            encoder_hidden_states=text,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]
        actual = model(
            hidden_states=hidden,
            encoder_hidden_states=text,
            pooled_projections=pooled,
            timestep=timestep,
            guidance=guidance,
            image_rotary_emb=rope,
        )
    if isinstance(actual, (tuple, list)):
        actual = actual[0]

    assert actual.shape == expected.shape == (1, 16, 4)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
