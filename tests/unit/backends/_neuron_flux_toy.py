"""A tiny random Flux DiT in diffusers format, and fixed inputs, for the neuron Flux tests."""

from __future__ import annotations

import torch

# 4 heads x 8 so TP4 shards one head per rank; rope axes sum to the head dim.
TINY = dict(patch_size=1, in_channels=8, num_layers=1, num_single_layers=2, attention_head_dim=8,
            num_attention_heads=4, joint_attention_dim=32, pooled_projection_dim=16,
            guidance_embeds=True, axes_dims_rope=(2, 2, 4))
TINY_SHAPE = dict(height=64, width=64, text_seq_len=8)  # 4x4 packed image tokens


def write_tiny_flux(root) -> str:
    from diffusers import FluxTransformer2DModel

    torch.manual_seed(0)
    model = FluxTransformer2DModel(**TINY)
    path = f"{root}/transformer"
    model.save_pretrained(path, safe_serialization=True)
    return path


def tiny_inputs(config):
    from difflet.backends.neuron.flux.transformer import flux_position_ids

    g = torch.Generator().manual_seed(1)
    txt_ids, img_ids = flux_position_ids(config)
    return dict(
        hidden_states=torch.randn(1, config.image_seq_len, TINY["in_channels"], generator=g),
        encoder_hidden_states=torch.randn(1, config.text_seq_len, TINY["joint_attention_dim"], generator=g),
        pooled_projections=torch.randn(1, TINY["pooled_projection_dim"], generator=g),
        timestep=torch.tensor([0.7]),
        guidance=torch.tensor([3.5]),
        img_ids=img_ids, txt_ids=txt_ids,
    )


def diffusers_reference(path, config) -> torch.Tensor:
    from diffusers import FluxTransformer2DModel

    model = FluxTransformer2DModel.from_pretrained(path).eval()
    with torch.no_grad():
        return model(**tiny_inputs(config), return_dict=False)[0]


def first(out):
    return out[0] if isinstance(out, (tuple, list)) else out
