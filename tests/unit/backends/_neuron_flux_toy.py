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


# ---- T5 (v1.1: gated-gelu, as T5-XXL) ----
TINY_T5 = dict(vocab_size=64, d_model=16, d_kv=4, d_ff=32, num_layers=2, num_heads=4,
               feed_forward_proj="gated-gelu", relative_attention_num_buckets=8,
               relative_attention_max_distance=16, dropout_rate=0.0, tie_word_embeddings=False)
T5_SEQ = 8


def write_tiny_t5(root) -> str:
    from transformers import T5Config, T5EncoderModel

    torch.manual_seed(0)
    model = T5EncoderModel(T5Config(**TINY_T5)).eval()
    path = f"{root}/text_encoder_2"
    model.save_pretrained(path, safe_serialization=True)
    return path


def t5_inputs():
    """A padded prompt (pad id 0 after five tokens), no mask: as FluxPipeline encodes."""
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(1, TINY_T5["vocab_size"], (1, T5_SEQ), generator=g)
    ids[:, 5:] = 0
    return (ids,)


def t5_reference(path) -> torch.Tensor:
    from transformers import T5EncoderModel

    model = T5EncoderModel.from_pretrained(path).eval()
    (ids,) = t5_inputs()
    with torch.no_grad():
        return model(input_ids=ids).last_hidden_state


# ---- VAE (2D AutoencoderKL; Flux's has no quant convs) ----
TINY_VAE = dict(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(8, 16),
                down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
                up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"), layers_per_block=1,
                norm_num_groups=4, use_quant_conv=False, use_post_quant_conv=False)
VAE_HW = 16  # image side; latent side = 16 / 2 (two blocks: scale factor 2)


def write_tiny_vae(root) -> str:
    from diffusers import AutoencoderKL

    torch.manual_seed(0)
    path = f"{root}/vae"
    AutoencoderKL(**TINY_VAE).save_pretrained(path, safe_serialization=True)
    return path


def vae_latents():
    g = torch.Generator().manual_seed(3)
    return torch.randn(1, TINY_VAE["latent_channels"], VAE_HW // 2, VAE_HW // 2, generator=g)


def vae_reference(path) -> torch.Tensor:
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(path).eval()
    with torch.no_grad():
        return vae.decode(vae_latents(), return_dict=False)[0]
