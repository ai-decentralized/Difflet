"""A tiny random HunyuanVideo DiT in diffusers format, fixed inputs and diffusers' own outputs, for
the neuron HunyuanVideo tests (not collected by pytest).

``write_tiny_hv1(root)`` saves a seeded diffusers ``HunyuanVideoTransformer3DModel`` (1 dual-stream
and 2 single-stream blocks, 4 heads x 16, the rope axes summing to the head dim) as
``root/transformer``. ``tiny_inputs(valid_text)`` is one call at ``TINY_SHAPE``: a ``[1, 4, 5, 8,
8]`` latent (5 x 4 x 4 = 80 image tokens after the 2x2 patch, divisible by cp 2 and 4), 16 text
tokens of which the first ``valid_text`` are valid, timestep 500 and guidance 6000 (scale 6 x
1000, as the pipeline sends it). The references run diffusers in fp32 and never touch
``difflet.ops``.

The Difflet port binds ``difflet.ops`` when it is imported, so every run of it goes through
``_neuron_gloo.run_ranks`` (``hv1_dit_worker``): a fresh process with ``DIFFLET_BACKEND=neuron``,
which also keeps Dynamo's caches out of the parent.
"""

from __future__ import annotations

import torch

TINY_HV1 = dict(in_channels=4, out_channels=4, num_attention_heads=4, attention_head_dim=16,
                num_layers=1, num_single_layers=2, num_refiner_layers=1, patch_size=2,
                patch_size_t=1, text_embed_dim=32, pooled_projection_dim=16,
                rope_axes_dim=(4, 6, 6), guidance_embeds=True)
#: 17 frames -> 5 latent frames; 64 x 64 -> an 8 x 8 latent, 4 x 4 after the patch.
TINY_SHAPE = dict(height=64, width=64, num_frames=17, text_seq_len=16)


def write_tiny_hv1(root) -> str:
    from diffusers import HunyuanVideoTransformer3DModel

    torch.manual_seed(0)
    path = f"{root}/transformer"
    HunyuanVideoTransformer3DModel(**TINY_HV1).save_pretrained(path, safe_serialization=True)
    return path


def tiny_inputs(valid_text: int = TINY_SHAPE["text_seq_len"]) -> dict[str, torch.Tensor]:
    """The DiT's six inputs, by the names both forwards take."""
    g = torch.Generator().manual_seed(1)
    text = TINY_SHAPE["text_seq_len"]
    mask = torch.zeros(1, text, dtype=torch.int64)
    mask[:, :valid_text] = 1
    return dict(
        hidden_states=torch.randn(1, TINY_HV1["in_channels"], 5, 8, 8, generator=g),
        timestep=torch.tensor([500.0]),
        encoder_hidden_states=torch.randn(1, text, TINY_HV1["text_embed_dim"], generator=g),
        encoder_attention_mask=mask,
        pooled_projections=torch.randn(1, TINY_HV1["pooled_projection_dim"], generator=g),
        guidance=torch.tensor([6000.0]),
    )


def diffusers_reference(path, inputs: dict[str, torch.Tensor], **config) -> torch.Tensor:
    """diffusers' forward on ``inputs``, fp32 on the host; ``config`` overrides the saved config
    (``num_single_layers=1`` loads the first single-stream block only)."""
    from diffusers import HunyuanVideoTransformer3DModel

    model = HunyuanVideoTransformer3DModel.from_pretrained(path, **config).eval()
    with torch.no_grad():
        return model(**inputs, return_dict=False)[0]


def diffusers_teacache_reference(path, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Block 0's modulated image input in diffusers' model, the TeaCache signal:
    ``transformer_blocks[0].norm1(x_embedder(x), time_text_embed(t, pooled, guidance))``."""
    from diffusers import HunyuanVideoTransformer3DModel

    model = HunyuanVideoTransformer3DModel.from_pretrained(path).eval()
    with torch.no_grad():
        temb, _ = model.time_text_embed(inputs["timestep"], inputs["pooled_projections"],
                                        inputs["guidance"])
        hidden = model.x_embedder(inputs["hidden_states"])
        return model.transformer_blocks[0].norm1(hidden, emb=temb)[0]
