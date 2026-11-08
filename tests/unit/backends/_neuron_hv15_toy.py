"""A tiny HunyuanVideo-1.5 snapshot with the MXFP8 geometry, fixed inputs and diffusers' own
fp32 outputs, for the HunyuanVideo-1.5 MXFP8 tests (not collected by pytest).

``TINY_FP8_HV15`` is the bf16 tests' tiny DiT (``test_neuron_hv15_dit.py``) with 32-wide heads:
4 x 32 = inner 128 and MLP 512, so every target's K is 128 or 512 and ``net.2``'s per rank is
512 / tp, a multiple of the MX layout's 128 at tp 1, 2 and 4. ``write_tiny_hv15_snapshot(root)``
writes what the pipeline reads without the VAE (the fp32 ``transformer/``, as the hub's is, so
the bf16 copy path runs; ``scheduler/``, ``guider/`` and ``model_index.json``).

Both MLP-downs (row-parallel: the image one a default target, the text one an override's) get
a K spread (``ROW_K_SPREAD``). Under diffusers' kaiming-uniform init every 32-value block lands
on the same E8M0 exponent, and MX rounding of a normal E4M3 value does not depend on its
power-of-two scale, so a whole-K quantization and a per rank K range one agree bit for bit and
a writer or loader that quantized a row-parallel target whole would go unseen. With the first
half of K scaled down, the two differ in about 37% of the values (those subnormal in one
layout, 2.4e-7 at most): too little for any useful forward tolerance to see, so the tests
compare the weights (``HV15_ROW_TARGETS``).

``hv15_reference`` is diffusers' forward in fp32 with the empty glyph stream and zero image
embeddings; with a spec, the weights it targets fake-quantized (a row-parallel one per rank K
range), what dequant mode computes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from difflet.quant import mx
from difflet.quant.spec import QuantSpec

TINY_FP8_HV15 = dict(in_channels=9, out_channels=4, num_attention_heads=4, attention_head_dim=32,
                     num_layers=2, num_refiner_layers=1, mlp_ratio=4.0, patch_size=1,
                     patch_size_t=1, qk_norm="rms_norm", text_embed_dim=32, text_embed_2_dim=24,
                     image_embed_dim=16, rope_axes_dim=(8, 12, 12), rope_theta=256.0,
                     target_size=640, task_type="t2v", use_meanflow=False)
#: 5 frames at 32 x 48: a 2 x 2 x 3 latent (the VAE's 4x temporal and 16x spatial compression),
#: 12 latent tokens; ``image_tokens`` is the image-embedding stream's length (zeros for t2v).
TINY_SHAPE_HV15 = dict(height=32, width=48, num_frames=5, text_seq_len=16, text_2_seq_len=8,
                       image_tokens=9)
_LATENT = (2, 2, 3)
#: The default targets per dual-stream block: the image stream's q/k/v and MLP.
TINY_TARGETS_PER_BLOCK = {"attn.to_q", "attn.to_k", "attn.to_v", "ff.net.0.proj", "ff.net.2"}
#: The factor on the first half of K of both MLP-downs.
ROW_K_SPREAD = 2.0 ** -14
#: The row-parallel target the tests read back: the image MLP-down, the default set's only one.
HV15_ROW_TARGETS = ("transformer_blocks.0.ff.net.2",)
_ROW_TARGET = r"transformer_blocks\.\d+\.(ff|ff_context)\.net\.2"


def tiny_hv15_inputs(*, seed: int = 1, valid_text: int = 6) -> dict[str, torch.Tensor]:
    """The DiT's four inputs at ``TINY_SHAPE_HV15``: a noise latent beside the t2v call's zero
    condition channels, timestep 700, and a text stream whose first ``valid_text`` tokens are
    valid."""
    g = torch.Generator().manual_seed(seed)
    text = TINY_SHAPE_HV15["text_seq_len"]
    noise = TINY_FP8_HV15["out_channels"]
    mask = torch.zeros(1, text, dtype=torch.int64)
    mask[:, :valid_text] = 1
    return dict(
        hidden_states=torch.cat([torch.randn(1, noise, *_LATENT, generator=g),
                                 torch.zeros(1, TINY_FP8_HV15["in_channels"] - noise, *_LATENT)],
                                dim=1),
        timestep=torch.tensor([700.0]),
        encoder_hidden_states=torch.randn(1, text, TINY_FP8_HV15["text_embed_dim"], generator=g),
        encoder_attention_mask=mask,
    )


def write_tiny_hv15_snapshot(root, *, config: dict = TINY_FP8_HV15) -> Path:
    """The seeded fp32 DiT with the K spread as ``root/transformer``, the scheduler, the guider
    (classifier-free guidance at 6.0) and ``model_index.json``; returns ``root``."""
    from diffusers import FlowMatchEulerDiscreteScheduler, HunyuanVideo15Transformer3DModel

    root = Path(root)
    torch.manual_seed(0)
    model = HunyuanVideo15Transformer3DModel(**config)
    with torch.no_grad():
        for name, module in model.named_modules():
            if re.fullmatch(_ROW_TARGET, name):
                module.weight[:, : module.in_features // 2] *= ROW_K_SPREAD
    model.save_pretrained(root / "transformer", safe_serialization=True)
    FlowMatchEulerDiscreteScheduler(shift=7.0).save_pretrained(root / "scheduler")
    (root / "guider").mkdir(parents=True, exist_ok=True)
    (root / "guider" / "guider_config.json").write_text(json.dumps(dict(
        _class_name="ClassifierFreeGuidance", enabled=True, guidance_rescale=0.0,
        guidance_scale=6.0, start=0.0, stop=1.0, use_original_formulation=False)))
    (root / "model_index.json").write_text(json.dumps(dict(
        _class_name="HunyuanVideo15Pipeline",
        scheduler=["diffusers", "FlowMatchEulerDiscreteScheduler"])))
    return root


def hv15_reference(transformer_dir, inputs: dict, *,
                   spec: QuantSpec | None = None) -> torch.Tensor:
    """diffusers' fp32 forward on ``inputs`` with the empty glyph stream and zero image
    embeddings. With ``spec``, every linear it targets has its weight fake-quantized, a
    row-parallel one per rank K range (``spec.k_shards``): what dequant mode computes, since it
    multiplies the activations by the weights the quantized directory holds, dequantized.
    diffusers' module names are the checkpoint keys, so the spec matches them as it is."""
    from diffusers import HunyuanVideo15Transformer3DModel

    model = HunyuanVideo15Transformer3DModel.from_pretrained(
        str(transformer_dir), torch_dtype=torch.float32).eval()
    batch, config = inputs["hidden_states"].shape[0], model.config
    glyphs, images = TINY_SHAPE_HV15["text_2_seq_len"], TINY_SHAPE_HV15["image_tokens"]
    with torch.no_grad():
        if spec is not None:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and spec.matches(name):
                    module.weight.copy_(mx.fake_quantize_mx_sharded(module.weight,
                                                                    spec.k_shards(name)))
        return model(**inputs,
                     encoder_hidden_states_2=torch.zeros(batch, glyphs, config.text_embed_2_dim),
                     encoder_attention_mask_2=torch.zeros(batch, glyphs, dtype=torch.int64),
                     image_embeds=torch.zeros(batch, images, config.image_embed_dim),
                     return_dict=False)[0]
