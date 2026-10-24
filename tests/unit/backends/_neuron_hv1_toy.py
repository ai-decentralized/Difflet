"""A tiny random HunyuanVideo DiT in diffusers format, fixed inputs and diffusers' own outputs, for
the neuron HunyuanVideo tests (not collected by pytest).

``write_tiny_hv1(root)`` saves a seeded diffusers ``HunyuanVideoTransformer3DModel`` (1 dual-stream
and 2 single-stream blocks, 4 heads x 16, the rope axes summing to the head dim) as
``root/transformer``. ``tiny_inputs(valid_text)`` is one call at ``TINY_SHAPE``: a ``[1, 4, 5, 8,
8]`` latent (5 x 4 x 4 = 80 image tokens after the 2x2 patch, divisible by cp 2 and 4), 16 text
tokens of which the first ``valid_text`` are valid, timestep 500 and guidance 6000 (scale 6 x
1000, as the pipeline sends it). The references run diffusers in fp32 and never touch
``difflet.ops``.

``write_tiny_hv1_pipeline(root)`` adds what the composed application reads (a diffusers
``HunyuanVideoPipeline`` snapshot: tiny VAE, the real scheduler config, tiny random Llama and
CLIP text models with the real tokenizers); ``diffusers_generate`` is diffusers' own pipeline on
it.

The Difflet port binds ``difflet.ops`` when it is imported, so every run of it goes through
``_neuron_gloo.run_ranks`` (``hv1_dit_worker``, ``hv1_app_worker``): a fresh process with
``DIFFLET_BACKEND=neuron``, which also keeps Dynamo's caches out of the parent.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

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


# --------------------------------------------------------------------------
# the composed application: a tiny pipeline snapshot
# --------------------------------------------------------------------------

#: The real snapshot, for its tokenizers: the Llama template's crop of 95 tokens is defined on
#: the real Llama tokenizer, and CLIP's pooled token is its EOS.
HV1_SNAPSHOT = Path(
    "/home/ubuntu/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo/snapshots/"
    "e8c2aaa66fe3742a32c11a6766aecbf07c56e773"
)
TINY_PROMPT = "a red fox runs through fresh snow"
#: The snapshot's model_index.json and scheduler/scheduler_config.json.
HV1_MODEL_INDEX = {
    "_class_name": "HunyuanVideoPipeline", "_diffusers_version": "0.32.0.dev0",
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "LlamaModel"],
    "text_encoder_2": ["transformers", "CLIPTextModel"],
    "tokenizer": ["transformers", "LlamaTokenizerFast"],
    "tokenizer_2": ["transformers", "CLIPTokenizer"],
    "transformer": ["diffusers", "HunyuanVideoTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLHunyuanVideo"],
}
HV1_SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler", "_diffusers_version": "0.32.0.dev0",
    "base_image_seq_len": 256, "base_shift": 0.5, "invert_sigmas": False,
    "max_image_seq_len": 4096, "max_shift": 1.15, "num_train_timesteps": 1000, "shift": 7.0,
    "shift_terminal": None, "use_beta_sigmas": False, "use_dynamic_shifting": False,
    "use_exponential_sigmas": False, "use_karras_sigmas": False,
}
#: A tiny AutoencoderKLHunyuanVideo with the real compression (8 spatial, 4 temporal) and the
#: DiT's 4 latent channels.
TINY_HV1_VAE = dict(in_channels=3, out_channels=3, latent_channels=TINY_HV1["in_channels"],
                    block_out_channels=(8, 8, 8, 8), layers_per_block=1, norm_num_groups=4,
                    scaling_factor=0.476986, spatial_compression_ratio=8,
                    temporal_compression_ratio=4)


def hv1_tokenizers_available() -> bool:
    return (HV1_SNAPSHOT / "tokenizer").is_dir() and (HV1_SNAPSHOT / "tokenizer_2").is_dir()


def write_tiny_hv1_pipeline(root) -> str:
    """A diffusers ``HunyuanVideoPipeline`` snapshot under ``root``: the tiny transformer
    (``write_tiny_hv1``), a tiny VAE, the snapshot's scheduler, and random-weight Llama (3
    layers, hidden 32 = the DiT's ``text_embed_dim``; the encode reads hidden layer -3) and CLIP
    (hidden 16 = ``pooled_projection_dim``) text models with the real snapshot's tokenizers
    copied in (the caller skips when ``hv1_tokenizers_available()`` is False). Returns ``root``."""
    from diffusers import AutoencoderKLHunyuanVideo
    from transformers import CLIPTextConfig, CLIPTextModel, LlamaConfig, LlamaModel

    root = Path(root)
    write_tiny_hv1(root)
    torch.manual_seed(0)
    AutoencoderKLHunyuanVideo(**TINY_HV1_VAE).save_pretrained(root / "vae",
                                                              safe_serialization=True)
    llama = LlamaConfig(vocab_size=128320, hidden_size=TINY_HV1["text_embed_dim"],
                        intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8, max_position_embeddings=512,
                        bos_token_id=128000, eos_token_id=128001, rope_theta=500000.0)
    LlamaModel(llama).save_pretrained(root / "text_encoder", safe_serialization=True)
    clip = CLIPTextConfig(vocab_size=49408, hidden_size=TINY_HV1["pooled_projection_dim"],
                          intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                          max_position_embeddings=77,
                          projection_dim=TINY_HV1["pooled_projection_dim"],
                          hidden_act="quick_gelu", bos_token_id=0, eos_token_id=2, pad_token_id=1)
    CLIPTextModel(clip).save_pretrained(root / "text_encoder_2", safe_serialization=True)
    for name in ("tokenizer", "tokenizer_2"):
        shutil.copytree(HV1_SNAPSHOT / name, root / name)
    (root / "scheduler").mkdir(parents=True, exist_ok=True)
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(HV1_SCHEDULER_CONFIG))
    (root / "model_index.json").write_text(json.dumps(HV1_MODEL_INDEX))
    return str(root)


def diffusers_pipeline(root, **components):
    """diffusers' ``HunyuanVideoPipeline`` from the tiny snapshot, fp32 on the host
    (``components`` replace or drop parts: ``transformer=None``)."""
    from diffusers import HunyuanVideoPipeline

    pipe = HunyuanVideoPipeline.from_pretrained(str(root), torch_dtype=torch.float32,
                                                **components)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def diffusers_generate(root, *, prompt: str = TINY_PROMPT, steps: int = 3, seed: int = 42,
                       guidance_scale: float = 6.0) -> torch.Tensor:
    """diffusers' fp32 latents for ``prompt`` at ``TINY_SHAPE`` (the reference ``generate``)."""
    pipe = diffusers_pipeline(root)
    with torch.no_grad():
        return pipe(prompt=prompt, height=TINY_SHAPE["height"], width=TINY_SHAPE["width"],
                    num_frames=TINY_SHAPE["num_frames"], num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator().manual_seed(seed), output_type="latent",
                    max_sequence_length=TINY_SHAPE["text_seq_len"]).frames
