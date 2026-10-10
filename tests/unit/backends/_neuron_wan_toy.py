"""A tiny diffusers-layout Wan2.1 model for the W1 tests (not collected by pytest).

``write_tiny_wan_model(root)`` writes what the neuron Wan application reads from a real snapshot,
at toy size, and returns the TP1 fp32 reference output on the fixed inputs:

* ``transformer/config.json`` + ``diffusion_pytorch_model.safetensors``: a seeded TP1
  ``WanTransformer3DModel`` (4 heads x 16, 2 blocks) whose state_dict is saved under the
  diffusers keys, so the FFN appears as ``net.0.proj``/``net.2`` and the loader's
  module -> checkpoint rename is exercised;
* ``scheduler/scheduler_config.json``: Wan2.1's UniPC config (flow_shift 3.0, flow sigmas);
* ``model_index.json`` without ``boundary_ratio`` (Wan2.1: one expert for every step);
* ``text_encoder/``: a seeded one-layer ``transformers.UMT5EncoderModel`` with ``d_model`` equal
  to the DiT's ``text_dim``, and ``tokenizer/``: a word-level fast tokenizer that appends
  ``</s>``, so the orchestrator's negative-prompt encode (``""`` at guidance > 1) really runs
  through the rank-0 host encoder.

The reference helpers build the unsharded model in the calling process under a spec-only TP1
mesh (``_neuron_toy.tp1_mesh``), before any process group exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tests.unit.backends._neuron_toy import tp1_mesh

#: The tiny DiT of tests/unit/models/wan/test_wan_application.py:20-34.
TINY_WAN_TRANSFORMER: dict = {
    "num_attention_heads": 4,
    "attention_head_dim": 16,
    "in_channels": 4,
    "out_channels": 4,
    "text_dim": 24,
    "freq_dim": 32,
    "ffn_dim": 48,
    "num_layers": 2,
    "patch_size": [1, 2, 2],
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "rope_max_seq_len": 1024,
    "eps": 1e-6,
}
#: 32x48x5 -> latent (1, 4, 2, 4, 6) -> 2 x 2 x 3 = 12 DiT tokens; text (1, 8, 24).
TINY_SHAPE = {"height": 32, "width": 48, "num_frames": 5}
TINY_TEXT_SEQ_LEN = 8
TINY_LATENT_SHAPE = (1, 4, 2, 4, 6)
TINY_TEXT_SHAPE = (1, TINY_TEXT_SEQ_LEN, TINY_WAN_TRANSFORMER["text_dim"])
TINY_BLOCK_NAMES = [f"blocks.{i}" for i in range(TINY_WAN_TRANSFORMER["num_layers"])]
TINY_WEIGHT_SEED = 0
TINY_INPUT_SEED = 1
TINY_PIPELINE_SEED = 2
TINY_ENCODER_SEED = 3
TINY_TIMESTEP = 750.0

#: Wan2.1-T2V-14B-Diffusers scheduler/scheduler_config.json (revision 38ec498c).
WAN21_SCHEDULER: dict = {
    "_class_name": "UniPCMultistepScheduler",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "disable_corrector": [],
    "dynamic_thresholding_ratio": 0.995,
    "final_sigmas_type": "zero",
    "flow_shift": 3.0,
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "predict_x0": True,
    "prediction_type": "flow_prediction",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "solver_order": 2,
    "solver_p": None,
    "solver_type": "bh2",
    "steps_offset": 0,
    "thresholding": False,
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": True,
    "use_karras_sigmas": False,
}

#: Word-level vocabulary of the tiny tokenizer; ids 0-2 are T5's pad, eos and unk.
_TOKENS = ["<pad>", "</s>", "<unk>", "a", "cat", "on", "the", "moon"]
_UMT5_CONFIG = {
    "vocab_size": len(_TOKENS),
    "d_model": TINY_WAN_TRANSFORMER["text_dim"],
    "d_kv": 8,
    "d_ff": 32,
    "num_heads": 3,
    "num_layers": 1,
    "relative_attention_num_buckets": 8,
    "relative_attention_max_distance": 16,
    "feed_forward_proj": "gated-gelu",
    "pad_token_id": 0,
    "eos_token_id": 1,
    "decoder_start_token_id": 0,
}

# module -> checkpoint direction, as the loader applies it (difflet/backends/tpu/wan/transformer.py)
_DIFFUSERS_FFN_KEYS = ((".ffn.net_in.", ".ffn.net.0.proj."), (".ffn.net_out.", ".ffn.net.2."))


def tiny_wan_inputs(*, dtype: torch.dtype = torch.float32):
    """The fixed DiT inputs ``(latents, timestep, text)``, identical in every process."""
    gen = torch.Generator().manual_seed(TINY_INPUT_SEED)
    latents = torch.randn(TINY_LATENT_SHAPE, generator=gen)
    text = torch.randn(TINY_TEXT_SHAPE, generator=gen)
    timestep = torch.tensor([TINY_TIMESTEP])
    return latents.to(dtype), timestep.to(dtype), text.to(dtype)


def tiny_wan_pipeline_inputs():
    """``(prompt_embeds, latents)`` for the two-step pipeline run, fp32, identical everywhere."""
    gen = torch.Generator().manual_seed(TINY_PIPELINE_SEED)
    prompt_embeds = torch.randn(TINY_TEXT_SHAPE, generator=gen)
    latents = torch.randn(TINY_LATENT_SHAPE, generator=gen)
    return prompt_embeds, latents


def tiny_wan_config(*, tp_degree: int = 1, dtype: torch.dtype = torch.float32):
    """``NeuronWanConfig`` of the tiny model at ``TINY_SHAPE``."""
    from difflet.backends.neuron.wan.config import NeuronWanConfig

    return NeuronWanConfig(
        **{**TINY_WAN_TRANSFORMER, "patch_size": tuple(TINY_WAN_TRANSFORMER["patch_size"])},
        **TINY_SHAPE,
        text_seq_len=TINY_TEXT_SEQ_LEN,
        tp_degree=tp_degree,
        torch_dtype=dtype,
    )


def tiny_wan_tp1_model():
    """The seeded unsharded DiT; call inside ``tp1_mesh()``."""
    from difflet.models.wan.modeling_wan import WanTransformer3DModel, WanTransformerConfig

    config = WanTransformerConfig.from_diffusers_dict(TINY_WAN_TRANSFORMER)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(TINY_WEIGHT_SEED)
        model = WanTransformer3DModel(config, dtype=torch.float32)
    return model.eval().requires_grad_(False)


def _diffusers_key(name: str) -> str:
    for module_key, checkpoint_key in _DIFFUSERS_FFN_KEYS:
        name = name.replace(module_key, checkpoint_key)
    return name


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_text_encoder(root: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import UMT5Config, UMT5EncoderModel

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(TINY_ENCODER_SEED)
        encoder = UMT5EncoderModel(UMT5Config(**_UMT5_CONFIG)).eval()
    encoder.save_pretrained(str(root / "text_encoder"))

    tokenizer = Tokenizer(
        models.WordLevel({token: i for i, token in enumerate(_TOKENS)}, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A </s>", special_tokens=[("</s>", 1)]
    )
    (root / "tokenizer").mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(root / "tokenizer" / "tokenizer.json"))
    _write_json(
        root / "tokenizer" / "tokenizer_config.json",
        {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "model_max_length": TINY_TEXT_SEQ_LEN,
        },
    )


def write_tiny_wan_model(root) -> torch.Tensor:
    """Write the tiny model under ``root``; return the TP1 fp32 output on ``tiny_wan_inputs()``."""
    from safetensors.torch import save_file

    root = Path(root)
    with tp1_mesh():
        model = tiny_wan_tp1_model()
        state = {
            _diffusers_key(name): tensor.detach().contiguous().clone()
            for name, tensor in model.state_dict().items()
        }
        with torch.no_grad():
            reference = model(*tiny_wan_inputs())
    if not any(".ffn.net.0.proj." in key for key in state):
        raise AssertionError("the tiny checkpoint must carry the diffusers FFN keys")
    _write_json(root / "transformer" / "config.json", TINY_WAN_TRANSFORMER)
    save_file(state, str(root / "transformer" / "diffusion_pytorch_model.safetensors"))
    _write_json(root / "scheduler" / "scheduler_config.json", WAN21_SCHEDULER)
    _write_json(root / "model_index.json", {"_class_name": "WanPipeline"})
    _write_text_encoder(root)
    return reference


def tiny_wan_pipeline_reference(
    root, *, num_inference_steps: int, guidance_scale: float
) -> torch.Tensor:
    """The same ``WanOrchestrator`` at TP1 on the host: the seeded unsharded DiT, the tiny UMT5
    called directly (no trimming, no broadcast), ``tiny_wan_pipeline_inputs()``; final latents."""
    from transformers import UMT5EncoderModel

    from difflet.models.wan.pipeline import WanOrchestrator

    root = Path(root)
    encoder = UMT5EncoderModel.from_pretrained(
        str(root / "text_encoder"), dtype=torch.float32
    ).eval()
    prompt_embeds, latents = tiny_wan_pipeline_inputs()
    with tp1_mesh():
        model = tiny_wan_tp1_model()

        def transformer(hidden_states, timestep, encoder_hidden_states):
            return model(hidden_states, timestep, encoder_hidden_states)

        transformer.dtype = torch.float32
        pipeline = WanOrchestrator(
            model_path=str(root),
            text_encoder=lambda input_ids, attention_mask: encoder(input_ids, attention_mask),
            transformer=transformer,
            dtype=torch.float32,
            max_text_length=TINY_TEXT_SEQ_LEN,
            **TINY_SHAPE,
        )
        out = pipeline(
            prompt_embeds=prompt_embeds,
            latents=latents,
            channels=TINY_LATENT_SHAPE[1],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="latent",
        )
    return out.latents
