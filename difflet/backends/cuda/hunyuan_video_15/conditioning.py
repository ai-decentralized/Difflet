"""Prompt conditioning as the reference computes it: device-local Qwen rotary
frequencies, and cuDNN SDPA off while prompts are encoded."""

from functools import wraps

import torch


def initialize_qwen_rotary(qwen):
    """Recompute the nonpersistent FP32 rotary frequencies on the encoder's device.

    Frequencies computed on the CPU and moved to CUDA differ in the last bits from the
    device's own FP32 power arithmetic, which the reference conditioning uses.
    """
    rotary = qwen.rotary_emb
    if rotary.config.rope_parameters.get("rope_type", "default") != "default":
        raise NotImplementedError("HunyuanVideo 1.5 requires default Qwen rotary embeddings")
    with torch.no_grad(), torch.device(next(qwen.parameters()).device):
        inv_freq, _ = rotary.compute_default_rope_parameters(rotary.config)
    for name in ("inv_freq", "original_inv_freq"):
        setattr(rotary, name, inv_freq.clone())


def configure_conditioning(pipe):
    """Encode prompts with cuDNN SDPA disabled, as the reference conditioning does.

    Only ``encode_prompt`` changes the policy, and it restores the caller's afterwards; the
    denoiser keeps its own. No model class or learned weight changes.
    """
    initialize_qwen_rotary(pipe.text_encoder)
    original = pipe.encode_prompt

    @wraps(original)
    def encode_prompt(*args, **kwargs):
        previous = torch.backends.cuda.cudnn_sdp_enabled()
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
            return original(*args, **kwargs)
        finally:
            torch.backends.cuda.enable_cudnn_sdp(previous)

    pipe.encode_prompt = encode_prompt
