"""The Flux single block's ``proj_out`` as one gemm at tp=1: a CUDA load-time module swap.

Diffusers runs one ``Linear(dim + mlp, dim)`` over ``cat([attn, mlp])``. The shared
``modeling_flux`` keeps the Megatron split (``proj_out_attn`` + ``proj_out_mlp``: two gemms and
an add), because the split is what merges two all-reduces at tp > 1 and what the Trainium and
TPU graphs trace. At tp=1 there is nothing to reduce, so after loading the CUDA backend may
rebuild diffusers' layout from the split weights and bind a forward that uses it to the block.
The shared model file is untouched, and the checkpoint keeps the split.

Off by default. Measured on one card, the fused path makes the eager DiT forward about 1%
faster; on one seed it also moved the end-to-end PSNR against diffusers by 10 dB, and until
that is explained the split path is the one that ships. Opt in with ``fuse_proj_out=True`` on
``CudaFluxTransformerApplication`` (``application_kwargs={"fuse_proj_out": True}`` through the
pipeline; ``--fuse-proj-out`` on the e2e gate script).
"""

from __future__ import annotations

import types

import torch
from torch import nn

from difflet.models.flux import modeling_flux as _modeling


def _fused_forward(
    self,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
    rotary_emb_text=None,
    rotary_emb_image=None,
):
    """``NeuronFluxSingleTransformerBlock.forward`` with the projection as one gemm."""
    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    attn_output = self.attn(
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        rotary_emb_text=rotary_emb_text,
        rotary_emb_image=rotary_emb_image,
    )
    # Diffusers' one gemm over the concatenation: accumulated in fp32, rounded once.
    proj_out = self.proj_out(torch.cat([attn_output, mlp_hidden_states], dim=2))
    hidden_states = residual + gate.unsqueeze(1) * proj_out
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return hidden_states


def fuse_proj_out(block: nn.Module) -> bool:
    """Rebuild diffusers' fused ``proj_out`` on one single block and bind the fused forward.

    The fused ``nn.Linear(dim + mlp, dim)`` takes ``cat([attn_weight, mlp_weight], dim=1)`` and
    the attn bias (exactly diffusers' ``proj_out``); the split modules are dropped. Returns True
    when it fused, False when there was nothing to do (already fused) or it must not (tp > 1:
    the split carries the sharding).
    """
    if getattr(block, "proj_out", None) is not None or block.proj_out_attn is None:
        return False
    if _modeling._safe_tp_size() != 1 or block.proj_out_attn.tensor_parallel_group is not None:
        return False
    attn, mlp = block.proj_out_attn, block.proj_out_mlp
    weight = torch.cat([attn.weight.detach(), mlp.weight.detach()], dim=1)
    # Built on the meta device so no throw-away [dim, dim + mlp] buffer is allocated and
    # kaiming-initialised on the GPU; the parameters are the cat and the bias themselves.
    fused = nn.Linear(weight.shape[1], weight.shape[0], bias=attn.bias is not None, device="meta")
    fused.weight = nn.Parameter(weight, requires_grad=False)
    if attn.bias is not None:
        fused.bias = nn.Parameter(attn.bias.detach().clone(), requires_grad=False)
    block.proj_out = fused
    block.proj_out_attn = None
    block.proj_out_mlp = None
    block.forward = types.MethodType(_fused_forward, block)
    return True


__all__ = ["fuse_proj_out"]
