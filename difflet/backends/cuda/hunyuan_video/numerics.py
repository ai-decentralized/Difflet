"""Hunyuan-specific reference rounding, applied after strict loading on the GPU flow."""

from __future__ import annotations

import torch


def _single_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    temb,
    attention_mask=None,
    image_rotary_emb=None,
    *args,
    **kwargs,
):
    """Preserve Hunyuan's two streams with the reference's one output projection."""
    text_length = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
    residual = hidden_states
    normalized, gate = self.norm(hidden_states, emb=temb)
    mlp = self.act_mlp(self.proj_mlp(normalized))
    attention, context = self.attn(
        hidden_states=normalized[:, :-text_length],
        encoder_hidden_states=normalized[:, -text_length:],
        attention_mask=attention_mask,
        image_rotary_emb=image_rotary_emb,
    )
    projected = self.proj_out(torch.cat([torch.cat([attention, context], dim=1), mlp], dim=-1))
    output = residual + gate.unsqueeze(1) * projected
    return output[:, :-text_length], output[:, -text_length:]


def fuse_single_projection(block: torch.nn.Module, *, tp_degree: int) -> bool:
    """Restore the checkpoint's concatenated projection after loading, only at tp=1."""
    import types

    if tp_degree != 1:
        raise NotImplementedError("The reference Hunyuan output projection requires tp=1")
    if getattr(block, "proj_out", None) is not None:
        return False
    attn, mlp = block.proj_out_attn, block.proj_out_mlp
    weight = torch.cat([attn.weight.detach(), mlp.weight.detach()], dim=1)
    fused = torch.nn.Linear(
        weight.shape[1], weight.shape[0], bias=attn.bias is not None, device="meta"
    )
    fused.weight = torch.nn.Parameter(weight, requires_grad=False)
    if attn.bias is not None:
        fused.bias = torch.nn.Parameter(attn.bias.detach().clone(), requires_grad=False)
    fused.train(block.training)
    block.proj_out = fused
    block.proj_out_attn = None
    block.proj_out_mlp = None
    block.forward = types.MethodType(_single_forward, block)
    return True


def restore_reference_row_bias(module: torch.nn.Module, *, tp_degree: int) -> int:
    """Keep bias inside the reference GEMM at TP=1, without changing shared collectives."""
    from difflet.backends.cuda.ops_impl.linear import RowParallelLinear

    if tp_degree != 1:
        raise NotImplementedError("Reference Hunyuan biased projections require tp=1")
    replacements = {}
    for parent in list(module.modules()):
        # Walk registrations, including aliases that named_children() deduplicates.
        for name, layer in list(parent._modules.items()):
            if not isinstance(layer, RowParallelLinear):
                continue
            if layer.bias is None or layer.skip_bias_add or not layer.reduce_output:
                continue
            if layer in replacements:
                setattr(parent, name, replacements[layer])
                continue
            # Standard Linear includes bias before its BF16 output rounding. Reuse the
            # strictly loaded parameters; constructing on meta allocates no new weights.
            replacement = torch.nn.Linear(
                layer.weight.shape[1], layer.weight.shape[0], device="meta"
            )
            replacement.weight = layer.weight
            replacement.bias = layer.bias
            replacement.train(layer.training)
            replacements[layer] = replacement
            setattr(parent, name, replacement)
    return len(replacements)


def preserve_reference_compiled_ops(block_lists, *, tp_degree: int) -> dict[str, int]:
    """Keep native linear/norm numerics inside otherwise compiled Hunyuan blocks.

    Share disabled unbound forwards across instances so a block stack does not
    acquire a different compiler callback for every parameter-bearing layer.
    Parameters, registrations, training flags and state-dict paths stay intact.
    """
    import types

    from diffusers.models.normalization import RMSNorm

    from difflet.backends.cuda.ops_impl.linear import ColumnParallelLinear, RowParallelLinear

    if tp_degree != 1:
        raise NotImplementedError("Reference Hunyuan compiled operators require tp=1")
    linears = (torch.nn.Linear, ColumnParallelLinear, RowParallelLinear)
    norms = (torch.nn.LayerNorm, RMSNorm)
    counts = {"linear": 0, "norm": 0}
    wrapped = {}
    seen = set()
    for blocks in block_lists:
        for block in blocks:
            for layer in block.modules():
                if layer in seen:
                    continue
                seen.add(layer)
                kind = (
                    "linear"
                    if isinstance(layer, linears)
                    else "norm"
                    if isinstance(layer, norms)
                    else None
                )
                if kind is None or getattr(layer.forward, "_torchdynamo_disable", False):
                    continue
                original = layer.forward.__func__
                if original not in wrapped:
                    wrapped[original] = torch.compiler.disable(original)
                layer.forward = types.MethodType(wrapped[original], layer)
                counts[kind] += 1
    return counts
