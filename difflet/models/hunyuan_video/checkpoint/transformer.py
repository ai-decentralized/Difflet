"""Convert upstream HunyuanVideo single-block output projections."""

from __future__ import annotations


def split_single_block_projections(state_dict, *, num_single_layers, inner_dim, clone=True):
    """Split fused attention/MLP columns in place, keeping the bias on attention only.

    ``clone=True`` preserves the Trainium converter's detached contiguous copies.
    At tp=1 the CUDA loader can copy directly from views with ``clone=False``.
    """
    for i in range(num_single_layers):
        prefix = f"single_transformer_blocks.{i}."
        weight = state_dict.pop(prefix + "proj_out.weight")
        bias = state_dict.pop(prefix + "proj_out.bias")
        if weight.ndim != 2 or weight.shape[0] != inner_dim or weight.shape[1] <= inner_dim:
            raise ValueError(f"{prefix}proj_out has incompatible shape {tuple(weight.shape)}")
        if bias.shape != (inner_dim,):
            raise ValueError(f"{prefix}proj_out bias has incompatible shape {tuple(bias.shape)}")
        pieces = {
            "proj_out_attn.weight": weight[:, :inner_dim],
            "proj_out_attn.bias": bias,
            "proj_out_mlp.weight": weight[:, inner_dim:],
        }
        for name, tensor in pieces.items():
            state_dict[prefix + name] = tensor.clone().detach().contiguous() if clone else tensor
    return state_dict
