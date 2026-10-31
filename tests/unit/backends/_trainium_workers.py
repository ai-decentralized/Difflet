"""Gloo rank workers for the trainium (AoT) backend's CPU tests (not collected by pytest).

They run under ``tests.unit.backends._neuron_gloo.run_ranks`` with ``TRAINIUM_CPU_ENV``, so
``difflet.ops`` resolves to NxD's layers, and they need ``neuronx_distributed``. They must never
touch a Neuron device: NxD's CPU mode and torch_xla's CPU device keep its model-parallel setup
off the runtime (outside CPU mode, ``initialize_model_parallel`` runs a NEFF to warm up the
collectives), and checkpoint sharding is plain torch.
"""

from __future__ import annotations

import torch

#: The child environment: the trainium backend, NxD in CPU mode, torch_xla on CPU.
TRAINIUM_CPU_ENV = {"DIFFLET_BACKEND": "trainium", "NXD_CPU_MODE": "1", "PJRT_DEVICE": "CPU"}

#: A HunyuanVideo DiT small enough for CPU whose widths divide tp=4 (4 heads of 4: inner 16).
HV1_TINY = dict(
    in_channels=2, out_channels=2, num_attention_heads=4, attention_head_dim=4,
    num_layers=2, num_single_layers=2, num_refiner_layers=1, mlp_ratio=2.0,
    patch_size=2, patch_size_t=1, guidance_embeds=True, text_embed_dim=6,
    pooled_projection_dim=5, rope_axes_dim=(2, 2, 0),
)


def _hv1_modulation_names(module) -> list[str]:
    names = []
    for i, _ in enumerate(module.transformer_blocks):
        names += [f"transformer_blocks.{i}.norm1.linear", f"transformer_blocks.{i}.norm1_context.linear"]
    for i, _ in enumerate(module.single_transformer_blocks):
        names.append(f"single_transformer_blocks.{i}.norm.linear")
    return names


def hv1_modulation_shard_worker(rank: int, world_size: int) -> dict:
    """Build the HunyuanVideo DiT as the trainium backbone does, then shard a full checkpoint of
    its modulation linears with NxD's ``get_sharded_checkpoint``: what each rank file holds."""
    from neuronx_distributed.parallel_layers import parallel_state
    from neuronx_distributed.trace.trace import get_sharded_checkpoint

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=world_size, skip_collective_init=True
    )
    from difflet.models.hunyuan_video import modeling_hunyuan_video as hv

    module = hv.HunyuanVideoTransformer3DModel(**HV1_TINY)
    hv.shard_modulation_linears(module)

    result = {}
    for name in _hv1_modulation_names(module):
        layer = module.get_submodule(name)
        out_features = layer.weight.shape[0] * world_size
        in_features = layer.weight.shape[1]
        # The checkpoint's full tensors, identical on every rank (same seed).
        gen = torch.Generator().manual_seed(len(result))
        weight = torch.randn(out_features, in_features, generator=gen)
        bias = torch.randn(out_features, generator=gen)
        x = torch.randn(2, in_features, generator=gen)
        checkpoint = {"weight": weight.clone(), "bias": bias.clone()}
        get_sharded_checkpoint(checkpoint, layer, rank, world_size)
        rows = out_features // world_size
        with torch.no_grad():
            layer.load_state_dict(checkpoint)
            out = layer(x)
        result[name] = {
            "type": type(layer).__name__,
            "gather_output": bool(getattr(layer, "gather_output", False)),
            "weight_shape": tuple(checkpoint["weight"].shape),
            "bias_shape": tuple(checkpoint["bias"].shape),
            "weight_is_my_rows": torch.equal(
                checkpoint["weight"], weight[rank * rows:(rank + 1) * rows]
            ),
            "bias_is_full": torch.equal(checkpoint["bias"], bias),
            # The gathered output against the unsharded layer's.
            "out_shape": tuple(out.shape),
            "out_max_err": float((out - torch.nn.functional.linear(x, weight, bias)).abs().max()),
        }
    return result
