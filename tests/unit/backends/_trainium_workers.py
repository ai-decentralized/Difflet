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


def qwen_masked_dit_worker(rank: int, world_size: int, transformer_dir: str) -> dict:
    """The trainium Qwen-Image DiT trace module (``_QwenImageMaskedTraceModule``) and the shared
    one it subclasses, loaded from the tiny checkpoint, on the CPU backend: their outputs on the
    tiny inputs with the prompt right-padded (and masked) and with every row valid."""
    from diffusers import QwenImageTransformer2DModel

    from difflet.backends.trainium.qwen_image.transformer import _QwenImageMaskedTraceModule
    from difflet.models.qwen_image.modeling_qwen_image import _QwenImageTransformerTraceModule
    from tests.unit.backends._neuron_qwen_toy import (
        tiny_padded_text,
        tiny_qwen_config,
        tiny_qwen_inputs,
    )

    weights = QwenImageTransformer2DModel.from_pretrained(
        transformer_dir, torch_dtype=torch.float32
    ).state_dict()
    config = tiny_qwen_config()

    def build(cls):
        module = cls(config).eval()
        missing, unexpected = module.load_state_dict(
            {f"transformer.{key}": value for key, value in weights.items()}, strict=False
        )
        assert not unexpected, unexpected
        assert not [key for key in missing if key.startswith("transformer.")], missing
        return module

    masked, unmasked = build(_QwenImageMaskedTraceModule), build(_QwenImageTransformerTraceModule)
    hidden, timestep, text = tiny_qwen_inputs()
    padded, mask = tiny_padded_text(text)
    full_mask = torch.ones(text.shape[:2], dtype=torch.bool)
    # numpy, not tensors: a tensor crosses the result queue as shared memory that is gone once
    # the rank exits.
    with torch.no_grad():
        return {
            "mask_text": masked.mask_text,
            "masked_padded": masked(hidden, timestep, padded, mask, timestep).numpy(),
            "unmasked_padded": unmasked(hidden, timestep, padded, mask, timestep).numpy(),
            "masked_full": masked(hidden, timestep, text, full_mask, timestep).numpy(),
            "unmasked_full": unmasked(hidden, timestep, text, full_mask, timestep).numpy(),
        }


def _hv1_modulation_names(module) -> list[str]:
    names = []
    for i, _ in enumerate(module.transformer_blocks):
        names += [f"transformer_blocks.{i}.norm1.linear", f"transformer_blocks.{i}.norm1_context.linear"]
    for i, _ in enumerate(module.single_transformer_blocks):
        names.append(f"single_transformer_blocks.{i}.norm.linear")
    return names


def hv1_modulation_shard_worker(rank: int, world_size: int) -> dict:
    """Build the HunyuanVideo DiT as the trainium backbone does (its fused-bias column-parallel
    modulation), then shard a full checkpoint of its modulation linears with NxD's
    ``get_sharded_checkpoint``: what each rank file holds, and what the layer returns."""
    import torch.nn.functional as F
    from neuronx_distributed.parallel_layers import parallel_state
    from neuronx_distributed.trace.trace import get_sharded_checkpoint

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=world_size, skip_collective_init=True
    )
    from difflet.backends.trainium.hunyuan_video.backbone import _ColumnParallelFusedBias
    from difflet.models.hunyuan_video import modeling_hunyuan_video as hv

    module = hv.HunyuanVideoTransformer3DModel(**HV1_TINY)
    hv.shard_modulation_linears(module, linear_cls=_ColumnParallelFusedBias)

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
        mine = slice(rank * rows, (rank + 1) * rows)
        with torch.no_grad():
            layer.load_state_dict(checkpoint)
            out = layer(x)
            # In bf16: the gathered output against each shard's own fused F.linear(x, W_r, b_r),
            # the rounding the unsharded nn.Linear has per element.
            layer.to(torch.bfloat16)
            xb, wb, bb = x.bfloat16(), weight.bfloat16(), bias.bfloat16()
            out_bf16 = layer(xb)
            fused_bf16 = torch.cat([F.linear(xb, w, b) for w, b in
                                    zip(wb.chunk(world_size), bb.chunk(world_size))], dim=-1)
        result[name] = {
            "type": type(layer).__name__,
            "weight_shape": tuple(checkpoint["weight"].shape),
            "bias_shape": tuple(checkpoint["bias"].shape),
            "weight_is_my_rows": torch.equal(checkpoint["weight"], weight[mine]),
            "bias_is_my_rows": torch.equal(checkpoint["bias"], bias[mine]),
            # The gathered output against the unsharded layer's.
            "out_shape": tuple(out.shape),
            "out_max_err": float((out - F.linear(x, weight, bias)).abs().max()),
            "bf16_equals_fused_shards": torch.equal(out_bf16, fused_bf16),
        }
    return result
