"""HunyuanVideo Trainium application skeleton.

M3 starts with registration, shape/cache plumbing, and explicit scope guards.
The transformer/text/VAE components land after the dual-stream attention
reference path is covered by CPU tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from nova.backends.trainium.core.config import NeuronConfig
from nova.utils.diffusers_adapter import load_diffusers_config


@dataclass(frozen=True)
class HunyuanVideoDiTInputBundle:
    """Host-side contract for one HunyuanVideo DiT call.

    M3 v0 keeps Llama3, CLIP, scheduler setup, and VAE decode outside the
    Trainium graph. The Trainium boundary is exactly this tuple.
    """

    hidden_states: torch.Tensor
    timestep: torch.Tensor
    encoder_hidden_states: torch.Tensor
    encoder_attention_mask: torch.Tensor
    pooled_projections: torch.Tensor
    guidance: torch.Tensor

    def as_model_inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            self.hidden_states,
            self.timestep,
            self.encoder_hidden_states,
            self.encoder_attention_mask,
            self.pooled_projections,
            self.guidance,
        )


def validate_hunyuan_video_dit_inputs(
    bundle: HunyuanVideoDiTInputBundle,
    *,
    config: Any,
    dtype: torch.dtype,
) -> None:
    """Validate the M3 v0 embedding/latent contract before Trainium dispatch."""

    batch_size = int(getattr(config.neuron_config, "batch_size", 1))
    text_seq_len = int(getattr(config, "text_seq_len", 256))
    latent_shape = (
        batch_size,
        int(config.in_channels),
        int(config.latent_frames),
        int(config.latent_height),
        int(config.latent_width),
    )
    expected = {
        "hidden_states": (latent_shape, dtype),
        "timestep": ((batch_size,), dtype),
        "encoder_hidden_states": (
            (batch_size, text_seq_len, int(config.text_embed_dim)),
            dtype,
        ),
        "encoder_attention_mask": ((batch_size, text_seq_len), torch.int64),
        "pooled_projections": ((batch_size, int(config.pooled_projection_dim)), dtype),
        "guidance": ((batch_size,), dtype),
    }
    tensors = {
        "hidden_states": bundle.hidden_states,
        "timestep": bundle.timestep,
        "encoder_hidden_states": bundle.encoder_hidden_states,
        "encoder_attention_mask": bundle.encoder_attention_mask,
        "pooled_projections": bundle.pooled_projections,
        "guidance": bundle.guidance,
    }
    for name, tensor in tensors.items():
        shape, tensor_dtype = expected[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"HunyuanVideo DiT input {name!r} has shape {tuple(tensor.shape)}, "
                f"expected {shape}."
            )
        if tensor.dtype != tensor_dtype:
            raise TypeError(
                f"HunyuanVideo DiT input {name!r} has dtype {tensor.dtype}, "
                f"expected {tensor_dtype}."
            )


def create_hunyuan_video_backbone_config(
    *,
    model_path: str,
    world_size: int,
    tp_degree: int,
    dtype: torch.dtype,
    height: int,
    width: int,
    num_frames: int,
    text_seq_len: int = 256,
    batch_size: int = 1,
):
    from nova.backends.trainium.hunyuan_video.backbone import (
        HunyuanVideoBackboneInferenceConfig,
    )

    transformer_path = os.path.join(model_path, "transformer")
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
        skip_sharding=True,
    )
    return HunyuanVideoBackboneInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_diffusers_config(transformer_path),
        height=height,
        width=width,
        num_frames=num_frames,
        text_seq_len=text_seq_len,
    )


def _normalize_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported HunyuanVideo dtype: {dtype!r}")


class NeuronHunyuanVideoApplication(nn.Module):
    def __init__(
        self,
        *,
        model_path: str,
        parallel,
        dtype: Any,
        shape: dict[str, int | None],
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.parallel = parallel
        self.dtype = _normalize_dtype(dtype)
        self.shape = {
            "height": int(shape.get("height") or 320),
            "width": int(shape.get("width") or 512),
            "num_frames": int(shape.get("num_frames") or 61),
        }
        self.kwargs = kwargs
        self.transformer_path = os.path.join(model_path, "transformer")
        self.transformer = None
        self.pipeline = None
        self.text_seq_len = int(kwargs.get("text_seq_len", 256))
        self.batch_size = int(kwargs.get("batch_size", 1))

        enable_transformer = bool(kwargs.get("enable_transformer", True))
        transformer_config_path = os.path.join(self.transformer_path, "config.json")
        if enable_transformer and os.path.exists(transformer_config_path):
            from nova.backends.trainium.hunyuan_video.backbone import (
                NeuronHunyuanVideoBackboneApplication,
            )

            config = create_hunyuan_video_backbone_config(
                model_path=model_path,
                world_size=parallel.tp_degree,
                tp_degree=parallel.tp_degree,
                dtype=self.dtype,
                height=self.shape["height"],
                width=self.shape["width"],
                num_frames=self.shape["num_frames"],
                text_seq_len=self.text_seq_len,
                batch_size=self.batch_size,
            )
            self.transformer = NeuronHunyuanVideoBackboneApplication(
                model_path=self.transformer_path,
                config=config,
            )
        from nova.models.hunyuan_video.pipeline import HunyuanVideoOrchestrator

        self.pipeline = HunyuanVideoOrchestrator(
            model_path=model_path,
            transformer=self if self.transformer is not None else None,
            dtype=self.dtype,
            height=self.shape["height"],
            width=self.shape["width"],
            num_frames=self.shape["num_frames"],
        )

    def _components(self):
        if self.transformer is not None:
            yield "transformer", self.transformer

    def compile(self, compiled_model_path: str, debug: bool = False) -> None:
        components = list(self._components())
        if not components:
            raise NotImplementedError(
                "HunyuanVideo compile requires transformer/config.json. "
                "The current app has no active compile component."
            )
        for name, component in components:
            component.compile(os.path.join(compiled_model_path, name), debug=debug)

    def has_compiled_artifacts(self, compiled_model_path: str) -> bool:
        components = list(self._components())
        if not components:
            return True
        for name, _component in components:
            component_path = os.path.join(compiled_model_path, name)
            if not os.path.exists(os.path.join(component_path, "model.pt")):
                return False
            if not os.path.exists(os.path.join(component_path, "neuron_config.json")):
                return False
        return True

    def load(
        self,
        compiled_model_path: str,
        start_rank_id: int | None = None,
        local_ranks_size: int | None = None,
        skip_warmup: bool = False,
    ) -> None:
        components = list(self._components())
        if not components:
            raise NotImplementedError("HunyuanVideo load requires compiled component artifacts")
        for name, component in components:
            component.load(
                os.path.join(compiled_model_path, name),
                start_rank_id=start_rank_id,
                local_ranks_size=local_ranks_size,
                skip_warmup=skip_warmup,
            )

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        if self.transformer is None:
            raise NotImplementedError("HunyuanVideo DiT contract requires an active transformer.")
        config = self.transformer.config
        batch_size = int(getattr(config.neuron_config, "batch_size", 1))
        text_seq_len = int(getattr(config, "text_seq_len", 256))
        return {
            "hidden_states": {
                "shape": (
                    batch_size,
                    int(config.in_channels),
                    int(config.latent_frames),
                    int(config.latent_height),
                    int(config.latent_width),
                ),
                "dtype": self.dtype,
            },
            "timestep": {"shape": (batch_size,), "dtype": self.dtype},
            "encoder_hidden_states": {
                "shape": (batch_size, text_seq_len, int(config.text_embed_dim)),
                "dtype": self.dtype,
            },
            "encoder_attention_mask": {
                "shape": (batch_size, text_seq_len),
                "dtype": torch.int64,
            },
            "pooled_projections": {
                "shape": (batch_size, int(config.pooled_projection_dim)),
                "dtype": self.dtype,
            },
            "guidance": {"shape": (batch_size,), "dtype": self.dtype},
        }

    def forward_dit(self, bundle: HunyuanVideoDiTInputBundle):
        if self.transformer is None:
            raise NotImplementedError("HunyuanVideo forward_dit requires an active transformer.")
        validate_hunyuan_video_dit_inputs(
            bundle,
            config=self.transformer.config,
            dtype=self.dtype,
        )
        return self.transformer(*bundle.as_model_inputs())

    def __call__(self, *args: Any, **kwargs: Any):
        if len(args) == 1 and isinstance(args[0], HunyuanVideoDiTInputBundle):
            return self.forward_dit(args[0])
        direct_keys = {
            "hidden_states",
            "timestep",
            "encoder_hidden_states",
            "encoder_attention_mask",
            "pooled_projections",
            "guidance",
        }
        if not args and direct_keys.issubset(kwargs):
            bundle = HunyuanVideoDiTInputBundle(**kwargs)
            return self.forward_dit(bundle)
        if self.transformer is not None and args:
            return self.transformer(*args, **kwargs)
        if self.pipeline is not None and self.pipeline.has_runtime_components():
            return self.pipeline(*args, **kwargs)
        del args, kwargs
        raise NotImplementedError("HunyuanVideo end-to-end inference is not implemented yet")
