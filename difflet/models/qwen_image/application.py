"""Qwen-Image Trainium application.

M4a starts with the DiT transformer boundary. The Qwen2.5-VL text encoder,
scheduler loop, and Qwen VAE remain host-side until transformer parity and
shape coverage are established.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from difflet.backends.trainium.core.config import NeuronConfig
from difflet.backends.trainium.core.multi_component_application import (
    ComponentSpec,
    MultiComponentApplication,
)
from difflet.utils.diffusers_adapter import load_diffusers_config


@dataclass(frozen=True)
class QwenImageDiTInputBundle:
    """Host-side contract for one Qwen-Image transformer call."""

    hidden_states: torch.Tensor
    timestep: torch.Tensor
    encoder_hidden_states: torch.Tensor
    encoder_hidden_states_mask: torch.Tensor
    guidance: torch.Tensor

    def as_model_inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            self.hidden_states,
            self.timestep,
            self.encoder_hidden_states,
            self.encoder_hidden_states_mask,
            self.guidance,
        )


def validate_qwen_image_dit_inputs(
    bundle: QwenImageDiTInputBundle,
    *,
    config: Any,
    dtype: torch.dtype,
) -> None:
    """Validate the M4a latent/text embedding contract before Trainium dispatch."""

    batch_size = int(getattr(config.neuron_config, "batch_size", 1))
    text_seq_len = int(getattr(config, "text_seq_len", 1024))
    expected = {
        "hidden_states": (
            (batch_size, int(config.image_seq_len), int(config.in_channels)),
            dtype,
        ),
        "timestep": ((batch_size,), dtype),
        "encoder_hidden_states": (
            (batch_size, text_seq_len, int(config.joint_attention_dim)),
            dtype,
        ),
        "encoder_hidden_states_mask": ((batch_size, text_seq_len), torch.bool),
        "guidance": ((batch_size,), dtype),
    }
    tensors = {
        "hidden_states": bundle.hidden_states,
        "timestep": bundle.timestep,
        "encoder_hidden_states": bundle.encoder_hidden_states,
        "encoder_hidden_states_mask": bundle.encoder_hidden_states_mask,
        "guidance": bundle.guidance,
    }
    for name, tensor in tensors.items():
        shape, tensor_dtype = expected[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"Qwen-Image DiT input {name!r} has shape {tuple(tensor.shape)}, "
                f"expected {shape}."
            )
        if tensor.dtype != tensor_dtype:
            raise TypeError(
                f"Qwen-Image DiT input {name!r} has dtype {tensor.dtype}, "
                f"expected {tensor_dtype}."
            )


def create_qwen_image_transformer_config(
    *,
    model_path: str,
    world_size: int,
    tp_degree: int,
    dtype: torch.dtype,
    height: int,
    width: int,
    text_seq_len: int = 1024,
    batch_size: int = 1,
    context_parallel_enabled: bool = False,
):
    from difflet.backends.trainium.qwen_image.transformer import (
        QwenImageTransformerInferenceConfig,
    )

    transformer_path = os.path.join(model_path, "transformer")
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
        skip_sharding=True,
    )
    return QwenImageTransformerInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_diffusers_config(transformer_path),
        height=height,
        width=width,
        text_seq_len=text_seq_len,
        context_parallel_enabled=context_parallel_enabled,
    )


def _normalize_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported Qwen-Image dtype: {dtype!r}")


class NeuronQwenImageApplication(MultiComponentApplication):
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
            "height": int(shape.get("height") or 1024),
            "width": int(shape.get("width") or 1024),
            "num_frames": None,
        }
        self.kwargs = kwargs
        self.transformer_path = os.path.join(model_path, "transformer")
        self.transformer = None
        self.teacache_probe = None
        self.teacache_probe_fused = False
        self.text_seq_len = int(kwargs.get("text_seq_len", 1024))
        self.batch_size = int(kwargs.get("batch_size", 1))

        enable_transformer = bool(kwargs.get("enable_transformer", True))
        transformer_config_path = os.path.join(self.transformer_path, "config.json")
        if enable_transformer and os.path.exists(transformer_config_path):
            from difflet.backends.trainium.qwen_image.transformer import (
                NeuronQwenImageTransformerApplication,
            )

            config = create_qwen_image_transformer_config(
                model_path=model_path,
                world_size=parallel.world_size,
                tp_degree=parallel.tp_degree,
                dtype=self.dtype,
                height=self.shape["height"],
                width=self.shape["width"],
                text_seq_len=self.text_seq_len,
                batch_size=self.batch_size,
                context_parallel_enabled=parallel.cp_degree > 1,
            )
            self.transformer = NeuronQwenImageTransformerApplication(
                model_path=self.transformer_path,
                config=config,
            )

            if bool(kwargs.get("teacache_fused", False)):
                from difflet.backends.trainium.qwen_image.teacache_probe_fused import (
                    NeuronQwenImageTeacacheProbeFusedApplication,
                )

                self.teacache_probe = NeuronQwenImageTeacacheProbeFusedApplication(
                    model_path=self.transformer_path,
                    config=config,
                )
                self.teacache_probe_fused = True

        from difflet.models.qwen_image.pipeline import QwenImageOrchestrator

        self.pipeline = QwenImageOrchestrator(
            model_path=model_path,
            transformer=self if self.transformer is not None else None,
            dtype=self.dtype,
            height=self.shape["height"],
            width=self.shape["width"],
            text_seq_len=self.text_seq_len,
            teacache_speedup=kwargs.get("teacache_speedup"),
            teacache_calibration_path=kwargs.get("teacache_calibration_path"),
        )

    def components(self) -> list[ComponentSpec]:
        components: list[ComponentSpec] = []
        if self.transformer is not None:
            components.append(ComponentSpec("transformer", self.transformer))
        if self.teacache_probe is not None:
            components.append(ComponentSpec("teacache_probe", self.teacache_probe))
        return components

    def no_components_message(self, action: str) -> str:
        if action == "compile":
            return (
                "Qwen-Image compile requires transformer/config.json. "
                "The current app has no active compile component."
            )
        if action == "load":
            return "Qwen-Image load requires compiled transformer artifacts."
        return super().no_components_message(action)

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        if self.transformer is None:
            raise NotImplementedError("Qwen-Image DiT contract requires an active transformer.")
        config = self.transformer.config
        batch_size = int(getattr(config.neuron_config, "batch_size", 1))
        text_seq_len = int(getattr(config, "text_seq_len", 1024))
        return {
            "hidden_states": {
                "shape": (batch_size, int(config.image_seq_len), int(config.in_channels)),
                "dtype": self.dtype,
            },
            "timestep": {"shape": (batch_size,), "dtype": self.dtype},
            "encoder_hidden_states": {
                "shape": (batch_size, text_seq_len, int(config.joint_attention_dim)),
                "dtype": self.dtype,
            },
            "encoder_hidden_states_mask": {
                "shape": (batch_size, text_seq_len),
                "dtype": torch.bool,
            },
            "guidance": {"shape": (batch_size,), "dtype": self.dtype},
        }

    def forward_dit(self, bundle: QwenImageDiTInputBundle):
        if self.transformer is None:
            raise NotImplementedError("Qwen-Image forward_dit requires an active transformer.")
        validate_qwen_image_dit_inputs(bundle, config=self.transformer.config, dtype=self.dtype)
        return self.transformer(*bundle.as_model_inputs())

    def teacache_delta(self, bundle: QwenImageDiTInputBundle) -> torch.Tensor:
        """fused-A entry (cclog 81): returns ONLY the scalar delta. prev_mod is a
        persistent on-device Parameter updated in place via alias — no host
        handle. Requires the fused probe (teacache_fused=True)."""
        if not self.teacache_probe_fused or self.teacache_probe is None:
            raise NotImplementedError(
                "teacache_delta requires the fused probe (teacache_fused=True)."
            )
        return self.teacache_probe.teacache_delta(*bundle.as_model_inputs())

    def __call__(self, *args: Any, **kwargs: Any):
        if len(args) == 1 and isinstance(args[0], QwenImageDiTInputBundle):
            return self.forward_dit(args[0])
        direct_keys = {
            "hidden_states",
            "timestep",
            "encoder_hidden_states",
            "encoder_hidden_states_mask",
            "guidance",
        }
        if not args and direct_keys.issubset(kwargs):
            bundle = QwenImageDiTInputBundle(**kwargs)
            return self.forward_dit(bundle)
        if self.transformer is not None and args:
            return self.transformer(*args, **kwargs)
        if self.pipeline is not None and self.pipeline.has_runtime_components():
            return self.pipeline(*args, **kwargs)
        del args, kwargs
        raise NotImplementedError("Qwen-Image end-to-end inference is not implemented yet")
