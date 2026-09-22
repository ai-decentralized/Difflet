"""LTX-2 Trainium application.

M4c starts at the dual-stream DiT transformer boundary. Text encoding,
connectors, scheduler setup, video VAE, audio VAE, and vocoder stay host-side
until the transformer path has compile/runtime evidence.
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

LTX_2_DEFAULT_TEXT_SEQ_LEN = 1024
LTX_2_DEFAULT_HEIGHT = 512
LTX_2_DEFAULT_WIDTH = 768
LTX_2_DEFAULT_NUM_FRAMES = 121


@dataclass(frozen=True)
class LTX2DiTInputBundle:
    """Host-side contract for one LTX-2 audiovisual transformer call."""

    hidden_states: torch.Tensor
    audio_hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor
    audio_encoder_hidden_states: torch.Tensor
    timestep: torch.Tensor
    sigma: torch.Tensor
    encoder_attention_mask: torch.Tensor
    audio_encoder_attention_mask: torch.Tensor
    video_coords: torch.Tensor
    audio_coords: torch.Tensor

    def as_model_inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            self.hidden_states,
            self.audio_hidden_states,
            self.encoder_hidden_states,
            self.audio_encoder_hidden_states,
            self.timestep,
            self.sigma,
            self.encoder_attention_mask,
            self.audio_encoder_attention_mask,
            self.video_coords,
            self.audio_coords,
        )


def validate_ltx_2_dit_inputs(
    bundle: LTX2DiTInputBundle,
    *,
    config: Any,
    dtype: torch.dtype,
) -> None:
    """Validate the M4c dual-stream latent/text embedding contract."""

    batch_size = int(getattr(config.neuron_config, "batch_size", 1))
    text_seq_len = int(getattr(config, "text_seq_len", LTX_2_DEFAULT_TEXT_SEQ_LEN))
    audio_text_seq_len = int(getattr(config, "audio_text_seq_len", text_seq_len))
    video_text_dim = int(getattr(config, "video_text_dim"))
    audio_text_dim = int(getattr(config, "audio_text_dim"))
    expected = {
        "hidden_states": ((batch_size, int(config.video_seq_len), int(config.in_channels)), dtype),
        "audio_hidden_states": (
            (batch_size, int(config.audio_seq_len), int(config.audio_in_channels)),
            dtype,
        ),
        "encoder_hidden_states": ((batch_size, text_seq_len, video_text_dim), dtype),
        "audio_encoder_hidden_states": (
            (batch_size, audio_text_seq_len, audio_text_dim),
            dtype,
        ),
        "timestep": ((batch_size,), dtype),
        "sigma": ((batch_size,), dtype),
        "encoder_attention_mask": ((batch_size, text_seq_len), torch.bool),
        "audio_encoder_attention_mask": ((batch_size, audio_text_seq_len), torch.bool),
        "video_coords": ((batch_size, 3, int(config.video_seq_len), 2), torch.float32),
        "audio_coords": ((batch_size, 1, int(config.audio_seq_len), 2), torch.float32),
    }
    tensors = {
        "hidden_states": bundle.hidden_states,
        "audio_hidden_states": bundle.audio_hidden_states,
        "encoder_hidden_states": bundle.encoder_hidden_states,
        "audio_encoder_hidden_states": bundle.audio_encoder_hidden_states,
        "timestep": bundle.timestep,
        "sigma": bundle.sigma,
        "encoder_attention_mask": bundle.encoder_attention_mask,
        "audio_encoder_attention_mask": bundle.audio_encoder_attention_mask,
        "video_coords": bundle.video_coords,
        "audio_coords": bundle.audio_coords,
    }
    for name, tensor in tensors.items():
        shape, tensor_dtype = expected[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"LTX-2 DiT input {name!r} has shape {tuple(tensor.shape)}, expected {shape}."
            )
        if tensor.dtype != tensor_dtype:
            raise TypeError(
                f"LTX-2 DiT input {name!r} has dtype {tensor.dtype}, expected {tensor_dtype}."
            )


def create_ltx_2_transformer_config(
    *,
    model_path: str,
    world_size: int,
    tp_degree: int,
    dtype: torch.dtype,
    height: int,
    width: int,
    num_frames: int,
    text_seq_len: int = LTX_2_DEFAULT_TEXT_SEQ_LEN,
    audio_text_seq_len: int | None = None,
    audio_num_frames: int | None = None,
    frame_rate: float = 24.0,
    batch_size: int = 1,
    cfg_parallel_enabled: bool = False,
):
    from difflet.backends.trainium.ltx_2.transformer import LTX2TransformerInferenceConfig

    transformer_path = os.path.join(model_path, "transformer")
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        world_size=world_size,
        torch_dtype=dtype,
    )
    return LTX2TransformerInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_diffusers_config(transformer_path),
        height=height,
        width=width,
        num_frames=num_frames,
        text_seq_len=text_seq_len,
        audio_text_seq_len=audio_text_seq_len or text_seq_len,
        audio_num_frames=audio_num_frames,
        frame_rate=frame_rate,
        cfg_parallel_enabled=cfg_parallel_enabled,
    )


def _normalize_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported LTX-2 dtype: {dtype!r}")


class NeuronLTX2Application(MultiComponentApplication):
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
            "height": int(shape.get("height") or LTX_2_DEFAULT_HEIGHT),
            "width": int(shape.get("width") or LTX_2_DEFAULT_WIDTH),
            "num_frames": int(shape.get("num_frames") or LTX_2_DEFAULT_NUM_FRAMES),
        }
        self.kwargs = kwargs
        self.transformer_path = os.path.join(model_path, "transformer")
        self.transformer = None
        self.teacache_probe = None
        self.text_seq_len = int(kwargs.get("text_seq_len", LTX_2_DEFAULT_TEXT_SEQ_LEN))
        self.audio_text_seq_len = int(kwargs.get("audio_text_seq_len", self.text_seq_len))
        self.audio_num_frames = kwargs.get("audio_num_frames")
        if self.audio_num_frames is not None:
            self.audio_num_frames = int(self.audio_num_frames)
        self.frame_rate = float(kwargs.get("frame_rate", 24.0))
        self.batch_size = int(kwargs.get("batch_size", 1))
        self.host_pipeline = None

        # CFG parallel adds a 2-way data-parallel lane (world_size = tp * 2) and
        # compiles the transformer at batch=2 ([uncond, cond]) before scattering
        # one branch per rank.
        cfg_parallel_enabled = bool(getattr(parallel, "cfg_parallel_enabled", False))
        self.cfg_parallel_enabled = cfg_parallel_enabled
        transformer_world_size = parallel.tp_degree * 2 if cfg_parallel_enabled else parallel.tp_degree
        transformer_batch_size = 2 if cfg_parallel_enabled else self.batch_size

        enable_transformer = bool(kwargs.get("enable_transformer", True))
        transformer_mode = str(kwargs.get("transformer_mode", "single"))
        transformer_config_path = os.path.join(self.transformer_path, "config.json")
        if enable_transformer and os.path.exists(transformer_config_path):
            config = create_ltx_2_transformer_config(
                model_path=model_path,
                world_size=transformer_world_size,
                tp_degree=parallel.tp_degree,
                dtype=self.dtype,
                height=self.shape["height"],
                width=self.shape["width"],
                num_frames=self.shape["num_frames"],
                text_seq_len=self.text_seq_len,
                audio_text_seq_len=self.audio_text_seq_len,
                audio_num_frames=self.audio_num_frames,
                frame_rate=self.frame_rate,
                batch_size=transformer_batch_size,
                cfg_parallel_enabled=cfg_parallel_enabled,
            )
            if transformer_mode == "single":
                from difflet.backends.trainium.ltx_2.transformer import (
                    NeuronLTX2TransformerApplication,
                )

                self.transformer = NeuronLTX2TransformerApplication(
                    model_path=self.transformer_path,
                    config=config,
                )
            elif transformer_mode == "segmented":
                from difflet.backends.trainium.ltx_2.segmented import (
                    LTX2SegmentedTransformerApplication,
                )

                self.transformer = LTX2SegmentedTransformerApplication(
                    model_path=self.transformer_path,
                    config=config,
                    block_load_mode=str(kwargs.get("segmented_block_load_mode", "streaming")),
                )
            else:
                raise ValueError(
                    "LTX-2 transformer_mode must be 'single' or 'segmented', "
                    f"got {transformer_mode!r}."
                )

            # TeaCache fused-A device probe. Opt-in, because it adds a NEFF to
            # the compiled artifact — the same ``teacache_fused`` contract Flux
            # and HunyuanVideo use. Single mode only: there the host CPU
            # transformer exists purely for this signal, so moving it on device
            # retires a full host copy of the model. In segmented mode that copy
            # is load-bearing for the frontend and final projection, so the
            # signal is already free and the host path stays.
            if bool(kwargs.get("teacache_fused", False)):
                if transformer_mode != "single":
                    raise ValueError(
                        "LTX-2 teacache_fused requires transformer_mode='single'. In "
                        "segmented mode the host transformer is already resident for "
                        "the frontend, so the TeaCache signal costs nothing on host."
                    )
                from difflet.backends.trainium.ltx_2.teacache_probe_fused import (
                    NeuronLTX2TeacacheProbeFusedApplication,
                )

                # Same config object as the backbone, so the probe shares its
                # weight-store entry and resolves to its shards.
                self.teacache_probe = NeuronLTX2TeacacheProbeFusedApplication(
                    model_path=self.transformer_path,
                    config=config,
                )

        if bool(kwargs.get("enable_host_pipeline", False)):
            self.host_pipeline = _load_ltx_2_host_pipeline(
                model_path=model_path,
                dtype=self.dtype,
                device=kwargs.get("host_device", "cpu"),
                load_decode_components=bool(kwargs.get("enable_decode_components", True)),
            )

        from difflet.models.ltx_2.pipeline import LTX2Orchestrator

        self.pipeline = LTX2Orchestrator(
            model_path=model_path,
            transformer=self if self.transformer is not None else None,
            vae=getattr(self.host_pipeline, "vae", None) if self.host_pipeline is not None else None,
            audio_vae=(
                getattr(self.host_pipeline, "audio_vae", None)
                if self.host_pipeline is not None
                else None
            ),
            vocoder=(
                getattr(self.host_pipeline, "vocoder", None)
                if self.host_pipeline is not None
                else None
            ),
            video_processor=(
                getattr(self.host_pipeline, "video_processor", None)
                if self.host_pipeline is not None
                else None
            ),
            host_pipeline=self.host_pipeline,
            dtype=self.dtype,
            height=self.shape["height"],
            width=self.shape["width"],
            num_frames=self.shape["num_frames"],
            text_seq_len=self.text_seq_len,
            audio_text_seq_len=self.audio_text_seq_len,
            audio_num_frames=self.audio_num_frames,
            frame_rate=self.frame_rate,
            teacache_calibration_path=self.kwargs.get("teacache_calibration_path"),
            # Probe-free modes: runtime-only (_RUNTIME_ONLY_APP_KWARGS), never
            # part of the compile-cache key.
            teacache_cadence=self.kwargs.get("teacache_cadence"),
            teacache_online_delta_alpha=self.kwargs.get("teacache_online_delta_alpha"),
        )

    def components(self) -> list[ComponentSpec]:
        components: list[ComponentSpec] = []
        if self.transformer is not None:
            component_specs = getattr(self.transformer, "component_specs", None)
            if component_specs is not None:
                components.extend(component_specs(prefix="transformer"))
            else:
                components.append(ComponentSpec("transformer", self.transformer))
        if self.teacache_probe is not None:
            # Same world_size as the backbone, and it shares the backbone's
            # weight-store entry, so it loads alongside the transformer.
            components.append(ComponentSpec("teacache_probe", self.teacache_probe))
        return components

    def load(self, compiled_model_path: str, *args: Any, **kwargs: Any) -> None:
        super().load(compiled_model_path, *args, **kwargs)
        set_compiled_model_path = getattr(self.transformer, "set_compiled_model_path", None)
        if set_compiled_model_path is not None:
            set_compiled_model_path(compiled_model_path)

    def no_components_message(self, action: str) -> str:
        if action == "compile":
            return (
                "LTX-2 compile requires transformer/config.json. "
                "The current app has no active compile component."
            )
        if action == "load":
            return "LTX-2 load requires compiled transformer artifacts."
        return super().no_components_message(action)

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        if self.transformer is None:
            raise NotImplementedError("LTX-2 DiT contract requires an active transformer.")
        config = self.transformer.config
        batch_size = int(getattr(config.neuron_config, "batch_size", 1))
        text_seq_len = int(getattr(config, "text_seq_len", LTX_2_DEFAULT_TEXT_SEQ_LEN))
        audio_text_seq_len = int(getattr(config, "audio_text_seq_len", text_seq_len))
        return {
            "hidden_states": {
                "shape": (batch_size, int(config.video_seq_len), int(config.in_channels)),
                "dtype": self.dtype,
            },
            "audio_hidden_states": {
                "shape": (batch_size, int(config.audio_seq_len), int(config.audio_in_channels)),
                "dtype": self.dtype,
            },
            "encoder_hidden_states": {
                "shape": (batch_size, text_seq_len, int(config.video_text_dim)),
                "dtype": self.dtype,
            },
            "audio_encoder_hidden_states": {
                "shape": (batch_size, audio_text_seq_len, int(config.audio_text_dim)),
                "dtype": self.dtype,
            },
            "timestep": {"shape": (batch_size,), "dtype": self.dtype},
            "sigma": {"shape": (batch_size,), "dtype": self.dtype},
            "encoder_attention_mask": {"shape": (batch_size, text_seq_len), "dtype": torch.bool},
            "audio_encoder_attention_mask": {
                "shape": (batch_size, audio_text_seq_len),
                "dtype": torch.bool,
            },
            "video_coords": {
                "shape": (batch_size, 3, int(config.video_seq_len), 2),
                "dtype": torch.float32,
            },
            "audio_coords": {
                "shape": (batch_size, 1, int(config.audio_seq_len), 2),
                "dtype": torch.float32,
            },
        }

    def forward_dit(self, bundle: LTX2DiTInputBundle):
        if self.transformer is None:
            raise NotImplementedError("LTX-2 forward_dit requires an active transformer.")
        validate_ltx_2_dit_inputs(bundle, config=self.transformer.config, dtype=self.dtype)
        return self.transformer(*bundle.as_model_inputs())

    def teacache_mod_input(self, hidden_states, timestep):
        """Delegate the TeaCache block-0 modulated-input signal to the backend runtime."""
        if self.transformer is None:
            raise NotImplementedError("LTX-2 teacache_mod_input requires an active transformer.")
        return self.transformer.teacache_mod_input(hidden_states, timestep)

    @property
    def teacache_probe_fused(self) -> bool:
        """Whether a fused device probe is mounted for the TeaCache signal."""
        return self.teacache_probe is not None

    def teacache_delta(self, hidden_states, timestep):
        """Relative-L1 of the block-0 modulated input, computed on device.

        Only the scalar crosses to host; ``prev_mod`` lives on the probe and is
        updated in place, so no modulated-input tensor is copied per step.
        """
        if self.teacache_probe is None:
            raise NotImplementedError(
                "LTX-2 teacache_delta requires teacache_fused=True (no probe is mounted)."
            )
        return self.teacache_probe.teacache_delta(hidden_states, timestep)

    def __call__(self, *args: Any, **kwargs: Any):
        if len(args) == 1 and isinstance(args[0], LTX2DiTInputBundle):
            return self.forward_dit(args[0])
        direct_keys = {
            "hidden_states",
            "audio_hidden_states",
            "encoder_hidden_states",
            "audio_encoder_hidden_states",
            "timestep",
            "sigma",
            "encoder_attention_mask",
            "audio_encoder_attention_mask",
            "video_coords",
            "audio_coords",
        }
        if not args and direct_keys.issubset(kwargs):
            bundle = LTX2DiTInputBundle(**kwargs)
            return self.forward_dit(bundle)
        if self.transformer is not None and args:
            return self.transformer(*args, **kwargs)
        if self.pipeline is not None and self.pipeline.has_runtime_components():
            return self.pipeline(*args, **kwargs)
        del args, kwargs
        raise NotImplementedError("LTX-2 end-to-end inference is not implemented yet")


def _load_ltx_2_host_pipeline(
    *,
    model_path: str,
    dtype: torch.dtype,
    device: str | torch.device,
    load_decode_components: bool,
) -> Any:
    from difflet.models.ltx_2.pipeline import disable_ltx_2_xla_lazy_import

    disable_ltx_2_xla_lazy_import()
    from diffusers import LTX2Pipeline

    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "transformer": None,
    }
    if not load_decode_components:
        load_kwargs.update({"vae": None, "audio_vae": None, "vocoder": None})
    pipe = LTX2Pipeline.from_pretrained(model_path, **load_kwargs)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe
