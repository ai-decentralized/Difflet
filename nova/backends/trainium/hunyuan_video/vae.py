"""Trainium application wrapper for the HunyuanVideo VAE decoder."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nova.backends.trainium.core.application_base import NeuronApplicationBase
from nova.backends.trainium.core.config import InferenceConfig
from nova.backends.trainium.core.multi_component_application import ComponentSpec
from nova.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from nova.models.hunyuan_video.vae.modeling_vae import (
    HunyuanVideoVAEDecoderConfig,
    HunyuanVideoVAEDecoderModel,
)


class HunyuanVideoVAEDecoderInferenceConfig(InferenceConfig):
    """Inference config for decoder-only HunyuanVideo VAE tile compile."""

    def add_derived_config(self):
        super().add_derived_config()
        self.tile_sample_min_height = int(getattr(self, "tile_sample_min_height", 256))
        self.tile_sample_min_width = int(getattr(self, "tile_sample_min_width", 256))
        self.tile_sample_min_num_frames = int(getattr(self, "tile_sample_min_num_frames", 16))
        self.tile_sample_stride_height = int(getattr(self, "tile_sample_stride_height", 192))
        self.tile_sample_stride_width = int(getattr(self, "tile_sample_stride_width", 192))
        self.tile_sample_stride_num_frames = int(getattr(self, "tile_sample_stride_num_frames", 12))
        if not hasattr(self, "scaling_factor"):
            self.scaling_factor = 0.476986

    def get_required_attributes(self) -> List[str]:
        return [
            "out_channels",
            "latent_channels",
            "up_block_types",
            "block_out_channels",
            "layers_per_block",
            "act_fn",
            "norm_num_groups",
            "scaling_factor",
            "spatial_compression_ratio",
            "temporal_compression_ratio",
            "mid_block_add_attention",
            "height",
            "width",
            "num_frames",
        ]

    @property
    def latent_height(self) -> int:
        return int(self.height) // int(self.spatial_compression_ratio)

    @property
    def latent_width(self) -> int:
        return int(self.width) // int(self.spatial_compression_ratio)

    @property
    def latent_frames(self) -> int:
        return (int(self.num_frames) - 1) // int(self.temporal_compression_ratio) + 1

    @property
    def tile_latent_height(self) -> int:
        return int(self.tile_sample_min_height) // int(self.spatial_compression_ratio)

    @property
    def tile_latent_width(self) -> int:
        return int(self.tile_sample_min_width) // int(self.spatial_compression_ratio)

    @property
    def tile_latent_frames(self) -> int:
        return int(self.tile_sample_min_num_frames) // int(self.temporal_compression_ratio) + 1

    @property
    def tile_latent_stride_height(self) -> int:
        return int(self.tile_sample_stride_height) // int(self.spatial_compression_ratio)

    @property
    def tile_latent_stride_width(self) -> int:
        return int(self.tile_sample_stride_width) // int(self.spatial_compression_ratio)

    @property
    def tile_latent_stride_num_frames(self) -> int:
        return int(self.tile_sample_stride_num_frames) // int(self.temporal_compression_ratio)

    def validate_config(self):
        super().validate_config()
        if int(self.height) % int(self.spatial_compression_ratio) != 0:
            raise ValueError("HunyuanVideo VAE height must be divisible by spatial compression ratio.")
        if int(self.width) % int(self.spatial_compression_ratio) != 0:
            raise ValueError("HunyuanVideo VAE width must be divisible by spatial compression ratio.")
        if int(self.spatial_compression_ratio) != 8:
            raise NotImplementedError("HunyuanVideo VAE spike expects spatial compression ratio 8.")
        if int(self.temporal_compression_ratio) != 4:
            raise NotImplementedError("HunyuanVideo VAE spike expects temporal compression ratio 4.")
        if int(self.tile_latent_stride_num_frames) <= 0:
            raise ValueError("HunyuanVideo VAE tile frame stride must be positive.")


class ModelWrapperHunyuanVideoVAEDecoder(ModelWrapper):
    """ModelBuilder wrapper for HunyuanVideo VAE tile decoder compile inputs."""

    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag: str = "",
        compiler_args: str | None = None,
        priority_model_idx: int | None = None,
        model_init_kwargs=None,
    ) -> None:
        super().__init__(
            config=config,
            model_cls=model_cls,
            tag=tag,
            compiler_args=compiler_args,
            priority_model_idx=priority_model_idx,
            model_init_kwargs=model_init_kwargs or {},
        )
        self.bucket_config = None

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        batch_size = int(getattr(self.config.neuron_config, "batch_size", 1))
        dtype = self.config.neuron_config.torch_dtype
        return [
            (
                torch.randn(
                    [
                        batch_size,
                        int(self.config.latent_channels),
                        int(self.config.tile_latent_frames),
                        int(self.config.tile_latent_height),
                        int(self.config.tile_latent_width),
                    ],
                    dtype=dtype,
                ),
            )
        ]

    def get_model_instance(self):
        def _create_model():
            cfg = HunyuanVideoVAEDecoderConfig(
                out_channels=int(self.config.out_channels),
                latent_channels=int(self.config.latent_channels),
                up_block_types=tuple(self.config.up_block_types),
                block_out_channels=tuple(self.config.block_out_channels),
                layers_per_block=int(self.config.layers_per_block),
                act_fn=str(self.config.act_fn),
                norm_num_groups=int(self.config.norm_num_groups),
                scaling_factor=float(self.config.scaling_factor),
                spatial_compression_ratio=int(self.config.spatial_compression_ratio),
                temporal_compression_ratio=int(self.config.temporal_compression_ratio),
                mid_block_add_attention=bool(self.config.mid_block_add_attention),
                tile_sample_min_height=int(self.config.tile_sample_min_height),
                tile_sample_min_width=int(self.config.tile_sample_min_width),
                tile_sample_min_num_frames=int(self.config.tile_sample_min_num_frames),
                tile_sample_stride_height=int(self.config.tile_sample_stride_height),
                tile_sample_stride_width=int(self.config.tile_sample_stride_width),
                tile_sample_stride_num_frames=int(self.config.tile_sample_stride_num_frames),
            )
            model = self.model_cls(cfg)
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(self, latents):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(latents)


@dataclass(frozen=True)
class _VAESegmentSpec:
    name: str
    model_cls: type[nn.Module]
    input_shape: tuple[int, ...]
    model_kwargs: dict[str, Any] | None = None


def _decoder_config_from_inference_config(config: InferenceConfig) -> HunyuanVideoVAEDecoderConfig:
    return HunyuanVideoVAEDecoderConfig(
        out_channels=int(config.out_channels),
        latent_channels=int(config.latent_channels),
        up_block_types=tuple(config.up_block_types),
        block_out_channels=tuple(config.block_out_channels),
        layers_per_block=int(config.layers_per_block),
        act_fn=str(config.act_fn),
        norm_num_groups=int(config.norm_num_groups),
        scaling_factor=float(config.scaling_factor),
        spatial_compression_ratio=int(config.spatial_compression_ratio),
        temporal_compression_ratio=int(config.temporal_compression_ratio),
        mid_block_add_attention=bool(config.mid_block_add_attention),
        tile_sample_min_height=int(config.tile_sample_min_height),
        tile_sample_min_width=int(config.tile_sample_min_width),
        tile_sample_min_num_frames=int(config.tile_sample_min_num_frames),
        tile_sample_stride_height=int(config.tile_sample_stride_height),
        tile_sample_stride_width=int(config.tile_sample_stride_width),
        tile_sample_stride_num_frames=int(config.tile_sample_stride_num_frames),
    )


def _make_decoder_model(config: InferenceConfig) -> HunyuanVideoVAEDecoderModel:
    model = HunyuanVideoVAEDecoderModel(_decoder_config_from_inference_config(config))
    return model


def _attach_module(root: nn.Module, path: str, module: nn.Module) -> None:
    parent = root
    parts = path.split(".")
    for name in parts[:-1]:
        child = parent._modules.get(name)
        if child is None:
            child = nn.Module()
            parent.add_module(name, child)
        parent = child
    parent.add_module(parts[-1], module)


def _get_module(root: nn.Module, path: str) -> nn.Module:
    current = root
    for name in path.split("."):
        current = current._modules[name]
    return current


class _HunyuanVAEBodyToUp2(nn.Module):
    """Tile decoder prefix through ``decoder.up_blocks[2]``.

    This prefix passed the CPU-vs-NEFF probe and keeps the expensive early
    decoder path in one graph.
    """

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__()
        full = _make_decoder_model(config)
        self.post_quant_conv = full.post_quant_conv
        self.decoder = nn.Module()
        self.decoder.conv_in = full.decoder.conv_in
        self.decoder.mid_block = full.decoder.mid_block
        self.decoder.up_blocks = nn.Module()
        for index in range(3):
            self.decoder.up_blocks.add_module(str(index), full.decoder.up_blocks[index])

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        hidden_states = self.post_quant_conv(latents)
        hidden_states = self.decoder.conv_in(hidden_states)
        hidden_states = self.decoder.mid_block(hidden_states)
        for index in range(3):
            hidden_states = self.decoder.up_blocks._modules[str(index)](hidden_states)
        return hidden_states


class _HunyuanVAEResnetNormAct(nn.Module):
    """One ``GroupNorm -> SiLU`` segment from ``up_blocks[3]``."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        resnet_index: int,
        norm_index: int,
    ) -> None:
        super().__init__()
        full = _make_decoder_model(config)
        block = full.decoder.up_blocks[3].resnets[int(resnet_index)]
        norm_name = f"norm{int(norm_index)}"
        base = f"decoder.up_blocks.3.resnets.{int(resnet_index)}"
        _attach_module(self, f"{base}.{norm_name}", getattr(block, norm_name))
        _attach_module(self, f"{base}.nonlinearity", block.nonlinearity)
        self._base = base
        self._norm_name = norm_name

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.contiguous()
        norm = _get_module(self, f"{self._base}.{self._norm_name}")
        act = _get_module(self, f"{self._base}.nonlinearity")
        return act(norm(hidden_states))


class _HunyuanVAEResnetConv(nn.Module):
    """One causal conv segment from ``up_blocks[3]``."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        resnet_index: int,
        conv_name: str,
    ) -> None:
        super().__init__()
        full = _make_decoder_model(config)
        block = full.decoder.up_blocks[3].resnets[int(resnet_index)]
        base = f"decoder.up_blocks.3.resnets.{int(resnet_index)}"
        _attach_module(self, f"{base}.{conv_name}", getattr(block, conv_name))
        self._path = f"{base}.{conv_name}"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        conv = _get_module(self, self._path)
        return conv(hidden_states.contiguous())


class _HunyuanVAEFinalNormAct(nn.Module):
    """Final ``conv_norm_out -> SiLU`` segment."""

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__()
        full = _make_decoder_model(config)
        self.decoder = nn.Module()
        self.decoder.conv_norm_out = full.decoder.conv_norm_out
        self.decoder.conv_act = full.decoder.conv_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.contiguous()
        return self.decoder.conv_act(self.decoder.conv_norm_out(hidden_states))


class _HunyuanVAEFinalConvOut(nn.Module):
    """Final causal RGB projection."""

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__()
        full = _make_decoder_model(config)
        self.decoder = nn.Module()
        self.decoder.conv_out = full.decoder.conv_out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.decoder.conv_out(hidden_states.contiguous())


class ModelWrapperHunyuanVideoVAEDecoderSegment(ModelWrapperHunyuanVideoVAEDecoder):
    """ModelBuilder wrapper for one segmented HunyuanVideo VAE subgraph."""

    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag: str = "",
        compiler_args: str | None = None,
        priority_model_idx: int | None = None,
        model_init_kwargs=None,
        input_shape: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(
            config=config,
            model_cls=model_cls,
            tag=tag,
            compiler_args=compiler_args,
            priority_model_idx=priority_model_idx,
            model_init_kwargs=model_init_kwargs or {},
        )
        if input_shape is None:
            raise ValueError("HunyuanVideo VAE segment wrapper requires input_shape.")
        self.input_shape = tuple(int(dim) for dim in input_shape)

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        dtype = self.config.neuron_config.torch_dtype
        return [(torch.randn(self.input_shape, dtype=dtype),)]

    def get_model_instance(self):
        kwargs = dict(self.model_init_kwargs)

        def _create_model():
            model = self.model_cls(self.config, **kwargs)
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.model = self.model_cls(self.config, **self.model_init_kwargs)
        self.model = self.model.to(dtype=self.config.neuron_config.torch_dtype)
        self.model.load_state_dict(state_dict, strict=strict, assign=assign)


class _NeuronHunyuanVideoVAEDecoderSegmentApplication(NeuronApplicationBase):
    """Single-NEFF application used by the segmented VAE decoder."""

    _model_cls = nn.Module

    def __init__(self, *args, segment: _VAESegmentSpec, compiler_args: str, **kwargs):
        self.segment = segment
        self._compiler_args = compiler_args
        super().__init__(*args, **kwargs)
        self.model = ModelWrapperHunyuanVideoVAEDecoderSegment(
            config=self.config,
            model_cls=segment.model_cls,
            tag=segment.name,
            compiler_args=compiler_args,
            priority_model_idx=0,
            model_init_kwargs=segment.model_kwargs or {},
            input_shape=segment.input_shape,
        )
        self.models.append(self.model)

    @classmethod
    def get_config_cls(cls):
        return HunyuanVideoVAEDecoderInferenceConfig

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.models[0](hidden_states)

    def compile(
        self,
        compiled_model_path,
        debug=False,
        pre_shard_weights_hook=None,
        dry_run=False,
        disable_fail_fast=False,
    ):
        # These segments intentionally inline weights. The VAE workaround
        # relies on real NEFF boundaries, and direct torch_neuronx tracing is
        # the path verified by the parity probes.
        del debug, pre_shard_weights_hook, disable_fail_fast
        compiled_path = Path(compiled_model_path)
        compiled_path.mkdir(parents=True, exist_ok=True)
        self.config.save(compiled_path)
        if dry_run:
            return

        import torch_neuronx

        model = self.segment.model_cls(
            self.config,
            **(self.segment.model_kwargs or {}),
        )
        model = model.to(dtype=self.config.neuron_config.torch_dtype).eval()
        state_dict = self.checkpoint_loader_fn()
        model.load_state_dict(state_dict, strict=False)
        example = self.models[0].input_generator()[0][0]
        with torch.no_grad():
            traced = torch_neuronx.trace(
                model,
                example,
                compiler_args=self.get_compiler_args(),
            )
        torch.jit.save(traced, compiled_path / "model.pt")

    def load(
        self,
        compiled_model_path,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup=False,
    ):
        del start_rank_id, local_ranks_size
        compiled_path = Path(compiled_model_path)
        self.traced_model = torch.jit.load(compiled_path / "model.pt")
        self.models[0].model = self.traced_model
        self.is_loaded_to_neuron = True
        if not self.config.neuron_config.skip_warmup and not skip_warmup:
            example = self.models[0].input_generator()[0]
            self.models[0](*example)

    def get_compiler_args(self) -> str:
        return self._compiler_args

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        from nova.models.hunyuan_video.checkpoint import convert_vae_decoder_state_dict

        return convert_vae_decoder_state_dict(state_dict, config=config)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass


class NeuronHunyuanVideoVAEDecoderApplication(NeuronApplicationBase):
    """Compile/load wrapper for ``HunyuanVideoVAEDecoderModel``."""

    _model_cls = HunyuanVideoVAEDecoderModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.segmented = bool(getattr(self.config, "segment_causal_norm_conv", True))
        self.segment_apps: dict[str, _NeuronHunyuanVideoVAEDecoderSegmentApplication] = {}
        if self.segmented:
            compiler_args = self.get_compiler_args()
            for segment in self._segment_specs():
                self.segment_apps[segment.name] = _NeuronHunyuanVideoVAEDecoderSegmentApplication(
                    model_path=self.model_path,
                    config=self.config,
                    segment=segment,
                    compiler_args=compiler_args,
                )
        else:
            self.model_wrapper = self.get_model_wrapper_cls()
            self.model = self.model_wrapper(
                config=self.config,
                model_cls=self._model_cls,
                tag=self._model_cls.__name__,
                compiler_args=self.get_compiler_args(),
                priority_model_idx=0,
            )
            self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype
        self.config_obj = SimpleNamespace(scaling_factor=float(self.config.scaling_factor))

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    @classmethod
    def get_config_cls(cls):
        return HunyuanVideoVAEDecoderInferenceConfig

    def get_model_wrapper_cls(self):
        return ModelWrapperHunyuanVideoVAEDecoder

    def component_specs(self, prefix: str = "") -> list[ComponentSpec]:
        if not self.segmented:
            name = prefix.rstrip("/") or "vae_decoder"
            return [ComponentSpec(name, self)]
        base = prefix.rstrip("/")
        return [
            ComponentSpec(f"{base}/{name}" if base else name, app)
            for name, app in self.segment_apps.items()
        ]

    def compile(
        self,
        compiled_model_path,
        debug=False,
        pre_shard_weights_hook=None,
        dry_run=False,
        disable_fail_fast=False,
    ):
        if not self.segmented:
            return super().compile(
                compiled_model_path,
                debug=debug,
                pre_shard_weights_hook=pre_shard_weights_hook,
                dry_run=dry_run,
                disable_fail_fast=disable_fail_fast,
            )
        del pre_shard_weights_hook, disable_fail_fast
        root = Path(compiled_model_path)
        root.mkdir(parents=True, exist_ok=True)
        self.config.save(root)
        for name, app in self.segment_apps.items():
            app.compile(str(root / name), debug=debug, dry_run=dry_run)

    def load(
        self,
        compiled_model_path,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup=False,
    ):
        if not self.segmented:
            return super().load(
                compiled_model_path,
                start_rank_id=start_rank_id,
                local_ranks_size=local_ranks_size,
                skip_warmup=skip_warmup,
            )
        root = Path(compiled_model_path)
        for name, app in self.segment_apps.items():
            app.load(
                str(root / name),
                start_rank_id=start_rank_id,
                local_ranks_size=local_ranks_size,
                skip_warmup=skip_warmup,
            )

    def forward(self, *model_inputs, **kwargs):
        if not self.segmented:
            return self.models[0](*model_inputs, **kwargs)
        if kwargs:
            raise TypeError("Segmented HunyuanVideo VAE decoder accepts positional tensors only.")
        if len(model_inputs) != 1:
            raise TypeError("Segmented HunyuanVideo VAE decoder expects one latent tensor.")

        hidden_states = self._run_segment("body_up2", model_inputs[0])
        for resnet_index in range(3):
            residual = hidden_states
            hidden_states = self._run_segment(f"up3_r{resnet_index}_norm1_act", hidden_states)
            hidden_states = self._run_segment(f"up3_r{resnet_index}_conv1", hidden_states)
            hidden_states = self._run_segment(f"up3_r{resnet_index}_norm2_act", hidden_states)
            hidden_states = self._run_segment(f"up3_r{resnet_index}_conv2", hidden_states)
            if resnet_index == 0:
                residual = self._run_segment("up3_r0_shortcut", residual)
            hidden_states = hidden_states + residual

        hidden_states = self._run_segment("final_norm_act", hidden_states)
        return self._run_segment("final_conv_out", hidden_states)

    def _run_segment(self, name: str, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.segment_apps[name](hidden_states.to(dtype=self.dtype))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def decode(self, latents: torch.Tensor, return_dict: bool = True):
        decoded = self._temporal_tiled_decode(latents)
        if not return_dict:
            return (decoded,)
        return SimpleNamespace(sample=decoded)

    def _decode_tile(self, latents: torch.Tensor) -> torch.Tensor:
        orig_t, orig_h, orig_w = latents.shape[-3:]
        target = (
            int(self.config.tile_latent_frames),
            int(self.config.tile_latent_height),
            int(self.config.tile_latent_width),
        )
        pad_t = target[0] - orig_t
        pad_h = target[1] - orig_h
        pad_w = target[2] - orig_w
        if pad_t < 0 or pad_h < 0 or pad_w < 0:
            raise ValueError(
                "HunyuanVideo VAE tile exceeds compiled tile shape: "
                f"got {(orig_t, orig_h, orig_w)}, expected <= {target}."
            )
        if pad_t or pad_h or pad_w:
            latents = F.pad(latents, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")

        decoded = self.forward(latents.to(dtype=self.dtype))
        sample_frames = (orig_t - 1) * int(self.config.temporal_compression_ratio) + 1
        sample_height = orig_h * int(self.config.spatial_compression_ratio)
        sample_width = orig_w * int(self.config.spatial_compression_ratio)
        return decoded[:, :, :sample_frames, :sample_height, :sample_width]

    def _spatial_tiled_decode(self, latents: torch.Tensor) -> torch.Tensor:
        _, _, _, height, width = latents.shape
        sample_height = height * int(self.config.spatial_compression_ratio)
        sample_width = width * int(self.config.spatial_compression_ratio)
        tile_h = int(self.config.tile_latent_height)
        tile_w = int(self.config.tile_latent_width)
        stride_h = int(self.config.tile_latent_stride_height)
        stride_w = int(self.config.tile_latent_stride_width)
        blend_h = int(self.config.tile_sample_min_height) - int(self.config.tile_sample_stride_height)
        blend_w = int(self.config.tile_sample_min_width) - int(self.config.tile_sample_stride_width)

        rows = []
        for i in range(0, height, stride_h):
            row = []
            for j in range(0, width, stride_w):
                tile = latents[:, :, :, i : i + tile_h, j : j + tile_w]
                row.append(self._decode_tile(tile))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = _blend_v(rows[i - 1][j], tile, blend_h)
                if j > 0:
                    tile = _blend_h(row[j - 1], tile, blend_w)
                result_row.append(
                    tile[
                        :,
                        :,
                        :,
                        : int(self.config.tile_sample_stride_height),
                        : int(self.config.tile_sample_stride_width),
                    ]
                )
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

    def _temporal_tiled_decode(self, latents: torch.Tensor) -> torch.Tensor:
        _, _, num_frames, height, width = latents.shape
        num_sample_frames = (num_frames - 1) * int(self.config.temporal_compression_ratio) + 1
        tile_frames = int(self.config.tile_latent_frames)
        stride_frames = int(self.config.tile_latent_stride_num_frames)
        blend_frames = int(self.config.tile_sample_min_num_frames) - int(
            self.config.tile_sample_stride_num_frames
        )

        row = []
        for i in range(0, num_frames, stride_frames):
            tile = latents[:, :, i : i + tile_frames, :, :]
            if height > int(self.config.tile_latent_height) or width > int(self.config.tile_latent_width):
                decoded = self._spatial_tiled_decode(tile)
            else:
                decoded = self._decode_tile(tile)
            if i > 0:
                decoded = decoded[:, :, 1:, :, :]
            row.append(decoded)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = _blend_t(row[i - 1], tile, blend_frames)
                result_row.append(tile[:, :, : int(self.config.tile_sample_stride_num_frames), :, :])
            else:
                result_row.append(tile[:, :, : int(self.config.tile_sample_stride_num_frames) + 1, :, :])
        return torch.cat(result_row, dim=2)[:, :, :num_sample_frames]

    def get_compiler_args(self) -> str:
        compiler_args = (
            "--model-type=unet-inference -O1 "
            "--auto-cast=none "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return compiler_args

    def _segment_specs(self) -> list[_VAESegmentSpec]:
        batch = int(getattr(self.config.neuron_config, "batch_size", 1))
        latent_shape = (
            batch,
            int(self.config.latent_channels),
            int(self.config.tile_latent_frames),
            int(self.config.tile_latent_height),
            int(self.config.tile_latent_width),
        )
        sample_frames = (
            (int(self.config.tile_latent_frames) - 1)
            * int(self.config.temporal_compression_ratio)
            + 1
        )
        sample_height = int(self.config.tile_latent_height) * int(
            self.config.spatial_compression_ratio
        )
        sample_width = int(self.config.tile_latent_width) * int(
            self.config.spatial_compression_ratio
        )
        up2_channels = int(self.config.block_out_channels[1])
        up3_channels = int(self.config.block_out_channels[0])
        up2_shape = (batch, up2_channels, sample_frames, sample_height, sample_width)
        up3_shape = (batch, up3_channels, sample_frames, sample_height, sample_width)

        specs: list[_VAESegmentSpec] = [
            _VAESegmentSpec("body_up2", _HunyuanVAEBodyToUp2, latent_shape),
        ]
        for resnet_index in range(3):
            norm1_shape = up2_shape if resnet_index == 0 else up3_shape
            specs.extend(
                [
                    _VAESegmentSpec(
                        f"up3_r{resnet_index}_norm1_act",
                        _HunyuanVAEResnetNormAct,
                        norm1_shape,
                        {"resnet_index": resnet_index, "norm_index": 1},
                    ),
                    _VAESegmentSpec(
                        f"up3_r{resnet_index}_conv1",
                        _HunyuanVAEResnetConv,
                        norm1_shape,
                        {"resnet_index": resnet_index, "conv_name": "conv1"},
                    ),
                    _VAESegmentSpec(
                        f"up3_r{resnet_index}_norm2_act",
                        _HunyuanVAEResnetNormAct,
                        up3_shape,
                        {"resnet_index": resnet_index, "norm_index": 2},
                    ),
                    _VAESegmentSpec(
                        f"up3_r{resnet_index}_conv2",
                        _HunyuanVAEResnetConv,
                        up3_shape,
                        {"resnet_index": resnet_index, "conv_name": "conv2"},
                    ),
                ]
            )
            if resnet_index == 0:
                specs.append(
                    _VAESegmentSpec(
                        "up3_r0_shortcut",
                        _HunyuanVAEResnetConv,
                        up2_shape,
                        {"resnet_index": 0, "conv_name": "conv_shortcut"},
                    )
                )

        specs.extend(
            [
                _VAESegmentSpec("final_norm_act", _HunyuanVAEFinalNormAct, up3_shape),
                _VAESegmentSpec("final_conv_out", _HunyuanVAEFinalConvOut, up3_shape),
            ]
        )
        return specs

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        from nova.models.hunyuan_video.checkpoint import convert_vae_decoder_state_dict

        return convert_vae_decoder_state_dict(state_dict, config=config)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass


def _blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[-2], b.shape[-2], int(blend_extent))
    for y in range(blend_extent):
        b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[
            :, :, :, y, :
        ] * (y / blend_extent)
    return b


def _blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[-1], b.shape[-1], int(blend_extent))
    for x in range(blend_extent):
        b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[
            :, :, :, :, x
        ] * (x / blend_extent)
    return b


def _blend_t(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[-3], b.shape[-3], int(blend_extent))
    for x in range(blend_extent):
        b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[
            :, :, x, :, :
        ] * (x / blend_extent)
    return b
