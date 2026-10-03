"""Trainium application wrapper for the Wan VAE decoder."""

from __future__ import annotations

import os
from typing import List, Tuple

import torch

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.bucketing import (
    CompileShape,
    ShapeBucketedInputGenerator,
    canonicalize_shapes,
    dedupe_example_inputs,
    resolve_compile_shapes,
)
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig, WanVAEDecoderModel


class WanVAEDecoderInferenceConfig(InferenceConfig):
    """Inference config for decoder-only Wan VAE compile."""

    def add_derived_config(self):
        super().add_derived_config()
        if not hasattr(self, "decoder_base_dim"):
            self.decoder_base_dim = getattr(self, "base_dim", 96)
        if not hasattr(self, "out_channels"):
            self.out_channels = 3
        if not hasattr(self, "scale_factor_temporal"):
            self.scale_factor_temporal = 4
        if not hasattr(self, "scale_factor_spatial"):
            self.scale_factor_spatial = 8
        # Bucket shape set (frames component = PIXEL frames, matching
        # config.num_frames semantics for the Wan VAE decoder).
        shapes = getattr(self, "compile_shapes", None)
        if shapes:
            self.compile_shapes = canonicalize_shapes(shapes)
            self.height, self.width, self.num_frames = self.compile_shapes[0]

    def get_required_attributes(self) -> List[str]:
        return [
            "base_dim",
            "z_dim",
            "dim_mult",
            "num_res_blocks",
            "attn_scales",
            "temperal_downsample",
            "dropout",
            "height",
            "width",
            "num_frames",
        ]

    @property
    def latent_height(self) -> int:
        return int(self.height) // int(self.scale_factor_spatial)

    @property
    def latent_width(self) -> int:
        return int(self.width) // int(self.scale_factor_spatial)

    @property
    def latent_frames(self) -> int:
        return (int(self.num_frames) - 1) // int(self.scale_factor_temporal) + 1

    def validate_config(self):
        super().validate_config()
        for height, width, _num_frames in resolve_compile_shapes(self):
            if int(height) % int(self.scale_factor_spatial) != 0:
                raise ValueError(
                    f"Wan VAE height must be divisible by spatial scale factor; got {height}."
                )
            if int(width) % int(self.scale_factor_spatial) != 0:
                raise ValueError(
                    f"Wan VAE width must be divisible by spatial scale factor; got {width}."
                )
        if int(self.scale_factor_temporal) != 4:
            raise NotImplementedError("Wan VAE spike expects temporal scale factor 4.")
        if getattr(self, "is_residual", False):
            raise NotImplementedError("Wan VAE spike supports only is_residual=False.")
        if getattr(self, "patch_size", None) is not None:
            raise NotImplementedError("Wan VAE spike does not support patchified VAE.")


class ModelWrapperWanVAEDecoder(ShapeBucketedInputGenerator, ModelWrapper):
    """ModelBuilder wrapper for Wan VAE decoder compile inputs.

    Decodes the full latent in one shot, so each compile shape genuinely
    becomes its own bucket NEFF (unlike the tiled HunyuanVideo decoder).
    """

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
            config,
            model_cls,
            tag,
            compiler_args,
            priority_model_idx,
            model_init_kwargs or {},
        )
        self.bucket_config = None

    def example_inputs_for_shape(self, shape: CompileShape) -> Tuple[torch.Tensor, ...]:
        height, width, num_frames = shape
        batch_size = int(getattr(self.config.neuron_config, "batch_size", 1))
        dtype = self.config.neuron_config.torch_dtype
        return (
            torch.randn(
                [
                    batch_size,
                    int(self.config.z_dim),
                    (int(num_frames) - 1) // int(self.config.scale_factor_temporal) + 1,
                    int(height) // int(self.config.scale_factor_spatial),
                    int(width) // int(self.config.scale_factor_spatial),
                ],
                dtype=dtype,
            ),
        )

    def get_model_instance(self):
        def _create_model():
            cfg = WanVAEDecoderConfig(
                base_dim=int(self.config.base_dim),
                decoder_base_dim=int(self.config.decoder_base_dim),
                z_dim=int(self.config.z_dim),
                dim_mult=list(self.config.dim_mult),
                num_res_blocks=int(self.config.num_res_blocks),
                attn_scales=list(self.config.attn_scales),
                temperal_downsample=list(self.config.temperal_downsample),
                dropout=float(self.config.dropout),
                latents_mean=list(getattr(self.config, "latents_mean", [])),
                latents_std=list(getattr(self.config, "latents_std", [])),
                is_residual=bool(getattr(self.config, "is_residual", False)),
                out_channels=int(getattr(self.config, "out_channels", 3)),
                patch_size=getattr(self.config, "patch_size", None),
                scale_factor_temporal=int(getattr(self.config, "scale_factor_temporal", 4)),
                scale_factor_spatial=int(getattr(self.config, "scale_factor_spatial", 8)),
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


class NeuronWanVAEDecoderApplication(NeuronApplicationBase):
    """Compile/load wrapper for ``WanVAEDecoderModel``."""

    _model_cls = WanVAEDecoderModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    @classmethod
    def get_config_cls(cls):
        return WanVAEDecoderInferenceConfig

    def get_model_wrapper_cls(self):
        return ModelWrapperWanVAEDecoder

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self) -> str:
        compiler_args = (
            "--model-type=unet-inference -O1 "
            "--auto-cast=none "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return compiler_args

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        from difflet.models.wan.checkpoint import convert_vae_decoder_state_dict

        return convert_vae_decoder_state_dict(state_dict, config=config)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass


class ModelWrapperWanVAEChunk(ModelWrapperWanVAEDecoder):
    """Signature-routed partition/phase buckets, independent of video length."""

    def input_generator(self):
        from difflet.models.wan.vae.chunked import split_input_shapes

        config = WanVAEDecoderConfig.from_diffusers_dict(vars(self.config))
        dtype = self.config.neuron_config.torch_dtype
        batch = int(self.config.neuron_config.batch_size)
        examples = []
        for height, width, _frames in resolve_compile_shapes(self.config):
            shape = (batch, config.z_dim, 1, height // 8, width // 8)
            for signature in split_input_shapes(config, shape):
                self.state_count = len(signature) - 1
                examples.append(tuple(torch.zeros(s, dtype=dtype) for s in signature))
        return dedupe_example_inputs(examples)

    def forward(self, latents, *state):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        if not state:
            if not hasattr(self, "state_count"):
                from difflet.models.wan.vae.chunked import WanVAESplitChunkModel
                config = WanVAEDecoderConfig.from_diffusers_dict(vars(self.config))
                with torch.device("meta"):
                    self.state_count = WanVAESplitChunkModel(config).state_count
            state = tuple(latents.new_zeros((1,)) for _ in range(self.state_count))
        if hasattr(self, "bucket_loader"):
            return self.bucket_loader.run((latents, *state))
        return self._forward(latents, *state)


class _WanBucketLoader:
    """Keep one prefix and one tail NEFF resident; weights stay shared."""

    def __init__(self, nxd_model, weights, start_rank):
        import ast
        from pathlib import Path
        import tempfile

        self.weights = weights
        self.start_rank = start_rank
        self.routes = {
            tuple(tuple(shape) for shape in ast.literal_eval(signature)): tuple(route)
            for signature, route in nxd_model.input_shape_map.items()
        }
        self.artifacts = {}
        # The SDK's Python __getstate__ decodes binary NEFF strings as UTF-8;
        # use its supported file exporters instead. Original scripted models
        # remain uninitialized and therefore own no device I/O buffers.
        with tempfile.TemporaryDirectory(prefix="wan-vae-neff-") as directory:
            neff = Path(directory) / "model.neff"
            metadata = Path(directory) / "model.metaneff"
            for name, bucket in nxd_model.models.named_children():
                for index, model in enumerate(bucket.models):
                    model.save_neff(str(neff))
                    model.save_metaneff(str(metadata))
                    self.artifacts[(name, index)] = (neff.read_bytes(), metadata.read_bytes())
        self.flatteners = dict(nxd_model.flattener_map.named_children())
        self.packer = nxd_model.packer
        self.resident = {}

    def run(self, inputs):
        signature = tuple(tuple(t.shape) for t in inputs)
        if signature not in self.routes:
            raise ValueError("Wan VAE input signature is absent from the compiled artifact")
        route = self.routes[signature]
        partition = inputs[0].ndim
        previous = self.resident.get(partition)
        if previous is None or previous[0] != route:
            if previous is not None:
                del self.resident[partition]
                # SPMDModel.unload() alone retains tensor pools. Destroy the
                # object before allocating the replacement's buffers.
                del previous
            model = torch.classes.neuron.SPMDModel(*self.artifacts[route], 1, 1)
            model.initialize([], self.weights, self.start_rank)
            self.resident[partition] = (route, model)
        model = self.resident[partition][1]
        flat_inputs = self.flatteners[f"{route[0]}_{route[1]}"](list(inputs))
        return self.packer(model.forward(flat_inputs))


class NeuronWanVAEChunkedApplication(NeuronWanVAEDecoderApplication):
    """FP32 streaming decoder; NxD shares weights across partition/phase buckets.

    Causal state is explicit input/output and currently crosses the host on
    each call. This bounds compilation size; it does not claim zero-copy state.
    """

    def __init__(self, *args, **kwargs):
        from difflet.models.wan.vae.chunked import WanVAESplitChunkModel

        self._model_cls = WanVAESplitChunkModel
        super().__init__(*args, **kwargs)
        if self.neuron_config.tp_degree != 1 or self.neuron_config.world_size != 1:
            raise ValueError("Chunked Wan VAE currently requires standalone TP1/W1")
        if self.dtype != torch.float32:
            raise ValueError("Chunked Wan VAE requires float32 to match the official VAE")
        self.config.wan_vae_chunked_version = 3

    def get_model_wrapper_cls(self):
        return ModelWrapperWanVAEChunk

    def compile(self, *args, **kwargs):
        from difflet.backends.trainium.utils.compile_serial import serial_bucket_compilation

        # Each 480P FP32 bucket can use tens of GiB of host compiler memory.
        with serial_bucket_compilation():
            return super().compile(*args, **kwargs)

    def load_weights(self, compiled_model_path, start_rank_id=None, local_ranks_size=None):
        from safetensors.torch import load_file
        from difflet.backends.trainium.core.application_base import _runtime_start_rank_id

        start_rank_id = self.neuron_config.start_rank_id if start_rank_id is None else start_rank_id
        local_ranks_size = self.neuron_config.local_ranks_size if local_ranks_size is None else local_ranks_size
        if local_ranks_size != 1:
            raise ValueError("Chunked Wan VAE loading requires one local rank")
        path = os.path.join(compiled_model_path, "weights", "tp0_sharded_checkpoint.safetensors")
        checkpoint = [load_file(path)] if os.path.exists(path) else self.get_builder().shard_checkpoint()
        # NxD.initialize eagerly allocates I/O buffers for EVERY bucket. With
        # large causal histories that exceeds a core's HBM. Load shared weights
        # once and initialize only the active graph for each partition.
        weights = torch.ops.neuron._parallel_load(checkpoint)
        self.model.bucket_loader = _WanBucketLoader(
            self.traced_model.nxd_model, weights,
            _runtime_start_rank_id(start_rank_id, local_ranks_size),
        )

    def warmup(self):
        for inputs in self.model.input_generator():
            self.model(*inputs)

    def forward(self, latents):
        from difflet.models.wan.vae.chunked import decode_split_chunks

        expected = {
            (int(self.neuron_config.batch_size), int(self.config.z_dim), h // 8, w // 8)
            for h, w, _frames in resolve_compile_shapes(self.config)
        }
        if latents.ndim != 5 or (latents.shape[0], latents.shape[1], latents.shape[3], latents.shape[4]) not in expected:
            raise ValueError("Wan VAE latent batch/channels/spatial shape does not match the compiled profile")
        return decode_split_chunks(self.model, latents.to(dtype=self.dtype))
