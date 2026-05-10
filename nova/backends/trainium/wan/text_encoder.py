"""Trainium application wrapper for the Wan UMT5 text encoder."""

from __future__ import annotations

import os
from typing import List, Tuple

import torch

from nova.backends.trainium.core.application_base import NeuronApplicationBase
from nova.backends.trainium.core.config import InferenceConfig
from nova.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from nova.models.wan.umt5.modeling_umt5 import WanUmT5EncoderModel


class WanTextEncoderInferenceConfig(InferenceConfig):
    """Inference config for the Wan UMT5 text encoder component."""

    def add_derived_config(self):
        super().add_derived_config()
        # diffusers Wan2.2 pipeline's default is 512 tokens; matches the
        # text_seq_len that backbone (cclogs/13 §2) expects on its
        # encoder_hidden_states input.
        if not hasattr(self, "text_seq_len"):
            self.text_seq_len = 512
        if not hasattr(self, "batch_size"):
            self.batch_size = self.neuron_config.batch_size

    def get_required_attributes(self) -> List[str]:
        return [
            "vocab_size",
            "d_model",
            "d_kv",
            "d_ff",
            "num_heads",
            "num_layers",
            "relative_attention_num_buckets",
            "relative_attention_max_distance",
            "is_gated_act",
            "dense_act_fn",
            "layer_norm_epsilon",
        ]

    @property
    def inner_dim(self) -> int:
        return self.num_heads * self.d_kv

    def validate_config(self):
        super().validate_config()
        if not getattr(self, "is_gated_act", True):
            raise NotImplementedError(
                "Wan UMT5 spike currently supports only gated FFN "
                "(is_gated_act=True)."
            )


class ModelWrapperWanTextEncoder(ModelWrapper):
    """ModelBuilder wrapper for Wan UMT5 compile inputs.

    Compile contract:
      input_ids:       (batch_size, text_seq_len) int64
      attention_mask:  (batch_size, text_seq_len) int32
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

    def input_generator(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = int(getattr(self.config.neuron_config, "batch_size", 1))
        text_seq_len = int(getattr(self.config, "text_seq_len", 512))
        return [
            (
                torch.zeros((batch_size, text_seq_len), dtype=torch.int64),
                torch.ones((batch_size, text_seq_len), dtype=torch.int32),
            )
        ]

    def get_model_instance(self):
        def _create_model():
            from nova.models.wan.umt5.modeling_umt5 import WanUmT5Config

            cfg = WanUmT5Config(
                vocab_size=self.config.vocab_size,
                d_model=self.config.d_model,
                d_kv=self.config.d_kv,
                d_ff=self.config.d_ff,
                num_heads=self.config.num_heads,
                num_layers=self.config.num_layers,
                relative_attention_num_buckets=self.config.relative_attention_num_buckets,
                relative_attention_max_distance=self.config.relative_attention_max_distance,
                is_gated_act=getattr(self.config, "is_gated_act", True),
                dense_act_fn=getattr(self.config, "dense_act_fn", "gelu_new"),
                layer_norm_epsilon=self.config.layer_norm_epsilon,
            )
            model = self.model_cls(cfg)
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(self, input_ids, attention_mask):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(input_ids, attention_mask)


class NeuronWanTextEncoderApplication(NeuronApplicationBase):
    """Compile/load wrapper for ``WanUmT5EncoderModel``."""

    _model_cls = WanUmT5EncoderModel

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
        return WanTextEncoderInferenceConfig

    def get_model_wrapper_cls(self):
        return ModelWrapperWanTextEncoder

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self) -> str:
        compiler_args = (
            "--model-type=transformer -O1 "
            "--tensorizer-options='--enable-ccop-compute-overlap' "
            "--auto-cast=none "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return compiler_args

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        from nova.models.wan.checkpoint import convert_text_encoder_state_dict

        return convert_text_encoder_state_dict(state_dict, config=config)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass
