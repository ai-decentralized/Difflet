"""HunyuanVideo TeaCache probe Neuron application.

A sibling application to ``NeuronHunyuanVideoBackboneApplication``. Compiles
and loads the small probe NEFF (block-0 modulated input + device-side L2 diff)
as an INDEPENDENT artifact, leaving the DiT compile cache untouched.

Per cclog 72: decoupled architecture, additive design. The diagnostic in
``scripts/diag_hv_load_only.py`` confirmed that piling a second ModelWrapper
into ``NeuronHunyuanVideoBackboneApplication`` breaks the single-traced-model
load semantics — both wrappers end up pointing at the same DiT-only
``traced_model`` and the probe forward signature mismatches at runtime. The
clean fix is to give the probe its OWN ``NeuronApplicationBase`` instance with
its OWN compiled directory and its OWN ``traced_model``.
"""

from __future__ import annotations

import os

import torch

from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.backends.trainium.core.teacache_probe import PrefixProbeApplication
from difflet.backends.trainium.hunyuan_video.backbone import (
    HunyuanVideoBackboneInferenceConfig,
)
from difflet.backends.trainium.hunyuan_video.teacache_probe_model import (
    PROBE_STATE_TENSORS,
    HunyuanVideoTeacacheProbeFusedModel,
    HunyuanVideoTeacacheProbeModel,
)


class _HunyuanVideoPrefixApplication(PrefixProbeApplication):
    weight_prefixes = ("x_embedder.", "time_text_embed.", "transformer_blocks.0.norm1.")

    @classmethod
    def convert_hf_to_neuron_state_dict(cls, state_dict, config):
        return {k: v for k, v in state_dict.items() if k.startswith(cls.weight_prefixes)}


class ModelWrapperHunyuanVideoTeacacheProbe(ModelWrapper):
    """ModelBuilder wrapper for the standalone TeaCache probe NEFF.

    The model and checkpoint contain only the modulation prefix.
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

    @property
    def _mod_input_seq_len(self) -> int:
        c = self.config
        return (
            (int(c.latent_frames) // int(c.patch_size_t))
            * (int(c.latent_height) // int(c.patch_size))
            * (int(c.latent_width) // int(c.patch_size))
        )

    def input_generator(self) -> list[tuple[torch.Tensor, ...]]:
        batch_size = int(getattr(self.config.neuron_config, "batch_size", 1))
        dtype = self.config.neuron_config.torch_dtype
        text_seq_len = int(getattr(self.config, "text_seq_len", 256))
        seq_len = self._mod_input_seq_len
        inner_dim = int(self.config.inner_dim)

        return [
            (
                torch.randn(
                    [
                        batch_size,
                        self.config.in_channels,
                        self.config.latent_frames,
                        self.config.latent_height,
                        self.config.latent_width,
                    ],
                    dtype=dtype,
                ),
                torch.ones([batch_size], dtype=dtype),
                torch.randn([batch_size, text_seq_len, self.config.text_embed_dim], dtype=dtype),
                torch.ones([batch_size, text_seq_len], dtype=torch.int64),
                torch.randn([batch_size, self.config.pooled_projection_dim], dtype=dtype),
                torch.ones([batch_size], dtype=dtype),
                torch.zeros([batch_size, seq_len, inner_dim], dtype=dtype),
            )
        ]

    def get_model_instance(self):
        def _create_model():
            model = self.model_cls(self.config)
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        guidance,
        prev_mod_input,
    ):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
            prev_mod_input,
        )


class NeuronHunyuanVideoTeacacheProbeApplication(_HunyuanVideoPrefixApplication):
    """Standalone compile/load wrapper for the HunyuanVideo TeaCache probe.

    Mirrors ``NeuronHunyuanVideoBackboneApplication`` but with a single
    ModelWrapper (the probe). Compiled artifact lives in its own subdirectory
    (e.g., ``compiled_path/teacache_probe/``) — does NOT share with the DiT
    artifact.
    """

    _model_cls = HunyuanVideoTeacacheProbeModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = ModelWrapperHunyuanVideoTeacacheProbe
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
        return HunyuanVideoBackboneInferenceConfig

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
    def update_state_dict_for_tied_weights(state_dict):
        pass

    def teacache_mod_input(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        guidance,
    ):
        """Calibration entry: matches the CPU model's teacache_mod_input signature.

        Internally invokes the probe with a zero ``prev_mod_input`` and discards
        the resulting delta, returning only ``mod_input``.
        """
        seq_len = (
            (int(self.config.latent_frames) // int(self.config.patch_size_t))
            * (int(self.config.latent_height) // int(self.config.patch_size))
            * (int(self.config.latent_width) // int(self.config.patch_size))
        )
        inner_dim = int(self.config.inner_dim)
        zero_prev = torch.zeros(
            (int(hidden_states.shape[0]), seq_len, inner_dim),
            dtype=self.dtype,
            device=hidden_states.device,
        )
        _, mod_input = self.models[0](
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
            zero_prev,
        )
        return mod_input

    def teacache_mod_input_with_delta(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        guidance,
        prev_mod_input,
    ):
        """T1 production entry: returns (delta_scalar, mod_input_handle)."""
        return self.models[0](
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
            prev_mod_input,
        )


# ---------------------------------------------------------------------------
# fused-A (cclog 80): prev_mod as a persistent on-device nn.Parameter, updated
# in place via input_output_aliases. Returns ONLY the scalar delta to host,
# removing the ~25.8 ms/step mod_input output marshaling (cclog 79 Test 1).
# ---------------------------------------------------------------------------


def _probe_seq_len(config) -> int:
    return (
        (int(config.latent_frames) // int(config.patch_size_t))
        * (int(config.latent_height) // int(config.patch_size))
        * (int(config.latent_width) // int(config.patch_size))
    )


class _FusedProbeModelInstance(BaseModelInstance):
    """ModelInstance whose get() aliases the prev_mod Parameter (output index 1
    = mod_input) back to itself — the tkg-nki3 cross-step-state pattern. Built
    post-load so it references the loaded module's Parameter object."""

    def __init__(self, module_builder):
        super().__init__(module_builder, input_output_aliases={})

    def get(self, bucket_rank, **kwargs):
        # forward returns (delta [out 0], mod_input [out 1]); alias out 1 →
        # prev_mod Parameter (in-place HBM update, stripped from host outputs).
        return self.module, {self.module.prev_mod: 1}


class ModelWrapperHunyuanVideoTeacacheProbeFused(ModelWrapper):
    """ModelBuilder wrapper for the fused-A probe NEFF."""

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

    def input_generator(self) -> list[tuple[torch.Tensor, ...]]:
        batch_size = int(getattr(self.config.neuron_config, "batch_size", 1))
        dtype = self.config.neuron_config.torch_dtype
        text_seq_len = int(getattr(self.config, "text_seq_len", 256))
        # prev_mod is NOT an input (it is an internal Parameter), so the input
        # set is the 6 bundle tensors only.
        return [
            (
                torch.randn(
                    [
                        batch_size,
                        self.config.in_channels,
                        self.config.latent_frames,
                        self.config.latent_height,
                        self.config.latent_width,
                    ],
                    dtype=dtype,
                ),
                torch.ones([batch_size], dtype=dtype),
                torch.randn([batch_size, text_seq_len, self.config.text_embed_dim], dtype=dtype),
                torch.ones([batch_size, text_seq_len], dtype=torch.int64),
                torch.randn([batch_size, self.config.pooled_projection_dim], dtype=dtype),
                torch.ones([batch_size], dtype=dtype),
            )
        ]

    def get_model_instance(self):
        config = self.config
        dtype = self.config.neuron_config.torch_dtype
        seq_len = _probe_seq_len(config)
        inner_dim = int(config.inner_dim)
        batch_size = int(getattr(config.neuron_config, "batch_size", 1))

        def _create_model():
            model = HunyuanVideoTeacacheProbeFusedModel(
                config, seq_len=seq_len, inner_dim=inner_dim, batch_size=batch_size
            )
            return model.to(dtype=dtype).eval()

        return _FusedProbeModelInstance(module_builder=_create_model)

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        guidance,
    ):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
        )


class NeuronHunyuanVideoTeacacheProbeFusedApplication(_HunyuanVideoPrefixApplication):
    """fused-A standalone probe app: prev_mod persistent on device, returns only delta."""

    _model_cls = HunyuanVideoTeacacheProbeFusedModel
    state_tensor_names = PROBE_STATE_TENSORS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = ModelWrapperHunyuanVideoTeacacheProbeFused
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
        return HunyuanVideoBackboneInferenceConfig

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
    def update_state_dict_for_tied_weights(state_dict):
        pass

    def teacache_delta(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        guidance,
    ):
        """T1 fused entry: returns the scalar delta only. prev_mod is updated
        in place on device (alias) — no host handle juggling."""
        out = self.models[0](
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
        )
        return out[0] if isinstance(out, (tuple, list)) else out
