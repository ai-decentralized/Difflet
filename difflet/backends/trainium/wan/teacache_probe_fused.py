"""Wan TeaCache fused-A probe.

Same recipe as the HunyuanVideo (cclog 80), Qwen-Image (cclog 81) and Flux
(cclog 85) fused probes: ``prev_mod`` is a persistent on-device ``nn.Parameter``
updated in place via ``input_output_aliases``; the forward returns
``(rel_l1, mod_input)`` and only the 4-byte scalar reaches host.

The block-0 modulated input itself is *not* reimplemented here. It is
``WanTransformer3DModel.teacache_mod_input`` (modeling_wan.py:927), the same
method the host CPU shadow calls, so the device probe and the shadow compute
the identical signal by construction and cannot drift apart.

Weight naming — "same weights => same names by construction"
------------------------------------------------------------
The probe model *is* a ``WanTransformer3DModel`` (a subclass), not a module that
wraps one, so every transformer parameter keeps the exact attribute path it has
in the backbone. Those are the names the probe NEFF looks up at
``nxd_model.initialize`` (``torch_neuronx`` names INPUT_WEIGHT tensors by module
path), which is what lets the shared weight store hand the probe the backbone's
pre-sharded checkpoint instead of making a second ~28 GB copy per expert. A
wrapper design nests everything under ``trace_module.transformer.*`` and fails on
device with ``Missing weight tensor with key ...`` (Flux hit exactly this on
2026-08-30; GitHub issue #39).

The only tensor the probe adds is ``prev_mod``. It is aliased to output 1, so NxD
classifies it as INPUT_STATE: allocated zero-filled by ``StateInitializer`` at
load and never looked up in the checkpoint. It is declared in
``state_tensor_names`` so the unit tests can prove every *other* tensor in the
probe's state dict is served by the backbone's converted checkpoint.

Wan specifics
-------------
- **Two experts.** Wan 2.2 runs a high-noise and a low-noise transformer, so a
  probe is mounted per stage; each probe shares its own expert's shards. The
  controller already resets its residual on the stage switch
  (pipeline.py:388-390).
- **One bucket.** The backbone is shape-bucketed, but ``prev_mod`` is a
  fixed-shape parameter, so the probe compiles at the primary compile shape
  only (``config.height``/``width``/``num_frames``, which
  ``add_derived_config`` pins to ``compile_shapes[0]``). The pipeline falls back
  to the host CPU shadow for any other request shape.
- **Parallelism.** ``teacache_mod_input`` stops before attention, so it performs
  no CP/SP scatter and no CFG-parallel batch scatter; it returns the full
  unsharded sequence on every rank. TeaCache already refuses CFG-parallel in the
  Wan pipeline (pipeline.py:341-346), which is where the single-skip-decision
  soundness argument lives.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.backends.trainium.wan.backbone import (
    NeuronWanBackboneApplication,
    WanBackboneInferenceConfig,
)
from difflet.models.wan.modeling_wan import WanTransformer3DModel

#: Tensors the probe adds on top of the backbone. Aliased in place by the
#: ModelInstance below, so they are NEFF state (zero-initialised at load), not
#: checkpoint weights.
PROBE_STATE_TENSORS: frozenset[str] = frozenset({"prev_mod"})


def probe_seq_len(config) -> int:
    """Patchified video sequence length at the config's primary compile shape.

    Mirrors ``WanTransformer3DModel.forward`` (modeling_wan.py:827-836):
    ``ppf * pph * ppw`` over the LATENT grid. ``config.num_frames`` is already
    latent frames for the backbone config; height/width are pixel dims the
    backbone divides by 8 (backbone.py:136-140).
    """
    p_t, p_h, p_w = tuple(config.patch_size)
    return (
        (int(config.num_frames) // int(p_t))
        * ((int(config.height) // 8) // int(p_h))
        * ((int(config.width) // 8) // int(p_w))
    )


class WanTeacacheProbeFusedModel(WanTransformer3DModel):
    """The backbone transformer plus a persistent ``prev_mod`` state.

    Forward returns ``(rel_l1, mod_input)``; ``mod_input`` is aliased back into
    ``prev_mod`` in place. Being a subclass (not a wrapper) keeps every
    transformer parameter at its backbone attribute path — see the module
    docstring.
    """

    def __init__(self, config, *, seq_len: int, inner_dim: int, batch_size: int = 1) -> None:
        super().__init__(config)
        self.prev_mod = nn.Parameter(
            torch.zeros(batch_size, seq_len, inner_dim), requires_grad=False
        )

    def forward(  # type: ignore[override] — the probe traces its own signature
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mod_input = self.teacache_mod_input(hidden_states, timestep, encoder_hidden_states)
        # Relative-L1 signal (mean|mod-prev| / mean|prev|), reduced in float32;
        # mod_input (out[1]) keeps the model dtype so it aliases back into the
        # same-dtype prev_mod Parameter.
        m = mod_input.float()
        p = self.prev_mod.float()
        rel_l1 = (m - p).abs().mean() / (p.abs().mean() + 1e-8)
        return rel_l1, mod_input


class _WanFusedProbeModelInstance(BaseModelInstance):
    def __init__(self, module_builder):
        super().__init__(module_builder, input_output_aliases={})

    def get(self, bucket_rank, **kwargs):
        # out 0 = rel_l1 (kept); out 1 = mod_input aliased to prev_mod (in-place).
        return self.module, {self.module.prev_mod: 1}


class ModelWrapperWanTeacacheProbeFused(ModelWrapper):
    """Single-bucket compile wrapper for the Wan probe.

    Deliberately not ``ShapeBucketedInputGenerator``: ``prev_mod`` is one fixed
    shape, so the probe serves the primary compile shape only.
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

    def input_generator(self):
        config = self.config
        dtype = config.neuron_config.torch_dtype
        batch_size = 1  # serial CFG: one branch per probe call
        text_seq_len = int(getattr(config, "text_seq_len", 512))
        return [
            (
                torch.randn(
                    [
                        batch_size,
                        int(config.in_channels),
                        int(config.num_frames),
                        int(config.height) // 8,
                        int(config.width) // 8,
                    ],
                    dtype=dtype,
                ),
                torch.randn([batch_size], dtype=dtype),
                torch.randn([batch_size, text_seq_len, int(config.text_dim)], dtype=dtype),
            )
        ]

    def get_model_instance(self):
        config = self.config
        dtype = config.neuron_config.torch_dtype
        seq = probe_seq_len(config)
        inner = int(config.num_attention_heads) * int(config.attention_head_dim)

        def _create():
            model = WanTeacacheProbeFusedModel(
                config, seq_len=seq, inner_dim=inner, batch_size=1
            )
            return model.to(dtype=dtype).eval()

        return _WanFusedProbeModelInstance(module_builder=_create)

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        dtype = self.config.neuron_config.torch_dtype
        return self._forward(
            hidden_states.to(dtype),
            timestep.to(dtype),
            encoder_hidden_states.to(dtype),
        )


class NeuronWanTeacacheProbeFusedApplication(NeuronApplicationBase):
    """Wan fused-A probe app (one per expert stage)."""

    _model_cls = WanTeacacheProbeFusedModel
    state_tensor_names = PROBE_STATE_TENSORS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = ModelWrapperWanTeacacheProbeFused
        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype
        self.teacache_probe_fused = True

    @classmethod
    def get_config_cls(cls):
        return WanBackboneInferenceConfig

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self) -> str:
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return "--model-type=transformer -O1 --auto-cast=none"

    # The probe's weights ARE the backbone's weights under the backbone's names,
    # so its checkpoint converter is the backbone's converter object — the two
    # cannot drift apart.
    convert_hf_to_neuron_state_dict = staticmethod(
        NeuronWanBackboneApplication.convert_hf_to_neuron_state_dict
    )

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    def teacache_delta(self, hidden_states, timestep, encoder_hidden_states) -> torch.Tensor:
        """Relative-L1 of the block-0 modulated input vs the previous probed step."""
        out = self.models[0](hidden_states, timestep, encoder_hidden_states)
        return out[0] if isinstance(out, (tuple, list)) else out


__all__ = [
    "PROBE_STATE_TENSORS",
    "ModelWrapperWanTeacacheProbeFused",
    "NeuronWanTeacacheProbeFusedApplication",
    "WanTeacacheProbeFusedModel",
    "probe_seq_len",
]
