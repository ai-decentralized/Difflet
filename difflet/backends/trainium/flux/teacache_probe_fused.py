"""Flux TeaCache fused-A probe (cclog 85).

Same recipe as the HunyuanVideo (cclog 80) and Qwen-Image (cclog 81) fused
probes: ``prev_mod`` is a persistent on-device ``nn.Parameter`` updated in place
via ``input_output_aliases``; the forward returns ``(rel_l1, mod_input)`` and
only the 4-byte scalar reaches host. The block-0 modulated input is computed by
``FluxTeacacheProbeFusedModel.teacache_mod_input``, which reuses the real
``NeuronFluxTransformer2DModel.forward`` prefix (x_embedder -> time_text_embed ->
transformer_blocks[0].norm1) and stops at the modulated image input, so it is
faithful to the real forward by construction (modeling_flux.py:401-411, 693).

Weight naming — "same weights => same names by construction"
------------------------------------------------------------
The probe model *is* a ``NeuronFluxTransformer2DModel`` (a subclass), not a
module that wraps one. Every transformer parameter therefore keeps the exact
attribute path it has in the backbone, and the names the probe NEFF looks up at
``nxd_model.initialize`` (``torch_neuronx`` names INPUT_WEIGHT tensors by module
path) are the backbone's shard keys. That is what lets the shared weight store
(``core/shared_weights.py``) hand the probe the backbone's pre-sharded
checkpoint: no layout tag in the store key, and no second ~22.7 GB copy of the
transformer (GitHub issue #39; the campaign fix ``3f04080`` is superseded).
The earlier wrapper design nested everything under
``trace_module.transformer.*`` and failed on device with
``Missing weight tensor with key trace_module.transformer.transformer_blocks.0.norm1.linear.bias``
(2026-08-30) once the store deduped it onto the backbone's shards.

The only tensor the probe adds is ``prev_mod``. It is aliased to output 1, so
NxD classifies it as INPUT_STATE: allocated zero-filled by ``StateInitializer``
at load and never looked up in the checkpoint. It is listed in
``NeuronFluxTeacacheProbeFusedApplication.state_tensor_names`` so the unit
tests can prove that every *other* tensor in the probe's state dict is served
by the backbone's converted checkpoint.

Flux is predicted HV-good for TeaCache because the block-0 AdaLN modulation
``temb`` incorporates the pooled CLIP text (time_text_embed(timestep[, guidance],
pooled_projections)) — unlike Qwen's text-independent block-0 (cclog 82/84). The
signal-correlation gate (cclog 84 lesson) must still confirm this on hardware
before trusting the adaptive controller over a fixed cadence.

Per-architecture differences from the Qwen/HV probes:
- Flux is a Difflet fork (NeuronFluxTransformer2DModel with NXD parallel layers),
  not a diffusers wrapper, so the probe subclasses the Difflet transformer.
- The block-0 IMAGE modulation does NOT depend on encoder_hidden_states (the T5
  sequence enters only via context_embedder + attention, after norm1), so the
  probe inputs are (hidden_states, timestep, pooled_projections, guidance) — one
  fewer tensor than Qwen.
- Timestep is fed at the diffusers [0,1] scale; the hook re-applies *1000
  internally exactly as the real forward does at modeling_flux.py:403 (cclog 83
  correctness gotcha — never double-scale).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from neuronx_distributed.trace.model_builder import BaseModelInstance
from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.model_wrapper import ModelWrapper
from difflet.models.flux.modeling_flux import (
    FluxBackboneInferenceConfig,
    NeuronFluxBackboneApplication,
    NeuronFluxTransformer2DModel,
)

#: Tensors the probe adds on top of the backbone. Aliased in place by the
#: ModelInstance below, so they are NEFF state (zero-initialised at load), not
#: checkpoint weights.
PROBE_STATE_TENSORS: frozenset[str] = frozenset({"prev_mod"})


class FluxTeacacheProbeFusedModel(NeuronFluxTransformer2DModel):
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

    def teacache_mod_input(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor,
    ) -> torch.Tensor:
        # Mirror NeuronFluxTransformer2DModel.forward (modeling_flux.py:401-411,
        # 693), stopping at the block-0 modulated image input. The forward
        # rescales timestep/guidance by 1000 internally, so this hook does too;
        # callers must pass the [0,1] diffusers-scale timestep (cclog 83).
        hs = self.x_embedder(hidden_states)
        timestep = timestep * 1000
        if self.config.guidance_embeds:
            guidance = guidance * 1000
            temb = self.time_text_embed(timestep, guidance, pooled_projections)
        else:
            temb = self.time_text_embed(timestep, pooled_projections)
        block0 = self.transformer_blocks[0]
        # NeuronAdaLayerNormZero.forward returns
        # (norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp); index 0
        # is the modulated image input. Stops before attention, so
        # image_rotary_emb is not needed.
        norm_hidden_states, *_ = block0.norm1(hs, emb=temb, hlomarker=True)
        return norm_hidden_states

    def forward(  # type: ignore[override] — the probe traces its own signature
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mod_input = self.teacache_mod_input(
            hidden_states, timestep, pooled_projections, guidance
        )
        # cclog 83 relative-L1 signal (mean|mod-prev|/mean|prev|), reduced in
        # float32; mod_input (out[1]) stays bf16 so it aliases back into the
        # bf16 prev_mod Parameter.
        m = mod_input.float()
        p = self.prev_mod.float()
        rel_l1 = (m - p).abs().mean() / (p.abs().mean() + 1e-8)
        return rel_l1, mod_input


class _FluxFusedProbeModelInstance(BaseModelInstance):
    def __init__(self, module_builder):
        super().__init__(module_builder, input_output_aliases={})

    def get(self, bucket_rank, **kwargs):
        # out 0 = rel_l1 (kept); out 1 = mod_input aliased to prev_mod (in-place).
        return self.module, {self.module.prev_mod: 1}


def _seq_len(config) -> int:
    return int(config.height) * int(config.width) // ((2 * int(config.vae_scale_factor)) ** 2)


class ModelWrapperFluxTeacacheProbeFused(ModelWrapper):
    def __init__(self, config, model_cls, tag="", compiler_args=None,
                 priority_model_idx=None, model_init_kwargs=None):
        super().__init__(config, model_cls, tag, compiler_args,
                         priority_model_idx, model_init_kwargs or {})
        self.bucket_config = None

    def input_generator(self):
        dtype = self.config.neuron_config.torch_dtype
        bs = 1  # serial CFG: one branch per probe call
        seq = _seq_len(self.config)
        return [
            (
                torch.randn([bs, seq, self.config.in_channels], dtype=dtype),
                torch.randn([bs], dtype=dtype),
                torch.randn([bs, self.config.pooled_projection_dim], dtype=dtype),
                torch.randn([bs], dtype=dtype)
                if self.config.guidance_embeds
                else torch.tensor([], dtype=dtype),
            )
        ]

    def get_model_instance(self):
        config = self.config
        dtype = self.config.neuron_config.torch_dtype
        seq = _seq_len(config)
        inner = int(config.num_attention_heads) * int(config.attention_head_dim)

        def _create():
            m = FluxTeacacheProbeFusedModel(config, seq_len=seq, inner_dim=inner, batch_size=1)
            return m.to(dtype=dtype).eval()

        return _FluxFusedProbeModelInstance(module_builder=_create)

    def forward(self, hidden_states, timestep, pooled_projections, guidance):
        if self.model is None:
            raise RuntimeError("Forward called before load.")
        # Cast floating inputs to the compiled dtype (the backbone wrapper does
        # the same at modeling_flux.py:1369-1380). The pipeline builds guidance
        # as float32; the NEFF expects bf16.
        dtype = self.config.neuron_config.torch_dtype
        hidden_states = hidden_states.to(dtype)
        timestep = timestep.to(dtype)
        pooled_projections = pooled_projections.to(dtype)
        if guidance is not None and guidance.numel() > 0:
            guidance = guidance.to(dtype)
        return self._forward(hidden_states, timestep, pooled_projections, guidance)


class NeuronFluxTeacacheProbeFusedApplication(NeuronApplicationBase):
    """Flux fused-A probe app."""

    _model_cls = FluxTeacacheProbeFusedModel
    state_tensor_names = PROBE_STATE_TENSORS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = ModelWrapperFluxTeacacheProbeFused
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
        return FluxBackboneInferenceConfig

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self) -> str:
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return "--model-type=transformer -O1 --auto-cast=none"

    # The probe's weights ARE the backbone's weights under the backbone's names,
    # so its checkpoint converter is the backbone's converter object — the two
    # cannot drift apart. (global_rank.rank + single-block proj_out splits.)
    convert_hf_to_neuron_state_dict = staticmethod(
        NeuronFluxBackboneApplication.convert_hf_to_neuron_state_dict
    )

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    def teacache_delta(self, hidden_states, timestep, pooled_projections, guidance):
        out = self.models[0](hidden_states, timestep, pooled_projections, guidance)
        return out[0] if isinstance(out, (tuple, list)) else out
