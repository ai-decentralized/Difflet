"""HunyuanVideo-1.5 TeaCache fused-A probe (cclog 86).

Mirrors the HV-1.0 (cclog 80) / Qwen (cclog 81) / Flux (cclog 85) fused probes:
``prev_mod`` is a persistent on-device ``nn.Parameter`` updated in place via
``input_output_aliases``; the forward returns ``(rel_l1, mod_input)`` and only
the scalar reaches host.

IMPORTANT (cclog 86 finding, verified against diffusers
``transformer_hunyuan_video15.py``): HV-1.5's block-0 AdaLN modulation ``temb`` is
**timestep-only** (``HunyuanVideo15TimeEmbedding.forward(timestep, timestep_r)``,
no pooled text/guidance). All text/image conditioning enters only as joint-attention
TOKENS, never into the modulation vector. Per the cclog-84 discriminator this is the
**Qwen-WEAK topology**, NOT the HV-1.0/Flux-good one — so the rel-L1 signal is expected
to be a weak predictor and the adaptive controller is expected NOT to transfer. This
probe exists to MEASURE the signal (the cclog-84 gate); the production deliverable for
HV-1.5 is the probe-free fixed cadence (`TeaCacheCalibration.cadence`).

Probe inputs are the minimal block-0 subset (hidden_states, timestep, timestep_r) —
text/image embeds are irrelevant to the timestep-only modulation. The hook reuses the
diffusers model's own submodules (time_embed -> x_embedder -> transformer_blocks[0].norm1),
faithful to the real forward by construction.

Monolithic runtime only. The segmented runtime (segmented15.py, per-block process
loading) has no single graph to attach a persistent prev_mod Parameter to — teacache
is unsupported there and the application raises for it.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from neuronx_distributed.trace.model_builder import BaseModelInstance
from nova.backends.trainium.core.application_base import NeuronApplicationBase
from nova.backends.trainium.core.config import InferenceConfig
from nova.backends.trainium.core.model_wrapper import ModelWrapper
from nova.backends.trainium.hunyuan_video.backbone15 import (
    HunyuanVideo15BackboneInferenceConfig,
    _model_kwargs_from_config,
)


class _HunyuanVideo15ProbeTraceModule(nn.Module):
    """Holds the diffusers HV-1.5 transformer; exposes the block-0 modulated input."""

    def __init__(self, config: HunyuanVideo15BackboneInferenceConfig) -> None:
        super().__init__()
        from diffusers.models.transformers.transformer_hunyuan_video15 import (
            HunyuanVideo15Transformer3DModel,
        )

        self.use_meanflow = bool(getattr(config, "use_meanflow", False))
        self.transformer = HunyuanVideo15Transformer3DModel(**_model_kwargs_from_config(config))

    def teacache_mod_input(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor,
    ) -> torch.Tensor:
        # Block-0 modulation is TIMESTEP-ONLY (transformer_hunyuan_video15.py:647 +
        # HunyuanVideo15TimeEmbedding). No pooled text / guidance — this is why the
        # signal is expected weak (cclog 86 / cclog-84 discriminator).
        t = self.transformer
        temb = t.time_embed(timestep, timestep_r=timestep_r if self.use_meanflow else None)
        h = t.x_embedder(hidden_states)
        # AdaLayerNormZero.forward returns
        # (norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp); index 0 is
        # the modulated image input.
        norm_hidden_states, *_ = t.transformer_blocks[0].norm1(h, emb=temb)
        return norm_hidden_states


class HunyuanVideo15TeacacheProbeFusedModel(nn.Module):
    """prev_mod nn.Parameter; forward returns (rel_l1, mod_input), aliased in place."""

    def __init__(self, config, *, seq_len: int, inner_dim: int, batch_size: int = 1) -> None:
        super().__init__()
        self.trace_module = _HunyuanVideo15ProbeTraceModule(config)
        self.prev_mod = nn.Parameter(
            torch.zeros(batch_size, seq_len, inner_dim), requires_grad=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mod_input = self.trace_module.teacache_mod_input(hidden_states, timestep, timestep_r)
        # cclog 83 relative-L1 (mean|mod-prev|/mean|prev|), reduced in float32;
        # mod_input (out[1]) stays bf16 so it aliases back into the bf16 prev_mod.
        m = mod_input.float()
        p = self.prev_mod.float()
        rel_l1 = (m - p).abs().mean() / (p.abs().mean() + 1e-8)
        return rel_l1, mod_input


class _HV15FusedProbeModelInstance(BaseModelInstance):
    def __init__(self, module_builder):
        super().__init__(module_builder, input_output_aliases={})

    def get(self, bucket_rank, **kwargs):
        # out 0 = rel_l1 (kept); out 1 = mod_input aliased to prev_mod (in-place).
        return self.module, {self.module.prev_mod: 1}


def _probe_seq_len(config) -> int:
    return (
        (int(config.latent_frames) // int(config.patch_size_t))
        * (int(config.latent_height) // int(config.patch_size))
        * (int(config.latent_width) // int(config.patch_size))
    )


class ModelWrapperHunyuanVideo15TeacacheProbeFused(ModelWrapper):
    def __init__(self, config, model_cls, tag="", compiler_args=None,
                 priority_model_idx=None, model_init_kwargs=None):
        super().__init__(config, model_cls, tag, compiler_args,
                         priority_model_idx, model_init_kwargs or {})
        self.bucket_config = None

    def input_generator(self):
        bs = int(getattr(self.config.neuron_config, "batch_size", 1))
        dtype = self.config.neuron_config.torch_dtype
        return [
            (
                torch.randn(
                    [
                        bs,
                        self.config.in_channels,
                        self.config.latent_frames,
                        self.config.latent_height,
                        self.config.latent_width,
                    ],
                    dtype=dtype,
                ),
                torch.ones([bs], dtype=dtype),
                torch.ones([bs], dtype=dtype),
            )
        ]

    def get_model_instance(self):
        config = self.config
        dtype = self.config.neuron_config.torch_dtype
        seq = _probe_seq_len(config)
        inner = int(config.inner_dim)
        bs = int(getattr(config.neuron_config, "batch_size", 1))

        def _create():
            m = HunyuanVideo15TeacacheProbeFusedModel(
                config, seq_len=seq, inner_dim=inner, batch_size=bs
            )
            return m.to(dtype=dtype).eval()

        return _HV15FusedProbeModelInstance(module_builder=_create)

    def forward(self, hidden_states, timestep, timestep_r):
        if self.model is None:
            raise RuntimeError("Forward called before load.")
        dtype = self.config.neuron_config.torch_dtype
        return self._forward(
            hidden_states.to(dtype), timestep.to(dtype), timestep_r.to(dtype)
        )


class NeuronHunyuanVideo15TeacacheProbeFusedApplication(NeuronApplicationBase):
    """HV-1.5 fused-A probe app (monolithic only)."""

    _model_cls = HunyuanVideo15TeacacheProbeFusedModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = ModelWrapperHunyuanVideo15TeacacheProbeFused
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
        return HunyuanVideo15BackboneInferenceConfig

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self) -> str:
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return "--model-type=transformer -O1 --auto-cast=none"

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        # The probe nests the diffusers model under trace_module.transformer.* ;
        # prev_mod stays the init Parameter (overwritten in place via the alias).
        del config
        return {
            (
                f"trace_module.{key}"
                if key.startswith("transformer.")
                else f"trace_module.transformer.{key}"
            ): value
            for key, value in state_dict.items()
        }

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    def teacache_delta(self, hidden_states, timestep, timestep_r):
        out = self.models[0](hidden_states, timestep, timestep_r)
        return out[0] if isinstance(out, (tuple, list)) else out
