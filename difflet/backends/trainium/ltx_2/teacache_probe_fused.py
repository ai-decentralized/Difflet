"""LTX-2 TeaCache fused-A probe (single-mode transformer only).

Same recipe as the HunyuanVideo (cclog 80), Qwen-Image (cclog 81) and Flux
(cclog 85) fused probes: ``prev_mod`` is a persistent on-device ``nn.Parameter``
updated in place via ``input_output_aliases``; the forward returns
``(rel_l1, mod_input)`` and only the 4-byte scalar reaches host.

Why this one pays for itself
----------------------------
In single mode the host CPU transformer (``transformer.py:_load_cpu_transformer``)
has exactly one caller: ``teacache_mod_input``. It is a full bf16 copy of the
48-block diffusers model held in host RAM purely to compute the block-0 signal.
Moving the signal on device makes that copy dead in single mode.

Segmented mode is the opposite case and deliberately keeps its host path: there
the same CPU copy is load-bearing for ``_prepare_frontend`` and
``_final_projection`` (segmented.py:1075, 1211), so the signal is already free
and no probe is mounted.

Weight naming — "same weights => same names by construction"
------------------------------------------------------------
The single-mode backbone is itself a wrapper: ``_LTX2TransformerTraceModule``
holds the stock diffusers model at ``self.transformer``, and its converter bakes
that prefix into every checkpoint key (transformer.py:831-834). So the probe
SUBCLASSES the trace module and adds nothing but ``prev_mod``: the inner model
keeps the attribute path ``transformer.*``, and ``tp_rank_util.rank`` /
``global_rank.rank`` are created by the inherited ``__init__`` under the same
names the converter emits. Adding another wrapper layer here would nest
everything one level deeper and fail on device with ``Missing weight tensor with
key ...``, which is what Flux hit on 2026-08-30 (GitHub issue #39).

Name parity is not optional: the shared weight store keys only on
``(source, dtype, tp_degree, world_size, context_parallel, sequence_parallel,
cfg_parallel)`` with no component tag, so a probe built from the backbone's own
config lands on the backbone's store entry and is handed its shards directly.

The only tensor the probe adds is ``prev_mod``, aliased to output 1, so NxD
classifies it as INPUT_STATE: zero-filled by ``StateInitializer`` at load and
never looked up in the checkpoint. It is declared in ``state_tensor_names``.

Parallelism
-----------
Only the per-block ``attn*``/``ff`` submodules are TP-sharded; ``proj_in``,
``time_embed``, ``transformer_blocks[0].scale_shift_table`` and ``norm1`` stay
replicated (transformer.py:320-361), so the probe's prefix math is identical on
every rank. The probe's forward never runs the CFG-parallel batch scatter: the
pipeline probes the single un-doubled latent at batch 1, exactly as the host
signal did (pipeline.py:631-648). The inherited ``__init__`` still creates the
``global_rank`` buffer under CFG parallel so the checkpoint keys stay in parity.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.backends.trainium.ltx_2.transformer import (
    LTX2TransformerInferenceConfig,
    NeuronLTX2TransformerApplication,
    _LTX2TransformerTraceModule,
)

#: Tensors the probe adds on top of the backbone. Aliased in place by the
#: ModelInstance below, so they are NEFF state (zero-initialised at load), not
#: checkpoint weights.
PROBE_STATE_TENSORS: frozenset[str] = frozenset({"prev_mod"})


def probe_inner_dim(config) -> int:
    """Hidden width of the block-0 modulated input.

    ``LTX2TransformerInferenceConfig`` has no ``inner_dim`` attribute; the
    diffusers model derives it the same way for ``proj_in`` and
    ``scale_shift_table``.
    """
    return int(config.num_attention_heads) * int(config.attention_head_dim)


class LTX2TeacacheProbeFusedModel(_LTX2TransformerTraceModule):
    """The single-mode trace module plus a persistent ``prev_mod`` state.

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
        self, hidden_states: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Block-0 modulated video self-attention input.

        The same prefix the host path computed on the CPU copy
        (transformer.py:789-806): ``norm1(proj_in(latent)) * (1 + scale_msa) +
        shift_msa``. The modulation is timestep-only, so it is identical for the
        cond/uncond CFG halves and the caller passes the un-doubled latent.
        Stops before attention, so no coords, text or audio inputs are needed.
        """
        model = self.transformer
        batch_size = hidden_states.shape[0]
        hidden_states = model.proj_in(hidden_states)
        temb, _embedded_timestep = model.time_embed(
            timestep.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        block0 = model.transformer_blocks[0]
        video_ada_params = block0.get_mod_params(block0.scale_shift_table, temb, batch_size)
        shift_msa, scale_msa = video_ada_params[0], video_ada_params[1]
        return block0.norm1(hidden_states) * (1 + scale_msa) + shift_msa

    def forward(  # type: ignore[override] — the probe traces its own signature
        self, hidden_states: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mod_input = self.teacache_mod_input(hidden_states, timestep)
        # Relative-L1 signal (mean|mod-prev| / mean|prev|), reduced in float32;
        # mod_input (out[1]) keeps the model dtype so it aliases back into the
        # same-dtype prev_mod Parameter.
        m = mod_input.float()
        p = self.prev_mod.float()
        rel_l1 = (m - p).abs().mean() / (p.abs().mean() + 1e-8)
        return rel_l1, mod_input


class _LTX2FusedProbeModelInstance(BaseModelInstance):
    def __init__(self, module_builder):
        super().__init__(module_builder, input_output_aliases={})

    def get(self, bucket_rank, **kwargs):
        # out 0 = rel_l1 (kept); out 1 = mod_input aliased to prev_mod (in-place).
        return self.module, {self.module.prev_mod: 1}


class ModelWrapperLTX2TeacacheProbeFused(ModelWrapper):
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
        # Probe at batch 1: the signal is timestep-only, so the pipeline passes
        # the single un-doubled latent even when the backbone compiles at
        # batch 2 for CFG parallel.
        return [
            (
                torch.randn(
                    [1, int(config.video_seq_len), int(config.in_channels)], dtype=dtype
                ),
                torch.ones([1], dtype=dtype),
            )
        ]

    def get_model_instance(self):
        config = self.config
        dtype = config.neuron_config.torch_dtype
        seq = int(config.video_seq_len)
        inner = probe_inner_dim(config)

        def _create():
            model = LTX2TeacacheProbeFusedModel(
                config, seq_len=seq, inner_dim=inner, batch_size=1
            )
            return model.to(dtype=dtype).eval()

        return _LTX2FusedProbeModelInstance(module_builder=_create)

    def forward(self, hidden_states, timestep):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        dtype = self.config.neuron_config.torch_dtype
        return self._forward(hidden_states.to(dtype), timestep.to(dtype))


class NeuronLTX2TeacacheProbeFusedApplication(NeuronApplicationBase):
    """LTX-2 fused-A probe app (single-mode transformer only)."""

    _model_cls = LTX2TeacacheProbeFusedModel
    state_tensor_names = PROBE_STATE_TENSORS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = ModelWrapperLTX2TeacacheProbeFused
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
        return LTX2TransformerInferenceConfig

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self) -> str:
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return "--model-type=transformer -O1 --auto-cast=none"

    # The probe's weights ARE the backbone's weights under the backbone's names,
    # so its checkpoint converter is the backbone's converter object — the two
    # cannot drift apart.
    convert_hf_to_neuron_state_dict = staticmethod(
        NeuronLTX2TransformerApplication.convert_hf_to_neuron_state_dict
    )

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    def teacache_delta(self, hidden_states, timestep) -> torch.Tensor:
        """Relative-L1 of the block-0 modulated input vs the previous probed step."""
        out = self.models[0](hidden_states, timestep)
        return out[0] if isinstance(out, (tuple, list)) else out


__all__ = [
    "LTX2TeacacheProbeFusedModel",
    "ModelWrapperLTX2TeacacheProbeFused",
    "NeuronLTX2TeacacheProbeFusedApplication",
    "PROBE_STATE_TENSORS",
    "probe_inner_dim",
]
