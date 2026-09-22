"""Small stateful TeaCache graphs: replicated prefix weights, device-side reduction."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuronx_distributed.trace.model_builder import BaseModelInstance

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.model_wrapper import ModelWrapper


class RelativeL1Probe(nn.Module):
    def __init__(self, *, batch_size: int, seq_len: int, inner_dim: int):
        super().__init__()
        self.prev_mod = nn.Parameter(
            torch.zeros(batch_size, seq_len, inner_dim), requires_grad=False
        )

    def delta_and_state(self, mod_input):
        # Buckets share one state allocation. Only live tokens enter the reduction;
        # the padded output has the same shape/dtype as the aliased Parameter.
        seq_len = mod_input.shape[1]
        previous = self.prev_mod[:, :seq_len].float()
        delta = (mod_input.float() - previous).abs().mean() / previous.abs().mean().clamp_min(1e-8)
        state = F.pad(mod_input, (0, 0, 0, self.prev_mod.shape[1] - seq_len))
        return delta, state


class _ProbeModelInstance(BaseModelInstance):
    def __init__(self, module_builder):
        super().__init__(module_builder, input_output_aliases={})

    def get(self, bucket_rank, **kwargs):
        return self.module, {self.module.prev_mod: 1}


class FusedProbeWrapper(ModelWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket_config = None

    def get_model_instance(self):
        def create():
            return (
                self.model_cls(self.config).to(dtype=self.config.neuron_config.torch_dtype).eval()
            )

        return _ProbeModelInstance(create)

    def forward(self, hidden_states, timestep):
        if self.model is None:
            raise RuntimeError("TeaCache probe called before load.")
        return self._forward(hidden_states, timestep)


class PrefixProbeApplication(NeuronApplicationBase):
    """Load only a probe's canonical checkpoint subset into a separate store."""

    shared_weights_layout = "teacache-prefix-v1"
    weight_prefixes: tuple[str, ...] = ()

    @classmethod
    def get_state_dict(cls, model_name_or_path, config):
        # Avoid loading tens of GB of attention/FFN weights just to discard them.
        from safetensors import safe_open

        shards = sorted(Path(model_name_or_path).glob("*.safetensors"))
        if not shards:
            return super().get_state_dict(model_name_or_path, config)
        state = {}
        for path in shards:
            with safe_open(path, framework="pt", device="cpu") as shard:
                for name in shard.keys():
                    if name.startswith(cls.weight_prefixes):
                        state[name] = shard.get_tensor(name)
        return cls.convert_hf_to_neuron_state_dict(state, config)


class FusedProbeApplication(PrefixProbeApplication):
    """Common lifecycle for probes that load only their small checkpoint subset.

    Prefix layers are replicated on each rank in the backbone's world. They do
    not use the backbone's TP/CP/CFG collectives, so each rank reduces the full
    signal. prev_mod is NEFF state, never a host or checkpoint tensor.
    """

    state_tensor_names = frozenset({"prev_mod"})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.wrapper_cls(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    def get_compiler_args(self):
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return "--model-type=transformer -O1 --auto-cast=none"

    def forward(self, hidden_states, timestep):
        return self.models[0](hidden_states, timestep)

    def teacache_delta(self, hidden_states, timestep):
        out = self(hidden_states, timestep)
        return out[0] if isinstance(out, (tuple, list)) else out
