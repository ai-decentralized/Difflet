"""Flux backbone with an in-graph Gram tap (H2').

The cache framework's whole cost model is built from inner products of the
per-step transformer outputs. An inner product is a full contraction, so it
collapses a 512 KB tensor into a scalar and commutes with any partition axis.
That makes it the one quantity that can be measured *inside* the backbone
invocation that is happening anyway — no extra graph dispatch, which the
H1a-H1g dispatch-floor result showed is the only affordable placement.

This module compiles the real backbone plus that tap so the perturbation it
causes to the ~267 ms step can be measured. Two placements are provided:

``post`` — tap the gathered output. Every rank holds the full tensor, so each
    computes redundant full-length inner products and no collective is added.
``shard`` — contract only this rank's slice of the output channels and sum the
    partial results across ranks. The arithmetic drops fourfold and one small
    collective appears in its place, so this reproduces the *cost structure* of
    a pre-gather tap while leaving ``proj_out`` untouched.

A true pre-gather tap additionally removes the gather from the critical path and
needs ``proj_out`` rebuilt with ``gather_output=False``; that is deferred until
the cheaper pair below shows the tap is affordable at all.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from neuronx_distributed.parallel_layers.mappings import (
    reduce_from_tensor_model_parallel_region,
)
from neuronx_distributed.trace.model_builder import BaseModelInstance

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.models.flux.modeling_flux import (
    ModelWrapperFluxBackbone,
    NeuronFluxBackboneApplication,
    NeuronFluxTransformer2DModel,
)

TAP_PLACEMENTS = ("post", "bf16", "noalias", "shard")


def _seq_len(config) -> int:
    return int(config.height) * int(config.width) // ((2 * int(config.vae_scale_factor)) ** 2)


def _out_dim(config) -> int:
    # Mirrors NeuronFluxTransformer2DModel: out_channels falls back to
    # in_channels when the checkpoint config omits it (modeling_flux.py:273).
    out_channels = getattr(config, "out_channels", None) or int(config.in_channels)
    return int(config.patch_size) * int(config.patch_size) * int(out_channels)


class FluxBackboneGramTapModel(nn.Module):
    """Real backbone; emits this step's Gram row against a resident anchor.

    One resident anchor is enough for what this spike asks — the tap's arithmetic
    and its alias write-back are both present, and H1a already proved a K=2 bank
    rotates correctly.

    Two runtime contracts shape the output tuple, both learned the hard way:

    * An aliased output is written in place and does **not** come back to the
      host, so the returned values are the tuple's prefix. Aliases therefore have
      to sit at the end, or the packer looks for a value the runtime never sent.
    * A tensor returned both plainly and as an alias target is deduplicated by
      XLA, which leaves the alias index dangling (the H1b finding).

    Both are satisfied by keeping the bank in fp32: the write-back is a genuine
    convert rather than a copy the compiler can fold away, and the Gram
    accumulation stops losing precision to bf16 on the way in.
    """

    def __init__(self, config, *, placement: str = "post") -> None:
        super().__init__()
        if placement not in TAP_PLACEMENTS:
            raise ValueError(f"placement must be one of {TAP_PLACEMENTS}")
        self.placement = placement
        self.transformer = NeuronFluxTransformer2DModel(config)
        seq, dim = _seq_len(config), _out_dim(config)
        self.tp_degree = int(config.neuron_config.tp_degree)
        self.shard_width = dim // self.tp_degree
        if placement == "bf16":
            # Flat so the alias target can be a reshape of the output: a reshape
            # is a distinct HLO value (dodging the dedup contract) yet a bitcast
            # on contiguous memory, so it costs nothing and the bank stays bf16.
            self.anchor = nn.Parameter(
                torch.zeros(1, seq * dim, dtype=torch.bfloat16), requires_grad=False
            )
        else:
            # ``noalias`` deliberately mirrors ``post`` — same fp32 bank, same
            # reductions — and differs only in dropping the write-back, so the
            # difference between the two prices the write-back on its own.
            width = dim if placement in ("post", "noalias") else self.shard_width
            self.anchor = nn.Parameter(
                torch.zeros(1, seq, width, dtype=torch.float32), requires_grad=False
            )

    def _slice_for_rank(self, tensor: torch.Tensor) -> torch.Tensor:
        """This rank's slice of the output channels, taken from the replica.

        Not yet implemented: the builder traces one program that every rank runs
        (SPMD), so a Python-time rank lookup would bake rank 0's slice into all
        four. A correct version has to read rank from device state the way the
        backbone does via ``global_rank.rank``. Deferred until the ``post``
        placement shows the tap is affordable at all.
        """
        raise NotImplementedError(
            "shard placement needs device-side rank selection; see docstring"
        )

    def _gram_row(self, current: torch.Tensor) -> torch.Tensor:
        """Two inner products, accumulated in fp32: <anchor, c> and <c, c>."""
        row = torch.stack([(self.anchor * current).sum(), (current * current).sum()])
        if self.placement == "shard":
            # Each rank held only a partial contraction; one collective of two
            # floats turns them into the true inner products. The payload is
            # tiny, so this costs the collective's fixed floor, not bandwidth.
            row = reduce_from_tensor_model_parallel_region(row)
        return row

    def forward(self, *model_inputs):
        output = self.transformer(*model_inputs)
        if self.placement == "noalias":
            # Identical to ``post`` except that nothing is written back.
            return output, self._gram_row(output.float())
        if self.placement == "bf16":
            flat = output.reshape(1, -1)
            row = torch.stack(
                [
                    (self.anchor * flat).sum(dtype=torch.float32),
                    (flat * flat).sum(dtype=torch.float32),
                ]
            )
            return output, row, flat
        tapped = output if self.placement == "post" else self._slice_for_rank(output)
        current = tapped.float()
        # Host receives (output, gram_row); `current` is the alias target and
        # stays on device, which is why it has to be last.
        return output, self._gram_row(current), current


class _GramTapModelInstance(BaseModelInstance):
    def __init__(self, module_builder):
        super().__init__(module_builder, input_output_aliases={})

    def get(self, bucket_rank, **kwargs):
        module = self.module
        # out 0 = output, out 1 = gram row (both returned). Where a bank is
        # written back it is out 2, which must be last: an aliased output never
        # returns to host, so anything after it would leave a hole.
        if module.placement == "noalias":
            return module, {}
        return module, {module.anchor: 2}


class ModelWrapperFluxGramTap(ModelWrapperFluxBackbone):
    """Backbone wrapper whose traced module carries the tap and its bank."""

    def __init__(self, *args, placement: str = "post", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.placement = placement

    def get_model_instance(self):
        config, placement = self.config, self.placement
        dtype = config.neuron_config.torch_dtype

        def _create_model():
            model = FluxBackboneGramTapModel(config, placement=placement)
            model = model.to(dtype=dtype)
            if placement in ("post", "noalias", "shard"):
                # `.to(dtype)` above demoted the bank; fp32 is the point there.
                model.anchor = nn.Parameter(model.anchor.float(), requires_grad=False)
            return model.eval()

        return _GramTapModelInstance(module_builder=_create_model)


class NeuronFluxGramTapApplication(NeuronFluxBackboneApplication):
    """Backbone application compiled with the in-graph Gram tap."""

    # This graph nests the transformer one level down, so its shards are keyed
    # ``transformer.*`` rather than the stock names. Without this declaration the
    # shared-weight store hashes it identically to the plain backbone and hands
    # it the plain backbone's shards — the H1a incident, recurring silently.
    weight_namespace = "flux-gram-tap"

    def __init__(self, *args, placement: str = "post", **kwargs) -> None:
        self._tap_placement = placement
        super().__init__(*args, **kwargs)

    def get_model_wrapper_cls(self):
        placement = self._tap_placement

        def _factory(**kwargs):
            return ModelWrapperFluxGramTap(placement=placement, **kwargs)

        return _factory

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        # The traced module nests the real transformer one level down and adds
        # the resident anchor, which is runtime state rather than a weight.
        converted = NeuronFluxBackboneApplication.convert_hf_to_neuron_state_dict(
            state_dict, config
        )
        rank = converted.pop("global_rank.rank", None)
        nested = {f"transformer.{key}": value for key, value in converted.items()}
        if rank is not None:
            nested["transformer.global_rank.rank"] = rank
        seq, dim = _seq_len(config), _out_dim(config)
        placement = getattr(config, "gram_tap_placement", "post")
        if placement == "bf16":
            nested["anchor"] = torch.zeros(1, seq * dim, dtype=torch.bfloat16)
        else:
            width = (
                dim
                if placement in ("post", "noalias")
                else dim // int(config.neuron_config.tp_degree)
            )
            nested["anchor"] = torch.zeros(1, seq, width, dtype=torch.float32)
        return nested


__all__ = [
    "FluxBackboneGramTapModel",
    "ModelWrapperFluxGramTap",
    "NeuronFluxGramTapApplication",
    "TAP_PLACEMENTS",
]
