"""Whether the LTX-2 DiT fits one NeuronCore at a given tp. Torch-free.

The DiT is 18.9 B parameters, 35.2 GiB in bf16 (the checkpoint stores most tensors in bf16 and
the modulation tables in fp32; the device holds all of it in bf16). Its TP layers shard each
block's six attentions and two feed-forwards over the TP group, so one rank holds about
35.2 / tp GiB plus the replicated rest (the patch projections, the caption projections, the
timestep embeddings and the per-block modulation tables and q/k norms): about 10 GiB at tp4, and
twice the sharded part at tp2, while tp1 cannot fit a 24 GiB core. A topology that narrows the
TP group narrows every rank's share the same way (CFG on four cores, DP replicas). The
application base checks the exact meta-built bytes when the DiT loads
(``difflet/backends/neuron/hbm.py``); the CLI driver uses the estimate here to refuse the launch
before torchrun, or a DP router's workers, start any rank.

The estimate reads only the safetensors headers and counts everything as sharded over tp, so it
is a lower bound: a rejection means the weights clearly cannot fit.
"""

from __future__ import annotations

from pathlib import Path

from difflet.backends.neuron.hbm import Component, components_bytes, components_fit

#: The DiT runs in bf16 on the device (decision D42, ``NeuronLtx2TransformerApplication``).
DIT_DTYPE_BYTES = 2
#: The device text encoder's default dtype (bf16, the host path's).
TEXT_ENCODER_DTYPE_BYTES = 2
DIT = Component("transformer", DIT_DTYPE_BYTES)
#: Gemma-3's language model in ``<model>/text_encoder`` (transformers' checkpoint): no vision
#: tower, no projector, no ``lm_head``, and only the shards ``model.safetensors.index.json``
#: lists (``model-*.safetensors``): the directory may also list a diffusers index of another
#: checkpoint.
TEXT_ENCODER = Component(
    "text_encoder", TEXT_ENCODER_DTYPE_BYTES,
    skipped_prefixes=("vision_tower.", "multi_modal_projector.", "language_model.lm_head."),
    index="model.safetensors.index.json",
)


def text_encoder_lm_numel(model_dir: str | Path) -> int:
    """Element count of the language model in ``<model_dir>/text_encoder``: what ``NeuronLtx2TextEncoderApplication``
    loads (``TEXT_ENCODER``)."""
    return TEXT_ENCODER.numel(model_dir)


def dit_bytes(model_dir: str | Path, *, tp_degree: int,
              dit_dtype_bytes: int = DIT_DTYPE_BYTES) -> int:
    """Lower bound of one rank's DiT weight bytes: ``<model_dir>/transformer`` over tp."""
    return components_bytes(model_dir, (DIT.in_dtype(dit_dtype_bytes),), tp_degree=tp_degree)


def dit_fits(model_dir: str | Path, *, tp_degree: int,
             dit_dtype_bytes: int = DIT_DTYPE_BYTES) -> tuple[bool, int, int]:
    """``(fits, estimate, limit)``: ``dit_bytes`` against ``FIT_FRACTION`` of one core's HBM."""
    return components_fit(model_dir, (DIT.in_dtype(dit_dtype_bytes),), tp_degree=tp_degree)


def device_text_encoder_components(encoder_dtype_bytes: int = TEXT_ENCODER_DTYPE_BYTES,
                                   dit_dtype_bytes: int = DIT_DTYPE_BYTES
                                   ) -> tuple[Component, Component]:
    """The language model beside the DiT on the same cores."""
    return TEXT_ENCODER.in_dtype(encoder_dtype_bytes), DIT.in_dtype(dit_dtype_bytes)


def device_text_encoder_bytes(model_dir: str | Path, *, tp_degree: int,
                              encoder_dtype_bytes: int = TEXT_ENCODER_DTYPE_BYTES,
                              dit_dtype_bytes: int = DIT_DTYPE_BYTES) -> int:
    """Lower bound of one rank's weight bytes with the language model on the device beside the
    DiT, both sharded over ``tp_degree``."""
    return components_bytes(
        model_dir, device_text_encoder_components(encoder_dtype_bytes, dit_dtype_bytes),
        tp_degree=tp_degree)


def device_text_encoder_fits(model_dir: str | Path, *, tp_degree: int,
                             encoder_dtype_bytes: int = TEXT_ENCODER_DTYPE_BYTES,
                             dit_dtype_bytes: int = DIT_DTYPE_BYTES) -> tuple[bool, int, int]:
    """``(fits, estimate, limit)``: ``device_text_encoder_bytes`` against ``FIT_FRACTION`` of
    one core's HBM."""
    return components_fit(
        model_dir, device_text_encoder_components(encoder_dtype_bytes, dit_dtype_bytes),
        tp_degree=tp_degree)


__all__ = [
    "DIT",
    "DIT_DTYPE_BYTES",
    "TEXT_ENCODER",
    "TEXT_ENCODER_DTYPE_BYTES",
    "device_text_encoder_bytes",
    "device_text_encoder_components",
    "device_text_encoder_fits",
    "dit_bytes",
    "dit_fits",
    "text_encoder_lm_numel",
]
