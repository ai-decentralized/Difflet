"""Whether Qwen-Image's device components fit one NeuronCore at a given tp. Torch-free.

The DiT is 20.43 B parameters, 38.05 GiB in bf16. Its TP layers shard it over the TP group, so
one rank holds about 38.05 / tp GiB plus the replicated embeddings and norms: measured on trn2,
10.1 GiB at tp4 and 19.75 GiB at tp2, while tp1 (38 GiB) cannot fit a 24 GiB core
(docs/verification/qwen-image-neuron-evidence.md). A topology that narrows the TP group narrows
every rank's share the same way: CP or CFG on four cores, or DP replicas (dp2tp2 fits, dp4tp1
does not). The application base checks the exact meta-built bytes when the DiT loads
(``difflet/backends/neuron/hbm.py``), so an API caller is refused before anything is
materialised. The CLI driver uses the estimate here to refuse the launch before torchrun, or a
DP router's workers, start any rank.

The opt-in device text encoder (``NeuronQwenTextEncoderApplication``) adds Qwen2.5-VL's language
model, 7.07 B parameters (13.2 GiB in bf16; the 0.68 B vision tower and the 0.54 B ``lm_head``
stay on disk), sharded over the same TP group on the same cores: about 3.3 GiB per rank at tp4
beside 9.5 GiB of DiT, well inside 24 GiB, but 6.6 GiB beside 19.0 GiB at tp2, which is over
the budget. The base's own check sees one component at a time, so it would pass the DiT and
refuse the encoder only after the DiT had loaded; ``device_text_encoder_fits`` adds the two up
before either loads. Its callers: ``TorchNeuronQwenImageApplication.__init__`` (an API caller
gets the error before any rank loads) and the Qwen CLI driver, which falls back to the host
encoder before torchrun starts.

The estimates read only the safetensors headers and count everything as sharded over tp, so
they are lower bounds: a rejection means the weights clearly cannot fit.
"""

from __future__ import annotations

from pathlib import Path

from difflet.backends.neuron.hbm import Component, components_bytes, components_fit

#: The DiT runs in bf16 on the device (decision D42, ``NeuronQwenImageTransformerApplication``).
DIT_DTYPE_BYTES = 2
#: The device text encoder's default dtype (bf16, as diffusers and the host path run it).
TEXT_ENCODER_DTYPE_BYTES = 2
DIT = Component("transformer", DIT_DTYPE_BYTES)
#: ``<model>/text_encoder`` without the vision tower and the ``lm_head``, which the device text
#: encoder never loads.
TEXT_ENCODER = Component("text_encoder", TEXT_ENCODER_DTYPE_BYTES,
                         skipped_prefixes=("visual.", "model.visual.", "lm_head."))


def text_encoder_lm_numel(model_dir: str | Path) -> int:
    """Element count of the language model in ``<model_dir>/text_encoder``: what ``NeuronQwenTextEncoderApplication``
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
