"""Whether Wan's umT5 fits on the device beside the DiT at a given tp. Torch-free.

``NeuronWanTextEncoderApplication`` puts umT5 on every rank of the TP group in the application's
``text_encoder_dtype`` (bf16 by default, 11.4 GB for the 5.68 B-parameter encoder; fp32 on
request, 22.7 GB), sharded across the group, next to this rank's DiT shard. At tp4 the fp32
encoder is about 5.3 GiB beside 6.7 GiB of DiT, well inside 24 GiB. Any topology that leaves
fewer cores to the TP group doubles both: CFG or context parallelism on four cores, or DP replicas
of tp2 (dp2tp2), each core then holding about 10.6 GiB of fp32 encoder (5.3 GiB in bf16) and
13.3 GiB of DiT. In fp32 that is more than its HBM, and the device would only find out after
minutes of allocator retries (``difflet/backends/neuron/hbm.py``); in bf16 it fits.

The application base checks each component as it loads it, so it would pass the DiT and catch the
encoder only after the DiT had loaded. This check adds the two up before either loads. The
estimate depends on tp and on the checkpoint's own sizes, not on the mode flags, so DP and any
other narrow TP are covered the same way as CFG and CP. It reads only the safetensors headers. It
is a lower bound: everything is counted as sharded over the TP group, though the DiT's embeddings
and norms are replicated, so a rejection means the components clearly cannot fit. Both callers
use it: ``TorchNeuronWanApplication.__init__``, so an API caller gets the error before any rank
loads, and the Wan CLI driver, which moves umT5 to the host before torchrun starts.
"""

from __future__ import annotations

from pathlib import Path

from difflet.backends.neuron.hbm import Component, components_bytes, components_fit

#: umT5's element size on the device by default: bf16, ``TorchNeuronWanApplication``'s
#: ``text_encoder_dtype`` default, which the CLI keeps (fp32 is 4, on request).
TEXT_ENCODER_DTYPE_BYTES = 2
TEXT_ENCODER = Component("text_encoder", TEXT_ENCODER_DTYPE_BYTES)


def device_text_encoder_components(experts: tuple[str, ...] = ("transformer",),
                                   dit_dtype_bytes: int = 2,
                                   encoder_dtype_bytes: int = TEXT_ENCODER_DTYPE_BYTES,
                                   ) -> tuple[Component, ...]:
    """umT5 in ``encoder_dtype_bytes`` and the DiT ``experts`` (the checkpoint subdirectories
    that stay resident with it: ``("transformer",)``, plus ``"transformer_2"`` for Wan 2.2's
    low-noise expert, or none)."""
    return (TEXT_ENCODER.in_dtype(encoder_dtype_bytes),
            *(Component(name, int(dit_dtype_bytes)) for name in experts))


def device_text_encoder_bytes(
    model_dir: str | Path,
    *,
    tp_degree: int,
    experts: tuple[str, ...] = ("transformer",),
    dit_dtype_bytes: int = 2,
    encoder_dtype_bytes: int = TEXT_ENCODER_DTYPE_BYTES,
) -> int:
    """Lower bound of one rank's weight bytes with umT5 (in ``encoder_dtype_bytes``) on the
    device beside the DiT ``experts``, all sharded over ``tp_degree``."""
    return components_bytes(
        model_dir, device_text_encoder_components(experts, dit_dtype_bytes, encoder_dtype_bytes),
        tp_degree=tp_degree)


def device_text_encoder_fits(
    model_dir: str | Path,
    *,
    tp_degree: int,
    experts: tuple[str, ...] = ("transformer",),
    dit_dtype_bytes: int = 2,
    encoder_dtype_bytes: int = TEXT_ENCODER_DTYPE_BYTES,
) -> tuple[bool, int, int]:
    """``(fits, estimate, limit)``: the ``device_text_encoder_bytes`` estimate against
    ``FIT_FRACTION`` of one core's HBM."""
    return components_fit(
        model_dir, device_text_encoder_components(experts, dit_dtype_bytes, encoder_dtype_bytes),
        tp_degree=tp_degree)


__all__ = [
    "TEXT_ENCODER",
    "TEXT_ENCODER_DTYPE_BYTES",
    "device_text_encoder_bytes",
    "device_text_encoder_components",
    "device_text_encoder_fits",
]
