"""Whether Wan's umT5 fits on the device beside the DiT at a given tp. Torch-free.

``NeuronWanTextEncoderApplication`` puts umT5 on every rank of the TP group in fp32 (22.7 GB for
the 5.68 B-parameter encoder, sharded across the group) next to this rank's DiT shard. At tp4 that
is about 5.3 GiB of encoder beside 8.1 GiB of DiT, well inside 24 GiB. Any topology that leaves
fewer cores to the TP group doubles both: CFG or context parallelism on four cores, or DP replicas
of tp2 (dp2tp2), each core then holding about 10.6 GiB of encoder and 14 GiB of DiT. That is more
than its HBM, and the device would only find out after minutes of allocator retries
(``difflet/backends/neuron/hbm.py``).

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

from difflet.backends.neuron.hbm import FIT_FRACTION, per_core_hbm_bytes, safetensors_numel

#: umT5 runs in fp32 on the device whatever the DiT's dtype (``NeuronWanTextEncoderApplication``).
TEXT_ENCODER_DTYPE_BYTES = 4


def device_text_encoder_bytes(
    model_dir: str | Path,
    *,
    tp_degree: int,
    experts: tuple[str, ...] = ("transformer",),
    dit_dtype_bytes: int = 2,
) -> int:
    """Lower bound of one rank's weight bytes with umT5 on the device beside the DiT ``experts``
    (the checkpoint subdirectories that stay resident with it: ``("transformer",)``, plus
    ``"transformer_2"`` for Wan 2.2's low-noise expert, or none), all sharded over ``tp_degree``."""
    root = Path(model_dir)
    total = safetensors_numel(root / "text_encoder") * TEXT_ENCODER_DTYPE_BYTES
    total += sum(safetensors_numel(root / name) for name in experts) * int(dit_dtype_bytes)
    return total // max(int(tp_degree), 1)


def device_text_encoder_fits(
    model_dir: str | Path,
    *,
    tp_degree: int,
    experts: tuple[str, ...] = ("transformer",),
    dit_dtype_bytes: int = 2,
) -> tuple[bool, int, int]:
    """``(fits, estimate, limit)``: the ``device_text_encoder_bytes`` estimate against
    ``FIT_FRACTION`` of one core's HBM."""
    estimate = device_text_encoder_bytes(
        model_dir, tp_degree=tp_degree, experts=experts, dit_dtype_bytes=dit_dtype_bytes
    )
    limit = int(per_core_hbm_bytes()[0] * FIT_FRACTION)
    return estimate <= limit, estimate, limit


__all__ = ["TEXT_ENCODER_DTYPE_BYTES", "device_text_encoder_bytes", "device_text_encoder_fits"]
