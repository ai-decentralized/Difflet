"""Whether the HunyuanVideo DiT fits one NeuronCore at a given tp. Torch-free.

The DiT is 12.8 B parameters, 23.9 GiB in bf16. One 24 GiB core can give the weights at most
``FIT_FRACTION`` of its HBM (22.8 GiB), so tp1 cannot fit; its TP layers shard each block's
attentions, feed-forwards and AdaLN modulation linears over the TP group, so one rank holds about
23.9 / tp GiB plus the replicated rest (the patch and text embedders, parts of the token refiner,
the timestep and guidance embeddings, the q/k norms): 12.0 GiB at tp2 and 6.1 GiB at tp4 (15.2
and 10.8 GiB while the 6.80 GB of modulation linears were replicated). A topology that narrows
the TP group (CP or SP on the same four cores) narrows every rank's share the same way. The
application base checks the exact meta-built bytes when the DiT loads
(``difflet/backends/neuron/hbm.py``); the CLI driver uses the estimate here to refuse the launch
before torchrun starts any rank.

The estimate reads only the safetensors headers and counts everything as sharded over tp, so it
is a lower bound: a rejection means the weights clearly cannot fit.
"""

from __future__ import annotations

from pathlib import Path

from difflet.backends.neuron.hbm import Component, components_bytes, components_fit

#: The DiT runs in bf16 on the device (decision D42, ``require_bf16_on_device``).
DIT_DTYPE_BYTES = 2
DIT = Component("transformer", DIT_DTYPE_BYTES)


def dit_bytes(model_dir: str | Path, *, tp_degree: int,
              dit_dtype_bytes: int = DIT_DTYPE_BYTES) -> int:
    """Lower bound of one rank's DiT weight bytes: ``<model_dir>/transformer`` over tp."""
    return components_bytes(model_dir, (DIT.in_dtype(dit_dtype_bytes),), tp_degree=tp_degree)


def dit_fits(model_dir: str | Path, *, tp_degree: int,
             dit_dtype_bytes: int = DIT_DTYPE_BYTES) -> tuple[bool, int, int]:
    """``(fits, estimate, limit)``: ``dit_bytes`` against ``FIT_FRACTION`` of one core's HBM."""
    return components_fit(model_dir, (DIT.in_dtype(dit_dtype_bytes),), tp_degree=tp_degree)


__all__ = ["DIT", "DIT_DTYPE_BYTES", "dit_bytes", "dit_fits"]
