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

The estimate reads only the safetensors headers and counts everything as sharded over tp, so it
is a lower bound: a rejection means the weights clearly cannot fit.
"""

from __future__ import annotations

from pathlib import Path

from difflet.backends.neuron.hbm import FIT_FRACTION, per_core_hbm_bytes, safetensors_numel

#: The DiT runs in bf16 on the device (decision D42, ``NeuronQwenImageTransformerApplication``).
DIT_DTYPE_BYTES = 2


def dit_bytes(model_dir: str | Path, *, tp_degree: int,
              dit_dtype_bytes: int = DIT_DTYPE_BYTES) -> int:
    """Lower bound of one rank's DiT weight bytes: ``<model_dir>/transformer`` over tp."""
    total = safetensors_numel(Path(model_dir) / "transformer") * int(dit_dtype_bytes)
    return total // max(int(tp_degree), 1)


def dit_fits(model_dir: str | Path, *, tp_degree: int,
             dit_dtype_bytes: int = DIT_DTYPE_BYTES) -> tuple[bool, int, int]:
    """``(fits, estimate, limit)``: ``dit_bytes`` against ``FIT_FRACTION`` of one core's HBM."""
    estimate = dit_bytes(model_dir, tp_degree=tp_degree, dit_dtype_bytes=dit_dtype_bytes)
    limit = int(per_core_hbm_bytes()[0] * FIT_FRACTION)
    return estimate <= limit, estimate, limit


__all__ = ["DIT_DTYPE_BYTES", "dit_bytes", "dit_fits"]
