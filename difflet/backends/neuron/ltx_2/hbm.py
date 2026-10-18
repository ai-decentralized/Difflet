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

from difflet.backends.neuron.hbm import FIT_FRACTION, per_core_hbm_bytes, safetensors_numel

#: The DiT runs in bf16 on the device (decision D42, ``NeuronLtx2TransformerApplication``).
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
