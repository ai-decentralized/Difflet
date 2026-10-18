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


#: The device text encoder's default dtype (bf16, the host path's).
TEXT_ENCODER_DTYPE_BYTES = 2
#: Tensors of ``<model>/text_encoder`` (transformers' Gemma-3 checkpoint) the device encoder
#: never loads.
_TEXT_ENCODER_SKIPPED = ("vision_tower.", "multi_modal_projector.", "language_model.lm_head.")
_TEXT_ENCODER_INDEX = "model.safetensors.index.json"


def text_encoder_lm_numel(model_dir: str | Path) -> int:
    """Element count of Gemma-3's language model in ``<model_dir>/text_encoder`` (no vision
    tower, no projector): what ``NeuronLtx2TextEncoderApplication`` loads. Only the
    transformers checkpoint's shards (``model-*.safetensors``) are read: the directory may also
    list a diffusers index of another checkpoint."""
    import json
    import math
    import struct

    directory = Path(model_dir) / "text_encoder"
    index = directory / _TEXT_ENCODER_INDEX
    files = (sorted(set(json.loads(index.read_text())["weight_map"].values()))
             if index.is_file() else [p.name for p in sorted(directory.glob("*.safetensors"))])
    total = 0
    for name in files:
        with (directory / name).open("rb") as handle:
            (header_len,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(header_len))
        for key, info in header.items():
            if key != "__metadata__" and not key.startswith(_TEXT_ENCODER_SKIPPED):
                total += math.prod(info["shape"]) if info["shape"] else 1
    return total


def device_text_encoder_bytes(model_dir: str | Path, *, tp_degree: int,
                              encoder_dtype_bytes: int = TEXT_ENCODER_DTYPE_BYTES,
                              dit_dtype_bytes: int = DIT_DTYPE_BYTES) -> int:
    """Lower bound of one rank's weight bytes with Gemma-3's language model on the device beside
    the DiT, both sharded over ``tp_degree``."""
    total = text_encoder_lm_numel(model_dir) * int(encoder_dtype_bytes)
    total += safetensors_numel(Path(model_dir) / "transformer") * int(dit_dtype_bytes)
    return total // max(int(tp_degree), 1)


def device_text_encoder_fits(model_dir: str | Path, *, tp_degree: int,
                             encoder_dtype_bytes: int = TEXT_ENCODER_DTYPE_BYTES,
                             dit_dtype_bytes: int = DIT_DTYPE_BYTES) -> tuple[bool, int, int]:
    """``(fits, estimate, limit)``: ``device_text_encoder_bytes`` against ``FIT_FRACTION`` of
    one core's HBM."""
    estimate = device_text_encoder_bytes(model_dir, tp_degree=tp_degree,
                                         encoder_dtype_bytes=encoder_dtype_bytes,
                                         dit_dtype_bytes=dit_dtype_bytes)
    limit = int(per_core_hbm_bytes()[0] * FIT_FRACTION)
    return estimate <= limit, estimate, limit


__all__ = [
    "DIT_DTYPE_BYTES",
    "TEXT_ENCODER_DTYPE_BYTES",
    "device_text_encoder_bytes",
    "device_text_encoder_fits",
    "dit_bytes",
    "dit_fits",
    "text_encoder_lm_numel",
]
