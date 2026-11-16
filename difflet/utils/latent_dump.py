"""Opt-in dump of denoise-loop latents and pre-encode frames for offline quality comparison.

Set ``DIFFLET_DUMP_LATENTS_DIR`` to a directory and the denoise loops write their initial and final
latents there as ``<name>.pt`` (CPU tensors, model dtype preserved), and the output writers write
the exact uint8 frames they are about to encode as ``frames.npy`` with ``frames.json`` beside it. An
environment variable rather than a CLI flag so it reaches staged subprocesses unchanged. Unset, it
is a no-op.

Names: models with one latent stream use ``initial`` / ``final``; audio-video models use
``initial_video`` / ``initial_audio`` / ``final_video`` / ``final_audio``. Only rank 0 writes (every
rank of a torchrun stage holds the same tensors), and every file appears atomically (written to a
temporary name, then renamed), so a reader never sees a torn file.

The frames are the ones the file encoder receives, quantized with the writer's own rule, so their
hash is the hash of what the output file was made from, and their PSNR/SSIM is not mixed with the
codec's error. ``frames.json`` records the rule (``truncate``: ``(x*255).astype(uint8)`` as
``export_to_video`` does; ``round``: ``round(x*255)`` as diffusers' PIL conversion does;
``round_half_up``: ``floor(x*255+0.5)``), the shape and the sha256 of the array bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

DUMP_LATENTS_ENV = "DIFFLET_DUMP_LATENTS_DIR"
FRAME_RULES = ("truncate", "round", "round_half_up")


def _dump_dir() -> Path | None:
    out_dir = os.environ.get(DUMP_LATENTS_ENV)
    if not out_dir:
        return None
    if _rank() != 0:
        return None
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def _atomic(path: Path, write) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write(tmp)
    os.replace(tmp, path)


def dump_latents(name: str, tensor: torch.Tensor) -> None:
    path = _dump_dir()
    if path is None:
        return
    _atomic(path / f"{name}.pt", lambda p: torch.save(tensor.detach().cpu(), p))


def frames_uint8(frames: Any, rule: str):
    """Float frames in [0, 1] (any array or tensor) -> uint8 with ``rule``."""
    import numpy as np

    if rule not in FRAME_RULES:
        raise ValueError(f"unknown frame rule {rule!r}; expected one of {FRAME_RULES}")
    x = frames.detach().float().cpu().numpy() if isinstance(frames, torch.Tensor) \
        else np.asarray(frames, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0) * 255.0
    if rule == "round":
        x = np.round(x)
    elif rule == "round_half_up":
        x = np.floor(x + 0.5)
    return x.astype(np.uint8)


def dump_frames(frames: Any, *, rule: str, layout: str, name: str = "frames") -> None:
    """Write the uint8 frames the encoder receives as ``<name>.npy`` [F, H, W, C].

    ``frames`` is either uint8 already (``rule`` names how the writer quantized it) or float in
    [0, 1] (quantized here with ``rule``). ``layout`` is the axis order of ``frames``: ``FHWC``,
    ``HWC`` (one image) or ``CFHW``/``FCHW``/``CHW``; it is converted to ``FHWC``.
    """
    path = _dump_dir()
    if path is None:
        return
    write_frames(path, frames, rule=rule, layout=layout, name=name)


def write_frames(directory: Path, frames: Any, *, rule: str, layout: str,
                 name: str = "frames") -> dict[str, Any]:
    """``dump_frames`` into ``directory`` unconditionally (also used to write references);
    returns the ``<name>.json`` metadata."""
    import numpy as np

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(frames, torch.Tensor):
        array = frames.detach().cpu()
        array = array.numpy() if array.dtype == torch.uint8 else array.float().numpy()
    else:
        array = np.asarray(frames)
    if array.dtype != np.uint8:
        array = frames_uint8(array, rule)
    order = {"FHWC": (0, 1, 2, 3), "CFHW": (1, 2, 3, 0), "FCHW": (0, 2, 3, 1)}
    if layout in ("HWC", "CHW"):
        array = array[None] if layout == "HWC" else np.transpose(array, (1, 2, 0))[None]
    elif layout in order:
        array = np.transpose(array, order[layout])
    else:
        raise ValueError(f"unknown frame layout {layout!r}")
    array = np.ascontiguousarray(array)

    def _save(p: Path) -> None:
        with open(p, "wb") as handle:  # a file object: np.save would append .npy to a path
            np.save(handle, array)

    _atomic(directory / f"{name}.npy", _save)
    meta = {"rule": rule, "layout": "FHWC", "shape": list(array.shape), "dtype": "uint8",
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    _atomic(directory / f"{name}.json", lambda p: p.write_text(json.dumps(meta, indent=2)))
    return meta
