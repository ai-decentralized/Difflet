"""Torch-free request checks for LTX-2 CUDA."""

from __future__ import annotations

import operator
from pathlib import Path

from difflet.models.ltx_2.spec import SPEC

APPLICATION_OPTIONS = frozenset(
    {
        "device",
        "torch_compile",
        "local_files_only",
        "load_text_encoders",
        "load_vae",
        "text_seq_len",
        "frame_rate",
        "generation_profile",
    }
)


def positive_integer(value, name):
    try:
        number = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"LTX-2 {name} must be a positive integer") from exc
    if isinstance(value, bool) or number < 1:
        raise ValueError(f"LTX-2 {name} must be a positive integer")
    return number


def validate_request(*, shape=None, parallel=None, application_kwargs=None, model_id=None):
    if model_id is not None and model_id.rstrip("/") not in SPEC.cli_model_ids:
        if not Path(model_id).expanduser().is_dir():
            raise NotImplementedError("LTX-2 CUDA supports Lightricks/LTX-2 or a local snapshot")
    if parallel is not None:
        for name, label in [
            ("tp_degree", "tensor parallelism"),
            ("cp_degree", "context parallelism"),
            ("dp_degree", "data parallelism"),
        ]:
            if (getattr(parallel, name, None) or 1) != 1:
                raise NotImplementedError(f"LTX-2 {label} is not supported on CUDA; use {name}=1")
        if getattr(parallel, "sp_enabled", False):
            raise NotImplementedError("LTX-2 sequence parallelism is not supported on CUDA")
        if getattr(parallel, "cfg_parallel_enabled", False):
            raise NotImplementedError("LTX-2 CFG parallelism is not implemented on CUDA")
    for key, value in (application_kwargs or {}).items():
        if value is not None and value is not False and key not in APPLICATION_OPTIONS:
            raise NotImplementedError(f"{key} is not supported for LTX-2 on CUDA")
    if shape is None:
        return None
    result = {}
    for name, default in [
        ("height", SPEC.default_height),
        ("width", SPEC.default_width),
        ("num_frames", SPEC.default_num_frames),
    ]:
        value = default if shape.get(name) is None else shape[name]
        result[name] = positive_integer(value, name)
    if result["height"] % 32 or result["width"] % 32:
        raise ValueError("LTX-2 height and width must be positive multiples of 32")
    if (result["num_frames"] - 1) % 8:
        raise ValueError("LTX-2 num_frames must be 8k+1")
    return result
