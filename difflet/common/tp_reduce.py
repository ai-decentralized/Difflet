"""Precision of the tensor-parallel all-reduce.

NxD's ``RowParallelLinear`` all-reduces in ``torch.float32`` unless told
otherwise. Difflet reduces in bf16 by default (``--tp-reduce-dtype``); fp32
stays selectable. Each model keys its compile artifacts with
:func:`tp_reduce_cache_fields` against the dtype it used before the flag
existed, so every pre-flag artifact keeps its cache key.
"""

from __future__ import annotations

import torch

TP_REDUCE_DTYPES: dict[str, torch.dtype] = {"bf16": torch.bfloat16, "fp32": torch.float32}


def parse_tp_reduce_dtype(value: str | None) -> torch.dtype:
    """Map a ``--tp-reduce-dtype`` value (``None`` = default bf16) to a dtype."""
    if value is None:
        return torch.bfloat16
    try:
        return TP_REDUCE_DTYPES[value]
    except KeyError:
        raise ValueError(
            f"tp_reduce_dtype must be one of {', '.join(TP_REDUCE_DTYPES)}; got {value!r}"
        ) from None


def tp_reduce_cache_fields(
    requested: torch.dtype, *, legacy: torch.dtype | None
) -> dict[str, str]:
    """Cache-key fields for a component's all-reduce dtype.

    ``legacy`` is the dtype the component reduced in before the flag existed
    (``None`` when one artifact mixed dtypes). Requesting the legacy dtype adds
    nothing, so the pre-flag key is unchanged.
    """
    if legacy is not None and requested == legacy:
        return {}
    return {"tp_reduce_dtype": str(requested).removeprefix("torch.")}
