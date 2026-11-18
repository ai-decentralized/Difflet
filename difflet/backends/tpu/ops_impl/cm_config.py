"""Per-shape parameters of the collective matmuls: a tuning table, a cost model,
then the caller's defaults (``collective_matmul``).

The Pallas collective-matmul kernels have parameters that suit some shapes and
not others: whether to run the kernel or the JAX ring at all, the column tile
width, keeping the weights resident in VMEM. XLA picks its tiling per shape; the
kernels had one fixed choice per model. :func:`choose_ag` / :func:`choose_rs`
resolve them per shape, at trace time (they are baked into the compiled
program), in this order:

1. **Tuning table** -- ``DIFFLET_TPU_CM_TUNING=<json>``: measured best configs per
   (chip, shape), written by an offline sweep. Unset: no table.
2. **Cost model** -- ``DIFFLET_TPU_CM_COST_MODEL`` (off by default until it is
   calibrated against such a sweep), its parts comma-separated, ``1``/``all``
   for both:

   * ``kernel`` -- the kernel or the JAX ring from per-hop compute vs transfer;
     the weights resident when re-streaming them per hop costs more than the
     hop's matmuls and they fit. Not calibrated and known wrong for LTX-2: it
     sends the narrow gathers to the ring, which lost end to end (+2.6 ms/step,
     2026-11-13) -- the ring's dynamic-update-slice assembly and its small
     matmuls are not in the model.
   * ``tile`` -- the column tile as torch_tpu sizes its flash-attention tiles
     (``CreateTiling``): the largest that fits the chip's VMEM budget, by the
     kernel's own VMEM estimate, up to a cap (``hw["max_tile"]``, the one
     number a tile sweep calibrates).
3. **The caller's default** -- today's behaviour, unchanged: the model's kernel
   default, ``DIFFLET_TPU_SP_AG_RESIDENT_ROWS``, the kernel's default tile.

Every rank must resolve the same config (a collective kernel whose programs differ
across ranks hangs or corrupts): the choice is a pure function of the shape and
the table file, and ``prepare`` checks that every rank loaded the same table.
The table's hash is part of the ops' names, so a changed table never reuses a
program compiled for another (torch_tpu caches by op name).

``DIFFLET_TPU_CM_LOG_SHAPES=<path>``: append each shape resolved, with its config
and where it came from, as JSON lines -- the input of a tuning sweep.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os

TUNING_ENV = "DIFFLET_TPU_CM_TUNING"
COST_MODEL_ENV = "DIFFLET_TPU_CM_COST_MODEL"
LOG_SHAPES_ENV = "DIFFLET_TPU_CM_LOG_SHAPES"


@dataclasses.dataclass(frozen=True)
class AgShape:
    """An all-gather matmul call: this device's x (batch, rows, k) against
    weights of the given output widths."""
    chip: str
    batch: int
    rows: int
    k: int
    widths: tuple
    dtype: str
    ring: int


@dataclasses.dataclass(frozen=True)
class AgConfig:
    use_kernel: bool
    tile_width: int = 256        # the kernel's column tile limit
    resident: bool = False       # weights loaded into VMEM once instead of per hop


@dataclasses.dataclass(frozen=True)
class RsShape:
    """A matmul reduce-scatter call: x_i (batch, rows, k_i) against weights (n, k_i)."""
    chip: str
    batch: int
    rows: int
    ks: tuple
    n: int
    dtype: str
    ring: int


@dataclasses.dataclass(frozen=True)
class RsConfig:
    use_kernel: bool
    tile_width: int = 256


_MODEL_PARTS = ("kernel", "tile")


def _model_parts() -> frozenset:
    """The cost model's parts switched on by ``COST_MODEL_ENV``."""
    raw = os.environ.get(COST_MODEL_ENV, "").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return frozenset()
    if raw in {"1", "true", "yes", "on", "all"}:
        return frozenset(_MODEL_PARTS)
    parts = frozenset(p.strip() for p in raw.split(",") if p.strip())
    unknown = parts - set(_MODEL_PARTS)
    if unknown:
        raise ValueError(f"{COST_MODEL_ENV}: unknown parts {sorted(unknown)}; "
                         f"use 1/all or a comma-separated subset of {list(_MODEL_PARTS)}")
    return parts


@functools.lru_cache(maxsize=None)
def _load_table(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    table = {}
    for kind, shape_cls, cfg_cls in (("ag", AgShape, AgConfig), ("rs", RsShape, RsConfig)):
        for entry in raw.get(kind, []):
            shape = {k: tuple(v) if isinstance(v, list) else v for k, v in entry["shape"].items()}
            table[(kind, shape_cls(**shape))] = cfg_cls(**entry["config"])
    return {"entries": table, "hw": raw.get("hw", {})}


def _table() -> dict:
    path = os.environ.get(TUNING_ENV, "").strip()
    return _load_table(path) if path else {"entries": {}, "hw": {}}


def table_tag() -> str:
    """8 hex of the tuning table's contents and the cost-model switch ("" when
    neither is in use): part of the ops' names and cache keys."""
    path = os.environ.get(TUNING_ENV, "").strip()
    parts = _model_parts()
    if not path and not parts:
        return ""
    digest = hashlib.sha256()
    if path:
        with open(path, "rb") as f:
            digest.update(f.read())
    digest.update(",".join(sorted(parts)).encode())
    return digest.hexdigest()[:8]


def _log(kind: str, shape, config, source: str) -> None:
    path = os.environ.get(LOG_SHAPES_ENV, "").strip()
    if not path:
        return
    record = {"kind": kind, "shape": dataclasses.asdict(shape),
              "config": dataclasses.asdict(config), "source": source}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


#: Hardware constants of the cost model (v5e; link numbers measured 2026-11-03).
#: A calibration sweep overrides them through the table's "hw" section.
_HW_DEFAULTS = {
    "flops": 1.97e14,           # bf16 peak per chip
    "link_bw": 4.67e10,         # bytes/s per link and direction
    "link_alpha": 2.5e-6,       # s per hop
    "hbm_bw": 8.19e11,          # bytes/s
    "kernel_margin": 1.3,       # kernel only when a hop's compute is this x its transfer
    "max_tile": 512,            # column tile cap (uncalibrated: the sweep sets it)
}

#: VMEM per TensorCore by ``device_kind`` prefix (longest match wins). The kernels
#: plan for 3/4 of it (v5e: 96 MiB, the kernels' VMEM_BUDGET_BYTES); torch_tpu's
#: flash attention plans for half, its estimate leaving out less.
_VMEM_CAPACITY = {
    "TPU v4": 32 * 2**20,
    "TPU v5 lite": 128 * 2**20,
    "TPU v5": 95 * 2**20,
    "TPU v6 lite": 128 * 2**20,
}


def vmem_budget(chip: str, hw: dict | None = None) -> int:
    """VMEM bytes the kernels may plan for on ``chip`` (``hw["vmem_bytes"]`` wins)."""
    if hw and "vmem_bytes" in hw:
        return int(hw["vmem_bytes"])
    from difflet.backends.tpu.ops_impl import collective_matmul_pallas as kernel

    match = max((k for k in _VMEM_CAPACITY if chip.startswith(k)), key=len, default=None)
    return _VMEM_CAPACITY[match] * 3 // 4 if match else kernel.VMEM_BUDGET_BYTES


def _itemsize(dtype: str) -> int:
    return 2 if dtype in ("bfloat16", "bf16", "float16") else 4


def _ag_choose_kernel(shape: AgShape, default: AgConfig, p: dict) -> AgConfig:
    """The kernel when a hop's matmuls hide its transfer, else the JAX ring; the
    weights resident when re-streaming them per hop costs more than the hop's
    matmuls and they fit."""
    from difflet.backends.tpu.ops_impl import collective_matmul_pallas as kernel

    half = shape.rows // 2
    item = _itemsize(shape.dtype)
    compute_hop = 2.0 * (2 * half) * shape.k * sum(shape.widths) / p["flops"]
    transfer_hop = p["link_alpha"] + half * shape.k * item / p["link_bw"]
    if compute_hop < p["kernel_margin"] * transfer_hop:
        return dataclasses.replace(default, use_kernel=False)
    restream = sum(shape.widths) * shape.k * item / p["hbm_bw"] > compute_hop
    fits = kernel.all_gather_vmem_bytes((shape.batch, shape.rows, shape.k), shape.widths,
                                        item, default.tile_width, True) <= p["vmem_budget"]
    return dataclasses.replace(default, use_kernel=True, resident=bool(restream and fits))


def _ag_choose_tile(shape: AgShape, config: AgConfig, p: dict) -> AgConfig:
    """The largest column tile up to ``max_tile`` that divides every width and fits
    the VMEM budget (torch_tpu's ``CreateTiling``: start large, shrink until the
    estimate fits); the caller's tile when none does."""
    from difflet.backends.tpu.ops_impl import collective_matmul_pallas as kernel

    x_shape = (shape.batch, shape.rows, shape.k)
    for tile in kernel.column_tiles(shape.widths, int(p["max_tile"])):
        if kernel.all_gather_vmem_bytes(x_shape, shape.widths, _itemsize(shape.dtype), tile,
                                        config.resident) <= p["vmem_budget"]:
            return dataclasses.replace(config, tile_width=tile)
    return config


def _ag_cost_model(shape: AgShape, default: AgConfig, hw: dict, parts=frozenset(_MODEL_PARTS)
                   ) -> AgConfig:
    p = {**_HW_DEFAULTS, **hw, "vmem_budget": vmem_budget(shape.chip, hw)}
    config = _ag_choose_kernel(shape, default, p) if "kernel" in parts else default
    if "tile" in parts and config.use_kernel:
        config = _ag_choose_tile(shape, config, p)
    return config


def choose_ag(shape: AgShape, default: AgConfig) -> tuple[AgConfig, str]:
    """(config, source): the table's, the cost model's, or ``default``."""
    table = _table()
    hit = table["entries"].get(("ag", shape))
    if hit is not None:
        config, source = hit, "table"
    elif _model_parts():
        config, source = _ag_cost_model(shape, default, table["hw"], _model_parts()), "model"
    else:
        config, source = default, "default"
    _log("ag", shape, config, source)
    return config, source


def choose_rs(shape: RsShape, default: RsConfig) -> tuple[RsConfig, str]:
    """(config, source): the table's or ``default`` (no RS cost model yet: the RS
    kernel lost to the ring on every shape but the comm-bound ones, 2026-11-11/13)."""
    hit = _table()["entries"].get(("rs", shape))
    config, source = (hit, "table") if hit is not None else (default, "default")
    _log("rs", shape, config, source)
    return config, source


__all__ = [
    "AgConfig", "AgShape", "COST_MODEL_ENV", "LOG_SHAPES_ENV", "RsConfig", "RsShape",
    "TUNING_ENV", "choose_ag", "choose_rs", "table_tag", "vmem_budget",
]
