# Orthogonal Parallel Mesh (dp/cfg/cp/tp) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor Difflet's process-group layer into four orthogonal named axes {dp, cfg, cp, tp} with per-axis subgroups, and move every CFG/CP collective off NxD's parasitic `dp_group` onto its own axis group — groups only, no DP feature yet.

**Architecture:** A pure-math `MeshSpec` (rank ↔ (dp,cfg,cp,tp) mapping, axis subgroup meshes) lives in `difflet/pipeline/parallel_mesh.py`; a Trainium `ProcessGroupManager` singleton (`difflet/backends/trainium/core/parallel_mesh.py`) materializes one `torch.distributed` group per non-trivial axis with `xla_pg_options` meshes (same pattern as the existing NxDI-fork `attention_process_groups.py`). Model code reaches groups only through `difflet.ops` (`get_cfg_group`/`get_cp_group`/`get_cfg_rank_spmd`/`get_cp_rank_spmd`/`init_parallel_mesh`); `get_data_parallel_group`/`get_dp_rank_spmd` are removed from the ops surface entirely.

**Tech Stack:** Python 3.12, torch 2.9 + torch_xla/NxD in `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`, pytest (unit tests are CPU-only via `DIFFLET_BACKEND=cpu`), Trainium trn2.3xlarge (4 logical cores) for device regression.

## Global Constraints

- Rank layout (tp innermost, dp outermost): `rank = tp + T*(cp + C*(cfg + G*dp))`.
- Bit-identity: for `(dp=1,cfg=2,tp=2)` and `(dp=1,cp=2,tp=2)` the axis-group meshes MUST equal NxD's legacy dp-group meshes (`[[j, j+T, …]]` column groups) so device outputs are bit-identical to pre-refactor (baseline commit `93fac4c`).
- CFG×CP mutual exclusion in `DiffletParallelConfig` and model validation STAYS (user decision). The mesh may *express* combined specs; pipelines keep rejecting them.
- Per-model CFG policy unchanged: FLUX.1-dev / HunyuanVideo / HunyuanVideo-1.5 / Qwen-Image stay blocked from CFG-parallel via existing `_DISTILLED_MODELS` + entry guards (user decision: HYV-1.5 and Qwen do NOT get cfg=2). Flux's dormant Python-API CFG path stays in code (user decision: "leave it") but migrates to `cfg_group`.
- `dp_group` must carry NO per-layer/per-step collective; nothing may reference `get_data_parallel_group`/`get_dp_rank_spmd` outside the NxDI-fork files (`backends/trainium/core/**`, `backends/trainium/utils/distributed.py`) and the manager itself.
- NxDI-fork files (`attention_process_groups.py`, `attention_base.py`, `utils/distributed.py`) are NOT modified — their intra-TP "dp/cp" machinery is unrelated to the pipeline axes.
- Compile-cache keys: `to_cache_dict` stays additive-only (omit `dp_degree` at 1) so default configs keep byte-identical cache keys.
- Unit tests: `cd /home/ubuntu/Difflet/.claude/worktrees/restruct-group && PYTHONPATH=. /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/pytest tests/unit -q`.
- **After each task: `git add` the task's files, commit with a conventional message, and `git push origin worktree-restruct-group`.**
- Final report saved under `docs/` (Task 13).

---

### Task 1: MeshSpec pure math

**Files:**
- Create: `difflet/pipeline/parallel_mesh.py`
- Test: `tests/unit/pipeline/test_parallel_mesh.py`

**Interfaces:**
- Produces: `MeshSpec(dp=1, cfg=1, cp=1, tp=1)` frozen dataclass with `world_size` property, `coords_of(rank) -> MeshCoords(dp,cfg,cp,tp)` (NamedTuple), `rank_of(*, dp=0, cfg=0, cp=0, tp=0) -> int`, `axis_size(axis) -> int`, `axis_stride(axis) -> int`, `axis_rank(rank, axis) -> int`, `axis_groups(axis) -> list[list[int]]`. `AXES = ("dp", "cfg", "cp", "tp")`.

- [ ] **Step 1: Write failing tests** — `tests/unit/pipeline/test_parallel_mesh.py`:

```python
"""Rank <-> (dp, cfg, cp, tp) mapping and axis-subgroup math for MeshSpec."""

import pytest

from difflet.pipeline.parallel_mesh import AXES, MeshCoords, MeshSpec


def test_axes_order_outer_to_inner():
    assert AXES == ("dp", "cfg", "cp", "tp")


@pytest.mark.parametrize("axis", AXES)
def test_axis_must_be_positive(axis):
    with pytest.raises(ValueError):
        MeshSpec(**{axis: 0})


def test_world_size_is_product():
    assert MeshSpec(dp=2, cfg=2, cp=3, tp=4).world_size == 48


@pytest.mark.parametrize(
    "spec",
    [
        MeshSpec(dp=1, cfg=2, cp=2, tp=2),
        MeshSpec(dp=1, cfg=1, cp=4, tp=2),
        MeshSpec(dp=4, cfg=1, cp=1, tp=2),
        MeshSpec(dp=2, cfg=2, cp=1, tp=2),
    ],
)
def test_rank_roundtrip_all_required_combos(spec):
    for rank in range(spec.world_size):
        c = spec.coords_of(rank)
        assert spec.rank_of(dp=c.dp, cfg=c.cfg, cp=c.cp, tp=c.tp) == rank


def test_rank_formula_tp_innermost_dp_outermost():
    spec = MeshSpec(dp=2, cfg=2, cp=2, tp=2)
    # rank = tp + T*(cp + C*(cfg + G*dp))
    assert spec.rank_of(dp=0, cfg=0, cp=0, tp=1) == 1
    assert spec.rank_of(dp=0, cfg=0, cp=1, tp=0) == 2
    assert spec.rank_of(dp=0, cfg=1, cp=0, tp=0) == 4
    assert spec.rank_of(dp=1, cfg=0, cp=0, tp=0) == 8
    assert spec.coords_of(13) == MeshCoords(dp=1, cfg=1, cp=0, tp=1)


def test_coords_and_rank_validate_ranges():
    spec = MeshSpec(cfg=2, tp=2)
    with pytest.raises(ValueError):
        spec.coords_of(4)
    with pytest.raises(ValueError):
        spec.coords_of(-1)
    with pytest.raises(ValueError):
        spec.rank_of(cfg=2)


def test_axis_groups_partition_and_vary_only_that_axis():
    spec = MeshSpec(dp=2, cfg=2, cp=2, tp=2)
    for axis in AXES:
        groups = spec.axis_groups(axis)
        flat = sorted(r for g in groups for r in g)
        assert flat == list(range(spec.world_size))          # exact partition
        for group in groups:
            assert len(group) == spec.axis_size(axis)
            base = spec.coords_of(group[0])
            for i, rank in enumerate(group):
                c = spec.coords_of(rank)
                assert getattr(c, axis) == i                 # increasing axis coord
                for other in AXES:
                    if other != axis:
                        assert getattr(c, other) == getattr(base, other)


def test_legacy_cfg2_mesh_matches_nxd_dp_columns():
    # (dp=1, cfg=2, cp=1, tp=T): cfg groups must be [[j, j+T] for j in range(T)]
    spec = MeshSpec(cfg=2, tp=4)
    assert spec.axis_groups("cfg") == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_legacy_cp4_mesh_matches_nxd_dp_columns():
    # (dp=1, cfg=1, cp=4, tp=2): cp groups must be [[j, j+T, j+2T, j+3T]]
    spec = MeshSpec(cp=4, tp=2)
    assert spec.axis_groups("cp") == [[0, 2, 4, 6], [1, 3, 5, 7]]


def test_axis_rank_matches_coords():
    spec = MeshSpec(dp=1, cfg=2, cp=2, tp=2)
    for rank in range(spec.world_size):
        c = spec.coords_of(rank)
        for axis in AXES:
            assert spec.axis_rank(rank, axis) == getattr(c, axis)


def test_unknown_axis_rejected():
    with pytest.raises(ValueError):
        MeshSpec().axis_groups("pp")
```

- [ ] **Step 2: Run to verify failure** — `PYTHONPATH=. pytest tests/unit/pipeline/test_parallel_mesh.py -q` → FAIL: `ModuleNotFoundError: difflet.pipeline.parallel_mesh`.

- [ ] **Step 3: Implement** — `difflet/pipeline/parallel_mesh.py`:

```python
"""Orthogonal {dp, cfg, cp, tp} device-mesh math (backend-agnostic).

Rank layout (tp innermost, dp outermost):

    rank = tp + T*(cp + C*(cfg + G*dp))

An axis's subgroup is the set of ranks that differ only in that axis's
coordinate. For every configuration expressible pre-refactor (exactly one of
cfg/cp non-trivial, dp=1) the non-trivial axis's groups coincide with NxD's
legacy data-parallel column groups ``[[j, j+T, ...] for j in range(T)]``, which
is what makes the migration bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

AXES = ("dp", "cfg", "cp", "tp")  # outermost -> innermost


class MeshCoords(NamedTuple):
    dp: int
    cfg: int
    cp: int
    tp: int


@dataclass(frozen=True)
class MeshSpec:
    """Sizes of the four orthogonal parallel axes."""

    dp: int = 1
    cfg: int = 1
    cp: int = 1
    tp: int = 1

    def __post_init__(self) -> None:
        for axis in AXES:
            size = getattr(self, axis)
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ValueError(f"{axis} must be an int >= 1, got {size!r}")

    @property
    def world_size(self) -> int:
        return self.dp * self.cfg * self.cp * self.tp

    def axis_size(self, axis: str) -> int:
        if axis not in AXES:
            raise ValueError(f"unknown axis {axis!r}; expected one of {AXES}")
        return getattr(self, axis)

    def axis_stride(self, axis: str) -> int:
        strides = {
            "tp": 1,
            "cp": self.tp,
            "cfg": self.tp * self.cp,
            "dp": self.tp * self.cp * self.cfg,
        }
        if axis not in strides:
            raise ValueError(f"unknown axis {axis!r}; expected one of {AXES}")
        return strides[axis]

    def coords_of(self, rank: int) -> MeshCoords:
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank {rank} out of range [0, {self.world_size})")
        return MeshCoords(
            dp=rank // (self.tp * self.cp * self.cfg),
            cfg=(rank // (self.tp * self.cp)) % self.cfg,
            cp=(rank // self.tp) % self.cp,
            tp=rank % self.tp,
        )

    def rank_of(self, *, dp: int = 0, cfg: int = 0, cp: int = 0, tp: int = 0) -> int:
        coords = {"dp": dp, "cfg": cfg, "cp": cp, "tp": tp}
        for axis, coord in coords.items():
            if not 0 <= coord < self.axis_size(axis):
                raise ValueError(
                    f"{axis} coordinate {coord} out of range [0, {self.axis_size(axis)})"
                )
        return tp + self.tp * (cp + self.cp * (cfg + self.cfg * dp))

    def axis_rank(self, rank: int, axis: str) -> int:
        return getattr(self.coords_of(rank), self.axis_size(axis) and axis)

    def axis_groups(self, axis: str) -> list[list[int]]:
        """Full axis mesh: one group per combination of the other coordinates.

        Group members are ordered by increasing axis coordinate; groups are
        ordered by their first (axis-coordinate-0) rank.
        """
        size = self.axis_size(axis)
        stride = self.axis_stride(axis)
        groups = []
        for base in range(self.world_size):
            if (base // stride) % size == 0:
                groups.append([base + i * stride for i in range(size)])
        return groups
```

Note: `axis_rank` must validate the axis before `coords_of` — implement as:

```python
    def axis_rank(self, rank: int, axis: str) -> int:
        self.axis_size(axis)  # validates axis name
        return getattr(self.coords_of(rank), axis)
```

- [ ] **Step 4: Run tests** — `PYTHONPATH=. pytest tests/unit/pipeline/test_parallel_mesh.py -q` → all PASS.
- [ ] **Step 5: Commit & push** — `git add difflet/pipeline/parallel_mesh.py tests/unit/pipeline/test_parallel_mesh.py && git commit -m "feat(mesh): MeshSpec rank<->(dp,cfg,cp,tp) math with per-axis subgroup meshes" && git push origin worktree-restruct-group`.

---

### Task 2: DiffletParallelConfig gains dp_degree + mesh_spec

**Files:**
- Modify: `difflet/pipeline/parallel_config.py`
- Test: `tests/unit/pipeline/test_parallel_config_mesh.py`

**Interfaces:**
- Produces: `DiffletParallelConfig.dp_degree: int = 1` (validated ≥1), `DiffletParallelConfig.mesh_spec -> MeshSpec`, `world_size` includes `dp_degree`, `to_cache_dict()` omits `dp_degree` when 1.

- [ ] **Step 1: Write failing tests** — `tests/unit/pipeline/test_parallel_config_mesh.py`:

```python
"""dp_degree axis + mesh_spec derivation on DiffletParallelConfig."""

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.pipeline.parallel_mesh import MeshSpec


def test_default_dp_degree_is_one_and_cache_key_unchanged():
    cfg = DiffletParallelConfig(tp_degree=2)
    assert cfg.dp_degree == 1
    assert "dp_degree" not in cfg.to_cache_dict()


def test_dp_degree_in_cache_dict_when_set():
    cfg = DiffletParallelConfig(tp_degree=2, dp_degree=4)
    assert cfg.to_cache_dict()["dp_degree"] == 4


def test_dp_degree_validation():
    with pytest.raises(ValueError):
        DiffletParallelConfig(dp_degree=0)


def test_world_size_includes_dp():
    assert DiffletParallelConfig(tp_degree=2, dp_degree=4).world_size == 8


def test_mesh_spec_cfg_parallel():
    cfg = DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True)
    assert cfg.mesh_spec == MeshSpec(dp=1, cfg=2, cp=1, tp=2)
    assert cfg.mesh_spec.world_size == cfg.world_size


def test_mesh_spec_cp():
    cfg = DiffletParallelConfig(tp_degree=2, cp_degree=2)
    assert cfg.mesh_spec == MeshSpec(dp=1, cfg=1, cp=2, tp=2)
    assert cfg.mesh_spec.world_size == cfg.world_size


def test_cfg_cp_still_mutually_exclusive():
    with pytest.raises(ValueError):
        DiffletParallelConfig(cp_degree=2, cfg_parallel_enabled=True)
```

- [ ] **Step 2: Run to verify failure** → FAIL (`dp_degree` unexpected kwarg).
- [ ] **Step 3: Implement** in `difflet/pipeline/parallel_config.py`: add `from difflet.pipeline.parallel_mesh import MeshSpec`; add field `dp_degree: int = 1` after `sp_enabled`; in `__post_init__` add:

```python
        if self.dp_degree < 1:
            raise ValueError("dp_degree must be >= 1")
```

Change `world_size`:

```python
    @property
    def world_size(self) -> int:
        # Megatron-style SP reuses the tensor-parallel group; it does not add a
        # new world-size axis, so it never appears in this product.
        return self.mesh_spec.world_size

    @property
    def mesh_spec(self) -> MeshSpec:
        return MeshSpec(
            dp=self.dp_degree,
            cfg=2 if self.cfg_parallel_enabled else 1,
            cp=self.cp_degree,
            tp=self.tp_degree,
        )
```

In `to_cache_dict`, after the `sp_enabled` pop:

```python
        if self.dp_degree == 1:
            d.pop("dp_degree")
```

- [ ] **Step 4: Run** — `PYTHONPATH=. pytest tests/unit/pipeline -q` → all PASS (existing parallel_config suites must stay green).
- [ ] **Step 5: Commit & push** — `git commit -m "feat(mesh): dp_degree axis + mesh_spec on DiffletParallelConfig (cache-key additive)"` then push.

---

### Task 3: Trainium ProcessGroupManager

**Files:**
- Create: `difflet/backends/trainium/core/parallel_mesh.py`
- Test: `tests/unit/backends/test_trainium_parallel_mesh.py`

**Interfaces:**
- Consumes: `MeshSpec` from Task 1.
- Produces (all importable from the module): `mesh_spec_from_config(config) -> MeshSpec`, `init_parallel_mesh(config) -> None` (idempotent for equal specs; raises `RuntimeError` on conflicting re-init), `get_mesh_spec() -> MeshSpec`, `get_cfg_group()`, `get_cp_group()`, `get_dp_group()` (raise `AssertionError` when uninitialized or axis trivial), `get_cp_mesh() -> list[list[int]]`, `get_cfg_rank_spmd(global_rank) -> Tensor(int32)`, `get_cp_rank_spmd(global_rank) -> Tensor(int32)`, `destroy_parallel_mesh()` (test-only reset of module state).

- [ ] **Step 1: Write failing tests** — `tests/unit/backends/test_trainium_parallel_mesh.py`. Uses the neuron venv's torch; stubs NxD's `parallel_state` and `torch.distributed.new_group` via monkeypatch:

```python
"""ProcessGroupManager: spec derivation, per-axis group construction, SPMD ranks."""

from types import SimpleNamespace

import pytest
import torch

import difflet.backends.trainium.core.parallel_mesh as pm
from difflet.pipeline.parallel_mesh import MeshSpec


class _FakeWorldGroup:
    def __init__(self, n):
        self._n = n

    def size(self):
        return self._n


@pytest.fixture(autouse=True)
def _reset_mesh(monkeypatch):
    pm.destroy_parallel_mesh()
    created = []

    def fake_new_group(ranks, pg_options=None):
        created.append((list(ranks), pg_options))
        return SimpleNamespace(ranks=list(ranks), pg_options=pg_options)

    monkeypatch.setattr(pm.torch.distributed, "new_group", fake_new_group)
    yield created
    pm.destroy_parallel_mesh()


def _patch_world(monkeypatch, tp, world):
    monkeypatch.setattr(pm, "_nxd_tp_size", lambda: tp)
    monkeypatch.setattr(pm, "_nxd_world_group", lambda: _FakeWorldGroup(world))


def _config(**kwargs):
    return SimpleNamespace(**kwargs)


def test_spec_cfg_parallel(monkeypatch):
    _patch_world(monkeypatch, tp=2, world=4)
    spec = pm.mesh_spec_from_config(_config(cfg_parallel_enabled=True))
    assert spec == MeshSpec(dp=1, cfg=2, cp=1, tp=2)


def test_spec_cp_inferred_from_world(monkeypatch):
    _patch_world(monkeypatch, tp=2, world=8)
    spec = pm.mesh_spec_from_config(_config(context_parallel_enabled=True))
    assert spec == MeshSpec(dp=1, cfg=1, cp=4, tp=2)


def test_spec_product_must_match_world(monkeypatch):
    _patch_world(monkeypatch, tp=2, world=8)  # no cp/cfg flag but world > tp
    with pytest.raises(ValueError):
        pm.mesh_spec_from_config(_config())


def test_init_builds_only_nontrivial_axes(monkeypatch, _reset_mesh=None):
    _patch_world(monkeypatch, tp=2, world=4)
    pm.init_parallel_mesh(_config(cfg_parallel_enabled=True))
    assert pm.get_mesh_spec() == MeshSpec(cfg=2, tp=2)
    assert pm.get_cfg_group() is not None
    with pytest.raises(AssertionError):
        pm.get_cp_group()
    with pytest.raises(AssertionError):
        pm.get_dp_group()


def test_cfg_group_mesh_matches_legacy_nxd_dp_mesh(monkeypatch, _reset_mesh):
    _patch_world(monkeypatch, tp=2, world=4)
    pm.init_parallel_mesh(_config(cfg_parallel_enabled=True))
    (ranks, pg_options), = _reset_mesh
    assert ranks == [0, 2]
    assert pg_options == {"xla_pg_options": {"mesh": [[0, 2], [1, 3]]}}


def test_init_idempotent_and_conflict_raises(monkeypatch):
    _patch_world(monkeypatch, tp=2, world=4)
    pm.init_parallel_mesh(_config(cfg_parallel_enabled=True))
    pm.init_parallel_mesh(_config(cfg_parallel_enabled=True))  # same spec: no-op
    with pytest.raises(RuntimeError):
        pm.init_parallel_mesh(_config(context_parallel_enabled=True))


def test_cp_mesh_accessor(monkeypatch):
    _patch_world(monkeypatch, tp=2, world=8)
    pm.init_parallel_mesh(_config(context_parallel_enabled=True))
    assert pm.get_cp_mesh() == [[0, 2, 4, 6], [1, 3, 5, 7]]


def test_spmd_rank_helpers_match_mesh_coords(monkeypatch):
    _patch_world(monkeypatch, tp=2, world=4)
    pm.init_parallel_mesh(_config(cfg_parallel_enabled=True))
    ranks = torch.arange(4)
    cfg_ranks = pm.get_cfg_rank_spmd(ranks)
    assert cfg_ranks.dtype == torch.int32
    assert cfg_ranks.tolist() == [0, 0, 1, 1]
    assert pm.get_cp_rank_spmd(ranks).tolist() == [0, 0, 0, 0]


def test_spmd_cp_rank_legacy_equivalence(monkeypatch):
    # (cfg=1, cp=4, tp=2): cp_rank must equal legacy rank // tp
    _patch_world(monkeypatch, tp=2, world=8)
    pm.init_parallel_mesh(_config(context_parallel_enabled=True))
    ranks = torch.arange(8)
    assert pm.get_cp_rank_spmd(ranks).tolist() == [r // 2 for r in range(8)]
```

- [ ] **Step 2: Run to verify failure** → FAIL: module not found.
- [ ] **Step 3: Implement** — `difflet/backends/trainium/core/parallel_mesh.py`:

```python
"""Orthogonal {dp, cfg, cp, tp} process groups for the Trainium backend.

One singleton per process (same pattern as the NxDI-fork
``attention_process_groups.py``). ``init_parallel_mesh(config)`` derives the
MeshSpec from the model config plus NxD's already-initialized tp/world sizes,
then builds ONE torch.distributed group per NON-trivial axis with the full axis
mesh in ``xla_pg_options`` (that mesh becomes the collective's replica groups
under SPMD tracing). Trivial axes build no group at all — a guidance-distilled
model (cfg=1) never constructs a cfg group.

TP is NOT built here: with tp innermost the mesh's tp axis coincides with
NxD's tensor_model_parallel_group, which every TP layer already uses.

The dp axis group exists only when dp > 1 and intentionally has NO consumer in
any per-layer/per-step code path; it is reserved for the future DP feature
(replica-level routing / load-time weight broadcast).
"""

from __future__ import annotations

import torch
import torch.distributed

from difflet.pipeline.parallel_mesh import MeshSpec

_MESH_SPEC: MeshSpec | None = None
_AXIS_GROUPS: dict[str, object] = {}


def _nxd_tp_size() -> int:
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_size,
    )

    return int(get_tensor_model_parallel_size())


def _nxd_world_group():
    from neuronx_distributed.parallel_layers.parallel_state import get_world_group

    return get_world_group()


def mesh_spec_from_config(config) -> MeshSpec:
    """Derive the mesh from model-config flags + NxD's tp/world sizes.

    cp is inferred as ``world / (tp * cfg * dp)`` when context parallelism is
    enabled, because the backbone configs carry only the boolean flag (the
    explicit degree never reached the backend pre-refactor either).
    """
    tp = _nxd_tp_size()
    world = int(_nxd_world_group().size())
    cfg = 2 if bool(getattr(config, "cfg_parallel_enabled", False)) else 1
    dp = int(getattr(config, "dp_degree", 1))
    denom = tp * cfg * dp
    if world % denom != 0:
        raise ValueError(
            f"world_size {world} is not divisible by tp*cfg*dp = {denom} "
            f"(tp={tp}, cfg={cfg}, dp={dp})"
        )
    cp = world // denom if bool(getattr(config, "context_parallel_enabled", False)) else 1
    spec = MeshSpec(dp=dp, cfg=cfg, cp=cp, tp=tp)
    if spec.world_size != world:
        raise ValueError(
            f"mesh spec {spec} product {spec.world_size} != world_size {world}"
        )
    return spec


def init_parallel_mesh(config) -> None:
    global _MESH_SPEC
    spec = mesh_spec_from_config(config)
    if _MESH_SPEC is not None:
        if spec != _MESH_SPEC:
            raise RuntimeError(
                f"parallel mesh already initialized with {_MESH_SPEC}; "
                f"cannot re-initialize with {spec}"
            )
        return
    _MESH_SPEC = spec
    for axis in ("cfg", "cp", "dp"):
        if spec.axis_size(axis) > 1:
            mesh = spec.axis_groups(axis)
            _AXIS_GROUPS[axis] = torch.distributed.new_group(
                mesh[0], pg_options={"xla_pg_options": {"mesh": mesh}}
            )


def _require_spec() -> MeshSpec:
    assert _MESH_SPEC is not None, "parallel mesh is not initialized"
    return _MESH_SPEC


def get_mesh_spec() -> MeshSpec:
    return _require_spec()


def _axis_group(axis: str):
    spec = _require_spec()
    assert axis in _AXIS_GROUPS, (
        f"{axis} axis is trivial ({axis}={spec.axis_size(axis)}); no {axis} group exists"
    )
    return _AXIS_GROUPS[axis]


def get_cfg_group():
    return _axis_group("cfg")


def get_cp_group():
    return _axis_group("cp")


def get_dp_group():
    return _axis_group("dp")


def get_cp_mesh() -> list[list[int]]:
    """Replica groups of the cp axis (for ring collectives)."""
    spec = _require_spec()
    return spec.axis_groups("cp")


def get_cfg_rank_spmd(global_rank: torch.Tensor) -> torch.Tensor:
    """cfg coordinate of a traced global rank: (rank // (T*C)) % G."""
    spec = _require_spec()
    return torch.remainder(
        torch.div(global_rank, spec.tp * spec.cp, rounding_mode="floor"), spec.cfg
    ).to(torch.int32)


def get_cp_rank_spmd(global_rank: torch.Tensor) -> torch.Tensor:
    """cp coordinate of a traced global rank: (rank // T) % C."""
    spec = _require_spec()
    return torch.remainder(
        torch.div(global_rank, spec.tp, rounding_mode="floor"), spec.cp
    ).to(torch.int32)


def destroy_parallel_mesh() -> None:
    """Reset module state (unit tests only; does not destroy torch groups)."""
    global _MESH_SPEC
    _MESH_SPEC = None
    _AXIS_GROUPS.clear()
```

- [ ] **Step 4: Run** — `PYTHONPATH=. pytest tests/unit/backends/test_trainium_parallel_mesh.py -q` → PASS.
- [ ] **Step 5: Commit & push** — `git commit -m "feat(mesh): Trainium ProcessGroupManager building per-axis subgroups from one spec"`.

---

### Task 4: ops surface swap (remove dp exports, add axis ops)

**Files:**
- Modify: `difflet/ops/__init__.py` (`_EXPORTS`)
- Modify: `difflet/backends/trainium/ops_impl/collectives.py`
- Modify: `difflet/backends/cpu/ops_impl/collectives.py`
- Modify: `tests/unit/backends/test_cpu_ops_impl.py:204-209`
- Test: extend `tests/unit/backends/test_cpu_ops_impl.py` + new asserts in `tests/unit/ops` if present

**Interfaces:**
- Produces via `difflet.ops`: `init_parallel_mesh(config)`, `get_cfg_group()`, `get_cp_group()`, `get_cfg_rank_spmd(global_rank)`, `get_cp_rank_spmd(global_rank)`.
- Removes from `difflet.ops`: `get_data_parallel_group`, `get_dp_rank_spmd`.

- [ ] **Step 1: Write failing test** — in `tests/unit/backends/test_cpu_ops_impl.py`, replace lines 204–209's dp assertions with:

```python
    assert cpu_col.get_cfg_rank_spmd(0) == 0
    assert cpu_col.get_cp_rank_spmd(0) == 0
    assert cpu_col.get_cfg_group().size() == 1
    assert cpu_col.get_cp_group().size() == 1
    assert cpu_col.init_parallel_mesh(object()) is None
    assert cpu_col.get_world_group().size() == 1
    assert not hasattr(cpu_col, "get_data_parallel_group")
    assert not hasattr(cpu_col, "get_dp_rank_spmd")
```

- [ ] **Step 2: Run to verify failure**.
- [ ] **Step 3: Implement.**
  - `difflet/ops/__init__.py`: in `_EXPORTS` delete the `"get_data_parallel_group"` and `"get_dp_rank_spmd"` entries; add:

```python
    "init_parallel_mesh": ("collectives", "init_parallel_mesh"),
    "get_cfg_group": ("collectives", "get_cfg_group"),
    "get_cp_group": ("collectives", "get_cp_group"),
    "get_cfg_rank_spmd": ("collectives", "get_cfg_rank_spmd"),
    "get_cp_rank_spmd": ("collectives", "get_cp_rank_spmd"),
```

  - `difflet/backends/trainium/ops_impl/collectives.py`: drop `get_data_parallel_group` from the NxD import and `from difflet.backends.trainium.utils.distributed import get_dp_rank_spmd`; add

```python
from difflet.backends.trainium.core.parallel_mesh import (
    get_cfg_group,
    get_cfg_rank_spmd,
    get_cp_group,
    get_cp_rank_spmd,
    init_parallel_mesh,
)
```

  and update `__all__` accordingly (remove the two dp names, add the five new ones).
  - `difflet/backends/cpu/ops_impl/collectives.py`: delete `get_data_parallel_group` and `get_dp_rank_spmd`; add

```python
def init_parallel_mesh(config):
    """Single-process CPU backend: the mesh is trivially (1,1,1,1)."""
    del config


def get_cfg_group():
    return _SingleProcessGroup()


def get_cp_group():
    return _SingleProcessGroup()


def get_cfg_rank_spmd(global_rank):
    del global_rank
    return 0


def get_cp_rank_spmd(global_rank):
    del global_rank
    return 0
```

  and update `__all__` (remove the two dp names, add the five new ones).
- [ ] **Step 4: Run** — `PYTHONPATH=. pytest tests/unit/backends tests/unit/ops -q` → PASS. NOTE: model modules still import the removed names at this point — that is Tasks 6–10; only run the backends/ops suites here.
- [ ] **Step 5: Commit & push** — `git commit -m "feat(ops): axis-explicit group ops; drop get_data_parallel_group/get_dp_rank_spmd from the surface"`.

---

### Task 5: Migrate ring collectives (ops_impl/attention.py)

**Files:**
- Modify: `difflet/backends/trainium/ops_impl/attention.py:159-199` (`ring_attention`), `:254-338` (`joint_ring_attention`)

**Interfaces:**
- Consumes: `get_cp_mesh()` from Task 3.

- [ ] **Step 1: Edit `ring_attention`** — replace the NxD import + mesh/num_workers derivation:

```python
    from difflet.backends.trainium.core.parallel_mesh import get_cp_mesh

    mesh = get_cp_mesh()  # List[List[int]] of global ranks, one ring per cp group
    num_workers = len(mesh[0])
    replica_groups = tuple(tuple(int(r) for r in grp) for grp in mesh)
```

Update the docstring line "The ring membership IS the data-parallel group the model scattered Q with" → "The ring membership IS the cp-axis subgroup the model scattered Q with".

- [ ] **Step 2: Edit `joint_ring_attention`** — same replacement for its `get_data_parallel_group(as_list=True)` / `get_data_parallel_size()` block:

```python
    from difflet.backends.trainium.core.parallel_mesh import get_cp_mesh

    mesh = get_cp_mesh()  # List[List[int]] of global ranks, one ring per cp group
    num_workers = len(mesh[0])
```

(keep the `xm` import; update the comment "each rank sends its current K,V to the next member of its cp group" — already says cp, fine).
- [ ] **Step 3: Verify no references remain** — `grep -n "get_data_parallel" difflet/backends/trainium/ops_impl/attention.py` → no matches.
- [ ] **Step 4: Run** — `PYTHONPATH=. pytest tests/unit/backends -q` → PASS (file must still import; device behavior covered in Task 12).
- [ ] **Step 5: Commit & push** — `git commit -m "refactor(cp): ring/joint-ring replica groups from the cp axis mesh, not NxD dp_group"`.

---

### Task 6: Migrate Wan

**Files:**
- Modify: `difflet/models/wan/modeling_wan.py` (imports ~50-66; `WanAttention.__init__` ~418-419; gather_kv ~543-545; model `__init__` ~694-698; `forward` CFG ~782-809, CP ~831-847, exit ~897-911; `teacache_mod_input` if it replicates the CFG/CP prefix — check and migrate identically)
- Test: `tests/unit/models/wan/test_modeling_wan_groups.py`

**Interfaces:**
- Consumes: `difflet.ops.{init_parallel_mesh, get_cfg_group, get_cp_group, get_cfg_rank_spmd, get_cp_rank_spmd}`.

- [ ] **Step 1: Write failing wiring test** — `tests/unit/models/wan/test_modeling_wan_groups.py` (same cpu-backend reload preamble as `test_modeling_wan_forward.py`):

```python
"""Wan CFG/CP collectives must be wired to the cfg/cp axis ops, not dp."""

import inspect


def test_wan_uses_axis_ops_not_dp(  # uses the reloaded module fixture pattern
):
    import tests.unit.models.wan.test_modeling_wan_forward as fwd
    src = inspect.getsource(fwd.wan)
    assert "get_data_parallel_group" not in src
    assert "get_dp_rank_spmd" not in src
    assert "get_cfg_group" in src
    assert "get_cp_group" in src
    assert "get_cfg_rank_spmd" in src
    assert "get_cp_rank_spmd" in src
    assert "init_parallel_mesh" in src
```

(Adjust to the actual reload fixture — import `difflet.models.wan.modeling_wan` directly with the same env dance if cleaner.)
- [ ] **Step 2: Edits.**
  - Imports: replace `get_data_parallel_group, get_dp_rank_spmd` with `get_cfg_group, get_cfg_rank_spmd, get_cp_group, get_cp_rank_spmd, init_parallel_mesh`; drop `get_tensor_model_parallel_size` if its only remaining use was the dp_rank call (verify with grep first).
  - `WanAttention.__init__` (~418): `self.data_parallel_group = get_data_parallel_group()` → `self.cp_group = get_cp_group()`; gather site (~543): `process_group=self.data_parallel_group` → `process_group=self.cp_group`.
  - Model `__init__` (~694-698):

```python
        # CFG parallel scatters the batch dim (uncond/cond) over the cfg axis;
        # CP scatters the sequence dim over the cp axis. Each collective fires
        # only in its own axis subgroup — dp carries nothing.
        if self.context_parallel_enabled or self.cfg_parallel_enabled:
            init_parallel_mesh(config)
            self.global_rank = SPMDRank(world_size=get_world_group().size())
        if self.cfg_parallel_enabled:
            self.cfg_group = get_cfg_group()
        if self.context_parallel_enabled:
            self.cp_group = get_cp_group()
```

  NOTE: `init_parallel_mesh` must run BEFORE `self.blocks` is built (WanAttention grabs `get_cp_group()` in its own `__init__`).
  - `forward` CFG block: `dp_rank = get_dp_rank_spmd(global_rank=..., tp_degree=...)` → `cfg_rank = get_cfg_rank_spmd(self.global_rank.get_rank())`; all three scatters use `rank=cfg_rank, process_group=self.cfg_group`.
  - `forward` CP block: → `cp_rank = get_cp_rank_spmd(self.global_rank.get_rank())`; scatters use `rank=cp_rank, process_group=self.cp_group`.
  - Exit gathers: CP gather → `process_group=self.cp_group`; CFG merge → `process_group=self.cfg_group`.
  - Mirror the same substitutions in `teacache_mod_input` if it contains the scatter prefix (check `grep -n "dp_rank\|data_parallel_group" difflet/models/wan/modeling_wan.py` afterwards → zero matches).
- [ ] **Step 3: Run** — `PYTHONPATH=. pytest tests/unit/models/wan -q` → PASS (forward tests prove CPU numerics unchanged; new wiring test proves axis ops).
- [ ] **Step 4: Commit & push** — `git commit -m "refactor(wan): CFG merge on cfg_group, CP on cp_group; de-parasitize dp_group"`.

---

### Task 7: Migrate Flux

**Files:**
- Modify: `difflet/models/flux/modeling_flux.py` (imports ~56-58; model `__init__` 245-246 → conditional; forward 417-455 CFG, 500-515 CP, 597-605 exit; `FluxAttention.__init__` 1028; attention CP sites 1327-1390; `split_along_dim` 1451-1456 and its callers)
- Test: `tests/unit/models/flux/test_modeling_flux_groups.py` (same static-wiring shape as Task 6's test, against `difflet.models.flux.modeling_flux`)

- [ ] **Step 1: Write failing wiring test** (identical pattern to Task 6 step 1: source must not contain `get_data_parallel_group`/`get_dp_rank_spmd`, must contain the new names).
- [ ] **Step 2: Edits.**
  - Imports: same swap as Wan.
  - Model `__init__` 245-250 — move the flag reads first, make groups conditional (the dormant CFG path stays wired, to `cfg_group`):

```python
        self.context_parallel_enabled = getattr(self.config, 'context_parallel_enabled', False)
        self.cp_mode = getattr(self.config, 'cp_mode', 'gather_kv')
        self.cfg_parallel_enabled = getattr(self.config, 'cfg_parallel_enabled', False)
        if self.context_parallel_enabled or self.cfg_parallel_enabled:
            init_parallel_mesh(self.config)
            self.global_rank = SPMDRank(world_size=get_world_group().size())
        self.cfg_group = get_cfg_group() if self.cfg_parallel_enabled else None
        self.cp_group = get_cp_group() if self.context_parallel_enabled else None
```

  - Forward: delete the unconditional `dp_rank = get_dp_rank_spmd(...)` (line 417-420); inside the CFG branch compute `cfg_rank = get_cfg_rank_spmd(self.global_rank.get_rank())` and scatter with `rank=cfg_rank, process_group=self.cfg_group`; inside the CP branch compute `cp_rank = get_cp_rank_spmd(self.global_rank.get_rank())` and pass `rank=cp_rank, process_group=self.cp_group` to the four `split_along_dim` calls; exit gathers → cfg_group / cp_group respectively.
  - `split_along_dim(tensor, dim, rank, data_parallel_group)` → rename the parameter to `process_group` (it forwards to `scatter_to_process_group_spmd(..., process_group=process_group)`); update its 4 call sites.
  - `FluxAttention.__init__` 1028: `self.data_parallel_group = get_data_parallel_group()` →

```python
        self.cp_group = get_cp_group() if context_parallel_enabled else None
```

  (place after `self.context_parallel_enabled = context_parallel_enabled`); rename all `self.data_parallel_group` uses in the CP attention paths (1327, 1338, 1359, 1370, 1374, 1390) to `self.cp_group`.
- [ ] **Step 3: Verify** — `grep -n "data_parallel\|dp_rank" difflet/models/flux/modeling_flux.py` → zero matches; `PYTHONPATH=. pytest tests/unit/models/flux -q` → PASS.
- [ ] **Step 4: Commit & push** — `git commit -m "refactor(flux): CFG/CP collectives on their own axis groups; conditional group init"`.

---

### Task 8: Migrate HunyuanVideo

**Files:**
- Modify: `difflet/models/hunyuan_video/modeling_hunyuan_video.py` (imports 40-46; `HunyuanVideoAttention.__init__` 331-333; gather 493-494; model `__init__` 1031-1033; forward 1151-1168; exit gather 1223-1224)
- Test: `tests/unit/models/hunyuan_video/test_modeling_hyv_groups.py` (same wiring-test shape)

- [ ] **Step 1: Failing wiring test** (CP-only: require `get_cp_group`, `get_cp_rank_spmd`, `init_parallel_mesh`; forbid dp names).
- [ ] **Step 2: Edits.**
  - Imports: swap `get_data_parallel_group, get_dp_rank_spmd` → `get_cp_group, get_cp_rank_spmd, init_parallel_mesh`.
  - Attention 331-333: `self.cp_group = get_cp_group() if context_parallel_enabled else None`; use at 493-494.
  - Model `__init__` 1031-1033:

```python
        if context_parallel_enabled:
            init_parallel_mesh(config)
            self.cp_group = get_cp_group()
            self.global_rank = SPMDRank(world_size=get_world_group().size())
```

  NOTE: as in Wan, this must execute before the transformer blocks are constructed — verify block construction order and move the init above it if needed.
  - Forward 1151-1155: `dp_rank = get_dp_rank_spmd(...)` → `cp_rank = get_cp_rank_spmd(self.global_rank.get_rank())`; the three scatters and exit gather use `self.cp_group`.
- [ ] **Step 3: Verify + run** — `grep -n "data_parallel\|dp_rank" difflet/models/hunyuan_video/modeling_hunyuan_video.py` → zero; `PYTHONPATH=. pytest tests/unit/models -q` → PASS.
- [ ] **Step 4: Commit & push** — `git commit -m "refactor(hunyuan): CP collectives on cp_group"`.

---

### Task 9: Migrate Qwen-Image

**Files:**
- Modify: `difflet/backends/trainium/qwen_image/transformer.py` (imports 22-23; `_QwenImageRopeModule` 143-172; `_QwenImageTrainiumAttnProcessor` 314-396; trace module 465-471 and wiring 496-504)
- Test: `tests/unit/models/qwen_image/test_qwen_groups.py` (wiring-test shape; note this module lives under backends — import path `difflet.backends.trainium.qwen_image.transformer`, guard with `pytest.importorskip("neuronx_distributed")`)

- [ ] **Step 1: Failing wiring test.**
- [ ] **Step 2: Edits.**
  - Imports: swap to `get_cp_group, get_cp_rank_spmd, init_parallel_mesh`.
  - Trace module `__init__` 465-471:

```python
        if self.context_parallel_enabled:
            init_parallel_mesh(config)
            self.cp_group = get_cp_group()
            self.global_rank = SPMDRank(world_size=get_world_group().size())
        else:
            self.cp_group = None
            self.global_rank = None
```

  - Rename the `data_parallel_group=` kwarg to `cp_group=` on `_QwenImageRopeModule` and `_QwenImageTrainiumAttnProcessor` (definition + the two wiring sites at 496-504); inside both classes rename `self.data_parallel_group` → `self.cp_group`.
  - `_QwenImageRopeModule.forward`: `get_dp_rank_spmd(...)` → `cp_rank = get_cp_rank_spmd(self._global_rank_ref[0].get_rank())`; scatters use `process_group=self.cp_group`.
  - Processor gather (~393-396): `process_group=self.cp_group`.
- [ ] **Step 3: Verify + run** — `grep -n "data_parallel\|dp_rank" difflet/backends/trainium/qwen_image/transformer.py` → zero; unit suite for qwen/registry stays green.
- [ ] **Step 4: Commit & push** — `git commit -m "refactor(qwen): CP collectives on cp_group"`.

---

### Task 10: Migrate LTX-2

**Files:**
- Modify: `difflet/backends/trainium/ltx_2/transformer.py` (imports 16-29; `_LTX2TransformerTraceModule.__init__` 544-552; forward 585-606 scatter, 633-639 gathers)
- Test: `tests/unit/models/ltx_2/test_ltx2_groups.py` (wiring-test shape, CFG-only: require `get_cfg_group`/`get_cfg_rank_spmd`/`init_parallel_mesh`)

- [ ] **Step 1: Failing wiring test.**
- [ ] **Step 2: Edits.**
  - Imports: swap `get_data_parallel_group, get_dp_rank_spmd` → `get_cfg_group, get_cfg_rank_spmd, init_parallel_mesh`.
  - `__init__` 545-552 (STG rejection stays — cfg axis is ONLY cond/uncond):

```python
        if self.cfg_parallel_enabled:
            if bool(getattr(config, "perturbed_attn", False)):
                raise NotImplementedError(
                    "LTX-2 CFG-parallel does not support perturbed_attn (STG); "
                    "disable spatio-temporal guidance when cfg_parallel_enabled."
                )
            init_parallel_mesh(config)
            self.cfg_group = get_cfg_group()
            self.global_rank = SPMDRank(world_size=get_world_group().size())
```

  - Forward 585-606: `dp_rank = get_dp_rank_spmd(...)` → `cfg_rank = get_cfg_rank_spmd(self.global_rank.get_rank())`; `_scatter` uses `rank=cfg_rank, process_group=self.cfg_group`; output gathers 633-639 use `self.cfg_group`.
- [ ] **Step 3: Verify + run** — grep zero dp references; `PYTHONPATH=. pytest tests/unit -q` → FULL suite PASS (all models migrated now).
- [ ] **Step 4: Commit & push** — `git commit -m "refactor(ltx2): CFG merge on cfg_group"`.

---

### Task 11: Parasite-audit + dp-carries-nothing tests

**Files:**
- Create: `tests/unit/test_no_dp_parasites.py`

- [ ] **Step 1: Write the audit test** (fails if any model/op file regresses to dp; also proves `difflet.ops` no longer exports the names):

```python
"""No CFG/CP/TP collective may ride the dp axis.

Static audit: outside the NxDI-fork core (attention_base / attention_process_groups /
utils/distributed) and the mesh manager itself, nothing may reference NxD's
get_data_parallel_group or the merged-axis get_dp_rank_spmd. Together with the
manager's "no dp group unless dp>1, and no consumer of get_dp_group in any
forward path" this enforces that dp carries no per-layer/per-step collective.
"""

from pathlib import Path

import difflet
import difflet.ops as ops

REPO = Path(difflet.__file__).resolve().parent

FORBIDDEN = ("get_data_parallel_group", "get_dp_rank_spmd")

# NxDI verbatim-fork files (LLM attention machinery, not the pipeline axes).
ALLOWED = {
    REPO / "backends/trainium/core/modules/attention/attention_base.py",
    REPO / "backends/trainium/core/modules/attention/attention_process_groups.py",
    REPO / "backends/trainium/core/modules/attention/sink.py",
    REPO / "backends/trainium/utils/distributed.py",
    REPO / "backends/trainium/core/config.py",
    REPO / "backends/trainium/core/application_base.py",
    REPO / "utils/tensor_capture_utils.py",
}


def test_ops_surface_has_no_dp_exports():
    assert "get_data_parallel_group" not in ops.__all__
    assert "get_dp_rank_spmd" not in ops.__all__
    for name in ("init_parallel_mesh", "get_cfg_group", "get_cp_group",
                 "get_cfg_rank_spmd", "get_cp_rank_spmd"):
        assert name in ops.__all__


def test_no_source_file_references_dp_group():
    offenders = []
    for path in REPO.rglob("*.py"):
        if path in ALLOWED:
            continue
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            if token in text:
                offenders.append(f"{path.relative_to(REPO)}: {token}")
    assert offenders == [], "\n".join(offenders)


def test_dp_axis_group_has_no_forward_consumer():
    # get_dp_group exists on the manager (reserved for the DP feature) but no
    # model / pipeline / ops code may call it in this increment.
    offenders = []
    manager = REPO / "backends/trainium/core/parallel_mesh.py"
    for path in REPO.rglob("*.py"):
        if path == manager:
            continue
        if "get_dp_group" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(REPO)))
    assert offenders == [], "\n".join(offenders)
```

Before finalizing ALLOWED, run the grep and shrink the set to files that actually (legitimately) contain the tokens — the smaller the better.
- [ ] **Step 2: Run** — `PYTHONPATH=. pytest tests/unit/test_no_dp_parasites.py -q` → PASS; then FULL `PYTHONPATH=. pytest tests/unit -q` → PASS.
- [ ] **Step 3: Commit & push** — `git commit -m "test: enforce dp_group carries no collectives (static audit + ops surface)"`.

---

### Task 12: On-device bit-identity regression + smoke

**Files:**
- Create: `scripts/mesh_regression_smoke.py`, `scripts/mesh_regression_smoke.sh`

**Protocol:** regression targets per user decision: `(dp=1, cfg=2, tp=2)` and `(dp=1, cp=2, tp=2)`, world=4 (fits the 4 logical cores). Baseline = commit `93fac4c` (pre-refactor). Real Wan weights are not on disk, so the script synthesizes a **tiny seeded diffusers `WanTransformer3DModel` checkpoint** (`save_pretrained`) once, shared by baseline and refactor runs — bit-identity needs identical weights, not real ones.

- [ ] **Step 1: Write `scripts/mesh_regression_smoke.py`** — modeled on `scripts/wan_ring_parity_smoke.py`: subcommand `--make-checkpoint DIR` (seed 0, tiny dims: heads=4, head_dim=64, layers=2, in_channels=16, text_dim=64, ffn_dim=256, patch [1,2,2]); run mode `--mode {cfg,cp} --ckpt DIR --out OUT.pt --work-dir WD`: builds `create_wan_backbone_config(world_size=4, tp_degree=2, context_parallel_enabled=(mode=="cp"), cfg_parallel_enabled=(mode=="cfg"), batch_size=2 if mode=="cfg" else 1, height=256, width=512, num_frames=1)`, `app.compile(WD)`, `app.load(WD)`, fixed-seed (1234) inputs (batch 2 for cfg mode), saves float32 output tensor.
- [ ] **Step 2: Write `scripts/mesh_regression_smoke.sh`** — same env preamble as `wan_ring_parity_smoke.sh` (venv PATH, PYTHONPATH, `NEURON_RT_NUM_CORES=4`, `DIFFLET_BACKEND=trainium`); flow:
  1. `git worktree add /tmp/mesh_baseline 93fac4c` (skip if exists).
  2. Make the shared tiny checkpoint once.
  3. For mode in cfg, cp: run with `PYTHONPATH=/tmp/mesh_baseline` → `baseline_<mode>.pt`; run with `PYTHONPATH=$ROOT` → `refactor_<mode>.pt` (separate work dirs per revision+mode).
  4. Compare: `torch.equal(a, b)` (byte-exact; report PSNR=inf on success) — non-zero exit on mismatch.
- [ ] **Step 3: Run it on device** — `bash scripts/mesh_regression_smoke.sh` → both modes byte-identical. This doubles as the on-device smoke (compile → load → forward through the refactored group layer).
- [ ] **Step 4: Commit & push** — `git commit -m "test(device): bit-identity mesh regression (cfg=2 and cp=2 vs pre-refactor) + smoke"`.

---

### Task 13: Final report + docs

**Files:**
- Create: `docs/reports/2026-07-05-parallel-mesh-refactor.md` (design summary, rank mapping, audit table of migrated call sites, unit-suite result, device regression PSNR/bit-identity evidence with command transcripts, known limitations: dp>1 unwired, cfg×cp expressible-but-rejected, HYV-1.5/Qwen cfg policy per user decision)
- Modify: `DEVELOPER.md` — short section documenting the mesh (`MeshSpec`, `init_parallel_mesh`, per-axis ops) replacing any dp-group guidance.

- [ ] **Step 1: Write the report** with actual results pasted in.
- [ ] **Step 2: Update DEVELOPER.md.**
- [ ] **Step 3: Full suite once more** — `PYTHONPATH=. pytest tests/unit -q` → PASS.
- [ ] **Step 4: Commit & push** — `git commit -m "docs: parallel-mesh refactor report + developer guide"`.

---

## Self-Review Notes

- Spec coverage: R1 mesh+mapping → Tasks 1–3; R2 call-site moves + dp carries nothing → Tasks 5–11; R3 mesh-spec product assert + 4 combos → Tasks 1–3 tests; R4 per-model policy → unchanged guards (global constraints) + LTX STG rejection preserved (Task 10); regression tests per user decision → Task 12; assertion test → Task 11; user additions (device+smoke, report, per-step push) → Tasks 12–13 + global constraint.
- Names used consistently: `init_parallel_mesh`, `get_cfg_group`, `get_cp_group`, `get_cfg_rank_spmd`, `get_cp_rank_spmd`, `get_cp_mesh`, `MeshSpec`, `MeshCoords`, `AXES`.
- Line numbers are anchors from the audit; verify with grep before each edit (files shift as tasks land).
