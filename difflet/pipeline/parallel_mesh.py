"""Orthogonal {dp, cfg, cp, tp} device-mesh math (backend-agnostic).

Rank layout (tp innermost, dp outermost):

    rank = tp + T*(cp + C*(cfg + G*dp))

An axis's subgroup is the set of ranks that differ only in that axis's
coordinate. For every configuration expressible pre-refactor (exactly one of
cfg/cp non-trivial, dp=1) the non-trivial axis's groups coincide with NxD's
legacy data-parallel column groups ``[[j, j+T, ...] for j in range(T)]``, which
is what makes the group migration bit-identical on device.
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
        self.axis_size(axis)  # validates the axis name
        return getattr(self.coords_of(rank), axis)

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
