"""Cache predictors: the independent "what replaces the model output?" axis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from difflet.pipeline.cache.types import CacheHistory, CacheStepContext, Coordinate


def _torch_working_copy(value: Any):
    import torch

    if not torch.is_tensor(value):
        raise TypeError(f"cache predictors require torch.Tensor anchors, got {type(value)!r}")
    original_dtype = value.dtype
    if value.is_floating_point() and value.dtype in (torch.float16, torch.bfloat16):
        value = value.float()
    else:
        value = value.clone()
    return value, original_dtype


@dataclass(frozen=True)
class LegacyResidualPredictor:
    """Continue the most recent real-anchor slope for one isolated step."""

    coord: Coordinate = "index"
    required_history: int = 2
    max_consecutive_predictions: int | None = 1

    def __post_init__(self) -> None:
        if self.coord not in ("index", "timestep", "sigma"):
            raise ValueError("coord must be one of: index, timestep, sigma")

    def predict(self, context: CacheStepContext, history: CacheHistory) -> Any:
        if not history.ready(self.required_history):
            raise RuntimeError("legacy residual prediction requires two real anchors")
        previous, latest = history.tail(2)
        x0 = previous.coordinate(self.coord)
        x1 = latest.coordinate(self.coord)
        target = context.coordinate(self.coord)
        denominator = x1 - x0
        if denominator == 0.0:
            raise ValueError("legacy residual predictor received duplicate coordinates")
        y0, _ = _torch_working_copy(previous.output)
        y1, dtype = _torch_working_copy(latest.output)
        result = y1 + (y1 - y0) * ((target - x1) / denominator)
        return result.to(dtype=dtype).detach()


@dataclass(frozen=True)
class TaylorSeerPredictor:
    """Newton divided-difference extrapolation over real anchors."""

    order: int = 1
    coord: Coordinate = "index"
    max_consecutive_predictions: int | None = None
    _coefficient_key: Any = field(default=None, init=False, repr=False, compare=False)
    _coefficient_coordinates: tuple[float, ...] = field(
        default=(), init=False, repr=False, compare=False
    )
    _coefficients: tuple[Any, ...] = field(
        default=(), init=False, repr=False, compare=False
    )
    _coefficient_dtype: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.order, bool) or self.order not in (1, 2):
            raise ValueError("TaylorSeer order must be 1 or 2")
        if self.coord not in ("index", "timestep", "sigma"):
            raise ValueError("coord must be one of: index, timestep, sigma")

    @property
    def required_history(self) -> int:
        return self.order + 1

    def reset_cache(self) -> None:
        """Release request-derived coefficients without changing configuration."""

        object.__setattr__(self, "_coefficient_key", None)
        object.__setattr__(self, "_coefficient_coordinates", ())
        object.__setattr__(self, "_coefficients", ())
        object.__setattr__(self, "_coefficient_dtype", None)

    def _prepare_coefficients(
        self,
        history: CacheHistory,
    ) -> tuple[tuple[float, ...], tuple[Any, ...], Any]:
        required = self.required_history
        anchors = history.tail(required)
        coordinates = tuple(anchor.coordinate(self.coord) for anchor in anchors)
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("TaylorSeer received duplicate anchor coordinates")
        key = tuple(
            (anchor.step_index, coordinate, id(anchor.output))
            for anchor, coordinate in zip(anchors, coordinates)
        )
        if key == self._coefficient_key:
            return (
                self._coefficient_coordinates,
                self._coefficients,
                self._coefficient_dtype,
            )

        values = []
        original_dtype = None
        for anchor in anchors:
            value, dtype = _torch_working_copy(anchor.output)
            original_dtype = dtype if original_dtype is None else original_dtype
            values.append(value)

        # Newton divided differences. ``coefficients[k]`` becomes the order-k
        # coefficient while lower entries retain the coefficients needed by
        # nested multiplication in ``predict``.
        coefficients = list(values)
        for level in range(1, required):
            for index in range(required - 1, level - 1, -1):
                denominator = coordinates[index] - coordinates[index - level]
                if denominator == 0.0:
                    raise ValueError("TaylorSeer received duplicate anchor coordinates")
                coefficients[index] = (
                    coefficients[index] - coefficients[index - 1]
                ) / denominator

        result = (coordinates, tuple(coefficients), original_dtype)
        object.__setattr__(self, "_coefficient_key", key)
        object.__setattr__(self, "_coefficient_coordinates", result[0])
        object.__setattr__(self, "_coefficients", result[1])
        object.__setattr__(self, "_coefficient_dtype", result[2])
        return result

    def predict(self, context: CacheStepContext, history: CacheHistory) -> Any:
        required = self.required_history
        if not history.ready(required):
            raise RuntimeError(
                f"TaylorSeer order={self.order} requires {required} real anchors"
            )
        coordinates, coefficients, original_dtype = self._prepare_coefficients(history)

        target = context.coordinate(self.coord)
        result = coefficients[-1]
        for index in range(required - 2, -1, -1):
            result = coefficients[index] + (target - coordinates[index]) * result
        return result.to(dtype=original_dtype).detach()


__all__ = ["LegacyResidualPredictor", "TaylorSeerPredictor"]
