"""Model registry for NovaPipeline."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

from nova.pipeline.parallel_config import NovaParallelConfig

ApplicationFactory = Callable[..., Any]
Detector = Callable[[str], bool]

_REGISTRY: dict[str, "ModelEntry"] = {}
_BUILTINS_LOADED = False


@dataclass(frozen=True)
class ModelEntry:
    name: str
    application_factory: ApplicationFactory | str
    hf_paths: tuple[str, ...] = ()
    detector: Detector | None = None
    default_parallel: NovaParallelConfig = field(default_factory=NovaParallelConfig)
    default_shape: dict[str, int | None] = field(default_factory=dict)
    backends: tuple[str, ...] = ("trainium",)
    # Per-model HF download allow-list. ``None`` (the default) lets
    # ``resolve_model_path`` use ``DEFAULT_DIFFUSERS_PATTERNS``. Override only
    # when a model needs files outside the standard diffusers layout (e.g. a
    # model that ships custom code in a non-standard directory).
    download_patterns: tuple[str, ...] | None = None

    def matches(self, model_id: str) -> bool:
        normalized = model_id.rstrip("/")
        if normalized in self.hf_paths:
            return True
        if self.detector is not None:
            return bool(self.detector(model_id))
        return False

    def resolve_shape(
        self,
        *,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
    ) -> dict[str, int | None]:
        shape = dict(self.default_shape)
        if height is not None:
            shape["height"] = height
        if width is not None:
            shape["width"] = width
        if num_frames is not None:
            shape["num_frames"] = num_frames
        return shape

    def create_application(
        self,
        *,
        model_path: str,
        parallel: NovaParallelConfig,
        dtype: Any,
        shape: dict[str, int | None],
        backend: str,
        application_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        factory = _resolve_factory(self.application_factory)
        return factory(
            model_path=model_path,
            parallel=parallel,
            dtype=dtype,
            shape=shape,
            backend=backend,
            **(application_kwargs or {}),
        )

    def require_backend(self, backend: str) -> None:
        if backend not in self.backends:
            supported = ", ".join(self.backends)
            raise ValueError(
                f"model {self.name!r} does not support backend {backend!r}; "
                f"supported backends: {supported}"
            )


def register_model(
    *,
    name: str,
    application_factory: ApplicationFactory | str | None = None,
    hf_paths: list[str] | tuple[str, ...] = (),
    detector: Detector | None = None,
    default_parallel: NovaParallelConfig | None = None,
    default_shape: dict[str, int | None] | None = None,
    backends: list[str] | tuple[str, ...] = ("trainium",),
) -> Callable[[type], type]:
    """Register a model entry.

    Can be used as a decorator, or called with a dummy class for metadata-only
    entries whose application factory is a lazy import string.
    """

    def decorator(cls: type) -> type:
        factory = application_factory or getattr(cls, "application_factory", None)
        if factory is None:
            raise ValueError(f"model {name!r} is missing an application_factory")
        _REGISTRY[name] = ModelEntry(
            name=name,
            application_factory=factory,
            hf_paths=tuple(hf_paths),
            detector=detector,
            default_parallel=default_parallel or NovaParallelConfig(),
            default_shape=default_shape or {},
            backends=tuple(backends),
        )
        return cls

    return decorator


def resolve_model(model_id: str, model_type: str | None = None) -> ModelEntry:
    _ensure_builtin_models_registered()
    if model_type is not None:
        try:
            return _REGISTRY[model_type]
        except KeyError as exc:
            raise ValueError(f"unknown model_type {model_type!r}") from exc

    matches = [entry for entry in _REGISTRY.values() if entry.matches(model_id)]
    if not matches:
        names = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ValueError(f"could not resolve model type for {model_id!r}; registered: {names}")
    if len(matches) > 1:
        names = ", ".join(entry.name for entry in matches)
        raise ValueError(f"model id {model_id!r} matched multiple entries: {names}")
    return matches[0]


def registered_models() -> tuple[ModelEntry, ...]:
    _ensure_builtin_models_registered()
    return tuple(_REGISTRY.values())


def _resolve_factory(factory: ApplicationFactory | str) -> ApplicationFactory:
    if callable(factory):
        return factory
    module_name, sep, attr_name = factory.partition(":")
    if sep != ":" or not module_name or not attr_name:
        raise ValueError(f"invalid application factory reference {factory!r}")
    module = importlib.import_module(module_name)
    resolved = getattr(module, attr_name)
    if not callable(resolved):
        raise TypeError(f"application factory {factory!r} is not callable")
    return resolved


def _ensure_builtin_models_registered() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    _register_builtin_flux()


def _register_builtin_flux() -> None:
    def is_flux(model_id: str) -> bool:
        value = model_id.lower()
        return "flux" in value or "black-forest-labs/flux" in value

    @register_model(
        name="flux",
        application_factory="nova.models.flux.entry:create_flux_application",
        hf_paths=(
            "black-forest-labs/FLUX.1-dev",
            "black-forest-labs/FLUX.1-schnell",
        ),
        detector=is_flux,
        default_parallel=NovaParallelConfig(tp_degree=8),
        default_shape={"height": 1024, "width": 1024, "num_frames": None},
    )
    class _FluxRegistration:
        pass
