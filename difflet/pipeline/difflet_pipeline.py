"""Unified public pipeline entry point."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from difflet.backends import BackendRuntime, get_backend
from difflet.pipeline.compile_cache import CacheSpec, cache_path, has_valid_manifest, write_manifest
from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.pipeline.path_resolver import resolve_model_path
from difflet.registry import ModelEntry, resolve_model


class DiffletPipeline:
    """Thin wrapper around a model-specific Neuron application."""

    def __init__(
        self,
        *,
        app: Any,
        model_id: str,
        model_path: str,
        model_entry: ModelEntry,
        compiled_path: Path,
        cache_spec: CacheSpec,
        shape: dict[str, int | None],
        parallel: DiffletParallelConfig,
        dtype: Any,
        backend: BackendRuntime,
    ) -> None:
        self.app = app
        self.model_id = model_id
        self.model_path = model_path
        self.model_entry = model_entry
        self.compiled_path = compiled_path
        self.cache_spec = cache_spec
        self.shape = shape
        self.parallel = parallel
        self.dtype = dtype
        self.backend = backend

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        parallel: DiffletParallelConfig | None = None,
        dtype: Any | None = None,
        compile_cache_dir: str | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        model_type: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        force_compile: bool = False,
        skip_compile: bool = False,
        load: bool = True,
        start_rank_id: int | None = None,
        local_ranks_size: int | None = None,
        skip_warmup: bool = False,
        debug_compile: bool = False,
        backend: str | None = None,
        application_kwargs: dict[str, Any] | None = None,
        teacache_speedup: float | None = None,
        teacache_calibration: Any | None = None,
        teacache_calibration_path: str | None = None,
        model_path_override: str | None = None,
        resolved_source_id: str | None = None,
        compiled_path_override: str | Path | None = None,
    ) -> "DiffletPipeline":
        application_kwargs = _merge_teacache_kwargs(
            application_kwargs,
            teacache_speedup=teacache_speedup,
            teacache_calibration=teacache_calibration,
            teacache_calibration_path=teacache_calibration_path,
        )
        cache_application_kwargs = _cache_application_kwargs(application_kwargs)
        entry = resolve_model(model_id, model_type=model_type)
        backend_runtime = get_backend(backend)
        entry.require_backend(backend_runtime.name)
        parallel_cfg = parallel or entry.default_parallel
        backend_runtime.prepare_runtime(parallel_cfg)
        dtype = dtype if dtype is not None else _default_dtype()
        shape = entry.resolve_shape(height=height, width=width, num_frames=num_frames)
        overrides = (model_path_override, resolved_source_id, compiled_path_override)
        bound_mode = any(value is not None for value in overrides)
        if bound_mode and not all(value is not None for value in overrides):
            raise ValueError(
                "model_path_override, resolved_source_id, and compiled_path_override "
                "must be provided together"
            )
        if bound_mode:
            model_path = str(Path(model_path_override).expanduser().resolve())
            cache_revision = resolved_source_id
        else:
            model_path = resolve_model_path(
                model_id,
                revision=revision,
                local_files_only=local_files_only,
                allow_patterns=entry.download_patterns,
            )
            cache_revision = revision
        spec = CacheSpec(
            model_id=model_id,
            model_path=model_path,
            model_name=entry.name,
            parallel=parallel_cfg,
            dtype=dtype,
            height=shape.get("height"),
            width=shape.get("width"),
            num_frames=shape.get("num_frames"),
            revision=cache_revision,
            application_kwargs=cache_application_kwargs,
        )
        compiled_path = (
            Path(compiled_path_override).expanduser().resolve()
            if bound_mode
            else cache_path(compile_cache_dir, spec)
        )
        app = entry.create_application(
            model_path=model_path,
            parallel=parallel_cfg,
            dtype=dtype,
            shape=shape,
            backend=backend_runtime.name,
            application_kwargs=application_kwargs,
        )

        cache_ready = has_valid_manifest(compiled_path, spec) and _compiled_artifacts_ready(
            app, compiled_path
        )
        if not skip_compile and (force_compile or not cache_ready):
            compiled_path.mkdir(parents=True, exist_ok=True)
            _compile_app(app, compiled_path, debug=debug_compile)
            write_manifest(compiled_path, spec)

        if load:
            _load_app(
                app,
                compiled_path,
                backend=backend_runtime,
                start_rank_id=start_rank_id,
                local_ranks_size=local_ranks_size,
                skip_warmup=skip_warmup,
            )

        return cls(
            app=app,
            model_id=model_id,
            model_path=model_path,
            model_entry=entry,
            compiled_path=compiled_path,
            cache_spec=spec,
            shape=shape,
            parallel=parallel_cfg,
            dtype=dtype,
            backend=backend_runtime,
        )

    @classmethod
    def precompile(cls, model_id: str, **kwargs: Any) -> "DiffletPipeline":
        kwargs.setdefault("load", False)
        return cls.from_pretrained(model_id, **kwargs)

    def compile(self, *, force: bool = False, debug: bool = False) -> None:
        # Use the same cache-validity check as `from_pretrained` so a stale or
        # missing manifest still triggers recompile even when the artifact dir
        # already exists.
        cache_ready = has_valid_manifest(
            self.compiled_path, self.cache_spec
        ) and _compiled_artifacts_ready(self.app, self.compiled_path)
        if force or not cache_ready:
            self.compiled_path.mkdir(parents=True, exist_ok=True)
            _compile_app(self.app, self.compiled_path, debug=debug)
            write_manifest(self.compiled_path, self.cache_spec)

    def load(
        self,
        *,
        start_rank_id: int | None = None,
        local_ranks_size: int | None = None,
        skip_warmup: bool = False,
    ) -> None:
        _load_app(
            self.app,
            self.compiled_path,
            backend=self.backend,
            start_rank_id=start_rank_id,
            local_ranks_size=local_ranks_size,
            skip_warmup=skip_warmup,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.app(*args, **kwargs)


def _default_dtype() -> Any:
    import torch

    return torch.bfloat16


def _merge_teacache_kwargs(
    application_kwargs: dict[str, Any] | None,
    *,
    teacache_speedup: float | None,
    teacache_calibration: Any | None = None,
    teacache_calibration_path: str | None = None,
) -> dict[str, Any] | None:
    if teacache_calibration is not None and teacache_calibration_path is not None:
        raise ValueError(
            "teacache_calibration and teacache_calibration_path are mutually exclusive"
        )
    merged = dict(application_kwargs or {})
    for key, value in (
        ("teacache_speedup", teacache_speedup),
        ("teacache_calibration", teacache_calibration),
        ("teacache_calibration_path", teacache_calibration_path),
    ):
        if value is None:
            continue
        if key in merged and merged[key] != value:
            raise ValueError(
                f"{key} was provided both as a top-level argument and in " "application_kwargs."
            )
        merged[key] = value
    return merged or None


def _cache_application_kwargs(application_kwargs: dict[str, Any] | None) -> dict[str, Any] | None:
    if not application_kwargs:
        return None
    cache_kwargs = dict(application_kwargs)
    probe_enabled = bool(
        cache_kwargs.pop("teacache_speedup", None) is not None
        or cache_kwargs.pop("teacache_fused", False)
    )
    cache_kwargs.pop("teacache_calibration", None)
    cache_kwargs.pop("teacache_calibration_path", None)
    cache_kwargs.pop("teacache_cadence", None)
    cache_kwargs.pop("teacache_online_delta_alpha", None)
    for runtime_key in (
        "cache_plan_file",
        "cache_mask_file",
        "cache_predictor",
        "cache_predictor_order",
        "cache_predictor_coord",
        "cache_recovery_warmup_steps",
        "cache_recovery_cooldown_steps",
        "cache_recovery_max_consecutive",
        "cache_recovery_steps",
        "cache_require_final_anchor",
    ):
        cache_kwargs.pop(runtime_key, None)
    if probe_enabled:
        cache_kwargs["teacache_probe_enabled"] = True
    return cache_kwargs or None


def _compile_app(app: Any, compiled_path: Path, *, debug: bool) -> None:
    signature = inspect.signature(app.compile)
    if "debug" in signature.parameters:
        app.compile(str(compiled_path), debug=debug)
    else:
        app.compile(str(compiled_path))


def _compiled_artifacts_ready(app: Any, compiled_path: Path) -> bool:
    checker = getattr(app, "has_compiled_artifacts", None)
    if checker is None:
        return True
    return bool(checker(str(compiled_path)))


def _load_app(
    app: Any,
    compiled_path: Path,
    *,
    backend: BackendRuntime,
    start_rank_id: int | None,
    local_ranks_size: int | None,
    skip_warmup: bool,
) -> None:
    start_rank_id, local_ranks_size = backend.resolve_load_rank_range(
        start_rank_id=start_rank_id,
        local_ranks_size=local_ranks_size,
    )
    signature = inspect.signature(app.load)
    kwargs: dict[str, Any] = {}
    if "start_rank_id" in signature.parameters:
        kwargs["start_rank_id"] = start_rank_id
    if "local_ranks_size" in signature.parameters:
        kwargs["local_ranks_size"] = local_ranks_size
    if "skip_warmup" in signature.parameters:
        kwargs["skip_warmup"] = skip_warmup
    if kwargs:
        app.load(str(compiled_path), **kwargs)
    else:
        app.load(str(compiled_path))


def _resolve_load_rank_range(
    *,
    start_rank_id: int | None,
    local_ranks_size: int | None,
) -> tuple[int | None, int | None]:
    return get_backend("trainium").resolve_load_rank_range(
        start_rank_id=start_rank_id,
        local_ranks_size=local_ranks_size,
    )
