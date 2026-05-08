"""Unified public pipeline entry point."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

from nova.pipeline.compile_cache import CacheSpec, cache_path, has_valid_manifest, write_manifest
from nova.pipeline.parallel_config import NovaParallelConfig
from nova.pipeline.path_resolver import resolve_model_path
from nova.registry import ModelEntry, resolve_model


class NovaPipeline:
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
        parallel: NovaParallelConfig,
        dtype: Any,
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

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        parallel: NovaParallelConfig | None = None,
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
        application_kwargs: dict[str, Any] | None = None,
    ) -> "NovaPipeline":
        entry = resolve_model(model_id, model_type=model_type)
        parallel_cfg = parallel or entry.default_parallel
        dtype = dtype if dtype is not None else _default_dtype()
        shape = entry.resolve_shape(height=height, width=width, num_frames=num_frames)
        model_path = resolve_model_path(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            allow_patterns=entry.download_patterns,
        )
        spec = CacheSpec(
            model_id=model_id,
            model_path=model_path,
            model_name=entry.name,
            parallel=parallel_cfg,
            dtype=dtype,
            height=shape.get("height"),
            width=shape.get("width"),
            num_frames=shape.get("num_frames"),
            revision=revision,
        )
        compiled_path = cache_path(compile_cache_dir, spec)
        app = entry.create_application(
            model_path=model_path,
            parallel=parallel_cfg,
            dtype=dtype,
            shape=shape,
            application_kwargs=application_kwargs,
        )

        cache_ready = has_valid_manifest(compiled_path, spec)
        if not skip_compile and (force_compile or not cache_ready):
            compiled_path.mkdir(parents=True, exist_ok=True)
            _compile_app(app, compiled_path, debug=debug_compile)
            write_manifest(compiled_path, spec)

        if load:
            _load_app(
                app,
                compiled_path,
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
        )

    @classmethod
    def precompile(cls, model_id: str, **kwargs: Any) -> "NovaPipeline":
        kwargs.setdefault("load", False)
        return cls.from_pretrained(model_id, **kwargs)

    def compile(self, *, force: bool = False, debug: bool = False) -> None:
        # Use the same cache-validity check as `from_pretrained` so a stale or
        # missing manifest still triggers recompile even when the artifact dir
        # already exists.
        if force or not has_valid_manifest(self.compiled_path, self.cache_spec):
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
            start_rank_id=start_rank_id,
            local_ranks_size=local_ranks_size,
            skip_warmup=skip_warmup,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.app(*args, **kwargs)


def _default_dtype() -> Any:
    import torch

    return torch.bfloat16


def _compile_app(app: Any, compiled_path: Path, *, debug: bool) -> None:
    signature = inspect.signature(app.compile)
    if "debug" in signature.parameters:
        app.compile(str(compiled_path), debug=debug)
    else:
        app.compile(str(compiled_path))


def _load_app(
    app: Any,
    compiled_path: Path,
    *,
    start_rank_id: int | None,
    local_ranks_size: int | None,
    skip_warmup: bool,
) -> None:
    start_rank_id, local_ranks_size = _resolve_load_rank_range(
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
    if start_rank_id is not None or local_ranks_size is not None:
        return start_rank_id, local_ranks_size

    world_size = _env_int("WORLD_SIZE")
    rank = _env_int("RANK")
    if world_size is not None and world_size > 1 and rank is not None:
        # torchrun/PJRT MPMD launches one Python process per NeuronCore. Each
        # process sees one local Neuron device, so loading all ranks from every
        # process trips Neuron's "Invalid device index" check. Load exactly the
        # rank owned by this process unless the caller overrides the range.
        return rank, 1

    return None, None


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
